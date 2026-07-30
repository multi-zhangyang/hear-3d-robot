import type {
  ActionReceipt,
  JsonValue,
  SpatialMemoryKind,
  SpatialMemoryRecord,
  Vec3,
  VoxelCoordinate,
  WorldSnapshot
} from "../domain/schema.js";

const MAX_RECORDS = 512;

export interface SpatialMemoryQuery {
  kind?: SpatialMemoryKind | undefined;
  near?: Vec3 | undefined;
  radius?: number | undefined;
  coordinate?: VoxelCoordinate | undefined;
  entityId?: string | undefined;
  text?: string | undefined;
  limit: number;
}

/**
 * Durable, provenance-carrying spatial memory derived only from accepted
 * action receipts and authoritative world snapshots.
 *
 * Agent SDK Session remains responsible for conversation history. This index
 * is deliberately narrower: coordinates, entities, terrain observations and
 * robot poses that must survive context compaction and process resume.
 */
export class SpatialMemory {
  readonly #worldId: string;
  readonly #records = new Map<string, SpatialMemoryRecord>();

  constructor(worldId: string, restore: readonly SpatialMemoryRecord[] = []) {
    this.#worldId = worldId;
    for (const record of restore) {
      if (record.world_id !== worldId) {
        throw new Error(`Spatial memory belongs to ${record.world_id}, not ${worldId}`);
      }
      this.#records.set(record.id, structuredClone(record));
    }
    this.#trim();
  }

  snapshot(): SpatialMemoryRecord[] {
    return [...this.#records.values()]
      .sort(recordOrder)
      .map((record) => structuredClone(record));
  }

  observe(receipt: ActionReceipt, world: WorldSnapshot): number {
    if (!receipt.accepted || receipt.kind === "checker" || receipt.name === "recall_spatial_memory") {
      return 0;
    }
    const records = this.#derive(receipt, world);
    for (const record of records) this.#records.set(record.id, record);
    this.#trim();
    return records.length;
  }

  query(input: SpatialMemoryQuery): {
    world_id: string;
    current_world_revision: number | null;
    matched: number;
    records: SpatialMemoryRecord[];
  } {
    const needle = input.text?.trim().toLowerCase();
    const candidates = [...this.#records.values()].filter((record) => {
      if (input.kind && record.kind !== input.kind) return false;
      if (input.entityId && record.entity_id !== input.entityId) return false;
      if (input.coordinate && !sameCoordinate(record.coordinate, input.coordinate)) return false;
      if (needle && !`${record.label}\n${record.summary}\n${JSON.stringify(record.data)}`
        .toLowerCase().includes(needle)) return false;
      if (input.near) {
        if (!record.position) return false;
        if (input.radius !== undefined && distance(record.position, input.near) > input.radius) {
          return false;
        }
      }
      return true;
    });
    candidates.sort((left, right) => {
      if (input.near) {
        const leftDistance = left.position ? distance(left.position, input.near) : Infinity;
        const rightDistance = right.position ? distance(right.position, input.near) : Infinity;
        if (leftDistance !== rightDistance) return leftDistance - rightDistance;
      }
      return right.world_revision - left.world_revision
        || right.observed_frame - left.observed_frame
        || right.recorded_at.localeCompare(left.recorded_at);
    });
    const records = candidates.slice(0, input.limit).map((record) => structuredClone(record));
    return {
      world_id: this.#worldId,
      current_world_revision: records[0]?.world_revision ?? null,
      matched: candidates.length,
      records
    };
  }

  #derive(receipt: ActionReceipt, world: WorldSnapshot): SpatialMemoryRecord[] {
    const records: SpatialMemoryRecord[] = [];
    const detail = asRecord(receipt.detail);
    const input = asRecord(receipt.input);
    const record = (
      id: string,
      kind: SpatialMemoryKind,
      label: string,
      summary: string,
      options: {
        position?: Vec3 | null;
        coordinate?: VoxelCoordinate | null;
        entityId?: string | null;
        data?: JsonValue;
      } = {}
    ): SpatialMemoryRecord => ({
      id,
      world_id: this.#worldId,
      kind,
      label,
      summary,
      position: options.position ?? null,
      coordinate: options.coordinate ?? null,
      entity_id: options.entityId ?? null,
      data: options.data ?? null,
      observed_frame: receipt.world_after_frame,
      world_revision: receipt.world_revision,
      voxel_revision: world.voxels?.revision ?? null,
      source_transaction_id: receipt.transaction_id,
      source_agent_id: receipt.agent_id,
      source_action: receipt.name,
      recorded_at: receipt.committed_at
    });

    if (receipt.name === "scan_voxels" && detail) {
      for (const value of asArray(detail.blocks)) {
        const block = asRecord(value);
        const coordinate = voxelCoordinate(block?.coordinate);
        if (!block || !coordinate) continue;
        const material = typeof block.material === "string" ? block.material : "empty";
        records.push(record(
          `voxel:${coordinateKey(coordinate)}`,
          "voxel",
          `Voxel ${coordinateKey(coordinate)}`,
          `${material} voxel observed at ${coordinateKey(coordinate)}`,
          {
            coordinate,
            position: vec3(block.center),
            data: json({
              ...block,
              affordance_contract_version:
                detail.affordance_contract_version
                ?? null
            })
          }
        ));
      }
    }

    if (receipt.name === "inspect_voxel" && detail) {
      const coordinate = voxelCoordinate(detail.coordinate ?? input?.coordinate);
      if (coordinate) {
        const material = typeof detail.material === "string" ? detail.material : "empty";
        records.push(record(
          `voxel:${coordinateKey(coordinate)}`,
          "voxel",
          `Voxel ${coordinateKey(coordinate)}`,
          `${material} voxel inspected at ${coordinateKey(coordinate)}`,
          { coordinate, position: vec3(detail.center), data: json(detail) }
        ));
      }
    }

    if ((receipt.name === "break_voxel" || receipt.name === "place_voxel") && detail) {
      const mutation = asRecord(detail.mutation);
      const coordinate = voxelCoordinate(mutation?.coordinate ?? input?.coordinate);
      if (coordinate) {
        const material = typeof mutation?.after === "string" ? mutation.after : "empty";
        records.push(record(
          `voxel:${coordinateKey(coordinate)}`,
          "voxel",
          `Voxel ${coordinateKey(coordinate)}`,
          `${material} after ${receipt.name} at ${coordinateKey(coordinate)}`,
          {
            coordinate,
            position: null,
            data: json({ mutation, inventory: detail.inventory })
          }
        ));
      }
    }

    if (receipt.name === "sense_scene" && detail) {
      for (const value of asArray(detail.visible_objects)) {
        const entity = asRecord(value);
        if (entity && typeof entity.id === "string") {
          records.push(this.#entityRecord(record, entity.id, entity, "visible object"));
        }
      }
      for (const [field, label] of [
        ["known_zones", "zone"],
        ["known_static_obstacles", "static obstacle"]
      ] as const) {
        for (const value of asArray(detail[field])) {
          const entity = asRecord(value);
          if (entity && typeof entity.id === "string") {
            records.push(this.#entityRecord(record, entity.id, entity, label));
          }
        }
      }
    }

    if (receipt.name === "inspect_entity" && detail) {
      const entityId = typeof input?.entity_id === "string"
        ? input.entity_id
        : typeof detail.id === "string" ? detail.id : null;
      if (entityId) records.push(this.#entityRecord(record, entityId, detail, "inspected entity"));
    }

    if (receipt.name === "survey_terrain" && detail) {
      const robotCell = asRecord(detail.robot_cell);
      const label = robotCell
        ? `Terrain near ${Number(robotCell.column)},${Number(robotCell.row)}`
        : `Terrain survey at frame ${receipt.world_after_frame}`;
      records.push(record(
        `terrain:${receipt.transaction_id}`,
        "terrain",
        label,
        `Local terrain survey at world revision ${receipt.world_revision}`,
        { position: world.robot.position, data: json(detail) }
      ));
    }

    if (receipt.name === "plan_base_path" && detail) {
      const target = vec3(detail.resolved_target ?? detail.requested_target);
      records.push(record(
        `navigation:${receipt.transaction_id}`,
        "navigation",
        target ? `Path to ${pointLabel(target)}` : "Navigation plan",
        `Recast/Rapier path accepted at revision ${receipt.world_revision}`,
        { position: target, data: json(detail) }
      ));
    }

    if (receipt.name === "query_contacts" && detail) {
      records.push(record(
        `contact:${receipt.transaction_id}`,
        "contact",
        `Contacts at frame ${receipt.world_after_frame}`,
        "Rapier robot contacts",
        { position: world.robot.position, data: json(detail) }
      ));
    }

    if (receipt.name === "read_proprioception" || receipt.channels.length > 0) {
      records.push(record(
        `robot:${receipt.world_revision}:${receipt.world_after_frame}`,
        "robot_pose",
        `Robot at ${pointLabel(world.robot.position)}`,
        `${receipt.name} left the robot at world revision ${receipt.world_revision}`,
        {
          position: world.robot.position,
          data: json({
            yaw: world.robot.yaw,
            joints: world.robot.joints,
            attachment: world.robot.attachment,
            result_code: receipt.code
          })
        }
      ));
    }
    return records;
  }

  #entityRecord(
    make: (
      id: string,
      kind: SpatialMemoryKind,
      label: string,
      summary: string,
      options?: {
        position?: Vec3 | null;
        coordinate?: VoxelCoordinate | null;
        entityId?: string | null;
        data?: JsonValue;
      }
    ) => SpatialMemoryRecord,
    entityId: string,
    entity: Record<string, unknown>,
    observedAs: string
  ): SpatialMemoryRecord {
    const position = vec3(entity.position ?? entity.center);
    return make(
      `entity:${entityId}`,
      "entity",
      entityId,
      `${observedAs} ${entityId}${position ? ` at ${pointLabel(position)}` : ""}`,
      { position, entityId, data: json(entity) }
    );
  }

  #trim(): void {
    if (this.#records.size <= MAX_RECORDS) return;
    const ordered = [...this.#records.values()].sort(recordOrder);
    for (const record of ordered.slice(0, ordered.length - MAX_RECORDS)) {
      this.#records.delete(record.id);
    }
  }
}

function recordOrder(left: SpatialMemoryRecord, right: SpatialMemoryRecord): number {
  return left.observed_frame - right.observed_frame
    || left.world_revision - right.world_revision
    || left.id.localeCompare(right.id);
}

function sameCoordinate(
  left: VoxelCoordinate | null,
  right: VoxelCoordinate
): boolean {
  return left !== null
    && left.column === right.column
    && left.level === right.level
    && left.row === right.row;
}

function coordinateKey(coordinate: VoxelCoordinate): string {
  return `${coordinate.column}:${coordinate.level}:${coordinate.row}`;
}

function pointLabel(point: Vec3): string {
  return `${point.x.toFixed(2)},${point.y.toFixed(2)},${point.z.toFixed(2)}`;
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function vec3(value: unknown): Vec3 | null {
  const record = asRecord(value);
  return record
    && typeof record.x === "number"
    && typeof record.y === "number"
    && typeof record.z === "number"
    ? { x: record.x, y: record.y, z: record.z }
    : null;
}

function voxelCoordinate(value: unknown): VoxelCoordinate | null {
  const record = asRecord(value);
  return record
    && Number.isInteger(record.column)
    && Number.isInteger(record.level)
    && Number.isInteger(record.row)
    ? {
        column: Number(record.column),
        level: Number(record.level),
        row: Number(record.row)
      }
    : null;
}

function json(value: unknown): JsonValue {
  const encoded = JSON.stringify(value);
  return encoded === undefined ? null : JSON.parse(encoded) as JsonValue;
}
