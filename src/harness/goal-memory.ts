import type {
  Goal,
  JsonValue,
  SpatialMemoryRecord,
  VoxelCoordinate
} from "../domain/schema.js";
import { voxelAffordanceContractStale } from "../world/voxel-affordance.js";

/**
 * Select source-backed long-term memories that name entities or voxel cells in
 * the current structured goal. This is retrieval, not authority: callers must
 * present record revisions and provenance and require current observation
 * before any physical action.
 */
export function goalRelevantSpatialMemory(
  goal: Goal,
  records: readonly SpatialMemoryRecord[],
  limit = 12
): SpatialMemoryRecord[] {
  const coordinates = new Set<string>();
  const entityIds = new Set<string>();

  for (const predicate of goal.predicates) {
    if (predicate.type === "voxel_at") {
      coordinates.add(coordinateKey(predicate.coordinate));
      continue;
    }
    if (predicate.type === "robot_in_zone") {
      entityIds.add(predicate.zone_id);
      continue;
    }
    if (predicate.type === "object_in_zone") {
      entityIds.add(predicate.object_id);
      entityIds.add(predicate.zone_id);
      continue;
    }
    if (predicate.type === "object_at"
      || predicate.type === "object_property"
      || predicate.type === "object_attached") {
      entityIds.add(predicate.object_id);
    }
  }

  if (coordinates.size === 0 && entityIds.size === 0) return [];
  const selected = records
    .filter((record) =>
      (record.coordinate !== null && coordinates.has(coordinateKey(record.coordinate)))
      || (record.entity_id !== null && entityIds.has(record.entity_id))
    )
    .sort((left, right) =>
      right.world_revision - left.world_revision
      || right.observed_frame - left.observed_frame
      || right.recorded_at.localeCompare(left.recorded_at)
      || left.id.localeCompare(right.id)
    );
  const unique = new Map<string, SpatialMemoryRecord>();
  for (const record of selected) {
    if (!unique.has(record.id)) unique.set(record.id, record);
  }
  return [...unique.values()]
    .slice(0, Math.max(0, Math.trunc(limit)))
    .map((record) => structuredClone(record));
}

/**
 * Automatic context always receives identity, provenance and staleness. Fresh
 * receipt data from the exact current world/voxel revision is also safe to
 * expose: it is the observation an agent just paid to obtain, and hiding its
 * interaction points from the parent makes compacted, stale geometry look more
 * useful than current evidence. Executable affordances disappear immediately
 * when either revision changes.
 */
export function goalMemoryContextRecords(
  records: readonly SpatialMemoryRecord[],
  current: { worldRevision: number; voxelRevision: number | null }
): JsonValue[] {
  return records.map((record) => {
    const revisionStale = record.world_revision !== current.worldRevision
      || (record.voxel_revision !== null
        && record.voxel_revision !== current.voxelRevision);
    const contractStale = voxelAffordanceContractStale(record.source_action, record.data);
    const stale = revisionStale || contractStale;
    return {
      id: record.id,
      world_id: record.world_id,
      kind: record.kind,
      label: record.label,
      summary: record.summary,
      position: record.position,
      coordinate: record.coordinate,
      entity_id: record.entity_id,
      observed_frame: record.observed_frame,
      world_revision: record.world_revision,
      voxel_revision: record.voxel_revision,
      source_transaction_id: record.source_transaction_id,
      source_agent_id: record.source_agent_id,
      source_action: record.source_action,
      recorded_at: record.recorded_at,
      stale,
      stale_reason: contractStale
        ? "voxel_affordance_contract_changed"
        : revisionStale ? "world_revision_changed" : null,
      must_reobserve_before_actuation: stale,
      omitted_dynamic_data: stale && record.data !== null,
      ...(!stale && record.data !== null
        ? {
            current_data: record.data,
            authority: "accepted receipt at the current world and voxel revision"
          }
        : {})
    };
  });
}

function coordinateKey(coordinate: VoxelCoordinate): string {
  return `${coordinate.column}:${coordinate.level}:${coordinate.row}`;
}
