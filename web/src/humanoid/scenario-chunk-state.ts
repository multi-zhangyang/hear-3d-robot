import type {
  Quaternion,
  ScenarioChunkDeltaState,
  ScenarioDefinition,
  Vec3
} from "../types";

export interface ScenarioVisualBlock {
  id: string;
  center: Vec3;
  size: Vec3;
  material: string;
}

export interface ScenarioVisualZone {
  id: string;
  color: string;
  center: Vec3;
  size: Vec3;
}

export interface ScenarioVisualObject {
  id: string;
  kind: string;
  color: string;
  position: Vec3;
  rotation: Quaternion;
  size: Vec3;
  portable: boolean;
}

interface ScenarioChunkVisuals {
  blocks: ScenarioVisualBlock[];
  zones: ScenarioVisualZone[];
  objects: ScenarioVisualObject[];
}

export function changedScenarioVisualChunkIds(
  current: ScenarioWorldVisualState,
  next: ScenarioWorldVisualState
): string[] {
  const ids = new Set([...current.chunks.keys(), ...next.chunks.keys()]);
  return [...ids].filter((id) => (
    JSON.stringify(current.chunks.get(id) ?? null)
      !== JSON.stringify(next.chunks.get(id) ?? null)
  )).sort();
}

export interface ScenarioWorldVisualState {
  revision: number;
  chunks: Map<string, ScenarioChunkVisuals>;
  objects: Map<string, ScenarioVisualObject>;
}

export function resolveScenarioWorldVisualState(
  scenario: ScenarioDefinition,
  deltas: ScenarioChunkDeltaState
): ScenarioWorldVisualState {
  if (deltas.scenario_seed !== scenario.seed) {
    throw new Error("场景分块状态与世界种子不一致");
  }
  const chunkIds = new Set(scenario.chunk_manifest.chunks.map(({ id }) => id));
  const blockOwners = baselineOwners(scenario, "obstacles");
  const objectOwners = baselineOwners(scenario, "objects");
  const zoneOwners = baselineOwners(scenario, "zones");
  const blocks = new Map(scenario.obstacles.map((block) => [block.id, {
    owner: requiredOwner(blockOwners, block.id),
    value: {
      id: block.id,
      center: structuredClone(block.center),
      size: structuredClone(block.size),
      material: "solid"
    }
  }]));
  const objects = new Map(scenario.objects.map((object) => [object.id, {
    owner: requiredOwner(objectOwners, object.id),
    value: {
      ...structuredClone(object),
      rotation: { x: 0, y: 0, z: 0, w: 1 }
    }
  }]));
  const zones = new Map(scenario.zones.map((zone) => [zone.id, {
    owner: requiredOwner(zoneOwners, zone.id),
    value: structuredClone(zone)
  }]));

  for (const chunk of deltas.chunks) {
    if (!chunkIds.has(chunk.chunk_id)) {
      throw new Error(`场景分块状态包含未知区块：${chunk.chunk_id}`);
    }
    for (const block of chunk.blocks) {
      if (!block.present) blocks.delete(block.id);
      else blocks.set(block.id, {
        owner: chunk.chunk_id,
        value: {
          id: block.id,
          center: structuredClone(block.center),
          size: structuredClone(block.size),
          material: block.material
        }
      });
    }
    for (const object of chunk.dynamic_entities) {
      if (!object.present) objects.delete(object.id);
      else objects.set(object.id, {
        owner: chunk.chunk_id,
        value: {
          id: object.id,
          kind: object.kind,
          color: object.color,
          position: structuredClone(object.position),
          rotation: structuredClone(object.rotation),
          size: structuredClone(object.size),
          portable: object.portable
        }
      });
    }
    for (const zone of chunk.zones) {
      if (!zone.present || !zone.enabled) zones.delete(zone.id);
      else zones.set(zone.id, {
        owner: chunk.chunk_id,
        value: {
          id: zone.id,
          color: zone.color,
          center: structuredClone(zone.center),
          size: structuredClone(zone.size)
        }
      });
    }
  }

  const chunks = new Map<string, ScenarioChunkVisuals>(
    scenario.chunk_manifest.chunks.map(({ id }) => [id, {
      blocks: [],
      zones: [],
      objects: []
    }])
  );
  for (const { owner, value } of blocks.values()) chunks.get(owner)!.blocks.push(value);
  for (const { owner, value } of zones.values()) chunks.get(owner)!.zones.push(value);
  for (const { owner, value } of objects.values()) chunks.get(owner)!.objects.push(value);
  for (const contents of chunks.values()) {
    contents.blocks.sort(compareIds);
    contents.zones.sort(compareIds);
    contents.objects.sort(compareIds);
  }
  return {
    revision: deltas.revision,
    chunks,
    objects: new Map([...objects].map(([id, entry]) => [id, entry.value]))
  };
}

function baselineOwners(
  scenario: ScenarioDefinition,
  category: "obstacles" | "objects" | "zones"
): Map<string, string> {
  const result = new Map<string, string>();
  for (const chunk of scenario.chunk_manifest.chunks) {
    for (const id of chunk.entity_ids[category]) result.set(id, chunk.id);
  }
  return result;
}

function requiredOwner(owners: ReadonlyMap<string, string>, id: string): string {
  const owner = owners.get(id);
  if (!owner) throw new Error(`场景实体缺少区块归属：${id}`);
  return owner;
}

function compareIds(left: { id: string }, right: { id: string }): number {
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
}
