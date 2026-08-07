import { z } from "zod";
import { ScenarioSchema, type Scenario } from "./schema.js";
import { scenarioChunkIdForPoint } from "./scenario-chunk.js";
import {
  restoreScenarioChunkDeltaState,
  ScenarioBlockDeltaSchema,
  ScenarioDynamicEntityDeltaSchema,
  StableScenarioEntityIdSchema,
  ScenarioZoneDeltaSchema,
  type ScenarioBlockDelta,
  type ScenarioChunkDelta,
  type ScenarioChunkDeltaState,
  type ScenarioDynamicEntityDelta,
  type ScenarioZoneDelta
} from "./scenario-chunk-delta-schema.js";

const BlockValueSchema = ScenarioBlockDeltaSchema.omit({ origin: true, present: true });
const ZoneValueSchema = ScenarioZoneDeltaSchema.omit({ origin: true, present: true });
const DynamicEntityValueSchema = ScenarioDynamicEntityDeltaSchema.omit({
  origin: true,
  present: true
});

const ScenarioChunkDeltaMutationSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("create_block"), block: BlockValueSchema }).strict(),
  z.object({ type: z.literal("put_block"), block: BlockValueSchema }).strict(),
  z.object({
    type: z.literal("remove_block"),
    entity_id: StableScenarioEntityIdSchema
  }).strict(),
  z.object({ type: z.literal("create_zone"), zone: ZoneValueSchema }).strict(),
  z.object({ type: z.literal("put_zone"), zone: ZoneValueSchema }).strict(),
  z.object({
    type: z.literal("remove_zone"),
    entity_id: StableScenarioEntityIdSchema
  }).strict(),
  z.object({
    type: z.literal("create_dynamic_entity"),
    entity: DynamicEntityValueSchema
  }).strict(),
  z.object({
    type: z.literal("put_dynamic_entity"),
    entity: DynamicEntityValueSchema
  }).strict(),
  z.object({
    type: z.literal("remove_dynamic_entity"),
    entity_id: StableScenarioEntityIdSchema
  }).strict()
]);

export type ScenarioChunkDeltaMutation = z.infer<typeof ScenarioChunkDeltaMutationSchema>;

export type ResolvedScenarioChunkEntity =
  | { category: "block"; chunk_id: string; state: ScenarioBlockDelta }
  | { category: "zone"; chunk_id: string; state: ScenarioZoneDelta }
  | { category: "dynamic_entity"; chunk_id: string; state: ScenarioDynamicEntityDelta };

export interface ResolvedScenarioChunkContents {
  chunk_id: string;
  blocks: readonly ScenarioBlockDelta[];
  zones: readonly ScenarioZoneDelta[];
  dynamic_entities: readonly ScenarioDynamicEntityDelta[];
}

type EntityCategory = ResolvedScenarioChunkEntity["category"];

interface DeltaMaps {
  blocks: Map<string, ScenarioBlockDelta>;
  zones: Map<string, ScenarioZoneDelta>;
  dynamicEntities: Map<string, ScenarioDynamicEntityDelta>;
}

export function applyScenarioChunkDeltaMutation(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState,
  input: ScenarioChunkDeltaMutation
): ScenarioChunkDeltaState {
  return applyScenarioChunkDeltaMutations(scenario, persisted, [input]);
}

/**
 * Applies one logical world transaction at one chunk revision. Every entity
 * transition is validated against the state produced by the previous entry,
 * while ownership changes are published together in the final sparse overlay.
 */
export function applyScenarioChunkDeltaMutations(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState,
  inputs: readonly ScenarioChunkDeltaMutation[]
): ScenarioChunkDeltaState {
  const state = restoreScenarioChunkDeltaState(scenario, persisted);
  const mutations = inputs.map((input) => ScenarioChunkDeltaMutationSchema.parse(input));
  if (mutations.length === 0) return state;
  const maps = deltaMaps(state);
  const touchedChunkIds = new Set<string>();
  let changed = false;

  for (const mutation of mutations) {
    const transition = applyMutationToMaps(scenario, maps, mutation);
    if (!transition) continue;
    changed = true;
    touchedChunkIds.add(transition.newOwner);
    if (transition.oldOwner) touchedChunkIds.add(transition.oldOwner);
  }

  if (!changed) return state;
  const nextRevision = state.revision + 1;
  if (!Number.isSafeInteger(nextRevision)) throw new Error("Scenario chunk revision is exhausted");
  return rebuildDeltaState(
    scenario,
    state,
    maps,
    touchedChunkIds,
    nextRevision
  );
}

function applyMutationToMaps(
  scenario: Scenario,
  maps: DeltaMaps,
  mutation: ScenarioChunkDeltaMutation
): { oldOwner?: string; newOwner: string } | null {
  let replacement: ScenarioBlockDelta | ScenarioZoneDelta | ScenarioDynamicEntityDelta;
  let oldOwner: string | undefined;

  switch (mutation.type) {
    case "create_block":
      assertEntityCanBeCreated(scenario, maps, mutation.block.id);
      replacement = { ...mutation.block, origin: "created", present: true };
      break;
    case "put_block": {
      const existing = requireEntity(scenario, maps, mutation.block.id, "block");
      replacement = { ...mutation.block, origin: existing.state.origin, present: true };
      oldOwner = existing.chunk_id;
      break;
    }
    case "remove_block": {
      const existing = requireEntity(scenario, maps, mutation.entity_id, "block");
      if (!existing.state.present) return null;
      replacement = { ...existing.state, present: false };
      oldOwner = existing.chunk_id;
      break;
    }
    case "create_zone":
      assertEntityCanBeCreated(scenario, maps, mutation.zone.id);
      replacement = { ...mutation.zone, origin: "created", present: true };
      break;
    case "put_zone": {
      const existing = requireEntity(scenario, maps, mutation.zone.id, "zone");
      replacement = { ...mutation.zone, origin: existing.state.origin, present: true };
      oldOwner = existing.chunk_id;
      break;
    }
    case "remove_zone": {
      const existing = requireEntity(scenario, maps, mutation.entity_id, "zone");
      if (!existing.state.present) return null;
      replacement = { ...existing.state, present: false };
      oldOwner = existing.chunk_id;
      break;
    }
    case "create_dynamic_entity":
      assertEntityCanBeCreated(scenario, maps, mutation.entity.id);
      replacement = { ...mutation.entity, origin: "created", present: true };
      break;
    case "put_dynamic_entity": {
      const existing = requireEntity(scenario, maps, mutation.entity.id, "dynamic_entity");
      replacement = { ...mutation.entity, origin: existing.state.origin, present: true };
      oldOwner = existing.chunk_id;
      break;
    }
    case "remove_dynamic_entity": {
      const existing = requireEntity(scenario, maps, mutation.entity_id, "dynamic_entity");
      if (!existing.state.present) return null;
      replacement = { ...existing.state, present: false };
      oldOwner = existing.chunk_id;
      break;
    }
  }

  const existingDelta = deltaRecord(maps, categoryOf(replacement), replacement.id);
  if (existingDelta && JSON.stringify(existingDelta) === JSON.stringify(replacement)) return null;
  if (!existingDelta && replacement.origin === "scenario") {
    const baseline = resolveEntity(scenario, maps, replacement.id);
    if (JSON.stringify(baseline.state) === JSON.stringify(replacement)) return null;
  }

  putDeltaRecord(maps, replacement);
  const newOwner = ownerForRecord(scenario, replacement);
  return { ...(oldOwner ? { oldOwner } : {}), newOwner };
}

export function resolveScenarioChunkDeltaEntity(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState,
  entityId: string
): ResolvedScenarioChunkEntity {
  const state = restoreScenarioChunkDeltaState(scenario, persisted);
  const maps = deltaMaps(state);
  return resolveEntity(scenario, maps, entityId);
}

/**
 * Resolves one current chunk from the immutable Scenario baseline plus its
 * sparse durable overlay. Physics bubbles, Recast tiles and render culling can
 * share this boundary without owning persistence or migration rules.
 */
export function resolveScenarioChunkDeltaContents(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState,
  chunkId: string
): ResolvedScenarioChunkContents {
  const state = restoreScenarioChunkDeltaState(scenario, persisted);
  if (!scenario.chunk_manifest.chunks.some(({ id }) => id === chunkId)) {
    throw new Error(`Unknown scenario chunk: ${chunkId}`);
  }
  const maps = deltaMaps(state);
  const ids = new Set([
    ...scenario.obstacles.map(({ id }) => id),
    ...scenario.zones.map(({ id }) => id),
    ...scenario.objects.map(({ id }) => id),
    ...maps.blocks.keys(),
    ...maps.zones.keys(),
    ...maps.dynamicEntities.keys()
  ]);
  const blocks: ScenarioBlockDelta[] = [];
  const zones: ScenarioZoneDelta[] = [];
  const dynamicEntities: ScenarioDynamicEntityDelta[] = [];
  for (const id of ids) {
    const resolved = resolveEntity(scenario, maps, id);
    if (!resolved.state.present || resolved.chunk_id !== chunkId) continue;
    if (resolved.category === "block") blocks.push(resolved.state);
    else if (resolved.category === "zone") zones.push(resolved.state);
    else dynamicEntities.push(resolved.state);
  }
  blocks.sort(compareEntityIds);
  zones.sort(compareEntityIds);
  dynamicEntities.sort(compareEntityIds);
  return {
    chunk_id: chunkId,
    blocks,
    zones,
    dynamic_entities: dynamicEntities
  };
}

/** Resolves the durable sparse overlay into the geometry consumed by physics and navigation. */
export function materializeScenarioChunkDeltaState(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState
): Scenario {
  const state = restoreScenarioChunkDeltaState(scenario, persisted);
  const contents = scenario.chunk_manifest.chunks.map(({ id }) => (
    resolveScenarioChunkDeltaContents(scenario, state, id)
  ));
  const blocks = stableMaterializedOrder(
    contents.flatMap(({ blocks }) => blocks),
    scenario.obstacles
  );
  const entities = stableMaterializedOrder(
    contents.flatMap(({ dynamic_entities }) => dynamic_entities),
    scenario.objects
  );
  const zones = stableMaterializedOrder(
    contents.flatMap(({ zones }) => zones).filter((zone) => zone.enabled),
    scenario.zones
  );
  return ScenarioSchema.parse({
    ...structuredClone(scenario),
    obstacles: blocks.map((block) => ({
      id: block.id,
      center: structuredClone(block.center),
      size: structuredClone(block.size)
    })),
    objects: entities.map((entity) => {
      const baseline = scenario.objects.find(({ id }) => id === entity.id);
      return {
        id: entity.id,
        kind: entity.kind,
        color: entity.color,
        position: structuredClone(entity.position),
        size: structuredClone(entity.size),
        portable: entity.portable,
        ...(baseline?.capability
          ? { capability: structuredClone(baseline.capability) }
          : {})
      };
    }),
    zones: zones.map((zone) => ({
      id: zone.id,
      color: zone.color,
      center: structuredClone(zone.center),
      size: structuredClone(zone.size)
    })),
    chunk_manifest: undefined
  });
}

function stableMaterializedOrder<T extends { id: string }>(
  current: readonly T[],
  baseline: readonly { id: string }[]
): T[] {
  const indexes = new Map(baseline.map((entry, index) => [entry.id, index]));
  return [...current].sort((left, right) => {
    const leftIndex = indexes.get(left.id);
    const rightIndex = indexes.get(right.id);
    if (leftIndex !== undefined && rightIndex !== undefined) return leftIndex - rightIndex;
    if (leftIndex !== undefined) return -1;
    if (rightIndex !== undefined) return 1;
    return compareEntityIds(left, right);
  });
}

function resolveEntity(
  scenario: Scenario,
  maps: DeltaMaps,
  entityId: string
): ResolvedScenarioChunkEntity {
  const block = maps.blocks.get(entityId);
  if (block) return { category: "block", chunk_id: ownerForRecord(scenario, block), state: block };
  const zone = maps.zones.get(entityId);
  if (zone) return { category: "zone", chunk_id: ownerForRecord(scenario, zone), state: zone };
  const dynamic = maps.dynamicEntities.get(entityId);
  if (dynamic) {
    return {
      category: "dynamic_entity",
      chunk_id: ownerForRecord(scenario, dynamic),
      state: dynamic
    };
  }

  const obstacle = scenario.obstacles.find(({ id }) => id === entityId);
  if (obstacle) {
    const baseline: ScenarioBlockDelta = {
      id: obstacle.id,
      origin: "scenario",
      present: true,
      center: structuredClone(obstacle.center),
      size: structuredClone(obstacle.size),
      material: "solid",
      properties: {}
    };
    return { category: "block", chunk_id: ownerForRecord(scenario, baseline), state: baseline };
  }
  const descriptor = scenario.objects.find(({ id }) => id === entityId);
  if (descriptor) {
    const baseline: ScenarioDynamicEntityDelta = {
      id: descriptor.id,
      origin: "scenario",
      present: true,
      kind: descriptor.kind,
      color: descriptor.color,
      position: structuredClone(descriptor.position),
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      linear_velocity: { x: 0, y: 0, z: 0 },
      angular_velocity: { x: 0, y: 0, z: 0 },
      size: structuredClone(descriptor.size),
      portable: descriptor.portable,
      properties: {}
    };
    return {
      category: "dynamic_entity",
      chunk_id: ownerForRecord(scenario, baseline),
      state: baseline
    };
  }
  const scenarioZone = scenario.zones.find(({ id }) => id === entityId);
  if (scenarioZone) {
    const baseline: ScenarioZoneDelta = {
      id: scenarioZone.id,
      origin: "scenario",
      present: true,
      color: scenarioZone.color,
      center: structuredClone(scenarioZone.center),
      size: structuredClone(scenarioZone.size),
      enabled: true,
      properties: {}
    };
    return { category: "zone", chunk_id: ownerForRecord(scenario, baseline), state: baseline };
  }
  throw new Error(`Unknown scenario entity: ${entityId}`);
}

function requireEntity<C extends EntityCategory>(
  scenario: Scenario,
  maps: DeltaMaps,
  entityId: string,
  category: C
): Extract<ResolvedScenarioChunkEntity, { category: C }> {
  const resolved = resolveEntity(scenario, maps, entityId);
  if (resolved.category !== category) {
    throw new Error(`Scenario entity ${entityId} is ${resolved.category}, not ${category}`);
  }
  return resolved as Extract<ResolvedScenarioChunkEntity, { category: C }>;
}

function assertEntityCanBeCreated(
  scenario: Scenario,
  maps: DeltaMaps,
  entityId: string
): void {
  if (maps.blocks.has(entityId) || maps.zones.has(entityId) || maps.dynamicEntities.has(entityId)
    || scenario.obstacles.some(({ id }) => id === entityId)
    || scenario.zones.some(({ id }) => id === entityId)
    || scenario.objects.some(({ id }) => id === entityId)) {
    throw new Error(`Scenario entity ID already exists: ${entityId}`);
  }
}

function deltaMaps(state: ScenarioChunkDeltaState): DeltaMaps {
  const result: DeltaMaps = {
    blocks: new Map(),
    zones: new Map(),
    dynamicEntities: new Map()
  };
  for (const chunk of state.chunks) {
    for (const record of chunk.blocks) result.blocks.set(record.id, structuredClone(record));
    for (const record of chunk.zones) result.zones.set(record.id, structuredClone(record));
    for (const record of chunk.dynamic_entities) {
      result.dynamicEntities.set(record.id, structuredClone(record));
    }
  }
  return result;
}

function deltaRecord(
  maps: DeltaMaps,
  category: EntityCategory,
  entityId: string
): ScenarioBlockDelta | ScenarioZoneDelta | ScenarioDynamicEntityDelta | undefined {
  if (category === "block") return maps.blocks.get(entityId);
  if (category === "zone") return maps.zones.get(entityId);
  return maps.dynamicEntities.get(entityId);
}

function putDeltaRecord(
  maps: DeltaMaps,
  record: ScenarioBlockDelta | ScenarioZoneDelta | ScenarioDynamicEntityDelta
): void {
  const category = categoryOf(record);
  if (category === "block") maps.blocks.set(record.id, record as ScenarioBlockDelta);
  else if (category === "zone") maps.zones.set(record.id, record as ScenarioZoneDelta);
  else maps.dynamicEntities.set(record.id, record as ScenarioDynamicEntityDelta);
}

function categoryOf(
  record: ScenarioBlockDelta | ScenarioZoneDelta | ScenarioDynamicEntityDelta
): EntityCategory {
  if ("position" in record) return "dynamic_entity";
  return "enabled" in record ? "zone" : "block";
}

function ownerForRecord(
  scenario: Scenario,
  record: ScenarioBlockDelta | ScenarioZoneDelta | ScenarioDynamicEntityDelta
): string {
  const point = "position" in record ? record.position : record.center;
  return scenarioChunkIdForPoint(scenario, scenario.chunk_manifest, point);
}

function rebuildDeltaState(
  scenario: Scenario,
  previous: ScenarioChunkDeltaState,
  maps: DeltaMaps,
  touchedChunkIds: ReadonlySet<string>,
  revision: number
): ScenarioChunkDeltaState {
  const groups = new Map<string, ScenarioChunkDelta>();
  const previousRevisions = new Map(
    previous.chunks.map((chunk) => [chunk.chunk_id, chunk.revision])
  );
  const add = (
    category: "blocks" | "zones" | "dynamic_entities",
    record: ScenarioBlockDelta | ScenarioZoneDelta | ScenarioDynamicEntityDelta
  ): void => {
    const chunkId = ownerForRecord(scenario, record);
    let group = groups.get(chunkId);
    if (!group) {
      group = {
        chunk_id: chunkId,
        revision: touchedChunkIds.has(chunkId)
          ? revision
          : (previousRevisions.get(chunkId) ?? revision),
        blocks: [],
        zones: [],
        dynamic_entities: []
      };
      groups.set(chunkId, group);
    }
    if (category === "blocks") group.blocks.push(record as ScenarioBlockDelta);
    else if (category === "zones") group.zones.push(record as ScenarioZoneDelta);
    else group.dynamic_entities.push(record as ScenarioDynamicEntityDelta);
  };
  for (const record of maps.blocks.values()) add("blocks", record);
  for (const record of maps.zones.values()) add("zones", record);
  for (const record of maps.dynamicEntities.values()) add("dynamic_entities", record);
  for (const group of groups.values()) {
    group.blocks.sort(compareEntityIds);
    group.zones.sort(compareEntityIds);
    group.dynamic_entities.sort(compareEntityIds);
  }
  const manifestOrder = new Map(
    scenario.chunk_manifest.chunks.map((chunk, index) => [chunk.id, index])
  );
  const chunks = [...groups.values()].sort((left, right) => (
    manifestOrder.get(left.chunk_id)! - manifestOrder.get(right.chunk_id)!
  ));
  const changedChunkIds = [...touchedChunkIds].sort((left, right) => (
    manifestOrder.get(left)! - manifestOrder.get(right)!
  ));
  return restoreScenarioChunkDeltaState(scenario, {
    ...previous,
    revision,
    changed_chunk_ids: changedChunkIds,
    chunks
  });
}

function compareEntityIds(left: { id: string }, right: { id: string }): number {
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
}
