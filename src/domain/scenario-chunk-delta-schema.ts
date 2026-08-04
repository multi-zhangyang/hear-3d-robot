import { createHash } from "node:crypto";
import { z } from "zod";
import {
  JsonValueSchema,
  Vec3Schema,
  type Scenario,
  type Vec3
} from "./schema.js";
import { scenarioChunkIdForPoint } from "./scenario-chunk.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const STABLE_ENTITY_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;

export const StableScenarioEntityIdSchema = z.string().regex(STABLE_ENTITY_ID_PATTERN);

const Size3Schema = z.object({
  x: z.number().finite().positive(),
  y: z.number().finite().positive(),
  z: z.number().finite().positive()
}).strict();

const QuaternionSchema = z.object({
  x: z.number().finite(),
  y: z.number().finite(),
  z: z.number().finite(),
  w: z.number().finite()
}).strict().refine((value) => (
  Math.abs(Math.hypot(value.x, value.y, value.z, value.w) - 1) <= 1e-6
), "Rotation quaternion must be normalized");

const MAX_PROPERTIES_BYTES = 16 * 1024;
const MAX_PROPERTIES_DEPTH = 8;
const PropertiesSchema = z.record(z.string(), JsonValueSchema).superRefine(
  (properties, context) => {
    if (Buffer.byteLength(JSON.stringify(properties)) > MAX_PROPERTIES_BYTES) {
      context.addIssue({
        code: "custom",
        message: `Entity properties cannot exceed ${MAX_PROPERTIES_BYTES} UTF-8 bytes`
      });
    }
    if (jsonDepth(properties) > MAX_PROPERTIES_DEPTH) {
      context.addIssue({
        code: "custom",
        message: `Entity properties cannot exceed nesting depth ${MAX_PROPERTIES_DEPTH}`
      });
    }
  }
);
const EntityOriginSchema = z.enum(["scenario", "created"]);

export const ScenarioPhysicalAuthoritySchema = z.object({
  source: z.literal("humanoid_mujoco"),
  transaction_id: z.string().trim().min(1),
  world_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative()
}).strict();

export const ScenarioBlockDeltaSchema = z.object({
  id: StableScenarioEntityIdSchema,
  origin: EntityOriginSchema,
  present: z.boolean(),
  center: Vec3Schema,
  size: Size3Schema,
  material: z.string().trim().min(1),
  properties: PropertiesSchema
}).strict();

export const ScenarioZoneDeltaSchema = z.object({
  id: StableScenarioEntityIdSchema,
  origin: EntityOriginSchema,
  present: z.boolean(),
  color: z.string().trim().min(1),
  center: Vec3Schema,
  size: Size3Schema,
  enabled: z.boolean(),
  properties: PropertiesSchema
}).strict();

export const ScenarioDynamicEntityDeltaSchema = z.object({
  id: StableScenarioEntityIdSchema,
  origin: EntityOriginSchema,
  present: z.boolean(),
  kind: z.string().trim().min(1),
  color: z.string().trim().min(1),
  position: Vec3Schema,
  rotation: QuaternionSchema,
  linear_velocity: Vec3Schema,
  angular_velocity: Vec3Schema,
  size: Size3Schema,
  portable: z.boolean(),
  properties: PropertiesSchema,
  physical_authority: ScenarioPhysicalAuthoritySchema.optional()
}).strict();

export type ScenarioBlockDelta = z.infer<typeof ScenarioBlockDeltaSchema>;
export type ScenarioZoneDelta = z.infer<typeof ScenarioZoneDeltaSchema>;
export type ScenarioDynamicEntityDelta = z.infer<typeof ScenarioDynamicEntityDeltaSchema>;

const ScenarioChunkDeltaSchema = z.object({
  chunk_id: z.string().trim().min(1),
  revision: z.number().int().positive(),
  blocks: z.array(ScenarioBlockDeltaSchema),
  zones: z.array(ScenarioZoneDeltaSchema),
  dynamic_entities: z.array(ScenarioDynamicEntityDeltaSchema)
}).strict();

const ScenarioChunkDeltaStateSchema = z.object({
  version: z.literal(1),
  scenario_seed: z.number().int().nonnegative(),
  scenario_sha256: z.string().regex(SHA256_PATTERN),
  manifest_version: z.literal(1),
  revision: z.number().int().nonnegative(),
  changed_chunk_ids: z.array(z.string().trim().min(1)).default([]),
  chunks: z.array(ScenarioChunkDeltaSchema)
}).strict();

export type ScenarioChunkDelta = z.infer<typeof ScenarioChunkDeltaSchema>;

/** Sparse, atomically replaced overlay bound to one exact materialized Scenario. */
export type ScenarioChunkDeltaState = z.infer<typeof ScenarioChunkDeltaStateSchema>;

type DeltaCategory = "block" | "zone" | "dynamic_entity";
type DeltaRecord = ScenarioBlockDelta | ScenarioZoneDelta | ScenarioDynamicEntityDelta;

interface CategorizedRecord {
  category: DeltaCategory;
  record: DeltaRecord;
}

export function createScenarioChunkDeltaState(scenario: Scenario): ScenarioChunkDeltaState {
  return ScenarioChunkDeltaStateSchema.parse({
    version: 1,
    scenario_seed: scenario.seed,
    scenario_sha256: scenarioBaselineSha256(scenario),
    manifest_version: scenario.chunk_manifest.version,
    revision: 0,
    changed_chunk_ids: [],
    chunks: []
  });
}

export function restoreScenarioChunkDeltaState(
  scenario: Scenario,
  persisted: unknown
): ScenarioChunkDeltaState {
  const state = ScenarioChunkDeltaStateSchema.parse(persisted);
  const issues = chunkDeltaIntegrityIssues(scenario, state);
  if (issues.length > 0) {
    throw new Error(`Invalid scenario chunk delta state: ${issues.join("; ")}`);
  }
  return state;
}

function chunkDeltaIntegrityIssues(
  scenario: Scenario,
  state: ScenarioChunkDeltaState
): string[] {
  const issues: string[] = [];
  if (state.scenario_seed !== scenario.seed) {
    issues.push("scenario seed does not match the run definition");
  }
  if (state.scenario_sha256 !== scenarioBaselineSha256(scenario)) {
    issues.push("scenario baseline hash does not match the run definition");
  }
  if (state.manifest_version !== scenario.chunk_manifest.version) {
    issues.push("chunk manifest version does not match the run definition");
  }

  const chunkIndexes = new Map(
    scenario.chunk_manifest.chunks.map((chunk, index) => [chunk.id, index])
  );
  const chunkIds = new Set<string>();
  const entityAssignments = new Map<string, { category: DeltaCategory; chunkId: string }>();
  const baselineCategories = baselineEntityCategories(scenario);
  let previousChunkIndex = -1;
  let maximumRevision = 0;

  const changedChunkIds = new Set<string>();
  let previousChangedIndex = -1;
  for (const chunkId of state.changed_chunk_ids) {
    const index = chunkIndexes.get(chunkId);
    if (index === undefined) {
      issues.push(`unknown changed chunk: ${chunkId}`);
      continue;
    }
    if (changedChunkIds.has(chunkId) || index <= previousChangedIndex) {
      issues.push("changed chunk IDs must be unique and use manifest order");
    }
    changedChunkIds.add(chunkId);
    previousChangedIndex = index;
  }

  for (const chunk of state.chunks) {
    const chunkIndex = chunkIndexes.get(chunk.chunk_id);
    if (chunkIndex === undefined) {
      issues.push(`unknown chunk delta owner: ${chunk.chunk_id}`);
    } else {
      if (chunkIndex <= previousChunkIndex) {
        issues.push("chunk deltas must be unique and use manifest order");
      }
      previousChunkIndex = chunkIndex;
    }
    if (chunkIds.has(chunk.chunk_id)) issues.push(`duplicate chunk delta: ${chunk.chunk_id}`);
    chunkIds.add(chunk.chunk_id);
    maximumRevision = Math.max(maximumRevision, chunk.revision);
    if (chunk.revision > state.revision) {
      issues.push(`${chunk.chunk_id} revision cannot exceed the state revision`);
    }
    if (changedChunkIds.has(chunk.chunk_id) && chunk.revision !== state.revision) {
      issues.push(`${chunk.chunk_id} changed marker must match the state revision`);
    }
    if (chunk.revision === state.revision && !changedChunkIds.has(chunk.chunk_id)) {
      issues.push(`${chunk.chunk_id} newest revision is missing its changed marker`);
    }
    const categorized = categorizedRecords(chunk);
    if (categorized.length === 0) {
      issues.push(`${chunk.chunk_id} is empty and must not be persisted`);
    }
    for (const records of [chunk.blocks, chunk.zones, chunk.dynamic_entities]) {
      if (!isUniqueSorted(records.map(({ id }) => id))) {
        issues.push(`${chunk.chunk_id} entity records must be unique and sorted`);
      }
    }
    for (const { category, record } of categorized) {
      const previous = entityAssignments.get(record.id);
      if (previous) {
        issues.push(
          `entity ${record.id} is duplicated in ${previous.chunkId} and ${chunk.chunk_id}`
        );
      } else {
        entityAssignments.set(record.id, { category, chunkId: chunk.chunk_id });
      }

      const baselineCategory = baselineCategories.get(record.id);
      if (record.origin === "scenario") {
        if (baselineCategory === undefined) {
          issues.push(`scenario delta references unknown ${category} entity: ${record.id}`);
        } else if (baselineCategory !== category) {
          issues.push(
            `scenario entity ${record.id} is ${baselineCategory}, not ${category}`
          );
        }
      } else if (baselineCategory !== undefined) {
        issues.push(`created entity ${record.id} conflicts with a scenario ${baselineCategory}`);
      }

      try {
        const expectedOwner = scenarioChunkIdForPoint(
          scenario,
          scenario.chunk_manifest,
          recordAnchor(category, record)
        );
        if (chunk.chunk_id !== expectedOwner) {
          issues.push(`entity ${record.id} must belong to ${expectedOwner}`);
        }
      } catch (error) {
        issues.push(error instanceof Error ? `${record.id}: ${error.message}` : String(error));
      }
    }
  }

  if (state.revision === 0 && state.chunks.length > 0) {
    issues.push("revision zero cannot contain chunk deltas");
  }
  if (state.revision === 0 && state.changed_chunk_ids.length > 0) {
    issues.push("revision zero cannot contain changed chunk markers");
  }
  if (state.revision > 0 && state.chunks.length === 0) {
    issues.push("a revised chunk delta state cannot be empty");
  }
  if (state.revision > 0 && state.changed_chunk_ids.length === 0) {
    issues.push("a revised chunk delta state must identify its changed chunks");
  }
  if (maximumRevision !== state.revision) {
    issues.push("state revision must equal the newest chunk revision");
  }
  return issues;
}

function baselineEntityCategories(scenario: Scenario): Map<string, DeltaCategory> {
  const result = new Map<string, DeltaCategory>();
  for (const { id } of scenario.obstacles) result.set(id, "block");
  for (const { id } of scenario.zones) result.set(id, "zone");
  for (const { id } of scenario.objects) result.set(id, "dynamic_entity");
  return result;
}

function categorizedRecords(chunk: ScenarioChunkDelta): CategorizedRecord[] {
  return [
    ...chunk.blocks.map((record): CategorizedRecord => ({ category: "block", record })),
    ...chunk.zones.map((record): CategorizedRecord => ({ category: "zone", record })),
    ...chunk.dynamic_entities.map((record): CategorizedRecord => ({
      category: "dynamic_entity",
      record
    }))
  ];
}

function recordAnchor(category: DeltaCategory, record: DeltaRecord): Pick<Vec3, "x" | "z"> {
  return category === "dynamic_entity"
    ? (record as ScenarioDynamicEntityDelta).position
    : (record as ScenarioBlockDelta | ScenarioZoneDelta).center;
}

function isUniqueSorted(ids: readonly string[]): boolean {
  return ids.every((id, index) => index === 0 || ids[index - 1]! < id);
}

function scenarioBaselineSha256(scenario: Scenario): string {
  return createHash("sha256").update(canonicalJson(scenario)).digest("hex");
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalJson(record[key])}`
  )).join(",")}}`;
}

function jsonDepth(root: unknown): number {
  const stack: Array<{ value: unknown; depth: number }> = [{ value: root, depth: 0 }];
  let maximum = 0;
  while (stack.length > 0) {
    const current = stack.pop()!;
    maximum = Math.max(maximum, current.depth);
    if (current.depth > MAX_PROPERTIES_DEPTH || current.value === null
      || typeof current.value !== "object") continue;
    const children = Array.isArray(current.value)
      ? current.value
      : Object.values(current.value as Record<string, unknown>);
    for (const value of children) stack.push({ value, depth: current.depth + 1 });
  }
  return maximum;
}
