import { createHash } from "node:crypto";
import { z } from "zod";
import {
  QuaternionSchema,
  Vec3Schema,
  type Scenario
} from "./schema.js";
import {
  resolveScenarioChunkDeltaEntity,
  type ScenarioChunkDeltaMutation
} from "./scenario-chunk-delta.js";
import {
  ScenarioPhysicalAuthoritySchema,
  StableScenarioEntityIdSchema,
  restoreScenarioChunkDeltaState,
  type ScenarioChunkDeltaState
} from "./scenario-chunk-delta-schema.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

const ScenarioPhysicalDynamicEntitySchema = z.object({
  id: StableScenarioEntityIdSchema,
  position: Vec3Schema,
  rotation: QuaternionSchema,
  linear_velocity: Vec3Schema,
  angular_velocity: Vec3Schema,
  physical_authority: ScenarioPhysicalAuthoritySchema
}).strict();

export const ScenarioPhysicalWorldDeltaSchema = z.object({
  version: z.literal(1),
  transaction_id: z.string().trim().min(1),
  source_world_frame: z.number().int().nonnegative(),
  source_world_revision: z.number().int().nonnegative(),
  base_chunk_revision: z.number().int().nonnegative(),
  entities: z.array(ScenarioPhysicalDynamicEntitySchema).min(1),
  state_sha256: z.string().regex(SHA256_PATTERN)
}).strict().superRefine((record, context) => {
  const ids = new Set<string>();
  let previousId: string | undefined;
  record.entities.forEach((entity, index) => {
    if (ids.has(entity.id) || (previousId !== undefined && previousId >= entity.id)) {
      context.addIssue({
        code: "custom",
        path: ["entities", index, "id"],
        message: "Physical world delta entities must be unique and sorted"
      });
    }
    ids.add(entity.id);
    previousId = entity.id;
    const authority = entity.physical_authority;
    if (authority.transaction_id !== record.transaction_id
      || authority.world_frame !== record.source_world_frame
      || authority.world_revision !== record.source_world_revision) {
      context.addIssue({
        code: "custom",
        path: ["entities", index, "physical_authority"],
        message: "Physical entity authority does not match its world delta transaction"
      });
    }
  });
  if (record.state_sha256 !== scenarioPhysicalWorldDeltaSha256(record)) {
    context.addIssue({
      code: "custom",
      path: ["state_sha256"],
      message: "Physical world delta integrity hash does not match"
    });
  }
});

export type ScenarioPhysicalWorldDelta = z.infer<
  typeof ScenarioPhysicalWorldDeltaSchema
>;
export type ScenarioPhysicalDynamicEntity = z.infer<
  typeof ScenarioPhysicalDynamicEntitySchema
>;

export function createScenarioPhysicalWorldDelta(input: Omit<
  ScenarioPhysicalWorldDelta,
  "state_sha256"
>): ScenarioPhysicalWorldDelta {
  const payload = {
    ...structuredClone(input),
    entities: [...input.entities].sort(compareEntityIds)
  };
  return ScenarioPhysicalWorldDeltaSchema.parse({
    ...payload,
    state_sha256: scenarioPhysicalWorldDeltaSha256(payload)
  });
}

export function scenarioPhysicalWorldDeltaMutations(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState,
  input: ScenarioPhysicalWorldDelta
): ScenarioChunkDeltaMutation[] {
  const state = restoreScenarioChunkDeltaState(scenario, persisted);
  const record = ScenarioPhysicalWorldDeltaSchema.parse(input);
  if (state.revision < record.base_chunk_revision) {
    throw new Error(
      `Physical world delta base revision is ahead of durable chunks: `
      + `${record.base_chunk_revision} > ${state.revision}`
    );
  }
  const mutations: ScenarioChunkDeltaMutation[] = [];
  for (const entity of record.entities) {
    const current = resolveScenarioChunkDeltaEntity(scenario, state, entity.id);
    if (current.category !== "dynamic_entity") {
      throw new Error(`Physical world delta entity is not dynamic: ${entity.id}`);
    }
    if (!current.state.present || !current.state.portable) {
      throw new Error(`Physical world delta entity is not an active portable body: ${entity.id}`);
    }
    const authority = current.state.physical_authority;
    if (authority && authority.world_revision > record.source_world_revision) continue;
    if (authority && authority.world_revision === record.source_world_revision) {
      if (authority.transaction_id !== record.transaction_id
        || !samePhysicalState(current.state, entity)) {
        throw new Error(
          `Physical world delta conflicts at world revision `
          + `${record.source_world_revision}: ${entity.id}`
        );
      }
      continue;
    }
    if (!authority && state.revision > record.base_chunk_revision
      && !samePhysicalState(current.state, entity)) {
      throw new Error(
        `Physical world delta would overwrite a newer unversioned entity state: ${entity.id}`
      );
    }
    mutations.push({
      type: "put_dynamic_entity",
      entity: {
        id: current.state.id,
        kind: current.state.kind,
        color: current.state.color,
        position: structuredClone(entity.position),
        rotation: structuredClone(entity.rotation),
        linear_velocity: structuredClone(entity.linear_velocity),
        angular_velocity: structuredClone(entity.angular_velocity),
        size: structuredClone(current.state.size),
        portable: current.state.portable,
        properties: structuredClone(current.state.properties),
        physical_authority: structuredClone(entity.physical_authority)
      }
    });
  }
  return mutations;
}

export function assertScenarioPhysicalWorldDeltaApplied(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState,
  input: ScenarioPhysicalWorldDelta
): void {
  const state = restoreScenarioChunkDeltaState(scenario, persisted);
  const record = ScenarioPhysicalWorldDeltaSchema.parse(input);
  for (const entity of record.entities) {
    const current = resolveScenarioChunkDeltaEntity(scenario, state, entity.id);
    if (current.category !== "dynamic_entity") {
      throw new Error(`Applied physical entity is not dynamic: ${entity.id}`);
    }
    const authority = current.state.physical_authority;
    if (!authority || authority.world_revision < record.source_world_revision) {
      throw new Error(`Physical world delta was not durably applied: ${entity.id}`);
    }
    if (authority.world_revision === record.source_world_revision
      && (authority.transaction_id !== record.transaction_id
        || !samePhysicalState(current.state, entity))) {
      throw new Error(`Applied physical world delta does not match its source: ${entity.id}`);
    }
  }
}

export function scenarioPhysicalWorldDeltaSha256(
  input: Omit<ScenarioPhysicalWorldDelta, "state_sha256"> | ScenarioPhysicalWorldDelta
): string {
  const { state_sha256: _stateSha256, ...payload } = input as (
    Partial<Pick<ScenarioPhysicalWorldDelta, "state_sha256">>
    & Omit<ScenarioPhysicalWorldDelta, "state_sha256">
  );
  return createHash("sha256").update(canonicalJson(payload)).digest("hex");
}

function samePhysicalState(
  current: {
    position: ScenarioPhysicalDynamicEntity["position"];
    rotation: ScenarioPhysicalDynamicEntity["rotation"];
    linear_velocity: ScenarioPhysicalDynamicEntity["linear_velocity"];
    angular_velocity: ScenarioPhysicalDynamicEntity["angular_velocity"];
  },
  expected: ScenarioPhysicalDynamicEntity
): boolean {
  return canonicalJson({
    position: current.position,
    rotation: current.rotation,
    linear_velocity: current.linear_velocity,
    angular_velocity: current.angular_velocity
  }) === canonicalJson({
    position: expected.position,
    rotation: expected.rotation,
    linear_velocity: expected.linear_velocity,
    angular_velocity: expected.angular_velocity
  });
}

function compareEntityIds(left: { id: string }, right: { id: string }): number {
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalJson(record[key])}`
  )).join(",")}}`;
}
