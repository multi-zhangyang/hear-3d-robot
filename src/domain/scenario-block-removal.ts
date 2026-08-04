import { createHash } from "node:crypto";
import { z } from "zod";
import type { Scenario } from "./schema.js";
import {
  applyScenarioChunkDeltaMutations,
  resolveScenarioChunkDeltaEntity,
  type ScenarioChunkDeltaMutation
} from "./scenario-chunk-delta.js";
import {
  restoreScenarioChunkDeltaState,
  type ScenarioBlockDelta,
  type ScenarioChunkDeltaState
} from "./scenario-chunk-delta-schema.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const REMOVAL_AUTHORITY_PROPERTY = "hear_block_removal_v1";

export const MINIMUM_BLOCK_REMOVAL_STABLE_FRAMES = 8 as const;
export const MINIMUM_BLOCK_REMOVAL_NORMAL_FORCE_N = 5 as const;

const BlockRemovalContactEvidenceSchema = z.object({
  predicate_index: z.number().int().min(0).max(15),
  predicate_type: z.enum(["body_contact_solid", "hand_contact_solid"]),
  surface_kind: z.enum(["body", "hand_surface"]),
  surface: z.string().trim().min(1),
  planned_stable_frames: z.number().int().min(1).max(500),
  observed_stable_frames: z.number().int().min(1).max(500),
  required_stable_frames: z.literal(MINIMUM_BLOCK_REMOVAL_STABLE_FRAMES),
  planned_minimum_normal_force_n: z.number().finite().positive(),
  observed_maximum_normal_force_n: z.number().finite().nonnegative(),
  required_minimum_normal_force_n: z.literal(MINIMUM_BLOCK_REMOVAL_NORMAL_FORCE_N)
}).strict().superRefine((evidence, context) => {
  if ((evidence.predicate_type === "body_contact_solid")
    !== (evidence.surface_kind === "body")) {
    context.addIssue({
      code: "custom",
      path: ["surface_kind"],
      message: "Block-removal contact surface does not match its predicate type"
    });
  }
  if (evidence.planned_stable_frames < evidence.required_stable_frames
    || evidence.observed_stable_frames < evidence.planned_stable_frames) {
    context.addIssue({
      code: "custom",
      path: ["observed_stable_frames"],
      message: "Block removal requires the Harness stability threshold to be physically met"
    });
  }
  if (evidence.observed_maximum_normal_force_n < Math.max(
    evidence.required_minimum_normal_force_n,
    evidence.planned_minimum_normal_force_n
  )) {
    context.addIssue({
      code: "custom",
      path: ["observed_maximum_normal_force_n"],
      message: "Block removal requires force-qualified physical contact"
    });
  }
});

export const ScenarioBlockRemovalTransactionSchema = z.object({
  version: z.literal(1),
  transaction_id: z.string().trim().min(1),
  solid_id: z.string().trim().min(1),
  block_id: z.string().trim().min(1),
  execution_transaction_id: z.string().trim().min(1),
  planning_transaction_id: z.string().trim().min(1),
  source_world_frame: z.number().int().nonnegative(),
  source_world_revision: z.number().int().nonnegative(),
  base_chunk_revision: z.number().int().nonnegative(),
  projected_chunk_revision: z.number().int().positive(),
  base_chunk_state_sha256: z.string().regex(SHA256_PATTERN),
  projected_chunk_state_sha256: z.string().regex(SHA256_PATTERN),
  expected_block_state_sha256: z.string().regex(SHA256_PATTERN),
  contact_evidence: BlockRemovalContactEvidenceSchema,
  transaction_sha256: z.string().regex(SHA256_PATTERN)
}).strict().superRefine((transaction, context) => {
  if (transaction.solid_id !== transaction.block_id) {
    context.addIssue({
      code: "custom",
      path: ["block_id"],
      message: "Only scenario blocks with an identical MuJoCo solid identity can be removed"
    });
  }
  if (transaction.projected_chunk_revision !== transaction.base_chunk_revision + 1) {
    context.addIssue({
      code: "custom",
      path: ["projected_chunk_revision"],
      message: "Block removal must advance exactly one atomic chunk revision"
    });
  }
  if (transaction.transaction_sha256
    !== scenarioBlockRemovalTransactionSha256(transaction)) {
    context.addIssue({
      code: "custom",
      path: ["transaction_sha256"],
      message: "Block-removal transaction integrity hash does not match"
    });
  }
});

export type ScenarioBlockRemovalTransaction = z.infer<
  typeof ScenarioBlockRemovalTransactionSchema
>;

export function createScenarioBlockRemovalTransaction(input: {
  scenario: Scenario;
  chunks: ScenarioChunkDeltaState;
  transactionId: string;
  solidId: string;
  executionTransactionId: string;
  planningTransactionId: string;
  sourceWorldFrame: number;
  sourceWorldRevision: number;
  contactEvidence: Omit<
    ScenarioBlockRemovalTransaction["contact_evidence"],
    "required_stable_frames" | "required_minimum_normal_force_n"
  >;
}): ScenarioBlockRemovalTransaction {
  const chunks = restoreScenarioChunkDeltaState(input.scenario, input.chunks);
  const block = requiredPresentBlock(input.scenario, chunks, input.solidId);
  const payload = {
    version: 1 as const,
    transaction_id: input.transactionId.trim(),
    solid_id: input.solidId.trim(),
    block_id: block.id,
    execution_transaction_id: input.executionTransactionId.trim(),
    planning_transaction_id: input.planningTransactionId.trim(),
    source_world_frame: input.sourceWorldFrame,
    source_world_revision: input.sourceWorldRevision,
    base_chunk_revision: chunks.revision,
    projected_chunk_revision: chunks.revision + 1,
    base_chunk_state_sha256: scenarioChunkDeltaStateSha256(chunks),
    expected_block_state_sha256: blockStateSha256(block),
    contact_evidence: {
      ...structuredClone(input.contactEvidence),
      required_stable_frames: 8 as const,
      required_minimum_normal_force_n: 5 as const
    }
  };
  const mutations = blockRemovalMutations(block, payload);
  const projected = applyScenarioChunkDeltaMutations(input.scenario, chunks, mutations);
  if (projected.revision !== payload.projected_chunk_revision) {
    throw new Error("Block removal did not advance its projected chunk revision");
  }
  const withProjection: Omit<
    ScenarioBlockRemovalTransaction,
    "transaction_sha256"
  > = {
    ...payload,
    projected_chunk_state_sha256: scenarioChunkDeltaStateSha256(projected)
  };
  return ScenarioBlockRemovalTransactionSchema.parse({
    ...withProjection,
    transaction_sha256: scenarioBlockRemovalTransactionSha256(withProjection)
  });
}

export function scenarioBlockRemovalMutations(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState,
  rawTransaction: ScenarioBlockRemovalTransaction
): ScenarioChunkDeltaMutation[] {
  const chunks = restoreScenarioChunkDeltaState(scenario, persisted);
  const transaction = ScenarioBlockRemovalTransactionSchema.parse(rawTransaction);
  const resolved = resolveScenarioChunkDeltaEntity(
    scenario,
    chunks,
    transaction.block_id
  );
  if (resolved.category !== "block") {
    throw new Error(`Block-removal target is not a scenario block: ${transaction.block_id}`);
  }
  if (!resolved.state.present) {
    if (sameRemovalAuthority(resolved.state, transaction)) return [];
    throw new Error(`Scenario block was removed by another transaction: ${transaction.block_id}`);
  }
  if (chunks.revision !== transaction.base_chunk_revision
    || scenarioChunkDeltaStateSha256(chunks) !== transaction.base_chunk_state_sha256) {
    throw new Error(
      `Block-removal base chunk state changed: ${transaction.block_id}`
    );
  }
  if (blockStateSha256(resolved.state) !== transaction.expected_block_state_sha256) {
    throw new Error(`Block-removal target state changed: ${transaction.block_id}`);
  }
  const mutations = blockRemovalMutations(resolved.state, transaction);
  const projected = applyScenarioChunkDeltaMutations(scenario, chunks, mutations);
  if (projected.revision !== transaction.projected_chunk_revision
    || scenarioChunkDeltaStateSha256(projected)
      !== transaction.projected_chunk_state_sha256) {
    throw new Error(`Block-removal projection changed: ${transaction.block_id}`);
  }
  return mutations;
}

export function projectScenarioBlockRemoval(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState,
  transaction: ScenarioBlockRemovalTransaction
): ScenarioChunkDeltaState {
  return applyScenarioChunkDeltaMutations(
    scenario,
    persisted,
    scenarioBlockRemovalMutations(scenario, persisted, transaction)
  );
}

export function assertScenarioBlockRemovalApplied(
  scenario: Scenario,
  persisted: ScenarioChunkDeltaState,
  rawTransaction: ScenarioBlockRemovalTransaction
): void {
  const chunks = restoreScenarioChunkDeltaState(scenario, persisted);
  const transaction = ScenarioBlockRemovalTransactionSchema.parse(rawTransaction);
  const resolved = resolveScenarioChunkDeltaEntity(
    scenario,
    chunks,
    transaction.block_id
  );
  if (resolved.category !== "block"
    || resolved.state.present
    || !sameRemovalAuthority(resolved.state, transaction)) {
    throw new Error(`Block-removal transaction is not durably applied: ${transaction.block_id}`);
  }
  if (chunks.revision === transaction.projected_chunk_revision
    && scenarioChunkDeltaStateSha256(chunks)
      !== transaction.projected_chunk_state_sha256) {
    throw new Error(`Applied block-removal state does not match its projection: ${transaction.block_id}`);
  }
}

export function scenarioBlockRemovalTransactionSha256(
  input: Omit<ScenarioBlockRemovalTransaction, "transaction_sha256">
    | ScenarioBlockRemovalTransaction
): string {
  const { transaction_sha256: _transactionSha256, ...payload } = input as (
    Partial<Pick<ScenarioBlockRemovalTransaction, "transaction_sha256">>
      & Omit<ScenarioBlockRemovalTransaction, "transaction_sha256">
  );
  return sha256(payload);
}

function scenarioChunkDeltaStateSha256(
  state: ScenarioChunkDeltaState
): string {
  return sha256(state);
}

function requiredPresentBlock(
  scenario: Scenario,
  chunks: ScenarioChunkDeltaState,
  blockId: string
): ScenarioBlockDelta {
  const resolved = resolveScenarioChunkDeltaEntity(scenario, chunks, blockId.trim());
  if (resolved.category !== "block") {
    throw new Error(`Scenario entity is not a removable block: ${blockId}`);
  }
  if (!resolved.state.present) {
    throw new Error(`Scenario block is already absent: ${blockId}`);
  }
  return resolved.state;
}

function blockRemovalMutations(
  block: ScenarioBlockDelta,
  transaction: Pick<
    ScenarioBlockRemovalTransaction,
    "transaction_id" | "execution_transaction_id" | "source_world_revision"
  >
): ScenarioChunkDeltaMutation[] {
  return [{
    type: "put_block",
    block: {
      id: block.id,
      center: structuredClone(block.center),
      size: structuredClone(block.size),
      material: block.material,
      properties: {
        ...structuredClone(block.properties),
        [REMOVAL_AUTHORITY_PROPERTY]: {
          version: 1,
          transaction_id: transaction.transaction_id,
          execution_transaction_id: transaction.execution_transaction_id,
          source_world_revision: transaction.source_world_revision
        }
      }
    }
  }, {
    type: "remove_block",
    entity_id: block.id
  }];
}

function sameRemovalAuthority(
  block: ScenarioBlockDelta,
  transaction: ScenarioBlockRemovalTransaction
): boolean {
  const authority = block.properties[REMOVAL_AUTHORITY_PROPERTY];
  if (!authority || typeof authority !== "object" || Array.isArray(authority)) return false;
  return authority.version === 1
    && authority.transaction_id === transaction.transaction_id
    && authority.execution_transaction_id === transaction.execution_transaction_id
    && authority.source_world_revision === transaction.source_world_revision;
}

function blockStateSha256(block: ScenarioBlockDelta): string {
  return sha256(block);
}

function sha256(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalJson(record[key])}`
  )).join(",")}}`;
}
