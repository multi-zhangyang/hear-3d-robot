import { createHash } from "node:crypto";
import { z } from "zod";
import { JsonValueSchema, type JsonValue } from "./schema.js";
import {
  ScenarioPhysicalWorldDeltaSchema,
  scenarioPhysicalWorldDeltaSha256,
  type ScenarioPhysicalWorldDelta
} from "./scenario-physical-delta.js";
import {
  ScenarioBlockRemovalTransactionSchema,
  scenarioBlockRemovalTransactionSha256,
  type ScenarioBlockRemovalTransaction
} from "./scenario-block-removal.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

const PendingActionCommitSchema = z.object({
  transaction_id: z.string().trim().min(1),
  runtime_event_id: z.string().trim().min(1),
  action_record: JsonValueSchema,
  action_record_sha256: z.string().regex(SHA256_PATTERN),
  goal_evidence_ref: z.string().trim().min(1),
  goal_evidence_record: JsonValueSchema,
  goal_evidence_sha256: z.string().regex(SHA256_PATTERN),
  experience_ref: z.string().trim().min(1).optional(),
  experience_record: JsonValueSchema.optional(),
  experience_sha256: z.string().regex(SHA256_PATTERN).optional(),
  physical_world_delta: ScenarioPhysicalWorldDeltaSchema.optional(),
  physical_world_delta_sha256: z.string().regex(SHA256_PATTERN).optional(),
  block_removal: ScenarioBlockRemovalTransactionSchema.optional(),
  block_removal_sha256: z.string().regex(SHA256_PATTERN).optional(),
  runtime_event: z.object({
    event_id: z.string().trim().min(1),
    run_id: z.string().trim().min(1),
    type: z.literal("humanoid_action_committed"),
    at: z.string().datetime(),
    data: JsonValueSchema
  }).strict(),
  runtime_event_sha256: z.string().regex(SHA256_PATTERN),
  staged_at: z.string().datetime()
}).strict().superRefine((entry, context) => {
  if (entry.runtime_event.event_id !== entry.runtime_event_id) {
    context.addIssue({
      code: "custom",
      path: ["runtime_event", "event_id"],
      message: "Action outbox event identity does not match its entry"
    });
  }
  const action = jsonObject(entry.action_record);
  if (action?.transactionId !== entry.transaction_id) {
    context.addIssue({
      code: "custom",
      path: ["action_record", "transactionId"],
      message: "Action outbox transaction identity does not match its record"
    });
  }
  if (action?.runtime_event_id !== entry.runtime_event_id) {
    context.addIssue({
      code: "custom",
      path: ["action_record", "runtime_event_id"],
      message: "Action outbox event identity does not match its action record"
    });
  }
  const evidence = jsonObject(entry.goal_evidence_record);
  const evidenceIdentity = jsonObject(evidence?.evidence ?? null);
  if (evidenceIdentity?.ref !== entry.goal_evidence_ref) {
    context.addIssue({
      code: "custom",
      path: ["goal_evidence_record", "evidence", "ref"],
      message: "Action outbox Goal evidence identity does not match its entry"
    });
  }
  if (evidenceIdentity?.kind !== "action_receipt"
    || entry.goal_evidence_ref !== `action:${entry.transaction_id}`) {
    context.addIssue({
      code: "custom",
      path: ["goal_evidence_record", "evidence"],
      message: "Action outbox Goal evidence is not the transaction's action receipt"
    });
  }
  const evidencePayload = jsonObject(evidence?.payload ?? null);
  if (evidencePayload?.transaction_id !== entry.transaction_id) {
    context.addIssue({
      code: "custom",
      path: ["goal_evidence_record", "payload", "transaction_id"],
      message: "Action outbox Goal evidence transaction does not match its entry"
    });
  }
  const receipt = actionCommitReceipt(entry.action_record);
  if (!receipt || evidencePayload?.receipt === undefined
    || !sameActionCommitPayload(evidencePayload.receipt, receipt)) {
    context.addIssue({
      code: "custom",
      path: ["goal_evidence_record", "payload", "receipt"],
      message: "Action outbox Goal evidence receipt does not match its action record"
    });
  }
  const experienceFields = [
    entry.experience_ref,
    entry.experience_record,
    entry.experience_sha256
  ].filter((value) => value !== undefined).length;
  if (experienceFields !== 0 && experienceFields !== 3) {
    context.addIssue({
      code: "custom",
      path: ["experience_record"],
      message: "Action outbox experience identity, record and hash must appear together"
    });
  }
  if (entry.experience_record !== undefined) {
    const experience = jsonObject(entry.experience_record);
    if (entry.experience_ref !== `action:${entry.transaction_id}`
      || experience?.source_ref !== entry.experience_ref
      || experience?.transaction_id !== entry.transaction_id) {
      context.addIssue({
        code: "custom",
        path: ["experience_record"],
        message: "Action outbox embodied experience does not match its transaction"
      });
    }
    if (entry.experience_sha256
      !== actionCommitPayloadSha256(entry.experience_record)) {
      context.addIssue({
        code: "custom",
        path: ["experience_sha256"],
        message: "Action outbox embodied experience integrity hash does not match"
      });
    }
  }
  const physicalDeltaFields = [
    entry.physical_world_delta,
    entry.physical_world_delta_sha256
  ].filter((value) => value !== undefined).length;
  if (physicalDeltaFields !== 0 && physicalDeltaFields !== 2) {
    context.addIssue({
      code: "custom",
      path: ["physical_world_delta"],
      message: "Action outbox physical world delta and hash must appear together"
    });
  }
  if (entry.physical_world_delta !== undefined) {
    if (entry.physical_world_delta.transaction_id !== entry.transaction_id
      || (action?.action !== "execute_whole_body_motion"
        && action?.action !== "execute_humanoid_navigation")
      || entry.physical_world_delta.source_world_revision !== action?.worldAfterRevision) {
      context.addIssue({
        code: "custom",
        path: ["physical_world_delta"],
        message: "Action outbox physical world delta does not match its execution receipt"
      });
    }
    if (entry.physical_world_delta_sha256
      !== scenarioPhysicalWorldDeltaSha256(entry.physical_world_delta)) {
      context.addIssue({
        code: "custom",
        path: ["physical_world_delta_sha256"],
        message: "Action outbox physical world delta integrity hash does not match"
      });
    }
  }
  const blockRemovalFields = [
    entry.block_removal,
    entry.block_removal_sha256
  ].filter((value) => value !== undefined).length;
  if (blockRemovalFields !== 0 && blockRemovalFields !== 2) {
    context.addIssue({
      code: "custom",
      path: ["block_removal"],
      message: "Action outbox block removal and hash must appear together"
    });
  }
  if (entry.block_removal !== undefined) {
    if (entry.block_removal.transaction_id !== entry.transaction_id
      || action?.action !== "remove_world_block"
      || action?.accepted !== true) {
      context.addIssue({
        code: "custom",
        path: ["block_removal"],
        message: "Action outbox block removal does not match its accepted action receipt"
      });
    }
    const actionDetail = jsonObject(action?.detail ?? null);
    if (actionDetail?.removal_transaction === undefined
      || !sameActionCommitPayload(
        actionDetail.removal_transaction,
        entry.block_removal
      )) {
      context.addIssue({
        code: "custom",
        path: ["action_record", "detail", "removal_transaction"],
        message: "Action outbox block removal is not bound to its receipt"
      });
    }
    if (entry.block_removal_sha256
      !== scenarioBlockRemovalTransactionSha256(entry.block_removal)) {
      context.addIssue({
        code: "custom",
        path: ["block_removal_sha256"],
        message: "Action outbox block-removal integrity hash does not match"
      });
    }
  }
  const eventData = jsonObject(entry.runtime_event.data);
  if (eventData?.receipt === undefined
    || !sameActionCommitPayload(eventData.receipt, entry.action_record)) {
    context.addIssue({
      code: "custom",
      path: ["runtime_event", "data", "receipt"],
      message: "Action outbox runtime event receipt does not match its action record"
    });
  }
  if (entry.physical_world_delta !== undefined) {
    const eventDelta = eventData?.physical_world_delta;
    if (eventDelta === undefined
      || actionCommitPayloadSha256(eventDelta)
        !== actionCommitPayloadSha256(entry.physical_world_delta)) {
      context.addIssue({
        code: "custom",
        path: ["runtime_event", "data", "physical_world_delta"],
        message: "Action outbox event does not contain its physical world delta"
      });
    }
  }
  if (entry.block_removal !== undefined) {
    const eventRemoval = eventData?.block_removal;
    if (eventRemoval === undefined
      || actionCommitPayloadSha256(eventRemoval)
        !== actionCommitPayloadSha256(entry.block_removal)) {
      context.addIssue({
        code: "custom",
        path: ["runtime_event", "data", "block_removal"],
        message: "Action outbox event does not contain its block removal"
      });
    }
  }
  if (entry.action_record_sha256 !== actionCommitPayloadSha256(entry.action_record)) {
    context.addIssue({
      code: "custom",
      path: ["action_record_sha256"],
      message: "Action outbox record integrity hash does not match"
    });
  }
  if (entry.runtime_event_sha256 !== actionCommitPayloadSha256(entry.runtime_event)) {
    context.addIssue({
      code: "custom",
      path: ["runtime_event_sha256"],
      message: "Action outbox event integrity hash does not match"
    });
  }
  if (entry.goal_evidence_sha256 !== actionCommitPayloadSha256(entry.goal_evidence_record)) {
    context.addIssue({
      code: "custom",
      path: ["goal_evidence_sha256"],
      message: "Action outbox Goal evidence integrity hash does not match"
    });
  }
});

export const ActionCommitOutboxSchema = z.object({
  version: z.literal(1),
  pending: z.record(z.string().trim().min(1), PendingActionCommitSchema)
}).strict().superRefine((outbox, context) => {
  for (const [transactionId, entry] of Object.entries(outbox.pending)) {
    if (transactionId !== entry.transaction_id) {
      context.addIssue({
        code: "custom",
        path: ["pending", transactionId],
        message: "Action outbox key does not match its transaction identity"
      });
    }
  }
});

export type PendingActionCommit = z.infer<typeof PendingActionCommitSchema>;
export type ActionCommitOutbox = z.infer<typeof ActionCommitOutboxSchema>;

export const EmptyActionCommitOutbox: ActionCommitOutbox = {
  version: 1,
  pending: {}
};

export function stageActionCommit(
  persisted: ActionCommitOutbox,
  input: {
    transactionId: string;
    runtimeEventId: string;
    actionRecord: JsonValue;
    goalEvidenceRef: string;
    goalEvidenceRecord: JsonValue;
    experienceRef?: string;
    experienceRecord?: JsonValue;
    physicalWorldDelta?: ScenarioPhysicalWorldDelta;
    blockRemoval?: ScenarioBlockRemovalTransaction;
    runtimeEvent: PendingActionCommit["runtime_event"];
    stagedAt?: string;
  }
): ActionCommitOutbox {
  const outbox = restoreActionCommitOutbox(persisted);
  if ((input.experienceRef === undefined)
    !== (input.experienceRecord === undefined)) {
    throw new Error("Action outbox experience identity and record must appear together");
  }
  const transactionId = input.transactionId.trim();
  const runtimeEventId = input.runtimeEventId.trim();
  const entry = PendingActionCommitSchema.parse({
    transaction_id: transactionId,
    runtime_event_id: runtimeEventId,
    action_record: structuredClone(input.actionRecord),
    action_record_sha256: actionCommitPayloadSha256(input.actionRecord),
    goal_evidence_ref: input.goalEvidenceRef,
    goal_evidence_record: structuredClone(input.goalEvidenceRecord),
    goal_evidence_sha256: actionCommitPayloadSha256(input.goalEvidenceRecord),
    ...(input.experienceRef !== undefined && input.experienceRecord !== undefined
      ? {
          experience_ref: input.experienceRef,
          experience_record: structuredClone(input.experienceRecord),
          experience_sha256: actionCommitPayloadSha256(input.experienceRecord)
        }
      : {}),
    ...(input.physicalWorldDelta
      ? {
          physical_world_delta: structuredClone(input.physicalWorldDelta),
          physical_world_delta_sha256: scenarioPhysicalWorldDeltaSha256(
            input.physicalWorldDelta
          )
        }
      : {}),
    ...(input.blockRemoval
      ? {
          block_removal: structuredClone(input.blockRemoval),
          block_removal_sha256: scenarioBlockRemovalTransactionSha256(
            input.blockRemoval
          )
        }
      : {}),
    runtime_event: structuredClone(input.runtimeEvent),
    runtime_event_sha256: actionCommitPayloadSha256(input.runtimeEvent),
    staged_at: input.stagedAt ?? new Date().toISOString()
  });
  const existing = outbox.pending[transactionId];
  if (existing) {
    if (existing.action_record_sha256 !== entry.action_record_sha256
      || existing.runtime_event_sha256 !== entry.runtime_event_sha256
      || existing.goal_evidence_sha256 !== entry.goal_evidence_sha256
      || existing.goal_evidence_ref !== entry.goal_evidence_ref
      || existing.experience_ref !== entry.experience_ref
      || existing.experience_sha256 !== entry.experience_sha256
      || existing.physical_world_delta_sha256 !== entry.physical_world_delta_sha256
      || existing.block_removal_sha256 !== entry.block_removal_sha256
      || existing.runtime_event_id !== entry.runtime_event_id) {
      throw new Error(`Action outbox transaction conflict: ${transactionId}`);
    }
    return outbox;
  }
  return ActionCommitOutboxSchema.parse({
    ...outbox,
    pending: {
      ...outbox.pending,
      [transactionId]: entry
    }
  });
}

export function acknowledgeActionCommit(
  persisted: ActionCommitOutbox,
  transactionId: string
): ActionCommitOutbox {
  const outbox = restoreActionCommitOutbox(persisted);
  const normalized = transactionId.trim();
  if (!outbox.pending[normalized]) return outbox;
  const pending = { ...outbox.pending };
  delete pending[normalized];
  return ActionCommitOutboxSchema.parse({ ...outbox, pending });
}

export function restoreActionCommitOutbox(persisted: unknown): ActionCommitOutbox {
  return ActionCommitOutboxSchema.parse(persisted);
}

export function pendingActionCommits(
  persisted: ActionCommitOutbox
): PendingActionCommit[] {
  const outbox = restoreActionCommitOutbox(persisted);
  return Object.values(outbox.pending)
    .sort((left, right) => left.staged_at.localeCompare(right.staged_at))
    .map((entry) => structuredClone(entry));
}

export function actionCommitPayloadSha256(value: JsonValue): string {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

/** Returns the canonical receipt shared by the action journal and Goal evidence. */
export function actionCommitReceipt(value: JsonValue): JsonValue | undefined {
  const action = jsonObject(value);
  if (!action || typeof action.runtime_event_id !== "string") return undefined;
  const { runtime_event_id: _runtimeEventId, ...receipt } = action;
  return structuredClone(receipt);
}

export function actionCommitReceiptSha256(value: JsonValue): string | undefined {
  const receipt = actionCommitReceipt(value);
  return receipt === undefined ? undefined : actionCommitPayloadSha256(receipt);
}

function canonicalJson(value: JsonValue): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}

function jsonObject(value: JsonValue): Record<string, JsonValue> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}

function sameActionCommitPayload(left: JsonValue, right: JsonValue): boolean {
  return actionCommitPayloadSha256(left) === actionCommitPayloadSha256(right);
}
