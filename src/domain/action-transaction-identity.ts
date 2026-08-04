import { z } from "zod";
import { HUMANOID_ACTION_NAMES } from "./humanoid-action.js";
import {
  actionCommitPayloadSha256,
  actionCommitReceipt,
  actionCommitReceiptSha256,
  type PendingActionCommit
} from "./action-commit-outbox.js";
import type { JsonValue } from "./schema.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const HumanoidActionNameSchema = z.enum(HUMANOID_ACTION_NAMES);

export const ActionTransactionIdentitySchema = z.object({
  version: z.literal(1),
  run_id: z.string().trim().min(1),
  transaction_id: z.string().trim().min(1),
  agent_id: z.string().trim().min(1),
  action: HumanoidActionNameSchema,
  action_fingerprint_sha256: z.string().regex(SHA256_PATTERN),
  receipt_sha256: z.string().regex(SHA256_PATTERN),
  action_record_sha256: z.string().regex(SHA256_PATTERN),
  runtime_event_id: z.string().trim().min(1),
  runtime_event_sha256: z.string().regex(SHA256_PATTERN),
  goal_evidence_ref: z.string().trim().min(1),
  goal_evidence_sha256: z.string().regex(SHA256_PATTERN),
  committed_at: z.string().datetime(),
  identity_sha256: z.string().regex(SHA256_PATTERN)
}).strict().superRefine((identity, context) => {
  if (identity.identity_sha256 !== actionTransactionIdentitySha256(identity)) {
    context.addIssue({
      code: "custom",
      path: ["identity_sha256"],
      message: "Action transaction identity integrity hash does not match"
    });
  }
});

export type ActionTransactionIdentity = z.infer<
  typeof ActionTransactionIdentitySchema
>;

export function createActionTransactionIdentity(
  commit: PendingActionCommit
): ActionTransactionIdentity {
  const receipt = jsonObject(actionCommitReceipt(commit.action_record));
  const receiptSha256 = actionCommitReceiptSha256(commit.action_record);
  if (!receipt || !receiptSha256
    || typeof receipt.agentId !== "string"
    || typeof receipt.action !== "string"
    || typeof receipt.fingerprint !== "string"
    || typeof receipt.committedAt !== "string") {
    throw new Error(`Action commit has no canonical receipt: ${commit.transaction_id}`);
  }
  const payload = {
    version: 1 as const,
    run_id: commit.runtime_event.run_id,
    transaction_id: commit.transaction_id,
    agent_id: receipt.agentId,
    action: HumanoidActionNameSchema.parse(receipt.action),
    action_fingerprint_sha256: actionTransactionFingerprintSha256(
      receipt.fingerprint
    ),
    receipt_sha256: receiptSha256,
    action_record_sha256: commit.action_record_sha256,
    runtime_event_id: commit.runtime_event_id,
    runtime_event_sha256: commit.runtime_event_sha256,
    goal_evidence_ref: commit.goal_evidence_ref,
    goal_evidence_sha256: commit.goal_evidence_sha256,
    committed_at: receipt.committedAt
  };
  return ActionTransactionIdentitySchema.parse({
    ...payload,
    identity_sha256: actionTransactionIdentitySha256(payload)
  });
}

export function rebuildActionTransactionIdentities(
  records: readonly JsonValue[],
  runId: string
): Map<string, ActionTransactionIdentity> {
  const identities = new Map<string, ActionTransactionIdentity>();
  for (const record of records) {
    const identity = ActionTransactionIdentitySchema.parse(record);
    if (identity.run_id !== runId) {
      throw new Error(
        `Action transaction identity belongs to another run: ${identity.transaction_id}`
      );
    }
    if (identities.has(identity.transaction_id)) {
      throw new Error(
        `Duplicate durable action transaction identity: ${identity.transaction_id}`
      );
    }
    identities.set(identity.transaction_id, identity);
  }
  return identities;
}

export function actionTransactionFingerprintSha256(fingerprint: string): string {
  return actionCommitPayloadSha256(fingerprint);
}

function actionTransactionIdentitySha256(input: Omit<
  ActionTransactionIdentity,
  "identity_sha256"
> | ActionTransactionIdentity): string {
  const { identity_sha256: _identitySha256, ...payload } = input as (
    Partial<Pick<ActionTransactionIdentity, "identity_sha256">>
    & Omit<ActionTransactionIdentity, "identity_sha256">
  );
  return actionCommitPayloadSha256(json(payload));
}

function json(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) {
    throw new Error("Action transaction identity is not serializable");
  }
  return JSON.parse(serialized) as JsonValue;
}

function jsonObject(value: JsonValue | undefined): Record<string, JsonValue> | null {
  return value !== undefined
    && value !== null
    && typeof value === "object"
    && !Array.isArray(value)
    ? value
    : null;
}
