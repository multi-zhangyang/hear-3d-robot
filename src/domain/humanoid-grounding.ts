import { z } from "zod";
import { actionCommitPayloadSha256 } from "./action-commit-outbox.js";
import { JsonValueSchema, type JsonValue } from "./schema.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

const HumanoidGroundingObligationSchema = z.object({
  id: z.enum([
    "planning_plan",
    "world_authority",
    "skill_binding",
    "active_goal",
    "semantic_preconditions",
    "target_evidence",
    "interaction_evidence"
  ]),
  scope: z.enum(["plan", "world", "skill", "goal", "object"]),
  required: z.boolean(),
  status: z.enum(["satisfied", "failed", "not_applicable"]),
  code: z.string().trim().min(1),
  evidence_sha256: z.string().regex(SHA256_PATTERN),
  detail: JsonValueSchema
}).strict().superRefine((obligation, context) => {
  if (obligation.required && obligation.status === "not_applicable") {
    context.addIssue({
      code: "custom",
      path: ["status"],
      message: "A required grounding obligation cannot be inapplicable"
    });
  }
});

export type HumanoidGroundingObligation = z.infer<
  typeof HumanoidGroundingObligationSchema
>;

export const HumanoidGroundingReceiptSchema = z.object({
  protocol: z.literal("humanoid-grounding-receipt-v1"),
  receipt_id: z.string().trim().min(1),
  transaction_id: z.string().trim().min(1),
  planning_transaction_id: z.string().trim().min(1),
  plan_id: z.string().trim().min(1),
  call_id: z.string().trim().min(1).nullable(),
  world_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  authority_state_sha256: z.string().regex(SHA256_PATTERN),
  obligations: z.array(HumanoidGroundingObligationSchema).min(1),
  accepted: z.boolean(),
  failed_obligation_ids: z.array(
    HumanoidGroundingObligationSchema.shape.id
  ),
  receipt_sha256: z.string().regex(SHA256_PATTERN)
}).strict().superRefine((receipt, context) => {
  const ids = receipt.obligations.map(({ id }) => id);
  if (new Set(ids).size !== ids.length) {
    context.addIssue({
      code: "custom",
      path: ["obligations"],
      message: "Grounding obligation identifiers must be unique"
    });
  }
  const failed = receipt.obligations.filter(
    ({ required, status }) => required && status === "failed"
  ).map(({ id }) => id);
  if (receipt.accepted !== (failed.length === 0)
    || JSON.stringify(receipt.failed_obligation_ids) !== JSON.stringify(failed)) {
    context.addIssue({
      code: "custom",
      path: ["failed_obligation_ids"],
      message: "Grounding receipt decision does not match its obligations"
    });
  }
  if (receipt.receipt_sha256 !== humanoidGroundingReceiptSha256(receipt)) {
    context.addIssue({
      code: "custom",
      path: ["receipt_sha256"],
      message: "Grounding receipt integrity hash does not match"
    });
  }
});

export type HumanoidGroundingReceipt = z.infer<
  typeof HumanoidGroundingReceiptSchema
>;

export function createHumanoidGroundingObligation(input: Omit<
  HumanoidGroundingObligation,
  "evidence_sha256"
>): HumanoidGroundingObligation {
  const detail = json(input.detail);
  return HumanoidGroundingObligationSchema.parse({
    ...input,
    detail,
    evidence_sha256: actionCommitPayloadSha256(detail)
  });
}

export function createHumanoidGroundingReceipt(input: Omit<
  HumanoidGroundingReceipt,
  "accepted" | "failed_obligation_ids" | "receipt_sha256"
>): HumanoidGroundingReceipt {
  const obligations = input.obligations.map((obligation) => (
    HumanoidGroundingObligationSchema.parse(obligation)
  ));
  const failed = obligations.filter(
    ({ required, status }) => required && status === "failed"
  ).map(({ id }) => id);
  const withoutHash = {
    ...input,
    obligations,
    accepted: failed.length === 0,
    failed_obligation_ids: failed
  };
  return HumanoidGroundingReceiptSchema.parse({
    ...withoutHash,
    receipt_sha256: humanoidGroundingReceiptSha256(withoutHash)
  });
}

function humanoidGroundingReceiptSha256(receipt: unknown): string {
  const {
    receipt_sha256: _receiptSha256,
    ...payload
  } = receipt as HumanoidGroundingReceipt;
  return actionCommitPayloadSha256(json(payload));
}

function json(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) {
    throw new Error("Grounding evidence must be JSON-serializable");
  }
  return JsonValueSchema.parse(JSON.parse(serialized) as unknown);
}
