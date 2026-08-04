import { createHash } from "node:crypto";
import { z } from "zod";
import { JsonValueSchema, type JsonValue } from "../../domain/schema.js";
import type { HumanoidExecutionReceipt } from "./world-contract.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

const HumanoidExecutionCodeSchema = z.enum([
  "motion_completed",
  "navigation_completed",
  "plan_stale",
  "motion_failed",
  "navigation_blocked",
  "motion_option_succeeded",
  "motion_goal_unmet",
  "motion_goal_uncertain",
  "motion_execution_drifted",
  "motion_constraint_violated",
  "plan_revalidation_failed"
]);

export const HumanoidPlanTerminalSchema = z.object({
  version: z.literal(1),
  plan_id: z.string().trim().min(1),
  accepted: z.boolean(),
  code: HumanoidExecutionCodeSchema,
  total_frames: z.number().int().nonnegative(),
  final_frame: z.number().int().nonnegative(),
  final_world_revision: z.number().int().nonnegative(),
  detail: JsonValueSchema,
  result_sha256: z.string().regex(SHA256_PATTERN)
}).strict().superRefine((terminal, context) => {
  if (terminal.result_sha256 !== humanoidPlanTerminalSha256(terminal)) {
    context.addIssue({
      code: "custom",
      path: ["result_sha256"],
      message: "Humanoid plan terminal result integrity hash does not match"
    });
  }
});

export type HumanoidPlanTerminal = z.infer<typeof HumanoidPlanTerminalSchema>;

export function createHumanoidPlanTerminal(input: {
  planId: string;
  totalFrames: number;
  receipt: HumanoidExecutionReceipt;
}): HumanoidPlanTerminal {
  const payload = {
    version: 1 as const,
    plan_id: input.planId,
    accepted: input.receipt.accepted,
    code: input.receipt.code,
    total_frames: input.totalFrames,
    final_frame: input.receipt.finalSnapshot.frame,
    final_world_revision: input.receipt.finalSnapshot.worldRevision,
    detail: json(input.receipt.detail)
  };
  return HumanoidPlanTerminalSchema.parse({
    ...payload,
    result_sha256: humanoidPlanTerminalSha256(payload)
  });
}

export function humanoidPlanTerminalReceipt(
  persisted: HumanoidPlanTerminal,
  finalSnapshot: HumanoidExecutionReceipt["finalSnapshot"]
): HumanoidExecutionReceipt {
  const terminal = HumanoidPlanTerminalSchema.parse(persisted);
  if (finalSnapshot.frame !== terminal.final_frame
    || finalSnapshot.worldRevision !== terminal.final_world_revision) {
    throw new Error(`Humanoid terminal plan state changed before acknowledgement: ${terminal.plan_id}`);
  }
  return {
    accepted: terminal.accepted,
    code: terminal.code,
    frames: 0,
    finalSnapshot: structuredClone(finalSnapshot),
    detail: structuredClone(terminal.detail) as HumanoidExecutionReceipt["detail"],
    terminalResultSha256: terminal.result_sha256
  };
}

function humanoidPlanTerminalSha256(input: {
  version: 1;
  plan_id: string;
  accepted: boolean;
  code: HumanoidExecutionReceipt["code"];
  total_frames: number;
  final_frame: number;
  final_world_revision: number;
  detail: JsonValue;
}): string {
  return createHash("sha256").update(canonicalJson(json({
    version: input.version,
    plan_id: input.plan_id,
    accepted: input.accepted,
    code: input.code,
    total_frames: input.total_frames,
    final_frame: input.final_frame,
    final_world_revision: input.final_world_revision,
    detail: input.detail
  }))).digest("hex");
}

function json(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) throw new Error("Humanoid terminal result is not serializable");
  return JSON.parse(serialized) as JsonValue;
}

function canonicalJson(value: JsonValue): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalJson(value[key]!)}`
  )).join(",")}}`;
}
