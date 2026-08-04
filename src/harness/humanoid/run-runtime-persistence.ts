import type { JsonValue } from "../../domain/schema.js";
import { actionCommitPayloadSha256 } from "../../domain/action-commit-outbox.js";
import type { ActionExecutionLedgerEntry } from "../../domain/action-execution-ledger.js";
import { PersistedHumanoidActionReceiptSchema } from "../../domain/humanoid-run.js";
import type { HumanoidWorld } from "../../world/humanoid/world.js";
import type { HumanoidActionReceipt } from "./runtime.js";

type HumanoidPersistenceCut = Awaited<
  ReturnType<HumanoidWorld["capturePersistenceState"]>
>;

export function requiresHumanoidClockPause(action: string): boolean {
  return action === "observe_humanoid"
    || action === "execute_whole_body_motion"
    || action === "execute_humanoid_navigation"
    || action === "remove_world_block";
}

export function object(value: JsonValue): Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : { value };
}

export function json(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) return null;
  return JSON.parse(serialized) as JsonValue;
}

export function physicalCheckpointSha256(cut: HumanoidPersistenceCut): string {
  return actionCommitPayloadSha256(json(cut.worldCheckpoint));
}

export function physicalExecutionCheckpointDue(
  entry: ActionExecutionLedgerEntry,
  cut: HumanoidPersistenceCut,
  interval: number
): boolean {
  if (!Number.isInteger(interval) || interval <= 0) {
    throw new Error("Physical execution checkpoint interval must be a positive integer");
  }
  const committedFrameCount = cut.world.worldRevision
    - entry.admission.world_revision;
  if (committedFrameCount <= 0) return false;
  if (committedFrameCount % interval === 0) return true;
  const planId = entry.admission.plan_id;
  return entry.action === "execute_whole_body_motion"
    ? cut.worldCheckpoint.motions.some((motion) => (
        motion.plan.id === planId && motion.terminal !== null
      ))
    : cut.worldCheckpoint.routes.some((route) => (
        route.id === planId && route.terminal !== null
      ));
}

export function previousCycleEvidence(value: JsonValue): Set<string> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return new Set();
  const ids = value.evidence_transaction_ids;
  if (!Array.isArray(ids)) return new Set();
  return new Set(ids.filter((entry): entry is string => typeof entry === "string"));
}

export function cycleSummary(value: JsonValue): string {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const summary = value.summary;
    if (typeof summary === "string" && summary.trim()) return summary.trim();
  }
  return "完成一次有物理回执的人形自主循环";
}

export function completedPhysicalExecution(receipt: HumanoidActionReceipt): boolean {
  return receipt.accepted && (
    receipt.action === "execute_whole_body_motion"
      ? receipt.code === "motion_option_succeeded"
      : receipt.action === "execute_humanoid_navigation"
        && receipt.code === "navigation_completed"
  );
}

export function physicalActuationReceipt(receipt: HumanoidActionReceipt): boolean {
  return receipt.frameCount > 0 && physicalExecutionReceipt(receipt);
}

export function physicalExecutionReceipt(receipt: HumanoidActionReceipt): boolean {
  return receipt.action === "execute_whole_body_motion"
    || receipt.action === "execute_humanoid_navigation";
}

export function assertPendingActionReceipt(
  value: JsonValue,
  expectedReceipt: HumanoidActionReceipt
): void {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Pending humanoid action commit contains a non-object record");
  }
  const { runtime_event_id: _runtimeEventId, ...rawReceipt } = value;
  const persisted = PersistedHumanoidActionReceiptSchema.parse(rawReceipt);
  const expected = PersistedHumanoidActionReceiptSchema.parse(expectedReceipt);
  if (JSON.stringify(persisted) !== JSON.stringify(expected)) {
    throw new Error(`Pending humanoid action commit conflicts with ${expected.transactionId}`);
  }
}

export function embodiedActionJournalReceipt(
  value: JsonValue
): HumanoidActionReceipt | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Humanoid action journal contains a non-object record");
  }
  if (value.action !== "execute_whole_body_motion"
    && value.action !== "execute_humanoid_navigation"
    && value.action !== "remove_world_block") return undefined;
  const { runtime_event_id: _runtimeEventId, ...receipt } = value;
  return PersistedHumanoidActionReceiptSchema.parse(receipt);
}
