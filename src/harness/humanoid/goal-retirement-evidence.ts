import type { GoalEvidenceArtifact } from "./goal-evidence.js";

export interface RecoverableGoalEvidence {
  evidenceRef: string;
  transactionId: string;
  code: "manipulation_base_placement_required" | "planning_attempt_recoverable";
  receiptCode: string;
  recovery: string;
  reachableBasePlacementCount?: number;
}

export type RecoverableGoalActionReceipt = Omit<
  RecoverableGoalEvidence,
  "evidenceRef"
>;

export function recoverableBlockedGoalEvidence(
  artifacts: readonly GoalEvidenceArtifact[]
): RecoverableGoalEvidence | null {
  for (const artifact of artifacts) {
    if (artifact.evidence.kind !== "action_receipt") continue;
    const payload = record(artifact.payload);
    const recoverable = recoverableBlockedGoalActionReceipt(payload?.receipt);
    if (recoverable) return { evidenceRef: artifact.evidence.ref, ...recoverable };
  }
  return null;
}

export function recoverableBlockedGoalActionReceipt(
  value: unknown
): RecoverableGoalActionReceipt | null {
  const receipt = record(value);
  const detail = record(receipt?.detail);
  const placements = detail?.reachable_base_placements;
  if (receipt?.accepted === false
    && receipt.code === "manipulation_base_placement_required"
    && Array.isArray(placements)
    && placements.length > 0
    && typeof receipt.transactionId === "string") {
    return {
      transactionId: receipt.transactionId,
      code: "manipulation_base_placement_required",
      receiptCode: receipt.code,
      recovery: `${placements.length} IK-validated base placement(s) remain`,
      reachableBasePlacementCount: placements.length
    };
  }
  if (receipt?.accepted === false
    && receipt.frameCount === 0
    && typeof receipt.transactionId === "string"
    && typeof receipt.code === "string"
    && (typeof receipt.action === "string" && receipt.action.startsWith("plan_")
      || receipt.code === "plan_revalidation_failed"
      || receipt.code === "repeated_planning_failure")) {
    return {
      transactionId: receipt.transactionId,
      code: "planning_attempt_recoverable",
      receiptCode: receipt.code,
      recovery: "the receipt rejected one zero-frame planning attempt, not the active Goal"
    };
  }
  return null;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}
