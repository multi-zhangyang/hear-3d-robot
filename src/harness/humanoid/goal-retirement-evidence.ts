import type { GoalEvidenceArtifact } from "./goal-evidence.js";

export interface RecoverableGoalEvidence {
  evidenceRef: string;
  transactionId: string;
  code: "manipulation_base_placement_required";
  reachableBasePlacementCount: number;
}

export function recoverableBlockedGoalEvidence(
  artifacts: readonly GoalEvidenceArtifact[]
): RecoverableGoalEvidence | null {
  for (const artifact of artifacts) {
    if (artifact.evidence.kind !== "action_receipt") continue;
    const payload = record(artifact.payload);
    const receipt = record(payload?.receipt);
    const detail = record(receipt?.detail);
    const placements = detail?.reachable_base_placements;
    if (receipt?.accepted === false
      && receipt.code === "manipulation_base_placement_required"
      && Array.isArray(placements)
      && placements.length > 0
      && typeof receipt.transactionId === "string") {
      return {
        evidenceRef: artifact.evidence.ref,
        transactionId: receipt.transactionId,
        code: "manipulation_base_placement_required",
        reachableBasePlacementCount: placements.length
      };
    }
  }
  return null;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}
