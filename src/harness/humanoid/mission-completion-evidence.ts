import {
  goalConstraintSha256,
  goalSha256
} from "../../domain/goal-identity.js";
import {
  HumanoidCheckerResultSchema,
  type HumanoidCheckerResult,
  type HumanoidRunCheckpoint
} from "../../domain/humanoid-run.js";
import {
  GoalEvidenceArtifactSchema,
  type GoalEvidenceArtifact
} from "./goal-evidence.js";

export interface HumanoidMissionCompletionEvidence {
  epoch_id: string;
  candidate_id: string;
  goal_evaluation_ref: string;
  world_frame: number;
  world_revision: number;
  checker: HumanoidCheckerResult;
}

export function resolveHumanoidMissionCompletion(
  checkpoint: Pick<HumanoidRunCheckpoint, "mission_goal" | "goal_dag">,
  rawArtifacts: Iterable<unknown>
): HumanoidMissionCompletionEvidence | null {
  const artifacts = indexedArtifacts(rawArtifacts);
  const missionConstraintSha256 = goalConstraintSha256(checkpoint.mission_goal);
  for (let index = checkpoint.goal_dag.epochs.length - 1; index >= 0; index -= 1) {
    const epoch = checkpoint.goal_dag.epochs[index]!;
    const candidate = checkpoint.goal_dag.candidates[epoch.candidate_id];
    const candidateGoalSha256 = candidate
      ? goalSha256(candidate.goal)
      : undefined;
    if (epoch.status !== "completed"
      || !candidate
      || candidate.status !== "completed"
      || candidate.content_sha256 !== candidateGoalSha256
      || goalConstraintSha256(candidate.goal) !== missionConstraintSha256) {
      continue;
    }
    for (const ref of epoch.physical_evidence_refs.resolution) {
      const artifact = artifacts.get(ref);
      if (!artifact || artifact.evidence.kind !== "goal_evaluation") continue;
      const payload = record(artifact.payload);
      const checker = HumanoidCheckerResultSchema.parse(payload?.evaluation);
      if (artifact.evidence.goal_content_sha256 !== candidateGoalSha256
        || artifact.evidence.world_frame !== checker.worldFrame
        || artifact.evidence.world_revision !== checker.worldRevision
        || epoch.resolved_world_revision !== checker.worldRevision
        || payload?.epoch_id !== epoch.epoch_id
        || payload.goal_content_sha256 !== candidateGoalSha256
        || checker.success !== true
        || goalSha256(checker.goal) !== candidateGoalSha256
        || goalConstraintSha256(checker.goal) !== missionConstraintSha256) {
        throw new Error(
          `Mission Goal completion evidence is inconsistent: ${artifact.evidence.ref}`
        );
      }
      const registered = checkpoint.goal_dag.evidence[ref];
      if (!registered || JSON.stringify(registered) !== JSON.stringify(artifact.evidence)) {
        throw new Error(
          `Mission Goal completion evidence is not registered by the Goal DAG: ${ref}`
        );
      }
      return {
        epoch_id: epoch.epoch_id,
        candidate_id: candidate.candidate_id,
        goal_evaluation_ref: ref,
        world_frame: checker.worldFrame,
        world_revision: checker.worldRevision,
        checker: structuredClone(checker)
      };
    }
    throw new Error(
      `Completed Mission Goal epoch has no durable successful evaluation: ${epoch.epoch_id}`
    );
  }
  return null;
}

function indexedArtifacts(
  rawArtifacts: Iterable<unknown>
): Map<string, GoalEvidenceArtifact> {
  const artifacts = new Map<string, GoalEvidenceArtifact>();
  for (const raw of rawArtifacts) {
    const artifact = GoalEvidenceArtifactSchema.parse(raw);
    const ref = artifact.evidence.ref;
    const existing = artifacts.get(ref);
    if (existing && JSON.stringify(existing) !== JSON.stringify(artifact)) {
      throw new Error(`Goal evidence reference was rebound: ${ref}`);
    }
    if (!existing) artifacts.set(ref, artifact);
  }
  return artifacts;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}
