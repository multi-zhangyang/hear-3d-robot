import type { Goal, HumanoidRunCheckpoint } from "./types";

export interface GoalSelectionPhase {
  kind: "candidate_generation" | "candidate_selection";
  candidateCount: number;
}

export function activeCheckpointGoal(checkpoint: HumanoidRunCheckpoint): Goal | null {
  if (checkpoint.version === 4 || checkpoint.version === 5) return checkpoint.goal ?? null;
  const dag = checkpoint.goal_dag;
  if (!dag || dag.status !== "active" || !dag.current_epoch_id) return null;
  const epoch = dag.epochs.find((candidate) => candidate.epoch_id === dag.current_epoch_id);
  if (!epoch || epoch.status !== "active") return null;
  const candidate = dag.candidates[epoch.candidate_id];
  return candidate?.status === "active" ? candidate.goal : null;
}

export function missionCheckpointGoal(checkpoint: HumanoidRunCheckpoint): Goal | null {
  return checkpoint.version === 6
    ? checkpoint.mission_goal ?? null
    : checkpoint.goal ?? null;
}

export function goalSelectionPhase(
  checkpoint: HumanoidRunCheckpoint
): GoalSelectionPhase | null {
  if (checkpoint.version !== 6
    || checkpoint.goal_dag?.status !== "awaiting_model_selection") return null;
  const candidateCount = Object.values(checkpoint.goal_dag.candidates)
    .filter((candidate) => candidate.status === "proposed").length;
  return candidateCount > 0
    ? { kind: "candidate_selection", candidateCount }
    : { kind: "candidate_generation", candidateCount: 0 };
}

export function goalSelectionLabel(checkpoint: HumanoidRunCheckpoint): string | null {
  const phase = goalSelectionPhase(checkpoint);
  if (!phase) return null;
  return phase.kind === "candidate_selection"
    ? `目标管理智能体正在选择 · ${phase.candidateCount} 个候选`
    : "目标管理智能体正在生成候选";
}
