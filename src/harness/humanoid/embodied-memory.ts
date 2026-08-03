import {
  HumanoidEmbodiedEpisodeSchema,
  HumanoidEmbodiedMemoryStateSchema,
  type HumanoidEmbodiedEpisode,
  type HumanoidEmbodiedMemoryState
} from "../../domain/humanoid-run.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { HumanoidActionReceipt } from "./runtime.js";

export const MAX_RECENT_EMBODIED_EPISODES = 64;
const CONTEXT_EMBODIED_EPISODES = 12;
export const MAX_CHECKPOINT_ACTION_RECEIPTS = 32;

export function appendEmbodiedEpisode(input: {
  state: HumanoidEmbodiedMemoryState;
  sequence: number;
  execution: HumanoidActionReceipt;
  modelSummary: string;
  world: HumanoidWorldSnapshot;
  goalSuccess: boolean;
}): {
  state: HumanoidEmbodiedMemoryState;
  episode: HumanoidEmbodiedEpisode;
} {
  const state = HumanoidEmbodiedMemoryStateSchema.parse(input.state);
  if (!input.execution.accepted
    || (input.execution.action !== "execute_whole_body_motion"
      && input.execution.action !== "execute_humanoid_navigation")
    || input.execution.worldAfterRevision !== input.world.worldRevision) {
    throw new Error("Embodied memory requires current accepted execution evidence");
  }
  const executionDetail = jsonRecord(input.execution.detail);
  const planningAction = executionDetail?.planning_action;
  const candidateCount = executionDetail?.candidate_count;
  const selectedRank = executionDetail?.selected_rank;
  const selectedCandidateId = executionDetail?.selected_candidate_id;
  const resultDetail = jsonRecord(executionDetail?.result);
  const optionDetail = jsonRecord(resultDetail?.option);
  const candidateSelection = planningAction === "plan_whole_body_motion_candidates"
    && typeof candidateCount === "number"
    && typeof selectedRank === "number"
    && typeof selectedCandidateId === "string"
    ? {
        candidate_count: candidateCount,
        selected_rank: selectedRank,
        selected_candidate_id: selectedCandidateId
      }
    : {};
  const motionOption = physicalOptionMemory(optionDetail);
  const episode = HumanoidEmbodiedEpisodeSchema.parse({
    sequence: input.sequence,
    source_ref: `episode:${input.sequence}`,
    transaction_id: input.execution.transactionId,
    action: input.execution.action,
    ...(planningAction === "plan_whole_body_motion"
      || planningAction === "plan_whole_body_motion_candidates"
      || planningAction === "plan_humanoid_navigation"
      ? { planning_action: planningAction }
      : {}),
    ...candidateSelection,
    ...(motionOption ? { motion_option: motionOption } : {}),
    code: input.execution.code,
    model_summary: input.modelSummary,
    world_before_revision: input.execution.worldBeforeRevision,
    world_after_revision: input.execution.worldAfterRevision,
    frame_count: input.execution.frameCount,
    result_frame: input.world.frame,
    result_root_position: input.world.robot.rootPosition,
    fallen: input.world.robot.fallen,
    support: input.world.robot.balance.support,
    upright: input.world.robot.balance.upright,
    goal_success: input.goalSuccess,
    recorded_at: input.execution.committedAt
  });
  const recent = [...state.recent_episodes, episode];
  const remove = Math.max(0, recent.length - MAX_RECENT_EMBODIED_EPISODES);
  return {
    state: HumanoidEmbodiedMemoryStateSchema.parse({
      version: 1,
      total_episodes: state.total_episodes + 1,
      pruned_episodes: state.pruned_episodes + remove,
      recent_episodes: recent.slice(remove)
    }),
    episode
  };
}

function physicalOptionMemory(
  option: Record<string, unknown> | undefined
): {
  option_id: string;
  status: "succeeded";
  termination_reason: "physical_success";
  full_frame_count: number;
  executed_prefix_frame_count: number;
  predicted_termination_frame: number;
  actual_termination_frame: number;
  artifact_sha256: string;
  rollout_sha256?: string;
} | undefined {
  if (!option
    || typeof option.option_id !== "string"
    || option.status !== "succeeded"
    || option.termination_reason !== "physical_success"
    || !positiveInteger(option.full_frame_count)
    || !positiveInteger(option.executed_prefix_frame_count)
    || !positiveInteger(option.predicted_termination_frame)
    || !positiveInteger(option.actual_termination_frame)
    || typeof option.artifact_sha256 !== "string") {
    return undefined;
  }
  return {
    option_id: option.option_id,
    status: option.status,
    termination_reason: option.termination_reason,
    full_frame_count: option.full_frame_count,
    executed_prefix_frame_count: option.executed_prefix_frame_count,
    predicted_termination_frame: option.predicted_termination_frame,
    actual_termination_frame: option.actual_termination_frame,
    artifact_sha256: option.artifact_sha256,
    ...(typeof option.rollout_sha256 === "string"
      ? { rollout_sha256: option.rollout_sha256 }
      : {})
  };
}

function positiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

function jsonRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

export function recentEmbodiedEpisodes(
  state: HumanoidEmbodiedMemoryState,
  limit = CONTEXT_EMBODIED_EPISODES
): HumanoidEmbodiedEpisode[] {
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new Error("Embodied memory limit must be a non-negative integer");
  }
  return structuredClone(state.recent_episodes.slice(-limit));
}

export function retainRecentActionReceipts<T>(
  receipts: Readonly<Record<string, T>>,
  limit = MAX_CHECKPOINT_ACTION_RECEIPTS
): { receipts: Record<string, T>; removed: number } {
  if (!Number.isSafeInteger(limit) || limit <= 0) {
    throw new Error("Action receipt checkpoint limit must be a positive integer");
  }
  const entries = Object.entries(receipts);
  const retained = entries.slice(-limit).map(([id, receipt]) => (
    [id, structuredClone(receipt)] as const
  ));
  return {
    receipts: Object.fromEntries(retained),
    removed: entries.length - retained.length
  };
}
