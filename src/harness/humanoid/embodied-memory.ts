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
  const episode = HumanoidEmbodiedEpisodeSchema.parse({
    sequence: input.sequence,
    transaction_id: input.execution.transactionId,
    action: input.execution.action,
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
