import type {
  HumanoidActionReceipt,
  HumanoidEmbodiedEpisode,
  HumanoidRunCheckpoint
} from "../types";
import { goalSelectionLabel } from "../goal-state";
import { modelOutputLabel } from "../ui-text";
import { presentAction, presentEmbodiedEpisode, presentFramework } from "./presenter";

export type CycleStageKind = "sense" | "plan" | "execute" | "mutate" | "verify" | "memory";
type CycleStageState = "waiting" | "active" | "success" | "warning";

interface PresentedCycleStage {
  kind: CycleStageKind;
  title: string;
  detail: string;
  meta: string | null;
  state: CycleStageState;
}

export interface PresentedCycle {
  id: string;
  index: number;
  state: "active" | "completed" | "interrupted";
  at: string | null;
  worldBeforeRevision: number | null;
  worldAfterRevision: number | null;
  goalReached: boolean | null;
  phaseLabel: string | null;
  stages: PresentedCycleStage[];
  liveModelOutput: {
    agent: string;
    detail: string;
  } | null;
}

const STAGE_TITLES: Record<CycleStageKind, string> = {
  sense: "感知",
  plan: "候选与预演",
  execute: "物理执行",
  mutate: "世界提交",
  verify: "目标验收",
  memory: "具身记忆"
};

export function presentAutonomousCycles(input: {
  checkpoint: HumanoidRunCheckpoint;
  actions: HumanoidActionReceipt[];
  framework: unknown[];
  limit?: number;
}): PresentedCycle[] {
  const episodes = [...input.checkpoint.embodied_memory.recent_episodes]
    .sort((left, right) => left.sequence - right.sequence || left.recorded_at.localeCompare(right.recorded_at));
  const actions = [...input.actions].sort((left, right) => (
    left.committedAt.localeCompare(right.committedAt)
      || left.transactionId.localeCompare(right.transactionId)
  ));
  const moments = presentFramework(input.framework).sort((left, right) => left.at.localeCompare(right.at));
  const cycles: PresentedCycle[] = [];

  for (const episode of episodes) {
    const cycleActions = actionsForEpisode(actions, episode);
    cycles.push(completedCycle(episode, cycleActions));
  }

  const activeCycle = input.checkpoint.active_cycle ?? null;
  const currentActions = activeCycle
    ? actions.filter((action) => action.cycle?.cycle_id === activeCycle.cycle_id)
    : [];
  const currentMoments = moments.filter((moment) => activeCycle
    ? moment.cycleId === activeCycle.cycle_id
    : moment.cycleId === null);
  const live = input.checkpoint.status === "starting" || input.checkpoint.status === "running";
  if (live || currentActions.length > 0 || currentMoments.length > 0) {
    cycles.push(currentCycle(
      activeCycle?.cycle_index ?? input.checkpoint.cycle_index + 1,
      activeCycle?.cycle_id ?? null,
      activeCycle?.started_world_revision ?? null,
      currentActions,
      currentMoments,
      live,
      input.checkpoint.world.worldRevision,
      goalSelectionLabel(input.checkpoint)
    ));
  }

  return cycles.reverse().slice(0, input.limit ?? 4);
}

function completedCycle(
  episode: HumanoidEmbodiedEpisode,
  actions: HumanoidActionReceipt[]
): PresentedCycle {
  const executionId = episode.causal_trace?.execution_transaction_id
    ?? episode.transaction_id;
  const execution = actions.find((action) => action.transactionId === executionId);
  const planId = episode.causal_trace?.planning_transaction_id
    ?? planningTransactionId(execution);
  const planning = planId
    ? actions.find((action) => action.transactionId === planId)
    : undefined;
  return {
    id: episode.causal_trace?.cycle.cycle_id
      ?? `legacy-cycle-${episode.sequence}-${episode.transaction_id}`,
    index: episode.causal_trace?.cycle.cycle_index ?? episode.sequence,
    state: "completed",
    at: episode.recorded_at,
    worldBeforeRevision: episode.world_before_revision,
    worldAfterRevision: episode.result_world_revision ?? episode.world_after_revision,
    goalReached: episode.goal_success,
    phaseLabel: null,
    stages: stagesFor({ actions, planning, execution, episode }),
    liveModelOutput: null
  };
}

function currentCycle(
  index: number,
  cycleId: string | null,
  startedWorldRevision: number | null,
  actions: HumanoidActionReceipt[],
  moments: ReturnType<typeof presentFramework>,
  live: boolean,
  currentWorldRevision: number,
  phaseLabel: string | null
): PresentedCycle {
  const execution = actions.filter(isExecution).at(-1);
  const planId = planningTransactionId(execution);
  const planning = planId
    ? actions.find((action) => action.transactionId === planId)
    : actions.filter(isPlanning).at(-1);
  const latestModelOutput = moments.filter((moment) => moment.kind === "model_output").at(-1);
  const worldBeforeRevision = startedWorldRevision
    ?? actions[0]?.worldBeforeRevision
    ?? currentWorldRevision;
  const worldAfterRevision = actions.at(-1)?.worldAfterRevision ?? currentWorldRevision;
  return {
    id: cycleId ?? `goal-transition-${index}`,
    index,
    state: live ? "active" : "interrupted",
    at: actions.at(-1)?.committedAt ?? latestModelOutput?.at ?? null,
    worldBeforeRevision,
    worldAfterRevision,
    goalReached: null,
    phaseLabel,
    stages: stagesFor({ actions, planning, execution }),
    liveModelOutput: latestModelOutput
      ? { agent: latestModelOutput.agent, detail: latestModelOutput.detail }
      : null
  };
}

function stagesFor(input: {
  actions: HumanoidActionReceipt[];
  planning?: HumanoidActionReceipt | undefined;
  execution?: HumanoidActionReceipt | undefined;
  episode?: HumanoidEmbodiedEpisode | undefined;
}): PresentedCycleStage[] {
  const observations = input.actions.filter((action) => action.action === "observe_humanoid");
  const latestObservation = observations.at(-1);
  const plans = input.actions.filter(isPlanning);
  const rejectedPlans = plans.filter((action) => !action.accepted).length;
  const sense = latestObservation
    ? actionStage("sense", latestObservation, observations.length > 1
      ? `${observations.length} 次有效感知`
      : null)
    : waitingStage("sense");
  const plan = input.planning
    ? actionStage("plan", input.planning, rejectedPlans > 0
      ? `${rejectedPlans} 个候选未通过约束`
      : null)
    : waitingStage("plan", plans.length > 0 && plans.every((action) => !action.accepted)
      ? "候选均未通过物理约束"
      : undefined, plans.length > 0 ? "warning" : "waiting");
  const execute = input.execution
    ? actionStage("execute", input.execution)
    : waitingStage("execute");
  const mutations = input.actions.filter((action) => action.action === "remove_world_block");
  const mutation = mutations.at(-1);
  const mutate = mutation
    ? actionStage("mutate", mutation, mutations.length > 1
      ? `${mutations.length} 次权威世界提交`
      : null)
    : waitingStage("mutate", "本轮无需改变世界");
  const verify = input.episode
    ? {
        kind: "verify" as const,
        title: STAGE_TITLES.verify,
        detail: input.episode.goal_success
          ? "真实世界状态已满足本轮 Goal。"
          : "本轮物理状态已验收，当前 Goal 尚未达成。",
        meta: `世界版本 ${input.episode.world_before_revision} → ${input.episode.world_after_revision}`,
        state: "success" as const
      }
    : waitingStage("verify");
  const memory = input.episode
    ? episodeStage(input.episode)
    : waitingStage("memory");
  return [sense, plan, execute, mutate, verify, memory];
}

function actionStage(
  kind: Extract<CycleStageKind, "sense" | "plan" | "execute" | "mutate">,
  action: HumanoidActionReceipt,
  extraMeta?: string | null
): PresentedCycleStage {
  const presented = presentAction(action);
  return {
    kind,
    title: STAGE_TITLES[kind],
    detail: presented.detail,
    meta: [presented.meta, extraMeta].filter(Boolean).join(" · ") || null,
    state: action.accepted ? "success" : "warning"
  };
}

function episodeStage(episode: HumanoidEmbodiedEpisode): PresentedCycleStage {
  const presented = presentEmbodiedEpisode(episode);
  return {
    kind: "memory",
    title: STAGE_TITLES.memory,
    detail: modelOutputLabel(episode.model_summary, 160) ?? presented.detail,
    meta: presented.meta,
    state: episode.fallen ? "warning" : "success"
  };
}

function waitingStage(
  kind: CycleStageKind,
  detail = "等待真实回执",
  state: CycleStageState = "waiting"
): PresentedCycleStage {
  return { kind, title: STAGE_TITLES[kind], detail, meta: null, state };
}

function isPlanning(action: HumanoidActionReceipt): boolean {
  return action.action === "plan_whole_body_motion"
    || action.action === "plan_whole_body_motion_candidates"
    || action.action === "plan_humanoid_navigation";
}

function actionsForEpisode(
  actions: HumanoidActionReceipt[],
  episode: HumanoidEmbodiedEpisode
): HumanoidActionReceipt[] {
  const cycleId = episode.causal_trace?.cycle.cycle_id;
  if (cycleId) {
    return actions.filter((action) => action.cycle?.cycle_id === cycleId);
  }
  const execution = actions.find((action) => (
    action.transactionId === episode.transaction_id && isExecution(action)
  ));
  const planningId = planningTransactionId(execution);
  return actions.filter((action) => (
    action.transactionId === execution?.transactionId
      || action.transactionId === planningId
  ));
}

function isExecution(action: HumanoidActionReceipt): boolean {
  return action.action === "execute_whole_body_motion"
    || action.action === "execute_humanoid_navigation";
}

function planningTransactionId(action: HumanoidActionReceipt | undefined): string | null {
  const input = record(action?.input);
  const detail = record(action?.detail);
  return stringOf(input?.planning_transaction_id)
    ?? stringOf(detail?.planning_transaction_id);
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringOf(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}
