import {
  HumanoidEmbodiedEpisodeSchema,
  HumanoidEmbodiedExperienceSchema,
  HumanoidEmbodiedMemoryStateSchema,
  type HumanoidEmbodiedEpisode,
  type HumanoidEmbodiedExperience,
  type HumanoidEmbodiedMemoryState
} from "../../domain/humanoid-run.js";
import { goalSha256 } from "../../domain/goal-identity.js";
import type { Goal, GoalPredicate } from "../../domain/schema.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  embodiedMemoryIdForCycle,
  sameAutonomousCycle,
  type AutonomousCycleRef
} from "../../domain/autonomous-cycle.js";

export const MAX_RECENT_EMBODIED_EPISODES = 64;
export const MAX_RECENT_EMBODIED_EXPERIENCES = 128;
const CONTEXT_EMBODIED_EPISODES = 12;
const CONTEXT_EMBODIED_EXPERIENCES = 16;
export const MAX_CHECKPOINT_ACTION_RECEIPTS = 12;

export function appendEmbodiedEpisode(input: {
  state: HumanoidEmbodiedMemoryState;
  sequence: number;
  execution: HumanoidActionReceipt;
  modelSummary: string;
  world: HumanoidWorldSnapshot;
  goalSuccess: boolean;
  cycle: AutonomousCycleRef;
  goalEvidenceRefs: readonly string[];
  worldMutations?: readonly HumanoidActionReceipt[];
}): {
  state: HumanoidEmbodiedMemoryState;
  episode: HumanoidEmbodiedEpisode;
} {
  const state = HumanoidEmbodiedMemoryStateSchema.parse(input.state);
  if (!input.execution.accepted
    || (input.execution.action !== "execute_whole_body_motion"
      && input.execution.action !== "execute_humanoid_navigation")
    || input.execution.worldAfterRevision > input.world.worldRevision) {
    throw new Error("Embodied memory requires current accepted execution evidence");
  }
  if (!input.execution.decision
    || !sameAutonomousCycle(input.execution.cycle, input.cycle)) {
    throw new Error("Embodied memory requires cycle-bound model execution authority");
  }
  const executionDetail = jsonRecord(input.execution.detail);
  const executionInput = jsonRecord(input.execution.input);
  const planningAction = executionDetail?.planning_action;
  const planningTransactionId = executionDetail?.planning_transaction_id
    ?? executionInput?.planning_transaction_id;
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
  const worldMutations = (input.worldMutations ?? []).map((mutation) => (
    embodiedWorldMutation(mutation, input.execution, input.cycle, input.world.worldRevision)
  ));
  if (typeof planningTransactionId !== "string" || planningTransactionId.length === 0) {
    throw new Error("Embodied memory requires an explicit planning transaction");
  }
  const episode = HumanoidEmbodiedEpisodeSchema.parse({
    sequence: input.sequence,
    source_ref: `episode:${input.sequence}`,
    causal_trace: {
      cycle: input.cycle,
      planning_transaction_id: planningTransactionId,
      execution_transaction_id: input.execution.transactionId,
      ...(worldMutations.length > 0
        ? {
            world_mutation_transaction_ids: worldMutations.map(
              (mutation) => mutation.transaction_id
            )
          }
        : {}),
      execution_decision: input.execution.decision,
      goal_evidence_refs: [...new Set(input.goalEvidenceRefs)],
      memory_id: embodiedMemoryIdForCycle(input.cycle)
    },
    transaction_id: input.execution.transactionId,
    action: input.execution.action,
    ...(planningAction === "plan_whole_body_motion"
      || planningAction === "plan_whole_body_motion_candidates"
      || planningAction === "plan_humanoid_navigation"
      ? { planning_action: planningAction }
      : {}),
    ...candidateSelection,
    ...(motionOption ? { motion_option: motionOption } : {}),
    ...(worldMutations.length > 0 ? { world_mutations: worldMutations } : {}),
    code: input.execution.code,
    model_summary: input.modelSummary,
    world_before_revision: input.execution.worldBeforeRevision,
    world_after_revision: input.execution.worldAfterRevision,
    frame_count: input.execution.frameCount,
    result_frame: input.world.frame,
    result_world_revision: input.world.worldRevision,
    result_root_position: input.world.robot.rootPosition,
    fallen: input.world.robot.fallen,
    support: input.world.robot.balance.support,
    upright: input.world.robot.balance.upright,
    goal_success: input.goalSuccess,
    recorded_at: input.worldMutations?.at(-1)?.committedAt
      ?? input.execution.committedAt
  });
  const recent = [...state.recent_episodes, episode];
  const remove = Math.max(0, recent.length - MAX_RECENT_EMBODIED_EPISODES);
  return {
    state: HumanoidEmbodiedMemoryStateSchema.parse({
      ...state,
      version: 2,
      total_episodes: state.total_episodes + 1,
      pruned_episodes: state.pruned_episodes + remove,
      recent_episodes: recent.slice(remove)
    }),
    episode
  };
}

function embodiedWorldMutation(
  mutation: HumanoidActionReceipt,
  execution: HumanoidActionReceipt,
  cycle: AutonomousCycleRef,
  resultWorldRevision: number
) {
  if (!mutation.accepted
    || mutation.action !== "remove_world_block"
    || mutation.code !== "world_block_removal_authorized"
    || !mutation.decision
    || mutation.agentId !== execution.agentId
    || !sameAutonomousCycle(mutation.cycle, cycle)
    || mutation.worldBeforeRevision < execution.worldAfterRevision
    || mutation.worldAfterRevision < mutation.worldBeforeRevision
    || mutation.worldAfterRevision > resultWorldRevision) {
    throw new Error("Embodied memory requires an accepted causal world mutation");
  }
  const detail = jsonRecord(mutation.detail);
  const transaction = jsonRecord(detail?.removal_transaction);
  if (!transaction
    || transaction.transaction_id !== mutation.transactionId
    || transaction.execution_transaction_id !== execution.transactionId
    || typeof transaction.solid_id !== "string"
    || typeof transaction.base_chunk_revision !== "number"
    || typeof transaction.projected_chunk_revision !== "number") {
    throw new Error("Embodied memory world mutation is not bound to its execution");
  }
  return {
    transaction_id: mutation.transactionId,
    action: "remove_world_block" as const,
    decision: mutation.decision,
    code: "world_block_removal_authorized" as const,
    execution_transaction_id: execution.transactionId,
    solid_id: transaction.solid_id,
    world_before_revision: mutation.worldBeforeRevision,
    world_after_revision: mutation.worldAfterRevision,
    chunk_before_revision: transaction.base_chunk_revision,
    chunk_after_revision: transaction.projected_chunk_revision
  };
}

export function rememberEmbodiedActionExperience(input: {
  state: HumanoidEmbodiedMemoryState;
  execution: HumanoidActionReceipt;
  goal: Goal;
}): {
  state: HumanoidEmbodiedMemoryState;
  experience: HumanoidEmbodiedExperience;
  created: boolean;
} {
  const state = HumanoidEmbodiedMemoryStateSchema.parse(input.state);
  if (input.execution.action !== "execute_whole_body_motion"
    && input.execution.action !== "execute_humanoid_navigation"
    && input.execution.action !== "remove_world_block") {
    throw new Error("Embodied experience requires a physical execution or world mutation receipt");
  }
  if (!input.execution.decision || !input.execution.cycle) {
    throw new Error("Embodied experience requires cycle-bound model authority");
  }
  const sourceRef = `action:${input.execution.transactionId}`;
  const existing = state.recent_experiences.find(
    (experience) => experience.source_ref === sourceRef
  );
  if (existing) {
    return {
      state,
      experience: structuredClone(existing),
      created: false
    };
  }
  const detail = jsonRecord(input.execution.detail);
  const planningAction = detail?.planning_action;
  const outcome = input.execution.accepted
    ? "succeeded" as const
    : input.execution.frameCount > 0
      ? "physically_failed" as const
      : "rejected" as const;
  const predicates = input.goal.predicates;
  const experience = HumanoidEmbodiedExperienceSchema.parse({
    sequence: state.total_experiences + 1,
    source_ref: sourceRef,
    transaction_id: input.execution.transactionId,
    cycle: input.execution.cycle,
    action: input.execution.action,
    ...(planningAction === "plan_whole_body_motion"
      || planningAction === "plan_whole_body_motion_candidates"
      || planningAction === "plan_humanoid_navigation"
      ? { planning_action: planningAction }
      : {}),
    accepted: input.execution.accepted,
    code: input.execution.code,
    outcome,
    world_before_revision: input.execution.worldBeforeRevision,
    world_after_revision: input.execution.worldAfterRevision,
    frame_count: input.execution.frameCount,
    goal_content_sha256: goalSha256(input.goal),
    goal_summary: input.goal.summary,
    predicate_types: uniqueSorted(predicates.map((predicate) => predicate.type)),
    object_ids: uniqueSorted(predicates.flatMap(predicateObjectIds)),
    solid_ids: uniqueSorted([
      ...predicates.flatMap(predicateSolidIds),
      ...(typeof detail?.solid_id === "string" ? [detail.solid_id] : [])
    ]),
    zone_ids: uniqueSorted(predicates.flatMap(predicateZoneIds)),
    recorded_at: input.execution.committedAt
  });
  const recent = [...state.recent_experiences, experience];
  const remove = Math.max(0, recent.length - MAX_RECENT_EMBODIED_EXPERIENCES);
  const next = {
    ...state,
    version: 2 as const,
    total_experiences: state.total_experiences + 1,
    pruned_experiences: state.pruned_experiences + remove,
    recent_experiences: recent.slice(remove),
    outcome_counts: incrementOutcomeCounts(state.outcome_counts, outcome),
    predicate_outcome_counts: incrementOutcomeIndex(
      state.predicate_outcome_counts,
      experience.predicate_types,
      outcome
    ),
    object_outcome_counts: incrementOutcomeIndex(
      state.object_outcome_counts,
      experience.object_ids,
      outcome
    ),
    zone_outcome_counts: incrementOutcomeIndex(
      state.zone_outcome_counts,
      experience.zone_ids,
      outcome
    )
  };
  return {
    state: HumanoidEmbodiedMemoryStateSchema.parse(next),
    experience,
    created: true
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

export function recentEmbodiedExperiences(
  state: HumanoidEmbodiedMemoryState,
  limit = CONTEXT_EMBODIED_EXPERIENCES
): HumanoidEmbodiedExperience[] {
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new Error("Embodied experience limit must be a non-negative integer");
  }
  const parsed = HumanoidEmbodiedMemoryStateSchema.parse(state);
  return structuredClone(parsed.recent_experiences.slice(-limit));
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

type ExperienceOutcome = HumanoidEmbodiedExperience["outcome"];
type OutcomeCounts = HumanoidEmbodiedMemoryState["outcome_counts"];

function incrementOutcomeCounts(
  counts: OutcomeCounts,
  outcome: ExperienceOutcome
): OutcomeCounts {
  return {
    ...counts,
    [outcome]: counts[outcome] + 1
  };
}

function incrementOutcomeIndex(
  index: Readonly<Record<string, OutcomeCounts>>,
  keys: readonly string[],
  outcome: ExperienceOutcome
): Record<string, OutcomeCounts> {
  const next: Record<string, OutcomeCounts> = structuredClone(index);
  for (const key of keys) {
    next[key] = incrementOutcomeCounts(next[key] ?? {
      succeeded: 0,
      rejected: 0,
      physically_failed: 0
    }, outcome);
  }
  return next;
}

function predicateObjectIds(predicate: GoalPredicate): string[] {
  return predicate.type === "object_at"
    || predicate.type === "object_in_zone"
    || predicate.type === "object_placed"
    || predicate.type === "object_grasped"
    ? [predicate.object_id]
    : [];
}

function predicateZoneIds(predicate: GoalPredicate): string[] {
  return predicate.type === "robot_in_zone"
    || predicate.type === "object_in_zone"
    || predicate.type === "object_placed"
    ? [predicate.zone_id]
    : [];
}

function predicateSolidIds(predicate: GoalPredicate): string[] {
  return predicate.type === "block_removed" ? [predicate.block_id] : [];
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) => (
    left < right ? -1 : left > right ? 1 : 0
  ));
}
