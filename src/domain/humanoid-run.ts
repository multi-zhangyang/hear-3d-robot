import { z } from "zod";
import {
  ContextMemoryStateSchema,
  GoalSchema,
  JsonValueSchema,
  RunLifecycleEventSchema,
  RunStatusSchema,
  TaskNodeSchema,
  Vec3Schema,
  type JsonValue
} from "./schema.js";
import { HUMANOID_ACTION_NAMES } from "./humanoid-action.js";
import {
  ActionCommitOutboxSchema,
  EmptyActionCommitOutbox,
  actionCommitPayloadSha256,
  actionCommitReceiptSha256
} from "./action-commit-outbox.js";
import {
  ActionExecutionLedgerSchema,
  EmptyActionExecutionLedger
} from "./action-execution-ledger.js";
import { goalSha256 } from "./goal-identity.js";
import { GoalDAGSchema, createGoalDAG } from "./goal-epoch.js";
import { ModelDecisionRefSchema } from "./model-call-authority.js";
import {
  EmptyModelUsageState,
  ModelUsageStateSchema
} from "./model-usage.js";
import {
  ActiveAutonomousCycleSchema,
  AutonomousCycleRefSchema,
  EmbodiedMemoryIdSchema,
  embodiedMemoryIdForCycle
} from "./autonomous-cycle.js";
import {
  HumanoidWorldCheckpointSchema,
  LegacyHumanoidWorldCheckpointSchema,
  type HumanoidWorldCheckpoint
} from "../world/humanoid/checkpoint.js";
import {
  HumanoidWorldSnapshotSchema,
  LegacyHumanoidWorldSnapshotSchema,
  PreGraspHumanoidWorldSnapshotSchema
} from "../world/humanoid/snapshot-schema.js";
import type { HumanoidWorldSnapshot } from "../world/humanoid/world-contract.js";

const HumanoidActionNameSchema = z.enum(HUMANOID_ACTION_NAMES);

const HumanoidBodyChannelSchema = z.enum([
  "locomotion",
  "left_leg",
  "right_leg",
  "torso",
  "left_arm",
  "right_arm"
]);

export const PersistedHumanoidActionReceiptSchema = z.object({
  transactionId: z.string().min(1),
  agentId: z.string().min(1),
  decision: ModelDecisionRefSchema.optional(),
  cycle: AutonomousCycleRefSchema.optional(),
  action: HumanoidActionNameSchema,
  input: JsonValueSchema,
  fingerprint: z.string().min(1),
  accepted: z.boolean(),
  code: z.string().min(1),
  worldBeforeRevision: z.number().int().nonnegative(),
  worldAfterRevision: z.number().int().nonnegative(),
  frameCount: z.number().int().nonnegative(),
  channels: z.array(HumanoidBodyChannelSchema),
  detail: JsonValueSchema,
  committedAt: z.string().datetime()
}).strict();

export const HumanoidCheckerResultSchema = z.object({
  success: z.boolean(),
  goal: GoalSchema,
  worldFrame: z.number().int().nonnegative(),
  worldRevision: z.number().int().nonnegative(),
  checks: z.array(z.object({
    name: z.string().min(1),
    passed: z.boolean(),
    actual: JsonValueSchema
  }).strict()),
  checkedAt: z.string().datetime()
}).strict();

export const HumanoidGoalProgressSchema = z.object({
  version: z.literal(1),
  goal_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  predicate_count: z.number().int().positive(),
  last_world_frame: z.number().int().nonnegative(),
  last_world_revision: z.number().int().nonnegative(),
  predicate_streaks: z.array(z.number().int().min(0).max(500))
}).strict().superRefine((progress, context) => {
  if (progress.predicate_streaks.length !== progress.predicate_count) {
    context.addIssue({
      code: "custom",
      path: ["predicate_streaks"],
      message: "Humanoid goal progress must contain one streak per predicate"
    });
  }
});

export type HumanoidGoalProgress = z.infer<typeof HumanoidGoalProgressSchema>;

const HumanoidEpisodeCausalTraceSchema = z.object({
  cycle: AutonomousCycleRefSchema,
  planning_transaction_id: z.string().trim().min(1),
  execution_transaction_id: z.string().trim().min(1),
  world_mutation_transaction_ids: z.array(z.string().trim().min(1)).optional(),
  execution_decision: ModelDecisionRefSchema,
  goal_evidence_refs: z.array(z.string().trim().min(1)).min(1),
  memory_id: EmbodiedMemoryIdSchema
}).strict().superRefine((trace, context) => {
  if (trace.execution_decision.tool_call_id !== trace.execution_transaction_id) {
    context.addIssue({
      code: "custom",
      path: ["execution_decision", "tool_call_id"],
      message: "Embodied execution decision must identify its execution transaction"
    });
  }
  if (!trace.goal_evidence_refs.includes(`action:${trace.execution_transaction_id}`)) {
    context.addIssue({
      code: "custom",
      path: ["goal_evidence_refs"],
      message: "Embodied causal trace requires its durable action evidence"
    });
  }
  for (const transactionId of trace.world_mutation_transaction_ids ?? []) {
    if (!trace.goal_evidence_refs.includes(`action:${transactionId}`)) {
      context.addIssue({
        code: "custom",
        path: ["goal_evidence_refs"],
        message: "Embodied causal trace requires durable evidence for every world mutation"
      });
    }
  }
  if (new Set(trace.world_mutation_transaction_ids ?? []).size
    !== (trace.world_mutation_transaction_ids ?? []).length) {
    context.addIssue({
      code: "custom",
      path: ["world_mutation_transaction_ids"],
      message: "Embodied world mutation identities must be unique"
    });
  }
  if (new Set(trace.goal_evidence_refs).size !== trace.goal_evidence_refs.length) {
    context.addIssue({
      code: "custom",
      path: ["goal_evidence_refs"],
      message: "Embodied causal evidence references must be unique"
    });
  }
});

const HumanoidEpisodeWorldMutationSchema = z.object({
  transaction_id: z.string().trim().min(1),
  action: z.literal("remove_world_block"),
  decision: ModelDecisionRefSchema,
  code: z.literal("world_block_removal_authorized"),
  execution_transaction_id: z.string().trim().min(1),
  solid_id: z.string().trim().min(1),
  world_before_revision: z.number().int().nonnegative(),
  world_after_revision: z.number().int().nonnegative(),
  chunk_before_revision: z.number().int().nonnegative(),
  chunk_after_revision: z.number().int().positive()
}).strict().superRefine((mutation, context) => {
  if (mutation.decision.tool_call_id !== mutation.transaction_id) {
    context.addIssue({
      code: "custom",
      path: ["decision", "tool_call_id"],
      message: "Embodied world mutation decision must identify its transaction"
    });
  }
  if (mutation.world_after_revision < mutation.world_before_revision
    || mutation.chunk_after_revision !== mutation.chunk_before_revision + 1) {
    context.addIssue({
      code: "custom",
      path: ["chunk_after_revision"],
      message: "Embodied world mutation revisions are inconsistent"
    });
  }
});

export const HumanoidEmbodiedEpisodeSchema = z.object({
  sequence: z.number().int().positive(),
  source_ref: z.string().regex(/^episode:[1-9]\d*$/).optional(),
  causal_trace: HumanoidEpisodeCausalTraceSchema.optional(),
  transaction_id: z.string().min(1),
  action: z.enum([
    "execute_humanoid_skill",
    "execute_whole_body_motion",
    "execute_humanoid_navigation"
  ]),
  planning_action: z.enum([
    "plan_humanoid_skill",
    "plan_whole_body_motion",
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
  ]).optional(),
  candidate_count: z.number().int().min(1).max(3).optional(),
  selected_rank: z.number().int().min(1).max(3).optional(),
  selected_candidate_id: z.string().min(1).optional(),
  motion_option: z.object({
    option_id: z.string().min(1),
    status: z.literal("succeeded"),
    termination_reason: z.literal("physical_success"),
    full_frame_count: z.number().int().positive(),
    executed_prefix_frame_count: z.number().int().positive(),
    predicted_termination_frame: z.number().int().positive(),
    actual_termination_frame: z.number().int().positive(),
    artifact_sha256: z.string().regex(/^[a-f0-9]{64}$/),
    rollout_sha256: z.string().regex(/^[a-f0-9]{64}$/).optional()
  }).strict().optional(),
  world_mutations: z.array(HumanoidEpisodeWorldMutationSchema).max(8).optional(),
  code: z.string().min(1),
  model_summary: z.string().trim().min(1),
  world_before_revision: z.number().int().nonnegative(),
  world_after_revision: z.number().int().nonnegative(),
  frame_count: z.number().int().nonnegative(),
  result_frame: z.number().int().nonnegative(),
  result_world_revision: z.number().int().nonnegative().optional(),
  result_root_position: Vec3Schema,
  fallen: z.boolean(),
  support: z.enum(["none", "left", "right", "double"]),
  upright: z.number().finite(),
  goal_success: z.boolean(),
  recorded_at: z.string().datetime()
}).strict().superRefine((episode, context) => {
  if (episode.causal_trace) {
    if (episode.causal_trace.execution_transaction_id !== episode.transaction_id
      || episode.causal_trace.cycle.cycle_index !== episode.sequence
      || episode.causal_trace.memory_id
        !== embodiedMemoryIdForCycle(episode.causal_trace.cycle)) {
      context.addIssue({
        code: "custom",
        path: ["causal_trace"],
        message: "Embodied causal trace does not belong to this episode"
      });
    }
    if (!episode.planning_action) {
      context.addIssue({
        code: "custom",
        path: ["planning_action"],
        message: "A causal embodied episode requires its planning action"
      });
    }
    const traceMutations = episode.causal_trace.world_mutation_transaction_ids ?? [];
    const episodeMutations = (episode.world_mutations ?? []).map(
      (mutation) => mutation.transaction_id
    );
    if (JSON.stringify(traceMutations) !== JSON.stringify(episodeMutations)) {
      context.addIssue({
        code: "custom",
        path: ["world_mutations"],
        message: "Embodied world mutations do not match the causal trace"
      });
    }
    if ((episode.world_mutations ?? []).some((mutation) => (
      mutation.execution_transaction_id !== episode.transaction_id
    ))) {
      context.addIssue({
        code: "custom",
        path: ["world_mutations"],
        message: "Embodied world mutation does not consume this episode execution"
      });
    }
  }
  if (episode.result_world_revision !== undefined
    && (episode.result_world_revision < episode.world_after_revision
      || episode.world_mutations?.some((mutation) => (
        mutation.world_after_revision > episode.result_world_revision!
      )))) {
    context.addIssue({
      code: "custom",
      path: ["result_world_revision"],
      message: "Embodied episode result precedes its causal actions"
    });
  }
  const selection = [
    episode.candidate_count,
    episode.selected_rank,
    episode.selected_candidate_id
  ];
  const present = selection.filter((value) => value !== undefined).length;
  if (episode.planning_action === "plan_whole_body_motion_candidates") {
    if (present !== selection.length) {
      context.addIssue({
        code: "custom",
        path: ["candidate_count"],
        message: "Candidate-backed embodied memory requires complete selection evidence"
      });
    } else if (episode.selected_rank! > episode.candidate_count!) {
      context.addIssue({
        code: "custom",
        path: ["selected_rank"],
        message: "Selected candidate rank cannot exceed the candidate count"
      });
    }
  } else if (present !== 0) {
    context.addIssue({
      code: "custom",
      path: ["candidate_count"],
      message: "Candidate selection evidence requires a candidate planning action"
    });
  }
  if (episode.code === "motion_option_succeeded" && !episode.motion_option) {
    context.addIssue({
      code: "custom",
      path: ["motion_option"],
      message: "A successful humanoid motion option requires physical termination evidence"
    });
  }
  if (episode.motion_option) {
    if ((episode.action !== "execute_whole_body_motion"
        && episode.action !== "execute_humanoid_skill")
      || episode.motion_option.executed_prefix_frame_count
        > episode.motion_option.full_frame_count
      || episode.motion_option.actual_termination_frame
        !== episode.motion_option.executed_prefix_frame_count
      || episode.motion_option.predicted_termination_frame
        > episode.motion_option.full_frame_count) {
      context.addIssue({
        code: "custom",
        path: ["motion_option"],
        message: "Humanoid embodied option evidence is inconsistent with its execution"
      });
    }
  }
});

const HumanoidEmbodiedMemoryStateV1Schema = z.object({
  version: z.literal(1),
  total_episodes: z.number().int().nonnegative(),
  pruned_episodes: z.number().int().nonnegative(),
  recent_episodes: z.array(HumanoidEmbodiedEpisodeSchema).max(64)
}).strict();

const HumanoidExperienceOutcomeSchema = z.enum([
  "succeeded",
  "rejected",
  "physically_failed"
]);

const HumanoidExperienceOutcomeCountsSchema = z.object({
  succeeded: z.number().int().nonnegative(),
  rejected: z.number().int().nonnegative(),
  physically_failed: z.number().int().nonnegative()
}).strict();

export const HumanoidEmbodiedExperienceSchema = z.object({
  sequence: z.number().int().positive(),
  source_ref: z.string().regex(/^action:\S+$/),
  transaction_id: z.string().trim().min(1),
  cycle: AutonomousCycleRefSchema,
  action: z.enum([
    "execute_humanoid_skill",
    "execute_whole_body_motion",
    "execute_humanoid_navigation",
    "remove_world_block"
  ]),
  planning_action: z.enum([
    "plan_humanoid_skill",
    "plan_whole_body_motion",
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
  ]).optional(),
  accepted: z.boolean(),
  code: z.string().trim().min(1),
  outcome: HumanoidExperienceOutcomeSchema,
  world_before_revision: z.number().int().nonnegative(),
  world_after_revision: z.number().int().nonnegative(),
  frame_count: z.number().int().nonnegative(),
  goal_content_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  goal_summary: z.string().trim().min(1),
  predicate_types: z.array(z.string().trim().min(1)),
  object_ids: z.array(z.string().trim().min(1)),
  solid_ids: z.array(z.string().trim().min(1)).default([]),
  zone_ids: z.array(z.string().trim().min(1)),
  recorded_at: z.string().datetime()
}).strict().superRefine((experience, context) => {
  if (experience.source_ref !== `action:${experience.transaction_id}`) {
    context.addIssue({
      code: "custom",
      path: ["source_ref"],
      message: "Embodied experience source must identify its action transaction"
    });
  }
  for (const [field, values] of [
    ["predicate_types", experience.predicate_types],
    ["object_ids", experience.object_ids],
    ["solid_ids", experience.solid_ids],
    ["zone_ids", experience.zone_ids]
  ] as const) {
    const sorted = [...new Set(values)].sort(compareCodePoints);
    if (JSON.stringify(sorted) !== JSON.stringify(values)) {
      context.addIssue({
        code: "custom",
        path: [field],
        message: "Embodied experience indexes must be unique and sorted"
      });
    }
  }
  const expectedOutcome = experience.accepted
    ? "succeeded"
    : experience.frame_count > 0
      ? "physically_failed"
      : "rejected";
  if (experience.outcome !== expectedOutcome) {
    context.addIssue({
      code: "custom",
      path: ["outcome"],
      message: "Embodied experience outcome does not match its physical receipt"
    });
  }
});

const HumanoidEmbodiedMemoryStateV2Schema = z.object({
  version: z.literal(2),
  total_episodes: z.number().int().nonnegative(),
  pruned_episodes: z.number().int().nonnegative(),
  recent_episodes: z.array(HumanoidEmbodiedEpisodeSchema).max(64),
  total_experiences: z.number().int().nonnegative(),
  pruned_experiences: z.number().int().nonnegative(),
  recent_experiences: z.array(HumanoidEmbodiedExperienceSchema).max(128),
  outcome_counts: HumanoidExperienceOutcomeCountsSchema,
  predicate_outcome_counts: z.record(
    z.string().trim().min(1),
    HumanoidExperienceOutcomeCountsSchema
  ),
  object_outcome_counts: z.record(
    z.string().trim().min(1),
    HumanoidExperienceOutcomeCountsSchema
  ),
  solid_outcome_counts: z.record(
    z.string().trim().min(1),
    HumanoidExperienceOutcomeCountsSchema
  ).default({}),
  zone_outcome_counts: z.record(
    z.string().trim().min(1),
    HumanoidExperienceOutcomeCountsSchema
  )
}).strict().superRefine((state, context) => {
  const counted = Object.values(state.outcome_counts).reduce(
    (total, count) => total + count,
    0
  );
  if (counted !== state.total_experiences) {
    context.addIssue({
      code: "custom",
      path: ["outcome_counts"],
      message: "Embodied experience lifetime counts do not match the total"
    });
  }
});

export const HumanoidEmbodiedMemoryStateSchema = z.union([
  HumanoidEmbodiedMemoryStateV1Schema,
  HumanoidEmbodiedMemoryStateV2Schema
]).transform((state) => state.version === 2 ? state : ({
  version: 2 as const,
  total_episodes: state.total_episodes,
  pruned_episodes: state.pruned_episodes,
  recent_episodes: state.recent_episodes,
  total_experiences: 0,
  pruned_experiences: 0,
  recent_experiences: [],
  outcome_counts: emptyExperienceOutcomeCounts(),
  predicate_outcome_counts: {},
  object_outcome_counts: {},
  solid_outcome_counts: {},
  zone_outcome_counts: {}
}));

export type HumanoidEmbodiedEpisode = z.infer<typeof HumanoidEmbodiedEpisodeSchema>;
export type HumanoidEmbodiedExperience = z.infer<
  typeof HumanoidEmbodiedExperienceSchema
>;
export type HumanoidEmbodiedMemoryState = z.output<
  typeof HumanoidEmbodiedMemoryStateSchema
>;

export const EmptyHumanoidEmbodiedMemoryState: HumanoidEmbodiedMemoryState = {
  version: 2,
  total_episodes: 0,
  pruned_episodes: 0,
  recent_episodes: [],
  total_experiences: 0,
  pruned_experiences: 0,
  recent_experiences: [],
  outcome_counts: emptyExperienceOutcomeCounts(),
  predicate_outcome_counts: {},
  object_outcome_counts: {},
  solid_outcome_counts: {},
  zone_outcome_counts: {}
};

function emptyExperienceOutcomeCounts(): z.infer<
  typeof HumanoidExperienceOutcomeCountsSchema
> {
  return { succeeded: 0, rejected: 0, physically_failed: 0 };
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

const HumanoidRunCheckpointBaseShape = {
  runtime: z.literal("humanoid_g1"),
  run_id: z.string().min(1),
  scenario_id: z.string().min(1),
  capability_catalog: z.array(z.string().min(1)),
  status: RunStatusSchema,
  root_id: z.string().min(1),
  active_agent_id: z.string().min(1).nullable(),
  active_agent_ids: z.array(z.string().min(1)),
  nodes: z.record(z.string().min(1), TaskNodeSchema),
  world: HumanoidWorldSnapshotSchema,
  world_checkpoint: HumanoidWorldCheckpointSchema,
  committed_actions: z.record(z.string().min(1), PersistedHumanoidActionReceiptSchema),
  action_runtime_state: JsonValueSchema.nullable().default(null),
  context_memory: ContextMemoryStateSchema,
  embodied_memory: HumanoidEmbodiedMemoryStateSchema,
  pending_lifecycle_events: z.array(RunLifecycleEventSchema),
  cycle_index: z.number().int().nonnegative(),
  total_model_calls: z.number().int().nonnegative(),
  model_usage: ModelUsageStateSchema.default(EmptyModelUsageState),
  checker: HumanoidCheckerResultSchema.nullable(),
  last_cycle: JsonValueSchema.nullable(),
  final_output: z.string().nullable(),
  error: z.string().nullable(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
} as const;

const LegacyHumanoidRunCheckpointBaseShape = {
  ...HumanoidRunCheckpointBaseShape,
  world: LegacyHumanoidWorldSnapshotSchema,
  world_checkpoint: LegacyHumanoidWorldCheckpointSchema
} as const;

export const HumanoidRunCheckpointV4Schema = z.object({
  version: z.literal(4),
  ...LegacyHumanoidRunCheckpointBaseShape,
  goal: GoalSchema
}).strict();

export type HumanoidRunCheckpointV4 = z.infer<
  typeof HumanoidRunCheckpointV4Schema
>;

export const HumanoidRunCheckpointV5Schema = z.object({
  version: z.literal(5),
  ...LegacyHumanoidRunCheckpointBaseShape,
  goal: GoalSchema,
  goal_progress: HumanoidGoalProgressSchema
}).strict().superRefine((checkpoint, context) => {
  const progress = checkpoint.goal_progress;
  if (progress.goal_sha256 !== goalSha256(checkpoint.goal)) {
    context.addIssue({
      code: "custom",
      path: ["goal_progress", "goal_sha256"],
      message: "Humanoid goal progress does not belong to the checkpoint goal"
    });
  }
  if (progress.predicate_count !== checkpoint.goal.predicates.length) {
    context.addIssue({
      code: "custom",
      path: ["goal_progress", "predicate_count"],
      message: "Humanoid goal progress predicate count does not match the goal"
    });
  }
  checkpoint.goal.predicates.forEach((predicate, index) => {
    const streak = progress.predicate_streaks[index];
    if (streak === undefined) return;
    if (predicate.type === "end_effector_at") {
      if (streak > predicate.stable_frames) {
        context.addIssue({
          code: "custom",
          path: ["goal_progress", "predicate_streaks", index],
          message: "Humanoid end-effector stability exceeds its goal requirement"
        });
      }
    } else if (streak !== 0) {
      context.addIssue({
        code: "custom",
        path: ["goal_progress", "predicate_streaks", index],
        message: "An instantaneous goal predicate cannot carry stability progress"
      });
    }
  });
  if (checkpoint.world.frame !== checkpoint.world_checkpoint.frame
    || checkpoint.world.worldRevision !== checkpoint.world_checkpoint.worldRevision) {
    context.addIssue({
      code: "custom",
      path: ["world_checkpoint"],
      message: "Humanoid world snapshot and physical checkpoint are not aligned"
    });
  }
  if (progress.last_world_frame !== checkpoint.world.frame
    || progress.last_world_revision !== checkpoint.world.worldRevision) {
    context.addIssue({
      code: "custom",
      path: ["goal_progress"],
      message: "Humanoid goal progress is not aligned with the authoritative world frame"
    });
  }
});

export type HumanoidRunCheckpointV5 = z.infer<
  typeof HumanoidRunCheckpointV5Schema
>;

export const LegacyHumanoidRunCheckpointSchema = z.discriminatedUnion(
  "version",
  [HumanoidRunCheckpointV4Schema, HumanoidRunCheckpointV5Schema]
);

export type LegacyHumanoidRunCheckpoint = z.infer<
  typeof LegacyHumanoidRunCheckpointSchema
>;

const HumanoidRunCheckpointV6Schema = z.object({
  version: z.literal(6),
  ...HumanoidRunCheckpointBaseShape,
  mission_goal: GoalSchema,
  goal_dag: GoalDAGSchema,
  goal_progress: HumanoidGoalProgressSchema.nullable(),
  active_cycle: ActiveAutonomousCycleSchema.nullable().default(null),
  action_commit_outbox: ActionCommitOutboxSchema.default(EmptyActionCommitOutbox),
  action_execution_ledger: ActionExecutionLedgerSchema.default(EmptyActionExecutionLedger)
}).strict().superRefine((checkpoint, context) => {
  if (checkpoint.active_cycle) {
    if (checkpoint.goal_dag.status !== "active"
      || checkpoint.goal_dag.current_epoch_id !== checkpoint.active_cycle.goal_epoch_id
      || checkpoint.active_cycle.cycle_index !== checkpoint.cycle_index + 1
      || checkpoint.active_cycle.started_world_frame > checkpoint.world.frame
      || checkpoint.active_cycle.started_world_revision > checkpoint.world.worldRevision) {
      context.addIssue({
        code: "custom",
        path: ["active_cycle"],
        message: "Active autonomous cycle is not aligned with the Goal epoch and world"
      });
    }
  }
  for (const [transactionId, receipt] of Object.entries(checkpoint.committed_actions)) {
    if (receipt.transactionId !== transactionId) {
      context.addIssue({
        code: "custom",
        path: ["committed_actions", transactionId],
        message: "Committed action key does not match its transaction identity"
      });
    }
  }
  for (const [transactionId, pending] of Object.entries(
    checkpoint.action_commit_outbox.pending
  )) {
    if (pending.runtime_event.run_id !== checkpoint.run_id) {
      context.addIssue({
        code: "custom",
        path: ["action_commit_outbox", "pending", transactionId, "runtime_event", "run_id"],
        message: "Pending action event does not belong to the checkpoint run"
      });
    }
    const receipt = checkpoint.committed_actions[transactionId];
    const pendingReceiptSha256 = actionCommitReceiptSha256(pending.action_record);
    if (!receipt || !pendingReceiptSha256
      || actionCommitPayloadSha256(receipt as JsonValue) !== pendingReceiptSha256) {
      context.addIssue({
        code: "custom",
        path: ["action_commit_outbox", "pending", transactionId, "action_record"],
        message: "Pending action commit does not match its checkpoint receipt"
      });
    }
  }
  for (const [transactionId, execution] of Object.entries(
    checkpoint.action_execution_ledger.active
  )) {
    if (execution.run_id !== checkpoint.run_id) {
      context.addIssue({
        code: "custom",
        path: ["action_execution_ledger", "active", transactionId, "run_id"],
        message: "Active action execution does not belong to the checkpoint run"
      });
    }
    if (execution.status !== "terminal") {
      if (checkpoint.committed_actions[transactionId]) {
        context.addIssue({
          code: "custom",
          path: ["action_execution_ledger", "active", transactionId],
          message: "A nonterminal physical execution cannot have a committed receipt"
        });
      }
      continue;
    }
    const pending = checkpoint.action_commit_outbox.pending[transactionId];
    if (!pending || !execution.terminal
      || execution.terminal.action_record_sha256 !== pending.action_record_sha256
      || execution.terminal.receipt_sha256
        !== actionCommitReceiptSha256(pending.action_record)
      || execution.terminal.runtime_event_id !== pending.runtime_event_id
      || execution.terminal.runtime_event_sha256 !== pending.runtime_event_sha256
      || execution.terminal.goal_evidence_ref !== pending.goal_evidence_ref
      || execution.terminal.goal_evidence_sha256 !== pending.goal_evidence_sha256) {
      context.addIssue({
        code: "custom",
        path: ["action_execution_ledger", "active", transactionId, "terminal"],
        message: "Terminal physical execution is not bound to its pending durable commit"
      });
    }
  }
  const activeEpoch = checkpoint.goal_dag.epochs.find(
    (epoch) => epoch.epoch_id === checkpoint.goal_dag.current_epoch_id
  );
  const activeCandidate = activeEpoch
    ? checkpoint.goal_dag.candidates[activeEpoch.candidate_id]
    : undefined;
  if (checkpoint.goal_dag.status === "active") {
    if (!activeCandidate || !checkpoint.goal_progress) {
      context.addIssue({
        code: "custom",
        path: ["goal_progress"],
        message: "An active Goal epoch requires physical predicate progress"
      });
      return;
    }
    validateGoalProgress(
      activeCandidate.goal,
      checkpoint.goal_progress,
      checkpoint.world,
      checkpoint.world_checkpoint,
      context
    );
    if (checkpoint.checker
      && goalSha256(checkpoint.checker.goal) !== activeCandidate.content_sha256) {
      context.addIssue({
        code: "custom",
        path: ["checker", "goal"],
        message: "Humanoid checker does not belong to the active Goal epoch"
      });
    }
  } else if (checkpoint.goal_progress !== null || checkpoint.checker !== null) {
    context.addIssue({
      code: "custom",
      path: ["goal_progress"],
      message: "A run awaiting model Goal selection cannot retain active progress or checker state"
    });
  }
});

export type HumanoidRunCheckpoint = z.infer<
  typeof HumanoidRunCheckpointV6Schema
>;

export const PreGraspHumanoidRunCheckpointV6Schema = z.object({
  version: z.literal(6),
  ...HumanoidRunCheckpointBaseShape,
  world: PreGraspHumanoidWorldSnapshotSchema,
  world_checkpoint: LegacyHumanoidWorldCheckpointSchema,
  mission_goal: GoalSchema,
  goal_dag: GoalDAGSchema,
  goal_progress: HumanoidGoalProgressSchema.nullable(),
  active_cycle: ActiveAutonomousCycleSchema.nullable().default(null),
  action_commit_outbox: ActionCommitOutboxSchema.default(EmptyActionCommitOutbox),
  action_execution_ledger: ActionExecutionLedgerSchema.default(
    EmptyActionExecutionLedger
  )
}).strict();

function validateGoalProgress(
  goal: z.infer<typeof GoalSchema>,
  progress: HumanoidGoalProgress,
  world: { frame: number; worldRevision: number },
  worldCheckpoint: { frame: number; worldRevision: number },
  context: z.RefinementCtx
): void {
  if (progress.goal_sha256 !== goalSha256(goal)) {
    context.addIssue({
      code: "custom",
      path: ["goal_progress", "goal_sha256"],
      message: "Humanoid goal progress does not belong to the active Goal"
    });
  }
  if (progress.predicate_count !== goal.predicates.length) {
    context.addIssue({
      code: "custom",
      path: ["goal_progress", "predicate_count"],
      message: "Humanoid goal progress predicate count does not match the active Goal"
    });
  }
  goal.predicates.forEach((predicate, index) => {
    const streak = progress.predicate_streaks[index];
    if (streak === undefined) return;
    if (predicate.type === "end_effector_at") {
      if (streak > predicate.stable_frames) {
        context.addIssue({
          code: "custom",
          path: ["goal_progress", "predicate_streaks", index],
          message: "Humanoid end-effector stability exceeds its Goal requirement"
        });
      }
    } else if (streak !== 0) {
      context.addIssue({
        code: "custom",
        path: ["goal_progress", "predicate_streaks", index],
        message: "An instantaneous Goal predicate cannot carry stability progress"
      });
    }
  });
  if (world.frame !== worldCheckpoint.frame
    || world.worldRevision !== worldCheckpoint.worldRevision) {
    context.addIssue({
      code: "custom",
      path: ["world_checkpoint"],
      message: "Humanoid world snapshot and physical checkpoint are not aligned"
    });
  }
  if (progress.last_world_frame !== world.frame
    || progress.last_world_revision !== world.worldRevision) {
    context.addIssue({
      code: "custom",
      path: ["goal_progress"],
      message: "Humanoid Goal progress is not aligned with the authoritative world frame"
    });
  }
}

function migrateHumanoidRunCheckpointV4(
  checkpoint: HumanoidRunCheckpointV4
): z.infer<typeof HumanoidRunCheckpointV5Schema> {
  const { version: _version, ...source } = checkpoint;
  return HumanoidRunCheckpointV5Schema.parse({
    version: 5,
    ...source,
    goal_progress: {
      version: 1,
      goal_sha256: goalSha256(checkpoint.goal),
      predicate_count: checkpoint.goal.predicates.length,
      last_world_frame: checkpoint.world.frame,
      last_world_revision: checkpoint.world.worldRevision,
      predicate_streaks: checkpoint.goal.predicates.map(() => 0)
    }
  });
}

function migrateHumanoidRunCheckpointV5(
  _checkpoint: HumanoidRunCheckpointV5
): HumanoidRunCheckpoint {
  throw new Error(
    "Legacy 29DoF humanoid checkpoint requires physical migration before V6 recovery"
  );
}

export function completeHumanoidRunCheckpointPhysicalMigration(input: {
  checkpoint: LegacyHumanoidRunCheckpoint;
  world: HumanoidWorldSnapshot;
  worldCheckpoint: HumanoidWorldCheckpoint;
}): HumanoidRunCheckpoint {
  const legacy = LegacyHumanoidRunCheckpointSchema.parse(input.checkpoint);
  const checkpoint = legacy.version === 4
    ? migrateHumanoidRunCheckpointV4(legacy)
    : legacy;
  const world = HumanoidWorldSnapshotSchema.parse(input.world);
  const worldCheckpoint = HumanoidWorldCheckpointSchema.parse(input.worldCheckpoint);
  const {
    version: _version,
    goal,
    goal_progress: _legacyProgress,
    checker: _legacyChecker,
    world: _legacyWorld,
    world_checkpoint: _legacyWorldCheckpoint,
    ...source
  } = checkpoint;
  return HumanoidRunCheckpointV6Schema.parse({
    version: 6,
    ...source,
    world,
    world_checkpoint: worldCheckpoint,
    mission_goal: goal,
    goal_dag: createGoalDAG(),
    goal_progress: null,
    checker: null
  });
}

export const HumanoidRunCheckpointSchema = z.union([
  HumanoidRunCheckpointV6Schema,
  HumanoidRunCheckpointV5Schema,
  HumanoidRunCheckpointV4Schema
]).transform((checkpoint): HumanoidRunCheckpoint => (
  checkpoint.version === 6
    ? checkpoint
    : checkpoint.version === 5
      ? migrateHumanoidRunCheckpointV5(checkpoint)
      : migrateHumanoidRunCheckpointV5(migrateHumanoidRunCheckpointV4(checkpoint))
));

export type HumanoidCheckerResult = z.infer<typeof HumanoidCheckerResultSchema>;
