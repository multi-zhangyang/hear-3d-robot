import { z } from "zod";
import {
  ContextMemoryStateSchema,
  GoalSchema,
  JsonValueSchema,
  RunLifecycleEventSchema,
  RunStatusSchema,
  TaskNodeSchema,
  Vec3Schema
} from "./schema.js";
import { HUMANOID_ACTION_NAMES } from "./humanoid-action.js";
import { goalSha256 } from "./goal-identity.js";
import { HumanoidWorldCheckpointSchema } from "../world/humanoid/checkpoint.js";
import { HumanoidWorldSnapshotSchema } from "../world/humanoid/snapshot-schema.js";

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

const HumanoidCheckerResultSchema = z.object({
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

export const HumanoidEmbodiedEpisodeSchema = z.object({
  sequence: z.number().int().positive(),
  source_ref: z.string().regex(/^episode:[1-9]\d*$/).optional(),
  transaction_id: z.string().min(1),
  action: z.enum(["execute_whole_body_motion", "execute_humanoid_navigation"]),
  planning_action: z.enum([
    "plan_whole_body_motion",
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
  ]).optional(),
  candidate_count: z.number().int().min(2).max(3).optional(),
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
  code: z.string().min(1),
  model_summary: z.string().trim().min(1),
  world_before_revision: z.number().int().nonnegative(),
  world_after_revision: z.number().int().nonnegative(),
  frame_count: z.number().int().nonnegative(),
  result_frame: z.number().int().nonnegative(),
  result_root_position: Vec3Schema,
  fallen: z.boolean(),
  support: z.enum(["none", "left", "right", "double"]),
  upright: z.number().finite(),
  goal_success: z.boolean(),
  recorded_at: z.string().datetime()
}).strict().superRefine((episode, context) => {
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
    if (episode.action !== "execute_whole_body_motion"
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

export const HumanoidEmbodiedMemoryStateSchema = z.object({
  version: z.literal(1),
  total_episodes: z.number().int().nonnegative(),
  pruned_episodes: z.number().int().nonnegative(),
  recent_episodes: z.array(HumanoidEmbodiedEpisodeSchema).max(64)
}).strict();

export type HumanoidEmbodiedEpisode = z.infer<typeof HumanoidEmbodiedEpisodeSchema>;
export type HumanoidEmbodiedMemoryState = z.infer<
  typeof HumanoidEmbodiedMemoryStateSchema
>;

export const EmptyHumanoidEmbodiedMemoryState: HumanoidEmbodiedMemoryState = {
  version: 1,
  total_episodes: 0,
  pruned_episodes: 0,
  recent_episodes: []
};

const HumanoidRunCheckpointCommonShape = {
  runtime: z.literal("humanoid_g1"),
  run_id: z.string().min(1),
  scenario_id: z.string().min(1),
  goal: GoalSchema,
  capability_catalog: z.array(z.string().min(1)),
  status: RunStatusSchema,
  root_id: z.string().min(1),
  active_agent_id: z.string().min(1).nullable(),
  active_agent_ids: z.array(z.string().min(1)),
  nodes: z.record(z.string().min(1), TaskNodeSchema),
  world: HumanoidWorldSnapshotSchema,
  world_checkpoint: HumanoidWorldCheckpointSchema,
  committed_actions: z.record(z.string().min(1), PersistedHumanoidActionReceiptSchema),
  context_memory: ContextMemoryStateSchema,
  embodied_memory: HumanoidEmbodiedMemoryStateSchema,
  pending_lifecycle_events: z.array(RunLifecycleEventSchema),
  cycle_index: z.number().int().nonnegative(),
  total_model_calls: z.number().int().nonnegative(),
  checker: HumanoidCheckerResultSchema.nullable(),
  last_cycle: JsonValueSchema.nullable(),
  final_output: z.string().nullable(),
  error: z.string().nullable(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
} as const;

export const HumanoidRunCheckpointV4Schema = z.object({
  version: z.literal(4),
  ...HumanoidRunCheckpointCommonShape
}).strict();

export type HumanoidRunCheckpointV4 = z.infer<
  typeof HumanoidRunCheckpointV4Schema
>;

export const HumanoidRunCheckpointV5Schema = z.object({
  version: z.literal(5),
  ...HumanoidRunCheckpointCommonShape,
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

export type HumanoidRunCheckpoint = z.infer<
  typeof HumanoidRunCheckpointV5Schema
>;

function migrateHumanoidRunCheckpointV4(
  checkpoint: HumanoidRunCheckpointV4
): HumanoidRunCheckpoint {
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

export const HumanoidRunCheckpointSchema = z.union([
  HumanoidRunCheckpointV5Schema,
  HumanoidRunCheckpointV4Schema
]).transform((checkpoint): HumanoidRunCheckpoint => (
  checkpoint.version === 5
    ? checkpoint
    : migrateHumanoidRunCheckpointV4(checkpoint)
));

export type HumanoidCheckerResult = z.infer<typeof HumanoidCheckerResultSchema>;
