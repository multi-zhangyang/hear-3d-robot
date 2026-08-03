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
import { HumanoidWorldCheckpointSchema } from "../world/humanoid/checkpoint.js";
import { HumanoidWorldSnapshotSchema } from "../world/humanoid/snapshot-schema.js";

const HumanoidActionNameSchema = z.enum(HUMANOID_ACTION_NAMES);

const HumanoidBodyChannelSchema = z.enum([
  "locomotion",
  "torso",
  "left_arm",
  "right_arm"
]);

const PersistedHumanoidActionReceiptSchema = z.object({
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

export const HumanoidEmbodiedEpisodeSchema = z.object({
  sequence: z.number().int().positive(),
  transaction_id: z.string().min(1),
  action: z.enum(["execute_whole_body_motion", "execute_humanoid_navigation"]),
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
}).strict();

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

export const HumanoidRunCheckpointSchema = z.object({
  version: z.literal(4),
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
}).strict();

export type HumanoidRunCheckpoint = z.infer<typeof HumanoidRunCheckpointSchema>;
export type HumanoidCheckerResult = z.infer<typeof HumanoidCheckerResultSchema>;
