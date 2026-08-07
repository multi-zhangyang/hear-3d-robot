import { z } from "zod";
import {
  buildScenarioChunkManifest,
  scenarioChunkIntegrityIssues,
  ScenarioChunkManifestSchema
} from "./scenario-chunk.js";
import { ScenarioObjectCapabilitySchema } from "./object-capability.js";

export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() => z.union([
  z.string(),
  z.number(),
  z.boolean(),
  z.null(),
  z.array(JsonValueSchema),
  z.record(z.string(), JsonValueSchema)
]));

export type JsonValue = string | number | boolean | null | JsonValue[] | {
  [key: string]: JsonValue;
};

export const Vec3Schema = z.object({
  x: z.number().finite(),
  y: z.number().finite(),
  z: z.number().finite()
}).strict();
export type Vec3 = z.infer<typeof Vec3Schema>;

export const QuaternionSchema = z.object({
  x: z.number().finite(),
  y: z.number().finite(),
  z: z.number().finite(),
  w: z.number().finite()
}).strict();
export type Quaternion = z.infer<typeof QuaternionSchema>;

export const HUMANOID_END_EFFECTORS = [
  "left_wrist",
  "right_wrist",
  "left_ankle",
  "right_ankle"
] as const;

export const HumanoidEndEffectorSchema = z.enum(HUMANOID_END_EFFECTORS);
export type HumanoidEndEffector = z.infer<typeof HumanoidEndEffectorSchema>;

const Size3Schema = Vec3Schema.refine(
  ({ x, y, z: depth }) => x > 0 && y > 0 && depth > 0,
  "size components must be positive"
);

const GoalPredicateUnionSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("robot_at"),
    target: Vec3Schema,
    tolerance: z.number().finite().positive()
  }).strict(),
  z.object({
    type: z.literal("robot_in_zone"),
    zone_id: z.string().trim().min(1),
    tolerance: z.number().finite().nonnegative()
  }).strict(),
  z.object({
    type: z.literal("block_removed"),
    block_id: z.string().trim().min(1)
  }).strict(),
  z.object({
    type: z.literal("object_in_zone"),
    object_id: z.string().trim().min(1),
    zone_id: z.string().trim().min(1),
    expected: z.boolean(),
    tolerance: z.number().finite().nonnegative()
  }).strict(),
  z.object({
    type: z.literal("object_placed"),
    object_id: z.string().trim().min(1),
    zone_id: z.string().trim().min(1),
    tolerance: z.number().finite().nonnegative()
  }).strict(),
  z.object({
    type: z.literal("object_at"),
    object_id: z.string().trim().min(1),
    target: Vec3Schema,
    tolerance: z.number().finite().positive()
  }).strict(),
  z.object({
    type: z.literal("object_grasped"),
    object_id: z.string().trim().min(1),
    hand: z.enum(["left", "right", "either"])
  }).strict(),
  z.object({
    type: z.literal("object_inside"),
    object_id: z.string().trim().min(1),
    container_id: z.string().trim().min(1),
    expected: z.boolean(),
    tolerance: z.number().finite().nonnegative()
  }).strict().refine(
    (predicate) => predicate.object_id !== predicate.container_id,
    { path: ["container_id"], message: "object cannot be inside itself" }
  ),
  z.object({
    type: z.literal("object_on"),
    object_id: z.string().trim().min(1),
    support_id: z.string().trim().min(1),
    expected: z.boolean(),
    tolerance: z.number().finite().nonnegative()
  }).strict().refine(
    (predicate) => predicate.object_id !== predicate.support_id,
    { path: ["support_id"], message: "object cannot support itself" }
  ),
  z.object({
    type: z.literal("articulation_state"),
    object_id: z.string().trim().min(1),
    joint_id: z.string().trim().min(1),
    state: z.enum(["open", "closed"]),
    tolerance: z.number().finite().min(0).max(0.49)
  }).strict(),
  z.object({
    type: z.literal("end_effector_at"),
    end_effector: HumanoidEndEffectorSchema,
    frame: z.enum(["world", "pelvis"]),
    target: Vec3Schema,
    tolerance: z.number().finite().positive().max(5),
    stable_frames: z.number().int().min(1).max(500),
    orientation: QuaternionSchema.optional(),
    orientation_tolerance_rad: z.number().finite().positive().max(Math.PI).optional()
  }).strict()
]);

export const GoalPredicateSchema = GoalPredicateUnionSchema.superRefine(
  (predicate, context) => {
    if (predicate.type !== "end_effector_at") return;
    const hasOrientation = predicate.orientation !== undefined;
    const hasOrientationTolerance = predicate.orientation_tolerance_rad !== undefined;
    if (hasOrientation !== hasOrientationTolerance) {
      context.addIssue({
        code: "custom",
        path: [hasOrientation ? "orientation_tolerance_rad" : "orientation"],
        message: "End-effector orientation and tolerance must be provided together"
      });
    }
    if (predicate.orientation
      && Math.hypot(
        predicate.orientation.x,
        predicate.orientation.y,
        predicate.orientation.z,
        predicate.orientation.w
      ) <= 1e-9) {
      context.addIssue({
        code: "custom",
        path: ["orientation"],
        message: "End-effector orientation must be a non-zero quaternion"
      });
    }
  }
);
export type GoalPredicate = z.infer<typeof GoalPredicateSchema>;

export const GoalSchema = z.object({
  summary: z.string().trim().min(1),
  predicates: z.array(GoalPredicateSchema).min(1)
}).strict();
export type Goal = z.infer<typeof GoalSchema>;

const ScenarioObjectSchema = z.object({
  id: z.string().trim().min(1),
  kind: z.string().trim().min(1),
  color: z.string().trim().min(1),
  position: Vec3Schema,
  size: Size3Schema,
  portable: z.boolean(),
  capability: ScenarioObjectCapabilitySchema.optional()
}).strict();

const ScenarioZoneSchema = z.object({
  id: z.string().trim().min(1),
  color: z.string().trim().min(1),
  center: Vec3Schema,
  size: Size3Schema
}).strict();

const RobotStartSchema = z.object({
  x: z.number().finite(),
  z: z.number().finite(),
  yaw: z.number().finite()
}).strict();

const ScenarioCoreSchema = z.object({
  title: z.string().trim().min(1),
  seed: z.number().int().nonnegative(),
  bounds: z.object({
    width: z.number().finite().positive(),
    depth: z.number().finite().positive()
  }).strict(),
  visibility_radius: z.number().finite().positive(),
  robot: RobotStartSchema,
  obstacles: z.array(z.object({
    id: z.string().trim().min(1),
    center: Vec3Schema,
    size: Size3Schema
  }).strict()),
  objects: z.array(ScenarioObjectSchema),
  zones: z.array(ScenarioZoneSchema),
  default_goal: GoalSchema
}).strict();

export const ScenarioSchema = ScenarioCoreSchema.extend({
  chunk_manifest: ScenarioChunkManifestSchema.optional()
}).transform((scenario) => ({
  ...scenario,
  chunk_manifest: scenario.chunk_manifest ?? buildScenarioChunkManifest(scenario)
})).superRefine((scenario, context) => {
  for (const message of scenarioChunkIntegrityIssues(scenario, scenario.chunk_manifest)) {
    context.addIssue({
      code: "custom",
      path: ["chunk_manifest"],
      message
    });
  }
});
export type Scenario = z.infer<typeof ScenarioSchema>;

const ProceduralWorldShapeSchema = z.object({
  bounds: z.object({
    width: z.number().finite().min(16).max(96),
    depth: z.number().finite().min(16).max(96)
  }).strict(),
  cell: z.number().finite().min(2.4).max(4),
  obstacle_density: z.number().finite().min(0).max(0.32),
  minimum_obstacle_height: z.number().finite().min(0.35).max(2.5),
  maximum_obstacle_height: z.number().finite().min(0.5).max(4),
  visibility_radius: z.number().finite().positive(),
  objects: z.array(ScenarioObjectSchema.omit({ position: true })),
  zones: z.array(ScenarioZoneSchema.omit({ center: true })),
  default_goal: GoalSchema
}).strict().refine(
  (shape) => shape.maximum_obstacle_height >= shape.minimum_obstacle_height,
  "maximum obstacle height must be greater than or equal to its minimum"
).refine(
  (shape) => Math.floor(shape.bounds.width / shape.cell) >= 6
    && Math.floor(shape.bounds.depth / shape.cell) >= 6,
  "procedural world requires at least six cells on each axis"
);

const ScenarioTemplateSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("authored"),
    runtime: z.literal("humanoid_g1"),
    title: z.string().trim().min(1),
    scenario: ScenarioCoreSchema.omit({ seed: true })
  }).strict(),
  z.object({
    kind: z.literal("procedural"),
    runtime: z.literal("humanoid_g1"),
    title: z.string().trim().min(1),
    generate: ProceduralWorldShapeSchema
  }).strict()
]);
export type ScenarioTemplate = z.infer<typeof ScenarioTemplateSchema>;

export const ScenarioCatalogSchema = z.object({
  scenarios: z.record(z.string().trim().min(1), ScenarioTemplateSchema)
}).strict();

const AgentStatusSchema = z.enum([
  "ready",
  "active",
  "waiting",
  "completed",
  "blocked",
  "failed"
]);

export const TaskNodeSchema = z.object({
  id: z.string().trim().min(1),
  name: z.string().trim().min(1),
  parent_id: z.string().trim().min(1).nullable(),
  child_ids: z.array(z.string().trim().min(1)),
  objective: z.string().trim().min(1),
  success_criteria: z.array(z.string().trim().min(1)),
  evidence_requirements: z.array(JsonValueSchema),
  goal_predicate_indexes: z.array(z.number().int().nonnegative()),
  capabilities: z.array(z.string().trim().min(1)),
  may_delegate: z.boolean(),
  references: z.array(z.object({
    name: z.string().trim().min(1),
    transaction_id: z.string().trim().min(1)
  }).strict()),
  depth: z.number().int().nonnegative(),
  status: AgentStatusSchema,
  steps_used: z.number().int().nonnegative(),
  model_calls_used: z.number().int().nonnegative(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
  last_result: JsonValueSchema.optional()
}).strict();
export type TaskNode = z.infer<typeof TaskNodeSchema>;

const ContextEvidenceSchema = z.object({
  summary: z.string().trim().min(1),
  transaction_ids: z.array(z.string().trim().min(1))
}).strict();

export const ContextCompactionSummarySchema = z.object({
  mission_state: z.string().trim().min(1),
  constraints: z.array(z.string().trim().min(1)),
  decisions: z.array(z.string().trim().min(1)),
  completed: z.array(ContextEvidenceSchema),
  pending: z.array(z.string().trim().min(1)),
  blockers: z.array(ContextEvidenceSchema),
  next_actions: z.array(z.string().trim().min(1))
}).strict();
export type ContextCompactionSummary = z.infer<typeof ContextCompactionSummarySchema>;

const ContextScopeStateSchema = z.object({
  scope_id: z.string().trim().min(1),
  agent_id: z.string().trim().min(1),
  agent_name: z.string().trim().min(1),
  raw_item_count: z.number().int().nonnegative(),
  raw_chain_hash: z.string().regex(/^[a-f0-9]{64}$/).nullable(),
  compacted_item_count: z.number().int().nonnegative(),
  retained_item_count: z.number().int().nonnegative().default(0),
  retained_chain_hash: z.string().regex(/^[a-f0-9]{64}$/).nullable().default(null),
  active_estimated_tokens: z.number().int().nonnegative(),
  token_estimator_correction_milli: z.number().int().positive().default(1_000),
  context_window_tokens: z.number().int().positive().optional(),
  compact_trigger_tokens: z.number().int().positive().optional(),
  compact_recent_model_turns: z.number().int().nonnegative().optional(),
  compact_max_output_tokens: z.number().int().positive().optional(),
  compaction_count: z.number().int().nonnegative(),
  summary: ContextCompactionSummarySchema.nullable(),
  summary_origin: z.enum(["model", "authority_projection"]).nullable().default(null),
  summary_world_revision: z.number().int().nonnegative().nullable(),
  last_compacted_at: z.string().datetime().nullable()
}).strict().superRefine((scope, context) => {
  const budgetKeys = [
    "context_window_tokens",
    "compact_trigger_tokens",
    "compact_recent_model_turns",
    "compact_max_output_tokens"
  ] as const;
  const present = budgetKeys.filter((key) => scope[key] !== undefined);
  if (present.length === 0 || present.length === budgetKeys.length) return;
  for (const key of budgetKeys) {
    if (scope[key] !== undefined) continue;
    context.addIssue({
      code: "custom",
      path: [key],
      message: "Context scope budgets must be persisted as one complete envelope"
    });
  }
});
export type ContextScopeState = z.infer<typeof ContextScopeStateSchema>;

export const ContextMemoryStateSchema = z.object({
  version: z.literal(1),
  context_window_tokens: z.number().int().positive(),
  compact_trigger_tokens: z.number().int().positive(),
  compact_recent_model_turns: z.number().int().nonnegative(),
  compact_max_output_tokens: z.number().int().positive(),
  active_scope_id: z.string().min(1).nullable(),
  active_estimated_tokens: z.number().int().nonnegative(),
  total_compactions: z.number().int().nonnegative(),
  last_compacted_at: z.string().datetime().nullable(),
  scopes: z.record(z.string(), ContextScopeStateSchema)
}).strict();
export type ContextMemoryState = z.infer<typeof ContextMemoryStateSchema>;

export const EmptyContextMemoryState: ContextMemoryState = {
  version: 1,
  context_window_tokens: 262_144,
  compact_trigger_tokens: 222_822,
  compact_recent_model_turns: 4,
  compact_max_output_tokens: 13_107,
  active_scope_id: null,
  active_estimated_tokens: 0,
  total_compactions: 0,
  last_compacted_at: null,
  scopes: {}
};

export const RunStatusSchema = z.enum([
  "starting",
  "running",
  "paused",
  "succeeded",
  "failed",
  "interrupted"
]);

const RunLifecycleEventTypeSchema = z.enum([
  "run_started",
  "run_resumed",
  "run_paused",
  "run_succeeded",
  "run_failed",
  "run_interrupted"
]);
export type RunLifecycleEventType = z.infer<typeof RunLifecycleEventTypeSchema>;

export const RunLifecycleEventSchema = z.object({
  event_id: z.string().trim().min(1),
  run_id: z.string().trim().min(1),
  type: RunLifecycleEventTypeSchema,
  at: z.string().datetime(),
  data: JsonValueSchema
}).strict();
export type RunLifecycleEvent = z.infer<typeof RunLifecycleEventSchema>;
