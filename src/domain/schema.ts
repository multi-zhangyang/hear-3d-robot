import { z } from "zod";

export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(JsonValueSchema),
    z.record(z.string(), JsonValueSchema)
  ])
);

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export const Vec3Schema = z.object({
  x: z.number().finite(),
  y: z.number().finite(),
  z: z.number().finite()
});

export type Vec3 = z.infer<typeof Vec3Schema>;

export const QuaternionSchema = z.object({
  x: z.number().finite(),
  y: z.number().finite(),
  z: z.number().finite(),
  w: z.number().finite()
});

export const Size3Schema = Vec3Schema.refine(
  ({ x, y, z: depth }) => x > 0 && y > 0 && depth > 0,
  "size components must be positive"
);

export const GoalPredicateSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("robot_at"),
    target: Vec3Schema,
    tolerance: z.number().finite().positive()
  }).strict(),
  z.object({
    type: z.literal("robot_in_zone"),
    zone_id: z.string().min(1),
    tolerance: z.number().finite().nonnegative()
  }).strict(),
  z.object({
    type: z.literal("terrain_explored"),
    minimum_fraction: z.number().finite().min(0).max(1)
  }).strict(),
  z.object({
    type: z.literal("voxel_at"),
    coordinate: z.object({
      column: z.number().int().nonnegative(),
      level: z.number().int().nonnegative(),
      row: z.number().int().nonnegative()
    }).strict(),
    material: z.enum(["grass", "dirt", "stone", "sand", "placed"]).nullable()
  }).strict(),
  z.object({
    type: z.literal("object_in_zone"),
    object_id: z.string().min(1),
    zone_id: z.string().min(1),
    expected: z.boolean(),
    tolerance: z.number().finite().nonnegative()
  }).strict(),
  z.object({
    type: z.literal("object_at"),
    object_id: z.string().min(1),
    target: Vec3Schema,
    tolerance: z.number().finite().positive()
  }).strict(),
  z.object({
    type: z.literal("object_property"),
    object_id: z.string().min(1),
    property: z.enum(["locked", "enabled"]),
    expected: z.boolean()
  }).strict(),
  z.object({
    type: z.literal("object_attached"),
    object_id: z.string().min(1),
    expected: z.boolean()
  }).strict()
]);

export const GoalSchema = z.object({
  summary: z.string().trim().min(1),
  predicates: z.array(GoalPredicateSchema).min(1)
}).strict();

export type Goal = z.infer<typeof GoalSchema>;

export const ScenarioObjectSchema = z.object({
  id: z.string().min(1),
  kind: z.string().trim().min(1),
  color: z.string().min(1),
  position: Vec3Schema,
  size: Size3Schema,
  portable: z.boolean(),
  locked: z.boolean().optional(),
  key_id: z.string().min(1).optional(),
  container_id: z.string().min(1).optional()
});

export const ScenarioZoneSchema = z.object({
  id: z.string().min(1),
  color: z.string().min(1),
  center: Vec3Schema,
  size: Size3Schema
});

export const ScenarioAffordanceSchema = z.discriminatedUnion("type", [
  z.object({
    id: z.string().trim().min(1),
    type: z.literal("keyed_lock"),
    container_id: z.string().trim().min(1),
    key_id: z.string().trim().min(1),
    socket: z.object({
      center: Vec3Schema,
      half_extents: Size3Schema,
      insertion_axis: Vec3Schema.refine(
        ({ x, y, z: depth }) => Math.hypot(x, y, depth) > 0,
        "insertion axis must be non-zero"
      ),
      maximum_axis_angle: z.number().finite().positive().max(Math.PI)
    }).strict()
  }).strict()
]);

export const RobotJointsSchema = z.object({
  head_yaw: z.number().finite(),
  head_pitch: z.number().finite(),
  shoulder: z.number().finite(),
  elbow: z.number().finite(),
  wrist: z.number().finite(),
  gripper_aperture: z.number().finite().nonnegative()
});

/**
 * Voxel terrain as one height level per grid column, row-major.
 *
 * Stored as a grid rather than as a list of boxes because the grid is what the
 * generator produces and what the world view draws, and because it stays small
 * enough to keep in the run definition whole. The merged box decomposition the
 * physics world and navmesh need is derived from it.
 */
export const TerrainSchema = z.object({
  cell: z.number().finite().positive(),
  columns: z.number().int().positive(),
  rows: z.number().int().positive(),
  block: z.number().finite().positive(),
  chunk_size: z.number().int().min(8).max(32).default(16),
  maximum_height: z.number().int().min(1).max(64).default(24),
  heights: z.array(z.number().int().nonnegative())
}).strict().refine(
  (terrain) => terrain.heights.length === terrain.columns * terrain.rows,
  "terrain heights must contain exactly columns × rows entries"
);

export type Terrain = z.infer<typeof TerrainSchema>;

export const VoxelMaterialSchema = z.enum(["grass", "dirt", "stone", "sand", "placed"]);
export type VoxelMaterial = z.infer<typeof VoxelMaterialSchema>;

export const VoxelCoordinateSchema = z.object({
  column: z.number().int().nonnegative(),
  level: z.number().int().nonnegative(),
  row: z.number().int().nonnegative()
}).strict();
export type VoxelCoordinate = z.infer<typeof VoxelCoordinateSchema>;

export const VoxelMutationSchema = z.object({
  coordinate: VoxelCoordinateSchema,
  before: VoxelMaterialSchema.nullable(),
  after: VoxelMaterialSchema.nullable(),
  revision: z.number().int().positive(),
  source_command_id: z.string().min(1),
  source_agent_id: z.string().min(1)
}).strict();
export type VoxelMutation = z.infer<typeof VoxelMutationSchema>;

export const VoxelChunkReferenceSchema = z.object({
  column: z.number().int().nonnegative(),
  row: z.number().int().nonnegative()
}).strict();

export const VoxelWorldStateSchema = z.object({
  version: z.literal(1),
  revision: z.number().int().nonnegative(),
  chunk_size: z.number().int().min(8).max(32),
  load_radius_chunks: z.number().int().min(1).max(8),
  loaded_chunks: z.array(VoxelChunkReferenceSchema),
  mutations: z.array(VoxelMutationSchema),
  inventory: z.record(VoxelMaterialSchema, z.number().int().nonnegative())
}).strict();
export type VoxelWorldState = z.infer<typeof VoxelWorldStateSchema>;

export const RobotStartSchema = z.object({
  x: z.number().finite(),
  z: z.number().finite(),
  yaw: z.number().finite(),
  /**
   * The configuration the arm and head begin in. Optional so an authored
   * scenario can omit it and start from the neutral stance; a generated world
   * always supplies one, because starting every run in the same pose is one of
   * the things that made successive runs look alike.
   */
  joints: RobotJointsSchema.optional()
});

export const ScenarioSchema = z.object({
  title: z.string().min(1),
  /**
   * The integer every generated aspect of this world was drawn from. Stored so
   * a run is a record of the world it actually faced: resuming rebuilds this
   * world, and two runs of the same template differ exactly here.
   */
  seed: z.number().int().nonnegative(),
  /**
   * Independent entropy for ordering model-facing movement choices. It is
   * deliberately separate from `seed`: fixing a world for inspection should
   * not force every new autonomous run through the same route. The value is
   * persisted with the concrete scenario so a resumed run keeps the choices
   * it had already observed.
   */
  motion_seed: z.number().int().nonnegative().default(0),
  bounds: z.object({ width: z.number().positive(), depth: z.number().positive() }),
  terrain: TerrainSchema.optional(),
  visibility_radius: z.number().positive(),
  robot: RobotStartSchema,
  obstacles: z.array(z.object({
    id: z.string().min(1),
    center: Vec3Schema,
    size: Size3Schema
  })),
  objects: z.array(ScenarioObjectSchema),
  zones: z.array(ScenarioZoneSchema),
  affordances: z.array(ScenarioAffordanceSchema).default([]),
  default_goal: GoalSchema
});

export type Scenario = z.infer<typeof ScenarioSchema>;

/** The shape of the voxel field a generated template asks for. */
export const TerrainShapeSchema = z.object({
  size: z.number().int().min(8).max(384),
  cell: z.number().finite().positive(),
  block: z.number().finite().positive(),
  chunk_size: z.number().int().min(8).max(32).default(16),
  maximum_height: z.number().int().min(1).max(64).default(24),
  relief: z.number().int().min(1).max(6),
  density: z.number().min(0).max(0.6)
}).strict();

/**
 * A world description with the positions left out.
 *
 * `authored` is a scenario written by hand, fixed in every detail. `generated`
 * says what kind of world to build and lets the seed decide where everything
 * sits, so selecting it twice gives two different worlds.
 */
export const ScenarioTemplateSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("authored"),
    title: z.string().min(1),
    scenario: ScenarioSchema.omit({ seed: true }).extend({
      seed: z.number().int().nonnegative().default(0)
    })
  }).strict(),
  z.object({
    kind: z.literal("generated"),
    title: z.string().min(1),
    generate: z.object({
      terrain: TerrainShapeSchema,
      visibility_radius: z.number().positive(),
      objects: z.array(ScenarioObjectSchema.omit({ position: true })),
      zones: z.array(ScenarioZoneSchema.omit({ center: true })),
      default_goal: GoalSchema
    }).strict()
  }).strict()
]);

export type ScenarioTemplate = z.infer<typeof ScenarioTemplateSchema>;

export const ScenarioCatalogSchema = z.object({
  scenarios: z.record(z.string(), ScenarioTemplateSchema)
});

export const BodyChannelSchema = z.enum(["base", "head", "arm", "gripper"]);
export type BodyChannel = z.infer<typeof BodyChannelSchema>;

export const EvidenceEffectSchema = z.enum([
  "observation",
  "memory",
  "plan",
  "body_motion",
  "world_mutation"
]);
export type EvidenceEffect = z.infer<typeof EvidenceEffectSchema>;

export const EvidenceFreshnessSchema = z.enum(["current_world", "historical_record"]);
export type EvidenceFreshness = z.infer<typeof EvidenceFreshnessSchema>;

export const EvidenceTargetSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("world") }).strict(),
  z.object({ kind: z.literal("robot") }).strict(),
  z.object({ kind: z.literal("terrain") }).strict(),
  z.object({ kind: z.literal("entity"), entity_id: z.string().trim().min(1) }).strict(),
  z.object({ kind: z.literal("voxel"), coordinate: VoxelCoordinateSchema }).strict(),
  z.object({ kind: z.literal("position"), position: Vec3Schema }).strict(),
  z.object({ kind: z.literal("body"), channel: BodyChannelSchema }).strict()
]);
export type EvidenceTarget = z.infer<typeof EvidenceTargetSchema>;

export const EvidenceRequirementSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("receipt"),
    criterion_index: z.number().int().nonnegative(),
    actions: z.array(z.string().trim().min(1)).min(1),
    effect: EvidenceEffectSchema,
    target: EvidenceTargetSchema,
    freshness: EvidenceFreshnessSchema
  }).strict(),
  z.object({
    kind: z.literal("goal_predicate"),
    criterion_index: z.number().int().nonnegative(),
    predicate_index: z.number().int().nonnegative()
  }).strict()
]);
export type EvidenceRequirement = z.infer<typeof EvidenceRequirementSchema>;

export const AgentReferenceSchema = z.object({
  name: z.string().trim().min(1),
  transaction_id: z.string().trim().min(1)
}).strict();

export type AgentReference = z.infer<typeof AgentReferenceSchema>;

export const AgentSpecSchema = z.object({
  name: z.string().trim().min(1),
  objective: z.string().trim().min(1),
  success_criteria: z.array(z.string().trim().min(1)).min(1),
  evidence_requirements: z.array(EvidenceRequirementSchema).default([]),
  // Supervisors own explicit final-state predicates rather than merely
  // claiming them in prose. The harness evaluates these indexes against the
  // live world before accepting complete_assignment.
  goal_predicate_indexes: z.array(z.number().int().nonnegative()).default([]),
  capabilities: z.array(z.string().trim().min(1)),
  may_delegate: z.boolean(),
  references: z.array(AgentReferenceSchema)
}).strict();

export type AgentSpec = z.infer<typeof AgentSpecSchema>;

export const AgentStatusSchema = z.enum([
  "ready",
  "active",
  "waiting",
  "completed",
  "blocked",
  "failed"
]);

export const TaskNodeSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  parent_id: z.string().min(1).nullable(),
  source_call_id: z.string().min(1).optional(),
  child_ids: z.array(z.string().min(1)),
  objective: z.string().min(1),
  success_criteria: z.array(z.string().min(1)),
  evidence_requirements: z.array(EvidenceRequirementSchema).default([]),
  goal_predicate_indexes: z.array(z.number().int().nonnegative()).default([]),
  capabilities: z.array(z.string().min(1)),
  may_delegate: z.boolean(),
  references: z.array(AgentReferenceSchema),
  depth: z.number().int().nonnegative(),
  status: AgentStatusSchema,
  steps_used: z.number().int().nonnegative(),
  model_calls_used: z.number().int().nonnegative(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
  last_result: JsonValueSchema.optional()
});

export type TaskNode = z.infer<typeof TaskNodeSchema>;

const LinkStateSchema = z.object({
  position: Vec3Schema,
  rotation: QuaternionSchema,
  linear_velocity: Vec3Schema,
  angular_velocity: Vec3Schema
}).strict();

const JointStatusSchema = z.object({
  position: z.number().finite(),
  velocity: z.number().finite(),
  target: z.number().finite(),
  minimum: z.number().finite(),
  maximum: z.number().finite(),
  maximum_velocity: z.number().finite().positive()
}).strict();

const CommandStateSchema = z.object({
  id: z.string().min(1),
  agent_id: z.string().min(1),
  agent_name: z.string().min(1),
  skill: z.string().min(1),
  channels: z.array(BodyChannelSchema),
  phase: z.string().min(1),
  target: JsonValueSchema.optional(),
  focus: JsonValueSchema.optional()
}).strict();

export const WorldObjectStateSchema = z.object({
  id: z.string(),
  kind: z.string(),
  color: z.string(),
  position: Vec3Schema,
  rotation: QuaternionSchema,
  linear_velocity: Vec3Schema,
  angular_velocity: Vec3Schema,
  size: Size3Schema,
  portable: z.boolean(),
  locked: z.boolean(),
  container_id: z.string().nullable(),
  enabled: z.boolean(),
  visible: z.boolean()
});

export const WorldSnapshotSchema = z.object({
  frame: z.number().int().nonnegative(),
  simulated_time: z.number().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  robot: z.object({
    position: Vec3Schema,
    yaw: z.number().finite(),
    joints: RobotJointsSchema,
    contacts: z.object({
      left_object_id: z.string().nullable(),
      right_object_id: z.string().nullable(),
      left_force: z.number().finite().nonnegative(),
      right_force: z.number().finite().nonnegative()
    }).strict(),
    attachment: z.object({
      object_id: z.string().min(1),
      constraint_id: z.string().min(1),
      source_command_id: z.string().min(1)
    }).strict().nullable(),
    odometry: z.object({
      left_wheel: z.object({
        position: z.number().finite(),
        velocity: z.number().finite()
      }).strict(),
      right_wheel: z.object({
        position: z.number().finite(),
        velocity: z.number().finite()
      }).strict()
    }).strict(),
    links: z.record(z.string(), LinkStateSchema),
    joint_status: z.record(z.string(), JointStatusSchema),
    gripper: z.object({
      aperture: z.number().finite().nonnegative(),
      target_aperture: z.number().finite().nonnegative(),
      maximum_force: z.number().finite().positive(),
      left_contact_object_id: z.string().nullable(),
      right_contact_object_id: z.string().nullable(),
      left_contact_force: z.number().finite().nonnegative(),
      right_contact_force: z.number().finite().nonnegative()
    }).strict()
  }),
  objects: z.array(WorldObjectStateSchema),
  zones: z.array(ScenarioZoneSchema),
  obstacles: ScenarioSchema.shape.obstacles,
  /**
   * How much of the world the robot has actually seen.
   *
   * The seen set is a bitmap, one bit per terrain column, base64-encoded. A
   * frame has to carry the whole set rather than what changed, because a
   * checkpoint is one frame and resuming from it must recover the full frontier
   * — but the set also appears in every streamed frame, so listing indices
   * would cost more than the rest of the frame put together. A few hundred
   * bytes of bitmap is complete and cheap at both.
   */
  explored: z.object({
    cells: z.string(),
    seen: z.number().int().nonnegative(),
    total: z.number().int().nonnegative()
  }).strict(),
  /** Mutable voxel overlay plus the backend chunks currently present in physics. */
  voxels: VoxelWorldStateSchema.nullable().default(null),
  navigation: z.object({
    plan_id: z.string().nullable(),
    status: z.enum(["idle", "planned", "executing", "completed", "blocked", "stopped"]),
    target: Vec3Schema.nullable(),
    face: Vec3Schema.nullable(),
    waypoints: z.array(Vec3Schema),
    waypoint_index: z.number().int().nonnegative().nullable(),
    distance: z.number().finite().nonnegative().nullable(),
    planned_at_frame: z.number().int().nonnegative().nullable(),
    actual_path: z.array(Vec3Schema)
  }).strict(),
  plans: z.object({
    base: z.array(z.object({
      id: z.string().min(1),
      created_revision: z.number().int().nonnegative(),
      target: Vec3Schema,
      face: Vec3Schema.nullable(),
      waypoints: z.array(Vec3Schema),
      distance: z.number().finite().nonnegative()
    }).strict()),
    arm: z.array(z.object({
      id: z.string().min(1),
      created_revision: z.number().int().nonnegative(),
      /**
       * Older checkpoints contain only end-effector plans and therefore have
       * no discriminator. Defaulting keeps those snapshots resumable while a
       * joint-space plan can explicitly avoid pretending it owns a fixed
       * world-space point.
       */
      kind: z.enum(["end_effector", "joint_targets"]).default("end_effector"),
      target: z.object({
        position: Vec3Schema,
        orientation: QuaternionSchema.optional(),
        seed: z.object({
          shoulder: z.number().finite().optional(),
          elbow: z.number().finite().optional(),
          wrist: z.number().finite().optional()
        }).strict().optional()
      }).strict().nullable(),
      joints: z.object({
        shoulder: z.number().finite(),
        elbow: z.number().finite(),
        wrist: z.number().finite()
      }).strict(),
      waypoints: z.array(z.object({
        shoulder: z.number().finite(),
        elbow: z.number().finite(),
        wrist: z.number().finite()
      }).strict()).default([])
    }).strict())
  }).strict(),
  affordance_events: z.array(z.object({
    frame: z.number().int().nonnegative(),
    affordance_id: z.string().nullable(),
    code: z.string().min(1),
    entity_id: z.string().min(1),
    source_command_id: z.string().nullable(),
    detail: JsonValueSchema
  }).strict()),
  last_command: CommandStateSchema.extend({
    accepted: z.boolean(),
    result_code: z.string().min(1),
    ended_at_frame: z.number().int().nonnegative()
  }).strict().nullable(),
  active_command: CommandStateSchema.nullable(),
  active_commands: z.array(CommandStateSchema).default([])
});

export type WorldSnapshot = z.infer<typeof WorldSnapshotSchema>;

export const CheckerResultSchema = z.object({
  success: z.boolean(),
  goal: GoalSchema,
  world_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  checks: z.array(z.object({
    name: z.string(),
    passed: z.boolean(),
    actual: JsonValueSchema
  })),
  checked_at: z.string().datetime()
});

export type CheckerResult = z.infer<typeof CheckerResultSchema>;

export const ActionReceiptSchema = z.object({
  transaction_id: z.string().min(1),
  agent_id: z.string().min(1),
  agent_name: z.string().min(1),
  kind: z.enum(["tool", "skill", "checker"]),
  name: z.string().min(1),
  input: JsonValueSchema,
  accepted: z.boolean(),
  code: z.string().min(1),
  detail: JsonValueSchema,
  world_before_frame: z.number().int().nonnegative(),
  world_before_revision: z.number().int().nonnegative().optional(),
  world_after_frame: z.number().int().nonnegative(),
  frame_count: z.number().int().nonnegative(),
  // The revision the measurement in `detail` describes. A receipt handed to
  // another node is only usable if that node can tell whether the world has
  // moved since — an object's position read before the robot carried it is a
  // true record and a wrong target.
  world_revision: z.number().int().nonnegative(),
  channels: z.array(BodyChannelSchema),
  gates: z.array(z.object({
    name: z.string().min(1),
    status: z.enum(["passed", "rejected"]),
    detail: JsonValueSchema
  }).strict()),
  committed_at: z.string().datetime()
});

export type ActionReceipt = z.infer<typeof ActionReceiptSchema>;

export const SpatialMemoryKindSchema = z.enum([
  "robot_pose",
  "entity",
  "voxel",
  "terrain",
  "navigation",
  "contact"
]);
export type SpatialMemoryKind = z.infer<typeof SpatialMemoryKindSchema>;

export const SpatialMemoryRecordSchema = z.object({
  id: z.string().min(1),
  world_id: z.string().min(1),
  kind: SpatialMemoryKindSchema,
  label: z.string().min(1),
  summary: z.string().min(1),
  position: Vec3Schema.nullable(),
  coordinate: VoxelCoordinateSchema.nullable(),
  entity_id: z.string().min(1).nullable(),
  data: JsonValueSchema,
  observed_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  voxel_revision: z.number().int().nonnegative().nullable(),
  source_transaction_id: z.string().min(1),
  source_agent_id: z.string().min(1),
  source_action: z.string().min(1),
  recorded_at: z.string().datetime()
}).strict();
export type SpatialMemoryRecord = z.infer<typeof SpatialMemoryRecordSchema>;

const ContextEvidenceSchema = z.object({
  summary: z.string().trim().min(1),
  transaction_ids: z.array(z.string().trim().min(1))
}).strict();

/**
 * Model-written working memory for one hierarchy node.
 *
 * This is deliberately not a source of truth. Transaction ids are validated
 * against committed receipts before a summary is accepted, while the current
 * world identity is injected separately by the harness on every model call.
 */
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

export const ContextScopeStateSchema = z.object({
  scope_id: z.string().min(1),
  agent_id: z.string().min(1),
  agent_name: z.string().min(1),
  raw_item_count: z.number().int().nonnegative(),
  raw_chain_hash: z.string().regex(/^[a-f0-9]{64}$/).nullable(),
  compacted_item_count: z.number().int().nonnegative(),
  retained_item_count: z.number().int().nonnegative().default(0),
  retained_chain_hash: z.string().regex(/^[a-f0-9]{64}$/).nullable().default(null),
  active_estimated_tokens: z.number().int().nonnegative(),
  compaction_count: z.number().int().nonnegative(),
  summary: ContextCompactionSummarySchema.nullable(),
  summary_origin: z.enum(["model", "authority_projection"]).nullable().default(null),
  summary_world_revision: z.number().int().nonnegative().nullable(),
  summary_voxel_revision: z.number().int().nonnegative().nullable(),
  last_compacted_at: z.string().datetime().nullable()
}).strict();
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
  context_window_tokens: 65_536,
  compact_trigger_tokens: 18_000,
  compact_recent_model_turns: 4,
  compact_max_output_tokens: 4_096,
  active_scope_id: null,
  active_estimated_tokens: 0,
  total_compactions: 0,
  last_compacted_at: null,
  scopes: {}
};

export const InflightActionSchema = z.object({
  transaction_id: z.string().min(1),
  agent_id: z.string().min(1),
  agent_name: z.string().min(1),
  kind: z.literal("skill"),
  name: z.string().min(1),
  input: JsonValueSchema,
  channels: z.array(BodyChannelSchema),
  world_before_frame: z.number().int().nonnegative(),
  world_before_revision: z.number().int().nonnegative().optional(),
  started_at: z.string().datetime()
}).strict();

export const RunStatusSchema = z.enum([
  "starting",
  "running",
  "succeeded",
  "failed",
  "interrupted"
]);

export const RunLifecycleEventTypeSchema = z.enum([
  "run_started",
  "run_resumed",
  "run_succeeded",
  "run_failed",
  "run_interrupted"
]);
export type RunLifecycleEventType = z.infer<typeof RunLifecycleEventTypeSchema>;

export const RunLifecycleEventSchema = z.object({
  event_id: z.string().min(1),
  run_id: z.string().min(1),
  type: RunLifecycleEventTypeSchema,
  at: z.string().datetime(),
  data: JsonValueSchema
}).strict();
export type RunLifecycleEvent = z.infer<typeof RunLifecycleEventSchema>;

export const RunCheckpointSchema = z.object({
  version: z.literal(3),
  run_id: z.string(),
  scenario_id: z.string(),
  goal: GoalSchema,
  capability_catalog: z.array(z.string().min(1)),
  status: RunStatusSchema,
  root_id: z.string(),
  active_agent_id: z.string().nullable(),
  active_agent_ids: z.array(z.string().min(1)).default([]),
  nodes: z.record(z.string(), TaskNodeSchema),
  world: WorldSnapshotSchema,
  inflight_action: InflightActionSchema.nullable(),
  inflight_actions: z.record(z.string(), InflightActionSchema).default({}),
  committed_actions: z.record(z.string(), ActionReceiptSchema),
  spatial_memory: z.array(SpatialMemoryRecordSchema).default([]),
  context_memory: ContextMemoryStateSchema.default(EmptyContextMemoryState),
  pending_lifecycle_events: z.array(RunLifecycleEventSchema).default([]),
  total_model_calls: z.number().int().nonnegative(),
  checker: CheckerResultSchema.nullable(),
  final_output: z.string().nullable(),
  error: z.string().nullable(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
});

export type RunCheckpoint = z.infer<typeof RunCheckpointSchema>;
