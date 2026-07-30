export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

export type BodyChannel = "base" | "head" | "arm" | "gripper";
export type VoxelMaterial = "grass" | "dirt" | "stone" | "sand" | "placed";

export interface VoxelCoordinate {
  column: number;
  level: number;
  row: number;
}

export interface VoxelMutation {
  coordinate: VoxelCoordinate;
  before: VoxelMaterial | null;
  after: VoxelMaterial | null;
  revision: number;
  source_command_id: string;
  source_agent_id: string;
}

export interface VoxelWorldState {
  version: 1;
  revision: number;
  chunk_size: number;
  load_radius_chunks: number;
  loaded_chunks: Array<{ column: number; row: number }>;
  mutations: VoxelMutation[];
  inventory: Record<VoxelMaterial, number>;
}

export interface CommandFocus {
  position: Vec3;
  kind?: string;
  id?: string;
  label?: string;
}

export interface RobotContactState {
  left_object_id: string | null;
  right_object_id: string | null;
  left_force: number;
  right_force: number;
}

export interface RobotAttachmentState {
  object_id: string;
  constraint_id: string;
  source_command_id: string;
}

export interface WheelOdometry {
  position: number;
  velocity: number;
}

export interface RobotOdometryState {
  left_wheel: WheelOdometry;
  right_wheel: WheelOdometry;
}

export interface RobotJointState {
  position: number;
  velocity: number;
  target: number;
  minimum: number;
  maximum: number;
  maximum_velocity: number;
}

export type GoalPredicate =
  | { type: "robot_at"; target: Vec3; tolerance: number }
  | { type: "robot_in_zone"; zone_id: string; tolerance: number }
  | { type: "terrain_explored"; minimum_fraction: number }
  | { type: "voxel_at"; coordinate: VoxelCoordinate; material: VoxelMaterial | null }
  | { type: "object_in_zone"; object_id: string; zone_id: string; expected: boolean; tolerance: number }
  | { type: "object_at"; object_id: string; target: Vec3; tolerance: number }
  | { type: "object_property"; object_id: string; property: "locked" | "enabled"; expected: boolean }
  | { type: "object_attached"; object_id: string; expected: boolean };

export interface Goal {
  summary: string;
  predicates: GoalPredicate[];
}

export interface RobotLinkState {
  position: Vec3;
  rotation: Quaternion;
  linear_velocity: Vec3;
  angular_velocity: Vec3;
}

export interface NavigationState {
  plan_id: string | null;
  status: "idle" | "planned" | "executing" | "completed" | "blocked" | "stopped";
  target: Vec3 | null;
  face: Vec3 | null;
  waypoints: Vec3[];
  waypoint_index: number | null;
  distance: number | null;
  planned_at_frame: number | null;
  actual_path: Vec3[];
}

export interface GripperState {
  aperture: number;
  target_aperture: number;
  maximum_force: number;
  left_contact_object_id: string | null;
  right_contact_object_id: string | null;
  left_contact_force: number;
  right_contact_force: number;
}

export interface WorldSnapshot {
  frame: number;
  simulated_time: number;
  world_revision: number;
  robot: {
    position: Vec3;
    yaw: number;
    joints: {
      head_yaw: number;
      head_pitch: number;
      shoulder: number;
      elbow: number;
      wrist: number;
      gripper_aperture: number;
    };
    contacts: RobotContactState;
    attachment: RobotAttachmentState | null;
    odometry: RobotOdometryState;
    joint_status: Record<string, RobotJointState>;
    links: Record<string, RobotLinkState>;
    gripper: GripperState;
  };
  objects: Array<{
    id: string;
    kind: string;
    color: string;
    position: Vec3;
    rotation: Quaternion;
    linear_velocity: Vec3;
    angular_velocity: Vec3;
    size: Vec3;
    portable: boolean;
    locked: boolean;
    container_id: string | null;
    enabled: boolean;
    visible: boolean;
  }>;
  zones: Array<{ id: string; color: string; center: Vec3; size: Vec3 }>;
  obstacles: Array<{ id: string; center: Vec3; size: Vec3 }>;
  explored: { cells: string; seen: number; total: number };
  /** Optional only for checkpoints created before editable voxel worlds. */
  voxels?: VoxelWorldState | null;
  active_command: CommandState | null;
  active_commands?: CommandState[];
  last_command: (CommandState & {
    accepted: boolean;
    result_code: string;
    ended_at_frame: number;
  }) | null;
  navigation: NavigationState;
  plans: {
    base: Array<{
      id: string;
      created_revision: number;
      target: Vec3;
      face: Vec3 | null;
      waypoints: Vec3[];
      distance: number;
    }>;
    arm: Array<{
      id: string;
      created_revision: number;
      kind: "end_effector" | "joint_targets";
      target: {
        position: Vec3;
        orientation?: Quaternion;
        seed?: { shoulder?: number; elbow?: number; wrist?: number };
      } | null;
      joints: { shoulder: number; elbow: number; wrist: number };
      waypoints?: Array<{ shoulder: number; elbow: number; wrist: number }>;
    }>;
  };
  affordance_events: AffordanceEvent[];
}

export interface AffordanceEvent {
  frame: number;
  affordance_id: string | null;
  code: string;
  entity_id: string;
  source_command_id: string | null;
  detail: unknown;
}

export interface CommandState {
  id: string;
  agent_id: string;
  agent_name: string;
  skill: string;
  phase: string;
  channels: BodyChannel[];
  focus?: CommandFocus;
  target?: unknown;
}

export interface TaskNode {
  id: string;
  name: string;
  parent_id: string | null;
  source_call_id?: string;
  child_ids: string[];
  objective: string;
  success_criteria: string[];
  goal_predicate_indexes: number[];
  capabilities: string[];
  may_delegate: boolean;
  references: Array<{
    name: string;
    transaction_id: string;
  }>;
  depth: number;
  status: "ready" | "active" | "waiting" | "completed" | "blocked" | "failed";
  steps_used: number;
  model_calls_used: number;
  created_at: string;
  updated_at: string;
  last_result?: unknown;
}

export interface CheckerResult {
  success: boolean;
  goal: Goal;
  world_frame: number;
  world_revision: number;
  checks: Array<{ name: string; passed: boolean; actual: unknown }>;
  checked_at: string;
}

export interface ActionReceipt {
  transaction_id: string;
  agent_id: string;
  agent_name: string;
  kind: "tool" | "skill" | "checker";
  name: string;
  input: unknown;
  accepted: boolean;
  code: string;
  detail: unknown;
  world_before_frame: number;
  world_before_revision?: number;
  world_after_frame: number;
  frame_count: number;
  world_revision: number;
  committed_at: string;
  channels: BodyChannel[];
  gates: Array<{
    name: string;
    status: "passed" | "rejected";
    detail: unknown;
  }>;
}

export interface ContextCompactionSummary {
  mission_state: string;
  constraints: string[];
  decisions: string[];
  completed: Array<{ summary: string; transaction_ids: string[] }>;
  pending: string[];
  blockers: Array<{ summary: string; transaction_ids: string[] }>;
  next_actions: string[];
}

export interface ContextScopeState {
  scope_id: string;
  agent_id: string;
  agent_name: string;
  raw_item_count: number;
  raw_chain_hash: string | null;
  compacted_item_count: number;
  retained_item_count: number;
  retained_chain_hash: string | null;
  active_estimated_tokens: number;
  compaction_count: number;
  summary: ContextCompactionSummary | null;
  summary_origin: "model" | "authority_projection" | null;
  summary_world_revision: number | null;
  summary_voxel_revision: number | null;
  last_compacted_at: string | null;
}

export interface ContextMemoryState {
  version: 1;
  context_window_tokens: number;
  compact_trigger_tokens: number;
  compact_recent_model_turns: number;
  compact_max_output_tokens: number;
  active_scope_id: string | null;
  active_estimated_tokens: number;
  total_compactions: number;
  last_compacted_at: string | null;
  scopes: Record<string, ContextScopeState>;
}

export interface RunCheckpoint {
  version: 3;
  run_id: string;
  scenario_id: string;
  goal: Goal;
  capability_catalog: string[];
  status: "starting" | "running" | "succeeded" | "failed" | "interrupted";
  root_id: string;
  active_agent_id: string | null;
  active_agent_ids?: string[];
  nodes: Record<string, TaskNode>;
  world: WorldSnapshot;
  inflight_action: {
    transaction_id: string;
    agent_id: string;
    agent_name: string;
    kind: "skill";
    name: string;
    input: unknown;
    channels: BodyChannel[];
    world_before_frame: number;
    world_before_revision?: number;
    started_at: string;
  } | null;
  inflight_actions?: Record<string, NonNullable<RunCheckpoint["inflight_action"]>>;
  committed_actions: Record<string, ActionReceipt>;
  /** Optional only for checkpoints written before long-run context memory. */
  context_memory?: ContextMemoryState;
  total_model_calls: number;
  checker: CheckerResult | null;
  final_output: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface TerrainDefinition {
  cell: number;
  columns: number;
  rows: number;
  block: number;
  chunk_size: number;
  maximum_height: number;
  heights: number[];
}

export interface ScenarioDefinition {
  title: string;
  seed: number;
  motion_seed: number;
  bounds: { width: number; depth: number };
  terrain?: TerrainDefinition;
  visibility_radius: number;
  robot: {
    x: number;
    z: number;
    yaw: number;
    joints?: {
      head_yaw: number;
      head_pitch: number;
      shoulder: number;
      elbow: number;
      wrist: number;
      gripper_aperture: number;
    };
  };
  obstacles: Array<{ id: string; center: Vec3; size: Vec3 }>;
  objects: Array<{
    id: string;
    kind: string;
    color: string;
    position: Vec3;
    size: Vec3;
    portable: boolean;
    locked?: boolean;
    key_id?: string;
    container_id?: string;
  }>;
  zones: Array<{ id: string; color: string; center: Vec3; size: Vec3 }>;
  affordances: Array<{
    id: string;
    type: "keyed_lock";
    container_id: string;
    key_id: string;
    socket: {
      center: Vec3;
      half_extents: Vec3;
      insertion_axis: Vec3;
      maximum_axis_angle: number;
    };
  }>;
  default_goal: Goal;
}

export interface RunDetails {
  definition: {
    version: 1;
    run_id: string;
    mission: string;
    scenario_id: string;
    scenario: ScenarioDefinition;
    goal: Goal;
    created_at: string;
  };
  checkpoint: RunCheckpoint;
  actions: ActionReceipt[];
  provider: unknown[];
  framework: unknown[];
  event_cursor: string | null;
}

export interface RunListItem {
  run_id: string;
  scenario_id: string | null;
  mission: string | null;
  status: RunCheckpoint["status"] | "local_artifact";
  created_at: string | null;
  updated_at: string | null;
  error: string | null;
}

export interface Bootstrap {
  provider:
    | { configured: true; protocol: string; model: string; endpoint: string }
    | { configured: false; error: string };
  authentication_required: boolean;
  capability_catalog: string[];
  scenarios: Array<{
    id: string;
    title: string;
    kind: "authored" | "generated";
    extent: { width: number; depth: number };
    objects: Array<{ id: string; kind: string; color: string }>;
    zones: Array<{ id: string; color: string }>;
    suggested_goal: Goal;
  }>;
}

export interface RuntimeEvent {
  event_id: string;
  run_id: string;
  type: string;
  at: string;
  data: unknown;
  durable?: boolean;
}

export type StreamState = "inactive" | "connecting" | "connected" | "disconnected";

export interface ProviderActivity {
  status: string;
  at: string | null;
  source: string | null;
}
