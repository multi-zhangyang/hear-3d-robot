export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export type HumanoidEndEffector =
  | "left_wrist"
  | "right_wrist"
  | "left_ankle"
  | "right_ankle";

interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

export type GoalPredicate =
  | { type: "robot_at"; target: Vec3; tolerance: number }
  | { type: "robot_in_zone"; zone_id: string; tolerance: number }
  | {
      type: "object_in_zone";
      object_id: string;
      zone_id: string;
      expected: boolean;
      tolerance: number;
    }
  | { type: "object_at"; object_id: string; target: Vec3; tolerance: number }
  | {
      type: "end_effector_at";
      end_effector: HumanoidEndEffector;
      frame: "world" | "pelvis";
      target: Vec3;
      tolerance: number;
      stable_frames: number;
    };

export interface Goal {
  summary: string;
  predicates: GoalPredicate[];
}

export type HumanoidBodyChannel =
  | "locomotion"
  | "left_leg"
  | "right_leg"
  | "torso"
  | "left_arm"
  | "right_arm";

interface HumanoidJointState {
  position: number;
  velocity: number;
  minimum: number;
  maximum: number;
}

interface HumanoidLinkState {
  position: Vec3;
  rotation: Quaternion;
  linearVelocity: Vec3;
  angularVelocity: Vec3;
}

interface HumanoidContactState {
  position: Vec3;
  normal: Vec3;
  normalForce: number;
  firstBody: string | null;
  secondBody: string | null;
  firstObject: string | null;
  secondObject: string | null;
}

interface HumanoidFootState {
  touching: boolean;
  contactCount: number;
  normalForce: number;
  points: Vec3[];
}

export interface HumanoidWorldSnapshot {
  frame: number;
  worldRevision: number;
  motionGenerator: {
    protocol: "humanoid-motion-generator-v1";
    implementation: string;
    motionClass: "constraint_solver" | "generative_model";
    sampling: "deterministic" | "stochastic";
  };
  robot: {
    simulatedTime: number;
    controller: {
      protocol: "humanoid-controller-v1";
      implementation: string;
      actuation: "joint_position_pd";
      controlStepSeconds: number;
      physicsStepSeconds: number;
    };
    rootPosition: Vec3;
    rootRotation: Quaternion;
    joints: Record<string, HumanoidJointState>;
    links: Record<string, HumanoidLinkState>;
    objects: Record<string, HumanoidLinkState & { id: string }>;
    contactCount: number;
    contacts: HumanoidContactState[];
    feet: { left: HumanoidFootState; right: HumanoidFootState };
    balance: {
      centerOfMass: Vec3;
      support: "double" | "left" | "right" | "none";
      supportMargin: number | null;
      upright: number;
    };
    nonFootEnvironmentContacts: string[];
    fallen: boolean;
  };
  navigation: {
    planId: string | null;
    status: "idle" | "planned" | "executing" | "completed" | "blocked";
    target: Vec3 | null;
    waypoints: Vec3[];
    waypointIndex: number | null;
  };
}

export interface TaskNode {
  id: string;
  name: string;
  parent_id: string | null;
  child_ids: string[];
  objective: string;
  success_criteria: string[];
  evidence_requirements: unknown[];
  goal_predicate_indexes: number[];
  capabilities: string[];
  may_delegate: boolean;
  references: Array<{ name: string; transaction_id: string }>;
  depth: number;
  status: "ready" | "active" | "waiting" | "completed" | "blocked" | "failed";
  steps_used: number;
  model_calls_used: number;
  created_at: string;
  updated_at: string;
  last_result?: unknown;
}

interface ContextCompactionSummary {
  mission_state: string;
  constraints: string[];
  decisions: string[];
  completed: Array<{ summary: string; transaction_ids: string[] }>;
  pending: string[];
  blockers: Array<{ summary: string; transaction_ids: string[] }>;
  next_actions: string[];
}

interface ContextScopeState {
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

export interface HumanoidCheckerResult {
  success: boolean;
  goal: Goal;
  worldFrame: number;
  worldRevision: number;
  checks: Array<{ name: string; passed: boolean; actual: unknown }>;
  checkedAt: string;
}

export interface HumanoidGoalProgress {
  version: 1;
  goal_sha256: string;
  predicate_count: number;
  last_world_frame: number;
  last_world_revision: number;
  predicate_streaks: number[];
}

export interface HumanoidEmbodiedEpisode {
  sequence: number;
  source_ref?: string;
  transaction_id: string;
  action: "execute_whole_body_motion" | "execute_humanoid_navigation";
  planning_action?: "plan_whole_body_motion"
    | "plan_whole_body_motion_candidates"
    | "plan_humanoid_navigation";
  candidate_count?: number;
  selected_rank?: number;
  selected_candidate_id?: string;
  motion_option?: {
    option_id: string;
    status: "succeeded";
    termination_reason: "physical_success";
    full_frame_count: number;
    executed_prefix_frame_count: number;
    predicted_termination_frame: number;
    actual_termination_frame: number;
    artifact_sha256: string;
  };
  code: string;
  model_summary: string;
  world_before_revision: number;
  world_after_revision: number;
  frame_count: number;
  result_frame: number;
  result_root_position: Vec3;
  fallen: boolean;
  support: "none" | "left" | "right" | "double";
  upright: number;
  goal_success: boolean;
  recorded_at: string;
}

export interface HumanoidEmbodiedMemoryState {
  version: 1;
  total_episodes: number;
  pruned_episodes: number;
  recent_episodes: HumanoidEmbodiedEpisode[];
}

type HumanoidActionName =
  | "observe_humanoid"
  | "plan_whole_body_motion"
  | "plan_whole_body_motion_candidates"
  | "execute_whole_body_motion"
  | "plan_humanoid_navigation"
  | "execute_humanoid_navigation";

export interface HumanoidActionReceipt {
  transactionId: string;
  agentId: string;
  action: HumanoidActionName;
  input: unknown;
  fingerprint: string;
  accepted: boolean;
  code: string;
  worldBeforeRevision: number;
  worldAfterRevision: number;
  frameCount: number;
  channels: HumanoidBodyChannel[];
  detail: unknown;
  committedAt: string;
}

type RunStatus = "starting" | "running" | "succeeded" | "failed" | "interrupted";

export interface HumanoidRunCheckpoint {
  version: 4 | 5;
  runtime: "humanoid_g1";
  run_id: string;
  scenario_id: string;
  goal: Goal;
  capability_catalog: string[];
  status: RunStatus;
  root_id: string;
  active_agent_id: string | null;
  active_agent_ids: string[];
  nodes: Record<string, TaskNode>;
  world: HumanoidWorldSnapshot;
  world_checkpoint: unknown;
  committed_actions: Record<string, HumanoidActionReceipt>;
  context_memory: ContextMemoryState;
  embodied_memory: HumanoidEmbodiedMemoryState;
  pending_lifecycle_events: unknown[];
  cycle_index: number;
  total_model_calls: number;
  checker: HumanoidCheckerResult | null;
  goal_progress?: HumanoidGoalProgress;
  last_cycle: unknown;
  final_output: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface ScenarioDefinition {
  title: string;
  seed: number;
  bounds: { width: number; depth: number };
  visibility_radius: number;
  robot: { x: number; z: number; yaw: number };
  obstacles: Array<{ id: string; center: Vec3; size: Vec3 }>;
  objects: Array<{
    id: string;
    kind: string;
    color: string;
    position: Vec3;
    size: Vec3;
    portable: boolean;
  }>;
  zones: Array<{ id: string; color: string; center: Vec3; size: Vec3 }>;
  default_goal: Goal;
}

export interface HumanoidRunDetails {
  definition: {
    version: 1;
    run_id: string;
    mission: string;
    scenario_id: string;
    scenario: ScenarioDefinition;
    goal: Goal;
    runtime: "humanoid_g1";
    created_at: string;
  };
  checkpoint: HumanoidRunCheckpoint;
  actions: HumanoidActionReceipt[];
  provider: unknown[];
  framework: unknown[];
  event_cursor: string | null;
}

export function isHumanoidRunDetails(value: unknown): value is HumanoidRunDetails {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const details = value as Record<string, unknown>;
  if (typeof details.definition !== "object" || details.definition === null) return false;
  if (typeof details.checkpoint !== "object" || details.checkpoint === null) return false;
  const definition = details.definition as Record<string, unknown>;
  const checkpoint = details.checkpoint as Record<string, unknown>;
  return definition.runtime === "humanoid_g1"
    && checkpoint.runtime === "humanoid_g1"
    && (checkpoint.version === 4 || checkpoint.version === 5)
    && (checkpoint.version !== 5 || isHumanoidGoalProgress(checkpoint.goal_progress));
}

function isHumanoidGoalProgress(value: unknown): boolean {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const progress = value as Record<string, unknown>;
  return progress.version === 1
    && typeof progress.goal_sha256 === "string"
    && typeof progress.predicate_count === "number"
    && typeof progress.last_world_frame === "number"
    && typeof progress.last_world_revision === "number"
    && Array.isArray(progress.predicate_streaks)
    && progress.predicate_streaks.length === progress.predicate_count;
}

export interface RunListItem {
  run_id: string;
  scenario_id: string | null;
  mission: string | null;
  status: RunStatus | "local_artifact";
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
    kind: "authored" | "generated" | "procedural";
    runtime: "humanoid_g1";
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
  cursor?: string;
}

export type StreamState = "inactive" | "connecting" | "connected" | "disconnected";

export interface ProviderActivity {
  status: string;
  at: string | null;
  source: string | null;
}
