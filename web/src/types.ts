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

export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

type G1HandContactSurfaceName =
  | "left_hand_palm_link"
  | "left_hand_thumb_0_link"
  | "left_hand_thumb_1_link"
  | "left_hand_thumb_2_link"
  | "left_hand_middle_0_link"
  | "left_hand_middle_1_link"
  | "left_hand_index_0_link"
  | "left_hand_index_1_link"
  | "right_hand_palm_link"
  | "right_hand_thumb_0_link"
  | "right_hand_thumb_1_link"
  | "right_hand_thumb_2_link"
  | "right_hand_middle_0_link"
  | "right_hand_middle_1_link"
  | "right_hand_index_0_link"
  | "right_hand_index_1_link";

export type GoalPredicate =
  | { type: "robot_at"; target: Vec3; tolerance: number }
  | { type: "robot_in_zone"; zone_id: string; tolerance: number }
  | { type: "block_removed"; block_id: string }
  | {
      type: "object_in_zone";
      object_id: string;
      zone_id: string;
      expected: boolean;
      tolerance: number;
    }
  | {
      type: "object_placed";
      object_id: string;
      zone_id: string;
      tolerance: number;
    }
  | { type: "object_at"; object_id: string; target: Vec3; tolerance: number }
  | {
      type: "object_grasped";
      object_id: string;
      hand: "left" | "right" | "either";
    }
  | {
      type: "end_effector_at";
      end_effector: HumanoidEndEffector;
      frame: "world" | "pelvis";
      target: Vec3;
      tolerance: number;
      stable_frames: number;
      orientation?: Quaternion;
      orientation_tolerance_rad?: number;
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
  firstSolid?: string | null;
  secondSolid?: string | null;
  firstHandLink?: G1HandContactSurfaceName | null;
  secondHandLink?: G1HandContactSurfaceName | null;
}

interface HumanoidHandJointState extends HumanoidJointState {
  target: number;
  stiffnessNewtonMetersPerRadian: number;
  dampingNewtonMeterSecondsPerRadian: number;
  appliedNewtonMeters: number;
  minimumNewtonMeters: number;
  maximumNewtonMeters: number;
  saturated: boolean;
}

interface HumanoidFootState {
  touching: boolean;
  contactCount: number;
  normalForce: number;
  points: Vec3[];
}

interface HumanoidGraspAssessment {
  protocol: "humanoid-grasp-assessment-v1";
  frame: number;
  object_id: string;
  hand: "left" | "right";
  phase: "idle" | "stabilizing" | "lifting" | "holding" | "verified";
  grasp_verified: boolean;
  reason: string;
  reset_reason: string | null;
  evidence: {
    contact: {
      status: "missing" | "insufficient_links" | "insufficient_normal"
        | "insufficient_geometry" | "not_opposed" | "opposed";
      observed_contact_count: number;
      force_qualified_contact_count: number;
      distinct_force_qualified_links: G1HandContactSurfaceName[];
      distinct_normal_qualified_links: G1HandContactSurfaceName[];
      opposing_pair: unknown | null;
    };
    support: {
      status: "supported" | "unsupported" | "insufficient_normal";
      candidate_contact_count: number;
      force_qualified_contact_count: number;
      upward_contact_count: number;
      baseline_projection_m: number | null;
      current_projection_m: number;
      lift_m: number | null;
    };
    relative_pose: {
      stable_frames: number;
      translation_drift_m: number | null;
      rotation_drift_rad: number | null;
    };
    lifted_hold_frames: number;
  };
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
    morphology?: {
      id: "unitree_g1_43dof_with_hands";
      bodyJointCount: 29;
      handJointCount: 14;
      totalJointCount: 43;
      source: {
        repository: "google-deepmind/mujoco_menagerie";
        commit: "71f066ad0be9cd271f7ed58c030243ef157af9f4";
        model: "unitree_g1/g1_with_hands.xml";
      };
    };
    simulatedTime: number;
    controller: {
      protocol: "humanoid-controller-v1";
      implementation: string;
      actuation: "joint_position_pd";
      controlStepSeconds: number;
      physicsStepSeconds: number;
    };
    controllerExecution?: {
      protocol: "humanoid-controller-execution-v1";
      mode: "learned_policy" | "reference_control";
      activeImplementation: string;
      transition: {
        fromImplementation: string;
        toImplementation: string;
        progress: number;
        durationSeconds: number;
      } | null;
    };
    rootPosition: Vec3;
    rootRotation: Quaternion;
    joints: Record<string, HumanoidJointState>;
    links: Record<string, HumanoidLinkState>;
    hands?: {
      controller: {
        protocol: "g1-hand-controller-v1";
        implementation: "mujoco_continuous_position_pd";
        actuation: "joint_position_pd";
        jointCount: 14;
      };
      joints: Record<string, HumanoidHandJointState>;
      links: Record<string, HumanoidLinkState>;
    };
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
  grasp: {
    contractSha256: string;
    assessments: HumanoidGraspAssessment[];
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
  token_estimator_correction_milli?: number;
  context_window_tokens?: number;
  compact_trigger_tokens?: number;
  compact_recent_model_turns?: number;
  compact_max_output_tokens?: number;
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

interface ModelUsageTotals {
  requests: number;
  reported_requests: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cached_input_tokens: number;
  reasoning_tokens: number;
}

export interface ModelUsageState {
  version: 1;
  total: ModelUsageTotals;
  by_agent: Record<string, ModelUsageTotals>;
  updated_at: string | null;
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

interface AutonomousCycleRef {
  cycle_id: string;
  cycle_index: number;
  goal_epoch_id: string;
}

export interface ActiveAutonomousCycle extends AutonomousCycleRef {
  started_world_frame: number;
  started_world_revision: number;
  started_at: string;
}

interface ModelDecisionRef {
  agent_id: string;
  agent_manifest_sha256: string;
  agent_manifest_epoch_id: string;
  model_call_id: string;
  response_id: string;
  response_output_sha256: string;
  tool_call_id: string;
  tool_arguments_sha256: string;
}

type GoalModelSource = ModelDecisionRef;

type GoalResolutionStatus =
  | "completed"
  | "blocked"
  | "abandoned"
  | "superseded"
  | "expired";

interface GoalCandidate {
  candidate_id: string;
  proposal_id: string;
  source: GoalModelSource;
  goal: Goal;
  mission_link: string;
  identity_sha256: string;
  content_sha256: string;
  integrity_sha256: string;
  dependency_candidate_ids: string[];
  status: "proposed" | "active" | GoalResolutionStatus;
  physical_evidence_refs: { proposal: string[]; resolution: string[] };
  created_world_revision: number;
  resolved_world_revision: number | null;
}

interface GoalEpoch {
  epoch_id: string;
  epoch_index: number;
  previous_epoch_id: string | null;
  candidate_id: string;
  candidate_source: GoalModelSource;
  selected_by: GoalModelSource;
  candidate_identity_sha256: string;
  candidate_content_sha256: string;
  dependency_candidate_ids: string[];
  identity_sha256: string;
  status: "active" | GoalResolutionStatus;
  retired_by: GoalModelSource | null;
  retirement_reason: string | null;
  physical_evidence_refs: { selection: string[]; resolution: string[] };
  created_world_revision: number;
  resolved_world_revision: number | null;
}

export interface GoalDAG {
  version: 2;
  status: "awaiting_model_selection" | "active";
  candidates: Record<string, GoalCandidate>;
  candidate_sequences: Record<string, number>;
  next_candidate_sequence: number;
  epochs: GoalEpoch[];
  current_epoch_id: string | null;
  next_epoch_index: number;
  evidence: Record<string, unknown>;
  archive: {
    record_count: number;
    last_record_sha256: string | null;
    last_epoch_id: string | null;
    retained_candidate_ids: string[];
    summary: GoalHistorySummary | null;
  };
  state_sha256: string;
}

interface GoalHistorySummary {
  version: 1;
  archived_epoch_count: number;
  last_record_sha256: string | null;
  records_without_alternate_history: number;
  outcomes: {
    selected: GoalHistorySelectedOutcomes;
    not_selected: number;
    predicate_outcomes: Array<GoalHistoryDimensionOutcome & {
      predicate_type: string;
    }>;
    entity_outcomes: Array<GoalHistoryDimensionOutcome & {
      entity_kind: "object" | "zone" | "solid" | "end_effector";
      entity_id: string;
    }>;
  };
}

interface GoalHistorySelectedOutcomes {
  total: number;
  completed: number;
  blocked: number;
  abandoned: number;
  superseded: number;
  expired: number;
}

interface GoalHistoryDimensionOutcome {
  selected: GoalHistorySelectedOutcomes;
  not_selected: number;
  last_selected: {
    epoch_sequence: number;
    status: "completed" | "blocked" | "abandoned" | "superseded" | "expired";
    world_revision: number;
  } | null;
  last_not_selected: {
    epoch_sequence: number;
    world_revision: number;
  } | null;
}

export interface HumanoidEmbodiedEpisode {
  sequence: number;
  source_ref?: string;
  causal_trace?: {
    cycle: AutonomousCycleRef;
    planning_transaction_id: string;
    execution_transaction_id: string;
    world_mutation_transaction_ids?: string[];
    execution_decision: ModelDecisionRef;
    goal_evidence_refs: string[];
    memory_id: string;
  };
  transaction_id: string;
  action: "execute_humanoid_skill"
    | "execute_whole_body_motion"
    | "execute_humanoid_navigation";
  planning_action?: "plan_humanoid_skill"
    | "plan_whole_body_motion"
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
  world_mutations?: Array<{
    transaction_id: string;
    action: "remove_world_block";
    decision: ModelDecisionRef;
    code: "world_block_removal_authorized";
    execution_transaction_id: string;
    solid_id: string;
    world_before_revision: number;
    world_after_revision: number;
    chunk_before_revision: number;
    chunk_after_revision: number;
  }>;
  code: string;
  model_summary: string;
  world_before_revision: number;
  world_after_revision: number;
  frame_count: number;
  result_frame: number;
  result_world_revision?: number;
  result_root_position: Vec3;
  fallen: boolean;
  support: "none" | "left" | "right" | "double";
  upright: number;
  goal_success: boolean;
  recorded_at: string;
}

export interface HumanoidEmbodiedMemoryState {
  version: 2;
  total_episodes: number;
  pruned_episodes: number;
  recent_episodes: HumanoidEmbodiedEpisode[];
  total_experiences: number;
  pruned_experiences: number;
  recent_experiences: HumanoidEmbodiedExperience[];
  outcome_counts: HumanoidExperienceOutcomeCounts;
  predicate_outcome_counts: Record<string, HumanoidExperienceOutcomeCounts>;
  object_outcome_counts: Record<string, HumanoidExperienceOutcomeCounts>;
  solid_outcome_counts: Record<string, HumanoidExperienceOutcomeCounts>;
  zone_outcome_counts: Record<string, HumanoidExperienceOutcomeCounts>;
}

interface HumanoidEmbodiedExperience {
  sequence: number;
  source_ref: string;
  transaction_id: string;
  cycle: AutonomousCycleRef;
  action: "execute_whole_body_motion"
    | "execute_humanoid_skill"
    | "execute_humanoid_navigation"
    | "remove_world_block";
  planning_action?: "plan_humanoid_skill"
    | "plan_whole_body_motion"
    | "plan_whole_body_motion_candidates"
    | "plan_humanoid_navigation";
  accepted: boolean;
  code: string;
  outcome: "succeeded" | "rejected" | "physically_failed";
  world_before_revision: number;
  world_after_revision: number;
  frame_count: number;
  goal_content_sha256: string;
  goal_summary: string;
  predicate_types: string[];
  object_ids: string[];
  solid_ids: string[];
  zone_ids: string[];
  recorded_at: string;
}

interface HumanoidExperienceOutcomeCounts {
  succeeded: number;
  rejected: number;
  physically_failed: number;
}

type HumanoidActionName =
  | "observe_humanoid"
  | "submit_humanoid_skill_plan"
  | "begin_humanoid_skill"
  | "plan_humanoid_skill"
  | "execute_humanoid_skill"
  | "plan_whole_body_motion"
  | "plan_whole_body_motion_candidates"
  | "execute_whole_body_motion"
  | "plan_humanoid_navigation"
  | "execute_humanoid_navigation"
  | "remove_world_block";

export interface HumanoidActionReceipt {
  transactionId: string;
  agentId: string;
  decision?: ModelDecisionRef;
  cycle?: AutonomousCycleRef;
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

type RunStatus = "starting" | "running" | "paused" | "succeeded" | "failed" | "interrupted";

export type HumanoidRunMode = "mission" | "continuous";

export interface HumanoidRunCheckpoint {
  version: 4 | 5 | 6;
  runtime: "humanoid_g1";
  run_id: string;
  scenario_id: string;
  goal?: Goal;
  mission_goal?: Goal;
  goal_dag?: GoalDAG;
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
  model_usage?: ModelUsageState;
  checker: HumanoidCheckerResult | null;
  goal_progress?: HumanoidGoalProgress | null;
  active_cycle?: ActiveAutonomousCycle | null;
  last_cycle: unknown;
  final_output: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

interface ScenarioChunkDefinition {
  id: string;
  coordinate: { column: number; row: number };
  bounds: {
    minimum: { x: number; z: number };
    maximum: { x: number; z: number };
  };
  entity_ids: {
    obstacles: string[];
    objects: string[];
    zones: string[];
  };
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
  chunk_manifest: {
    version: 1;
    chunk_size: number;
    grid: { columns: number; rows: number };
    chunks: ScenarioChunkDefinition[];
  };
}

export interface ScenarioChunkDeltaState {
  version: 1;
  scenario_seed: number;
  scenario_sha256: string;
  manifest_version: 1;
  revision: number;
  changed_chunk_ids: string[];
  chunks: Array<{
    chunk_id: string;
    revision: number;
    blocks: Array<{
      id: string;
      origin: "scenario" | "created";
      present: boolean;
      center: Vec3;
      size: Vec3;
      material: string;
      properties: Record<string, unknown>;
    }>;
    zones: Array<{
      id: string;
      origin: "scenario" | "created";
      present: boolean;
      color: string;
      center: Vec3;
      size: Vec3;
      enabled: boolean;
      properties: Record<string, unknown>;
    }>;
    dynamic_entities: Array<{
      id: string;
      origin: "scenario" | "created";
      present: boolean;
      kind: string;
      color: string;
      position: Vec3;
      rotation: Quaternion;
      linear_velocity: Vec3;
      angular_velocity: Vec3;
      size: Vec3;
      portable: boolean;
      properties: Record<string, unknown>;
      physical_authority?: {
        source: "humanoid_mujoco";
        transaction_id: string;
        world_frame: number;
        world_revision: number;
      };
    }>;
  }>;
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
    run_mode: HumanoidRunMode;
    created_at: string;
  };
  checkpoint: HumanoidRunCheckpoint;
  scenario_chunks: ScenarioChunkDeltaState;
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
  const scenarioChunks = details.scenario_chunks as Record<string, unknown> | null;
  return definition.runtime === "humanoid_g1"
    && checkpoint.runtime === "humanoid_g1"
    && scenarioChunks !== null
    && scenarioChunks.version === 1
    && typeof scenarioChunks.revision === "number"
    && Array.isArray(scenarioChunks.chunks)
    && (checkpoint.version === 4 || checkpoint.version === 5 || checkpoint.version === 6)
    && (checkpoint.version !== 5 || isHumanoidGoalProgress(checkpoint.goal_progress))
    && (checkpoint.version !== 6 || isGoalEpochCheckpoint(checkpoint));
}

function isGoalEpochCheckpoint(checkpoint: Record<string, unknown>): boolean {
  const missionGoal = checkpoint.mission_goal as Record<string, unknown> | null;
  const dag = checkpoint.goal_dag as Record<string, unknown> | null;
  if (!missionGoal || typeof missionGoal.summary !== "string" || !Array.isArray(missionGoal.predicates)
    || !dag || (dag.version !== 1 && dag.version !== 2) || !Array.isArray(dag.epochs)
    || typeof dag.candidates !== "object" || dag.candidates === null
    || (dag.status !== "awaiting_model_selection" && dag.status !== "active")) return false;
  if (dag.status === "awaiting_model_selection") {
    return checkpoint.goal_progress === null && checkpoint.checker === null;
  }
  return typeof dag.current_epoch_id === "string"
    && isHumanoidGoalProgress(checkpoint.goal_progress);
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
    | { configured: true; protocol: string; model: string }
    | { configured: false; error: string };
  authentication_required: boolean;
  capability_catalog: string[];
  scenarios: Array<{
    id: string;
    title: string;
    kind: "authored" | "generated" | "procedural";
    runtime: "humanoid_g1";
    extent: { width: number; depth: number };
    chunk_grid: {
      manifest_version: 1;
      chunk_size: number;
      columns: number;
      rows: number;
    };
    objects: Array<{ id: string; kind: string; color: string; portable: boolean }>;
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
  agentId?: string;
}
