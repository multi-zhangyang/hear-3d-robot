import type {
  NeuralLayer,
  NeuralPathway,
  NeuralSignalKind
} from "../../domain/neural-hierarchy.js";
import type { TaskNode } from "../../domain/schema.js";

export const HUMANOID_NEURAL_AGENT_IDS = {
  executive: "humanoid-executive",
  goalManager: "humanoid-goal-manager",
  actionSelection: "humanoid-action-selection-gate",
  perceptionManager: "humanoid-perceptual-association-manager",
  sensorFusion: "humanoid-sensor-fusion",
  sceneInterpreter: "humanoid-scene-interpreter",
  memoryRetriever: "humanoid-relevant-memory-retriever",
  sensorimotorManager: "humanoid-sensorimotor-manager",
  affordance: "humanoid-affordance-specialist",
  risk: "humanoid-risk-interoception-critic",
  predictive: "humanoid-cerebellar-predictive-critic",
  premotor: "humanoid-premotor-skill-composer",
  motorIntent: "humanoid-motor-intent-compiler",
  rolloutGate: "humanoid-rollout-gate",
  executionDispatcher: "humanoid-certified-execution-dispatcher",
  executor: "humanoid-executor",
  reflex: "humanoid-controller-reflex",
  body: "humanoid-mujoco-body",
  recovery: "humanoid-recovery-controller"
} as const;

export type HumanoidNeuralAgentKey = keyof typeof HUMANOID_NEURAL_AGENT_IDS;
export type HumanoidNeuralAgentId = typeof HUMANOID_NEURAL_AGENT_IDS[
  HumanoidNeuralAgentKey
];

export type HumanoidNeuralExecutionKind =
  | "model_agent"
  | "deterministic_service"
  | "learned_controller";

/**
 * Control ownership is deliberately separate from execution kind. A model
 * child is normally exposed through Agent.asTool(), so its parent keeps the
 * control domain. Recovery also keeps that single parent: it runs a separate,
 * exclusive SDK episode under a parent-issued Harness lease rather than an
 * SDK handoff. Non-model nodes are invoked by the Harness, never offered as
 * peer Agents.
 */
export type HumanoidNeuralOrchestrationKind =
  | "root_runner"
  | "agent_tool"
  | "exclusive_lease_episode"
  | "runtime_service"
  | "controller_loop"
  | "physical_plant";

export type HumanoidNeuralSessionMode = "independent_file_session" | "none";

/**
 * Cadence is a semantic activation contract, not a fixed timer. Model layers
 * are event-driven; only the controller and plant are allowed to run at their
 * configured continuous rates.
 */
export type HumanoidNeuralCadence =
  | "mission_event"
  | "goal_event"
  | "world_event"
  | "skill_event"
  | "rollout_event"
  | "execution_transaction"
  | "recovery_event"
  | "controller_tick"
  | "physics_tick";

export type HumanoidNeuralCorrectionScope =
  | "none"
  | "local"
  | "pathway"
  | "supervisory";

export interface HumanoidNeuralNodeDescriptor {
  key: HumanoidNeuralAgentKey;
  id: HumanoidNeuralAgentId;
  name: string;
  parentKey: HumanoidNeuralAgentKey | null;
  childKeys: readonly HumanoidNeuralAgentKey[];
  layer: NeuralLayer;
  pathway: NeuralPathway;
  executionKind: HumanoidNeuralExecutionKind;
  orchestrationKind: HumanoidNeuralOrchestrationKind;
  sessionMode: HumanoidNeuralSessionMode;
  cadence: HumanoidNeuralCadence;
  maximumCorrectionScope: HumanoidNeuralCorrectionScope;
  objective: string;
  criteria: readonly string[];
  capabilities: readonly string[];
  mayDelegate: boolean;
  parallelSafe: boolean;
  parallelGroup?: "perception_interpretation" | "sensorimotor_assessment";
  physicalWriteAuthority: boolean;
}

export const HUMANOID_NEURAL_NODES: readonly HumanoidNeuralNodeDescriptor[] = [
  {
    key: "executive",
    id: HUMANOID_NEURAL_AGENT_IDS.executive,
    name: "Executive Goal Valuation Manager",
    parentKey: null,
    childKeys: ["goalManager", "actionSelection"],
    layer: "executive",
    pathway: "executive_control",
    executionKind: "model_agent",
    orchestrationKind: "root_runner",
    sessionMode: "independent_file_session",
    cadence: "mission_event",
    maximumCorrectionScope: "supervisory",
    objective: "Maintain the mission, value candidate Goals, and delegate bounded control episodes.",
    criteria: ["Never emit trajectories or directly write physical state."],
    capabilities: ["manage_goal", "delegate_action_selection", "complete_cycle"],
    mayDelegate: true,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "goalManager",
    id: HUMANOID_NEURAL_AGENT_IDS.goalManager,
    name: "Goal Valuation Specialist",
    parentKey: "executive",
    childKeys: [],
    layer: "executive",
    pathway: "goal_valuation",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "goal_event",
    maximumCorrectionScope: "supervisory",
    objective: "Propose, select, continue, or retire observable Goal epochs.",
    criteria: ["Bind every Goal transition to current world evidence."],
    capabilities: ["recall_goal_history", "manage_goal_epoch"],
    mayDelegate: false,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "actionSelection",
    id: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    name: "Action Selection Gate",
    parentKey: "executive",
    childKeys: ["perceptionManager", "sensorimotorManager"],
    layer: "action_selection",
    pathway: "sensorimotor_selection",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "skill_event",
    maximumCorrectionScope: "pathway",
    objective: "Gate competing sensorimotor programs and maintain one skill commitment.",
    criteria: ["Escalate only errors that lower layers cannot correct locally."],
    capabilities: ["select_pathway", "gate_skill_commitment"],
    mayDelegate: true,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "perceptionManager",
    id: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
    name: "Perceptual Association Manager",
    parentKey: "actionSelection",
    childKeys: ["sensorFusion", "sceneInterpreter", "memoryRetriever"],
    layer: "perceptual_association",
    pathway: "perceptual_association",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "world_event",
    maximumCorrectionScope: "pathway",
    objective: "Fuse current sensation with scene interpretation and relevant history.",
    criteria: [
      "Parallel specialists remain read-only and world-version bounded.",
      "Perceptual belief reports state and candidates without selecting a motor program."
    ],
    capabilities: ["orchestrate_perception"],
    mayDelegate: true,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "sensorFusion",
    id: HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
    name: "Sensor Fusion Service",
    parentKey: "perceptionManager",
    childKeys: [],
    layer: "perceptual_association",
    pathway: "ascending_feedback",
    executionKind: "deterministic_service",
    orchestrationKind: "runtime_service",
    sessionMode: "none",
    cadence: "world_event",
    maximumCorrectionScope: "none",
    objective: "Capture vision, proprioception, contact, and world revision from MuJoCo.",
    criteria: ["Do not infer goals or motor programs."],
    capabilities: ["observe_humanoid"],
    mayDelegate: false,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "sceneInterpreter",
    id: HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter,
    name: "Scene and World Model Interpreter",
    parentKey: "perceptionManager",
    childKeys: [],
    layer: "perceptual_association",
    pathway: "perceptual_association",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "world_event",
    maximumCorrectionScope: "local",
    objective: "Interpret spatial relations, occlusion, contacts, and task-relevant changes.",
    criteria: ["Return beliefs with confidence and evidence, never invented coordinates."],
    capabilities: ["interpret_scene"],
    mayDelegate: false,
    parallelSafe: true,
    parallelGroup: "perception_interpretation",
    physicalWriteAuthority: false
  },
  {
    key: "memoryRetriever",
    id: HUMANOID_NEURAL_AGENT_IDS.memoryRetriever,
    name: "Relevant Embodied Memory Retriever",
    parentKey: "perceptionManager",
    childKeys: [],
    layer: "perceptual_association",
    pathway: "perceptual_association",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "world_event",
    maximumCorrectionScope: "local",
    objective: "Retrieve only prior experiences relevant to the present state and active Goal.",
    criteria: ["Historical state never substitutes for current sensory authority."],
    capabilities: ["recall_embodied_history"],
    mayDelegate: false,
    parallelSafe: true,
    parallelGroup: "perception_interpretation",
    physicalWriteAuthority: false
  },
  {
    key: "sensorimotorManager",
    id: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
    name: "Sensorimotor Manager",
    parentKey: "actionSelection",
    childKeys: [
      "affordance",
      "risk",
      "predictive",
      "premotor",
      "executionDispatcher",
      "recovery"
    ],
    layer: "sensorimotor",
    pathway: "sensorimotor_selection",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "skill_event",
    maximumCorrectionScope: "pathway",
    objective: "Integrate affordance, risk, prediction, and premotor proposals into action.",
    criteria: ["Parallel critics are advisory; one serial physical writer remains authoritative."],
    capabilities: ["orchestrate_sensorimotor", "execute_accepted_plan"],
    mayDelegate: true,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "affordance",
    id: HUMANOID_NEURAL_AGENT_IDS.affordance,
    name: "Affordance Specialist",
    parentKey: "sensorimotorManager",
    childKeys: [],
    layer: "sensorimotor",
    pathway: "sensorimotor_selection",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "skill_event",
    maximumCorrectionScope: "local",
    objective: "Identify currently reachable and task-relevant object and terrain affordances.",
    criteria: ["Affordances remain hypotheses until deterministic planning validates them."],
    capabilities: ["assess_affordances"],
    mayDelegate: false,
    parallelSafe: true,
    parallelGroup: "sensorimotor_assessment",
    physicalWriteAuthority: false
  },
  {
    key: "risk",
    id: HUMANOID_NEURAL_AGENT_IDS.risk,
    name: "Risk and Interoception Critic",
    parentKey: "sensorimotorManager",
    childKeys: [],
    layer: "sensorimotor",
    pathway: "interoceptive_risk",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "skill_event",
    maximumCorrectionScope: "local",
    objective: "Assess balance, collision, contact, fatigue proxy, and commitment risk.",
    criteria: ["High risk inhibits action but does not invent an alternate motor command."],
    capabilities: ["assess_risk"],
    mayDelegate: false,
    parallelSafe: true,
    parallelGroup: "sensorimotor_assessment",
    physicalWriteAuthority: false
  },
  {
    key: "predictive",
    id: HUMANOID_NEURAL_AGENT_IDS.predictive,
    name: "Cerebellar Predictive Critic",
    parentKey: "sensorimotorManager",
    childKeys: [],
    layer: "predictive_rollout",
    pathway: "cerebellar_prediction",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "rollout_event",
    maximumCorrectionScope: "local",
    objective: "Judge whether one bounded rollout chunk safely advances the active Skill toward its terminal contract.",
    criteria: [
      "Report prediction error independently of executive preference.",
      "Intermediate chunks need progress and safety, not premature completion of the final Goal predicate."
    ],
    capabilities: ["interpret_rollout", "estimate_prediction_error"],
    mayDelegate: false,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "premotor",
    id: HUMANOID_NEURAL_AGENT_IDS.premotor,
    name: "Premotor Skill Composer",
    parentKey: "sensorimotorManager",
    childKeys: ["motorIntent"],
    layer: "premotor",
    pathway: "premotor_composition",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "skill_event",
    maximumCorrectionScope: "local",
    objective: "Compose a short skill DAG and delegate one bounded motor intent at a time.",
    criteria: ["Specify semantics and termination, never joint trajectories."],
    capabilities: ["compose_skill", "delegate_motor_intent"],
    mayDelegate: true,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "motorIntent",
    id: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
    name: "Motor Intent Compiler",
    parentKey: "premotor",
    childKeys: ["rolloutGate"],
    layer: "motor_planning",
    pathway: "motor_intent",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "rollout_event",
    maximumCorrectionScope: "local",
    objective: "Compile one semantic intent into an existing deterministic planning tool call.",
    criteria: ["Copy current bindings and let solvers derive geometry and trajectories."],
    capabilities: ["plan_skill", "plan_navigation", "plan_whole_body_motion"],
    mayDelegate: true,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "rolloutGate",
    id: HUMANOID_NEURAL_AGENT_IDS.rolloutGate,
    name: "MuJoCo Predictive Rollout Gate",
    parentKey: "motorIntent",
    childKeys: [],
    layer: "predictive_rollout",
    pathway: "cerebellar_prediction",
    executionKind: "deterministic_service",
    orchestrationKind: "runtime_service",
    sessionMode: "none",
    cadence: "rollout_event",
    maximumCorrectionScope: "none",
    objective: "Validate candidate motion in a cloned MuJoCo state before admission.",
    criteria: ["Only certified plans may reach the physical writer."],
    capabilities: ["mujoco_rollout_validation"],
    mayDelegate: false,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "executionDispatcher",
    id: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
    name: "Certified Execution Dispatcher",
    parentKey: "sensorimotorManager",
    childKeys: ["executor"],
    layer: "controller",
    pathway: "physical_execution",
    executionKind: "model_agent",
    orchestrationKind: "agent_tool",
    sessionMode: "independent_file_session",
    cadence: "skill_event",
    maximumCorrectionScope: "local",
    objective: "Dispatch the one already-certified motor intent through the serial physical writer.",
    criteria: [
      "Run only after Action Selection authorization and invoke exactly one required execution tool.",
      "Make no planning, selection, or recovery decision."
    ],
    capabilities: ["dispatch_certified_execution"],
    mayDelegate: true,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "executor",
    id: HUMANOID_NEURAL_AGENT_IDS.executor,
    name: "Serial Physical Execution Gate",
    parentKey: "executionDispatcher",
    childKeys: ["reflex"],
    layer: "controller",
    pathway: "physical_execution",
    executionKind: "deterministic_service",
    orchestrationKind: "runtime_service",
    sessionMode: "none",
    cadence: "execution_transaction",
    maximumCorrectionScope: "local",
    objective: "Consume one accepted plan and own the only physical mutation transaction.",
    criteria: ["All body writes pass through one mutex-protected authority."],
    capabilities: ["execute_plan", "remove_world_block"],
    mayDelegate: false,
    parallelSafe: false,
    physicalWriteAuthority: true
  },
  {
    key: "reflex",
    id: HUMANOID_NEURAL_AGENT_IDS.reflex,
    name: "Learned Controller and Reflex Loop",
    parentKey: "executor",
    childKeys: ["body"],
    layer: "reflex",
    pathway: "ascending_feedback",
    executionKind: "learned_controller",
    orchestrationKind: "controller_loop",
    sessionMode: "none",
    cadence: "controller_tick",
    maximumCorrectionScope: "local",
    objective: "Track references, reject disturbances, and report local control error at control rate.",
    criteria: ["No LLM is present in the per-frame loop."],
    capabilities: ["learned_control", "reference_control", "contact_reflex"],
    mayDelegate: false,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "body",
    id: HUMANOID_NEURAL_AGENT_IDS.body,
    name: "MuJoCo Embodied Plant",
    parentKey: "reflex",
    childKeys: [],
    layer: "body",
    pathway: "ascending_feedback",
    executionKind: "deterministic_service",
    orchestrationKind: "physical_plant",
    sessionMode: "none",
    cadence: "physics_tick",
    maximumCorrectionScope: "none",
    objective: "Evolve the authoritative physical state under admitted controller commands.",
    criteria: ["Expose sensation and contacts; never accept a model-authored state write."],
    capabilities: ["mujoco_physics", "proprioception", "contact_sensing"],
    mayDelegate: false,
    parallelSafe: false,
    physicalWriteAuthority: false
  },
  {
    key: "recovery",
    id: HUMANOID_NEURAL_AGENT_IDS.recovery,
    name: "Recovery Control Specialist",
    parentKey: "sensorimotorManager",
    childKeys: [],
    layer: "sensorimotor",
    pathway: "interoceptive_risk",
    executionKind: "model_agent",
    orchestrationKind: "exclusive_lease_episode",
    sessionMode: "independent_file_session",
    cadence: "recovery_event",
    maximumCorrectionScope: "pathway",
    objective: "Use an exclusive parent-issued lease to decide one bounded recovery response.",
    criteria: [
      "Remain a child of Sensorimotor Manager and close the lease after recovery or escalation."
    ],
    capabilities: ["recover_balance", "recover_path", "escalate_failure"],
    mayDelegate: false,
    parallelSafe: false,
    physicalWriteAuthority: false
  }
] as const;

export const HUMANOID_NEURAL_NODE_BY_ID: ReadonlyMap<
  string,
  HumanoidNeuralNodeDescriptor
> = new Map(
  HUMANOID_NEURAL_NODES.map((node) => [node.id, node] as const)
);

export const HUMANOID_NEURAL_CAPABILITIES = [...new Set(
  HUMANOID_NEURAL_NODES.flatMap((node) => node.capabilities)
)];

/**
 * Concrete operator/runtime surface of the V3 hierarchy.  Keep this separate
 * from HUMANOID_NEURAL_CAPABILITIES: node capabilities describe structural
 * responsibilities, while this catalog describes calls the running Harness
 * can actually perform. Motor Intent intentionally has no raw dense-motion
 * authoring or physical execution tool.
 */
export const HUMANOID_NEURAL_RUNTIME_CAPABILITIES = [
  "recall_goal_history",
  "submit_goal_candidates",
  "select_goal_candidate",
  "retire_goal_epoch",
  "continue_goal_epoch",
  "observe_humanoid",
  "recall_embodied_history",
  "submit_humanoid_skill_plan",
  "begin_humanoid_skill",
  "plan_humanoid_skill",
  "plan_whole_body_motion_candidates",
  "plan_humanoid_navigation",
  "execute_humanoid_skill",
  "execute_whole_body_motion",
  "execute_humanoid_navigation",
  "remove_world_block"
] as const;

export const HUMANOID_NEURAL_SIGNAL_CONTRACTS = [
  signalContract("executive", "goalManager", "descending", "goal_context"),
  signalContract("goalManager", "executive", "ascending", "goal_selected", "escalation"),
  signalContract("executive", "actionSelection", "descending", "goal_context", "goal_selected", "skill_commitment", "execution_receipt", "skill_completed", "skill_failed", "prediction_error", "escalation"),
  signalContract("actionSelection", "executive", "ascending", "perceptual_belief", "skill_commitment", "skill_completed", "skill_failed", "escalation"),
  signalContract("actionSelection", "perceptionManager", "descending", "goal_context", "goal_selected", "skill_commitment", "execution_receipt", "skill_completed", "skill_failed"),
  signalContract("perceptionManager", "actionSelection", "ascending", "perceptual_belief", "escalation"),
  signalContract("actionSelection", "sensorimotorManager", "descending", "goal_context", "goal_selected", "perceptual_belief", "skill_commitment", "prediction_error", "skill_failed", "escalation"),
  signalContract("sensorimotorManager", "actionSelection", "ascending", "skill_proposal", "skill_commitment", "rollout_result", "prediction_error", "execution_receipt", "skill_completed", "skill_failed", "escalation"),
  signalContract("perceptionManager", "sensorFusion", "descending", "goal_context"),
  signalContract("sensorFusion", "perceptionManager", "ascending", "sensory_evidence"),
  signalContract("perceptionManager", "sceneInterpreter", "descending", "goal_context", "sensory_evidence"),
  signalContract("sceneInterpreter", "perceptionManager", "ascending", "scene_interpretation"),
  signalContract("perceptionManager", "memoryRetriever", "descending", "goal_context", "sensory_evidence"),
  signalContract("memoryRetriever", "perceptionManager", "ascending", "memory_retrieval"),
  signalContract("sensorimotorManager", "affordance", "descending", "goal_context", "goal_selected", "perceptual_belief", "skill_commitment"),
  signalContract("affordance", "sensorimotorManager", "ascending", "affordance_hypothesis"),
  signalContract("sensorimotorManager", "risk", "descending", "goal_context", "goal_selected", "perceptual_belief", "skill_commitment"),
  signalContract("risk", "sensorimotorManager", "ascending", "risk_assessment", "escalation"),
  signalContract("sensorimotorManager", "predictive", "descending", "goal_context", "skill_commitment", "rollout_result", "execution_receipt"),
  signalContract("predictive", "sensorimotorManager", "ascending", "forward_prediction", "prediction_error", "escalation"),
  signalContract("sensorimotorManager", "premotor", "descending", "goal_context", "goal_selected", "perceptual_belief", "affordance_hypothesis", "risk_assessment", "skill_proposal", "skill_commitment"),
  signalContract("premotor", "sensorimotorManager", "ascending", "rollout_result", "escalation"),
  signalContract("premotor", "motorIntent", "descending", "skill_proposal", "skill_commitment", "perceptual_belief"),
  signalContract("motorIntent", "premotor", "ascending", "rollout_result", "escalation"),
  signalContract("motorIntent", "rolloutGate", "descending", "motor_intent"),
  signalContract("rolloutGate", "motorIntent", "ascending", "rollout_result", "escalation"),
  signalContract("sensorimotorManager", "executionDispatcher", "descending", "skill_commitment"),
  signalContract("executionDispatcher", "sensorimotorManager", "ascending", "execution_receipt", "prediction_error", "skill_completed", "skill_failed", "escalation"),
  signalContract("executionDispatcher", "executor", "descending", "skill_commitment", "motor_intent", "rollout_result"),
  signalContract("executor", "executionDispatcher", "ascending", "execution_receipt", "prediction_error", "skill_completed", "skill_failed", "escalation"),
  signalContract("executor", "reflex", "descending", "skill_commitment", "motor_intent"),
  signalContract("reflex", "executor", "ascending", "execution_receipt", "prediction_error", "skill_completed", "skill_failed", "escalation"),
  signalContract("reflex", "body", "descending", "motor_intent"),
  signalContract("body", "reflex", "ascending", "sensory_evidence", "prediction_error"),
  signalContract("sensorimotorManager", "recovery", "descending", "perceptual_belief", "risk_assessment", "prediction_error", "skill_failed", "escalation"),
  signalContract("recovery", "sensorimotorManager", "ascending", "skill_proposal", "escalation"),
  signalContract("rolloutGate", "predictive", "reentrant", "rollout_result"),
  signalContract("reflex", "sensorimotorManager", "reentrant", "prediction_error", "execution_receipt", "skill_completed", "skill_failed", "escalation"),
  signalContract("body", "perceptionManager", "reentrant", "sensory_evidence")
] as const;

export function humanoidNeuralHierarchyNodes(input: {
  mission: string;
  goalSummary: string;
  predicateIndexes: readonly number[];
  at: string;
}): Record<string, TaskNode> {
  return Object.fromEntries(HUMANOID_NEURAL_NODES.map((descriptor) => {
    const parentId = descriptor.parentKey === null
      ? null
      : HUMANOID_NEURAL_AGENT_IDS[descriptor.parentKey];
    const childIds = descriptor.childKeys.map(
      (key) => HUMANOID_NEURAL_AGENT_IDS[key]
    );
    const root = descriptor.key === "executive";
    const node: TaskNode = {
      id: descriptor.id,
      name: descriptor.name,
      parent_id: parentId,
      child_ids: childIds,
      objective: root ? input.mission : descriptor.objective,
      success_criteria: root
        ? [input.goalSummary, ...descriptor.criteria]
        : [...descriptor.criteria],
      evidence_requirements: [],
      goal_predicate_indexes: root ? [...input.predicateIndexes] : [],
      capabilities: [...descriptor.capabilities],
      may_delegate: descriptor.mayDelegate,
      references: [],
      depth: humanoidNeuralNodeDepth(descriptor.key),
      status: root ? "active" : "ready",
      steps_used: 0,
      model_calls_used: 0,
      created_at: input.at,
      updated_at: input.at
    };
    return [node.id, node];
  }));
}

export function reconcileHumanoidNeuralHierarchyNodes(input: {
  nodes: Readonly<Record<string, TaskNode>>;
  mission: string;
  goalSummary: string;
  predicateIndexes: readonly number[];
  at: string;
}): Record<string, TaskNode> {
  const canonical = humanoidNeuralHierarchyNodes(input);
  for (const [agentId, next] of Object.entries(canonical)) {
    const previous = input.nodes[agentId];
    if (!previous) continue;
    next.model_calls_used = previous.model_calls_used;
    next.steps_used = previous.steps_used;
    next.created_at = previous.created_at;
    next.updated_at = previous.updated_at;
    if (["completed", "failed", "waiting", "active"].includes(previous.status)) {
      next.status = previous.status;
    }
  }
  return canonical;
}

export function assertHumanoidNeuralHierarchyContract(): void {
  const byKey = new Map(HUMANOID_NEURAL_NODES.map((node) => [node.key, node]));
  const ids = new Set(HUMANOID_NEURAL_NODES.map((node) => node.id));
  if (byKey.size !== HUMANOID_NEURAL_NODES.length
    || ids.size !== HUMANOID_NEURAL_NODES.length) {
    throw new Error("Neural hierarchy node keys and ids must be unique");
  }
  const childOwners = new Map<HumanoidNeuralAgentKey, HumanoidNeuralAgentKey>();
  let physicalWriters = 0;
  for (const node of HUMANOID_NEURAL_NODES) {
    if (node.physicalWriteAuthority) physicalWriters += 1;
    if (node.executionKind === "model_agent"
      && node.sessionMode !== "independent_file_session") {
      throw new Error(`Every model Agent requires its own FileSession: ${node.id}`);
    }
    if (node.executionKind !== "model_agent" && node.sessionMode !== "none") {
      throw new Error(`Non-model neural nodes cannot own SDK Sessions: ${node.id}`);
    }
    const expectedOrchestrationKind: HumanoidNeuralOrchestrationKind
      = node.key === "executive"
        ? "root_runner"
        : node.key === "recovery"
          ? "exclusive_lease_episode"
          : node.executionKind === "model_agent"
            ? "agent_tool"
            : node.executionKind === "learned_controller"
              ? "controller_loop"
              : node.key === "body"
                ? "physical_plant"
                : "runtime_service";
    if (node.orchestrationKind !== expectedOrchestrationKind) {
      throw new Error(
        `Neural node has an invalid orchestration kind: ${node.id} -> ${node.orchestrationKind}`
      );
    }
    if (node.mayDelegate !== (node.childKeys.length > 0
      && node.executionKind === "model_agent")) {
      throw new Error(`Only model managers with owned children may delegate: ${node.id}`);
    }
    if (node.parallelSafe !== (node.parallelGroup !== undefined)) {
      throw new Error(
        `Parallel-safe neural nodes must belong to an explicit manager fan-out: ${node.id}`
      );
    }
    if (node.parentKey === null && node.key !== "executive") {
      throw new Error(`Only Executive may be the neural hierarchy root: ${node.id}`);
    }
    if (node.parentKey !== null && !byKey.has(node.parentKey)) {
      throw new Error(`Neural hierarchy node has no parent: ${node.id}`);
    }
    for (const childKey of node.childKeys) {
      const child = byKey.get(childKey);
      if (!child || child.parentKey !== node.key) {
        throw new Error(`Neural hierarchy child does not point back to its parent: ${childKey}`);
      }
      const existing = childOwners.get(childKey);
      if (existing) {
        throw new Error(`Neural hierarchy node has multiple parents: ${childKey}`);
      }
      childOwners.set(childKey, node.key);
    }
  }
  for (const node of HUMANOID_NEURAL_NODES) {
    if (node.parentKey !== null && childOwners.get(node.key) !== node.parentKey) {
      throw new Error(`Neural hierarchy parent does not own child: ${node.id}`);
    }
  }
  if (physicalWriters !== 1
    || !byKey.get("executor")?.physicalWriteAuthority) {
    throw new Error("Neural hierarchy requires exactly one physical write authority");
  }
  const correctionScopeRank: Readonly<Record<
    HumanoidNeuralCorrectionScope,
    number
  >> = {
    none: 0,
    local: 1,
    pathway: 2,
    supervisory: 3
  };
  for (const node of HUMANOID_NEURAL_NODES) {
    if (node.executionKind === "model_agent"
      && ["execution_transaction", "controller_tick", "physics_tick"]
        .includes(node.cadence)) {
      throw new Error(`A model Agent cannot run in a physical-rate loop: ${node.id}`);
    }
    if (node.parentKey !== null) {
      const parent = byKey.get(node.parentKey)!;
      if (correctionScopeRank[node.maximumCorrectionScope]
        > correctionScopeRank[parent.maximumCorrectionScope]) {
        throw new Error(
          `A child cannot own a broader correction scope than its parent: ${node.id}`
        );
      }
    }
  }
  if (byKey.get("executor")?.cadence !== "execution_transaction"
    || byKey.get("reflex")?.cadence !== "controller_tick"
    || byKey.get("body")?.cadence !== "physics_tick") {
    throw new Error(
      "Execution, controller, and body nodes require distinct transaction/control/physics cadences"
    );
  }
  if (byKey.get("executionDispatcher")?.parentKey !== "sensorimotorManager"
    || byKey.get("executor")?.parentKey !== "executionDispatcher") {
    throw new Error(
      "Certified Execution Dispatcher must isolate Sensorimotor reasoning from the Serial Executor"
    );
  }
  if (byKey.get("recovery")?.orchestrationKind !== "exclusive_lease_episode"
    || byKey.get("recovery")?.parentKey !== "sensorimotorManager") {
    throw new Error(
      "Recovery must remain a Sensorimotor child in an exclusive leased episode"
    );
  }
  if (byKey.get("rolloutGate")?.parentKey !== "motorIntent"
    || byKey.get("predictive")?.childKeys.includes("rolloutGate")) {
    throw new Error(
      "Motor Intent must own Rollout Gate; Predictive Critic receives only reentrant rollout feedback"
    );
  }
  const runtimeCapabilities = new Set<string>(
    HUMANOID_NEURAL_RUNTIME_CAPABILITIES
  );
  if (runtimeCapabilities.has("plan_whole_body_motion")
    || !runtimeCapabilities.has("plan_whole_body_motion_candidates")
    || !runtimeCapabilities.has("execute_whole_body_motion")) {
    throw new Error(
      "Motor Intent must use bounded candidate planning while Serial Executor owns physical execution"
    );
  }
  const parallelGroups = new Map<
    NonNullable<HumanoidNeuralNodeDescriptor["parallelGroup"]>,
    HumanoidNeuralNodeDescriptor[]
  >();
  for (const node of HUMANOID_NEURAL_NODES) {
    if (!node.parallelGroup) continue;
    const parent = node.parentKey === null ? undefined : byKey.get(node.parentKey);
    if (!parent || parent.parallelSafe || parent.executionKind !== "model_agent") {
      throw new Error(
        `Parallel fan-out must be owned and joined by one serial model manager: ${node.id}`
      );
    }
    if (node.physicalWriteAuthority || node.executionKind !== "model_agent") {
      throw new Error(
        `Parallel neural branches must be read-only model specialists: ${node.id}`
      );
    }
    const siblings = parallelGroups.get(node.parallelGroup) ?? [];
    siblings.push(node);
    parallelGroups.set(node.parallelGroup, siblings);
  }
  for (const [group, nodes] of parallelGroups) {
    const parentKeys = new Set(nodes.map((node) => node.parentKey));
    if (nodes.length < 2 || parentKeys.size !== 1) {
      throw new Error(
        `Parallel neural group must contain at least two siblings under one manager: ${group}`
      );
    }
  }
}

/**
 * Derive SDK tool concurrency from the hierarchy contract itself. A manager is
 * serial unless it owns a complete group of explicitly read-only siblings.
 */
export function humanoidNeuralManagerParallelToolConcurrency(
  managerKey: HumanoidNeuralAgentKey
): number {
  const groupWidths = new Map<
    NonNullable<HumanoidNeuralNodeDescriptor["parallelGroup"]>,
    number
  >();
  for (const node of HUMANOID_NEURAL_NODES) {
    if (node.parentKey !== managerKey || !node.parallelSafe || !node.parallelGroup) {
      continue;
    }
    groupWidths.set(node.parallelGroup, (groupWidths.get(node.parallelGroup) ?? 0) + 1);
  }
  return Math.max(1, ...groupWidths.values());
}

const SIGNAL_KINDS_BY_ROUTE: ReadonlyMap<string, ReadonlySet<NeuralSignalKind>>
  = new Map(HUMANOID_NEURAL_SIGNAL_CONTRACTS.map((contract) => [
    `${contract.sourceAgentId}->${contract.targetAgentId}:${contract.direction}`,
    new Set<NeuralSignalKind>(contract.signalKinds)
  ]));

export function assertHumanoidNeuralSignalRoute(input: {
  sourceNodeId: string;
  targetNodeId: string;
  direction: "descending" | "ascending" | "reentrant";
  kind: NeuralSignalKind;
}): {
  source: HumanoidNeuralNodeDescriptor;
  target: HumanoidNeuralNodeDescriptor;
} {
  const nodeById: ReadonlyMap<string, HumanoidNeuralNodeDescriptor>
    = HUMANOID_NEURAL_NODE_BY_ID;
  const source = nodeById.get(input.sourceNodeId);
  const target = nodeById.get(input.targetNodeId);
  if (!source || !target) {
    throw new Error(
      `Neural signal route references an unknown node: ${input.sourceNodeId} -> ${input.targetNodeId}`
    );
  }
  if (input.direction === "descending") {
    if (target.parentKey !== source.key) {
      throw new Error("Descending neural signals must travel from parent to direct child");
    }
  } else if (input.direction === "ascending") {
    if (source.parentKey !== target.key) {
      throw new Error("Ascending neural signals must travel from child to direct parent");
    }
  } else {
    const allowedKinds = SIGNAL_KINDS_BY_ROUTE.get(
      `${source.id}->${target.id}:reentrant`
    );
    if (!allowedKinds || !allowedKinds.has(input.kind)) {
      throw new Error(
        `Reentrant neural signal is not part of the hierarchy contract: ${source.id} -> ${target.id}`
      );
    }
  }
  const allowedKinds = SIGNAL_KINDS_BY_ROUTE.get(
    `${source.id}->${target.id}:${input.direction}`
  );
  if (!allowedKinds?.has(input.kind)) {
    throw new Error(
      `Neural signal kind ${input.kind} is forbidden on ${source.id} -> ${target.id}`
    );
  }
  return { source, target };
}

function signalContract(
  source: HumanoidNeuralAgentKey,
  target: HumanoidNeuralAgentKey,
  direction: "descending" | "ascending" | "reentrant",
  ...signalKinds: NeuralSignalKind[]
): {
  sourceAgentId: HumanoidNeuralAgentId;
  targetAgentId: HumanoidNeuralAgentId;
  direction: "descending" | "ascending" | "reentrant";
  signalKinds: readonly NeuralSignalKind[];
} {
  return {
    sourceAgentId: HUMANOID_NEURAL_AGENT_IDS[source],
    targetAgentId: HUMANOID_NEURAL_AGENT_IDS[target],
    direction,
    signalKinds
  };
}

function humanoidNeuralNodeDepth(key: HumanoidNeuralAgentKey): number {
  const byKey = new Map(HUMANOID_NEURAL_NODES.map((node) => [node.key, node]));
  const visited = new Set<HumanoidNeuralAgentKey>();
  let depth = 0;
  let cursor = byKey.get(key);
  while (cursor?.parentKey !== null && cursor?.parentKey !== undefined) {
    if (visited.has(cursor.key)) {
      throw new Error(`Neural hierarchy contains a cycle at ${cursor.id}`);
    }
    visited.add(cursor.key);
    depth += 1;
    cursor = byKey.get(cursor.parentKey);
  }
  return depth;
}

assertHumanoidNeuralHierarchyContract();
