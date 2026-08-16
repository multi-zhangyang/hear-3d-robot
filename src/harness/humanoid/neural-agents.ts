import {
  Agent,
  RunContext,
  Runner,
  tool,
  type CallModelInputFilter,
  type FunctionTool,
  type Model,
  type ModelSettings,
  type RunStreamEvent,
  type Session,
  type Tool,
  type ToolUseBehavior
} from "@openai/agents";
import { Mutex } from "async-mutex";
import { z } from "zod";
import {
  providerConfigForAgent,
  type AgentModelProfile,
  type ModelProviderConfig,
  type ProviderConfig
} from "../../config/load.js";
import { JsonValueSchema, type JsonValue } from "../../domain/schema.js";
import type {
  NeuralAuthorityLease,
  NeuralHarnessPhase,
  NeuralPredictionError,
  NeuralHierarchyState,
  NeuralPlanningAction,
  NeuralRolloutCertificate,
  NeuralSafetyInterrupt,
  NeuralSensingAuthority,
  NeuralSignal,
  NeuralSignalKind,
  NeuralSkillCommitment
} from "../../domain/neural-hierarchy.js";
import { NeuralSkillCommitmentSchema } from "../../domain/neural-hierarchy.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import {
  HUMANOID_SKILL_IDS,
  HUMANOID_SKILL_CONTRACTS,
  HumanoidSkillIdSchema,
  HumanoidSkillInvocationSchema
} from "../../domain/humanoid-skill.js";
import { HumanoidSkillPlanProposalSchema } from
  "../../domain/humanoid-skill-plan.js";
import {
  agentInvocationMarker,
  currentAgentHarnessInvocation,
  currentAgentHarnessInvocationChain,
  scopeAgentToolInvocation,
  stableAgentToolInvocationId,
  withAgentInvocation
} from "../agent-scope.js";
import { createToolInputRecovery } from "../tool-input-recovery.js";
import {
  humanoidNeuralAgentToolName,
  type NeuralAgentHierarchy,
  type NeuralRuntimeService
} from "../neural-agent-manifest.js";
import {
  createGoalManagerTools,
  type GoalManagerRuntime
} from "./goal-manager-tools.js";
import {
  createHumanoidActionTools,
  humanoidActionReceiptModelOutput,
  invokeDeterministicHumanoidAction,
  type HumanoidEmbodiedRecallInvoker
} from "./tools.js";
import type {
  HumanoidActionInvoker,
  HumanoidActionReceipt
} from "./runtime.js";
import {
  HUMANOID_NEURAL_AGENT_IDS,
  HUMANOID_NEURAL_NODE_BY_ID,
  HUMANOID_NEURAL_NODES,
  HUMANOID_NEURAL_SIGNAL_CONTRACTS,
  humanoidNeuralManagerParallelToolConcurrency,
  type HumanoidNeuralAgentId,
  type HumanoidNeuralAgentKey
} from "./neural-hierarchy-contract.js";
import {
  HUMANOID_EXPERIENCE_OUTCOMES,
  HUMANOID_GOAL_PREDICATE_TYPES,
  type HumanoidEmbodiedRecallRequest
} from "./embodied-recall.js";
import { modelReceiptDetail } from "./receipt-context.js";

const EmptyDelegationSchema = z.object({}).strict();
const NEURAL_OUTPUT_SUBMISSION_TOOL_NAME = "submit_neural_output";
const MODEL_EPISODE_SIGNAL_TTL_REVISIONS = 10_000;
const GENERIC_NEURAL_SUBMISSION_AGENT_KEYS: ReadonlySet<HumanoidNeuralAgentKey>
  = new Set([
    "perceptionManager",
    "sceneInterpreter",
    "sensorimotorManager",
    "affordance",
    "risk",
    "predictive",
    "recovery"
  ]);
type NeuralDelegationSourceSignalContract = Readonly<{
  requiredKinds: readonly NeuralSignalKind[];
  allowedKinds: readonly NeuralSignalKind[];
}>;
const CURRENT_SENSORY_EVIDENCE_SOURCE_CONTRACT:
  NeuralDelegationSourceSignalContract = {
    requiredKinds: ["sensory_evidence"],
    allowedKinds: ["sensory_evidence"]
  };
const MOTOR_INTENT_PLANNING_ACTIONS: ReadonlySet<string> = new Set([
  "plan_humanoid_skill",
  "plan_whole_body_motion_candidates",
  "plan_humanoid_navigation"
]);
const MOTOR_INTENT_TRANSIT_RECOVERY_MODALITIES = [
  "whole_body_clearance",
  "alternate_navigation"
] as const;
type MotorIntentTransitRecoveryModality =
  typeof MOTOR_INTENT_TRANSIT_RECOVERY_MODALITIES[number];
const MotorIntentTransitRecoveryAttemptSchema = z.object({
  protocol: z.literal("motor_intent_transit_recovery_attempt_v1"),
  modality: z.enum(MOTOR_INTENT_TRANSIT_RECOVERY_MODALITIES),
  action: z.enum([
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
  ])
}).strict();
const MOTOR_INTENT_SKILL_STATE_ACTIONS: ReadonlySet<string> = new Set([
  "submit_humanoid_skill_plan",
  "begin_humanoid_skill"
]);
const NeuralJsonTextSchema = z.string().trim().min(2).max(128_000)
  .superRefine((value, context) => {
    try {
      z.json().parse(JSON.parse(value));
    } catch {
      context.addIssue({
        code: "custom",
        message: "Value must contain one valid JSON payload"
      });
    }
  });
const NeuralPayloadSchema = JsonValueSchema.superRefine((value, context) => {
  if (JSON.stringify(value).length > 128_000) {
    context.addIssue({
      code: "custom",
      message: "Neural payload exceeds 128000 serialized characters"
    });
  }
});
const BoundedSkillParametersSchema = z.record(
  z.string().trim().min(1),
  JsonValueSchema
);
const BoundedSkillProposalSchema = z.object({
  skill: HumanoidSkillIdSchema.describe(
    "Exactly one bounded Skill id from the live humanoid Skill catalog"
  ),
  phase: z.string().trim().min(1).max(256).describe(
    "One process phase declared by the selected Skill contract"
  ),
  params: BoundedSkillParametersSchema.describe(
    "Exact invocation parameters for the selected Skill; do not repeat the skill id"
  ),
  rationale: z.string().trim().min(1).max(8_000)
}).strict().superRefine((proposal, context) => {
  if (Object.prototype.hasOwnProperty.call(proposal.params, "skill")) {
    context.addIssue({
      code: "custom",
      path: ["params", "skill"],
      message: "params must not repeat or override proposed_skill.skill"
    });
  }
  const invocation = HumanoidSkillInvocationSchema.safeParse({
    ...proposal.params,
    skill: proposal.skill
  });
  if (!invocation.success) {
    for (const issue of invocation.error.issues.slice(0, 16)) {
      const path = issue.path[0] === "skill" ? issue.path : ["params", ...issue.path];
      context.addIssue({
        code: "custom",
        path,
        message: `Invalid ${proposal.skill} invocation: ${issue.message}`
      });
    }
  }
  const validPhases = HUMANOID_SKILL_CONTRACTS[proposal.skill].process.map(
    (entry) => entry.phase
  );
  if (!validPhases.includes(proposal.phase)) {
    context.addIssue({
      code: "custom",
      path: ["phase"],
      message: `Phase must be one of the ${proposal.skill} contract phases: ${validPhases.join(", ")}`
    });
  }
});
const BoundedSkillPhaseSchema = z.object({
  skill: HumanoidSkillIdSchema,
  phase: z.string().trim().min(1).max(256)
}).strict();
const BoundedSkillProposalPayloadSchema = z.object({
  proposed_skill: BoundedSkillProposalSchema,
  phase_sequence: z.array(BoundedSkillPhaseSchema).max(32).optional()
}).strict();
const EstablishSkillCommitmentSchema = z.object({
  skill: z.string().trim().min(1).max(2_000),
  termination_contract_json: NeuralJsonTextSchema,
  source_signal_ids: z.array(z.string().uuid()).max(64).default([]).describe(
    "Optional only when more than one current Sensorimotor proposal is semantically admissible. The Harness binds the unique latest direct proposal automatically."
  )
}).strict();
const TransitionSkillCommitmentSchema = z.object({
  commitment_id: z.string().uuid(),
  source_signal_ids: z.array(z.string().uuid()).max(64).default([]).describe(
    "Harness-bound lifecycle feedback. Supply ids only when multiple current feedback signals are semantically admissible."
  ),
  reason: z.string().trim().min(1).max(2_000)
}).strict();
const AcknowledgeSafetyInterruptSchema = z.object({
  interrupt_id: z.string().uuid(),
  reason: z.string().trim().min(1).max(2_000)
}).strict();
const AuthorizeSkillExecutionSchema = z.object({
  commitment_id: z.string().uuid(),
  reason: z.string().trim().min(1).max(2_000)
}).strict();
const CycleCompletionSchema = z.object({
  summary: z.string().trim().min(1).max(8_000),
  perceptual_belief_signal_ids: z.array(z.string().uuid()).max(1).default([]).describe(
    "Optional only if multiple current post-execution beliefs are admissible. The Harness binds the unique direct belief automatically."
  ),
  next_intent: z.string().trim().min(1).max(4_000).optional()
}).strict();
const SatisfiedGoalCompletionSchema = z.object({
  summary: z.string().trim().min(1).max(8_000)
}).strict();
const ExecutionTaskSchema = z.object({
  objective: z.string().trim().min(1),
  // The model owns only the direct parent -> child decision edge. Certificate
  // internals and planning identifiers are durable Harness state and must not
  // be copied through another model response.
  source_signal_ids: z.array(z.string().uuid()).max(1).default([]).describe(
    "Harness-bound direct executing commitment; omit when it is unique."
  )
}).strict();

const RelevantMemoryIntentSchema = z.object({
  retrieval_mode: z.enum(["active_goal", "recent"])
    .describe(
      "active_goal retrieves semantically matching experience for the current Goal; recent retrieves a chronological history window"
    ),
  outcome_scope: z.enum(["all", "successful", "unsuccessful"])
    .describe(
      "High-level outcome preference. The Harness maps it to the durable memory outcome vocabulary"
    )
}).strict();

const ResolvedExecutionSchema = z.object({
  kind: z.literal("execute_plan"),
  planning_action: z.enum([
    "plan_humanoid_skill",
    "plan_whole_body_motion",
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
  ]),
  planning_transaction_id: z.string().trim().min(1)
}).strict();

const NeuralAgentOutputSchema = z.object({
  signal_kind: z.enum([
    "goal_context",
    "goal_selected",
    "scene_interpretation",
    "memory_retrieval",
    "perceptual_belief",
    "affordance_hypothesis",
    "risk_assessment",
    "forward_prediction",
    "prediction_error",
    "skill_proposal",
    "skill_commitment",
    "motor_intent",
    "rollout_result",
    "execution_receipt",
    "skill_completed",
    "skill_failed",
    "escalation"
  ]),
  summary: z.string().trim().min(1).max(8_000),
  payload_json: NeuralJsonTextSchema,
  source_signal_ids: z.array(z.string().uuid()).max(64).default([]),
  confidence: z.number().min(0).max(1)
}).strict();

const ExecutiveOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum([
    "goal_context",
    "perceptual_belief",
    "skill_commitment",
    "skill_completed",
    "skill_failed",
    "escalation"
  ])
}).strict();

const ActionSelectionOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum([
    "perceptual_belief",
    "skill_commitment",
    "skill_completed",
    "skill_failed",
    "escalation"
  ])
}).strict();

const PerceptionOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum(["perceptual_belief", "escalation"]),
  summary: z.string().trim().min(1).max(1_000)
}).strict();

const CompactPerceptualBeliefPayloadSchema = z.object({
  protocol: z.literal("compact_perceptual_belief_v1"),
  world_revision: z.number().int().nonnegative(),
  goal_state: z.enum(["unsatisfied", "satisfied", "unknown"]),
  observations: z.array(z.string().trim().min(1).max(320)).min(1).max(8),
  uncertainties: z.array(z.string().trim().min(1).max(320)).max(4).default([]),
  changed_entity_ids: z.array(z.string().trim().min(1).max(160)).max(16).default([]),
  safety_relevant: z.array(z.string().trim().min(1).max(320)).max(4).default([]),
  escalation_reason: z.string().trim().max(1_000).optional()
}).strict();

const PremotorMotorProgramSchema = z.object({
  protocol: z.literal("premotor_motor_program_v1"),
  skill_plan: HumanoidSkillPlanProposalSchema
}).strict();

const SensorimotorOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum([
    "skill_proposal",
    "rollout_result",
    "prediction_error",
    "execution_receipt",
    "skill_completed",
    "skill_failed",
    "escalation"
  ])
}).strict().superRefine(validateBoundedSkillProposalOutput);

const GoalValuationOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum(["goal_selected", "escalation"])
}).strict();

const SceneInterpretationOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.literal("scene_interpretation")
}).strict();

const MemoryRetrievalOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.literal("memory_retrieval")
}).strict();

const AffordanceOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.literal("affordance_hypothesis")
}).strict();

const RiskOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum(["risk_assessment", "escalation"])
}).strict();

const PredictiveOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum(["forward_prediction", "prediction_error", "escalation"]),
  accepted: z.boolean()
}).strict().superRefine((output, context) => {
  if ((output.signal_kind === "forward_prediction") !== output.accepted) {
    context.addIssue({
      code: "custom",
      path: ["accepted"],
      message: "Only an explicitly accepted forward prediction may authorize execution"
    });
  }
});

const PremotorOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum(["rollout_result", "escalation"])
}).strict();

const MotorIntentOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum(["rollout_result", "escalation"])
}).strict();

const RecoveryOutputSchema = NeuralAgentOutputSchema.extend({
  signal_kind: z.enum(["skill_proposal", "escalation"])
}).strict().superRefine(validateBoundedSkillProposalOutput);

export interface HumanoidNeuralAgentRuntime extends HumanoidActionInvoker,
  HumanoidEmbodiedRecallInvoker, GoalManagerRuntime {
  currentWorldRevision(): number;
  neuralExecutionAvailable(): boolean;
  cycleCompletionReadiness(): {
    status: "ready" | "not_ready";
    evidence_transaction_ids: string[];
    observed_after_execution: boolean;
  };
  autonomyReadiness():
    | "goal_selection"
    | "goal_transition"
    | "complete_satisfied_goal"
    | "observe_or_plan"
    | "plan"
    | "post_failure_observation"
    | "replan_or_retire"
    | "execute_plan"
    | "post_execution"
    | "complete_cycle";
  recoveryFailureEvidence(): JsonValue | null;
  validateCycleEvidence(evidenceTransactionIds: readonly string[]): HumanoidActionReceipt;
  validateSatisfiedGoal(): JsonValue;
  neuralHierarchyState(): NeuralHierarchyState;
  neuralHarnessPhase(): NeuralHierarchyState["harness_phase"];
  neuralSkillCommitmentOutcome(commitment: NeuralSkillCommitment): {
    status: "completed" | "failed" | "in_progress";
    detail: JsonValue;
  };
  neuralSkillProposalAdmission?(signal: NeuralSignal): {
    accepted: boolean;
    reason?: string;
    detail?: JsonValue;
    invocation?: JsonValue;
    relation?: "direct" | "prerequisite" | "recovery" | "safety";
    predicate_index?: number | null;
  };
  neuralNodeEnabled(input: {
    nodeId: string;
    phases: readonly NeuralHarnessPhase[];
    signalKinds?: readonly NeuralSignalKind[];
    requireCommitment?: boolean;
  }): boolean;
  contextAnchor(agentId: string): JsonValue;
  pendingNeuralSignals(input?: {
    targetNodeId?: string;
    kinds?: readonly NeuralSignalKind[];
    invocationId?: string;
  }): NeuralSignal[];
  pendingNeuralSafetyInterrupts(): NeuralSafetyInterrupt[];
  acknowledgeNeuralSafetyInterrupt(input: {
    interruptId: string;
    acknowledgedByNodeId: string;
  }): Promise<{
    interrupt: NeuralSafetyInterrupt;
    commitment: NeuralSkillCommitment | null;
  }>;
  publishNeuralSignal(input: {
    kind: NeuralSignalKind;
    pathway: NeuralSignal["pathway"];
    direction: NeuralSignal["direction"];
    sourceNodeId: string;
    targetNodeId: string;
    ttlRevisions: number;
    priority: number;
    causalParentIds?: readonly string[];
    authorityLeaseId?: string | null;
    sourceAuthorityLeaseId?: string | null;
    invocationId?: string;
    parentInvocationId?: string | null;
    parentEpisodeId?: string;
    payload: JsonValue;
  }): Promise<NeuralSignal>;
  consumeNeuralSignals(
    consumerNodeId: string,
    signalIds: readonly string[]
  ): Promise<void>;
  transitionNeuralHarnessPhase(input: {
    phase: NeuralHarnessPhase;
    enteredByNodeId: string;
    reason: string;
    goalEpochId?: string | null;
    commitmentId?: string | null;
  }): Promise<NeuralHierarchyState["harness_phase"]>;
  issueNeuralAuthorityLease(input: {
    issuingParentNodeId: string;
    targetChildNodeId: string;
    allowedSignalKinds: readonly NeuralSignalKind[];
    correctionScope?: NeuralAuthorityLease["correction_scope"];
    ttlRevisions?: number;
    ttlMs?: number;
    exclusive?: boolean;
    suspendLeaseIds?: readonly string[];
    invocationId?: string;
    parentInvocationId?: string | null;
    parentEpisodeId: string;
  }): Promise<NeuralAuthorityLease>;
  closeNeuralAuthorityLease(input: {
    leaseId: string;
    closedByNodeId: string;
    reason: string;
    status?: "closed" | "revoked" | "expired";
    resumeSuspended?: boolean;
  }): Promise<void>;
  establishNeuralSkillCommitment(input: {
    ownerNodeId: string;
    goalEpochId: string;
    skill: string;
    terminationContract: JsonValue;
    sourceSignalIds: readonly string[];
  }): Promise<NeuralSkillCommitment>;
  transitionNeuralSkillCommitment(input: {
    ownerNodeId: string;
    commitmentId: string;
    state: NeuralSkillCommitment["state"];
    sourceSignalIds?: readonly string[];
  }): Promise<NeuralSkillCommitment>;
  recordNeuralPredictionError(input: {
    observerNodeId: string;
    sourceSignalId: string;
    magnitude: number;
    tolerance: number;
    correctionScope: NeuralPredictionError["correction_scope"];
    detail: JsonValue;
  }): Promise<NeuralPredictionError>;
  issueNeuralRolloutCertificate(input: {
    issuedByNodeId: string;
    commitmentId: string;
    goalEpochId: string;
    planningTransactionId: string;
    planningAction: NeuralPlanningAction;
    rolloutSignalId: string;
    predictiveSignalId: string;
    rolloutPayloadSha256: string;
    rolloutInvocationId: string;
    predictiveInvocationId: string;
    ttlRevisions?: number;
  }): Promise<NeuralRolloutCertificate>;
  captureNeuralObservation(input: {
    transactionId: string;
    authority: NeuralSensingAuthority;
    signal?: AbortSignal;
  }): Promise<HumanoidActionReceipt>;
}

export interface HumanoidNeuralAgentHierarchy extends Omit<
  NeuralAgentHierarchy,
  "root" | "agents"
> {
  root: Agent<unknown, "text">;
  agents: ReadonlyMap<string, Agent<any, any>>;
  agent(agentId: HumanoidNeuralAgentId): Agent<any, any> | undefined;
  outputSchema(agentId: HumanoidNeuralAgentId): z.ZodObject | undefined;
}

const CERTIFIED_EXECUTION_DISPATCHER_TOOLS = new WeakMap<
  HumanoidNeuralAgentRuntime,
  {
    tool: FunctionTool<unknown, any, unknown>;
    parent: Agent<unknown, "text">;
  }
>();

const DIRECT_ACTION_SELECTION_TOOLS = new WeakMap<
  HumanoidNeuralAgentRuntime,
  {
    tool: FunctionTool<unknown, any, unknown>;
    parent: Agent<unknown, "text">;
  }
>();

const DETERMINISTIC_CYCLE_TOOLS = new WeakMap<
  HumanoidNeuralAgentRuntime,
  {
    completion: FunctionTool<unknown, any, unknown>;
    satisfied: FunctionTool<unknown, any, unknown>;
    parent: Agent<unknown, "text">;
  }
>();

const DIRECT_GOAL_MANAGER_TOOLS = new WeakMap<
  HumanoidNeuralAgentRuntime,
  {
    tool: FunctionTool<unknown, any, unknown>;
    parent: Agent<unknown, "text">;
  }
>();

export function createHumanoidNeuralAgentHierarchy(input: {
  createModel: (agentId: string, provider: ModelProviderConfig) => Model;
  createSession: (agentId: string) => Session;
  callModelInputFilter: CallModelInputFilter;
  provider: ProviderConfig;
  runtime: HumanoidNeuralAgentRuntime;
  onAgentStream?: (agentId: string, event: RunStreamEvent) => void | Promise<void>;
}): HumanoidNeuralAgentHierarchy {
  const models = new Set<Model>();
  const sessions = new Map<string, Session>();
  const sessionOwners = new Set<Session>();
  const agents = new Map<string, Agent<any, any>>();
  const outputSchemas = new Map<string, z.ZodObject>();
  const ownModel = (key: HumanoidNeuralAgentKey): Model => {
    const agentId = HUMANOID_NEURAL_AGENT_IDS[key];
    const profile = humanoidNeuralAgentProfile(agentId);
    const provider = providerConfigForAgent(input.provider, agentId, profile);
    const configuredModel = input.createModel(
      agentId,
      provider
    );
    if (models.has(configuredModel)) {
      throw new Error(`Neural Agents cannot share one Model facade: ${agentId}`);
    }
    models.add(configuredModel);
    return configuredModel;
  };
  const ownSession = (key: HumanoidNeuralAgentKey): Session => {
    const agentId = HUMANOID_NEURAL_AGENT_IDS[key];
    const session = input.createSession(agentId);
    if (sessionOwners.has(session)) {
      throw new Error(`Neural Agents cannot share one Session: ${agentId}`);
    }
    sessionOwners.add(session);
    sessions.set(agentId, session);
    return session;
  };
  for (const node of HUMANOID_NEURAL_NODES) {
    if (node.executionKind === "model_agent") ownSession(node.key);
  }
  const settingsFor = (
    key: HumanoidNeuralAgentKey,
    options: {
      parallel?: boolean;
      toolChoice?: "auto" | "required";
      thinking?: "enabled" | "disabled";
    } = {}
  ): ModelSettings => {
    const agentId = HUMANOID_NEURAL_AGENT_IDS[key];
    const provider = providerConfigForAgent(
      input.provider,
      agentId,
      humanoidNeuralAgentProfile(agentId)
    );
    // Formal progress still requires a verified tool result in each node's
    // ToolUseBehavior. Reasoning turns stay provider-neutral and auto-select
    // tools. The deterministic execution phase is advanced directly by the
    // Harness and never changes one Agent Session's reasoning mode.
    const toolChoice = options.toolChoice ?? provider.toolChoice ?? "auto";
    return {
      temperature: provider.temperature,
      ...(provider.reasoningEffort === undefined || options.thinking === "disabled"
        ? {}
        : { reasoning: { effort: provider.reasoningEffort } }),
      ...(provider.maxOutputTokens === undefined
        ? {}
        : { maxTokens: provider.maxOutputTokens }),
      ...(provider.protocol === "openai_compatible"
        && provider.model.toLowerCase().includes("deepseek")
        ? {
            providerData: {
              thinking: { type: options.thinking ?? "enabled" },
              providerOptions: {
                "configured-openai-compatible": {
                  thinking: { type: options.thinking ?? "enabled" }
                }
              }
            }
          }
        : {}),
      parallelToolCalls: options.parallel === true
        && humanoidNeuralManagerParallelToolConcurrency(key) > 1,
      toolChoice
    };
  };
  const register = <TOutput extends z.ZodObject>(
    key: HumanoidNeuralAgentKey,
    outputType: TOutput,
    tools: Tool[],
    options: {
      parallel?: boolean;
      toolChoice?: "auto" | "required";
      thinking?: "enabled" | "disabled";
      extraInstructions?: readonly string[];
      toolUseBehavior?: ToolUseBehavior;
    } = {}
  ): Agent<unknown, "text"> => {
    const descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(
      HUMANOID_NEURAL_AGENT_IDS[key]
    );
    if (!descriptor || descriptor.executionKind !== "model_agent") {
      throw new Error(`Cannot register a non-model neural node: ${key}`);
    }
    const submissionTool = neuralOutputSubmissionTool(
      descriptor.key,
      outputType,
      input.runtime
    );
    // Compatible chat-completions providers receive persistent Session history
    // together with the current tool definitions. A state-dependent tool table
    // can therefore retain an earlier tool call while omitting that tool's
    // definition from a later request. Keep every real Agent capability stable
    // and enforce phase/signal/commitment authority at invoke time instead.
    // Managers that synthesize a join and model-only leaf specialists own the
    // generic typed submission endpoint. Structural pass-through nodes and
    // domain-tool nodes terminate through their owned tool result; advertising
    // a generic endpoint there creates a competing, permanently illegal route.
    const capabilityTools = (GENERIC_NEURAL_SUBMISSION_AGENT_KEYS.has(key)
      ? [...tools, submissionTool]
      : [...tools]
    ).map((candidate) => stableNeuralToolCapability(
      candidate,
      descriptor.id,
      input.runtime
    ));
    const agent = new Agent<unknown, "text">({
      name: descriptor.name,
      instructions: scopedInstructions(descriptor.id, [
        ...baseInstructions(descriptor.key),
        ...(options.extraInstructions ?? [])
  ]),
      model: ownModel(key),
      modelSettings: settingsFor(key, options),
      tools: capabilityTools,
      // A formal Harness transition is a terminal tool result, not a second
      // competing response_format path. With tool_choice=auto, a structured
      // SDK output type invites compatible reasoning models to emit a JSON
      // final answer instead of invoking the tool they selected. Keep the SDK
      // output textual and validate only the terminal tool receipt below.
      outputType: "text",
      resetToolChoice: false,
      toolUseBehavior: neuralOutputToolUseBehavior(
        outputType,
        options.toolUseBehavior
      )
    });
    agents.set(descriptor.id, agent);
    outputSchemas.set(descriptor.id, outputType);
    return agent;
  };
  const asChildTool = (inputTool: {
    parentKey: HumanoidNeuralAgentKey;
    childKey: HumanoidNeuralAgentKey;
    child: Agent<unknown, "text">;
    description: string;
    requiredSignalKind?: NeuralSignalKind;
    sourceSignalContract?: NeuralDelegationSourceSignalContract;
    isEnabled?: () => boolean;
    phases?: readonly NeuralHarnessPhase[];
    requireCommitment?: boolean;
    prepareModelInput?: (input: {
      invocationId: string;
      authorityLeaseId: string;
      signal?: AbortSignal;
    }) => Promise<void>;
  }): FunctionTool<unknown, any, unknown> => {
    const parentId = HUMANOID_NEURAL_AGENT_IDS[inputTool.parentKey];
    const childId = HUMANOID_NEURAL_AGENT_IDS[inputTool.childKey];
    const childOutputSchema = outputSchemas.get(childId);
    if (!childOutputSchema) {
      throw new Error(`Neural child output schema is unavailable: ${childId}`);
    }
    const childDescriptor = HUMANOID_NEURAL_NODE_BY_ID.get(childId)!;
    const delegationSchema = neuralDelegationSchema(
      inputTool.parentKey,
      inputTool.childKey,
      inputTool.requiredSignalKind,
      inputTool.sourceSignalContract
    );
    const maximumToolConcurrency
      = humanoidNeuralManagerParallelToolConcurrency(childDescriptor.key);
    const nestedParallelism = maximumToolConcurrency > 1;
    let descendingSignal: NeuralSignal | undefined;
    let authorityLease: NeuralAuthorityLease | undefined;
    let invocationInputSignalIds: string[] = [];
    let scopedInvocationId: ReturnType<typeof stableAgentToolInvocationId> | undefined;
    let scopedSignal: AbortSignal | undefined;
    // A compatible model may end one SDK run turn in prose after a child tool
    // returns. The parent retries the same structural edge, but that retry is
    // still the same logical child episode and must retain its signal scope.
    let continuedChildEpisode: {
      parentInvocationId: string;
      invocationId: ReturnType<typeof stableAgentToolInvocationId>;
      authorityLeaseId: string;
      descendingSignalId: string;
      invocationInputSignalIds: readonly string[];
    } | undefined;
    const invocationMutex = new Mutex();
    const childTool = scopeAgentToolInvocation(
      childId,
      inputTool.child.asTool({
      toolName: humanoidNeuralAgentToolName(inputTool.childKey),
      toolDescription: inputTool.description,
      parameters: delegationSchema,
      // A child failure is a failed control edge. Preserve the originating
      // exception through Agent.asTool instead of turning it into generic prose
      // that the owning manager could mistake for a neural result.
      errorFunction: null,
      inputBuilder: async ({ params }) => {
        const invocation = requiredHarnessInvocation(childId);
        const parentEpisodeId = requiredParentEpisodeId(parentId);
        const continuation = continuedChildEpisode?.parentInvocationId
          === parentEpisodeId
          && continuedChildEpisode.invocationId === invocation.invocationId
          ? continuedChildEpisode
          : undefined;
        if (continuation) {
          const state = input.runtime.neuralHierarchyState();
          const retainedLease = state.authority_leases[
            continuation.authorityLeaseId
          ];
          const retainedSignal = state.signals[
            continuation.descendingSignalId
          ];
          if (retainedLease?.status === "active"
            && retainedLease.invocation_id === invocation.invocationId
            && retainedLease.parent_invocation_id === invocation.parentInvocationId
            && retainedSignal?.status === "pending"
            && retainedSignal.authority_lease_id === retainedLease.lease_id
            && retainedSignal.invocation_id === invocation.invocationId) {
            authorityLease = retainedLease;
            descendingSignal = retainedSignal;
            invocationInputSignalIds = [...continuation.invocationInputSignalIds];
            return neuralAgentToolTurnContinuationInput(
              input.runtime,
              parentId,
              childId,
              invocation.invocationId
            );
          }
          // A process pause or lease deadline may invalidate the in-memory
          // continuation. Recreate the edge once from durable state below;
          // never treat an expired lease as current authority.
          continuedChildEpisode = undefined;
        }
        await prepareHarnessPhaseForChild(
          input.runtime,
          inputTool.parentKey,
          inputTool.childKey,
          params.signal_kind
        );
        const activeCommitment = input.runtime.neuralHierarchyState()
          .active_skill_commitment;
        const attachOwnedCommitment = inputTool.parentKey === "actionSelection"
          && inputTool.childKey === "sensorimotorManager"
          && activeCommitment !== null
          && !["completed", "failed", "released"].includes(activeCommitment.state)
          && params.signal_kind !== "skill_commitment";
        const suppliedSourceSignals = params.source_signal_ids.map(
          (signalId) => input.runtime.neuralHierarchyState().signals[signalId]
        );
        if (suppliedSourceSignals.some((signal) => signal === undefined)) {
          throw new Error("Neural delegation references an unknown source signal");
        }
        const selectedSourceSignals = inputTool.parentKey === "sensorimotorManager"
          && inputTool.childKey === "predictive"
          && params.signal_kind === "rollout_result"
          ? [currentDirectPremotorRollout(input.runtime, parentEpisodeId)]
          : suppliedSourceSignals as NeuralSignal[];
        const lifecycleFeedback = inputTool.parentKey === "executive"
          && inputTool.childKey === "actionSelection"
          && input.runtime.neuralHarnessPhase().phase === "feedback"
          && activeCommitment?.state === "executing"
          ? currentCommitmentLifecycleFeedback(
              input.runtime,
              activeCommitment,
              { pendingOnly: false }
            )
          : undefined;
        const exactSourceSignals = lifecycleFeedback
          && !selectedSourceSignals.some(
            (signal) => signal.signal_id === lifecycleFeedback.signal_id
          )
          ? [...selectedSourceSignals, lifecycleFeedback]
          : selectedSourceSignals;
        const currentBelief = inputTool.parentKey === "actionSelection"
          && inputTool.childKey === "sensorimotorManager"
          ? currentActionSelectionBelief(
              input.runtime,
              exactSourceSignals as NeuralSignal[]
            )
          : undefined;
        const proposalCorrection = inputTool.parentKey === "actionSelection"
          && inputTool.childKey === "sensorimotorManager"
          ? currentSkillProposalAdmissionCorrection(input.runtime)
          : undefined;
        const delegatedMotorProgram: JsonValue | undefined
          = inputTool.parentKey === "premotor"
          && inputTool.childKey === "motorIntent"
          ? JsonValueSchema.parse(
              PremotorMotorProgramSchema.parse(
                (params as { motor_program?: unknown }).motor_program
              )
            )
          : undefined;
        const forwardedParentInputKinds: readonly NeuralSignalKind[]
          = inputTool.parentKey === "sensorimotorManager"
            && inputTool.childKey === "premotor"
            ? [
                "perceptual_belief",
                "skill_commitment"
              ]
            : inputTool.parentKey === "sensorimotorManager"
              && inputTool.childKey === "predictive"
              ? ["skill_commitment"]
            : inputTool.parentKey === "premotor"
              && inputTool.childKey === "motorIntent"
              ? ["perceptual_belief", "skill_commitment"]
              : [];
        const forwardedParentInputs = forwardedParentInputKinds.length > 0
          ? input.runtime.pendingNeuralSignals({
              targetNodeId: parentId,
              kinds: forwardedParentInputKinds
            }).filter((signal) => (
              signal.kind !== params.signal_kind
                && (signal.direction === "descending"
                  ? signal.invocation_id === parentEpisodeId
                  : signal.parent_episode_id === parentEpisodeId)
                && isCurrentNeuralSignal(input.runtime, signal)
            ))
          : [];
        const acceptedProposal = inputTool.parentKey === "sensorimotorManager"
          && inputTool.childKey === "premotor"
          && activeCommitment !== null
          ? acceptedSkillProposal(
              input.runtime.neuralHierarchyState(),
              activeCommitment
            )
          : undefined;
        const routedCausalParentIds = exactSourceSignals.map(
          (signal) => signal.signal_id
        );
        if (proposalCorrection) {
          routedCausalParentIds.push(proposalCorrection.proposalSignalId);
        }
        if (inputTool.childKey === "predictive"
          && params.signal_kind === "rollout_result") {
          const reentrantRollout = resolveReentrantRolloutAncestor(
            input.runtime.neuralHierarchyState(),
            exactSourceSignals as NeuralSignal[],
            childId
          );
          routedCausalParentIds.push(reentrantRollout.signal_id);
        }
        if (inputTool.parentKey === "actionSelection"
          && inputTool.childKey === "perceptionManager"
          && ["feedback", "perception"].includes(
            input.runtime.neuralHarnessPhase().phase
          )
          && activeCommitment !== null
          && (activeCommitment.state === "completed"
            || activeCommitment.state === "failed")) {
          const lifecycleKind = activeCommitment.state === "completed"
            ? "skill_completed"
            : "skill_failed";
          const directLifecycleFeedback = currentCommitmentLifecycleFeedback(
            input.runtime,
            activeCommitment,
            { pendingOnly: false }
          );
          if (!directLifecycleFeedback
            || directLifecycleFeedback.kind !== lifecycleKind) {
            throw new Error(
              `Post-execution Perception requires one committed Sensorimotor ${lifecycleKind} transition`
            );
          }
          // A new Action Selection episode sees only Executive's direct signal.
          // Resolve the durable transition edge inside the Harness and bind it
          // into the next descending signal even after its delivery TTL has
          // elapsed; never ask the model to quote a grandchild's signal id
          // across episodes. The closed commitment is the authority that makes
          // this historical signal a live causal predecessor, not a fallback.
          routedCausalParentIds.push(directLifecycleFeedback.signal_id);
        }
        const descendingPayload = ((inputTool.parentKey === "actionSelection"
          && inputTool.childKey === "sensorimotorManager")
          || (inputTool.parentKey === "sensorimotorManager"
            && inputTool.childKey === "executionDispatcher"))
          && activeCommitment !== null
          && params.signal_kind === "skill_commitment"
          ? activeCommitment
            : {
              control: {
                protocol: "structural_neural_delegation_v1",
                parent_node_id: parentId,
                child_node_id: childId,
                harness_phase: input.runtime.neuralHarnessPhase().phase,
                signal_kind: params.signal_kind
              },
              causal_inputs: exactSourceSignals.map((signal) => ({
                signal_id: signal!.signal_id,
                kind: signal!.kind,
                source_node_id: signal!.source_node_id,
                world_revision: signal!.world_revision,
                payload: causalSemanticProjection(signal!.payload)
              })),
              ...(proposalCorrection
                ? { direct_parent_correction: proposalCorrection.payload }
                : {}),
              ...(delegatedMotorProgram
                ? { motor_program: delegatedMotorProgram }
                : {}),
              ...(inputTool.parentKey === "executive"
                && inputTool.childKey === "actionSelection"
                && input.runtime.neuralHarnessPhase().phase === "safety_interrupt"
                ? {
                    afferent_safety_interrupts:
                      input.runtime.pendingNeuralSafetyInterrupts()
                  }
                : {})
            };
        authorityLease = await input.runtime.issueNeuralAuthorityLease({
          issuingParentNodeId: parentId,
          targetChildNodeId: childId,
          allowedSignalKinds: [...new Set<NeuralSignalKind>([
            params.signal_kind,
            ...(lifecycleFeedback ? [lifecycleFeedback.kind] : []),
            ...(attachOwnedCommitment ? ["skill_commitment" as const] : []),
            ...(currentBelief ? ["perceptual_belief" as const] : []),
            ...(acceptedProposal ? ["skill_proposal" as const] : []),
            ...forwardedParentInputs.map((signal) => signal.kind)
          ])],
          ttlRevisions: params.ttl_revisions,
          ttlMs: 120_000,
          invocationId: invocation.invocationId,
          parentInvocationId: invocation.parentInvocationId,
          parentEpisodeId
        });
        descendingSignal = await input.runtime.publishNeuralSignal({
          kind: params.signal_kind,
          pathway: childDescriptor.pathway,
          direction: "descending",
          sourceNodeId: parentId,
          targetNodeId: childId,
          ttlRevisions: params.ttl_revisions,
          priority: params.priority,
          causalParentIds: [...new Set(routedCausalParentIds)],
          authorityLeaseId: authorityLease.lease_id,
          invocationId: invocation.invocationId,
          parentInvocationId: invocation.parentInvocationId,
          payload: descendingPayload
        });
        if (lifecycleFeedback) {
          // Physical completion may outlive the Action Selection episode that
          // launched it. Rebind that exact durable feedback into this new SDK
          // invocation as a direct Executive-owned edge, so the child can
          // legally close its commitment without quoting a prior episode's
          // grandchild signal or weakening the lifecycle transition contract.
          await input.runtime.publishNeuralSignal({
            kind: lifecycleFeedback.kind,
            pathway: "executive_control",
            direction: "descending",
            sourceNodeId: parentId,
            targetNodeId: childId,
            ttlRevisions: params.ttl_revisions,
            priority: 100,
            causalParentIds: [lifecycleFeedback.signal_id],
            authorityLeaseId: authorityLease.lease_id,
            invocationId: invocation.invocationId,
            parentInvocationId: invocation.parentInvocationId,
            payload: lifecycleFeedback.payload
          });
        }
        if (attachOwnedCommitment) {
          await input.runtime.publishNeuralSignal({
            kind: "skill_commitment",
            pathway: "sensorimotor_selection",
            direction: "descending",
            sourceNodeId: parentId,
            targetNodeId: childId,
            ttlRevisions: params.ttl_revisions,
            priority: Math.max(params.priority, 90),
            causalParentIds: params.source_signal_ids,
            authorityLeaseId: authorityLease.lease_id,
            invocationId: invocation.invocationId,
            parentInvocationId: invocation.parentInvocationId,
            payload: activeCommitment
          });
        }
        if (currentBelief) {
          await input.runtime.publishNeuralSignal({
            kind: "perceptual_belief",
            pathway: "perceptual_association",
            direction: "descending",
            sourceNodeId: parentId,
            targetNodeId: childId,
            ttlRevisions: params.ttl_revisions,
            priority: Math.max(params.priority, 85),
            causalParentIds: [currentBelief.signal_id],
            authorityLeaseId: authorityLease.lease_id,
            invocationId: invocation.invocationId,
            parentInvocationId: invocation.parentInvocationId,
            payload: currentBelief.payload
          });
        }
        if (acceptedProposal) {
          await input.runtime.publishNeuralSignal({
            kind: "skill_proposal",
            pathway: "premotor_composition",
            direction: "descending",
            sourceNodeId: parentId,
            targetNodeId: childId,
            ttlRevisions: params.ttl_revisions,
            priority: Math.max(params.priority, acceptedProposal.priority),
            causalParentIds: [acceptedProposal.signal_id],
            authorityLeaseId: authorityLease.lease_id,
            invocationId: invocation.invocationId,
            parentInvocationId: invocation.parentInvocationId,
            payload: acceptedProposal.payload
          });
        }
        for (const signal of forwardedParentInputs) {
          await input.runtime.publishNeuralSignal({
            kind: signal.kind,
            pathway: signal.pathway,
            direction: "descending",
            sourceNodeId: parentId,
            targetNodeId: childId,
            ttlRevisions: params.ttl_revisions,
            priority: Math.max(params.priority, signal.priority),
            causalParentIds: [signal.signal_id],
            authorityLeaseId: authorityLease.lease_id,
            invocationId: invocation.invocationId,
            parentInvocationId: invocation.parentInvocationId,
            payload: signal.payload
          });
        }
        invocationInputSignalIds = input.runtime.pendingNeuralSignals({
          targetNodeId: childId,
          invocationId: invocation.invocationId
        }).map((signal) => signal.signal_id);
        await inputTool.prepareModelInput?.({
          invocationId: invocation.invocationId,
          authorityLeaseId: authorityLease.lease_id,
          ...(scopedSignal ? { signal: scopedSignal } : {})
        });
        return neuralInvocationInput(
          input.runtime,
          parentId,
          childId,
          invocation.invocationId
        );
      },
      includeInputSchema: false,
      needsApproval: false,
      runConfig: {
        callModelInputFilter: input.callModelInputFilter,
        toolExecution: { maxFunctionToolConcurrency: maximumToolConcurrency },
        modelSettings: { parallelToolCalls: nestedParallelism }
      },
      runOptions: {
        session: requiredSession(sessions, inputTool.childKey),
        maxTurns: null,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: maximumToolConcurrency }
      },
      resumeState: { contextStrategy: "replace" },
      customOutputExtractor: ({ finalOutput }) => {
        const output = parseNeuralAgentFinalOutput(
          childOutputSchema,
          finalOutput
        );
        return output.success
          ? JSON.stringify(output.data)
          : neuralAgentTurnContinuationReceipt(
              childId,
              humanoidNeuralAgentToolName(inputTool.childKey)
            );
      },
      ...(input.onAgentStream
        ? {
            onStream: ({ event }) => input.onAgentStream!(
              HUMANOID_NEURAL_AGENT_IDS[inputTool.childKey],
              event
            )
          }
        : {})
      }),
      (toolCallId) => scopedInvocationId
        ?? stableAgentToolInvocationId(childId, toolCallId)
    );
    const sdkEnabled = childTool.isEnabled;
    childTool.isEnabled = async (context, agent) => (
      sdkEnabled ? await sdkEnabled(context, agent) : true
    );
    // A direct child edge is a stable capability of its owning Agent. Keep the
    // SDK tool surface identical across turns and persistent Session episodes;
    // changing it from Harness state leaves earlier tool calls in history while
    // removing their definitions from the next compatible request. Phase and
    // signal admissibility remain hard execute-time checks in the wrapper below.
    const inputRecovery = createToolInputRecovery();
    const invoke = childTool.invoke;
    childTool.invoke = (context, rawInput, details) => invocationMutex.runExclusive(async () => {
      // Agent.asTool captures its SDK error handler when constructed. Validate
      // the compatible provider's raw arguments before that internal parser so
      // malformed JSON remains a normal model correction turn owned by the
      // calling parent, without opening a child lease or episode.
      const rejection = inputRecovery.preflight(
        rawInput,
        delegationSchema,
        childTool.name
      );
      if (rejection !== undefined) return rejection;
      const modelDelegation = delegationSchema.parse(JSON.parse(rawInput));
      const parentInvocation = requiredHarnessInvocation(parentId);
      const currentParentSignals = currentDelegationSourceSignals(
        input.runtime,
        parentId,
        parentInvocation.invocationId
      );
      const delegation = {
        ...modelDelegation,
        source_signal_ids: canonicalDirectDelegationSourceSignalIds({
          currentParentSignals,
          requestedSourceSignalIds: modelDelegation.source_signal_ids,
          signalKind: modelDelegation.signal_kind,
          sourceSignalContract: inputTool.sourceSignalContract
        })
      };
      const unknownSourceSignalIds = delegation.source_signal_ids.filter(
        (signalId) => input.runtime.neuralHierarchyState().signals[signalId] === undefined
      );
      if (unknownSourceSignalIds.length > 0) {
        return JSON.stringify({
          accepted: false,
          code: "unknown_neural_source_signal",
          tool: childTool.name,
          unknown_source_signal_ids: unknownSourceSignalIds,
          automatic_actuation: false,
          next_response_contract: {
            mode: "corrected_tool_call_only",
            tool: childTool.name,
            preserve_valid_fields: true,
            narration_allowed: false
          },
          recovery: "Call this same delegation tool once with only exact signal_id values present in the current invocation. Use [] when no causal source signal exists; never invent or substitute a UUID."
        });
      }
      const currentParentSignalIds = new Set(currentParentSignals.map(
        (signal) => signal.signal_id
      ));
      const outOfScopeSourceSignalIds = delegation.source_signal_ids.filter(
        (signalId) => !currentParentSignalIds.has(signalId)
      );
      if (outOfScopeSourceSignalIds.length > 0) {
        return JSON.stringify({
          accepted: false,
          code: "neural_source_signal_out_of_scope",
          tool: childTool.name,
          parent_node_id: parentId,
          parent_episode_id: parentInvocation.invocationId,
          rejected_source_signal_ids: outOfScopeSourceSignalIds,
          current_parent_episode_signal_ids: currentParentSignals.map(
            (signal) => signal.signal_id
          ),
          automatic_actuation: false,
          next_response_contract: {
            mode: "corrected_tool_call_only",
            tool: childTool.name,
            preserve_valid_fields: true,
            narration_allowed: false
          },
          recovery: "Call this same direct-child tool once using only pending signal_id values directed to this parent in the current parent episode. Do not cite a consumed, expired, sibling-owned, foreign-parent, or earlier-episode signal; use [] when no causal source signal is required."
        });
      }
      if (inputTool.isEnabled && !inputTool.isEnabled()) {
        return JSON.stringify({
          accepted: false,
          code: "neural_child_not_enabled",
          tool: childTool.name,
          child_node_id: childId,
          harness_phase: input.runtime.neuralHarnessPhase().phase,
          automatic_actuation: false,
          recovery: "Choose a child tool enabled by the current Harness phase and directed signals; do not bypass the phase barrier."
        });
      }
      if (inputTool.phases && !input.runtime.neuralNodeEnabled({
        nodeId: parentId,
        phases: inputTool.phases,
        ...(inputTool.requireCommitment ? { requireCommitment: true } : {})
      }) && parentId !== HUMANOID_NEURAL_AGENT_IDS.executive) {
        return JSON.stringify({
          accepted: false,
          code: "neural_parent_authority_invalid",
          tool: childTool.name,
          parent_node_id: parentId,
          child_node_id: childId,
          harness_phase: input.runtime.neuralHarnessPhase().phase,
          automatic_actuation: false,
          recovery: "Use only the child edge authorized for the current Harness phase and active commitment."
        });
      }
      const sourceSignalContract = inputTool.sourceSignalContract;
      if (sourceSignalContract) {
        const sourceSignals = delegation.source_signal_ids.map(
          (signalId) => input.runtime.neuralHierarchyState().signals[signalId]!
        );
        const sourceKinds = new Set(sourceSignals.map((signal) => signal.kind));
        const missingSourceKinds = sourceSignalContract.requiredKinds.filter(
          (kind) => !sourceKinds.has(kind)
        );
        const disallowedSourceSignals = sourceSignals.filter(
          (signal) => !sourceSignalContract.allowedKinds.includes(signal.kind)
        );
        if (missingSourceKinds.length > 0 || disallowedSourceSignals.length > 0) {
          const admissibleSourceSignals = currentParentSignals.filter(
            (signal) => sourceSignalContract.allowedKinds.includes(signal.kind)
          );
          return JSON.stringify({
            accepted: false,
            code: "neural_source_signal_kind_mismatch",
            tool: childTool.name,
            parent_node_id: parentId,
            child_node_id: childId,
            parent_episode_id: parentInvocation.invocationId,
            required_source_signal_kinds: sourceSignalContract.requiredKinds,
            allowed_source_signal_kinds: sourceSignalContract.allowedKinds,
            missing_source_signal_kinds: missingSourceKinds,
            rejected_source_signals: disallowedSourceSignals.map((signal) => ({
              signal_id: signal.signal_id,
              kind: signal.kind
            })),
            admissible_current_source_signals: admissibleSourceSignals.map(
              (signal) => ({ signal_id: signal.signal_id, kind: signal.kind })
            ),
            automatic_actuation: false,
            next_response_contract: {
              mode: "corrected_tool_call_only",
              tool: childTool.name,
              preserve_valid_fields: true,
              narration_allowed: false
            },
            recovery: "Call this same direct-child tool once using the exact admissible_current_source_signals required by this structural edge. A signal UUID cannot be relabeled as another semantic kind."
          });
        }
      }
      const childInvocationId = continuedChildEpisode?.parentInvocationId
        === parentInvocation.invocationId
        ? continuedChildEpisode.invocationId
        : durablePendingChildInvocationId({
            runtime: input.runtime,
            parentNodeId: parentId,
            childNodeId: childId,
            parentInvocationId: parentInvocation.invocationId
          }) ?? stableAgentToolInvocationId(childId, details?.toolCall?.callId);
      scopedInvocationId = childInvocationId;
      scopedSignal = details?.signal;
      try {
        let childOutput: Record<string, JsonValue> | undefined;
        while (!childOutput) {
          const output = await invoke(
            context,
            JSON.stringify(delegation),
            details
          );
          const candidate = outputObject(output);
          if (!candidate) {
            throw new Error(
              `${childId} returned a non-neural Agent.asTool result: ${String(output)}`
            );
          }
          if (!isNeuralAgentTurnContinuationReceipt(candidate)) {
            childOutput = candidate;
            break;
          }
          if (!authorityLease || !descendingSignal) {
            throw new Error(
              `Neural child continuation lost its active control edge: ${childId}`
            );
          }
          continuedChildEpisode = {
            parentInvocationId: parentInvocation.invocationId,
            invocationId: childInvocationId,
            authorityLeaseId: authorityLease.lease_id,
            descendingSignalId: descendingSignal.signal_id,
            invocationInputSignalIds: [...invocationInputSignalIds]
          };
          // Agent.asTool has completed one SDK run, but its independent
          // Session and the Harness edge are still live. Retry the child here
          // with the explicit continuation input instead of returning the
          // receipt to its parent and waking the whole hierarchy again.
        }
        continuedChildEpisode = undefined;
        const childSpecificOutput = childOutputSchema.parse(childOutput);
        const routingOutput = NeuralAgentOutputSchema.passthrough().parse(
          childSpecificOutput
        );
        const modelParsed = parseNeuralAgentOutput(
          NeuralAgentOutputSchema.parse({
            signal_kind: routingOutput.signal_kind,
            summary: routingOutput.summary,
            payload_json: routingOutput.payload_json,
            source_signal_ids: routingOutput.source_signal_ids,
            confidence: routingOutput.confidence
          })
        );
        if (!authorityLease) {
          throw new Error(`Neural child invocation did not receive a lease: ${childId}`);
        }
        const invocationId = authorityLease.invocation_id;
        const join = canonicalManagerJoinEvidence({
          runtime: input.runtime,
          managerKey: inputTool.childKey,
          outputKind: modelParsed.signal_kind,
          invocationId,
          modelSourceSignalIds: modelParsed.source_signal_ids
        });
        const parsed = {
          ...modelParsed,
          source_signal_ids: join.sourceSignalIds
        };
        const causality = [...new Set([
          ...(descendingSignal ? [descendingSignal.signal_id] : []),
          ...parsed.source_signal_ids
        ])];
        const joinedSignals = join.signals;
        for (const joined of joinedSignals) {
          causality.push(joined.signal_id);
        }
        const routedPayload = inputTool.childKey === "perceptionManager"
          && parsed.signal_kind === "perceptual_belief"
          ? materializePerceptualBelief(parsed.payload, joinedSignals)
          : parsed.payload;
        if (inputTool.childKey === "motorIntent"
          && parsed.signal_kind === "rollout_result") {
          const motorIntentRollouts = input.runtime.pendingNeuralSignals({
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
            kinds: ["rollout_result"]
          }).filter((signal) => signal.parent_episode_id === invocationId);
          const realRollout = motorIntentRollouts.find(
            (signal) => parsed.source_signal_ids.includes(signal.signal_id)
          ) ?? motorIntentRollouts[0];
          if (!realRollout) {
            throw new Error("Motor Intent returned without a deterministic rollout signal");
          }
          causality.push(realRollout.signal_id);
          await input.runtime.consumeNeuralSignals(
            HUMANOID_NEURAL_AGENT_IDS.motorIntent,
            [realRollout.signal_id]
          );
        }
        if (inputTool.childKey === "predictive"
          && parsed.signal_kind === "forward_prediction") {
          const predictive = PredictiveOutputSchema.parse(childSpecificOutput);
          if (!predictive.accepted) {
            throw new Error("Predictive Critic did not accept the rollout");
          }
          const realRollout = input.runtime.pendingNeuralSignals({
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.predictive,
            kinds: ["rollout_result"]
          }).find((signal) => descendingSignal?.causal_parent_ids.includes(
            signal.signal_id
          ) && signal.direction === "reentrant"
            && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.rolloutGate);
          if (!realRollout || realRollout.direction !== "reentrant") {
            throw new Error("Predictive acceptance has no deterministic rollout signal");
          }
          causality.push(realRollout.signal_id);
        }
        if (inputTool.childKey === "sensorimotorManager") {
          const forwardedKinds: NeuralSignalKind[] = [
            "skill_proposal",
            "rollout_result",
            "forward_prediction",
            "execution_receipt",
            "skill_completed",
            "skill_failed",
            "prediction_error",
            "escalation"
          ];
          const forwarded = input.runtime.pendingNeuralSignals({
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            kinds: forwardedKinds
          }).filter((signal) => (
            (signal.parent_episode_id === invocationId
              || parsed.source_signal_ids.includes(signal.signal_id))
              && isCurrentNeuralSignal(input.runtime, signal)
          ));
          for (const signal of forwarded) {
            if (signal.kind === parsed.signal_kind
              && JSON.stringify(signal.payload) === JSON.stringify(parsed.payload)) {
              causality.push(signal.signal_id);
            }
          }
          if (parsed.signal_kind === "rollout_result") {
            const forwardedRollout = forwarded.find((signal) => (
              signal.kind === "rollout_result"
                && parsed.source_signal_ids.includes(signal.signal_id)
            ));
            if (!forwardedRollout) {
              throw new Error("Sensorimotor rollout output omitted its exact child rollout signal");
            }
            const hierarchyState = input.runtime.neuralHierarchyState();
            const activeCertificate = Object.values(
              hierarchyState.rollout_certificates
            ).find((certificate) => {
              if (certificate.status !== "active"
                || certificate.commitment_id
                  !== hierarchyState.active_skill_commitment?.commitment_id
                || !parsed.source_signal_ids.includes(
                  certificate.predictive_signal_id
                )) {
                return false;
              }
              const certifiedRawRollout = hierarchyState.signals[
                certificate.rollout_signal_id
              ];
              return certifiedRawRollout?.kind === "rollout_result"
                && certifiedRawRollout.source_node_id
                  === HUMANOID_NEURAL_AGENT_IDS.rolloutGate
                && modelPayloadSha256(certifiedRawRollout.payload)
                  === certificate.rollout_payload_sha256
                && neuralSignalHasAncestorId(
                  hierarchyState,
                  forwardedRollout,
                  certifiedRawRollout.signal_id
                );
            });
            const acceptedPrediction = forwarded.find((signal) => (
              signal.kind === "forward_prediction"
                && signal.signal_id === activeCertificate?.predictive_signal_id
            ));
            if (!activeCertificate || !acceptedPrediction) {
              throw new Error(
                "Sensorimotor cannot return rollout_result before Predictive acceptance"
              );
            }
            causality.push(acceptedPrediction.signal_id);
          }
        }
        if (inputTool.childKey === "actionSelection"
          && parsed.signal_kind === "skill_commitment") {
          const commitment = input.runtime.neuralHierarchyState()
            .active_skill_commitment;
          if (!commitment
            || !["committed", "executing", "released"].includes(commitment.state)) {
            throw new Error(
              "Action Selection cannot return skill_commitment without a durable commitment"
            );
          }
          if (commitment?.state === "executing") {
            const acceptedRollout = input.runtime.pendingNeuralSignals({
              targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
              kinds: ["rollout_result"]
            }).find((signal) => parsed.source_signal_ids.includes(signal.signal_id));
            if (!acceptedRollout) {
              throw new Error(
                "Action Selection executing output omitted accepted rollout feedback"
              );
            }
            causality.push(acceptedRollout.signal_id);
          }
        }
        if (inputTool.childKey === "actionSelection"
          && parsed.signal_kind === "perceptual_belief") {
          const hierarchyState = input.runtime.neuralHierarchyState();
          const postExecutionBelief = input.runtime.pendingNeuralSignals({
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
            kinds: ["perceptual_belief"]
          }).find((signal) => parsed.source_signal_ids.includes(signal.signal_id));
          const epochHasSkillCompletion = Object.values(
            hierarchyState.signals
          ).some((signal) => signal.kind === "skill_completed");
          const boundToLiveCompletion = postExecutionBelief !== undefined
            && epochHasSkillCompletion
            && neuralSignalHasAncestorKind(
              hierarchyState,
              postExecutionBelief,
              "skill_completed"
            );
          const completionReadiness = input.runtime.cycleCompletionReadiness();
          const boundToDurableFreshEpochExecution = postExecutionBelief !== undefined
            && !epochHasSkillCompletion
            && completionReadiness.status === "ready"
            && completionReadiness.observed_after_execution
            && input.runtime.autonomyReadiness() === "complete_cycle"
            && postExecutionBelief.world_revision >= input.runtime.validateCycleEvidence(
              completionReadiness.evidence_transaction_ids
            ).worldAfterRevision
            && neuralSignalHasAncestorKind(
              hierarchyState,
              postExecutionBelief,
              "sensory_evidence"
            );
          if (!postExecutionBelief
            || (!boundToLiveCompletion && !boundToDurableFreshEpochExecution)) {
            throw new Error(
              "Action Selection post-execution output omitted its causally bound Perception result"
            );
          }
          causality.push(postExecutionBelief.signal_id);
        }
        const ascendingSignal = await input.runtime.publishNeuralSignal({
          kind: parsed.signal_kind,
          pathway: childDescriptor.pathway,
          direction: "ascending",
          sourceNodeId: childId,
          targetNodeId: parentId,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 50,
          causalParentIds: [...new Set(causality)],
          sourceAuthorityLeaseId: authorityLease.lease_id,
          invocationId,
          parentInvocationId: authorityLease.parent_invocation_id,
          payload: routedPayload
        });
        if (inputTool.childKey === "predictive"
          && parsed.signal_kind === "forward_prediction") {
          const rolloutSignals = input.runtime.pendingNeuralSignals({
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.predictive,
            kinds: ["rollout_result"]
          });
          const rollout = rolloutSignals.find(
            (signal) => signal.direction === "reentrant"
              && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.rolloutGate
              && ascendingSignal.causal_parent_ids.includes(signal.signal_id)
          );
          if (!rollout) {
            throw new Error("Predictive acceptance has no real rollout input");
          }
          if (!ascendingSignal.causal_parent_ids.includes(rollout.signal_id)) {
            throw new Error("Predictive acceptance omitted its rollout causal parent");
          }
          const receipt = planningReceiptFromRollout(rollout.payload);
          const commitment = input.runtime.neuralHierarchyState()
            .active_skill_commitment;
          if (!commitment || commitment.state !== "committed") {
            throw new Error("Predictive acceptance has no committed Skill");
          }
          await input.runtime.issueNeuralRolloutCertificate({
            issuedByNodeId: childId,
            commitmentId: commitment.commitment_id,
            goalEpochId: commitment.goal_epoch_id,
            planningTransactionId: receipt.transactionId,
            planningAction: receipt.action,
            rolloutSignalId: rollout.signal_id,
            predictiveSignalId: ascendingSignal.signal_id,
            rolloutPayloadSha256: modelPayloadSha256(rollout.payload),
            rolloutInvocationId: rollout.invocation_id,
            predictiveInvocationId: ascendingSignal.invocation_id,
            ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS
          });
        }
        await advanceHarnessPhaseAfterChild(
          input.runtime,
          inputTool.childKey,
          parsed.signal_kind
        );
        const stillPending = new Set(input.runtime.pendingNeuralSignals({
          targetNodeId: childId,
          invocationId
        }).map((signal) => signal.signal_id));
        const preservedInputKinds: ReadonlySet<NeuralSignalKind> = new Set(
          inputTool.childKey === "predictive"
            && parsed.signal_kind === "forward_prediction"
            ? ["rollout_result"]
            : []
        );
        const consumed = new Set(invocationInputSignalIds.filter((signalId) => {
          if (!stillPending.has(signalId)) return false;
          const signal = input.runtime.neuralHierarchyState().signals[signalId];
          return !signal || !preservedInputKinds.has(signal.kind);
        }));
        for (const signal of joinedManagerSignals(
          input.runtime,
          inputTool.childKey,
          parsed.signal_kind,
          invocationId
        )) {
          consumed.add(signal.signal_id);
        }
        if (consumed.size > 0) {
          await input.runtime.consumeNeuralSignals(childId, [...consumed]);
        }
        return JSON.stringify({
          ...parsed,
          payload: routedPayload,
          // Rebind the child result to the one direct edge the parent owns.
          // The child's internal source ids remain durable causal parents in
          // Harness state; exposing both sets made compatible models select
          // the wrong authority namespace on the next call.
          source_signal_ids: [ascendingSignal.signal_id]
        });
      } catch (error) {
        // Continuation state is valid only while this local Agent.asTool loop
        // still owns the caller. If the replacement SDK turn fails, no future
        // invocation can legally reuse that edge, so let `finally` close it.
        continuedChildEpisode = undefined;
        throw error;
      } finally {
        const retainedForContinuation = authorityLease !== undefined
          && continuedChildEpisode?.authorityLeaseId === authorityLease.lease_id;
        if (authorityLease && !retainedForContinuation) {
          await input.runtime.closeNeuralAuthorityLease({
            leaseId: authorityLease.lease_id,
            closedByNodeId: parentId,
            reason: "parent_child_invocation_returned"
          });
        }
        authorityLease = undefined;
        descendingSignal = undefined;
        invocationInputSignalIds = [];
        scopedInvocationId = undefined;
        scopedSignal = undefined;
      }
    });
    // Agent.asTool delegation is an episode-stable capability surface.  The
    // SDK freezes the visible tool set for a run turn sequence, so changing
    // isEnabled after a child returns makes a later/recovered model call see
    // "Tool not found" even though the Harness explicitly requests that edge.
    // The invocation wrapper above remains the single dynamic phase/lease gate.
    return childTool;
  };

  const goalManager = register(
    "goalManager",
    GoalValuationOutputSchema,
    createGoalManagerTools(input.runtime),
    {
      toolUseBehavior: goalValuationToolUseBehavior(),
      extraInstructions: [
        "Use the existing Goal DAG tools for every formal Goal mutation.",
        "When no Goal is active, use the current world-bounded autonomy frontier and outcome history to submit 1–3 distinct observable Goal candidates, then use a separate model response to select exactly one candidate sequence. An exact actionable mission Goal may be the only candidate when no physically grounded alternative exists; never invent filler Goals. Do not wait for an operator to choose.",
        "In continuous mode, continuous_drive_state.drive_phase other than open_ended means the exact mission_goal remains the required long-horizon Goal; include it unchanged in the candidate slate and select it. Route discovery and local waypoints belong to bounded navigation Skills, never replacement Goals. Only after drive_phase=open_ended may novelty and useful interaction outrank repeating the bootstrap mission.",
        "Treat observable_goal_surface.predicate_types as the exhaustive controller-admitted Goal predicate whitelist for this invocation. Empty manipulation, articulation, or block-removal surfaces mean those Goals are physically unavailable; never propose them from object visibility alone.",
        "After continuous_drive_state.drive_phase becomes open_ended, every successor Goal must require a real change from the current MuJoCo state. Use the current robot root, zone geometry, and reachable exploration frontiers; never select a position or zone the robot already satisfies.",
        "A Goal is a durable desired physical state, not one motor step. Keep the exact mission Goal active across perception, navigation, manipulation, replanning, and recovery; do not replace it with nearby exploration waypoints merely because a frontier has higher information gain.",
        "In mission mode, value candidates by causal progress toward the mission Goal. In either mode, Goal valuation chooses only observable predicates; it never chooses a Skill, hand, interaction point, route, posture, or controller command.",
        "When a recovery escalation returns, continue the active Goal only if current physical evidence still shows a viable unfinished predicate; otherwise retire it with exact evidence and autonomously select a successor.",
        "After a terminal Goal tool result, return goal_selected or escalation as structured output."
      ]
    }
  );
  const sceneInterpreter = register(
    "sceneInterpreter",
    SceneInterpretationOutputSchema,
    [],
    {
      toolChoice: "auto",
      extraInstructions: [
        "Return scene_interpretation only from current sensory evidence.",
        "Describe geometry, topology, object state, visibility, and uncertainty. Never choose or recommend an action, waypoint, route side, Skill, hand, interaction point, motor program, or plan; those decisions belong to Sensorimotor and Motor Intent.",
        "Report sensed frontiers and their measured scores as observations, never as an imperative next step."
      ]
    }
  );
  const memoryRetriever = register(
    "memoryRetriever",
    MemoryRetrievalOutputSchema,
    [relevantMemoryRecallTool(input.runtime)],
    {
      toolChoice: "auto",
      toolUseBehavior: relevantMemoryToolUseBehavior(),
      extraInstructions: [
        "Choose one high-level retrieval intent. The Harness constructs the legal storage query and the typed memory_retrieval result ends this episode.",
        "Use active_goal for normal perception and post-execution feedback. Use recent only when the parent explicitly requests a chronological view.",
        "Historical records never replace current sensing. An empty recall result is a valid final retrieval result."
      ]
    }
  );
  const perceptionSensorFusionTool = sensorFusionTool(input.runtime);
  const perceptionSceneTool = asChildTool({
    parentKey: "perceptionManager",
    childKey: "sceneInterpreter",
    child: sceneInterpreter,
    description: "Interpret the current authoritative sensory signal.",
    requiredSignalKind: "sensory_evidence",
    sourceSignalContract: CURRENT_SENSORY_EVIDENCE_SOURCE_CONTRACT,
    phases: ["perception", "feedback"],
    isEnabled: () => hasCurrentManagerEpisodeSignal(
      input.runtime,
      HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      "sensory_evidence"
    ) && !hasCurrentManagerEpisodeSignal(
      input.runtime,
      HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      "scene_interpretation"
    )
  });
  const perceptionMemoryTool = asChildTool({
    parentKey: "perceptionManager",
    childKey: "memoryRetriever",
    child: memoryRetriever,
    description: "Retrieve only relevant historical embodied experience.",
    requiredSignalKind: "sensory_evidence",
    sourceSignalContract: CURRENT_SENSORY_EVIDENCE_SOURCE_CONTRACT,
    phases: ["perception", "feedback"],
    isEnabled: () => hasCurrentManagerEpisodeSignal(
      input.runtime,
      HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      "sensory_evidence"
    ) && !hasCurrentManagerEpisodeSignal(
      input.runtime,
      HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      "memory_retrieval"
    )
  });
  // Capture the structural invokers before register() wraps their public
  // capability admission. This path is called only from the owning
  // Perception Agent.asTool input builder, where the parent invocation and
  // lease already exist; it never exposes a second root or sibling channel.
  const invokePerceptionSensorFusion = perceptionSensorFusionTool.invoke;
  const invokePerceptionScene = perceptionSceneTool.invoke;
  const invokePerceptionMemory = perceptionMemoryTool.invoke;
  const preparePerceptionModelInput = async (prepared: {
    invocationId: string;
    signal?: AbortSignal;
  }): Promise<void> => {
    prepared.signal?.throwIfAborted();
    const currentChildSignals = (
      kinds: readonly NeuralSignalKind[]
    ): NeuralSignal[] => input.runtime.pendingNeuralSignals({
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      kinds
    }).filter((signal) => signal.parent_episode_id === prepared.invocationId
      && isCurrentNeuralSignal(input.runtime, signal));

    let sensorySignals = currentChildSignals(["sensory_evidence"]).filter(
      (signal) => signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorFusion
    );
    if (sensorySignals.length === 0) {
      const rawInput = JSON.stringify({});
      await invokePerceptionSensorFusion(
        new RunContext(),
        rawInput,
        {
          toolCall: {
            type: "function_call",
            callId: stableAgentToolInvocationId(
              HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
              `perception-input:${prepared.invocationId}`
            ),
            name: perceptionSensorFusionTool.name,
            arguments: rawInput
          },
          ...(prepared.signal ? { signal: prepared.signal } : {})
        }
      );
      sensorySignals = currentChildSignals(["sensory_evidence"]).filter(
        (signal) => signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorFusion
      );
    }
    if (sensorySignals.length !== 1) {
      throw new Error(
        `Perception input requires one current Sensor Fusion signal; found ${sensorySignals.length}`
      );
    }
    const sensorySignal = sensorySignals[0]!;
    const delegationInput = JSON.stringify({
      signal_kind: "sensory_evidence",
      source_signal_ids: [sensorySignal.signal_id],
      ttl_revisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
      priority: 80
    });
    const branches: Promise<unknown>[] = [];
    const sceneSignals = currentChildSignals(["scene_interpretation"]).filter(
      (signal) => signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter
    );
    if (sceneSignals.length === 0) {
      branches.push(invokePerceptionScene(
        new RunContext(),
        delegationInput,
        {
          toolCall: {
            type: "function_call",
            callId: stableAgentToolInvocationId(
              HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter,
              `perception-input:${prepared.invocationId}`
            ),
            name: perceptionSceneTool.name,
            arguments: delegationInput
          },
          ...(prepared.signal ? { signal: prepared.signal } : {})
        }
      ));
    }
    const memorySignals = currentChildSignals(["memory_retrieval"]).filter(
      (signal) => signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.memoryRetriever
    );
    if (memorySignals.length === 0) {
      branches.push(invokePerceptionMemory(
        new RunContext(),
        delegationInput,
        {
          toolCall: {
            type: "function_call",
            callId: stableAgentToolInvocationId(
              HUMANOID_NEURAL_AGENT_IDS.memoryRetriever,
              `perception-input:${prepared.invocationId}`
            ),
            name: perceptionMemoryTool.name,
            arguments: delegationInput
          },
          ...(prepared.signal ? { signal: prepared.signal } : {})
        }
      ));
    }
    const settled = await Promise.allSettled(branches);
    const failures = settled.flatMap((result) => (
      result.status === "rejected" ? [result.reason] : []
    ));
    if (failures.length > 0) {
      throw new AggregateError(
        failures,
        "Perception Scene/Memory fan-out failed"
      );
    }
    prepared.signal?.throwIfAborted();
    const completedScene = currentChildSignals(["scene_interpretation"]).filter(
      (signal) => signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter
        && neuralSignalHasAncestorId(
          input.runtime.neuralHierarchyState(),
          signal,
          sensorySignal.signal_id
        )
    );
    const completedMemory = currentChildSignals(["memory_retrieval"]).filter(
      (signal) => signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.memoryRetriever
        && neuralSignalHasAncestorId(
          input.runtime.neuralHierarchyState(),
          signal,
          sensorySignal.signal_id
        )
    );
    if (completedScene.length !== 1 || completedMemory.length !== 1) {
      throw new Error(
        "Perception input fan-out did not produce one Scene and one Memory return"
      );
    }
  };
  const perceptionManager = register(
    "perceptionManager",
    PerceptionOutputSchema,
    [
      perceptionSensorFusionTool,
      perceptionSceneTool,
      perceptionMemoryTool
    ],
    {
      parallel: true,
      extraInstructions: [
        "The Harness has already captured Sensor Fusion and completed the mandatory parallel Scene/Memory fan-out before your first reasoning turn. Do not call those structural tools again in this episode.",
        "Return perceptual_belief only after Sensor Fusion, Scene, and Memory have all returned. The Harness owns their fork/join provenance and binds the three formal source ids to your aggregate; concentrate on the semantic belief rather than protocol UUID bookkeeping.",
        "Perception owns state estimation, not action selection. Submit only compact_perceptual_belief_v1: bounded observations, uncertainty, changed entity ids, and safety facts. Never copy raw Sensor Fusion, Scene, Memory, or reachable-base arrays into payload; the Harness attaches those exact child signals after validating your compact belief.",
        "Never recommend or select a Skill, hand, interaction point, motor program, or next action. Action Selection and Sensorimotor own those decisions.",
        "Never send one sibling's raw output or Session to the other."
      ]
    }
  );
  const affordance = register(
    "affordance",
    AffordanceOutputSchema,
    [],
    { toolChoice: "auto", extraInstructions: [
      "Return affordance_hypothesis for the committed Goal only.",
      "Evaluate live reachable_base_placements as exact atomic tuples: interaction_point_id, hand_surface, root_world_target, and root_yaw_radians must stay together. Never combine a hand or interaction point from history with the geometry of another live tuple.",
      "Historical failures are evidence about the failed tuple, not authority to overwrite a different current geometry candidate."
    ] }
  );
  const risk = register(
    "risk",
    RiskOutputSchema,
    [],
    { toolChoice: "auto", extraInstructions: [
      "During skill_proposal there is no committed Skill yet. Assess current balance, contact, collision, and environmental bounds without inventing or selecting a Skill, hand, interaction point, or motor program.",
      "State hazards, uncertainty, and admissible safety bounds only. Never recommend an action, waypoint, frontier, route side, Skill, or plan; Action Selection and Sensorimotor own those choices.",
      "Return risk_assessment when at least one lower-level option may proceed under stated bounds.",
      "Return escalation when risk requires inhibition or Recovery; never invent an alternate motor command."
    ] }
  );
  const predictive = register(
    "predictive",
    PredictiveOutputSchema,
    [],
    { toolChoice: "auto", extraInstructions: [
      "Run only after a real rollout_result.",
      "Judge admission of the exact bounded execution chunk represented by this rollout, not completion of the longer-horizon Goal.",
      "For a chunked plan, forward_prediction with accepted=true means this chunk is aligned with the active Skill, safely feasible, and makes bounded progress toward its termination contract.",
      "A positive remaining_distance or an unsatisfied final robot_in_zone predicate is expected before intermediate navigation chunks execute and is not by itself a prediction error.",
      "Evaluate an obstacle against the current chunk and validated route, not an imagined straight-line future path beyond this rollout.",
      "Return prediction_error or escalation with accepted=false only when this exact chunk violates alignment, progress, feasibility, rollout acceptance, or safety bounds."
    ] }
  );
  const motorIntent = register(
    "motorIntent",
    MotorIntentOutputSchema,
    motorIntentPlanningTools(input.runtime),
    {
      toolUseBehavior: planningReceiptToolUseBehavior(input.runtime),
      extraInstructions: [
        "Compile the exact Premotor-authored motor_program in your direct descending signal. When planning_tool_state requires submit_humanoid_skill_plan, copy motor_program.skill_plan verbatim; the Harness rejects any changed objective, strategy, node, dependency, Skill, or parameter. You do not redesign the Skill DAG.",
        "Complete the remaining embodied Skill lifecycle in this one SDK episode: when planning_tool_state requires begin_humanoid_skill, copy one ready_skill_binding verbatim; only then call the enabled semantic planning tool with the real bound skill_transaction_id.",
        "submit_humanoid_skill_plan and begin_humanoid_skill are lifecycle transitions, not rollout results. Continue after each accepted transition and inspect the next planning_tool_state exposed by the Harness.",
        "Compile object-relative preparation with object-relative Skills. approach(object_id=...) is the navigation Skill that moves the base to a manipulation stance chosen from live reachable_base_placements; navigate_to_zone moves only the robot root into a semantic zone and cannot prepare an uncarried object for grasping or placement.",
        "For an object_placed termination contract, preserve one causal object chain in the Skill DAG: an uncarried object needs an object-targeted approach/reach/grasp/lift path before carry_to_zone/place. You choose the hand, interaction point, standoff, and exact bounded nodes from current geometry; never substitute the destination zone for the source object in the first ready node.",
        "If Skill-plan admission rejects a ready node, change the contradictory invocation before retrying. Repeating the same rejected Skill and parameters is not recovery.",
        "A rejected physical plan is not terminal while planning_tool_state exposes an untried same-commitment recovery modality. In particular, when transit_clearance.status=required, use the supplied collision geometry, fixed feet, current wrist, and skill_transaction_id to try a materially different whole-body clearance posture and, if needed, alternate navigation. Do not repeat a recovery modality already rejected in this Motor Intent episode: after both modalities produce deterministic rejection signals, the Harness ends this bounded local episode with typed escalation to Premotor. For a whole-body clearance candidate, copy both required_candidate_contract.every_keyframe_channels entries into every keyframe, choose a future collision-side wrist world position with the required displacement, and copy that exact position into the matching terminal predicate; describing fixed feet in intent text does not satisfy the physical contract.",
        "The planning tool already performs the deterministic MuJoCo rollout gate.",
        "Never emit joint trajectories, controller commands, or physical writes."
      ]
    }
  );
  const premotor = register(
    "premotor",
    PremotorOutputSchema,
    [asChildTool({
      parentKey: "premotor",
      childKey: "motorIntent",
      child: motorIntent,
      description: "Compile one bounded committed skill into an existing planning call.",
      requiredSignalKind: "skill_commitment",
      phases: ["motor_planning"],
      requireCommitment: true
    })],
    {
      toolUseBehavior: directChildNeuralOutcomeToolUseBehavior(
        humanoidNeuralAgentToolName("motorIntent"),
        PremotorOutputSchema
      ),
      extraInstructions: [
        "Run only for an Action Selection-authorized commitment.",
        "Compose one short Skill DAG that begins with the exact committed bounded Skill and preserves its invocation parameters. Call Motor Intent with motor_program={protocol:'premotor_motor_program_v1', skill_plan:{objective, strategies, selected_strategy_id}}. This structured handoff is your semantic output; reasoning text is not transferred.",
        "Keep the selected strategy locally executable. Dependencies may express real near-term prerequisites, but never replace the committed first ready node with another Skill or expand into a long-horizon mission plan.",
        "The Motor Intent tool's typed rollout_result or escalation directly completes this episode; do not resubmit or relabel it."
      ]
    }
  );
  const recovery = register(
    "recovery",
    RecoveryOutputSchema,
    [],
    {
      toolChoice: "auto",
      toolUseBehavior: actionSelectionToolUseBehavior(input.runtime),
      extraInstructions: [
        "This is an independent bounded recovery episode under a Harness authority lease.",
        "The current failure receipt and post-failure perceptual belief are already in your directed input. Historical evidence may arrive only through the Perception Manager -> Memory Retriever branch materialized in that belief; never query or assume a shared memory store.",
        "Read durable_failure_evidence.failure_receipt as the exact failed physical/planning attempt. Do not return the same bounded invocation again while that failure remains unresolved; choose a materially different prerequisite/target/Skill or escalate.",
        "For an unresolved acknowledged stationary_fall, the only admissible local proposal is one bounded stabilize Skill beginning at recover_support, even when a prior partial get-up has already cleared the coarse fallen flag. Do not propose navigation, manipulation, or task continuation until whole-body recovery has stood the robot up and handed control back to the body controller. Escalate only when the live catalog or controller capability evidence makes stabilize recovery unavailable.",
        "Return a recovery skill_proposal or escalation. You never write physical state."
      ]
    }
  );
  const executionDispatcher = register(
    "executionDispatcher",
    SensorimotorOutputSchema,
    [serialExecutionTool(input.runtime)],
    {
      toolChoice: "required",
      thinking: "disabled",
      toolUseBehavior: certifiedExecutionDispatcherToolUseBehavior(input.runtime),
      extraInstructions: [
        "This is a pure execution node. The Harness has already fixed the commitment, rollout certificate, and serial writer.",
        "Call execute_certified_motor_intent exactly once with a short execution objective. The Harness binds the unique direct skill_commitment; do not copy signal UUIDs.",
        "Do not plan, compare options, reason about alternatives, submit a separate neural output, or return prose."
      ]
    }
  );
  const executionDispatcherTool = asChildTool({
    parentKey: "sensorimotorManager",
    childKey: "executionDispatcher",
    child: executionDispatcher,
    description: "Dispatch the already-certified commitment through the serial physical writer.",
    requiredSignalKind: "skill_commitment",
    phases: ["execution"],
    requireCommitment: true,
    isEnabled: () => input.runtime.neuralExecutionAvailable()
      && input.runtime.neuralHarnessPhase().phase === "execution"
      && input.runtime.neuralHierarchyState().active_skill_commitment?.state
        === "executing"
  });
  const sensorimotorPremotorTool = asChildTool({
    parentKey: "sensorimotorManager",
    childKey: "premotor",
    child: premotor,
    description: "Compose and compile one bounded motor skill after assessment.",
    phases: ["motor_assessment", "motor_planning"],
    requireCommitment: true,
    isEnabled: () => !hasCurrentRecoveryDemand(input.runtime)
      && input.runtime.neuralHierarchyState().active_skill_commitment?.state
        === "committed"
      && hasCurrentManagerEpisodeSignal(
        input.runtime,
        HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        "skill_commitment"
      )
      && !hasCurrentManagerEpisodeSignalsAny(
        input.runtime,
        HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        ["rollout_result", "escalation"]
      )
  });
  const sensorimotorAffordanceTool = asChildTool({
    parentKey: "sensorimotorManager",
    childKey: "affordance",
    child: affordance,
    description: "Assess current affordances read-only.",
    phases: ["skill_proposal", "motor_assessment"],
    isEnabled: () => !hasCurrentRecoveryDemand(input.runtime)
      && !neuralSkillCommitmentIsOpen(
        input.runtime.neuralHierarchyState().active_skill_commitment
      )
      && hasCurrentManagerEpisodeSignal(
        input.runtime,
        HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        "perceptual_belief"
      )
      && !hasCurrentManagerEpisodeSignal(
        input.runtime,
        HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        "affordance_hypothesis"
      )
  });
  const sensorimotorRiskTool = asChildTool({
    parentKey: "sensorimotorManager",
    childKey: "risk",
    child: risk,
    description: "Assess balance, contact, collision, and commitment risk read-only.",
    phases: ["skill_proposal", "motor_assessment"],
    isEnabled: () => !hasCurrentRecoveryDemand(input.runtime)
      && !neuralSkillCommitmentIsOpen(
        input.runtime.neuralHierarchyState().active_skill_commitment
      )
      && hasCurrentManagerEpisodeSignal(
        input.runtime,
        HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        "perceptual_belief"
      )
      && !hasCurrentManagerEpisodeSignalsAny(
        input.runtime,
        HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        ["risk_assessment", "escalation"]
      )
  });
  const invokeSensorimotorAffordance = sensorimotorAffordanceTool.invoke;
  const invokeSensorimotorRisk = sensorimotorRiskTool.invoke;
  const invokeSensorimotorPremotor = sensorimotorPremotorTool.invoke;
  const prepareSensorimotorModelInput = async (prepared: {
    invocationId: string;
    authorityLeaseId: string;
    signal?: AbortSignal;
  }): Promise<void> => {
    const phase = input.runtime.neuralHarnessPhase().phase;
    if (hasCurrentRecoveryDemand(input.runtime)) return;
    prepared.signal?.throwIfAborted();
    const currentSignals = (
      kinds: readonly NeuralSignalKind[]
    ): NeuralSignal[] => input.runtime.pendingNeuralSignals({
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      kinds
    }).filter((signal) => (
      (signal.direction === "descending"
        ? signal.invocation_id === prepared.invocationId
        : signal.parent_episode_id === prepared.invocationId)
        && isCurrentNeuralSignal(input.runtime, signal)
    ));
    const retireSupersededControlSubtrees = async (
      roots: readonly NeuralSignal[]
    ): Promise<void> => {
      if (roots.length === 0) return;
      const state = input.runtime.neuralHierarchyState();
      const rootIds = new Set(roots.map((signal) => signal.signal_id));
      const byTarget = new Map<string, string[]>();
      const targetNodeIds = new Set(Object.values(state.signals).map(
        (signal) => signal.target_node_id
      ));
      for (const targetNodeId of targetNodeIds) {
        for (const signal of input.runtime.pendingNeuralSignals({
          targetNodeId
        })) {
          if (!rootIds.has(signal.signal_id)
            && !roots.some((root) => neuralSignalHasAncestorId(
              state,
              signal,
              root.signal_id
            ))) continue;
          const targetSignals = byTarget.get(targetNodeId) ?? [];
          targetSignals.push(signal.signal_id);
          byTarget.set(targetNodeId, targetSignals);
        }
      }
      for (const [targetNodeId, signalIds] of byTarget) {
        await input.runtime.consumeNeuralSignals(targetNodeId, signalIds);
      }
    };
    if ((phase === "motor_assessment" || phase === "motor_planning")
      && input.runtime.neuralHierarchyState().active_skill_commitment?.state
        === "committed") {
      const directCommitments = currentSignals(["skill_commitment"]).filter(
        (signal) => signal.direction === "descending"
          && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
      );
      const supersededCommitmentInputs = directCommitments.filter(
        (signal) => signal.authority_lease_id !== prepared.authorityLeaseId
      );
      await retireSupersededControlSubtrees(supersededCommitmentInputs);
      const commitments = currentSignals(["skill_commitment"]).filter(
        (signal) => signal.direction === "descending"
          && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
          && signal.authority_lease_id === prepared.authorityLeaseId
      );
      if (commitments.length !== 1) {
        throw new Error(
          `Sensorimotor motor input requires one current-authority commitment; found ${commitments.length}`
        );
      }
      const commitment = commitments[0]!;
      const currentPremotorReturns = (): NeuralSignal[] => currentSignals([
        "rollout_result",
        "escalation"
      ]).filter((signal) => signal.source_node_id
          === HUMANOID_NEURAL_AGENT_IDS.premotor
        && neuralSignalHasAncestorId(
          input.runtime.neuralHierarchyState(),
          signal,
          commitment.signal_id
        ));
      if (currentPremotorReturns().length === 0) {
        const delegationInput = JSON.stringify({
          signal_kind: "skill_commitment",
          source_signal_ids: [commitment.signal_id],
          ttl_revisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 90
        });
        await invokeSensorimotorPremotor(
          new RunContext(),
          delegationInput,
          {
            toolCall: {
              type: "function_call",
              callId: stableAgentToolInvocationId(
                HUMANOID_NEURAL_AGENT_IDS.premotor,
                `sensorimotor-input:${prepared.invocationId}`
              ),
              name: sensorimotorPremotorTool.name,
              arguments: delegationInput
            },
            ...(prepared.signal ? { signal: prepared.signal } : {})
          }
        );
      }
      prepared.signal?.throwIfAborted();
      const premotorReturns = currentPremotorReturns();
      if (premotorReturns.length !== 1) {
        throw new Error(
          `Sensorimotor motor input requires one causally bound Premotor return; found ${premotorReturns.length}`
        );
      }
      return;
    }
    if (phase !== "skill_proposal"
      || neuralSkillCommitmentIsOpen(
        input.runtime.neuralHierarchyState().active_skill_commitment
      )) return;
    const directBeliefs = currentSignals(["perceptual_belief"]).filter(
      (signal) => signal.direction === "descending"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
    );
    // A recovered SDK Session deliberately reuses its durable invocation id,
    // but the new process owns a new lease. Retire inputs from the abandoned
    // control edge before forking specialists so one episode cannot join
    // beliefs from two process lifetimes.
    const supersededInputs = directBeliefs.filter(
      (signal) => signal.authority_lease_id !== prepared.authorityLeaseId
    );
    await retireSupersededControlSubtrees(supersededInputs);
    const beliefs = currentSignals(["perceptual_belief"]).filter(
      (signal) => signal.direction === "descending"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
        && signal.authority_lease_id === prepared.authorityLeaseId
        && isSemanticPerceptualBeliefSignal(signal)
    );
    if (beliefs.length !== 1) {
      throw new Error(
        `Sensorimotor input requires one current-authority semantic perceptual belief; found ${beliefs.length}`
      );
    }
    const belief = beliefs[0]!;
    const delegationInput = JSON.stringify({
      signal_kind: "perceptual_belief",
      source_signal_ids: [belief.signal_id],
      ttl_revisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
      priority: 80
    });
    const branches: Promise<unknown>[] = [];
    if (currentSignals(["affordance_hypothesis"]).length === 0) {
      branches.push(invokeSensorimotorAffordance(
        new RunContext(),
        delegationInput,
        {
          toolCall: {
            type: "function_call",
            callId: stableAgentToolInvocationId(
              HUMANOID_NEURAL_AGENT_IDS.affordance,
              `sensorimotor-input:${prepared.invocationId}`
            ),
            name: sensorimotorAffordanceTool.name,
            arguments: delegationInput
          },
          ...(prepared.signal ? { signal: prepared.signal } : {})
        }
      ));
    }
    if (currentSignals(["risk_assessment", "escalation"]).filter(
      (signal) => signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.risk
    ).length === 0) {
      branches.push(invokeSensorimotorRisk(
        new RunContext(),
        delegationInput,
        {
          toolCall: {
            type: "function_call",
            callId: stableAgentToolInvocationId(
              HUMANOID_NEURAL_AGENT_IDS.risk,
              `sensorimotor-input:${prepared.invocationId}`
            ),
            name: sensorimotorRiskTool.name,
            arguments: delegationInput
          },
          ...(prepared.signal ? { signal: prepared.signal } : {})
        }
      ));
    }
    const settled = await Promise.allSettled(branches);
    const failures = settled.flatMap((result) => (
      result.status === "rejected" ? [result.reason] : []
    ));
    if (failures.length > 0) {
      throw new AggregateError(
        failures,
        "Sensorimotor Affordance/Risk fan-out failed"
      );
    }
    prepared.signal?.throwIfAborted();
    const affordances = currentSignals(["affordance_hypothesis"]).filter(
      (signal) => signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.affordance
        && neuralSignalHasAncestorId(
          input.runtime.neuralHierarchyState(),
          signal,
          belief.signal_id
        )
    );
    const risks = currentSignals(["risk_assessment", "escalation"]).filter(
      (signal) => signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.risk
        && neuralSignalHasAncestorId(
          input.runtime.neuralHierarchyState(),
          signal,
          belief.signal_id
        )
    );
    if (affordances.length !== 1 || risks.length !== 1) {
      throw new Error(
        "Sensorimotor input fan-out did not produce one Affordance and one Risk return"
      );
    }
  };
  const sensorimotorManager = register(
    "sensorimotorManager",
    SensorimotorOutputSchema,
    [
      sensorimotorAffordanceTool,
      sensorimotorRiskTool,
      sensorimotorPremotorTool,
      asChildTool({
        parentKey: "sensorimotorManager",
        childKey: "predictive",
        child: predictive,
        description: "Interpret a completed MuJoCo rollout result.",
        requiredSignalKind: "rollout_result",
        phases: ["rollout_review"],
        requireCommitment: true,
        isEnabled: () => !hasCurrentRecoveryDemand(input.runtime)
          && hasCurrentSignal(
            input.runtime,
            HUMANOID_NEURAL_AGENT_IDS.predictive,
            "rollout_result"
          )
          && !hasCurrentSignalsAny(
            input.runtime,
            HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            ["forward_prediction", "prediction_error", "escalation"]
          )
      }),
      recoveryAuthorityTool(input.runtime, recovery, sessions, input),
      executionDispatcherTool
    ],
    {
      parallel: true,
      toolUseBehavior: sensorimotorToolUseBehavior(input.runtime),
      extraInstructions: [
        "In skill_proposal, the Harness has already completed the mandatory parallel Affordance/Risk fan-out before your first reasoning turn. Do not call those structural tools again in that episode.",
        "A skill_proposal must semantically join the current perceptual_belief with both Affordance and Risk results. The Harness binds all three exact episode-local signals; do not copy their UUIDs.",
        "If directed input contains direct_parent_correction, treat it as binding causal feedback from Action Selection. Do not repeat the rejected invocation while its physical preconditions are unchanged; select the structured required prerequisite Skill when present, using the current observation and specialist evidence.",
        "Return exactly one next bounded catalog Skill as proposed_skill={skill, phase, params, rationale}. Future steps may appear only in phase_sequence; never make a compound skill_name or the whole task the proposal.",
        "Every normal proposal must make measurable physical progress toward an active Goal predicate or establish its real prerequisite. For robot_in_zone, prefer navigate_to_zone; a navigate_to_point or explore frontier is admissible only when its actual target reduces distance to that zone. Information gain never authorizes movement away from the active Goal.",
        "For an object placement Goal whose object is not grasped, preparation is object-relative: propose approach(object_id), then reach/grasp/lift, before carry_to_zone/place. Never approach the destination zone to make the source object reachable.",
        "For approach, params are exactly object_id, interaction_point_id, hand, and standoff_m. Preserve the selected live hand+interaction-point pair, but never add root_world_target, root_yaw_radians, or base_placement; the Harness binds that pair to its authoritative IK placement.",
        "After Action Selection accepts the proposal, the Harness completes the mandatory Premotor -> Motor Intent branch before your first committed-branch reasoning turn. Do not call Premotor again in that episode; inspect its exact rollout_result and delegate Predictive. Affordance/Risk belong only to skill_proposal and must not be repeated.",
        "Predictive judges admission of the current bounded rollout chunk, not whether that chunk already completes the final Goal predicate.",
        "Predictive acceptance completes the motor-assessment episode and returns a certified rollout to Action Selection; do not call or synthesize execution in that episode.",
        "In execution, delegate the one direct executing skill_commitment to Certified Execution Dispatcher. The Dispatcher owns the required non-thinking tool call; do not execute or restate it yourself.",
        "Recovery freezes normal selection, runs one independent episode, and returns before execution resumes."
      ]
    }
  );
  const actionSelection = register(
    "actionSelection",
    ActionSelectionOutputSchema,
    [
      acknowledgeSafetyInterruptTool(input.runtime),
      establishSkillCommitmentTool(input.runtime),
      authorizeSkillExecutionTool(input.runtime),
      transitionSkillCommitmentTool(input.runtime, "completed"),
      transitionSkillCommitmentTool(input.runtime, "failed"),
      transitionSkillCommitmentTool(input.runtime, "released"),
      asChildTool({
        parentKey: "actionSelection",
        childKey: "perceptionManager",
        child: perceptionManager,
        description: "Build one current perceptual belief through the owned perception subtree.",
        phases: ["perception", "feedback"],
        prepareModelInput: preparePerceptionModelInput,
        isEnabled: () => input.runtime.neuralHarnessPhase().phase !== "feedback"
          || ["completed", "released"].includes(
            input.runtime.neuralHierarchyState().active_skill_commitment?.state ?? ""
          )
      }),
      asChildTool({
        parentKey: "actionSelection",
        childKey: "sensorimotorManager",
        child: sensorimotorManager,
        description: "Select, rollout, and admit one bounded sensorimotor skill.",
        phases: [
          "skill_proposal",
          "motor_assessment",
          "motor_planning",
          "rollout_review",
          "execution",
          "recovery"
        ],
        requireCommitment: false,
        prepareModelInput: prepareSensorimotorModelInput,
        isEnabled: () => input.runtime.neuralHarnessPhase().phase !== "recovery"
          || input.runtime.neuralHierarchyState().active_skill_commitment === null
          || ["completed", "failed", "released"].includes(
            input.runtime.neuralHierarchyState().active_skill_commitment!.state
          )
      })
    ],
    {
      toolChoice: "auto",
      toolUseBehavior: actionSelectionToolUseBehavior(input.runtime),
      extraInstructions: [
        "In safety_interrupt, acknowledge the exact pending Body→Reflex afferent interrupt first. This is independent plant provenance, not an Agent signal and not a reason to invent source_signal_ids. The Harness atomically fails any interrupted commitment and then requires fresh Perception at the fallen world revision.",
        "Treat neural_hierarchy.harness_phase.phase as binding route authority: in perception or feedback, delegate Perception Manager before any Sensorimotor call; Sensorimotor is not a valid child route until the Harness itself enters skill_proposal, motor_assessment, motor_planning, rollout_review, execution, or recovery.",
        "Perception precedes sensorimotor selection whenever the belief is absent or stale.",
        "Sensorimotor may return only a skill_proposal before authorization.",
        "A skill_proposal must select exactly one next bounded catalog Skill in proposed_skill.skill. A multi-step skill_sequence may describe future causal order, but skill_name, skill_id, or the whole sequence is never the proposed Skill commitment.",
        "You alone establish one durable skill commitment from the newest direct Sensorimotor proposal. The Harness binds that unique proposal; do not copy child UUIDs. Then invoke Sensorimotor again with the resulting skill_commitment signal.",
        "A replacement Sensorimotor proposal supersedes every earlier proposal in this Action Selection episode. Establish only the newest direct proposal and its exact bounded Skill; never restore a rejected proposal because it matched the previous commitment or intention.",
        "Authorize execution only after the committed branch returns a real accepted rollout_result. Call authorize_skill_execution with the active commitment id and your reason; the Harness binds the unique direct certified rollout, so do not copy a rollout or Predictive signal id into that call.",
        "After authorize_skill_execution returns skill_executing, your bounded decision episode ends. The Harness deterministically routes the unique certified commitment through Sensorimotor to Executor without another model decision; your next wake is typed physical feedback, never a request to wait or redispatch execution.",
        "A Premotor, planning, or Predictive escalation is a local recovery demand, not a Recovery escalation. First call release_skill_commitment with the direct Sensorimotor failure signal. If the Harness enters perception, delegate Perception and use that newer belief with the failure; otherwise delegate the same direct failure back to Sensorimotor so its exclusive Recovery child can propose a replacement. Only an escalation returned by Recovery may propagate to Executive.",
        "A successful physical chunk is not automatically Skill completion. complete_skill_commitment is enabled only when the exact committed Skill binding has a successful physical receipt and its authoritative Skill-plan node postcondition is complete. Model-authored prose in the termination contract cannot override that lifecycle. Otherwise release the exhausted bounded plan, obtain fresh Perception, and continue the same Goal through a new bounded Skill commitment.",
        "In recovery, forward the direct failure signal to Sensorimotor. When the failure came from physical execution, also cite the new post-failure perceptual_belief returned after you closed the old commitment. If Recovery returns a replacement skill_proposal, you alone replace the failed commitment; if it escalates, return that typed escalation unchanged to Executive.",
        "In feedback, complete or fail the active commitment before delegating Perception. After completed execution, forward perceptual_belief to Executive. After failed execution, use the new belief to enter Recovery while preserving the active Goal.",
        "Children may not replace the active Goal or establish their own commitment."
      ]
    }
  );
  const actionSelectionTool = asChildTool({
    parentKey: "executive",
    childKey: "actionSelection",
    child: actionSelection,
    description: "Run one bounded hierarchical action-selection episode.",
    phases: [
      "perception",
      "skill_proposal",
      "commitment_authorization",
      "motor_assessment",
      "motor_planning",
      "rollout_review",
      "execution",
      "feedback",
      "safety_interrupt",
      "recovery",
      "cycle_completion"
    ]
  });
  const cycleCompletionTool = neuralCycleCompletionTool(input.runtime);
  const satisfiedGoalCompletionTool = neuralSatisfiedGoalCompletionTool(
    input.runtime
  );
  const goalManagerTool = asChildTool({
    parentKey: "executive",
    childKey: "goalManager",
    child: goalManager,
    description: "Value, select, continue, or retire the current Goal.",
    phases: ["bootstrapping", "goal_valuation", "cycle_completion"]
  });
  const executive = register(
    "executive",
    ExecutiveOutputSchema,
    [
      cycleCompletionTool,
      satisfiedGoalCompletionTool,
      goalManagerTool,
      actionSelectionTool
    ],
    {
      toolUseBehavior: executiveToolUseBehavior(),
      extraInstructions: [
        "You are the only root. Goal Valuation and Action Selection are children, never peers.",
        "Do not wake lower layers unless an event requires work; never poll all Agents.",
        "A typed Action Selection return ends this Executive episode. A newly selected Goal stays in the same local Executive episode: immediately delegate Action Selection after the Harness enters perception. Never relabel or resubmit a child result.",
        "A supervisory prediction_error carrying causal_interrupt_ids is a Body→Reflex afferent safety interrupt. Delegate Action Selection with prediction_error and no invented source signal so it can acknowledge the exact durable interrupt before Perception and Recovery.",
        "In recovery, preserve the active Goal and delegate the direct failure signal through Action Selection; after physical failure, Action Selection must first close the old commitment and obtain current Perception before Recovery is reachable through Sensorimotor.",
        "After execution, delegate Action Selection once. That one child episode must resolve the commitment and obtain current Perception before returning; on failure it continues from the fresh belief into Recovery instead of returning to Executive or using stale pre-action state.",
        "After post-execution Sensor Fusion, decide whether to complete from the direct perceptual_belief. The Harness binds the unique current belief and resolves physical transaction ids internally; do not copy UUIDs or self-certify success."
      ]
    }
  );

  const services = createRuntimeServices();
  CERTIFIED_EXECUTION_DISPATCHER_TOOLS.set(
    input.runtime,
    {
      tool: executionDispatcherTool,
      parent: sensorimotorManager
    }
  );
  DIRECT_ACTION_SELECTION_TOOLS.set(input.runtime, {
    tool: actionSelectionTool,
    parent: executive
  });
  DETERMINISTIC_CYCLE_TOOLS.set(input.runtime, {
    completion: cycleCompletionTool,
    satisfied: satisfiedGoalCompletionTool,
    parent: executive
  });
  DIRECT_GOAL_MANAGER_TOOLS.set(input.runtime, {
    tool: goalManagerTool,
    parent: executive
  });
  return {
    root: executive,
    agents,
    services,
    session: (agentId) => sessions.get(agentId),
    agent: (agentId) => agents.get(agentId),
    outputSchema: (agentId) => outputSchemas.get(agentId)
  };
}

function relevantMemoryRecallTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof RelevantMemoryIntentSchema, string> {
  const name = "retrieve_relevant_embodied_memory";
  return tool({
    name,
    description: "Retrieve one bounded historical memory view. Express only the retrieval intent; the Harness derives exact legal filters from the active Goal.",
    parameters: RelevantMemoryIntentSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    execute: async (intent) => {
      const invocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS.memoryRetriever
      );
      const sourceSignalIds = runtime.pendingNeuralSignals({
        targetNodeId: HUMANOID_NEURAL_AGENT_IDS.memoryRetriever,
        invocationId: invocation.invocationId
      }).map((signal) => signal.signal_id);
      const request = relevantMemoryRecallRequest(runtime, intent);
      const recalled = await runtime.recallEmbodiedHistory(request);
      const payload = relevantMemoryProjection(recalled, intent, request);
      const counts = memoryProjectionCounts(payload);
      return JSON.stringify(MemoryRetrievalOutputSchema.parse({
        signal_kind: "memory_retrieval",
        summary: `Harness retrieved ${counts.experiences} relevant experiences and ${counts.actions} historical actions`,
        payload_json: JSON.stringify(payload),
        source_signal_ids: sourceSignalIds,
        confidence: 1
      }));
    }
  });
}

function relevantMemoryRecallRequest(
  runtime: HumanoidNeuralAgentRuntime,
  intent: z.infer<typeof RelevantMemoryIntentSchema>
): HumanoidEmbodiedRecallRequest {
  const limit = 16;
  if (intent.retrieval_mode === "recent") return { limit };
  const anchor = jsonRecord(runtime.contextAnchor(
    HUMANOID_NEURAL_AGENT_IDS.memoryRetriever
  ));
  const activeGoal = jsonRecord(anchor?.active_goal);
  const predicates = Array.isArray(activeGoal?.predicates)
    ? activeGoal.predicates.map(jsonRecord).filter(
        (predicate): predicate is Record<string, JsonValue> => predicate !== null
      )
    : [];
  const predicateTypes = uniqueStrings(predicates.map(
    (predicate) => typeof predicate.type === "string" ? predicate.type : undefined
  )).filter((value): value is typeof HUMANOID_GOAL_PREDICATE_TYPES[number] => (
    HUMANOID_GOAL_PREDICATE_TYPES.includes(
      value as typeof HUMANOID_GOAL_PREDICATE_TYPES[number]
    )
  ));
  const objectIds = uniqueStrings(predicates.flatMap((predicate) => [
    stringField(predicate, "object_id"),
    stringField(predicate, "container_id"),
    stringField(predicate, "support_id")
  ]));
  const solidIds = uniqueStrings(predicates.map(
    (predicate) => stringField(predicate, "block_id")
  ));
  const zoneIds = uniqueStrings(predicates.map(
    (predicate) => stringField(predicate, "zone_id")
  ));
  const outcomes: Array<typeof HUMANOID_EXPERIENCE_OUTCOMES[number]>
    = intent.outcome_scope === "successful"
      ? ["succeeded"]
      : intent.outcome_scope === "unsuccessful"
        ? ["rejected", "physically_failed"]
        : [...HUMANOID_EXPERIENCE_OUTCOMES];
  return {
    outcomes,
    ...(predicateTypes.length > 0 ? { predicate_types: predicateTypes } : {}),
    ...(objectIds.length > 0 ? { object_ids: objectIds } : {}),
    ...(solidIds.length > 0 ? { solid_ids: solidIds } : {}),
    ...(zoneIds.length > 0 ? { zone_ids: zoneIds } : {}),
    limit
  };
}

function relevantMemoryProjection(
  recalled: JsonValue,
  intent: z.infer<typeof RelevantMemoryIntentSchema>,
  request: HumanoidEmbodiedRecallRequest
): JsonValue {
  const record = jsonRecord(recalled);
  if (!record) throw new Error("Embodied memory recall returned a non-object payload");
  const actions = Array.isArray(record.actions)
    ? record.actions.flatMap((value) => {
        const action = jsonRecord(value);
        if (!action) return [];
        return [{
          source_ref: action.source_ref ?? null,
          transaction_id: action.transactionId ?? null,
          action: action.action ?? null,
          accepted: action.accepted ?? null,
          code: action.code ?? null,
          world_before_revision: action.worldBeforeRevision ?? null,
          world_after_revision: action.worldAfterRevision ?? null,
          frame_count: action.frameCount ?? null,
          historical_only: true
        }];
      })
    : [];
  return {
    historical_only: true,
    retrieval_intent: intent,
    resolved_query: {
      ...request,
      source_refs: []
    },
    current_world_revision: record.current_world_revision ?? null,
    ordered_source_refs: arrayValue(record.ordered_source_refs),
    experiences: arrayValue(record.experiences),
    actions,
    missing_source_refs: arrayValue(record.missing_source_refs),
    next_before_sequence: record.next_before_sequence ?? null,
    next_before_experience_sequence:
      record.next_before_experience_sequence ?? null
  };
}

function memoryProjectionCounts(payload: JsonValue): {
  experiences: number;
  actions: number;
} {
  const record = jsonRecord(payload);
  return {
    experiences: Array.isArray(record?.experiences) ? record.experiences.length : 0,
    actions: Array.isArray(record?.actions) ? record.actions.length : 0
  };
}

function relevantMemoryToolUseBehavior(): ToolUseBehavior {
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output"
        || result.tool.name !== "retrieve_relevant_embodied_memory") continue;
      const output = MemoryRetrievalOutputSchema.safeParse(
        outputObject(result.output)
      );
      if (!output.success) continue;
      return {
        isFinalOutput: true,
        isInterrupted: undefined,
        finalOutput: JSON.stringify(output.data)
      };
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function jsonRecord(value: unknown): Record<string, JsonValue> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, JsonValue>
    : null;
}

function arrayValue(value: JsonValue | undefined): JsonValue[] {
  return Array.isArray(value) ? value : [];
}

function stringField(
  value: Record<string, JsonValue>,
  key: string
): string | undefined {
  return typeof value[key] === "string" ? value[key] : undefined;
}

function uniqueStrings(values: readonly (string | undefined)[]): string[] {
  return [...new Set(values.filter((value): value is string => (
    typeof value === "string" && value.length > 0
  )))];
}

function sensorFusionTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof EmptyDelegationSchema, unknown> {
  const name = humanoidNeuralAgentToolName("sensorFusion");
  return tool({
    name,
    description: "Capture one authoritative current humanoid observation; this deterministic service must finish before model interpretation.",
    parameters: EmptyDelegationSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => ["perception", "feedback"].includes(
      runtime.neuralHarnessPhase().phase
    ) && runtime.neuralNodeEnabled({
      nodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      phases: ["perception", "feedback"]
    }) && !hasCurrentManagerEpisodeSignal(
      runtime,
      HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      "sensory_evidence"
    ),
    execute: async (_params, _context, details) => withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
      async () => {
      const invocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS.sensorFusion
      );
      const lease = await runtime.issueNeuralAuthorityLease({
        issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
        targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
        allowedSignalKinds: ["goal_context"],
        ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          ttlMs: 60_000,
          invocationId: invocation.invocationId,
          parentInvocationId: invocation.parentInvocationId,
          parentEpisodeId: requiredParentEpisodeId(
            HUMANOID_NEURAL_AGENT_IDS.perceptionManager
          )
      });
      try {
        const parentEpisodeId = requiredParentEpisodeId(
          HUMANOID_NEURAL_AGENT_IDS.perceptionManager
        );
        const managerInputs = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
          invocationId: parentEpisodeId
        }).filter((signal) => signal.direction === "descending");
        if (runtime.neuralHarnessPhase().phase === "feedback") {
          const commitment = runtime.neuralHierarchyState().active_skill_commitment;
          if (commitment?.state !== "completed") {
            throw new Error(
              "Post-execution Sensor Fusion requires Action Selection to complete the active commitment first"
            );
          }
          if (!managerInputs.some((signal) => neuralSignalHasAncestorKind(
            runtime.neuralHierarchyState(),
            signal,
            "skill_completed"
          ))) {
            throw new Error(
              "Post-execution Sensor Fusion is not causally linked to skill completion feedback"
            );
          }
        }
        const request = await runtime.publishNeuralSignal({
          kind: "goal_context",
          pathway: "perceptual_association",
          direction: "descending",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 80,
          causalParentIds: managerInputs.map((signal) => signal.signal_id),
          authorityLeaseId: lease.lease_id,
          invocationId: invocation.invocationId,
          parentInvocationId: invocation.parentInvocationId,
          payload: {
            harness_phase: runtime.neuralHarnessPhase().phase,
            capture: "authoritative_current_observation"
          }
        });
        const transactionId = details?.toolCall?.callId;
        if (!transactionId) {
          throw new Error("Sensor Fusion requires one Harness transaction identity");
        }
        if (details?.toolCall?.name !== name) {
          throw new Error("Sensor Fusion Harness tool identity mismatch");
        }
        const receipt = await runtime.captureNeuralObservation({
          transactionId,
          authority: {
            protocol: "neural-sensing-authority-v1",
            transaction_id: transactionId,
            manager_node_id: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
            sensor_node_id: HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
            manager_invocation_id: parentEpisodeId,
            sensor_invocation_id: invocation.invocationId,
            authority_lease_id: lease.lease_id,
            request_signal_id: request.signal_id,
            issued_world_revision: request.world_revision
          },
          ...(details?.signal ? { signal: details.signal } : {})
        });
        const receiptPayload = z.json().parse(JSON.parse(
          humanoidActionReceiptModelOutput(receipt)
        ));
        const sensoryEvidence = await runtime.publishNeuralSignal({
          kind: "sensory_evidence",
          pathway: "ascending_feedback",
          direction: "ascending",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 80,
          causalParentIds: [request.signal_id],
          sourceAuthorityLeaseId: lease.lease_id,
          invocationId: invocation.invocationId,
          parentInvocationId: invocation.parentInvocationId,
          payload: receiptPayload
        });
        return JSON.stringify({
          receipt: receiptPayload,
          source_signal_ids: [sensoryEvidence.signal_id]
        });
      } finally {
        await runtime.closeNeuralAuthorityLease({
          leaseId: lease.lease_id,
          closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
          reason: "sensor_fusion_capture_returned"
        });
      }
      },
      typeof details?.resumeState === "string",
      stableAgentToolInvocationId(
        HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
        details?.toolCall?.callId
      )
    )
  });
}

function motorIntentPlanningTools(
  runtime: HumanoidNeuralAgentRuntime
): Tool[] {
  return createHumanoidActionTools(
    runtime,
    HUMANOID_NEURAL_AGENT_IDS.motorIntent,
    [
      "submit_humanoid_skill_plan",
      "begin_humanoid_skill",
      "plan_humanoid_skill",
      "plan_whole_body_motion_candidates",
      "plan_humanoid_navigation"
    ],
    // The Agents SDK freezes an Agent's tool surface for the whole episode.
    // Motor Intent deliberately performs submit -> begin -> plan in one
    // episode, so every lifecycle endpoint must remain registered while the
    // Harness state machine still owns when each call is admissible.
    { availability: "stable" }
  ).map((planningTool) => {
    if (planningTool.type !== "function") return planningTool;
    if (planningTool.name === "submit_humanoid_skill_plan") {
      const invoke = planningTool.invoke;
      planningTool.invoke = async (context, rawInput, details) => {
        const expected = currentPremotorMotorProgram(runtime);
        let submittedInput: unknown;
        try {
          submittedInput = JSON.parse(rawInput);
        } catch {
          return invoke(context, rawInput, details);
        }
        const submitted = HumanoidSkillPlanProposalSchema.safeParse(
          submittedInput
        );
        if (!submitted.success
          || modelPayloadSha256(submitted.data)
            !== modelPayloadSha256(expected.skill_plan)) {
          return JSON.stringify({
            accepted: false,
            code: "premotor_motor_program_mismatch",
            tool: planningTool.name,
            required_protocol: expected.protocol,
            required_skill_plan_sha256: modelPayloadSha256(expected.skill_plan),
            automatic_actuation: false,
            next_response_contract: {
              mode: "corrected_tool_call_only",
              tool: planningTool.name,
              copy_directed_motor_program_verbatim: true,
              narration_allowed: false
            },
            recovery: "Call submit_humanoid_skill_plan once with the exact motor_program.skill_plan from the direct Premotor signal. Motor Intent compiles that program and cannot rewrite the Premotor strategy."
          });
        }
        return invoke(context, rawInput, details);
      };
      return planningTool;
    }
    // Skill-plan registration and phase binding are control-state transitions,
    // not physical intents. Let the SDK Agent loop consume their receipts and
    // continue until the runtime exposes one genuinely enabled planner.
    if (!MOTOR_INTENT_PLANNING_ACTIONS.has(planningTool.name)) {
      return planningTool;
    }
    const invoke = planningTool.invoke;
    planningTool.invoke = async (context, rawInput, details) => withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.rolloutGate,
      async () => {
      const invocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS.rolloutGate
      );
      const lease = await runtime.issueNeuralAuthorityLease({
        issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
        targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.rolloutGate,
        allowedSignalKinds: ["motor_intent"],
        ttlMs: 180_000,
        invocationId: invocation.invocationId,
        parentInvocationId: invocation.parentInvocationId,
        parentEpisodeId: requiredParentEpisodeId(
          HUMANOID_NEURAL_AGENT_IDS.motorIntent
        )
      });
      let intentSignal: NeuralSignal | undefined;
      try {
        const recoveryAttempt = motorIntentTransitRecoveryAttempt(
          runtime,
          planningTool.name
        );
        intentSignal = await runtime.publishNeuralSignal({
          kind: "motor_intent",
          pathway: "motor_intent",
          direction: "descending",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.rolloutGate,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 90,
          authorityLeaseId: lease.lease_id,
          invocationId: invocation.invocationId,
          parentInvocationId: invocation.parentInvocationId,
          payload: {
            planning_tool: planningTool.name,
            parameters: z.json().parse(JSON.parse(rawInput))
          }
        });
        const output = await invoke(context, rawInput, details);
        const receiptPayload = z.json().parse(
          typeof output === "string" ? JSON.parse(output) : output
        );
        const rolloutResult = await runtime.publishNeuralSignal({
          kind: "rollout_result",
          pathway: "cerebellar_prediction",
          direction: "ascending",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.rolloutGate,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 90,
          causalParentIds: [intentSignal.signal_id],
          sourceAuthorityLeaseId: lease.lease_id,
          invocationId: invocation.invocationId,
          parentInvocationId: invocation.parentInvocationId,
          payload: receiptPayload
        });
        const predictiveRollout = await runtime.publishNeuralSignal({
          kind: "rollout_result",
          pathway: "cerebellar_prediction",
          direction: "reentrant",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.rolloutGate,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.predictive,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 90,
          causalParentIds: [rolloutResult.signal_id],
          sourceAuthorityLeaseId: lease.lease_id,
          invocationId: invocation.invocationId,
          parentInvocationId: invocation.parentInvocationId,
          payload: receiptPayload
        });
        const outputRecord = outputObject(output);
        if (!outputRecord) {
          throw new Error("MuJoCo Rollout Gate returned a non-object planning receipt");
        }
        return JSON.stringify({
          ...outputRecord,
          ...(recoveryAttempt
            ? { motor_intent_recovery_attempt: recoveryAttempt }
            : {}),
          planning_tool_state: currentMotorIntentPlanningToolState(runtime),
          source_signal_ids: [
            rolloutResult.signal_id,
            predictiveRollout.signal_id
          ]
        });
      } finally {
        await runtime.closeNeuralAuthorityLease({
          leaseId: lease.lease_id,
          closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
          reason: intentSignal
            ? "deterministic_planning_rollout_returned"
            : "motor_intent_publication_failed"
        });
      }
      },
      typeof details?.resumeState === "string",
      stableAgentToolInvocationId(
        HUMANOID_NEURAL_AGENT_IDS.rolloutGate,
        details?.toolCall?.callId
      )
    );
    return planningTool;
  });
}

function establishSkillCommitmentTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof EstablishSkillCommitmentSchema, string> {
  return tool({
    name: "establish_skill_commitment",
    description: "Action Selection exclusively binds one Sensorimotor skill proposal to the current Goal epoch before motor compilation.",
    parameters: EstablishSkillCommitmentSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => runtime.neuralHarnessPhase().phase === "commitment_authorization"
      && (() => {
        const active = runtime.neuralHierarchyState().active_skill_commitment;
        return active === null || ["completed", "failed", "released"].includes(active.state);
      })(),
    execute: async (params) => {
      const invocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS.actionSelection
      );
      const harnessPhase = runtime.neuralHarnessPhase();
      if (!harnessPhase.goal_epoch_id) {
        throw new Error("Skill commitment requires one Harness-owned active Goal epoch");
      }
      const proposalSignals = currentManagerChildSignals(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        ["skill_proposal"]
      ).filter((signal) => signal.direction === "ascending"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection)
        .sort((left, right) => right.sequence - left.sequence);
      const latestProposal = proposalSignals[0];
      if (!latestProposal) {
        return JSON.stringify({
          accepted: false,
          code: "sensorimotor_proposal_unavailable",
          tool: "establish_skill_commitment",
          action_selection_episode_id: invocation.invocationId,
          automatic_actuation: false,
          next_response_contract: {
            mode: "new_sensorimotor_proposal_required",
            narration_allowed: false
          },
          recovery: "Delegate Sensorimotor once in the current Action Selection episode before establishing a commitment."
        });
      }
      const citedProposal = latestProposal;
      const commitmentSourceSignalIds = [citedProposal.signal_id];
      const proposedSkill = boundedProposedSkillFromPayload(citedProposal.payload);
      if (typeof proposedSkill !== "string" || proposedSkill !== params.skill) {
        return JSON.stringify({
          accepted: false,
          code: "skill_commitment_proposal_mismatch",
          tool: "establish_skill_commitment",
          required_bounded_skill: proposedSkill ?? null,
          rejected_skill: params.skill,
          automatic_actuation: false,
          next_response_contract: proposedSkill
            ? {
                mode: "corrected_tool_call_only",
                tool: "establish_skill_commitment",
                required_skill: proposedSkill,
                preserve_goal_epoch_id: true,
                preserve_source_signal_ids: true,
                narration_allowed: false
              }
            : null,
          recovery: proposedSkill
            ? `Call establish_skill_commitment once more with skill exactly '${proposedSkill}'. A compound task description cannot own one bounded commitment.`
            : "Sensorimotor did not return a machine-readable bounded Skill. Return to Sensorimotor selection instead of inventing a commitment."
        });
      }
      const admission = runtime.neuralSkillProposalAdmission?.(citedProposal);
      if (admission && !admission.accepted) {
        const recoveryCorrection = admission.relation === "recovery";
        await runtime.transitionNeuralHarnessPhase({
          phase: recoveryCorrection ? "recovery" : "skill_proposal",
          enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
          reason: recoveryCorrection
            ? "recovery_proposal_admission_rejected"
            : "skill_proposal_admission_rejected",
          commitmentId: null
        });
        return JSON.stringify({
          accepted: false,
          code: "skill_proposal_goal_misaligned",
          tool: "establish_skill_commitment",
          rejected_proposal_signal_id: citedProposal.signal_id,
          rejected_invocation: admission.invocation ?? null,
          reason: admission.reason ?? "The proposed Skill does not advance the active Goal",
          rejection_detail: admission.detail ?? null,
          automatic_actuation: false,
          next_response_contract: {
            mode: recoveryCorrection
              ? "new_recovery_proposal_required"
              : "new_sensorimotor_proposal_required",
            preserve_goal_epoch_id: true,
            narration_allowed: false
          },
          recovery: recoveryCorrection
            ? "Do not establish this commitment. Delegate the current failure and this direct correction through Sensorimotor to Recovery again; choose a materially different bounded invocation or escalate."
            : "Do not establish this commitment. Delegate Sensorimotor again for one bounded Skill whose real invocation advances the active Goal or establishes its next physical prerequisite."
        });
      }
      const commitment = await runtime.establishNeuralSkillCommitment({
        ownerNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        goalEpochId: harnessPhase.goal_epoch_id,
        skill: params.skill,
        terminationContract: parseNeuralJsonText(
          params.termination_contract_json,
          "Skill termination contract"
        ),
        sourceSignalIds: commitmentSourceSignalIds
      });
      await runtime.transitionNeuralHarnessPhase({
        phase: "motor_assessment",
        enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        reason: "action_selection_established_local_motor_commitment",
        goalEpochId: commitment.goal_epoch_id,
        commitmentId: commitment.commitment_id
      });
      return JSON.stringify({
        status: "skill_committed",
        commitment,
        next_phase: runtime.neuralHarnessPhase().phase,
        source_signal_ids: commitmentSourceSignalIds
      });
    }
  });
}

function neuralCycleCompletionTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof CycleCompletionSchema, string> {
  return tool({
    name: "complete_neural_autonomous_cycle",
    description: "Executive closes one physical cycle from the unique direct post-execution perceptual belief. The Harness binds its neural signal and physical evidence internally.",
    parameters: CycleCompletionSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => {
      const completion = runtime.cycleCompletionReadiness();
      return runtime.neuralHarnessPhase().phase === "cycle_completion"
        && completion.status === "ready"
        && completion.observed_after_execution
        && runtime.autonomyReadiness() === "complete_cycle";
    },
    execute: (params) => {
      const readiness = runtime.cycleCompletionReadiness();
      if (readiness.status !== "ready" || !readiness.observed_after_execution) {
        throw new Error("Executive cycle completion is no longer ready");
      }
      const evidenceTransactionIds = readiness.evidence_transaction_ids;
      const execution = runtime.validateCycleEvidence(evidenceTransactionIds);
      const invocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS.executive
      );
      const eligiblePostExecutionBeliefs = currentDelegationSourceSignals(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.executive,
        invocation.invocationId
      ).filter((signal) => signal.kind === "perceptual_belief"
        && signal.world_revision >= execution.worldAfterRevision)
        .sort((left, right) => right.sequence - left.sequence);
      const postExecutionBeliefs = eligiblePostExecutionBeliefs.length === 1
        ? eligiblePostExecutionBeliefs
        : eligiblePostExecutionBeliefs.filter((signal) => (
            params.perceptual_belief_signal_ids.includes(signal.signal_id)
          ));
      if (postExecutionBeliefs.length !== 1) {
        throw new Error(
          "Executive cycle completion requires one current direct perceptual belief observed after durable physical execution"
        );
      }
      const sourceSignalIds = [postExecutionBeliefs[0]!.signal_id];
      return JSON.stringify({
        status: "cycle_completed",
        summary: params.summary,
        evidence_transaction_ids: evidenceTransactionIds,
        source_signal_ids: sourceSignalIds,
        world_revision: execution.worldAfterRevision,
        executed_action: execution.action,
        ...(params.next_intent ? { next_intent: params.next_intent } : {})
      });
    }
  });
}

function neuralSatisfiedGoalCompletionTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof SatisfiedGoalCompletionSchema, string> {
  return tool({
    name: "complete_neural_satisfied_goal",
    description: "Executive closes an already physically satisfied Goal without inventing another movement.",
    parameters: SatisfiedGoalCompletionSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => runtime.neuralHarnessPhase().phase === "cycle_completion"
      && runtime.autonomyReadiness() === "complete_satisfied_goal",
    execute: (params) => JSON.stringify({
      status: "satisfied_goal_completed",
      summary: params.summary,
      verification: runtime.validateSatisfiedGoal()
    })
  });
}

function acknowledgeSafetyInterruptTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof AcknowledgeSafetyInterruptSchema, string> {
  return tool({
    name: "acknowledge_safety_interrupt",
    description: "Action Selection acknowledges one exact Body→Reflex afferent safety interrupt and atomically fails the interrupted commitment before current Perception.",
    parameters: AcknowledgeSafetyInterruptSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => runtime.neuralHarnessPhase().phase === "safety_interrupt"
      && runtime.pendingNeuralSafetyInterrupts().some(
        (interrupt) => interrupt.status === "pending"
      ),
    execute: async (params) => {
      const pending = runtime.pendingNeuralSafetyInterrupts().filter(
        (interrupt) => interrupt.status === "pending"
      );
      const latest = pending[0];
      if (!latest || latest.interrupt_id !== params.interrupt_id) {
        return JSON.stringify({
          accepted: false,
          code: "safety_interrupt_not_current",
          tool: "acknowledge_safety_interrupt",
          required_interrupt_id: latest?.interrupt_id ?? null,
          rejected_interrupt_id: params.interrupt_id,
          automatic_actuation: false,
          recovery: "Acknowledge only the newest pending safety interrupt shown in neural_hierarchy.pending_safety_interrupts."
        });
      }
      const acknowledged = await runtime.acknowledgeNeuralSafetyInterrupt({
        interruptId: latest.interrupt_id,
        acknowledgedByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection
      });
      return JSON.stringify({
        status: "skill_failed",
        protocol: "stationary_safety_interrupt_acknowledgement_v1",
        reason: params.reason,
        safety_interrupt: acknowledged.interrupt,
        failed_commitment: acknowledged.commitment,
        source_signal_ids: [],
        source_interrupt_ids: [acknowledged.interrupt.interrupt_id],
        next_phase: "perception",
        automatic_actuation: false
      });
    }
  });
}

function transitionSkillCommitmentTool(
  runtime: HumanoidNeuralAgentRuntime,
  state: "completed" | "failed" | "released"
): FunctionTool<unknown, typeof TransitionSkillCommitmentSchema, string> {
  const phases: Readonly<Record<typeof state, readonly NeuralHarnessPhase[]>> = {
    // The Agents SDK fixes an episode's tool surface before a nested physical
    // child returns. Advertise lifecycle transitions at execution entry so the
    // same Action Selection episode can consume the later physical feedback.
    // execute() remains the authority boundary and rejects any premature use.
    completed: ["execution", "feedback"],
    failed: ["execution", "feedback", "recovery"],
    released: [
      "skill_proposal",
      "motor_assessment",
      "rollout_review",
      "execution",
      "feedback",
      "recovery"
    ]
  };
  const names = {
    completed: "complete_skill_commitment",
    failed: "fail_skill_commitment",
    released: "release_skill_commitment"
  } as const;
  const allowedFrom: Readonly<Record<
    typeof state,
    ReadonlySet<NeuralSkillCommitment["state"]>
  >> = {
    completed: new Set(["executing"]),
    failed: new Set(["committed", "executing"]),
    released: new Set(["proposed", "committed", "executing"])
  };
  return tool({
    name: names[state],
    description: `Action Selection exclusively transitions the active skill commitment to ${state}.`,
    parameters: TransitionSkillCommitmentSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => {
      const active = runtime.neuralHierarchyState().active_skill_commitment;
      return active !== null
        && allowedFrom[state].has(active.state)
        && phases[state].includes(runtime.neuralHarnessPhase().phase);
    },
    execute: async (params) => {
      const hierarchy = runtime.neuralHierarchyState();
      const active = hierarchy.active_skill_commitment;
      if (!active || active.commitment_id !== params.commitment_id) {
        throw new Error("Action Selection transition does not reference the active commitment");
      }
      if (state === "completed") {
        const outcome = runtime.neuralSkillCommitmentOutcome(active);
        if (outcome.status !== "completed") {
          return JSON.stringify({
            accepted: false,
            code: "skill_termination_contract_unsatisfied",
            commitment_id: active.commitment_id,
            outcome,
            automatic_actuation: false,
            recovery: "The physical chunk succeeded but the semantic Skill termination contract is still false. Release this exhausted plan commitment with the direct execution feedback, then select the next bounded continuation Skill without changing the Goal."
          });
        }
      }
      const pending = runtime.pendingNeuralSignals({
        targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection
      });
      const pendingIds = new Set(pending.map((signal) => signal.signal_id));
      const requiredKind: Partial<Record<typeof state, NeuralSignalKind>> = {
        completed: "skill_completed",
        failed: "skill_failed"
      };
      const kind = requiredKind[state];
      let transitionSourceSignalIds = params.source_signal_ids;
      if (kind) {
        // Lifecycle provenance is Harness state, not a UUID-selection task for
        // the model. Compatible providers can legally retain an older direct
        // child id in Session history after the SDK opens a replacement parent
        // episode. Resolve the newest current descendant of this exact
        // commitment instead of failing an otherwise valid physical outcome.
        const lifecycleSignal = pending.filter((signal) => (
          signal.kind === kind
            && neuralSignalBindsCommitment(hierarchy, signal, active)
        )).sort((left, right) => right.sequence - left.sequence)[0];
        if (!lifecycleSignal) {
          throw new Error(
            `Commitment ${state} has no current ${kind} signal bound to the active Skill`
          );
        }
        transitionSourceSignalIds = [lifecycleSignal.signal_id];
      } else if (!params.source_signal_ids.some(
        (signalId) => pendingIds.has(signalId)
      )) {
        throw new Error("Commitment transition requires current child feedback");
      }
      const commitment = await runtime.transitionNeuralSkillCommitment({
        ownerNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        commitmentId: params.commitment_id,
        state,
        sourceSignalIds: transitionSourceSignalIds
      });
      if (state === "completed" || state === "failed") {
        await runtime.transitionNeuralHarnessPhase({
          phase: state === "completed" ? "feedback" : "perception",
          enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
          reason: state === "completed"
            ? "action_selection_closed_commitment_for_feedback"
            : "action_selection_failed_commitment_observation_required",
          goalEpochId: commitment.goal_epoch_id,
          commitmentId: null
        });
      } else if (state === "released"
        && runtime.autonomyReadiness() === "post_failure_observation") {
        // A rejected plan may not actuate the plant, but its geometry and
        // availability evidence belongs to an older world cut. Re-enter
        // Perception before Recovery so the replacement decision joins the
        // durable rejection with a current belief instead of recycling the
        // pre-rejection proposal context.
        await runtime.transitionNeuralHarnessPhase({
          phase: "perception",
          enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
          reason: "released_failed_commitment_requires_current_perception",
          goalEpochId: commitment.goal_epoch_id,
          commitmentId: null
        });
      }
      return JSON.stringify({
        status: `skill_${state}`,
        commitment,
        next_phase: runtime.neuralHarnessPhase().phase,
        source_signal_ids: transitionSourceSignalIds
      });
    }
  });
}

function authorizeSkillExecutionTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof AuthorizeSkillExecutionSchema, string> {
  return tool({
    name: "authorize_skill_execution",
    description: "Authorize the active Skill after the unique direct Sensorimotor rollout has a Predictive certificate. The Harness resolves that direct evidence; provide no signal ids.",
    parameters: AuthorizeSkillExecutionSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => runtime.neuralHarnessPhase().phase === "rollout_review",
    execute: async (params) => {
      const hierarchy = runtime.neuralHierarchyState();
      const active = hierarchy.active_skill_commitment;
      if (!active || active.commitment_id !== params.commitment_id
        || active.state !== "committed") {
        throw new Error(
          "Action Selection execution authorization does not reference the active committed Skill"
        );
      }
      const certificates = Object.values(hierarchy.rollout_certificates).filter(
        (candidate) => candidate.status === "active"
          && candidate.commitment_id === active.commitment_id
      );
      if (certificates.length !== 1) {
        // Tool discovery is fixed at the beginning of an Agents SDK episode,
        // while the nested Predictive child may issue its certificate later
        // in that same episode. A premature compatible-model call is therefore
        // an ordinary rejected decision, not a Harness/runtime failure.
        return JSON.stringify({
          accepted: false,
          code: "predictive_certificate_pending",
          active_certificate_count: certificates.length,
          commitment_id: active.commitment_id,
          required_next_tool: humanoidNeuralAgentToolName("sensorimotorManager"),
          automatic_actuation: false,
          recovery: "Delegate the committed branch to Sensorimotor so its Predictive child can certify the real rollout before execution authorization."
        });
      }
      const certificate = certificates[0]!;
      const certifiedRawRollout = hierarchy.signals[
        certificate.rollout_signal_id
      ];
      if (!certifiedRawRollout
        || certifiedRawRollout.kind !== "rollout_result"
        || certifiedRawRollout.source_node_id
          !== HUMANOID_NEURAL_AGENT_IDS.rolloutGate
        || modelPayloadSha256(certifiedRawRollout.payload)
          !== certificate.rollout_payload_sha256) {
        throw new Error("Execution authorization lost its certified raw rollout");
      }
      // The Predictive certificate and its direct Sensorimotor return are
      // durable control authority. A process pause can end the SDK invocation
      // that received the return without invalidating either artifact. Resolve
      // the one still-live structural child edge by certificate causality,
      // rather than requiring its parent_episode_id to equal the replacement
      // Action Selection episode created after resume.
      const rollouts = Object.values(hierarchy.signals).filter((signal) => (
        signal.status === "pending"
          && isCurrentNeuralSignal(runtime, signal)
          && signal.kind === "rollout_result"
          && signal.direction === "ascending"
          && signal.source_node_id
            === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
          && signal.target_node_id
            === HUMANOID_NEURAL_AGENT_IDS.actionSelection
        && signal.causal_parent_ids.includes(certificate.predictive_signal_id)
        && neuralSignalHasAncestorId(
          hierarchy,
          signal,
          certifiedRawRollout.signal_id
        )
      ));
      if (rollouts.length !== 1) {
        throw new Error(
          `Execution authorization requires one live direct certified Sensorimotor rollout; found ${rollouts.length}`
        );
      }
      const sourceSignalIds = [rollouts[0]!.signal_id];
      const commitment = await runtime.transitionNeuralSkillCommitment({
        ownerNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        commitmentId: params.commitment_id,
        state: "executing",
        sourceSignalIds
      });
      return JSON.stringify({
        status: "skill_executing",
        commitment,
        reason: params.reason,
        source_signal_ids: sourceSignalIds
      });
    }
  });
}

/**
 * Invoke Goal Valuation directly when the durable state exposes exactly one
 * legal Executive edge. Goal choice still belongs to the Goal Manager model;
 * the Harness merely avoids spending a separate root turn on saying which
 * child must make that choice.
 */
export async function orchestrateDirectNeuralGoalValuation(
  runtime: HumanoidNeuralAgentRuntime,
  signal?: AbortSignal
): Promise<boolean> {
  const phase = runtime.neuralHarnessPhase();
  if (phase.phase !== "bootstrapping" && phase.phase !== "goal_valuation") {
    return false;
  }
  const pendingEscalation = runtime.pendingNeuralSignals({
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
    kinds: ["escalation"]
  }).some((candidate) => candidate.direction === "ascending"
    && isCurrentNeuralSignal(runtime, candidate));
  if (pendingEscalation) return false;
  const binding = DIRECT_GOAL_MANAGER_TOOLS.get(runtime);
  if (!binding) {
    throw new Error("Direct Goal Manager is not bound to the active hierarchy");
  }
  const orchestrationKey = [
    "direct-goal-valuation-v1",
    runtime.neuralHierarchyState().epoch_id,
    phase.goal_epoch_id ?? "no-goal",
    phase.sequence
  ].join(":");
  const executiveInvocationId = stableAgentToolInvocationId(
    HUMANOID_NEURAL_AGENT_IDS.executive,
    orchestrationKey
  );
  const toolCallId = stableAgentToolInvocationId(
    HUMANOID_NEURAL_AGENT_IDS.goalManager,
    orchestrationKey
  );
  return withAgentInvocation(
    HUMANOID_NEURAL_AGENT_IDS.executive,
    async () => {
      const context = new RunContext();
      const enabled = binding.tool.isEnabled
        ? await binding.tool.isEnabled(context, binding.parent)
        : true;
      if (!enabled) throw new Error("Goal Manager is disabled during valuation");
      const rawInput = JSON.stringify({
        signal_kind: "goal_context",
        source_signal_ids: [],
        ttl_revisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
        priority: 100
      });
      const output = await binding.tool.invoke(context, rawInput, {
        toolCall: {
          type: "function_call",
          callId: toolCallId,
          name: binding.tool.name,
          arguments: rawInput
        },
        ...(signal ? { signal } : {})
      });
      const parsed = outputObject(output);
      const result = parsed
        ? directChildNeuralOutput(GoalValuationOutputSchema, parsed)
        : undefined;
      if (!result) {
        throw new Error("Direct Goal Manager returned no typed valuation result");
      }
      return result.signal_kind === "goal_selected";
    },
    false,
    executiveInvocationId
  );
}

/**
 * Commit a cycle only after the physical checker and post-execution belief
 * have already reduced the transition to a single legal operation. The
 * Executive still owns Goal valuation and escalation choices; repeating a
 * deterministic completion receipt through another model response is not a
 * cognitive decision.
 */
export async function orchestrateDeterministicNeuralCycleCompletion(
  runtime: HumanoidNeuralAgentRuntime,
  signal?: AbortSignal
): Promise<string | undefined> {
  if (runtime.neuralHarnessPhase().phase !== "cycle_completion") {
    return undefined;
  }
  const readiness = runtime.autonomyReadiness();
  if (readiness !== "complete_cycle"
    && readiness !== "complete_satisfied_goal") return undefined;
  const binding = DETERMINISTIC_CYCLE_TOOLS.get(runtime);
  if (!binding) {
    throw new Error("Deterministic cycle tools are not bound to the active hierarchy");
  }
  const belief = runtime.pendingNeuralSignals({
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
    kinds: ["perceptual_belief"]
  }).filter((candidate) => (
    candidate.direction === "ascending"
      && isCurrentNeuralSignal(runtime, candidate)
  )).sort((left, right) => right.sequence - left.sequence)[0];
  if (readiness === "complete_cycle" && !belief) {
    throw new Error(
      "Deterministic cycle completion requires current post-execution perception"
    );
  }
  const bindingTool = readiness === "complete_cycle"
    ? binding.completion
    : binding.satisfied;
  const orchestrationKey = [
    "deterministic-cycle-completion-v1",
    runtime.neuralHierarchyState().epoch_id,
    runtime.neuralHarnessPhase().goal_epoch_id ?? "no-goal",
    runtime.neuralHarnessPhase().sequence,
    readiness
  ].join(":");
  const beliefEpisode = z.string().uuid().safeParse(belief?.parent_episode_id);
  const executiveInvocationId = beliefEpisode.success
    ? beliefEpisode.data as ReturnType<typeof stableAgentToolInvocationId>
    : stableAgentToolInvocationId(
        HUMANOID_NEURAL_AGENT_IDS.executive,
        orchestrationKey
      );
  const toolCallId = stableAgentToolInvocationId(
    HUMANOID_NEURAL_AGENT_IDS.executive,
    `${orchestrationKey}:tool`
  );
  return withAgentInvocation(
    HUMANOID_NEURAL_AGENT_IDS.executive,
    async () => {
      const context = new RunContext();
      const enabled = bindingTool.isEnabled
        ? await bindingTool.isEnabled(context, binding.parent)
        : true;
      if (!enabled) {
        throw new Error(`Cycle completion is disabled for ${readiness}`);
      }
      const rawInput = JSON.stringify(readiness === "complete_cycle"
        ? {
            summary: "Completed the active Goal cycle from verified physical execution and current perception.",
            perceptual_belief_signal_ids: belief ? [belief.signal_id] : []
          }
        : {
            summary: "Completed the active Goal from its already verified physical state."
          });
      const output = await bindingTool.invoke(context, rawInput, {
        toolCall: {
          type: "function_call",
          callId: toolCallId,
          name: bindingTool.name,
          arguments: rawInput
        },
        ...(signal ? { signal } : {})
      });
      const parsed = outputObject(output);
      if (!parsed || (parsed.status !== "cycle_completed"
        && parsed.status !== "satisfied_goal_completed")) {
        throw new Error("Deterministic cycle tool returned no terminal receipt");
      }
      const sourceSignalIds = z.array(z.string().uuid()).max(64).catch([]).parse(
        parsed.source_signal_ids
      );
      return JSON.stringify({
        signal_kind: "skill_completed",
        summary: typeof parsed.summary === "string"
          ? parsed.summary
          : String(parsed.status),
        payload_json: JSON.stringify(parsed),
        source_signal_ids: sourceSignalIds,
        confidence: 1
      });
    },
    false,
    executiveInvocationId
  );
}

/**
 * Advance phases where the Executive has no semantic branch to choose. The
 * structural root still owns the invocation, while Action Selection remains
 * a real independent Agent.asTool episode with its own Session and model
 * decisions. This removes an otherwise redundant Executive model turn that
 * could only say "delegate Action Selection" and occasionally ended in prose.
 */
export async function orchestrateDirectNeuralActionSelection(
  runtime: HumanoidNeuralAgentRuntime,
  signal?: AbortSignal
): Promise<boolean> {
  const state = runtime.neuralHierarchyState();
  const phase = state.harness_phase.phase;
  if (![
    "perception",
    "skill_proposal",
    "commitment_authorization",
    "motor_assessment",
    "motor_planning",
    "rollout_review",
    "feedback",
    "safety_interrupt",
    "recovery"
  ].includes(phase)) return false;
  const binding = DIRECT_ACTION_SELECTION_TOOLS.get(runtime);
  if (!binding) {
    throw new Error("Direct Action Selection is not bound to the active hierarchy");
  }
  const source = directActionSelectionSource(runtime, phase);
  const signalKind = directActionSelectionSignalKind(runtime, phase, source);
  const rawInput = JSON.stringify({
    signal_kind: signalKind,
    source_signal_ids: source ? [source.signal_id] : [],
    ttl_revisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
    priority: 100
  });
  const orchestrationKey = [
    "direct-action-selection-v1",
    state.epoch_id,
    state.harness_phase.goal_epoch_id ?? "no-goal",
    state.harness_phase.sequence,
    state.harness_phase.commitment_id ?? "no-commitment"
  ].join(":");
  const sourceParentEpisode = z.string().uuid().safeParse(
    source?.parent_episode_id
  );
  const executiveInvocationId = sourceParentEpisode.success
    ? sourceParentEpisode.data as ReturnType<typeof stableAgentToolInvocationId>
    : stableAgentToolInvocationId(
        HUMANOID_NEURAL_AGENT_IDS.executive,
        orchestrationKey
      );
  const toolCallId = stableAgentToolInvocationId(
    HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    orchestrationKey
  );
  return withAgentInvocation(
    HUMANOID_NEURAL_AGENT_IDS.executive,
    async () => {
      const context = new RunContext();
      const enabled = binding.tool.isEnabled
        ? await binding.tool.isEnabled(context, binding.parent)
        : true;
      if (!enabled) {
        throw new Error(`Action Selection is disabled in ${phase}`);
      }
      const output = await binding.tool.invoke(context, rawInput, {
        toolCall: {
          type: "function_call",
          callId: toolCallId,
          name: binding.tool.name,
          arguments: rawInput
        },
        ...(signal ? { signal } : {})
      });
      const parsed = outputObject(output);
      if (!parsed || !directChildNeuralOutput(
        ActionSelectionOutputSchema,
        parsed
      )) {
        throw new Error(
          "Direct Action Selection returned no typed hierarchical result"
        );
      }
      return true;
    },
    false,
    executiveInvocationId
  );
}

function directActionSelectionSource(
  runtime: HumanoidNeuralAgentRuntime,
  phase: NeuralHarnessPhase
): NeuralSignal | undefined {
  const preferredKinds: readonly NeuralSignalKind[] = phase === "perception"
    ? [
        "goal_selected",
        "skill_completed",
        "skill_failed",
        "execution_receipt",
        "goal_context"
      ]
    : phase === "skill_proposal" || phase === "commitment_authorization"
      ? ["perceptual_belief", "goal_selected", "goal_context"]
      : phase === "feedback"
        ? ["skill_completed", "skill_failed", "execution_receipt"]
        : phase === "safety_interrupt" || phase === "recovery"
          ? ["prediction_error", "skill_failed", "escalation"]
          : ["skill_commitment", "rollout_result", "goal_context"];
  return runtime.pendingNeuralSignals({
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
    kinds: preferredKinds
  }).filter((candidate) => (
    candidate.direction === "ascending"
      && isCurrentNeuralSignal(runtime, candidate)
  )).sort((left, right) => right.sequence - left.sequence)[0];
}

function directActionSelectionSignalKind(
  runtime: HumanoidNeuralAgentRuntime,
  phase: NeuralHarnessPhase,
  source: NeuralSignal | undefined
): NeuralSignalKind {
  if (phase === "feedback") {
    const commitment = runtime.neuralHierarchyState().active_skill_commitment;
    const feedback = commitment
      ? currentCommitmentLifecycleFeedback(runtime, commitment, {
          pendingOnly: false
        })
      : undefined;
    if (feedback?.kind === "skill_completed" || feedback?.kind === "skill_failed") {
      return feedback.kind;
    }
    throw new Error("Feedback phase has no authoritative physical completion signal");
  }
  if (phase === "safety_interrupt") {
    return "prediction_error";
  }
  if (phase === "recovery") {
    // Recovery authority is typed by the failure that opened it. In
    // particular, skill_failed requires a causally newer perceptual belief,
    // while escalation must remain eligible for supervisory propagation.
    // Relabelling either as prediction_error here weakens those contracts and
    // can send a supervisory escalation back around the local recovery loop.
    if (source?.kind === "prediction_error"
      || source?.kind === "skill_failed"
      || source?.kind === "escalation") {
      return source.kind;
    }
    const durableFailure = jsonRecord(
      jsonRecord(runtime.recoveryFailureEvidence())?.failure_receipt
    );
    if (typeof durableFailure?.action === "string"
      && durableFailure.action.startsWith("execute_")) {
      return "skill_failed";
    }
    return "prediction_error";
  }
  if (["motor_assessment", "motor_planning", "rollout_review"].includes(phase)) {
    return "skill_commitment";
  }
  if (phase === "perception"
    && (source?.kind === "goal_selected"
      || source?.kind === "skill_completed"
      || source?.kind === "skill_failed")) {
    return source.kind;
  }
  return "goal_context";
}

/**
 * Advance the one phase that contains no cognitive choice. The Harness opens
 * the fixed Executive -> Action Selection -> Sensorimotor control path, then
 * invokes the independent non-thinking Execution Dispatcher Agent. Its real
 * required tool call is the model authority for the physical transaction;
 * cognitive Agent Sessions never change reasoning mode.
 */
export async function orchestrateCertifiedNeuralExecution(
  runtime: HumanoidNeuralAgentRuntime,
  signal?: AbortSignal
): Promise<boolean> {
  const state = runtime.neuralHierarchyState();
  if (state.harness_phase.phase !== "execution") return false;
  const commitment = state.active_skill_commitment;
  if (!commitment || commitment.state !== "executing"
    || state.harness_phase.commitment_id !== commitment.commitment_id) {
    throw new Error(
      "Deterministic execution phase requires one Action Selection-authorized commitment"
    );
  }
  const certificates = Object.values(state.rollout_certificates).filter(
    (candidate) => candidate.status === "active"
      && candidate.commitment_id === commitment.commitment_id
  );
  if (certificates.length !== 1) {
    throw new Error(
      `Deterministic execution phase requires one active Predictive certificate; found ${certificates.length}`
    );
  }
  if (!runtime.neuralExecutionAvailable()) {
    throw new Error("Certified deterministic execution is not physically available");
  }
  const certificate = certificates[0]!;
  const dispatcherBinding = CERTIFIED_EXECUTION_DISPATCHER_TOOLS.get(runtime);
  if (!dispatcherBinding) {
    throw new Error(
      "Certified Execution Dispatcher is not bound to the active neural hierarchy"
    );
  }
  const { tool: dispatcherTool, parent: sensorimotorManager } = dispatcherBinding;
  const orchestrationKey = [
    "certified-execution-v1",
    commitment.commitment_id,
    certificate.certificate_id
  ].join(":");
  const executiveInvocationId = stableAgentToolInvocationId(
    HUMANOID_NEURAL_AGENT_IDS.executive,
    orchestrationKey
  );
  const actionInvocationId = stableAgentToolInvocationId(
    HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    orchestrationKey
  );
  const sensorimotorInvocationId = stableAgentToolInvocationId(
    HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
    orchestrationKey
  );
  const dispatcherToolCallId = stableAgentToolInvocationId(
    HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
    orchestrationKey
  );

  return withAgentInvocation(
    HUMANOID_NEURAL_AGENT_IDS.executive,
    () => withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      async () => {
        const actionInvocation = requiredHarnessInvocation(
          HUMANOID_NEURAL_AGENT_IDS.actionSelection
        );
        const actionLease = await runtime.issueNeuralAuthorityLease({
          issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
          targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
          allowedSignalKinds: ["skill_commitment"],
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          ttlMs: 600_000,
          invocationId: actionInvocation.invocationId,
          parentInvocationId: actionInvocation.parentInvocationId,
          parentEpisodeId: executiveInvocationId
        });
        let executiveAdmission: NeuralSignal | undefined;
        try {
          executiveAdmission = await runtime.publishNeuralSignal({
            kind: "skill_commitment",
            pathway: "executive_control",
            direction: "descending",
            sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
            ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
            priority: 100,
            causalParentIds: [
              certificate.predictive_signal_id,
              certificate.rollout_signal_id
            ],
            authorityLeaseId: actionLease.lease_id,
            invocationId: actionLease.invocation_id,
            parentInvocationId: actionLease.parent_invocation_id,
            payload: commitment
          });

          return await withAgentInvocation(
            HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            async () => {
              const sensorimotorInvocation = requiredHarnessInvocation(
                HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
              );
              const sensorimotorLease = await runtime.issueNeuralAuthorityLease({
                issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
                targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
                allowedSignalKinds: ["skill_commitment"],
                ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
                ttlMs: 600_000,
                invocationId: sensorimotorInvocation.invocationId,
                parentInvocationId: sensorimotorInvocation.parentInvocationId,
                parentEpisodeId: actionInvocation.invocationId
              });
              let sensorimotorAdmission: NeuralSignal | undefined;
              try {
                sensorimotorAdmission = await runtime.publishNeuralSignal({
                  kind: "skill_commitment",
                  pathway: "sensorimotor_selection",
                  direction: "descending",
                  sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
                  targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
                  ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
                  priority: 100,
                  causalParentIds: [
                    executiveAdmission!.signal_id,
                    certificate.predictive_signal_id,
                    certificate.rollout_signal_id
                  ],
                  authorityLeaseId: sensorimotorLease.lease_id,
                  invocationId: sensorimotorLease.invocation_id,
                  parentInvocationId: sensorimotorLease.parent_invocation_id,
                  payload: commitment
                });
                const rawInput = JSON.stringify({
                  signal_kind: "skill_commitment",
                  source_signal_ids: [sensorimotorAdmission.signal_id],
                  ttl_revisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
                  priority: 100
                });
                const dispatcherContext = new RunContext();
                const dispatcherEnabled = dispatcherTool.isEnabled
                  ? await dispatcherTool.isEnabled(
                      dispatcherContext,
                      sensorimotorManager
                    )
                  : true;
                if (!dispatcherEnabled) {
                  throw new Error(
                    "Certified Execution Dispatcher is disabled in execution phase"
                  );
                }
                const output = await dispatcherTool.invoke(
                  dispatcherContext,
                  rawInput,
                  {
                    toolCall: {
                      type: "function_call",
                      callId: dispatcherToolCallId,
                      name: dispatcherTool.name,
                      arguments: rawInput
                    },
                    ...(signal ? { signal } : {})
                  }
                );
                const parsed = outputObject(output);
                if (!parsed) {
                  throw new Error(
                    "Certified Execution Dispatcher returned a non-object receipt"
                  );
                }
                const sourceSignalIds = z.array(z.string().uuid()).max(64).parse(
                  parsed.source_signal_ids
                );
                const afterExecution = runtime.neuralHierarchyState();
                const completionSignals = sourceSignalIds.map(
                  (signalId) => afterExecution.signals[signalId]
                ).filter((candidate): candidate is NeuralSignal => (
                  candidate !== undefined
                    && (candidate.kind === "skill_completed"
                      || candidate.kind === "skill_failed")
                    && candidate.source_node_id
                      === HUMANOID_NEURAL_AGENT_IDS.executionDispatcher
                    && candidate.target_node_id
                      === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
                ));
                if (completionSignals.length !== 1) {
                  throw new Error(
                    `Certified execution requires one Dispatcher completion signal; found ${completionSignals.length}`
                  );
                }
                const receipt = JsonValueSchema.parse(parsed.payload);
                await runtime.publishNeuralSignal({
                  kind: completionSignals[0]!.kind,
                  pathway: "sensorimotor_selection",
                  direction: "ascending",
                  sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
                  targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
                  ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
                  priority: 100,
                  causalParentIds: [
                    sensorimotorAdmission.signal_id,
                    ...sourceSignalIds
                  ],
                  sourceAuthorityLeaseId: sensorimotorLease.lease_id,
                  invocationId: sensorimotorLease.invocation_id,
                  parentInvocationId: sensorimotorLease.parent_invocation_id,
                  payload: receipt
                });
                await runtime.consumeNeuralSignals(
                  HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
                  [sensorimotorAdmission.signal_id, ...sourceSignalIds]
                );
                return true;
              } finally {
                await runtime.closeNeuralAuthorityLease({
                  leaseId: sensorimotorLease.lease_id,
                  closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
                  reason: "certified_dispatcher_execution_returned"
                });
              }
            },
            false,
            sensorimotorInvocationId
          );
        } finally {
          if (executiveAdmission) {
            await runtime.consumeNeuralSignals(
              HUMANOID_NEURAL_AGENT_IDS.actionSelection,
              [executiveAdmission.signal_id]
            );
          }
          await runtime.closeNeuralAuthorityLease({
            leaseId: actionLease.lease_id,
            closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
            reason: "certified_action_execution_returned"
          });
        }
      },
      false,
      actionInvocationId
    ),
    false,
    executiveInvocationId
  );
}

function serialExecutionTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof ExecutionTaskSchema, string> {
  const name = humanoidNeuralAgentToolName("executor");
  // Sensorimotor may run its explicitly read-only specialist group in
  // parallel, but physical admission remains a process-local single-writer
  // boundary even if a provider emits duplicate execution calls in one turn.
  const executionMutex = new Mutex();
  return tool({
    name,
    description: "Execute the certified plan bound to the current direct skill_commitment signal. Pass no certificate or planning internals.",
    parameters: ExecutionTaskSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => runtime.neuralExecutionAvailable()
      && runtime.neuralHarnessPhase().phase === "execution"
      && runtime.neuralHierarchyState().active_skill_commitment?.state === "executing"
      && hasCurrentManagerEpisodeSignal(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
        "skill_commitment"
      )
      && Object.values(runtime.neuralHierarchyState().rollout_certificates).filter(
        (candidate) => candidate.status === "active"
          && candidate.commitment_id === runtime.neuralHierarchyState()
            .active_skill_commitment?.commitment_id
      ).length === 1,
    execute: (params, _context, details) => executionMutex.runExclusive(async () => {
      const managerInvocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS.executionDispatcher
      );
      return withAgentInvocation(HUMANOID_NEURAL_AGENT_IDS.executor, async () => {
      const executorInvocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS.executor
      );
      const hierarchy = runtime.neuralHierarchyState();
      const commitment = hierarchy.active_skill_commitment;
      if (!commitment || commitment.state !== "executing"
        || hierarchy.harness_phase.commitment_id !== commitment.commitment_id) {
        throw new Error("Serial execution requires an Action Selection-authorized commitment");
      }
      const directCommitmentSignals = runtime.pendingNeuralSignals({
        targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
        kinds: ["skill_commitment"]
      }).filter((signal) => signal.status === "pending"
        && signal.kind === "skill_commitment"
        && signal.direction === "descending"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.executionDispatcher
        && signal.invocation_id === managerInvocation.invocationId);
      if (directCommitmentSignals.length !== 1) {
        throw new Error(
          "Serial execution requires the one direct Sensorimotor skill_commitment signal from this Dispatcher episode"
        );
      }
      const directCommitment = NeuralSkillCommitmentSchema.parse(
        directCommitmentSignals[0]!.payload
      );
      if (directCommitment.commitment_id !== commitment.commitment_id
        || directCommitment.state !== "executing") {
        throw new Error("Serial execution signal does not carry the executing commitment");
      }
      const certificates = Object.values(hierarchy.rollout_certificates).filter(
        (candidate) => candidate.status === "active"
          && candidate.commitment_id === commitment.commitment_id
      );
      if (certificates.length !== 1) {
        throw new Error(
          `Serial execution requires one active rollout certificate; found ${certificates.length}`
        );
      }
      const certificate = certificates[0]!;
      const execution = ResolvedExecutionSchema.parse({
        kind: "execute_plan",
        planning_action: certificate.planning_action,
        planning_transaction_id: certificate.planning_transaction_id
      });
      const resolvedSourceInput = {
        ...params,
        source_signal_ids: [directCommitmentSignals[0]!.signal_id],
        rollout_certificate_id: certificate.certificate_id,
        execution
      };
      const lease = await runtime.issueNeuralAuthorityLease({
        issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
        targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
        allowedSignalKinds: ["skill_commitment", "motor_intent", "rollout_result"],
        ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
        ttlMs: 180_000,
        invocationId: executorInvocation.invocationId,
        parentInvocationId: executorInvocation.parentInvocationId,
        parentEpisodeId: requiredParentEpisodeId(
          HUMANOID_NEURAL_AGENT_IDS.executionDispatcher
        )
      });
      const selected = executionAction(execution);
      try {
        const executionTransactionId = details?.toolCall?.callId;
        if (!executionTransactionId) {
          throw new Error("Serial Executor requires an SDK tool call identity");
        }
        const admission = await runtime.publishNeuralSignal({
          kind: "skill_commitment",
          pathway: "physical_execution",
          direction: "descending",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 100,
          causalParentIds: [
            directCommitmentSignals[0]!.signal_id,
            certificate.predictive_signal_id,
            certificate.rollout_signal_id
          ],
          authorityLeaseId: lease.lease_id,
          invocationId: lease.invocation_id,
          parentInvocationId: lease.parent_invocation_id,
          payload: commitment
        });
        const lowerLoop = await executeCertifiedLowerMotorLoop({
          runtime,
          sourceToolName: name,
          sourceInput: resolvedSourceInput,
          selected,
          certificate,
          commitment,
          admission,
          executionTransactionId,
          ...(details ? { details } : {})
        });
        const output = lowerLoop.output;
        const receiptPayload = z.json().parse(JSON.parse(output));
        const feedback = await publishExecutorPhysicalFeedback({
          runtime,
          receiptPayload,
          executionTransactionId,
          executorLease: lease,
          lowerLoop
        });
        const outputRecord = outputObject(output);
        if (!outputRecord) {
          throw new Error("Serial Executor returned a non-object physical receipt");
        }
        return JSON.stringify({
          ...outputRecord,
          source_signal_ids: feedback.sourceSignalIds
        });
      } finally {
        await runtime.closeNeuralAuthorityLease({
          leaseId: lease.lease_id,
          closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
          reason: "serial_execution_returned"
        });
      }
      }, typeof details?.resumeState === "string", stableAgentToolInvocationId(
        HUMANOID_NEURAL_AGENT_IDS.executor,
        details?.toolCall?.callId
      ));
    })
  });
}

function recoveryAuthorityTool(
  runtime: HumanoidNeuralAgentRuntime,
  recovery: Agent<any, any>,
  sessions: ReadonlyMap<string, Session>,
  outer: {
    runtime: HumanoidNeuralAgentRuntime;
    callModelInputFilter: CallModelInputFilter;
    onAgentStream?: (agentId: string, event: RunStreamEvent) => void | Promise<void>;
  }
): FunctionTool<unknown, typeof EmptyDelegationSchema, unknown> {
  const runner = new Runner({
    tracingDisabled: true,
    traceIncludeSensitiveData: false,
    callModelInputFilter: outer.callModelInputFilter,
    toolExecution: { maxFunctionToolConcurrency: 1 },
    toolNotFoundBehavior: "return_error_to_model",
    reasoningItemIdPolicy: "omit",
    modelSettings: { parallelToolCalls: false },
    workflowName: "HEAR bounded recovery authority lease"
  });
  const recoveryId = HUMANOID_NEURAL_AGENT_IDS.recovery;
  return tool({
    name: humanoidNeuralAgentToolName("recovery"),
    description: "Suspend ordinary sensorimotor selection and run one independent recovery SDK episode under a bounded authority lease.",
    parameters: EmptyDelegationSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => ["motor_assessment", "rollout_review", "recovery"].includes(
      runtime.neuralHarnessPhase().phase
    ) && hasCurrentRecoveryDemand(runtime),
    execute: async (_params, _context, details) => withAgentInvocation(
      recoveryId,
      async () => {
        const recoveryInvocation = requiredHarnessInvocation(recoveryId);
        if (runtime.neuralHarnessPhase().phase !== "recovery") {
          await runtime.transitionNeuralHarnessPhase({
            phase: "recovery",
            enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            reason: "sensorimotor_granted_recovery_decision_domain"
          });
        }
        const recoveryDemands = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          kinds: ["prediction_error", "skill_failed", "escalation"]
        });
        const relevant = recoveryDemands.filter((signal) => (
          signal.direction === "descending"
            && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
            && signal.invocation_id === recoveryInvocation.parentInvocationId
        )).sort((left, right) => (
          right.world_revision - left.world_revision
            || right.sequence - left.sequence
        ))[0];
        const currentBelief = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          kinds: ["perceptual_belief"]
        }).filter((signal) => (
          signal.direction === "descending"
            && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
            && signal.invocation_id === recoveryInvocation.parentInvocationId
            && isCurrentNeuralSignal(runtime, signal)
        )).sort((left, right) => (
          right.world_revision - left.world_revision
            || right.sequence - left.sequence
        ))[0];
        const hierarchyState = runtime.neuralHierarchyState();
        const stationarySafetyInterrupt = runtime.pendingNeuralSafetyInterrupts()
          .find((interrupt) => interrupt.kind === "stationary_fall"
            && interrupt.status === "acknowledged"
            && currentBelief !== undefined
            && currentBelief.world_revision >= interrupt.world_revision);
        const recoveryReadiness = runtime.autonomyReadiness();
        const durableFailureEvidence = runtime.recoveryFailureEvidence();
        const freshDurableFailureBelief = !relevant
          && currentBelief
          && recoveryReadiness === "replan_or_retire"
          && !neuralSkillCommitmentIsOpen(hierarchyState.active_skill_commitment)
          ? currentBelief
          : undefined;
        const recoveryRoot = relevant ?? freshDurableFailureBelief;
        if (!recoveryRoot) {
          throw new Error(
            "Recovery requires failure feedback routed through the current Action Selection -> Sensorimotor episode"
          );
        }
        // Once durable action state says the failed/rejected attempt has been
        // observed, every recovery kind must carry that newest belief. A
        // process-resumed structural edge may be prediction_error even when
        // the durable root failure was physical; checking only skill_failed
        // would silently drop the post-failure world state on resume.
        const postFailureBelief = currentBelief;
        if (recoveryReadiness === "replan_or_retire" && !postFailureBelief) {
          throw new Error(
            "Durable Recovery requires a current causally bound post-failure perceptual belief"
          );
        }
        if (recoveryReadiness === "replan_or_retire"
          && durableFailureEvidence === null) {
          throw new Error(
            "Durable Recovery requires the current Cycle failure receipt"
          );
        }
        const durableObservation = jsonRecord(
          jsonRecord(durableFailureEvidence)?.post_failure_observation_receipt
        );
        const durableObservationRevision = durableObservation
          ?.world_after_revision;
        if (recoveryReadiness === "replan_or_retire"
          && (typeof durableObservationRevision !== "number"
            || !Number.isSafeInteger(durableObservationRevision)
            || !postFailureBelief
            || postFailureBelief.world_revision < durableObservationRevision)) {
          throw new Error(
            "Durable Recovery perceptual belief predates its post-failure observation receipt"
          );
        }
        const suspendLeaseIds = Object.values(hierarchyState.authority_leases)
          .filter((lease) => (
            lease.status === "active"
              && lease.issuing_parent_node_id
                === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
              && lease.target_child_node_id !== recoveryId
          ))
          .map((lease) => lease.lease_id);
        const lease = await runtime.issueNeuralAuthorityLease({
          issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          targetChildNodeId: recoveryId,
          allowedSignalKinds: [recoveryRoot.kind],
          correctionScope: "pathway",
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          ttlMs: 120_000,
          exclusive: true,
          suspendLeaseIds,
          invocationId: recoveryInvocation.invocationId,
          parentInvocationId: recoveryInvocation.parentInvocationId,
          parentEpisodeId: requiredParentEpisodeId(
            HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
          )
        });
        try {
          const descending = await runtime.publishNeuralSignal({
            kind: recoveryRoot.kind,
            pathway: "interoceptive_risk",
            direction: "descending",
            sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            targetNodeId: recoveryId,
            ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
            priority: 100,
            causalParentIds: [
              recoveryRoot.signal_id,
              ...(postFailureBelief
                && postFailureBelief.signal_id !== recoveryRoot.signal_id
                ? [postFailureBelief.signal_id]
                : [])
            ],
            authorityLeaseId: lease.lease_id,
            invocationId: recoveryInvocation.invocationId,
            parentInvocationId: recoveryInvocation.parentInvocationId,
            payload: stationarySafetyInterrupt && postFailureBelief
              ? {
                  recovery_basis:
                    "stationary_safety_interrupt_and_post_failure_observation",
                  autonomy_readiness: recoveryReadiness,
                  failure: relevant?.payload ?? null,
                  durable_failure_evidence: durableFailureEvidence,
                  safety_interrupt: stationarySafetyInterrupt,
                  post_failure_belief: postFailureBelief.payload
                }
              : freshDurableFailureBelief
                ? {
                    recovery_basis:
                      "durable_failure_receipt_and_post_failure_observation",
                    autonomy_readiness: "replan_or_retire",
                    durable_failure_evidence: durableFailureEvidence,
                    post_failure_belief: freshDurableFailureBelief.payload
                  }
                : postFailureBelief
                  ? {
                      failure: recoveryRoot.payload,
                      ...(durableFailureEvidence === null
                        ? {}
                        : { durable_failure_evidence: durableFailureEvidence }),
                      post_failure_belief: postFailureBelief.payload
                    }
                  : recoveryRoot.payload
          });
          const runOptions = {
            session: requiredSession(sessions, "recovery"),
            maxTurns: null,
            reasoningItemIdPolicy: "omit" as const,
            toolExecution: { maxFunctionToolConcurrency: 1 },
            ...(details?.signal ? { signal: details.signal } : {})
          };
          let turnInput = neuralInvocationInput(
            outer.runtime,
            HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            recoveryId,
            recoveryInvocation.invocationId
          );
          let validatedOutput: z.infer<typeof RecoveryOutputSchema> | undefined;
          for (;;) {
            let finalOutput: unknown;
            if (!outer.onAgentStream) {
              const result = await runner.run(recovery, turnInput, runOptions);
              finalOutput = result.finalOutput;
            } else {
              const stream = await runner.run(
                recovery,
                turnInput,
                { ...runOptions, stream: true as const }
              );
              for await (const event of stream) {
                await outer.onAgentStream(recoveryId, event);
              }
              await stream.completed;
              finalOutput = stream.finalOutput;
            }
            const candidate = parseNeuralAgentFinalOutput(
              RecoveryOutputSchema,
              finalOutput
            );
            if (candidate.success) {
              validatedOutput = candidate.data;
              break;
            }
            // Recovery owns an exclusive bounded decision domain. A prose-only
            // SDK turn has no authority, so continue this same child episode
            // locally instead of returning a correction receipt that makes
            // Sensorimotor rebuild and republish the recovery edge.
            turnInput = neuralAgentToolTurnContinuationInput(
              outer.runtime,
              HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
              recoveryId,
              recoveryInvocation.invocationId
            );
          }
          const parsed = parseNeuralAgentOutput(validatedOutput);
          const recoveryResult = await runtime.publishNeuralSignal({
            kind: parsed.signal_kind,
            pathway: "interoceptive_risk",
            direction: "ascending",
            sourceNodeId: recoveryId,
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
            priority: 100,
            causalParentIds: [...new Set([
              descending.signal_id,
              ...parsed.source_signal_ids
            ])],
            sourceAuthorityLeaseId: lease.lease_id,
            invocationId: recoveryInvocation.invocationId,
            parentInvocationId: recoveryInvocation.parentInvocationId,
            payload: parsed.payload
          });
          // Recovery owns only the descending signal created for this bounded
          // child episode.  The enclosing Sensorimotor Agent.asTool invocation
          // owns its Action Selection inputs and consumes them after the
          // ascending result is published.  Consuming the earlier snapshot of
          // every pending recovery demand here crossed invocation boundaries:
          // an old Premotor escalation could expire during the model call and
          // then abort an otherwise valid Recovery result on return.
          await runtime.consumeNeuralSignals(recoveryId, [descending.signal_id]);
          await runtime.transitionNeuralHarnessPhase({
            phase: parsed.signal_kind === "escalation"
              ? "goal_valuation"
              : "skill_proposal",
            enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            reason: parsed.signal_kind === "escalation"
              ? "recovery_escalated_to_supervisory_control"
              : "recovery_proposal_returned_to_owner"
          });
          return JSON.stringify({
            ...parsed,
            source_signal_ids: [recoveryResult.signal_id]
          });
        } finally {
          await runtime.closeNeuralAuthorityLease({
            leaseId: lease.lease_id,
            closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            reason: "bounded_recovery_episode_returned",
            resumeSuspended: true
          });
        }
      },
      typeof details?.resumeState === "string",
      stableAgentToolInvocationId(recoveryId, details?.toolCall?.callId)
    )
  });
}

async function executeCertifiedLowerMotorLoop(input: {
  runtime: HumanoidNeuralAgentRuntime;
  sourceToolName: string;
  sourceInput: unknown;
  selected: ReturnType<typeof executionAction>;
  certificate: NeuralRolloutCertificate;
  commitment: NeuralSkillCommitment;
  admission: NeuralSignal;
  executionTransactionId: string;
  details?: {
    toolCall?: {
      callId?: string;
      name?: string;
      arguments: string;
    };
    signal?: AbortSignal;
    resumeState?: unknown;
  };
}): Promise<{
  output: string;
  executionReceiptSignal: NeuralSignal;
  predictionErrorSignal?: NeuralSignal;
  completionSignal: NeuralSignal;
  summary: JsonValue;
}> {
  return withAgentInvocation(
    HUMANOID_NEURAL_AGENT_IDS.reflex,
    async () => {
      const reflexInvocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS.reflex
      );
      const reflexLease = await input.runtime.issueNeuralAuthorityLease({
        issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
        targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
        allowedSignalKinds: ["skill_commitment", "motor_intent"],
        ttlMs: 600_000,
        invocationId: reflexInvocation.invocationId,
        parentInvocationId: reflexInvocation.parentInvocationId,
        parentEpisodeId: requiredParentEpisodeId(
          HUMANOID_NEURAL_AGENT_IDS.executor
        )
      });
      let motorIntent: NeuralSignal | undefined;
      try {
        motorIntent = await input.runtime.publishNeuralSignal({
          kind: "motor_intent",
          pathway: "physical_execution",
          direction: "descending",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 100,
          causalParentIds: [input.admission.signal_id],
          authorityLeaseId: reflexLease.lease_id,
          invocationId: reflexLease.invocation_id,
          parentInvocationId: reflexLease.parent_invocation_id,
          payload: {
            protocol: "certified-motor-intent-v1",
            commitment_id: input.commitment.commitment_id,
            planning_transaction_id: input.certificate.planning_transaction_id,
            planning_action: input.certificate.planning_action,
            rollout_certificate_id: input.certificate.certificate_id,
            execution_transaction_id: input.executionTransactionId,
            action: input.selected.action,
            parameters: input.selected.input
          }
        });

        const bodyResult = await withAgentInvocation(
          HUMANOID_NEURAL_AGENT_IDS.body,
          async () => {
            const bodyInvocation = requiredHarnessInvocation(
              HUMANOID_NEURAL_AGENT_IDS.body
            );
            const bodyLease = await input.runtime.issueNeuralAuthorityLease({
              issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
              targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
              allowedSignalKinds: ["motor_intent"],
              ttlMs: 600_000,
              invocationId: bodyInvocation.invocationId,
              parentInvocationId: bodyInvocation.parentInvocationId,
              parentEpisodeId: requiredParentEpisodeId(
                HUMANOID_NEURAL_AGENT_IDS.reflex
              )
            });
            let bodyIntent: NeuralSignal | undefined;
            try {
              bodyIntent = await input.runtime.publishNeuralSignal({
                kind: "motor_intent",
                pathway: "physical_execution",
                direction: "descending",
                sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
                targetNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
                ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
                priority: 100,
                causalParentIds: [motorIntent!.signal_id],
                authorityLeaseId: bodyLease.lease_id,
                invocationId: bodyLease.invocation_id,
                parentInvocationId: bodyLease.parent_invocation_id,
                payload: {
                  protocol: "controller-reference-command-v1",
                  commitment_id: input.commitment.commitment_id,
                  execution_transaction_id: input.executionTransactionId,
                  action: input.selected.action,
                  parameters: input.selected.input,
                  controller_contract: "learned_reference_controller_reflex_loop_v1"
                }
              });
              let physicalReceipt: HumanoidActionReceipt | undefined;
              const output = await invokeDeterministicHumanoidAction({
                runtime: input.runtime,
                // Serial Executor retains the sole physical write authority;
                // Body is the plant being advanced inside that transaction.
                actorAgentId: HUMANOID_NEURAL_AGENT_IDS.executor,
                sourceToolName: input.sourceToolName,
                sourceInput: input.sourceInput,
                action: input.selected.action,
                actionInput: input.selected.input,
                contractId: "execution_gate_v1",
                neuralRolloutCertificate: {
                  certificate_id: input.certificate.certificate_id,
                  commitment_id: input.certificate.commitment_id,
                  goal_epoch_id: input.certificate.goal_epoch_id,
                  planning_transaction_id: input.certificate.planning_transaction_id,
                  planning_action: input.certificate.planning_action,
                  rollout_signal_id: input.certificate.rollout_signal_id,
                  predictive_signal_id: input.certificate.predictive_signal_id,
                  rollout_invocation_id: input.certificate.rollout_invocation_id,
                  predictive_invocation_id: input.certificate.predictive_invocation_id,
                  rollout_payload_sha256: input.certificate.rollout_payload_sha256
                },
                ...(input.details ? { details: input.details } : {}),
                onReceipt: (receipt) => {
                  physicalReceipt = receipt;
                }
              });
              if (!physicalReceipt) {
                throw new Error("MuJoCo Body returned no authoritative physical receipt");
              }
              return {
                output,
                physicalReceipt,
                lowerLoop: await publishCertifiedLowerMotorFeedback({
                  runtime: input.runtime,
                  physicalReceipt,
                  commitment: input.commitment,
                  executionTransactionId: input.executionTransactionId,
                  motorIntent: motorIntent!,
                  bodyIntent,
                  bodyLease,
                  reflexLease
                })
              };
            } finally {
              await input.runtime.closeNeuralAuthorityLease({
                leaseId: bodyLease.lease_id,
                closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
                reason: bodyIntent
                  ? "mujoco_body_transaction_returned"
                  : "body_motor_intent_publication_failed"
              });
            }
          },
          false,
          stableAgentToolInvocationId(
            HUMANOID_NEURAL_AGENT_IDS.body,
            input.executionTransactionId
          )
        );
        return {
          output: bodyResult.output,
          ...bodyResult.lowerLoop
        };
      } finally {
        await input.runtime.closeNeuralAuthorityLease({
          leaseId: reflexLease.lease_id,
          closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
          reason: motorIntent
            ? "controller_reflex_loop_returned"
            : "reflex_motor_intent_publication_failed"
        });
      }
    },
    false,
    stableAgentToolInvocationId(
      HUMANOID_NEURAL_AGENT_IDS.reflex,
      input.executionTransactionId
    )
  );
}

interface CertifiedLowerMotorFeedback {
  executionReceiptSignal: NeuralSignal;
  predictionErrorSignal?: NeuralSignal;
  completionSignal: NeuralSignal;
  summary: JsonValue;
}

async function publishCertifiedLowerMotorFeedback(input: {
  runtime: HumanoidNeuralAgentRuntime;
  physicalReceipt: HumanoidActionReceipt;
  commitment: NeuralSkillCommitment;
  executionTransactionId: string;
  motorIntent: NeuralSignal;
  bodyIntent: NeuralSignal;
  bodyLease: NeuralAuthorityLease;
  reflexLease: NeuralAuthorityLease;
}): Promise<CertifiedLowerMotorFeedback> {
  const sensation = lowerMotorSensation(input.physicalReceipt);
  let sensorySignal = executionFeedbackSignal(input.runtime, {
    kind: "sensory_evidence",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
    executionTransactionId: input.executionTransactionId
  });
  if (sensorySignal) {
    if (!sensorySignal.causal_parent_ids.includes(input.bodyIntent.signal_id)
      || modelPayloadSha256(sensorySignal.payload) !== modelPayloadSha256(sensation)) {
      throw new Error(
        `Recovered Body feedback conflicts with physical receipt ${input.executionTransactionId}`
      );
    }
  } else {
    sensorySignal = await input.runtime.publishNeuralSignal({
      kind: "sensory_evidence",
      pathway: "ascending_feedback",
      direction: "ascending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
      ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
      priority: 95,
      causalParentIds: [input.bodyIntent.signal_id],
      sourceAuthorityLeaseId: input.bodyLease.lease_id,
      invocationId: input.bodyLease.invocation_id,
      parentInvocationId: input.bodyLease.parent_invocation_id,
      payload: sensation
    });
  }

  let bodyPredictionError: NeuralSignal | undefined;
  if (!physicalExecutionSucceeded(input.physicalReceipt)) {
    bodyPredictionError = executionFeedbackSignal(input.runtime, {
      kind: "prediction_error",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
      causalParentSignalId: sensorySignal.signal_id
    });
    bodyPredictionError ??= await input.runtime.publishNeuralSignal({
      kind: "prediction_error",
      pathway: "ascending_feedback",
      direction: "ascending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
      ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
      priority: 100,
      causalParentIds: [sensorySignal.signal_id],
      sourceAuthorityLeaseId: input.bodyLease.lease_id,
      invocationId: input.bodyLease.invocation_id,
      parentInvocationId: input.bodyLease.parent_invocation_id,
      payload: lowerMotorPredictionError(input.physicalReceipt)
    });
  }

  let reflexReceipt = executionFeedbackSignal(input.runtime, {
    kind: "execution_receipt",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
    executionTransactionId: input.executionTransactionId
  });
  reflexReceipt ??= await input.runtime.publishNeuralSignal({
    kind: "execution_receipt",
    pathway: "ascending_feedback",
    direction: "ascending",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
    ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
    priority: 95,
    causalParentIds: [sensorySignal.signal_id],
    sourceAuthorityLeaseId: input.reflexLease.lease_id,
    invocationId: input.reflexLease.invocation_id,
    parentInvocationId: input.reflexLease.parent_invocation_id,
    payload: {
      protocol: "reflex-execution-receipt-v1",
      commitment_id: input.commitment.commitment_id,
      execution_transaction_id: input.executionTransactionId,
      controller: sensation.controller,
      body_signal_id: sensorySignal.signal_id,
      physical: sensation
    }
  });

  let reflexPredictionError: NeuralSignal | undefined;
  if (bodyPredictionError) {
    const recordedErrors = input.runtime.neuralHierarchyState().prediction_errors.filter(
      (candidate) => candidate.observer_node_id === HUMANOID_NEURAL_AGENT_IDS.reflex
        && candidate.source_signal_id === bodyPredictionError!.signal_id
    );
    if (recordedErrors.length > 1) {
      throw new Error(
        `Physical transaction has duplicate Reflex prediction errors: ${input.executionTransactionId}`
      );
    }
    const error = recordedErrors[0] ?? await input.runtime.recordNeuralPredictionError({
      observerNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
      sourceSignalId: bodyPredictionError.signal_id,
      magnitude: lowerMotorPredictionErrorMagnitude(sensation.reflex_arc),
      tolerance: 0.2,
      correctionScope: "local",
      detail: bodyPredictionError.payload
    });
    reflexPredictionError = executionFeedbackSignal(input.runtime, {
      kind: "prediction_error",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
      causalParentSignalId: bodyPredictionError.signal_id
    });
    reflexPredictionError ??= await input.runtime.publishNeuralSignal({
      kind: "prediction_error",
      pathway: "ascending_feedback",
      direction: "ascending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
      ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
      priority: 100,
      causalParentIds: [bodyPredictionError.signal_id],
      sourceAuthorityLeaseId: input.reflexLease.lease_id,
      invocationId: input.reflexLease.invocation_id,
      parentInvocationId: input.reflexLease.parent_invocation_id,
      payload: {
        protocol: "reflex-prediction-error-v1",
        error,
        physical: sensation
      }
    });
  }

  const completed = physicalExecutionSucceeded(input.physicalReceipt);
  let completionSignal = executionFeedbackSignal(input.runtime, {
    kind: completed ? "skill_completed" : "skill_failed",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
    executionTransactionId: input.executionTransactionId
  });
  completionSignal ??= await input.runtime.publishNeuralSignal({
    kind: completed ? "skill_completed" : "skill_failed",
    pathway: "ascending_feedback",
    direction: "ascending",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
    ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
    priority: 100,
    causalParentIds: [...new Set([
      reflexReceipt.signal_id,
      ...(reflexPredictionError ? [reflexPredictionError.signal_id] : [])
    ])],
    sourceAuthorityLeaseId: input.reflexLease.lease_id,
    invocationId: input.reflexLease.invocation_id,
    parentInvocationId: input.reflexLease.parent_invocation_id,
    payload: {
      protocol: "lower-motor-loop-completion-v1",
      commitment_id: input.commitment.commitment_id,
      execution_transaction_id: input.executionTransactionId,
      accepted: input.physicalReceipt.accepted,
      code: input.physicalReceipt.code,
      body_signal_id: sensorySignal.signal_id,
      reflex_receipt_signal_id: reflexReceipt.signal_id,
      reflex_arc: sensation.reflex_arc
    }
  });
  return {
    executionReceiptSignal: reflexReceipt,
    ...(reflexPredictionError ? { predictionErrorSignal: reflexPredictionError } : {}),
    completionSignal,
    summary: {
      protocol: "certified-lower-motor-loop-v1",
      motor_intent_signal_id: input.motorIntent.signal_id,
      body_sensory_signal_id: sensorySignal.signal_id,
      reflex_execution_signal_id: reflexReceipt.signal_id,
      completion_signal_id: completionSignal.signal_id,
      ...(reflexPredictionError
        ? { prediction_error_signal_id: reflexPredictionError.signal_id }
        : {}),
      controller: sensation.controller,
      reflex_arc: sensation.reflex_arc
    }
  };
}

async function publishExecutorPhysicalFeedback(input: {
  runtime: HumanoidNeuralAgentRuntime;
  receiptPayload: JsonValue;
  executionTransactionId: string;
  executorLease: NeuralAuthorityLease;
  lowerLoop: CertifiedLowerMotorFeedback;
}): Promise<{
  executionReceiptSignal: NeuralSignal;
  predictionErrorSignal?: NeuralSignal;
  completionSignal: NeuralSignal;
  sourceSignalIds: string[];
}> {
  let executionReceipt = executionFeedbackSignal(input.runtime, {
    kind: "execution_receipt",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
    executionTransactionId: input.executionTransactionId
  });
  executionReceipt ??= await input.runtime.publishNeuralSignal({
    kind: "execution_receipt",
    pathway: "physical_execution",
    direction: "ascending",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
    ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
    priority: 90,
    causalParentIds: [input.lowerLoop.executionReceiptSignal.signal_id],
    sourceAuthorityLeaseId: input.executorLease.lease_id,
    invocationId: input.executorLease.invocation_id,
    parentInvocationId: input.executorLease.parent_invocation_id,
    payload: {
      ...(jsonRecord(input.receiptPayload) ?? {}),
      execution_transaction_id: input.executionTransactionId,
      lower_motor_loop: input.lowerLoop.summary
    }
  });

  let executionPredictionError: NeuralSignal | undefined;
  if (input.lowerLoop.predictionErrorSignal) {
    executionPredictionError = executionFeedbackSignal(input.runtime, {
      kind: "prediction_error",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
      causalParentSignalId: input.lowerLoop.predictionErrorSignal.signal_id
    });
    executionPredictionError ??= await input.runtime.publishNeuralSignal({
      kind: "prediction_error",
      pathway: "ascending_feedback",
      direction: "ascending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
      ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
      priority: 100,
      causalParentIds: [input.lowerLoop.predictionErrorSignal.signal_id],
      sourceAuthorityLeaseId: input.executorLease.lease_id,
      invocationId: input.executorLease.invocation_id,
      parentInvocationId: input.executorLease.parent_invocation_id,
      payload: input.lowerLoop.predictionErrorSignal.payload
    });
  }
  const skillCompleted = input.lowerLoop.completionSignal.kind === "skill_completed";
  let completionSignal = executionFeedbackSignal(input.runtime, {
    kind: skillCompleted ? "skill_completed" : "skill_failed",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
    executionTransactionId: input.executionTransactionId
  });
  completionSignal ??= await input.runtime.publishNeuralSignal({
    kind: skillCompleted ? "skill_completed" : "skill_failed",
    pathway: "physical_execution",
    direction: "ascending",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
    ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
    priority: 95,
    causalParentIds: [...new Set([
      input.lowerLoop.completionSignal.signal_id,
      executionReceipt.signal_id,
      ...(executionPredictionError ? [executionPredictionError.signal_id] : [])
    ])],
    sourceAuthorityLeaseId: input.executorLease.lease_id,
    invocationId: input.executorLease.invocation_id,
    parentInvocationId: input.executorLease.parent_invocation_id,
    payload: input.receiptPayload
  });

  const phase = input.runtime.neuralHarnessPhase().phase;
  if (phase === "execution") {
    await input.runtime.transitionNeuralHarnessPhase({
      // A physical failure first returns through Action Selection so it can
      // close the executing commitment. Recovery starts only after a fresh
      // Sensor Fusion observation is causally bound to that failure.
      phase: "feedback",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
      reason: skillCompleted
        ? "physical_execution_completed"
        : "physical_execution_failed_requires_commitment_resolution"
    });
  } else if (phase !== "feedback") {
    throw new Error(`Physical feedback cannot enter Harness phase ${phase}`);
  }
  return {
    executionReceiptSignal: executionReceipt,
    ...(executionPredictionError
      ? { predictionErrorSignal: executionPredictionError }
      : {}),
    completionSignal,
    sourceSignalIds: [
      executionReceipt.signal_id,
      ...(executionPredictionError ? [executionPredictionError.signal_id] : []),
      completionSignal.signal_id
    ]
  };
}

function executionFeedbackSignal(
  runtime: HumanoidNeuralAgentRuntime,
  input: {
    kind: NeuralSignalKind;
    sourceNodeId: string;
    targetNodeId: string;
    executionTransactionId?: string;
    causalParentSignalId?: string;
  }
): NeuralSignal | undefined {
  const matches = Object.values(runtime.neuralHierarchyState().signals).filter(
    (signal) => signal.kind === input.kind
      && signal.source_node_id === input.sourceNodeId
      && signal.target_node_id === input.targetNodeId
      && (input.executionTransactionId === undefined
        || neuralFeedbackTransactionId(signal.payload)
          === input.executionTransactionId)
      && (input.causalParentSignalId === undefined
        || signal.causal_parent_ids.includes(input.causalParentSignalId))
  );
  if (matches.length > 1) {
    throw new Error(
      `Physical transaction has duplicate ${input.kind} feedback on ${input.sourceNodeId} -> ${input.targetNodeId}`
    );
  }
  return matches[0];
}

function neuralFeedbackTransactionId(payload: JsonValue): string | undefined {
  const record = jsonRecord(payload);
  for (const key of [
    "execution_transaction_id",
    "transaction_id",
    "transactionId"
  ] as const) {
    if (typeof record?.[key] === "string") return record[key];
  }
  const physical = jsonRecord(record?.physical);
  return typeof physical?.transaction_id === "string"
    ? physical.transaction_id
    : undefined;
}

/**
 * Finish the neural return path for a physical transaction that committed
 * after its owning SDK episode was interrupted. The physical receipt and the
 * original descending motor-intent chain remain the authority; this recovery
 * episode only republishes missing deterministic feedback edges.
 */
export async function recoverCommittedNeuralPhysicalExecutionFeedback(
  runtime: HumanoidNeuralAgentRuntime,
  receipt: HumanoidActionReceipt
): Promise<boolean> {
  const state = runtime.neuralHierarchyState();
  const commitment = state.active_skill_commitment;
  if (!commitment || commitment.state !== "executing") return false;
  const reflexArc = state.reflex_arc;
  if (reflexArc.execution_transaction_id !== receipt.transactionId) return false;
  if (receipt.agentId !== HUMANOID_NEURAL_AGENT_IDS.executor
    || reflexArc.commitment_id !== commitment.commitment_id
    || reflexArc.status === "idle"
    || reflexArc.status === "active"
    || reflexArc.terminal_code !== receipt.code
    || (reflexArc.status === "succeeded") !== physicalExecutionSucceeded(receipt)) {
    throw new Error(
      `Committed physical receipt conflicts with terminal Reflex arc ${receipt.transactionId}`
    );
  }
  // lowerMotorSensation validates that the durable receipt itself carries the
  // same transaction-bound controller/reflex summary recorded by the runtime.
  lowerMotorSensation(receipt);

  const completedKind = physicalExecutionSucceeded(receipt)
    ? "skill_completed"
    : "skill_failed";
  const existingCompletion = executionFeedbackSignal(runtime, {
    kind: completedKind,
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    executionTransactionId: receipt.transactionId
  });
  if (existingCompletion) return true;

  const certificateMatches = Object.values(state.rollout_certificates).filter(
    (certificate) => certificate.commitment_id === commitment.commitment_id
      && certificate.execution_transaction_id === receipt.transactionId
      && certificate.status === "consumed"
  );
  if (certificateMatches.length !== 1) {
    throw new Error(
      `Recovered physical transaction requires one consumed rollout certificate: ${receipt.transactionId}`
    );
  }
  const certificate = certificateMatches[0]!;
  const bodyIntents = Object.values(state.signals).filter((signal) => {
    const payload = jsonRecord(signal.payload);
    return signal.kind === "motor_intent"
      && signal.direction === "descending"
      && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.reflex
      && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.body
      && payload?.execution_transaction_id === receipt.transactionId
      && payload.commitment_id === commitment.commitment_id;
  });
  if (bodyIntents.length !== 1) {
    throw new Error(
      `Recovered physical transaction requires one durable Body motor intent: ${receipt.transactionId}`
    );
  }
  const bodyIntent = bodyIntents[0]!;
  if (bodyIntent.causal_parent_ids.length !== 1) {
    throw new Error("Recovered Body motor intent has no unique Reflex parent");
  }
  const motorIntent = state.signals[bodyIntent.causal_parent_ids[0]!];
  const motorPayload = motorIntent ? jsonRecord(motorIntent.payload) : undefined;
  if (!motorIntent
    || motorIntent.kind !== "motor_intent"
    || motorIntent.direction !== "descending"
    || motorIntent.source_node_id !== HUMANOID_NEURAL_AGENT_IDS.executor
    || motorIntent.target_node_id !== HUMANOID_NEURAL_AGENT_IDS.reflex
    || motorPayload?.execution_transaction_id !== receipt.transactionId
    || motorPayload.commitment_id !== commitment.commitment_id
    || motorPayload.rollout_certificate_id !== certificate.certificate_id
    || motorIntent.causal_parent_ids.length !== 1) {
    throw new Error(
      `Recovered physical transaction has no certified Executor motor intent: ${receipt.transactionId}`
    );
  }
  const admission = state.signals[motorIntent.causal_parent_ids[0]!];
  const admittedCommitment = admission
    ? NeuralSkillCommitmentSchema.safeParse(admission.payload)
    : undefined;
  if (!admission
    || admission.kind !== "skill_commitment"
    || admission.direction !== "descending"
    || admission.source_node_id !== HUMANOID_NEURAL_AGENT_IDS.executionDispatcher
    || admission.target_node_id !== HUMANOID_NEURAL_AGENT_IDS.executor
    || !admittedCommitment?.success
    || admittedCommitment.data.commitment_id !== commitment.commitment_id
    || admittedCommitment.data.state !== "executing") {
    throw new Error(
      `Recovered physical transaction has no Dispatcher execution admission: ${receipt.transactionId}`
    );
  }
  const dispatcherAdmissions = admission.causal_parent_ids.flatMap((signalId) => {
    const signal = state.signals[signalId];
    const parsed = signal
      ? NeuralSkillCommitmentSchema.safeParse(signal.payload)
      : undefined;
    return signal
      && signal.kind === "skill_commitment"
      && signal.direction === "descending"
      && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
      && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.executionDispatcher
      && parsed?.success
      && parsed.data.commitment_id === commitment.commitment_id
      ? [signal]
      : [];
  });
  if (dispatcherAdmissions.length !== 1) {
    throw new Error(
      `Recovered physical transaction has no unique Sensorimotor Dispatcher admission: ${receipt.transactionId}`
    );
  }

  // The Body and Reflex invocations that owned a physical transaction are
  // process-local lexical scopes. If the process stops while MuJoCo is still
  // executing, their durable leases remain active even though those scopes can
  // never unwind their finally blocks. Physical recovery deliberately creates
  // a new deterministic return episode below, so retire only the two leases
  // proven by this transaction's original motor-intent chain before opening
  // that replacement path. Other Agent leases remain available for normal SDK
  // RunState/Session recovery.
  await revokeDetachedPhysicalExecutionLease({
    runtime,
    signal: bodyIntent,
    issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
    targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
    executionTransactionId: receipt.transactionId
  });
  await revokeDetachedPhysicalExecutionLease({
    runtime,
    signal: motorIntent,
    issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
    targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
    executionTransactionId: receipt.transactionId
  });

  const receiptPayload = z.json().parse(JSON.parse(
    humanoidActionReceiptModelOutput(receipt)
  ));
  const recoveryKey = `physical-feedback-recovery:${receipt.transactionId}`;
  const rootInvocationId = stableAgentToolInvocationId(
    HUMANOID_NEURAL_AGENT_IDS.executive,
    recoveryKey
  );
  await withAgentInvocation(
    HUMANOID_NEURAL_AGENT_IDS.executive,
    () => withRecoveredFeedbackLease({
      runtime,
      parentNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      childNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      allowedSignalKinds: ["goal_selected"],
      recoveryKey,
      operation: () => withRecoveredFeedbackLease({
        runtime,
        parentNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        childNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        allowedSignalKinds: ["skill_commitment"],
        recoveryKey,
        operation: (sensorimotorLease) => withRecoveredFeedbackLease({
          runtime,
          parentNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          childNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
          allowedSignalKinds: ["skill_commitment"],
          recoveryKey,
          operation: (dispatcherLease) => withRecoveredFeedbackLease({
            runtime,
            parentNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
            childNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
            allowedSignalKinds: ["skill_commitment"],
            recoveryKey,
            operation: (executorLease) => withRecoveredFeedbackLease({
              runtime,
              parentNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
              childNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
              allowedSignalKinds: ["motor_intent"],
              recoveryKey,
              operation: (reflexLease) => withRecoveredFeedbackLease({
                runtime,
                parentNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
                childNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
                allowedSignalKinds: ["motor_intent"],
                recoveryKey,
                operation: async (bodyLease) => {
                  const lowerLoop = await publishCertifiedLowerMotorFeedback({
                    runtime,
                    physicalReceipt: receipt,
                    commitment,
                    executionTransactionId: receipt.transactionId,
                    motorIntent,
                    bodyIntent,
                    bodyLease,
                    reflexLease
                  });
                  const executorFeedback = await publishExecutorPhysicalFeedback({
                    runtime,
                    receiptPayload,
                    executionTransactionId: receipt.transactionId,
                    executorLease,
                    lowerLoop
                  });
                  const dispatcherCompletion = await publishRecoveredDispatcherCompletion({
                    runtime,
                    receiptPayload,
                    executionTransactionId: receipt.transactionId,
                    dispatcherLease,
                    executorFeedback
                  });
                  await publishRecoveredSensorimotorCompletion({
                    runtime,
                    receiptPayload,
                    executionTransactionId: receipt.transactionId,
                    sensorimotorLease,
                    dispatcherCompletion
                  });
                }
              })
            })
          })
        })
      })
    }),
    true,
    rootInvocationId
  );
  return true;
}

async function revokeDetachedPhysicalExecutionLease(input: {
  runtime: HumanoidNeuralAgentRuntime;
  signal: NeuralSignal;
  issuingParentNodeId: HumanoidNeuralAgentId;
  targetChildNodeId: HumanoidNeuralAgentId;
  executionTransactionId: string;
}): Promise<void> {
  const leaseId = input.signal.authority_lease_id;
  const lease = leaseId === null
    ? undefined
    : input.runtime.neuralHierarchyState().authority_leases[leaseId];
  if (!lease
    || lease.issuing_parent_node_id !== input.issuingParentNodeId
    || lease.target_child_node_id !== input.targetChildNodeId
    || lease.invocation_id !== input.signal.invocation_id
    || lease.parent_invocation_id !== input.signal.parent_invocation_id
    || lease.parent_episode_id !== input.signal.parent_episode_id) {
    throw new Error(
      `Recovered physical transaction has invalid detached neural lease: ${input.executionTransactionId}`
    );
  }
  if (lease.status !== "active" && lease.status !== "suspended") return;
  await input.runtime.closeNeuralAuthorityLease({
    leaseId: lease.lease_id,
    closedByNodeId: input.issuingParentNodeId,
    status: "revoked",
    reason: "physical_execution_invocation_detached_after_restart"
  });
}

async function publishRecoveredSensorimotorCompletion(input: {
  runtime: HumanoidNeuralAgentRuntime;
  receiptPayload: JsonValue;
  executionTransactionId: string;
  sensorimotorLease: NeuralAuthorityLease;
  dispatcherCompletion: NeuralSignal;
}): Promise<NeuralSignal> {
  const kind = input.dispatcherCompletion.kind === "skill_completed"
    ? "skill_completed"
    : "skill_failed";
  const existing = executionFeedbackSignal(input.runtime, {
    kind,
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    executionTransactionId: input.executionTransactionId
  });
  if (existing) return existing;
  return input.runtime.publishNeuralSignal({
    kind,
    pathway: "sensorimotor_selection",
    direction: "ascending",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
    priority: 100,
    causalParentIds: [input.dispatcherCompletion.signal_id],
    sourceAuthorityLeaseId: input.sensorimotorLease.lease_id,
    invocationId: input.sensorimotorLease.invocation_id,
    parentInvocationId: input.sensorimotorLease.parent_invocation_id,
    payload: input.receiptPayload
  });
}

async function publishRecoveredDispatcherCompletion(input: {
  runtime: HumanoidNeuralAgentRuntime;
  receiptPayload: JsonValue;
  executionTransactionId: string;
  dispatcherLease: NeuralAuthorityLease;
  executorFeedback: {
    executionReceiptSignal: NeuralSignal;
    predictionErrorSignal?: NeuralSignal;
    completionSignal: NeuralSignal;
  };
}): Promise<NeuralSignal> {
  const kind = input.executorFeedback.completionSignal.kind === "skill_completed"
    ? "skill_completed"
    : "skill_failed";
  const existing = executionFeedbackSignal(input.runtime, {
    kind,
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
    executionTransactionId: input.executionTransactionId
  });
  if (existing) return existing;
  return input.runtime.publishNeuralSignal({
    kind,
    pathway: "physical_execution",
    direction: "ascending",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executionDispatcher,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
    ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
    priority: 100,
    causalParentIds: [...new Set([
      input.executorFeedback.executionReceiptSignal.signal_id,
      ...(input.executorFeedback.predictionErrorSignal
        ? [input.executorFeedback.predictionErrorSignal.signal_id]
        : []),
      input.executorFeedback.completionSignal.signal_id
    ])],
    sourceAuthorityLeaseId: input.dispatcherLease.lease_id,
    invocationId: input.dispatcherLease.invocation_id,
    parentInvocationId: input.dispatcherLease.parent_invocation_id,
    payload: input.receiptPayload
  });
}

async function withRecoveredFeedbackLease<T>(input: {
  runtime: HumanoidNeuralAgentRuntime;
  parentNodeId: HumanoidNeuralAgentId;
  childNodeId: HumanoidNeuralAgentId;
  allowedSignalKinds: readonly NeuralSignalKind[];
  recoveryKey: string;
  operation: (lease: NeuralAuthorityLease) => Promise<T>;
}): Promise<T> {
  const invocationId = stableAgentToolInvocationId(
    input.childNodeId,
    input.recoveryKey
  );
  return withAgentInvocation(input.childNodeId, async () => {
    const invocation = requiredHarnessInvocation(input.childNodeId);
    const lease = await input.runtime.issueNeuralAuthorityLease({
      issuingParentNodeId: input.parentNodeId,
      targetChildNodeId: input.childNodeId,
      allowedSignalKinds: input.allowedSignalKinds,
      ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
      ttlMs: 120_000,
      invocationId: invocation.invocationId,
      parentInvocationId: invocation.parentInvocationId,
      parentEpisodeId: requiredParentEpisodeId(input.parentNodeId)
    });
    try {
      return await input.operation(lease);
    } finally {
      await input.runtime.closeNeuralAuthorityLease({
        leaseId: lease.lease_id,
        closedByNodeId: input.parentNodeId,
        reason: "recovered_physical_feedback_returned"
      });
    }
  }, true, invocationId);
}

function physicalExecutionSucceeded(receipt: HumanoidActionReceipt): boolean {
  return receipt.accepted && (
    receipt.code === "motion_completed"
      || receipt.code === "navigation_completed"
      || receipt.code === "motion_option_succeeded"
      || receipt.code === "recovery_completed"
  );
}

function lowerMotorSensation(receipt: HumanoidActionReceipt): {
  protocol: "mujoco-body-sensation-v1";
  transaction_id: string;
  action: string;
  accepted: boolean;
  code: string;
  world_before_revision: number;
  world_after_revision: number;
  frame_count: number;
  final: JsonValue;
  trajectory: JsonValue;
  controller: JsonValue;
  reflex_arc: JsonValue;
} {
  const detail = jsonRecord(receipt.detail) ?? {};
  const trajectory = jsonRecord(detail.physical_trajectory) ?? {};
  const controller = trajectory.controller_usage ?? null;
  const reflexArc = jsonRecord(detail.reflex_arc);
  if (!reflexArc
    || reflexArc.protocol !== "neural-reflex-arc-summary-v1"
    || reflexArc.execution_transaction_id !== receipt.transactionId) {
    throw new Error(
      `Certified physical receipt has no matching reflex arc: ${receipt.transactionId}`
    );
  }
  return {
    protocol: "mujoco-body-sensation-v1",
    transaction_id: receipt.transactionId,
    action: receipt.action,
    accepted: receipt.accepted,
    code: receipt.code,
    world_before_revision: receipt.worldBeforeRevision,
    world_after_revision: receipt.worldAfterRevision,
    frame_count: receipt.frameCount,
    final: detail.final ?? null,
    trajectory: {
      trajectory_sha256: trajectory.trajectory_sha256 ?? null,
      complete_from_admission: trajectory.complete_from_admission ?? null,
      start_frame: trajectory.start_frame ?? null,
      end_frame: trajectory.end_frame ?? null,
      root_path_length_m: trajectory.root_path_length_m ?? null,
      root_planar_path_length_m: trajectory.root_planar_path_length_m ?? null,
      contact_transition_count: trajectory.contact_transition_count ?? null
    },
    controller,
    reflex_arc: reflexArc
  };
}

function lowerMotorPredictionError(receipt: HumanoidActionReceipt): JsonValue {
  const detail = jsonRecord(receipt.detail) ?? {};
  return {
    protocol: "mujoco-body-prediction-error-v1",
    transaction_id: receipt.transactionId,
    action: receipt.action,
    accepted: receipt.accepted,
    code: receipt.code,
    world_revision: receipt.worldAfterRevision,
    frame_count: receipt.frameCount,
    final: detail.final ?? null,
    reason: detail.reason ?? detail.failure_class ?? receipt.code,
    reflex_arc: detail.reflex_arc ?? null
  };
}

function lowerMotorPredictionErrorMagnitude(reflexArcValue: JsonValue): number {
  const reflexArc = jsonRecord(reflexArcValue);
  if (!reflexArc) return 1;
  const controllerTicks = finiteNonnegative(reflexArc.controller_ticks);
  const trackingError = finiteNonnegative(
    reflexArc.last_weighted_joint_tracking_error
  );
  const controllerDelta = finiteNonnegative(
    reflexArc.last_controller_reference_delta
  );
  const nonFootContactTicks = finiteNonnegative(
    reflexArc.non_foot_contact_ticks
  );
  return Math.min(1, Math.max(
    0.25,
    reflexArc.fallen === true ? 1 : 0,
    trackingError / 0.35,
    controllerDelta / 0.7,
    controllerTicks > 0 ? nonFootContactTicks / controllerTicks : 0
  ));
}

function finiteNonnegative(value: JsonValue | undefined): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : 0;
}

function executionAction(
  execution: z.infer<typeof ResolvedExecutionSchema>
): {
  action: "execute_humanoid_skill" | "execute_whole_body_motion"
    | "execute_humanoid_navigation";
  input: Record<string, string>;
} {
  const action = {
    plan_humanoid_skill: "execute_humanoid_skill",
    plan_whole_body_motion: "execute_whole_body_motion",
    plan_whole_body_motion_candidates: "execute_whole_body_motion",
    plan_humanoid_navigation: "execute_humanoid_navigation"
  } as const;
  return {
    action: action[execution.planning_action],
    input: { planning_transaction_id: execution.planning_transaction_id }
  };
}

function planningReceiptFromRollout(payload: JsonValue): {
  transactionId: string;
  action: NeuralPlanningAction;
} {
  const receipt = z.object({
    transactionId: z.string().trim().min(1),
    action: z.enum([
      "plan_humanoid_skill",
      "plan_whole_body_motion",
      "plan_whole_body_motion_candidates",
      "plan_humanoid_navigation"
    ]),
    accepted: z.literal(true)
  }).passthrough().parse(payload);
  return { transactionId: receipt.transactionId, action: receipt.action };
}

function acceptedSkillProposal(
  state: NeuralHierarchyState,
  commitment: NeuralSkillCommitment
): NeuralSignal {
  const proposals = commitment.source_signal_ids.flatMap((signalId) => {
    const signal = state.signals[signalId];
    return signal?.kind === "skill_proposal"
      && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
      && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
      ? [signal]
      : [];
  });
  if (proposals.length !== 1) {
    throw new Error(
      `Active commitment must bind one accepted Sensorimotor proposal; found ${proposals.length}`
    );
  }
  return proposals[0]!;
}

/**
 * A physical execution can finish after the SDK episode that launched it was
 * paused and resumed. The durable Sensorimotor completion then belongs to the
 * previous Action Selection invocation, while the new Executive episode owns
 * the only live structural edge back into Action Selection. Rebind that exact
 * lifecycle signal as causal input instead of asking Executive to reproduce a
 * grandchild signal id or letting the new manager episode guess the outcome.
 */
function currentCommitmentLifecycleFeedback(
  runtime: HumanoidNeuralAgentRuntime,
  commitment: NeuralSkillCommitment,
  options: { pendingOnly?: boolean } = {}
): NeuralSignal | undefined {
  const outcome = runtime.neuralSkillCommitmentOutcome(commitment);
  if (outcome.status === "in_progress") return undefined;
  const detail = jsonRecord(outcome.detail);
  const signalId = detail?.completion_signal_id;
  if (typeof signalId !== "string") {
    throw new Error(
      `Resolved ${outcome.status} Skill outcome has no completion signal identity`
    );
  }
  const signal = runtime.neuralHierarchyState().signals[signalId];
  if (!signal
    || signal.kind !== (outcome.status === "completed"
      ? "skill_completed"
      : "skill_failed")
    || signal.source_node_id !== HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
    || signal.target_node_id !== HUMANOID_NEURAL_AGENT_IDS.actionSelection) {
    throw new Error(
      `Resolved ${outcome.status} Skill feedback is not a direct Sensorimotor signal`
    );
  }
  if (options.pendingOnly !== false && !runtime.pendingNeuralSignals({
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    kinds: [signal.kind]
  }).some((candidate) => candidate.signal_id === signal.signal_id)) {
    throw new Error(
      `Resolved ${outcome.status} Skill feedback is no longer pending for Action Selection`
    );
  }
  if ((commitment.state === "completed" || commitment.state === "failed")
    && !commitment.transition_signal_ids.some((transitionSignalId) => {
      const transition = runtime.neuralHierarchyState().signals[transitionSignalId];
      return transitionSignalId === signal.signal_id
        || transition !== undefined && neuralSignalHasAncestorId(
          runtime.neuralHierarchyState(),
          transition,
          signal.signal_id
        );
    })) {
    throw new Error(
      `Closed ${outcome.status} Skill commitment is not causally bound to its physical feedback`
    );
  }
  return signal;
}

function neuralInvocationInput(
  runtime: HumanoidNeuralAgentRuntime,
  parentAgentId: string,
  childAgentId: string,
  invocationId: string
): string {
  const signals = runtime.pendingNeuralSignals({
    targetNodeId: childAgentId
  }).filter((signal) => signal.invocation_id === invocationId
    || signal.parent_episode_id === invocationId);
  const anchor = humanoidNeuralContextProjection(
    runtime.contextAnchor(childAgentId),
    childAgentId,
    signals
  );
  return [
    `PARENT_CONTROL_AUTHORITY=${parentAgentId}`,
    `CHILD_STRUCTURAL_ID=${childAgentId}`,
    `HARNESS_INVOCATION_ID=${invocationId}`,
    "Only the following bounded typed state is current. No parent, sibling, or compacted transcript is transferred.",
    JSON.stringify({
      anchor,
      directed_signals: signals
    })
  ].join("\n");
}

export function humanoidNeuralContextProjection(
  anchor: JsonValue,
  agentId: string,
  signals: readonly NeuralSignal[]
): JsonValue {
  const record = jsonRecord(anchor);
  const neuralHierarchy = jsonRecord(record?.neural_hierarchy);
  if (!record || !neuralHierarchy) return anchor;
  const { directed_signals: _allPendingSignals, ...scopedHierarchy } = neuralHierarchy;
  const key = HUMANOID_NEURAL_NODE_BY_ID.get(agentId)?.key;
  if (!key) return { ...record, neural_hierarchy: scopedHierarchy };

  // Agent-as-tool children receive a responsibility projection, not a copy of
  // the runtime's universal operator view. The exact parent->child signals are
  // injected next to this anchor and remain the semantic handoff authority.
  // These fields only supply live state that the child cannot derive from that
  // edge. This keeps sibling Sessions isolated and prevents irrelevant skill
  // catalogs, world models, and history from being replayed at every depth.
  const common = pickJsonFields(record, [
    "world_frame",
    "world_revision",
    "active_agent"
  ]);
  const goal = pickJsonFields(record, ["active_goal"]);
  const perceptualState = pickJsonFields(record, ["active_goal", "goal_state"]);
  const hierarchy = { neural_hierarchy: scopedHierarchy };

  switch (key) {
    case "executive":
      return jsonValue({
        ...common,
        ...pickJsonFields(record, [
          "mission",
          "run_mode",
          "scenario_id",
          "mission_goal",
          "goal_dag",
          "active_goal",
          "active_cycle",
          "cycle_completion",
          "cycle_index",
          "previous_cycle_transition",
          "goal_state"
        ]),
        ...hierarchy
      });
    case "goalManager":
      return jsonValue({
        ...common,
        ...pickJsonFields(record, [
          "mission",
          "run_mode",
          "scenario_id",
          "mission_goal",
          "goal_dag",
          "active_goal",
          "goal_context",
          "goal_state",
          "cycle_index",
          "previous_cycle_transition"
        ]),
        ...hierarchy
      });
    case "actionSelection":
      return jsonValue({
        ...common,
        ...goal,
        ...pickJsonFields(record, ["mission_goal"]),
        ...hierarchy
      });
    case "perceptionManager":
      return jsonValue({ ...common, ...perceptualState, ...hierarchy });
    case "sceneInterpreter":
      return jsonValue({ ...common, ...goal, ...hierarchy });
    case "memoryRetriever":
      return jsonValue({
        ...common,
        ...goal,
        ...hierarchy
      });
    case "sensorimotorManager":
      return jsonValue({
        ...common,
        ...goal,
        ...hierarchy
      });
    case "affordance":
      return jsonValue({
        ...common,
        ...goal,
        interaction: compactInteractionProjection(record.interaction),
        ...hierarchy
      });
    case "risk":
      return jsonValue({
        ...common,
        ...goal,
        interaction: compactRiskProjection(record.interaction),
        ...hierarchy
      });
    case "predictive":
    case "premotor":
      return jsonValue({ ...common, ...goal, ...hierarchy });
    case "motorIntent":
      return jsonValue({
        ...common,
        ...goal,
        ...pickJsonFields(record, ["planning_tool_state"]),
        grounding_snapshot: compactMotorGrounding(
          record.grounding_snapshot,
          signals,
          record.planning_tool_state
        ),
        ...hierarchy
      });
    case "recovery":
      return jsonValue({
        ...common,
        ...goal,
        ...hierarchy
      });
    default:
      return jsonValue({ ...common, ...hierarchy });
  }
}

function pickJsonFields(
  source: Record<string, JsonValue>,
  keys: readonly string[]
): Record<string, JsonValue> {
  return Object.fromEntries(keys.flatMap((key) => (
    Object.prototype.hasOwnProperty.call(source, key)
      ? [[key, source[key]!] as const]
      : []
  )));
}

function compactInteractionProjection(value: JsonValue | undefined): JsonValue {
  const interaction = jsonRecord(value);
  if (!interaction) return null;
  const skillCatalog = jsonRecord(interaction.skill_catalog);
  const entries = Array.isArray(skillCatalog?.entries)
    ? skillCatalog.entries.flatMap((value) => {
        const entry = jsonRecord(value);
        if (!entry) return [];
        return [pickJsonFields(entry, [
          "id",
          "parameters",
          "required_affordances",
          "preconditions",
          "process",
          "success_conditions",
          "available",
          "unavailable_reasons",
          "observable_target_ids",
          "observable_solid_ids",
          "observable_zone_ids",
          "remembered_target_ids",
          "destination_ids",
          "learned_policy_ready",
          "learned_policy_missing_capabilities"
        ])];
      })
    : [];
  return jsonValue({
    ...pickJsonFields(interaction, [
      "frame",
      "world_revision",
      "zones",
      "manipulable_objects",
      "carrying"
    ]),
    skill_catalog: skillCatalog
      ? {
          ...pickJsonFields(skillCatalog, ["protocol", "contract_sha256"]),
          entries
        }
      : null
  });
}

function compactRiskProjection(value: JsonValue | undefined): JsonValue {
  const interaction = jsonRecord(value);
  if (!interaction) return null;
  return jsonValue(pickJsonFields(interaction, [
    "frame",
    "world_revision",
    "zones",
    "manipulable_objects",
    "carrying",
    "grasp_authority"
  ]));
}

function compactMotorGrounding(
  value: JsonValue | undefined,
  signals: readonly NeuralSignal[],
  planningToolState: JsonValue | undefined
): JsonValue {
  const grounding = jsonRecord(value);
  if (!grounding) return null;
  const skillIds = new Set<string>();
  collectSkillIds(planningToolState, skillIds);
  for (const signal of signals) collectSkillIds(signal.payload, skillIds);
  expandGoalCausalSkillIds(skillIds);
  const skillAuthority = jsonRecord(grounding.skill_authority);
  const skills = Array.isArray(skillAuthority?.skills)
    ? skillAuthority.skills.filter((value) => {
        const skill = jsonRecord(value);
        return skill && typeof skill.id === "string"
          && (skillIds.size === 0 || skillIds.has(skill.id));
      })
    : [];
  return jsonValue({
    ...pickJsonFields(grounding, [
      "protocol",
      "frame",
      "world_revision",
      "control_authority",
      "robot",
      "navigation",
      "spatial_belief",
      "zones",
      "objects",
      "solids",
      "carrying",
      "grasp",
      "manipulation_geometry"
    ]),
    skill_authority: skillAuthority
      ? {
          ...pickJsonFields(skillAuthority, [
            "protocol",
            "contract_sha256",
            "world_frame",
            "world_revision"
          ]),
          skills
        }
      : null
  });
}

function collectSkillIds(value: unknown, target: Set<string>): void {
  if (Array.isArray(value)) {
    for (const item of value) collectSkillIds(item, target);
    return;
  }
  const record = jsonRecord(value);
  if (!record) return;
  for (const [key, nested] of Object.entries(record)) {
    if ((key === "type" || key === "predicate_type")
      && typeof nested === "string") {
      collectGoalPredicateSkillIds(nested, target);
    }
    if (["skill", "skill_id", "target_skill"].includes(key)
      && typeof nested === "string") {
      const id = nested.match(/^[a-z][a-z0-9_]*/i)?.[0];
      if (id) target.add(id);
      continue;
    }
    collectSkillIds(nested, target);
  }
}

function collectGoalPredicateSkillIds(
  predicateType: string,
  target: Set<string>
): void {
  if (predicateType === "robot_at") {
    target.add("navigate_to_point");
    return;
  }
  if (predicateType === "robot_in_zone") {
    target.add("navigate_to_zone");
    return;
  }
  if (predicateType === "object_placed"
    || predicateType === "object_in_zone"
    || predicateType === "object_at"
    || predicateType === "object_inside"
    || predicateType === "object_on") {
    for (const skill of [
      "approach",
      "reach",
      "grasp",
      "lift",
      "carry",
      "carry_to_zone",
      "place",
      "regrasp",
      "bimanual_support",
      "bimanual_carry"
    ]) target.add(skill);
    return;
  }
  if (predicateType === "object_grasped") {
    for (const skill of [
      "approach",
      "reach",
      "grasp",
      "regrasp",
      "bimanual_support"
    ]) target.add(skill);
  }
}

function expandGoalCausalSkillIds(target: Set<string>): void {
  const seed = new Set(target);
  if ([...seed].some((skill) => [
    "place",
    "carry",
    "carry_to_zone",
    "bimanual_carry"
  ].includes(skill))) {
    for (const skill of ["approach", "reach", "grasp", "lift"]) {
      target.add(skill);
    }
  }
  if (seed.has("lift") || seed.has("grasp")) {
    target.add("approach");
    target.add("reach");
  }
  if (seed.has("reach")) target.add("approach");
}

function causalSemanticProjection(value: JsonValue): JsonValue {
  const record = jsonRecord(value);
  if (!record) return value;
  const causalInputs = Array.isArray(record.causal_inputs)
    ? record.causal_inputs
    : undefined;
  const control = jsonRecord(record.control);
  const legacyFreeTextDelegation = typeof record.intent === "string";
  if ((!control && !legacyFreeTextDelegation) || causalInputs === undefined) return value;
  const semanticInputs = causalInputs.flatMap((candidate) => {
    const input = jsonRecord(candidate);
    if (!input) return [];
    const payload = input.payload;
    return payload === undefined ? [] : [causalSemanticProjection(payload)];
  });
  if (semanticInputs.length === 1) return semanticInputs[0]!;
  if (semanticInputs.length > 1) {
    return jsonValue({
      ...(control ? { control } : {}),
      semantic_inputs: semanticInputs
    });
  }
  // Runs created before structural_neural_delegation_v1 can still contain a
  // prose `intent`. Treat it only as an obsolete routing wrapper: preserving
  // that text would let stale parent-authored motion choices cross an epoch.
  return jsonValue(control ? { control } : {});
}

function jsonValue(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}

function requiredHarnessInvocation(expectedAgentId: string): NonNullable<
  ReturnType<typeof currentAgentHarnessInvocation>
> {
  const invocation = currentAgentHarnessInvocation();
  if (!invocation || invocation.agentId !== expectedAgentId) {
    throw new Error(`Missing invocation scope for hierarchy node ${expectedAgentId}`);
  }
  return invocation;
}

function requiredParentEpisodeId(expectedParentAgentId: string): string {
  const invocation = currentAgentHarnessInvocation();
  if (!invocation || invocation.parentInvocationId === null
    || invocation.parentAgentId !== expectedParentAgentId) {
    throw new Error(
      `Hierarchy node ${invocation?.agentId ?? "unknown"} is not running under `
        + `its structural parent ${expectedParentAgentId}`
    );
  }
  return invocation.parentInvocationId;
}

/**
 * Re-enters an unfinished structural child episode after process or transport
 * recovery. A child publishes its descending inputs before its model runs and
 * consumes them only after a verified ascending result. Those pending signals
 * therefore form the durable continuation marker; relying only on an
 * in-memory closure gives the same recovered SDK Session a new neural
 * invocation identity after restart and severs its causal joins.
 */
function durablePendingChildInvocationId(input: {
  runtime: HumanoidNeuralAgentRuntime;
  parentNodeId: string;
  childNodeId: string;
  parentInvocationId: string;
}): ReturnType<typeof stableAgentToolInvocationId> | undefined {
  const state = input.runtime.neuralHierarchyState();
  const phase = state.harness_phase;
  const candidates = Object.values(state.signals).filter((signal) => {
    if ((signal.status !== "pending" && signal.status !== "expired")
      || signal.direction !== "descending"
      || signal.source_node_id !== input.parentNodeId
      || signal.target_node_id !== input.childNodeId
      || signal.parent_invocation_id !== input.parentInvocationId
      || signal.parent_episode_id !== input.parentInvocationId
      || signal.authority_lease_id === null) return false;
    const lease = state.authority_leases[signal.authority_lease_id];
    return lease !== undefined
      && (lease.status === "active"
        || lease.status === "closed"
        || lease.status === "expired")
      && lease.issuing_parent_node_id === input.parentNodeId
      && lease.target_child_node_id === input.childNodeId
      && lease.invocation_id === signal.invocation_id
      && lease.parent_invocation_id === input.parentInvocationId
      && lease.parent_episode_id === input.parentInvocationId
      && lease.goal_epoch_id === phase.goal_epoch_id
      && lease.commitment_id === phase.commitment_id;
  }).sort((left, right) => right.sequence - left.sequence);
  const selected = candidates[0];
  if (!selected) return undefined;
  return selected.invocation_id as ReturnType<typeof stableAgentToolInvocationId>;
}

async function prepareHarnessPhaseForChild(
  runtime: HumanoidNeuralAgentRuntime,
  parentKey: HumanoidNeuralAgentKey,
  childKey: HumanoidNeuralAgentKey,
  signalKind: NeuralSignalKind
): Promise<void> {
  const phase = runtime.neuralHarnessPhase();
  const transition = async (
    next: NeuralHarnessPhase,
    reason: string
  ): Promise<void> => {
    if (phase.phase === next) return;
    await runtime.transitionNeuralHarnessPhase({
      phase: next,
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS[parentKey],
      reason
    });
  };
  if (childKey === "goalManager") {
    await transition("goal_valuation", "executive_delegated_goal_valuation");
    return;
  }
  if (childKey === "perceptionManager") {
    await transition(
      phase.phase === "feedback" ? "feedback" : "perception",
      "action_selection_delegated_perception"
    );
    return;
  }
  if (childKey === "sensorimotorManager") {
    const directRecoveryDemand = parentKey === "actionSelection"
      && (signalKind === "prediction_error"
        || signalKind === "skill_failed"
        || signalKind === "escalation");
    if (phase.phase === "recovery" || directRecoveryDemand) {
      // Recovery is a strict Executive -> Action Selection -> Sensorimotor
      // control episode. A lower-loop failure first closes/releases the old
      // commitment, then this exact direct signal opens the Recovery decision
      // domain before Sensorimotor receives it. Preserve that domain until
      // Sensorimotor has opened and closed the exclusive Recovery child lease.
      await transition(
        "recovery",
        "action_selection_delegated_local_recovery_demand"
      );
      return;
    }
    const commitment = runtime.neuralHierarchyState().active_skill_commitment;
    await transition(
      commitment?.state === "executing"
        ? "execution"
        : commitment?.state === "committed"
          ? "motor_assessment"
          : "skill_proposal",
      commitment
        ? "action_selection_delegated_committed_sensorimotor_branch"
        : "action_selection_requested_skill_proposal"
    );
    return;
  }
  if (childKey === "premotor") {
    await transition("motor_planning", "sensorimotor_assessment_joined");
    return;
  }
  if (childKey === "predictive") {
    await transition("rollout_review", "mujoco_rollout_completed");
  }
}

async function advanceHarnessPhaseAfterChild(
  runtime: HumanoidNeuralAgentRuntime,
  childKey: HumanoidNeuralAgentKey,
  signalKind: NeuralSignalKind
): Promise<void> {
  const phase = runtime.neuralHarnessPhase();
  if (childKey === "goalManager" && signalKind === "goal_selected") {
    await runtime.transitionNeuralHarnessPhase({
      phase: "perception",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "goal_epoch_selected"
    });
    return;
  }
  if (childKey === "perceptionManager" && signalKind === "perceptual_belief") {
    const postFailureBelief = currentManagerChildSignals(
      runtime,
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      ["perceptual_belief"]
    ).some((signal) => neuralSignalHasAncestorKind(
      runtime.neuralHierarchyState(),
      signal,
      "skill_failed"
    ));
    const readiness = runtime.autonomyReadiness();
    const completionReady = readiness === "complete_cycle"
      || readiness === "complete_satisfied_goal";
    // Fresh agent epochs and interrupted/resumed SDK runs can retain a pending
    // failure signal from an older Action Selection invocation. That signal is
    // not authority for the current episode and must not suppress the current
    // post-failure belief. Scope the check to this concrete manager episode,
    // exactly as all other parent-child joins are scoped.
    const currentEpisodeHasFailureSignal = hasCurrentManagerEpisodeSignalsAny(
      runtime,
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      ["prediction_error", "skill_failed", "escalation"]
    );
    const freshDurableFailureBelief = readiness === "replan_or_retire"
      && !currentEpisodeHasFailureSignal;
    await runtime.transitionNeuralHarnessPhase({
      phase: phase.phase === "feedback" || completionReady
        ? "cycle_completion"
        : postFailureBelief || freshDurableFailureBelief
          ? "recovery"
          : "skill_proposal",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: phase.phase === "feedback"
        ? "post_execution_perception_joined"
        : completionReady
          ? "fresh_epoch_post_execution_perception_joined"
        : postFailureBelief
          ? "post_failure_perception_joined_before_recovery"
        : freshDurableFailureBelief
          ? "fresh_epoch_durable_failure_perception_joined"
        : "current_perceptual_belief_joined"
    });
    return;
  }
  if (childKey === "sensorimotorManager" && signalKind === "skill_proposal") {
    await runtime.transitionNeuralHarnessPhase({
      phase: "commitment_authorization",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "sensorimotor_skill_proposal_returned"
    });
    return;
  }
  if (childKey === "actionSelection" && signalKind === "skill_commitment") {
    const commitment = runtime.neuralHierarchyState().active_skill_commitment;
    if (commitment) {
      const failureObservationRequired = phase.phase === "perception"
        && runtime.autonomyReadiness() === "post_failure_observation";
      const nextPhase: NeuralHarnessPhase = failureObservationRequired
        ? "perception"
        : commitment.state === "executing"
          ? "execution"
          : commitment.state === "committed"
            ? "motor_assessment"
            : "skill_proposal";
      await runtime.transitionNeuralHarnessPhase({
        phase: nextPhase,
        enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
        reason: failureObservationRequired
          ? "released_commitment_returned_before_required_failure_observation"
          : `action_selection_returned_${commitment.state}_commitment`,
        goalEpochId: commitment.goal_epoch_id,
        commitmentId: ["completed", "failed", "released"].includes(commitment.state)
          ? null
          : commitment.commitment_id
      });
    }
    return;
  }
  if (childKey === "actionSelection"
    && (signalKind === "skill_completed" || signalKind === "skill_failed")) {
    if (phase.phase === "cycle_completion") return;
    const commitment = runtime.neuralHierarchyState().active_skill_commitment;
    await runtime.transitionNeuralHarnessPhase({
      phase: signalKind === "skill_completed" ? "feedback" : "perception",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: `action_selection_resolved_${signalKind}`,
      goalEpochId: commitment?.goal_epoch_id ?? phase.goal_epoch_id,
      commitmentId: null
    });
    return;
  }
  if (childKey === "motorIntent" && signalKind === "rollout_result") {
    const currentPhase = runtime.neuralHarnessPhase().phase;
    if (["execution", "feedback", "recovery", "cycle_completion", "terminal"]
      .includes(currentPhase)) {
      // The nested SDK result can finish unwinding after its parent has already
      // authorized a later phase. A completed lower-layer callback may confirm
      // its causal output, but it must never move the global phase backward.
      return;
    }
    if (currentPhase === "rollout_review") return;
    await runtime.transitionNeuralHarnessPhase({
      phase: "rollout_review",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      reason: "motor_intent_rollout_returned"
    });
    return;
  }
  if (childKey === "premotor" && signalKind === "escalation") {
    await runtime.transitionNeuralHarnessPhase({
      phase: "recovery",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      reason: "premotor_failure_returned_to_action_selection_for_recovery"
    });
    return;
  }
  if (childKey === "predictive" && signalKind === "forward_prediction") {
    // prepareHarnessPhaseForChild already entered rollout_review before the
    // Predictive episode. Do not rewrite the phase while nested Agent.asTool
    // calls unwind; Action Selection may have advanced it to execution.
    return;
  }
}

function requiredSession(
  sessions: ReadonlyMap<string, Session>,
  key: HumanoidNeuralAgentKey
): Session {
  const agentId = HUMANOID_NEURAL_AGENT_IDS[key];
  const session = sessions.get(agentId);
  if (!session) throw new Error(`Neural Agent has no owned Session: ${agentId}`);
  return session;
}

function hasCurrentSignal(
  runtime: HumanoidNeuralAgentRuntime,
  targetNodeId: string,
  kind: NeuralSignalKind
): boolean {
  const state = runtime.neuralHierarchyState();
  const storedCommitment = state.active_skill_commitment;
  const commitment = neuralSkillCommitmentIsOpen(storedCommitment)
    ? storedCommitment
    : null;
  const commitmentBoundKinds: ReadonlySet<NeuralSignalKind> = new Set([
    "skill_commitment",
    "affordance_hypothesis",
    "risk_assessment",
    "forward_prediction",
    "prediction_error",
    "motor_intent",
    "rollout_result",
    "execution_receipt",
    "skill_completed",
    "skill_failed"
  ]);
  return runtime.pendingNeuralSignals({ targetNodeId, kinds: [kind] }).some(
    (signal) => {
      if (!commitment || !commitmentBoundKinds.has(kind)) {
        return isCurrentNeuralSignal(runtime, signal);
      }
      const provenanceLeaseId = signal.authority_lease_id
        ?? signal.source_authority_lease_id;
      const lease = provenanceLeaseId
        ? state.authority_leases[provenanceLeaseId]
        : undefined;
      return signal.world_revision >= commitment.established_world_revision
        && lease?.commitment_id === commitment.commitment_id;
    }
  );
}

function currentActionSelectionBelief(
  runtime: HumanoidNeuralAgentRuntime,
  sourceSignals: readonly NeuralSignal[]
): NeuralSignal | undefined {
  const state = runtime.neuralHierarchyState();
  const beliefs = runtime.pendingNeuralSignals({
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    kinds: ["perceptual_belief"]
  }).filter((signal) => isCurrentNeuralSignal(runtime, signal)
    && isSemanticPerceptualBeliefSignal(signal));
  const causallyBound = beliefs.filter((belief) => sourceSignals.some((source) => (
    belief.signal_id === source.signal_id
      || neuralSignalHasAncestorId(state, belief, source.signal_id)
      || (source.kind === "skill_failed"
        && neuralSignalsShareAncestorKind(
          state,
          belief,
          source,
          "skill_failed"
        ))
  )));
  if (sourceSignals.some((signal) => signal.kind === "skill_failed")) {
    return [...causallyBound].sort(
      (left, right) => right.world_revision - left.world_revision
        || right.sequence - left.sequence
    )[0];
  }
  return [...(causallyBound.length > 0 ? causallyBound : beliefs)].sort(
    (left, right) => right.world_revision - left.world_revision
      || right.sequence - left.sequence
  )[0];
}

function isSemanticPerceptualBeliefSignal(signal: NeuralSignal): boolean {
  const payload = jsonRecord(signal.payload);
  if (payload?.protocol === "compact_perceptual_belief_v1") return true;
  if (payload?.protocol !== "materialized_perceptual_belief_v1") return false;
  return jsonRecord(payload.belief)?.protocol === "compact_perceptual_belief_v1";
}

function hasCurrentManagerEpisodeSignal(
  runtime: HumanoidNeuralAgentRuntime,
  managerNodeId: string,
  kind: NeuralSignalKind
): boolean {
  const invocationId = currentManagerEpisodeId(runtime, managerNodeId);
  if (!invocationId) return false;
  return runtime.pendingNeuralSignals({
    targetNodeId: managerNodeId,
    kinds: [kind]
  }).some((signal) => (
    (signal.direction === "descending"
      ? signal.invocation_id === invocationId
      : signal.parent_episode_id === invocationId)
      && isCurrentNeuralSignal(runtime, signal)
  ));
}

function hasCurrentManagerEpisodeSignals(
  runtime: HumanoidNeuralAgentRuntime,
  managerNodeId: string,
  kinds: readonly NeuralSignalKind[]
): boolean {
  return kinds.every((kind) => hasCurrentManagerEpisodeSignal(
    runtime,
    managerNodeId,
    kind
  ));
}

function hasCurrentManagerEpisodeSignalsAny(
  runtime: HumanoidNeuralAgentRuntime,
  managerNodeId: string,
  kinds: readonly NeuralSignalKind[]
): boolean {
  return kinds.some((kind) => hasCurrentManagerEpisodeSignal(
    runtime,
    managerNodeId,
    kind
  ));
}

function currentManagerEpisodeSignals(
  runtime: HumanoidNeuralAgentRuntime,
  managerNodeId: string,
  kinds: readonly NeuralSignalKind[]
): NeuralSignal[] {
  const invocationId = currentManagerEpisodeId(runtime, managerNodeId);
  if (!invocationId) return [];
  return runtime.pendingNeuralSignals({
    targetNodeId: managerNodeId,
    kinds
  }).filter((signal) => (
    (signal.direction === "descending"
      ? signal.invocation_id === invocationId
      : signal.parent_episode_id === invocationId)
      && isCurrentNeuralSignal(runtime, signal)
  ));
}

/**
 * The model-visible Session may remember UUIDs from earlier episodes, but a
 * direct-child delegation may cite only signals currently owned by this exact
 * parent episode. Descending inputs bind through the parent's invocation id;
 * child/reentrant returns bind through parent_episode_id. This is the causal
 * authority boundary that prevents a valid old UUID from becoming shared
 * memory or a cross-layer control channel.
 */
function currentDelegationSourceSignals(
  runtime: HumanoidNeuralAgentRuntime,
  parentNodeId: string,
  parentEpisodeId: string
): NeuralSignal[] {
  return runtime.pendingNeuralSignals({ targetNodeId: parentNodeId }).filter(
    (signal) => (
      (signal.direction === "descending"
        ? signal.invocation_id === parentEpisodeId
        : signal.parent_episode_id === parentEpisodeId)
        && isCurrentNeuralSignal(runtime, signal)
    )
  );
}

/**
 * Direct-edge provenance is Harness state. A model selects the structural child
 * and outgoing signal kind; it selects UUIDs only when the current parent
 * episode contains more than one genuinely admissible semantic source. This
 * prevents a persistent Session from turning an otherwise correct decision
 * into a correction round merely by repeating an older episode's UUID.
 */
function canonicalDirectDelegationSourceSignalIds(input: {
  currentParentSignals: readonly NeuralSignal[];
  requestedSourceSignalIds: readonly string[];
  signalKind: NeuralSignalKind;
  sourceSignalContract: NeuralDelegationSourceSignalContract | undefined;
}): string[] {
  const requested = [...new Set(input.requestedSourceSignalIds)];
  if (input.sourceSignalContract) {
    const allowedKinds = new Set(input.sourceSignalContract.allowedKinds);
    const requiredKinds = new Set(input.sourceSignalContract.requiredKinds);
    const admissible = input.currentParentSignals.filter(
      (signal) => allowedKinds.has(signal.kind)
    );
    const required = input.sourceSignalContract.requiredKinds.map((kind) => (
      admissible.filter((signal) => signal.kind === kind)
    ));
    const optional = admissible.filter((signal) => !requiredKinds.has(signal.kind));
    if (optional.length === 0
      && required.every((signals) => signals.length === 1)) {
      return required.map((signals) => signals[0]!.signal_id);
    }
  }

  const exactKind = input.currentParentSignals.filter(
    (signal) => signal.kind === input.signalKind
  );
  const semanticExactKind = exactKind.filter(
    (signal) => !isStructuralNeuralDelegationSignal(signal)
  );
  const canonicalExactKind = semanticExactKind.length > 0
    ? semanticExactKind
    : exactKind;
  if (canonicalExactKind.length === 1) {
    return [canonicalExactKind[0]!.signal_id];
  }
  if (exactKind.length === 0 && input.currentParentSignals.length <= 1) {
    return input.currentParentSignals.map((signal) => signal.signal_id);
  }
  return requested;
}

function isStructuralNeuralDelegationSignal(signal: NeuralSignal): boolean {
  return jsonRecord(jsonRecord(signal.payload)?.control)?.protocol
    === "structural_neural_delegation_v1";
}

function currentManagerChildSignals(
  runtime: HumanoidNeuralAgentRuntime,
  managerNodeId: string,
  kinds: readonly NeuralSignalKind[]
): NeuralSignal[] {
  const invocationId = currentManagerEpisodeId(runtime, managerNodeId);
  if (!invocationId) return [];
  return runtime.pendingNeuralSignals({
    targetNodeId: managerNodeId,
    kinds
  }).filter((signal) => signal.direction !== "descending"
    && signal.parent_episode_id === invocationId
    && isCurrentNeuralSignal(runtime, signal));
}

function currentSkillProposalAdmissionCorrection(
  runtime: HumanoidNeuralAgentRuntime
): { proposalSignalId: string; payload: JsonValue } | undefined {
  if (runtime.neuralHarnessPhase().reason !== "skill_proposal_admission_rejected"
    && runtime.neuralHarnessPhase().reason
      !== "recovery_proposal_admission_rejected") {
    return undefined;
  }
  const proposal = currentManagerChildSignals(
    runtime,
    HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    ["skill_proposal"]
  ).filter((signal) => signal.direction === "ascending"
    && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
    && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection)
    .sort((left, right) => right.sequence - left.sequence)[0];
  if (!proposal) return undefined;
  const admission = runtime.neuralSkillProposalAdmission?.(proposal);
  if (!admission || admission.accepted) return undefined;
  return {
    proposalSignalId: proposal.signal_id,
    payload: jsonValue({
      protocol: "skill_proposal_admission_correction_v1",
      rejected_proposal_signal_id: proposal.signal_id,
      rejected_invocation: admission.invocation ?? null,
      reason: admission.reason
        ?? "The proposed Skill does not advance the active Goal",
      detail: admission.detail ?? null,
      required_response: {
        mode: "replace_rejected_proposal",
        preserve_goal_epoch: true,
        repeat_rejected_invocation: false,
        choose_currently_ready_bounded_skill: true
      }
    })
  };
}

/**
 * SDK tool discovery is allowed to run outside the short-lived async-local
 * callback that opened an Agent.asTool episode.  The durable direct-parent
 * authority lease is therefore the source of truth for dynamic tool
 * availability; async-local identity remains the stronger execute-time check.
 */
function currentManagerEpisodeId(
  runtime: HumanoidNeuralAgentRuntime,
  managerNodeId: string
): string | undefined {
  const invocation = currentAgentHarnessInvocation();
  if (invocation?.agentId === managerNodeId) return invocation.invocationId;
  const state = runtime.neuralHierarchyState();
  const phase = state.harness_phase;
  const currentWorldRevision = runtime.currentWorldRevision();
  const leases = Object.values(state.authority_leases).filter((lease) => (
    lease.status === "active"
      && lease.target_child_node_id === managerNodeId
      && lease.goal_epoch_id === phase.goal_epoch_id
      && lease.commitment_id === phase.commitment_id
      && currentWorldRevision >= lease.issued_world_revision
      && currentWorldRevision <= lease.expires_world_revision
      && Date.now() <= Date.parse(lease.expires_at)
  ));
  return leases.length === 1 ? leases[0]!.invocation_id : undefined;
}

function isCurrentNeuralSignal(
  runtime: HumanoidNeuralAgentRuntime,
  signal: NeuralSignal
): boolean {
  const state = runtime.neuralHierarchyState();
  const storedCommitment = state.active_skill_commitment;
  const commitment = neuralSkillCommitmentIsOpen(storedCommitment)
    ? storedCommitment
    : null;
  const commitmentBoundKinds: ReadonlySet<NeuralSignalKind> = new Set([
    "skill_commitment",
    "affordance_hypothesis",
    "risk_assessment",
    "forward_prediction",
    "prediction_error",
    "motor_intent",
    "rollout_result",
    "execution_receipt",
    "skill_completed",
    "skill_failed"
  ]);
  if (!commitment || !commitmentBoundKinds.has(signal.kind)) {
    const liveInvocationIds = new Set<string>(currentAgentHarnessInvocationChain().map(
      (invocation) => invocation.invocationId
    ));
    return signal.status === "pending"
      && (liveInvocationIds.has(signal.invocation_id)
        || liveInvocationIds.has(signal.parent_episode_id)
        || runtime.currentWorldRevision()
          <= signal.world_revision + signal.ttl_revisions);
  }
  const provenanceLeaseId = signal.authority_lease_id
    ?? signal.source_authority_lease_id;
  const lease = provenanceLeaseId
    ? state.authority_leases[provenanceLeaseId]
    : undefined;
  return signal.world_revision >= commitment.established_world_revision
    && lease?.commitment_id === commitment.commitment_id;
}

function hasCurrentSignalsAny(
  runtime: HumanoidNeuralAgentRuntime,
  targetNodeId: string,
  kinds: readonly NeuralSignalKind[]
): boolean {
  return kinds.some((kind) => hasCurrentSignal(runtime, targetNodeId, kind));
}

function hasCurrentRecoveryDemand(
  runtime: HumanoidNeuralAgentRuntime
): boolean {
  return runtime.neuralHarnessPhase().phase === "recovery"
    || hasCurrentManagerEpisodeSignalsAny(
      runtime,
      HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      ["prediction_error", "skill_failed", "escalation"]
    );
}

function joinedManagerSignals(
  runtime: HumanoidNeuralAgentRuntime,
  childKey: HumanoidNeuralAgentKey,
  outputKind: NeuralSignalKind,
  invocationId: string
): NeuralSignal[] {
  const targetNodeId = HUMANOID_NEURAL_AGENT_IDS[childKey];
  let kinds: readonly NeuralSignalKind[] = [];
  if (childKey === "perceptionManager" && outputKind === "perceptual_belief") {
    kinds = ["sensory_evidence", "scene_interpretation", "memory_retrieval"];
  } else if (childKey === "sensorimotorManager" && outputKind === "skill_proposal") {
    kinds = [
      "perceptual_belief",
      "affordance_hypothesis",
      "risk_assessment",
      "skill_proposal"
    ];
  } else if (childKey === "sensorimotorManager"
    && (outputKind === "execution_receipt"
      || outputKind === "skill_completed"
      || outputKind === "skill_failed")) {
    kinds = [
      "rollout_result",
      "forward_prediction",
      "execution_receipt",
      "skill_completed",
      "skill_failed"
    ];
  } else if (childKey === "executionDispatcher"
    && (outputKind === "skill_completed" || outputKind === "skill_failed")) {
    kinds = [
      "execution_receipt",
      "prediction_error",
      "skill_completed",
      "skill_failed"
    ];
  }
  return kinds.length === 0 ? [] : durableManagerEpisodeSignals(
    runtime,
    targetNodeId,
    kinds,
    invocationId
  );
}

/**
 * A Manager episode owns its exact fork/join inputs until the enclosing
 * Agent.asTool invocation has published the aggregate result. World-revision
 * TTL is still used for ordinary routing and orphan cleanup, but it cannot
 * invalidate a persisted sibling result while the same episode is unwinding
 * through the SDK wrapper after a long-running parallel join.
 */
function durableManagerEpisodeSignals(
  runtime: HumanoidNeuralAgentRuntime,
  targetNodeId: string,
  kinds: readonly NeuralSignalKind[],
  invocationId: string
): NeuralSignal[] {
  const acceptedKinds = new Set(kinds);
  return Object.values(runtime.neuralHierarchyState().signals)
    .filter((signal) => signal.status === "pending"
      && signal.target_node_id === targetNodeId
      && acceptedKinds.has(signal.kind)
      && (signal.direction === "descending"
        ? signal.invocation_id === invocationId
        : signal.parent_episode_id === invocationId))
    .sort((left, right) => (
      right.priority - left.priority || left.sequence - right.sequence
    ));
}

/**
 * A Manager-authored aggregate is legal only after the Harness can prove the
 * complete sibling fork/join inside this exact Manager episode. Prompted tool
 * order is not authority: the returned output must cite one current result
 * from every required branch.
 */
function requireManagerJoinEvidence(
  runtime: HumanoidNeuralAgentRuntime,
  managerKey: HumanoidNeuralAgentKey,
  outputKind: NeuralSignalKind,
  invocationId: string,
  sourceSignalIds: readonly string[]
): NeuralSignal[] {
  const signals = managerJoinEvidence(
    runtime,
    managerKey,
    outputKind,
    invocationId,
    sourceSignalIds
  );
  const cited = new Set(sourceSignalIds);
  for (const signal of signals) {
    if (!cited.has(signal.signal_id)) {
      throw new Error(
        `${HUMANOID_NEURAL_AGENT_IDS[managerKey]} ${outputKind} omitted joined `
          + `${signal.kind} signal `
          + `${signal.signal_id} from source_signal_ids`
      );
    }
  }
  return signals;
}

/**
 * Fork/join provenance is structural Harness state, not a semantic choice for
 * the model. A Manager may cite additional current evidence, but the Harness
 * always binds the exact required child returns to the aggregate output.
 */
function canonicalManagerJoinEvidence(input: {
  runtime: HumanoidNeuralAgentRuntime;
  managerKey: HumanoidNeuralAgentKey;
  outputKind: NeuralSignalKind;
  invocationId: string;
  modelSourceSignalIds: readonly string[];
}): { signals: NeuralSignal[]; sourceSignalIds: string[] } {
  const signals = managerJoinEvidence(
    input.runtime,
    input.managerKey,
    input.outputKind,
    input.invocationId,
    input.modelSourceSignalIds
  );
  return {
    signals,
    sourceSignalIds: [...new Set([
      ...signals.map((signal) => signal.signal_id),
      ...input.modelSourceSignalIds
    ])].slice(0, 64)
  };
}

function managerJoinEvidence(
  runtime: HumanoidNeuralAgentRuntime,
  managerKey: HumanoidNeuralAgentKey,
  outputKind: NeuralSignalKind,
  invocationId: string,
  sourceSignalIds: readonly string[]
): NeuralSignal[] {
  if (managerKey === "sensorimotorManager" && outputKind === "skill_proposal") {
    const recoveryProposal = durableManagerEpisodeSignals(
      runtime,
      HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      ["skill_proposal"],
      invocationId
    ).find((signal) => signal.direction === "ascending"
      && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.recovery
      && sourceSignalIds.includes(signal.signal_id));
    if (recoveryProposal) return [];
  }
  const requirements: readonly NeuralSignalKind[] = managerKey === "perceptionManager"
    && outputKind === "perceptual_belief"
    ? ["sensory_evidence", "scene_interpretation", "memory_retrieval"]
    : managerKey === "sensorimotorManager" && outputKind === "skill_proposal"
      ? ["perceptual_belief", "affordance_hypothesis", "risk_assessment"]
      : [];
  if (requirements.length === 0) return [];
  const targetNodeId = HUMANOID_NEURAL_AGENT_IDS[managerKey];
  const available = durableManagerEpisodeSignals(
    runtime,
    targetNodeId,
    requirements,
    invocationId
  );
  return requirements.map((kind) => {
    const signal = available.find((candidate) => (
      candidate.kind === kind
        && (kind === "perceptual_belief"
          ? candidate.direction === "descending"
            && candidate.invocation_id === invocationId
            && isSemanticPerceptualBeliefSignal(candidate)
          : candidate.direction === "ascending"
            && candidate.parent_episode_id === invocationId)
    ));
    if (!signal) {
      throw new Error(
        `${targetNodeId} cannot return ${outputKind} before joining ${kind} `
          + `inside Manager episode ${invocationId}`
      );
    }
    return signal;
  });
}

function neuralDelegationSchema(
  parentKey: HumanoidNeuralAgentKey,
  childKey: HumanoidNeuralAgentKey,
  requiredSignalKind?: NeuralSignalKind,
  sourceSignalContract?: NeuralDelegationSourceSignalContract
) {
  const parentId = HUMANOID_NEURAL_AGENT_IDS[parentKey];
  const childId = HUMANOID_NEURAL_AGENT_IDS[childKey];
  const contract = HUMANOID_NEURAL_SIGNAL_CONTRACTS.find((candidate) => (
    candidate.sourceAgentId === parentId
      && candidate.targetAgentId === childId
      && candidate.direction === "descending"
  ));
  if (!contract) {
    throw new Error(`Missing descending neural signal contract: ${parentId} -> ${childId}`);
  }
  if (requiredSignalKind !== undefined
    && !(contract.signalKinds as readonly NeuralSignalKind[]).includes(
      requiredSignalKind
    )) {
    throw new Error(
      `Required signal kind ${requiredSignalKind} is outside neural contract: `
        + `${parentId} -> ${childId}`
    );
  }
  if (sourceSignalContract) {
    const allowedKinds = new Set(sourceSignalContract.allowedKinds);
    if (sourceSignalContract.allowedKinds.length === 0
      || sourceSignalContract.requiredKinds.some((kind) => !allowedKinds.has(kind))) {
      throw new Error(
        `Invalid source signal contract: ${parentId} -> ${childId}`
      );
    }
  }
  const signalKinds = [
    ...(requiredSignalKind === undefined
      ? contract.signalKinds
      : [requiredSignalKind])
  ] as [
    NeuralSignalKind,
    ...NeuralSignalKind[]
  ];
  const carriesPremotorProgram = parentKey === "premotor"
    && childKey === "motorIntent";
  const common = {
    signal_kind: z.enum(signalKinds).describe(
      carriesPremotorProgram
        ? "The committed typed signal on the fixed Premotor-to-Motor-Intent edge. Premotor owns the semantic Skill DAG; Motor Intent owns only compilation and deterministic planner selection."
        : "The allowed typed signal on this fixed structural parent-child edge. It does not authorize choosing a lower layer's Skill, hand, interaction point, route, posture, coordinates, or motor parameters."
    ),
    source_signal_ids: z.array(z.string().uuid()).max(64).default([]).describe([
      "Usually omit this field. The Harness binds a unique current causal source on this fixed edge. Supply ids only when the invocation exposes multiple genuinely admissible semantic candidates; consumed, expired, sibling-owned, foreign-parent, and earlier-episode ids are rejected.",
      ...(sourceSignalContract
        ? [
            `This edge requires source kinds: ${sourceSignalContract.requiredKinds.join(", ")}.`,
            `Only these source kinds are admissible: ${sourceSignalContract.allowedKinds.join(", ")}. A UUID never changes its semantic kind.`
          ]
        : []),
      "The Harness supplies all child state and derives the child's responsibility from the structural edge and current phase."
    ].join(" ")),
    ttl_revisions: z.number().int().min(1).max(1_000_000)
      .default(MODEL_EPISODE_SIGNAL_TTL_REVISIONS),
    priority: z.number().int().min(0).max(100).default(50)
  };
  return carriesPremotorProgram
    ? z.object({
        ...common,
        motor_program: PremotorMotorProgramSchema.describe(
          "Premotor-owned semantic Skill DAG. Motor Intent must compile it verbatim and cannot infer it from the parent transcript."
        )
      }).strict()
    : z.object(common).strict();
}

export function humanoidNeuralAgentProfile(
  agentId: string
): Exclude<AgentModelProfile, "compactor"> {
  const profiles: Partial<Record<HumanoidNeuralAgentKey, Exclude<
    AgentModelProfile,
    "compactor"
  >>> = {
    executive: "executive",
    goalManager: "executive",
    actionSelection: "executive",
    perceptionManager: "associative",
    sceneInterpreter: "associative",
    memoryRetriever: "associative",
    sensorimotorManager: "sensorimotor",
    affordance: "associative",
    risk: "associative",
    predictive: "sensorimotor",
    premotor: "sensorimotor",
    motorIntent: "motor_intent",
    executionDispatcher: "sensorimotor",
    recovery: "sensorimotor"
  };
  const descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(agentId);
  const profile = descriptor ? profiles[descriptor.key] : undefined;
  if (!profile) throw new Error(`Neural runtime node has no model profile: ${agentId}`);
  return profile;
}

function stableNeuralToolCapability(
  candidate: Tool,
  agentId: string,
  runtime: HumanoidNeuralAgentRuntime
): Tool {
  if (candidate.type !== "function") return candidate;
  const admission = candidate.isEnabled;
  const invoke = candidate.invoke;
  const owningAgents = new WeakMap<object, Agent<any, any>>();
  candidate.isEnabled = async (context, agent) => {
    owningAgents.set(context, agent);
    return true;
  };
  candidate.invoke = async (context, rawInput, details) => {
    const owner = owningAgents.get(context);
    if (!owner || !await admission(context, owner)) {
      return JSON.stringify({
        accepted: false,
        code: "neural_tool_not_admitted",
        agent_id: agentId,
        tool: candidate.name,
        harness_phase: runtime.neuralHarnessPhase().phase,
        automatic_actuation: false,
        next_response_contract: {
          mode: "choose_admitted_capability",
          narration_allowed: false
        },
        recovery: "Choose the capability authorized by the current directed signals, Harness phase, and commitment state."
      });
    }
    return invoke(context, rawInput, details);
  };
  return candidate;
}

function baseInstructions(key: HumanoidNeuralAgentKey): string[] {
  const descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(HUMANOID_NEURAL_AGENT_IDS[key]);
  if (!descriptor) throw new Error(`Unknown neural node: ${key}`);
  const outputInstructions = key === "executionDispatcher"
    ? [
        "Your only valid response is the required execute_certified_motor_intent tool call. Its result terminates this episode."
      ]
    : GENERIC_NEURAL_SUBMISSION_AGENT_KEYS.has(key)
      ? [
        `Submit every final neural signal through ${NEURAL_OUTPUT_SUBMISSION_TOOL_NAME}; never return the envelope as assistant text.`,
        "Pass the structured signal body directly in the submission tool's payload field; do not JSON-stringify it."
        ]
      : [
          "Terminate only through an owned direct-child, lifecycle, or domain tool result; never restate that result through a generic neural submission."
        ];
  return [
    `You are ${descriptor.name}, structural id ${descriptor.id}.`,
    `Your one control parent is ${descriptor.parentKey === null
      ? "none; you are the root"
      : HUMANOID_NEURAL_AGENT_IDS[descriptor.parentKey]}.`,
    descriptor.objective,
    ...descriptor.criteria,
    `Activation cadence: ${descriptor.cadence}; maximum correction scope: ${descriptor.maximumCorrectionScope}.`,
    "This is a strict hierarchy. Never contact siblings, read their Sessions, or bypass your parent.",
    "Treat only directed world-versioned neural signals and your bounded invocation anchor as current.",
    "When delegating, choose only one owned child edge and an allowed signal_kind. The Harness binds the unique causal source; use source_signal_ids only when multiple current semantic candidates remain. The structural edge and current Harness phase define the child's bounded responsibility; there is no free-text delegation channel and you cannot prescribe a lower layer's Skill, hand, interaction point, route, posture, coordinates, or motor parameters.",
    "Never copy a UUID merely for protocol bookkeeping. If multiple admissible sources are present, select only exact signal_id values from the current invocation; never invent a UUID or placeholder.",
    ...outputInstructions,
    "The Harness canonicalizes the payload after schema validation. Escalate only errors outside your local correction scope."
  ];
}

function neuralOutputSubmissionTool<TOutput extends z.ZodObject>(
  key: HumanoidNeuralAgentKey,
  outputType: TOutput,
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, TOutput, unknown> {
  // Function-calling models are much more reliable when structured data stays
  // structured. Requiring a JSON document inside a string made the provider
  // escape an entire second wire format and caused otherwise valid leaf-node
  // signals to fail before the Harness could see them. Keep payload_json as the
  // internal/Agent output representation for compatibility, but expose one
  // native JSON payload field at the model boundary and canonicalize it here.
  const submissionParameters = neuralSubmissionParameters(outputType, key);
  return tool({
    name: NEURAL_OUTPUT_SUBMISSION_TOOL_NAME,
    description: "Submit this node's final typed neural signal to its structural parent.",
    parameters: submissionParameters as never,
    // Arbitrary JSON objects require `additionalProperties`; strict function
    // schemas reject that shape. The SDK parses this plain JSON Schema, then
    // `submissionSchema` performs the full executable Zod validation below.
    strict: false,
    timeoutBehavior: "raise_exception",
    // Valid JSON that fails executable Zod/authority checks gets an explicit
    // correction receipt. Syntactically malformed JSON is rejected one layer
    // earlier by the Agents SDK and also remains non-terminal below.
    errorFunction: (_context, error) => neuralSubmissionRejection(error),
    // The Agents SDK materializes an Agent's tool surface at episode start and
    // does not add a dynamically enabled tool after earlier calls complete.
    // Keep the endpoint discoverable for manager-authored join episodes, but
    // remove it entirely from Sensorimotor recovery. Recovery owns that
    // decision domain and its direct typed child result terminates the parent
    // episode through sensorimotorToolUseBehavior; exposing this competing
    // endpoint lets compatible models restate the child proposal and loop on a
    // barrier that can never be satisfied in a recovery episode.
    isEnabled: () => key !== "goalManager"
      && key !== "motorIntent"
      && (key !== "actionSelection"
        || runtime.neuralHarnessPhase().phase === "cycle_completion")
      && (key !== "sensorimotorManager"
        || runtime.neuralHarnessPhase().phase !== "recovery"),
    execute: (params: unknown) => {
      const submitted = z.record(z.string(), z.unknown()).parse(params);
      const { payload, ...envelope } = submitted;
      const signalKind = NeuralAgentOutputSchema.shape.signal_kind.parse(
        envelope.signal_kind
      );
      const modelSourceSignalIds = z.array(z.string().uuid()).max(64).parse(
        envelope.source_signal_ids ?? []
      );
      const invocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS[key]
      );
      const join = canonicalManagerJoinEvidence({
        runtime,
        managerKey: key,
        outputKind: signalKind,
        invocationId: invocation.invocationId,
        modelSourceSignalIds
      });
      const validatedPayload = validateNeuralSubmissionPayload(
        key,
        signalKind,
        payload
      );
      const parsed = outputType.parse({
        ...envelope,
        signal_kind: signalKind,
        source_signal_ids: join.sourceSignalIds,
        payload_json: JSON.stringify(validatedPayload)
      });
      assertNeuralOutputSubmissionAuthority(
        runtime,
        key,
        NeuralAgentOutputSchema.parse({
          signal_kind: parsed.signal_kind,
          summary: parsed.summary,
          payload_json: parsed.payload_json,
          source_signal_ids: parsed.source_signal_ids,
          confidence: parsed.confidence
        })
      );
      return JSON.stringify(parsed);
    }
  }) as unknown as FunctionTool<unknown, TOutput, unknown>;
}

function neuralSubmissionRejection(
  error: unknown
): string {
  if (error instanceof NeuralSubmissionBarrierError) {
    return JSON.stringify({
      accepted: false as const,
      code: "neural_submission_barrier_unsatisfied" as const,
      tool: NEURAL_OUTPUT_SUBMISSION_TOOL_NAME,
      neural_output: null,
      error: error.message,
      automatic_actuation: false as const,
      next_response_contract: {
        mode: "invoke_required_child_tools" as const,
        required_tools: neuralSubmissionBarrierTools(error.key, error.phase),
        resubmit_tool: NEURAL_OUTPUT_SUBMISSION_TOOL_NAME,
        resubmit_allowed_after_children: true as const,
        narration_allowed: false as const
      }
    });
  }
  return JSON.stringify({
    accepted: false as const,
    code: "invalid_tool_input" as const,
    tool: NEURAL_OUTPUT_SUBMISSION_TOOL_NAME,
    neural_output: null,
    error: error instanceof Error
      ? error.message
      : "The neural output arguments were not valid JSON or did not match the declared schema.",
    validation_issues: error instanceof z.ZodError
      ? error.issues.slice(0, 16).map((issue) => ({
          path: issue.path.join("."),
          code: issue.code,
          message: issue.message
        }))
      : [{
          path: "",
          code: "invalid_json",
          message: "Tool arguments must be valid JSON and match the declared neural output schema."
        }],
    automatic_actuation: false as const,
    next_response_contract: {
      mode: "corrected_tool_call_only" as const,
      tool: NEURAL_OUTPUT_SUBMISSION_TOOL_NAME,
      preserve_valid_fields: true as const,
      narration_allowed: false as const
    }
  });
}

class NeuralSubmissionBarrierError extends Error {
  readonly key: HumanoidNeuralAgentKey;
  readonly phase: NeuralHarnessPhase;

  constructor(key: HumanoidNeuralAgentKey, phase: NeuralHarnessPhase) {
    super(
      `${HUMANOID_NEURAL_AGENT_IDS[key]} cannot submit before its formal child/state barrier`
    );
    this.name = "NeuralSubmissionBarrierError";
    this.key = key;
    this.phase = phase;
  }
}

function neuralSubmissionBarrierTools(
  key: HumanoidNeuralAgentKey,
  phase: NeuralHarnessPhase
): readonly string[] {
  if (key === "perceptionManager") {
    return [
      humanoidNeuralAgentToolName("sensorFusion"),
      humanoidNeuralAgentToolName("sceneInterpreter"),
      humanoidNeuralAgentToolName("memoryRetriever")
    ];
  }
  if (key === "sensorimotorManager") {
    if (phase === "skill_proposal") {
      return [
        humanoidNeuralAgentToolName("affordance"),
        humanoidNeuralAgentToolName("risk")
      ];
    }
    if (phase === "motor_assessment" || phase === "motor_planning") {
      return [humanoidNeuralAgentToolName("premotor")];
    }
    if (phase === "rollout_review") {
      return [humanoidNeuralAgentToolName("predictive")];
    }
    if (phase === "execution") {
      return [humanoidNeuralAgentToolName("executionDispatcher")];
    }
    if (phase === "recovery") {
      return [humanoidNeuralAgentToolName("recovery")];
    }
  }
  if (key === "premotor") {
    return [humanoidNeuralAgentToolName("motorIntent")];
  }
  if (key === "actionSelection") {
    return [
      humanoidNeuralAgentToolName("perceptionManager"),
      humanoidNeuralAgentToolName("sensorimotorManager")
    ];
  }
  if (key === "executive") {
    return [
      humanoidNeuralAgentToolName("goalManager"),
      humanoidNeuralAgentToolName("actionSelection")
    ];
  }
  return [];
}

function neuralSubmissionParameters(
  outputType: z.ZodObject,
  key: HumanoidNeuralAgentKey
): Record<string, unknown> {
  const schema = structuredClone(
    z.toJSONSchema(outputType)
  ) as Record<string, unknown> & {
    allOf?: unknown[];
    properties?: Record<string, unknown>;
    required?: string[];
  };
  if (!schema.properties) {
    throw new Error("Neural submission output schema has no object properties");
  }
  delete schema.properties.payload_json;
  // This is intentionally a native, open JSON value. The function tool is
  // declared non-strict because JSON Schema cannot express arbitrary object
  // keys under strict Structured Outputs. The native payload and reconstructed
  // node output are both Zod-validated inside the Harness before routing.
  schema.properties.payload = key === "perceptionManager"
    ? compactPerceptualBeliefPayloadJsonSchema()
    : {
        description: key === "sensorimotorManager" || key === "recovery"
          ? "Structured JSON body. A skill_proposal must match the bounded proposed_skill contract shown by the conditional schema."
          : "Structured JSON body of this neural signal"
      };
  if (key === "sensorimotorManager" || key === "recovery") {
    schema.allOf = [
      ...(schema.allOf ?? []),
      {
        if: {
          properties: { signal_kind: { const: "skill_proposal" } },
          required: ["signal_kind"]
        },
        then: {
          properties: {
            payload: boundedSkillProposalPayloadJsonSchema()
          },
          required: ["payload"]
        }
      }
    ];
  }
  schema.required = [
    ...(schema.required ?? []).filter((key) => key !== "payload_json"),
    "payload"
  ];
  return schema;
}

function compactPerceptualBeliefPayloadJsonSchema(): Record<string, unknown> {
  const boundedTextArray = (maximumItems: number) => ({
    type: "array",
    maxItems: maximumItems,
    items: { type: "string", minLength: 1, maxLength: 320 }
  });
  return {
    type: "object",
    additionalProperties: false,
    description: "A bounded cortical-style belief code. The Harness binds verbatim child evidence; do not copy raw child payloads here.",
    properties: {
      protocol: { type: "string", const: "compact_perceptual_belief_v1" },
      world_revision: { type: "integer", minimum: 0 },
      goal_state: {
        type: "string",
        enum: ["unsatisfied", "satisfied", "unknown"]
      },
      observations: {
        ...boundedTextArray(8),
        minItems: 1,
        description: "Task-relevant observed facts only; no action, Skill, hand, route, or pose selection"
      },
      uncertainties: boundedTextArray(4),
      changed_entity_ids: {
        type: "array",
        maxItems: 16,
        items: { type: "string", minLength: 1, maxLength: 160 }
      },
      safety_relevant: boundedTextArray(4),
      escalation_reason: {
        type: "string",
        maxLength: 1_000,
        description: "Optional; omit or use an empty string when no escalation exists"
      }
    },
    required: [
      "protocol",
      "world_revision",
      "goal_state",
      "observations",
      "uncertainties",
      "changed_entity_ids",
      "safety_relevant"
    ]
  };
}

function boundedSkillProposalPayloadJsonSchema(): Record<string, unknown> {
  type SkillInvocationJsonSchema = Record<string, unknown> & {
    oneOf?: Array<Record<string, unknown> & {
      properties?: Record<string, unknown> & {
        skill?: { const?: unknown };
      };
      required?: string[];
    }>;
  };
  const invocationSchema = z.toJSONSchema(
    HumanoidSkillInvocationSchema
  ) as SkillInvocationJsonSchema;
  const invocationVariants = invocationSchema.oneOf ?? [];
  const proposedSkillVariants = HUMANOID_SKILL_IDS.map((skill) => {
    const invocationVariant = invocationVariants.find(
      (candidate) => candidate.properties?.skill?.const === skill
    );
    if (!invocationVariant?.properties) {
      throw new Error(`Humanoid Skill JSON Schema is missing ${skill}`);
    }
    const params = structuredClone(invocationVariant);
    delete params.properties!.skill;
    params.required = (params.required ?? []).filter((key) => key !== "skill");
    const phases = HUMANOID_SKILL_CONTRACTS[skill].process.map(
      (entry) => entry.phase
    );
    return {
      type: "object",
      additionalProperties: false,
      properties: {
        skill: { type: "string", const: skill },
        phase: {
          type: "string",
          enum: phases,
          description: `${skill} process phase`
        },
        params,
        rationale: { type: "string", minLength: 1, maxLength: 8_000 }
      },
      required: ["skill", "phase", "params", "rationale"]
    };
  });
  const phaseReferences = HUMANOID_SKILL_IDS.map((skill) => ({
    type: "object",
    additionalProperties: false,
    properties: {
      skill: { type: "string", const: skill },
      phase: {
        type: "string",
        enum: HUMANOID_SKILL_CONTRACTS[skill].process.map((entry) => entry.phase)
      }
    },
    required: ["skill", "phase"]
  }));
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      proposed_skill: {
        oneOf: proposedSkillVariants,
        description: "Exactly one bounded Skill with its real invocation parameters and process phase"
      },
      phase_sequence: {
        type: "array",
        items: { oneOf: phaseReferences },
        maxItems: 32,
        description: "Optional future causal order; never replaces proposed_skill"
      }
    },
    required: ["proposed_skill"]
  };
}

function validateNeuralSubmissionPayload(
  key: HumanoidNeuralAgentKey,
  signalKind: unknown,
  payload: unknown
): JsonValue {
  const validated = NeuralPayloadSchema.parse(payload);
  const structured = key === "perceptionManager"
    && signalKind === "perceptual_belief"
    ? validatePerceptualBeliefAuthority(
        JsonValueSchema.parse(CompactPerceptualBeliefPayloadSchema.parse(validated))
      )
    : key === "affordance"
    && signalKind === "affordance_hypothesis"
    ? validateAffordanceManipulationRecommendation(validated)
    : signalKind === "skill_proposal"
      && (key === "sensorimotorManager" || key === "recovery")
      ? BoundedSkillProposalPayloadSchema.parse(validated)
      : validated;
  return JsonValueSchema.parse(structured);
}

function validatePerceptualBeliefAuthority(payload: JsonValue): JsonValue {
  const root = jsonRecord(payload);
  if (!root) return payload;
  const prescriptiveFields = [
    "recommendation",
    "immediate_next_skill",
    "next_action",
    "proposed_skill",
    "selected_action",
    "selected_hand",
    "selected_interaction_point",
    "selected_skill",
    "motor_command"
  ].filter((field) => Object.hasOwn(root, field));
  if (prescriptiveFields.length === 0) return payload;
  throw new Error(
    `Perceptual belief crossed its hierarchy authority boundary with prescriptive fields: ${prescriptiveFields.join(", ")}. `
      + "Report only bounded observed state and uncertainty. The Harness binds verbatim child evidence; Action Selection and Sensorimotor exclusively own Skill, hand, interaction-point, and next-action selection."
  );
}

function materializePerceptualBelief(
  compactBelief: JsonValue,
  joinedSignals: readonly NeuralSignal[]
): JsonValue {
  const evidence = Object.fromEntries(joinedSignals.map((signal) => [
    signal.kind,
    {
      signal_id: signal.signal_id,
      source_node_id: signal.source_node_id,
      world_revision: signal.world_revision,
      payload: causalSemanticProjection(signal.payload)
    }
  ]));
  return JsonValueSchema.parse({
    protocol: "materialized_perceptual_belief_v1",
    belief: compactBelief,
    authoritative_evidence: evidence
  });
}

function validateAffordanceManipulationRecommendation(
  payload: JsonValue
): JsonValue {
  const root = jsonRecord(payload);
  if (!root) return payload;
  const contradictions: string[] = [];
  let expectedPair: {
    hand: string | undefined;
    point: string | undefined;
  } | undefined;
  const comparePair = (
    placement: Record<string, JsonValue> | null | undefined,
    recommendation: Record<string, JsonValue> | null | undefined,
    pointField: string,
    handField: string,
    label: string
  ) => {
    if (!placement || !recommendation) return;
    const point = stringField(placement, "interaction_point_id");
    const surface = stringField(placement, "hand_surface");
    const hand = surface?.startsWith("left_")
      ? "left"
      : surface?.startsWith("right_")
        ? "right"
        : undefined;
    expectedPair ??= { hand, point };
    const recommendedPoint = stringField(recommendation, pointField);
    const recommendedHand = stringField(recommendation, handField);
    if (point && recommendedPoint && point !== recommendedPoint) {
      contradictions.push(`${label}.${pointField}=${recommendedPoint}`);
    }
    if (hand && recommendedHand && hand !== recommendedHand) {
      contradictions.push(`${label}.${handField}=${recommendedHand}`);
    }
  };

  const affordances = jsonRecord(root.affordances);
  const graspable = jsonRecord(affordances?.graspable);
  const legacyPlacement = jsonRecord(graspable?.reachable_base_placement);
  comparePair(
    legacyPlacement,
    graspable,
    "recommended_interaction_point",
    "recommended_hand",
    "graspable"
  );
  const immediate = jsonRecord(root.immediate_next_skill);
  const candidate = immediate ? stringField(immediate, "candidate") : undefined;
  const parameters = jsonRecord(immediate?.parameters);
  if (candidate === "approach") {
    comparePair(
      legacyPlacement,
      parameters,
      "interaction_point_id",
      "hand",
      "immediate approach"
    );
  }
  const currentReachability = jsonRecord(graspable?.current_reachability);
  if (candidate === "reach" && currentReachability?.reachable_now === false) {
    contradictions.push(
      "immediate reach selected while current_reachability.reachable_now=false"
    );
  }

  const hypotheses = Array.isArray(root.affordance_hypotheses)
    ? root.affordance_hypotheses
    : [];
  for (const hypothesisValue of hypotheses) {
    const hypothesis = jsonRecord(hypothesisValue);
    const reachability = jsonRecord(hypothesis?.reachability);
    const placement = jsonRecord(reachability?.recommended_base_placement);
    const feasibility = jsonRecord(hypothesis?.grasp_feasibility);
    comparePair(
      placement,
      feasibility,
      "recommended_interaction_point",
      "recommended_hand",
      `affordance_hypotheses[${hypothesis ? stringField(hypothesis, "object_id") ?? "object" : "object"}].grasp_feasibility`
    );
    comparePair(
      placement,
      jsonRecord(root.recommendation),
      "interaction_point",
      "hand",
      "recommendation"
    );
  }
  if (contradictions.length === 0) return payload;
  throw new Error(
    `Affordance manipulation recommendation contradicts its reachable_base_placement (${contradictions.join(", ")}). `
      + `Use the exact placement pair hand=${expectedPair?.hand ?? "unknown"}, interaction_point_id=${expectedPair?.point ?? "unknown"}; `
      + "when reachable_now=false, recommend approach before reach."
  );
}

function validateBoundedSkillProposalOutput(
  output: { signal_kind: NeuralSignalKind; payload_json: string },
  context: z.RefinementCtx
): void {
  if (output.signal_kind !== "skill_proposal") return;
  let payload: unknown;
  try {
    payload = JSON.parse(output.payload_json);
  } catch {
    return;
  }
  const parsed = BoundedSkillProposalPayloadSchema.safeParse(payload);
  if (parsed.success) return;
  for (const issue of parsed.error.issues.slice(0, 16)) {
    context.addIssue({
      code: "custom",
      path: ["payload_json", ...issue.path],
      message: issue.message
    });
  }
}

function neuralOutputSubmissionAvailable(
  runtime: HumanoidNeuralAgentRuntime,
  key: HumanoidNeuralAgentKey
): boolean {
  switch (key) {
    case "goalManager":
    case "motorIntent":
    case "executionDispatcher":
      // Their formal mutation/planning tools terminate the SDK episode and
      // synthesize the typed output from the real tool receipt.
      return false;
    case "executive":
      if (runtime.neuralHarnessPhase().phase === "cycle_completion") return false;
      return currentManagerChildSignals(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.executive,
        [
          "goal_selected",
          "perceptual_belief",
          "skill_commitment",
          "skill_completed",
          "skill_failed",
          "escalation"
        ]
      ).length > 0;
    case "actionSelection":
      // Commitment transitions terminate through their state-mutation tools.
      // The only free-form return is post-execution belief (or an actual child
      // escalation). During ordinary selection a belief must continue down to
      // Sensorimotor instead of being bounced back to Executive.
      return runtime.neuralHarnessPhase().phase === "cycle_completion"
        && currentManagerChildSignals(
          runtime,
          HUMANOID_NEURAL_AGENT_IDS.actionSelection,
          ["perceptual_belief", "escalation"]
        ).length > 0;
    case "perceptionManager":
      return hasCurrentManagerEpisodeSignals(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
        ["sensory_evidence", "scene_interpretation", "memory_retrieval"]
      ) || hasCurrentManagerEpisodeSignal(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
        "escalation"
      );
    case "sensorimotorManager": {
      const phase = runtime.neuralHarnessPhase().phase;
      if (phase === "skill_proposal") {
        return hasCurrentManagerEpisodeSignals(
          runtime,
          HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          ["perceptual_belief", "affordance_hypothesis", "risk_assessment"]
        );
      }
      if (phase === "rollout_review") {
        return hasCurrentManagerEpisodeSignals(
          runtime,
          HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          ["rollout_result", "forward_prediction"]
        );
      }
      return hasCurrentManagerEpisodeSignalsAny(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        [
          "execution_receipt",
          "skill_completed",
          "skill_failed",
          "prediction_error",
          "escalation"
        ]
      );
    }
    case "premotor":
      return currentManagerChildSignals(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.premotor,
        ["motor_intent", "rollout_result", "escalation"]
      ).length > 0;
    default:
      // Read-only leaf critics and interpreters originate their own bounded
      // interpretation from the one descending signal in this invocation.
      return true;
  }
}

function assertNeuralOutputSubmissionAuthority(
  runtime: HumanoidNeuralAgentRuntime,
  key: HumanoidNeuralAgentKey,
  output: z.infer<typeof NeuralAgentOutputSchema>
): void {
  if (!neuralOutputSubmissionAvailable(runtime, key)) {
    throw new NeuralSubmissionBarrierError(
      key,
      runtime.neuralHarnessPhase().phase
    );
  }
  const requiredKinds: readonly NeuralSignalKind[] = key === "executive"
    ? output.signal_kind === "goal_context"
      ? ["goal_selected"]
      : [output.signal_kind]
    : key === "actionSelection"
      ? [output.signal_kind]
      : key === "sensorimotorManager"
        ? output.signal_kind === "skill_proposal"
          ? ["perceptual_belief", "affordance_hypothesis", "risk_assessment"]
          : output.signal_kind === "rollout_result"
            ? ["rollout_result", "forward_prediction"]
            : [output.signal_kind]
      : key === "perceptionManager"
        ? output.signal_kind === "perceptual_belief"
          ? ["sensory_evidence", "scene_interpretation", "memory_retrieval"]
          : [output.signal_kind]
      : key === "premotor"
        ? output.signal_kind === "skill_proposal"
          ? ["motor_intent"]
          : [output.signal_kind]
        : [];
  if (requiredKinds.length > 0) {
    const current = currentManagerEpisodeSignals(
      runtime,
      HUMANOID_NEURAL_AGENT_IDS[key],
      requiredKinds
    );
    const cited = new Set(output.source_signal_ids);
    for (const kind of requiredKinds) {
      const matching = current.filter((signal) => signal.kind === kind);
      if (matching.length === 0) {
        throw new Error(
          `${HUMANOID_NEURAL_AGENT_IDS[key]} submission lacks formal ${kind} child/state signal`
        );
      }
      if (!matching.some((signal) => cited.has(signal.signal_id))) {
        throw new Error(
          `${HUMANOID_NEURAL_AGENT_IDS[key]} submission omitted its formal ${kind} child/state signal`
        );
      }
    }
  }
  const invocation = requiredHarnessInvocation(HUMANOID_NEURAL_AGENT_IDS[key]);
  requireManagerJoinEvidence(
    runtime,
    key,
    output.signal_kind,
    invocation.invocationId,
    output.source_signal_ids
  );
}

function neuralOutputToolUseBehavior(
  outputType: z.ZodObject,
  existing?: ToolUseBehavior
): ToolUseBehavior {
  return async (context, results) => {
    // submit_neural_output is terminal only after both its tool-result contract
    // and this exact Agent's output schema succeed. SDK parser/validation
    // failures are ordinary correction turns and must go back to the same
    // Agent, never through a parent or fallback path.
    const ordinaryResults = results.filter((result) => result.type !== "function_output"
      || result.tool.name !== NEURAL_OUTPUT_SUBMISSION_TOOL_NAME);
    if (typeof existing === "function") {
      const decision = await existing(context, ordinaryResults);
      if (decision.isFinalOutput || decision.isInterrupted) return decision;
    } else if (existing === "stop_on_first_tool") {
      const first = ordinaryResults.find((result) => result.type === "function_output");
      if (first?.type === "function_output") {
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: typeof first.output === "string"
            ? first.output
            : JSON.stringify(first.output)
        };
      }
    } else if (existing && typeof existing === "object") {
      const terminal = ordinaryResults.find((result) => result.type === "function_output"
        && existing.stopAtToolNames.includes(result.tool.name));
      if (terminal?.type === "function_output") {
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: typeof terminal.output === "string"
            ? terminal.output
            : JSON.stringify(terminal.output)
        };
      }
    }
    const submissions = results.filter((result) => result.type === "function_output"
      && result.tool.name === NEURAL_OUTPUT_SUBMISSION_TOOL_NAME);
    for (const submission of submissions) {
      if (submission.type !== "function_output") continue;
      const output = outputType.safeParse(outputObject(submission.output));
      if (!output.success) continue;
      return {
        isFinalOutput: true,
        isInterrupted: undefined,
        finalOutput: JSON.stringify(output.data)
      };
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function directChildNeuralOutcomeToolUseBehavior(
  childToolName: string,
  outputType: z.ZodObject
): ToolUseBehavior {
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output" || result.tool.name !== childToolName) {
        continue;
      }
      const childOutput = outputObject(result.output);
      if (!childOutput) continue;
      const output = directChildNeuralOutput(outputType, childOutput);
      if (!output) continue;
      return {
        isFinalOutput: true,
        isInterrupted: undefined,
        finalOutput: JSON.stringify(output)
      };
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function scopedInstructions(agentId: string, instructions: readonly string[]): string {
  return `${agentInvocationMarker(agentId)}\n${instructions.join("\n")}`;
}

function createRuntimeServices(): ReadonlyMap<string, NeuralRuntimeService> {
  return new Map(HUMANOID_NEURAL_NODES
    .filter((node) => node.executionKind !== "model_agent")
    .map((node) => [node.id, {
      id: node.id,
      name: node.name,
      implementationContract: runtimeImplementationContract(node.key)
    }] as const));
}

function runtimeImplementationContract(key: HumanoidNeuralAgentKey): string {
  const contracts: Partial<Record<HumanoidNeuralAgentKey, string>> = {
    sensorFusion: "authoritative_humanoid_sensor_fusion_v1",
    rolloutGate: "existing_mujoco_planning_rollout_gate_v1",
    executor: "serial_physical_execution_gate_v1",
    reflex: "learned_reference_controller_reflex_loop_v1",
    body: "authoritative_mujoco_body_v1"
  };
  const contract = contracts[key];
  if (!contract) throw new Error(`Neural model node has no runtime contract: ${key}`);
  return contract;
}

function planningReceiptToolUseBehavior(
  runtime: HumanoidNeuralAgentRuntime
): ToolUseBehavior {
  let invocationId: string | undefined;
  const rejectedRecoveryEvidenceByModality = new Map<
    MotorIntentTransitRecoveryModality,
    {
      action: string;
      sourceSignalIds: string[];
      receipt: JsonValue;
    }
  >;
  return (_context, results) => {
    const currentInvocationId = requiredHarnessInvocation(
      HUMANOID_NEURAL_AGENT_IDS.motorIntent
    ).invocationId;
    if (invocationId !== currentInvocationId) {
      invocationId = currentInvocationId;
      rejectedRecoveryEvidenceByModality.clear();
    }
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const output = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output);
      const receiptRecord = outputObject(output);
      const receipt = receiptRecord as Partial<HumanoidActionReceipt> | undefined;
      const physicalPlanningResult = receipt?.transactionId
        && MOTOR_INTENT_PLANNING_ACTIONS.has(String(receipt.action));
      const rejectedSkillStateTransition = receipt?.transactionId
        && receipt.accepted === false
        && MOTOR_INTENT_SKILL_STATE_ACTIONS.has(String(receipt.action));
      const sourceSignalIds = z.array(z.string().uuid()).max(64).catch([]).parse(
        receiptRecord?.source_signal_ids
      );
      const planningAction = String(receipt?.action ?? "");
      const recoveryAttempt = MotorIntentTransitRecoveryAttemptSchema.safeParse(
        receiptRecord?.motor_intent_recovery_attempt
      );
      if (physicalPlanningResult
        && receipt.accepted === false
        && recoveryAttempt.success
        && recoveryAttempt.data.action === planningAction) {
        rejectedRecoveryEvidenceByModality.set(recoveryAttempt.data.modality, {
          action: planningAction,
          sourceSignalIds,
          receipt: jsonValue({
            transaction_id: receipt.transactionId,
            action: receipt.action,
            accepted: receipt.accepted,
            code: receipt.code,
            world_before_revision: receipt.worldBeforeRevision,
            world_after_revision: receipt.worldAfterRevision,
            detail: modelReceiptDetail(receipt.detail ?? null)
          })
        });
      }
      const localRecoveryExhausted = physicalPlanningResult
        && receipt.accepted === false
        && motorIntentTransitRecoveryExhausted(
          runtime,
          rejectedRecoveryEvidenceByModality
        );
      if (physicalPlanningResult
        && receipt.accepted === false
        && !localRecoveryExhausted
        && motorIntentRecoveryPlanningAvailable(
          runtime,
          rejectedRecoveryEvidenceByModality
        )) {
        // The SDK's native Agent loop is the owner of intra-episode tool
        // recovery. A physical rejection has already updated the Runtime's
        // planning state; keep the same Motor Intent episode alive so the
        // model can consume that state and select one still-untried bounded
        // recovery modality. Once both collision-clearance modalities have
        // produced deterministic rejection signals in this exact episode,
        // the commitment is outside Motor Intent's correction scope and must
        // ascend through Premotor instead of looping over fresh coordinates.
        continue;
      }
      if ((physicalPlanningResult || rejectedSkillStateTransition)
        && typeof receipt?.accepted === "boolean") {
        const routedSourceSignalIds = localRecoveryExhausted
          ? [...new Set([
              ...[...rejectedRecoveryEvidenceByModality.values()]
                .flatMap((evidence) => evidence.sourceSignalIds),
              ...sourceSignalIds
            ])].slice(0, 64)
          : sourceSignalIds;
        const { source_signal_ids: _sourceSignalIds, ...receiptPayload } = receiptRecord!;
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify({
            signal_kind: receipt.accepted ? "rollout_result" : "escalation",
            summary: localRecoveryExhausted
              ? "motor_intent_local_recovery_exhausted"
              : receipt.code ?? "planning result",
            payload_json: JSON.stringify(localRecoveryExhausted
              ? {
                  ...receiptPayload,
                  local_recovery_exhaustion: {
                    protocol: "motor_intent_local_recovery_exhausted_v1",
                    commitment_id: runtime.neuralHierarchyState()
                      .active_skill_commitment?.commitment_id ?? null,
                    rejected_recovery_modalities: [
                      ...rejectedRecoveryEvidenceByModality.keys()
                    ],
                    rejected_planning_actions: [...new Set(
                      [...rejectedRecoveryEvidenceByModality.values()]
                        .map((evidence) => evidence.action)
                    )],
                    rejected_rollout_signal_ids: routedSourceSignalIds,
                    rejected_recovery_receipts: Object.fromEntries(
                      [...rejectedRecoveryEvidenceByModality].map(
                        ([modality, evidence]) => [modality, evidence.receipt]
                      )
                    )
                  }
                }
              : receiptPayload),
            source_signal_ids: routedSourceSignalIds,
            confidence: receipt.accepted ? 1 : 0.5
          })
        };
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function currentMotorIntentPlanningToolState(
  runtime: HumanoidNeuralAgentRuntime
): JsonValue {
  const anchor = jsonRecord(runtime.contextAnchor(
    HUMANOID_NEURAL_AGENT_IDS.motorIntent
  ));
  return anchor?.planning_tool_state ?? null;
}

function currentPremotorMotorProgram(
  runtime: HumanoidNeuralAgentRuntime
): z.infer<typeof PremotorMotorProgramSchema> {
  const invocation = requiredHarnessInvocation(
    HUMANOID_NEURAL_AGENT_IDS.motorIntent
  );
  const state = runtime.neuralHierarchyState();
  const candidates = runtime.pendingNeuralSignals({
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
    invocationId: invocation.invocationId
  }).flatMap((signal) => {
    if (signal.direction !== "descending"
      || signal.source_node_id !== HUMANOID_NEURAL_AGENT_IDS.premotor
      || signal.authority_lease_id === null) return [];
    const lease = state.authority_leases[signal.authority_lease_id];
    const program = PremotorMotorProgramSchema.safeParse(
      jsonRecord(signal.payload)?.motor_program
    );
    return lease?.status === "active" && program.success
      ? [{ signal, lease, program: program.data }]
      : [];
  }).sort((left, right) => (
    right.lease.issued_world_revision - left.lease.issued_world_revision
      || right.signal.sequence - left.signal.sequence
  ));
  const newest = candidates[0];
  if (!newest) {
    throw new Error("Motor Intent has no active Premotor motor_program authority");
  }
  const sameAuthority = candidates.filter(
    (candidate) => candidate.lease.lease_id === newest.lease.lease_id
  );
  if (sameAuthority.length !== 1) {
    throw new Error(
      `Motor Intent requires one Premotor motor_program on its current authority edge; found ${sameAuthority.length}`
    );
  }
  return newest.program;
}

function motorIntentRecoveryPlanningAvailable(
  runtime: HumanoidNeuralAgentRuntime,
  rejectedRecoveryEvidenceByModality: ReadonlyMap<
    MotorIntentTransitRecoveryModality,
    {
      readonly sourceSignalIds: readonly string[];
    }
  >
): boolean {
  const state = jsonRecord(currentMotorIntentPlanningToolState(runtime));
  const transitClearance = jsonRecord(state?.transit_clearance);
  if (transitClearance?.status !== "required") return false;
  const actions = Array.isArray(state?.planning_actions)
    ? state.planning_actions
    : [];
  return actions.some((value) => {
    const action = jsonRecord(value);
    const modality = motorIntentTransitRecoveryModality(action?.action);
    return action?.available === true
      && modality !== null
      && !rejectedRecoveryEvidenceByModality.has(modality);
  });
}

function motorIntentTransitRecoveryExhausted(
  runtime: HumanoidNeuralAgentRuntime,
  rejectedRecoveryEvidenceByModality: ReadonlyMap<
    MotorIntentTransitRecoveryModality,
    {
      readonly sourceSignalIds: readonly string[];
    }
  >
): boolean {
  const state = jsonRecord(currentMotorIntentPlanningToolState(runtime));
  const transitClearance = jsonRecord(state?.transit_clearance);
  return transitClearance?.status === "required"
    && MOTOR_INTENT_TRANSIT_RECOVERY_MODALITIES.every((modality) => (
      (rejectedRecoveryEvidenceByModality.get(modality)?.sourceSignalIds.length ?? 0) > 0
    ));
}

function motorIntentTransitRecoveryAttempt(
  runtime: HumanoidNeuralAgentRuntime,
  action: string
): z.infer<typeof MotorIntentTransitRecoveryAttemptSchema> | null {
  const state = jsonRecord(currentMotorIntentPlanningToolState(runtime));
  const transitClearance = jsonRecord(state?.transit_clearance);
  const modality = motorIntentTransitRecoveryModality(action);
  if (transitClearance?.status !== "required" || modality === null) return null;
  return MotorIntentTransitRecoveryAttemptSchema.parse({
    protocol: "motor_intent_transit_recovery_attempt_v1",
    modality,
    action
  });
}

function motorIntentTransitRecoveryModality(
  action: unknown
): MotorIntentTransitRecoveryModality | null {
  if (action === "plan_whole_body_motion_candidates") {
    return "whole_body_clearance";
  }
  return action === "plan_humanoid_navigation"
    ? "alternate_navigation"
    : null;
}

function goalValuationToolUseBehavior(): ToolUseBehavior {
  const statuses = new Set([
    "goal_candidate_selected",
    "goal_epoch_retired",
    "goal_epoch_continued"
  ]);
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const output = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output);
      const parsed = outputObject(output);
      if (parsed && statuses.has(String(parsed.status))) {
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify({
            signal_kind: "goal_selected",
            summary: String(parsed.status),
            payload_json: JSON.stringify(parsed),
            source_signal_ids: [],
            confidence: 1
          })
        };
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function actionSelectionToolUseBehavior(
  runtime: HumanoidNeuralAgentRuntime
): ToolUseBehavior {
  const statuses = new Set(["skill_executing"]);
  const perceptionToolName = humanoidNeuralAgentToolName("perceptionManager");
  const sensorimotorToolName = humanoidNeuralAgentToolName("sensorimotorManager");
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const parsed = outputObject(result.output);
      if (!parsed) continue;
      if (result.tool.name === sensorimotorToolName
        && parsed.signal_kind === "escalation"
        && directNeuralResultBindsChildOutput(
          runtime,
          parsed,
          HUMANOID_NEURAL_AGENT_IDS.recovery,
          "escalation"
        )) {
        const finalOutput = directChildNeuralOutput(
          ActionSelectionOutputSchema,
          parsed
        );
        if (!finalOutput || finalOutput.signal_kind !== "escalation") continue;
        // Recovery escalation already crossed the direct Sensorimotor edge.
        // Preserve that typed result so it can cross the final direct edge to
        // Executive without another model-authored restatement.
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify(finalOutput)
        };
      }
      if (result.tool.name === perceptionToolName
        && runtime.neuralHarnessPhase().phase === "cycle_completion") {
        const finalOutput = ActionSelectionOutputSchema.safeParse({
          signal_kind: parsed.signal_kind,
          summary: parsed.summary,
          payload_json: JSON.stringify(parsed.payload),
          source_signal_ids: parsed.source_signal_ids,
          confidence: parsed.confidence
        });
        if (!finalOutput.success
          || finalOutput.data.signal_kind !== "perceptual_belief") continue;
        // Post-execution Perception is already a typed, causally routed child
        // result. Returning it directly closes this bounded manager-as-tool
        // episode; asking the model to restate it caused compatible models to
        // repeat the previous skill_completed signal after the commitment had
        // already closed.
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify(finalOutput.data)
        };
      }
      if (!statuses.has(String(parsed.status))) continue;
      const status = String(parsed.status);
      const sourceSignalIds = z.array(z.string().uuid()).max(64).catch([]).parse(
        parsed.source_signal_ids
      );
      return {
        isFinalOutput: true,
        isInterrupted: undefined,
        finalOutput: JSON.stringify({
          signal_kind: "skill_commitment",
          summary: status,
          payload_json: JSON.stringify(parsed),
          source_signal_ids: sourceSignalIds,
          confidence: 1
        })
      };
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function sensorimotorToolUseBehavior(
  runtime: HumanoidNeuralAgentRuntime
): ToolUseBehavior {
  const predictiveToolName = humanoidNeuralAgentToolName("predictive");
  const premotorToolName = humanoidNeuralAgentToolName("premotor");
  const executionDispatcherToolName = humanoidNeuralAgentToolName(
    "executionDispatcher"
  );
  const recoveryToolName = humanoidNeuralAgentToolName("recovery");
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const parsed = outputObject(result.output);
      if (!parsed) continue;
      if (result.tool.name === premotorToolName
        && parsed.signal_kind === "escalation") {
        const finalOutput = directChildNeuralOutput(
          SensorimotorOutputSchema,
          parsed
        );
        if (!finalOutput || finalOutput.signal_kind !== "escalation") continue;
        // Premotor already returned a typed, Harness-routed failure for this
        // exact committed motor program. End this Sensorimotor episode on that
        // direct child result; asking the model to quote the nested Motor Intent
        // signal crosses authority namespaces and can only restate stale data.
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify(finalOutput)
        };
      }
      if (result.tool.name === recoveryToolName
        && (parsed.signal_kind === "skill_proposal"
          || parsed.signal_kind === "escalation")) {
        const finalOutput = directChildNeuralOutput(
          SensorimotorOutputSchema,
          parsed
        );
        if (!finalOutput) continue;
        // A Recovery proposal is authorized by its exclusive failure lease,
        // not by the ordinary Affordance/Risk fork. End the Sensorimotor
        // episode on the already-routed child result so Action Selection can
        // either bind the replacement Skill or forward the escalation.
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify(finalOutput)
        };
      }
      if (result.tool.name === predictiveToolName
        && parsed.signal_kind === "forward_prediction") {
        const state = runtime.neuralHierarchyState();
        const commitmentId = state.active_skill_commitment?.commitment_id;
        const certificates = Object.values(state.rollout_certificates).filter(
          (candidate) => candidate.status === "active"
            && candidate.commitment_id === commitmentId
        );
        if (certificates.length !== 1) {
          throw new Error(
            `Sensorimotor Predictive join requires one active rollout certificate; found ${certificates.length}`
          );
        }
        const certificate = certificates[0]!;
        const current = currentManagerChildSignals(
          runtime,
          HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          ["rollout_result", "forward_prediction"]
        );
        const prediction = current.find((signal) => (
          signal.kind === "forward_prediction"
            && signal.signal_id === certificate.predictive_signal_id
        ));
        const certifiedRollout = state.signals[certificate.rollout_signal_id];
        if (!certifiedRollout
          || certifiedRollout.kind !== "rollout_result"
          || certifiedRollout.source_node_id !== HUMANOID_NEURAL_AGENT_IDS.rolloutGate
          || modelPayloadSha256(certifiedRollout.payload)
            !== certificate.rollout_payload_sha256) {
          throw new Error("Sensorimotor Predictive join lost its certified raw rollout");
        }
        const rollouts = current.filter((signal) => (
          signal.kind === "rollout_result"
            && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.premotor
            && neuralSignalHasAncestorId(
              state,
              signal,
              certifiedRollout.signal_id
            )
            && prediction !== undefined
            && neuralSignalHasAncestorId(
              state,
              prediction,
              certifiedRollout.signal_id
            )
        ));
        if (!prediction || rollouts.length !== 1) {
          throw new Error(
            `Sensorimotor Predictive join requires one direct certified rollout; found ${rollouts.length}`
          );
        }
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify({
            signal_kind: "rollout_result",
            summary: "predictive_certified_rollout",
            payload_json: JSON.stringify(rollouts[0]!.payload),
            source_signal_ids: [
              rollouts[0]!.signal_id,
              prediction.signal_id
            ],
            confidence: typeof parsed.confidence === "number"
              ? parsed.confidence
              : 1
          })
        };
      }
      if (result.tool.name === predictiveToolName
        && (parsed.signal_kind === "prediction_error"
          || parsed.signal_kind === "escalation")) {
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify({
            signal_kind: parsed.signal_kind,
            summary: typeof parsed.summary === "string"
              ? parsed.summary
              : `predictive_${parsed.signal_kind}`,
            payload_json: typeof parsed.payload_json === "string"
              ? parsed.payload_json
              : JSON.stringify(parsed.payload ?? {}),
            source_signal_ids: z.array(z.string().uuid()).max(64).parse(
              parsed.source_signal_ids
            ),
            confidence: typeof parsed.confidence === "number"
              ? parsed.confidence
              : 1
          })
        };
      }
      if (result.tool.name === executionDispatcherToolName
        && (parsed.signal_kind === "skill_completed"
          || parsed.signal_kind === "skill_failed")) {
        const finalOutput = directChildNeuralOutput(
          SensorimotorOutputSchema,
          parsed
        );
        if (!finalOutput) continue;
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify(finalOutput)
        };
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function certifiedExecutionDispatcherToolUseBehavior(
  runtime: HumanoidNeuralAgentRuntime
): ToolUseBehavior {
  const executorToolName = humanoidNeuralAgentToolName("executor");
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output"
        || result.tool.name !== executorToolName) continue;
      const parsed = outputObject(result.output);
      if (!parsed) continue;
      const sourceSignalIds = z.array(z.string().uuid()).max(64).parse(
        parsed.source_signal_ids
      );
      const state = runtime.neuralHierarchyState();
      const completionSignals = sourceSignalIds.map(
        (signalId) => state.signals[signalId]
      ).filter((signal): signal is NeuralSignal => signal !== undefined
        && (signal.kind === "skill_completed" || signal.kind === "skill_failed")
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.executor
        && signal.target_node_id
          === HUMANOID_NEURAL_AGENT_IDS.executionDispatcher);
      if (completionSignals.length !== 1) {
        throw new Error(
          `Serial execution must return one physical completion signal; found ${completionSignals.length}`
        );
      }
      const { source_signal_ids: _sourceSignalIds, ...payload } = parsed;
      return {
        isFinalOutput: true,
        isInterrupted: undefined,
        finalOutput: JSON.stringify({
          signal_kind: completionSignals[0]!.kind,
          summary: typeof parsed.code === "string"
            ? parsed.code
            : completionSignals[0]!.kind,
          payload_json: JSON.stringify(payload),
          source_signal_ids: sourceSignalIds,
          confidence: parsed.accepted === true ? 1 : 0.5
        })
      };
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function directChildNeuralOutput<TSchema extends z.ZodObject>(
  schema: TSchema,
  parsed: Record<string, unknown>
): z.infer<TSchema> | undefined {
  const output = schema.safeParse({
    signal_kind: parsed.signal_kind,
    summary: parsed.summary,
    payload_json: typeof parsed.payload_json === "string"
      ? parsed.payload_json
      : JSON.stringify(parsed.payload ?? {}),
    source_signal_ids: parsed.source_signal_ids,
    confidence: parsed.confidence
  });
  return output.success ? output.data : undefined;
}

function executiveToolUseBehavior(): ToolUseBehavior {
  const statuses = new Set(["cycle_completed", "satisfied_goal_completed"]);
  const goalToolName = humanoidNeuralAgentToolName("goalManager");
  const actionSelectionToolName = humanoidNeuralAgentToolName("actionSelection");
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const parsed = outputObject(result.output);
      if (!parsed) continue;
      if (result.tool.name === actionSelectionToolName) {
        const executiveOutput = directChildNeuralOutput(
          ExecutiveOutputSchema,
          parsed
        );
        if (!executiveOutput) continue;
        // A direct manager-as-tool return is already a typed, Harness-routed
        // result. End this event-bounded Executive episode immediately instead
        // of asking the model to relabel and resubmit the same child signal.
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify(executiveOutput)
        };
      }
      if (result.tool.name === goalToolName) {
        const childOutput = NeuralAgentOutputSchema.passthrough().safeParse(parsed);
        if (!childOutput.success) continue;
        if (childOutput.data.signal_kind === "goal_selected") {
          // Goal selection and initial action dispatch are one supervisory
          // event. The Harness has already entered perception, so keep the
          // Executive Agent loop open and let it invoke Action Selection
          // directly instead of scheduling a redundant root model turn.
          continue;
        }
        const executiveOutput = ExecutiveOutputSchema.safeParse({
          signal_kind: "goal_context",
          summary: childOutput.data.summary,
          payload_json: JSON.stringify(childOutput.data.payload),
          source_signal_ids: childOutput.data.source_signal_ids,
          confidence: childOutput.data.confidence
        });
        if (!executiveOutput.success) continue;
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify(executiveOutput.data)
        };
      }
      if (!statuses.has(String(parsed.status))) continue;
      const sourceSignalIds = z.array(z.string().uuid()).max(64).catch([]).parse(
        parsed.source_signal_ids
      );
      return {
        isFinalOutput: true,
        isInterrupted: undefined,
        finalOutput: JSON.stringify({
          signal_kind: "skill_completed",
          summary: typeof parsed.summary === "string" && parsed.summary.trim()
            ? parsed.summary
            : String(parsed.status),
          payload_json: JSON.stringify(parsed),
          source_signal_ids: sourceSignalIds,
          confidence: 1
        })
      };
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function neuralSignalHasAncestorKind(
  state: NeuralHierarchyState,
  signal: NeuralSignal,
  kind: NeuralSignalKind
): boolean {
  const visited = new Set<string>();
  const pending = [...signal.causal_parent_ids];
  while (pending.length > 0) {
    const signalId = pending.pop()!;
    if (visited.has(signalId)) continue;
    visited.add(signalId);
    const ancestor = state.signals[signalId];
    if (!ancestor) continue;
    if (ancestor.kind === kind) return true;
    pending.push(...ancestor.causal_parent_ids);
  }
  return false;
}

function neuralSignalBindsCommitment(
  state: NeuralHierarchyState,
  signal: NeuralSignal,
  commitment: NeuralSkillCommitment
): boolean {
  const visited = new Set<string>();
  const pending = [signal.signal_id];
  while (pending.length > 0) {
    const signalId = pending.pop()!;
    if (visited.has(signalId)) continue;
    visited.add(signalId);
    const candidate = state.signals[signalId];
    if (!candidate) continue;
    if (candidate.kind === "skill_commitment") {
      const bound = NeuralSkillCommitmentSchema.safeParse(candidate.payload);
      if (bound.success
        && bound.data.commitment_id === commitment.commitment_id
        && bound.data.goal_epoch_id === commitment.goal_epoch_id) {
        return true;
      }
    }
    pending.push(...candidate.causal_parent_ids);
  }
  return false;
}

function neuralSignalHasAncestorId(
  state: NeuralHierarchyState,
  signal: NeuralSignal,
  ancestorSignalId: string
): boolean {
  const visited = new Set<string>();
  const pending = [...signal.causal_parent_ids];
  while (pending.length > 0) {
    const signalId = pending.pop()!;
    if (signalId === ancestorSignalId) return true;
    if (visited.has(signalId)) continue;
    visited.add(signalId);
    const ancestor = state.signals[signalId];
    if (ancestor) pending.push(...ancestor.causal_parent_ids);
  }
  return false;
}

function neuralSignalsShareAncestorKind(
  state: NeuralHierarchyState,
  left: NeuralSignal,
  right: NeuralSignal,
  kind: NeuralSignalKind
): boolean {
  const leftAncestors = new Set<string>();
  const leftVisited = new Set<string>();
  const leftPending = [left.signal_id];
  while (leftPending.length > 0) {
    const signalId = leftPending.pop()!;
    if (leftVisited.has(signalId)) continue;
    leftVisited.add(signalId);
    const signal = state.signals[signalId];
    if (!signal) continue;
    if (signal.kind === kind) leftAncestors.add(signalId);
    leftPending.push(...signal.causal_parent_ids);
  }
  if (leftAncestors.size === 0) return false;

  const visited = new Set<string>();
  const rightPending = [right.signal_id];
  while (rightPending.length > 0) {
    const signalId = rightPending.pop()!;
    if (visited.has(signalId)) continue;
    visited.add(signalId);
    const signal = state.signals[signalId];
    if (!signal) continue;
    if (signal.kind === kind && leftAncestors.has(signalId)) return true;
    rightPending.push(...signal.causal_parent_ids);
  }
  return false;
}

function directNeuralResultBindsChildOutput(
  runtime: HumanoidNeuralAgentRuntime,
  parsed: Record<string, unknown>,
  ancestorNodeId: string,
  ancestorKind: NeuralSignalKind
): boolean {
  const sourceSignalIds = z.array(z.string().uuid()).max(64).catch([]).parse(
    parsed.source_signal_ids
  );
  const state = runtime.neuralHierarchyState();
  return sourceSignalIds.some((sourceSignalId) => {
    const source = state.signals[sourceSignalId];
    if (!source) return false;
    // The Agent.asTool wrapper binds the returned child signal as one immediate
    // causal parent of the new direct edge. Do not scan arbitrary historical
    // ancestry: a later Skill may itself descend from an earlier Recovery epoch,
    // but that does not make its new Premotor failure a Recovery-authored result.
    return [source, ...source.causal_parent_ids.flatMap((signalId) => {
      const parent = state.signals[signalId];
      return parent ? [parent] : [];
    })].some((signal) => signal.source_node_id === ancestorNodeId
      && signal.kind === ancestorKind);
  });
}

function currentDirectPremotorRollout(
  runtime: HumanoidNeuralAgentRuntime,
  sensorimotorEpisodeId: string
): NeuralSignal {
  const rollouts = runtime.pendingNeuralSignals({
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
    kinds: ["rollout_result"]
  }).filter((signal) => signal.direction === "ascending"
    && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.premotor
    && signal.parent_episode_id === sensorimotorEpisodeId
    && isCurrentNeuralSignal(runtime, signal));
  if (rollouts.length !== 1) {
    throw new Error(
      `Predictive delegation requires one current direct Premotor rollout; found ${rollouts.length}`
    );
  }
  return rollouts[0]!;
}

function resolveReentrantRolloutAncestor(
  state: NeuralHierarchyState,
  ownedSignals: readonly NeuralSignal[],
  predictiveNodeId: string
): NeuralSignal {
  const activeCommitmentId = state.active_skill_commitment?.commitment_id ?? null;
  const ownedCommitmentIds = new Set(ownedSignals.map(
    (signal) => neuralSignalCommitmentId(state, signal)
  ).filter((commitmentId): commitmentId is string => commitmentId !== null));
  const candidates = Object.values(state.signals).filter((candidate) => (
    candidate.status === "pending"
      && candidate.kind === "rollout_result"
      && candidate.direction === "reentrant"
      && candidate.source_node_id === HUMANOID_NEURAL_AGENT_IDS.rolloutGate
      && candidate.target_node_id === predictiveNodeId
      && (activeCommitmentId === null
        || neuralSignalCommitmentId(state, candidate) === activeCommitmentId)
      && (ownedCommitmentIds.size === 0
        || ownedCommitmentIds.has(neuralSignalCommitmentId(state, candidate) ?? ""))
      // Rollout Gate emits two branches from one physical preview: the ordinary
      // ascending result is wrapped with Motor Intent planning state on its way
      // through Premotor, while the reentrant result intentionally retains the
      // raw receipt. Their payload hashes therefore differ by design. Bind the
      // branches at their shared rollout-result lineage so compatible models
      // never have to smuggle a cross-branch signal id through a parent Agent's
      // context.
      && candidate.causal_parent_ids.some((rolloutSignalId) => (
        ownedSignals.some((ownedSignal) => (
          ownedSignal.signal_id === rolloutSignalId
            || neuralSignalHasAncestorId(state, ownedSignal, rolloutSignalId)
        ))
      ))
  ));
  if (candidates.length !== 1) {
    throw new Error(
      `Predictive delegation requires one unique reentrant rollout ancestor; found ${candidates.length}`
    );
  }
  return candidates[0]!;
}

function neuralSignalCommitmentId(
  state: NeuralHierarchyState,
  signal: NeuralSignal
): string | null {
  const leaseId = signal.authority_lease_id ?? signal.source_authority_lease_id;
  return leaseId === null
    ? null
    : state.authority_leases[leaseId]?.commitment_id ?? null;
}

function neuralSkillCommitmentIsOpen(
  commitment: NeuralSkillCommitment | null
): boolean {
  return commitment !== null
    && !["completed", "failed", "released"].includes(commitment.state);
}

function boundedProposedSkillFromPayload(payload: JsonValue): string | undefined {
  const parsed = BoundedSkillProposalPayloadSchema.safeParse(payload);
  return parsed.success ? parsed.data.proposed_skill.skill : undefined;
}

function parseNeuralAgentFinalOutput<TOutput extends z.ZodObject>(
  outputSchema: TOutput,
  output: unknown
) {
  let candidate = output;
  if (typeof output === "string") {
    try {
      candidate = JSON.parse(output);
    } catch {
      return outputSchema.safeParse(output);
    }
  }
  return outputSchema.safeParse(candidate);
}

function neuralAgentTurnContinuationReceipt(
  childNodeId: string,
  childToolName: string
): string {
  return JSON.stringify({
    accepted: false,
    code: "neural_agent_turn_requires_tool",
    child_node_id: childNodeId,
    automatic_actuation: false,
    next_response_contract: {
      mode: "repeat_same_child_tool",
      tool: childToolName,
      narration_allowed: false
    },
    recovery: "The child SDK turn ended in assistant text without a verified terminal tool receipt. Invoke the same direct-child tool again so its independent Session can continue; do not reinterpret its prose as a neural signal."
  });
}

function neuralAgentToolTurnContinuationInput(
  runtime: HumanoidNeuralAgentRuntime,
  parentNodeId: string,
  childNodeId: string,
  invocationId: string
): string {
  return [
    neuralInvocationInput(runtime, parentNodeId, childNodeId, invocationId),
    "SDK_TURN_CONTINUATION=verified_tool_receipt_required",
    "Your preceding turn ended in assistant text, so it made no Harness transition. Continue this same invocation and call exactly one currently enabled formal tool now. Do not narrate, promise, or describe the call; emit the tool call itself."
  ].join("\n");
}

function isNeuralAgentTurnContinuationReceipt(
  output: Record<string, JsonValue>
): boolean {
  return output.accepted === false
    && output.code === "neural_agent_turn_requires_tool";
}

function outputObject(output: unknown): Record<string, JsonValue> | undefined {
  let parsed = output;
  if (typeof output === "string") {
    try {
      parsed = JSON.parse(output);
    } catch {
      return undefined;
    }
  }
  return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
    ? parsed as Record<string, JsonValue>
    : undefined;
}

function parseNeuralJsonText(value: string, label: string): JsonValue {
  try {
    return z.json().parse(JSON.parse(value));
  } catch (cause) {
    throw new Error(`${label} is not valid JSON`, { cause });
  }
}

function parseNeuralAgentOutput<T extends {
  signal_kind: NeuralSignalKind;
  summary: string;
  payload_json: string;
  source_signal_ids: string[];
  confidence: number;
}>(output: T): Omit<T, "payload_json"> & { payload: JsonValue } {
  const { payload_json: payloadJson, ...identity } = output;
  return {
    ...identity,
    payload: parseNeuralJsonText(payloadJson, "neural Agent output payload")
  };
}
