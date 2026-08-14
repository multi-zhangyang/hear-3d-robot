import {
  Agent,
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
  providerConfigForProfile,
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
  createHumanoidEmbodiedRecallTool,
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
  type HumanoidNeuralAgentId,
  type HumanoidNeuralAgentKey
} from "./neural-hierarchy-contract.js";
import {
  HUMANOID_EXPERIENCE_OUTCOMES,
  HUMANOID_GOAL_PREDICATE_TYPES,
  type HumanoidEmbodiedRecallRequest
} from "./embodied-recall.js";

const EmptyDelegationSchema = z.object({}).strict();
const NEURAL_OUTPUT_SUBMISSION_TOOL_NAME = "submit_neural_output";
const MODEL_EPISODE_SIGNAL_TTL_REVISIONS = 10_000;
const MOTOR_INTENT_PLANNING_ACTIONS: ReadonlySet<string> = new Set([
  "plan_humanoid_skill",
  "plan_whole_body_motion_candidates",
  "plan_humanoid_navigation"
]);
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
  goal_epoch_id: z.string().trim().min(1),
  skill: z.string().trim().min(1).max(2_000),
  termination_contract_json: NeuralJsonTextSchema,
  source_signal_ids: z.array(z.string().uuid()).min(1).max(64)
}).strict();
const TransitionSkillCommitmentSchema = z.object({
  commitment_id: z.string().uuid(),
  source_signal_ids: z.array(z.string().uuid()).min(1).max(64),
  reason: z.string().trim().min(1).max(2_000)
}).strict();
const AuthorizeSkillExecutionSchema = z.object({
  commitment_id: z.string().uuid(),
  reason: z.string().trim().min(1).max(2_000)
}).strict();
const CycleCompletionSchema = z.object({
  summary: z.string().trim().min(1).max(8_000),
  source_signal_ids: z.array(z.string().uuid()).min(1).max(64),
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
  source_signal_ids: z.array(z.string().uuid()).length(1)
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
  signal_kind: z.enum(["perceptual_belief", "escalation"])
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
  coordinatorPhase(): string;
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
}

export interface HumanoidNeuralAgentHierarchy extends Omit<
  NeuralAgentHierarchy,
  "root" | "agents"
> {
  root: Agent<unknown, typeof ExecutiveOutputSchema>;
  agents: ReadonlyMap<string, Agent<any, any>>;
  agent(agentId: HumanoidNeuralAgentId): Agent<any, any> | undefined;
}

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
  const ownModel = (key: HumanoidNeuralAgentKey): Model => {
    const agentId = HUMANOID_NEURAL_AGENT_IDS[key];
    const profile = humanoidNeuralAgentProfile(agentId);
    const model = input.createModel(
      agentId,
      providerConfigForProfile(input.provider, profile)
    );
    if (models.has(model)) {
      throw new Error(`Neural Agents cannot share one Model facade: ${agentId}`);
    }
    models.add(model);
    return model;
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
      thinking?: "enabled" | "disabled";
      toolChoice?: "auto" | "required";
    } = {}
  ): ModelSettings => {
    const provider = providerConfigForProfile(
      input.provider,
      humanoidNeuralAgentProfile(HUMANOID_NEURAL_AGENT_IDS[key])
    );
    // Every neural node advances the control graph through a formal tool: a
    // child delegation, a state mutation, or submit_neural_output. `auto`
    // lets a compatible model terminate in prose and silently bypass that
    // graph, so the neural Harness itself requires tool use on every turn.
    const toolChoice = options.toolChoice ?? "required";
    const deepSeekCompatible = provider.protocol === "openai_compatible"
      && provider.model.toLowerCase().includes("deepseek");
    const thinking = options.thinking
      ?? (deepSeekCompatible && toolChoice === "required" ? "disabled" : "enabled");
    return {
      temperature: provider.temperature,
      ...(provider.reasoningEffort === undefined
        ? {}
        : { reasoning: { effort: provider.reasoningEffort } }),
      ...(provider.maxOutputTokens === undefined
        ? {}
        : { maxTokens: provider.maxOutputTokens }),
      ...(deepSeekCompatible
        ? {
            // DeepSeek thinking rejects tool_choice=required. Keep the same
            // model and disable thinking only for these formal control turns,
            // matching its OpenAI-compatible transport contract.
            providerData: {
              thinking: { type: thinking },
              providerOptions: {
                "configured-openai-compatible": {
                  thinking: { type: thinking }
                }
              }
            }
          }
        : {}),
      parallelToolCalls: options.parallel ?? false,
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
      extraInstructions?: readonly string[];
      toolUseBehavior?: ToolUseBehavior;
    } = {}
  ): Agent<unknown, TOutput> => {
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
    const agent = new Agent({
      name: descriptor.name,
      instructions: scopedInstructions(descriptor.id, [
        ...baseInstructions(descriptor.key),
        ...(options.extraInstructions ?? [])
  ]),
      model: ownModel(key),
      modelSettings: settingsFor(key, options),
      tools: [...tools, submissionTool],
      outputType,
      resetToolChoice: false,
      toolUseBehavior: neuralOutputToolUseBehavior(
        outputType,
        options.toolUseBehavior
      )
    });
    agents.set(descriptor.id, agent);
    return agent;
  };
  const asChildTool = <TOutput extends z.ZodObject>(inputTool: {
    parentKey: HumanoidNeuralAgentKey;
    childKey: HumanoidNeuralAgentKey;
    child: Agent<unknown, TOutput>;
    description: string;
    isEnabled?: () => boolean;
    phases?: readonly NeuralHarnessPhase[];
    requireCommitment?: boolean;
  }): FunctionTool<unknown, any, unknown> => {
    const parentId = HUMANOID_NEURAL_AGENT_IDS[inputTool.parentKey];
    const childId = HUMANOID_NEURAL_AGENT_IDS[inputTool.childKey];
    const childDescriptor = HUMANOID_NEURAL_NODE_BY_ID.get(childId)!;
    const delegationSchema = neuralDelegationSchema(
      inputTool.parentKey,
      inputTool.childKey
    );
    const nestedParallelism = childDescriptor.key === "perceptionManager"
      || childDescriptor.key === "sensorimotorManager";
    const maximumToolConcurrency = nestedParallelism ? 2 : 1;
    let descendingSignal: NeuralSignal | undefined;
    let authorityLease: NeuralAuthorityLease | undefined;
    let invocationInputSignalIds: string[] = [];
    const invocationMutex = new Mutex();
    const childTool = scopeAgentToolInvocation(
      childId,
      inputTool.child.asTool({
      toolName: humanoidNeuralAgentToolName(inputTool.childKey),
      toolDescription: inputTool.description,
      parameters: delegationSchema,
      inputBuilder: async ({ params }) => {
        const invocation = requiredHarnessInvocation(childId);
        await prepareHarnessPhaseForChild(
          input.runtime,
          inputTool.parentKey,
          inputTool.childKey
        );
        const activeCommitment = input.runtime.neuralHierarchyState()
          .active_skill_commitment;
        const attachOwnedCommitment = inputTool.parentKey === "actionSelection"
          && inputTool.childKey === "sensorimotorManager"
          && activeCommitment !== null
          && !["completed", "failed", "released"].includes(activeCommitment.state)
          && params.signal_kind !== "skill_commitment";
        const parentEpisodeId = requiredParentEpisodeId(parentId);
        const suppliedSourceSignals = params.source_signal_ids.map(
          (signalId) => input.runtime.neuralHierarchyState().signals[signalId]
        );
        if (suppliedSourceSignals.some((signal) => signal === undefined)) {
          throw new Error("Neural delegation references an unknown source signal");
        }
        const exactSourceSignals = inputTool.parentKey === "sensorimotorManager"
          && inputTool.childKey === "predictive"
          && params.signal_kind === "rollout_result"
          ? [currentDirectPremotorRollout(input.runtime, parentEpisodeId)]
          : suppliedSourceSignals as NeuralSignal[];
        const currentBelief = inputTool.parentKey === "actionSelection"
          && inputTool.childKey === "sensorimotorManager"
          ? currentActionSelectionBelief(
              input.runtime,
              exactSourceSignals as NeuralSignal[]
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
          && input.runtime.neuralHarnessPhase().phase === "feedback") {
          const hierarchyState = input.runtime.neuralHierarchyState();
          const directCompletion = Object.values(hierarchyState.signals).filter(
            (signal) => signal.kind === "skill_completed"
              && signal.source_node_id
                === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
              && signal.target_node_id === parentId
              && signal.status === "pending"
              && exactSourceSignals.some((source) => source !== undefined
                && (source.signal_id === signal.signal_id
                  || neuralSignalHasAncestorId(
                    hierarchyState,
                    source,
                    signal.signal_id
                  )))
          );
          if (directCompletion.length !== 1) {
            throw new Error(
              `Post-execution Perception requires one causally routed Sensorimotor completion; found ${directCompletion.length}`
            );
          }
          // A new Action Selection episode sees only Executive's direct signal.
          // Resolve the previous Sensorimotor edge inside the Harness and bind
          // it into the next descending signal; never ask the model to quote a
          // grandchild's signal id across episodes.
          routedCausalParentIds.push(directCompletion[0]!.signal_id);
        }
        const descendingPayload = inputTool.parentKey === "actionSelection"
          && inputTool.childKey === "sensorimotorManager"
          && activeCommitment !== null
          && params.signal_kind === "skill_commitment"
          ? activeCommitment
            : {
              intent: params.intent,
              causal_inputs: exactSourceSignals.map((signal) => ({
                signal_id: signal!.signal_id,
                kind: signal!.kind,
                source_node_id: signal!.source_node_id,
                world_revision: signal!.world_revision,
                payload: causalSemanticProjection(signal!.payload)
              }))
            };
        authorityLease = await input.runtime.issueNeuralAuthorityLease({
          issuingParentNodeId: parentId,
          targetChildNodeId: childId,
          allowedSignalKinds: [...new Set<NeuralSignalKind>([
            params.signal_kind,
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
        const output = inputTool.child.outputType.parse(finalOutput);
        return JSON.stringify(output);
      },
      ...(input.onAgentStream
        ? {
            onStream: ({ event }) => input.onAgentStream!(
              HUMANOID_NEURAL_AGENT_IDS[inputTool.childKey],
              event
            )
          }
        : {})
      })
    );
    // A hierarchy child failure is a control-path failure, not model-visible
    // prose. The SDK function-tool default converts exceptions into
    // "An error occurred...", which destroys the originating node and makes
    // the parent parse an error sentence as a neural JSON envelope.
    childTool.errorFunction = null;
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
      const delegation = delegationSchema.parse(JSON.parse(rawInput));
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
      try {
        const output = await invoke(context, rawInput, details);
        const childOutput = outputObject(output);
        if (!childOutput) {
          throw new Error(
            `${childId} returned a non-neural Agent.asTool result: ${String(output)}`
          );
        }
        const childSpecificOutput = inputTool.child.outputType.parse(childOutput);
        const routingOutput = NeuralAgentOutputSchema.passthrough().parse(
          childSpecificOutput
        );
        const parsed = parseNeuralAgentOutput(
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
        const causality = [...new Set([
          ...(descendingSignal ? [descendingSignal.signal_id] : []),
          ...parsed.source_signal_ids
        ])];
        for (const joined of requireManagerJoinEvidence(
          input.runtime,
          inputTool.childKey,
          parsed.signal_kind,
          invocationId,
          parsed.source_signal_ids
        )) {
          causality.push(joined.signal_id);
        }
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
            const activeCertificate = Object.values(
              input.runtime.neuralHierarchyState().rollout_certificates
            ).find((certificate) => certificate.status === "active"
              && certificate.commitment_id
                === input.runtime.neuralHierarchyState().active_skill_commitment?.commitment_id
              && parsed.source_signal_ids.includes(certificate.predictive_signal_id)
              && certificate.rollout_payload_sha256
                === modelPayloadSha256(forwardedRollout.payload));
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
            && input.runtime.coordinatorPhase() === "complete_cycle"
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
          payload: parsed.payload
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
          // Rebind the child result to the one direct edge the parent owns.
          // The child's internal source ids remain durable causal parents in
          // Harness state; exposing both sets made compatible models select
          // the wrong authority namespace on the next call.
          source_signal_ids: [ascendingSignal.signal_id]
        });
      } finally {
        if (authorityLease) {
          await input.runtime.closeNeuralAuthorityLease({
            leaseId: authorityLease.lease_id,
            closedByNodeId: parentId,
            reason: "parent_child_invocation_returned"
          });
        }
        authorityLease = undefined;
        descendingSignal = undefined;
        invocationInputSignalIds = [];
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
      toolChoice: "required",
      toolUseBehavior: goalValuationToolUseBehavior(),
      extraInstructions: [
        "Use the existing Goal DAG tools for every formal Goal mutation.",
        "After a terminal Goal tool result, return goal_selected or escalation as structured output."
      ]
    }
  );
  const sceneInterpreter = register(
    "sceneInterpreter",
    SceneInterpretationOutputSchema,
    [],
    { extraInstructions: ["Return scene_interpretation only from current sensory evidence."] }
  );
  const memoryRetriever = register(
    "memoryRetriever",
    MemoryRetrievalOutputSchema,
    [relevantMemoryRecallTool(input.runtime)],
    {
      toolUseBehavior: relevantMemoryToolUseBehavior(),
      extraInstructions: [
        "Choose one high-level retrieval intent. The Harness constructs the legal storage query and the typed memory_retrieval result ends this episode.",
        "Use active_goal for normal perception and post-execution feedback. Use recent only when the parent explicitly requests a chronological view.",
        "Historical records never replace current sensing. An empty recall result is a valid final retrieval result."
      ]
    }
  );
  const perceptionManager = register(
    "perceptionManager",
    PerceptionOutputSchema,
    [
      sensorFusionTool(input.runtime),
      asChildTool({
        parentKey: "perceptionManager",
        childKey: "sceneInterpreter",
        child: sceneInterpreter,
        description: "Interpret the current authoritative sensory signal.",
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
      }),
      asChildTool({
        parentKey: "perceptionManager",
        childKey: "memoryRetriever",
        child: memoryRetriever,
        description: "Retrieve only relevant historical embodied experience.",
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
      })
    ],
    {
      parallel: true,
      extraInstructions: [
        "Call capture_sensor_fusion before any interpretation.",
        "After sensing, fan out scene and memory tools in one response; both branches are mandatory for perceptual_belief.",
        "Return perceptual_belief only after copying the Sensor Fusion, Scene, and Memory source_signal_ids values into source_signal_ids.",
        "Perception owns state estimation, not action selection: report every live reachable-base candidate verbatim, but never recommend or select a Skill, hand, interaction point, motor program, or next action. Action Selection and Sensorimotor own those decisions.",
        "Never send one sibling's raw output or Session to the other."
      ]
    }
  );
  const affordance = register(
    "affordance",
    AffordanceOutputSchema,
    [],
    { extraInstructions: [
      "Return affordance_hypothesis for the committed Goal only.",
      "Evaluate live reachable_base_placements as exact atomic tuples: interaction_point_id, hand_surface, root_world_target, and root_yaw_radians must stay together. Never combine a hand or interaction point from history with the geometry of another live tuple.",
      "Historical failures are evidence about the failed tuple, not authority to overwrite a different current geometry candidate."
    ] }
  );
  const risk = register(
    "risk",
    RiskOutputSchema,
    [],
    { extraInstructions: [
      "During skill_proposal there is no committed Skill yet. Assess current balance, contact, collision, and environmental bounds without inventing or selecting a Skill, hand, interaction point, or motor program.",
      "Return risk_assessment when at least one lower-level option may proceed under stated bounds.",
      "Return escalation when risk requires inhibition or Recovery; never invent an alternate motor command."
    ] }
  );
  const predictive = register(
    "predictive",
    PredictiveOutputSchema,
    [],
    { extraInstructions: [
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
      toolChoice: "required",
      toolUseBehavior: planningReceiptToolUseBehavior(),
      extraInstructions: [
        "Complete the existing embodied Skill lifecycle in this one SDK episode: when planning_tool_state requires submit_humanoid_skill_plan, submit the committed short Skill DAG; when it requires begin_humanoid_skill, copy one ready_skill_binding verbatim; only then call the enabled semantic planning tool with the real bound skill_transaction_id.",
        "submit_humanoid_skill_plan and begin_humanoid_skill are lifecycle transitions, not rollout results. Continue after each accepted transition and inspect the next planning_tool_state exposed by the Harness.",
        "Compile object-relative preparation with object-relative Skills. approach(object_id=...) is the navigation Skill that moves the base to a manipulation stance chosen from live reachable_base_placements; navigate_to_zone moves only the robot root into a semantic zone and cannot prepare an uncarried object for grasping or placement.",
        "For an object_placed termination contract, preserve one causal object chain in the Skill DAG: an uncarried object needs an object-targeted approach/reach/grasp/lift path before carry_to_zone/place. You choose the hand, interaction point, standoff, and exact bounded nodes from current geometry; never substitute the destination zone for the source object in the first ready node.",
        "If Skill-plan admission rejects a ready node, change the contradictory invocation before retrying. Repeating the same rejected Skill and parameters is not recovery.",
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
        "Compose one short Skill DAG, then delegate one bounded motor intent.",
        "The Motor Intent tool's typed rollout_result or escalation directly completes this episode; do not resubmit or relabel it."
      ]
    }
  );
  const recovery = register(
    "recovery",
    RecoveryOutputSchema,
    [singleRecallTool(input.runtime, "recovery")],
    {
      toolUseBehavior: actionSelectionToolUseBehavior(input.runtime),
      extraInstructions: [
        "This is an independent bounded recovery episode under a Harness authority lease.",
        "The current failure receipt is already in your directed input. If you recall analogous successful experience, use query_mode=semantic and source_refs=null; use query_mode=chronological_or_exact only when reading an exact historical source, never combine both modes.",
        "Return a recovery skill_proposal or escalation. You never write physical state."
      ]
    }
  );
  const sensorimotorManager = register(
    "sensorimotorManager",
    SensorimotorOutputSchema,
    [
      asChildTool({
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
      }),
      asChildTool({
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
      }),
      asChildTool({
        parentKey: "sensorimotorManager",
        childKey: "premotor",
        child: premotor,
        description: "Compose and compile one bounded motor skill after assessment.",
        phases: ["motor_assessment", "motor_planning"],
        requireCommitment: true,
        isEnabled: () => !hasCurrentRecoveryDemand(input.runtime)
          && hasCurrentManagerEpisodeSignal(
            input.runtime,
            HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            "skill_commitment"
          )
      }),
      asChildTool({
        parentKey: "sensorimotorManager",
        childKey: "predictive",
        child: predictive,
        description: "Interpret a completed MuJoCo rollout result.",
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
      serialExecutionTool(input.runtime)
    ],
    {
      parallel: true,
      toolUseBehavior: sensorimotorToolUseBehavior(input.runtime),
      extraInstructions: [
        "Affordance and Risk are the only parallel pre-action fan-out; join both yourself.",
        "A skill_proposal must cite the current perceptual_belief plus both Affordance and Risk harness signal ids in source_signal_ids.",
        "Return exactly one next bounded catalog Skill as proposed_skill={skill, phase, params, rationale}. Future steps may appear only in phase_sequence; never make a compound skill_name or the whole task the proposal.",
        "For an object placement Goal whose object is not grasped, preparation is object-relative: propose approach(object_id), then reach/grasp/lift, before carry_to_zone/place. Never approach the destination zone to make the source object reachable.",
        "For approach, params are exactly object_id, interaction_point_id, hand, and standoff_m. Preserve the selected live hand+interaction-point pair, but never add root_world_target, root_yaw_radians, or base_placement; the Harness binds that pair to its authoritative IK placement.",
        "Premotor waits for Action Selection to accept that joined proposal as the active commitment. When the current phase is motor_assessment or motor_planning and the invocation contains the direct skill_commitment, call Premotor immediately; Affordance/Risk belong only to skill_proposal and must not be repeated. Predictive waits for a real rollout result.",
        "Predictive judges admission of the current bounded rollout chunk, not whether that chunk already completes the final Goal predicate.",
        "Predictive acceptance completes the motor-assessment episode and returns a certified rollout to Action Selection; do not call or synthesize execution in that episode.",
        "Call execute_certified_motor_intent only in a later execution-phase invocation after Action Selection has moved the commitment to executing. Pass only the one direct skill_commitment signal from the current invocation; the Harness resolves its certificate and plan.",
        "Recovery freezes normal selection, runs one independent episode, and returns before execution resumes."
      ]
    }
  );
  const actionSelection = register(
    "actionSelection",
    ActionSelectionOutputSchema,
    [
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
        isEnabled: () => input.runtime.neuralHarnessPhase().phase !== "recovery"
          || input.runtime.neuralHierarchyState().active_skill_commitment === null
          || ["completed", "failed", "released"].includes(
            input.runtime.neuralHierarchyState().active_skill_commitment!.state
          )
      })
    ],
    {
      toolUseBehavior: actionSelectionToolUseBehavior(input.runtime),
      extraInstructions: [
        "Treat neural_hierarchy.harness_phase.phase as binding route authority: in perception or feedback, delegate Perception Manager before any Sensorimotor call; Sensorimotor is not a valid child route until the Harness itself enters skill_proposal, motor_assessment, motor_planning, rollout_review, execution, or recovery.",
        "Perception precedes sensorimotor selection whenever the belief is absent or stale.",
        "Sensorimotor may return only a skill_proposal before authorization.",
        "A skill_proposal must select exactly one next bounded catalog Skill in proposed_skill.skill. A multi-step skill_sequence may describe future causal order, but skill_name, skill_id, or the whole sequence is never the proposed Skill commitment.",
        "You alone establish one durable skill commitment. Copy the outer delegate_sensorimotor_manager.source_signal_ids into establish_skill_commitment.source_signal_ids; nested Affordance/Risk ids do not authorize commitment. Then invoke Sensorimotor again with the resulting skill_commitment signal.",
        "A replacement Sensorimotor proposal supersedes every earlier proposal in this Action Selection episode. Establish only the newest direct proposal and its exact bounded Skill; never restore a rejected proposal because it matched the previous commitment or intention.",
        "Authorize execution only after the committed branch returns a real accepted rollout_result. Call authorize_skill_execution with the active commitment id and your reason; the Harness binds the unique direct certified rollout, so do not copy a rollout or Predictive signal id into that call.",
        "A Premotor, planning, or Predictive escalation is a local recovery demand, not a Recovery escalation. First call release_skill_commitment with the direct Sensorimotor failure signal, then delegate that same direct signal back to Sensorimotor so its exclusive Recovery child can propose a replacement. Only an escalation returned by Recovery may propagate to Executive.",
        "A successful physical chunk is not automatically Skill completion. complete_skill_commitment is enabled only when the exact committed Skill binding has a successful physical receipt and its authoritative Skill-plan node postcondition is complete. Model-authored prose in the termination contract cannot override that lifecycle. Otherwise release the exhausted bounded plan, obtain fresh Perception, and continue the same Goal through a new bounded Skill commitment.",
        "In recovery, forward the direct failure signal to Sensorimotor. When the failure came from physical execution, also cite the new post-failure perceptual_belief returned after you closed the old commitment. If Recovery returns a replacement skill_proposal, you alone replace the failed commitment; if it escalates, return that typed escalation unchanged to Executive.",
        "In feedback, complete or fail the active commitment before delegating Perception. After completed execution, forward perceptual_belief to Executive. After failed execution, use the new belief to enter Recovery while preserving the active Goal.",
        "Children may not replace the active Goal or establish their own commitment."
      ]
    }
  );
  const executive = register(
    "executive",
    ExecutiveOutputSchema,
    [
      neuralCycleCompletionTool(input.runtime),
      neuralSatisfiedGoalCompletionTool(input.runtime),
      asChildTool({
        parentKey: "executive",
        childKey: "goalManager",
        child: goalManager,
        description: "Value, select, continue, or retire the current Goal.",
        phases: ["bootstrapping", "goal_valuation", "cycle_completion"]
      }),
      asChildTool({
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
          "recovery",
          "cycle_completion"
        ]
      })
    ],
    {
      toolUseBehavior: executiveToolUseBehavior(),
      extraInstructions: [
        "You are the only root. Goal Valuation and Action Selection are children, never peers.",
        "Do not wake lower layers unless an event requires work; never poll all Agents.",
        "A typed direct child return ends this Executive episode; the Harness scheduler starts the next phase in a fresh event. Never relabel or resubmit a child result.",
        "In recovery, preserve the active Goal and delegate the direct failure signal through Action Selection; after physical failure, Action Selection must first close the old commitment and obtain current Perception before Recovery is reachable through Sensorimotor.",
        "After execution, delegate Action Selection once to resolve its commitment, then again for current Perception when the Harness requests it. A failed execution continues from that fresh belief into Recovery, not directly from stale pre-action state.",
        "After post-execution Sensor Fusion, cite the direct perceptual_belief and decide whether to complete; the Harness resolves exact physical transaction ids. Never self-certify success."
      ]
    }
  );

  const services = createRuntimeServices();
  return {
    root: executive,
    agents,
    services,
    session: (agentId) => sessions.get(agentId),
    agent: (agentId) => agents.get(agentId)
  };
}

function singleRecallTool(
  runtime: HumanoidEmbodiedRecallInvoker,
  key: "memoryRetriever" | "recovery"
): FunctionTool<unknown, any, string> {
  const agentId = HUMANOID_NEURAL_AGENT_IDS[key];
  const recall = createHumanoidEmbodiedRecallTool(runtime);
  let completedEpisodeId: string | undefined;
  const sdkEnabled = recall.isEnabled;
  recall.isEnabled = async (context, agent) => {
    const invocation = currentAgentHarnessInvocation();
    const episodeId = invocation?.parentInvocationId ?? invocation?.invocationId;
    return invocation?.agentId === agentId
      && episodeId !== completedEpisodeId
      && await sdkEnabled(context, agent);
  };
  const invoke = recall.invoke;
  recall.invoke = async (context, rawInput, details) => {
    const invocation = requiredHarnessInvocation(agentId);
    // SDK tool calls receive their own invocation marker.  The structural
    // parent episode is the stable identity shared by all model turns in this
    // child Agent run, so gate recall on that identity rather than on each new
    // tool-call marker.
    const episodeId = invocation.parentInvocationId ?? invocation.invocationId;
    if (completedEpisodeId === episodeId) {
      throw new Error(`${agentId} already completed its one bounded recall`);
    }
    const output = await invoke(context, rawInput, details);
    const record = outputObject(output);
    if (record?.accepted !== false) {
      completedEpisodeId = episodeId;
    }
    return output;
  };
  return recall;
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
        const output = await invokeDeterministicHumanoidAction({
          runtime,
          actorAgentId: HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
          sourceToolName: name,
          sourceInput: {},
          action: "observe_humanoid",
          actionInput: {},
          contractId: "grounding_monitor_v1",
          ...(details ? { details } : {})
        });
        const receiptPayload = z.json().parse(JSON.parse(output));
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
      const proposalSignals = currentManagerChildSignals(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        ["skill_proposal"]
      ).filter((signal) => signal.direction === "ascending"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection)
        .sort((left, right) => right.sequence - left.sequence);
      const latestProposal = proposalSignals[0];
      const citedProposalSignals = proposalSignals.filter((signal) => (
        params.source_signal_ids.includes(signal.signal_id)
      ));
      const staleCitedProposals = citedProposalSignals.filter(
        (signal) => signal.signal_id !== latestProposal?.signal_id
      );
      if (latestProposal && staleCitedProposals.length > 0) {
        const requiredSkill = boundedProposedSkillFromPayload(latestProposal.payload);
        return JSON.stringify({
          accepted: false,
          code: "stale_sensorimotor_proposal",
          tool: "establish_skill_commitment",
          action_selection_episode_id: invocation.invocationId,
          latest_proposal_signal_id: latestProposal.signal_id,
          latest_proposed_skill: requiredSkill ?? null,
          rejected_proposal_signal_ids: staleCitedProposals.map(
            (signal) => signal.signal_id
          ),
          rejected_source_signal_ids: params.source_signal_ids,
          automatic_actuation: false,
          next_response_contract: {
            mode: "corrected_tool_call_only",
            tool: "establish_skill_commitment",
            required_source_signal_ids: [latestProposal.signal_id],
            required_skill: requiredSkill ?? null,
            preserve_goal_epoch_id: true,
            preserve_termination_contract: true,
            narration_allowed: false
          },
          recovery: "The cited Sensorimotor proposal was superseded inside this Action Selection episode. Retry once using only the latest direct Sensorimotor proposal signal id and its bounded Skill; do not reuse an earlier rejected proposal."
        });
      }
      const citedProposal = latestProposal
        && params.source_signal_ids.includes(latestProposal.signal_id)
        ? latestProposal
        : undefined;
      if (!citedProposal) {
        return JSON.stringify({
          accepted: false,
          code: "sensorimotor_proposal_not_cited",
          tool: "establish_skill_commitment",
          action_selection_episode_id: invocation.invocationId,
          current_proposal_signal_ids: proposalSignals.map(
            (signal) => signal.signal_id
          ),
          rejected_source_signal_ids: params.source_signal_ids,
          automatic_actuation: false,
          next_response_contract: {
            mode: "corrected_tool_call_only",
            tool: "establish_skill_commitment",
            preserve_valid_fields: true,
            narration_allowed: false
          },
          recovery: "Call establish_skill_commitment once more, preserving the Goal, Skill, and termination contract, and include the exact current proposal signal id returned in delegate_sensorimotor_manager.source_signal_ids. Nested Affordance/Risk source ids do not authorize a commitment."
        });
      }
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
        await runtime.transitionNeuralHarnessPhase({
          phase: "skill_proposal",
          enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
          reason: "skill_proposal_admission_rejected",
          commitmentId: null
        });
        return JSON.stringify({
          accepted: false,
          code: "skill_proposal_goal_misaligned",
          tool: "establish_skill_commitment",
          rejected_proposal_signal_id: citedProposal.signal_id,
          rejected_invocation: admission.invocation ?? null,
          reason: admission.reason ?? "The proposed Skill does not advance the active Goal",
          automatic_actuation: false,
          next_response_contract: {
            mode: "new_sensorimotor_proposal_required",
            preserve_goal_epoch_id: true,
            narration_allowed: false
          },
          recovery: "Do not establish this commitment. Delegate Sensorimotor again for one bounded Skill whose real invocation advances the active Goal or establishes its next physical prerequisite."
        });
      }
      const commitment = await runtime.establishNeuralSkillCommitment({
        ownerNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        goalEpochId: params.goal_epoch_id,
        skill: params.skill,
        terminationContract: parseNeuralJsonText(
          params.termination_contract_json,
          "Skill termination contract"
        ),
        sourceSignalIds: params.source_signal_ids
      });
      return JSON.stringify({
        status: "skill_committed",
        commitment,
        source_signal_ids: params.source_signal_ids
      });
    }
  });
}

function neuralCycleCompletionTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof CycleCompletionSchema, string> {
  return tool({
    name: "complete_neural_autonomous_cycle",
    description: "Executive closes one physical cycle after citing the direct post-execution belief. The Harness resolves and validates the exact physical evidence transactions.",
    parameters: CycleCompletionSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    isEnabled: () => {
      const completion = runtime.cycleCompletionReadiness();
      return runtime.neuralHarnessPhase().phase === "cycle_completion"
        && completion.status === "ready"
        && completion.observed_after_execution
        && runtime.coordinatorPhase() === "complete_cycle";
    },
    execute: (params) => {
      const readiness = runtime.cycleCompletionReadiness();
      if (readiness.status !== "ready" || !readiness.observed_after_execution) {
        throw new Error("Executive cycle completion is no longer ready");
      }
      const evidenceTransactionIds = readiness.evidence_transaction_ids;
      const execution = runtime.validateCycleEvidence(evidenceTransactionIds);
      const postExecutionBeliefs = runtime.pendingNeuralSignals({
        targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
        kinds: ["perceptual_belief"]
      }).filter((signal) => (
        params.source_signal_ids.includes(signal.signal_id)
          && signal.world_revision >= execution.worldAfterRevision
      ));
      if (postExecutionBeliefs.length === 0) {
        throw new Error(
          "Executive cycle completion requires the exact perceptual belief observed after the durable physical execution"
        );
      }
      return JSON.stringify({
        status: "cycle_completed",
        summary: params.summary,
        evidence_transaction_ids: evidenceTransactionIds,
        source_signal_ids: params.source_signal_ids,
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
      && runtime.coordinatorPhase() === "complete_satisfied_goal",
    execute: (params) => JSON.stringify({
      status: "satisfied_goal_completed",
      summary: params.summary,
      verification: runtime.validateSatisfiedGoal()
    })
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
      if (!params.source_signal_ids.some((signalId) => pendingIds.has(signalId))) {
        throw new Error("Commitment transition requires current child feedback");
      }
      const requiredKind: Partial<Record<typeof state, NeuralSignalKind>> = {
        completed: "skill_completed",
        failed: "skill_failed"
      };
      const kind = requiredKind[state];
      if (kind && !pending.some((signal) => (
        signal.kind === kind && params.source_signal_ids.includes(signal.signal_id)
      ))) {
        throw new Error(`Commitment ${state} requires a current ${kind} signal`);
      }
      const commitment = await runtime.transitionNeuralSkillCommitment({
        ownerNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        commitmentId: params.commitment_id,
        state,
        sourceSignalIds: params.source_signal_ids
      });
      return JSON.stringify({
        status: `skill_${state}`,
        commitment,
        source_signal_ids: params.source_signal_ids
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
        throw new Error(
          `Execution authorization requires one active Predictive certificate; found ${certificates.length}`
        );
      }
      const certificate = certificates[0]!;
      const rollouts = currentManagerChildSignals(
        runtime,
        HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        ["rollout_result"]
      ).filter((signal) => signal.source_node_id
          === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && signal.causal_parent_ids.includes(certificate.predictive_signal_id)
        && modelPayloadSha256(signal.payload) === certificate.rollout_payload_sha256);
      if (rollouts.length !== 1) {
        throw new Error(
          `Execution authorization requires one direct certified Sensorimotor rollout; found ${rollouts.length}`
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

function serialExecutionTool(
  runtime: HumanoidNeuralAgentRuntime
): FunctionTool<unknown, typeof ExecutionTaskSchema, string> {
  const name = humanoidNeuralAgentToolName("executor");
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
        HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        "skill_commitment"
      )
      && Object.values(runtime.neuralHierarchyState().rollout_certificates).filter(
        (candidate) => candidate.status === "active"
          && candidate.commitment_id === runtime.neuralHierarchyState()
            .active_skill_commitment?.commitment_id
      ).length === 1,
    execute: async (params, _context, details) => {
      const managerInvocation = requiredHarnessInvocation(
        HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
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
      const directCommitmentSignals = params.source_signal_ids.map(
        (signalId) => hierarchy.signals[signalId]
      ).filter((signal): signal is NeuralSignal => signal !== undefined
        && signal.status === "pending"
        && signal.kind === "skill_commitment"
        && signal.direction === "descending"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && signal.invocation_id === managerInvocation.invocationId);
      if (directCommitmentSignals.length !== 1) {
        throw new Error(
          "Serial execution requires the one direct Action Selection skill_commitment signal from this episode"
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
        rollout_certificate_id: certificate.certificate_id,
        execution
      };
      const lease = await runtime.issueNeuralAuthorityLease({
        issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
        allowedSignalKinds: ["skill_commitment", "motor_intent", "rollout_result"],
        ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
        ttlMs: 180_000,
        invocationId: executorInvocation.invocationId,
        parentInvocationId: executorInvocation.parentInvocationId,
        parentEpisodeId: requiredParentEpisodeId(
          HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
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
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
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
        const skillCompleted = lowerLoop.completionSignal.kind === "skill_completed";
        const causalParentIds = [lowerLoop.executionReceiptSignal.signal_id];
        const executionReceipt = await runtime.publishNeuralSignal({
          kind: "execution_receipt",
          pathway: "physical_execution",
          direction: "ascending",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 90,
          causalParentIds,
          sourceAuthorityLeaseId: lease.lease_id,
          invocationId: lease.invocation_id,
          parentInvocationId: lease.parent_invocation_id,
          payload: {
            ...(jsonRecord(receiptPayload) ?? {}),
            execution_transaction_id: executionTransactionId,
            lower_motor_loop: lowerLoop.summary
          }
        });
        let executionPredictionError: NeuralSignal | undefined;
        if (lowerLoop.predictionErrorSignal) {
          executionPredictionError = await runtime.publishNeuralSignal({
            kind: "prediction_error",
            pathway: "ascending_feedback",
            direction: "ascending",
            sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
            priority: 100,
            causalParentIds: [lowerLoop.predictionErrorSignal.signal_id],
            sourceAuthorityLeaseId: lease.lease_id,
            invocationId: lease.invocation_id,
            parentInvocationId: lease.parent_invocation_id,
            payload: lowerLoop.predictionErrorSignal.payload
          });
        }
        const completionSignal = await runtime.publishNeuralSignal({
          kind: skillCompleted ? "skill_completed" : "skill_failed",
          pathway: "physical_execution",
          direction: "ascending",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 95,
          causalParentIds: [...new Set([
            lowerLoop.completionSignal.signal_id,
            executionReceipt.signal_id,
            ...(executionPredictionError ? [executionPredictionError.signal_id] : [])
          ])],
          sourceAuthorityLeaseId: lease.lease_id,
          invocationId: lease.invocation_id,
          parentInvocationId: lease.parent_invocation_id,
          payload: receiptPayload
        });
        await runtime.transitionNeuralHarnessPhase({
          // A physical failure must first return through Action Selection so
          // it can close the executing commitment. Recovery starts only after
          // a new Sensor Fusion observation is causally bound to that failure.
          phase: "feedback",
          enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          reason: skillCompleted
            ? "physical_execution_completed"
            : "physical_execution_failed_requires_commitment_resolution"
        });
        const outputRecord = outputObject(output);
        if (!outputRecord) {
          throw new Error("Serial Executor returned a non-object physical receipt");
        }
        return JSON.stringify({
          ...outputRecord,
          source_signal_ids: [
            executionReceipt.signal_id,
            ...(executionPredictionError ? [executionPredictionError.signal_id] : []),
            completionSignal.signal_id
          ]
        });
      } finally {
        await runtime.closeNeuralAuthorityLease({
          leaseId: lease.lease_id,
          closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          reason: "serial_execution_returned"
        });
      }
      }, typeof details?.resumeState === "string", stableAgentToolInvocationId(
        HUMANOID_NEURAL_AGENT_IDS.executor,
        details?.toolCall?.callId
      ));
    }
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
        const relevant = recoveryDemands.find((signal) => signal.direction === "descending"
          && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
          && signal.invocation_id === recoveryInvocation.parentInvocationId);
        const currentBelief = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          kinds: ["perceptual_belief"]
        }).find((signal) => signal.direction === "descending"
          && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
          && signal.invocation_id === recoveryInvocation.parentInvocationId
          && isCurrentNeuralSignal(runtime, signal));
        const hierarchyState = runtime.neuralHierarchyState();
        const freshDurableFailureBelief = !relevant
          && currentBelief
          && runtime.coordinatorPhase() === "replan_or_retire"
          && !neuralSkillCommitmentIsOpen(hierarchyState.active_skill_commitment)
          ? currentBelief
          : undefined;
        const recoveryRoot = relevant ?? freshDurableFailureBelief;
        if (!recoveryRoot) {
          throw new Error(
            "Recovery requires failure feedback routed through the current Action Selection -> Sensorimotor episode"
          );
        }
        const postFailureBelief = relevant?.kind === "skill_failed"
          || freshDurableFailureBelief
          ? currentBelief
          : undefined;
        if (relevant?.kind === "skill_failed"
          && runtime.coordinatorPhase() === "replan_or_retire"
          && !postFailureBelief) {
          throw new Error(
            "Physical failure Recovery requires a current causally bound post-failure perceptual belief"
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
            payload: freshDurableFailureBelief
              ? {
                  recovery_basis:
                    "durable_failure_receipt_and_post_failure_observation",
                  coordinator_phase: "replan_or_retire",
                  post_failure_belief: freshDurableFailureBelief.payload
                }
              : postFailureBelief
              ? {
                  failure: recoveryRoot.payload,
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
          let parsed: ReturnType<typeof parseNeuralAgentOutput>;
          if (!outer.onAgentStream) {
            const result = await runner.run(
              recovery,
              neuralInvocationInput(
                outer.runtime,
                HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
                recoveryId,
                recoveryInvocation.invocationId
              ),
              runOptions
            );
            parsed = parseNeuralAgentOutput(
              RecoveryOutputSchema.parse(result.finalOutput)
            );
          } else {
            const stream = await runner.run(
              recovery,
              neuralInvocationInput(
                outer.runtime,
                HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
                recoveryId,
                recoveryInvocation.invocationId
              ),
              { ...runOptions, stream: true as const }
            );
            for await (const event of stream) {
              await outer.onAgentStream(recoveryId, event);
            }
            await stream.completed;
            parsed = parseNeuralAgentOutput(
              RecoveryOutputSchema.parse(stream.finalOutput)
            );
          }
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
          await runtime.consumeNeuralSignals(
            HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            recoveryDemands.map((signal) => signal.signal_id)
          );
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
              const sensation = lowerMotorSensation(physicalReceipt);
              const sensorySignal = await input.runtime.publishNeuralSignal({
                kind: "sensory_evidence",
                pathway: "ascending_feedback",
                direction: "ascending",
                sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
                targetNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
                ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
                priority: 95,
                causalParentIds: [bodyIntent.signal_id],
                sourceAuthorityLeaseId: bodyLease.lease_id,
                invocationId: bodyLease.invocation_id,
                parentInvocationId: bodyLease.parent_invocation_id,
                payload: sensation
              });
              let bodyPredictionError: NeuralSignal | undefined;
              if (!physicalExecutionSucceeded(physicalReceipt)) {
                bodyPredictionError = await input.runtime.publishNeuralSignal({
                  kind: "prediction_error",
                  pathway: "ascending_feedback",
                  direction: "ascending",
                  sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.body,
                  targetNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
                  ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
                  priority: 100,
                  causalParentIds: [sensorySignal.signal_id],
                  sourceAuthorityLeaseId: bodyLease.lease_id,
                  invocationId: bodyLease.invocation_id,
                  parentInvocationId: bodyLease.parent_invocation_id,
                  payload: lowerMotorPredictionError(physicalReceipt)
                });
              }
              return {
                output,
                physicalReceipt,
                sensation,
                sensorySignal,
                ...(bodyPredictionError ? { bodyPredictionError } : {})
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

        const reflexReceipt = await input.runtime.publishNeuralSignal({
          kind: "execution_receipt",
          pathway: "ascending_feedback",
          direction: "ascending",
          sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
          ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
          priority: 95,
          causalParentIds: [bodyResult.sensorySignal.signal_id],
          sourceAuthorityLeaseId: reflexLease.lease_id,
          invocationId: reflexLease.invocation_id,
          parentInvocationId: reflexLease.parent_invocation_id,
          payload: {
            protocol: "reflex-execution-receipt-v1",
            commitment_id: input.commitment.commitment_id,
            execution_transaction_id: input.executionTransactionId,
            controller: bodyResult.sensation.controller,
            body_signal_id: bodyResult.sensorySignal.signal_id,
            physical: bodyResult.sensation
          }
        });
        let reflexPredictionError: NeuralSignal | undefined;
        if (bodyResult.bodyPredictionError) {
          const error = await input.runtime.recordNeuralPredictionError({
            observerNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
            sourceSignalId: bodyResult.bodyPredictionError.signal_id,
            magnitude: 1,
            tolerance: 0.2,
            correctionScope: "local",
            detail: bodyResult.bodyPredictionError.payload
          });
          reflexPredictionError = await input.runtime.publishNeuralSignal({
            kind: "prediction_error",
            pathway: "ascending_feedback",
            direction: "ascending",
            sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
            ttlRevisions: MODEL_EPISODE_SIGNAL_TTL_REVISIONS,
            priority: 100,
            causalParentIds: [bodyResult.bodyPredictionError.signal_id],
            sourceAuthorityLeaseId: reflexLease.lease_id,
            invocationId: reflexLease.invocation_id,
            parentInvocationId: reflexLease.parent_invocation_id,
            payload: {
              protocol: "reflex-prediction-error-v1",
              error,
              physical: bodyResult.sensation
            }
          });
        }
        const completed = physicalExecutionSucceeded(bodyResult.physicalReceipt);
        const completionSignal = await input.runtime.publishNeuralSignal({
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
          sourceAuthorityLeaseId: reflexLease.lease_id,
          invocationId: reflexLease.invocation_id,
          parentInvocationId: reflexLease.parent_invocation_id,
          payload: {
            protocol: "lower-motor-loop-completion-v1",
            commitment_id: input.commitment.commitment_id,
            execution_transaction_id: input.executionTransactionId,
            accepted: bodyResult.physicalReceipt.accepted,
            code: bodyResult.physicalReceipt.code,
            body_signal_id: bodyResult.sensorySignal.signal_id,
            reflex_receipt_signal_id: reflexReceipt.signal_id
          }
        });
        return {
          output: bodyResult.output,
          executionReceiptSignal: reflexReceipt,
          ...(reflexPredictionError
            ? { predictionErrorSignal: reflexPredictionError }
            : {}),
          completionSignal,
          summary: {
            protocol: "certified-lower-motor-loop-v1",
            motor_intent_signal_id: motorIntent.signal_id,
            body_sensory_signal_id: bodyResult.sensorySignal.signal_id,
            reflex_execution_signal_id: reflexReceipt.signal_id,
            completion_signal_id: completionSignal.signal_id,
            ...(reflexPredictionError
              ? { prediction_error_signal_id: reflexPredictionError.signal_id }
              : {}),
            controller: bodyResult.sensation.controller
          }
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

function physicalExecutionSucceeded(receipt: HumanoidActionReceipt): boolean {
  return receipt.accepted && (
    receipt.code === "motion_completed"
      || receipt.code === "navigation_completed"
      || receipt.code === "motion_option_succeeded"
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
} {
  const detail = jsonRecord(receipt.detail) ?? {};
  const trajectory = jsonRecord(detail.physical_trajectory) ?? {};
  const controller = trajectory.controller_usage ?? null;
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
    controller
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
    reason: detail.reason ?? detail.failure_class ?? receipt.code
  };
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

function neuralInvocationInput(
  runtime: HumanoidNeuralAgentRuntime,
  parentAgentId: string,
  childAgentId: string,
  invocationId: string
): string {
  const signals = runtime.pendingNeuralSignals({
    targetNodeId: childAgentId,
    invocationId
  });
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
  const goal = pickJsonFields(record, ["active_goal", "goal_state"]);
  const control = pickJsonFields(record, [
    "active_cycle",
    "cycle_completion",
    "coordinator_phase",
    "execution_authority",
    "recovery_authority"
  ]);
  const body = pickJsonFields(record, ["robot"]);
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
          "coordinator_phase",
          "execution_authority",
          "recovery_authority",
          "cycle_index",
          "previous_cycle_transition",
          "robot",
          "goal_state",
          "recent_receipts"
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
          "previous_cycle_transition",
          "embodied_experience_memory"
        ]),
        ...hierarchy
      });
    case "actionSelection":
      return jsonValue({
        ...common,
        ...goal,
        ...control,
        ...body,
        ...pickJsonFields(record, ["mission_goal", "recent_receipts"]),
        ...hierarchy
      });
    case "perceptionManager":
      return jsonValue({ ...common, ...goal, ...control, ...body, ...hierarchy });
    case "sceneInterpreter":
      return jsonValue({ ...common, ...goal, ...hierarchy });
    case "memoryRetriever":
      return jsonValue({
        ...common,
        ...goal,
        ...pickJsonFields(record, ["embodied_experience_memory"]),
        ...hierarchy
      });
    case "sensorimotorManager":
      return jsonValue({
        ...common,
        ...goal,
        ...control,
        ...body,
        ...pickJsonFields(record, ["recent_receipts"]),
        ...hierarchy
      });
    case "affordance":
      return jsonValue({
        ...common,
        ...goal,
        ...body,
        interaction: compactInteractionProjection(record.interaction),
        ...hierarchy
      });
    case "risk":
      return jsonValue({
        ...common,
        ...goal,
        ...body,
        interaction: compactRiskProjection(record.interaction),
        ...hierarchy
      });
    case "predictive":
    case "premotor":
      return jsonValue({ ...common, ...goal, ...body, ...control, ...hierarchy });
    case "motorIntent":
      return jsonValue({
        ...common,
        ...goal,
        ...body,
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
        ...control,
        ...body,
        interaction: compactInteractionProjection(record.interaction),
        ...pickJsonFields(record, [
          "embodied_experience_memory",
          "recent_receipts"
        ]),
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
  if (typeof record.intent !== "string" || causalInputs === undefined) return value;
  const semanticInputs = causalInputs.flatMap((candidate) => {
    const input = jsonRecord(candidate);
    if (!input) return [];
    const payload = input.payload;
    return payload === undefined ? [] : [causalSemanticProjection(payload)];
  });
  if (semanticInputs.length === 1) return semanticInputs[0]!;
  if (semanticInputs.length > 1) {
    return jsonValue({
      intent: record.intent,
      semantic_inputs: semanticInputs
    });
  }
  return jsonValue({ intent: record.intent });
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

async function prepareHarnessPhaseForChild(
  runtime: HumanoidNeuralAgentRuntime,
  parentKey: HumanoidNeuralAgentKey,
  childKey: HumanoidNeuralAgentKey
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
    if (phase.phase === "recovery") {
      // Recovery is a strict Executive -> Action Selection -> Sensorimotor
      // control episode. Preserve its decision domain until Sensorimotor has
      // opened and closed the exclusive Recovery child lease.
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
    const coordinator = runtime.coordinatorPhase();
    const completionReady = coordinator === "complete_cycle"
      || coordinator === "complete_satisfied_goal";
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
    const freshDurableFailureBelief = coordinator === "replan_or_retire"
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
      const nextPhase: NeuralHarnessPhase = commitment.state === "executing"
        ? "execution"
        : commitment.state === "committed"
          ? "motor_assessment"
          : "skill_proposal";
      await runtime.transitionNeuralHarnessPhase({
        phase: nextPhase,
        enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
        reason: `action_selection_returned_${commitment.state}_commitment`,
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
  }).filter((signal) => isCurrentNeuralSignal(runtime, signal));
  const causallyBound = beliefs.filter((belief) => sourceSignals.some((source) => (
    belief.signal_id === source.signal_id
      || neuralSignalHasAncestorId(state, belief, source.signal_id)
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
    || hasCurrentSignalsAny(
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
  }
  return kinds.length === 0
    ? []
    : runtime.pendingNeuralSignals({ targetNodeId, kinds }).filter((signal) => (
      signal.direction === "descending"
        ? signal.invocation_id === invocationId
        : signal.parent_episode_id === invocationId
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
  if (managerKey === "sensorimotorManager" && outputKind === "skill_proposal") {
    const recoveryProposal = runtime.pendingNeuralSignals({
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      kinds: ["skill_proposal"]
    }).find((signal) => signal.direction === "ascending"
      && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.recovery
      && signal.parent_episode_id === invocationId
      && sourceSignalIds.includes(signal.signal_id)
      && isCurrentNeuralSignal(runtime, signal));
    if (recoveryProposal) {
      // Recovery owns a separate, exclusive failure-domain lease. Its proposal
      // is already the formal child result and must not be forced through the
      // ordinary Affordance/Risk assessment fork.
      return [];
    }
  }
  const requirements: readonly NeuralSignalKind[] = managerKey === "perceptionManager"
    && outputKind === "perceptual_belief"
    ? ["sensory_evidence", "scene_interpretation", "memory_retrieval"]
    : managerKey === "sensorimotorManager" && outputKind === "skill_proposal"
      ? ["perceptual_belief", "affordance_hypothesis", "risk_assessment"]
      : [];
  if (requirements.length === 0) return [];
  const targetNodeId = HUMANOID_NEURAL_AGENT_IDS[managerKey];
  const available = runtime.pendingNeuralSignals({
    targetNodeId,
    kinds: requirements
  });
  const cited = new Set(sourceSignalIds);
  return requirements.map((kind) => {
    const signal = available.find((candidate) => (
      candidate.kind === kind
        && (kind === "perceptual_belief"
          ? candidate.direction === "descending"
            && candidate.invocation_id === invocationId
          : candidate.direction === "ascending"
            && candidate.parent_episode_id === invocationId)
    ));
    if (!signal) {
      throw new Error(
        `${targetNodeId} cannot return ${outputKind} before joining ${kind} `
          + `inside Manager episode ${invocationId}`
      );
    }
    if (!cited.has(signal.signal_id)) {
      throw new Error(
        `${targetNodeId} ${outputKind} omitted joined ${kind} signal `
          + `${signal.signal_id} from source_signal_ids`
      );
    }
    return signal;
  });
}

function neuralDelegationSchema(
  parentKey: HumanoidNeuralAgentKey,
  childKey: HumanoidNeuralAgentKey
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
  const signalKinds = [...contract.signalKinds] as [
    NeuralSignalKind,
    ...NeuralSignalKind[]
  ];
  return z.object({
    signal_kind: z.enum(signalKinds),
    intent: z.string().trim().min(1).max(8_000).describe(
      "Concise responsibility for the child. Do not copy context anchors, directed signals, observations, or JSON into this field; the Harness injects the child's authoritative state separately."
    ),
    source_signal_ids: z.array(z.string().uuid()).max(64).default([]),
    ttl_revisions: z.number().int().min(1).max(1_000_000)
      .default(MODEL_EPISODE_SIGNAL_TTL_REVISIONS),
    priority: z.number().int().min(0).max(100).default(50)
  }).strict();
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
    recovery: "sensorimotor"
  };
  const descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(agentId);
  const profile = descriptor ? profiles[descriptor.key] : undefined;
  if (!profile) throw new Error(`Neural runtime node has no model profile: ${agentId}`);
  return profile;
}

function baseInstructions(key: HumanoidNeuralAgentKey): string[] {
  const descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(HUMANOID_NEURAL_AGENT_IDS[key]);
  if (!descriptor) throw new Error(`Unknown neural node: ${key}`);
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
    "When delegating, write only a concise plain-text intent. Never copy your context anchor, observations, directed signals, or JSON into a child tool argument; the Harness injects the child's authoritative state.",
    "For source_signal_ids copy only exact signal_id values present in the current invocation. Use [] when none exists; never invent a UUID or placeholder.",
    `Submit every final neural signal through ${NEURAL_OUTPUT_SUBMISSION_TOOL_NAME}; never return the envelope as assistant text.`,
    "Pass the structured signal body directly in the submission tool's payload field; do not JSON-stringify it.",
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
      const validatedPayload = validateNeuralSubmissionPayload(
        key,
        envelope.signal_kind,
        payload
      );
      const parsed = outputType.parse({
        ...envelope,
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
      return [humanoidNeuralAgentToolName("executor")];
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
  schema.properties.payload = {
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

function boundedSkillProposalPayloadJsonSchema(): Record<string, unknown> {
  const phaseReference = {
    type: "object",
    additionalProperties: false,
    properties: {
      skill: { type: "string", enum: [...HUMANOID_SKILL_IDS] },
      phase: { type: "string", minLength: 1, maxLength: 256 }
    },
    required: ["skill", "phase"]
  };
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      proposed_skill: {
        type: "object",
        additionalProperties: false,
        properties: {
          skill: {
            type: "string",
            enum: [...HUMANOID_SKILL_IDS],
            description: "Exactly one bounded Skill id"
          },
          phase: {
            type: "string",
            minLength: 1,
            maxLength: 256,
            description: "One phase from that Skill's live process contract"
          },
          params: {
            type: "object",
            additionalProperties: true,
            description: "Exact catalog parameters for this one Skill; never include another skill field"
          },
          rationale: { type: "string", minLength: 1, maxLength: 8_000 }
        },
        required: ["skill", "phase", "params", "rationale"]
      },
      phase_sequence: {
        type: "array",
        items: phaseReference,
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
    ? validatePerceptualBeliefAuthority(validated)
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
      + "Report observed state, uncertainty, and all live reachability candidates verbatim; Action Selection and Sensorimotor exclusively own Skill, hand, interaction-point, and next-action selection."
  );
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

function planningReceiptToolUseBehavior(): ToolUseBehavior {
  return (_context, results) => {
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
      if ((physicalPlanningResult || rejectedSkillStateTransition)
        && typeof receipt?.accepted === "boolean") {
        const sourceSignalIds = z.array(z.string().uuid()).max(64).catch([]).parse(
          receiptRecord?.source_signal_ids
        );
        const { source_signal_ids: _sourceSignalIds, ...receiptPayload } = receiptRecord!;
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: JSON.stringify({
            signal_kind: receipt.accepted ? "rollout_result" : "escalation",
            summary: receipt.code ?? "planning result",
            payload_json: JSON.stringify(receiptPayload),
            source_signal_ids: sourceSignalIds,
            confidence: receipt.accepted ? 1 : 0.5
          })
        };
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
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
  const statuses = new Set([
    "skill_committed",
    "skill_executing",
    "skill_completed",
    "skill_failed",
    "skill_released"
  ]);
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
          signal_kind: status === "skill_completed"
            ? "skill_completed"
            : status === "skill_failed"
              ? "skill_failed"
              : "skill_commitment",
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
  const executorToolName = humanoidNeuralAgentToolName("executor");
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
        const rollouts = current.filter((signal) => (
          signal.kind === "rollout_result"
            && modelPayloadSha256(signal.payload)
              === certificate.rollout_payload_sha256
            && prediction !== undefined
            && neuralSignalHasAncestorId(
              state,
              prediction,
              signal.signal_id
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
      if (result.tool.name === executorToolName) {
        const sourceSignalIds = z.array(z.string().uuid()).max(64).parse(
          parsed.source_signal_ids
        );
        const state = runtime.neuralHierarchyState();
        const completionSignals = sourceSignalIds.map(
          (signalId) => state.signals[signalId]
        ).filter((signal): signal is NeuralSignal => signal !== undefined
          && (signal.kind === "skill_completed" || signal.kind === "skill_failed"));
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
  const ownedPayloadHashes = new Set(ownedSignals.map(
    (signal) => modelPayloadSha256(signal.payload)
  ));
  const ownedCommitmentIds = new Set(ownedSignals.map(
    (signal) => neuralSignalCommitmentId(state, signal)
  ).filter((commitmentId): commitmentId is string => commitmentId !== null));
  const candidates = Object.values(state.signals).filter((candidate) => (
    candidate.status === "pending"
      && candidate.kind === "rollout_result"
      && candidate.direction === "reentrant"
      && candidate.source_node_id === HUMANOID_NEURAL_AGENT_IDS.rolloutGate
      && candidate.target_node_id === predictiveNodeId
      && ownedPayloadHashes.has(modelPayloadSha256(candidate.payload))
      && (activeCommitmentId === null
        || neuralSignalCommitmentId(state, candidate) === activeCommitmentId)
      && (ownedCommitmentIds.size === 0
        || ownedCommitmentIds.has(neuralSignalCommitmentId(state, candidate) ?? ""))
      && ownedSignals.some((ownedSignal) => (
        ownedSignal.signal_id === candidate.signal_id
          || neuralSignalHasAncestorId(state, ownedSignal, candidate.signal_id)
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
