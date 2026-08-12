import {
  Agent,
  tool,
  type CallModelInputFilter,
  type FunctionTool,
  type Model,
  type ModelSettings,
  type RunStreamEvent,
  type Session,
  type ToolInputParameters,
  type ToolUseBehavior
} from "@openai/agents";
import { z } from "zod";
import type { JsonValue } from "../../domain/schema.js";
import {
  providerConfigForRole,
  type AgentModelRole,
  type ModelProviderConfig,
  type ProviderConfig
} from "../../config/load.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import type { HumanoidActionInvoker } from "./runtime.js";
import {
  createHumanoidActionTools,
  createHumanoidEmbodiedRecallTool,
  invokeDeterministicHumanoidAction,
  type HumanoidEmbodiedRecallInvoker
} from "./tools.js";
import {
  createGoalManagerTools,
  type GoalManagerRuntime
} from "./goal-manager-tools.js";
import {
  agentInvocationMarker,
  scopeAgentToolInvocation
} from "../agent-scope.js";
import { createToolInputRecovery } from "../tool-input-recovery.js";
import { DelegatedDecisionRecovery } from "../delegated-decision-recovery.js";
import { modelDecisionStallFrom } from "../model-telemetry.js";
import type { HumanoidCycleCompletionReadiness } from "./cycle-causal-evidence.js";
import type { HumanoidCoordinatorPhase } from "./run-runtime.js";
import { GOAL_HISTORY_PREDICATE_TYPES } from "./goal-history.js";

const SpecialistDelegationSchema = z.object({}).strict();
const ExecutePlanTaskSchema = z.object({
  kind: z.literal("execute_plan"),
  planning_action: z.enum([
    "plan_humanoid_skill",
    "plan_whole_body_motion",
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
  ]).describe("逐字复制已接受规划回执 action"),
  planning_transaction_id: z.string().trim().min(1)
    .describe("逐字复制已接受规划回执 transactionId")
}).strict();
const RemoveWorldBlockTaskSchema = z.object({
  kind: z.literal("remove_world_block"),
  solid_id: z.string().trim().min(1).describe("目标 solid_id"),
  execution_transaction_id: z.string().trim().min(1)
    .describe("先前成功物理执行的 transactionId")
}).strict();
const ExecutionTaskSchema = z.object({
  objective: z.string().trim().min(1),
  execution: z.discriminatedUnion("kind", [
    ExecutePlanTaskSchema,
    RemoveWorldBlockTaskSchema
  ])
}).strict();
const CycleCompletionSchema = z.object({
  summary: z.string().trim().min(1),
  evidence_transaction_ids: z.array(z.string().trim().min(1)).min(1),
  next_intent: z.string().trim().min(1).optional()
}).strict().superRefine((value, context) => {
  if (new Set(value.evidence_transaction_ids).size !== value.evidence_transaction_ids.length) {
    context.addIssue({
      code: "custom",
      path: ["evidence_transaction_ids"],
      message: "Cycle evidence transaction identifiers must be unique"
    });
  }
});
const SatisfiedGoalCompletionSchema = z.object({
  summary: z.string().trim().min(1)
}).strict();
export const HUMANOID_AGENT_IDS = {
  goalManager: "humanoid-goal-manager",
  coordinator: "humanoid-coordinator",
  sentry: "humanoid-sentry",
  motion: "humanoid-motion-reference",
  executor: "humanoid-executor"
} as const;

export function goalManagerInvocationInput(
  authority: JsonValue
): string {
  const root = jsonRecord(authority);
  const goalDAG = jsonRecord(root.goal_dag);
  const goalContext = jsonRecord(root.goal_context);
  const autonomy = jsonRecord(goalContext.autonomy);
  const autonomyHistory = jsonRecord(autonomy.history);
  const observation = jsonRecord(goalContext.observation);
  const candidates = jsonRecord(goalDAG.candidates);
  const projection = jsonRecord(goalDAG.context_projection);
  const candidateReferences = Object.entries(candidates).map(([candidateId, value]) => {
    const candidate = jsonRecord(value);
    return {
      candidate_sequence: candidate.candidate_sequence ?? null,
      proposal_id: candidate.proposal_id ?? null,
      candidate_id: candidateId,
      status: candidate.status ?? null,
      goal: candidate.goal ?? null,
      mission_link: candidate.mission_link ?? null,
      dependency_candidate_ids: Array.isArray(candidate.dependency_candidate_ids)
        ? candidate.dependency_candidate_ids
        : [],
      dependency_candidates: Array.isArray(candidate.dependency_candidates)
        ? candidate.dependency_candidates
        : []
    };
  });
  const solids = Array.isArray(observation.solids)
    ? observation.solids.map(jsonRecord)
    : [];
  const objects = Array.isArray(observation.objects)
    ? observation.objects.map(jsonRecord)
    : [];
  const interaction = jsonRecord(root.interaction);
  const spatialBelief = jsonRecord(root.spatial_belief);
  const carrying = jsonRecord(interaction.carrying);
  const carryBindings = Array.isArray(carrying.bindings)
    ? carrying.bindings.map(jsonRecord)
    : [];
  const recentReceipts = Array.isArray(root.recent_receipts)
    ? root.recent_receipts.map(jsonRecord)
    : [];
  const exact = {
    run_mode: root.run_mode ?? null,
    mission_goal: root.mission_goal ?? null,
    goal_dag_status: goalDAG.status ?? null,
    coordinator_phase: root.coordinator_phase ?? null,
    recovery_authority: root.recovery_authority ?? null,
    previous_cycle_transition: root.previous_cycle_transition ?? null,
    goal_state: root.goal_state ?? null,
    current_goal_evidence_ref: goalContext.evidence_ref ?? null,
    existing_goal_candidate_ids: Object.keys(candidates).sort(),
    existing_goal_candidates: candidateReferences,
    current_goal_epoch_id: goalDAG.current_epoch_id ?? null,
    recent_action_evidence: recentReceipts.flatMap((receipt) => (
      typeof receipt.transaction_id === "string"
        ? [{
            evidence_ref: `action:${receipt.transaction_id}`,
            transaction_id: receipt.transaction_id,
            action: receipt.action ?? null,
            accepted: receipt.accepted ?? null,
            code: receipt.code ?? null,
            world_after_revision: receipt.world_after_revision ?? null,
            frame_count: receipt.frame_count ?? null,
            detail: receipt.detail ?? {}
          }]
        : []
    )),
    candidate_history: {
      total: projection.total_candidate_count ?? Object.keys(candidates).length,
      visible: Object.keys(candidates).length,
      truncated: projection.history_truncated === true,
      lifetime_outcomes: autonomyHistory.lifetime_outcomes ?? null
    },
    observable_goal_surface: {
      predicate_types: [...GOAL_HISTORY_PREDICATE_TYPES],
      zone_ids: stringArray(observation.zone_ids),
      visible_object_ids: stringArray(observation.visible_object_ids),
      portable_object_ids: objects.flatMap((object) => (
        object.portable === true && typeof object.id === "string" ? [object.id] : []
      )),
      object_affordances: objects.flatMap((object) => (
        typeof object.id === "string"
          ? [{
              object_id: object.id,
              affordances: stringArray(object.affordances),
              articulation: object.articulation ?? null
            }]
          : []
      )),
      removable_block_ids: solids.flatMap((solid) => (
        solid.kind === "block" && typeof solid.id === "string" ? [solid.id] : []
      )),
      carrying: {
        phase: carrying.phase ?? null,
        object_ids: carryBindings.flatMap((binding) => (
          typeof binding.object_id === "string" ? [binding.object_id] : []
        )),
        continuation_verified: carrying.continuation_verified ?? null
      },
      exploration: {
        visibility_model: spatialBelief.visibility_model ?? null,
        frontier_model: spatialBelief.frontier_model ?? null,
        coverage_ratio: spatialBelief.coverage_ratio ?? null,
        observed_cell_count: spatialBelief.observed_cell_count ?? null,
        free_cell_count: spatialBelief.free_cell_count ?? null,
        occupied_cell_count: spatialBelief.occupied_cell_count ?? null,
        reachable_free_cell_count: spatialBelief.reachable_free_cell_count ?? null,
        frontier_candidates: Array.isArray(spatialBelief.frontiers)
          ? spatialBelief.frontiers.slice(0, 12)
          : []
      }
    }
  };
  return [
    "请基于当前权威状态独立管理下一 Goal。",
    "CURRENT GOAL MANAGER INVOCATION",
    "以下状态与标识来自本次调用的权威物理状态。候选提交和选择会由 Harness 绑定本次证据；工具中仍需填写的标识必须逐字复制。",
    JSON.stringify(exact)
  ].join("\n\n");
}

export function motionInvocationInput(authority: JsonValue): string {
  const root = jsonRecord(authority);
  const goalDAG = jsonRecord(root.goal_dag);
  return [
    "请独立决定并规划推进当前 active Goal 的下一个可真实执行阶段。必要时先重新观察；不得沿用上级坐标或动作参数。",
    "CURRENT MOTION DELEGATION",
    JSON.stringify({
      run_mode: root.run_mode ?? null,
      current_goal_epoch_id: goalDAG.current_epoch_id ?? null,
      active_goal: root.active_goal ?? null,
      coordinator_phase: root.coordinator_phase ?? null,
      active_cycle: root.active_cycle ?? null,
      planning_tool_state: root.planning_tool_state ?? null
    })
  ].join("\n\n");
}

export const HUMANOID_AGENT_TOOL_CONTRACTS = {
  goalManager: {
    dispatchKind: "model_agent",
    toolName: "delegate_goal_manager",
    targetRole: "goal_manager",
    targetAgentId: HUMANOID_AGENT_IDS.goalManager,
    inputBuilderContract: "goal_manager_authority_envelope_v2",
    inputBuilder: () => goalManagerInvocationInput({}),
    runOptions: {
      sessionAgentId: HUMANOID_AGENT_IDS.goalManager,
      contextSource: "parent_run_context",
      maxTurns: "unbounded"
    },
    resumeContextStrategy: "merge",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "nested_agent_final_output_text"
  },
  sentry: {
    dispatchKind: "deterministic_service",
    toolName: "delegate_humanoid_sentry",
    targetRole: "sentry",
    targetAgentId: HUMANOID_AGENT_IDS.sentry,
    targetName: "异步物理 Grounding Monitor",
    inputBuilderContract: "grounding_monitor_direct_v1",
    implementationContract: "observe_humanoid_from_coordinator_tool_call_v1",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "formal_action_receipt"
  },
  motion: {
    dispatchKind: "model_agent",
    toolName: "delegate_motion_reference",
    targetRole: "motion",
    targetAgentId: HUMANOID_AGENT_IDS.motion,
    inputBuilderContract: "motion_authority_envelope_v1",
    inputBuilder: () => motionInvocationInput({}),
    runOptions: {
      sessionAgentId: HUMANOID_AGENT_IDS.motion,
      contextSource: "parent_run_context",
      maxTurns: "unbounded"
    },
    resumeContextStrategy: "merge",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "nested_agent_final_output_text"
  },
  executor: {
    dispatchKind: "deterministic_service",
    toolName: "delegate_physics_executor",
    targetRole: "executor",
    targetAgentId: HUMANOID_AGENT_IDS.executor,
    targetName: "确定性物理 Execution Gate",
    inputBuilderContract: "validated_execution_gate_v1",
    implementationContract: "accepted_plan_to_runtime_action_v1",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "formal_action_receipt"
  }
} as const;

type ModelAgentToolContract =
  | typeof HUMANOID_AGENT_TOOL_CONTRACTS.goalManager
  | typeof HUMANOID_AGENT_TOOL_CONTRACTS.motion;

export const HUMANOID_CAPABILITIES = [
  "recall_goal_history",
  "submit_goal_candidates",
  "select_goal_candidate",
  "retire_goal_epoch",
  "continue_goal_epoch",
  "observe_humanoid",
  "submit_humanoid_skill_plan",
  "begin_humanoid_skill",
  "recall_embodied_history",
  "plan_humanoid_skill",
  "execute_humanoid_skill",
  "plan_whole_body_motion",
  "plan_whole_body_motion_candidates",
  "execute_whole_body_motion",
  "plan_humanoid_navigation",
  "execute_humanoid_navigation",
  "remove_world_block"
] as const;

type HumanoidHierarchyRuntime = HumanoidActionInvoker
& HumanoidEmbodiedRecallInvoker & GoalManagerRuntime & {
  contextAnchor(agentId: string): JsonValue;
  validateCycleEvidence(evidenceTransactionIds: readonly string[]): HumanoidActionReceipt;
  cycleCompletionReadiness(): HumanoidCycleCompletionReadiness;
  coordinatorPhase(): HumanoidCoordinatorPhase;
  executorDelegationAvailable(): boolean;
  goalRetirementDelegationAvailable(): boolean;
  sentryDelegationAvailable?(): boolean;
  motionDelegationAvailable?(): boolean;
  validateSatisfiedGoal(): JsonValue;
};

export interface HumanoidAgentHierarchy {
  goalManager: Agent;
  coordinator: Agent;
  motion: Agent;
  sentry: HumanoidDeterministicService;
  executor: HumanoidDeterministicService;
  goalManagerSession: Session;
  coordinatorSession: Session;
  session(agentId: string): Session | undefined;
}

interface HumanoidDeterministicService {
  id: string;
  name: string;
  kind: "deterministic_service";
  implementationContract: string;
}

export interface DelegatedDecisionFollowUp {
  agentId: string;
  attempt: number;
  reason: string;
}

export function createHumanoidAgentHierarchy(input: {
  createModel: (agentId: string, provider: ModelProviderConfig) => Model;
  createSession: (agentId: string) => Session;
  callModelInputFilter: CallModelInputFilter;
  provider: ProviderConfig;
  runtime: HumanoidHierarchyRuntime;
  onAgentStream?: (agentId: string, event: RunStreamEvent) => void | Promise<void>;
  onDelegatedDecisionFollowUp?: (
    event: DelegatedDecisionFollowUp
  ) => void | Promise<void>;
}): HumanoidAgentHierarchy {
  const models = new Set<Model>();
  const sessions = new Map<string, Session>();
  const sessionOwners = new Set<Session>();
  const delegatedDecisionRecovery = new DelegatedDecisionRecovery();
  const ownModel = (agentId: string): Model => {
    const model = input.createModel(
      agentId,
      providerConfigForRole(input.provider, humanoidAgentRole(agentId))
    );
    if (models.has(model)) {
      throw new Error(`Humanoid hierarchy cannot share one Model facade: ${agentId}`);
    }
    models.add(model);
    return model;
  };
  const ownSession = (agentId: string): Session => {
    const session = input.createSession(agentId);
    if (sessionOwners.has(session)) {
      throw new Error(`Humanoid hierarchy cannot share one Session: ${agentId}`);
    }
    sessionOwners.add(session);
    sessions.set(agentId, session);
    return session;
  };
  const modelSettings = (agentId: string): ModelSettings => {
    const provider = providerConfigForRole(input.provider, humanoidAgentRole(agentId));
    return {
      temperature: provider.temperature,
      ...(provider.reasoningEffort === undefined
        ? {}
        : { reasoning: { effort: provider.reasoningEffort } }),
      ...(provider.maxOutputTokens === undefined
        ? {}
        : { maxTokens: provider.maxOutputTokens }),
      parallelToolCalls: false,
      toolChoice: provider.toolChoice ?? "auto"
    };
  };

  const goalManager = new Agent({
    name: "自主目标管理智能体",
    instructions: scopedInstructions(
      HUMANOID_AGENT_IDS.goalManager,
      goalManagerInstructions()
    ),
    model: ownModel(HUMANOID_AGENT_IDS.goalManager),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.goalManager),
    tools: createGoalManagerTools(input.runtime),
    resetToolChoice: false,
    toolUseBehavior: verifiedStatusToolUseBehavior({
      select_goal_candidate: "goal_candidate_selected",
      retire_goal_epoch: "goal_epoch_retired",
      continue_goal_epoch: "goal_epoch_continued"
    })
  });
  const motion = new Agent({
    name: "全身运动参考智能体",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.motion, motionInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.motion),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.motion),
    tools: [
      ...createHumanoidActionTools(
        input.runtime,
        HUMANOID_AGENT_IDS.motion,
        ["observe_humanoid"],
        { availability: "stable" }
      ),
      createHumanoidEmbodiedRecallTool(input.runtime),
      ...createHumanoidActionTools(
        input.runtime,
        HUMANOID_AGENT_IDS.motion,
        [
          "submit_humanoid_skill_plan",
          "begin_humanoid_skill",
          "plan_humanoid_skill",
          "plan_whole_body_motion_candidates",
          "plan_humanoid_navigation"
        ],
        { availability: "stable" }
      )
    ],
    resetToolChoice: false,
    toolUseBehavior: receiptToolUseBehavior([
      "plan_humanoid_skill",
      "plan_whole_body_motion_candidates",
      "plan_humanoid_navigation"
    ])
  });
  const sentry: HumanoidDeterministicService = {
    id: HUMANOID_AGENT_IDS.sentry,
    name: HUMANOID_AGENT_TOOL_CONTRACTS.sentry.targetName,
    kind: "deterministic_service",
    implementationContract:
      HUMANOID_AGENT_TOOL_CONTRACTS.sentry.implementationContract
  };
  const executor: HumanoidDeterministicService = {
    id: HUMANOID_AGENT_IDS.executor,
    name: HUMANOID_AGENT_TOOL_CONTRACTS.executor.targetName,
    kind: "deterministic_service",
    implementationContract:
      HUMANOID_AGENT_TOOL_CONTRACTS.executor.implementationContract
  };
  const goalManagerSession = ownSession(HUMANOID_AGENT_IDS.goalManager);
  ownSession(HUMANOID_AGENT_IDS.motion);
  const coordinatorSession = ownSession(HUMANOID_AGENT_IDS.coordinator);
  const agentToolSession = (contract: ModelAgentToolContract): Session => {
    const session = sessions.get(contract.runOptions.sessionAgentId);
    if (!session) {
      throw new Error(
        `Agent-as-tool ${contract.toolName} requires Session owner `
        + contract.runOptions.sessionAgentId
      );
    }
    return session;
  };
  const goalManagerContract = HUMANOID_AGENT_TOOL_CONTRACTS.goalManager;
  const motionContract = HUMANOID_AGENT_TOOL_CONTRACTS.motion;
  const coordinator = new Agent({
    name: "人形自主协调智能体",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.coordinator, coordinatorInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.coordinator),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.coordinator),
    tools: [
      createHumanoidEmbodiedRecallTool(input.runtime),
      recoverAgentToolInput(scopeAgentToolInvocation(
        HUMANOID_AGENT_IDS.goalManager,
        requireDelegatedDecision(
          delegatedDecisionRecovery,
          HUMANOID_AGENT_IDS.goalManager,
          goalManager.asTool({
        toolName: goalManagerContract.toolName,
        toolDescription: "让独立目标管理智能体基于当前物理证据提交 2–3 个候选并显式选择下一 Goal，或证据化退役当前不可达 Goal。",
        customOutputExtractor: ({ finalOutput }) => serializeAgentToolOutput(finalOutput),
        errorFunction: null,
        parameters: SpecialistDelegationSchema,
        inputBuilder: () => delegatedDecisionRecovery.invocationInput(
          HUMANOID_AGENT_IDS.goalManager,
          goalManagerInvocationInput(
            input.runtime.contextAnchor(HUMANOID_AGENT_IDS.goalManager)
          )
        ),
        includeInputSchema: goalManagerContract.includeInputSchema,
        needsApproval: goalManagerContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: {
          session: agentToolSession(goalManagerContract),
          maxTurns: null
        },
        resumeState: { contextStrategy: goalManagerContract.resumeContextStrategy },
        isEnabled: () => coordinatorToolAvailable(
          goalManagerContract.toolName,
          input.runtime
        ),
        ...(input.onAgentStream
          ? {
              onStream: ({ event }) => input.onAgentStream!(
                HUMANOID_AGENT_IDS.goalManager,
                event
              )
            }
          : {})
          }),
          isFormalGoalManagerResult,
          input.onDelegatedDecisionFollowUp
        )
      ), SpecialistDelegationSchema),
      groundingMonitorTool(input.runtime),
      recoverAgentToolInput(scopeAgentToolInvocation(
        HUMANOID_AGENT_IDS.motion,
        requireDelegatedDecision(
          delegatedDecisionRecovery,
          HUMANOID_AGENT_IDS.motion,
          motion.asTool({
        toolName: motionContract.toolName,
        toolDescription: "让独立运动参考智能体读取当前状态，提出多个按偏好排序且分别经过完整物理预演的连续全身动作候选，或规划双足路线。",
        customOutputExtractor: ({ finalOutput }) => serializeAgentToolOutput(finalOutput),
        errorFunction: null,
        parameters: SpecialistDelegationSchema,
        inputBuilder: () => delegatedDecisionRecovery.invocationInput(
          HUMANOID_AGENT_IDS.motion,
          motionInvocationInput(
            input.runtime.contextAnchor(HUMANOID_AGENT_IDS.motion)
          )
        ),
        includeInputSchema: motionContract.includeInputSchema,
        needsApproval: motionContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: {
          session: agentToolSession(motionContract),
          maxTurns: null
        },
        resumeState: { contextStrategy: motionContract.resumeContextStrategy },
        ...(input.onAgentStream
          ? { onStream: ({ event }) => input.onAgentStream!(HUMANOID_AGENT_IDS.motion, event) }
          : {})
          }),
          (output) => isFormalActionReceipt(
            output,
            HUMANOID_AGENT_IDS.motion,
            [
              "plan_humanoid_skill",
              "plan_whole_body_motion_candidates",
              "plan_humanoid_navigation"
            ]
          ),
          input.onDelegatedDecisionFollowUp
        )
      ), SpecialistDelegationSchema),
      executionGateTool(input.runtime),
      cycleCompletionTool(input.runtime),
      satisfiedGoalCompletionTool(input.runtime)
    ],
    resetToolChoice: false,
    toolUseBehavior: coordinatorStepToolUseBehavior()
  });
  guardCoordinatorToolExecution(coordinator, input.runtime);

  return {
    goalManager,
    coordinator,
    sentry,
    motion,
    executor,
    goalManagerSession,
    coordinatorSession,
    session: (agentId) => sessions.get(agentId)
  };
}

function groundingMonitorTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof SpecialistDelegationSchema, string> {
  const contract = HUMANOID_AGENT_TOOL_CONTRACTS.sentry;
  const groundingTool = tool<typeof SpecialistDelegationSchema, unknown, string>({
    name: contract.toolName,
    description: "从当前 MuJoCo、本体感觉、接触与头部传感器直接生成受限权威观察；该服务不调用模型，也不产生目标或动作。",
    parameters: SpecialistDelegationSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    execute: (params, _context, details) => invokeDeterministicHumanoidAction({
      runtime,
      actorAgentId: HUMANOID_AGENT_IDS.sentry,
      sourceToolName: contract.toolName,
      sourceInput: params,
      action: "observe_humanoid",
      actionInput: {},
      contractId: "grounding_monitor_v1",
      ...(details ? { details } : {})
    })
  });
  return recoverAgentToolInput(
    scopeAgentToolInvocation(HUMANOID_AGENT_IDS.sentry, groundingTool),
    SpecialistDelegationSchema
  );
}

function executionGateTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof ExecutionTaskSchema, string> {
  const contract = HUMANOID_AGENT_TOOL_CONTRACTS.executor;
  const executionTool = tool<typeof ExecutionTaskSchema, unknown, string>({
    name: contract.toolName,
    description: "确定性消费当前 Harness 已接受的规划授权并映射到唯一正式物理动作，或提交已有稳定接触证明的方块拆除事务；不会调用模型、重规划或改写参数。",
    parameters: ExecutionTaskSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    execute: (params, _context, details) => {
      const selected = executionGateAction(params.execution);
      return invokeDeterministicHumanoidAction({
        runtime,
        actorAgentId: HUMANOID_AGENT_IDS.executor,
        sourceToolName: contract.toolName,
        sourceInput: params,
        action: selected.action,
        actionInput: selected.input,
        contractId: "execution_gate_v1",
        ...(details ? { details } : {})
      });
    }
  });
  return recoverAgentToolInput(
    scopeAgentToolInvocation(HUMANOID_AGENT_IDS.executor, executionTool),
    ExecutionTaskSchema
  );
}

function executionGateAction(
  execution: z.infer<typeof ExecutionTaskSchema>["execution"]
): {
  action: "execute_humanoid_skill"
    | "execute_whole_body_motion"
    | "execute_humanoid_navigation"
    | "remove_world_block";
  input: Record<string, string>;
} {
  if (execution.kind === "remove_world_block") {
    return {
      action: "remove_world_block",
      input: {
        solid_id: execution.solid_id,
        execution_transaction_id: execution.execution_transaction_id
      }
    };
  }
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

function guardCoordinatorToolExecution(
  coordinator: Agent,
  runtime: HumanoidHierarchyRuntime
): void {
  for (const coordinatorTool of coordinator.tools) {
    if (coordinatorTool.type !== "function") continue;
    const name = coordinatorTool.name;
    const isEnabled = coordinatorTool.isEnabled;
    coordinatorTool.isEnabled = async (context, agent) => (
      coordinatorToolAvailable(name, runtime)
        && (isEnabled ? await isEnabled(context, agent) : true)
    );
    const invoke = coordinatorTool.invoke;
    coordinatorTool.invoke = async (context, toolInput, details) => {
      if (coordinatorToolAvailable(name, runtime)) {
        return invoke(context, toolInput, details);
      }
      return JSON.stringify({
        accepted: false,
        code: "coordinator_phase_rejected",
        tool: name,
        coordinator_phase: runtime.coordinatorPhase(),
        automatic_actuation: false,
        recovery: "Use a coordinator tool authorized by the current Harness phase."
      });
    };
  }
}

function coordinatorToolAvailable(
  name: string,
  runtime: HumanoidHierarchyRuntime
): boolean {
  const completion = runtime.cycleCompletionReadiness();
  const phase = runtime.coordinatorPhase();
  if (name === "complete_autonomous_cycle") {
    return completion.status === "ready"
      && completion.observed_after_execution
      && phase === "complete_cycle";
  }
  if (name === "complete_satisfied_goal") return phase === "complete_satisfied_goal";
  if (name === "delegate_goal_manager") {
    return phase === "goal_selection"
      || runtime.goalRetirementDelegationAvailable();
  }
  if (name === "delegate_humanoid_sentry") {
    const phaseAllowsObservation = phase === "observe_or_plan"
      || phase === "plan"
      || phase === "post_failure_observation"
      || phase === "replan_or_retire"
      || phase === "post_execution";
    return phaseAllowsObservation
      && (runtime.sentryDelegationAvailable?.() ?? true);
  }
  if (name === "delegate_motion_reference") {
    return (phase === "observe_or_plan"
      || phase === "plan"
      || phase === "replan_or_retire")
      && (runtime.motionDelegationAvailable?.() ?? true);
  }
  if (name === "delegate_physics_executor") {
    return runtime.executorDelegationAvailable();
  }
  return phase === "observe_or_plan" || phase === "plan";
}

export function humanoidAgentRole(agentId: string): Exclude<AgentModelRole, "compactor"> {
  switch (agentId) {
    case HUMANOID_AGENT_IDS.goalManager:
      return "goal_manager";
    case HUMANOID_AGENT_IDS.coordinator:
      return "coordinator";
    case HUMANOID_AGENT_IDS.sentry:
      return "sentry";
    case HUMANOID_AGENT_IDS.motion:
      return "motion";
    case HUMANOID_AGENT_IDS.executor:
      return "executor";
    default:
      throw new Error(`Unknown humanoid agent identity: ${agentId}`);
  }
}

function scopedInstructions(agentId: string, instructions: string): string {
  return `${agentInvocationMarker(agentId)}\n${instructions}`;
}

function jsonRecord(value: unknown): Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, JsonValue>
    : {};
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function cycleCompletionTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof CycleCompletionSchema, string> {
  return tool<typeof CycleCompletionSchema, unknown, string>({
    name: "complete_autonomous_cycle",
    description: "完成一次真实自主循环。证据必须包含最近一次未被后续物理执行取代的成功回执；若该执行产生世界修改，还必须同时引用对应 mutation 回执。",
    parameters: CycleCompletionSchema,
    strict: true,
    execute: (input) => {
      const execution = runtime.validateCycleEvidence(input.evidence_transaction_ids);
      return JSON.stringify({
        status: "cycle_completed",
        summary: input.summary,
        evidence_transaction_ids: input.evidence_transaction_ids,
        world_revision: execution.worldAfterRevision,
        executed_action: execution.action,
        ...(input.next_intent ? { next_intent: input.next_intent } : {})
      });
    }
  });
}

function satisfiedGoalCompletionTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof SatisfiedGoalCompletionSchema, string> {
  return tool<typeof SatisfiedGoalCompletionSchema, unknown, string>({
    name: "complete_satisfied_goal",
    description: "当前 active Goal 已由实时物理状态直接满足且本周期无需重复执行时，提交 Harness 复验并完成该 Goal。",
    parameters: SatisfiedGoalCompletionSchema,
    strict: true,
    execute: (input) => JSON.stringify({
      status: "satisfied_goal_completed",
      summary: input.summary,
      verification: runtime.validateSatisfiedGoal()
    })
  });
}

function receiptToolUseBehavior(
  terminalActions: readonly string[]
): ToolUseBehavior {
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const output = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output);
      try {
        const receipt = JSON.parse(output) as Partial<HumanoidActionReceipt>;
        if (receipt.transactionId
          && receipt.action === result.tool.name
          && typeof receipt.accepted === "boolean"
          && (!receipt.accepted
            || terminalActions.includes(result.tool.name))) {
          return {
            isFinalOutput: true,
            isInterrupted: undefined,
            finalOutput: output
          };
        }
      } catch {
        // The SDK feeds malformed model arguments back through errorFunction.
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function verifiedStatusToolUseBehavior(
  statusByTool: Readonly<Record<string, string>>
): ToolUseBehavior {
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const expectedStatus = statusByTool[result.tool.name];
      if (!expectedStatus) continue;
      const output = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output);
      try {
        const parsed = JSON.parse(output) as { status?: unknown };
        if (parsed.status === expectedStatus) {
          return {
            isFinalOutput: true,
            isInterrupted: undefined,
            finalOutput: output
          };
        }
      } catch {
        // Rejected or malformed terminal output is fed back to the deciding model.
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function coordinatorStepToolUseBehavior(): ToolUseBehavior {
  const terminalStatusByTool: Readonly<Record<string, string>> = {
    complete_autonomous_cycle: "cycle_completed",
    complete_satisfied_goal: "satisfied_goal_completed"
  };
  const receiptTools = new Set([
    "delegate_humanoid_sentry",
    "delegate_motion_reference",
    "delegate_physics_executor"
  ]);
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const output = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output);
      const parsed = delegatedOutputObject(output);
      if (!parsed) continue;
      const terminalStatus = terminalStatusByTool[result.tool.name];
      const terminal = terminalStatus !== undefined
        && parsed.status === terminalStatus;
      const goalTransition = result.tool.name === "delegate_goal_manager"
        && (parsed.status === "goal_candidate_selected"
          || parsed.status === "goal_epoch_retired"
          || parsed.status === "goal_epoch_continued");
      const physicalStep = receiptTools.has(result.tool.name)
        && typeof parsed.transactionId === "string"
        && typeof parsed.action === "string"
        && typeof parsed.accepted === "boolean";
      if (terminal || goalTransition || physicalStep) {
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: output
        };
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function requireDelegatedDecision<
  TContext,
  TParameters extends ToolInputParameters,
  TResult,
  TTool extends FunctionTool<TContext, TParameters, TResult>
>(
  recovery: DelegatedDecisionRecovery,
  agentId: string,
  agentTool: TTool,
  isFormalDecision: (output: unknown) => boolean,
  onFollowUp?: (event: DelegatedDecisionFollowUp) => void | Promise<void>
): TTool {
  const invoke = agentTool.invoke;
  agentTool.invoke = async (context, input, details) => {
    details?.signal?.throwIfAborted();
    try {
      const output = await invoke(context, input, details);
      return recovery.accept(agentId, output, isFormalDecision);
    } catch (error) {
      const stall = modelDecisionStallFrom(error);
      if (!stall || stall.agentId !== agentId || stall.evidence) throw error;
      // A delegated Agent gets one bounded SDK invocation for the current
      // physical authority. Return a missing-decision stall to the outer
      // Harness instead of polling the same specialist forever without a new
      // world observation. The next hierarchy step may invoke it again with
      // strengthened native tool-choice and a freshly rebuilt authority.
      if (recovery.attempt(agentId) === 0) recovery.recordFailure(agentId, stall);
      await onFollowUp?.({
        agentId,
        attempt: recovery.attempt(agentId),
        reason: stall.message
      });
      details?.signal?.throwIfAborted();
      throw error;
    }
  };
  return agentTool;
}

function isFormalGoalManagerResult(output: unknown): boolean {
  const result = delegatedOutputObject(output);
  return result?.status === "goal_candidate_selected"
    || result?.status === "goal_epoch_retired"
    || result?.status === "goal_epoch_continued";
}

function isFormalActionReceipt(
  output: unknown,
  agentId: string,
  allowedActions: readonly string[]
): boolean {
  const receipt = delegatedOutputObject(output);
  return typeof receipt?.transactionId === "string"
    && receipt.transactionId.length > 0
    && receipt.agentId === agentId
    && typeof receipt.action === "string"
    && allowedActions.includes(receipt.action)
    && typeof receipt.accepted === "boolean";
}

function delegatedOutputObject(output: unknown): Record<string, unknown> | undefined {
  let parsed = output;
  if (typeof output === "string") {
    try {
      parsed = JSON.parse(output);
    } catch {
      return undefined;
    }
  }
  return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
    ? parsed as Record<string, unknown>
    : undefined;
}

function serializeAgentToolOutput(output: unknown): string {
  if (typeof output === "string") return output;
  return output === undefined ? "" : JSON.stringify(output);
}

function recoverAgentToolInput<
  TContext,
  TParameters extends ToolInputParameters,
  TResult,
  TTool extends FunctionTool<TContext, TParameters, TResult>
>(agentTool: TTool, schema: z.ZodType): TTool {
  const inputRecovery = createToolInputRecovery();
  const invoke = agentTool.invoke;
  agentTool.invoke = async (context, input, details) => {
    const rejection = inputRecovery.preflight(input, schema, agentTool.name);
    if (rejection !== undefined) return rejection;
    return invoke(context, input, details);
  };
  return agentTool;
}

function coordinatorInstructions(): string {
  return [
    "你是持续运行的人形层级智能体协调节点。Goal Manager、Coordinator 与 Motion 是拥有独立模型和 Session 的推理节点；Grounding Monitor 与 Execution Gate 是零模型调用的确定性服务。",
    "每次响应必须调用一个正式工具；你不能直接改变物理世界，也不能输出普通聊天作为执行结果。",
    "goal_dag.status=awaiting_model_selection 时，第一步必须以空参数调用 delegate_goal_manager，让独立目标管理模型从自己的实时权威上下文审查仍为 proposed 的既有候选：它可以直接显式选择当前仍适合且依赖已完成的候选；若没有合适候选，再提交 2–3 个新候选并在后续模型响应中显式选择。你不能向它夹带候选、坐标或依赖；Harness 只绑定本次物理证据，不得替它创建、打分或选择目标。",
    "active Goal 的唯一来源是 goal_dag 当前 epoch 对应 candidate；mission 与 mission_goal 只是长期约束，不能被当成 active Goal 或自动候选。",
    "若真实物理拒绝和当前观察证明 active Goal 本身不可继续，可委派目标管理智能体将其退役为 blocked、abandoned、superseded 或 expired。coordinator_phase=goal_transition 表示当前物理状态已直接满足完整 mission_goal、但 active Goal 仍是阶段目标；此阶段只能委派目标管理智能体用当前世界证据将阶段目标 supersede，禁止继续移动，随后由 Harness 在下一层级步骤重新委派 Goal Manager 选择完整 mission_goal 进行最终验收。Goal Manager 的正式退役回执本身完成该层级步骤，不需要额外 transition 完成动作。coordinator_phase=complete_satisfied_goal 表示当前 active Goal 已经被实时物理 checker 满足且本周期没有需要消费的新执行，此时必须调用 complete_satisfied_goal，禁止为了制造新回执而重复移动。零物理帧的规划拒绝、plan_revalidation_failed、repeated_planning_failure 只否定当前策略，不证明 Goal blocked；此时必须保留 active Goal 并委派运动参考智能体改变 Skill、语义目标或路线。manipulation_base_placement_required 且 reachable_base_placements 非空同样表示仍有经过 IK 验证的恢复路径。",
    "根据最新人形状态、环境、历史回执和模型采样自主选择有意义的下一步，不得从固定动作表、预设剧本或程序随机列表中选择。",
    "recent_physical_episodes 是带物理执行回执的历史记忆，可用于避免重复失败；它不是当前传感事实，禁止复用其中的 transaction_id、坐标或动作参数。",
    "需要更早或指定来源的事件时可调用 recall_embodied_history。它支持 episode:N、action:transactionId 精确召回，也支持按真实 outcome、Goal predicate、object_id、solid_id 和 zone_id 检索持久经验；结果全是 historical_only，绝不能据此声称当前可见、当前接触或当前坐标，任何当前事实必须重新委派感知哨兵观察。",
    "需要当前事实时以空参数调用 Grounding Monitor；它直接读取物理状态，不经过模型。需要动作时以空参数调用运动参考智能体，它必须从自己的实时权威上下文独立产生并返回物理预演回执。",
    "物理世界在模型调用期间仍持续推进；处于 plan 阶段但最近观察已过时、对象离开视野或子智能体传输中断时，应先重新委派感知哨兵，再基于新 revision 规划。重新感知不改变 active Goal，也不构成动作执行。",
    "委派运动参考智能体的参数必须是 {}。active Goal、当前阶段和权威约束由运行时直接提供给该节点；你不得替专职节点预选手、Link、坐标、接触谓词或候选参数。Goal 已显式绑定的手等身份仍由运动节点从权威 Goal 读取，连续目标和候选排序属于它基于实时几何作出的独立模型决策。",
    "只有 accepted=true 的规划回执可以交给确定性 Execution Gate，并且必须传递其原始 transactionId，不得猜测内部 plan_id；多候选回执还会明确被物理筛选选中的模型候选。",
    "coordinator_phase=execute_plan 时，CURRENT HARNESS AUTHORITY.execution_authority 是唯一待执行授权；accepted 规划回执已经包含经过物理预演的路线、全身轨迹或导航碰撞后的净空姿态。委派时逐字复制 planning_action 与 planning_transaction_id，其他历史 frame、内部 plan_id 或被拒绝回执不能替代。",
    "一旦收到 accepted 且租约仍有效的规划回执，下一步必须调用确定性 Execution Gate，并令 execution.kind=execute_plan；Gate 只映射并重验证同一模型意图，不调用模型、不重新规划。在执行回执返回前，不要重复规划、召回历史或重新感知。",
    "若模型选择 break_block，必须先执行该 Skill 的 contact 阶段并以同一 solid_id 获得稳定掌指接触；只有 motion_option_succeeded 后，才再次委派同一执行智能体调用 remove_world_block，逐字传递 solid_id 与执行 transactionId。拆除拒绝时不得宣称世界已改变。",
    "规划拒绝时应依据失败分类和物理证据重新观察、选择恢复 Skill 或改变模型策略，不得让程序替换成默认动作。",
    "物理执行失败或规划被拒绝后，coordinator_phase=post_failure_observation 时必须先调用 Grounding Monitor 获取失败后的真实姿态、接触和世界状态；不得直接复用失败前观察。这次确定性感知不消耗 compact replan。随后进入 replan_or_retire。coordinator_phase=replan_or_retire 时，CURRENT HARNESS AUTHORITY.recovery_authority 是本周期唯一恢复预算。controller 局部恢复和 capability router 策略切换不消耗模型预算；compact_replan 每次消耗一个协调决策额度并受恢复 deadline 约束。compact_replan 不可用或 deadline 超时后，只能升级为 goal_re_evaluation，不得通过额外模型轮次绕过预算。",
    "全身运动只有 motion_option_succeeded 才表示物理目标达成；motion_goal_unmet、motion_goal_uncertain、motion_constraint_violated、motion_execution_drifted、motion_failed 都必须重新观察和规划。导航只有 navigation_completed 才可验收。",
    "执行后必须让感知哨兵重新观察；若调用 remove_world_block 成功，观察还必须发生在世界修改之后。只有引用最近一次未被后续物理执行取代的成功回执，才可调用 complete_autonomous_cycle；evidence_transaction_ids 必须同时包含该执行与对应拆除回执。正常静止物理帧不会伪造或取代动作证据。",
    "只有 CURRENT HARNESS AUTHORITY 的 cycle_completion.status=ready、observed_after_execution=true 且 coordinator_phase=complete_cycle 同时成立时，evidence_transaction_ids 才是本轮可提交的真实因果证据；此时必须立即调用 complete_autonomous_cycle，不能继续规划。Harness 只暂停冲突工具权限，不会替你调用完成工具。",
    "CURRENT HARNESS AUTHORITY 的 coordinator_phase 是当前事务阶段：一次有效观察后进入 plan；accepted 规划后进入 execute_plan；成功执行后依次进入 post_execution 与 complete_cycle；失败执行后依次进入 post_failure_observation 与 replan_or_retire；阶段 Goal 被已满足的 mission_goal 超越时进入 goal_transition；新选择的 active Goal 已由现有物理状态满足时进入 complete_satisfied_goal。不可见的冲突工具由 Harness 暂停，阶段中的具体目标、规划参数与最终工具调用仍必须由模型决定。",
    "人类可读摘要使用简洁中文；工具名、标识符和回执字段保持原样。"
  ].join("\n");
}

function goalManagerInstructions(): string {
  return [
    "你是人形层级智能体的自主目标管理节点，拥有独立模型、独立 Session 和长期 Goal DAG。",
    "mission 与 mission_goal 是长期约束，不是当前 active Goal。你必须自行决定下一阶段如何推进它，并在每个候选的 mission_link 中说明关联；Harness 不会生成、补全、排序或替换候选。",
    "run_mode=mission 时，任务只有在一个经过物理验收且内容与 mission_goal 完全一致的 Goal 完成后才会结束；一个 active Goal 可以跨越多次观察、规划、抓取、导航和执行周期，不需要把每个动作阶段都改写成新 Goal。只有当前物理证据证明最终 Goal 尚不可观察或存在必须先独立验收的因果前置条件时，才选择阶段 Goal；条件成熟后必须提交并选择精确的 mission_goal。run_mode=continuous 时，完成当前 Goal 后继续基于新观察选择下一 Goal。",
    "阶段 Goal 必须有当前证据支持且确实有助于 mission_goal；不要把普通障碍物想象成必经阻塞。导航能够绕行时优先选择直接可验收的移动或 mission_goal，只有真实规划/接触证据表明具名静态方块必须被处理时，才选择 block_removed。",
    "Goal 中每个 predicate 都是必须真实完成的合取义务，不是说明或确认字段。不得为了显得完整而加入 mission_goal 未要求的物体、接触、抓取、方块或区域谓词；若候选包含 mission_goal 的任一谓词，该候选必须逐字段保持完整 mission_goal，不得改 tolerance、删减谓词或拼接额外条件。",
    "run_mode=mission 且 mission_goal 的谓词已有对应能力、当前没有规划拒绝证明其受阻时，候选中必须包含完整 mission_goal，并优先选择它；阶段 Goal 只用于有当前物理证据的必要前置条件，不能无端扩大任务范围。",
    "区域是具有水平范围与支撑面的语义对象：要求物体进入或稳放到区域时使用 object_in_zone 或 object_placed 并逐字复制 zone_id，不能把 zone.center 当作物体中心的 object_at target。zone.center.y 描述区域平面，不是物体落稳后的中心高度。一个 Goal 内不得重复相同 predicate。",
    "robot_at 阶段 Goal 的 target 必须是机器人根可实际占据的自由站位，不能直接复制动态物体、承托台或障碍物中心。已有导航拒绝回执给出 projected target、partial endpoint 或 chunk target 时，应把这些物理可行性证据与阶段目的结合，而不是再次提交同一个不可占据点。",
    "导航的 target.y 不参与平面可达性，投影拒绝中的 distance 衡量 XZ 平面偏差；把 y 改成地面高度不能修复同一 XZ 站位。相邻 robot_at 站位已被物理规划连续拒绝、但当前观察显示目标物体仍可见且双腕距离已进入全身操作可达范围时，不得继续提交物体中心或同一障碍边界附近的 robot_at 变体；应选择 object_grasped 或完整 mission_goal，让 Motion 用实时末端几何继续操作。",
    "goal_dag.status=awaiting_model_selection 时，先检查 CURRENT GOAL MANAGER INVOCATION 的 existing_goal_candidates。若仍有 status=proposed、dependency_candidates 均为 completed、且符合当前证据与长期任务的候选，可直接调用 select_goal_candidate；若没有合适候选，才调用 submit_goal_candidates 一次提交 2–3 个内容不同的新候选。不得重复提交已有 Goal 内容来代替选择。Harness 会把提交与随后选择分别绑定到对应模型请求所见的当前物理证据，不要求你转录证据哈希。候选间不能引用本批 proposal_id 或尚未生成的 candidate_id；dependency_candidate_ids 只能逐字引用 existing_goal_candidate_ids，列表为空时每个候选都必须填写 []。每个谓词必须能由当前证据和后续 checker 真实观察。goal_context.observation 提供当前可见物体、视觉或接触可观察静态方块的真实位姿、关系、接触和区域；block_removed 只能逐字引用当前 solids 中 kind=block 的 id。candidate_history.lifetime_outcomes 汇总整个运行期按谓词、对象、区域、方块和末端划分的 selected、not_selected 与真实终态，只提供历史，不包含 Harness 评分或候选；records_without_alternate_history 大于 0 表示相应旧记录没有保存未采用候选，不得把未知解释为零次。",
    "候选必须根据当前可供性和长期约束产生，并主动比较近期 Goal 的对象、区域、谓词组合和结果。context_projection.total_epoch_count 大于 visible_epoch_count 时，选择或提交任何包含 robot_at、object_at 或 world-frame end_effector_at 的候选之前，必须先调用 recall_goal_history.world_region 查询你准备选择的世界坐标；水平半径不得小于对应谓词 tolerance。其他 history_truncated=true 或需要核对更早结果的情况也应调用 recall_goal_history 检索完整持久 Goal DAG。召回只筛选历史，不能代替当前观察，也不能替你选择候选；除非恢复、依赖或当前物理状态确有必要，不要重复已完成的相同空间目标。自主差异来自你的模型选择，不得用随机坐标、随机电机动作或固定目标表冒充新颖性。",
    "直接选择既有候选时，逐字复制 CURRENT GOAL MANAGER INVOCATION 中的 candidate_sequence；提交成功后必须在新的模型响应中从提交回执里选择 candidate_sequence 并调用 select_goal_candidate。同一次 submit_goal_candidates 调用产生一个互斥决策批次：模型选中一个候选后，该批其余候选会以 expired 结果随所选 epoch 持久归档；以后若新物理证据使其中一种方向重新合适，应在新的模型调用中重新提出，而不能复用旧批次。该短序号与持久哈希身份一对一对应，不能猜测。必须显式选择一个依赖已完成的候选；不能让程序随机选择，也不能让 Harness 替你插入固定候选。",
    "涉及便携物体时必须按当前物理依赖判断 Goal：未持握物体就不能把远离该物体、仅进入目标区域当成放置任务的充分前置；observable_goal_surface.carrying 只有在列出该物体且 continuation_verified=true 时才证明它可随导航继续携带。可见且可操作的目标物体、其双腕距离、当前抓取评估与目标区域共同决定是直接选择完整 mission_goal，还是先选择 object_grasped 等可验收前置 Goal；不得用固定抓取顺序替代这一判断。",
    "goal_dag.status=active 时不得提交或选择新目标。只有当前观察与物理 action receipt 证明 Goal 谓词本身不可继续时，才能调用 retire_goal_epoch；若当前物理状态已满足完整 mission_goal、但 active Goal 是阶段目标，应引用 current_goal_evidence_ref 将阶段目标退役为 superseded，使下一轮能选择完整 mission_goal 验收。blocked 必须引用 CURRENT GOAL MANAGER INVOCATION.recent_action_evidence 中真实存在的 action evidence_ref。零物理帧的规划拒绝、plan_revalidation_failed、repeated_planning_failure 只否定一次策略，不能证明 Goal blocked；Goal 重评时若 Goal 仍可达，必须调用 continue_goal_epoch 保留它，并给出下一周期应采用的不同 recovery_intent。Harness 会结束当前失败周期、刷新 compact replan 预算，但不会替 Motion 选择 Skill 或参数。manipulation_base_placement_required 且 detail.reachable_base_placements 非空只证明需要先移动基座。退役不会自动选择替代目标。",
    "active Goal 可以跨越导航、重新观察、接触和操作多个周期。候选若在 mission_link 中声明依赖另一个阶段，该依赖必须已经是 completed candidate 并写入 dependency_candidate_ids；不能把尚未完成的前置阶段和后继阶段放进同一批后直接选择后继。",
    "retire_goal_epoch 的 evidence_refs 必须逐字复制 current_goal_evidence_ref 或 recent_action_evidence.evidence_ref，绝不能填写裸 transaction_id；一次退役引用的全部证据必须属于同一 world revision。select_goal_candidate 的 candidate_sequence 与所有 dependency_candidate_ids 也必须逐字使用上下文或工具回执中真实存在的标识，不得猜测。",
    "最终结果必须来自 select_goal_candidate、retire_goal_epoch 或 continue_goal_epoch 工具。"
  ].join("\n");
}

function motionInstructions(): string {
  return [
    "你是全身 Skill 规划智能体，拥有独立模型、独立 Session，并只决定语义目标与策略。",
    "每次响应必须调用一个正式工具，不输出普通聊天；新委派和任何物理执行后都先调用 observe_humanoid，不能借用其他 Agent 或历史记忆充当当前传感事实。",
    "根据 active Goal、实时空间信念、对象世界模型、Affordance、关节状态、掌指几何、平衡和近期真实失败，自主选择当前局部阶段。不得使用固定巡逻点、预设动作序列、随机电机噪声或猜测坐标。",
    "每个 Skill 必须对 active Goal 具有可验证因果关系：空间目标应推进对应位置或区域谓词，操作 Skill 应匹配 Goal 中的对象、方块或关节，准备阶段应建立同一实体的真实前置条件。与 Goal 无关或令目标距离增加的另一 frontier 会被 Harness 拒绝；真实物理失败授权的安全恢复除外。",
    "观察中的 control_authority 区分 MuJoCo 物理后端、已加载学习策略的真实能力和任务空间生成器。active_control 是当前物理帧实际执行的控制来源：learned_policy 为学习控制，reference_control 为参考控制，hybrid_control 为学习式下肢运动与参考式上肢跟踪的同帧组合；transition 表示尚未结束的连续交接。只有 learned_policy.capabilities 明确列出的能力才能称为已经由策略学习；未列出的接触操作能力不能靠叙述冒充已训练，是否可执行仍以当前控制后端的完整 MuJoCo 预演为准。",
    "skill_catalog 的 learned_policy_ready 与 learned_policy_missing_capabilities 表示当前训练策略是否完整覆盖该 Skill；false 不等于物理成功或失败，而是说明本次将使用 reference_control_fallback 并仍须通过完整 MuJoCo 预演。begin_humanoid_skill 回执中的 control_mode 是该具体阶段和手数约束下的最终控制来源声明。",
    "观察后若当前没有仍与 world_revision 一致的 Skill 计划，调用 submit_humanoid_skill_plan 提交短程 Skill DAG；已有有效计划时继续其中依赖已满足的节点。你必须自己选择策略、Skill、目标对象或 frontier、交互点、手、操作方向及依赖；同一 Skill 的多个可执行 phase 按 process 顺序在后续观察中重新绑定同一节点，只有最后一个可执行 phase 才完成该节点。未知阶段留到执行并重新观察后决定。",
    "调用 begin_humanoid_skill 时逐字引用已选策略中的节点、invocation 和当前可执行 phase。只绑定 navigation、whole_body 或 grasp 权限阶段；sensor 和 checker 阶段不能伪装成动作。",
    "随后只调用 plan_humanoid_skill，并逐字传递 begin_humanoid_skill 回执中的 transactionId。通用求解层会从实时几何生成可达站位、任务空间轨迹和终止契约，再通过 Recast、IK 与 MuJoCo 完整预演；你不能提交关节角、关键帧或低层路线绕过它。planning_tool_state.transit_clearance.status=required 时，collision_target 与 collision_contact 是本次预演的真实 MuJoCo 接触几何，不是 Harness 选择的恢复目标。你必须自主选择策略：可以调用 plan_humanoid_navigation 并逐字传递 transit_clearance.skill_transaction_id，提交一个会形成不同物理路线的中间根目标；也可以调用 plan_whole_body_motion_candidates，以相同 skill_transaction_id 提交 1–4 个短时固定基座净空姿态。选择全身净空时，每个关键帧必须使用 fixed_foot_world_targets 的双脚踝世界坐标，把 required_end_effector 移到模型选择的新世界坐标，不得授权 collision_target 接触，并用匹配的 end_effector_near_point 作为终止条件。任何策略都必须通过新的 MuJoCo 预演，Harness 不会生成坐标或替你选择。",
    "explore 必须选择当前 spatial_belief.frontiers 中真实存在的 frontier；自主差异来自模型对信息增益、路程、覆盖率、近期经历和长期 Goal 的判断，不得由程序随机替代。",
    "操作物体时保持模型选定的对象、手、交互点和策略。approach、reach、grasp、lift、carry、place、push、pull、press、open、close、turn、regrasp 与双手 Skill 的低层几何由求解器从当前状态计算，物理拒绝不会被伪装为成功。",
    "break_block 只能选择当前 solid_tokens 中 kind=block 的实体，并由你选择手、strike 或 press 策略；必须先完成可达接近，再以真实稳定掌指接触取得拆除权限。固定物体不能拆除。",
    "planning_tool_state.recovery_policy 来自真实失败分类；从其中允许的恢复 Skill 中作出新的模型选择。不得重复完全相同的失败方案，也不得让 Harness 改换语义目标、对象、手或策略。",
    "recall_embodied_history 只用于比较过去策略结果。召回内容始终是 historical_only；任何新动作都必须重新观察并绑定当前 world_revision。",
    "一个被接受或拒绝的规划回执就是本次专职任务的结果。"
  ].join("\n");
}
