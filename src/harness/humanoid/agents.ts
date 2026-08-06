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
import type { HumanoidCycleCompletionReadiness } from "./cycle-causal-evidence.js";
import type { HumanoidCoordinatorPhase } from "./run-runtime.js";
import { GOAL_HISTORY_PREDICATE_TYPES } from "./goal-history.js";

const SpecialistDelegationSchema = z.object({}).strict();
const ExecutionTaskSchema = z.object({
  task: z.enum(["execute_plan", "remove_world_block"]),
  objective: z.string().trim().min(1),
  planning_action: z.enum([
    "plan_whole_body_motion",
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
  ]).nullable().describe("execute_plan 时填写规划回执 action；remove_world_block 时必须为 null"),
  planning_transaction_id: z.string().trim().min(1).nullable()
    .describe("execute_plan 时逐字复制已接受规划回执 transactionId；remove_world_block 时必须为 null"),
  solid_id: z.string().trim().min(1).nullable()
    .describe("remove_world_block 时填写目标 solid_id；execute_plan 时必须为 null"),
  execution_transaction_id: z.string().trim().min(1).nullable()
    .describe("仅 remove_world_block 使用，填写先前成功物理执行的 transactionId；execute_plan 时必须为 null")
}).strict().superRefine((task, context) => {
  const planningMode = task.task === "execute_plan";
  for (const [field, value] of [
    ["planning_action", task.planning_action],
    ["planning_transaction_id", task.planning_transaction_id]
  ] as const) {
    if ((value !== null) !== planningMode) {
      context.addIssue({
        code: "custom",
        path: [field],
        message: `Execution-plan field ${field} does not match task mode`
      });
    }
  }
  for (const [field, value] of [
    ["solid_id", task.solid_id],
    ["execution_transaction_id", task.execution_transaction_id]
  ] as const) {
    if ((value !== null) === planningMode) {
      context.addIssue({
        code: "custom",
        path: [field],
        message: `Block-removal field ${field} does not match task mode`
      });
    }
  }
});
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
const GoalTransitionCompletionSchema = z.object({
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
      truncated: projection.history_truncated === true
    },
    observable_goal_surface: {
      predicate_types: [...GOAL_HISTORY_PREDICATE_TYPES],
      zone_ids: stringArray(observation.zone_ids),
      visible_object_ids: stringArray(observation.visible_object_ids),
      portable_object_ids: objects.flatMap((object) => (
        object.portable === true && typeof object.id === "string" ? [object.id] : []
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

function sentryInvocationInput(): string {
  return "请读取当前权威 MuJoCo 状态，并只返回正式观察工具的真实回执。";
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

function executionTaskJsonInputBuilder(
  { params }: { params: z.infer<typeof ExecutionTaskSchema> }
): string {
  return [
    "CURRENT EXECUTION AUTHORITY",
    "该 JSON 是上级模型选择且 Harness 验证过的本次唯一执行授权。它不会替你调用工具；你必须调用匹配的正式执行工具，并逐字复制授权中的 transactionId，不能改写为内部 plan_id。",
    JSON.stringify(params)
  ].join("\n\n");
}

export const HUMANOID_AGENT_TOOL_CONTRACTS = {
  goalManager: {
    toolName: "delegate_goal_manager",
    targetRole: "goal_manager",
    targetAgentId: HUMANOID_AGENT_IDS.goalManager,
    inputBuilderContract: "goal_manager_authority_envelope_v2",
    inputBuilder: () => goalManagerInvocationInput({}),
    runOptions: {
      sessionAgentId: HUMANOID_AGENT_IDS.goalManager,
      contextSource: "parent_run_context",
      maxTurns: "sdk_default"
    },
    resumeContextStrategy: "merge",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "nested_agent_final_output_text"
  },
  sentry: {
    toolName: "delegate_humanoid_sentry",
    targetRole: "sentry",
    targetAgentId: HUMANOID_AGENT_IDS.sentry,
    inputBuilderContract: "live_authority_delegation_v1",
    inputBuilder: () => sentryInvocationInput(),
    runOptions: {
      sessionAgentId: HUMANOID_AGENT_IDS.sentry,
      contextSource: "parent_run_context",
      maxTurns: "sdk_default"
    },
    resumeContextStrategy: "merge",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "nested_agent_final_output_text"
  },
  motion: {
    toolName: "delegate_motion_reference",
    targetRole: "motion",
    targetAgentId: HUMANOID_AGENT_IDS.motion,
    inputBuilderContract: "motion_authority_envelope_v1",
    inputBuilder: () => motionInvocationInput({}),
    runOptions: {
      sessionAgentId: HUMANOID_AGENT_IDS.motion,
      contextSource: "parent_run_context",
      maxTurns: "sdk_default"
    },
    resumeContextStrategy: "merge",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "nested_agent_final_output_text"
  },
  executor: {
    toolName: "delegate_physics_executor",
    targetRole: "executor",
    targetAgentId: HUMANOID_AGENT_IDS.executor,
    inputBuilderContract: "validated_execution_task_json_v1",
    inputBuilder: executionTaskJsonInputBuilder,
    runOptions: {
      sessionAgentId: HUMANOID_AGENT_IDS.executor,
      contextSource: "parent_run_context",
      maxTurns: "sdk_default"
    },
    resumeContextStrategy: "merge",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "nested_agent_final_output_text"
  }
} as const;

type HumanoidAgentToolContract =
  typeof HUMANOID_AGENT_TOOL_CONTRACTS[keyof typeof HUMANOID_AGENT_TOOL_CONTRACTS];

export const HUMANOID_CAPABILITIES = [
  "recall_goal_history",
  "submit_goal_candidates",
  "select_goal_candidate",
  "retire_goal_epoch",
  "observe_humanoid",
  "recall_embodied_history",
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
  validateGoalTransition(): JsonValue;
};

export interface HumanoidAgentHierarchy {
  goalManager: Agent;
  coordinator: Agent;
  sentry: Agent;
  motion: Agent;
  executor: Agent;
  goalManagerSession: Session;
  coordinatorSession: Session;
  session(agentId: string): Session | undefined;
}

export function createHumanoidAgentHierarchy(input: {
  createModel: (agentId: string, provider: ModelProviderConfig) => Model;
  createSession: (agentId: string) => Session;
  callModelInputFilter: CallModelInputFilter;
  provider: ProviderConfig;
  runtime: HumanoidHierarchyRuntime;
  onAgentStream?: (agentId: string, event: RunStreamEvent) => void | Promise<void>;
}): HumanoidAgentHierarchy {
  const models = new Set<Model>();
  const sessions = new Map<string, Session>();
  const sessionOwners = new Set<Session>();
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
      ...(provider.maxOutputTokens === undefined
        ? {}
        : { maxTokens: provider.maxOutputTokens }),
      parallelToolCalls: false,
      toolChoice: "required"
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
      retire_goal_epoch: "goal_epoch_retired"
    })
  });
  const sentry = new Agent({
    name: "人形感知哨兵",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.sentry, sentryInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.sentry),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.sentry),
    tools: createHumanoidActionTools(
      input.runtime,
      HUMANOID_AGENT_IDS.sentry,
      ["observe_humanoid"]
    ),
    resetToolChoice: false,
    toolUseBehavior: receiptToolUseBehavior(["observe_humanoid"])
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
        ["observe_humanoid"]
      ),
      createHumanoidEmbodiedRecallTool(input.runtime),
      ...createHumanoidActionTools(
        input.runtime,
        HUMANOID_AGENT_IDS.motion,
        ["plan_whole_body_motion_candidates", "plan_humanoid_navigation"]
      )
    ],
    resetToolChoice: false,
    toolUseBehavior: receiptToolUseBehavior([
      "plan_whole_body_motion_candidates",
      "plan_humanoid_navigation"
    ])
  });
  const executor = new Agent({
    name: "人形物理执行智能体",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.executor, executorInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.executor),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.executor),
    tools: createHumanoidActionTools(
      input.runtime,
      HUMANOID_AGENT_IDS.executor,
      ["execute_whole_body_motion", "execute_humanoid_navigation", "remove_world_block"]
    ),
    resetToolChoice: false,
    toolUseBehavior: receiptToolUseBehavior([
      "execute_whole_body_motion",
      "execute_humanoid_navigation",
      "remove_world_block"
    ])
  });

  const goalManagerSession = ownSession(HUMANOID_AGENT_IDS.goalManager);
  ownSession(HUMANOID_AGENT_IDS.sentry);
  ownSession(HUMANOID_AGENT_IDS.motion);
  ownSession(HUMANOID_AGENT_IDS.executor);
  const coordinatorSession = ownSession(HUMANOID_AGENT_IDS.coordinator);
  const agentToolSession = (contract: HumanoidAgentToolContract): Session => {
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
  const sentryContract = HUMANOID_AGENT_TOOL_CONTRACTS.sentry;
  const motionContract = HUMANOID_AGENT_TOOL_CONTRACTS.motion;
  const executorContract = HUMANOID_AGENT_TOOL_CONTRACTS.executor;
  const coordinator = new Agent({
    name: "人形自主协调智能体",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.coordinator, coordinatorInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.coordinator),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.coordinator),
    tools: [
      createHumanoidEmbodiedRecallTool(input.runtime),
      recoverAgentToolInput(scopeAgentToolInvocation(
        HUMANOID_AGENT_IDS.goalManager,
        goalManager.asTool({
        toolName: goalManagerContract.toolName,
        toolDescription: "让独立目标管理智能体基于当前物理证据提交 2–3 个候选并显式选择下一 Goal，或证据化退役当前不可达 Goal。",
        parameters: SpecialistDelegationSchema,
        inputBuilder: () => goalManagerInvocationInput(
          input.runtime.contextAnchor(HUMANOID_AGENT_IDS.goalManager)
        ),
        includeInputSchema: goalManagerContract.includeInputSchema,
        needsApproval: goalManagerContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: { session: agentToolSession(goalManagerContract) },
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
        })
      ), SpecialistDelegationSchema),
      recoverAgentToolInput(scopeAgentToolInvocation(
        HUMANOID_AGENT_IDS.sentry,
        sentry.asTool({
        toolName: sentryContract.toolName,
        toolDescription: "让独立感知智能体读取当前 MuJoCo 人形、接触、平衡、物体和导航状态。",
        parameters: SpecialistDelegationSchema,
        inputBuilder: () => sentryInvocationInput(),
        includeInputSchema: sentryContract.includeInputSchema,
        needsApproval: sentryContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: { session: agentToolSession(sentryContract) },
        resumeState: { contextStrategy: sentryContract.resumeContextStrategy },
        ...(input.onAgentStream
          ? { onStream: ({ event }) => input.onAgentStream!(HUMANOID_AGENT_IDS.sentry, event) }
          : {})
        })
      ), SpecialistDelegationSchema),
      recoverAgentToolInput(scopeAgentToolInvocation(
        HUMANOID_AGENT_IDS.motion,
        motion.asTool({
        toolName: motionContract.toolName,
        toolDescription: "让独立运动参考智能体读取当前状态，提出多个按偏好排序且分别经过完整物理预演的连续全身动作候选，或规划双足路线。",
        parameters: SpecialistDelegationSchema,
        inputBuilder: () => motionInvocationInput(
          input.runtime.contextAnchor(HUMANOID_AGENT_IDS.motion)
        ),
        includeInputSchema: motionContract.includeInputSchema,
        needsApproval: motionContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: { session: agentToolSession(motionContract) },
        resumeState: { contextStrategy: motionContract.resumeContextStrategy },
        ...(input.onAgentStream
          ? { onStream: ({ event }) => input.onAgentStream!(HUMANOID_AGENT_IDS.motion, event) }
          : {})
        })
      ), SpecialistDelegationSchema),
      recoverAgentToolInput(scopeAgentToolInvocation(
        HUMANOID_AGENT_IDS.executor,
        executor.asTool({
        toolName: executorContract.toolName,
        toolDescription: "让独立执行智能体消费已接受规划并在 MuJoCo 中真实执行，或用同周期稳定接触回执提交方块拆除事务。",
        parameters: ExecutionTaskSchema,
        inputBuilder: executorContract.inputBuilder,
        includeInputSchema: executorContract.includeInputSchema,
        needsApproval: executorContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: { session: agentToolSession(executorContract) },
        resumeState: { contextStrategy: executorContract.resumeContextStrategy },
        ...(input.onAgentStream
          ? { onStream: ({ event }) => input.onAgentStream!(HUMANOID_AGENT_IDS.executor, event) }
          : {})
        })
      ), ExecutionTaskSchema),
      cycleCompletionTool(input.runtime),
      goalTransitionCompletionTool(input.runtime)
    ],
    resetToolChoice: false,
    toolUseBehavior: verifiedStatusToolUseBehavior({
      complete_autonomous_cycle: "cycle_completed",
      complete_goal_transition: "goal_transition_completed"
    })
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
  if (name === "complete_goal_transition") return phase === "goal_selection";
  if (name === "delegate_goal_manager") {
    return phase === "goal_selection"
      || phase === "replan_or_retire" && runtime.goalRetirementDelegationAvailable();
  }
  if (name === "delegate_humanoid_sentry") {
    const phaseAllowsObservation = phase === "observe_or_plan"
      || phase === "plan"
      || phase === "replan_or_retire"
      || phase === "post_execution";
    return phaseAllowsObservation
      && (runtime.sentryDelegationAvailable?.() ?? true);
  }
  if (name === "delegate_motion_reference") {
    return phase === "observe_or_plan"
      || phase === "plan"
      || phase === "replan_or_retire";
  }
  if (name === "delegate_physics_executor") {
    return runtime.executorDelegationAvailable();
  }
  return phase === "observe_or_plan"
    || phase === "plan"
    || phase === "replan_or_retire";
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

function goalTransitionCompletionTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof GoalTransitionCompletionSchema, string> {
  return tool<typeof GoalTransitionCompletionSchema, unknown, string>({
    name: "complete_goal_transition",
    description: "仅在目标管理智能体刚刚以物理证据退役 active Goal 后结束本轮；不选择替代 Goal，也不代表任务成功。",
    parameters: GoalTransitionCompletionSchema,
    strict: true,
    execute: (input) => JSON.stringify({
      status: "goal_transition_completed",
      summary: input.summary,
      transition: runtime.validateGoalTransition()
    })
  });
}

function receiptToolUseBehavior(
  terminalActions: readonly string[]
): ToolUseBehavior {
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output"
        || !terminalActions.includes(result.tool.name)) continue;
      const output = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output);
      try {
        const receipt = JSON.parse(output) as Partial<HumanoidActionReceipt>;
        if (receipt.transactionId
          && receipt.action === result.tool.name
          && typeof receipt.accepted === "boolean") {
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
    "你是持续运行的人形层级智能体协调节点，每个专职节点都有独立模型和独立 Session。",
    "每次响应必须调用一个正式工具；你不能直接改变物理世界，也不能输出普通聊天作为执行结果。",
    "goal_dag.status=awaiting_model_selection 时，第一步必须以空参数调用 delegate_goal_manager，让独立目标管理模型从自己的实时权威上下文审查仍为 proposed 的既有候选：它可以直接显式选择当前仍适合且依赖已完成的候选；若没有合适候选，再提交 2–3 个新候选并在后续模型响应中显式选择。你不能向它夹带候选、坐标或依赖；Harness 只绑定本次物理证据，不得替它创建、打分或选择目标。",
    "active Goal 的唯一来源是 goal_dag 当前 epoch 对应 candidate；mission 与 mission_goal 只是长期约束，不能被当成 active Goal 或自动候选。",
    "若真实物理拒绝和当前观察证明 active Goal 不可继续，可委派目标管理智能体将其退役为 blocked、abandoned、superseded 或 expired；退役后调用 complete_goal_transition 结束本轮，下一轮仍必须由目标管理模型重新提出并选择。manipulation_base_placement_required 且 reachable_base_placements 非空表示当前根姿态不可达但仍存在经过 IK 验证的恢复路径，不是 Goal blocked 证据；此时必须继续委派运动参考智能体，让它自主选择样本并规划导航。",
    "根据最新人形状态、环境、历史回执和模型采样自主选择有意义的下一步，不得从固定动作表、预设剧本或程序随机列表中选择。",
    "recent_physical_episodes 是带物理执行回执的历史记忆，可用于避免重复失败；它不是当前传感事实，禁止复用其中的 transaction_id、坐标或动作参数。",
    "需要更早或指定来源的事件时可调用 recall_embodied_history。它支持 episode:N、action:transactionId 精确召回，也支持按真实 outcome、Goal predicate、object_id、solid_id 和 zone_id 检索持久经验；结果全是 historical_only，绝不能据此声称当前可见、当前接触或当前坐标，任何当前事实必须重新委派感知哨兵观察。",
    "需要当前事实时以空参数调用感知哨兵；需要动作时以空参数调用运动参考智能体，它必须从自己的实时权威上下文独立产生并返回物理预演回执。",
    "物理世界在模型调用期间仍持续推进；处于 plan 阶段但最近观察已过时、对象离开视野或子智能体传输中断时，应先重新委派感知哨兵，再基于新 revision 规划。重新感知不改变 active Goal，也不构成动作执行。",
    "委派运动参考智能体的参数必须是 {}。active Goal、当前阶段和权威约束由运行时直接提供给该节点；你不得替专职节点预选手、Link、坐标、接触谓词或候选参数。Goal 已显式绑定的手等身份仍由运动节点从权威 Goal 读取，连续目标和候选排序属于它基于实时几何作出的独立模型决策。",
    "只有 accepted=true 的规划回执可以交给物理执行智能体，并且必须传递其原始 transactionId，不得猜测内部 plan_id；多候选回执还会明确被物理筛选选中的模型候选。",
    "coordinator_phase=execute_plan 时，CURRENT HARNESS AUTHORITY.execution_authority 是唯一待执行授权；plan_humanoid_navigation 的 accepted 回执本身就是可执行规划，不需要也不得另找全身规划。委派时逐字复制其中 planning_action 与 planning_transaction_id，其他历史 frame、plan_id 或被拒绝回执一律不能替代。",
    "一旦收到 accepted 且租约仍有效的规划回执，下一步必须以 task=execute_plan 直接委派物理执行智能体；Harness 会在最新权威状态上重验证同一模型意图。在执行回执返回前，不要重复规划、召回历史或重新感知。",
    "若模型选择的本轮意图要求拆除静态方块，必须先让全身候选以同一 solid_id 的 body_contact_solid 或 hand_contact_solid 作为必需终止条件并真实执行；只有 motion_option_succeeded 后，才再次委派同一执行智能体调用 remove_world_block，逐字传递该 solid_id 与执行 transactionId。拆除拒绝时不得宣称世界已改变。",
    "规划拒绝时应依据 failures 与 evidence 重新观察或提出不同的连续全身约束，不得让程序替换成默认动作。",
    "全身运动只有 motion_option_succeeded 才表示物理目标达成；motion_goal_unmet、motion_goal_uncertain、motion_constraint_violated、motion_execution_drifted、motion_failed 都必须重新观察和规划。导航只有 navigation_completed 才可验收。",
    "执行后必须让感知哨兵重新观察；若调用 remove_world_block 成功，观察还必须发生在世界修改之后。只有引用最近一次未被后续物理执行取代的成功回执，才可调用 complete_autonomous_cycle；evidence_transaction_ids 必须同时包含该执行与对应拆除回执。正常静止物理帧不会伪造或取代动作证据。",
    "只有 CURRENT HARNESS AUTHORITY 的 cycle_completion.status=ready、observed_after_execution=true 且 coordinator_phase=complete_cycle 同时成立时，evidence_transaction_ids 才是本轮可提交的真实因果证据；此时必须立即调用 complete_autonomous_cycle，不能继续规划。Harness 只暂停冲突工具权限，不会替你调用完成工具。",
    "CURRENT HARNESS AUTHORITY 的 coordinator_phase 是当前事务阶段：一次有效观察后进入 plan；accepted 规划后进入 execute_plan；成功执行后依次进入 post_execution 与 complete_cycle。不可见的冲突工具由 Harness 暂停，阶段中的具体目标、规划参数与最终工具调用仍必须由模型决定。",
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
    "goal_dag.status=awaiting_model_selection 时，先检查 CURRENT GOAL MANAGER INVOCATION 的 existing_goal_candidates。若仍有 status=proposed、依赖均已完成且符合当前证据与长期任务的候选，可直接调用 select_goal_candidate；若没有合适的既有候选，才调用 submit_goal_candidates 一次提交 2–3 个内容不同的新候选。不得重复提交已有 Goal 内容来代替选择。Harness 会把提交与随后选择分别绑定到对应模型请求所见的当前物理证据，不要求你转录证据哈希。候选间不能引用本批 proposal_id 或尚未生成的 candidate_id；dependency_candidate_ids 只能逐字引用 existing_goal_candidate_ids，列表为空时每个候选都必须填写 []。每个谓词必须能由当前证据和后续 checker 真实观察。goal_context.observation 提供当前可见物体、视觉或接触可观察静态方块的真实位姿、关系、接触和区域；block_removed 只能逐字引用当前 solids 中 kind=block 的 id。goal_context.autonomy 提供有界历史计数与能力面，它不包含 Harness 评分或候选。",
    "候选必须根据当前可供性和长期约束产生，并主动比较近期 Goal 的对象、区域、谓词组合和结果；context_projection.history_truncated=true 或需要核对更早结果时，调用 recall_goal_history 检索完整持久 Goal DAG。召回只用于历史比较，不能代替当前观察。除非恢复、依赖或当前物理状态确有必要，不要重复相同目标内容。自主差异来自你的模型选择，不得用随机坐标、随机电机动作或固定目标表冒充新颖性。",
    "直接选择既有候选时，逐字复制 CURRENT GOAL MANAGER INVOCATION 中的 candidate_sequence；提交成功后必须在新的模型响应中从提交回执里选择 candidate_sequence 并调用 select_goal_candidate。该短序号与持久哈希身份一对一对应，不能猜测。必须显式选择一个依赖已完成的候选；不能让程序随机选择，也不能让 Harness 替你插入固定候选。",
    "涉及便携物体时必须按当前物理依赖判断 Goal：未持握物体就不能把远离该物体、仅进入目标区域当成放置任务的充分前置；observable_goal_surface.carrying 只有在列出该物体且 continuation_verified=true 时才证明它可随导航继续携带。可见且可操作的目标物体、其双腕距离、当前抓取评估与目标区域共同决定是直接选择完整 mission_goal，还是先选择 object_grasped 等可验收前置 Goal；不得用固定抓取顺序替代这一判断。",
    "goal_dag.status=active 时不得提交或选择新目标。只有当前观察与物理 action receipt 支持时，才能调用 retire_goal_epoch；blocked 必须引用 CURRENT GOAL MANAGER INVOCATION.recent_action_evidence 中真实存在的 action evidence_ref。manipulation_base_placement_required 且 detail.reachable_base_placements 非空只证明需要先移动基座，不能证明 Goal blocked。退役不会自动选择替代目标。",
    "active Goal 可以跨越导航、重新观察、接触和操作多个周期。候选若在 mission_link 中声明依赖另一个阶段，该依赖必须已经是 completed candidate 并写入 dependency_candidate_ids；不能把尚未完成的前置阶段和后继阶段放进同一批后直接选择后继。",
    "retire_goal_epoch 的 evidence_refs 必须逐字复制 current_goal_evidence_ref 或 recent_action_evidence.evidence_ref，绝不能填写裸 transaction_id；一次退役引用的全部证据必须属于同一 world revision。select_goal_candidate 的 candidate_sequence 与所有 dependency_candidate_ids 也必须逐字使用上下文或工具回执中真实存在的标识，不得猜测。",
    "最终结果必须来自 select_goal_candidate 或 retire_goal_epoch 工具。"
  ].join("\n");
}

function sentryInstructions(): string {
  return [
    "你是人形层级智能体的感知哨兵，拥有独立模型与上下文。",
    "调用 observe_humanoid 返回当前 MuJoCo 根姿态、关键 Link、双脚和平衡、末端位置、手部协调目标、掌指碰撞面几何、必要接触、头部传感器可见物体、带来源版本的持久 3D 物体记录，以及当前帧支撑脱离和抓取稳定证据；不得补写、猜测或模拟视野外状态。物理 Option 成功、失败或不可确定后都应以新的观察作为下一次规划依据。",
    "工具回执就是本次专职任务的结果。"
  ].join("\n");
}

function motionInstructions(): string {
  return [
    "你是全身运动参考智能体，拥有独立模型与上下文。",
    "每次模型响应直接调用一个正式工具，不复述任务、工具 schema 或推理过程。只做一次几何判断并立即提交给物理预演裁决；禁止在同一响应中反复重算同一轨迹、反复起草参数或多次声明准备调用工具。一次委派只规划当前状态下可以真实执行的下一阶段，不试图在一个动作制品中完成整条任务链。",
    "工具返回 invalid_tool_input 或 repeated_invalid_tool_input 后，下一响应只能重新调用同一工具一次：保留未被 validation_issues 指出的有效字段，只修正列出的字段及其必要依赖；禁止重新解释 schema、重新规划无关字段或输出说明文字。",
    "每次新委派的第一次规划前必须由你自己调用 observe_humanoid；任何物理执行之后也必须重新观察。其他 Agent 的观测不会注入你的独立 Session，未观察或执行后沿用旧观测的规划会被 Harness 拒绝，且 Harness 不会共享、补全或生成参数。观察后根据实时根姿态、关键 Link、末端、双脚接触、平衡、掌指碰撞面、可见物体、带 age_revisions 的记忆和导航状态决定连续全身目标。观察的 root.heading 与 CURRENT HARNESS AUTHORITY.robot.root_heading 给出根偏航角以及局部前向、左向在世界坐标中的单位向量；root_velocity 的 forward_mps 与 lateral_mps 必须按这两个真实向量解释。朝世界目标移动时用目标位移分别点乘这两个单位向量判断局部速度方向，不能把世界坐标增量直接当成机体前向。interaction 只提供当前权威的目标区域、携带生命周期、抓取阈值以及物体相对骨盆和双腕的几何关系，不包含推荐动作；你必须自行选择阶段、坐标、时序与候选顺序。委派目标或 active Goal 已明确抓取手时必须保持该手，不得改用另一只手。",
    "非导航动作必须使用 plan_whole_body_motion_candidates，一次提交一个批次顶层 objective、一个所有候选共享的 termination，以及 1 至 3 个按你偏好排序的连续全身 candidates；不得把 objective 或 termination 放进单个候选。局部接近、接触、抓取、抬升或释放默认只提交 1 个短而完整的候选；只有你确实有几何或时序上不同的替代策略时才提交第 2 或第 3 个，禁止为凑数复制候选。共同 termination 一旦绑定了手、对象、Link 或接触面，每个候选都必须保持这些身份；候选只能改变到达它的连续轨迹、位姿、时序或姿态策略，不能以换手或换接触对象制造差异。使用 plan_humanoid_navigation 输出你选择的世界目标。不要生成、猜测或复制任何关节角；运动后端负责由任务空间目标求解连续全身参考。",
    "严格按工具 Schema 的扁平字段提交：每个 candidate 只含 id、intent、duration_seconds、contacts、keyframes；contacts 的 hand_object 直接填写 type、hand_surface、object_id、required；每个 keyframe 的 channels 是数组。root_velocity 直接填写 type、forward_mps、lateral_mps，禁止 velocity 包装。hand_coordination 必须填写 type 与 coordination:{left:{thumb_opposition,thumb_curl,index_curl,middle_curl},right:{thumb_opposition,thumb_curl,index_curl,middle_curl}}，禁止数组或扁平八字段。末端通道必须直接填写 type、end_effector、frame、position、tolerance_m 及成对的 orientation、orientation_tolerance_rad；end_effector 只能是 left_wrist、right_wrist、left_ankle、right_ankle。禁止使用 value、target、target_pose、left_wrist 等包装对象，也不得给 candidate 重复添加 termination。",
    "若委派目标包含接近、接触、操作、离开或继续导航等多个阶段，或目标明显超出单个 8 秒 Option 的可达范围，先自主选择当前阶段的可达目标；需要长距离接近时调用 plan_humanoid_navigation，进入真实可操作范围后再由后续周期规划接触或操作。",
    "纯导航阶段不要混入手腕、脚踝或抓取目标；直接调用 plan_humanoid_navigation。target 是根节点可占据的开放地面坐标，不是物体或障碍物中心；arrival_heading 是到达后由真实双足控制器验收的朝向，准备观察或操作物体时使用 type=face_point 并指向该物体的当前世界坐标，不要求朝向时明确填 null。若拒绝回执给出 partial_endpoint、projected target 或 chunk_target，下一次调用应把该结构化物理可行性证据作为新决策依据。partial_endpoint 是失败预演实际到达的位置，不是 Harness 生成的替代目标。target.y 不参与平面路线选择，可使用当前根高度。",
    "termination 必须描述本轮可由当前传感状态验证的物理结果，例如根节点、具名手腕/脚踝末端到达位置或可选朝向、身体 Link 到达位置、接触、物体到达位置、区域、grasp_verified、object_released 或 object_settled_on_support。每个谓词是按 type 区分的严格联合类型，只填写该类型声明的字段。普通接近、接触和抓取直接使用 mode=all，由全部谓词隐式 AND；其结构只有 mode、option_id、predicates、stable_steps。只有确实需要前置或执行中不变量时才使用 mode=phased；每个 condition 都必须同时填写 predicate_indexes 与 not_predicate_indexes，没有否定项也要填写 []。stable_steps 只允许出现在 termination 顶层和 precondition 内；during 与 terminal 只能包含 condition，不得加入 stable_steps。不得引用当前不可见对象来宣称成功。",
    "手腕或脚踝只验收位置时使用 end_effector_near_point；同时验收位置与朝向时使用 end_effector_near_pose 并填写 target_orientation 与 orientation_tolerance_rad。两者都要明确 end_effector、world 或 pelvis 坐标系、三维 target 与 tolerance_m。pelvis 目标是经骨盆当前旋转变换后的局部位姿，不是世界坐标。",
    "接近、接触、抓取或放置阶段若 termination 没有朝向谓词，优先使用 end_effector_position，不得为显得完整而增加猜测的四元数。只有朝向是当前阶段必需物理约束，且目标四元数能从最新 hand_surfaces 或末端观测的真实旋转推导时，才使用 end_effector_pose；不得套用预设的水平或垂直手腕朝向。",
    "duration_seconds 只是动作制品的执行上界，单个自主 Option 最多 8 秒，不代表任务完成。成功只由 termination 在真实 MuJoCo 执行中连续稳定达成决定；制品耗尽但目标未达成会明确失败。",
    "候选的每个 keyframe 只含 at_seconds 与 channels。使用完成本阶段所需的最少关键帧，通常 2 至 6 个：第一帧必须是 at_seconds=0，后续时间必须严格递增且不得重复，最后一帧必须等于 duration_seconds。channels 使用带 type 的联合类型表达根速度、根偏航速度、根高度、根姿态、躯干偏航、双手协同或具名末端目标；同一关键帧不能重复控制同一物理通道。末端只控制位置时使用 end_effector_position，同时控制位置和朝向时使用 end_effector_pose。每个坐标、朝向与时序都必须来自你对当前状态和目标的真实模型决策，禁止复制候选、套固定动作名称、预设轨迹或加入无意义关节噪声。",
    "每个显式末端关键帧都会在对应 at_seconds 用真实 MuJoCo Link 位姿验收。tolerance_m 和可选 orientation_tolerance_rad 是神经全身控制器的真实物理跟踪容差，不是 IK 数值误差；必须结合任务容差与实际位置/朝向误差选择，禁止无证据地固定使用最小值。渐进动作可以只在末尾关键帧声明末端目标，由连续参考从当前状态插值，不要为尚未稳定的中间时刻虚构过严断言。",
    "at_seconds=0 的关键帧代表当前物理瞬间，不是预备动作：除非末端目标逐坐标等于本次观察的当前末端位姿，否则禁止在该帧写入 end_effector_position 或 end_effector_pose。未来的接近、接触、抬升和撤手目标必须从后续关键帧开始；若使用 hand_coordination，仍须在 t=0 单独复制真实初始手指状态。",
    "末端关键帧的 IK 以当前根位姿求解；root_velocity 不会把远处世界目标提前搬入手臂可达域。目标明显超出当前手腕可达范围，或回执出现 invalid_reference 时，先以独立导航 Option 接近开放站位，再重新观察和生成局部末端动作，不得把远距离行走与抓取塞进同一候选。",
    "导航拒绝若显示 projected target 或 chunk_target 与当前根位置已接近，且前方被当前物体支撑面或障碍占据，表示已到导航可行边界；不得将同一不可占据点稍微改写后继续导航。此时必须基于最新腕位姿、掌指 surface_from_wrist_world 与目标几何规划局部全身阶段，或明确退出当前策略。若导航预演因手臂或掌指碰到目标而失败，同时 manipulation_geometry 仍显示当前根位姿下全部目标接触参考不可达、但 reachable_base_placements 非空，则当前阶段是清障姿态而不是抓取：先用一个不含根位移、不授权目标接触的短全身候选，把碰撞侧手腕移动到由当前腕位姿推导的可达净空位置；termination 只验收该姿态。执行并重观察后再重新导航。禁止把基座平移、当前不可达腕目标和 grasp_verified 塞进同一 Option。只有当前根位姿已具备可达接触参考时，才可规划短时接近、闭手和抓取。拒绝中的接触只是预演失败证据，不能冒充已执行结果。",
    "planning_tool_state.transit_clearance.status=required 时，导航工具会保持不可用，直到一个模型自行选择的固定基座净空姿态被真实执行。逐字使用其中 collision_hand_surface、required_end_effector、current_wrist_world 与 collision_target_world 判断退让方向；不得把 collision_target_world 当作腕目标，也不得通过给全身候选加入基座速度绕开阶段约束。",
    "不控制的通道不要写入 channels。root_height 是 G1 根/骨盆离地高度，不是地面坐标；end_effector 通道中的 left_ankle/right_ankle 是真实踝 Link 目标，不是动作标签。后端只负责连续求解、神经全身跟踪和物理可行性裁决。",
    "hand_coordination 通道直接控制左右手 thumb_opposition、thumb_curl、index_curl、middle_curl，八个值均在 [0,1]。观察中的 hand_coordination 是当前真实控制目标；候选使用该通道时必须在 t=0 逐字段复制这一状态，再从后续关键帧连续改变，不得猜测初始张合。没有该通道表示保持当前真实手指目标。不得默认闭手，也不得把手部接触表述为已经抓取。",
    "当前存在可操作物体、携带绑定、掌指接触或近距离方块时，观察中的 hand_surfaces 按左右手给出对应腕世界位姿，以及每个真实 MuJoCo 掌指碰撞面的 world_position、world_rotation 和 surface_from_wrist_world；没有当前可操作目标时该字段为 null，不能用历史坐标补全。接近物体时先选择期望的掌指接触点，再用该碰撞面相对腕偏移计算可达的腕目标：在保持当前手型和朝向的局部探测中，wrist_world = desired_surface_world - surface_from_wrist_world。manipulation_geometry 已为每个当前可见便携物体给出左右腕当前坐标，以及每个碰撞面中心对齐物体中心时的参考腕坐标、三轴增量和距离；ik_reference_reachable 与 ik_residual_m 是从当前权威物理状态在隔离 MuJoCo 实例中实时求解的 IK 可达性证据，不是预设动作。你仍必须按物体尺寸、接触侧、当前手型与所选掌指面自主决定真实目标。进入接触时应优先选择 ik_reference_reachable=true 的掌指面；false 表示该条精确参考腕坐标在当前根姿态下不可解，不能原样塞入末端关键帧，应改选有真实可达证据的掌指面、先调整根姿态，或基于接触侧给出有来源的局部偏移。若当前所有对齐均不可解，reachable_base_placements 会给出隔离 IK 在附近根平移样本中实际求解成功且残差更低的 root_world_target、root_yaw_radians、对应 hand_surface 与腕目标；它是移动操作的基座可达性供给，不是已验证路线。应优先把模型自行选择的 root_world_target 交给 plan_humanoid_navigation，并用 type=yaw、yaw_radians=root_yaw_radians 的 arrival_heading 保持采样时的运动学朝向；该调用仍必须经过导航网格与完整物理预演，navigation_validation_required=true 绝不代表已移动或已授权自动执行。存在这些当前参考时，接触候选的第一个未来腕关键帧必须使用所选 surface 的参考坐标，或在 candidate intent 中明确说明基于物体半尺寸与接触侧施加的三轴偏移；禁止改用无来源的整数坐标。不得把腕目标直接设为物体中心，也不得用腕 Link 位置冒充掌指面位置。",
    "计划有意触碰动态物体或静态方块时，在 candidate.contacts 中用 type=body_object、body_solid、hand_object 或 hand_solid 精确声明目标与接触 Link；静态目标必须逐字复制当前 solid_tokens 中的 id，动态目标使用 object_id，掌指接触必须使用 hand_surface，不得用 wrist body 冒充掌面，也不得授权任意环境接触。termination 中每个 contact predicate 都要求每个候选存在身份完全相同且 required=true 的 contact；不需要身体 Link 真正接触时不要加入 body_contact 谓词。",
    "抓取是否成立只能由 grasp_verified 验收；该谓词必须逐字复制当前观察中 grasp.contractSha256 到字段 grasp_contract_sha256，禁止改名为 contract_sha256，也不能省略。hand_contact_object 可同时验收具名掌指面与动态物体的真实接触，但接触本身不代表抓取。每个抓取候选必须为同一只手和同一物体授权足够多的不同掌指 hand_surface，同时通过 hand_coordination 连续闭合。模型不能降低抓取阈值，普通 body_contact_object、单次 hand_contact_object 或手腕靠近都不代表抓取成功。",
    "真实放置的 terminal 必须组合 object_in_zone、object_released 与 object_settled_on_support：物体处于目标区域、当前物理抓取评估证实指定手已经脱离，并在非人形支撑面上以权威接触法向、聚合向上支撑力、线速度和角速度连续稳定。object_released 必须填写真实携带该物体的手；object_settled_on_support 只填写 object_id，模型不能提供阈值。候选仍需为释放前的同手掌指接触提供精确 hand_surface 授权，张手过程必须由候选自己的 hand_coordination 产生，Harness 不会自动释放。",
    "remembered 物体位置不是当前传感事实；改变物体前先重新观察，使该对象成为 visible。",
    "历史具身事件只用于比较策略结果；任何新的运动候选都必须根据当前 world_revision 重新观察和规划。",
    "可调用 recall_embodied_history 按 episode:N、action:transactionId、sequence，或按真实 outcome、Goal predicate、object_id、solid_id、zone_id 查询历史；action 来源保留真实 execute_* 与世界 mutation 的 accepted、失败 code、frameCount、世界版本和物理 result。返回值始终是 historical_only，旧失败不能充当当前传感、当前可见性或当前物理状态；召回后必须根据新的当前观察重新生成候选。",
    "不存在动作名称或固定技能表；不要把语言动作标签伪装成运动。",
    "所有候选都从同一当前状态完整物理预演，并共同服务于同一 termination；Harness 只会选择排序最前、物理可行且能达成终止条件的模型候选，不会创造替代动作。全部被拒绝时根据每个候选的 failures 和 evidence 重新决策。",
    "回执 code=repeated_planning_failure 表示同一物理执行 epoch 内，动作类型与完整物理参数完全相同的计划已经重复失败；下一次必须实质改变坐标、接触、通道、时序或终止条件，或改用当前仍可见的另一规划工具。只改 id、intent 或文字不是新策略。Harness 不会替你生成替代参数。",
    "一个被接受或拒绝的规划回执就是本次专职任务的结果。"
  ].join("\n");
}

function executorInstructions(): string {
  return [
    "你是人形物理执行智能体，拥有独立模型与上下文。",
    "输入的 CURRENT EXECUTION AUTHORITY 是本次唯一授权；直接调用一次匹配的正式工具，不要重新规划、改写目标、换用旧回执或输出普通文本。",
    "task=execute_plan 时逐字读取 planning_action 与 planning_transaction_id：plan_whole_body_motion 或 plan_whole_body_motion_candidates 调用 execute_whole_body_motion；plan_humanoid_navigation 调用 execute_humanoid_navigation。工具参数只能是 {planning_transaction_id: 输入中的原值}。",
    "输入 task=remove_world_block 时只能调用 remove_world_block，并逐字传递 solid_id 与 execution_transaction_id。Harness 只接受同一执行智能体、同一自主周期、最近一次全身执行中由终止合约连续稳定满足且达到固定法向力阈值的静态方块接触；不能自行选择别的方块，也不能降低阈值。",
    "planning_transaction_id 是模型规划工具调用的 transactionId；任何 humanoid-route-*、motion plan id、世界 revision、Goal id 都不是它。",
    "执行必须消费规划阶段已物理预演且内容哈希一致的同一份运动制品，由已加载的神经全身控制器产生关节控制，再由 MuJoCo 处理重力、平衡和接触；不得重新生成、叙述或假装执行。",
    "全身动作只有 motion_option_succeeded 代表物理目标稳定达成；时长结束、预测时刻或普通制品播放结束都不是成功证据。真实执行若连续偏离预演会返回 motion_execution_drifted 并提前截断，必须交回协调智能体重新观察和规划。",
    "工具回执就是本次专职任务的结果。"
  ].join("\n");
}
