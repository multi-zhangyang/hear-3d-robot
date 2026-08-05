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
import { preflightAgentToolInput } from "../tool-input-recovery.js";
import type { HumanoidCycleCompletionReadiness } from "./cycle-causal-evidence.js";
import type { HumanoidCoordinatorPhase } from "./run-runtime.js";
import { GOAL_HISTORY_PREDICATE_TYPES } from "./goal-history.js";

const SpecialistTaskSchema = z.object({
  objective: z.string().trim().min(1)
}).strict();
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

function objectiveTextInputBuilder(
  { params }: { params: z.infer<typeof SpecialistTaskSchema> }
): string {
  return params.objective;
}

export function goalManagerInvocationInput(
  objective: string,
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
      status: candidate.status ?? null
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
  const exact = {
    run_mode: root.run_mode ?? null,
    mission_goal: root.mission_goal ?? null,
    goal_dag_status: goalDAG.status ?? null,
    current_goal_evidence_ref: goalContext.evidence_ref ?? null,
    existing_goal_candidate_ids: Object.keys(candidates).sort(),
    existing_goal_candidates: candidateReferences,
    current_goal_epoch_id: goalDAG.current_epoch_id ?? null,
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
    objective,
    "CURRENT GOAL MANAGER INVOCATION",
    "以下状态与标识来自本次调用的权威物理状态。候选提交和选择会由 Harness 绑定本次证据；工具中仍需填写的标识必须逐字复制。",
    JSON.stringify(exact)
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
    inputBuilderContract: "goal_manager_authority_envelope_v1",
    inputBuilder: goalManagerInvocationInput,
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
    inputBuilderContract: "objective_text_v1",
    inputBuilder: objectiveTextInputBuilder,
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
    inputBuilderContract: "objective_text_v1",
    inputBuilder: objectiveTextInputBuilder,
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
        parameters: SpecialistTaskSchema,
        inputBuilder: ({ params }) => goalManagerInvocationInput(
          params.objective,
          input.runtime.contextAnchor(HUMANOID_AGENT_IDS.goalManager)
        ),
        includeInputSchema: goalManagerContract.includeInputSchema,
        needsApproval: goalManagerContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: { session: agentToolSession(goalManagerContract) },
        resumeState: { contextStrategy: goalManagerContract.resumeContextStrategy },
        ...(input.onAgentStream
          ? {
              onStream: ({ event }) => input.onAgentStream!(
                HUMANOID_AGENT_IDS.goalManager,
                event
              )
            }
          : {})
        })
      ), SpecialistTaskSchema),
      recoverAgentToolInput(scopeAgentToolInvocation(
        HUMANOID_AGENT_IDS.sentry,
        sentry.asTool({
        toolName: sentryContract.toolName,
        toolDescription: "让独立感知智能体读取当前 MuJoCo 人形、接触、平衡、物体和导航状态。",
        parameters: SpecialistTaskSchema,
        inputBuilder: sentryContract.inputBuilder,
        includeInputSchema: sentryContract.includeInputSchema,
        needsApproval: sentryContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: { session: agentToolSession(sentryContract) },
        resumeState: { contextStrategy: sentryContract.resumeContextStrategy },
        ...(input.onAgentStream
          ? { onStream: ({ event }) => input.onAgentStream!(HUMANOID_AGENT_IDS.sentry, event) }
          : {})
        })
      ), SpecialistTaskSchema),
      recoverAgentToolInput(scopeAgentToolInvocation(
        HUMANOID_AGENT_IDS.motion,
        motion.asTool({
        toolName: motionContract.toolName,
        toolDescription: "让独立运动参考智能体读取当前状态，提出多个按偏好排序且分别经过完整物理预演的连续全身动作候选，或规划双足路线。",
        parameters: SpecialistTaskSchema,
        inputBuilder: motionContract.inputBuilder,
        includeInputSchema: motionContract.includeInputSchema,
        needsApproval: motionContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: { session: agentToolSession(motionContract) },
        resumeState: { contextStrategy: motionContract.resumeContextStrategy },
        ...(input.onAgentStream
          ? { onStream: ({ event }) => input.onAgentStream!(HUMANOID_AGENT_IDS.motion, event) }
          : {})
        })
      ), SpecialistTaskSchema),
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
  configureCoordinatorToolAvailability(coordinator, input.runtime);

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

function configureCoordinatorToolAvailability(
  coordinator: Agent,
  runtime: HumanoidHierarchyRuntime
): void {
  for (const coordinatorTool of coordinator.tools) {
    if (coordinatorTool.type !== "function") continue;
    const name = coordinatorTool.name;
    coordinatorTool.isEnabled = async () => {
      const completion = runtime.cycleCompletionReadiness();
      const phase = runtime.coordinatorPhase();
      if (name === "complete_autonomous_cycle") {
        return completion.status === "ready";
      }
      if (name === "complete_goal_transition") return phase === "goal_selection";
      if (name === "delegate_goal_manager") {
        return phase === "goal_selection" || phase === "replan_or_retire";
      }
      if (name === "delegate_humanoid_sentry") {
        return phase === "observe_or_plan" || phase === "post_execution";
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
    };
  }
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
  const invoke = agentTool.invoke;
  agentTool.invoke = async (context, input, details) => {
    const rejection = preflightAgentToolInput(input, schema, agentTool.name);
    if (rejection !== undefined) return rejection;
    return invoke(context, input, details);
  };
  return agentTool;
}

function coordinatorInstructions(): string {
  return [
    "你是持续运行的人形层级智能体协调节点，每个专职节点都有独立模型和独立 Session。",
    "每次响应必须调用一个正式工具；你不能直接改变物理世界，也不能输出普通聊天作为执行结果。",
    "goal_dag.status=awaiting_model_selection 时，第一步必须调用 delegate_goal_manager，让独立目标管理模型提交 2–3 个候选并显式选择；委派 objective 只说明长期任务结果，不得替它建议候选、坐标或依赖；Harness 只绑定本次物理证据，不得替它创建、打分或选择目标。",
    "active Goal 的唯一来源是 goal_dag 当前 epoch 对应 candidate；mission 与 mission_goal 只是长期约束，不能被当成 active Goal 或自动候选。",
    "若真实物理拒绝和当前观察证明 active Goal 不可继续，可委派目标管理智能体将其退役为 blocked、abandoned、superseded 或 expired；退役后调用 complete_goal_transition 结束本轮，下一轮仍必须由目标管理模型重新提出并选择。",
    "根据最新人形状态、环境、历史回执和模型采样自主选择有意义的下一步，不得从固定动作表、预设剧本或程序随机列表中选择。",
    "recent_physical_episodes 是带物理执行回执的历史记忆，可用于避免重复失败；它不是当前传感事实，禁止复用其中的 transaction_id、坐标或动作参数。",
    "需要更早或指定来源的事件时可调用 recall_embodied_history。它支持 episode:N、action:transactionId 精确召回，也支持按真实 outcome、Goal predicate、object_id、solid_id 和 zone_id 检索持久经验；结果全是 historical_only，绝不能据此声称当前可见、当前接触或当前坐标，任何当前事实必须重新委派感知哨兵观察。",
    "需要当前事实时调用感知哨兵；需要动作时调用运动参考智能体，它必须返回物理预演回执。",
    "只有 accepted=true 的规划回执可以交给物理执行智能体，并且必须传递其原始 transactionId，不得猜测内部 plan_id；多候选回执还会明确被物理筛选选中的模型候选。",
    "coordinator_phase=execute_plan 时，CURRENT HARNESS AUTHORITY.execution_authority 是唯一待执行授权；plan_humanoid_navigation 的 accepted 回执本身就是可执行规划，不需要也不得另找全身规划。委派时逐字复制其中 planning_action 与 planning_transaction_id，其他历史 frame、plan_id 或被拒绝回执一律不能替代。",
    "一旦收到 accepted 且租约仍有效的规划回执，下一步必须以 task=execute_plan 直接委派物理执行智能体；Harness 会在最新权威状态上重验证同一模型意图。在执行回执返回前，不要重复规划、召回历史或重新感知。",
    "若模型选择的本轮意图要求拆除静态方块，必须先让全身候选以同一 solid_id 的 body_contact_solid 或 hand_contact_solid 作为必需终止条件并真实执行；只有 motion_option_succeeded 后，才再次委派同一执行智能体调用 remove_world_block，逐字传递该 solid_id 与执行 transactionId。拆除拒绝时不得宣称世界已改变。",
    "规划拒绝时应依据 failures 与 evidence 重新观察或提出不同的连续全身约束，不得让程序替换成默认动作。",
    "全身运动只有 motion_option_succeeded 才表示物理目标达成；motion_goal_unmet、motion_goal_uncertain、motion_constraint_violated、motion_execution_drifted、motion_failed 都必须重新观察和规划。导航只有 navigation_completed 才可验收。",
    "执行后让感知哨兵重新观察，再决定下一轮。只有引用最近一次未被后续物理执行取代的成功回执，才可调用 complete_autonomous_cycle；若调用 remove_world_block 成功，evidence_transaction_ids 必须同时包含原物理执行与拆除回执。正常静止物理帧不会伪造或取代动作证据。",
    "CURRENT HARNESS AUTHORITY 的 cycle_completion.status=ready 时，evidence_transaction_ids 是本轮唯一可提交的真实因果证据；感知完成后必须立即调用 complete_autonomous_cycle，不能在本轮继续规划。Harness 只暂停冲突工具权限，不会替你调用完成工具。",
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
    "goal_dag.status=awaiting_model_selection 时，先调用 submit_goal_candidates 一次提交 2–3 个内容不同的候选；Harness 会把提交与随后选择分别绑定到对应模型请求所见的当前物理证据，不要求你转录证据哈希。候选间不能引用本批 proposal_id 或尚未生成的 candidate_id；dependency_candidate_ids 只能逐字引用 existing_goal_candidate_ids，列表为空时每个候选都必须填写 []。每个谓词必须能由当前证据和后续 checker 真实观察。goal_context.observation 提供当前可见物体、视觉或接触可观察静态方块的真实位姿、关系、接触和区域；block_removed 只能逐字引用当前 solids 中 kind=block 的 id。goal_context.autonomy 提供有界历史计数与能力面，它不包含 Harness 评分或候选。",
    "候选必须根据当前可供性和长期约束产生，并主动比较近期 Goal 的对象、区域、谓词组合和结果；context_projection.history_truncated=true 或需要核对更早结果时，调用 recall_goal_history 检索完整持久 Goal DAG。召回只用于历史比较，不能代替当前观察。除非恢复、依赖或当前物理状态确有必要，不要重复相同目标内容。自主差异来自你的模型选择，不得用随机坐标、随机电机动作或固定目标表冒充新颖性。",
    "提交成功后必须在新的模型响应中调用 select_goal_candidate，逐字复制回执中所选候选的 candidate_sequence；该短序号与持久哈希身份一对一对应，不能猜测。必须显式选择一个依赖已完成的候选；不能让程序随机选择，也不能让 Harness 替你插入固定候选。",
    "涉及便携物体时必须按当前物理依赖判断 Goal：未持握物体就不能把远离该物体、仅进入目标区域当成放置任务的充分前置；observable_goal_surface.carrying 只有在列出该物体且 continuation_verified=true 时才证明它可随导航继续携带。可见且可操作的目标物体、其双腕距离、当前抓取评估与目标区域共同决定是直接选择完整 mission_goal，还是先选择 object_grasped 等可验收前置 Goal；不得用固定抓取顺序替代这一判断。",
    "goal_dag.status=active 时不得提交或选择新目标。只有当前观察与物理 action receipt 支持时，才能调用 retire_goal_epoch；blocked 必须引用当前 revision 的 action receipt。退役不会自动选择替代目标。",
    "retire_goal_epoch 的 evidence_refs、select_goal_candidate 的 candidate_sequence 与所有 dependency_candidate_ids 必须逐字使用上下文或工具回执中真实存在的标识，不得猜测。",
    "最终结果必须来自 select_goal_candidate 或 retire_goal_epoch 工具。"
  ].join("\n");
}

function sentryInstructions(): string {
  return [
    "你是人形层级智能体的感知哨兵，拥有独立模型与上下文。",
    "调用 observe_humanoid 返回当前 MuJoCo 本体状态、头部传感器可见物体、带来源版本的持久 3D 物体记录，以及当前帧掌指接触、支撑脱离和抓取稳定证据；不得补写、猜测或模拟视野外状态。物理 Option 成功、失败或不可确定后都应以新的观察作为下一次规划依据。",
    "工具回执就是本次专职任务的结果。"
  ].join("\n");
}

function motionInstructions(): string {
  return [
    "你是全身运动参考智能体，拥有独立模型与上下文。",
    "每次模型响应直接调用一个正式工具，不复述任务、工具 schema 或推理过程。一次委派只规划当前状态下可以真实执行的下一阶段，不试图在一个动作制品中完成整条任务链。",
    "需要时先调用 observe_humanoid，再根据实时关节、Link、双脚接触、平衡、可见物体、带 age_revisions 的记忆和导航状态决定连续全身目标。interaction 只提供当前权威的目标区域、携带生命周期、抓取阈值以及物体相对骨盆和双腕的几何关系，不包含推荐动作；你必须自行选择阶段、手、坐标、时序与候选顺序。",
    "非导航动作必须使用 plan_whole_body_motion_candidates，一次提交共同 termination 和 2 至 3 个真正不同、按你偏好排序的连续全身候选；使用 plan_humanoid_navigation 输出你选择的世界目标。不要生成、猜测或复制任何关节角；运动后端负责由任务空间目标求解连续全身参考。",
    "若委派目标包含接近、接触、操作、离开或继续导航等多个阶段，或目标明显超出单个 8 秒 Option 的可达范围，先自主选择当前阶段的可达目标；需要长距离接近时调用 plan_humanoid_navigation，进入真实可操作范围后再由后续周期规划接触或操作。",
    "纯导航阶段不要混入手腕、脚踝或抓取目标；直接调用 plan_humanoid_navigation。导航 target.y 使用当前地面或规划回执给出的地面高度，不能复制机器人根高度或障碍物中心高度；若回执给出 partial_endpoint、projected target 或 chunk_target，只能基于这些真实可行性证据重新选择下一段地面目标。",
    "termination 必须描述本轮可由当前传感状态验证的物理结果，例如根节点、具名手腕/脚踝末端到达位置或可选朝向、身体 Link 到达位置、接触、物体到达位置、区域或 grasp_verified。简单目标可使用全部谓词隐式 AND；复杂目标可用受限 all/any/not 条件树和 precondition/during/terminal 阶段。stable_steps 用于排除瞬时碰撞；不得引用当前不可见对象来宣称成功。",
    "手腕或脚踝位姿目标优先使用 end_effector_near_point，并明确 end_effector、world 或 pelvis 坐标系、三维 target 与 tolerance_m；需要朝向验收时同时填写 target_orientation 四元数和 orientation_tolerance_rad，不需要时省略这两个可选字段。pelvis 目标是经骨盆当前旋转变换后的局部位姿，不是世界坐标。每个 termination 谓词只填写其严格 schema 定义的字段，不得补充其他类型字段或 null 占位。阶段条件使用递归 condition 中的 predicate_index 组合 all、any、not；不需要分阶段时 phases 填 null。",
    "duration_seconds 只是动作制品的执行上界，单个自主 Option 最多 8 秒，不代表任务完成。成功只由 termination 在真实 MuJoCo 执行中连续稳定达成决定；制品耗尽但目标未达成会明确失败。",
    "候选通过根速度、躯干朝向、左右手腕和左右踝 Link 的连续末端位姿目标组成任务空间关键帧；末端 orientation 与 orientation_tolerance_rad 是成对出现的可选朝向约束。每个坐标、朝向与时序都必须来自你对当前状态和目标的真实模型决策，禁止复制候选、套固定动作名称、预设轨迹或加入无意义关节噪声。",
    "每个显式末端关键帧都会在对应 at_seconds 用真实 MuJoCo Link 位姿验收。tolerance_m 和可选 orientation_tolerance_rad 是神经全身控制器的真实物理跟踪容差，不是 IK 数值误差；必须结合任务容差与实际位置/朝向误差选择，禁止无证据地固定使用最小值。渐进动作可以只在末尾关键帧声明末端目标，由连续参考从当前状态插值，不要为尚未稳定的中间时刻虚构过严断言。",
    "keyframes 中未控制的 root_velocity、root_yaw_velocity、root_height、root_roll、root_pitch、torso_yaw、hand_coordination、left_hand、right_hand、left_foot、right_foot 必须填 null，不能用 0 或空对象占位。root_height 是 G1 根/骨盆离地高度，不是地面目标坐标的 y；不主动控制高度时填 null。",
    "hand_coordination 直接控制左右手 thumb_opposition、thumb_curl、index_curl、middle_curl，八个值均在 [0,1]。使用手部协同时必须在 t=0 显式给出初始状态；null 表示保持当前真实手指目标。不得默认闭手，也不得把手部接触表述为已经抓取。",
    "脚部字段直接表示对应踝 Link 在 world 或 pelvis 坐标系中的目标位姿，不代表动作标签；后端只负责腿部运动学求解、神经全身跟踪和物理可行性裁决。",
    "只填写本次确实要控制的通道；不控制某个手腕或脚踝时不要复制它的当前位置，省略的通道由当前全身参考连续保持。",
    "计划有意触碰动态物体或静态方块时，必须用 contact_constraints 精确声明 object_id 或 solid_id 以及接触面：普通身体 Link 使用 body，掌面或指面使用 hand_surface；静态目标必须逐字复制当前 solid_tokens 中的 id，不得用 wrist body 冒充掌面，也不得授权任意环境接触。",
    "抓取只能用 grasp_verified 作为物理终止谓词，并逐字复制当前观察中 grasp.contractSha256；每个候选必须为同一只手和同一物体授权足够多的不同掌指 hand_surface，同时通过 hand_coordination 连续闭合。模型不能降低抓取阈值，普通 body_contact_object、接触一次或手腕靠近都不代表抓取成功。",
    "真实放置的 terminal 必须组合 object_in_zone、object_released 与 object_settled_on_support：物体处于目标区域、当前物理抓取评估证实指定手已经脱离，并在非人形支撑面上以权威接触法向、聚合向上支撑力、线速度和角速度连续稳定。object_released 必须填写真实携带该物体的手；object_settled_on_support 只填写 object_id，模型不能提供阈值。候选仍需为释放前的同手掌指接触提供精确 hand_surface 授权，张手过程必须由候选自己的 hand_coordination 产生，Harness 不会自动释放。",
    "remembered 物体位置不是当前传感事实；改变物体前先重新观察，使该对象成为 visible。",
    "历史具身事件只用于比较策略结果；任何新的运动候选都必须根据当前 world_revision 重新观察和规划。",
    "可调用 recall_embodied_history 按 episode:N、action:transactionId、sequence，或按真实 outcome、Goal predicate、object_id、solid_id、zone_id 查询历史；action 来源保留真实 execute_* 与世界 mutation 的 accepted、失败 code、frameCount、世界版本和物理 result。返回值始终是 historical_only，旧失败不能充当当前传感、当前可见性或当前物理状态；召回后必须根据新的当前观察重新生成候选。",
    "不存在动作名称或固定技能表；不要把语言动作标签伪装成运动。",
    "所有候选都从同一当前状态完整物理预演，并共同服务于同一 termination；Harness 只会选择排序最前、物理可行且能达成终止条件的模型候选，不会创造替代动作。全部被拒绝时根据每个候选的 failures 和 evidence 重新决策。",
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
