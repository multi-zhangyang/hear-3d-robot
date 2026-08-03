import {
  Agent,
  tool,
  type FunctionTool,
  type Model,
  type ModelSettings,
  type RunStreamEvent,
  type Session,
  type ToolUseBehavior
} from "@openai/agents";
import { z } from "zod";
import type { ProviderConfig } from "../../config/load.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import type { HumanoidActionRuntime } from "./runtime.js";
import { createHumanoidActionTools } from "./tools.js";
import { agentInvocationMarker } from "../agent-scope.js";

const SpecialistTaskSchema = z.object({
  objective: z.string().trim().min(1)
}).strict();
const ExecutionTaskSchema = z.object({
  objective: z.string().trim().min(1),
  planning_action: z.enum([
    "plan_whole_body_motion",
    "plan_humanoid_navigation"
  ]),
  planning_transaction_id: z.string().trim().min(1)
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

export const HUMANOID_AGENT_IDS = {
  coordinator: "humanoid-coordinator",
  sentry: "humanoid-sentry",
  motion: "humanoid-motion-reference",
  executor: "humanoid-executor"
} as const;

export const HUMANOID_CAPABILITIES = [
  "observe_humanoid",
  "plan_whole_body_motion",
  "execute_whole_body_motion",
  "plan_humanoid_navigation",
  "execute_humanoid_navigation"
] as const;

type HumanoidHierarchyRuntime = Pick<
  HumanoidActionRuntime,
  "invoke"
> & {
  validateCycleEvidence(evidenceTransactionIds: readonly string[]): HumanoidActionReceipt;
};

export interface HumanoidAgentHierarchy {
  coordinator: Agent;
  sentry: Agent;
  motion: Agent;
  executor: Agent;
  coordinatorSession: Session;
  session(agentId: string): Session | undefined;
}

export function createHumanoidAgentHierarchy(input: {
  createModel: (agentId: string) => Model;
  createSession: (agentId: string) => Session;
  provider: ProviderConfig;
  runtime: HumanoidHierarchyRuntime;
  onAgentStream?: (agentId: string, event: RunStreamEvent) => void | Promise<void>;
}): HumanoidAgentHierarchy {
  const models = new Set<Model>();
  const sessions = new Map<string, Session>();
  const sessionOwners = new Set<Session>();
  const ownModel = (agentId: string): Model => {
    const model = input.createModel(agentId);
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
  const modelSettings: ModelSettings = {
    temperature: input.provider.temperature,
    maxTokens: input.provider.maxOutputTokens,
    parallelToolCalls: false,
    toolChoice: "required"
  };

  const sentry = new Agent({
    name: "人形感知哨兵",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.sentry, sentryInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.sentry),
    modelSettings,
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
    modelSettings,
    tools: createHumanoidActionTools(
      input.runtime,
      HUMANOID_AGENT_IDS.motion,
      [
        "observe_humanoid",
        "plan_whole_body_motion",
        "plan_humanoid_navigation"
      ]
    ),
    resetToolChoice: false,
    toolUseBehavior: receiptToolUseBehavior([
      "plan_whole_body_motion",
      "plan_humanoid_navigation"
    ])
  });
  const executor = new Agent({
    name: "人形物理执行智能体",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.executor, executorInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.executor),
    modelSettings,
    tools: createHumanoidActionTools(
      input.runtime,
      HUMANOID_AGENT_IDS.executor,
      ["execute_whole_body_motion", "execute_humanoid_navigation"]
    ),
    resetToolChoice: false,
    toolUseBehavior: receiptToolUseBehavior([
      "execute_whole_body_motion",
      "execute_humanoid_navigation"
    ])
  });

  const sentrySession = ownSession(HUMANOID_AGENT_IDS.sentry);
  const motionSession = ownSession(HUMANOID_AGENT_IDS.motion);
  const executorSession = ownSession(HUMANOID_AGENT_IDS.executor);
  const coordinatorSession = ownSession(HUMANOID_AGENT_IDS.coordinator);
  const coordinator = new Agent({
    name: "人形自主协调智能体",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.coordinator, coordinatorInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.coordinator),
    modelSettings,
    tools: [
      sentry.asTool({
        toolName: "delegate_humanoid_sentry",
        toolDescription: "让独立感知智能体读取当前 MuJoCo 人形、接触、平衡、物体和导航状态。",
        parameters: SpecialistTaskSchema,
        inputBuilder: ({ params }) => params.objective,
        runOptions: { session: sentrySession },
        ...(input.onAgentStream
          ? { onStream: ({ event }) => input.onAgentStream!(HUMANOID_AGENT_IDS.sentry, event) }
          : {})
      }),
      motion.asTool({
        toolName: "delegate_motion_reference",
        toolDescription: "让独立运动参考智能体读取当前状态并提出一条经过完整物理预演的连续全身动作或双足路线。",
        parameters: SpecialistTaskSchema,
        inputBuilder: ({ params }) => params.objective,
        runOptions: { session: motionSession },
        ...(input.onAgentStream
          ? { onStream: ({ event }) => input.onAgentStream!(HUMANOID_AGENT_IDS.motion, event) }
          : {})
      }),
      executor.asTool({
        toolName: "delegate_physics_executor",
        toolDescription: "让独立执行智能体消费一个已接受规划回执，并用已加载的神经全身控制器与 MuJoCo 真实执行。",
        parameters: ExecutionTaskSchema,
        inputBuilder: ({ params }) => JSON.stringify(params),
        runOptions: { session: executorSession },
        ...(input.onAgentStream
          ? { onStream: ({ event }) => input.onAgentStream!(HUMANOID_AGENT_IDS.executor, event) }
          : {})
      }),
      cycleCompletionTool(input.runtime)
    ],
    resetToolChoice: false,
    toolUseBehavior: { stopAtToolNames: ["complete_autonomous_cycle"] }
  });

  return {
    coordinator,
    sentry,
    motion,
    executor,
    coordinatorSession,
    session: (agentId) => sessions.get(agentId)
  };
}

function scopedInstructions(agentId: string, instructions: string): string {
  return `${agentInvocationMarker(agentId)}\n${instructions}`;
}

function cycleCompletionTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof CycleCompletionSchema, string> {
  return tool<typeof CycleCompletionSchema, unknown, string>({
    name: "complete_autonomous_cycle",
    description: "完成一次真实自主循环。只有当前世界版本存在被引用的、已接受的人形物理执行回执时才会成功。",
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

function coordinatorInstructions(): string {
  return [
    "你是持续运行的人形层级智能体协调节点，每个专职节点都有独立模型和独立 Session。",
    "每次响应必须调用一个正式工具；你不能直接改变物理世界，也不能输出普通聊天作为执行结果。",
    "根据最新人形状态、环境、历史回执和模型采样自主选择有意义的下一步，不得从固定动作表、预设剧本或程序随机列表中选择。",
    "recent_physical_episodes 是带物理执行回执的历史记忆，可用于避免重复失败；它不是当前传感事实，禁止复用其中的 transaction_id、坐标或动作参数。",
    "需要当前事实时调用感知哨兵；需要动作时调用运动参考智能体，它必须返回物理预演回执。",
    "只有 accepted=true 的规划回执可以交给物理执行智能体，并且必须传递其原始 transactionId，不得猜测内部 plan_id。",
    "规划拒绝时应依据 failures 与 evidence 重新观察或提出不同的连续全身约束，不得让程序替换成默认动作。",
    "执行后重新观察可以帮助下一轮决策。只有引用当前世界版本的已接受执行回执，才可调用 complete_autonomous_cycle。",
    "人类可读摘要使用简洁中文；工具名、标识符和回执字段保持原样。"
  ].join("\n");
}

function sentryInstructions(): string {
  return [
    "你是人形层级智能体的感知哨兵，拥有独立模型与上下文。",
    "调用 observe_humanoid 返回当前 MuJoCo 本体状态、头部传感器可见物体和带来源版本的持久 3D 物体记录；不得补写、猜测或模拟视野外状态。",
    "工具回执就是本次专职任务的结果。"
  ].join("\n");
}

function motionInstructions(): string {
  return [
    "你是全身运动参考智能体，拥有独立模型与上下文。",
    "需要时先调用 observe_humanoid，再根据实时关节、Link、双脚接触、平衡、可见物体、带 age_revisions 的记忆和导航状态决定连续全身目标。",
    "使用 plan_whole_body_motion 输出根速度、躯干朝向和双手末端位置组成的任务空间关键帧，或使用 plan_humanoid_navigation 输出你选择的世界目标。不要生成、猜测或复制任何关节角；运动后端负责由任务空间目标求解连续全身参考。",
    "只填写本次确实要控制的通道；纯移动不要复制当前手腕位置或添加双手目标，省略的通道由当前全身参考连续保持。",
    "计划有意触碰物体时，必须用 contact_constraints 精确声明 body、object_id 与 required；不得授权某个 Link 接触任意环境。",
    "remembered 物体位置不是当前传感事实；改变物体前先重新观察，使该对象成为 visible。",
    "历史具身事件只用于比较策略结果；任何新的运动候选都必须根据当前 world_revision 重新观察和规划。",
    "不存在动作名称或固定技能表；不要把语言动作标签伪装成运动。",
    "规划必须由物理预演裁决。被拒绝时根据 failures 和 evidence 改变候选，不得要求 fallback 自动执行。",
    "一个被接受或拒绝的规划回执就是本次专职任务的结果。"
  ].join("\n");
}

function executorInstructions(): string {
  return [
    "你是人形物理执行智能体，拥有独立模型与上下文。",
    "输入包含 planning_action 与 planning_transaction_id。根据规划动作调用严格匹配的执行工具。",
    "只能传递 planning_transaction_id，不能猜测、复制或构造内部 plan_id。",
    "执行必须消费规划阶段已物理预演的同一份运动制品，由已加载的神经全身控制器产生关节控制，再由 MuJoCo 处理重力、平衡和接触；不得重新生成、叙述或假装执行。",
    "工具回执就是本次专职任务的结果。"
  ].join("\n");
}
