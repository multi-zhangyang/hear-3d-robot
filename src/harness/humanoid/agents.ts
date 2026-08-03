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
import {
  createHumanoidActionTools,
  createHumanoidEmbodiedRecallTool,
  type HumanoidEmbodiedRecallInvoker
} from "./tools.js";
import { agentInvocationMarker } from "../agent-scope.js";

const SpecialistTaskSchema = z.object({
  objective: z.string().trim().min(1)
}).strict();
const ExecutionTaskSchema = z.object({
  objective: z.string().trim().min(1),
  planning_action: z.enum([
    "plan_whole_body_motion",
    "plan_whole_body_motion_candidates",
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
  "recall_embodied_history",
  "plan_whole_body_motion",
  "plan_whole_body_motion_candidates",
  "execute_whole_body_motion",
  "plan_humanoid_navigation",
  "execute_humanoid_navigation"
] as const;

type HumanoidHierarchyRuntime = Pick<
  HumanoidActionRuntime,
  "invoke"
> & HumanoidEmbodiedRecallInvoker & {
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
      createHumanoidEmbodiedRecallTool(input.runtime),
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
        toolDescription: "让独立运动参考智能体读取当前状态，提出多个按偏好排序且分别经过完整物理预演的连续全身动作候选，或规划双足路线。",
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
    description: "完成一次真实自主循环。全身动作必须由物理 Option 稳定达成，导航必须真实完成，且回执属于当前世界版本。",
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
    "需要更早或指定来源的事件时可调用 recall_embodied_history。episode:N 指向已完成具身循环，action:transactionId 指向 actions.jsonl 中真实的 execute_* 回执，包括拒绝、失败和停滞；它们全是 historical_only，绝不能据此声称当前可见、当前接触或当前坐标，任何当前事实必须重新委派感知哨兵观察。",
    "需要当前事实时调用感知哨兵；需要动作时调用运动参考智能体，它必须返回物理预演回执。",
    "只有 accepted=true 的规划回执可以交给物理执行智能体，并且必须传递其原始 transactionId，不得猜测内部 plan_id；多候选回执还会明确被物理筛选选中的模型候选。",
    "一旦收到当前 world_revision 的 accepted 规划回执，下一步必须直接委派物理执行智能体；在执行回执返回前，不要重复规划、召回历史或重新感知。",
    "规划拒绝时应依据 failures 与 evidence 重新观察或提出不同的连续全身约束，不得让程序替换成默认动作。",
    "全身运动只有 motion_option_succeeded 才表示物理目标达成；motion_goal_unmet、motion_goal_uncertain、motion_constraint_violated、motion_execution_drifted、motion_failed 都必须重新观察和规划。导航只有 navigation_completed 才可验收。",
    "执行后让感知哨兵重新观察，再决定下一轮。只有引用当前世界版本的物理成功回执，才可调用 complete_autonomous_cycle。",
    "人类可读摘要使用简洁中文；工具名、标识符和回执字段保持原样。"
  ].join("\n");
}

function sentryInstructions(): string {
  return [
    "你是人形层级智能体的感知哨兵，拥有独立模型与上下文。",
    "调用 observe_humanoid 返回当前 MuJoCo 本体状态、头部传感器可见物体和带来源版本的持久 3D 物体记录；不得补写、猜测或模拟视野外状态。物理 Option 成功、失败或不可确定后都应以新的观察作为下一次规划依据。",
    "工具回执就是本次专职任务的结果。"
  ].join("\n");
}

function motionInstructions(): string {
  return [
    "你是全身运动参考智能体，拥有独立模型与上下文。",
    "需要时先调用 observe_humanoid，再根据实时关节、Link、双脚接触、平衡、可见物体、带 age_revisions 的记忆和导航状态决定连续全身目标。",
    "非导航动作必须使用 plan_whole_body_motion_candidates，一次提交共同 termination 和 2 至 3 个真正不同、按你偏好排序的连续全身候选；使用 plan_humanoid_navigation 输出你选择的世界目标。不要生成、猜测或复制任何关节角；运动后端负责由任务空间目标求解连续全身参考。",
    "termination 必须描述本轮可由当前传感状态验证的物理结果，例如根节点、具名手腕/脚踝末端或身体 Link 到达位置、身体与可见物体接触、可见物体到达位置或区域。简单目标可使用全部谓词隐式 AND；复杂目标可用受限 all/any/not 条件树和 precondition/during/terminal 阶段。stable_steps 用于排除瞬时碰撞；不得引用当前不可见对象来宣称成功。",
    "手腕或脚踝目标优先使用 end_effector_near_point，并明确 end_effector、world 或 pelvis 坐标系、三维 target 与 tolerance_m；pelvis 目标是经骨盆当前旋转变换后的局部坐标，不是世界坐标。每个 termination 谓词都填写完整字段集合：当前类型不使用的 body、end_effector、frame、object_id、zone_id、target、tolerance_m、minimum_normal_force、expected 必须为 null。阶段条件用 predicate_indexes 表示正向谓词、not_predicate_indexes 表示否定谓词，再由 op=all 或 any 组合；不需要分阶段时 phases 填 null。",
    "duration_seconds 只是动作制品的执行上界，单个自主 Option 最多 8 秒，不代表任务完成。成功只由 termination 在真实 MuJoCo 执行中连续稳定达成决定；制品耗尽但目标未达成会明确失败。",
    "候选通过根速度、躯干朝向、左右手腕和左右踝 Link 的连续三维末端目标组成任务空间关键帧。每个坐标与时序都必须来自你对当前状态和目标的真实模型决策，禁止复制候选、套固定动作名称、预设轨迹或加入无意义关节噪声。",
    "每个显式末端关键帧都会在对应 at_seconds 用真实 MuJoCo Link 位置验收。tolerance_m 是神经全身控制器的物理跟踪容差，不是 IK 数值误差；必须结合任务容差和 task_space_target_unmet 的实际 errorMeters 选择，禁止无证据地固定使用最小值。渐进动作可以只在末尾关键帧声明末端目标，由连续参考从当前状态插值，不要为尚未稳定的中间时刻虚构过严位置断言。",
    "keyframes 中未控制的 root_velocity、root_yaw_velocity、root_height、root_roll、root_pitch、torso_yaw、left_hand、right_hand、left_foot、right_foot 必须填 null，不能用 0 或空对象占位。root_height 是 G1 根/骨盆离地高度，不是地面目标坐标的 y；不主动控制高度时填 null。",
    "脚部字段直接表示对应踝 Link 在 world 或 pelvis 坐标系中的目标位置，不代表动作标签；后端只负责腿部运动学求解、神经全身跟踪和物理可行性裁决。",
    "只填写本次确实要控制的通道；不控制某个手腕或脚踝时不要复制它的当前位置，省略的通道由当前全身参考连续保持。",
    "计划有意触碰物体时，必须用 contact_constraints 精确声明 body、object_id 与 required；不得授权某个 Link 接触任意环境。",
    "remembered 物体位置不是当前传感事实；改变物体前先重新观察，使该对象成为 visible。",
    "历史具身事件只用于比较策略结果；任何新的运动候选都必须根据当前 world_revision 重新观察和规划。",
    "可调用 recall_embodied_history 按 episode:N、action:transactionId 或 sequence 查询历史；action 来源保留真实 execute_* 的 accepted、失败 code、frameCount、世界版本和物理 result。返回值始终是 historical_only，旧失败不能充当当前传感、当前可见性或当前物理状态；召回后必须根据新的当前观察重新生成候选。",
    "不存在动作名称或固定技能表；不要把语言动作标签伪装成运动。",
    "所有候选都从同一当前状态完整物理预演，并共同服务于同一 termination；Harness 只会选择排序最前、物理可行且能达成终止条件的模型候选，不会创造替代动作。全部被拒绝时根据每个候选的 failures 和 evidence 重新决策。",
    "一个被接受或拒绝的规划回执就是本次专职任务的结果。"
  ].join("\n");
}

function executorInstructions(): string {
  return [
    "你是人形物理执行智能体，拥有独立模型与上下文。",
    "输入包含 planning_action 与 planning_transaction_id。单候选或多候选全身规划都调用 execute_whole_body_motion；双足路线调用 execute_humanoid_navigation。",
    "只能传递 planning_transaction_id，不能猜测、复制或构造内部 plan_id。",
    "执行必须消费规划阶段已物理预演且内容哈希一致的同一份运动制品，由已加载的神经全身控制器产生关节控制，再由 MuJoCo 处理重力、平衡和接触；不得重新生成、叙述或假装执行。",
    "全身动作只有 motion_option_succeeded 代表物理目标稳定达成；时长结束、预测时刻或普通制品播放结束都不是成功证据。真实执行若连续偏离预演会返回 motion_execution_drifted 并提前截断，必须交回协调智能体重新观察和规划。",
    "工具回执就是本次专职任务的结果。"
  ].join("\n");
}
