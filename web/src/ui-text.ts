import type {
  Goal,
  GoalPredicate,
  HumanoidBodyChannel,
  HumanoidRunCheckpoint,
  RunListItem,
  StreamState,
  TaskNode
} from "./types";

const RUN_STATUS: Record<RunListItem["status"], string> = {
  starting: "启动中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  interrupted: "已暂停",
  local_artifact: "本地记录"
};

const NODE_STATUS: Record<TaskNode["status"], string> = {
  ready: "就绪",
  active: "执行中",
  waiting: "等待中",
  completed: "已完成",
  blocked: "已阻塞",
  failed: "失败"
};

const STREAM_STATUS: Record<StreamState, string> = {
  inactive: "未连接",
  connecting: "连接中",
  connected: "已同步",
  disconnected: "已断开"
};

const BODY_CHANNEL: Record<HumanoidBodyChannel, string> = {
  locomotion: "双足运动",
  torso: "躯干",
  left_arm: "左臂",
  right_arm: "右臂"
};

const ACTION_LABELS: Record<string, string> = {
  observe_humanoid: "感知人形世界",
  plan_whole_body_motion: "规划全身动作",
  execute_whole_body_motion: "执行全身动作",
  plan_humanoid_navigation: "规划双足路线",
  execute_humanoid_navigation: "执行双足导航",
  delegate_humanoid_sentry: "委派感知哨兵",
  delegate_motion_reference: "委派运动参考智能体",
  delegate_physics_executor: "委派物理执行智能体",
  complete_autonomous_cycle: "完成自主循环",
  tool: "工具"
};

const RESULT_CODES: Record<string, string> = {
  accepted: "已接受",
  humanoid_observed: "人形世界状态已感知",
  whole_body_plan_validated: "全身动作已通过物理预演",
  whole_body_plan_rejected: "全身动作未通过物理预演",
  humanoid_route_validated: "双足路线已通过物理预演",
  humanoid_route_rejected: "没有可行的双足路线",
  motion_completed: "全身动作已完成",
  motion_failed: "全身动作执行失败",
  navigation_completed: "双足导航分块已完成",
  navigation_blocked: "双足导航受阻",
  plan_stale: "动作计划已失效",
  planning_receipt_missing: "缺少规划回执",
  planning_receipt_action_mismatch: "规划回执与执行动作不匹配",
  planning_receipt_rejected: "规划回执未被物理预演接受",
  planning_receipt_missing_plan: "规划回执缺少动作计划",
  invalid_tool_input: "动作输入无效",
  invalid_reference: "全身运动参考无效",
  environment_contact: "动作产生未授权环境接触",
  fallen: "机器人在物理预演中失去平衡",
  required_contact_missing: "动作缺少要求的物理接触",
  unknown_contact_object: "接触目标不存在",
  contact_object_not_currently_visible: "接触目标当前不可见",
  unsupported_finish: "动作结束状态不受支持"
};

const SCENARIO_LABELS: Record<string, string> = {
  humanoid_frontier: "人形方块边境",
  humanoid_realm: "人形方块疆域",
  humanoid_courtyard: "人形庭院"
};

const AGENT_NAMES: Record<string, string> = {
  "humanoid coordinator": "自主协调智能体",
  "humanoid sentry": "人形感知哨兵",
  "humanoid motion reference": "全身运动参考智能体",
  "humanoid executor": "人形物理执行智能体"
};

const CHINESE_AGENT_NAMES: Record<string, string> = {
  "人形自主协调智能体": "自主协调智能体"
};

const ENTITY_NAMES: Record<string, string> = {
  amber_crate: "琥珀木箱",
  blue_crate: "蓝色木箱",
  copper_crate: "铜色木箱",
  moss_crate: "苔绿色木箱",
  violet_crate: "紫色木箱",
  courtyard_crate: "庭院木箱",
  frontier_beacon: "边境信标区",
  distant_beacon: "远方信标区",
  rest_clearing: "休整区",
  courtyard_beacon: "庭院信标区",
  stone_column: "石柱",
  low_block: "低矮障碍"
};

export function runStatusLabel(status: RunListItem["status"]): string {
  return RUN_STATUS[status];
}

export function nodeStatusLabel(status: TaskNode["status"]): string {
  return NODE_STATUS[status];
}

export function streamStatusLabel(status: StreamState): string {
  return STREAM_STATUS[status];
}

export function bodyChannelLabel(channel: HumanoidBodyChannel): string {
  return BODY_CHANNEL[channel];
}

export function motionGeneratorLabel(implementation: string): string {
  const normalized = implementation.toLowerCase();
  if (normalized === "task_space_constraints") return "任务约束";
  if (normalized.includes("ardy")) return "ARDY";
  if (normalized.includes("motionbricks")) return "MotionBricks";
  if (normalized.includes("omg")) return "OMG";
  return "运动生成器";
}

export function humanoidControllerLabel(implementation: string): string {
  const normalized = implementation.toLowerCase();
  if (normalized === "yahmp_onnx") return "YAHMP";
  if (normalized.includes("sonic")) return "SONIC";
  return "全身控制器";
}

export function actionLabel(value: string): string {
  return ACTION_LABELS[value] ?? "未识别动作";
}

export function resultCodeLabel(value: string): string {
  return RESULT_CODES[value] ?? "未识别的运行回执";
}

export function scenarioLabel(id: string | null, fallback?: string): string {
  if (id && SCENARIO_LABELS[id]) return SCENARIO_LABELS[id];
  return fallback?.trim() || id || "未知场景";
}

export function agentNameLabel(name: string): string {
  if (/[㐀-鿿]/u.test(name)) return CHINESE_AGENT_NAMES[name] ?? name;
  const normalized = name.trim().toLowerCase().replaceAll("_", " ").replaceAll("-", " ")
    .replace(/\s+/gu, " ");
  const exact = AGENT_NAMES[normalized];
  if (exact) return exact;
  for (const [base, label] of Object.entries(AGENT_NAMES)) {
    const match = normalized.match(new RegExp(`^${escapeRegExp(base)} (\\d+)$`, "u"));
    if (match) return `${label} ${match[1]}`;
  }
  const suffix = normalized.match(/(?:^| )(\d+)$/u)?.[1];
  return suffix ? `专项智能体 ${suffix}` : "专项智能体";
}

export function entityLabel(id: string): string {
  if (ENTITY_NAMES[id]) return ENTITY_NAMES[id];
  return /[㐀-鿿]/u.test(id) ? id : "场景实体";
}

export function nodePurposeLabel(node: TaskNode, goal: Goal): string {
  if (node.depth === 0) return goalSummaryLabel(goal);
  if (node.may_delegate) return "协调下级智能体 · 汇总物理回执";
  const capabilities = [...new Set(node.capabilities.map(actionLabel))].slice(0, 3);
  return capabilities.length > 0 ? capabilities.join(" · ") : "执行当前分配";
}

export function nodeResultLabel(node: TaskNode): string | null {
  if (node.last_result === undefined) return null;
  const output = modelOutputLabel(node.last_result);
  if (output) return output;
  if (node.status === "completed") return "当前分配已完成。";
  if (node.status === "blocked") return "当前分配已被阻塞，等待上级重新规划。";
  if (node.status === "failed") return "当前分配执行失败，等待上级处理。";
  return null;
}

const MODEL_OUTPUT_LIMIT = 240;
const SENSITIVE_ASSIGNMENT = /\b(?:provider(?:[_ -]?id)?|endpoint|base[_ -]?url|api[_ -]?key|authorization|model|response[_ -]?id|request[_ -]?id|trace[_ -]?id|prompt|instructions)\b\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)/giu;
const CREDENTIAL_ASSIGNMENT = /\b[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\b\s*=\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)/giu;
const CREDENTIAL_VALUE = /(?:\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{16,}|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]{5,})?)/giu;
const PROVIDER_ID_VALUE = /\b(?:response|resp|request|req|trace|gen|call|message|msg|transaction)[_-][A-Za-z0-9_-]{8,}\b/giu;
const URL_VALUE = /https?:\/\/[^\s<>"')\]}]+/giu;
const MODEL_ID_VALUE = /\b[a-z0-9][a-z0-9._-]{1,}\/[a-z0-9][a-z0-9._:-]{2,}\b/giu;

export function modelOutputLabel(value: unknown, maxLength = MODEL_OUTPUT_LIMIT): string | null {
  const extracted = extractModelOutput(value, 0);
  if (!extracted) return null;
  let text = extracted
    .replace(/```[\s\S]*?```/gu, " ")
    .replace(SENSITIVE_ASSIGNMENT, " ")
    .replace(CREDENTIAL_ASSIGNMENT, " ")
    .replace(CREDENTIAL_VALUE, "[敏感信息已隐藏]")
    .replace(PROVIDER_ID_VALUE, "[内部标识已隐藏]")
    .replace(URL_VALUE, "[链接已隐藏]")
    .replace(MODEL_ID_VALUE, "[模型标识已隐藏]");
  const rawJsonStart = text.search(/[{[](?=[^{}\[\]]{0,80}"[^"]+"\s*:)/u);
  if (rawJsonStart >= 0) text = text.slice(0, rawJsonStart);
  text = text.replace(/\s+/gu, " ").replace(/\s+([，。！？；：,.!?;:])/gu, "$1").trim();
  if (!text) return null;
  const characters = Array.from(text);
  const limit = Math.max(40, Math.floor(maxLength));
  return characters.length <= limit ? text : `${characters.slice(0, limit - 1).join("")}…`;
}

function extractModelOutput(value: unknown, depth: number): string | null {
  if (depth > 4) return null;
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return null;
    const fenced = text.match(/^```([^\r\n]*)[\r\n]+([\s\S]*?)\s*```$/u);
    if (fenced?.[2]) {
      const language = fenced[1]?.trim().toLowerCase();
      return language === "" || language === "json"
        ? extractModelOutput(fenced[2], depth + 1)
        : null;
    }
    if (text.startsWith("{") || text.startsWith("[")) {
      try {
        return extractModelOutput(JSON.parse(text), depth + 1);
      } catch {
        return text;
      }
    }
    return text;
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const object = value as Record<string, unknown>;
  for (const key of ["summary", "message", "result", "output", "final_output"]) {
    const extracted = extractModelOutput(object[key], depth + 1);
    if (extracted) return extracted;
  }
  return null;
}

export function goalSummaryLabel(goal: Goal): string {
  if (goal.predicates.length === 1) return predicateLabel(goal.predicates[0]!);
  return `完成 ${goal.predicates.length} 项结构化任务条件。`;
}

export function predicateLabel(predicate: GoalPredicate): string {
  switch (predicate.type) {
    case "robot_at":
      return `到达坐标 ${position(predicate.target)}`;
    case "robot_in_zone":
      return `进入区域 ${entityLabel(predicate.zone_id)}`;
    case "object_in_zone":
      return `${entityLabel(predicate.object_id)}${predicate.expected ? "位于" : "离开"}区域 ${entityLabel(predicate.zone_id)}`;
    case "object_at":
      return `将${entityLabel(predicate.object_id)}移动到 ${position(predicate.target)}`;
  }
}

export function missionResultLabel(checkpoint: HumanoidRunCheckpoint): string | null {
  if (checkpoint.status === "starting" || checkpoint.status === "running") {
    return "任务正在由层级智能体自主执行。";
  }
  if (checkpoint.status === "succeeded") {
    const passed = checkpoint.checker?.checks.filter((check) => check.passed).length ?? 0;
    return `${passed}/${checkpoint.goal.predicates.length} 项任务条件已通过。`;
  }
  if (checkpoint.status === "interrupted") return "任务已暂停，可继续运行。";
  return "任务未完成。";
}

export function runOptionLabel(run: RunListItem): string {
  return `${scenarioLabel(run.scenario_id)} · ${runStatusLabel(run.status)}`;
}

function position(point: { x: number; z: number }): string {
  return `[${point.x.toFixed(1)}, ${point.z.toFixed(1)}]`;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
