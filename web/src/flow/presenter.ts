import type { ActionReceipt, TaskNode } from "../types";
import {
  actionLabel,
  agentNameLabel,
  bodyChannelLabel,
  modelOutputLabel,
  nodeResultLabel,
  resultCodeLabel
} from "../ui-text";

export type FeedTone = "active" | "success" | "warning" | "neutral";
export type ActionCategory = "sense" | "plan" | "move" | "verify";

export interface PresentedAction {
  id: string;
  at: string;
  agent: string;
  title: string;
  detail: string;
  meta: string;
  tone: FeedTone;
  category: ActionCategory;
  channels: string[];
}

export interface ModelMoment {
  id: string;
  at: string;
  agent: string;
  title: string;
  detail: string;
  tone: FeedTone;
}

const SENSE_ACTIONS = new Set([
  "read_proprioception",
  "sense_scene",
  "survey_terrain",
  "scan_voxels",
  "inspect_voxel",
  "recall_spatial_memory",
  "inspect_entity",
  "query_contacts",
  "inspect_command"
]);
const PLAN_ACTIONS = new Set([
  "plan_base_path",
  "plan_arm_retraction",
  "plan_joint_targets",
  "solve_end_effector_position",
  "solve_end_effector_pose"
]);
const VERIFY_ACTIONS = new Set(["check_mission", "complete_mission"]);

export function presentAction(action: ActionReceipt): PresentedAction {
  const category: ActionCategory = SENSE_ACTIONS.has(action.name)
    ? "sense"
    : PLAN_ACTIONS.has(action.name) ? "plan"
      : VERIFY_ACTIONS.has(action.name) || action.kind === "checker" ? "verify" : "move";
  return {
    id: action.transaction_id,
    at: action.committed_at,
    agent: agentNameLabel(action.agent_name),
    title: action.accepted
      ? actionLabel(action.name)
      : `${actionLabel(action.name)}被拒绝`,
    detail: actionDetail(action),
    meta: action.frame_count > 0
      ? `${action.frame_count.toLocaleString("zh-CN")} 个物理帧`
      : resultCodeLabel(action.code),
    tone: action.accepted ? category === "move" ? "active" : "success" : "warning",
    category,
    channels: action.channels.map(bodyChannelLabel)
  };
}

export function presentFramework(entries: unknown[]): ModelMoment[] {
  const moments: ModelMoment[] = [];
  entries.forEach((entry, index) => {
    const outer = record(entry);
    const event = record(outer?.event);
    const item = record(event?.item);
    const raw = record(item?.rawItem);
    const rawAgent = stringOf(outer?.agent_name)
      ?? stringOf(record(item?.agent)?.name)
      ?? stringOf(outer?.scope)?.replace(/^agent:/, "")
      ?? "智能体";
    const agent = agentNameLabel(rawAgent);
    const at = stringOf(outer?.at) ?? new Date(0).toISOString();
    const recordId = stringOf(outer?.runtime_event_id) ?? `${index}-${at}`;

    if (event?.type === "agent_updated_stream_event") {
      const activeAgent = agentNameLabel(stringOf(event.agent) ?? rawAgent);
      moments.push({
        id: `agent-${recordId}`,
        at,
        agent: activeAgent,
        title: "接管当前执行流",
        detail: "当前由该智能体执行。",
        tone: "active"
      });
      return;
    }

    if (event?.type === "raw_model_stream_event") {
      const data = record(event.data);
      if (data?.type !== "response_done") return;
      const usage = record(data.usage);
      moments.push({
        id: `usage-${recordId}`,
        at,
        agent,
        title: "模型调用已完成",
        detail: usageDetail(usage),
        tone: "success"
      });
      return;
    }

    if (event?.type !== "run_item_stream_event") return;

    if (event.name === "reasoning_item_created") {
      return;
    }

    if (event.name === "tool_called") {
      const name = stringOf(raw?.name);
      if (!name) return;
      moments.push({
        id: `decision-${recordId}`,
        at,
        agent,
        title: `发起${actionLabel(name)}`,
        detail: decisionDetail(raw?.arguments),
        tone: "active"
      });
      return;
    }

    if (event.name === "message_output_created") {
      const text = textFromContent(raw?.content);
      const detail = modelOutputLabel(text);
      if (!detail) return;
      moments.push({
        id: `message-${recordId}`,
        at,
        agent,
        title: "模型输出",
        detail,
        tone: "neutral"
      });
      return;
    }

    if (event.name === "tool_output") {
      const name = stringOf(raw?.name) ?? "tool";
      const result = toolResult(name, raw, item);
      moments.push({
        id: `result-${recordId}`,
        at,
        agent,
        title: `${actionLabel(name)}回执`,
        detail: result.detail,
        tone: result.tone
      });
    }
  });
  return moments;
}

function toolResult(
  name: string,
  raw: Record<string, unknown> | null,
  item: Record<string, unknown> | null
): { detail: string; tone: FeedTone } {
  const parsed = toolPayload(raw, item);
  const accepted = typeof parsed?.accepted === "boolean" ? parsed.accepted : null;
  const status = toolStatus(stringOf(parsed?.status));
  const summary = modelOutputLabel(parsed, 150);
  const result = unique([
    accepted === true ? "动作已接受" : accepted === false ? "动作已拒绝" : null,
    stringOf(parsed?.code) ? resultCodeLabel(String(parsed?.code)) : null,
    status,
    ...toolFacts(parsed),
    summary
  ].filter((value): value is string => value !== null)).slice(0, 4);
  return {
    detail: result.join(" · ") || `${actionLabel(name)}已返回。`,
    tone: accepted === false || status === "执行受阻" || status === "执行失败"
      ? "warning"
      : accepted === true || status === "执行完成" ? "success" : "neutral"
  };
}

function toolPayload(
  raw: Record<string, unknown> | null,
  item: Record<string, unknown> | null
): Record<string, unknown> | null {
  const direct = record(raw?.output) ?? record(item?.output);
  const wrapped = direct;
  const text = stringOf(wrapped?.text)
    ?? stringOf(raw?.output)
    ?? stringOf(item?.output);
  return record(text ? parseJson(text) : null) ?? direct;
}

function toolStatus(status: string | null): string | null {
  if (status === "completed" || status === "succeeded") return "执行完成";
  if (status === "blocked") return "执行受阻";
  if (status === "failed") return "执行失败";
  if (status === "interrupted") return "执行已暂停";
  return null;
}

function toolFacts(payload: Record<string, unknown> | null): string[] {
  if (!payload) return [];
  const detail = record(payload.detail);
  const sampling = record(detail?.movement_sampling);
  const facts: string[] = [];
  const frameCount = numeric(payload.frame_count) ?? numeric(detail?.frame_count);
  if (frameCount !== null && frameCount > 0) {
    facts.push(`${frameCount.toLocaleString("zh-CN")} 个物理帧`);
  }
  const candidateCount = numeric(sampling?.choice_count);
  if (candidateCount !== null) facts.push(`${candidateCount} 个可达候选`);
  const target = pointText(payload.target) ?? pointText(detail?.target);
  if (target) facts.push(`目标 ${target}`);
  const checker = record(payload.checker);
  if (Array.isArray(checker?.checks)) {
    const passed = checker.checks.filter((check) => record(check)?.passed === true).length;
    facts.push(`${passed}/${checker.checks.length} 项条件通过`);
  }
  return facts;
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function usageDetail(usage: Record<string, unknown> | null): string {
  if (!usage) return "模型已响应。";
  const input = numeric(usage.inputTokens) ?? numeric(usage.input_tokens);
  const output = numeric(usage.outputTokens) ?? numeric(usage.output_tokens);
  const total = numeric(usage.totalTokens) ?? numeric(usage.total_tokens)
    ?? (input !== null && output !== null ? input + output : null);
  if (total === null) return "模型已响应。";
  return `${total.toLocaleString("zh-CN")} 个令牌${input === null || output === null
    ? ""
    : ` · 输入 ${input.toLocaleString("zh-CN")} · 输出 ${output.toLocaleString("zh-CN")}`}`;
}

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function nodeOutput(node: TaskNode): string | null {
  return nodeResultLabel(node);
}

export function shortTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "" : date.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  });
}

function actionDetail(action: ActionReceipt): string {
  const input = record(action.input);
  const detail = record(action.detail);
  if (!action.accepted) {
    return `动作被拒绝：${resultCodeLabel(action.code)}。`;
  }
  const target = pointText(input?.target);
  if (action.name === "plan_base_path" && target) return `已请求规划前往 ${target} 的可通行路线。`;
  if (action.name === "execute_base_plan") return "机器人已沿路线移动。";
  if (action.name === "survey_terrain") {
    const sampling = record(detail?.movement_sampling);
    const count = typeof sampling?.choice_count === "number" ? sampling.choice_count : null;
    return count === null ? "已更新机器人附近的可见地形。" : `已发现 ${count} 个可达探索边界。`;
  }
  if (action.name === "check_mission") {
    return action.code === "mission_satisfied" ? "所有结构化完成条件均已通过。" : "仍有结构化完成条件尚未满足。";
  }
  if (target) return `目标位置：${target}。`;
  if (action.channels.length > 0) {
    return `使用${action.channels.map(bodyChannelLabel).join("、")}控制通道完成动作。`;
  }
  return resultCodeLabel(action.code);
}

function decisionDetail(args: unknown): string {
  const parsed = typeof args === "string" ? parseJson(args) : args;
  const input = record(parsed);
  const target = pointText(input?.target);
  if (target) return `目标 ${target}`;
  const coordinate = voxelCoordinateText(input?.coordinate);
  if (coordinate) return `体素坐标 ${coordinate}`;
  return "等待工具回执。";
}

function voxelCoordinateText(value: unknown): string | null {
  const coordinate = record(value);
  if (!coordinate
    || !Number.isInteger(coordinate.column)
    || !Number.isInteger(coordinate.level)
    || !Number.isInteger(coordinate.row)) return null;
  return `[${coordinate.column}, ${coordinate.level}, ${coordinate.row}]`;
}

function pointText(value: unknown): string | null {
  const point = record(value);
  if (typeof point?.x !== "number" || typeof point.z !== "number") return null;
  return `(${point.x.toFixed(1)}, ${point.z.toFixed(1)})`;
}

function textFromContent(value: unknown): string | null {
  if (!Array.isArray(value)) return null;
  return value.map((entry) => stringOf(record(entry)?.text)).filter(Boolean).join(" ") || null;
}

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringOf(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
