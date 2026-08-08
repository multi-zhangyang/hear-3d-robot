import type { HumanoidActionReceipt, HumanoidEmbodiedEpisode, TaskNode } from "../types";
import {
  actionLabel,
  agentNameLabel,
  bodyChannelLabel,
  modelOutputLabel,
  nodeResultLabel,
  resultCodeLabel
} from "../ui-text";

type FeedTone = "active" | "success" | "warning" | "neutral";
export type ActionCategory = "sense" | "plan" | "move" | "mutate" | "verify";

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
  kind: "agent" | "model_completed" | "tool_called" | "model_output" | "tool_output";
  cycleId: string | null;
  title: string;
  detail: string;
  tone: FeedTone;
}

const HUMANOID_SENSE_ACTIONS = new Set(["observe_humanoid"]);
const HUMANOID_PLAN_ACTIONS = new Set([
  "submit_humanoid_skill_plan",
  "begin_humanoid_skill",
  "plan_humanoid_skill",
  "plan_whole_body_motion",
  "plan_whole_body_motion_candidates",
  "plan_humanoid_navigation"
]);
const HUMANOID_MUTATION_ACTIONS = new Set(["remove_world_block"]);

export function presentAction(action: HumanoidActionReceipt): PresentedAction {
  const name = action.action;
  const accepted = action.accepted;
  const channels = action.channels;
  const category: ActionCategory = HUMANOID_SENSE_ACTIONS.has(name)
    ? "sense"
    : HUMANOID_PLAN_ACTIONS.has(name)
      ? "plan"
      : HUMANOID_MUTATION_ACTIONS.has(name) ? "mutate" : "move";
  return {
    id: action.transactionId,
    at: action.committedAt,
    agent: agentNameLabel(action.agentId),
    title: accepted
      ? actionLabel(name)
      : `${actionLabel(name)}被拒绝`,
    detail: actionDetail(action),
    meta: receiptFrames(action) > 0
      ? `${receiptFrames(action).toLocaleString("zh-CN")} 个物理帧`
      : resultCodeLabel(action.code),
    tone: accepted ? category === "move" ? "active" : "success" : "warning",
    category,
    channels: channels.map(bodyChannelLabel)
  };
}

export function presentEmbodiedEpisode(episode: HumanoidEmbodiedEpisode): PresentedAction {
  const candidateSelection = episode.candidate_count !== undefined
    && episode.selected_rank !== undefined
    ? ` · 候选 ${episode.selected_rank}/${episode.candidate_count}`
    : "";
  const optionResult = episode.motion_option
    ? ` · 物理达成 ${episode.motion_option.actual_termination_frame}/${episode.motion_option.predicted_termination_frame} 帧`
    : "";
  const mutationResult = episode.world_mutations?.length
    ? ` · 世界提交 ${episode.world_mutations.length}`
    : "";
  return {
    id: `episode-${episode.sequence}-${episode.transaction_id}`,
    at: episode.recorded_at,
    agent: "具身记忆",
    title: "物理经历已记住",
    detail: modelOutputLabel(episode.model_summary, 160) ?? "已保存本次物理执行结果。",
    meta: `${episode.frame_count.toLocaleString("zh-CN")} 个物理帧${candidateSelection}${optionResult}${mutationResult} · 世界版本 ${episode.result_world_revision ?? episode.world_after_revision}`,
    tone: episode.fallen ? "warning" : episode.goal_success ? "success" : "neutral",
    category: "verify",
    channels: []
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
    const cycleId = stringOf(record(outer?.cycle)?.cycle_id);

    if (event?.type === "agent_updated_stream_event") {
      const activeAgent = agentNameLabel(stringOf(event.agent) ?? rawAgent);
      moments.push({
        id: `agent-${recordId}`,
        at,
        agent: activeAgent,
        kind: "agent",
        cycleId,
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
        kind: "model_completed",
        cycleId,
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
        kind: "tool_called",
        cycleId,
        title: `发起${actionLabel(name)}`,
        detail: decisionDetail(raw?.arguments),
        tone: "active"
      });
      return;
    }

    if (event.name === "message_output_created") {
      const text = textFromContent(raw?.content);
      const detail = presentedModelOutput(text);
      if (!detail) return;
      moments.push({
        id: `message-${recordId}`,
        at,
        agent,
        kind: "model_output",
        cycleId,
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
        kind: "tool_output",
        cycleId,
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
  const facts: string[] = [];
  const frameCount = numeric(payload.frame_count) ?? numeric(detail?.frame_count);
  if (frameCount !== null && frameCount > 0) {
    facts.push(`${frameCount.toLocaleString("zh-CN")} 个物理帧`);
  }
  const target = pointText(payload.target) ?? pointText(detail?.target);
  if (target) facts.push(`目标 ${target}`);
  const candidateCount = numeric(detail?.candidate_count);
  const selectedRank = numeric(detail?.selected_rank);
  if (candidateCount !== null) {
    facts.push(selectedRank === null
      ? `${candidateCount} 个候选均已预演`
      : `${candidateCount} 个候选 · 选择第 ${selectedRank} 个`);
  }
  const checker = record(payload.checker);
  if (Array.isArray(checker?.checks)) {
    const passed = checker.checks.filter((check) => record(check)?.passed === true).length;
    facts.push(`${passed}/${checker.checks.length} 项条件通过`);
  }
  const removal = record(detail?.removal_transaction);
  const chunkRevision = numeric(removal?.projected_chunk_revision);
  if (chunkRevision !== null) facts.push(`区块 R${chunkRevision}`);
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

function actionDetail(action: HumanoidActionReceipt): string {
  const name = action.action;
  const input = record(action.input);
  if (name === "execute_whole_body_motion") {
    const optionDetail = physicalOptionDetail(action);
    if (optionDetail) return optionDetail;
  }
  if (!action.accepted) {
    return `动作被拒绝：${resultCodeLabel(action.code)}。`;
  }
  const target = pointText(input?.target);
  if (name === "submit_humanoid_skill_plan") {
    const strategies = Array.isArray(input?.strategies) ? input.strategies.length : 0;
    return strategies > 0
      ? `模型提交了 ${strategies} 种技能策略，Harness 已绑定本轮选择。`
      : "模型已提交推进当前目标的技能策略。";
  }
  if (name === "begin_humanoid_skill") return "技能阶段已绑定当前 Goal 与实时世界版本。";
  if (name === "plan_humanoid_skill") return "技能路线已完成导航、全身控制与 MuJoCo 物理预演。";
  if (name === "execute_humanoid_skill") return "机器人已执行模型选择并通过预演的技能路线。";
  if (name === "plan_humanoid_navigation" && target) return `已对前往 ${target} 的双足路线完成物理预演。`;
  if (name === "execute_humanoid_navigation") return "机器人已执行模型选择的双足导航分块。";
  if (name === "plan_whole_body_motion_candidates") {
    const detail = record(action.detail);
    const candidateCount = numeric(detail?.candidate_count);
    const selectedRank = numeric(detail?.selected_rank);
    return candidateCount !== null && selectedRank !== null
      ? `${candidateCount} 个模型候选已分别完成 MuJoCo 预演，选择第 ${selectedRank} 个可行动作。`
      : "模型候选已分别完成 MuJoCo 物理预演。";
  }
  if (name === "plan_whole_body_motion") return "已对连续全身动作完成 MuJoCo 物理预演。";
  if (name === "execute_whole_body_motion") return "已在 MuJoCo 中执行连续全身动作。";
  if (name === "remove_world_block") {
    const detail = record(action.detail);
    const removal = record(detail?.removal_transaction);
    const chunkRevision = numeric(removal?.projected_chunk_revision);
    return chunkRevision === null
      ? "物理接触证据已提交到权威世界。"
      : `目标方块已从权威世界移除 · 区块 R${chunkRevision}。`;
  }
  if (name === "observe_humanoid") return "已更新头部感知、身体姿态、接触、平衡和持久对象状态。";
  if (target) return `目标位置：${target}。`;
  if (action.channels.length > 0) {
    return `使用${action.channels.map(bodyChannelLabel).join("、")}控制通道完成动作。`;
  }
  return resultCodeLabel(action.code);
}

function presentedModelOutput(value: unknown): string | null {
  const raw = typeof value === "string" ? value : JSON.stringify(value);
  if (/coordinator_phase/iu.test(raw) && /cycle_completion/iu.test(raw)) {
    return "当前物理执行、执行后感知与目标验收证据已经齐备。";
  }
  if (/goal_dag(?:_status|\.status)/iu.test(raw) && /current_goal_epoch_id/iu.test(raw)) {
    return "当前目标阶段已收束，正在等待目标管理智能体选择下一目标。";
  }
  return modelOutputLabel(value);
}

function physicalOptionDetail(action: HumanoidActionReceipt): string | null {
  const detail = record(action.detail);
  const result = record(detail?.result);
  const option = record(result?.option);
  if (!option) return null;
  const predictedFrame = numeric(option.predicted_termination_frame);
  const actualFrame = numeric(option.actual_termination_frame);
  const motion = record(result?.motion);
  const step = numeric(motion?.control_step_seconds);
  const timing = predictedFrame !== null && actualFrame !== null
    ? step !== null
      ? `预测 ${(predictedFrame * step).toFixed(2)} 秒 · 实际 ${(actualFrame * step).toFixed(2)} 秒`
      : `预测 ${predictedFrame} 帧 · 实际 ${actualFrame} 帧`
    : "";
  if (option.status === "succeeded") {
    return timing ? `物理目标稳定达成 · ${timing}` : "物理目标稳定达成。";
  }
  if (option.termination_reason === "motion_goal_uncertain") {
    return timing ? `目标状态不可确定 · ${timing}` : "目标状态不可确定，未判定成功。";
  }
  if (option.termination_reason === "motion_goal_unmet") {
    return timing ? `物理目标未达成 · ${timing}` : "动作上界已到，物理目标未达成。";
  }
  if (option.termination_reason === "execution_drift") {
    const driftStreak = numeric(option.drift_streak);
    return driftStreak === null
      ? "执行偏离物理预演，剩余动作已截断并交回重规划。"
      : `执行连续 ${driftStreak} 帧偏离物理预演，已截断并交回重规划。`;
  }
  if (option.termination_reason === "motion_constraint_violated") {
    return "持续物理约束被违反，动作已提前终止并交回重规划。";
  }
  return "物理执行失败，未判定目标达成。";
}

export function receiptFrames(action: HumanoidActionReceipt): number {
  return action.frameCount;
}

function decisionDetail(args: unknown): string {
  const parsed = typeof args === "string" ? parseJson(args) : args;
  const input = record(parsed);
  const target = pointText(input?.target);
  if (target) return `目标 ${target}`;
  return "等待工具回执。";
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
