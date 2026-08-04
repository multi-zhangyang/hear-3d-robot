import { providerActivityFrom } from "./stream-state";
import type { ProviderActivity, RuntimeEvent } from "./types";

export type ModelActivityPhase =
  | "offline"
  | "ready"
  | "active"
  | "verified"
  | "recovering"
  | "error";

export interface ModelActivityState {
  phase: ModelActivityPhase;
  agentId: string | null;
  startedAt: string | null;
  completedAt: string | null;
}

export function createModelActivity(configured: boolean): ModelActivityState {
  return {
    phase: configured ? "ready" : "offline",
    agentId: null,
    startedAt: null,
    completedAt: null
  };
}

export function modelActivityFromJournal(
  configured: boolean,
  entries: unknown[],
  runIsActive: boolean
): ModelActivityState {
  const folded = entries.reduce<ModelActivityState>((state, entry) => {
    const activity = providerActivityFrom(entry);
    return activity ? reduceProviderActivity(state, activity) : state;
  }, createModelActivity(configured));
  return settleModelActivity(folded, runIsActive);
}

export function reduceRuntimeModelActivity(
  current: ModelActivityState,
  event: RuntimeEvent
): ModelActivityState {
  if (event.type !== "model_request_started") return current;
  const data = record(event.data);
  return {
    phase: "active",
    agentId: stringOf(data?.agent_id) ?? current.agentId,
    startedAt: event.at,
    completedAt: null
  };
}

export function reduceProviderActivity(
  current: ModelActivityState,
  activity: ProviderActivity
): ModelActivityState {
  const status = activity.status;
  const at = activity.at;
  const agentId = activity.agentId ?? current.agentId;
  if (status === "configured") {
    return { phase: "ready", agentId, startedAt: null, completedAt: at };
  }
  if (status === "contacted" || status === "streaming_text") {
    return {
      phase: "active",
      agentId,
      startedAt: current.phase === "active" ? current.startedAt ?? at : at,
      completedAt: null
    };
  }
  if (status === "usable_stream" || status === "transport_recovered") {
    return { ...current, phase: "verified", agentId, completedAt: at };
  }
  if (status === "transport_interrupted" || status === "model_decision_recovery") {
    return { ...current, phase: "recovering", agentId, completedAt: at };
  }
  if (status === "no_text" || status.includes("error")) {
    return { ...current, phase: "error", agentId, completedAt: at };
  }
  return current;
}

export function settleModelActivity(
  current: ModelActivityState,
  runIsActive: boolean
): ModelActivityState {
  if (runIsActive || current.phase === "offline" || current.phase === "error") return current;
  if (current.phase === "active") return { ...current, phase: "ready" };
  if (current.phase === "recovering") return { ...current, phase: "error" };
  return current;
}

export function modelActivityLabel(phase: ModelActivityPhase): string {
  if (phase === "active") return "模型调用中";
  if (phase === "verified") return "模型已响应";
  if (phase === "recovering") return "模型恢复中";
  if (phase === "ready") return "模型已就绪";
  if (phase === "error") return "模型异常";
  return "模型未配置";
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringOf(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}
