import type {
  Bootstrap,
  Goal,
  HumanoidRunMode,
  HumanoidRunDetails,
  RunListItem,
  RuntimeEvent,
  StreamState
} from "./types";
import { isHumanoidRunDetails } from "./types";
import {
  eventStreamFailureDecision,
  eventStreamRetryDelay
} from "./event-stream-recovery";
import { nextRuntimeEventCursor } from "./stream-state";

const PASSWORD_STORAGE_KEY = "hear.password";
let password = readStoredPassword();

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export interface RefreshedRunCursor {
  cursor: string | null;
  active: boolean;
}

export function setPassword(value: string): void {
  password = value;
  if (value) writeStoredPassword(value);
  else deleteStoredPassword();
}

function writeStoredPassword(value: string): void {
  try {
    sessionPasswordStorage()?.setItem(PASSWORD_STORAGE_KEY, value);
  } catch {
    // Authentication still works for this tab when browser storage is denied
    // or full; the module-level value remains authoritative for requests.
  }
}

function deleteStoredPassword(): void {
  try {
    sessionPasswordStorage()?.removeItem(PASSWORD_STORAGE_KEY);
  } catch {
    // The in-memory password was already cleared.
  }
}

export function hasPassword(): boolean {
  return password.length > 0;
}

function readStoredPassword(): string {
  try {
    return sessionPasswordStorage()?.getItem(PASSWORD_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

function sessionPasswordStorage(): Storage | undefined {
  try {
    return globalThis.sessionStorage;
  } catch {
    return undefined;
  }
}

export async function getBootstrap(): Promise<Bootstrap> {
  return request<Bootstrap>("/api/bootstrap");
}

export async function getRuns(): Promise<RunListItem[]> {
  return (await request<{ runs: RunListItem[] }>("/api/runs")).runs;
}

export async function getRun(
  runId: string,
  options: {
    signal?: AbortSignal;
    actionLimit?: number;
    providerLimit?: number;
    frameworkLimit?: number;
  } = {}
): Promise<HumanoidRunDetails> {
  const query = new URLSearchParams();
  if (options.actionLimit !== undefined) {
    query.set("actions", String(Math.min(5000, Math.max(1, options.actionLimit))));
  }
  if (options.providerLimit !== undefined) {
    query.set("provider", String(Math.min(5000, Math.max(1, options.providerLimit))));
  }
  if (options.frameworkLimit !== undefined) {
    query.set("framework", String(Math.min(5000, Math.max(1, options.frameworkLimit))));
  }
  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  const details = await request<HumanoidRunDetails>(`/api/runs/${encodeURIComponent(runId)}${suffix}`, {
    ...(options.signal ? { signal: options.signal } : {})
  });
  const provider = recentEntries(details.provider, options.providerLimit);
  const framework = recentEntries(details.framework, options.frameworkLimit);
  if (!isHumanoidRunDetails(details)) throw new ApiError(409, "该记录不是人形机器人任务");
  const actions = recentEntries(details.actions, options.actionLimit);
  const committedActions = Object.fromEntries(actions.map((action) => [
    action.transactionId,
    details.checkpoint.committed_actions[action.transactionId] ?? action
  ]));
  return {
    ...details,
    actions,
    provider,
    framework,
    checkpoint: {
      ...details.checkpoint,
      committed_actions: committedActions
    }
  };
}

export async function startRun(input: {
  mission: string;
  scenario_id: string;
  goal: Goal;
  run_mode: HumanoidRunMode;
}): Promise<string> {
  const result = await request<{ run_id: string }>("/api/runs", {
    method: "POST",
    body: JSON.stringify({ ...input, confirmed: true })
  });
  return result.run_id;
}

export async function resumeRun(runId: string): Promise<void> {
  await request(`/api/runs/${encodeURIComponent(runId)}/resume`, {
    method: "POST",
    body: JSON.stringify({ confirmed: true })
  });
}

export async function stopRun(runId: string): Promise<void> {
  await request(`/api/runs/${encodeURIComponent(runId)}/stop`, {
    method: "POST",
    body: JSON.stringify({ confirmed: true })
  });
}

export function subscribeToRun(
  runId: string,
  onEvent: (event: RuntimeEvent) => void,
  onError: (error: Error) => void,
  onState?: (state: StreamState) => void,
  after?: string,
  refreshCursor?: () => Promise<RefreshedRunCursor>
): () => void {
  const controller = new AbortController();
  void consumeEventLoop(
    runId,
    controller.signal,
    onEvent,
    onError,
    onState,
    after,
    refreshCursor
  );
  return () => {
    controller.abort();
    onState?.("inactive");
  };
}

async function consumeEventLoop(
  runId: string,
  signal: AbortSignal,
  onEvent: (event: RuntimeEvent) => void,
  onError: (error: Error) => void,
  onState: ((state: StreamState) => void) | undefined,
  initialCursor: string | undefined,
  refreshCursor: (() => Promise<RefreshedRunCursor>) | undefined
): Promise<void> {
  let cursor = initialCursor;
  let retryAttempt = 0;
  if (cursor === undefined && refreshCursor) {
    const refreshed = await refreshEventCursor(signal, refreshCursor, onError, onState);
    if (refreshed === null) return;
    cursor = refreshed;
  }
  while (!signal.aborted) {
    try {
      onState?.("connecting");
      await consumeEvents(
        runId,
        signal,
        (event) => {
          retryAttempt = 0;
          cursor = nextRuntimeEventCursor(cursor, event);
          onEvent(event);
        },
        () => onState?.("connected"),
        cursor
      );
      if (!signal.aborted) throw new Error("Live event stream closed");
    } catch (error) {
      if (signal.aborted) return;
      const cause = error instanceof Error ? error : new Error(String(error));
      const decision = eventStreamFailureDecision(cause);
      if (decision === "refresh_snapshot") {
        if (!refreshCursor) {
          onState?.("disconnected");
          onError(cause);
          return;
        }
        const refreshed = await refreshEventCursor(signal, refreshCursor, onError, onState);
        if (refreshed === null) return;
        cursor = refreshed;
        continue;
      }
      onState?.("disconnected");
      onError(cause);
      if (decision === "stop") return;
      retryAttempt += 1;
      await abortableDelay(eventStreamRetryDelay(retryAttempt), signal);
    }
  }
}

async function refreshEventCursor(
  signal: AbortSignal,
  refreshCursor: () => Promise<RefreshedRunCursor>,
  onError: (error: Error) => void,
  onState: ((state: StreamState) => void) | undefined
): Promise<string | null> {
  let retryAttempt = 0;
  while (!signal.aborted) {
    try {
      onState?.("connecting");
      const refreshed = await refreshCursor();
      if (signal.aborted || !refreshed.active) return null;
      if (refreshed.cursor) return refreshed.cursor;
    } catch (error) {
      if (signal.aborted) return null;
      const cause = error instanceof Error ? error : new Error(String(error));
      onState?.("disconnected");
      onError(cause);
      if (eventStreamFailureDecision(cause) === "stop") return null;
    }
    retryAttempt += 1;
    await abortableDelay(eventStreamRetryDelay(retryAttempt), signal);
  }
  return null;
}

async function consumeEvents(
  runId: string,
  signal: AbortSignal,
  onEvent: (event: RuntimeEvent) => void,
  onConnected: () => void,
  after?: string
): Promise<void> {
  const query = after ? `?after=${encodeURIComponent(after)}` : "";
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/events${query}`, {
    headers: {
      ...authHeaders(),
      ...(after ? { "last-event-id": after } : {})
    },
    signal
  });
  if (!response.ok || !response.body) throw await responseError(response);
  onConnected();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");
    const records = buffer.split("\n\n");
    buffer = records.pop() ?? "";
    for (const record of records) {
      const lines = record.split("\n");
      const streamCursor = lines
        .filter((line) => line.startsWith("id:"))
        .map((line) => line.slice(3).replace(/^ /, ""))
        .at(-1);
      const data = lines
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (!data) continue;
      const parsed = JSON.parse(data) as RuntimeEvent | { run_id: string };
      if (!("event_id" in parsed)) continue;
      if (parsed.cursor !== undefined && (
        typeof parsed.cursor !== "string"
        || parsed.cursor.length === 0
        || /[\r\n\0]/.test(parsed.cursor)
      )) {
        throw new SyntaxError("Runtime event contains an invalid cursor");
      }
      if (streamCursor !== undefined) {
        if (streamCursor.length === 0 || /[\r\n\0]/.test(streamCursor)) {
          throw new SyntaxError("Runtime event contains an invalid SSE id");
        }
        if (parsed.cursor !== undefined && parsed.cursor !== streamCursor) {
          throw new SyntaxError("Runtime event cursor does not match its SSE id");
        }
        parsed.cursor = streamCursor;
      }
      onEvent(parsed);
    }
  }
}

async function abortableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  await new Promise<void>((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const onAbort = (): void => {
      globalThis.clearTimeout(timer);
      resolve();
    };
    const timer = globalThis.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

async function request<T = unknown>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...authHeaders(),
      ...init.headers
    }
  });
  if (!response.ok) throw await responseError(response);
  return response.json() as Promise<T>;
}

function recentEntries<T>(entries: T[], limit: number | undefined): T[] {
  if (limit === undefined || entries.length <= limit) return entries;
  if (limit <= 0) return [];
  return entries.slice(-limit);
}

function authHeaders(): Record<string, string> {
  return password ? { authorization: `Bearer ${password}` } : {};
}

async function responseError(response: Response): Promise<ApiError> {
  let message = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json() as { error?: string };
    if (body.error) message = body.error;
  } catch {
    // The HTTP status remains the error when a proxy returns a non-JSON body.
  }
  return new ApiError(response.status, message);
}
