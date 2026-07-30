import type {
  Bootstrap,
  Goal,
  RunDetails,
  RunListItem,
  RuntimeEvent,
  StreamState
} from "./types";
import { nextRuntimeEventCursor } from "./stream-state";

let password = sessionStorage.getItem("hear.password") ?? "";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function setPassword(value: string): void {
  password = value;
  if (value) sessionStorage.setItem("hear.password", value);
  else sessionStorage.removeItem("hear.password");
}

export function hasPassword(): boolean {
  return password.length > 0;
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
): Promise<RunDetails> {
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
  const details = await request<RunDetails>(`/api/runs/${encodeURIComponent(runId)}${suffix}`, {
    ...(options.signal ? { signal: options.signal } : {})
  });
  const actions = recentEntries(details.actions, options.actionLimit);
  const provider = recentEntries(details.provider, options.providerLimit);
  const framework = recentEntries(details.framework, options.frameworkLimit);
  const committedActions = Object.fromEntries(actions.map((action) => [
    action.transaction_id,
    details.checkpoint.committed_actions[action.transaction_id] ?? action
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
  after?: string
): () => void {
  const controller = new AbortController();
  void consumeEventLoop(runId, controller.signal, onEvent, onError, onState, after);
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
  initialCursor: string | undefined
): Promise<void> {
  let cursor = initialCursor;
  while (!signal.aborted) {
    try {
      onState?.("connecting");
      await consumeEvents(
        runId,
        signal,
        (event) => {
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
      onState?.("disconnected");
      onError(cause);
      if (cause instanceof ApiError && cause.status < 500) return;
      if (cause instanceof SyntaxError) return;
      await abortableDelay(800, signal);
    }
  }
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
    const records = buffer.split("\n\n");
    buffer = records.pop() ?? "";
    for (const record of records) {
      const data = record.split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (!data) continue;
      const parsed = JSON.parse(data) as RuntimeEvent | { run_id: string };
      if ("event_id" in parsed) onEvent(parsed);
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
      window.clearTimeout(timer);
      resolve();
    };
    const timer = window.setTimeout(() => {
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
