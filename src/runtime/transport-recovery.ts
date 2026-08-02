import { isDeepStrictEqual } from "node:util";
import type {
  ContextMemoryState,
  RunCheckpoint,
  TaskNode
} from "../domain/schema.js";

/**
 * A mission is a long conversation over one HTTP connection, and connections
 * break for reasons that have nothing to do with the mission: a socket closes
 * mid-body, a gateway times out, a rate limiter sheds load. The harness already
 * persists the agent state after every stream event, so none of that has to end
 * a run — but only if the failure can be told apart from a real one. Retrying a
 * malformed request or a rejected key forever would be worse than failing.
 *
 * These are standard HTTP and Node socket failure shapes, not any one vendor's:
 * classifying by transport error code is what keeps the runtime provider-neutral.
 */

/** Node/undici socket and DNS failures — the connection never carried a reply. */
const TRANSPORT_ERROR_CODES = new Set([
  "ECONNRESET",
  "ECONNREFUSED",
  "ECONNABORTED",
  "EPIPE",
  "ETIMEDOUT",
  "EHOSTUNREACH",
  "ENETUNREACH",
  "ENOTFOUND",
  "EAI_AGAIN",
  "UND_ERR_SOCKET",
  "UND_ERR_CONNECT_TIMEOUT",
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_BODY_TIMEOUT"
]);

/** Transient 4xx responses that explicitly permit a later attempt. */
const RETRYABLE_CLIENT_STATUS_CODES = new Set([408, 409, 425, 429]);
const INITIAL_TRANSPORT_BACKOFF_MS = 2_000;
const MAXIMUM_TRANSPORT_BACKOFF_MS = 30_000;
const MAXIMUM_SERVER_RETRY_AFTER_MS = 5 * 60_000;

export interface TransportRetryPlan {
  backoffMs: number;
  retryAfterMs: number | null;
  waitMs: number;
}

/**
 * Limits one uninterrupted outage instead of the lifetime of a mission. Any
 * completed model response proves that transport is available again and opens
 * a fresh recovery window for a later, independent outage.
 */
export class ConsecutiveTransportRecovery {
  readonly maximumAttempts: number;
  #attempts = 0;

  constructor(maximumAttempts: number) {
    if (!Number.isSafeInteger(maximumAttempts) || maximumAttempts < 1) {
      throw new Error("Maximum transport recovery attempts must be a positive safe integer");
    }
    this.maximumAttempts = maximumAttempts;
  }

  nextAttempt(): number | null {
    if (this.#attempts >= this.maximumAttempts) return null;
    this.#attempts += 1;
    return this.#attempts;
  }

  responseCompleted(): number {
    const recoveredAttempts = this.#attempts;
    this.#attempts = 0;
    return recoveredAttempts;
  }
}

/**
 * True when the failure is the connection rather than the conversation, so
 * resuming the mission from its persisted state is a real continuation and not
 * a retry of something that will fail identically.
 */
export function isTransportInterruption(error: unknown): boolean {
  for (const link of errorChain(error)) {
    const status = numberField(link, "statusCode") ?? numberField(link, "status");
    if (status !== undefined) {
      if (isRetryableStatus(status)) return true;
      if (status >= 400 && status <= 499) return false;
    }

    const code = stringField(link, "code");
    if (code && TRANSPORT_ERROR_CODES.has(code)) return true;

    // `fetch` collapses every connection failure into `TypeError: fetch failed`
    // or `TypeError: terminated`, carrying the real reason only in `cause` —
    // which is already in this chain, but is sometimes a plain string.
    if (link instanceof TypeError && /^(?:fetch failed|terminated)$/i.test(link.message)) {
      return true;
    }
  }
  return false;
}

/**
 * Combines bounded exponential backoff with the standard HTTP Retry-After
 * signal. A server hint may lengthen the wait but can never accelerate the
 * local backoff, and an extreme hint is capped so the mission remains
 * interruptible and resumable instead of sleeping inside one process forever.
 */
export function transportRetryPlan(
  error: unknown,
  attempt: number,
  now = Date.now()
): TransportRetryPlan {
  if (!Number.isSafeInteger(attempt) || attempt < 1) {
    throw new Error("Transport retry attempt must be a positive safe integer");
  }
  if (!Number.isFinite(now)) throw new Error("Transport retry clock must be finite");

  const backoffMs = Math.min(
    INITIAL_TRANSPORT_BACKOFF_MS * 2 ** Math.min(attempt - 1, 30),
    MAXIMUM_TRANSPORT_BACKOFF_MS
  );
  const retryAfterMs = retryAfterHint(error, now);
  return {
    backoffMs,
    retryAfterMs,
    waitMs: Math.max(backoffMs, retryAfterMs ?? 0)
  };
}

function isRetryableStatus(status: number): boolean {
  // Server-side failures may clear without changing the request. Other 4xx
  // responses describe a request/authentication problem and must terminate.
  return RETRYABLE_CLIENT_STATUS_CODES.has(status) || (status >= 500 && status <= 599);
}

function retryAfterHint(error: unknown, now: number): number | null {
  let longest: number | null = null;
  for (const link of errorChain(error)) {
    for (const headers of headerContainers(link)) {
      const value = headerValue(headers, "retry-after");
      if (value === undefined) continue;
      const parsed = parseRetryAfter(value, now);
      if (parsed !== null) longest = Math.max(longest ?? 0, parsed);
    }
  }
  return longest;
}

function headerContainers(value: unknown): unknown[] {
  if (value === null || typeof value !== "object") return [];
  const record = value as Record<string, unknown>;
  const containers = [record.responseHeaders, record.headers];
  const response = record.response;
  if (response !== null && typeof response === "object") {
    containers.push((response as Record<string, unknown>).headers);
  }
  return containers.filter((container) => container !== undefined);
}

function headerValue(headers: unknown, name: string): string | undefined {
  if (headers === null || typeof headers !== "object") return undefined;
  const getter = (headers as { get?: unknown }).get;
  if (typeof getter === "function") {
    try {
      const value = getter.call(headers, name) as unknown;
      if (typeof value === "string") return value;
    } catch {
      return undefined;
    }
  }
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() !== name) continue;
    if (typeof value === "string") return value;
    if (Array.isArray(value)) {
      return value.find((entry): entry is string => typeof entry === "string");
    }
  }
  return undefined;
}

function parseRetryAfter(value: string, now: number): number | null {
  const normalized = value.trim();
  if (/^\d+$/.test(normalized)) {
    const milliseconds = Number(normalized) * 1_000;
    if (!Number.isSafeInteger(milliseconds)) return MAXIMUM_SERVER_RETRY_AFTER_MS;
    return Math.min(milliseconds, MAXIMUM_SERVER_RETRY_AFTER_MS);
  }
  const timestamp = Date.parse(normalized);
  if (!Number.isFinite(timestamp)) return null;
  return Math.min(
    Math.max(0, timestamp - now),
    MAXIMUM_SERVER_RETRY_AFTER_MS
  );
}

/**
 * A failed opening request has no serialized RunState to resume. It may only
 * be sent again when the model never changed any authoritative mission state.
 * Model-call counters, timestamps, and the raw journal of the opening input are
 * telemetry, so they are deliberately excluded; everything that could affect
 * a later decision remains compared.
 */
export function canReplayInitialModelRequest(
  before: RunCheckpoint,
  after: RunCheckpoint
): boolean {
  return isDeepStrictEqual(replayAuthority(before), replayAuthority(after));
}

function replayAuthority(checkpoint: RunCheckpoint): unknown {
  return {
    status: checkpoint.status,
    root_id: checkpoint.root_id,
    active_agent_id: checkpoint.active_agent_id,
    active_agent_ids: checkpoint.active_agent_ids,
    nodes: Object.fromEntries(Object.entries(checkpoint.nodes).map(([id, node]) => [
      id,
      nodeReplayAuthority(node)
    ])),
    world: checkpoint.world,
    inflight_action: checkpoint.inflight_action,
    inflight_actions: checkpoint.inflight_actions,
    committed_actions: checkpoint.committed_actions,
    spatial_memory: checkpoint.spatial_memory,
    context_memory: contextReplayAuthority(checkpoint.context_memory),
    checker: checkpoint.checker,
    final_output: checkpoint.final_output,
    error: checkpoint.error
  };
}

function contextReplayAuthority(memory: ContextMemoryState): unknown {
  return {
    total_compactions: memory.total_compactions,
    last_compacted_at: memory.last_compacted_at,
    scopes: Object.fromEntries(Object.entries(memory.scopes)
      .filter(([, scope]) => scope.compaction_count > 0 || scope.summary !== null)
      .map(([id, scope]) => [id, {
        compaction_count: scope.compaction_count,
        summary: scope.summary,
        summary_origin: scope.summary_origin,
        summary_world_revision: scope.summary_world_revision,
        summary_voxel_revision: scope.summary_voxel_revision,
        last_compacted_at: scope.last_compacted_at
      }]))
  };
}

function nodeReplayAuthority(node: TaskNode): TaskNode {
  return {
    ...node,
    model_calls_used: 0,
    updated_at: "1970-01-01T00:00:00.000Z"
  };
}

/** Walks `cause` and aggregate `errors` links, guarding against cycles. */
function* errorChain(error: unknown, seen = new WeakSet<object>()): Generator<unknown> {
  if (error === null || typeof error !== "object") {
    yield error;
    return;
  }
  if (seen.has(error)) return;
  seen.add(error);
  yield error;

  const cause = (error as { cause?: unknown }).cause;
  if (cause !== undefined) yield* errorChain(cause, seen);

  // Some compatible transports reject with a response-shaped object whose
  // nested `error` carries the socket/status metadata instead of using Error.cause.
  const nested = (error as { error?: unknown }).error;
  if (nested !== undefined) yield* errorChain(nested, seen);

  const aggregated = (error as { errors?: unknown }).errors;
  if (Array.isArray(aggregated)) {
    for (const item of aggregated) yield* errorChain(item, seen);
  }
}

function stringField(value: unknown, key: string): string | undefined {
  if (value === null || typeof value !== "object") return undefined;
  const field = (value as Record<string, unknown>)[key];
  return typeof field === "string" ? field : undefined;
}

function numberField(value: unknown, key: string): number | undefined {
  if (value === null || typeof value !== "object") return undefined;
  const field = (value as Record<string, unknown>)[key];
  return typeof field === "number" && Number.isFinite(field) ? field : undefined;
}
