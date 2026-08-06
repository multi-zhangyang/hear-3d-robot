
/**
 * A mission is a long conversation over one HTTP connection, and connections
 * break for reasons that have nothing to do with the mission: a socket closes
 * mid-body, a gateway times out, a rate limiter sheds load. The harness persists
 * resumable SDK boundaries, so none of that has to end a run when the failure
 * can be distinguished from a malformed request or rejected credential. Using
 * standard HTTP and Node socket shapes keeps recovery provider-neutral without
 * rewriting a model decision.
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

export class PerAgentTransportRecovery {
  readonly maximumAttempts: number;
  readonly #recoveries = new Map<string, ConsecutiveTransportRecovery>();

  constructor(maximumAttempts: number) {
    if (!Number.isSafeInteger(maximumAttempts) || maximumAttempts < 1) {
      throw new Error("Maximum transport recovery attempts must be a positive safe integer");
    }
    this.maximumAttempts = maximumAttempts;
  }

  nextAttempt(agentId: string): number | null {
    return this.#recovery(agentId).nextAttempt();
  }

  responseCompleted(agentId: string): number {
    assertRecoveryAgentId(agentId);
    const recovery = this.#recoveries.get(agentId);
    if (!recovery) return 0;
    this.#recoveries.delete(agentId);
    return recovery.responseCompleted();
  }

  #recovery(agentId: string): ConsecutiveTransportRecovery {
    assertRecoveryAgentId(agentId);
    const existing = this.#recoveries.get(agentId);
    if (existing) return existing;
    const recovery = new ConsecutiveTransportRecovery(this.maximumAttempts);
    this.#recoveries.set(agentId, recovery);
    return recovery;
  }
}

function assertRecoveryAgentId(agentId: string): void {
  if (agentId.trim().length === 0) {
    throw new Error("Transport recovery Agent identity must not be empty");
  }
}

/**
 * True when the failure is the connection rather than the conversation, so
 * resuming the mission from its persisted state is a real continuation and not
 * a retry of something that will fail identically.
 */
export function isTransportInterruption(error: unknown): boolean {
  for (const link of errorChain(error)) {
    const status = transportStatusFromLink(link);
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

export function transportStatusCode(error: unknown): number | undefined {
  for (const link of errorChain(error)) {
    const status = transportStatusFromLink(link);
    if (status !== undefined) return status;
  }
  return undefined;
}

function transportStatusFromLink(value: unknown): number | undefined {
  return numberField(value, "statusCode")
    ?? numberField(value, "status")
    ?? normalizedTransportCode(value);
}

function normalizedTransportCode(value: unknown): number | undefined {
  if (value === null || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  if (record.name !== "ModelTransportError") return undefined;
  const code = numberField(value, "code");
  return code !== undefined && Number.isInteger(code) && code >= 100 && code <= 599
    ? code
    : undefined;
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
