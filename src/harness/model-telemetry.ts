import { randomUUID } from "node:crypto";
import type { Model, ModelResponse } from "@openai/agents";
import {
  modelPayloadSha256,
  modelToolArgumentsSha256
} from "../domain/model-call-authority.js";
import { errorMessage } from "../runtime/error-message.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import {
  agentIdFromInstructions,
  clearAgentInvocationTransportInterruption,
  currentAgentInvocationId,
  recordAgentInvocationDecisionInterruption,
  recordAgentInvocationTransportInterruption
} from "./agent-scope.js";
import type {
  ModelProgressReceipt,
  ModelProgressSnapshot,
  ModelTelemetryRuntime
} from "./context-runtime.js";
import { modelResponseDisposition } from "./sdk-events.js";

const MAX_DECISIONS_WITHOUT_AUTHORITY_CHANGE = 6;
const MAX_REPEATED_NO_PROGRESS_RECEIPTS = 4;
const MAX_DECISIONS_WITHOUT_PHYSICAL_PROGRESS = 18;
const SDK_ABSENT_RESPONSE_ID = "FAKE_ID";
const transportInterruptionAgentIds = new WeakMap<object, string>();

/**
 * Binds one model facade to exactly one hierarchy node while recording every
 * real request against that node's durable budget. The facade never selects a
 * tool or substitutes a response; it only enforces identity and transport
 * semantics around the configured model.
 */
export function withModelTelemetry(
  model: Model,
  runtime: ModelTelemetryRuntime,
  boundAgentId: string,
  onModelResponseCompleted?: (
    agentId: string,
    usage: { inputTokens: number }
  ) => void | Promise<void>,
  streamEventIdleTimeoutMs?: number,
  requestTimeoutMs?: number
): Model {
  const progressGuard = runtime.modelProgressSnapshot
    ? new AuthoritativeModelProgressGuard(runtime.modelProgressSnapshot(boundAgentId))
    : undefined;
  let progressRecoveryEpoch = runtime.modelProgressRecoveryEpoch?.() ?? 0;
  const observeProgress = progressGuard
    ? (agentId: string, snapshot: ModelProgressSnapshot) => {
        const nextEpoch = runtime.modelProgressRecoveryEpoch?.() ?? progressRecoveryEpoch;
        if (nextEpoch !== progressRecoveryEpoch) {
          progressGuard.resetAfterTransportInterruption(snapshot);
          progressRecoveryEpoch = nextEpoch;
        }
        progressGuard.observe(agentId, snapshot);
      }
    : undefined;
  return {
    getResponse: async (request) => {
      const agentId = requestAgentId(request.systemInstructions, runtime.rootAgentId);
      assertModelBinding(boundAgentId, agentId);
      runtime.activeNode(agentId);
      const modelCallId = await runtime.recordModelCallStarted(agentId);
      let response: ModelResponse;
      try {
        response = normalizeModelResponseIdentity(
          await modelResponseWithIdleTimeout(
            model,
            request,
            streamEventIdleTimeoutMs
          ),
          modelCallId ?? randomUUID()
        );
      } catch (error) {
        if (modelCallId) await runtime.recordModelCallFailed?.(modelCallId, agentId);
        throw preserveModelInterruption(error, agentId);
      }
      clearAgentInvocationTransportInterruption();
      if (modelCallId && response.responseId) {
        await runtime.recordModelCallCompleted?.(
          completedModelCall(modelCallId, agentId, response.responseId, response.output)
        );
      } else if (modelCallId) {
        await runtime.recordModelCallFailed?.(modelCallId, agentId);
      }
      await onModelResponseCompleted?.(agentId, response.usage);
      const hasDecision = responseHasDecision(
        response.output,
        request.outputType !== "text"
      );
      const progressSnapshot = runtime.modelProgressSnapshot?.(agentId);
      if (hasDecision && observeProgress && progressSnapshot) {
        observeProgress(agentId, progressSnapshot);
      }
      return response;
    },
    getStreamedResponse: (request) => claimAndStream(
      model,
      runtime,
      observeProgress,
      boundAgentId,
      request,
      onModelResponseCompleted,
      streamEventIdleTimeoutMs,
      requestTimeoutMs
    ),
    ...(model.getRetryAdvice
      ? { getRetryAdvice: (request) => model.getRetryAdvice!(request) }
      : {})
  };
}

async function modelResponseWithIdleTimeout(
  model: Model,
  request: Parameters<Model["getResponse"]>[0],
  timeoutMs: number | undefined
): Promise<ModelResponse> {
  if (timeoutMs === undefined) return model.getResponse(request);
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error("Model response event idle timeout must be finite and positive");
  }
  const controller = new AbortController();
  const timedRequest = {
    ...request,
    signal: request.signal
      ? AbortSignal.any([request.signal, controller.signal])
      : controller.signal
  };
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      const error = modelResponseEventTimeoutError(timeoutMs);
      controller.abort(error);
      reject(error);
    }, timeoutMs);
  });
  try {
    return await Promise.race([model.getResponse(timedRequest), timeout]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

async function* claimAndStream(
  model: Model,
  runtime: ModelTelemetryRuntime,
  observeProgress: ((agentId: string, snapshot: ModelProgressSnapshot) => void) | undefined,
  boundAgentId: string,
  request: Parameters<Model["getStreamedResponse"]>[0],
  onModelResponseCompleted?: (
    agentId: string,
    usage: { inputTokens: number }
  ) => void | Promise<void>,
  streamEventIdleTimeoutMs?: number,
  requestTimeoutMs?: number
) {
  const agentId = requestAgentId(request.systemInstructions, runtime.rootAgentId);
  assertModelBinding(boundAgentId, agentId);
  runtime.activeNode(agentId);
  const modelCallId = await runtime.recordModelCallStarted(agentId);
  const localResponseIdentity = modelCallId ?? randomUUID();
  let terminalRecorded = false;
  const streamAbort = new AbortController();
  const streamRequest = {
    ...request,
    signal: request.signal
      ? AbortSignal.any([request.signal, streamAbort.signal])
      : streamAbort.signal
  };
  const iterator = model.getStreamedResponse(streamRequest)[Symbol.asyncIterator]();
  const responseDeadline = modelResponseDeadline(requestTimeoutMs);
  let streamDone = false;
  try {
    for (;;) {
      const next = await nextModelStreamEvent(
        iterator,
        streamEventIdleTimeoutMs,
        streamAbort,
        responseDeadline
      );
      if (next.done) {
        streamDone = true;
        break;
      }
      const event = next.value;
      if (event.type === "response_done") {
        const response = normalizeStreamResponseIdentity(
          event.response,
          localResponseIdentity
        );
        if (modelCallId) {
          await runtime.recordModelCallCompleted?.(completedModelCall(
            modelCallId,
            agentId,
            response.id,
            response.output
          ));
          terminalRecorded = true;
        }
        clearAgentInvocationTransportInterruption();
        await onModelResponseCompleted?.(agentId, response.usage);
        const hasDecision = responseHasDecision(
          response.output,
          request.outputType !== "text"
        );
        const progressSnapshot = runtime.modelProgressSnapshot?.(agentId);
        if (hasDecision && observeProgress && progressSnapshot) {
          observeProgress(agentId, progressSnapshot);
        }
        yield response === event.response ? event : { ...event, response };
        return;
      }
      yield event;
    }
  } catch (error) {
    if (modelCallId && !terminalRecorded) {
      terminalRecorded = true;
      await runtime.recordModelCallFailed?.(modelCallId, agentId);
    }
    throw preserveModelInterruption(error, agentId);
  } finally {
    if (!streamDone) {
      streamAbort.abort(new Error("Model stream consumer completed"));
      try {
        await iterator.return?.();
      } finally {
        if (modelCallId && !terminalRecorded) {
          terminalRecorded = true;
          await runtime.recordModelCallFailed?.(modelCallId, agentId);
        }
      }
    }
  }
}

async function nextModelStreamEvent<T>(
  iterator: AsyncIterator<T>,
  timeoutMs: number | undefined,
  controller: AbortController,
  responseDeadline?: { at: number; timeoutMs: number }
): Promise<IteratorResult<T>> {
  if (timeoutMs !== undefined && (!Number.isFinite(timeoutMs) || timeoutMs <= 0)) {
    throw new Error("Model stream event idle timeout must be finite and positive");
  }
  const remainingResponseMs = responseDeadline
    ? Math.max(0, responseDeadline.at - Date.now())
    : undefined;
  if (timeoutMs === undefined && remainingResponseMs === undefined) {
    return iterator.next();
  }
  const effectiveTimeoutMs = Math.min(
    timeoutMs ?? Number.POSITIVE_INFINITY,
    remainingResponseMs ?? Number.POSITIVE_INFINITY
  );
  const responseDeadlineWins = remainingResponseMs !== undefined
    && remainingResponseMs <= (timeoutMs ?? Number.POSITIVE_INFINITY);
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      const error = responseDeadlineWins && responseDeadline
        ? modelResponseDeadlineError(responseDeadline.timeoutMs)
        : modelStreamEventTimeoutError(timeoutMs!);
      controller.abort(error);
      reject(error);
    }, effectiveTimeoutMs);
  });
  try {
    return await Promise.race([iterator.next(), timeout]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function modelResponseDeadline(
  timeoutMs: number | undefined
): { at: number; timeoutMs: number } | undefined {
  if (timeoutMs === undefined) return undefined;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error("Model request timeout must be finite and positive");
  }
  return { at: Date.now() + timeoutMs, timeoutMs };
}

function modelStreamEventTimeoutError(timeoutMs: number): TypeError {
  const cause = Object.assign(new Error(
    `Model stream produced no SDK event for ${timeoutMs}ms`
  ), {
    code: "ETIMEDOUT"
  });
  return new TypeError("Model stream stalled", { cause });
}

function modelResponseEventTimeoutError(timeoutMs: number): TypeError {
  const cause = Object.assign(new Error(
    `Model response produced no SDK event for ${timeoutMs}ms`
  ), {
    code: "ETIMEDOUT"
  });
  return new TypeError("Model response stalled", { cause });
}

function modelResponseDeadlineError(timeoutMs: number): TypeError {
  const cause = Object.assign(new Error(
    `Model response did not complete within ${timeoutMs}ms`
  ), {
    code: "ETIMEDOUT"
  });
  return new TypeError("Model response exceeded its request deadline", { cause });
}

function normalizeModelResponseIdentity(
  response: ModelResponse,
  localIdentity: string
): ModelResponse {
  if (usableResponseId(response.responseId)) return response;
  const responseId = `local-response:${localIdentity}`;
  return {
    ...response,
    responseId,
    output: outputWithResponseIdentity(response.output, responseId)
  };
}

function normalizeStreamResponseIdentity<T extends {
  id: string;
  output: ModelResponse["output"];
}>(response: T, localIdentity: string): T {
  if (usableResponseId(response.id)) return response;
  const responseId = `local-response:${localIdentity}`;
  return {
    ...response,
    id: responseId,
    output: outputWithResponseIdentity(response.output, responseId)
  };
}

function outputWithResponseIdentity(
  output: ModelResponse["output"],
  responseId: string
): ModelResponse["output"] {
  return output.map((item) => ({
    ...item,
    providerData: {
      ...(item.providerData ?? {}),
      responseId
    }
  })) as ModelResponse["output"];
}

function usableResponseId(responseId: string | undefined): responseId is string {
  return responseId !== undefined
    && responseId.trim().length > 0
    && responseId !== SDK_ABSENT_RESPONSE_ID;
}

function completedModelCall(
  modelCallId: string,
  agentId: string,
  responseId: string,
  output: ModelResponse["output"]
): Parameters<NonNullable<ModelTelemetryRuntime["recordModelCallCompleted"]>>[0] {
  return {
    modelCallId,
    agentId,
    responseId,
    responseOutputSha256: modelPayloadSha256(output),
    toolCalls: output.flatMap((item) => {
      if (item.type !== "function_call") return [];
      return [{
        toolCallId: item.callId,
        toolName: item.name,
        argumentsSha256: modelToolArgumentsSha256(item.arguments)
      }];
    })
  };
}

function preserveModelInterruption(error: unknown, agentId: string): Error {
  const normalized = asError(error);
  if (isTransportInterruption(normalized)) {
    bindModelTransportInterruptionToAgent(normalized, agentId);
  } else if (normalized instanceof ModelDecisionStallError) {
    recordAgentInvocationDecisionInterruption(normalized);
  }
  return normalized;
}

/** Associates provider-neutral transport metadata with its concrete hierarchy node. */
export function bindModelTransportInterruptionToAgent(
  error: unknown,
  agentId: string
): Error {
  const normalized = asError(error);
  if (!isTransportInterruption(normalized)) return normalized;
  transportInterruptionAgentIds.set(normalized, agentId);
  recordAgentInvocationTransportInterruption(normalized);
  return normalized;
}

export function modelTransportInterruptionAgentIdFrom(
  error: unknown
): string | undefined {
  const pending: unknown[] = [error];
  const visited = new Set<object>();
  while (pending.length > 0 && visited.size < 16) {
    const candidate = pending.shift();
    if (candidate === null || typeof candidate !== "object"
      || visited.has(candidate)) continue;
    visited.add(candidate);
    const agentId = transportInterruptionAgentIds.get(candidate);
    if (agentId) return agentId;
    const wrapper = candidate as {
      error?: unknown;
      cause?: unknown;
      originalError?: unknown;
      errors?: unknown;
    };
    pending.push(wrapper.error, wrapper.cause, wrapper.originalError);
    if (Array.isArray(wrapper.errors)) pending.push(...wrapper.errors);
  }
  return undefined;
}

function requestAgentId(systemInstructions: unknown, rootAgentId: string): string {
  return currentAgentInvocationId()
    ?? agentIdFromInstructions(systemInstructions, rootAgentId);
}

function assertModelBinding(boundAgentId: string, requestAgentId: string): void {
  if (boundAgentId !== requestAgentId) {
    throw new Error(
      `Model binding mismatch: ${boundAgentId} cannot execute ${requestAgentId}'s turn`
    );
  }
}

function responseHasDecision(
  output: ModelResponse["output"],
  structuredOutputDecision: boolean
): boolean {
  return structuredOutputDecision || modelResponseDisposition(output).hasDecision;
}

export type ModelDecisionStallEvidence = {
  reason: "authority_unchanged" | "repeated_no_progress_receipt" | "physical_progress_absent";
  world_revision: number;
  cycle_index: number;
  checker_success: boolean;
  decisions_since_physical_progress: number;
  decisions_without_new_receipt: number;
  repeated_receipt_count: number;
  receipt_pattern: string | null;
  recent_transaction_ids: string[];
};

interface ModelProgressGuardLimits {
  decisionsWithoutAuthorityChange: number;
  repeatedNoProgressReceipts: number;
  decisionsWithoutPhysicalProgress: number;
}

/**
 * Detects valid-tool loops from durable physical evidence rather than turn
 * count. Observation, planning and execution may take any number of model
 * turns, but repeated decisions must eventually produce a new receipt and a
 * real world/cycle transition. The guard never chooses or executes an action.
 */
export class AuthoritativeModelProgressGuard {
  readonly #limits: ModelProgressGuardLimits;
  #cycleIndex: number;
  #checkerSuccess: boolean;
  #goalStateSha256: string;
  #authorityStateSha256: string;
  #previousReceiptIds: Set<string>;
  #decisionsSincePhysicalProgress = 0;
  #decisionsWithoutNewReceipt = 0;
  #lastNoProgressReceiptPattern: string | null = null;
  #consecutiveNoProgressReceiptCount = 0;

  constructor(
    snapshot: ModelProgressSnapshot,
    limits: Partial<ModelProgressGuardLimits> = {}
  ) {
    this.#limits = {
      decisionsWithoutAuthorityChange: limits.decisionsWithoutAuthorityChange
        ?? MAX_DECISIONS_WITHOUT_AUTHORITY_CHANGE,
      repeatedNoProgressReceipts: limits.repeatedNoProgressReceipts
        ?? MAX_REPEATED_NO_PROGRESS_RECEIPTS,
      decisionsWithoutPhysicalProgress: limits.decisionsWithoutPhysicalProgress
        ?? MAX_DECISIONS_WITHOUT_PHYSICAL_PROGRESS
    };
    for (const value of Object.values(this.#limits)) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error("Model progress guard limits must be positive safe integers");
      }
    }
    this.#cycleIndex = snapshot.cycleIndex;
    this.#checkerSuccess = snapshot.checkerSuccess;
    this.#goalStateSha256 = snapshot.goalStateSha256;
    this.#authorityStateSha256 = snapshot.authorityStateSha256;
    this.#previousReceiptIds = new Set(snapshot.receipts.map((receipt) => receipt.transactionId));
  }

  observe(agentId: string, snapshot: ModelProgressSnapshot): void {
    const newReceipts = snapshot.receipts.filter((receipt) => (
      !this.#previousReceiptIds.has(receipt.transactionId)
    ));
    this.#previousReceiptIds = new Set(
      snapshot.receipts.map((receipt) => receipt.transactionId)
    );
    const authoritativeProgress = snapshot.cycleIndex !== this.#cycleIndex
      || (!this.#checkerSuccess && snapshot.checkerSuccess)
      || snapshot.goalStateSha256 !== this.#goalStateSha256
      || snapshot.authorityStateSha256 !== this.#authorityStateSha256
      || newReceipts.some(receiptHasPhysicalProgress);
    this.#cycleIndex = snapshot.cycleIndex;
    this.#checkerSuccess = snapshot.checkerSuccess;
    this.#goalStateSha256 = snapshot.goalStateSha256;
    this.#authorityStateSha256 = snapshot.authorityStateSha256;
    if (authoritativeProgress) {
      this.#resetProgressWindow();
      return;
    }

    this.#decisionsSincePhysicalProgress += 1;
    if (newReceipts.length === 0) {
      this.#decisionsWithoutNewReceipt += 1;
    } else {
      this.#decisionsWithoutNewReceipt = 0;
    }
    let repeatedPattern: string | null = null;
    let repeatedCount = 0;
    for (const receipt of newReceipts) {
      const pattern = receiptPattern(receipt);
      if (pattern === this.#lastNoProgressReceiptPattern) {
        this.#consecutiveNoProgressReceiptCount += 1;
      } else {
        this.#lastNoProgressReceiptPattern = pattern;
        this.#consecutiveNoProgressReceiptCount = 1;
      }
      if (this.#consecutiveNoProgressReceiptCount > repeatedCount) {
        repeatedPattern = pattern;
        repeatedCount = this.#consecutiveNoProgressReceiptCount;
      }
    }

    let reason: ModelDecisionStallEvidence["reason"] | undefined;
    if (repeatedCount >= this.#limits.repeatedNoProgressReceipts) {
      reason = "repeated_no_progress_receipt";
    } else if (this.#decisionsWithoutNewReceipt
      >= this.#limits.decisionsWithoutAuthorityChange) {
      reason = "authority_unchanged";
    } else if (this.#decisionsSincePhysicalProgress
      >= this.#limits.decisionsWithoutPhysicalProgress) {
      reason = "physical_progress_absent";
    }
    if (!reason) return;

    const evidence: ModelDecisionStallEvidence = {
      reason,
      world_revision: snapshot.worldRevision,
      cycle_index: snapshot.cycleIndex,
      checker_success: snapshot.checkerSuccess,
      decisions_since_physical_progress: this.#decisionsSincePhysicalProgress,
      decisions_without_new_receipt: this.#decisionsWithoutNewReceipt,
      repeated_receipt_count: repeatedCount,
      receipt_pattern: repeatedPattern,
      recent_transaction_ids: newReceipts.slice(-6).map((receipt) => receipt.transactionId)
    };
    this.#resetProgressWindow();
    throw new ModelDecisionStallError(
      agentId,
      `Configured model kept making valid tool decisions without authoritative progress: ${reason}`,
      evidence
    );
  }

  resetAfterTransportInterruption(snapshot: ModelProgressSnapshot): void {
    this.#cycleIndex = snapshot.cycleIndex;
    this.#checkerSuccess = snapshot.checkerSuccess;
    this.#goalStateSha256 = snapshot.goalStateSha256;
    this.#authorityStateSha256 = snapshot.authorityStateSha256;
    this.#previousReceiptIds = new Set(
      snapshot.receipts.map((receipt) => receipt.transactionId)
    );
    this.#resetProgressWindow();
  }

  #resetProgressWindow(): void {
    this.#decisionsSincePhysicalProgress = 0;
    this.#decisionsWithoutNewReceipt = 0;
    this.#lastNoProgressReceiptPattern = null;
    this.#consecutiveNoProgressReceiptCount = 0;
  }
}

export class ModelDecisionStallError extends Error {
  readonly agentId: string;
  readonly evidence: ModelDecisionStallEvidence | undefined;

  constructor(
    agentId: string,
    message: string,
    evidence?: ModelDecisionStallEvidence
  ) {
    super(message);
    this.name = "ModelDecisionStallError";
    this.agentId = agentId;
    this.evidence = evidence ? structuredClone(evidence) : undefined;
  }
}

export function modelDecisionStallFrom(
  error: unknown
): ModelDecisionStallError | undefined {
  const pending: unknown[] = [error];
  const visited = new Set<object>();
  while (pending.length > 0 && visited.size < 12) {
    const candidate = pending.shift();
    if (candidate instanceof ModelDecisionStallError) return candidate;
    if (candidate === null || typeof candidate !== "object"
      || visited.has(candidate)) continue;
    visited.add(candidate);
    const wrapper = candidate as {
      error?: unknown;
      cause?: unknown;
      originalError?: unknown;
      errors?: unknown;
    };
    pending.push(wrapper.error, wrapper.cause, wrapper.originalError);
    if (Array.isArray(wrapper.errors)) pending.push(...wrapper.errors);
  }
  return undefined;
}

function receiptHasPhysicalProgress(receipt: ModelProgressReceipt): boolean {
  return receipt.frameCount > 0
    && receipt.worldAfterRevision > receipt.worldBeforeRevision;
}

function receiptPattern(receipt: ModelProgressReceipt): string {
  return [
    receipt.agentId,
    receipt.action,
    receipt.accepted ? "accepted" : "rejected",
    receipt.code,
    receipt.frameCount > 0 ? "frames" : "no_frames"
  ].join("|");
}

/** Preserves provider-neutral status and socket metadata on non-Error throws. */
function asError(error: unknown): Error {
  if (error instanceof Error) return error;
  const normalized = new Error(errorMessage(error));
  normalized.name = "ModelTransportError";
  if (error !== null && typeof error === "object") {
    const source = error as Record<string, unknown>;
    const target = normalized as Error & Record<string, unknown>;
    for (const key of ["status", "statusCode", "code", "type", "request_id", "requestId"] as const) {
      const value = source[key];
      if (typeof value === "string" || typeof value === "number") target[key] = value;
    }
    if (source.cause !== undefined) normalized.cause = source.cause;
    if (source.error !== undefined) target.error = source.error;
    if (Array.isArray(source.errors)) target.errors = source.errors;
    const retryAfter = standardRetryAfter(source);
    if (retryAfter !== undefined) {
      target.responseHeaders = { "retry-after": retryAfter };
    }
  }
  return normalized;
}

function standardRetryAfter(source: Record<string, unknown>): string | undefined {
  const response = source.response;
  const candidates = [
    source.responseHeaders,
    source.headers,
    response !== null && typeof response === "object"
      ? (response as Record<string, unknown>).headers
      : undefined
  ];
  for (const headers of candidates) {
    if (headers === null || typeof headers !== "object") continue;
    const getter = (headers as { get?: unknown }).get;
    if (typeof getter === "function") {
      try {
        const value = getter.call(headers, "retry-after") as unknown;
        if (typeof value === "string") return value;
      } catch {
        continue;
      }
    }
    for (const [key, value] of Object.entries(headers)) {
      if (key.toLowerCase() === "retry-after" && typeof value === "string") return value;
    }
  }
  return undefined;
}
