import type { Model, ModelResponse } from "@openai/agents";
import { errorMessage } from "../runtime/error-message.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import {
  agentIdFromModelPayload,
  recordAgentInvocationTransportInterruption
} from "./agent-scope.js";
import type { ModelTelemetryRuntime } from "./context-runtime.js";
import { modelResponseDisposition } from "./sdk-events.js";

const MAX_CONSECUTIVE_NO_DECISION_RESPONSES = 4;
const MAX_ROOT_CONSECUTIVE_NO_DECISION_RESPONSES = 3;

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
  onModelResponseCompleted?: (agentId: string) => void | Promise<void>
): Model {
  const decisionGuard = new ModelDecisionGuard(
    boundAgentId === runtime.rootAgentId
      ? MAX_ROOT_CONSECUTIVE_NO_DECISION_RESPONSES
      : MAX_CONSECUTIVE_NO_DECISION_RESPONSES
  );
  return {
    getResponse: async (request) => {
      const agentId = agentIdFromModelPayload(request, runtime.rootAgentId);
      assertModelBinding(boundAgentId, agentId);
      runtime.activeNode(agentId);
      await runtime.recordModelCallStarted(agentId);
      try {
        const response = await model.getResponse(request);
        await onModelResponseCompleted?.(agentId);
        decisionGuard.observe(agentId, response.output);
        return response;
      } catch (error) {
        throw preserveModelInterruption(error);
      }
    },
    getStreamedResponse: (request) => claimAndStream(
      model,
      runtime,
      decisionGuard,
      boundAgentId,
      request,
      onModelResponseCompleted
    ),
    ...(model.getRetryAdvice
      ? { getRetryAdvice: (request) => model.getRetryAdvice!(request) }
      : {})
  };
}

async function* claimAndStream(
  model: Model,
  runtime: ModelTelemetryRuntime,
  decisionGuard: ModelDecisionGuard,
  boundAgentId: string,
  request: Parameters<Model["getStreamedResponse"]>[0],
  onModelResponseCompleted?: (agentId: string) => void | Promise<void>
) {
  const agentId = agentIdFromModelPayload(request, runtime.rootAgentId);
  assertModelBinding(boundAgentId, agentId);
  runtime.activeNode(agentId);
  await runtime.recordModelCallStarted(agentId);
  try {
    for await (const event of model.getStreamedResponse(request)) {
      if (event.type === "response_done") {
        await onModelResponseCompleted?.(agentId);
        decisionGuard.observe(agentId, event.response.output);
      }
      yield event;
    }
  } catch (error) {
    throw preserveModelInterruption(error);
  }
}

function preserveModelInterruption(error: unknown): Error {
  const normalized = asError(error);
  if (isTransportInterruption(normalized)) {
    recordAgentInvocationTransportInterruption(normalized);
  }
  return normalized;
}

function assertModelBinding(boundAgentId: string, requestAgentId: string): void {
  if (boundAgentId !== requestAgentId) {
    throw new Error(
      `Model binding mismatch: ${boundAgentId} cannot execute ${requestAgentId}'s turn`
    );
  }
}

/**
 * Tool choice is required for every HEAR agent. Some compatible endpoints can
 * nevertheless return reasoning or prose without the required call. The SDK
 * asks again, which is useful once but becomes an unbounded loop if prose keeps
 * resetting the counter. This guard counts every response without a tool
 * decision, never supplies a decision and never swaps models.
 */
class ModelDecisionGuard {
  readonly #consecutive = new Map<string, number>();

  constructor(
    readonly maximumConsecutiveResponses = MAX_CONSECUTIVE_NO_DECISION_RESPONSES
  ) {
    if (!Number.isInteger(maximumConsecutiveResponses) || maximumConsecutiveResponses < 1) {
      throw new Error("Model decision threshold must be a positive integer");
    }
  }

  observe(agentId: string, output: ModelResponse["output"]): void {
    const { hasDecision } = modelResponseDisposition(output);
    if (hasDecision) {
      this.#consecutive.set(agentId, 0);
      return;
    }
    const consecutive = (this.#consecutive.get(agentId) ?? 0) + 1;
    this.#consecutive.set(agentId, consecutive);
    if (consecutive >= this.maximumConsecutiveResponses) {
      this.#consecutive.set(agentId, 0);
      throw new ModelDecisionStallError(
        agentId,
        `Configured model returned ${consecutive} consecutive responses `
        + "without the required tool decision"
      );
    }
  }
}

export class ModelDecisionStallError extends Error {
  readonly agentId: string;

  constructor(agentId: string, message: string) {
    super(message);
    this.name = "ModelDecisionStallError";
    this.agentId = agentId;
  }
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
