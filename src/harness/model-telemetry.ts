import type { Model, ModelResponse } from "@openai/agents";
import { errorMessage } from "../runtime/error-message.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import {
  agentIdFromModelPayload,
  recordAgentInvocationTransportInterruption
} from "./agent-scope.js";
import type {
  ModelProgressReceipt,
  ModelProgressSnapshot,
  ModelTelemetryRuntime
} from "./context-runtime.js";
import { modelResponseDisposition } from "./sdk-events.js";

const MAX_CONSECUTIVE_NO_DECISION_RESPONSES = 4;
const MAX_ROOT_CONSECUTIVE_NO_DECISION_RESPONSES = 3;
const MAX_DECISIONS_WITHOUT_AUTHORITY_CHANGE = 6;
const MAX_REPEATED_NO_PROGRESS_RECEIPTS = 4;
const MAX_DECISIONS_WITHOUT_PHYSICAL_PROGRESS = 18;

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
  const progressGuard = runtime.modelProgressSnapshot
    ? new AuthoritativeModelProgressGuard(runtime.modelProgressSnapshot())
    : undefined;
  return {
    getResponse: async (request) => {
      const agentId = agentIdFromModelPayload(request, runtime.rootAgentId);
      assertModelBinding(boundAgentId, agentId);
      runtime.activeNode(agentId);
      await runtime.recordModelCallStarted(agentId);
      try {
        const response = await model.getResponse(request);
        await onModelResponseCompleted?.(agentId);
        const hasDecision = decisionGuard.observe(agentId, response.output);
        const progressSnapshot = runtime.modelProgressSnapshot?.();
        if (hasDecision && progressGuard && progressSnapshot) {
          progressGuard.observe(agentId, progressSnapshot);
        }
        return response;
      } catch (error) {
        throw preserveModelInterruption(error);
      }
    },
    getStreamedResponse: (request) => claimAndStream(
      model,
      runtime,
      decisionGuard,
      progressGuard,
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
  progressGuard: AuthoritativeModelProgressGuard | undefined,
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
        const hasDecision = decisionGuard.observe(agentId, event.response.output);
        const progressSnapshot = runtime.modelProgressSnapshot?.();
        if (hasDecision && progressGuard && progressSnapshot) {
          progressGuard.observe(agentId, progressSnapshot);
        }
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

  observe(agentId: string, output: ModelResponse["output"]): boolean {
    const { hasDecision } = modelResponseDisposition(output);
    if (hasDecision) {
      this.#consecutive.set(agentId, 0);
      return true;
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
    return false;
  }
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
  readonly #receiptPatterns = new Map<string, number>();
  #worldRevision: number;
  #cycleIndex: number;
  #checkerSuccess: boolean;
  #previousReceiptIds: Set<string>;
  #decisionsSincePhysicalProgress = 0;
  #decisionsWithoutNewReceipt = 0;

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
    this.#worldRevision = snapshot.worldRevision;
    this.#cycleIndex = snapshot.cycleIndex;
    this.#checkerSuccess = snapshot.checkerSuccess;
    this.#previousReceiptIds = new Set(snapshot.receipts.map((receipt) => receipt.transactionId));
    this.#seedFromDurableReceipts(snapshot.receipts);
  }

  observe(agentId: string, snapshot: ModelProgressSnapshot): void {
    const newReceipts = snapshot.receipts.filter((receipt) => (
      !this.#previousReceiptIds.has(receipt.transactionId)
    ));
    this.#previousReceiptIds = new Set(
      snapshot.receipts.map((receipt) => receipt.transactionId)
    );
    const physicalProgress = snapshot.worldRevision !== this.#worldRevision
      || snapshot.cycleIndex !== this.#cycleIndex
      || (!this.#checkerSuccess && snapshot.checkerSuccess)
      || newReceipts.some(receiptHasPhysicalProgress);
    this.#worldRevision = snapshot.worldRevision;
    this.#cycleIndex = snapshot.cycleIndex;
    this.#checkerSuccess = snapshot.checkerSuccess;
    if (physicalProgress) {
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
      const count = (this.#receiptPatterns.get(pattern) ?? 0) + 1;
      this.#receiptPatterns.set(pattern, count);
      if (count > repeatedCount) {
        repeatedPattern = pattern;
        repeatedCount = count;
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

  #seedFromDurableReceipts(receipts: readonly ModelProgressReceipt[]): void {
    const lastPhysical = receipts.findLastIndex(receiptHasPhysicalProgress);
    const suffix = receipts.slice(lastPhysical + 1);
    this.#decisionsSincePhysicalProgress = suffix.length;
    for (const receipt of suffix) {
      const pattern = receiptPattern(receipt);
      this.#receiptPatterns.set(pattern, (this.#receiptPatterns.get(pattern) ?? 0) + 1);
    }
  }

  #resetProgressWindow(): void {
    this.#receiptPatterns.clear();
    this.#decisionsSincePhysicalProgress = 0;
    this.#decisionsWithoutNewReceipt = 0;
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
