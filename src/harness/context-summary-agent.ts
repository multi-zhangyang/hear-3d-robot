import {
  Agent,
  Runner,
  tool,
  type AgentInputItem,
  type Model
} from "@openai/agents";
import { z } from "zod";
import {
  ContextCompactionSummarySchema,
  type ContextCompactionSummary,
  type JsonValue
} from "../domain/schema.js";
import {
  CONTEXT_COMPACTOR_MAX_ATTEMPTS,
  CONTEXT_COMPACTOR_TURNS_PER_ATTEMPT
} from "../runtime/context-budget.js";
import { errorMessage } from "../runtime/error-message.js";
import {
  estimateModelInputTokens,
  estimateToolTokens
} from "./token-budget.js";

const COMMIT_CONTEXT_TOOL = "commit_context_checkpoint";
const COMMIT_CONTEXT_DESCRIPTION = "Commit the only context checkpoint for this compaction pass.";
const COMPACTOR_INSTRUCTIONS = [
  "Compress historical agent and tool context into one durable working checkpoint.",
  `Call ${COMMIT_CONTEXT_TOOL} exactly once and return no prose.`,
  "Preserve operator constraints, decisions, unfinished work, blockers, and the next useful actions.",
  "Only values enumerated in the receipt allowlists may appear in transaction_ids.",
  "COMPLETED accepts only accepted receipts; BLOCKERS may cite accepted or rejected receipts.",
  "An SDK callId is not a receipt transaction id; omit an evidence entry when no allowed receipt proves it.",
  "Never turn a rejected receipt into completed evidence and never invent coordinates or success.",
  "The authority block is current; older observations and the prior checkpoint may be stale."
].join(" ");

export interface ContextSummaryRequest {
  priorSummary: ContextCompactionSummary | null;
  sourceItems: AgentInputItem[];
  authority: JsonValue;
  acceptedTransactionIds: string[];
  blockerTransactionIds: string[];
  maxInputTokens: number;
  signal?: AbortSignal;
}

export interface ContextSummaryUsage {
  requests: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export interface ContextSummaryResult {
  summary: ContextCompactionSummary;
  origin: "model" | "authority_projection";
  usage: ContextSummaryUsage;
}

export interface ContextSummaryGenerator {
  generate(request: ContextSummaryRequest): Promise<ContextSummaryResult>;
}

/**
 * A real Agents SDK model run that turns historical model/tool items into a
 * typed checkpoint. Prose-only failures are retried in fresh bounded turns so
 * one bad branch cannot recursively consume the provider window. Failed model
 * attempts never become a synthetic durable checkpoint: callers must preserve
 * the uncompressed Session and resume when a real checkpoint can be produced.
 */
export class AgentsSdkContextSummaryGenerator implements ContextSummaryGenerator {
  readonly #runner: Runner;
  readonly #model: Model;
  readonly #temperature: number;
  readonly #maxOutputTokens: number;
  readonly #onModelResponseCompleted: (() => void | Promise<void>) | undefined;

  constructor(input: {
    model: Model;
    temperature: number;
    maxOutputTokens: number;
    onModelResponseCompleted?: () => void | Promise<void>;
  }) {
    this.#model = input.model;
    this.#temperature = input.temperature;
    this.#maxOutputTokens = input.maxOutputTokens;
    this.#onModelResponseCompleted = input.onModelResponseCompleted;
    this.#runner = new Runner({
      tracingDisabled: true,
      traceIncludeSensitiveData: false,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 1 },
      toolNotFoundBehavior: "raise_error",
      workflowName: "HEAR context compaction"
    });
  }

  async generate(request: ContextSummaryRequest): Promise<ContextSummaryResult> {
    const accepted = [...new Set(request.acceptedTransactionIds)];
    const blockers = [...new Set(request.blockerTransactionIds)];
    const boundedRequest = {
      ...request,
      acceptedTransactionIds: accepted,
      blockerTransactionIds: blockers
    };
    const usage: ContextSummaryUsage = {
      requests: 0,
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0
    };
    const estimatedInputTokens = estimateContextSummaryRequestTokens(boundedRequest);
    if (estimatedInputTokens > request.maxInputTokens) {
      throw new ContextCompactionInterruption(
        `Context compactor request is estimated at ${estimatedInputTokens} input tokens, `
        + `above its ${request.maxInputTokens}-token hard limit; no request was sent`,
        { usage }
      );
    }
    const countedModel = withUsageCounter(
      this.#model,
      usage,
      this.#onModelResponseCompleted
    );
    let lastFailure: string | undefined;

    for (let attempt = 1; attempt <= CONTEXT_COMPACTOR_MAX_ATTEMPTS; attempt += 1) {
      request.signal?.throwIfAborted();
      const agent = compactorAgent({
        model: countedModel,
        temperature: this.#temperature,
        maxOutputTokens: this.#maxOutputTokens,
        accepted,
        blockers
      });
      const prompt = compactionPrompt(boundedRequest, attempt, lastFailure);
      const attemptInputTokens = estimateContextSummaryPromptTokens(prompt, accepted, blockers);
      if (attemptInputTokens > request.maxInputTokens) {
        lastFailure = `Retry prompt requires an estimated ${attemptInputTokens} input tokens, `
          + `above its ${request.maxInputTokens}-token hard limit; no oversized request was sent`;
        break;
      }
      try {
        // Each attempt starts from the same bounded evidence and no Session.
        // Invalid reasoning from a prior attempt is intentionally not replayed.
        const result = await this.#runner.run(
          agent,
          prompt,
          {
            maxTurns: CONTEXT_COMPACTOR_TURNS_PER_ATTEMPT,
            ...(request.signal ? { signal: request.signal } : {})
          }
        );
        request.signal?.throwIfAborted();
        if (typeof result.finalOutput !== "string") {
          throw new Error("Context Compactor returned no checkpoint tool output");
        }
        let output: unknown;
        try {
          output = JSON.parse(result.finalOutput);
        } catch {
          throw new Error("Context Compactor did not return a valid checkpoint tool result");
        }
        return {
          summary: ContextCompactionSummarySchema.parse(output),
          origin: "model",
          usage
        };
      } catch (error) {
        request.signal?.throwIfAborted();
        lastFailure = errorMessage(error).slice(0, 600);
      }
    }

    throw new ContextCompactionInterruption(
      lastFailure ?? "Context Compactor returned no valid checkpoint",
      { usage }
    );
  }
}

export class ContextCompactionInterruption extends Error {
  readonly code = "context_compaction_interrupted";
  readonly usage: ContextSummaryUsage | undefined;

  constructor(
    detail: string,
    options: { cause?: unknown; usage?: ContextSummaryUsage } = {}
  ) {
    super(`Context compaction interruption: ${detail}`, {
      ...(options.cause === undefined ? {} : { cause: options.cause })
    });
    this.name = "ContextCompactionInterruption";
    this.usage = options.usage ? { ...options.usage } : undefined;
  }
}

export function isContextCompactionInterruption(error: unknown): boolean {
  const pending: unknown[] = [error];
  const seen = new Set<unknown>();
  while (pending.length > 0) {
    const current = pending.shift();
    if (current === null || (typeof current !== "object" && typeof current !== "function")) {
      continue;
    }
    if (seen.has(current)) continue;
    seen.add(current);
    if (current instanceof ContextCompactionInterruption) return true;
    const record = current as Record<string, unknown>;
    if (record.code === "context_compaction_interrupted"
      || record.name === "ContextCompactionInterruption") return true;
    if (record.cause !== undefined) pending.push(record.cause);
    if (Array.isArray(record.errors)) pending.push(...record.errors);
  }
  return false;
}

function compactorAgent(input: {
  model: Model;
  temperature: number;
  maxOutputTokens: number;
  accepted: string[];
  blockers: string[];
}): Agent {
  const commit = contextCheckpointTool(input.accepted, input.blockers);
  return new Agent({
    name: "Context Compactor",
    instructions: COMPACTOR_INSTRUCTIONS,
    model: input.model,
    modelSettings: {
      temperature: input.temperature,
      maxTokens: input.maxOutputTokens,
      parallelToolCalls: false,
      // This agent exposes exactly one terminal tool. Naming it gives the
      // provider a stronger, SDK-native constraint than generic "required".
      toolChoice: COMMIT_CONTEXT_TOOL
    },
    tools: [commit],
    resetToolChoice: false,
    toolUseBehavior: (_context, toolResults) => {
      const completion = toolResults.find((result) =>
        result.type === "function_output"
          && result.tool.name === COMMIT_CONTEXT_TOOL
          && validCheckpointOutput(result.output)
      );
      if (!completion || completion.type !== "function_output") {
        return { isFinalOutput: false, isInterrupted: undefined };
      }
      return {
        isFinalOutput: true,
        isInterrupted: undefined,
        finalOutput: typeof completion.output === "string"
          ? completion.output
          : JSON.stringify(completion.output)
      };
    }
  });
}

function withUsageCounter(
  model: Model,
  usage: ContextSummaryUsage,
  onModelResponseCompleted?: () => void | Promise<void>
): Model {
  const addUsage = (value: { inputTokens: number; outputTokens: number; totalTokens: number }) => {
    usage.inputTokens += value.inputTokens;
    usage.outputTokens += value.outputTokens;
    usage.totalTokens += value.totalTokens;
  };
  return {
    getResponse: async (request) => {
      usage.requests += 1;
      const response = await model.getResponse(request);
      addUsage(response.usage);
      await onModelResponseCompleted?.();
      return response;
    },
    getStreamedResponse: (request) => {
      usage.requests += 1;
      const stream = model.getStreamedResponse(request);
      return (async function* () {
        for await (const event of stream) {
          if (event.type === "response_done") {
            addUsage(event.response.usage);
            await onModelResponseCompleted?.();
          }
          yield event;
        }
      })();
    },
    ...(model.getRetryAdvice
      ? { getRetryAdvice: (request) => model.getRetryAdvice!(request) }
      : {})
  };
}

/** Worst-case bounded attempt input, including retry diagnostics and dynamic schema. */
export function estimateContextSummaryRequestTokens(request: ContextSummaryRequest): number {
  const accepted = [...new Set(request.acceptedTransactionIds)];
  const blockers = [...new Set(request.blockerTransactionIds)];
  const boundedRequest = {
    ...request,
    acceptedTransactionIds: accepted,
    blockerTransactionIds: blockers
  };
  const first = estimateContextSummaryPromptTokens(
    compactionPrompt(boundedRequest),
    accepted,
    blockers
  );
  const retry = estimateContextSummaryPromptTokens(
    compactionPrompt(
      boundedRequest,
      CONTEXT_COMPACTOR_MAX_ATTEMPTS,
      "界".repeat(600)
    ),
    accepted,
    blockers
  );
  return Math.max(first, retry);
}

function estimateContextSummaryPromptTokens(
  prompt: string,
  accepted: string[],
  blockers: string[]
): number {
  const structuralEstimate = estimateModelInputTokens({
    input: [{ role: "user", content: prompt }],
    instructions: COMPACTOR_INSTRUCTIONS
  }) + estimateToolTokens([contextCheckpointTool(accepted, blockers)]);
  // Live usage includes provider/SDK framing that is not present in the
  // ModelInputData object. Keep a proportional margin plus a fixed envelope so
  // a batch near the boundary cannot cross it due to transport serialization.
  return Math.ceil(structuralEstimate * 1.1) + 256;
}

function compactionPrompt(
  request: ContextSummaryRequest,
  attempt = 1,
  lastFailure?: string
): string {
  const formalItems = request.sourceItems.filter((item) =>
    !("type" in item && item.type === "reasoning")
  );
  const omittedReasoning = request.sourceItems.length - formalItems.length;
  return [
    "Build the next long-run context checkpoint from these inputs.",
    `ATTEMPT ${attempt}: call ${COMMIT_CONTEXT_TOOL} now. Do not discuss missing evidence. `
      + "If no allowlisted receipt proves an evidence entry, omit that entry.",
    ...(lastFailure ? [`PREVIOUS FRESH ATTEMPT FAILED: ${lastFailure}`] : []),
    `CURRENT AUTHORITY:\n${JSON.stringify(request.authority)}`,
    `ACCEPTED RECEIPT TRANSACTION IDS FOR COMPLETED:\n${JSON.stringify(request.acceptedTransactionIds)}`,
    `RECEIPT TRANSACTION IDS FOR BLOCKERS:\n${JSON.stringify(request.blockerTransactionIds)}`,
    `PRIOR MODEL CHECKPOINT:\n${request.priorSummary ? JSON.stringify(request.priorSummary) : "null"}`,
    `OMITTED NON-ACTION REASONING ITEMS:\n${omittedReasoning}`,
    `NEW FORMAL HISTORICAL ITEMS:\n${JSON.stringify(formalItems)}`
  ].join("\n\n");
}

/**
 * Rebase model-written memory onto a new world identity without replacing its
 * durable semantics. Constraints, unfinished goals, and revision-independent
 * decisions survive. Receipt claims are retained only when the caller proves
 * them current, while concrete action arguments are discarded because plans,
 * poses, and coordinates are revision-bound capabilities rather than memory.
 */
export function rebaseContextSummary(input: {
  summary: ContextCompactionSummary;
  acceptedTransactionIds: string[];
  blockerTransactionIds: string[];
}): ContextCompactionSummary {
  return {
    mission_state: input.summary.mission_state,
    constraints: [...input.summary.constraints],
    decisions: input.summary.decisions.filter((decision) =>
      !isConcreteRevisionBoundAction(decision, false)
    ),
    completed: retainedEvidence(input.summary.completed, input.acceptedTransactionIds),
    pending: input.summary.pending.filter((item) =>
      !isConcreteRevisionBoundAction(item, true)
    ),
    blockers: retainedEvidence(input.summary.blockers, input.blockerTransactionIds),
    next_actions: input.summary.next_actions.filter((action) =>
      !isConcreteRevisionBoundAction(action, true)
    )
  };
}

const ACTION_LANGUAGE = /\b(?:call|execute|follow|use|move|navigate|drive|turn|rotate|set|place|break|grasp|pick|drop|reach|inspect|scan|survey|plan)\b|(?:调用|执行|沿用|使用|移动|前往|导航|驾驶|转向|旋转|设置|放置|破坏|抓取|拾取|释放|到达|检查|扫描|规划)/iu;
const PLAN_OR_ARGUMENT = /\b(?:plan[_\s-]?id|planning[_\s-]?transaction[_\s-]?id|transaction[_\s-]?id|call[_\s-]?id|tool[_\s-]?arguments?|arguments?|target|face[_\s-]?point)\b\s*(?::|=|#|\bis\b)|(?:计划|工具)(?:编号|参数)\s*(?::|=|为)/iu;
const COORDINATE_TUPLE = /(?:\(|\[)\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*,\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*,\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*(?:\)|\])/u;
const NAMED_COORDINATE = /(?:\b[xyz]\b\s*[:=]\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+).*){2,}|(?:坐标|位置)\s*(?::|=|为)?\s*(?:\(|\[)?\s*[+-]?\d/iu;
const TOOL_CALL_ARGUMENTS = /\b[a-z][a-z0-9_]*\s*\(\s*(?:\{|[a-z][a-z0-9_]*\s*(?::|=))/iu;

function isConcreteRevisionBoundAction(value: string, actionField: boolean): boolean {
  const concrete = PLAN_OR_ARGUMENT.test(value)
    || COORDINATE_TUPLE.test(value)
    || NAMED_COORDINATE.test(value)
    || TOOL_CALL_ARGUMENTS.test(value);
  return concrete && (actionField || ACTION_LANGUAGE.test(value));
}

function retainedEvidence(
  evidence: ContextCompactionSummary["completed"] | undefined,
  allowedTransactionIds: string[]
): ContextCompactionSummary["completed"] {
  const allowed = new Set(allowedTransactionIds);
  return (evidence ?? []).flatMap((item) => {
    const transactionIds = item.transaction_ids.filter((transactionId) => allowed.has(transactionId));
    return transactionIds.length > 0
      ? [{ summary: item.summary, transaction_ids: transactionIds }]
      : [];
  });
}

function summarySchema(
  acceptedTransactionIds: string[],
  blockerTransactionIds: string[]
) {
  const acceptedIds = acceptedTransactionIds.length > 0
    ? z.array(z.enum(acceptedTransactionIds as [string, ...string[]])).min(1)
    : z.array(z.string()).length(0);
  const blockerIds = blockerTransactionIds.length > 0
    ? z.array(z.enum(blockerTransactionIds as [string, ...string[]])).min(1)
    : z.array(z.string()).length(0);
  const completedEvidence = z.object({
    summary: z.string().trim().min(1),
    transaction_ids: acceptedIds
  }).strict();
  const blockerEvidence = z.object({
    summary: z.string().trim().min(1),
    transaction_ids: blockerIds
  }).strict();
  return z.object({
    mission_state: z.string().trim().min(1),
    constraints: z.array(z.string().trim().min(1)),
    decisions: z.array(z.string().trim().min(1)),
    completed: z.array(completedEvidence),
    pending: z.array(z.string().trim().min(1)),
    blockers: z.array(blockerEvidence),
    next_actions: z.array(z.string().trim().min(1))
  }).strict();
}

function contextCheckpointTool(
  acceptedTransactionIds: string[],
  blockerTransactionIds: string[]
) {
  return tool({
    name: COMMIT_CONTEXT_TOOL,
    description: COMMIT_CONTEXT_DESCRIPTION,
    parameters: summarySchema(acceptedTransactionIds, blockerTransactionIds),
    strict: true,
    execute: async (summary) => JSON.stringify(ContextCompactionSummarySchema.parse(summary))
  });
}

function validCheckpointOutput(value: unknown): boolean {
  if (typeof value !== "string") {
    return ContextCompactionSummarySchema.safeParse(value).success;
  }
  try {
    return ContextCompactionSummarySchema.safeParse(JSON.parse(value)).success;
  } catch {
    return false;
  }
}
