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
  fallbackReason?: string;
  usage: ContextSummaryUsage;
}

export interface ContextSummaryGenerator {
  generate(request: ContextSummaryRequest): Promise<ContextSummaryResult>;
}

/**
 * A real Agents SDK model run that turns historical model/tool items into a
 * typed checkpoint. Prose-only failures are retried in fresh bounded turns so
 * one bad branch cannot recursively consume the provider window. If every
 * model attempt fails, the safety projection retains only current authority
 * and receipt-backed evidence; it never selects or performs an action.
 */
export class AgentsSdkContextSummaryGenerator implements ContextSummaryGenerator {
  readonly #runner: Runner;
  readonly #model: Model;
  readonly #temperature: number;
  readonly #maxOutputTokens: number;

  constructor(input: {
    model: Model;
    temperature: number;
    maxOutputTokens: number;
  }) {
    this.#model = input.model;
    this.#temperature = input.temperature;
    this.#maxOutputTokens = input.maxOutputTokens;
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
    const estimatedInputTokens = estimateContextSummaryRequestTokens(boundedRequest);
    if (estimatedInputTokens > request.maxInputTokens) {
      throw new Error(
        `Context compactor request is estimated at ${estimatedInputTokens} input tokens, `
        + `above its ${request.maxInputTokens}-token hard limit`
      );
    }
    const usage: ContextSummaryUsage = {
      requests: 0,
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0
    };
    const countedModel = withUsageCounter(this.#model, usage);
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
      try {
        // Each attempt starts from the same bounded evidence and no Session.
        // Invalid reasoning from a prior attempt is intentionally not replayed.
        const result = await this.#runner.run(
          agent,
          compactionPrompt(boundedRequest, attempt, lastFailure),
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

    return {
      summary: authorityProjectionSummary(boundedRequest),
      origin: "authority_projection",
      fallbackReason: lastFailure ?? "Context Compactor returned no valid checkpoint",
      usage
    };
  }
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

function withUsageCounter(model: Model, usage: ContextSummaryUsage): Model {
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
      return response;
    },
    getStreamedResponse: (request) => {
      usage.requests += 1;
      const stream = model.getStreamedResponse(request);
      return (async function* () {
        for await (const event of stream) {
          if (event.type === "response_done") addUsage(event.response.usage);
          yield event;
        }
      })();
    },
    ...(model.getRetryAdvice
      ? { getRetryAdvice: (request) => model.getRetryAdvice!(request) }
      : {})
  };
}

/** Estimated input for the first compactor turn, including its dynamic schema. */
export function estimateContextSummaryRequestTokens(request: ContextSummaryRequest): number {
  const accepted = [...new Set(request.acceptedTransactionIds)];
  const blockers = [...new Set(request.blockerTransactionIds)];
  const prompt = compactionPrompt({
    ...request,
    acceptedTransactionIds: accepted,
    blockerTransactionIds: blockers
  });
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
 * Last-resort continuity made only from current harness authority and receipt
 * allowlists. It intentionally drops free-form decisions and never suggests a
 * concrete movement, pose, or tool argument.
 */
export function authorityProjectionSummary(
  request: ContextSummaryRequest
): ContextCompactionSummary {
  const authority = asRecord(request.authority);
  const goal = asRecord(authority.goal);
  const goalState = asRecord(authority.goal_state);
  const goalSummary = typeof goal.summary === "string"
    ? goal.summary
    : "Continue the structured mission";
  const worldRevision = typeof authority.world_revision === "number"
    ? authority.world_revision
    : "unknown";
  const voxelRevision = typeof authority.voxel_revision === "number"
    ? authority.voxel_revision
    : "unknown";
  const checks = Array.isArray(goalState.checks) ? goalState.checks : [];
  const unmet = checks.flatMap((value) => {
    const check = asRecord(value);
    return check.passed === false && typeof check.name === "string" ? [check.name] : [];
  });
  const satisfied = goalState.satisfied === true;
  return {
    mission_state: `${satisfied ? "satisfied" : "incomplete"}: ${goalSummary}; `
      + `world_revision=${worldRevision}; voxel_revision=${voxelRevision}`,
    constraints: [
      "Current harness authority, current revisions, and committed receipts override this safety projection.",
      "Re-observe dynamic geometry before actuation; this projection contains no model-selected action."
    ],
    decisions: [],
    completed: retainedEvidence(request.priorSummary?.completed, request.acceptedTransactionIds),
    pending: satisfied
      ? []
      : unmet.length > 0
        ? unmet.map((name) => `Unmet structured goal check: ${name}`)
        : ["The current structured goal remains incomplete."],
    blockers: retainedEvidence(request.priorSummary?.blockers, request.blockerTransactionIds),
    next_actions: satisfied
      ? ["Let the active model verify the current goal through the formal checker."]
      : ["Let the active model choose the next formal tool from current authority and re-observe before actuation."]
  };
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

function asRecord(value: JsonValue | undefined): Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value
    : {};
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
