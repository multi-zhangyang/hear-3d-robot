import { Usage, type Model } from "@openai/agents";
import { describe, expect, it } from "vitest";
import {
  CONTEXT_COMPACTOR_MAX_ATTEMPTS,
  CONTEXT_COMPACTOR_TURNS_PER_ATTEMPT,
  compactorInputTokenLimit,
  effectiveContextSummaryOutputTokens
} from "../runtime/context-budget.js";
import {
  AgentsSdkContextSummaryGenerator,
  ContextCompactionCapacityError,
  ContextCompactionInterruption,
  isRetryableContextCompactionInterruption,
  estimateContextSummaryRequestTokens,
  isContextCompactionInterruption,
  rebaseContextSummary
} from "./context-summary-agent.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";

describe("context summary fidelity", () => {
  it("uses the tightest source, compactor, and model output ceiling", () => {
    expect(effectiveContextSummaryOutputTokens(90, 120, 150)).toBe(90);
    expect(effectiveContextSummaryOutputTokens(150, 80, 120)).toBe(80);
    expect(effectiveContextSummaryOutputTokens(150, 120, 70)).toBe(70);
  });

  it("budgets one compactor request rather than accumulating repair turns", () => {
    expect(compactorInputTokenLimit(65_536, 4_096)).toBe(61_440);
  });

  it("rebases durable model semantics while removing stale evidence and action arguments", () => {
    const rebased = rebaseContextSummary({
      summary: {
        mission_state: "The exploration objective remains unfinished.",
        constraints: [
          "Keep one worker responsible for mapping.",
          "Re-observe before physical execution."
        ],
        decisions: [
          "Use current authority before selecting any action.",
          "Execute plan_id=nav-old on the next turn."
        ],
        completed: [
          { summary: "Mixed old and current evidence", transaction_ids: ["old-ok", "current-ok"] },
          { summary: "Old evidence only", transaction_ids: ["old-only"] }
        ],
        pending: [
          "Map the remaining reachable frontier.",
          "Move to x=7, y=0, z=-3."
        ],
        blockers: [
          { summary: "Mixed blockers", transaction_ids: ["old-blocker", "current-blocker"] }
        ],
        next_actions: [
          "Re-observe terrain before choosing a route.",
          "Move to (4, 0, -2).",
          "Call execute_base_plan({\"plan_id\":\"nav-old\"})."
        ]
      },
      acceptedTransactionIds: ["current-ok"],
      blockerTransactionIds: ["current-blocker"]
    });

    expect(rebased).toEqual({
      mission_state: "The exploration objective remains unfinished.",
      constraints: [
        "Keep one worker responsible for mapping.",
        "Re-observe before physical execution."
      ],
      decisions: ["Use current authority before selecting any action."],
      completed: [{
        summary: "Mixed old and current evidence",
        transaction_ids: ["current-ok"]
      }],
      pending: ["Map the remaining reachable frontier."],
      blockers: [{
        summary: "Mixed blockers",
        transaction_ids: ["current-blocker"]
      }],
      next_actions: ["Re-observe terrain before choosing a route."]
    });
  });

  it("interrupts after bounded real-model attempts instead of returning a synthetic summary", async () => {
    let calls = 0;
    const outputLimits: Array<number | undefined> = [];
    const model = {
      getResponse: async (request: Parameters<Model["getResponse"]>[0]) => {
        calls += 1;
        outputLimits.push(request.modelSettings.maxTokens);
        throw new Error("checkpoint tool omitted");
      },
      getStreamedResponse: () => {
        throw new Error("streaming is not used by this compactor test");
      }
    } as unknown as Model;
    const generator = new AgentsSdkContextSummaryGenerator({
      model,
      temperature: 0
    });

    const failure = await generator.generate({
      priorSummary: null,
      sourceItems: [{ role: "user", content: "Preserve this unfinished objective." }],
      authority: { world_revision: 3 },
      acceptedTransactionIds: [],
      blockerTransactionIds: [],
      maxInputTokens: 16_000,
      maxOutputTokens: 256
    }).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(ContextCompactionInterruption);
    expect(failure).toMatchObject({
      code: "context_compaction_interrupted",
      usage: { requests: CONTEXT_COMPACTOR_MAX_ATTEMPTS }
    });
    expect(calls).toBe(CONTEXT_COMPACTOR_MAX_ATTEMPTS);
    expect(outputLimits).toEqual(
      Array.from({ length: CONTEXT_COMPACTOR_MAX_ATTEMPTS }, () => undefined)
    );
    expect(isContextCompactionInterruption(new Error("wrapper", { cause: failure }))).toBe(true);
  });

  it("returns a transport interruption to the owning runtime after one request", async () => {
    const connectionFailure = Object.assign(new Error("upstream connection reset"), {
      code: "ECONNRESET"
    });
    let calls = 0;
    const model = {
      getResponse: async () => {
        calls += 1;
        throw connectionFailure;
      },
      getStreamedResponse: () => {
        throw new Error("streaming is not used by this compactor test");
      }
    } as unknown as Model;
    const generator = new AgentsSdkContextSummaryGenerator({
      model,
      temperature: 0
    });

    const failure = await generator.generate({
      priorSummary: null,
      sourceItems: [{ role: "user", content: "Preserve the active objective." }],
      authority: { world_revision: 4 },
      acceptedTransactionIds: [],
      blockerTransactionIds: [],
      maxInputTokens: 16_000,
      maxOutputTokens: 256
    }).catch((error: unknown) => error);

    expect(failure).toMatchObject({
      code: "context_compaction_interrupted",
      cause: connectionFailure,
      usage: { requests: 1 }
    });
    expect(isTransportInterruption(failure)).toBe(true);
    expect(isRetryableContextCompactionInterruption(failure)).toBe(true);
    expect(calls).toBe(1);
  });

  it("does not send a compactor request beyond the retry-safe input envelope", async () => {
    let calls = 0;
    const model = {
      getResponse: async () => {
        calls += 1;
        throw new Error("must not be called");
      },
      getStreamedResponse: () => {
        throw new Error("must not be called");
      }
    } as unknown as Model;
    const generator = new AgentsSdkContextSummaryGenerator({
      model,
      temperature: 0
    });
    const request = {
      priorSummary: null,
      sourceItems: [{ role: "user" as const, content: "Keep the raw history intact." }],
      authority: { world_revision: 9 },
      acceptedTransactionIds: [],
      blockerTransactionIds: [],
      maxInputTokens: Number.MAX_SAFE_INTEGER,
      maxOutputTokens: 256
    };
    const retrySafeEstimate = estimateContextSummaryRequestTokens(request);

    const failure = await generator.generate({
      ...request,
      maxInputTokens: retrySafeEstimate - 1
    }).catch((error: unknown) => error);

    expect(failure).toMatchObject({
      code: "context_compaction_interrupted",
      usage: { requests: 0 }
    });
    expect(calls).toBe(0);
  });

  it("preserves run-level cache details in compactor usage", async () => {
    const summary = {
      mission_state: "Continue the current task.",
      constraints: [],
      decisions: [],
      completed: [],
      pending: ["Inspect the latest state."],
      blockers: [],
      next_actions: ["Continue planning."]
    };
    const model = {
      getResponse: async () => ({
        responseId: "response-compactor-cache",
        output: [{
          type: "function_call",
          callId: "call-compactor-cache",
          name: "commit_context_checkpoint",
          arguments: JSON.stringify(summary)
        }],
        usage: new Usage({
          requests: 1,
          inputTokens: 200,
          outputTokens: 40,
          totalTokens: 240,
          inputTokensDetails: { cached_tokens: 160 }
        })
      }),
      getStreamedResponse: () => {
        throw new Error("streaming is not used by this compactor test");
      }
    } as unknown as Model;
    const generator = new AgentsSdkContextSummaryGenerator({
      model,
      temperature: 0
    });

    const result = await generator.generate({
      priorSummary: null,
      sourceItems: [{ role: "user", content: "Preserve the active task." }],
      authority: { world_revision: 3 },
      acceptedTransactionIds: [],
      blockerTransactionIds: [],
      maxInputTokens: 16_000,
      maxOutputTokens: 256
    });

    expect(result).toMatchObject({
      origin: "model",
      summary,
      usage: {
        requests: 1,
        inputTokens: 200,
        outputTokens: 40,
        totalTokens: 240,
        inputTokensDetails: [{ cached_tokens: 160 }]
      }
    });
  });

  it("filters unproved receipt claims from a real model checkpoint at the tool boundary", async () => {
    const modelSummary = {
      mission_state: "Continue the unfinished task.",
      constraints: ["Use current authority."],
      decisions: [],
      completed: [{
        summary: "One accepted action and one invented action.",
        transaction_ids: ["accepted-action", "invented-action"]
      }],
      pending: ["Finish the current goal."],
      blockers: [{
        summary: "A rejected action remains relevant.",
        transaction_ids: ["rejected-action", "invented-blocker"]
      }],
      next_actions: ["Observe the current state."]
    };
    const model = {
      getResponse: async () => ({
        responseId: "response-filtered-compactor",
        output: [{
          type: "function_call",
          callId: "call-filtered-compactor",
          name: "commit_context_checkpoint",
          arguments: JSON.stringify(modelSummary)
        }],
        usage: new Usage({
          requests: 1,
          inputTokens: 200,
          outputTokens: 40,
          totalTokens: 240
        })
      }),
      getStreamedResponse: () => {
        throw new Error("streaming is not used by this compactor test");
      }
    } as unknown as Model;
    const generator = new AgentsSdkContextSummaryGenerator({
      model,
      temperature: 0
    });

    const result = await generator.generate({
      priorSummary: null,
      sourceItems: [{ role: "user", content: "Preserve verified evidence only." }],
      authority: { world_revision: 3 },
      acceptedTransactionIds: ["accepted-action"],
      blockerTransactionIds: ["accepted-action", "rejected-action"],
      maxInputTokens: 16_000,
      maxOutputTokens: 256
    });

    expect(result.summary.completed).toEqual([{
      summary: "One accepted action and one invented action.",
      transaction_ids: ["accepted-action"]
    }]);
    expect(result.summary.blockers).toEqual([{
      summary: "A rejected action remains relevant.",
      transaction_ids: ["rejected-action"]
    }]);
  });

  it("interrupts on a schema-invalid checkpoint without starting another request", async () => {
    const summary = {
      mission_state: "Continue the task.",
      constraints: [],
      decisions: [],
      completed: [],
      pending: ["Finish the active goal."],
      blockers: [],
      next_actions: ["Inspect current authority."]
    };
    let calls = 0;
    const requests: string[] = [];
    const model = {
      getResponse: async (request: Parameters<Model["getResponse"]>[0]) => {
        calls += 1;
        requests.push(JSON.stringify(request.input));
        return {
          responseId: `response-${calls}`,
          output: [{
            type: "function_call",
            callId: `call-${calls}`,
            name: "commit_context_checkpoint",
            arguments: JSON.stringify(calls <= CONTEXT_COMPACTOR_TURNS_PER_ATTEMPT
              ? { mission_state: "Missing the required arrays." }
              : summary)
          }],
          usage: new Usage({
            requests: 1,
            inputTokens: 100,
            outputTokens: 20,
            totalTokens: 120
          })
        };
      },
      getStreamedResponse: () => {
        throw new Error("streaming is not used by this compactor test");
      }
    } as unknown as Model;
    const generator = new AgentsSdkContextSummaryGenerator({
      model,
      temperature: 0
    });

    const failure = await generator.generate({
      priorSummary: null,
      sourceItems: [{ role: "user", content: "Preserve the unfinished goal." }],
      authority: { world_revision: 12 },
      acceptedTransactionIds: [],
      blockerTransactionIds: [],
      maxInputTokens: 16_000,
      maxOutputTokens: 256
    }).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(ContextCompactionInterruption);
    expect(calls).toBe(1);
    expect(requests).toHaveLength(1);
  });

  it("surfaces an observed provider context ceiling after a length-truncated tool call", async () => {
    let calls = 0;
    const model = {
      getResponse: async () => {
        calls += 1;
        return {
          responseId: "response-truncated-compactor",
          output: [{
            type: "function_call",
            callId: "call-truncated-compactor",
            name: "commit_context_checkpoint",
            arguments: '{"mission_state":"unfinished'
          }],
          usage: new Usage({
            requests: 1,
            inputTokens: 3_900,
            outputTokens: 196,
            totalTokens: 4_096
          }),
          providerData: { finishReason: "length" }
        };
      },
      getStreamedResponse: () => {
        throw new Error("streaming is not used by this compactor test");
      }
    } as unknown as Model;
    const generator = new AgentsSdkContextSummaryGenerator({
      model,
      temperature: 0
    });

    const failure = await generator.generate({
      priorSummary: null,
      sourceItems: [{ role: "user", content: "Preserve this history." }],
      authority: { world_revision: 4 },
      acceptedTransactionIds: [],
      blockerTransactionIds: [],
      maxInputTokens: 8_000,
      maxOutputTokens: 256
    }).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(ContextCompactionCapacityError);
    expect(failure).toMatchObject({
      observedContextWindowTokens: 4_096,
      usage: { requests: 1, totalTokens: 4_096 }
    });
    expect(calls).toBe(1);
  });
});
