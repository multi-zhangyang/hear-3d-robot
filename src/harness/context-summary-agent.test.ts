import type { Model } from "@openai/agents";
import { describe, expect, it } from "vitest";
import { CONTEXT_COMPACTOR_MAX_ATTEMPTS } from "../runtime/context-budget.js";
import {
  AgentsSdkContextSummaryGenerator,
  ContextCompactionInterruption,
  estimateContextSummaryRequestTokens,
  isContextCompactionInterruption,
  rebaseContextSummary
} from "./context-summary-agent.js";

describe("context summary fidelity", () => {
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
    const model = {
      getResponse: async () => {
        calls += 1;
        throw new Error("checkpoint tool omitted");
      },
      getStreamedResponse: () => {
        throw new Error("streaming is not used by this compactor test");
      }
    } as unknown as Model;
    const generator = new AgentsSdkContextSummaryGenerator({
      model,
      temperature: 0,
      maxOutputTokens: 256
    });

    const failure = await generator.generate({
      priorSummary: null,
      sourceItems: [{ role: "user", content: "Preserve this unfinished objective." }],
      authority: { world_revision: 3 },
      acceptedTransactionIds: [],
      blockerTransactionIds: [],
      maxInputTokens: 16_000
    }).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(ContextCompactionInterruption);
    expect(failure).toMatchObject({
      code: "context_compaction_interrupted",
      usage: { requests: CONTEXT_COMPACTOR_MAX_ATTEMPTS }
    });
    expect(calls).toBe(CONTEXT_COMPACTOR_MAX_ATTEMPTS);
    expect(isContextCompactionInterruption(new Error("wrapper", { cause: failure }))).toBe(true);
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
      temperature: 0,
      maxOutputTokens: 256
    });
    const request = {
      priorSummary: null,
      sourceItems: [{ role: "user" as const, content: "Keep the raw history intact." }],
      authority: { world_revision: 9 },
      acceptedTransactionIds: [],
      blockerTransactionIds: [],
      maxInputTokens: Number.MAX_SAFE_INTEGER
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
});
