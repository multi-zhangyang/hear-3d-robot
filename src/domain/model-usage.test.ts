import { describe, expect, it } from "vitest";
import {
  EmptyModelUsageState,
  addModelUsage,
  modelUsageDeltaFromProviderEvent
} from "./model-usage.js";

describe("model usage", () => {
  it("normalizes AI SDK response and compactor usage without vendor logic", () => {
    const response = modelUsageDeltaFromProviderEvent({
      status: "usable_stream",
      usage: {
        inputTokens: 120,
        outputTokens: 30,
        totalTokens: 150,
        inputTokensDetails: { cached_tokens: 80 },
        outputTokensDetails: { reasoning_tokens: 12 }
      }
    }, "humanoid-motion-reference")!;
    const compactor = modelUsageDeltaFromProviderEvent({
      status: "context_compacted",
      agent_id: "humanoid-coordinator",
      usage: {
        requests: 2,
        inputTokens: 200,
        outputTokens: 40,
        totalTokens: 240
      }
    })!;
    const first = addModelUsage(EmptyModelUsageState, response, "2026-08-04T00:00:00.000Z");
    const second = addModelUsage(first, compactor, "2026-08-04T00:01:00.000Z");

    expect(second).toMatchObject({
      total: {
        requests: 3,
        reported_requests: 3,
        input_tokens: 320,
        output_tokens: 70,
        total_tokens: 390,
        cached_input_tokens: 80,
        reasoning_tokens: 12
      },
      by_agent: {
        "humanoid-motion-reference": { requests: 1, total_tokens: 150 },
        "humanoid-coordinator": { requests: 2, total_tokens: 240 }
      },
      updated_at: "2026-08-04T00:01:00.000Z"
    });
  });

  it("distinguishes an unreported usage request from a zero-token claim", () => {
    const delta = modelUsageDeltaFromProviderEvent({ usage: { requests: 1 } }, "agent")!;
    expect(delta.usage).toMatchObject({
      requests: 1,
      reported_requests: 0,
      total_tokens: 0
    });
    expect(modelUsageDeltaFromProviderEvent({ usage: { requests: 0 } }, "agent"))
      .toBeUndefined();
  });

  it("aggregates Agents SDK run-level token detail arrays", () => {
    const delta = modelUsageDeltaFromProviderEvent({
      usage: {
        requests: 2,
        inputTokens: 240,
        outputTokens: 50,
        totalTokens: 290,
        inputTokensDetails: [
          { cached_tokens: 80 },
          { cachedTokens: 120 }
        ],
        outputTokensDetails: [
          { reasoning_tokens: 9 },
          { reasoningTokens: 11 }
        ]
      }
    }, "agent")!;

    expect(delta.usage).toMatchObject({
      requests: 2,
      cached_input_tokens: 200,
      reasoning_tokens: 20
    });
  });
});
