import { describe, expect, it, vi } from "vitest";
import type { Model, ModelRequest } from "@openai/agents";
import {
  ModelDecisionProtocolRecovery,
  withModelDecisionProtocolRecovery
} from "./model-decision-protocol.js";

describe("model decision protocol recovery", () => {
  it("requires a tool on the same model facade until authority changes", async () => {
    let authority = "authority-a";
    const requests: ModelRequest[] = [];
    const model: Model = {
      getResponse: vi.fn(async (request) => {
        requests.push(request);
        return {
          usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          output: []
        };
      }),
      getStreamedResponse: vi.fn(() => ({
        async *[Symbol.asyncIterator]() {}
      }))
    };
    const recovery = new ModelDecisionProtocolRecovery(() => authority);
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-coordinator",
      recovery
    );
    const request = modelRequest("auto");

    await wrapped.getResponse(request);
    expect(requests.at(-1)?.modelSettings.toolChoice).toBe("auto");

    expect(recovery.requireToolDecision("humanoid-coordinator")).toBe(true);
    expect(recovery.requireToolDecision("humanoid-coordinator")).toBe(false);
    await wrapped.getResponse(request);
    expect(requests.at(-1)?.modelSettings.toolChoice).toBe("required");
    expect(request.modelSettings.toolChoice).toBe("auto");

    authority = "authority-b";
    await wrapped.getResponse(request);
    expect(requests.at(-1)?.modelSettings.toolChoice).toBe("auto");
  });

  it("applies the same recovery contract to streamed calls", async () => {
    const requests: ModelRequest[] = [];
    const model: Model = {
      getResponse: vi.fn(),
      getStreamedResponse: vi.fn((request) => {
        requests.push(request);
        return { async *[Symbol.asyncIterator]() {} };
      })
    };
    const recovery = new ModelDecisionProtocolRecovery(() => "authority-a");
    recovery.requireToolDecision("humanoid-motion-reference");
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-motion-reference",
      recovery
    );

    for await (const _event of wrapped.getStreamedResponse(modelRequest("none"))) {
      // No events are needed to inspect the forwarded request.
    }
    expect(requests[0]?.modelSettings.toolChoice).toBe("required");
  });

  it("negotiates back to the configured protocol when required is unsupported", async () => {
    const requests: ModelRequest[] = [];
    const unsupported = vi.fn();
    const model: Model = {
      getResponse: vi.fn(),
      getStreamedResponse: vi.fn((request) => {
        requests.push(request);
        return {
          async *[Symbol.asyncIterator]() {
            if (request.modelSettings.toolChoice === "required") {
              throw new Error("Thinking mode does not support this tool_choice");
            }
          }
        };
      })
    };
    const recovery = new ModelDecisionProtocolRecovery(() => "authority-a");
    recovery.requireToolDecision("humanoid-coordinator");
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-coordinator",
      recovery,
      unsupported
    );

    for await (const _event of wrapped.getStreamedResponse(modelRequest("auto"))) {
      // No events are needed to inspect protocol negotiation.
    }
    expect(requests.map((request) => request.modelSettings.toolChoice)).toEqual([
      "required",
      "auto"
    ]);
    expect(unsupported).toHaveBeenCalledWith("humanoid-coordinator");
    expect(recovery.requireToolDecision("humanoid-coordinator")).toBe(false);
  });
});

function modelRequest(toolChoice: "auto" | "none"): ModelRequest {
  return {
    input: "continue",
    modelSettings: { toolChoice },
    tools: [],
    outputType: "text"
  };
}
