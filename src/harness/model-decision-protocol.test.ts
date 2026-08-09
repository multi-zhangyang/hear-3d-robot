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

  it("uses the only visible function as a named protocol choice before any stall", async () => {
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
    const recovery = new ModelDecisionProtocolRecovery(() => "authority-a");
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-executor",
      recovery
    );
    const request = modelRequest("auto", ["delegate_physics_executor"]);

    await wrapped.getResponse(request);

    expect(requests[0]?.modelSettings.toolChoice).toBe("delegate_physics_executor");
    expect(request.modelSettings.toolChoice).toBe("auto");
  });

  it("negotiates an unsupported named choice once and remembers configured mode", async () => {
    const requests: ModelRequest[] = [];
    const unsupported = vi.fn();
    const model: Model = {
      getResponse: vi.fn(async (request) => {
        requests.push(request);
        if (request.modelSettings.toolChoice !== "auto") {
          throw new Error("Named tool_choice is not supported by this endpoint");
        }
        return {
          usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          output: []
        };
      }),
      getStreamedResponse: vi.fn(() => ({
        async *[Symbol.asyncIterator]() {}
      }))
    };
    const recovery = new ModelDecisionProtocolRecovery(() => "authority-a");
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-sentry",
      recovery,
      unsupported
    );
    const request = modelRequest("auto", ["delegate_humanoid_sentry"]);

    await wrapped.getResponse(request);
    await wrapped.getResponse(request);

    expect(requests.map((entry) => entry.modelSettings.toolChoice)).toEqual([
      "delegate_humanoid_sentry",
      "auto",
      "auto"
    ]);
    expect(unsupported).toHaveBeenCalledTimes(1);
    expect(unsupported).toHaveBeenCalledWith({
      agentId: "humanoid-sentry",
      mode: "named",
      toolName: "delegate_humanoid_sentry"
    });
  });

  it("falls through named and required constraints before configured auto", async () => {
    const requests: ModelRequest[] = [];
    const unsupported = vi.fn();
    const model: Model = {
      getResponse: vi.fn(async (request) => {
        requests.push(request);
        if (request.modelSettings.toolChoice !== "auto") {
          throw new Error("tool_choice is unsupported");
        }
        return {
          usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          output: []
        };
      }),
      getStreamedResponse: vi.fn(() => ({
        async *[Symbol.asyncIterator]() {}
      }))
    };
    const recovery = new ModelDecisionProtocolRecovery(() => "authority-a");
    recovery.requireToolDecision("humanoid-coordinator");
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-coordinator",
      recovery,
      unsupported
    );

    await wrapped.getResponse(modelRequest("auto", ["complete_autonomous_cycle"]));

    expect(requests.map((entry) => entry.modelSettings.toolChoice)).toEqual([
      "complete_autonomous_cycle",
      "required",
      "auto"
    ]);
    expect(unsupported.mock.calls).toEqual([
      [{
        agentId: "humanoid-coordinator",
        mode: "named",
        toolName: "complete_autonomous_cycle"
      }],
      [{ agentId: "humanoid-coordinator", mode: "required" }]
    ]);
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
    expect(unsupported).toHaveBeenCalledWith({
      agentId: "humanoid-coordinator",
      mode: "required"
    });
    expect(recovery.requireToolDecision("humanoid-coordinator")).toBe(false);
  });
});

function modelRequest(
  toolChoice: "auto" | "none",
  toolNames: readonly string[] = []
): ModelRequest {
  return {
    input: "continue",
    modelSettings: { toolChoice },
    tools: toolNames.map((name) => ({
      type: "function",
      name,
      description: `${name} test tool`,
      parameters: { type: "object", properties: {}, additionalProperties: false },
      strict: true
    })),
    outputType: "text",
    handoffs: [],
    tracing: false
  };
}
