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
        return functionCallResponse(request.tools[0]?.name ?? "observe_humanoid");
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
    const request = modelRequest("auto", ["observe_humanoid", "plan_humanoid_skill"]);

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
        return streamedFunctionCall(request.tools[0]?.name ?? "observe_humanoid");
      })
    };
    const recovery = new ModelDecisionProtocolRecovery(() => "authority-a");
    recovery.requireToolDecision("humanoid-motion-reference");
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-motion-reference",
      recovery
    );

    for await (const _event of wrapped.getStreamedResponse(modelRequest(
      "none",
      ["observe_humanoid", "plan_humanoid_skill"]
    ))) {
      // No events are needed to inspect the forwarded request.
    }
    expect(requests[0]?.modelSettings.toolChoice).toBe("required");
  });

  it("uses the only visible function as a named protocol choice before any stall", async () => {
    const requests: ModelRequest[] = [];
    const model: Model = {
      getResponse: vi.fn(async (request) => {
        requests.push(request);
        return functionCallResponse(request.tools[0]?.name ?? "delegate_physics_executor");
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
        return functionCallResponse("delegate_humanoid_sentry");
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
        return functionCallResponse("complete_autonomous_cycle");
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
            yield* streamedFunctionCall("observe_humanoid");
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

    for await (const _event of wrapped.getStreamedResponse(modelRequest(
      "auto",
      ["observe_humanoid", "plan_humanoid_skill"]
    ))) {
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
    expect(recovery.requiresToolDecision("humanoid-coordinator")).toBe(true);
  });

  it("retries a prose-only response in place without exposing it to the SDK", async () => {
    const requests: ModelRequest[] = [];
    const retries = vi.fn();
    const model: Model = {
      getResponse: vi.fn(),
      getStreamedResponse: vi.fn((request) => {
        requests.push(request);
        return requests.length === 1
          ? streamedText("我将调用运动参考智能体。")
          : streamedFunctionCall("delegate_motion_reference");
      })
    };
    const recovery = new ModelDecisionProtocolRecovery(() => "authority-a");
    recovery.rejectRequiredToolChoice("humanoid-coordinator");
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-coordinator",
      recovery,
      undefined,
      retries
    );
    const events = [];

    for await (const event of wrapped.getStreamedResponse(modelRequest(
      "auto",
      ["delegate_humanoid_sentry", "delegate_motion_reference"]
    ))) {
      events.push(event);
    }

    expect(requests).toHaveLength(2);
    expect(requests[1]?.modelSettings.toolChoice).toBe("auto");
    expect(JSON.stringify(requests[1]?.input)).toContain(
      "HARNESS NATIVE FUNCTION DECISION RECOVERY"
    );
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      type: "response_done",
      response: { output: [{ type: "function_call", name: "delegate_motion_reference" }] }
    });
    expect(retries).toHaveBeenCalledWith({
      agentId: "humanoid-coordinator",
      completedResponseCount: 1,
      availableToolNames: ["delegate_humanoid_sentry", "delegate_motion_reference"],
      constraint: "prompted_auto"
    });
  });

  it("keeps the logical decision requirement when native required is rejected", async () => {
    const requests: ModelRequest[] = [];
    const model: Model = {
      getResponse: vi.fn(async (request) => {
        requests.push(request);
        if (request.modelSettings.toolChoice === "required") {
          throw new Error("tool_choice required is not supported");
        }
        return requests.length === 2
          ? textResponse("准备调用工具")
          : functionCallResponse("plan_humanoid_skill");
      }),
      getStreamedResponse: vi.fn(() => ({ async *[Symbol.asyncIterator]() {} }))
    };
    const recovery = new ModelDecisionProtocolRecovery(() => "authority-a");
    recovery.requireToolDecision("humanoid-motion-reference");
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-motion-reference",
      recovery
    );

    await wrapped.getResponse(modelRequest(
      "auto",
      ["observe_humanoid", "plan_humanoid_skill"]
    ));

    expect(requests.map((request) => request.modelSettings.toolChoice)).toEqual([
      "required",
      "auto",
      "auto"
    ]);
    expect(JSON.stringify(requests[2]?.input)).toContain(
      "HARNESS NATIVE FUNCTION DECISION RECOVERY"
    );
    expect(recovery.requiresToolDecision("humanoid-motion-reference")).toBe(true);
  });

  it("narrows only a recovery request to the tools currently authorized by runtime", async () => {
    const requests: ModelRequest[] = [];
    const model: Model = {
      getResponse: vi.fn(async (request) => {
        requests.push(request);
        return requests.length === 1
          ? textResponse("先观察再规划")
          : functionCallResponse("observe_humanoid");
      }),
      getStreamedResponse: vi.fn(() => ({ async *[Symbol.asyncIterator]() {} }))
    };
    const recovery = new ModelDecisionProtocolRecovery(
      () => "authority-a",
      (_agentId, exposed) => exposed.filter((name) => (
        name === "observe_humanoid" || name === "recall_embodied_history"
      ))
    );
    recovery.rejectRequiredToolChoice("humanoid-motion-reference");
    const wrapped = withModelDecisionProtocolRecovery(
      model,
      "humanoid-motion-reference",
      recovery
    );

    await wrapped.getResponse(modelRequest("auto", [
      "observe_humanoid",
      "recall_embodied_history",
      "submit_humanoid_skill_plan",
      "begin_humanoid_skill"
    ]));

    expect(requests[0]?.tools.map((tool) => tool.name)).toEqual([
      "observe_humanoid",
      "recall_embodied_history",
      "submit_humanoid_skill_plan",
      "begin_humanoid_skill"
    ]);
    expect(requests[1]?.tools.map((tool) => tool.name)).toEqual([
      "observe_humanoid",
      "recall_embodied_history"
    ]);
  });
});

function functionCallResponse(name: string) {
  return {
    usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
    output: [{
      type: "function_call",
      callId: `call-${name}`,
      name,
      arguments: "{}"
    }]
  } as never;
}

function textResponse(text: string) {
  return {
    usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
    output: [{ type: "message", role: "assistant", content: text }]
  } as never;
}

function streamedFunctionCall(name: string) {
  return {
    async *[Symbol.asyncIterator]() {
      yield {
        type: "response_done",
        response: functionCallResponse(name)
      } as never;
    }
  };
}

function streamedText(text: string) {
  return {
    async *[Symbol.asyncIterator]() {
      yield {
        type: "response_done",
        response: textResponse(text)
      } as never;
    }
  };
}

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
