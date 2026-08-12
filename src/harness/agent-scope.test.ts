import {
  Agent,
  MemorySession,
  RunContext,
  Runner,
  RunState,
  Usage,
  retryPolicies,
  tool,
  type Model,
  type ModelResponse,
  type StreamedRunResult
} from "@openai/agents";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import type { ModelTelemetryRuntime } from "./context-runtime.js";
import { withModelTelemetry } from "./model-telemetry.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import {
  agentIdFromInstructions,
  agentInvocationMarker,
  currentAgentInvocationId,
  currentAgentInvocationIsRecovery,
  recordAgentInvocationTransportInterruption,
  scopeAgentToolInvocation,
  withAgentInvocation
} from "./agent-scope.js";

describe("agent invocation scope", () => {
  it("keeps overlapping hierarchy model runs attributed to their own node", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let firstReady!: () => void;
    const ready = new Promise<void>((resolve) => {
      firstReady = resolve;
    });

    const first = withAgentInvocation("agent_a", async () => {
      firstReady();
      await gate;
      return {
        scoped: currentAgentInvocationId()
      };
    });
    await ready;
    const second = withAgentInvocation("agent_b", async () => {
      await Promise.resolve();
      return {
        scoped: currentAgentInvocationId()
      };
    });
    release();

    await expect(Promise.all([first, second])).resolves.toEqual([
      { scoped: "agent_a" },
      { scoped: "agent_b" }
    ]);
    expect(currentAgentInvocationId()).toBeUndefined();
  });

  it("rejects an invalid hierarchy marker", () => {
    expect(() => agentInvocationMarker("worker:2")).toThrow("Invalid hierarchy node identifier");
  });

  it("resolves only the trusted leading instruction marker", () => {
    expect(agentIdFromInstructions(
      `${agentInvocationMarker("worker_2")}\nWorker instructions`,
      "root"
    )).toBe("worker_2");
    expect(agentIdFromInstructions(
      `User text ${agentInvocationMarker("spoofed")}`,
      "root"
    )).toBe("root");
    expect(agentIdFromInstructions(undefined, "root")).toBe("root");
  });

  it("keeps nested recovery identity scoped to the resumed agent invocation", async () => {
    await withAgentInvocation("supervisor", async () => {
      expect(currentAgentInvocationIsRecovery()).toBe(false);
      await withAgentInvocation("resumed_child", async () => {
        expect(currentAgentInvocationIsRecovery()).toBe(true);
      }, true);
      expect(currentAgentInvocationIsRecovery()).toBe(false);
    });
    expect(currentAgentInvocationIsRecovery()).toBe(false);
  });

  it("does not expose one failed invocation to a concurrent sibling", async () => {
    const interruption = transportError();
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const failing = scopeAgentToolInvocation("worker_a", tool({
      name: "worker_a",
      description: "worker a",
      parameters: z.object({}).strict(),
      execute: async () => {
        recordAgentInvocationTransportInterruption(interruption);
        await gate;
        throw interruption;
      }
    }));
    const sibling = scopeAgentToolInvocation("worker_b", tool({
      name: "worker_b",
      description: "worker b",
      parameters: z.object({}).strict(),
      execute: async () => {
        release();
        await Promise.resolve();
        return currentAgentInvocationId();
      }
    }));

    const results = await withAgentInvocation("supervisor", async () => Promise.allSettled([
      failing.invoke(new RunContext(), "{}"),
      sibling.invoke(new RunContext(), "{}")
    ]));

    expect(results[0]).toEqual({ status: "rejected", reason: interruption });
    expect(results[1]).toEqual({ status: "fulfilled", value: "worker_b" });
  });

  it("restores a real nested Agent.asTool run after the worker transport drops", async () => {
    const interruption = transportError();
    const workerScopeIds: Array<string | undefined> = [];
    let workerCalls = 0;
    let coordinatorCalls = 0;
    const recordModelCallStarted = vi.fn(async () => undefined);
    const telemetryRuntime: ModelTelemetryRuntime = {
      rootAgentId: "coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted
    };
    const workerModel = withModelTelemetry({
      getResponse: async () => {
        throw new Error("non-stream path is outside this test");
      },
      getStreamedResponse: async function* () {
        workerCalls += 1;
        workerScopeIds.push(currentAgentInvocationId());
        if (workerCalls === 1) throw interruption;
        yield responseDone(messageResponse("worker-complete", "observation complete"));
      }
    } satisfies Model, telemetryRuntime, "worker");
    const worker = new Agent({
      name: "worker",
      instructions: `${agentInvocationMarker("worker")}\nObserve the current state.`,
      model: workerModel
    });
    const workerSession = new MemorySession({ sessionId: "worker-session" });
    const delegate = scopeAgentToolInvocation("worker", worker.asTool({
      toolName: "delegate_worker",
      toolDescription: "delegate observation",
      runOptions: { session: workerSession },
      onStream: async () => undefined
    }));
    const coordinatorModel = {
      getResponse: async () => {
        throw new Error("non-stream path is outside this test");
      },
      getStreamedResponse: async function* (request) {
        coordinatorCalls += 1;
        const hasWorkerResult = request.input.some((item) => (
          typeof item === "object"
          && item !== null
          && "type" in item
          && item.type === "function_call_result"
        ));
        yield responseDone(hasWorkerResult
          ? messageResponse("coordinator-complete", "cycle complete")
          : functionCallResponse());
      }
    } satisfies Model;
    const coordinator = new Agent({
      name: "coordinator",
      instructions: `${agentInvocationMarker("coordinator")}\nDelegate once.`,
      model: coordinatorModel,
      tools: [delegate]
    });
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 1 }
    });

    const first = await runner.run(coordinator, "observe", { stream: true });
    let resumableState: string | undefined;
    let surfaced: unknown;
    try {
      for await (const _event of first) {
        resumableState = first.state.toString();
      }
      await first.completed;
    } catch (error) {
      surfaced = error;
    }

    expect(isTransportInterruption(surfaced)).toBe(true);
    expect((surfaced as { error?: unknown }).error).toBe(interruption);
    expect(coordinatorCalls).toBe(1);
    expect(resumableState).toBeDefined();
    const restored = await RunState.fromString(coordinator, resumableState!);
    const resumed = await runner.run(coordinator, restored, { stream: true });
    await drain(resumed);

    expect(resumed.finalOutput).toBe("cycle complete");
    expect(workerCalls).toBe(2);
    expect(workerScopeIds).toEqual(["worker", "worker"]);
    expect(recordModelCallStarted).toHaveBeenCalledTimes(2);
    expect(recordModelCallStarted).toHaveBeenNthCalledWith(1, "worker");
    expect(recordModelCallStarted).toHaveBeenNthCalledWith(2, "worker");
    expect(coordinatorCalls).toBe(3);
  });

  it("does not surface an interruption recovered by the SDK model retry", async () => {
    const interruption = transportError();
    let workerCalls = 0;
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => undefined
    };
    const workerModel = withModelTelemetry({
      getResponse: async () => {
        workerCalls += 1;
        if (workerCalls === 1) throw interruption;
        return messageResponse("worker-retried", "retry completed");
      },
      getStreamedResponse: async function* () {
        throw new Error("streaming is outside this test");
      }
    } satisfies Model, runtime, "worker");
    const worker = new Agent({
      name: "worker",
      instructions: `${agentInvocationMarker("worker")}\nObserve.`,
      model: workerModel,
      modelSettings: {
        retry: {
          maxRetries: 1,
          policy: retryPolicies.networkError(),
          backoff: { initialDelayMs: 0, maxDelayMs: 0, jitter: false }
        }
      }
    });
    const delegate = scopeAgentToolInvocation("worker", worker.asTool({
      toolName: "delegate_retrying_worker",
      toolDescription: "delegate with SDK retry"
    }));

    await expect(delegate.invoke(
      new RunContext(),
      JSON.stringify({ input: "inspect" })
    )).resolves.toBe("retry completed");
    expect(workerCalls).toBe(2);
  });

});

async function drain(result: StreamedRunResult<unknown, Agent>): Promise<void> {
  for await (const _event of result) {
    // Drain the SDK stream so nested tool execution and completion both settle.
  }
  await result.completed;
}

function transportError(): TypeError {
  return new TypeError("fetch failed", {
    cause: Object.assign(new Error("socket reset"), { code: "ECONNRESET" })
  });
}

function responseDone(response: ModelResponse) {
  return {
    type: "response_done" as const,
    response: {
      id: response.responseId ?? "response",
      output: response.output,
      usage: response.usage
    }
  };
}

function messageResponse(responseId: string, text: string): ModelResponse {
  return modelResponse(responseId, [{
    type: "message",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text }]
  }]);
}

function reasoningResponse(responseId: string): ModelResponse {
  return modelResponse(responseId, [{
    type: "reasoning",
    content: [{ type: "input_text", text: "still thinking" }]
  }]);
}

function functionCallResponse(): ModelResponse {
  return modelResponse("delegate-call", [{
    type: "function_call",
    callId: "delegate-worker-1",
    name: "delegate_worker",
    arguments: JSON.stringify({ input: "inspect" })
  }]);
}

function modelResponse(responseId: string, output: unknown[]): ModelResponse {
  return {
    responseId,
    output,
    usage: new Usage({
      requests: 1,
      inputTokens: 1,
      outputTokens: 1,
      totalTokens: 2
    })
  } as ModelResponse;
}
