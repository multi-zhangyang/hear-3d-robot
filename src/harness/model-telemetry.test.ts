import type { Model, ModelResponse } from "@openai/agents";
import { describe, expect, it, vi } from "vitest";
import type {
  ModelProgressReceipt,
  ModelProgressSnapshot,
  ModelTelemetryRuntime
} from "./context-runtime.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import {
  AuthoritativeModelProgressGuard,
  ModelDecisionStallError,
  modelDecisionStallFrom,
  modelTransportInterruptionAgentIdFrom,
  withModelTelemetry
} from "./model-telemetry.js";
import { agentInvocationMarker } from "./agent-scope.js";

describe("model decision interruption", () => {
  it("recovers the original stall through nested SDK tool wrappers", () => {
    const stall = new ModelDecisionStallError(
      "humanoid-executor",
      "model returned no tool decision"
    );
    const wrapped = {
      name: "ToolCallError",
      error: {
        name: "AgentToolError",
        cause: stall
      }
    };

    expect(modelDecisionStallFrom(wrapped)).toBe(stall);
    expect(modelDecisionStallFrom(new Error("unrelated"))).toBeUndefined();
  });
});

describe("authoritative model progress guard", () => {
  it("allows repeated observation-plan-execution chains when physics advances", () => {
    const snapshot = progressSnapshot();
    const guard = new AuthoritativeModelProgressGuard(snapshot);

    for (let cycle = 0; cycle < 12; cycle += 1) {
      expect(() => guard.observe("humanoid-coordinator", snapshot)).not.toThrow();
      snapshot.receipts.push(receipt(`observe-${cycle}`, {
        agentId: "humanoid-sentry",
        action: "observe_humanoid",
        code: "humanoid_observed"
      }));
      expect(() => guard.observe("humanoid-coordinator", snapshot)).not.toThrow();
      snapshot.receipts.push(receipt(`plan-${cycle}`, {
        agentId: "humanoid-motion-reference",
        action: "plan_whole_body_motion_candidates",
        code: "whole_body_candidates_validated"
      }));
      expect(() => guard.observe("humanoid-coordinator", snapshot)).not.toThrow();
      const before = snapshot.worldRevision;
      snapshot.worldRevision += 20;
      snapshot.receipts.push(receipt(`execute-${cycle}`, {
        agentId: "humanoid-executor",
        action: "execute_whole_body_motion",
        code: "motion_option_succeeded",
        frameCount: 20,
        worldBeforeRevision: before,
        worldAfterRevision: snapshot.worldRevision
      }));
      expect(() => guard.observe("humanoid-coordinator", snapshot)).not.toThrow();
      snapshot.cycleIndex += 1;
    }
  });

  it("starts a fresh repetition window after restoring durable receipts", () => {
    const snapshot = progressSnapshot();
    snapshot.receipts.push(
      receipt("rejected-plan-1", {
        action: "plan_whole_body_motion_candidates",
        accepted: false,
        code: "whole_body_candidates_rejected"
      }),
      receipt("rejected-plan-2", {
        action: "plan_whole_body_motion_candidates",
        accepted: false,
        code: "whole_body_candidates_rejected"
      })
    );
    const guard = new AuthoritativeModelProgressGuard(snapshot, {
      repeatedNoProgressReceipts: 3,
      decisionsWithoutAuthorityChange: 20,
      decisionsWithoutPhysicalProgress: 20
    });
    for (let index = 3; index < 5; index += 1) {
      snapshot.receipts.push(receipt(`rejected-plan-${index}`, {
        action: "plan_whole_body_motion_candidates",
        accepted: false,
        code: "whole_body_candidates_rejected"
      }));
      expect(() => guard.observe("humanoid-coordinator", snapshot)).not.toThrow();
    }
    snapshot.receipts.push(receipt("rejected-plan-5", {
      action: "plan_whole_body_motion_candidates",
      accepted: false,
      code: "whole_body_candidates_rejected"
    }));

    expect(() => guard.observe("humanoid-coordinator", snapshot)).toThrowError(
      expect.objectContaining({
        name: "ModelDecisionStallError",
        evidence: expect.objectContaining({
          reason: "repeated_no_progress_receipt",
          repeated_receipt_count: 3,
          recent_transaction_ids: ["rejected-plan-5"]
        })
      })
    );
  });

  it("allows the first safety observation after restoring repeated observations", () => {
    const snapshot = progressSnapshot();
    for (let index = 1; index <= 3; index += 1) {
      snapshot.receipts.push(receipt(`historic-observe-${index}`, {
        agentId: "humanoid-motion-reference",
        action: "observe_humanoid",
        code: "humanoid_observed"
      }));
    }
    const guard = new AuthoritativeModelProgressGuard(snapshot, {
      repeatedNoProgressReceipts: 4,
      decisionsWithoutAuthorityChange: 20,
      decisionsWithoutPhysicalProgress: 20
    });
    snapshot.receipts.push(receipt("recovery-observe-1", {
      agentId: "humanoid-motion-reference",
      action: "observe_humanoid",
      code: "humanoid_observed"
    }));

    expect(() => guard.observe("humanoid-motion-reference", snapshot)).not.toThrow();
  });

  it("bounds valid read-only decisions that never change authority", () => {
    const snapshot = progressSnapshot();
    const guard = new AuthoritativeModelProgressGuard(snapshot, {
      decisionsWithoutAuthorityChange: 2,
      repeatedNoProgressReceipts: 20,
      decisionsWithoutPhysicalProgress: 20
    });

    guard.observe("humanoid-motion-reference", snapshot);
    expect(() => guard.observe("humanoid-motion-reference", snapshot)).toThrowError(
      expect.objectContaining({
        evidence: expect.objectContaining({
          reason: "authority_unchanged",
          decisions_without_new_receipt: 2
        })
      })
    );
  });

  it("treats a Goal DAG transition as authoritative progress", () => {
    const snapshot = progressSnapshot();
    const guard = new AuthoritativeModelProgressGuard(snapshot, {
      decisionsWithoutAuthorityChange: 2,
      repeatedNoProgressReceipts: 20,
      decisionsWithoutPhysicalProgress: 20
    });

    guard.observe("humanoid-goal-manager", snapshot);
    snapshot.goalStateSha256 = "goal-state-1";
    expect(() => guard.observe("humanoid-goal-manager", snapshot)).not.toThrow();
    expect(() => guard.observe("humanoid-goal-manager", snapshot)).not.toThrow();
  });

  it("does not mistake idle physics clock revisions for Agent progress", () => {
    const snapshot = progressSnapshot();
    const guard = new AuthoritativeModelProgressGuard(snapshot, {
      decisionsWithoutAuthorityChange: 2,
      repeatedNoProgressReceipts: 20,
      decisionsWithoutPhysicalProgress: 20
    });

    snapshot.worldRevision += 50;
    guard.observe("humanoid-coordinator", snapshot);
    snapshot.worldRevision += 50;
    expect(() => guard.observe("humanoid-coordinator", snapshot)).toThrowError(
      expect.objectContaining({
        evidence: expect.objectContaining({ reason: "authority_unchanged" })
      })
    );
  });

  it("wires valid function-call responses into the existing recovery error", async () => {
    const snapshot = progressSnapshot();
    const recordModelCallStarted = vi.fn(async () => undefined);
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted,
      modelProgressSnapshot: () => structuredClone(snapshot)
    };
    const getResponse = vi.fn(async () => functionCallResponse());
    const model = {
      getResponse,
      getStreamedResponse: async function* () {
        throw new Error("streaming is outside this test");
      }
    } as unknown as Model;
    const wrapped = withModelTelemetry(model, runtime, runtime.rootAgentId);

    for (let index = 0; index < 5; index += 1) {
      await expect(wrapped.getResponse({ input: [] } as never)).resolves.toBeDefined();
    }
    await expect(wrapped.getResponse({ input: [] } as never)).rejects.toMatchObject({
      name: "ModelDecisionStallError",
      agentId: runtime.rootAgentId,
      evidence: { reason: "authority_unchanged" }
    } satisfies Partial<ModelDecisionStallError>);
    expect(getResponse).toHaveBeenCalledTimes(6);
    expect(recordModelCallStarted).toHaveBeenCalledTimes(6);
  });

  it("accepts SDK structured output as a formal model decision", async () => {
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => undefined
    };
    const model = {
      getResponse: async () => textResponse(
        '{"tool_name":"delegate_motion_reference","arguments_json":"{}"}'
      ),
      getStreamedResponse: async function* () {
        throw new Error("streaming is outside this test");
      }
    } as unknown as Model;
    const wrapped = withModelTelemetry(model, runtime, runtime.rootAgentId);

    for (let index = 0; index < 5; index += 1) {
      await expect(wrapped.getResponse({
        input: [],
        outputType: { type: "json_schema", name: "decision", schema: {} }
      } as never)).resolves.toBeDefined();
    }
  });

  it("does not count transport-retried delegation decisions as one semantic stall", async () => {
    const snapshot = progressSnapshot();
    let recoveryEpoch = 0;
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => undefined,
      modelProgressSnapshot: () => structuredClone(snapshot),
      modelProgressRecoveryEpoch: () => recoveryEpoch
    };
    const model = {
      getResponse: async () => functionCallResponse(),
      getStreamedResponse: async function* () {
        throw new Error("streaming is outside this test");
      }
    } as unknown as Model;
    const wrapped = withModelTelemetry(model, runtime, runtime.rootAgentId);

    for (let index = 0; index < 5; index += 1) {
      await expect(wrapped.getResponse({ input: [] } as never)).resolves.toBeDefined();
    }
    recoveryEpoch += 1;
    for (let index = 0; index < 5; index += 1) {
      await expect(wrapped.getResponse({ input: [] } as never)).resolves.toBeDefined();
    }
    await expect(wrapped.getResponse({ input: [] } as never)).rejects.toMatchObject({
      name: "ModelDecisionStallError",
      evidence: { reason: "authority_unchanged" }
    });
  });

  it("preserves the bound Agent identity on nested transport wrappers", async () => {
    const interruption = Object.assign(new Error("rate limited"), { statusCode: 429 });
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => undefined
    };
    const model = {
      getResponse: async () => { throw interruption; },
      getStreamedResponse: async function* () {
        throw new Error("streaming is outside this test");
      }
    } as unknown as Model;

    const surfaced = await withModelTelemetry(model, runtime, "humanoid-executor")
      .getResponse({
        systemInstructions: agentInvocationMarker("humanoid-executor"),
        input: []
      } as never)
      .catch((error: unknown) => error);

    expect(surfaced).toBe(interruption);
    expect(modelTransportInterruptionAgentIdFrom({
      error: { cause: surfaced }
    })).toBe("humanoid-executor");
  });

  it("replaces an SDK missing-ID sentinel with a traceable local response identity", async () => {
    const recordModelCallCompleted = vi.fn(async () => undefined);
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => "model-call-7",
      recordModelCallCompleted
    };
    const model = {
      getResponse: async () => { throw new Error("non-streaming is outside this test"); },
      getStreamedResponse: async function* () {
        yield {
          type: "response_done",
          response: {
            id: "FAKE_ID",
            usage: {
              inputTokens: 1,
              outputTokens: 1,
              totalTokens: 2
            },
            output: [{
              type: "function_call",
              callId: "call-7",
              name: "recall_embodied_history",
              arguments: "{}",
              providerData: { responseId: "FAKE_ID" }
            }]
          }
        } as never;
      }
    } as unknown as Model;
    const events: unknown[] = [];

    for await (const event of withModelTelemetry(
      model,
      runtime,
      runtime.rootAgentId
    ).getStreamedResponse({ input: [] } as never)) {
      events.push(event);
    }

    expect(events).toEqual([expect.objectContaining({
      type: "response_done",
      response: expect.objectContaining({
        id: "local-response:model-call-7",
        output: [expect.objectContaining({
          providerData: expect.objectContaining({
            responseId: "local-response:model-call-7"
          })
        })]
      })
    })]);
    expect(JSON.stringify(events)).not.toContain("FAKE_ID");
    expect(recordModelCallCompleted).toHaveBeenCalledWith(expect.objectContaining({
      modelCallId: "model-call-7",
      responseId: "local-response:model-call-7"
    }));
  });

  it("stops reading a provider stream after its terminal response", async () => {
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => "model-call-terminal"
    };
    const model = {
      getResponse: async () => { throw new Error("non-streaming is outside this test"); },
      getStreamedResponse: async function* () {
        yield {
          type: "response_done",
          response: {
            id: "response-terminal",
            usage: {
              inputTokens: 10,
              outputTokens: 2,
              totalTokens: 12
            },
            output: [{
              type: "function_call",
              callId: "call-terminal",
              name: "recall_embodied_history",
              arguments: "{}"
            }]
          }
        } as never;
        throw new Error("provider stream was read after response_done");
      }
    } as unknown as Model;
    const events: unknown[] = [];

    for await (const event of withModelTelemetry(
      model,
      runtime,
      runtime.rootAgentId
    ).getStreamedResponse({ input: [] } as never)) {
      events.push(event);
    }

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ type: "response_done" });
  });

  it("times out a stream whose transport heartbeats produce no SDK events", async () => {
    const recordModelCallFailed = vi.fn(async () => undefined);
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => "model-call-idle",
      recordModelCallFailed
    };
    let providerSignal: AbortSignal | undefined;
    const model = {
      getResponse: async () => { throw new Error("non-streaming is outside this test"); },
      getStreamedResponse: async function* (request) {
        providerSignal = request.signal;
        yield { type: "response_started" } as never;
        await new Promise<never>((_resolve, reject) => {
          request.signal?.addEventListener("abort", () => {
            reject(request.signal?.reason);
          }, { once: true });
        });
      }
    } as unknown as Model;
    const stream = withModelTelemetry(
      model,
      runtime,
      runtime.rootAgentId,
      undefined,
      10
    ).getStreamedResponse({ input: [] } as never)[Symbol.asyncIterator]();

    await expect(stream.next()).resolves.toMatchObject({
      done: false,
      value: { type: "response_started" }
    });
    const error = await stream.next().catch((failure: unknown) => failure);

    expect(isTransportInterruption(error)).toBe(true);
    expect(providerSignal?.aborted).toBe(true);
    expect(modelTransportInterruptionAgentIdFrom(error)).toBe(runtime.rootAgentId);
    expect(recordModelCallFailed).toHaveBeenCalledWith(
      "model-call-idle",
      runtime.rootAgentId
    );
  });

  it("times out a stream that keeps emitting events without completing a response", async () => {
    const recordModelCallFailed = vi.fn(async () => undefined);
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => "model-call-deadline",
      recordModelCallFailed
    };
    let providerSignal: AbortSignal | undefined;
    const model = {
      getResponse: async () => { throw new Error("non-streaming is outside this test"); },
      getStreamedResponse: async function* (request) {
        providerSignal = request.signal;
        for (;;) {
          await new Promise((resolve) => setTimeout(resolve, 4));
          yield { type: "response_started" } as never;
        }
      }
    } as unknown as Model;
    const stream = withModelTelemetry(
      model,
      runtime,
      runtime.rootAgentId,
      undefined,
      50,
      20
    ).getStreamedResponse({ input: [] } as never)[Symbol.asyncIterator]();

    await expect(stream.next()).resolves.toMatchObject({ done: false });
    const error = await (async () => {
      for (;;) {
        try {
          const next = await stream.next();
          if (next.done) return undefined;
        } catch (failure) {
          return failure;
        }
      }
    })();

    expect(isTransportInterruption(error)).toBe(true);
    expect(providerSignal?.aborted).toBe(true);
    expect(recordModelCallFailed).toHaveBeenCalledWith(
      "model-call-deadline",
      runtime.rootAgentId
    );
  });

  it("times out a non-streaming response whose transport never produces an SDK event", async () => {
    const recordModelCallFailed = vi.fn(async () => undefined);
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => "model-call-response-idle",
      recordModelCallFailed
    };
    let providerSignal: AbortSignal | undefined;
    const model = {
      getResponse: async (request) => {
        providerSignal = request.signal;
        return new Promise<never>((_resolve, reject) => {
          request.signal?.addEventListener("abort", () => {
            reject(request.signal?.reason);
          }, { once: true });
        });
      },
      getStreamedResponse: async function* () {
        throw new Error("streaming is outside this test");
      }
    } as unknown as Model;

    const error = await withModelTelemetry(
      model,
      runtime,
      runtime.rootAgentId,
      undefined,
      10
    ).getResponse({ input: [] } as never).catch((failure: unknown) => failure);

    expect(isTransportInterruption(error)).toBe(true);
    expect(providerSignal?.aborted).toBe(true);
    expect(modelTransportInterruptionAgentIdFrom(error)).toBe(runtime.rootAgentId);
    expect(recordModelCallFailed).toHaveBeenCalledWith(
      "model-call-response-idle",
      runtime.rootAgentId
    );
  });

  it("reads progress through a runtime method without losing its receiver", async () => {
    const runtime: ModelTelemetryRuntime & { snapshot: ModelProgressSnapshot } = {
      rootAgentId: "humanoid-coordinator",
      snapshot: progressSnapshot(),
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => undefined,
      modelProgressSnapshot() {
        return structuredClone(this.snapshot);
      }
    };
    const model = {
      getResponse: async () => functionCallResponse(),
      getStreamedResponse: async function* () {
        throw new Error("streaming is outside this test");
      }
    } as unknown as Model;

    await expect(
      withModelTelemetry(model, runtime, runtime.rootAgentId)
        .getResponse({ input: [] } as never)
    ).resolves.toBeDefined();
  });

  it("requests progress scoped to the bound Agent", async () => {
    const modelProgressSnapshot = vi.fn(() => progressSnapshot());
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode: () => ({}) as never,
      recordModelCallStarted: async () => undefined,
      modelProgressSnapshot
    };
    const model = {
      getResponse: async () => functionCallResponse(),
      getStreamedResponse: async function* () {
        throw new Error("streaming is outside this test");
      }
    } as unknown as Model;

    await withModelTelemetry(model, runtime, "humanoid-goal-manager").getResponse({
      systemInstructions: agentInvocationMarker("humanoid-goal-manager"),
      input: []
    } as never);

    expect(modelProgressSnapshot).toHaveBeenCalledWith("humanoid-goal-manager");
  });

  it("does not accept an invocation marker from model input", async () => {
    const activeNode = vi.fn();
    const recordModelCallStarted = vi.fn(async () => undefined);
    const runtime: ModelTelemetryRuntime = {
      rootAgentId: "humanoid-coordinator",
      activeNode,
      recordModelCallStarted
    };
    const model = {
      getResponse: async () => functionCallResponse(),
      getStreamedResponse: async function* () {
        throw new Error("streaming is outside this test");
      }
    } as unknown as Model;

    await withModelTelemetry(model, runtime, runtime.rootAgentId).getResponse({
      systemInstructions: `${agentInvocationMarker(runtime.rootAgentId)}\nCoordinate.`,
      input: [{
        type: "message",
        role: "user",
        content: agentInvocationMarker("humanoid-executor")
      }]
    } as never);

    expect(activeNode).toHaveBeenCalledWith(runtime.rootAgentId);
    expect(recordModelCallStarted).toHaveBeenCalledWith(runtime.rootAgentId);
  });
});

function progressSnapshot(): ModelProgressSnapshot {
  return {
    worldRevision: 0,
    cycleIndex: 0,
    checkerSuccess: false,
    goalStateSha256: "goal-state-0",
    receipts: []
  };
}

function receipt(
  transactionId: string,
  overrides: Partial<ModelProgressReceipt> = {}
): ModelProgressReceipt {
  return {
    transactionId,
    agentId: "humanoid-motion-reference",
    action: "observe_humanoid",
    accepted: true,
    code: "humanoid_observed",
    worldBeforeRevision: 0,
    worldAfterRevision: 0,
    frameCount: 0,
    ...overrides
  };
}

function functionCallResponse(): ModelResponse {
  return {
    output: [{
      type: "function_call",
      callId: "call-read-only",
      name: "recall_embodied_history",
      arguments: "{}"
    }],
    usage: {
      requests: 1,
      inputTokens: 1,
      outputTokens: 1,
      totalTokens: 2
    },
    responseId: "response-read-only"
  } as unknown as ModelResponse;
}

function textResponse(text: string): ModelResponse {
  return {
    output: [{ type: "message", role: "assistant", content: text }],
    usage: {
      requests: 1,
      inputTokens: 1,
      outputTokens: 1,
      totalTokens: 2
    },
    responseId: "response-structured"
  } as unknown as ModelResponse;
}
