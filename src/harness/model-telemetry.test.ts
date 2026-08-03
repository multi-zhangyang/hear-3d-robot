import type { Model, ModelResponse } from "@openai/agents";
import { describe, expect, it, vi } from "vitest";
import type {
  ModelProgressReceipt,
  ModelProgressSnapshot,
  ModelTelemetryRuntime
} from "./context-runtime.js";
import {
  AuthoritativeModelProgressGuard,
  ModelDecisionStallError,
  withModelTelemetry
} from "./model-telemetry.js";

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

  it("stops a repeated zero-progress receipt pattern with durable evidence", () => {
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
    snapshot.receipts.push(receipt("rejected-plan-3", {
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
          recent_transaction_ids: ["rejected-plan-3"]
        })
      })
    );
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
});

function progressSnapshot(): ModelProgressSnapshot {
  return {
    worldRevision: 0,
    cycleIndex: 0,
    checkerSuccess: false,
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
