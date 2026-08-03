import {
  MemorySession,
  RunContext,
  type Model
} from "@openai/agents";
import { describe, expect, it } from "vitest";
import type { ProviderConfig } from "../../config/load.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import { createHumanoidAgentHierarchy } from "./agents.js";

const provider: ProviderConfig = {
  protocol: "openai_compatible",
  baseUrl: "https://example.test/v1",
  model: "test-model",
  apiKey: "test-key",
  temperature: 0.4,
  maxOutputTokens: 2048,
  contextWindowTokens: 32_768,
  compactTriggerTokens: 8192,
  compactRecentModelTurns: 4,
  compactMaxOutputTokens: 2048
};

describe("humanoid agent hierarchy", () => {
  it("owns one Model facade and one Session per concrete hierarchy node", async () => {
    const modelOwners: string[] = [];
    const sessionOwners: string[] = [];
    const models: Model[] = [];
    const recallRequests: unknown[] = [];
    const execution = receipt({
      transactionId: "execute-accepted",
      action: "execute_whole_body_motion",
      agentId: "humanoid-executor",
      worldBeforeRevision: 8,
      worldAfterRevision: 20,
      frameCount: 12,
      channels: ["left_arm"]
    });
    const runtime = {
      invoke: async () => execution,
      recallEmbodiedHistory: async (request: unknown) => {
        recallRequests.push(request);
        return {
          historical_only: false,
          current_world_revision: 20,
          episodes: [{ source_ref: "episode:7", sequence: 7 }]
        };
      },
      validateCycleEvidence: (transactionIds: readonly string[]) => {
        expect(transactionIds).toEqual([execution.transactionId]);
        return structuredClone(execution);
      }
    } as never;
    const hierarchy = createHumanoidAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => {
        modelOwners.push(agentId);
        const model = modelStub();
        models.push(model);
        return model;
      },
      createSession: (agentId) => {
        sessionOwners.push(agentId);
        return new MemorySession({ sessionId: agentId });
      }
    });

    expect(modelOwners).toEqual([
      "humanoid-sentry",
      "humanoid-motion-reference",
      "humanoid-executor",
      "humanoid-coordinator"
    ]);
    expect(sessionOwners).toEqual([
      "humanoid-sentry",
      "humanoid-motion-reference",
      "humanoid-executor",
      "humanoid-coordinator"
    ]);
    expect(new Set(models).size).toBe(4);
    expect(hierarchy.coordinator.model).not.toBe(hierarchy.sentry.model);
    expect(hierarchy.sentry.model).not.toBe(hierarchy.motion.model);
    expect(hierarchy.motion.model).not.toBe(hierarchy.executor.model);
    expect(hierarchy.session("humanoid-sentry")).toBeDefined();
    expect(hierarchy.session("humanoid-sentry")).not.toBe(
      hierarchy.session("humanoid-motion-reference")
    );
    expect(hierarchy.coordinatorSession).toBe(
      hierarchy.session("humanoid-coordinator")
    );

    expect(hierarchy.coordinator.tools.map((entry) => entry.name)).toEqual([
      "recall_embodied_history",
      "delegate_humanoid_sentry",
      "delegate_motion_reference",
      "delegate_physics_executor",
      "complete_autonomous_cycle"
    ]);
    expect(hierarchy.coordinator.tools.map((entry) => entry.name)).not.toContain(
      "execute_whole_body_motion"
    );
    expect(hierarchy.sentry.tools.map((entry) => entry.name)).toEqual([
      "observe_humanoid"
    ]);
    expect(hierarchy.motion.tools.map((entry) => entry.name)).toEqual([
      "observe_humanoid",
      "recall_embodied_history",
      "plan_whole_body_motion_candidates",
      "plan_humanoid_navigation"
    ]);
    expect(hierarchy.executor.tools.map((entry) => entry.name)).toEqual([
      "execute_whole_body_motion",
      "execute_humanoid_navigation"
    ]);
    expect(hierarchy.sentry.tools.map((entry) => entry.name)).not.toContain(
      "recall_embodied_history"
    );
    expect(hierarchy.executor.tools.map((entry) => entry.name)).not.toContain(
      "recall_embodied_history"
    );

    const recall = hierarchy.coordinator.tools.find(
      (entry) => entry.name === "recall_embodied_history"
    );
    if (!recall || recall.type !== "function") {
      throw new Error("Embodied history recall tool is missing");
    }
    const recalled = await recall.invoke(
      new RunContext({ runId: "humanoid-recall-test" }),
      JSON.stringify({ source_refs: ["episode:7"], limit: 1 })
    );
    expect(JSON.parse(String(recalled))).toMatchObject({
      historical_only: true,
      episodes: [{ source_ref: "episode:7", sequence: 7 }]
    });
    expect(recallRequests).toEqual([{ source_refs: ["episode:7"], limit: 1 }]);

    const complete = hierarchy.coordinator.tools.find(
      (entry) => entry.name === "complete_autonomous_cycle"
    );
    if (!complete || complete.type !== "function") {
      throw new Error("Cycle completion tool is missing");
    }
    const output = await complete.invoke(
      new RunContext({ runId: "humanoid-hierarchy-test" }),
      JSON.stringify({
        summary: "完成一次真实全身动作",
        evidence_transaction_ids: [execution.transactionId],
        next_intent: "观察动作后的平衡变化"
      })
    );
    expect(JSON.parse(String(output))).toMatchObject({
      status: "cycle_completed",
      world_revision: 20,
      executed_action: "execute_whole_body_motion"
    });
  });

  it("rejects a shared Model facade instead of mixing agent ownership", () => {
    const shared = modelStub();
    expect(() => createHumanoidAgentHierarchy({
      provider,
      runtime: {
        invoke: async () => { throw new Error("outside test"); },
        recallEmbodiedHistory: async () => { throw new Error("outside test"); },
        validateCycleEvidence: () => { throw new Error("outside test"); }
      } as never,
      createModel: () => shared,
      createSession: (agentId) => new MemorySession({ sessionId: agentId })
    })).toThrow("cannot share one Model facade");
  });
});

function modelStub(): Model {
  return {
    getResponse: async () => {
      throw new Error("Model calls are outside this construction test");
    },
    getStreamedResponse: () => {
      throw new Error("Model calls are outside this construction test");
    }
  } as unknown as Model;
}

function receipt(
  overrides: Partial<HumanoidActionReceipt>
): HumanoidActionReceipt {
  return {
    transactionId: "transaction",
    agentId: "agent",
    action: "observe_humanoid",
    accepted: true,
    code: "accepted",
    fingerprint: "fingerprint",
    worldBeforeRevision: 0,
    worldAfterRevision: 0,
    frameCount: 0,
    channels: [],
    detail: {},
    committedAt: "2026-08-02T00:00:00.000Z",
    ...overrides
  };
}
