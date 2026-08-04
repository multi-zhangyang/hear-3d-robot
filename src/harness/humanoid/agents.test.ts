import {
  MemorySession,
  RunContext,
  type Model
} from "@openai/agents";
import { describe, expect, it } from "vitest";
import type { ProviderConfig } from "../../config/load.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  createHumanoidAgentHierarchy,
  goalManagerInvocationInput
} from "./agents.js";

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
  it("places exact live Goal identifiers in every Goal Manager invocation", () => {
    const evidenceRef = `goal-world:19:19:${"a".repeat(64)}`;
    const candidateId = `goal-candidate:${"b".repeat(64)}`;

    const rendered = goalManagerInvocationInput("推进长期任务。", {
      goal_dag: {
        status: "awaiting_model_selection",
        candidates: { [candidateId]: { status: "proposed" } },
        current_epoch_id: null
      },
      goal_context: {
        evidence_ref: evidenceRef,
        observation: {
          zone_ids: ["courtyard_beacon"],
          visible_object_ids: ["courtyard_crate"],
          solids: [
            { id: "stone_column", kind: "block" },
            { id: "wall", kind: "wall" }
          ]
        }
      }
    });

    expect(rendered).toContain(`"current_goal_evidence_ref":"${evidenceRef}"`);
    expect(rendered).toContain(`"existing_goal_candidate_ids":["${candidateId}"]`);
    expect(rendered).toContain(
      '"visible_object_ids":["courtyard_crate"],"removable_block_ids":["stone_column"]'
    );
    expect(rendered).toContain("候选提交和选择会由 Harness 绑定本次证据");
  });

  it("owns one Model facade and one Session per concrete hierarchy node", async () => {
    const modelOwners: string[] = [];
    const sessionOwners: string[] = [];
    const models: Model[] = [];
    const recallRequests: unknown[] = [];
    let cycleCompletion = {
      status: "not_ready" as "ready" | "not_ready",
      evidence_transaction_ids: [] as string[],
      execution_transaction_id: null as string | null,
      observed_after_execution: false,
      reason: "no execution" as string | null
    };
    let coordinatorPhase = "observe_or_plan" as
      | "goal_selection"
      | "observe_or_plan"
      | "plan"
      | "replan_or_retire"
      | "execute_plan"
      | "post_execution"
      | "complete_cycle";
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
      },
      cycleCompletionReadiness: () => structuredClone(cycleCompletion),
      coordinatorPhase: () => coordinatorPhase
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
      "humanoid-goal-manager",
      "humanoid-sentry",
      "humanoid-motion-reference",
      "humanoid-executor",
      "humanoid-coordinator"
    ]);
    expect(sessionOwners).toEqual([
      "humanoid-goal-manager",
      "humanoid-sentry",
      "humanoid-motion-reference",
      "humanoid-executor",
      "humanoid-coordinator"
    ]);
    expect(new Set(models).size).toBe(5);
    expect(hierarchy.goalManager.model).not.toBe(hierarchy.coordinator.model);
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
      "delegate_goal_manager",
      "delegate_humanoid_sentry",
      "delegate_motion_reference",
      "delegate_physics_executor",
      "complete_autonomous_cycle",
      "complete_goal_transition"
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
      "execute_humanoid_navigation",
      "remove_world_block"
    ]);
    expect(hierarchy.sentry.tools.map((entry) => entry.name)).not.toContain(
      "recall_embodied_history"
    );
    expect(hierarchy.executor.tools.map((entry) => entry.name)).not.toContain(
      "recall_embodied_history"
    );

    const coordinatorTool = (name: string) => {
      const selected = hierarchy.coordinator.tools.find((entry) => entry.name === name);
      if (!selected || selected.type !== "function") {
        throw new Error(`Coordinator tool is missing: ${name}`);
      }
      return selected;
    };
    const enabled = (name: string) => coordinatorTool(name).isEnabled(
      new RunContext({ runId: `enabled-${name}` }),
      hierarchy.coordinator
    );
    expect(await enabled("delegate_motion_reference")).toBe(true);
    expect(await enabled("complete_autonomous_cycle")).toBe(false);
    cycleCompletion = {
      status: "ready",
      evidence_transaction_ids: [execution.transactionId],
      execution_transaction_id: execution.transactionId,
      observed_after_execution: false,
      reason: null
    };
    coordinatorPhase = "post_execution";
    expect(await enabled("delegate_motion_reference")).toBe(false);
    expect(await enabled("delegate_humanoid_sentry")).toBe(true);
    expect(await enabled("complete_autonomous_cycle")).toBe(true);
    cycleCompletion.observed_after_execution = true;
    coordinatorPhase = "complete_cycle";
    expect(await enabled("delegate_humanoid_sentry")).toBe(false);
    expect(await enabled("delegate_physics_executor")).toBe(true);
    expect(hierarchy.motion.instructions).toEqual(expect.stringContaining(
      "object_in_zone、not grasp_verified 与 object_settled_on_support"
    ));

    const coordinatorBehavior = hierarchy.coordinator.toolUseBehavior;
    if (typeof coordinatorBehavior !== "function") {
      throw new Error("Coordinator must validate terminal tool output");
    }
    const rejectedTerminal = await coordinatorBehavior(
      new RunContext({ runId: "rejected-terminal" }),
      [{
        type: "function_output",
        tool: { name: "complete_goal_transition" },
        output: "An error occurred while running the tool"
      }] as never
    );
    expect(rejectedTerminal).toEqual({
      isFinalOutput: false,
      isInterrupted: undefined
    });
    const acceptedTerminal = await coordinatorBehavior(
      new RunContext({ runId: "accepted-terminal" }),
      [{
        type: "function_output",
        tool: { name: "complete_goal_transition" },
        output: JSON.stringify({ status: "goal_transition_completed" })
      }] as never
    );
    expect(acceptedTerminal).toMatchObject({
      isFinalOutput: true,
      finalOutput: JSON.stringify({ status: "goal_transition_completed" })
    });

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

  it("applies each resolved profile to only its owning Agent", () => {
    const { maxOutputTokens: _maxOutputTokens, ...unboundedProvider } = provider;
    const configured: ProviderConfig = {
      ...provider,
      agentModels: {
        goal_manager: { ...unboundedProvider, model: "goal-manager", temperature: 0.15 },
        coordinator: { ...provider, model: "coordinator", temperature: 0.1 },
        sentry: { ...provider, model: "sentry", temperature: 0.2 },
        motion: { ...provider, model: "motion", temperature: 0.3 },
        executor: { ...provider, model: "executor", temperature: 0.4 },
        compactor: { ...provider, model: "compactor", temperature: 0.5 }
      }
    };
    const owners = new Map<string, string>();
    const hierarchy = createHumanoidAgentHierarchy({
      provider: configured,
      runtime: {
        invoke: async () => { throw new Error("outside test"); },
        recallEmbodiedHistory: async () => { throw new Error("outside test"); },
        validateCycleEvidence: () => { throw new Error("outside test"); }
      } as never,
      createModel: (agentId, selected) => {
        owners.set(agentId, selected.model);
        return modelStub();
      },
      createSession: (agentId) => new MemorySession({ sessionId: agentId })
    });

    expect(Object.fromEntries(owners)).toEqual({
      "humanoid-goal-manager": "goal-manager",
      "humanoid-sentry": "sentry",
      "humanoid-motion-reference": "motion",
      "humanoid-executor": "executor",
      "humanoid-coordinator": "coordinator"
    });
    expect(hierarchy.coordinator.modelSettings.temperature).toBe(0.1);
    expect(hierarchy.goalManager.modelSettings.temperature).toBe(0.15);
    expect(hierarchy.goalManager.modelSettings).not.toHaveProperty("maxTokens");
    expect(hierarchy.sentry.modelSettings.temperature).toBe(0.2);
    expect(hierarchy.motion.modelSettings.temperature).toBe(0.3);
    expect(hierarchy.executor.modelSettings.temperature).toBe(0.4);
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
