import { describe, expect, it } from "vitest";
import type { AgentInputItem, ModelInputData } from "@openai/agents";
import type { ModelProviderConfig, ProviderConfig } from "../config/load.js";
import {
  EmptyContextMemoryState,
  type ContextMemoryState,
  type TaskNode
} from "../domain/schema.js";
import type { LongRunContextRuntime } from "./context-runtime.js";
import { compactorInputTokenLimit } from "../runtime/context-budget.js";
import { agentInvocationMarker } from "./agent-scope.js";
import { LongRunContextManager } from "./context-compaction.js";
import type { ContextSummaryRequest } from "./context-summary-agent.js";

describe("LongRunContextManager hierarchy identity", () => {
  it("binds worker memory and budgets from trusted Agent instructions", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const activeNodeCalls: string[] = [];
    const configuredAgents: string[] = [];
    const nodes = new Map([
      ["humanoid-coordinator", taskNode("humanoid-coordinator", "协调")],
      ["humanoid-sentry", taskNode("humanoid-sentry", "感知")]
    ]);
    const runtime = {
      rootAgentId: "humanoid-coordinator",
      signal: undefined,
      store: {},
      activeNode(agentId = "humanoid-coordinator") {
        activeNodeCalls.push(agentId);
        const node = nodes.get(agentId);
        if (!node) throw new Error(`Unknown test node: ${agentId}`);
        return node;
      },
      contextAnchor: (agentId: string) => ({ agent_id: agentId, world_revision: 0 }),
      contextMemory: () => structuredClone(memory),
      contextWorldIdentity: () => ({ worldRevision: 0 }),
      contextReceipts: () => ({}),
      assertContextSummaryEvidence: () => undefined,
      async updateContextMemory(state: ContextMemoryState) {
        memory = structuredClone(state);
      },
      async recordCompactionModelCall() {},
      async reconcileCompactionModelCalls() {},
      async recordProvider() {}
    } as unknown as LongRunContextRuntime;
    const provider = providerConfig();
    const manager = new LongRunContextManager({
      runtime,
      provider,
      configForAgent(agentId) {
        configuredAgents.push(agentId);
        return provider;
      },
      createGenerator: () => ({
        async generate() {
          throw new Error("Compaction was not expected in this test");
        }
      })
    });
    const modelData: ModelInputData = {
      instructions: `${agentInvocationMarker("humanoid-sentry")}\nObserve only.`,
      input: [{ role: "user", content: "Inspect the current robot." }] as AgentInputItem[]
    };

    const filtered = await manager.filter({
      modelData,
      agent: { name: "人形感知哨兵", tools: [] }
    } as never);

    expect(activeNodeCalls).toEqual(["humanoid-sentry"]);
    expect(configuredAgents).toEqual(["humanoid-sentry"]);
    expect(manager.snapshot.active_scope_id).toBe("humanoid-sentry");
    expect(Object.keys(manager.snapshot.scopes)).toEqual(["humanoid-sentry"]);
    expect(filtered.instructions).toContain(agentInvocationMarker("humanoid-sentry"));
  });

  it("surfaces revision-bound Goal identifiers ahead of the full authority payload", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const node = taskNode("humanoid-goal-manager", "目标管理");
    const evidenceRef = `goal-world:41:41:${"a".repeat(64)}`;
    const candidateId = `goal-candidate:${"b".repeat(64)}`;
    const runtime = {
      rootAgentId: "humanoid-coordinator",
      signal: undefined,
      store: {},
      activeNode: () => node,
      contextAnchor: () => ({
        world_frame: 41,
        world_revision: 41,
        coordinator_phase: "execute_plan",
        goal_context: { evidence_ref: evidenceRef },
        goal_dag: {
          candidates: { [candidateId]: { status: "proposed" } },
          current_epoch_id: null
        },
        execution_authority: {
          planning_action: "plan_humanoid_navigation",
          planning_transaction_id: "planning-call-41",
          executor_action: "execute_humanoid_navigation"
        }
      }),
      contextMemory: () => structuredClone(memory),
      contextWorldIdentity: () => ({ worldRevision: 41 }),
      contextReceipts: () => ({}),
      assertContextSummaryEvidence: () => undefined,
      async updateContextMemory(state: ContextMemoryState) {
        memory = structuredClone(state);
      },
      async recordCompactionModelCall() {},
      async reconcileCompactionModelCalls() {},
      async recordProvider() {}
    } as unknown as LongRunContextRuntime;
    const manager = new LongRunContextManager({
      runtime,
      provider: providerConfig(),
      createGenerator: () => ({
        async generate() {
          throw new Error("Compaction was not expected in this test");
        }
      })
    });

    const filtered = await manager.filter({
      modelData: {
        instructions: `${agentInvocationMarker(node.id)}\nManage Goals.`,
        input: [{ role: "user", content: "Choose the next Goal." }] as AgentInputItem[]
      },
      agent: { name: node.name, tools: [] }
    } as never);

    expect(filtered.instructions).toContain(
      "CURRENT EXACT IDENTIFIERS (copy values character-for-character; never invent aliases)"
    );
    expect(filtered.instructions).toContain(`goal_evidence_ref=${JSON.stringify(evidenceRef)}`);
    expect(filtered.instructions).toContain(
      `existing_goal_candidate_ids=${JSON.stringify([candidateId])}`
    );
    expect(filtered.instructions).toContain("current_world_revision=41");
    expect(filtered.instructions).toContain(
      'pending_planning_transaction_id="planning-call-41"'
    );
    expect(filtered.instructions).toContain(
      'required_executor_action="execute_humanoid_navigation"'
    );
  });

  it("calibrates the provider-neutral estimate from reported input usage", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const journal: unknown[] = [];
    const node = taskNode("humanoid-coordinator", "协调");
    const runtime = {
      rootAgentId: node.id,
      signal: undefined,
      store: {},
      activeNode: () => node,
      contextAnchor: () => ({ world_revision: 0 }),
      contextMemory: () => structuredClone(memory),
      contextWorldIdentity: () => ({ worldRevision: 0 }),
      contextReceipts: () => ({}),
      assertContextSummaryEvidence: () => undefined,
      async updateContextMemory(state: ContextMemoryState, record?: unknown) {
        memory = structuredClone(state);
        if (record !== undefined) journal.push(record);
      },
      async recordCompactionModelCall() {},
      async reconcileCompactionModelCalls() {},
      async recordProvider() {}
    } as unknown as LongRunContextRuntime;
    const manager = new LongRunContextManager({
      runtime,
      provider: providerConfig({
        contextWindowTokens: 8_192,
        compactTriggerTokens: 7_000
      }),
      createGenerator: () => ({
        async generate() {
          throw new Error("Compaction was not expected in this test");
        }
      })
    });
    const request = {
      modelData: {
        instructions: `${agentInvocationMarker(node.id)}\nCoordinate.`,
        input: [{ role: "user", content: "Inspect current state." }] as AgentInputItem[]
      },
      agent: { name: node.name, tools: [] }
    } as never;

    await manager.filter(request);
    const baseline = manager.snapshot.scopes[node.id]!.active_estimated_tokens;
    await manager.recordModelInputUsage(node.id, baseline * 2);

    expect(manager.snapshot.scopes[node.id]).toMatchObject({
      token_estimator_correction_milli: 2_000,
      active_estimated_tokens: baseline * 2
    });
    expect(journal).toContainEqual(expect.objectContaining({
      type: "context_token_estimator_calibrated",
      estimated_input_tokens: baseline,
      reported_input_tokens: baseline * 2,
      correction_milli: 2_000
    }));

    await manager.filter(request);
    expect(manager.snapshot.scopes[node.id]!.active_estimated_tokens)
      .toBe(baseline * 2);
  });

  it("rewrites a completed SDK Session from the logical hot tail", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    let worldRevision = 0;
    const node = taskNode("humanoid-coordinator", "协调");
    const runtime = {
      rootAgentId: node.id,
      signal: undefined,
      store: {},
      activeNode: () => node,
      contextAnchor: () => ({ world_revision: worldRevision }),
      contextMemory: () => structuredClone(memory),
      contextWorldIdentity: () => ({ worldRevision }),
      contextReceipts: () => ({}),
      assertContextSummaryEvidence: () => undefined,
      async updateContextMemory(state: ContextMemoryState) {
        memory = structuredClone(state);
      },
      async recordCompactionModelCall() {},
      async reconcileCompactionModelCalls() {},
      async recordProvider() {}
    } as unknown as LongRunContextRuntime;
    const manager = new LongRunContextManager({
      runtime,
      provider: providerConfig({
        contextWindowTokens: 8_192,
        compactTriggerTokens: 1_000,
        compactRecentModelTurns: 1,
        compactMaxOutputTokens: 120
      }),
      createGenerator: () => ({
        async generate() {
          worldRevision = 1;
          return {
            summary: {
              mission_state: "继续长期任务。",
              constraints: [],
              decisions: [],
              completed: [],
              pending: ["保留最新物理状态。"],
              blockers: [],
              next_actions: ["继续规划。"]
            },
            origin: "model",
            usage: { requests: 1, inputTokens: 500, outputTokens: 40, totalTokens: 540 }
          };
        }
      })
    });
    const physical = [
      { role: "user", content: "x".repeat(3_000) },
      { role: "assistant", content: "旧状态已读取。" },
      { role: "user", content: "保留这一轮。" }
    ] as AgentInputItem[];
    const request = {
      modelData: {
        instructions: `${agentInvocationMarker(node.id)}\nCoordinate.`,
        input: physical
      },
      agent: { name: node.name, tools: [] }
    } as never;
    const filtered = await manager.filter(request);
    const completion = {
      role: "assistant",
      content: "完成当前决策。"
    } as AgentInputItem;
    let persisted = [...structuredClone(physical), completion];
    let replacements = 0;
    const session = {
      getItems: async () => structuredClone(persisted),
      replaceItems: async (items: AgentInputItem[]) => {
        persisted = structuredClone(items);
        replacements += 1;
      }
    };

    expect(manager.snapshot.scopes[node.id]).toMatchObject({
      compaction_count: 1,
      summary_origin: "model",
      summary_world_revision: 0
    });
    await manager.compactSessionHistories(() => session as never);

    expect(replacements).toBe(1);
    expect(persisted).toEqual([...filtered.input, completion]);
    await manager.filter({
      ...request,
      modelData: { ...request.modelData, input: persisted }
    });
    expect(manager.snapshot.scopes[node.id]).toMatchObject({
      compaction_count: 1,
      summary_origin: "model"
    });
  });

  it("persists distinct scope budgets and applies one effective compactor envelope", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const requests: Array<{ agentId: string; request: ContextSummaryRequest }> = [];
    const nodes = new Map([
      ["humanoid-coordinator", taskNode("humanoid-coordinator", "协调")],
      ["humanoid-sentry", taskNode("humanoid-sentry", "感知")]
    ]);
    const coordinator = providerConfig({
      contextWindowTokens: 8_192,
      compactTriggerTokens: 5_000,
      compactRecentModelTurns: 3,
      compactMaxOutputTokens: 80
    });
    const sentry = providerConfig({
      contextWindowTokens: 4_096,
      compactTriggerTokens: 700,
      compactRecentModelTurns: 1,
      compactMaxOutputTokens: 120
    });
    const compactor = providerConfig({
      contextWindowTokens: 8_192,
      compactTriggerTokens: 2_000,
      compactRecentModelTurns: 2,
      compactMaxOutputTokens: 200,
      maxOutputTokens: 150
    });
    const runtime = {
      rootAgentId: "humanoid-coordinator",
      signal: undefined,
      store: {},
      activeNode(agentId = "humanoid-coordinator") {
        const node = nodes.get(agentId);
        if (!node) throw new Error(`Unknown test node: ${agentId}`);
        return node;
      },
      contextAnchor: (agentId: string) => ({ agent_id: agentId, world_revision: 0 }),
      contextMemory: () => structuredClone(memory),
      contextWorldIdentity: () => ({ worldRevision: 0 }),
      contextReceipts: () => ({}),
      assertContextSummaryEvidence: () => undefined,
      async updateContextMemory(state: ContextMemoryState) {
        memory = structuredClone(state);
      },
      async recordCompactionModelCall() {},
      async reconcileCompactionModelCalls() {},
      async recordProvider() {}
    } as unknown as LongRunContextRuntime;
    const manager = new LongRunContextManager({
      runtime,
      provider: coordinator,
      compactorProvider: compactor,
      configForAgent: (agentId) => agentId === "humanoid-sentry" ? sentry : coordinator,
      createGenerator: (agentId) => ({
        async generate(request) {
          requests.push({ agentId, request });
          return {
            summary: {
              mission_state: "继续执行当前任务。",
              constraints: [],
              decisions: [],
              completed: [],
              pending: ["读取最新观察。"],
              blockers: [],
              next_actions: ["继续规划。"]
            },
            origin: "model",
            usage: { requests: 1, inputTokens: 300, outputTokens: 40, totalTokens: 340 }
          };
        }
      })
    });

    await manager.filter({
      modelData: {
        instructions: `${agentInvocationMarker("humanoid-coordinator")}\nCoordinate.`,
        input: [{ role: "user", content: "Inspect the mission." }] as AgentInputItem[]
      },
      agent: { name: "协调", tools: [] }
    } as never);
    await manager.filter({
      modelData: {
        instructions: `${agentInvocationMarker("humanoid-sentry")}\nObserve.`,
        input: [
          { role: "user", content: "x".repeat(3_000) },
          { role: "assistant", content: "Observation acknowledged." },
          { role: "user", content: "Keep the newest evidence." }
        ] as AgentInputItem[]
      },
      agent: { name: "感知", tools: [] }
    } as never);

    expect(manager.snapshot.scopes["humanoid-coordinator"]).toMatchObject({
      context_window_tokens: 8_192,
      compact_trigger_tokens: 5_000,
      compact_recent_model_turns: 3,
      compact_max_output_tokens: 80
    });
    expect(manager.snapshot.scopes["humanoid-sentry"]).toMatchObject({
      context_window_tokens: 4_096,
      compact_trigger_tokens: 700,
      compact_recent_model_turns: 1,
      compact_max_output_tokens: 120
    });
    expect(manager.snapshot).toMatchObject({
      active_scope_id: "humanoid-sentry",
      context_window_tokens: 4_096,
      compact_trigger_tokens: 700,
      compact_recent_model_turns: 1,
      compact_max_output_tokens: 120
    });
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      agentId: "humanoid-sentry",
      request: {
        maxOutputTokens: 120,
        maxInputTokens: compactorInputTokenLimit(8_192, 120)
      }
    });
  });
});

function providerConfig(
  overrides: Partial<ModelProviderConfig> = {}
): ProviderConfig & ModelProviderConfig {
  return {
    protocol: "openai_compatible",
    baseUrl: "https://example.invalid/v1",
    model: "test-model",
    apiKey: "test-key",
    requestTimeoutMs: 1_000,
    temperature: 0.2,
    maxOutputTokens: 128,
    contextWindowTokens: 4_096,
    compactTriggerTokens: 2_048,
    compactRecentModelTurns: 2,
    compactMaxOutputTokens: 256,
    ...overrides
  };
}

function taskNode(id: string, name: string): TaskNode {
  const now = "2026-08-03T00:00:00.000Z";
  return {
    id,
    name,
    parent_id: id === "humanoid-coordinator" ? null : "humanoid-coordinator",
    child_ids: [],
    objective: "test",
    success_criteria: ["test"],
    evidence_requirements: [],
    goal_predicate_indexes: [],
    capabilities: [],
    may_delegate: false,
    references: [],
    depth: id === "humanoid-coordinator" ? 0 : 1,
    status: "ready",
    steps_used: 0,
    model_calls_used: 0,
    created_at: now,
    updated_at: now
  };
}
