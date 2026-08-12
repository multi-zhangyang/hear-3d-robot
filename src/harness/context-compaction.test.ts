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
import { modelTransportInterruptionAgentIdFrom } from "./model-telemetry.js";
import {
  ContextCompactionCapacityError,
  ContextCompactionInterruption,
  type ContextSummaryRequest
} from "./context-summary-agent.js";

describe("LongRunContextManager hierarchy identity", () => {
  it("binds worker memory and budgets from trusted Agent instructions", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const activeNodeCalls: string[] = [];
    const configuredAgents: string[] = [];
    const nodes = new Map([
      ["humanoid-coordinator", taskNode("humanoid-coordinator", "协调")],
      ["humanoid-motion-reference", taskNode("humanoid-motion-reference", "运动")]
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
      instructions: `${agentInvocationMarker("humanoid-motion-reference")}\nPlan motion only.`,
      input: [{ role: "user", content: "Inspect the current robot." }] as AgentInputItem[]
    };

    const filtered = await manager.filter({
      modelData,
      agent: { name: "全身运动参考智能体", tools: [] }
    } as never);

    expect(activeNodeCalls).toEqual(["humanoid-motion-reference"]);
    expect(configuredAgents).toEqual(["humanoid-motion-reference"]);
    expect(manager.snapshot.active_scope_id).toBe("humanoid-motion-reference");
    expect(Object.keys(manager.snapshot.scopes)).toEqual(["humanoid-motion-reference"]);
    expect(filtered.instructions).toContain(agentInvocationMarker("humanoid-motion-reference"));
  });

  it("surfaces revision-bound Goal identifiers in the final authority envelope", async () => {
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

    const authorityItem = filtered.input.at(-1);
    expect(authorityItem).toMatchObject({
      type: "message",
      role: "user"
    });
    const authorityText = authorityItem && "content" in authorityItem
      && typeof authorityItem.content === "string"
      ? authorityItem.content
      : "";
    expect(authorityText).toContain(
      "CURRENT EXACT IDENTIFIERS (copy values character-for-character; never invent aliases)"
    );
    expect(authorityText).toContain(`goal_evidence_ref=${JSON.stringify(evidenceRef)}`);
    expect(authorityText).toContain(
      `existing_goal_candidate_ids=${JSON.stringify([candidateId])}`
    );
    expect(authorityText).toContain("current_world_revision=41");
    expect(authorityText).toContain(
      'pending_planning_transaction_id="planning-call-41"'
    );
    expect(authorityText).toContain(
      'required_executor_action="execute_humanoid_navigation"'
    );
    expect(authorityText).toMatch(
      /END CURRENT HARNESS AUTHORITY\nFollow the stable Agent instructions now\. Return the required formal function call; prose is not a tool decision\.$/
    );
    expect(filtered.instructions).not.toContain("current_world_revision=41");
  });

  it("keeps the semantic prefix stable with only the current authority suffix", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    let worldRevision = 1;
    const node = taskNode("humanoid-coordinator", "协调");
    const runtime = {
      rootAgentId: node.id,
      signal: undefined,
      store: {
        readJournalTail: async () => ({ total: 0 })
      },
      activeNode: () => node,
      contextAnchor: () => ({
        world_frame: worldRevision,
        world_revision: worldRevision,
        coordinator_phase: "plan"
      }),
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
      provider: providerConfig(),
      createGenerator: () => ({
        async generate() {
          throw new Error("Compaction was not expected in this test");
        }
      })
    });
    const modelData: ModelInputData = {
      instructions: `${agentInvocationMarker(node.id)}\nCoordinate.`,
      input: [{ role: "user", content: "Continue." }] as AgentInputItem[]
    };

    const first = await manager.filter({
      modelData,
      agent: { name: node.name, tools: [] }
    } as never);
    worldRevision = 2;
    const second = await manager.filter({
      modelData: {
        ...modelData,
        input: [
          ...modelData.input,
          { role: "assistant", content: "Previous decision." }
        ] as AgentInputItem[]
      },
      agent: { name: node.name, tools: [] }
    } as never);

    expect(second.instructions).toBe(first.instructions);
    expect(second.instructions).not.toContain("world_revision");
    expect(second.input.slice(0, -1)).toEqual([
      ...modelData.input,
      { role: "assistant", content: "Previous decision." }
    ]);
    expect(second.input.filter(isHarnessAuthorityItem)).toHaveLength(1);
    const current = second.input.at(-1);
    expect(current && "content" in current ? current.content : "")
      .toContain("current_world_revision=2");
    expect(JSON.stringify(second.input)).not.toContain("current_world_revision=1");
    expect(manager.snapshot.scopes[node.id]).toMatchObject({
      raw_item_count: 2,
      retained_item_count: 2
    });

    worldRevision = 3;
    const terminal = { role: "assistant" as const, content: "Terminal decision." };
    const nextDelegation = { role: "user" as const, content: "Delegate again." };
    const third = await manager.filter({
      modelData: {
        ...modelData,
        input: [
          ...first.input,
          { role: "assistant", content: "Previous decision." },
          terminal,
          nextDelegation
        ] as AgentInputItem[]
      },
      agent: { name: node.name, tools: [] }
    } as never);

    expect(third.input.slice(0, -1)).toEqual([
      ...modelData.input,
      { role: "assistant", content: "Previous decision." },
      terminal,
      nextDelegation
    ]);
    expect(third.input.filter(isHarnessAuthorityItem)).toHaveLength(1);
    expect(JSON.stringify(third.input)).not.toContain("current_world_revision=1");
    expect(JSON.stringify(third.input.at(-1))).toContain("current_world_revision=3");
  });

  it("removes incomplete function-tool fragments before an OpenAI-compatible request", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const providerEvents: unknown[] = [];
    const node = taskNode("humanoid-motion-reference", "运动参考");
    const runtime = {
      rootAgentId: "humanoid-coordinator",
      signal: undefined,
      store: {},
      activeNode: () => node,
      contextAnchor: () => ({ world_revision: 7 }),
      contextMemory: () => structuredClone(memory),
      contextWorldIdentity: () => ({ worldRevision: 7 }),
      contextReceipts: () => ({}),
      assertContextSummaryEvidence: () => undefined,
      async updateContextMemory(state: ContextMemoryState) {
        memory = structuredClone(state);
      },
      async recordCompactionModelCall() {},
      async reconcileCompactionModelCalls() {},
      async recordProvider(event: unknown) {
        providerEvents.push(event);
      }
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
    const input = [
      { type: "message", role: "user", content: "Plan." },
      {
        type: "function_call",
        callId: "complete-call",
        name: "observe_humanoid",
        arguments: "{}",
        status: "completed"
      },
      {
        type: "function_call_result",
        callId: "complete-call",
        name: "observe_humanoid",
        output: "observed",
        status: "completed"
      },
      {
        type: "function_call",
        callId: "missing-result",
        name: "submit_humanoid_skill_plan",
        arguments: "{}",
        status: "completed"
      },
      { type: "message", role: "user", content: "Retry." },
      {
        type: "function_call_result",
        callId: "missing-call",
        name: "submit_humanoid_skill_plan",
        output: "Tool not found.",
        status: "completed"
      },
      { type: "message", role: "user", content: "Continue." }
    ] as AgentInputItem[];

    const filtered = await manager.filter({
      modelData: {
        instructions: `${agentInvocationMarker(node.id)}\nPlan motion.`,
        input
      },
      agent: { name: node.name, tools: [] }
    } as never);

    expect(filtered.input.slice(0, -1)).toEqual([
      input[0],
      input[1],
      input[2],
      input[4],
      input[6]
    ]);
    expect(providerEvents).toContainEqual(expect.objectContaining({
      status: "context_tool_history_normalized",
      source: "openai_tool_message_invariant",
      removed_tool_items: 2,
      incomplete_function_calls: 1,
      orphan_function_results: 1
    }));
    expect(manager.snapshot.scopes[node.id]).toMatchObject({
      raw_item_count: 5,
      retained_item_count: 5
    });
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

    await manager.recordModelInputUsage(node.id, baseline);
    expect(manager.snapshot.scopes[node.id]).toMatchObject({
      token_estimator_correction_milli: 1_000,
      active_estimated_tokens: baseline
    });
    expect(journal).toContainEqual(expect.objectContaining({
      type: "context_token_estimator_calibrated",
      estimated_input_tokens: baseline,
      reported_input_tokens: baseline,
      previous_correction_milli: 2_000,
      correction_milli: 1_000
    }));
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
    let persisted = [
      ...structuredClone(physical),
      structuredClone(filtered.input.at(-1)!),
      completion
    ];
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
    expect(isHarnessAuthorityItem(filtered.input.at(-1))).toBe(true);
    expect(persisted).toEqual([
      ...filtered.input.filter((item) => !isHarnessAuthorityItem(item)),
      completion
    ]);
    const next = await manager.filter({
      ...request,
      modelData: { ...request.modelData, input: persisted }
    });
    expect(next.instructions).toBe(filtered.instructions);
    expect(next.input.slice(0, persisted.length)).toEqual(persisted);
    expect(next.input.filter(isHarnessAuthorityItem)).toHaveLength(1);
    expect(manager.snapshot.scopes[node.id]).toMatchObject({
      compaction_count: 1,
      summary_origin: "model",
      summary_world_revision: 0
    });
  });

  it("keeps a completed checkpoint when its SDK request rolls back", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    let worldRevision = 3;
    let compactorCalls = 0;
    const node = taskNode("humanoid-motion-reference", "运动参考");
    const runtime = {
      rootAgentId: "humanoid-coordinator",
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
          compactorCalls += 1;
          return {
            summary: {
              mission_state: "继续当前运动目标。",
              constraints: [],
              decisions: [],
              completed: [],
              pending: ["根据新权威继续规划。"],
              blockers: [],
              next_actions: ["读取当前物理状态。"]
            },
            origin: "model" as const,
            usage: { requests: 1, inputTokens: 500, outputTokens: 40, totalTokens: 540 }
          };
        }
      })
    });
    const physical = [
      { role: "user", content: "x".repeat(3_000) },
      { role: "assistant", content: "旧观察。" },
      { role: "user", content: "保留当前规划轮。" }
    ] as AgentInputItem[];
    const request = {
      modelData: {
        instructions: `${agentInvocationMarker(node.id)}\nPlan motion.`,
        input: physical
      },
      agent: { name: node.name, tools: [] }
    } as never;

    const first = await manager.filter(request);
    expect(compactorCalls).toBe(1);
    expect(JSON.stringify(first.input)).not.toContain("x".repeat(3_000));

    manager.acceptSdkSessionRollback(node.id, physical);
    worldRevision = 4;
    const retried = await manager.filter(request);

    expect(compactorCalls).toBe(1);
    expect(JSON.stringify(retried.input)).not.toContain("x".repeat(3_000));
    expect(JSON.stringify(retried.input)).toContain("world_revision\\\":4");
    expect(manager.snapshot.scopes[node.id]).toMatchObject({
      compaction_count: 1,
      summary_origin: "model",
      summary_world_revision: 4
    });
  });

  it("rehydrates a compacted specialist Session after process interruption", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const journal: unknown[] = [];
    let compactorCalls = 0;
    const node = taskNode("humanoid-motion-reference", "运动参考");
    const runtime = {
      rootAgentId: "humanoid-coordinator",
      signal: undefined,
      store: {
        readJournalTail: async () => ({ total: journal.length }),
        readJournalPage: async (_name: string, from: number, limit: number) => ({
          entries: journal.slice(from, from + limit),
          total: journal.length,
          next: from + limit < journal.length ? from + limit : null
        })
      },
      activeNode: () => node,
      contextAnchor: () => ({
        world_revision: 8,
        coordinator_phase: "plan",
        execution_authority: null
      }),
      contextMemory: () => structuredClone(memory),
      contextWorldIdentity: () => ({ worldRevision: 8 }),
      contextReceipts: () => ({}),
      assertContextSummaryEvidence: () => undefined,
      async updateContextMemory(state: ContextMemoryState, record?: unknown) {
        memory = structuredClone(state);
        if (record !== undefined) journal.push(structuredClone(record));
      },
      async recordCompactionModelCall() {},
      async reconcileCompactionModelCalls() {},
      async recordProvider() {}
    } as unknown as LongRunContextRuntime;
    const configured = providerConfig({
      contextWindowTokens: 8_192,
      compactTriggerTokens: 1_000,
      compactRecentModelTurns: 1,
      compactMaxOutputTokens: 120
    });
    const createManager = () => new LongRunContextManager({
      runtime,
      provider: configured,
      createGenerator: () => ({
        async generate() {
          compactorCalls += 1;
          return {
            summary: {
              mission_state: "继续当前运动目标。",
              constraints: [],
              decisions: [],
              completed: [],
              pending: ["保持当前 Goal。"],
              blockers: [],
              next_actions: ["读取恢复后的实时权威。"]
            },
            origin: "model" as const,
            usage: { requests: 1, inputTokens: 500, outputTokens: 40, totalTokens: 540 }
          };
        }
      })
    });
    const physical = [
      { role: "user", content: "old-prefix:" + "x".repeat(3_000) },
      { role: "assistant", content: "旧运动观察。" },
      { role: "user", content: "当前规划轮。" }
    ] as AgentInputItem[];
    const request = {
      modelData: {
        instructions: `${agentInvocationMarker(node.id)}\nPlan motion.`,
        input: physical
      },
      agent: { name: node.name, tools: [] }
    } as never;

    const originalManager = createManager();
    await originalManager.filter(request);
    await originalManager.filter({
      ...request,
      modelData: {
        ...request.modelData,
        input: [
          ...physical,
          { role: "assistant", content: "abandoned-response-suffix" }
        ] as AgentInputItem[]
      }
    });
    const recovered = createManager();
    recovered.acceptSdkSessionRollback(node.id, physical);
    const filtered = await recovered.filter(request);

    expect(compactorCalls).toBe(1);
    expect(JSON.stringify(filtered.input)).not.toContain("old-prefix:");
    expect(JSON.stringify(filtered.input)).not.toContain("abandoned-response-suffix");
    expect(JSON.stringify(filtered.input)).toContain("当前规划轮");
    expect(filtered.input.filter(isHarnessAuthorityItem)).toHaveLength(1);
    expect(recovered.snapshot.scopes[node.id]).toMatchObject({
      compaction_count: 1,
      summary_origin: "model",
      raw_item_count: 2,
      retained_item_count: 2
    });
  });

  it("attributes a compactor transport failure to the specialist that owns the context", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const providerEvents: unknown[] = [];
    const node = taskNode("humanoid-motion-reference", "运动参考");
    const connectionFailure = Object.assign(new Error("gateway disconnected"), {
      code: "ECONNRESET"
    });
    const runtime = {
      rootAgentId: "humanoid-coordinator",
      signal: undefined,
      store: {},
      activeNode: () => node,
      contextAnchor: () => ({ world_revision: 5 }),
      contextMemory: () => structuredClone(memory),
      contextWorldIdentity: () => ({ worldRevision: 5 }),
      contextReceipts: () => ({}),
      assertContextSummaryEvidence: () => undefined,
      async updateContextMemory(state: ContextMemoryState) {
        memory = structuredClone(state);
      },
      async recordCompactionModelCall() {},
      async reconcileCompactionModelCalls() {},
      async recordProvider(event: unknown) {
        providerEvents.push(event);
      }
    } as unknown as LongRunContextRuntime;
    const manager = new LongRunContextManager({
      runtime,
      provider: providerConfig({
        contextWindowTokens: 8_192,
        compactTriggerTokens: 700,
        compactRecentModelTurns: 1,
        compactMaxOutputTokens: 120
      }),
      createGenerator: () => ({
        async generate() {
          throw new ContextCompactionInterruption("transport failed", {
            cause: connectionFailure,
            usage: { requests: 1, inputTokens: 0, outputTokens: 0, totalTokens: 0 }
          });
        }
      })
    });

    const failure = await manager.filter({
      modelData: {
        instructions: `${agentInvocationMarker(node.id)}\nPlan motion.`,
        input: [
          { role: "user", content: "x".repeat(3_000) },
          { role: "assistant", content: "旧观察。" },
          { role: "user", content: "继续。" }
        ] as AgentInputItem[]
      },
      agent: { name: node.name, tools: [] }
    } as never).catch((error: unknown) => error);

    expect(modelTransportInterruptionAgentIdFrom(failure)).toBe(node.id);
    expect(providerEvents).toContainEqual(expect.objectContaining({
      status: "context_compaction_transport_interrupted",
      agent_id: node.id,
      raw_history_preserved: true,
      automatic_actuation: false
    }));
  });

  it("shrinks a compaction batch from a real response usage ceiling", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const requests: ContextSummaryRequest[] = [];
    const providerEvents: unknown[] = [];
    const node = taskNode("humanoid-coordinator", "协调");
    const sourceProvider = providerConfig({
      contextWindowTokens: 16_384,
      compactTriggerTokens: 4_000,
      compactRecentModelTurns: 1,
      compactMaxOutputTokens: 256
    });
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
      async updateContextMemory(state: ContextMemoryState) {
        memory = structuredClone(state);
      },
      async recordCompactionModelCall() {},
      async reconcileCompactionModelCalls() {},
      async recordProvider(event: unknown) {
        providerEvents.push(event);
      }
    } as unknown as LongRunContextRuntime;
    const manager = new LongRunContextManager({
      runtime,
      provider: sourceProvider,
      compactorProvider: sourceProvider,
      createGenerator: () => ({
        async generate(request) {
          requests.push(request);
          if (requests.length === 1) {
            throw new ContextCompactionCapacityError(
              8_192,
              "tool JSON was truncated at the observed context ceiling",
              {
                usage: {
                  requests: 1,
                  inputTokens: 7_800,
                  outputTokens: 392,
                  totalTokens: 8_192
                }
              }
            );
          }
          return {
            summary: {
              mission_state: "继续当前任务。",
              constraints: [],
              decisions: [],
              completed: [],
              pending: ["保留未完成目标。"],
              blockers: [],
              next_actions: ["读取当前权威状态。"]
            },
            origin: "model",
            usage: { requests: 1, inputTokens: 2_000, outputTokens: 80, totalTokens: 2_080 }
          };
        }
      })
    });
    const input = Array.from({ length: 12 }, (_, index) => ({
      role: index % 2 === 0 ? "user" as const : "assistant" as const,
      content: `${index}:${"x".repeat(2_000)}`
    })) as AgentInputItem[];

    await manager.filter({
      modelData: {
        instructions: `${agentInvocationMarker(node.id)}\nCoordinate.`,
        input
      },
      agent: { name: node.name, tools: [] }
    } as never);

    expect(requests.length).toBeGreaterThanOrEqual(2);
    expect(requests[1]!.maxInputTokens).toBe(compactorInputTokenLimit(8_192, 128));
    expect(requests[1]!.sourceItems.length).toBeLessThan(requests[0]!.sourceItems.length);
    expect(providerEvents).toContainEqual(expect.objectContaining({
      status: "context_compaction_capacity_calibrated",
      configured_context_window_tokens: 16_384,
      observed_context_window_tokens: 8_192,
      calibrated_input_limit_tokens: compactorInputTokenLimit(8_192, 128)
    }));
  });

  it("persists distinct scope budgets and applies one effective compactor envelope", async () => {
    let memory = structuredClone(EmptyContextMemoryState);
    const requests: Array<{ agentId: string; request: ContextSummaryRequest }> = [];
    const nodes = new Map([
      ["humanoid-coordinator", taskNode("humanoid-coordinator", "协调")],
      ["humanoid-motion-reference", taskNode("humanoid-motion-reference", "运动")]
    ]);
    const coordinator = providerConfig({
      contextWindowTokens: 8_192,
      compactTriggerTokens: 5_000,
      compactRecentModelTurns: 3,
      compactMaxOutputTokens: 80
    });
    const motion = providerConfig({
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
      configForAgent: (agentId) => (
        agentId === "humanoid-motion-reference" ? motion : coordinator
      ),
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
        instructions: `${agentInvocationMarker("humanoid-motion-reference")}\nPlan.`,
        input: [
          { role: "user", content: "x".repeat(3_000) },
          { role: "assistant", content: "Observation acknowledged." },
          { role: "user", content: "Keep the newest evidence." }
        ] as AgentInputItem[]
      },
      agent: { name: "运动", tools: [] }
    } as never);

    expect(manager.snapshot.scopes["humanoid-coordinator"]).toMatchObject({
      context_window_tokens: 8_192,
      compact_trigger_tokens: 5_000,
      compact_recent_model_turns: 3,
      compact_max_output_tokens: 80
    });
    expect(manager.snapshot.scopes["humanoid-motion-reference"]).toMatchObject({
      context_window_tokens: 4_096,
      compact_trigger_tokens: 700,
      compact_recent_model_turns: 1,
      compact_max_output_tokens: 120
    });
    expect(manager.snapshot).toMatchObject({
      active_scope_id: "humanoid-motion-reference",
      context_window_tokens: 4_096,
      compact_trigger_tokens: 700,
      compact_recent_model_turns: 1,
      compact_max_output_tokens: 120
    });
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      agentId: "humanoid-motion-reference",
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

function isHarnessAuthorityItem(item: AgentInputItem | undefined): boolean {
  return item !== undefined
    && item.type === "message"
    && "role" in item
    && item.role === "user"
    && "content" in item
    && typeof item.content === "string"
    && item.content.startsWith("CURRENT HARNESS AUTHORITY\n");
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
