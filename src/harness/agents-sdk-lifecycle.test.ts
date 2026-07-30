import {
  RunContext,
  Runner,
  type AgentInputItem,
  type Model,
  type Session
} from "@openai/agents";
import { isDeepStrictEqual } from "node:util";
import { describe, expect, it } from "vitest";
import type { ProviderConfig } from "../config/load.js";
import type { AgentSpec, TaskNode } from "../domain/schema.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import {
  agentInvocationInput,
  createAgentHierarchy,
  DelegationSpecSchema
} from "./agents.js";
import { receiptEvidenceRequirement } from "./evidence-contract.js";
import type { HarnessRuntimeContext } from "./runtime-context.js";

const TEST_PROVIDER: ProviderConfig = {
  protocol: "openai_compatible",
  baseUrl: "https://example.test/v1",
  model: "test-model",
  apiKey: "test-key",
  temperature: 0,
  maxOutputTokens: 512,
  contextWindowTokens: 8192,
  compactTriggerTokens: 2048,
  compactRecentModelTurns: 2,
  compactMaxOutputTokens: 512
};

describe("Agents SDK delegation failure lifecycle", () => {
  it("propagates the original nested 503 without failing the node or duplicating Session input", async () => {
    const unavailable = Object.assign(new Error("upstream unavailable"), { status: 503 });
    const fixture = sdkFixture({ agent_transport: { type: "error", error: unavailable } });
    const spec = leafSpec("Transport worker");

    await expect(fixture.invoke("transport", spec)).rejects.toBe(unavailable);
    await expect(fixture.invoke("transport", spec)).rejects.toBe(unavailable);

    expect(fixture.runtime.failed).toEqual([]);
    expect(fixture.runtime.node("agent_transport").status).toBe("active");
    expect(await fixture.session("agent_transport").getItems()).toEqual([]);
  });

  it("keeps a non-retryable 400 on the normal child failure path", async () => {
    const invalidRequest = Object.assign(new Error("invalid request"), { status: 400 });
    const fixture = sdkFixture({ agent_bad_request: { type: "error", error: invalidRequest } });

    const output = await fixture.invoke("bad_request", leafSpec("Bad request worker"));

    expect(JSON.parse(String(output))).toMatchObject({
      accepted: false,
      code: "child_agent_failed",
      agent_id: "agent_bad_request"
    });
    expect(fixture.runtime.failed).toEqual(["agent_bad_request"]);
    expect(fixture.runtime.node("agent_bad_request").status).toBe("failed");
  });

  it("keeps invalid worker terminal output on the normal child failure path", async () => {
    const fixture = sdkFixture({ agent_invalid_terminal: { type: "invalid_terminal" } });

    const output = await fixture.invoke(
      "invalid_terminal",
      leafSpec("Invalid terminal worker")
    );

    expect(JSON.parse(String(output))).toMatchObject({
      accepted: false,
      code: "child_agent_failed",
      agent_id: "agent_invalid_terminal"
    });
    expect(fixture.runtime.failed).toEqual(["agent_invalid_terminal"]);
  });

  it("does not leak one sibling's transport interruption into a successful sibling", async () => {
    const unavailable = Object.assign(new Error("one sibling unavailable"), { status: 503 });
    const fixture = sdkFixture({
      agent_failing_sibling: { type: "error", error: unavailable },
      agent_successful_sibling: { type: "success" }
    });

    const [failed, succeeded] = await Promise.allSettled([
      fixture.invoke("failing_sibling", leafSpec("Failing sibling")),
      fixture.invoke("successful_sibling", leafSpec("Successful sibling"))
    ]);

    expect(failed).toEqual({ status: "rejected", reason: unavailable });
    expect(succeeded.status).toBe("fulfilled");
    if (succeeded.status === "fulfilled") {
      expect(JSON.parse(String(succeeded.value))).toMatchObject({
        status: "completed",
        summary: "真实子节点已完成"
      });
    }
    expect(fixture.runtime.failed).toEqual([]);
    expect(fixture.runtime.completed).toEqual(["agent_successful_sibling"]);
    expect(await fixture.session("agent_failing_sibling").getItems()).toEqual([]);
    expect((await fixture.session("agent_successful_sibling").getItems()).length).toBeGreaterThan(0);
  });

  it("drains delayed siblings before a transport error leaves one real SDK tool-call batch", async () => {
    const unavailable = Object.assign(new Error("one sibling unavailable"), { status: 503 });
    const failureStarted = deferred<void>();
    const successStarted = deferred<void>();
    const releaseSuccess = deferred<void>();
    const failingSpec = leafSpec("Immediate failing sibling");
    const successfulSpec = leafSpec("Delayed successful sibling");
    const fixture = sdkFixture({
      agent_immediate_failure: {
        type: "error",
        error: unavailable,
        started: () => failureStarted.resolve()
      },
      agent_delayed_success: {
        type: "success",
        started: () => successStarted.resolve(),
        waitFor: releaseSuccess.promise
      }
    }, undefined, [
      { callId: "immediate_failure", spec: failingSpec },
      { callId: "delayed_success", spec: successfulSpec }
    ]);
    const runner = new Runner({
      tracingDisabled: true,
      toolExecution: { maxFunctionToolConcurrency: 4 }
    });
    let surfaced: unknown;
    const running = runner.run(fixture.hierarchy.root, "Run both delegations.", {
      context: new RunContext({ runId: fixture.runtime.runId }),
      maxTurns: 2,
      toolExecution: { maxFunctionToolConcurrency: 4 }
    }).catch((error: unknown) => {
      surfaced = error;
      throw error;
    });

    await Promise.all([failureStarted.promise, successStarted.promise]);
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(surfaced).toBeUndefined();
    expect(fixture.runtime.completed).toEqual([]);

    releaseSuccess.resolve();
    const caught = await running.catch((error: unknown) => error);
    expect(isTransportInterruption(caught)).toBe(true);
    expect(surfaced).toBe(caught);
    expect(fixture.runtime.completed).toEqual(["agent_delayed_success"]);
    expect(fixture.runtime.failed).toEqual([]);
  });

  it("removes one crash-dangling opening input atomically before retrying the worker", async () => {
    const spec = leafSpec("Dangling input worker");
    const fixture = sdkFixture({ agent_dangling_input: { type: "success" } });
    const session = fixture.session("agent_dangling_input");
    const parsedSpec = DelegationSpecSchema.parse(spec) as AgentSpec;
    expect(agentInvocationInput("agent_dangling_input", parsedSpec, fixture.runtime))
      .toBe(agentInvocationInput("agent_dangling_input", spec, fixture.runtime));
    const opening: AgentInputItem = {
      type: "message",
      role: "user",
      content: agentInvocationInput("agent_dangling_input", spec, fixture.runtime)
    };
    await session.addItems([opening]);

    await fixture.invoke("dangling_input", spec);

    const items = await session.getItems();
    expect(session.replaceCalls).toBe(1);
    expect(items.filter((item) => isDeepStrictEqual(item, opening))).toHaveLength(1);
    expect(fixture.runtime.completed).toEqual(["agent_dangling_input"]);
  });

  it("propagates an operator abort and restores the nested Session baseline", async () => {
    const controller = new AbortController();
    const stopped = new Error("operator stopped");
    const fixture = sdkFixture({
      agent_aborted: { type: "abort", controller, reason: stopped }
    }, controller.signal);

    await expect(fixture.invoke("aborted", leafSpec("Aborted worker"))).rejects.toBe(stopped);

    expect(fixture.runtime.failed).toEqual([]);
    expect(await fixture.session("agent_aborted").getItems()).toEqual([]);
  });
});

type ModelBehavior =
  | { type: "success"; started?: () => void; waitFor?: Promise<void> }
  | { type: "invalid_terminal" }
  | { type: "error"; error: Error; started?: () => void }
  | { type: "abort"; controller: AbortController; reason: Error };

function sdkFixture(
  behaviors: Record<string, ModelBehavior>,
  signal?: AbortSignal,
  rootDelegations?: Array<{ callId: string; spec: AgentSpec }>
) {
  const runtime = new FakeDelegationRuntime(signal);
  const sessions = new Map<string, MemorySession>();
  const session = (agentId: string): MemorySession => {
    const existing = sessions.get(agentId);
    if (existing) return existing;
    const created = new MemorySession(`session:${agentId}`);
    sessions.set(agentId, created);
    return created;
  };
  const hierarchy = createAgentHierarchy({
    createModel: () => routingModel(behaviors, rootDelegations),
    createSession: session,
    provider: TEST_PROVIDER,
    runtime: runtime as unknown as HarnessRuntimeContext
  });
  const delegate = hierarchy.root.tools.find((tool) => tool.name === "delegate_agent");
  if (!delegate || delegate.type !== "function") throw new Error("Delegation tool is missing");

  return {
    hierarchy,
    runtime,
    session,
    invoke(callId: string, spec: AgentSpec) {
      const input = JSON.stringify(spec);
      return delegate.invoke(
        new RunContext({ runId: runtime.runId }),
        input,
        {
          toolCall: {
            type: "function_call",
            callId,
            name: "delegate_agent",
            arguments: input,
            status: "completed"
          },
          ...(signal ? { signal } : {})
        }
      );
    }
  };
}

function routingModel(
  behaviors: Record<string, ModelBehavior>,
  rootDelegations?: Array<{ callId: string; spec: AgentSpec }>
): Model {
  let rootRequests = 0;
  return {
    getResponse: async () => {
      if (!rootDelegations) {
        throw new Error("Delegated Agent.asTool must use the streaming SDK path");
      }
      rootRequests += 1;
      if (rootRequests > 1) throw new Error("Root requested another model turn after interruption");
      return {
        usage: { requests: 1, inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        output: rootDelegations.map(({ callId, spec }) => ({
          type: "function_call" as const,
          callId,
          name: "delegate_agent",
          arguments: JSON.stringify(spec),
          status: "completed" as const
        }))
      };
    },
    getStreamedResponse: (request) => (async function* () {
      const agentId = Object.keys(behaviors).find((candidate) =>
        JSON.stringify(request).includes(`HEAR_AGENT_INVOCATION_V1:${candidate}`)
      );
      if (!agentId) throw new Error("Model request has no routed hierarchy identity");
      const behavior = behaviors[agentId]!;
      if (behavior.type === "error") {
        behavior.started?.();
        throw behavior.error;
      }
      if (behavior.type === "abort") {
        behavior.controller.abort(behavior.reason);
        throw behavior.reason;
      }
      if (behavior.type === "success") {
        behavior.started?.();
        await behavior.waitFor;
      }
      const output = behavior.type === "success"
        ? [{
            type: "function_call" as const,
            callId: `complete_${agentId}`,
            name: "complete_assignment",
            arguments: JSON.stringify({
              status: "completed",
              summary: "真实子节点已完成",
              evidence: [{ criterion_index: 0, transaction_ids: [`${agentId}:receipt`] }],
              unmet_criteria: []
            }),
            status: "completed" as const
          }]
        : [{
            type: "message" as const,
            role: "assistant" as const,
            status: "completed" as const,
            content: [{ type: "output_text" as const, text: "missing terminal tool" }]
          }];
      yield {
        type: "response_done" as const,
        response: {
          id: `response_${agentId}`,
          usage: {
            requests: 1,
            inputTokens: 1,
            outputTokens: 1,
            totalTokens: 2
          },
          output
        }
      };
    })()
  } as Model;
}

function deferred<T>() {
  let resolve: (value: T | PromiseLike<T>) => void = () => undefined;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

function leafSpec(name: string): AgentSpec {
  return {
    name,
    objective: "Return one real receipt-backed observation.",
    success_criteria: ["The assigned observation is complete."],
    evidence_requirements: [receiptEvidenceRequirement(
      0,
      "sense_scene",
      { kind: "world" }
    )],
    goal_predicate_indexes: [],
    capabilities: ["sense_scene"],
    may_delegate: false,
    references: []
  };
}

class MemorySession implements Session {
  readonly #id: string;
  #items: AgentInputItem[] = [];
  replaceCalls = 0;

  constructor(id: string) {
    this.#id = id;
  }

  async getSessionId(): Promise<string> {
    return this.#id;
  }

  async getItems(limit?: number): Promise<AgentInputItem[]> {
    const selected = limit === undefined ? this.#items : this.#items.slice(-limit);
    return structuredClone(selected);
  }

  async addItems(items: AgentInputItem[]): Promise<void> {
    this.#items = this.#items.concat(structuredClone(items));
  }

  async popItem(): Promise<AgentInputItem | undefined> {
    return this.#items.pop();
  }

  async clearSession(): Promise<void> {
    this.#items = [];
  }

  async replaceItems(items: AgentInputItem[]): Promise<void> {
    this.replaceCalls += 1;
    this.#items = structuredClone(items);
  }
}

class FakeDelegationRuntime {
  readonly runId = "run_sdk_delegation_lifecycle";
  readonly rootAgentId = "root";
  readonly signal: AbortSignal | undefined;
  readonly failed: string[] = [];
  readonly completed: string[] = [];
  readonly #nodes = new Map<string, TaskNode>();
  readonly #calls = new Map<string, string>();

  constructor(signal?: AbortSignal) {
    this.signal = signal;
    this.#nodes.set(this.rootAgentId, taskNode(this.rootAgentId, null, {
      name: "Mission Coordinator",
      objective: "Coordinate the test mission.",
      success_criteria: ["The mission is complete."],
      evidence_requirements: [],
      goal_predicate_indexes: [],
      capabilities: ["sense_scene", "read_proprioception"],
      may_delegate: true,
      references: []
    }));
  }

  node(agentId: string): TaskNode {
    const node = this.#nodes.get(agentId);
    if (!node) throw new Error(`Unknown test node: ${agentId}`);
    return structuredClone(node);
  }

  goal() {
    return {
      summary: "Exercise SDK delegation lifecycle.",
      predicates: [{ type: "terrain_explored" as const, minimum_fraction: 1 }]
    };
  }

  referencedReceipts() { return []; }
  acceptedActionReferences() { return []; }
  checkerSatisfiedCurrentWorld() { return false; }
  canDelegate(parent: AgentSpec | null) { return parent === null || parent.may_delegate; }
  isCapabilityEnabled() { return false; }
  frameworkScope(agentId: string) { return `agent:${agentId}`; }
  async recordFramework() {}
  async recordProvider() {}
  async recordModelCallStarted() {}
  worldIdentity() { return { world_frame: 0, world_revision: 0 }; }

  activeNode(agentId?: string): TaskNode {
    return this.node(agentId ?? this.rootAgentId);
  }

  async beginDelegation(
    _parent: AgentSpec | null,
    spec: AgentSpec,
    sourceCallId: string
  ) {
    const existingId = this.#calls.get(sourceCallId);
    if (existingId) return { node: this.node(existingId), created: false };
    const id = `agent_${sourceCallId}`;
    const node = taskNode(id, this.rootAgentId, spec, sourceCallId);
    this.#nodes.set(id, node);
    this.#calls.set(sourceCallId, id);
    return { node: this.node(id), created: true };
  }

  assertChildEvidence() { return []; }

  async completeChild(agentId: string) {
    const node = this.#nodes.get(agentId)!;
    node.status = "completed";
    this.completed.push(agentId);
  }

  async failChild(agentId: string, error: string) {
    const node = this.#nodes.get(agentId)!;
    node.status = "failed";
    node.last_result = { error };
    this.failed.push(agentId);
  }

  async blockChild(agentId: string, reason: string) {
    const node = this.#nodes.get(agentId)!;
    node.status = "blocked";
    node.last_result = { blocked: reason };
  }
}

function taskNode(
  id: string,
  parentId: string | null,
  spec: AgentSpec,
  sourceCallId?: string
): TaskNode {
  const now = new Date().toISOString();
  return {
    id,
    name: spec.name,
    parent_id: parentId,
    child_ids: [],
    objective: spec.objective,
    success_criteria: [...spec.success_criteria],
    evidence_requirements: structuredClone(spec.evidence_requirements),
    goal_predicate_indexes: [...spec.goal_predicate_indexes],
    capabilities: [...spec.capabilities],
    may_delegate: spec.may_delegate,
    references: structuredClone(spec.references),
    depth: parentId === null ? 0 : 1,
    ...(sourceCallId ? { source_call_id: sourceCallId } : {}),
    status: "active",
    steps_used: 0,
    model_calls_used: 0,
    created_at: now,
    updated_at: now
  };
}
