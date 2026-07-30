import { RunContext, type Model } from "@openai/agents";
import { describe, expect, it } from "vitest";
import type { ProviderConfig } from "../config/load.js";
import type { AgentSpec } from "../domain/schema.js";
import {
  DelegationSpecSchema,
  ModelDecisionGuard,
  ModelDecisionStallError,
  WorkerOutcomeSchema,
  agentToolTopology,
  capabilityCatalog,
  createAgentHierarchy,
  rethrowDelegationInterruption
} from "./agents.js";
import { coordinatorInstructions, workerInstructions } from "./agent-prompts.js";
import { receiptEvidenceRequirement } from "./evidence-contract.js";
import type { HarnessRuntimeContext } from "./runtime-context.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";

const provider: ProviderConfig = {
  protocol: "openai_compatible",
  baseUrl: "https://example.test/v1",
  model: "initial-model",
  apiKey: "test-key",
  temperature: 0.2,
  maxOutputTokens: 1024,
  contextWindowTokens: 16_384,
  compactTriggerTokens: 4096,
  compactRecentModelTurns: 4,
  compactMaxOutputTokens: 1024
};

describe("agent tool topology", () => {
  it("requires a typed evidence contract for every bounded worker criterion", () => {
    const input = {
      name: "Proprioception leaf",
      objective: "Read current proprioception and return its accepted receipt.",
      success_criteria: ["Current robot state is observed."],
      evidence_requirements: [receiptEvidenceRequirement(
        0,
        "read_proprioception",
        { kind: "robot" }
      )],
      goal_predicate_indexes: [],
      capabilities: ["read_proprioception"],
      may_delegate: false
    };
    expect(DelegationSpecSchema.parse(input)).toEqual({ ...input, references: [] });
    expect(DelegationSpecSchema.safeParse({
      ...input,
      evidence_requirements: []
    }).success).toBe(false);
  });

  it("keeps root coordination-only and gives workers scoped actions plus recursive delegation", () => {
    const capabilities = capabilityCatalog();
    const topology = agentToolTopology();

    expect(topology.root).toEqual([
      "delegate_agent",
      "check_mission",
      "complete_mission"
    ]);
    expect(topology.root.filter((name) => capabilities.includes(name))).toEqual([]);
    expect(topology.worker).toEqual([
      ...capabilities,
      "delegate_agent",
      "complete_assignment",
      "report_blocked"
    ]);
    expect(topology.worker).not.toContain("check_mission");
  });

  it("requires worker evidence to reference real transaction identifiers by criterion", () => {
    const common = {
      status: "completed" as const,
      summary: "The assigned state was observed.",
      unmet_criteria: []
    };

    expect(WorkerOutcomeSchema.safeParse({
      ...common,
      evidence: [{
        criterion_index: 0,
        transaction_ids: ["agent_1:action_1"]
      }]
    }).success).toBe(true);
    expect(WorkerOutcomeSchema.safeParse({
      ...common,
      evidence: {}
    }).success).toBe(false);
  });

  it("creates one SDK Agent instance per concrete hierarchy node", async () => {
    let checkerSatisfied = false;
    const runtime = {
      runId: "run_agent_instances",
      rootAgentId: "root_agent",
      signal: undefined,
      goal: () => ({ summary: "Test goal", predicates: [] }),
      referencedReceipts: () => [],
      checkerSatisfiedCurrentWorld: () => checkerSatisfied,
      canDelegate: () => true
    } as unknown as HarnessRuntimeContext;
    const createdModels: Model[] = [];
    const createModel = (): Model => {
      const model = {
        getResponse: async () => { throw new Error("Model calls are outside this construction test"); },
        getStreamedResponse: () => { throw new Error("Model calls are outside this construction test"); }
      } as unknown as Model;
      createdModels.push(model);
      return model;
    };
    const hierarchy = createAgentHierarchy({ createModel, provider, runtime });
    const spec = (name: string): AgentSpec => ({
      name,
      objective: `Execute the bounded assignment for ${name}.`,
      success_criteria: ["One receipt-backed outcome is returned."],
      evidence_requirements: [receiptEvidenceRequirement(
        0,
        "sense_scene",
        { kind: "world" }
      )],
      goal_predicate_indexes: [],
      capabilities: ["sense_scene"],
      may_delegate: false,
      references: []
    });

    const first = hierarchy.createWorker("agent_a", spec("Observer A"));
    const second = hierarchy.createWorker("agent_b", spec("Observer B"));
    const firstAgain = hierarchy.createWorker("agent_a", spec("Observer A"));

    expect(first).not.toBe(second);
    expect(firstAgain).toBe(first);
    expect(first.model).not.toBe(second.model);
    expect(first.model).not.toBe(hierarchy.root.model);
    expect(createdModels).toHaveLength(3);
    expect(first.name).toBe("Observer A");
    expect(second.name).toBe("Observer B");
    expect(first.tools).not.toBe(second.tools);
    expect(first.tools.map((entry) => entry.name)).toEqual(agentToolTopology().worker);
    expect(second.tools.map((entry) => entry.name)).toEqual(agentToolTopology().worker);
    expect(hierarchy.root.modelSettings.parallelToolCalls).toBe(true);
    expect(first.modelSettings.parallelToolCalls).toBe(true);
    expect(() => hierarchy.createWorker("agent_a", spec("Changed observer"))).toThrow(
      "cannot change its agent specification"
    );

    const runContext = new RunContext({ runId: runtime.runId });
    const rootTools = Object.fromEntries(
      hierarchy.root.tools.map((entry) => [entry.name, entry])
    );
    expect(await rootTools.delegate_agent?.isEnabled(runContext, hierarchy.root)).toBe(true);
    expect(await rootTools.check_mission?.isEnabled(runContext, hierarchy.root)).toBe(true);
    expect(await rootTools.complete_mission?.isEnabled(runContext, hierarchy.root)).toBe(false);

    checkerSatisfied = true;
    expect(await rootTools.delegate_agent?.isEnabled(runContext, hierarchy.root)).toBe(false);
    expect(await rootTools.check_mission?.isEnabled(runContext, hierarchy.root)).toBe(false);
    expect(await rootTools.complete_mission?.isEnabled(runContext, hierarchy.root)).toBe(true);
  });

  it("preserves retry metadata from non-Error model rejections", async () => {
    const runtime = {
      runId: "run_transport_metadata",
      rootAgentId: "root_agent",
      signal: undefined,
      activeNode: () => ({ id: "root_agent" }),
      recordModelCallStarted: async () => undefined
    } as unknown as HarnessRuntimeContext;
    const hierarchy = createAgentHierarchy({
      createModel: () => ({
        getResponse: async () => {
          throw { statusCode: 503, error: { message: "upstream unavailable" } };
        },
        getStreamedResponse: () => {
          throw new Error("Streaming is outside this test");
        }
      }) as unknown as Model,
      provider,
      runtime
    });

    let caught: unknown;
    try {
      await hierarchy.root.model.getResponse({ input: [] } as never);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(Error);
    expect(caught).toMatchObject({ name: "ModelTransportError", statusCode: 503 });
    expect(isTransportInterruption(caught)).toBe(true);
  });

  it("rethrows nested transport and abort failures before business handling", () => {
    const unavailable = Object.assign(new Error("upstream unavailable"), { status: 503 });
    expect(() => rethrowDelegationInterruption(unavailable)).toThrow(unavailable);

    const controller = new AbortController();
    const stopped = new Error("operator stopped");
    controller.abort(stopped);
    expect(() => rethrowDelegationInterruption(new Error("nested run stopped"), controller.signal))
      .toThrow(stopped);

    expect(() => rethrowDelegationInterruption(new Error("invalid terminal outcome")))
      .not.toThrow();
  });

  it("permits only supervisors to fan out independent model-selected delegations", () => {
    const runtime = {
      referencedReceipts: () => []
    } as unknown as HarnessRuntimeContext;
    const supervisor: AgentSpec = {
      name: "Body supervisor",
      objective: "Coordinate independent body outcomes.",
      success_criteria: ["Every outcome has receipt evidence."],
      evidence_requirements: [{
        kind: "goal_predicate",
        criterion_index: 0,
        predicate_index: 0
      }],
      goal_predicate_indexes: [0],
      capabilities: ["drive_base", "set_head_target", "set_joint_targets"],
      may_delegate: true,
      references: []
    };
    const leaf: AgentSpec = {
      ...supervisor,
      name: "Head leaf",
      objective: "Move the head from live evidence.",
      evidence_requirements: [receiptEvidenceRequirement(
        0,
        "set_head_target",
        { kind: "body", channel: "head" }
      )],
      goal_predicate_indexes: [],
      capabilities: ["read_proprioception", "set_head_target"],
      may_delegate: false
    };

    expect(coordinatorInstructions()).toContain(
      "several delegate_agent calls when their outcomes are independent"
    );
    expect(coordinatorInstructions()).toContain(
      "do not mix check_mission or complete_mission with delegation calls"
    );
    expect(workerInstructions(supervisor, runtime)).toContain(
      "several delegate_agent calls only for independent outcomes"
    );
    expect(workerInstructions(leaf, runtime)).toContain(
      "Every response must invoke exactly one currently enabled formal tool"
    );
    expect(workerInstructions(leaf, runtime)).toContain(
      "the very next response must call complete_assignment"
    );
    expect(workerInstructions(leaf, runtime)).not.toContain(
      "several delegate_agent calls only for independent outcomes"
    );
  });

});

describe("model decision boundary", () => {
  it("allows an isolated reasoning-only response but rejects an unbounded non-decision loop", () => {
    const guard = new ModelDecisionGuard();
    const empty = [{ type: "reasoning", content: [] }] as never;

    guard.observe("agent_a", empty);
    guard.observe("agent_a", empty);
    guard.observe("agent_a", empty);
    expect(() => guard.observe("agent_a", empty)).toThrow(/4 consecutive responses/);
  });

  it("does not mistake repeated prose for an embodied decision", () => {
    const guard = new ModelDecisionGuard();
    const prose = [{
      type: "message",
      content: [{ type: "output_text", text: "I will move next." }]
    }] as never;

    guard.observe("agent_a", prose);
    guard.observe("agent_a", prose);
    guard.observe("agent_a", prose);
    expect(() => guard.observe("agent_a", prose)).toThrow(/required tool decision/);
  });

  it("resets after a real tool decision", () => {
    const guard = new ModelDecisionGuard();
    const empty = [{ type: "reasoning", content: [] }] as never;
    const decision = [{ type: "function_call" }] as never;

    guard.observe("agent_a", empty);
    guard.observe("agent_a", empty);
    guard.observe("agent_a", empty);
    guard.observe("agent_a", decision);
    guard.observe("agent_a", empty);
    guard.observe("agent_a", empty);
    guard.observe("agent_a", empty);
    expect(() => guard.observe("agent_a", decision)).not.toThrow();
  });

  it("does not carry a failed child's non-decisions into its parent", () => {
    const guard = new ModelDecisionGuard();
    const empty = [{ type: "reasoning", content: [] }] as never;

    guard.observe("child", empty);
    guard.observe("child", empty);
    guard.observe("child", empty);
    expect(() => guard.observe("parent", empty)).not.toThrow();
  });

  it("identifies and resets a stalled hierarchy node for a fresh model turn", () => {
    const guard = new ModelDecisionGuard();
    const empty = [{ type: "reasoning", content: [] }] as never;
    for (let index = 0; index < 3; index += 1) guard.observe("root", empty);
    try {
      guard.observe("root", empty);
      throw new Error("expected a decision stall");
    } catch (error) {
      expect(error).toBeInstanceOf(ModelDecisionStallError);
      expect((error as ModelDecisionStallError).agentId).toBe("root");
    }
    for (let index = 0; index < 3; index += 1) {
      expect(() => guard.observe("root", empty)).not.toThrow();
    }
  });

  it("supports a stricter coordinator threshold without changing worker defaults", () => {
    const guard = new ModelDecisionGuard(2);
    const empty = [{ type: "reasoning", content: [] }] as never;

    guard.observe("root", empty);
    expect(() => guard.observe("root", empty)).toThrow(/2 consecutive responses/);
  });
});
