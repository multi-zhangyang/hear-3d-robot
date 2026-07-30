import { Agent, type AgentInputItem } from "@openai/agents";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog, type ProviderConfig } from "../config/load.js";
import type { AgentSpec, Goal, Scenario } from "../domain/schema.js";
import { FileSession } from "../persistence/file-session.js";
import { RunStore } from "../persistence/run-store.js";
import {
  CONTEXT_COMPACTOR_MAX_TURNS,
  compactorInputTokenLimit
} from "../runtime/context-budget.js";
import { RapierWorld } from "../world/rapier-world.js";
import { capabilityCatalog } from "./agents.js";
import { agentInvocationMarker } from "./agent-scope.js";
import {
  estimateToolTokens,
  LongRunContextManager,
  sequencePrefixIndex
} from "./context-compaction.js";
import { receiptEvidenceRequirement } from "./evidence-contract.js";
import {
  estimateContextSummaryRequestTokens,
  type ContextSummaryGenerator,
  type ContextSummaryRequest,
  type ContextSummaryResult
} from "./context-summary-agent.js";
import { HierarchyProjection } from "./hierarchy-projection.js";
import { createCheckpoint, HarnessRuntimeContext } from "./runtime-context.js";

const provider: ProviderConfig = {
  protocol: "openai_compatible",
  baseUrl: "https://example.test/v1",
  model: "test-model",
  apiKey: "test-key",
  temperature: 0.2,
  maxOutputTokens: 512,
  contextWindowTokens: 12_288,
  compactTriggerTokens: 1600,
  compactRecentModelTurns: 0,
  compactMaxOutputTokens: 256
};

class RecordingGenerator implements ContextSummaryGenerator {
  readonly requests: ContextSummaryRequest[] = [];
  readonly #transactionId: string | undefined;
  readonly #origin: ContextSummaryResult["origin"];

  constructor(
    transactionId?: string,
    origin: ContextSummaryResult["origin"] = "model"
  ) {
    this.#transactionId = transactionId;
    this.#origin = origin;
  }

  async generate(request: ContextSummaryRequest): Promise<ContextSummaryResult> {
    this.requests.push(structuredClone(request));
    return {
      summary: {
        mission_state: "The hierarchy is still working on the operator mission.",
        constraints: ["Only committed receipts prove physical work."],
        decisions: ["Continue from current harness authority."],
        completed: this.#transactionId
          ? [{ summary: "Prior work completed", transaction_ids: [this.#transactionId] }]
          : [],
        pending: ["Take the next source-backed action."],
        blockers: [],
        next_actions: ["Observe the current world before actuation."]
      },
      origin: this.#origin,
      ...(this.#origin === "authority_projection"
        ? { fallbackReason: "Configured model omitted the checkpoint tool" }
        : {}),
      usage: { requests: 1, inputTokens: 2400, outputTokens: 120, totalTokens: 2520 }
    };
  }
}

class NodeRecordingGenerator extends RecordingGenerator {
  readonly modelInstance = {};

  constructor(readonly nodeId: string) {
    super();
  }
}

describe("long-run context compaction", () => {
  it("uses the newest repeated Session sequence when rotating compacted history", () => {
    const turn: AgentInputItem[] = [
      { role: "user", content: "repeat" },
      { role: "assistant", status: "completed", content: [{ type: "output_text", text: "same" }] }
    ];
    expect(sequencePrefixIndex([...turn, ...turn], turn)).toBe(turn.length);
  });

  it("budgets the tool surface that the provider receives outside modelData", () => {
    expect(estimateToolTokens([
      { name: "survey_terrain", description: "Observe reachable frontier choices." },
      { name: "execute_base_plan", description: "Execute one accepted physical path." }
    ])).toBeGreaterThan(320);
  });

  it("injects current harness authority before any compaction is needed", async () => {
    const fixture = await createFixture();
    try {
      const generator = new RecordingGenerator();
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider: { ...provider, compactTriggerTokens: 7000 },
        createGenerator: () => generator
      });
      const filtered = await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: {
          input: [{ role: "user", content: "Continue from the live world." }],
          instructions: "Coordinate."
        }
      });

      expect(generator.requests).toHaveLength(0);
      expect(filtered.instructions).toContain("CURRENT HARNESS AUTHORITY");
      expect(filtered.instructions).toContain('"goal_state"');
      expect(filtered.instructions).toContain('"world_revision":0');
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("recursively compacts a long source in complete bounded model batches", async () => {
    const fixture = await createFixture();
    try {
      const generator = new RecordingGenerator();
      const chunkedProvider = {
        ...provider,
        compactRecentModelTurns: 0
      };
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider: chunkedProvider,
        createGenerator: () => generator
      });
      const items = chunkedHistory();
      const filtered = await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: items, instructions: "Coordinate." }
      });

      const inputLimit = compactorInputTokenLimit(
        chunkedProvider.contextWindowTokens,
        chunkedProvider.compactMaxOutputTokens
      );
      expect(generator.requests.length).toBeGreaterThan(1);
      for (const request of generator.requests) {
        const estimated = estimateContextSummaryRequestTokens(request);
        expect(request.maxInputTokens).toBe(inputLimit);
        expect(estimated).toBeLessThanOrEqual(inputLimit);
        expect(
          estimated
          + chunkedProvider.compactMaxOutputTokens * CONTEXT_COMPACTOR_MAX_TURNS
        ).toBeLessThanOrEqual(chunkedProvider.contextWindowTokens);
      }
      expect(generator.requests.flatMap((request) => request.sourceItems)).toEqual(items);
      expect(filtered.input).toEqual([]);

      const persisted = await fixture.store.readCheckpoint();
      expect(persisted.context_memory.scopes[persisted.root_id]).toMatchObject({
        raw_item_count: 0,
        compacted_item_count: 0,
        compaction_count: generator.requests.length
      });
      const contextJournal = await fixture.store.readJournal("context") as Array<Record<string, unknown>>;
      expect(contextJournal.filter((record) => record.type === "context_compacted"))
        .toHaveLength(generator.requests.length);
      expect(contextJournal.at(-1)).toMatchObject({
        type: "context_history_compacted",
        compacted_items: items.length,
        to_item: 0,
        items: []
      });
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("only exposes accepted receipts to completed memory evidence", async () => {
    const fixture = await createFixture();
    try {
      const child = await fixture.runtime.beginDelegation(null, {
        name: "Memory evidence worker",
        objective: "Produce one accepted and one rejected receipt for memory validation.",
        success_criteria: ["Both receipt dispositions are recorded."],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "read_proprioception",
          { kind: "robot" }
        )],
        capabilities: ["read_proprioception", "plan_base_path"],
        may_delegate: false,
        references: []
      }, "memory_evidence_worker");
      const accepted = JSON.parse(await fixture.runtime.invokeTool(
        "read_proprioception",
        {},
        "accepted_memory_source",
        child.node.id
      )) as { transaction_id: string; accepted: boolean };
      const position = fixture.runtime.checkpoint.world.robot.position;
      const rejected = JSON.parse(await fixture.runtime.invokeTool(
        "plan_base_path",
        { target: position, face_point: position },
        "rejected_memory_source",
        child.node.id
      )) as { transaction_id: string; accepted: boolean };
      expect(accepted.accepted).toBe(true);
      expect(rejected.accepted).toBe(false);

      const generator = new RecordingGenerator();
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: () => generator
      });
      const items = longHistory();
      items[0] = {
        role: "user",
        content: `Accepted ${accepted.transaction_id}; rejected blocker ${rejected.transaction_id}.`
      };
      await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: items, instructions: "Coordinate." }
      });

      expect(generator.requests).toHaveLength(1);
      expect(generator.requests[0]?.acceptedTransactionIds).toEqual([accepted.transaction_id]);
      expect(generator.requests[0]?.blockerTransactionIds).toEqual([
        accepted.transaction_id,
        rejected.transaction_id
      ]);
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("persists concurrent sibling scopes without one checkpoint overwriting the other", async () => {
    const fixture = await createFixture();
    try {
      const spec = (name: string): AgentSpec => ({
        name,
        objective: `Maintain the independent context for ${name}.`,
        success_criteria: ["The node retains its own model history."],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "read_proprioception",
          { kind: "robot" }
        )],
        capabilities: ["read_proprioception"],
        may_delegate: false,
        references: []
      });
      const first = await fixture.runtime.beginDelegation(
        null,
        spec("Memory worker A"),
        "memory_a"
      );
      const second = await fixture.runtime.beginDelegation(
        null,
        spec("Memory worker B"),
        "memory_b",
        fixture.runtime.rootAgentId
      );
      const roomyProvider = { ...provider, compactTriggerTokens: 7000 };
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider: roomyProvider,
        createGenerator: () => new RecordingGenerator()
      });
      const inputFor = (agentId: string): AgentInputItem[] => [{
        role: "user",
        content: `${agentInvocationMarker(agentId)}\nKeep this node's durable context separate.`
      }];

      await Promise.all([
        manager.filter({
          agent: new Agent({ name: "Capability Worker", instructions: "Work independently." }),
          context: { runId: fixture.runtime.runId },
          modelData: { input: inputFor(first.node.id), instructions: "Work independently." }
        }),
        manager.filter({
          agent: new Agent({ name: "Capability Worker", instructions: "Work independently." }),
          context: { runId: fixture.runtime.runId },
          modelData: { input: inputFor(second.node.id), instructions: "Work independently." }
        })
      ]);

      const persisted = await fixture.store.readCheckpoint();
      expect(persisted.context_memory.scopes[first.node.id]).toMatchObject({
        agent_id: first.node.id,
        raw_item_count: 1
      });
      expect(persisted.context_memory.scopes[second.node.id]).toMatchObject({
        agent_id: second.node.id,
        raw_item_count: 1
      });
      const records = await fixture.store.readJournal("context") as Array<Record<string, unknown>>;
      expect(records.filter((record) => record.type === "context_history_delta"))
        .toHaveLength(2);
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("keeps one model-backed compactor per hierarchy node without sharing context", async () => {
    const fixture = await createFixture();
    try {
      const spec = (name: string): AgentSpec => ({
        name,
        objective: `Maintain the independent compact context for ${name}.`,
        success_criteria: ["The node retains its own compacted model history."],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "read_proprioception",
          { kind: "robot" }
        )],
        capabilities: ["read_proprioception"],
        may_delegate: false,
        references: []
      });
      const first = await fixture.runtime.beginDelegation(
        null,
        spec("Compaction worker A"),
        "compact_model_a"
      );
      const second = await fixture.runtime.beginDelegation(
        null,
        spec("Compaction worker B"),
        "compact_model_b",
        fixture.runtime.rootAgentId
      );
      const factoryCalls: string[] = [];
      const generators = new Map<string, NodeRecordingGenerator>();
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: (agentId) => {
          factoryCalls.push(agentId);
          const generator = new NodeRecordingGenerator(agentId);
          generators.set(agentId, generator);
          return generator;
        }
      });
      const historyFor = (agentId: string, privateText: string): AgentInputItem[] => [
        {
          role: "user",
          content: `${agentInvocationMarker(agentId)}\n${privateText}`
        },
        ...Array.from({ length: 18 }, (_, index): AgentInputItem => ({
          role: "assistant",
          status: "completed",
          content: [{
            type: "output_text",
            text: `${privateText} turn ${index + 1}. ${"evidence ".repeat(850)}`
          }]
        }))
      ];

      await Promise.all([
        manager.filter({
          agent: new Agent({ name: "Capability Worker", instructions: "Work independently." }),
          context: { runId: fixture.runtime.runId },
          modelData: {
            input: historyFor(first.node.id, "NODE_A_PRIVATE_CONTEXT"),
            instructions: "Work independently."
          }
        }),
        manager.filter({
          agent: new Agent({ name: "Capability Worker", instructions: "Work independently." }),
          context: { runId: fixture.runtime.runId },
          modelData: {
            input: historyFor(second.node.id, "NODE_B_PRIVATE_CONTEXT"),
            instructions: "Work independently."
          }
        })
      ]);

      const firstGenerator = generators.get(first.node.id);
      const secondGenerator = generators.get(second.node.id);
      expect(firstGenerator).toBeDefined();
      expect(secondGenerator).toBeDefined();
      expect(firstGenerator?.nodeId).toBe(first.node.id);
      expect(secondGenerator?.nodeId).toBe(second.node.id);
      expect(firstGenerator).not.toBe(secondGenerator);
      expect(firstGenerator?.modelInstance).not.toBe(secondGenerator?.modelInstance);
      expect(factoryCalls.filter((agentId) => agentId === first.node.id)).toHaveLength(1);
      expect(factoryCalls.filter((agentId) => agentId === second.node.id)).toHaveLength(1);
      expect(firstGenerator!.requests.length).toBeGreaterThan(1);
      expect(secondGenerator!.requests.length).toBeGreaterThan(1);
      expect(JSON.stringify(firstGenerator!.requests)).toContain("NODE_A_PRIVATE_CONTEXT");
      expect(JSON.stringify(firstGenerator!.requests)).not.toContain("NODE_B_PRIVATE_CONTEXT");
      expect(JSON.stringify(secondGenerator!.requests)).toContain("NODE_B_PRIVATE_CONTEXT");
      expect(JSON.stringify(secondGenerator!.requests)).not.toContain("NODE_A_PRIVATE_CONTEXT");
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("does not commit a compactor result returned after the mission was aborted", async () => {
    const controller = new AbortController();
    const fixture = await createFixture(controller.signal);
    const interrupted = new Error("lease fencing interrupted compaction");
    const generator = new RecordingGenerator();
    const abortingGenerator: ContextSummaryGenerator = {
      generate: async (request) => {
        const result = await generator.generate(request);
        controller.abort(interrupted);
        return result;
      }
    };
    try {
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: () => abortingGenerator
      });
      await expect(manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: longHistory(), instructions: "Coordinate." }
      })).rejects.toBe(interrupted);

      const persisted = await fixture.store.readCheckpoint();
      expect(persisted.context_memory.scopes[persisted.root_id]).toMatchObject({
        summary: null,
        compaction_count: 0
      });
      expect(await fixture.store.readJournal("context")).toEqual([
        expect.objectContaining({ type: "context_history_delta" })
      ]);
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("uses a model checkpoint, archives raw items, and restores the bounded scope", async () => {
    const fixture = await createFixture();
    let resumedWorld: RapierWorld | undefined;
    try {
      const generator = new RecordingGenerator();
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: () => generator
      });
      const items = longHistory();
      const filtered = await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate the mission." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: items, instructions: "Coordinate the mission." }
      });

      expect(generator.requests).toHaveLength(1);
      expect(generator.requests[0]?.sourceItems).toEqual(items);
      expect(filtered.input).toEqual([]);
      expect(filtered.instructions).toContain("LONG-RUN CONTEXT CHECKPOINT");
      expect(filtered.instructions).toContain("CURRENT HARNESS AUTHORITY");
      expect(filtered.instructions).toContain('"goal_state":{"satisfied":true');

      const persisted = await fixture.store.readCheckpoint();
      const scope = persisted.context_memory.scopes[persisted.root_id];
      expect(persisted.context_memory).toMatchObject({
        total_compactions: 1,
        active_scope_id: persisted.root_id
      });
      expect(scope).toMatchObject({
        raw_item_count: 0,
        compacted_item_count: 0,
        compaction_count: 1,
        summary: { mission_state: "The hierarchy is still working on the operator mission." }
      });

      const journal = await fixture.store.readJournal("context");
      expect(journal).toHaveLength(3);
      expect(journal[0]).toMatchObject({
        type: "context_history_delta",
        from_item: 0,
        to_item: items.length,
        items
      });
      expect(journal[1]).toMatchObject({
        type: "context_compacted",
        source_from_item: 0,
        source_to_item: items.length
      });
      expect(journal[2]).toMatchObject({
        type: "context_history_compacted",
        compacted_items: items.length,
        items: []
      });

      fixture.world.dispose();
      resumedWorld = await RapierWorld.create(fixture.scenario, persisted.world);
      const resumedHierarchy = new HierarchyProjection(
        persisted.nodes,
        persisted.root_id,
        persisted.active_agent_id
      );
      const resumedRuntime = new HarnessRuntimeContext({
        store: fixture.store,
        goal: fixture.goal,
        world: resumedWorld,
        hierarchy: resumedHierarchy,
        checkpoint: persisted
      });
      const resumedManager = new LongRunContextManager({
        runtime: resumedRuntime,
        provider,
        createGenerator: () => generator
      });
      const nextItem: AgentInputItem = {
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "Continue." }]
      };
      const resumedFiltered = await resumedManager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate the mission." }),
        context: { runId: fixture.runtime.runId },
        // A local SDK Session may contain only the post-compaction tail. The
        // persisted checkpoint must rebase onto it without forgetting the
        // model-written summary or replaying the old prefix.
        modelData: { input: [nextItem], instructions: "Coordinate the mission." }
      });
      expect(generator.requests).toHaveLength(1);
      expect(resumedFiltered.input).toEqual([nextItem]);
      expect((await fixture.store.readCheckpoint()).context_memory.total_compactions).toBe(1);
    } finally {
      resumedWorld?.dispose();
      if (!resumedWorld) fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("rotates the persisted SDK session only after compact evidence is durable", async () => {
    const fixture = await createFixture();
    try {
      const session = new FileSession(
        fixture.store.sessionPath(),
        fixture.runtime.runId
      );
      const items = longHistory();
      await session.addItems(items);
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: () => new RecordingGenerator(),
        sessionForAgent: () => session
      });

      await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: items, instructions: "Coordinate." }
      });

      expect(await session.getItems()).toEqual([]);
      expect(await fixture.store.readJournal("context")).toEqual(expect.arrayContaining([
        expect.objectContaining({ type: "context_history_delta", items }),
        expect.objectContaining({ type: "context_compacted" }),
        expect.objectContaining({
          type: "context_history_compacted",
          physical_item_count: items.length,
          items: []
        })
      ]));
      expect((await fixture.store.readCheckpoint()).context_memory.scopes[fixture.runtime.rootAgentId])
        .toMatchObject({ raw_item_count: 0, compacted_item_count: 0 });
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("compacts successive physical session epochs without restoring an archived prefix", async () => {
    const fixture = await createFixture();
    try {
      const session = new FileSession(
        fixture.store.sessionPath(),
        fixture.runtime.runId
      );
      const generator = new RecordingGenerator();
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: () => generator,
        sessionForAgent: () => session
      });
      const first = longHistory();
      const second = structuredClone(longHistory());
      second[0] = { role: "user", content: "Continue after the first compact checkpoint." };

      await session.addItems(first);
      await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: await session.getItems(), instructions: "Coordinate." }
      });
      await manager.compactSessionHistories(() => session);
      expect(await session.getItems()).toEqual([]);

      await session.addItems(second);
      const filtered = await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: await session.getItems(), instructions: "Coordinate." }
      });

      expect(filtered.input).toEqual([]);
      expect(generator.requests).toHaveLength(2);
      expect(generator.requests[1]?.sourceItems).toEqual(second);
      expect(await session.getItems()).toEqual([]);
      expect((await fixture.store.readCheckpoint()).context_memory).toMatchObject({
        total_compactions: 2,
        scopes: {
          [fixture.runtime.rootAgentId]: {
            raw_item_count: 0,
            compacted_item_count: 0,
            compaction_count: 2
          }
        }
      });
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("repairs an untrimmed physical session after a full-compaction crash window", async () => {
    const fixture = await createFixture();
    let resumedWorld: RapierWorld | undefined;
    try {
      const items = longHistory();
      const session = new FileSession(
        fixture.store.sessionPath(),
        fixture.runtime.runId
      );
      await session.addItems(items);
      const generator = new RecordingGenerator();
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: () => generator
      });

      await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: items, instructions: "Coordinate." }
      });
      const persisted = await fixture.store.readCheckpoint();
      expect(persisted.context_memory.scopes[persisted.root_id]).toMatchObject({
        raw_item_count: 0,
        raw_chain_hash: null,
        compaction_count: 1
      });
      expect(await session.getItems()).toEqual(items);

      fixture.world.dispose();
      resumedWorld = await RapierWorld.create(fixture.scenario, persisted.world);
      const resumedHierarchy = new HierarchyProjection(
        persisted.nodes,
        persisted.root_id,
        persisted.active_agent_id,
        persisted.active_agent_ids
      );
      const resumedRuntime = new HarnessRuntimeContext({
        store: fixture.store,
        goal: fixture.goal,
        world: resumedWorld,
        hierarchy: resumedHierarchy,
        checkpoint: persisted
      });
      const resumedSession = new FileSession(
        fixture.store.sessionPath(),
        fixture.runtime.runId
      );
      const resumedManager = new LongRunContextManager({
        runtime: resumedRuntime,
        provider,
        createGenerator: () => generator,
        sessionForAgent: () => resumedSession
      });
      const filtered = await resumedManager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: await resumedSession.getItems(), instructions: "Coordinate." }
      });

      expect(filtered.input).toEqual([]);
      expect(generator.requests).toHaveLength(1);
      expect(await resumedSession.getItems()).toEqual([]);
      expect((await fixture.store.readJournal("context")).filter((entry) =>
        typeof entry === "object" && entry !== null && !Array.isArray(entry)
          && entry.type === "context_compacted"
      )).toHaveLength(1);
    } finally {
      resumedWorld?.dispose();
      if (!resumedWorld) fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("keeps the raw ledger when a model checkpoint cites nonexistent evidence", async () => {
    const fixture = await createFixture();
    try {
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: () => new RecordingGenerator("invented-transaction")
      });
      const items = longHistory();
      await expect(manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate the mission." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: items, instructions: "Coordinate the mission." }
      })).rejects.toThrow("unknown transaction");

      const persisted = await fixture.store.readCheckpoint();
      expect(persisted.context_memory.total_compactions).toBe(0);
      expect(persisted.context_memory.scopes[persisted.root_id]).toMatchObject({
        raw_item_count: items.length,
        compacted_item_count: 0,
        summary: null
      });
      expect(await fixture.store.readJournal("context")).toEqual([
        expect.objectContaining({
          type: "context_history_delta",
          items
        })
      ]);
      expect(await fixture.store.readJournal("provider")).toEqual([
        expect.objectContaining({ status: "context_compaction_error" })
      ]);
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("labels an authority-only fallback without treating it as model memory", async () => {
    const fixture = await createFixture();
    try {
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: () => new RecordingGenerator(undefined, "authority_projection")
      });
      const items = longHistory();
      const filtered = await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: items, instructions: "Coordinate." }
      });

      const persisted = await fixture.store.readCheckpoint();
      expect(persisted.context_memory.scopes[persisted.root_id]).toMatchObject({
        summary_origin: "authority_projection",
        compacted_item_count: 0,
        raw_item_count: 0
      });
      expect(filtered.instructions).toContain("authority-only safety projection");
      expect(await fixture.store.readJournal("provider")).toEqual(expect.arrayContaining([
        expect.objectContaining({
          status: "context_compaction_fallback",
          source: "authority_projection",
          automatic_actuation: false
        }),
        expect.objectContaining({
          status: "context_compacted",
          source: "authority_projection"
        })
      ]));
      expect((await fixture.store.readJournal("context"))[0]).toMatchObject({
        type: "context_history_delta",
        items
      });
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("retains compact memory when a stalled SDK conversation starts a fresh branch", async () => {
    const fixture = await createFixture();
    try {
      const generator = new RecordingGenerator();
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider,
        createGenerator: () => generator
      });
      await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: longHistory(), instructions: "Coordinate." }
      });

      manager.startFreshSdkTurn(fixture.runtime.rootAgentId);
      // Deliberately reuse the old branch's opening item. A fresh turn must not
      // mistake this shared prefix for permission to rehydrate the abandoned
      // model/tool tail.
      const recovery = structuredClone(longHistory()[0]!);
      const filtered = await manager.filter({
        agent: new Agent({ name: "Mission Coordinator", instructions: "Coordinate." }),
        context: { runId: fixture.runtime.runId },
        modelData: { input: [recovery], instructions: "Coordinate." }
      });

      expect(filtered.input).toEqual([recovery]);
      expect(filtered.instructions).toContain("LONG-RUN CONTEXT CHECKPOINT");
      const persisted = await fixture.store.readCheckpoint();
      expect(persisted.context_memory.scopes[persisted.root_id]).toMatchObject({
        raw_item_count: 1,
        compacted_item_count: 0,
        compaction_count: 1,
        summary_origin: "authority_projection",
        summary: {
          mission_state: expect.stringContaining("satisfied:"),
          decisions: [],
          next_actions: ["Let the active model verify the current goal through the formal checker."]
        }
      });
      expect(await fixture.store.readJournal("context")).toEqual(expect.arrayContaining([
        expect.objectContaining({
          type: "context_history_delta",
          branch: true,
          from_item: 0,
          to_item: 1,
          items: [recovery]
        }),
        expect.objectContaining({
          type: "context_checkpoint_rebased",
          reason: expect.arrayContaining(["fresh_sdk_turn"]),
          summary: expect.objectContaining({ decisions: [] })
        })
      ]));
    } finally {
      fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("rehydrates uncompressed worker history after a process restart", async () => {
    const fixture = await createFixture();
    let resumedWorld: RapierWorld | undefined;
    const roomyProvider = { ...provider, compactTriggerTokens: 7000 };
    try {
      const generator = new RecordingGenerator();
      const manager = new LongRunContextManager({
        runtime: fixture.runtime,
        provider: roomyProvider,
        createGenerator: () => generator
      });
      const opening: AgentInputItem = { role: "user", content: "Continue hierarchy node A." };
      const prior: AgentInputItem[] = [
        opening,
        {
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "I will inspect the current world." }]
        },
        {
          type: "function_call_result",
          callId: "inspect_before_restart",
          name: "sense_scene",
          output: JSON.stringify({ accepted: true, code: "scene_observed" }),
          status: "completed"
        }
      ];
      const agent = new Agent({ name: "Mission Coordinator", instructions: "Coordinate." });
      await manager.filter({
        agent,
        context: { runId: fixture.runtime.runId },
        modelData: { input: [opening], instructions: "Coordinate." }
      });
      await manager.filter({
        agent,
        context: { runId: fixture.runtime.runId },
        modelData: { input: prior, instructions: "Coordinate." }
      });
      const persisted = await fixture.store.readCheckpoint();
      expect(persisted.context_memory.scopes[persisted.root_id]).toMatchObject({
        raw_item_count: prior.length,
        compacted_item_count: 0,
        summary: null
      });

      fixture.world.dispose();
      resumedWorld = await RapierWorld.create(fixture.scenario, persisted.world);
      const hierarchy = new HierarchyProjection(
        persisted.nodes,
        persisted.root_id,
        persisted.active_agent_id,
        persisted.active_agent_ids
      );
      const runtime = new HarnessRuntimeContext({
        store: fixture.store,
        goal: fixture.goal,
        world: resumedWorld,
        hierarchy,
        checkpoint: persisted
      });
      const resumed = new LongRunContextManager({
        runtime,
        provider: roomyProvider,
        createGenerator: () => generator
      });
      const restored = await resumed.filter({
        agent,
        context: { runId: fixture.runtime.runId },
        modelData: { input: [opening], instructions: "Coordinate." }
      });
      expect(restored.input).toEqual(prior);

      const next: AgentInputItem = {
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "Resume from the persisted observation." }]
      };
      const continued = await resumed.filter({
        agent,
        context: { runId: fixture.runtime.runId },
        modelData: { input: [opening, next], instructions: "Coordinate." }
      });
      expect(continued.input).toEqual([...prior, next]);
      const journal = await fixture.store.readJournal("context");
      expect(journal).toEqual(expect.arrayContaining([
        expect.objectContaining({
          type: "context_history_rehydrated",
          restored_item_count: prior.length
        }),
        expect.objectContaining({
          type: "context_history_delta",
          from_item: prior.length,
          to_item: prior.length + 1,
          items: [next]
        })
      ]));
    } finally {
      resumedWorld?.dispose();
      if (!resumedWorld) fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });
});

function longHistory(): AgentInputItem[] {
  return [
    { role: "user", content: "Execute the operator mission." },
    {
      type: "function_call",
      callId: "call_observe",
      name: "sense_scene",
      arguments: "{}",
      status: "completed"
    },
    {
      type: "function_call_result",
      callId: "call_observe",
      name: "sense_scene",
      output: JSON.stringify({ accepted: true, detail: "x".repeat(12_000) }),
      status: "completed"
    }
  ];
}

function chunkedHistory(): AgentInputItem[] {
  return [
    { role: "user", content: "Continue this long-running autonomous mission." },
    ...Array.from({ length: 18 }, (_, index): AgentInputItem => ({
      role: "assistant",
      status: "completed",
      content: [{
        type: "output_text",
        text: `Completed model turn ${index + 1}. ${"evidence ".repeat(850)}`
      }]
    }))
  ];
}

async function createFixture(signal?: AbortSignal): Promise<{
  runsDir: string;
  scenario: Scenario;
  goal: Goal;
  store: RunStore;
  world: RapierWorld;
  runtime: HarnessRuntimeContext;
}> {
  const runsDir = await mkdtemp(join(tmpdir(), "hear-context-"));
  const catalog = await loadRuntimeCatalog();
  const scenario = catalog.materialize("open_navigation", 41);
  const goal: Goal = {
    summary: "Keep the robot in its current physical state.",
    predicates: [{
      type: "robot_at",
      target: { x: scenario.robot.x, y: 0, z: scenario.robot.z },
      tolerance: 0.25
    }]
  };
  const store = await RunStore.create(runsDir, {
    mission: "Maintain a long-running autonomous mission",
    scenarioId: "open_navigation",
    scenario,
    goal
  });
  const world = await RapierWorld.create(scenario);
  const capabilities = capabilityCatalog();
  const hierarchy = HierarchyProjection.create("Maintain a long-running mission", capabilities);
  const checkpoint = createCheckpoint({ store, hierarchy, capabilityCatalog: capabilities, world });
  await store.writeCheckpoint(checkpoint);
  const runtime = new HarnessRuntimeContext({
    store,
    goal,
    world,
    hierarchy,
    checkpoint,
    ...(signal ? { signal } : {})
  });
  await runtime.start();
  return { runsDir, scenario, goal, store, world, runtime };
}
