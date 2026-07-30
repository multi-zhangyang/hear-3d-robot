import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import {
  RunCheckpointSchema,
  type AgentSpec,
  type Goal,
  type RunCheckpoint,
  type Scenario
} from "../domain/schema.js";
import { RunStore } from "../persistence/run-store.js";
import { RapierWorld } from "../world/rapier-world.js";
import { capabilityCatalog } from "./agents.js";
import { REPEATED_DENIAL_LIMIT } from "./denial-ledger.js";
import { receiptEvidenceRequirement } from "./evidence-contract.js";
import { HierarchyProjection } from "./hierarchy-projection.js";
import {
  createCheckpoint,
  HarnessRuntimeContext,
  PARTIAL_WORLD_CHECKPOINT_INTERVAL_FRAMES,
  type RuntimeEvent,
  type RuntimeEventSink
} from "./runtime-context.js";

interface RuntimeFixture {
  runsDir: string;
  scenario: Scenario;
  store: RunStore;
  world: RapierWorld;
  hierarchy: HierarchyProjection;
  runtime: HarnessRuntimeContext;
  activeSpec: AgentSpec;
  activeId: string;
}

interface UnstartedRuntimeFixture {
  runsDir: string;
  scenario: Scenario;
  store: RunStore;
  world: RapierWorld;
  runtime: HarnessRuntimeContext;
}

async function createUnstartedRuntimeFixture(
  eventSink?: RuntimeEventSink
): Promise<UnstartedRuntimeFixture> {
  const runsDir = await mkdtemp(join(tmpdir(), "hear-runtime-lifecycle-"));
  const catalog = await loadRuntimeCatalog();
  const scenario = catalog.materialize("open_navigation", 0);
  const goal: Goal = {
    summary: "Robot remains at the requested coordinate.",
    predicates: [
      { type: "robot_at", target: { x: 1, y: 0, z: 1 }, tolerance: 0.25 }
    ]
  };
  const store = await RunStore.create(runsDir, {
    mission: "Maintain the requested robot state",
    scenarioId: "open_navigation",
    scenario,
    goal
  });
  const world = await RapierWorld.create(scenario);
  const available = capabilityCatalog();
  const hierarchy = HierarchyProjection.create(
    "Maintain the requested robot state",
    available,
    goal.predicates.length
  );
  const checkpoint = createCheckpoint({
    store,
    hierarchy,
    capabilityCatalog: available,
    world
  });
  await store.writeCheckpoint(checkpoint);
  return {
    runsDir,
    scenario,
    store,
    world,
    runtime: new HarnessRuntimeContext({
      store,
      goal,
      world,
      hierarchy,
      checkpoint,
      ...(eventSink ? { eventSink } : {})
    })
  };
}

async function createRuntimeFixture(
  capabilities: string[],
  eventSink?: RuntimeEventSink,
  goalPredicateCount = 0,
  scenarioId = "open_navigation",
  scenarioSeed = 0
): Promise<RuntimeFixture> {
  const runsDir = await mkdtemp(join(tmpdir(), "hear-runtime-"));
  const catalog = await loadRuntimeCatalog();
  const scenario = catalog.materialize(scenarioId, scenarioSeed);
  const goal: Goal = {
    summary: "Robot remains at the requested coordinate.",
    predicates: [
      { type: "robot_at", target: { x: 1, y: 0, z: 1 }, tolerance: 0.25 }
    ]
  };
  const store = await RunStore.create(runsDir, {
    mission: "Maintain the requested robot state",
    scenarioId,
    scenario,
    goal
  });
  const world = await RapierWorld.create(scenario);
  const available = capabilityCatalog();
  const hierarchy = HierarchyProjection.create(
    "Maintain the requested robot state",
    available,
    goalPredicateCount
  );
  const checkpoint = createCheckpoint({
    store,
    hierarchy,
    capabilityCatalog: available,
    world
  });
  await store.writeCheckpoint(checkpoint);
  const runtime = new HarnessRuntimeContext({
    store,
    goal,
    world,
    hierarchy,
    checkpoint,
    ...(eventSink ? { eventSink } : {})
  });
  await runtime.start();
  const activeSpec: AgentSpec = {
    name: "Runtime contract node",
    objective: "Exercise only the granted harness capabilities",
    success_criteria: ["The harness records a source-backed action receipt"],
    evidence_requirements: [fixtureEvidenceRequirement(capabilities)],
    capabilities,
    may_delegate: false,
    references: []
  };
  const active = await runtime.beginDelegation(null, activeSpec, "delegate_runtime_contract");
  return {
    runsDir,
    scenario,
    store,
    world,
    hierarchy,
    runtime,
    activeSpec,
    activeId: active.node.id
  };
}

function fixtureEvidenceRequirement(capabilities: string[]) {
  const action = capabilities[0];
  if (!action) throw new Error("Runtime fixture needs a capability");
  if (action === "read_proprioception" || action === "query_contacts" || action === "inspect_command") {
    return receiptEvidenceRequirement(0, action, { kind: "robot" });
  }
  if (action === "sense_scene" || action === "scan_voxels" || action === "recall_spatial_memory") {
    return receiptEvidenceRequirement(0, action, { kind: "world" });
  }
  if (action === "survey_terrain") {
    return receiptEvidenceRequirement(0, action, { kind: "terrain" });
  }
  if (action === "inspect_entity") {
    return receiptEvidenceRequirement(0, action, { kind: "entity", entity_id: "red_block" });
  }
  if (action === "inspect_voxel" || action === "break_voxel" || action === "place_voxel") {
    return receiptEvidenceRequirement(0, action, {
      kind: "voxel",
      coordinate: { column: 0, level: 0, row: 0 }
    });
  }
  if (action === "plan_base_path" || action === "plan_arm_retraction"
    || action === "solve_end_effector_position" || action === "solve_end_effector_pose") {
    return receiptEvidenceRequirement(0, action, {
      kind: "position",
      position: { x: 0, y: 0, z: 0 }
    });
  }
  const channel = action === "set_head_target" ? "head"
    : action === "set_gripper_target" ? "gripper"
      : action === "drive_base" || action === "execute_base_plan" ? "base" : "arm";
  return receiptEvidenceRequirement(0, action, { kind: "body", channel });
}

async function disposeFixture(fixture: Pick<RuntimeFixture, "runsDir" | "world">): Promise<void> {
  fixture.world.dispose();
  await rm(fixture.runsDir, { recursive: true, force: true });
}

function outputRecord(output: string): Record<string, unknown> {
  return JSON.parse(output) as Record<string, unknown>;
}

function objectRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function point(value: unknown): { x: number; y: number; z: number } {
  const candidate = objectRecord(value);
  if (!candidate || typeof candidate.x !== "number" || typeof candidate.y !== "number"
    || typeof candidate.z !== "number") throw new Error("Expected a world-space point");
  return { x: candidate.x, y: candidate.y, z: candidate.z };
}

function runtimeEventTransactionId(value: unknown): string | undefined {
  const event = objectRecord(value);
  const data = objectRecord(event?.data);
  const receipt = objectRecord(data?.receipt);
  const transactionId = data?.transaction_id ?? receipt?.transaction_id;
  return typeof transactionId === "string" ? transactionId : undefined;
}

async function resumedRuntime(
  fixture: Pick<RuntimeFixture, "scenario" | "store">,
  checkpoint: RunCheckpoint,
  eventSink?: RuntimeEventSink
): Promise<{ world: RapierWorld; runtime: HarnessRuntimeContext }> {
  const world = await RapierWorld.create(fixture.scenario, checkpoint.world);
  const hierarchy = new HierarchyProjection(
    checkpoint.nodes,
    checkpoint.root_id,
    checkpoint.active_agent_id,
    checkpoint.active_agent_ids
  );
  return {
    world,
    runtime: new HarnessRuntimeContext({
      store: fixture.store,
      goal: fixture.store.definition.goal,
      world,
      hierarchy,
      checkpoint,
      ...(eventSink ? { eventSink } : {})
    })
  };
}

describe("HarnessRuntimeContext", () => {
  it("shares one stable identity between telemetry journals and durable SSE events", async () => {
    const delivered: RuntimeEvent[] = [];
    const fixture = await createUnstartedRuntimeFixture((event) => delivered.push(event));
    try {
      await fixture.runtime.recordProvider({ status: "contacted" });
      await fixture.runtime.recordFramework("agent:root", { type: "response_done" });

      const provider = objectRecord((await fixture.store.readJournal("provider"))[0]);
      const framework = objectRecord((await fixture.store.readJournal("framework"))[0]);
      const providerEvent = delivered.find((event) => event.type === "provider_event");
      const frameworkEvent = delivered.find((event) => event.type === "framework_event");
      expect(provider?.runtime_event_id).toBe(providerEvent?.event_id);
      expect(framework?.runtime_event_id).toBe(frameworkEvent?.event_id);
      expect(providerEvent?.durable).toBe(true);
      expect(frameworkEvent?.durable).toBe(true);
      expect(providerEvent?.cursor).toMatch(/^v1:\d+:[a-f0-9]{64}$/);
      expect(frameworkEvent?.cursor).toMatch(/^v1:\d+:[a-f0-9]{64}$/);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("recovers a started lifecycle event after its first journal append fails", async () => {
    const fixture = await createUnstartedRuntimeFixture();
    let resumedWorld: RapierWorld | undefined;
    const originalAppend = fixture.store.appendRuntimeEvents.bind(fixture.store);
    const append = vi.spyOn(fixture.store, "appendRuntimeEvents").mockImplementation(async (events) => {
      if (events.some((event) => event.type === "run_started")) {
        throw new Error("event journal unavailable");
      }
      return originalAppend(events);
    });
    try {
      await expect(fixture.runtime.start()).rejects.toThrow("event journal unavailable");
      append.mockRestore();

      const pending = await fixture.store.readCheckpoint();
      expect(pending.status).toBe("running");
      expect(pending.pending_lifecycle_events).toHaveLength(1);
      expect(pending.pending_lifecycle_events[0]).toMatchObject({ type: "run_started" });
      expect(await fixture.store.readJournal("events")).toEqual([]);

      const delivered: RuntimeEvent[] = [];
      const resumed = await resumedRuntime(fixture, pending, (event) => delivered.push(event));
      resumedWorld = resumed.world;
      await resumed.runtime.start(true);

      const lifecycle = (await fixture.store.readJournal("events"))
        .filter((entry) => ["run_started", "run_resumed"].includes(
          String(objectRecord(entry)?.type)
        ));
      expect(lifecycle.map((entry) => objectRecord(entry)?.type)).toEqual([
        "run_started",
        "run_resumed"
      ]);
      expect(new Set(lifecycle.map((entry) => objectRecord(entry)?.event_id)).size).toBe(2);
      expect(delivered.map((event) => event.type)).toEqual(["run_started", "run_resumed"]);
      expect((await fixture.store.readCheckpoint()).pending_lifecycle_events).toEqual([]);
    } finally {
      append.mockRestore();
      resumedWorld?.dispose();
      await disposeFixture(fixture);
    }
  });

  it("does not duplicate a lifecycle event when clearing its published outbox fails", async () => {
    const fixture = await createUnstartedRuntimeFixture();
    let resumedWorld: RapierWorld | undefined;
    const originalWrite = fixture.store.writeCheckpoint.bind(fixture.store);
    let writes = 0;
    const write = vi.spyOn(fixture.store, "writeCheckpoint")
      .mockImplementation(async (checkpoint) => {
        writes += 1;
        if (writes === 2) throw new Error("checkpoint clear unavailable");
        await originalWrite(checkpoint);
      });
    try {
      await expect(fixture.runtime.start()).rejects.toThrow("checkpoint clear unavailable");
      write.mockRestore();

      const pending = await fixture.store.readCheckpoint();
      expect(pending.pending_lifecycle_events).toHaveLength(1);
      expect((await fixture.store.readJournal("events")).filter(
        (entry) => objectRecord(entry)?.type === "run_started"
      )).toHaveLength(1);

      const delivered: RuntimeEvent[] = [];
      const resumed = await resumedRuntime(fixture, pending, (event) => delivered.push(event));
      resumedWorld = resumed.world;
      await resumed.runtime.start(true);

      expect((await fixture.store.readJournal("events")).filter(
        (entry) => objectRecord(entry)?.type === "run_started"
      )).toHaveLength(1);
      expect(delivered.map((event) => event.type)).toEqual(["run_resumed"]);
      expect((await fixture.store.readCheckpoint()).pending_lifecycle_events).toEqual([]);
    } finally {
      write.mockRestore();
      resumedWorld?.dispose();
      await disposeFixture(fixture);
    }
  });

  it("recovers an interrupted lifecycle event before resuming a terminal checkpoint", async () => {
    const fixture = await createRuntimeFixture(["set_head_target"]);
    let resumedWorld: RapierWorld | undefined;
    const originalAppend = fixture.store.appendRuntimeEvents.bind(fixture.store);
    const append = vi.spyOn(fixture.store, "appendRuntimeEvents").mockImplementation(async (events) => {
      if (events.some((event) => event.type === "run_interrupted")) {
        throw new Error("terminal event journal unavailable");
      }
      return originalAppend(events);
    });
    try {
      await expect(fixture.runtime.interrupt("operator stopped"))
        .rejects.toThrow("terminal event journal unavailable");
      append.mockRestore();

      const interrupted = await fixture.store.readCheckpoint();
      expect(interrupted.status).toBe("interrupted");
      expect(interrupted.pending_lifecycle_events).toHaveLength(1);
      expect(interrupted.pending_lifecycle_events[0]).toMatchObject({
        type: "run_interrupted",
        data: { reason: "operator stopped" }
      });

      const delivered: RuntimeEvent[] = [];
      const resumed = await resumedRuntime(fixture, interrupted, (event) => delivered.push(event));
      resumedWorld = resumed.world;
      await resumed.runtime.start(true);

      expect((await fixture.store.readJournal("events")).filter(
        (entry) => objectRecord(entry)?.type === "run_interrupted"
      )).toHaveLength(1);
      expect(delivered.map((event) => event.type)).toEqual([
        "run_interrupted",
        "run_resumed"
      ]);
      expect((await fixture.store.readCheckpoint()).pending_lifecycle_events).toEqual([]);
    } finally {
      append.mockRestore();
      resumedWorld?.dispose();
      await disposeFixture(fixture);
    }
  });

  it("commits one receipt when a skill transaction is reused with identical data", async () => {
    const fixture = await createRuntimeFixture(["set_head_target"]);
    try {
      const input = { yaw: 0.2, pitch: -0.1 };
      const revisionBeforeFirst = fixture.runtime.checkpoint.world.world_revision;
      const first = await fixture.runtime.invokeSkill(
        "set_head_target",
        input,
        "sdk_action_1"
      );
      const frameAfterFirst = fixture.runtime.checkpoint.world.frame;
      const second = await fixture.runtime.invokeSkill(
        "set_head_target",
        input,
        "sdk_action_1"
      );

      expect(second).toBe(first);
      expect(outputRecord(first)).toMatchObject({
        accepted: true,
        code: "head_target_reached",
        transaction_id: `${fixture.activeId}:sdk_action_1`
      });
      expect(fixture.runtime.checkpoint.world.frame).toBe(frameAfterFirst);
      expect(Object.keys(fixture.runtime.checkpoint.committed_actions)).toEqual([
        `${fixture.activeId}:sdk_action_1`
      ]);
      expect(fixture.runtime.checkpoint.committed_actions[`${fixture.activeId}:sdk_action_1`])
        .toMatchObject({ world_before_revision: revisionBeforeFirst });
      expect(await fixture.store.readJournal("actions")).toHaveLength(1);

      await expect(fixture.runtime.invokeSkill(
        "set_head_target",
        { yaw: 0.3, pitch: -0.1 },
        "sdk_action_1"
      )).rejects.toThrow("was reused with different action data");
      expect(await fixture.store.readJournal("actions")).toHaveLength(1);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("returns the original tool and checker receipts when their transactions are replayed", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      const firstTool = await fixture.runtime.invokeTool(
        "sense_scene",
        {},
        "sdk_tool_replay"
      );
      const replayedTool = await fixture.runtime.invokeTool(
        "sense_scene",
        {},
        "sdk_tool_replay"
      );
      expect(replayedTool).toBe(firstTool);
      expect(outputRecord(firstTool)).toMatchObject({
        accepted: true,
        code: "scene_observation",
        transaction_id: `${fixture.activeId}:sdk_tool_replay`
      });

      const firstChecker = await fixture.runtime.invokeChecker(
        {},
        "sdk_checker_replay",
        fixture.hierarchy.rootId
      );
      const replayedChecker = await fixture.runtime.invokeChecker(
        {},
        "sdk_checker_replay",
        fixture.hierarchy.rootId
      );
      expect(replayedChecker).toBe(firstChecker);
      expect(outputRecord(firstChecker)).toMatchObject({
        accepted: true,
        code: "mission_satisfied",
        transaction_id: `${fixture.hierarchy.rootId}:sdk_checker_replay`
      });

      expect(await fixture.store.readJournal("actions")).toHaveLength(2);
      expect(await fixture.store.readJournal("checker")).toHaveLength(1);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("rejects every mismatched transaction signature without replacing its receipt", async () => {
    const fixture = await createRuntimeFixture([
      "recall_spatial_memory",
      "sense_scene",
      "set_head_target"
    ]);
    try {
      const callId = "sdk_signature_conflict";
      const transactionId = `${fixture.activeId}:${callId}`;
      const original = await fixture.runtime.invokeTool(
        "recall_spatial_memory",
        { text: "first query" },
        callId,
        fixture.activeId
      );
      const storedBefore = structuredClone(
        fixture.runtime.checkpoint.committed_actions[transactionId]
      );

      await expect(fixture.runtime.invokeTool(
        "recall_spatial_memory",
        { text: "different query" },
        callId,
        fixture.activeId
      )).rejects.toThrow("was reused with different action data");
      await expect(fixture.runtime.invokeTool(
        "sense_scene",
        {},
        callId,
        fixture.activeId
      )).rejects.toThrow("was reused with different action data");
      await expect(fixture.runtime.invokeSkill(
        "set_head_target",
        { yaw: 0.2, pitch: -0.1 },
        callId,
        fixture.activeId
      )).rejects.toThrow("was reused with different action data");

      expect(outputRecord(original)).toMatchObject({ accepted: true, transaction_id: transactionId });
      expect(fixture.runtime.checkpoint.committed_actions[transactionId]).toEqual(storedBefore);
      expect(await fixture.store.readJournal("actions")).toHaveLength(1);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("joins concurrent duplicate tool and skill transactions without executing twice", async () => {
    const events: RuntimeEvent[] = [];
    const fixture = await createRuntimeFixture(
      ["sense_scene", "set_head_target"],
      (event) => {
        events.push(event);
      }
    );
    try {
      const toolCalls = await Promise.all([
        fixture.runtime.invokeTool("sense_scene", {}, "sdk_concurrent_tool", fixture.activeId),
        fixture.runtime.invokeTool("sense_scene", {}, "sdk_concurrent_tool", fixture.activeId)
      ]);
      expect(toolCalls[1]).toBe(toolCalls[0]);

      const headInput = { yaw: 0.35, pitch: -0.12 };
      const skillCalls = await Promise.all([
        fixture.runtime.invokeSkill(
          "set_head_target",
          headInput,
          "sdk_concurrent_skill",
          fixture.activeId
        ),
        fixture.runtime.invokeSkill(
          "set_head_target",
          headInput,
          "sdk_concurrent_skill",
          fixture.activeId
        )
      ]);
      expect(skillCalls[1]).toBe(skillCalls[0]);
      expect(outputRecord(skillCalls[0]!)).toMatchObject({
        accepted: true,
        code: "head_target_reached"
      });

      expect(await fixture.store.readJournal("actions")).toHaveLength(2);
      expect(events.filter((event) => event.type === "action_requested")).toHaveLength(2);
      expect(events.filter((event) => event.type === "action_reused")).toHaveLength(2);
      expect(events.filter((event) => event.type === "command_started")).toHaveLength(1);
      expect(fixture.runtime.activeNode(fixture.activeId).steps_used).toBe(2);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("rejects a direct action when the active node was not granted its capability", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      const before = fixture.runtime.checkpoint.world.frame;
      expect(fixture.runtime.isCapabilityEnabled("sense_scene")).toBe(true);
      expect(fixture.runtime.isCapabilityEnabled("set_head_target")).toBe(false);

      const output = await fixture.runtime.invokeSkill(
        "set_head_target",
        { yaw: 0.2, pitch: 0 },
        "sdk_unauthorized_action"
      );
      const transactionId = `${fixture.activeId}:sdk_unauthorized_action`;
      expect(outputRecord(output)).toMatchObject({
        accepted: false,
        code: "authority_denied",
        transaction_id: transactionId,
        world_frame: before
      });
      expect(fixture.runtime.checkpoint.committed_actions[transactionId]).toMatchObject({
        accepted: false,
        code: "authority_denied",
        frame_count: 0,
        gates: [
          {
            name: "capability_authority",
            status: "rejected",
            detail: {
              reason: "capability_not_granted",
              agent_id: fixture.activeId,
              name: "set_head_target"
            }
          },
          { name: "body_lease", status: "passed" }
        ]
      });
      expect(fixture.runtime.checkpoint.world.frame).toBe(before);
      expect(await fixture.store.readJournal("actions")).toHaveLength(1);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("resolves a granted planning transaction across sibling hierarchy nodes", async () => {
    const fixture = await createRuntimeFixture(["plan_base_path", "execute_base_plan"]);
    try {
      const planned = await fixture.runtime.invokeTool(
        "plan_base_path",
        { target: { x: 1, y: 0, z: 2 } },
        "sdk_cross_node_plan"
      );
      const planningTransactionId = `${fixture.activeId}:sdk_cross_node_plan`;
      expect(outputRecord(planned)).toMatchObject({
        accepted: true,
        transaction_id: planningTransactionId
      });
      await fixture.runtime.completeChild(fixture.activeId, "Base path planning completed");
      const coordinatorAnchor = fixture.runtime.contextAnchor(
        fixture.hierarchy.rootId
      ) as Record<string, unknown>;
      expect(coordinatorAnchor.pending_plan_receipts).toEqual([
        expect.objectContaining({
          planning_transaction_id: planningTransactionId,
          planning_action: "plan_base_path",
          required_execution_action: "execute_base_plan",
          plan_status: "valid",
          automatic_actuation: false
        })
      ]);

      const ungrantedExecutor: AgentSpec = {
        name: "Ungranted base executor",
        objective: "Attempt the requested base execution",
        success_criteria: ["The accepted base plan is physically executed."],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "execute_base_plan",
          { kind: "body", channel: "base" }
        )],
        capabilities: ["execute_base_plan"],
        may_delegate: false,
        references: []
      };
      const ungranted = await fixture.runtime.beginDelegation(
        null,
        ungrantedExecutor,
        "delegate_ungranted_executor"
      );
      const deniedExecution = await fixture.runtime.invokeSkill(
        "execute_base_plan",
        { planning_transaction_id: planningTransactionId },
        "sdk_ungranted_execution"
      );
      expect(outputRecord(deniedExecution)).toMatchObject({
        accepted: false,
        code: "planning_transaction_not_granted"
      });
      // "Not granted" alone leaves three possible faults — wrong id, expired
      // plan, someone else's plan — with three different fixes. A live run hit
      // this and stalled, so the denial has to name the owner and the way out.
      const ungrantedDetail = (outputRecord(deniedExecution).detail) as Record<string, unknown>;
      expect(ungrantedDetail.owning_agent_name).toBe(fixture.activeSpec.name);
      expect(String(ungrantedDetail.recovery)).toContain("plan_base_path");
      expect(String(ungrantedDetail.recovery)).toContain("report_blocked");
      await fixture.runtime.completeChild(ungranted.node.id, "Execution reference was not granted");

      const executorSpec: AgentSpec = {
        name: "Granted base executor",
        objective: "Execute the accepted base path in the current world revision",
        success_criteria: ["The accepted base plan is physically executed."],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "execute_base_plan",
          { kind: "body", channel: "base" }
        )],
        capabilities: ["execute_base_plan"],
        may_delegate: false,
        references: [{
          name: "plan_base_path",
          transaction_id: planningTransactionId
        }]
      };
      const executor = await fixture.runtime.beginDelegation(
        null,
        executorSpec,
        "delegate_granted_executor"
      );
      const executed = await fixture.runtime.invokeSkill(
        "execute_base_plan",
        { planning_transaction_id: planningTransactionId },
        "sdk_granted_execution"
      );

      expect(outputRecord(executed)).toMatchObject({
        accepted: true,
        code: "base_plan_completed",
        transaction_id: `${executor.node.id}:sdk_granted_execution`
      });
      expect(fixture.runtime.checkpoint.world.robot.position.z).toBeGreaterThan(1.5);
    } finally {
      await disposeFixture(fixture);
    }
  }, 20_000);

  it("points a mistaken plan handle back at the transaction id it should have used", async () => {
    const fixture = await createRuntimeFixture(["plan_base_path", "execute_base_plan"]);
    try {
      const planned = await fixture.runtime.invokeTool(
        "plan_base_path",
        { target: { x: 1, y: 0, z: 2 } },
        "sdk_plan_for_handle_mixup"
      );
      const planningTransactionId = `${fixture.activeId}:sdk_plan_for_handle_mixup`;
      const planId = (outputRecord(planned).detail as Record<string, unknown>).plan_id;
      expect(typeof planId).toBe("string");
      expect(planId).not.toBe(planningTransactionId);

      // A live run passed exactly this: the world's own handle for the solution,
      // sitting in the receipt detail and looking like what an executor wants.
      // Answering "unknown" sends the agent back to reread the same detail and
      // find the same wrong field, so the denial has to name the right one.
      const denied = await fixture.runtime.invokeSkill(
        "execute_base_plan",
        { planning_transaction_id: planId },
        "sdk_execute_with_plan_handle"
      );
      expect(outputRecord(denied)).toMatchObject({
        accepted: false,
        code: "unknown_planning_transaction"
      });
      const detail = outputRecord(denied).detail as Record<string, unknown>;
      expect(detail.your_latest_matching_transaction_id).toBe(planningTransactionId);
      expect(String(detail.recovery)).toContain("plan_id");
      expect(String(detail.recovery)).toContain("transaction_id");

      const executed = await fixture.runtime.invokeSkill(
        "execute_base_plan",
        { planning_transaction_id: String(detail.your_latest_matching_transaction_id) },
        "sdk_execute_after_hint"
      );
      expect(outputRecord(executed)).toMatchObject({
        accepted: true,
        code: "base_plan_completed"
      });
    } finally {
      await disposeFixture(fixture);
    }
  }, 20_000);

  it("executes a position-only arm plan by its harness transaction", async () => {
    const fixture = await createRuntimeFixture([
      "solve_end_effector_position",
      "execute_joint_plan"
    ]);
    try {
      const position = fixture.world.snapshot().robot.links.gripper?.position;
      expect(position).toBeDefined();
      const planned = await fixture.runtime.invokeTool(
        "solve_end_effector_position",
        { position },
        "sdk_position_only_plan"
      );
      const planningTransactionId = `${fixture.activeId}:sdk_position_only_plan`;
      expect(outputRecord(planned)).toMatchObject({
        accepted: true,
        code: "end_effector_solution",
        transaction_id: planningTransactionId,
        detail: {
          execution_required: {
            automatic_actuation: false,
            tool: "execute_joint_plan",
            planning_transaction_id: planningTransactionId,
            available_to_current_agent: true
          }
        }
      });
      expect(fixture.runtime.checkpoint.committed_actions[planningTransactionId]?.input)
        .toEqual({ position });
      const anchoredPlan = fixture.runtime.contextAnchor(fixture.activeId) as Record<
        string,
        unknown
      >;
      expect(anchoredPlan.pending_plan_receipts).toEqual([
        expect.objectContaining({
          planning_transaction_id: planningTransactionId,
          planning_action: "solve_end_effector_position",
          required_execution_action: "execute_joint_plan",
          plan_status: "valid",
          automatic_actuation: false,
          decision_owner: "model"
        })
      ]);

      const executed = await fixture.runtime.invokeSkill(
        "execute_joint_plan",
        { planning_transaction_id: planningTransactionId },
        "sdk_position_only_execution"
      );
      expect(outputRecord(executed)).toMatchObject({
        accepted: true,
        code: "joint_targets_reached"
      });
      const afterExecution = fixture.runtime.contextAnchor(fixture.activeId) as Record<
        string,
        unknown
      >;
      expect(afterExecution.pending_plan_receipts).toBeNull();
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("rejects capability calls from the root coordinator", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      await fixture.runtime.completeChild(fixture.activeId, "Observation work completed");
      const before = fixture.runtime.checkpoint.world.frame;
      const output = await fixture.runtime.invokeTool("sense_scene", {}, "sdk_root_bypass");
      const transactionId = `${fixture.hierarchy.rootId}:sdk_root_bypass`;

      expect(outputRecord(output)).toMatchObject({
        accepted: false,
        code: "authority_denied",
        transaction_id: transactionId,
        world_frame: before
      });
      expect(fixture.runtime.checkpoint.committed_actions[transactionId]).toMatchObject({
        gates: [{
          name: "capability_authority",
          status: "rejected",
          detail: { reason: "root_coordinator_must_delegate", name: "sense_scene" }
        }]
      });
      expect(fixture.runtime.checkpoint.world.frame).toBe(before);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("refuses an unchanged failed delegation after repeated denials at one revision", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      for (let index = 0; index < 3; index += 1) {
        const denied = await fixture.runtime.invokeSkill(
          "set_head_target",
          { yaw: 0.1 + index * 0.1, pitch: 0 },
          `sdk_failed_branch_${index}`
        );
        expect(outputRecord(denied)).toMatchObject({
          accepted: false,
          code: "authority_denied"
        });
      }
      await fixture.runtime.failChild(fixture.activeId, "The unchanged grant cannot perform the work");

      await expect(fixture.runtime.beginDelegation(
        null,
        fixture.activeSpec,
        "delegate_same_failed_branch"
      )).rejects.toThrow("Unchanged delegation already ended failed");

      const different = await fixture.runtime.beginDelegation(
        null,
        { ...fixture.activeSpec, objective: "Observe the current world instead" },
        "delegate_materially_different_branch"
      );
      expect(different.node.status).toBe("active");
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("allows an identical delegation retry when transport failed before any receipt", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      await fixture.runtime.failChild(fixture.activeId, "Provider connection closed before response");
      const retried = await fixture.runtime.beginDelegation(
        null,
        fixture.activeSpec,
        "delegate_transport_retry"
      );
      expect(retried.node.status).toBe("active");
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("requires supervisory nodes to delegate robot capabilities to leaves", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      await fixture.runtime.completeChild(fixture.activeId, "Initial leaf completed");
      const supervisorSpec: AgentSpec = {
        name: "Observation supervisor",
        objective: "Coordinate a source-backed observation",
        success_criteria: ["A child observes the current scene."],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "sense_scene",
          { kind: "world" }
        )],
        capabilities: ["sense_scene"],
        may_delegate: true,
        references: []
      };
      const supervisor = await fixture.runtime.beginDelegation(
        null,
        supervisorSpec,
        "delegate_observation_supervisor"
      );

      expect(fixture.runtime.isCapabilityEnabled("sense_scene")).toBe(false);
      const output = await fixture.runtime.invokeTool(
        "sense_scene",
        {},
        "sdk_supervisor_bypass"
      );
      const transactionId = `${supervisor.node.id}:sdk_supervisor_bypass`;
      expect(outputRecord(output)).toMatchObject({
        accepted: false,
        code: "authority_denied"
      });
      expect(fixture.runtime.checkpoint.committed_actions[transactionId]).toMatchObject({
        gates: [{
          name: "capability_authority",
          status: "rejected",
          detail: {
            reason: "supervisor_must_delegate",
            agent_id: supervisor.node.id,
            name: "sense_scene"
          }
        }]
      });
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("rejects a supervisor completion while its owned live goal predicate is unmet", async () => {
    const fixture = await createRuntimeFixture(["drive_base"], undefined, 1);
    try {
      await fixture.runtime.invokeSkill(
        "drive_base",
        {
          linear_meters_per_second: 0.4,
          angular_radians_per_second: 0,
          duration_seconds: 1
        },
        "sdk_move_off_goal"
      );
      await fixture.runtime.completeChild(fixture.activeId, "Robot moved off the goal");

      const criterion = "The current state has been observed.";
      const supervisorSpec: AgentSpec = {
        name: "Goal-owning supervisor",
        objective: "Restore and verify the requested robot state",
        success_criteria: [criterion],
        evidence_requirements: [{
          kind: "goal_predicate",
          criterion_index: 0,
          predicate_index: 0
        }],
        goal_predicate_indexes: [0],
        capabilities: ["sense_scene", "read_proprioception", "drive_base"],
        may_delegate: true,
        references: []
      };
      const supervisor = await fixture.runtime.beginDelegation(
        null,
        supervisorSpec,
        "delegate_goal_owner"
      );
      const observerSpec: AgentSpec = {
        name: "Goal observer",
        objective: "Observe the current state without changing it",
        success_criteria: [criterion],
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
      const observer = await fixture.runtime.beginDelegation(
        supervisorSpec,
        observerSpec,
        "delegate_goal_observer"
      );
      await fixture.runtime.invokeTool("sense_scene", {}, "sdk_goal_observation");
      const observationTransaction = `${observer.node.id}:sdk_goal_observation`;
      await fixture.runtime.completeChild(observer.node.id, "Observation completed");

      expect(() => fixture.runtime.assertChildEvidence(
        supervisor.node.id,
        "completed",
        [{ criterion_index: 0, transaction_ids: [observationTransaction] }],
        []
      )).toThrow("requires unmet goal predicate 0");
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("requires every new supervisor to own a structured goal predicate", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"], undefined, 1);
    try {
      await fixture.runtime.completeChild(fixture.activeId, "Initial leaf completed");
      await expect(fixture.runtime.beginDelegation(
        null,
        {
          name: "Unbound supervisor",
          objective: "Claim work without final-state ownership",
          success_criteria: ["The work is complete."],
          evidence_requirements: [receiptEvidenceRequirement(
            0,
            "sense_scene",
            { kind: "world" }
          )],
          goal_predicate_indexes: [],
          capabilities: ["sense_scene"],
          may_delegate: true,
          references: []
        },
        "delegate_unbound_supervisor"
      )).rejects.toThrow("must own at least one structured goal predicate");
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("invalidates Checker success after a child changes the world revision", async () => {
    const fixture = await createRuntimeFixture(["set_head_target"]);
    try {
      await fixture.runtime.completeChild(fixture.activeId, "Initial work completed");
      const checked = await fixture.runtime.invokeChecker({}, "sdk_current_world_check");
      const checker = fixture.runtime.checkpoint.checker;

      expect(outputRecord(checked)).toMatchObject({
        accepted: true,
        code: "mission_satisfied"
      });
      expect(checker).toMatchObject({
        success: true,
        world_frame: fixture.runtime.checkpoint.world.frame,
        world_revision: fixture.runtime.checkpoint.world.world_revision
      });
      expect(fixture.runtime.checkerSatisfiedCurrentWorld()).toBe(true);

      const mover = await fixture.runtime.beginDelegation(
        null,
        fixture.activeSpec,
        "delegate_after_checker"
      );
      const checkedRevision = checker?.world_revision;
      const moved = await fixture.runtime.invokeSkill(
        "set_head_target",
        { yaw: 0.25, pitch: -0.1 },
        "sdk_change_checked_world"
      );

      expect(outputRecord(moved)).toMatchObject({
        accepted: true,
        transaction_id: `${mover.node.id}:sdk_change_checked_world`
      });
      expect(fixture.runtime.checkpoint.world.world_revision).toBeGreaterThan(
        checkedRevision ?? -1
      );
      expect(fixture.runtime.checkpoint.checker).toBeNull();
      expect(fixture.runtime.checkerSatisfiedCurrentWorld()).toBe(false);
      await expect(fixture.runtime.succeed("Done")).rejects.toThrow(
        "Cannot complete a run without Checker success for the current world revision"
      );
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("rejects planning and execution as separate terminal criteria when delegation begins", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      await fixture.runtime.completeChild(fixture.activeId, "Initial setup completed");
      const target = { x: 2, y: 0, z: 2 };
      await expect(fixture.runtime.beginDelegation(
        null,
        {
          name: "Invalid movement evidence worker",
          objective: "Plan and execute one base movement",
          success_criteria: ["A path is planned.", "The path is executed."],
          evidence_requirements: [
            receiptEvidenceRequirement(0, "plan_base_path", {
              kind: "position",
              position: target
            }),
            receiptEvidenceRequirement(1, "execute_base_plan", {
              kind: "body",
              channel: "base"
            })
          ],
          goal_predicate_indexes: [],
          capabilities: ["plan_base_path", "execute_base_plan"],
          may_delegate: false,
          references: []
        },
        "delegate_invalid_movement_evidence"
      )).rejects.toThrow(
        "cannot declare plan_base_path together with execute_base_plan as terminal receipt criteria"
      );
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("completes a surveyed movement leaf from terminal execution evidence and verifies its source plan", async () => {
    const fixture = await createRuntimeFixture(
      ["sense_scene"],
      undefined,
      0,
      "voxel_survey",
      11
    );
    try {
      await fixture.runtime.completeChild(fixture.activeId, "Initial setup completed");
      const movementSpec: AgentSpec = {
        name: "Frontier movement worker",
        objective: "Select and execute one reachable terrain frontier",
        success_criteria: ["One model-selected base plan is physically completed."],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "execute_base_plan",
          { kind: "body", channel: "base" }
        )],
        goal_predicate_indexes: [],
        capabilities: ["survey_terrain", "plan_base_path", "execute_base_plan"],
        may_delegate: false,
        references: []
      };
      const movement = await fixture.runtime.beginDelegation(
        null,
        movementSpec,
        "delegate_frontier_movement"
      );

      const surveyed = outputRecord(await fixture.runtime.invokeTool(
        "survey_terrain",
        { radius_cells: 12 },
        "sdk_frontier_survey",
        movement.node.id
      ));
      expect(surveyed).toMatchObject({ accepted: true, code: "terrain_survey" });
      const frontier = objectRecord(surveyed.detail)?.frontier;
      if (!Array.isArray(frontier) || frontier.length === 0) {
        throw new Error("Terrain survey returned no reachable frontier");
      }
      const selected = objectRecord(frontier[0]);
      if (!selected) throw new Error("Terrain frontier is not a record");
      const target = point(selected.target);
      const facePoint = point(selected.face_point);

      const planned = outputRecord(await fixture.runtime.invokeTool(
        "plan_base_path",
        { target, face_point: facePoint },
        "sdk_frontier_plan",
        movement.node.id
      ));
      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
      const planningTransaction = `${movement.node.id}:sdk_frontier_plan`;

      const executed = outputRecord(await fixture.runtime.invokeSkill(
        "execute_base_plan",
        {
          planning_transaction_id: planningTransaction,
          options: { max_velocity: 0.8, max_duration_seconds: 30, tolerance: 0.08 }
        },
        "sdk_frontier_execute",
        movement.node.id
      ));
      expect(executed).toMatchObject({ accepted: true, code: "base_plan_completed" });
      const executionTransaction = `${movement.node.id}:sdk_frontier_execute`;

      const verified = fixture.runtime.assertChildEvidence(
        movement.node.id,
        "completed",
        [{ criterion_index: 0, transaction_ids: [executionTransaction] }],
        []
      );
      expect(verified).toEqual([
        expect.objectContaining({
          criterion_index: 0,
          transaction_id: executionTransaction,
          action: "execute_base_plan",
          result_code: "base_plan_completed",
          source_transaction_id: planningTransaction,
          source_action: "plan_base_path",
          source_target: { kind: "position", position: target }
        })
      ]);
    } finally {
      await disposeFixture(fixture);
    }
  }, 20_000);

  it("rejects an accepted receipt as evidence for an unmet blocked criterion", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      await fixture.runtime.invokeTool("sense_scene", {}, "sdk_accepted_blocker");
      const transactionId = `${fixture.activeId}:sdk_accepted_blocker`;
      expect(() => fixture.runtime.assertChildEvidence(
        fixture.activeId,
        "blocked",
        [{ criterion_index: 0, transaction_ids: [transactionId] }],
        [0]
      )).toThrow("Blocker evidence requires a rejected transaction");
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("rejects an accepted observation as blocker evidence for an unmet goal predicate", async () => {
    const fixture = await createRuntimeFixture(["drive_base"], undefined, 1);
    try {
      await fixture.runtime.invokeSkill(
        "drive_base",
        {
          linear_meters_per_second: 0.4,
          angular_radians_per_second: 0,
          duration_seconds: 1
        },
        "sdk_leave_goal"
      );
      await fixture.runtime.completeChild(fixture.activeId, "Robot left the goal");

      const blocker = await fixture.runtime.beginDelegation(
        null,
        {
          name: "Goal blocker leaf",
          objective: "Reach the requested robot position or prove a physical blocker",
          success_criteria: ["The requested robot position is reached."],
          evidence_requirements: [{
            kind: "goal_predicate",
            criterion_index: 0,
            predicate_index: 0
          }],
          goal_predicate_indexes: [0],
          capabilities: ["read_proprioception", "drive_base"],
          may_delegate: false,
          references: []
        },
        "delegate_goal_blocker"
      );
      await fixture.runtime.invokeTool(
        "read_proprioception",
        {},
        "sdk_irrelevant_goal_observation",
        blocker.node.id
      );
      const transactionId = `${blocker.node.id}:sdk_irrelevant_goal_observation`;

      expect(() => fixture.runtime.assertChildEvidence(
        blocker.node.id,
        "blocked",
        [{ criterion_index: 0, transaction_ids: [transactionId] }],
        [0]
      )).toThrow("requires a rejected physical transaction");
    } finally {
      await disposeFixture(fixture);
    }
  }, 10_000);

  it("rejects a planning protocol error as body-motion blocker evidence", async () => {
    const fixture = await createRuntimeFixture(["execute_base_plan"]);
    try {
      const output = outputRecord(await fixture.runtime.invokeSkill(
        "execute_base_plan",
        { planning_transaction_id: "missing_planning_transaction" },
        "sdk_protocol_blocker"
      ));
      expect(output).toMatchObject({
        accepted: false,
        code: "unknown_planning_transaction"
      });
      const transactionId = `${fixture.activeId}:sdk_protocol_blocker`;

      expect(() => fixture.runtime.assertChildEvidence(
        fixture.activeId,
        "blocked",
        [{ criterion_index: 0, transaction_ids: [transactionId] }],
        [0]
      )).toThrow("non-terminal code unknown_planning_transaction");
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("accepts current rejected evidence for an unmet blocker beside accepted evidence for met work", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      await fixture.runtime.completeChild(fixture.activeId, "Initial setup completed");
      const blockerSpec: AgentSpec = {
        name: "Entity blocker worker",
        objective: "Inspect one known entity and report one unavailable entity",
        success_criteria: ["The known zone is inspected.", "The missing entity is inspected."],
        evidence_requirements: [
          receiptEvidenceRequirement(0, "inspect_entity", {
            kind: "entity",
            entity_id: "arrival_zone"
          }),
          receiptEvidenceRequirement(1, "inspect_entity", {
            kind: "entity",
            entity_id: "missing_entity"
          })
        ],
        goal_predicate_indexes: [],
        capabilities: ["inspect_entity"],
        may_delegate: false,
        references: []
      };
      const blocker = await fixture.runtime.beginDelegation(
        null,
        blockerSpec,
        "delegate_entity_blocker"
      );
      const accepted = outputRecord(await fixture.runtime.invokeTool(
        "inspect_entity",
        { entity_id: "arrival_zone" },
        "sdk_known_entity",
        blocker.node.id
      ));
      const rejected = outputRecord(await fixture.runtime.invokeTool(
        "inspect_entity",
        { entity_id: "missing_entity" },
        "sdk_missing_entity",
        blocker.node.id
      ));
      expect(accepted).toMatchObject({ accepted: true, code: "entity_state" });
      expect(rejected).toMatchObject({ accepted: false, code: "unknown_entity" });
      const acceptedTransaction = `${blocker.node.id}:sdk_known_entity`;
      const rejectedTransaction = `${blocker.node.id}:sdk_missing_entity`;

      const verified = fixture.runtime.assertChildEvidence(
        blocker.node.id,
        "blocked",
        [
          { criterion_index: 0, transaction_ids: [acceptedTransaction] },
          { criterion_index: 1, transaction_ids: [rejectedTransaction] }
        ],
        [1]
      );
      expect(verified).toEqual([
        expect.objectContaining({
          criterion_index: 0,
          transaction_id: acceptedTransaction,
          result_code: "entity_state"
        }),
        expect.objectContaining({
          criterion_index: 1,
          transaction_id: rejectedTransaction,
          result_code: "unknown_entity",
          accepted: false,
          freshness: "current_world"
        })
      ]);
    } finally {
      await disposeFixture(fixture);
    }
  }, 10_000);

  it("accepts evidence only from the delegated subtree and requires every exact criterion", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    try {
      await fixture.runtime.invokeTool("sense_scene", {}, "sdk_sibling_observation");
      const siblingTransaction = `${fixture.activeId}:sdk_sibling_observation`;
      await fixture.runtime.completeChild(fixture.activeId, "Sibling observation completed");

      const ownCriterion = "The lead observes the current world state.";
      const descendantCriterion = "The descendant moves an authorized joint.";
      const leadSpec: AgentSpec = {
        name: "Evidence lead",
        objective: "Validate evidence across one delegated subtree",
        success_criteria: [ownCriterion, descendantCriterion],
        evidence_requirements: [
          receiptEvidenceRequirement(0, "sense_scene", { kind: "world" }),
          receiptEvidenceRequirement(1, "set_head_target", { kind: "body", channel: "head" })
        ],
        capabilities: ["sense_scene", "set_head_target"],
        may_delegate: true,
        references: []
      };
      const lead = await fixture.runtime.beginDelegation(
        null,
        leadSpec,
        "delegate_evidence_lead"
      );
      const observationSpec: AgentSpec = {
        name: "Evidence observation worker",
        objective: "Observe the current world state",
        success_criteria: [ownCriterion],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "sense_scene",
          { kind: "world" }
        )],
        capabilities: ["sense_scene"],
        may_delegate: false,
        references: []
      };
      const observation = await fixture.runtime.beginDelegation(
        leadSpec,
        observationSpec,
        "delegate_evidence_observation"
      );
      await fixture.runtime.invokeTool("sense_scene", {}, "sdk_descendant_observation");
      const ownTransaction = `${observation.node.id}:sdk_descendant_observation`;
      await fixture.runtime.completeChild(observation.node.id, "Observation completed");

      const descendantSpec: AgentSpec = {
        name: "Evidence motion worker",
        objective: "Move one authorized joint and report its transaction",
        success_criteria: [descendantCriterion],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "set_head_target",
          { kind: "body", channel: "head" }
        )],
        capabilities: ["set_head_target"],
        may_delegate: false,
        references: []
      };
      const descendant = await fixture.runtime.beginDelegation(
        leadSpec,
        descendantSpec,
        "delegate_evidence_motion"
      );
      await fixture.runtime.invokeSkill(
        "set_head_target",
        { yaw: 0.2, pitch: -0.1 },
        "sdk_descendant_motion"
      );
      const descendantTransaction = `${descendant.node.id}:sdk_descendant_motion`;
      await fixture.runtime.invokeSkill(
        "drive_base",
        {
          linear_meters_per_second: 0.2,
          angular_radians_per_second: 0,
          duration_seconds: 0.25
        },
        "sdk_descendant_rejected"
      );
      const rejectedTransaction = `${descendant.node.id}:sdk_descendant_rejected`;
      const currentObservation = await fixture.runtime.beginDelegation(
        leadSpec,
        observationSpec,
        "delegate_current_evidence_observation",
        lead.node.id
      );
      await fixture.runtime.invokeTool(
        "sense_scene",
        {},
        "sdk_current_descendant_observation",
        currentObservation.node.id
      );
      const currentOwnTransaction = `${currentObservation.node.id}:sdk_current_descendant_observation`;
      const completeEvidence = [
        { criterion_index: 0, transaction_ids: [currentOwnTransaction] },
        { criterion_index: 1, transaction_ids: [descendantTransaction] }
      ];

      expect(() => fixture.runtime.assertChildEvidence(
        lead.node.id,
        "completed",
        [
          { criterion_index: 0, transaction_ids: [ownTransaction] },
          { criterion_index: 1, transaction_ids: [descendantTransaction] }
        ],
        []
      )).toThrow("is from world revision");
      expect(() => fixture.runtime.assertChildEvidence(
        lead.node.id,
        "completed",
        completeEvidence,
        []
      )).not.toThrow();
      expect(() => fixture.runtime.assertChildEvidence(
        lead.node.id,
        "completed",
        [
          { criterion_index: 0, transaction_ids: [`${lead.node.id}:missing`] },
          { criterion_index: 1, transaction_ids: [descendantTransaction] }
        ],
        []
      )).toThrow("Evidence references an unknown transaction");
      expect(() => fixture.runtime.assertChildEvidence(
        lead.node.id,
        "completed",
        [
          { criterion_index: 0, transaction_ids: [siblingTransaction] },
          { criterion_index: 1, transaction_ids: [descendantTransaction] }
        ],
        []
      )).toThrow("belongs to another hierarchy branch");
      expect(() => fixture.runtime.assertChildEvidence(
        lead.node.id,
        "completed",
        [
          { criterion_index: 0, transaction_ids: [currentOwnTransaction] },
          { criterion_index: 1, transaction_ids: [rejectedTransaction] }
        ],
        []
      )).toThrow("Completed evidence references a rejected transaction");
      expect(() => fixture.runtime.assertChildEvidence(
        lead.node.id,
        "completed",
        [{ criterion_index: 0, transaction_ids: [currentOwnTransaction] }],
        []
      )).toThrow("Completed outcome has no verified evidence for criteria: 1");
      expect(() => fixture.runtime.assertChildEvidence(
        lead.node.id,
        "completed",
        [
          { criterion_index: 2, transaction_ids: [currentOwnTransaction] },
          { criterion_index: 1, transaction_ids: [descendantTransaction] }
        ],
        []
      )).toThrow("Evidence names an unknown criterion index: 2");
      expect(() => fixture.runtime.assertChildEvidence(
        lead.node.id,
        "blocked",
        [
          { criterion_index: 0, transaction_ids: [currentOwnTransaction] },
          { criterion_index: 1, transaction_ids: [descendantTransaction] }
        ],
        [1]
      )).toThrow("A supervisory agent cannot report blocked");
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("returns a persisted tool receipt when the SDK replays it after process recovery", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    let resumedWorld: RapierWorld | undefined;
    let originalWorldDisposed = false;
    try {
      const input = {};
      const callId = "sdk_tool_after_restart";
      const transactionId = `${fixture.activeId}:${callId}`;
      const original = await fixture.runtime.invokeTool(
        "sense_scene",
        input,
        callId,
        fixture.activeId
      );
      const persisted = RunCheckpointSchema.parse(await fixture.store.readCheckpoint());
      expect(persisted.committed_actions[transactionId]).toMatchObject({
        accepted: true,
        code: "scene_observation"
      });
      expect(await fixture.store.readJournal("actions")).toHaveLength(1);

      fixture.world.dispose();
      originalWorldDisposed = true;
      resumedWorld = await RapierWorld.create(fixture.scenario, persisted.world);
      const resumedHierarchy = new HierarchyProjection(
        persisted.nodes,
        persisted.root_id,
        persisted.active_agent_id,
        persisted.active_agent_ids
      );
      const resumedRuntime = new HarnessRuntimeContext({
        store: fixture.store,
        goal: fixture.store.definition.goal,
        world: resumedWorld,
        hierarchy: resumedHierarchy,
        checkpoint: persisted
      });
      await resumedRuntime.start(true);

      const replayed = await resumedRuntime.invokeTool(
        "sense_scene",
        input,
        callId,
        fixture.activeId
      );
      expect(replayed).toBe(original);
      expect(resumedRuntime.checkpoint.committed_actions[transactionId]).toEqual(
        persisted.committed_actions[transactionId]
      );
      expect(await fixture.store.readJournal("actions")).toHaveLength(1);
      expect(resumedRuntime.activeNode(fixture.activeId).steps_used).toBe(
        persisted.nodes[fixture.activeId]?.steps_used
      );
    } finally {
      resumedWorld?.dispose();
      if (!originalWorldDisposed) fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("reconciles a physical receipt committed before its audit journals without replaying motion", async () => {
    const fixture = await createRuntimeFixture(["set_head_target"]);
    const resumedWorlds: RapierWorld[] = [];
    let originalWorldDisposed = false;
    try {
      const callId = "sdk_checkpoint_before_action_journal";
      const transactionId = `${fixture.activeId}:${callId}`;
      const input = { yaw: 0.45, pitch: -0.2 };
      const frameBefore = fixture.runtime.checkpoint.world.frame;
      const append = fixture.store.append.bind(fixture.store);
      let rejectActionsOnce = true;
      const appendSpy = vi.spyOn(fixture.store, "append").mockImplementation(
        async (name, value) => {
          if (name === "actions" && rejectActionsOnce) {
            rejectActionsOnce = false;
            throw new Error("Process stopped after checkpoint commit");
          }
          await append(name, value);
        }
      );
      try {
        await expect(fixture.runtime.invokeSkill(
          "set_head_target",
          input,
          callId,
          fixture.activeId
        )).rejects.toThrow("Process stopped after checkpoint commit");
      } finally {
        appendSpy.mockRestore();
      }

      const committed = RunCheckpointSchema.parse(await fixture.store.readCheckpoint());
      expect(committed.world.frame).toBeGreaterThan(frameBefore);
      expect(committed.committed_actions[transactionId]).toMatchObject({
        accepted: true,
        code: "head_target_reached"
      });
      expect((await fixture.store.readJournal("actions")).filter(
        (entry) => objectRecord(entry)?.transaction_id === transactionId
      )).toHaveLength(0);

      fixture.world.dispose();
      originalWorldDisposed = true;
      const firstResume = await resumedRuntime(fixture, committed);
      resumedWorlds.push(firstResume.world);
      await firstResume.runtime.start(true);

      const actionsAfterRecovery = await fixture.store.readJournal("actions");
      const hierarchyAfterRecovery = await fixture.store.readJournal("hierarchy");
      const eventsAfterRecovery = await fixture.store.readJournal("events");
      expect(actionsAfterRecovery.filter(
        (entry) => objectRecord(entry)?.transaction_id === transactionId
      )).toHaveLength(1);
      expect(hierarchyAfterRecovery.filter((entry) => {
        const record = objectRecord(entry);
        return record?.type === "action_committed" && record.transaction_id === transactionId;
      })).toHaveLength(1);
      expect(eventsAfterRecovery.filter((entry) =>
        runtimeEventTransactionId(entry) === transactionId
          && ["command_finished", "action_committed"].includes(
            String(objectRecord(entry)?.type)
          )
      )).toHaveLength(2);
      for (const type of ["command_finished", "action_committed"]) {
        const event = eventsAfterRecovery.find((entry) =>
          objectRecord(entry)?.type === type
            && runtimeEventTransactionId(entry) === transactionId
        );
        expect(objectRecord(objectRecord(event)?.data)?.reconciled).toBe(true);
      }

      const frameAfterCommit = committed.world.frame;
      expect(outputRecord(await firstResume.runtime.invokeSkill(
        "set_head_target",
        input,
        callId,
        fixture.activeId
      ))).toMatchObject({ transaction_id: transactionId, accepted: true });
      expect(firstResume.runtime.checkpoint.world.frame).toBe(frameAfterCommit);

      const secondCheckpoint = RunCheckpointSchema.parse(await fixture.store.readCheckpoint());
      const secondResume = await resumedRuntime(fixture, secondCheckpoint);
      resumedWorlds.push(secondResume.world);
      await secondResume.runtime.start(true);

      expect((await fixture.store.readJournal("actions")).filter(
        (entry) => objectRecord(entry)?.transaction_id === transactionId
      )).toHaveLength(1);
      expect((await fixture.store.readJournal("hierarchy")).filter((entry) => {
        const record = objectRecord(entry);
        return record?.type === "action_committed" && record.transaction_id === transactionId;
      })).toHaveLength(1);
      expect((await fixture.store.readJournal("events")).filter((entry) =>
        runtimeEventTransactionId(entry) === transactionId
          && ["command_finished", "action_committed"].includes(
            String(objectRecord(entry)?.type)
          )
      )).toHaveLength(2);
      expect(secondResume.runtime.checkpoint.world.frame).toBe(frameAfterCommit);
    } finally {
      for (const world of resumedWorlds) world.dispose();
      if (!originalWorldDisposed) fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("reconciles only missing checker and post-commit records after partial journal persistence", async () => {
    const fixture = await createRuntimeFixture(["sense_scene"]);
    const resumedWorlds: RapierWorld[] = [];
    let originalWorldDisposed = false;
    try {
      await fixture.runtime.completeChild(fixture.activeId, "Observation work completed");
      const callId = "sdk_checker_after_partial_journals";
      const transactionId = `${fixture.hierarchy.rootId}:${callId}`;
      const append = fixture.store.append.bind(fixture.store);
      let rejectCheckerOnce = true;
      const appendSpy = vi.spyOn(fixture.store, "append").mockImplementation(
        async (name, value) => {
          if (name === "checker" && rejectCheckerOnce) {
            rejectCheckerOnce = false;
            throw new Error("Process stopped before checker journal commit");
          }
          await append(name, value);
        }
      );
      try {
        await expect(fixture.runtime.invokeChecker(
          {},
          callId,
          fixture.hierarchy.rootId
        )).rejects.toThrow("Process stopped before checker journal commit");
      } finally {
        appendSpy.mockRestore();
      }

      const committed = RunCheckpointSchema.parse(await fixture.store.readCheckpoint());
      expect(committed.committed_actions[transactionId]).toMatchObject({
        kind: "checker",
        accepted: true,
        code: "mission_satisfied"
      });
      expect(await fixture.store.readJournal("checker")).toHaveLength(0);

      fixture.world.dispose();
      originalWorldDisposed = true;
      const firstResume = await resumedRuntime(fixture, committed);
      resumedWorlds.push(firstResume.world);
      await firstResume.runtime.start(true);

      expect(await fixture.store.readJournal("checker")).toEqual([
        committed.committed_actions[transactionId]?.detail
      ]);
      expect((await fixture.store.readJournal("actions")).filter(
        (entry) => objectRecord(entry)?.transaction_id === transactionId
      )).toHaveLength(1);
      expect((await fixture.store.readJournal("hierarchy")).filter((entry) => {
        const record = objectRecord(entry);
        return record?.type === "action_committed" && record.transaction_id === transactionId;
      })).toHaveLength(1);
      const committedEvents = (await fixture.store.readJournal("events")).filter((entry) =>
        objectRecord(entry)?.type === "action_committed"
          && runtimeEventTransactionId(entry) === transactionId
      );
      expect(committedEvents).toHaveLength(1);
      expect(objectRecord(objectRecord(committedEvents[0])?.data)?.reconciled).toBe(true);

      const secondCheckpoint = RunCheckpointSchema.parse(await fixture.store.readCheckpoint());
      const secondResume = await resumedRuntime(fixture, secondCheckpoint);
      resumedWorlds.push(secondResume.world);
      await secondResume.runtime.start(true);

      expect(await fixture.store.readJournal("checker")).toHaveLength(1);
      expect((await fixture.store.readJournal("actions")).filter(
        (entry) => objectRecord(entry)?.transaction_id === transactionId
      )).toHaveLength(1);
      expect((await fixture.store.readJournal("events")).filter((entry) =>
        objectRecord(entry)?.type === "action_committed"
          && runtimeEventTransactionId(entry) === transactionId
      )).toHaveLength(1);
    } finally {
      for (const world of resumedWorlds) world.dispose();
      if (!originalWorldDisposed) fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("recovers a persisted v3 inflight command as rejected and never actuates it twice", async () => {
    let stopOnWorldFrames = true;
    const eventSink: RuntimeEventSink = (event) => {
      if (stopOnWorldFrames && event.type === "world_frames") {
        stopOnWorldFrames = false;
        throw new Error("Process stopped after partial world persistence");
      }
    };
    const fixture = await createRuntimeFixture(["drive_base"], eventSink);
    let resumedWorld: RapierWorld | undefined;
    let originalWorldDisposed = false;
    try {
      const input = {
        linear_meters_per_second: 0.2,
        angular_radians_per_second: 0,
        duration_seconds: 0.5
      };
      await expect(fixture.runtime.invokeSkill(
        "drive_base",
        input,
        "sdk_interrupted_action"
      )).rejects.toThrow("Process stopped after partial world persistence");

      const inflightCheckpoint = RunCheckpointSchema.parse(await fixture.store.readCheckpoint());
      const transactionId = `${fixture.activeId}:sdk_interrupted_action`;
      expect(inflightCheckpoint).toMatchObject({
        version: 3,
        status: "running",
        inflight_action: {
          transaction_id: transactionId,
          agent_id: fixture.activeId,
          name: "drive_base",
          world_before_frame: 0,
          world_before_revision: 0
        }
      });
      expect(inflightCheckpoint.world.frame).toBeGreaterThan(0);
      expect(inflightCheckpoint.world.active_command).toMatchObject({
        id: transactionId,
        skill: "drive_base",
        phase: "driving"
      });
      expect(inflightCheckpoint.committed_actions).toEqual({});

      fixture.world.dispose();
      originalWorldDisposed = true;
      resumedWorld = await RapierWorld.create(fixture.scenario, inflightCheckpoint.world);
      const resumedHierarchy = new HierarchyProjection(
        inflightCheckpoint.nodes,
        inflightCheckpoint.root_id,
        inflightCheckpoint.active_agent_id
      );
      const resumedRuntime = new HarnessRuntimeContext({
        store: fixture.store,
        goal: fixture.store.definition.goal,
        world: resumedWorld,
        hierarchy: resumedHierarchy,
        checkpoint: inflightCheckpoint
      });
      const appendMany = fixture.store.appendMany.bind(fixture.store);
      let rejectRecoveredActionsOnce = true;
      const appendManySpy = vi.spyOn(fixture.store, "appendMany").mockImplementation(
        async (name, values) => {
          if (name === "actions" && rejectRecoveredActionsOnce) {
            rejectRecoveredActionsOnce = false;
            throw new Error("Process stopped after interrupted receipt checkpoint");
          }
          await appendMany(name, values);
        }
      );
      try {
        await expect(resumedRuntime.start(true))
          .rejects.toThrow("Process stopped after interrupted receipt checkpoint");
      } finally {
        appendManySpy.mockRestore();
      }

      const interruptedCommit = RunCheckpointSchema.parse(
        await fixture.store.readCheckpoint()
      );
      expect(interruptedCommit.inflight_action).toBeNull();
      expect(interruptedCommit.committed_actions[transactionId]).toMatchObject({
        accepted: false,
        code: "command_interrupted",
        world_before_revision: 0
      });
      expect(await fixture.store.readJournal("actions")).toHaveLength(0);

      resumedWorld.dispose();
      resumedWorld = await RapierWorld.create(fixture.scenario, interruptedCommit.world);
      const reconciledHierarchy = new HierarchyProjection(
        interruptedCommit.nodes,
        interruptedCommit.root_id,
        interruptedCommit.active_agent_id,
        interruptedCommit.active_agent_ids
      );
      const reconciledRuntime = new HarnessRuntimeContext({
        store: fixture.store,
        goal: fixture.store.definition.goal,
        world: resumedWorld,
        hierarchy: reconciledHierarchy,
        checkpoint: interruptedCommit
      });
      await reconciledRuntime.start(true);

      const recovered = RunCheckpointSchema.parse(reconciledRuntime.checkpoint);
      expect(recovered.version).toBe(3);
      expect(recovered.inflight_action).toBeNull();
      expect(recovered.world.active_command).toBeNull();
      expect(recovered.world.last_command).toMatchObject({
        id: transactionId,
        accepted: false,
        result_code: "command_interrupted"
      });
      expect(recovered.committed_actions[transactionId]).toMatchObject({
        accepted: false,
        code: "command_interrupted",
        world_before_revision: 0,
        gates: expect.arrayContaining([
          { name: "exactly_once_recovery", status: "rejected", detail: { interrupted: true } }
        ])
      });
      expect(await fixture.store.readJournal("actions")).toHaveLength(1);
      expect((await fixture.store.readJournal("hierarchy")).filter((entry) => {
        const record = objectRecord(entry);
        return record?.type === "action_interrupted" && record.transaction_id === transactionId;
      })).toHaveLength(1);
      const recoveredEvents = (await fixture.store.readJournal("events")).filter((entry) =>
        runtimeEventTransactionId(entry) === transactionId
          && ["action_rejected", "command_finished", "action_committed"].includes(
            String(objectRecord(entry)?.type)
          )
      );
      expect(recoveredEvents).toHaveLength(1);
      expect(objectRecord(recoveredEvents[0])?.type).toBe("action_rejected");
      expect(objectRecord(objectRecord(recoveredEvents[0])?.data)).toMatchObject({
        recovered: true,
        reconciled: true
      });

      const frameAfterRecovery = recovered.world.frame;
      const reused = await reconciledRuntime.invokeSkill(
        "drive_base",
        input,
        "sdk_interrupted_action"
      );
      expect(outputRecord(reused)).toMatchObject({
        accepted: false,
        code: "command_interrupted",
        transaction_id: transactionId,
        world_frame: frameAfterRecovery
      });
      expect(reconciledRuntime.checkpoint.world.frame).toBe(frameAfterRecovery);
      expect(await fixture.store.readJournal("actions")).toHaveLength(1);
    } finally {
      resumedWorld?.dispose();
      if (!originalWorldDisposed) fixture.world.dispose();
      await rm(fixture.runsDir, { recursive: true, force: true });
    }
  });

  it("bounds partial world checkpoint writes without reducing live physics frames", async () => {
    const events: RuntimeEvent[] = [];
    const fixture = await createRuntimeFixture(["drive_base"], (event) => {
      events.push(event);
    });
    const originalWrite = fixture.store.writeCheckpoint.bind(fixture.store);
    const persistedFrames: number[] = [];
    const write = vi.spyOn(fixture.store, "writeCheckpoint")
      .mockImplementation(async (checkpoint) => {
        persistedFrames.push(checkpoint.world.frame);
        await originalWrite(checkpoint);
      });
    try {
      const before = fixture.runtime.checkpoint.world.frame;
      const result = outputRecord(await fixture.runtime.invokeSkill(
        "drive_base",
        {
          linear_meters_per_second: 0.2,
          angular_radians_per_second: 0,
          duration_seconds: 0.6
        },
        "sdk_bounded_partial_checkpoints"
      ));
      expect(result).toMatchObject({ accepted: true });

      const terminal = fixture.runtime.checkpoint.world.frame;
      const partial = [...new Set(persistedFrames)]
        .filter((frame) => frame > before && frame < terminal);
      expect(partial[0]).toBe(before + 3);
      for (let index = 1; index < partial.length; index += 1) {
        expect(partial[index]! - partial[index - 1]!)
          .toBeGreaterThanOrEqual(PARTIAL_WORLD_CHECKPOINT_INTERVAL_FRAMES);
      }
      expect(partial.length).toBeLessThan(
        Math.ceil((terminal - before) / 3)
      );
      expect((await fixture.store.readCheckpoint()).world.frame).toBe(terminal);

      const streamedFrames = events
        .filter((event) => event.type === "world_frames")
        .flatMap((event) => {
          if (typeof event.data !== "object" || event.data === null || Array.isArray(event.data)) {
            return [];
          }
          return Array.isArray(event.data.frames) ? event.data.frames : [];
        })
        .flatMap((frame) => {
          if (typeof frame !== "object" || frame === null || Array.isArray(frame)) return [];
          return typeof frame.frame === "number" ? [frame.frame] : [];
        });
      expect(new Set(streamedFrames).size).toBe(terminal - before);
    } finally {
      write.mockRestore();
      await disposeFixture(fixture);
    }
  });

  it("does not invent a before revision for a legacy inflight action", async () => {
    const fixture = await createRuntimeFixture(["drive_base"]);
    let restoredWorld: RapierWorld | undefined;
    try {
      const checkpoint = fixture.runtime.checkpoint;
      const transactionId = `${fixture.activeId}:legacy_inflight_action`;
      const inflight: NonNullable<RunCheckpoint["inflight_action"]> = {
        transaction_id: transactionId,
        agent_id: fixture.activeId,
        agent_name: fixture.runtime.activeNode(fixture.activeId).name,
        kind: "skill",
        name: "drive_base",
        input: {
          linear_meters_per_second: 0.2,
          angular_radians_per_second: 0,
          duration_seconds: 0.5
        },
        channels: ["base"],
        world_before_frame: checkpoint.world.frame,
        started_at: new Date().toISOString()
      };
      checkpoint.inflight_action = inflight;
      checkpoint.inflight_actions = { [transactionId]: inflight };
      await fixture.store.writeCheckpoint(checkpoint);

      const restored = await resumedRuntime(fixture, RunCheckpointSchema.parse(checkpoint));
      restoredWorld = restored.world;
      await restored.runtime.start(true);

      expect(restored.runtime.checkpoint.committed_actions[transactionId])
        .not.toHaveProperty("world_before_revision");
    } finally {
      restoredWorld?.dispose();
      await disposeFixture(fixture);
    }
  });

  it("tells a planner it cannot execute its own plan, instead of letting it replan", async () => {
    // A live run granted one child plan_base_path without execute_base_plan.
    // It planned four times waiting for a move it had no authority to make.
    const fixture = await createRuntimeFixture(["plan_base_path"]);
    try {
      const planned = outputRecord(await fixture.runtime.invokeTool(
        "plan_base_path",
        { target: { x: 1, y: 0, z: 1 } },
        "sdk_plan_only"
      ));
      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
      const detail = planned.detail as Record<string, unknown>;
      expect(detail.handoff).toContain("execute_base_plan");
      expect(detail.handoff).toContain("Do not plan again");
      expect(fixture.runtime.isCapabilityEnabled("plan_base_path", fixture.activeId)).toBe(false);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("marks a plan for explicit execution when the same agent may execute it", async () => {
    const fixture = await createRuntimeFixture(["plan_base_path", "execute_base_plan"]);
    try {
      const planned = outputRecord(await fixture.runtime.invokeTool(
        "plan_base_path",
        { target: { x: 1, y: 0, z: 1 } },
        "sdk_plan_and_execute"
      ));
      expect(planned).toMatchObject({ accepted: true });
      expect(planned.detail).toMatchObject({
        execution_required: {
          automatic_actuation: false,
          tool: "execute_base_plan",
          planning_transaction_id: `${fixture.activeId}:sdk_plan_and_execute`,
          available_to_current_agent: true
        }
      });
      expect((planned.detail as Record<string, unknown>).handoff).toBeUndefined();
      expect(fixture.runtime.isCapabilityEnabled("plan_base_path", fixture.activeId)).toBe(true);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("refuses to keep running an identical call that the world already denied", async () => {
    // A live run committed 594 denials of the same three IK targets while the
    // world stood still. Each was decidable in advance, so the harness now
    // declines to spend the provider budget re-proving it.
    const fixture = await createRuntimeFixture(["plan_base_path"]);
    try {
      // Inside the arena but inside the barrier obstacle, so it cannot be planned.
      const input = { target: { x: -20, y: 0, z: -20 } };
      const codes: string[] = [];
      for (let attempt = 0; attempt < REPEATED_DENIAL_LIMIT + 2; attempt += 1) {
        const output = await fixture.runtime.invokeTool(
          "plan_base_path",
          input,
          `sdk_repeat_${attempt}`
        );
        codes.push(outputRecord(output).code as string);
      }

      // Every attempt is still recorded — the agent is told, not silently dropped.
      expect(codes).toHaveLength(REPEATED_DENIAL_LIMIT + 2);
      expect(codes.slice(0, REPEATED_DENIAL_LIMIT).every((code) => code !== "repeated_denied_action"))
        .toBe(true);
      expect(codes.slice(REPEATED_DENIAL_LIMIT)).toEqual([
        "repeated_denied_action",
        "repeated_denied_action"
      ]);

      const receipts = await fixture.store.readJournal("actions");
      const refusal = receipts.at(-1) as Record<string, unknown>;
      const detail = refusal.detail as Record<string, unknown>;
      expect(detail.recovery).toContain("report_blocked");
      expect(detail.world_revision).toBe(fixture.runtime.checkpoint.world.world_revision);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("lets an agent proceed as soon as its request actually differs", async () => {
    const fixture = await createRuntimeFixture(["plan_base_path"]);
    try {
      const blocked = { target: { x: -20, y: 0, z: -20 } };
      for (let attempt = 0; attempt < REPEATED_DENIAL_LIMIT; attempt += 1) {
        await fixture.runtime.invokeTool("plan_base_path", blocked, `sdk_blocked_${attempt}`);
      }
      expect(outputRecord(await fixture.runtime.invokeTool(
        "plan_base_path",
        blocked,
        "sdk_blocked_final"
      ))).toMatchObject({ code: "repeated_denied_action" });

      // A reachable target is a different question and must be answered.
      const reachable = await fixture.runtime.invokeTool(
        "plan_base_path",
        { target: { x: 1, y: 0, z: 1 } },
        "sdk_reachable"
      );
      expect(outputRecord(reachable)).toMatchObject({
        accepted: true,
        code: "base_path_planned"
      });
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("refuses to keep re-reading a world that has not changed", async () => {
    // A live run spent eleven consecutive read_proprioception calls at one
    // revision. Each was accepted and each returned the identical snapshot, so
    // nothing in the world objected — the repetition itself was the waste.
    const fixture = await createRuntimeFixture(["read_proprioception", "plan_base_path"]);
    try {
      const codes: string[] = [];
      for (let attempt = 0; attempt < REPEATED_DENIAL_LIMIT + 2; attempt += 1) {
        const output = await fixture.runtime.invokeTool(
          "read_proprioception",
          {},
          `sdk_reread_${attempt}`
        );
        codes.push(outputRecord(output).code as string);
      }

      expect(codes[0]).toBe("proprioception");
      expect(codes.slice(1).every((code) => code === "repeated_accepted_action"))
        .toBe(true);

      // A different capability at the same revision is a different question.
      expect(outputRecord(await fixture.runtime.invokeTool(
        "plan_base_path",
        { target: { x: 1, y: 0, z: 1 } },
        "sdk_after_reread"
      ))).toMatchObject({ accepted: true, code: "base_path_planned" });
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("does not replay an already-satisfied head target after another channel changes", async () => {
    // A real parallel run completed the head command while arm/base siblings
    // were still moving. Their later revisions made the same head input look
    // new even though the live head joints still held the accepted target.
    const fixture = await createRuntimeFixture(["set_head_target", "drive_base"]);
    try {
      const headInput = {
        yaw: 0.4,
        pitch: -0.2,
        options: { tolerance: 0.01 }
      };
      const first = outputRecord(await fixture.runtime.invokeSkill(
        "set_head_target",
        headInput,
        "sdk_initial_head_target"
      ));
      expect(first).toMatchObject({ accepted: true, code: "head_target_reached" });
      const firstRevision = fixture.runtime.checkpoint.world.world_revision;

      expect(outputRecord(await fixture.runtime.invokeSkill(
        "drive_base",
        {
          linear_meters_per_second: 0.2,
          angular_radians_per_second: 0,
          duration_seconds: 0.25
        },
        "sdk_unrelated_base_revision"
      ))).toMatchObject({ accepted: true });
      expect(fixture.runtime.checkpoint.world.world_revision).toBeGreaterThan(firstRevision);

      const repeated = outputRecord(await fixture.runtime.invokeSkill(
        "set_head_target",
        headInput,
        "sdk_redundant_head_target"
      ));
      expect(repeated).toMatchObject({
        accepted: false,
        code: "repeated_accepted_action",
        detail: {
          previous_transaction_id: `${fixture.activeId}:sdk_initial_head_target`,
          previous_world_revision: firstRevision
        }
      });
      expect(String((repeated.detail as Record<string, unknown>).recovery))
        .toContain("unrelated world changes");
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("refuses a repeated unmet check and never refuses a satisfied one", async () => {
    // A live run's coordinator called check_mission three times against an
    // unchanged world instead of delegating the work the verdict implied.
    const fixture = await createRuntimeFixture(["drive_base"]);
    try {
      // The fixture's goal holds at the start pose, so the robot has to leave it
      // for the check to be unmet.
      expect(outputRecord(await fixture.runtime.invokeSkill(
        "drive_base",
        { linear_meters_per_second: 0.4, angular_radians_per_second: 0, duration_seconds: 2 },
        "sdk_leave_goal_pose"
      ))).toMatchObject({ accepted: true });
      await fixture.runtime.completeChild(fixture.activeId, "Left the goal pose");

      const codes: string[] = [];
      for (let attempt = 0; attempt < REPEATED_DENIAL_LIMIT + 2; attempt += 1) {
        codes.push(outputRecord(
          await fixture.runtime.invokeChecker({}, `sdk_recheck_${attempt}`)
        ).code as string);
      }
      expect(codes.slice(0, REPEATED_DENIAL_LIMIT).every((code) => code === "mission_incomplete"))
        .toBe(true);
      expect(codes.slice(REPEATED_DENIAL_LIMIT)).toEqual([
        "repeated_denied_action",
        "repeated_denied_action"
      ]);
      expect(fixture.runtime.checkerSatisfiedCurrentWorld()).toBe(false);
    } finally {
      await disposeFixture(fixture);
    }

    // A satisfied verdict is the one call that ends the run, so it stays
    // available however many times it is asked for.
    const satisfied = await createRuntimeFixture(["drive_base"]);
    try {
      await satisfied.runtime.completeChild(satisfied.activeId, "Already at the goal pose");
      for (let attempt = 0; attempt < REPEATED_DENIAL_LIMIT + 3; attempt += 1) {
        expect(outputRecord(
          await satisfied.runtime.invokeChecker({}, `sdk_confirm_${attempt}`)
        )).toMatchObject({ accepted: true, code: "mission_satisfied" });
      }
      expect(satisfied.runtime.checkerSatisfiedCurrentWorld()).toBe(true);
    } finally {
      await disposeFixture(satisfied);
    }
  }, 20_000);

  it("hands a granted reference its measurement and not just its name", async () => {
    const fixture = await createRuntimeFixture(["inspect_entity"]);
    try {
      const observed = outputRecord(await fixture.runtime.invokeTool(
        "inspect_entity",
        { entity_id: "center_column" },
        "sdk_observe_for_handoff"
      ));
      expect(observed).toMatchObject({ accepted: true, code: "entity_state" });

      const references = fixture.runtime.acceptedActionReferences([
        `${fixture.activeId}:sdk_observe_for_handoff`
      ]);
      expect(references).toEqual([
        { name: "inspect_entity", transaction_id: `${fixture.activeId}:sdk_observe_for_handoff` }
      ]);

      // The id alone is what a child used to receive, and it is not enough to
      // act on: the position it has to pass to plan_base_path lives in the
      // receipt, not in its name.
      const delivered = fixture.runtime.referencedReceipts(references) as Array<
        Record<string, unknown>
      >;
      expect(delivered).toHaveLength(1);
      expect(delivered[0]).toMatchObject({
        name: "inspect_entity",
        transaction_id: `${fixture.activeId}:sdk_observe_for_handoff`,
        code: "entity_state",
        input: { entity_id: "center_column" }
      });
      expect((delivered[0]?.result as Record<string, unknown>).position)
        .toEqual((observed.detail as Record<string, unknown>).position);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("marks a handed-over measurement stale once the world has moved past it", async () => {
    const fixture = await createRuntimeFixture(["inspect_entity", "drive_base"]);
    try {
      await fixture.runtime.invokeTool(
        "inspect_entity",
        { entity_id: "center_column" },
        "sdk_observe_before_motion"
      );
      const references = fixture.runtime.acceptedActionReferences([
        `${fixture.activeId}:sdk_observe_before_motion`
      ]);
      const fresh = fixture.runtime.referencedReceipts(references) as Array<
        Record<string, unknown>
      >;
      expect(fresh[0]).toMatchObject({ stale: false });

      // Any real body command moves the revision, which is exactly when a
      // position read earlier stops describing the present.
      await fixture.runtime.invokeSkill(
        "drive_base",
        { linear_meters_per_second: 0.2, angular_radians_per_second: 0, duration_seconds: 0.5 },
        "sdk_move_past_the_measurement"
      );

      const afterMotion = fixture.runtime.referencedReceipts(references) as Array<
        Record<string, unknown>
      >;
      expect(afterMotion[0]).toMatchObject({ stale: true });
      expect(String(afterMotion[0]?.recovery)).toContain("Re-observe");
      expect(afterMotion[0]?.measured_at_world_revision)
        .not.toBe(afterMotion[0]?.current_world_revision);
    } finally {
      await disposeFixture(fixture);
    }
  });

  it("marks a reference unavailable rather than passing off a rejection as a measurement", async () => {
    const fixture = await createRuntimeFixture(["inspect_entity"]);
    try {
      const missing = fixture.runtime.referencedReceipts([
        { name: "inspect_entity", transaction_id: `${fixture.activeId}:never_committed` }
      ]) as Array<Record<string, unknown>>;
      expect(missing).toEqual([
        {
          name: "inspect_entity",
          transaction_id: `${fixture.activeId}:never_committed`,
          available: false
        }
      ]);
    } finally {
      await disposeFixture(fixture);
    }
  });
});
