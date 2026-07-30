import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { RunCheckpointSchema, type AgentSpec, type Goal, type JsonValue } from "../domain/schema.js";
import { RunStore } from "../persistence/run-store.js";
import { RapierWorld } from "../world/rapier-world.js";
import { capabilityCatalog } from "./agents.js";
import { receiptEvidenceRequirement } from "./evidence-contract.js";
import { HierarchyProjection } from "./hierarchy-projection.js";
import { createCheckpoint, HarnessRuntimeContext, type RuntimeEvent } from "./runtime-context.js";

interface Fixture {
  root: string;
  runsDir: string;
  store: RunStore;
  world: RapierWorld;
  runtime: HarnessRuntimeContext;
  first: string;
  second: string;
}

async function fixture(
  capabilities: string[],
  events: RuntimeEvent[] = [],
  stop?: (event: RuntimeEvent) => void | Promise<void>
): Promise<Fixture> {
  const runsDir = await mkdtemp(join(tmpdir(), "hear-concurrency-"));
  const catalog = await loadRuntimeCatalog();
  const scenario = catalog.materialize("open_navigation", 0);
  const goal: Goal = {
    summary: "Keep the robot inside the world.",
    predicates: [{ type: "robot_in_zone", zone_id: "goal", tolerance: 20 }]
  };
  const store = await RunStore.create(runsDir, {
    mission: "Exercise independent body channels",
    scenarioId: "open_navigation",
    scenario,
    goal
  });
  const world = await RapierWorld.create(scenario);
  const hierarchy = HierarchyProjection.create("Exercise independent body channels", capabilityCatalog());
  const checkpoint = createCheckpoint({ store, hierarchy, capabilityCatalog: capabilityCatalog(), world });
  await store.writeCheckpoint(checkpoint);
  const runtime = new HarnessRuntimeContext({
    store,
    goal,
    world,
    hierarchy,
    checkpoint,
    eventSink: (event) => {
      events.push(event);
      return stop?.(event);
    }
  });
  await runtime.start();
  const spec = (name: string): AgentSpec => ({
    name,
    objective: `Operate the ${capabilities.join(" and ")} capability grant`,
    success_criteria: ["A source-backed action receipt is committed."],
    evidence_requirements: [capabilities[0] === "read_proprioception"
      ? receiptEvidenceRequirement(0, "read_proprioception", { kind: "robot" })
      : receiptEvidenceRequirement(0, "drive_base", { kind: "body", channel: "base" })],
    capabilities,
    may_delegate: false,
    references: []
  });
  const first = await runtime.beginDelegation(null, spec("Parallel worker A"), "parallel_a");
  const second = await runtime.beginDelegation(
    null,
    spec("Parallel worker B"),
    "parallel_b",
    hierarchy.rootId
  );
  return {
    root: hierarchy.rootId,
    runsDir,
    store,
    world,
    runtime,
    first: first.node.id,
    second: second.node.id
  };
}

async function dispose(value: Pick<Fixture, "runsDir" | "world">): Promise<void> {
  value.world.dispose();
  await rm(value.runsDir, { recursive: true, force: true });
}

function output(value: string): Record<string, unknown> {
  return JSON.parse(value) as Record<string, unknown>;
}

describe("concurrent hierarchy actuation", () => {
  it("keeps concurrent provider and framework telemetry on the originating sibling", async () => {
    const value = await fixture(["read_proprioception"]);
    try {
      await Promise.all([
        value.runtime.recordProvider({ status: "first_model" }, value.first),
        value.runtime.recordProvider({ status: "second_model" }, value.second),
        value.runtime.recordFramework("first-stream", { type: "first_event" }, value.first),
        value.runtime.recordFramework("second-stream", { type: "second_event" }, value.second)
      ]);

      const provider = await value.store.readJournal("provider") as Array<Record<string, JsonValue>>;
      const framework = await value.store.readJournal("framework") as Array<Record<string, JsonValue>>;
      expect(provider).toEqual(expect.arrayContaining([
        expect.objectContaining({ status: "first_model", agent_id: value.first }),
        expect.objectContaining({ status: "second_model", agent_id: value.second })
      ]));
      expect(framework).toEqual(expect.arrayContaining([
        expect.objectContaining({ scope: "first-stream", agent_id: value.first }),
        expect.objectContaining({ scope: "second-stream", agent_id: value.second })
      ]));
    } finally {
      await dispose(value);
    }
  });

  it("restores legacy singular focus and inflight state", async () => {
    const value = await fixture(["drive_base"]);
    try {
      const current = value.runtime.checkpoint;
      const legacy = structuredClone(current) as unknown as Record<string, unknown>;
      delete legacy.active_agent_ids;
      delete legacy.inflight_actions;
      delete legacy.spatial_memory;
      delete legacy.context_memory;
      delete legacy.pending_lifecycle_events;
      const world = legacy.world as Record<string, unknown>;
      delete world.active_commands;

      const parsed = RunCheckpointSchema.parse(legacy);
      const hierarchy = new HierarchyProjection(
        parsed.nodes,
        parsed.root_id,
        parsed.active_agent_id,
        parsed.active_agent_ids
      );
      expect(hierarchy.activeIds).toEqual([parsed.active_agent_id]);
      expect(parsed.inflight_actions).toEqual({});
      expect(parsed.spatial_memory).toEqual([]);
      expect(parsed.world.active_commands).toEqual([]);
      expect(parsed.context_memory.total_compactions).toBe(0);
      expect(parsed.pending_lifecycle_events).toEqual([]);
    } finally {
      await dispose(value);
    }
  });

  it("runs independent sibling channels together and attributes both receipts", async () => {
    const events: RuntimeEvent[] = [];
    const value = await fixture(["drive_base", "set_head_target"], events);
    try {
      const [base, head] = await Promise.all([
        value.runtime.invokeSkill("drive_base", {
          linear_meters_per_second: 0.12,
          angular_radians_per_second: 0,
          duration_seconds: 0.35
        }, "base_call", value.first),
        value.runtime.invokeSkill("set_head_target", {
          yaw: 0.35,
          pitch: -0.14
        }, "head_call", value.second)
      ]);

      expect(output(base)).toMatchObject({ accepted: true, transaction_id: `${value.first}:base_call` });
      expect(output(head)).toMatchObject({ accepted: true, transaction_id: `${value.second}:head_call` });
      expect(value.runtime.checkpoint.active_agent_ids).toEqual(expect.arrayContaining([
        value.first,
        value.second
      ]));
      const frames = events
        .filter((event) => event.type === "world_frames")
        .flatMap((event) => {
          const data = event.data as Record<string, JsonValue>;
          return Array.isArray(data.frames) ? data.frames : [];
        }) as Array<Record<string, JsonValue>>;
      expect(frames.some((frame) =>
        Array.isArray(frame.active_commands) && frame.active_commands.length === 2
      )).toBe(true);
      expect(value.runtime.checkpoint.inflight_actions).toEqual({});
    } finally {
      await dispose(value);
    }
  });

  it("lets three sibling agents lease base, head, and arm in shared physics frames", async () => {
    const events: RuntimeEvent[] = [];
    const value = await fixture(
      ["drive_base", "set_head_target", "plan_joint_targets", "execute_joint_plan"],
      events
    );
    const thirdEntry = await value.runtime.beginDelegation(
      null,
      {
        name: "Parallel worker C",
        objective: "Operate the arm capability grant",
        success_criteria: ["A source-backed action receipt is committed."],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "drive_base",
          { kind: "body", channel: "base" }
        )],
        capabilities: ["drive_base", "set_head_target", "plan_joint_targets", "execute_joint_plan"],
        may_delegate: false,
        references: []
      },
      "parallel_c",
      value.root
    );
    try {
      const plannedArm = output(await value.runtime.invokeTool("plan_joint_targets", {
        targets: { shoulder: 1.2, elbow: -2.1, wrist: 0.9 }
      }, "triple_arm_plan", thirdEntry.node.id));
      expect(plannedArm).toMatchObject({ accepted: true, code: "joint_target_plan" });
      const armPlanningTransaction = `${thirdEntry.node.id}:triple_arm_plan`;

      const [base, head, arm] = await Promise.all([
        value.runtime.invokeSkill("drive_base", {
          linear_meters_per_second: 0.12,
          angular_radians_per_second: 0,
          duration_seconds: 0.35
        }, "triple_base", value.first),
        value.runtime.invokeSkill("set_head_target", {
          yaw: 0.25,
          pitch: -0.12
        }, "triple_head", value.second),
        value.runtime.invokeSkill("execute_joint_plan", {
          planning_transaction_id: armPlanningTransaction
        }, "triple_arm", thirdEntry.node.id)
      ]);

      expect([base, head, arm].map(output)).toEqual([
        expect.objectContaining({ accepted: true, transaction_id: `${value.first}:triple_base` }),
        expect.objectContaining({ accepted: true, transaction_id: `${value.second}:triple_head` }),
        expect.objectContaining({ accepted: true, transaction_id: `${thirdEntry.node.id}:triple_arm` })
      ]);
      const frames = events
        .filter((event) => event.type === "world_frames")
        .flatMap((event) => {
          const data = event.data as Record<string, JsonValue>;
          return Array.isArray(data.frames) ? data.frames : [];
        }) as Array<Record<string, JsonValue>>;
      expect(frames.some((frame) => {
        if (!Array.isArray(frame.active_commands)) return false;
        const channels = new Set((frame.active_commands as Array<Record<string, JsonValue>>)
          .flatMap((command) => Array.isArray(command.channels) ? command.channels : []));
        return channels.has("base") && channels.has("head") && channels.has("arm");
      })).toBe(true);
      expect(value.runtime.checkpoint.inflight_actions).toEqual({});
    } finally {
      await dispose(value);
    }
  });

  it("enforces a stationary base for fixed-world IK planning and execution", async () => {
    let releaseLease!: () => void;
    let observeLease!: () => void;
    let blockedTransaction = "";
    let leaseObserved = new Promise<void>((resolve) => {
      observeLease = resolve;
    });
    let leaseRelease = new Promise<void>((resolve) => {
      releaseLease = resolve;
    });
    const value = await fixture(
      ["drive_base", "solve_end_effector_position", "execute_joint_plan"],
      [],
      async (event) => {
        if (event.type !== "body_lease_acquired") return;
        const transactionId = (event.data as Record<string, JsonValue>).transaction_id;
        if (transactionId !== blockedTransaction) return;
        observeLease();
        await leaseRelease;
      }
    );
    try {
      const position = value.world.snapshot().robot.links.gripper?.position;
      if (!position) throw new Error("Fixture robot has no gripper link");
      const planned = output(await value.runtime.invokeTool(
        "solve_end_effector_position",
        { position },
        "fixed_world_plan",
        value.first
      ));
      expect(planned).toMatchObject({ accepted: true, code: "end_effector_solution" });

      blockedTransaction = `${value.first}:fixed_world_execute`;
      const fixedExecution = value.runtime.invokeSkill("execute_joint_plan", {
        planning_transaction_id: `${value.first}:fixed_world_plan`
      }, "fixed_world_execute", value.first);
      await leaseObserved;
      const competingBase = output(await value.runtime.invokeSkill("drive_base", {
        linear_meters_per_second: 0.1,
        angular_radians_per_second: 0,
        duration_seconds: 0.2
      }, "base_during_fixed_ik", value.second));
      expect(competingBase).toMatchObject({ accepted: false, code: "body_channel_busy" });
      releaseLease();
      expect(output(await fixedExecution)).toMatchObject({
        accepted: true,
        transaction_id: `${value.first}:fixed_world_execute`
      });
      expect(value.runtime.checkpoint.committed_actions[`${value.first}:fixed_world_execute`]?.channels)
        .toEqual(["arm", "base"]);

      blockedTransaction = `${value.second}:base_during_fixed_plan`;
      leaseObserved = new Promise<void>((resolve) => {
        observeLease = resolve;
      });
      leaseRelease = new Promise<void>((resolve) => {
        releaseLease = resolve;
      });
      const movingBase = value.runtime.invokeSkill("drive_base", {
        linear_meters_per_second: 0.08,
        angular_radians_per_second: 0,
        duration_seconds: 0.2
      }, "base_during_fixed_plan", value.second);
      await leaseObserved;
      const competingPlan = output(await value.runtime.invokeTool(
        "solve_end_effector_position",
        { position: { ...position, y: position.y + 0.02 } },
        "fixed_plan_during_base",
        value.first
      ));
      expect(competingPlan).toMatchObject({ accepted: false, code: "body_channel_busy" });
      releaseLease();
      expect(output(await movingBase).accepted).toBe(true);
    } finally {
      releaseLease();
      await dispose(value);
    }
  });

  it("grants a conflicting body channel to only one sibling", async () => {
    const value = await fixture(["drive_base"]);
    try {
      const results = await Promise.all([
        value.runtime.invokeSkill("drive_base", {
          linear_meters_per_second: 0.1,
          angular_radians_per_second: 0,
          duration_seconds: 0.3
        }, "base_a", value.first),
        value.runtime.invokeSkill("drive_base", {
          linear_meters_per_second: 0.08,
          angular_radians_per_second: 0,
          duration_seconds: 0.3
        }, "base_b", value.second)
      ]);
      const parsed = results.map(output);
      expect(parsed.filter((result) => result.accepted === true)).toHaveLength(1);
      expect(parsed.filter((result) => result.code === "body_channel_busy")).toHaveLength(1);
    } finally {
      await dispose(value);
    }
  });

  it("recovers every interrupted sibling command without replaying actuation", async () => {
    let interrupt = true;
    let started = 0;
    let releaseStart!: () => void;
    const bothStarted = new Promise<void>((resolve) => {
      releaseStart = resolve;
    });
    const value = await fixture(
      ["drive_base", "set_head_target"],
      [],
      (event) => {
        if (event.type === "command_started") {
          started += 1;
          if (started === 2) {
            releaseStart();
            return;
          }
          return bothStarted;
        }
        if (interrupt && event.type === "world_frames") {
          interrupt = false;
          throw new Error("simulated concurrent process stop");
        }
      }
    );
    let resumedWorld: RapierWorld | undefined;
    let originalDisposed = false;
    try {
      const outcomes = await Promise.allSettled([
        value.runtime.invokeSkill("drive_base", {
          linear_meters_per_second: 0.12,
          angular_radians_per_second: 0,
          duration_seconds: 0.6
        }, "interrupted_base", value.first),
        value.runtime.invokeSkill("set_head_target", {
          yaw: 1.2,
          pitch: -0.2,
          options: { max_velocity: 0.5, max_duration_seconds: 6 }
        }, "interrupted_head", value.second)
      ]);
      expect(outcomes.every((result) => result.status === "rejected")).toBe(true);

      const interrupted = RunCheckpointSchema.parse(await value.store.readCheckpoint());
      expect(Object.keys(interrupted.inflight_actions)).toHaveLength(2);
      expect(interrupted.world.active_commands).toHaveLength(2);

      value.world.dispose();
      originalDisposed = true;
      resumedWorld = await RapierWorld.create(value.store.definition.scenario, interrupted.world);
      const hierarchy = new HierarchyProjection(
        interrupted.nodes,
        interrupted.root_id,
        interrupted.active_agent_id,
        interrupted.active_agent_ids
      );
      const resumed = new HarnessRuntimeContext({
        store: value.store,
        goal: value.store.definition.goal,
        world: resumedWorld,
        hierarchy,
        checkpoint: interrupted
      });
      await resumed.start(true);

      const recovered = resumed.checkpoint;
      expect(recovered.inflight_actions).toEqual({});
      expect(recovered.inflight_action).toBeNull();
      expect(recovered.world.active_commands).toEqual([]);
      expect(recovered.committed_actions[`${value.first}:interrupted_base`]).toMatchObject({
        accepted: false,
        code: "command_interrupted"
      });
      expect(recovered.committed_actions[`${value.second}:interrupted_head`]).toMatchObject({
        accepted: false,
        code: "command_interrupted"
      });
      expect(await value.store.readJournal("actions")).toHaveLength(2);
    } finally {
      resumedWorld?.dispose();
      if (!originalDisposed) value.world.dispose();
      await rm(value.runsDir, { recursive: true, force: true });
    }
  });
});
