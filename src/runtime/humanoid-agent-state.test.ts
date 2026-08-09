import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  afterEach,
  describe,
  expect,
  it,
  vi
} from "vitest";
import { EmptyActionCommitOutbox } from "../domain/action-commit-outbox.js";
import { EmptyActionExecutionLedger } from "../domain/action-execution-ledger.js";
import { FileSession } from "../persistence/file-session.js";
import {
  captureHumanoidSessionBaseline,
  captureHumanoidSessionStateIdentity,
  humanoidAgentStateFingerprint,
  humanoidSessionBaselineIdentity,
  restoreHumanoidSessionBaseline,
  restoreHumanoidSessionStateBaseline,
  restoreHumanoidSessionStateBaselineDetailed,
  type HumanoidAgentStateCheckpoint,
  type HumanoidSessionBaseline
} from "./humanoid-agent-state.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => (
    rm(directory, { recursive: true, force: true })
  )));
});

describe("humanoid Agent state recovery identity", () => {
  it("ignores passive world advancement but retains temporal Goal progress", () => {
    const before = checkpoint();
    const after = structuredClone(before);
    after.world.frame += 2_000;
    after.world.worldRevision += 2_000;
    after.world.rootPosition = [2.0004, 0.8098, 1.9997];
    after.goal_progress!.last_world_frame += 2_000;
    after.goal_progress!.last_world_revision += 2_000;

    expect(humanoidAgentStateFingerprint(after)).toBe(
      humanoidAgentStateFingerprint(before)
    );

    after.goal_progress!.predicate_streaks[0] = 2;
    expect(humanoidAgentStateFingerprint(after)).not.toBe(
      humanoidAgentStateFingerprint(before)
    );
  });

  it.each([
    ["Goal DAG", (value: FingerprintFixture) => {
      value.goal_dag.state_sha256 = "b".repeat(64);
    }],
    ["committed action", (value: FingerprintFixture) => {
      value.committed_actions["motion-2"] = {};
    }],
    ["action commit outbox", (value: FingerprintFixture) => {
      value.action_commit_outbox.pending["motion-2"] = {} as never;
    }],
    ["action execution ledger", (value: FingerprintFixture) => {
      value.action_execution_ledger.active["motion-2"] = {} as never;
    }],
    ["context compaction", (value: FingerprintFixture) => {
      value.context_memory.total_compactions += 1;
    }],
    ["cycle index", (value: FingerprintFixture) => {
      value.cycle_index += 1;
    }],
    ["active cycle", (value: FingerprintFixture) => {
      value.active_cycle = { cycle_id: "cycle-2" };
    }]
  ] as const)("invalidates state after a causal %s change", (_name, mutate) => {
    const before = checkpoint();
    const after = structuredClone(before);
    mutate(after);
    expect(humanoidAgentStateFingerprint(after)).not.toBe(
      humanoidAgentStateFingerprint(before)
    );
  });

  it("hashes record fields canonically", () => {
    const left = checkpoint();
    left.committed_actions.first = {};
    left.committed_actions.second = {};
    const right = checkpoint();
    right.committed_actions.second = {};
    right.committed_actions.first = {};

    expect(humanoidAgentStateFingerprint(left)).toBe(
      humanoidAgentStateFingerprint(right)
    );
  });

  it("restores every hierarchy Session to one call baseline", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-agent-session-baseline-"));
    temporaryDirectories.push(directory);
    const sessions = new Map([
      ["coordinator", new FileSession(join(directory, "coordinator.json"), "coordinator")],
      ["worker", new FileSession(join(directory, "worker.json"), "worker")]
    ]);
    await sessions.get("coordinator")!.addItems([
      { role: "user", content: "coordinator baseline" }
    ]);
    await sessions.get("worker")!.addItems([
      { role: "user", content: "worker baseline" }
    ]);
    const baseline = await captureHumanoidSessionBaseline(sessions);

    await Promise.all([...sessions.values()].map((session) => session.addItems([
      { role: "user", content: "failed transport branch" }
    ])));
    expect(await restoreHumanoidSessionBaseline(sessions, baseline)).toEqual([
      "coordinator",
      "worker"
    ]);

    expect(await sessions.get("coordinator")!.getItems()).toEqual([
      { role: "user", content: "coordinator baseline" }
    ]);
    expect(await sessions.get("worker")!.getItems()).toEqual([
      { role: "user", content: "worker baseline" }
    ]);
  });

  it("reports only Sessions whose durable history was actually rolled back", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-agent-session-changed-"));
    temporaryDirectories.push(directory);
    const sessions = new Map([
      ["coordinator", new FileSession(join(directory, "coordinator.json"), "coordinator")],
      ["worker", new FileSession(join(directory, "worker.json"), "worker")]
    ]);
    await sessions.get("coordinator")!.addItems([
      { role: "user", content: "coordinator baseline" }
    ]);
    await sessions.get("worker")!.addItems([
      { role: "user", content: "worker baseline" }
    ]);
    const baseline = await captureHumanoidSessionBaseline(sessions);
    await sessions.get("worker")!.addItems([
      { role: "user", content: "abandoned suffix" }
    ]);

    expect(await restoreHumanoidSessionBaseline(sessions, baseline)).toEqual(["worker"]);
    expect(await restoreHumanoidSessionBaseline(sessions, baseline)).toEqual([]);
  });

  it("rejects a changed Session set before restoring any call baseline", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-agent-session-set-"));
    temporaryDirectories.push(directory);
    const coordinator = new FileSession(
      join(directory, "coordinator.json"),
      "coordinator"
    );
    const replacement = new FileSession(
      join(directory, "replacement.json"),
      "replacement"
    );
    await coordinator.addItems([{ role: "user", content: "current suffix" }]);
    const sessions = new Map([
      ["coordinator", coordinator],
      ["replacement", replacement]
    ]);
    const baseline: HumanoidSessionBaseline = new Map([
      ["coordinator", [{ role: "user", content: "older baseline" }]],
      ["missing-worker", []]
    ]);

    await expect(restoreHumanoidSessionBaseline(sessions, baseline)).rejects.toThrow(
      "missing-worker"
    );
    expect(await coordinator.getItems()).toEqual([
      { role: "user", content: "current suffix" }
    ]);
  });

  it("trims every Session to the exact baseline paired with a RunState", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-agent-session-state-"));
    temporaryDirectories.push(directory);
    const sessions = new Map([
      ["coordinator", new FileSession(join(directory, "coordinator.json"), "coordinator")],
      ["worker", new FileSession(join(directory, "worker.json"), "worker")]
    ]);
    await sessions.get("coordinator")!.addItems([
      { role: "user", content: "coordinator state input" }
    ]);
    await sessions.get("worker")!.addItems([
      { role: "user", content: "worker history" }
    ]);
    const identity = humanoidSessionBaselineIdentity(
      await captureHumanoidSessionBaseline(sessions)
    );

    await sessions.get("coordinator")!.addItems([
      { role: "user", content: "root suffix after persisted state" }
    ]);
    await sessions.get("worker")!.addItems([
      { role: "user", content: "failed nested call input" }
    ]);

    expect(await restoreHumanoidSessionStateBaseline(sessions, identity)).toBe(true);
    expect(await sessions.get("coordinator")!.getItems()).toEqual([
      { role: "user", content: "coordinator state input" }
    ]);
    expect(await sessions.get("worker")!.getItems()).toEqual([
      { role: "user", content: "worker history" }
    ]);
  });

  it("reports the exact Session prefixes removed for context rollback", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-agent-session-state-detail-"));
    temporaryDirectories.push(directory);
    const coordinator = new FileSession(join(directory, "coordinator.json"), "coordinator");
    const motion = new FileSession(join(directory, "motion.json"), "motion");
    const sessions = new Map([
      ["coordinator", coordinator],
      ["motion", motion]
    ]);
    await coordinator.addItems([{ role: "user", content: "stable coordinator" }]);
    await motion.addItems([{ role: "user", content: "stable motion" }]);
    const baseline = humanoidSessionBaselineIdentity(
      await captureHumanoidSessionBaseline(sessions)
    );
    await motion.addItems([{ role: "assistant", content: "abandoned model suffix" }]);

    const restored = await restoreHumanoidSessionStateBaselineDetailed(
      sessions,
      baseline
    );

    expect(restored.compatible).toBe(true);
    expect([...restored.restored.keys()]).toEqual(["motion"]);
    expect(restored.restored.get("motion")).toEqual([
      { role: "user", content: "stable motion" }
    ]);
    expect(await motion.getItems()).toEqual(restored.restored.get("motion"));
  });

  it("captures the persisted identity without cloning Session item arrays", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-agent-session-identity-"));
    temporaryDirectories.push(directory);
    const session = new FileSession(join(directory, "coordinator.json"), "coordinator");
    const sessions = new Map([["coordinator", session]]);
    await session.addItems([{ role: "user", content: "stable history" }]);
    const snapshotIdentity = humanoidSessionBaselineIdentity(
      await captureHumanoidSessionBaseline(sessions)
    );
    const getItems = vi.spyOn(session, "getItems");

    const first = await captureHumanoidSessionStateIdentity(sessions);
    const second = await captureHumanoidSessionStateIdentity(sessions);

    expect(first).toEqual(snapshotIdentity);
    expect(second).toEqual(first);
    expect(getItems).not.toHaveBeenCalled();
    await session.addItems([{ role: "assistant", content: "new history" }]);
    const changed = await captureHumanoidSessionStateIdentity(sessions);
    expect(changed.coordinator?.item_count).toBe(2);
    expect(changed.coordinator?.items_sha256).not.toBe(
      first.coordinator?.items_sha256
    );
  });

  it("rejects a RunState baseline without partially rewriting divergent Sessions", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-agent-session-divergence-"));
    temporaryDirectories.push(directory);
    const sessions = new Map([
      ["coordinator", new FileSession(join(directory, "coordinator.json"), "coordinator")],
      ["worker", new FileSession(join(directory, "worker.json"), "worker")]
    ]);
    await sessions.get("coordinator")!.addItems([
      { role: "user", content: "coordinator baseline" }
    ]);
    await sessions.get("worker")!.addItems([
      { role: "user", content: "worker baseline" }
    ]);
    const identity = humanoidSessionBaselineIdentity(
      await captureHumanoidSessionBaseline(sessions)
    );
    await sessions.get("coordinator")!.addItems([
      { role: "user", content: "valid removable suffix" }
    ]);
    await sessions.get("worker")!.replaceItems([
      { role: "user", content: "divergent worker prefix" }
    ]);

    expect(await restoreHumanoidSessionStateBaseline(sessions, identity)).toBe(false);
    expect(await sessions.get("coordinator")!.getItems()).toEqual([
      { role: "user", content: "coordinator baseline" },
      { role: "user", content: "valid removable suffix" }
    ]);
    expect(await sessions.get("worker")!.getItems()).toEqual([
      { role: "user", content: "divergent worker prefix" }
    ]);
  });
});

interface FingerprintFixture extends HumanoidAgentStateCheckpoint {
  world: {
    frame: number;
    worldRevision: number;
    rootPosition: [number, number, number];
  };
  committed_actions: Record<string, unknown>;
}

function checkpoint(): FingerprintFixture {
  return {
    world: {
      frame: 100,
      worldRevision: 100,
      rootPosition: [2, 0.81, 2]
    },
    goal_dag: { state_sha256: "a".repeat(64) },
    goal_progress: {
      version: 1,
      goal_sha256: "c".repeat(64),
      predicate_count: 1,
      last_world_frame: 100,
      last_world_revision: 100,
      predicate_streaks: [1]
    },
    committed_actions: { "motion-1": {} },
    action_commit_outbox: structuredClone(EmptyActionCommitOutbox),
    action_execution_ledger: structuredClone(EmptyActionExecutionLedger),
    context_memory: { total_compactions: 3 },
    cycle_index: 7,
    active_cycle: { cycle_id: "cycle-1" }
  };
}
