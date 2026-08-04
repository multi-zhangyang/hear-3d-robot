import { access, appendFile, mkdtemp, readFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { RunStore } from "./run-store.js";
import { FileSession } from "./file-session.js";

describe("RunStore journal windows", () => {
  it("clears only the replaceable SDK agent state", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-store-state-"));
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 0);
    const store = await RunStore.create(runsDir, {
      mission: "Rotate one SDK branch",
      scenarioId: "humanoid_courtyard",
      scenario,
      goal: scenario.default_goal
    });
    expect(store.definition.run_mode).toBe("mission");

    await store.writeAgentState("serialized-state");
    expect(await store.readAgentState()).toBe("serialized-state");
    const fingerprint = "a".repeat(64);
    await store.writeAgentState("enveloped-state", fingerprint);
    expect(await store.readAgentStateRecord()).toEqual({
      state: "enveloped-state",
      checkpointFingerprint: fingerprint
    });
    expect(await store.readAgentState()).toBe("enveloped-state");
    await store.clearAgentState();
    expect(await store.readAgentState()).toBeUndefined();
    await store.clearAgentState();

    const persistedEvents = await store.appendRuntimeEvents([
      runtimeEvent(store.definition.run_id, "cursor-a"),
      runtimeEvent(store.definition.run_id, "cursor-b")
    ]);
    expect(persistedEvents.map((event) => event.cursor)).toEqual([
      expect.stringMatching(/^v1:0:[a-f0-9]{64}$/),
      expect.stringMatching(/^v1:1:[a-f0-9]{64}$/)
    ]);
    expect(await store.readJournal("events")).toEqual(persistedEvents);
  });

  it("isolates and clears one SDK session file per concrete worker", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-store-sessions-"));
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 0);
    const store = await RunStore.create(runsDir, {
      mission: "Persist independent worker sessions",
      scenarioId: "humanoid_courtyard",
      scenario,
      goal: scenario.default_goal
    });
    const firstPath = store.workerSessionPath("agent_first");
    const secondPath = store.workerSessionPath("agent_second");
    const first = new FileSession(firstPath, `${store.definition.run_id}:agent_first`);
    const second = new FileSession(secondPath, `${store.definition.run_id}:agent_second`);

    await Promise.all([
      first.addItems([{ role: "user", content: "first worker" }]),
      second.addItems([{ role: "user", content: "second worker" }])
    ]);
    expect(await first.getItems()).toEqual([{ role: "user", content: "first worker" }]);
    expect(await second.getItems()).toEqual([{ role: "user", content: "second worker" }]);

    await store.clearWorkerSessions();
    await expect(access(firstPath)).rejects.toMatchObject({ code: "ENOENT" });
    await expect(access(secondPath)).rejects.toMatchObject({ code: "ENOENT" });
    await store.clearWorkerSessions();
    expect(() => store.workerSessionPath("../escape")).toThrow(/Invalid hierarchy node/);
  });

  it("reads pages and tails without changing journal order", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-store-"));
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 0);
    const store = await RunStore.create(runsDir, {
      mission: "Inspect journal windows",
      scenarioId: "humanoid_courtyard",
      scenario,
      goal: scenario.default_goal
    });

    expect(await store.readJournalPage("events", 0, 2)).toEqual({
      entries: [],
      next: null,
      total: 0
    });

    await store.appendMany("events", [0, 1, 2, 3, 4]);
    expect(await store.readJournalPage("events", 1, 2)).toEqual({
      entries: [1, 2],
      next: 3,
      total: 5
    });
    expect(await store.readJournalPage("events", 4, 3)).toEqual({
      entries: [4],
      next: null,
      total: 5
    });
    expect(await store.readJournalTail("events", 2)).toEqual({
      entries: [3, 4],
      next: null,
      total: 5
    });
  });

  it("serializes concurrent appenders into complete ordered JSONL records", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-store-concurrent-"));
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 0);
    const store = await RunStore.create(runsDir, {
      mission: "Write a durable concurrent journal",
      scenarioId: "humanoid_courtyard",
      scenario,
      goal: scenario.default_goal
    });

    const appendCount = 32;
    await Promise.all(Array.from({ length: appendCount }, (_, index) =>
      store.append("events", { index, payload: `event-${index}` })
    ));
    const journal = await store.readJournal("events") as Array<{ index: number; payload: string }>;
    expect(journal).toHaveLength(appendCount);
    expect(journal.map((entry) => entry.index)).toEqual(
      Array.from({ length: appendCount }, (_, index) => index)
    );
  }, 20_000);

  it("serializes one journal across independently opened store instances", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-store-shared-journal-"));
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 0);
    const store = await RunStore.create(runsDir, {
      mission: "Share one indexed journal safely",
      scenarioId: "humanoid_courtyard",
      scenario,
      goal: scenario.default_goal
    });
    const storeCount = 8;
    const stores = await Promise.all(
      Array.from({ length: storeCount }, () => RunStore.open(store.runDir))
    );

    await Promise.all(stores.map((opened, index) =>
      opened.append("events", { index, payload: `shared-${index}` })
    ));

    const pages = await Promise.all(Array.from({ length: stores.length }, (_, from) =>
      store.readJournalPage("events", from, 1)
    ));
    expect(pages.every((page) => page.entries.length === 1 && page.total === stores.length))
      .toBe(true);
    expect(new Set(pages.map((page) =>
      (page.entries[0] as { index: number }).index
    ))).toEqual(new Set(Array.from({ length: stores.length }, (_, index) => index)));
    expect((await stat(join(store.runDir, "events.offsets"))).size).toBe(stores.length * 8);
  }, 15_000);

  it("revalidates disk state after a data-ahead partial append is repaired elsewhere", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-store-stale-index-"));
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 0);
    const store = await RunStore.create(runsDir, {
      mission: "Recover a writer after a partial journal append",
      scenarioId: "humanoid_courtyard",
      scenario,
      goal: scenario.default_goal
    });
    const dataPath = join(store.runDir, "events.jsonl");
    const indexPath = join(store.runDir, "events.offsets");
    const first = JSON.stringify({ index: 0 });
    const recovered = JSON.stringify({ index: 1, recovered: true });
    const final = JSON.stringify({ index: 2 });

    await store.append("events", { index: 0 });
    await appendFile(dataPath, `${recovered}\n`);
    const reader = await RunStore.open(store.runDir);
    expect(await reader.readJournalTail("events", 1)).toMatchObject({
      entries: [{ index: 1, recovered: true }],
      total: 2
    });

    await store.append("events", { index: 2 });
    const offsets = await readFile(indexPath);
    expect(Array.from({ length: 3 }, (_, index) =>
      Number(offsets.readBigUInt64LE(index * 8))
    )).toEqual([
      0,
      Buffer.byteLength(`${first}\n`),
      Buffer.byteLength(`${first}\n${recovered}\n`)
    ]);
    expect(await store.readJournal("events")).toEqual([
      { index: 0 },
      { index: 1, recovered: true },
      { index: 2 }
    ]);
    expect((await stat(dataPath)).size).toBe(Buffer.byteLength(`${first}\n${recovered}\n${final}\n`));
  });

  it("reopens indexed journals and repairs an unterminated crash tail before appending", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-store-recovery-"));
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 0);
    const store = await RunStore.create(runsDir, {
      mission: "Recover a durable journal index",
      scenarioId: "humanoid_courtyard",
      scenario,
      goal: scenario.default_goal
    });

    await store.appendMany("events", Array.from({ length: 2_000 }, (_, index) => ({ index })));
    const indexInfo = await stat(join(store.runDir, "events.offsets"));
    expect(indexInfo.size).toBe(2_000 * 8);

    await appendFile(join(store.runDir, "events.jsonl"), '{"interrupted":');
    const reopened = await RunStore.open(store.runDir);
    expect(await reopened.readJournalTail("events", 2)).toEqual({
      entries: [{ index: 1_998 }, { index: 1_999 }],
      next: null,
      total: 2_000
    });
    expect((await reopened.readJournal("events")).at(-1)).toEqual({ index: 1_999 });

    await reopened.append("events", { index: 2_000 });
    expect((await reopened.readJournal("events")).at(-1)).toEqual({ index: 2_000 });
    expect((await stat(join(store.runDir, "events.offsets"))).size).toBe(2_001 * 8);
  });
});

function runtimeEvent(runId: string, eventId: string) {
  return {
    event_id: eventId,
    run_id: runId,
    type: "test_event",
    at: "2026-07-30T00:00:00.000Z",
    data: { source: "test" }
  };
}
