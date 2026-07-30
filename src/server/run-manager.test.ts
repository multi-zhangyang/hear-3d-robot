import { appendFile, cp, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { loadRuntimeCatalog, type ProviderConfig } from "../config/load.js";
import type { JsonValue } from "../domain/schema.js";
import type { RuntimeEvent, RuntimeEventSink } from "../harness/runtime-context.js";
import type { MutationFence } from "../persistence/mutation-fence.js";
import { RunStore } from "../persistence/run-store.js";
import { RunManager } from "./run-manager.js";

const missionRunner = vi.hoisted(() => ({
  startMission: vi.fn(),
  resumeMission: vi.fn()
}));

vi.mock("../runtime/mission-runner.js", () => missionRunner);

const FIXTURE = resolve(
  process.cwd(),
  "tests/fixtures/runs/20000101T000000Z_fetch_red_block_00000000"
);
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

describe("RunManager event and process lifecycle", () => {
  it("takes one fenced details cut so a matching journal record cannot be skipped by its cursor", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const fence = new PausableMutationFence();
    const manager = new RunManager({ runsDir, catalog, mutationFence: fence });
    const writer = await RunStore.open(store.runDir, { mutationFence: fence });
    const baseline = await manager.details(runId, { actions: 20, provider: 20, framework: 20 });
    const pause = fence.pauseNext();
    const loading = manager.details(runId, { actions: 20, provider: 20, framework: 20 });
    await pause.entered;

    const record = {
      status: "contacted",
      at: "2026-07-30T12:00:00.000Z",
      runtime_event_id: "interleaved-provider-event"
    };
    const durable = {
      event_id: record.runtime_event_id,
      run_id: runId,
      type: "provider_event",
      at: record.at,
      data: record,
      durable: true
    } satisfies RuntimeEvent;
    let writeFinished = false;
    const writing = (async () => {
      await writer.append("provider", record);
      const [persisted] = await writer.appendRuntimeEvents([durable]);
      writeFinished = true;
      return persisted!;
    })();
    await Promise.resolve();
    expect(writeFinished).toBe(false);

    pause.release();
    const cut = await loading;
    const persistedDurable = await writing;
    expect(cut.event_cursor).toBe(baseline.event_cursor);
    expect(cut.provider).not.toContainEqual(record);

    const replayed: RuntimeEvent[] = [];
    const unsubscribe = await manager.subscribe(
      runId,
      cut.event_cursor ?? undefined,
      (event) => replayed.push(event)
    );
    unsubscribe();
    expect(replayed.filter((event) => event.event_id === durable.event_id))
      .toEqual([persistedDurable]);
  });

  it("replays one matching event when details sees its domain record first", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const manager = new RunManager({ runsDir, catalog });
    const record = {
      status: "contacted",
      at: "2026-07-30T12:00:00.000Z",
      runtime_event_id: "domain-ahead-provider-event"
    };
    const durable = {
      event_id: record.runtime_event_id,
      run_id: runId,
      type: "provider_event",
      at: record.at,
      data: record,
      durable: true
    } satisfies RuntimeEvent;

    await store.append("provider", record);
    const cut = await manager.details(runId, { actions: 20, provider: 20, framework: 20 });
    expect(cut.provider).toContainEqual(record);

    const [persistedDurable] = await store.appendRuntimeEvents([durable]);
    const replayed: RuntimeEvent[] = [];
    const unsubscribe = await manager.subscribe(
      runId,
      cut.event_cursor ?? undefined,
      (event) => replayed.push(event)
    );
    unsubscribe();
    expect(replayed.filter((event) => event.event_id === durable.event_id))
      .toEqual([persistedDurable]);
  });

  it("streams an old cursor suffix without replaying history for a current cursor", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const manager = new RunManager({ runsDir, catalog });
    const entries = await store.readJournal("events") as unknown as RuntimeEvent[];
    expect(entries.length).toBeGreaterThan(2);

    const fromCurrent: RuntimeEvent[] = [];
    const stopCurrent = await manager.subscribe(
      runId,
      entries.at(-1)!.event_id,
      (event) => fromCurrent.push(event)
    );
    stopCurrent();
    expect(fromCurrent).toEqual([]);

    const fromOlder: RuntimeEvent[] = [];
    const deliveryOrder: string[] = [];
    const cursorIndex = entries.length - 3;
    const stopOlder = await manager.subscribe(
      runId,
      entries[cursorIndex]!.event_id,
      (event) => {
        fromOlder.push(event);
        deliveryOrder.push(event.event_id);
      },
      undefined,
      () => deliveryOrder.push("ready")
    );
    stopOlder();
    expect(fromOlder.map((event) => event.event_id)).toEqual(
      entries.slice(cursorIndex + 1).map((event) => event.event_id)
    );
    expect(deliveryOrder).toEqual([
      "ready",
      ...entries.slice(cursorIndex + 1).map((event) => event.event_id)
    ]);

    const noReplay: RuntimeEvent[] = [];
    const stopLive = await manager.subscribe(runId, undefined, (event) => noReplay.push(event));
    stopLive();
    expect(noReplay).toEqual([]);
    await expect(manager.subscribe("missing_run", undefined, () => undefined))
      .rejects.toMatchObject({ code: "ENOENT" });
  });

  it("locates a versioned cursor in O(1) and replays its suffix across bounded pages", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const appended = Array.from({ length: 1_300 }, (_, index) =>
      runtimeEvent(runId, `indexed-event-${index}`));
    const persisted = await store.appendRuntimeEvents(appended);
    const manager = new RunManager({ runsDir, catalog });
    const scan = vi.spyOn(RunStore.prototype, "scanJournal")
      .mockRejectedValue(new Error("full journal scan is forbidden"));
    const pages = vi.spyOn(RunStore.prototype, "readJournalPage");
    const received: RuntimeEvent[] = [];
    try {
      const stop = await manager.subscribe(
        runId,
        persisted[100]!.cursor,
        (event) => received.push(event)
      );
      stop();
      expect(received.map((event) => event.event_id)).toEqual(
        persisted.slice(101).map((event) => event.event_id)
      );
      expect(scan).not.toHaveBeenCalled();
      expect(pages.mock.calls.length).toBeGreaterThan(1);
      expect(pages.mock.calls.every((call) => call[2] <= 64)).toBe(true);
      expect(pages.mock.calls.every((call) => call[3] === 512 * 1024)).toBe(true);
    } finally {
      pages.mockRestore();
      scan.mockRestore();
    }
  });

  it("distinguishes duplicate event IDs with versioned cursors", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const [first, second] = await store.appendRuntimeEvents([
      runtimeEvent(runId, "reused-event-id"),
      runtimeEvent(runId, "reused-event-id")
    ]);
    const manager = new RunManager({ runsDir, catalog });
    const received: RuntimeEvent[] = [];

    const stop = await manager.subscribe(
      runId,
      first!.cursor,
      (event) => received.push(event)
    );
    stop();

    expect(received).toEqual([second]);
    expect(first!.cursor).not.toBe(second!.cursor);
  });

  it("rejects duplicate legacy IDs and malformed reserved cursors before stream readiness", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    await store.appendRuntimeEvents([
      runtimeEvent(runId, "ambiguous-legacy-id"),
      runtimeEvent(runId, "ambiguous-legacy-id")
    ]);
    const manager = new RunManager({ runsDir, catalog });

    for (const cursor of [
      "ambiguous-legacy-id",
      "v1:not-an-index:broken",
      `v1:0:${"0".repeat(64)}`
    ]) {
      const ready = vi.fn();
      await expect(manager.subscribe(runId, cursor, () => undefined, undefined, ready))
        .rejects.toThrow(/(?:Ambiguous|Malformed|Unknown) .*event cursor/i);
      expect(ready).not.toHaveBeenCalled();
    }
  });

  it("rejects an unknown cursor before declaring the stream ready", async () => {
    const { runsDir, runId } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const manager = new RunManager({ runsDir, catalog });
    const ready = vi.fn();
    const received: RuntimeEvent[] = [];

    await expect(manager.subscribe(
      runId,
      "unknown-event-cursor",
      (event) => received.push(event),
      undefined,
      ready
    )).rejects.toThrow(/Unknown event cursor/);
    expect(ready).not.toHaveBeenCalled();
    expect(received).toEqual([]);
  });

  it("marks process-orphaned runs interrupted and leaves them resumable", async () => {
    const { runsDir, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const checkpoint = await store.readCheckpoint();
    expect(checkpoint.active_agent_ids).toEqual([]);
    expect(checkpoint.inflight_actions).toEqual({});
    expect(checkpoint.spatial_memory).toEqual([]);
    expect(Object.values(checkpoint.nodes).every(
      (node) => Array.isArray(node.evidence_requirements)
    )).toBe(true);
    await store.writeCheckpoint({ ...checkpoint, status: "running", error: null });

    const manager = new RunManager({ runsDir, catalog });
    await expect(manager.recoverOrphanedRuns()).resolves.toBe(1);
    await expect(manager.recoverOrphanedRuns()).resolves.toBe(0);

    const recovered = await store.readCheckpoint();
    expect(recovered.status).toBe("interrupted");
    expect(recovered.error).toMatch(/previous operator process/);
    expect((await store.readJournalTail("events", 1)).entries[0]).toMatchObject({
      type: "run_interrupted",
      data: { recovered_on_operator_start: true }
    });
  });

  it("does not hide a valid orphaned checkpoint write failure", async () => {
    const { runsDir, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const checkpoint = await store.readCheckpoint();
    await store.writeCheckpoint({ ...checkpoint, status: "running", error: null });
    const write = vi.spyOn(RunStore.prototype, "writeCheckpoint")
      .mockRejectedValueOnce(new Error("checkpoint storage unavailable"));
    try {
      const manager = new RunManager({ runsDir, catalog });
      await expect(manager.recoverOrphanedRuns())
        .rejects.toThrow("checkpoint storage unavailable");
    } finally {
      write.mockRestore();
    }
  });

  it("reconciles an orphan interruption after its first event append fails", async () => {
    const { runsDir, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const checkpoint = await store.readCheckpoint();
    await store.writeCheckpoint({ ...checkpoint, status: "running", error: null });
    const originalAppend = RunStore.prototype.appendRuntimeEvents;
    let rejected = false;
    const append = vi.spyOn(RunStore.prototype, "appendRuntimeEvents").mockImplementation(async function (
      events
    ) {
      if (!rejected && events.some((event) => event.type === "run_interrupted")) {
        rejected = true;
        throw new Error("event journal unavailable");
      }
      return originalAppend.call(this, events);
    });
    try {
      const manager = new RunManager({ runsDir, catalog });
      await expect(manager.recoverOrphanedRuns()).rejects.toThrow("event journal unavailable");

      const pending = await store.readCheckpoint();
      expect(pending.status).toBe("interrupted");
      expect(pending.pending_lifecycle_events).toHaveLength(1);

      await expect(manager.recoverOrphanedRuns()).resolves.toBe(0);
      const recovered = await store.readCheckpoint();
      expect(recovered.pending_lifecycle_events).toEqual([]);
      expect((await store.readJournal("events")).filter(
        (entry) => (entry as Record<string, unknown>).type === "run_interrupted"
      )).toHaveLength(1);
    } finally {
      append.mockRestore();
    }
  });

  it("backfills only complete crash-safe events and cancels a disconnected replay", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const manager = new RunManager({ runsDir, catalog });
    const entries = await store.readJournal("events") as unknown as RuntimeEvent[];
    await appendFile(join(store.runDir, "events.jsonl"), '{"event_id":"interrupted"');

    const replayed: RuntimeEvent[] = [];
    const stop = await manager.subscribe(
      runId,
      entries.at(-3)!.event_id,
      (event) => replayed.push(event)
    );
    stop();
    expect(replayed.map((event) => event.event_id)).toEqual(
      entries.slice(-2).map((event) => event.event_id)
    );

    const disconnected = new AbortController();
    let deliveredBeforeDisconnect = 0;
    await expect(manager.subscribe(runId, entries[0]!.event_id, () => {
      deliveredBeforeDisconnect += 1;
      disconnected.abort(new Error("client disconnected"));
    }, disconnected.signal)).rejects.toThrow("client disconnected");
    expect(deliveredBeforeDisconnect).toBe(1);
  });

  it("replays one durable event larger than the replay byte budget", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const manager = new RunManager({ runsDir, catalog });
    const baseline = await manager.details(runId, { actions: 1, provider: 1, framework: 1 });
    const [oversized] = await store.appendRuntimeEvents([{
      ...runtimeEvent(runId, "oversized-durable-event"),
      data: { payload: "x".repeat(1024 * 1024) }
    }]);
    const received: RuntimeEvent[] = [];

    const stop = await manager.subscribe(
      runId,
      baseline.event_cursor!,
      (event) => received.push(event)
    );
    stop();

    expect(received).toEqual([oversized]);
  });

  it("orders live events after a paused journal backfill without losing either source", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const entries = await store.readJournal("events") as unknown as RuntimeEvent[];
    const cursorIndex = entries.length - 3;
    let emit: RuntimeEventSink | undefined;
    let finishMission!: (value: { runId: string; runDir: string; output: string }) => void;
    missionRunner.startMission.mockImplementationOnce((input: { eventSink?: RuntimeEventSink }) => {
      emit = input.eventSink;
      return new Promise((resolveMission) => {
        finishMission = resolveMission;
      });
    });
    const manager = new RunManager({ runsDir, catalog, provider: TEST_PROVIDER });
    const starting = manager.start({
      mission: store.definition.mission,
      scenarioId: store.definition.scenario_id,
      goal: store.definition.goal
    });
    if (!emit) throw new Error("Mission event sink was not installed");
    emit(runtimeEvent(runId, "mission-created"));
    await expect(starting).resolves.toBe(runId);

    const live = runtimeEvent(runId, "live-during-backfill");
    const originalScan = RunStore.prototype.scanJournal;
    const scan = vi.spyOn(RunStore.prototype, "scanJournal").mockImplementationOnce(
      async function (name, visit) {
        await originalScan.call(this, name, visit);
        emit?.(entries[cursorIndex + 2]!);
        emit?.(live);
      }
    );
    const received: RuntimeEvent[] = [];
    try {
      const stop = await manager.subscribe(
        runId,
        entries[cursorIndex]!.event_id,
        async (event) => {
          await Promise.resolve();
          received.push(event);
        }
      );
      stop();
      expect(received.map((event) => event.event_id)).toEqual([
        ...entries.slice(cursorIndex + 1).map((event) => event.event_id),
        live.event_id
      ]);
    } finally {
      scan.mockRestore();
      finishMission({ runId, runDir: store.runDir, output: "done" });
      await manager.drain();
      missionRunner.startMission.mockReset();
    }
  });

  it("fails a replay whose bounded live-event buffer is overrun", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const entries = await store.readJournal("events") as unknown as RuntimeEvent[];
    const cursorIndex = entries.length - 3;
    let emit: RuntimeEventSink | undefined;
    let finishMission!: (value: { runId: string; runDir: string; output: string }) => void;
    missionRunner.startMission.mockImplementationOnce((input: { eventSink?: RuntimeEventSink }) => {
      emit = input.eventSink;
      return new Promise((resolveMission) => {
        finishMission = resolveMission;
      });
    });
    const manager = new RunManager({ runsDir, catalog, provider: TEST_PROVIDER });
    const starting = manager.start({
      mission: store.definition.mission,
      scenarioId: store.definition.scenario_id,
      goal: store.definition.goal
    });
    if (!emit) throw new Error("Mission event sink was not installed");
    emit(runtimeEvent(runId, "mission-created"));
    await starting;

    const originalScan = RunStore.prototype.scanJournal;
    const scan = vi.spyOn(RunStore.prototype, "scanJournal").mockImplementationOnce(
      async function (name, visit) {
        await originalScan.call(this, name, visit);
        for (let liveIndex = 0; liveIndex < 300; liveIndex += 1) {
          emit?.(runtimeEvent(runId, `live-overflow-${liveIndex}`));
        }
      }
    );
    const ready = vi.fn();
    try {
      await expect(manager.subscribe(
        runId,
        entries[cursorIndex]!.event_id,
        async () => Promise.resolve(),
        undefined,
        ready
      )).rejects.toThrow(/fell behind during journal backfill/);
      expect(ready).not.toHaveBeenCalled();
    } finally {
      scan.mockRestore();
      finishMission({ runId, runDir: store.runDir, output: "done" });
      await manager.drain();
      missionRunner.startMission.mockReset();
    }
  });

  it("allows one oversized live event while a journal cursor is being replayed", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const entries = await store.readJournal("events") as unknown as RuntimeEvent[];
    const cursorIndex = entries.length - 3;
    let emit: RuntimeEventSink | undefined;
    let finishMission!: (value: { runId: string; runDir: string; output: string }) => void;
    missionRunner.startMission.mockImplementationOnce((input: { eventSink?: RuntimeEventSink }) => {
      emit = input.eventSink;
      return new Promise((resolveMission) => {
        finishMission = resolveMission;
      });
    });
    const manager = new RunManager({ runsDir, catalog, provider: TEST_PROVIDER });
    const starting = manager.start({
      mission: store.definition.mission,
      scenarioId: store.definition.scenario_id,
      goal: store.definition.goal
    });
    if (!emit) throw new Error("Mission event sink was not installed");
    emit(runtimeEvent(runId, "mission-created"));
    await starting;

    const originalScan = RunStore.prototype.scanJournal;
    const scan = vi.spyOn(RunStore.prototype, "scanJournal").mockImplementationOnce(
      async function (name, visit) {
        await originalScan.call(this, name, visit);
        emit?.({
          ...runtimeEvent(runId, "oversized-live-frame"),
          type: "world_frames",
          durable: false,
          data: { frames: [], payload: "x".repeat(1024 * 1024) }
        });
      }
    );
    const received: RuntimeEvent[] = [];
    try {
      const stop = await manager.subscribe(
        runId,
        entries[cursorIndex]!.event_id,
        (event) => received.push(event)
      );
      stop();
      expect(received.at(-1)?.event_id).toBe("oversized-live-frame");
    } finally {
      scan.mockRestore();
      finishMission({ runId, runDir: store.runDir, output: "done" });
      await manager.drain();
      missionRunner.startMission.mockReset();
    }
  });

  it("disconnects when another live event follows an oversized backfill buffer entry", async () => {
    const { runsDir, runId, store } = await copiedFixture();
    const catalog = await loadRuntimeCatalog();
    const entries = await store.readJournal("events") as unknown as RuntimeEvent[];
    const cursorIndex = entries.length - 3;
    let emit: RuntimeEventSink | undefined;
    let finishMission!: (value: { runId: string; runDir: string; output: string }) => void;
    missionRunner.startMission.mockImplementationOnce((input: { eventSink?: RuntimeEventSink }) => {
      emit = input.eventSink;
      return new Promise((resolveMission) => {
        finishMission = resolveMission;
      });
    });
    const manager = new RunManager({ runsDir, catalog, provider: TEST_PROVIDER });
    const starting = manager.start({
      mission: store.definition.mission,
      scenarioId: store.definition.scenario_id,
      goal: store.definition.goal
    });
    if (!emit) throw new Error("Mission event sink was not installed");
    emit(runtimeEvent(runId, "mission-created"));
    await starting;

    const originalScan = RunStore.prototype.scanJournal;
    const scan = vi.spyOn(RunStore.prototype, "scanJournal").mockImplementationOnce(
      async function (name, visit) {
        await originalScan.call(this, name, visit);
        emit?.({
          ...runtimeEvent(runId, "oversized-live-frame"),
          type: "world_frames",
          durable: false,
          data: { frames: [], payload: "x".repeat(1024 * 1024) }
        });
        emit?.(runtimeEvent(runId, "event-after-oversized-frame"));
      }
    );
    const ready = vi.fn();
    try {
      await expect(manager.subscribe(
        runId,
        entries[cursorIndex]!.event_id,
        () => undefined,
        undefined,
        ready
      )).rejects.toThrow(/fell behind during journal backfill/);
      expect(ready).not.toHaveBeenCalled();
    } finally {
      scan.mockRestore();
      finishMission({ runId, runDir: store.runDir, output: "done" });
      await manager.drain();
      missionRunner.startMission.mockReset();
    }
  });
});

async function copiedFixture(): Promise<{
  runsDir: string;
  runId: string;
  store: RunStore;
}> {
  const runsDir = await mkdtemp(join(tmpdir(), "hear-manager-"));
  const runId = basename(FIXTURE);
  const destination = join(runsDir, runId);
  await cp(FIXTURE, destination, { recursive: true });
  return { runsDir, runId, store: await RunStore.open(destination) };
}

function runtimeEvent(runId: string, eventId: string): RuntimeEvent {
  return {
    event_id: eventId,
    run_id: runId,
    type: "test_event",
    at: "2026-07-30T00:00:00.000Z",
    data: { source: "test" }
  };
}

class PausableMutationFence implements MutationFence {
  #tail: Promise<void> = Promise.resolve();
  #nextPause: {
    enter: () => void;
    release: Promise<void>;
  } | undefined;

  pauseNext(): { entered: Promise<void>; release: () => void } {
    if (this.#nextPause) throw new Error("A mutation fence pause is already armed");
    const entered = deferredSignal();
    const release = deferredSignal();
    this.#nextPause = { enter: entered.resolve, release: release.promise };
    return { entered: entered.promise, release: release.resolve };
  }

  runMutation<T>(operation: () => Promise<T>): Promise<T> {
    const pause = this.#nextPause;
    this.#nextPause = undefined;
    const execute = async (): Promise<T> => {
      if (pause) {
        pause.enter();
        await pause.release;
      }
      return operation();
    };
    const result = this.#tail.then(execute, execute);
    this.#tail = result.then(() => undefined, () => undefined);
    return result;
  }
}

function deferredSignal(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}
