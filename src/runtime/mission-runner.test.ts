import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ProviderConfig } from "../config/load.js";
import { loadRuntimeCatalog } from "../config/load.js";
import type { Goal } from "../domain/schema.js";
import {
  HarnessRuntimeContext,
  type RuntimeEventSink
} from "../harness/runtime-context.js";
import { FileSession } from "../persistence/file-session.js";
import { listRunDirectories, RunStore } from "../persistence/run-store.js";
import { startMission } from "./mission-runner.js";

const fakeRunner = vi.hoisted(() => ({
  finalOutput: "Mission completed from the current world state",
  behavior: "success" as
    | "success"
    | "opening_transport_failure"
    | "context_compaction_failure",
  runCalls: 0,
  sessionSizesBeforeRequest: [] as number[]
}));

vi.mock("@openai/agents", async (importOriginal) => {
  const original = await importOriginal<typeof import("@openai/agents")>();

  class ConfigurableRunner {
    async run(
      _agent: unknown,
      _input: unknown,
      options?: {
        session?: {
          getItems(): Promise<unknown[]>;
          addItems(items: Array<{ role: string; content: string }>): Promise<void>;
        };
      }
    ): Promise<unknown> {
      fakeRunner.runCalls += 1;
      if (fakeRunner.behavior === "opening_transport_failure") {
        if (!options?.session) throw new Error("Test Runner requires a Session");
        fakeRunner.sessionSizesBeforeRequest.push((await options.session.getItems()).length);
        await options.session.addItems([{
          role: "user",
          content: "SDK-persisted opening mission input"
        }]);
        throw Object.assign(new Error("opening model transport unavailable"), {
          status: 503
        });
      }
      if (fakeRunner.behavior === "context_compaction_failure") {
        throw Object.assign(
          new Error("Context compaction interruption: checkpoint tool omitted"),
          {
            name: "ContextCompactionInterruption",
            code: "context_compaction_interrupted"
          }
        );
      }
      return {
        completed: Promise.resolve(),
        finalOutput: fakeRunner.finalOutput,
        state: { toString: () => "{}" },
        async *[Symbol.asyncIterator]() {
          // The regression concerns terminal persistence after a completed SDK
          // run, so no intermediate stream event is needed here.
        }
      };
    }
  }

  return { ...original, Runner: ConfigurableRunner };
});

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

const originalSucceed = HarnessRuntimeContext.prototype.succeed;

afterEach(() => {
  fakeRunner.behavior = "success";
  fakeRunner.runCalls = 0;
  fakeRunner.sessionSizesBeforeRequest = [];
  vi.restoreAllMocks();
});

describe("mission terminal lifecycle persistence", () => {
  it("does not overwrite durable success when the lifecycle sink rejects", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-mission-success-sink-"));
    try {
      arrangeSuccessfulChecker();
      const sink: RuntimeEventSink = (event) => {
        if (event.type === "run_succeeded") throw new Error("success sink unavailable");
      };

      await expect(runMission(runsDir, sink)).rejects.toThrow("success sink unavailable");
      await expectPersistedSuccessOnly(runsDir);
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("does not overwrite durable success when clearing its outbox rejects", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-mission-success-clear-"));
    const originalWrite = RunStore.prototype.writeCheckpoint;
    vi.spyOn(RunStore.prototype, "writeCheckpoint").mockImplementation(async function (
      this: RunStore,
      checkpoint
    ) {
      if (checkpoint.status === "succeeded"
        && checkpoint.pending_lifecycle_events.length === 0) {
        throw new Error("success outbox clear unavailable");
      }
      await originalWrite.call(this, checkpoint);
    });
    try {
      arrangeSuccessfulChecker();

      await expect(runMission(runsDir)).rejects.toThrow("success outbox clear unavailable");
      await expectPersistedSuccessOnly(runsDir);
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });
});

describe("initial model transport recovery", () => {
  it("restores the opening Session baseline after bounded retries are exhausted", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-mission-opening-transport-"));
    fakeRunner.behavior = "opening_transport_failure";
    vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: () => void) => {
      queueMicrotask(callback);
      return {} as NodeJS.Timeout;
    }) as typeof setTimeout);

    try {
      await expect(runMission(runsDir)).rejects.toThrow("opening model transport unavailable");

      expect(fakeRunner.runCalls).toBe(9);
      expect(fakeRunner.sessionSizesBeforeRequest).toEqual(Array<number>(9).fill(0));

      const runDirectories = await listRunDirectories(runsDir);
      expect(runDirectories).toHaveLength(1);
      const store = await RunStore.open(runDirectories[0]!);
      const checkpoint = await store.readCheckpoint();
      const session = new FileSession(store.sessionPath(), checkpoint.run_id);

      expect(await store.readAgentState()).toBeUndefined();
      expect(await session.getItems()).toEqual([]);
      expect(checkpoint).toMatchObject({
        status: "interrupted",
        error: expect.stringContaining("opening model transport unavailable")
      });
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });
});

describe("context compaction recovery", () => {
  it("persists an unsafe compaction outcome as interrupted instead of failed", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-mission-context-interruption-"));
    fakeRunner.behavior = "context_compaction_failure";
    try {
      await expect(runMission(runsDir)).rejects.toMatchObject({
        code: "context_compaction_interrupted"
      });

      const runDirectories = await listRunDirectories(runsDir);
      expect(runDirectories).toHaveLength(1);
      const store = await RunStore.open(runDirectories[0]!);
      const checkpoint = await store.readCheckpoint();
      const events = await store.readJournal("events");

      expect(checkpoint).toMatchObject({
        status: "interrupted",
        error: expect.stringContaining("Context compaction interruption"),
        final_output: null
      });
      expect(events.some((event) => eventRecord(event)?.type === "run_interrupted")).toBe(true);
      expect(events.some((event) => eventRecord(event)?.type === "run_failed")).toBe(false);
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });
});

function arrangeSuccessfulChecker(): void {
  vi.spyOn(HarnessRuntimeContext.prototype, "succeed").mockImplementation(async function (
    finalOutput
  ) {
    await this.invokeChecker({}, "test_terminal_checker");
    await originalSucceed.call(this, finalOutput);
  });
}

async function runMission(runsDir: string, eventSink?: RuntimeEventSink): Promise<unknown> {
  const catalog = await loadRuntimeCatalog();
  const goal: Goal = {
    summary: "Robot remains at its initial coordinate.",
    predicates: [
      { type: "robot_at", target: { x: 1, y: 0, z: 1 }, tolerance: 0.01 }
    ]
  };
  return startMission({
    runsDir,
    mission: "Keep the robot at its current coordinate",
    scenarioId: "open_navigation",
    goal,
    catalog,
    provider: TEST_PROVIDER,
    seed: 0,
    ...(eventSink ? { eventSink } : {})
  });
}

async function expectPersistedSuccessOnly(runsDir: string): Promise<void> {
  const runDirectories = await listRunDirectories(runsDir);
  expect(runDirectories).toHaveLength(1);
  const store = await RunStore.open(runDirectories[0]!);
  const checkpoint = await store.readCheckpoint();
  const events = await store.readJournal("events");

  expect(checkpoint).toMatchObject({
    status: "succeeded",
    final_output: fakeRunner.finalOutput,
    error: null
  });
  expect(events.filter((event) => eventRecord(event)?.type === "run_succeeded")).toHaveLength(1);
  expect(events.some((event) => eventRecord(event)?.type === "run_failed")).toBe(false);
}

function eventRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}
