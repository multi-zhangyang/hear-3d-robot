import { MemorySession } from "@openai/agents";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi
} from "vitest";
import {
  loadProviderConfig,
  type ProviderConfig,
  type RuntimeCatalog
} from "../config/load.js";
import { ScenarioSchema } from "../domain/schema.js";
import {
  createHumanoidAgentManifest
} from "../harness/agent-manifest.js";
import { createHumanoidAgentHierarchy } from "../harness/humanoid/agents.js";
import { createHumanoidRunCheckpoint } from "../harness/humanoid/run-checkpoint.js";
import { createConfiguredModel } from "../model/factory.js";
import { FileSession } from "../persistence/file-session.js";
import { RunStore } from "../persistence/run-store.js";
import { HumanoidWorld } from "../world/humanoid/world.js";
import { resumeHumanoidMission } from "./humanoid-mission-runner.js";
import { RunPauseRequestedError } from "./run-pause.js";

const runnerControl = vi.hoisted(() => ({
  constructorError: undefined as Error | undefined,
  run: vi.fn()
}));

vi.mock("@openai/agents", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@openai/agents")>();
  return {
    ...actual,
    Runner: class {
      constructor() {
        if (runnerControl.constructorError) throw runnerControl.constructorError;
      }

      run(...args: unknown[]) {
        return runnerControl.run(...args);
      }
    }
  };
});

vi.mock("./transport-recovery.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./transport-recovery.js")>();
  return {
    ...actual,
    transportRetryPlan(error: unknown, attempt: number) {
      return { ...actual.transportRetryPlan(error, attempt), waitMs: 0 };
    }
  };
});

const scenario = ScenarioSchema.parse({
  title: "人形任务恢复场",
  seed: 31,
  bounds: { width: 10, depth: 10 },
  visibility_radius: 6,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [],
  zones: [],
  default_goal: {
    summary: "保持当前位置与站立状态",
    predicates: [{
      type: "robot_at",
      target: { x: 2, y: 0, z: 2 },
      tolerance: 0.4
    }]
  }
});
const catalog: RuntimeCatalog = {
  templates: {},
  materialize: () => structuredClone(scenario)
};
const temporaryDirectories: string[] = [];

beforeEach(() => {
  runnerControl.constructorError = undefined;
  runnerControl.run.mockReset().mockRejectedValue(
    new Error("Runner should not reach the model in this test")
  );
});

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => (
    rm(directory, { recursive: true, force: true })
  )));
});

describe("humanoid mission initialization recovery", () => {
  it("persists an intentional operator pause separately from a crash interruption", async () => {
    const store = await createCheckpointedRun();
    const controller = new AbortController();
    controller.abort(new RunPauseRequestedError("operator requested a safe pause"));

    await expect(resume(store, provider(), controller.signal)).rejects.toThrow(
      "operator requested a safe pause"
    );

    const checkpoint = await store.readHumanoidCheckpoint();
    expect(checkpoint).toMatchObject({
      status: "paused",
      error: null,
      active_agent_id: null,
      active_agent_ids: []
    });
    expect(await lifecycleTypes(store)).toEqual(["run_resumed", "run_paused"]);
  }, 60_000);

  it("creates an epoch only for the checkpoint-before-manifest crash window", async () => {
    const store = await createCheckpointedRun();
    const controller = new AbortController();
    controller.abort(new Error("operator paused before the model turn"));

    await expect(resume(store, provider(), controller.signal)).rejects.toThrow(
      "operator paused before the model turn"
    );

    const manifest = await store.readAgentManifest();
    expect(manifest.epoch_id).toMatch(/^[0-9a-f-]{36}$/);
    const checkpoint = await store.readHumanoidCheckpoint();
    expect(checkpoint).toMatchObject({
      status: "interrupted",
      error: expect.stringContaining("operator paused before the model turn"),
      active_agent_id: null,
      active_agent_ids: [],
      pending_lifecycle_events: []
    });
    expect(await lifecycleTypes(store)).toEqual(["run_resumed", "run_interrupted"]);
    expect(runnerControl.run).not.toHaveBeenCalled();
  }, 60_000);

  it.each(runtimeStateCases())(
    "refuses to invent an epoch when a missing manifest has %s",
    async (_name, persistState) => {
      const store = await createCheckpointedRun();
      await persistState(store);

      await expect(resume(store, provider())).rejects.toThrow(
        "Agent manifest is missing while Agent state or Session state exists"
      );

      const checkpoint = await store.readHumanoidCheckpoint();
      expect(checkpoint.status).toBe("interrupted");
      expect(checkpoint.error).toContain(
        "refusing to create a replacement epoch"
      );
      await expect(store.readAgentManifest()).rejects.toThrow("Agent manifest is missing");
      expect(await lifecycleTypes(store)).toEqual(["run_interrupted"]);
      expect(runnerControl.run).not.toHaveBeenCalled();
    }
  );

  it("keeps a malformed manifest recoverable and preserves its evidence", async () => {
    const store = await createCheckpointedRun();
    await writeFile(
      join(store.runDir, "agent-manifest.json"),
      "{\"version\":2,\"corrupt\":true}\n",
      "utf8"
    );

    await expect(resume(store, provider())).rejects.toThrow(
      "Unable to read the persisted agent manifest"
    );

    const checkpoint = await store.readHumanoidCheckpoint();
    expect(checkpoint.status).toBe("interrupted");
    expect(checkpoint.error).toContain("Unable to read the persisted agent manifest");
    expect(await lifecycleTypes(store)).toEqual(["run_interrupted"]);
    expect(runnerControl.run).not.toHaveBeenCalled();
  });

  it("interrupts an incompatible agent identity without starting the run", async () => {
    const store = await createCheckpointedRun();
    const firstProvider = provider("first-model");
    const manifest = createManifest(firstProvider);
    await store.writeAgentManifest(manifest);

    await expect(resume(store, provider("second-model"))).rejects.toMatchObject({
      name: "AgentManifestIncompatibleError",
      code: "agent_manifest_incompatible"
    });

    const checkpoint = await store.readHumanoidCheckpoint();
    expect(checkpoint.status).toBe("interrupted");
    expect(checkpoint.error).toContain("Agent manifest is incompatible");
    expect(await store.readAgentManifest()).toEqual(manifest);
    expect(await lifecycleTypes(store)).toEqual(["run_interrupted"]);
    expect(runnerControl.run).not.toHaveBeenCalled();
  });

  it("persists an initialization failure instead of leaving a starting checkpoint", async () => {
    const store = await createCheckpointedRun();
    runnerControl.constructorError = new Error("Runner initialization unavailable");

    await expect(resume(store, provider())).rejects.toThrow(
      "Runner initialization unavailable"
    );

    const checkpoint = await store.readHumanoidCheckpoint();
    expect(checkpoint.status).toBe("interrupted");
    expect(checkpoint.error).toContain("Runner initialization unavailable");
    expect(await lifecycleTypes(store)).toEqual(["run_interrupted"]);
    await expect(store.readAgentManifest()).rejects.toThrow("Agent manifest is missing");
  });

  it("interrupts after the bounded transport recovery window is exhausted", async () => {
    const store = await createCheckpointedRun();
    const transportError = Object.assign(new Error("connection reset"), {
      code: "ECONNRESET"
    });
    runnerControl.run.mockRejectedValue(transportError);

    await expect(resume(store, provider())).rejects.toBe(transportError);

    const checkpoint = await store.readHumanoidCheckpoint();
    expect(checkpoint.status).toBe("interrupted");
    expect(checkpoint.error).toContain("connection reset");
    expect(runnerControl.run).toHaveBeenCalledTimes(9);
    const providerEvents = await store.readJournal("provider");
    expect(providerEvents.filter((entry) => (
      isRecord(entry) && entry.status === "transport_interrupted"
    ))).toHaveLength(8);
    expect(await lifecycleTypes(store)).toEqual(["run_resumed", "run_interrupted"]);
  }, 30_000);
});

function runtimeStateCases(): Array<[
  string,
  (store: RunStore) => Promise<void>
]> {
  return [
    ["serialized Agent state", async (store) => {
      await store.writeAgentState("persisted-sdk-state");
    }],
    ["a coordinator Session", async (store) => {
      const session = new FileSession(store.sessionPath(), "orphaned-session");
      await session.addItems([{ role: "user", content: "persisted coordinator context" }]);
    }],
    ["a worker Session", async (store) => {
      const session = new FileSession(
        store.workerSessionPath("humanoid-sentry"),
        "orphaned-worker-session"
      );
      await session.addItems([{ role: "user", content: "persisted worker context" }]);
    }]
  ];
}

async function createCheckpointedRun(): Promise<RunStore> {
  const runsDir = await mkdtemp(join(tmpdir(), "hear-mission-init-"));
  temporaryDirectories.push(runsDir);
  const store = await RunStore.create(runsDir, {
    mission: "安全恢复人形自主任务",
    scenarioId: "humanoid-recovery-test",
    scenario,
    goal: scenario.default_goal,
    runtime: "humanoid_g1"
  });
  const world = await HumanoidWorld.create(scenario);
  try {
    await store.writeCheckpoint(createHumanoidRunCheckpoint({
      store,
      goal: scenario.default_goal,
      world
    }));
  } finally {
    await world.dispose();
  }
  return store;
}

function provider(model = "test-model"): ProviderConfig {
  return loadProviderConfig({
    AI_PROVIDER: "openai_compatible",
    AI_BASE_URL: "https://models.example.test/v1",
    AI_MODEL: model,
    AI_API_KEY: "test-credential"
  });
}

function createManifest(config: ProviderConfig) {
  return createHumanoidAgentManifest({
    hierarchy: createHumanoidAgentHierarchy({
      provider: config,
      runtime: {
        invoke: async () => { throw new Error("outside manifest construction"); },
        recallEmbodiedHistory: async () => { throw new Error("outside manifest construction"); },
        validateCycleEvidence: () => { throw new Error("outside manifest construction"); }
      } as never,
      createModel: (_agentId, agentProvider) => createConfiguredModel(agentProvider),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    }),
    provider: config,
    epochId: "11111111-1111-4111-8111-111111111111"
  });
}

async function resume(
  store: RunStore,
  config: ProviderConfig,
  signal?: AbortSignal
) {
  return resumeHumanoidMission({
    runDir: store.runDir,
    catalog,
    provider: config,
    ...(signal ? { signal } : {})
  });
}

async function lifecycleTypes(store: RunStore): Promise<unknown[]> {
  return (await store.readJournal("events"))
    .filter((entry) => (
      isRecord(entry) && typeof entry.type === "string" && entry.type.startsWith("run_")
    ))
    .map((entry) => isRecord(entry) ? entry.type : undefined);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
