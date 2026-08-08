import {
  MemorySession,
  UserError,
  type RunStreamEvent
} from "@openai/agents";
import { mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
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
import {
  createHumanoidAgentHierarchy,
  HUMANOID_AGENT_IDS
} from "../harness/humanoid/agents.js";
import { createHumanoidRunCheckpoint } from "../harness/humanoid/run-checkpoint.js";
import { ModelDecisionStallError } from "../harness/model-telemetry.js";
import { createConfiguredModel } from "../model/factory.js";
import { FileSession } from "../persistence/file-session.js";
import { RunStore } from "../persistence/run-store.js";
import { HumanoidWorld } from "../world/humanoid/world.js";
import { captureHumanoidSessionStateIdentity } from "./humanoid-agent-state.js";
import {
  assertHumanoidControllerSourceCompatible,
  humanoidModelProgressSnapshot,
  nextModelDecisionFollowUpState,
  recoverableDynamicToolRunStateError,
  resumeHumanoidMission,
  startHumanoidMission,
  shouldPersistHumanoidAgentState
} from "./humanoid-mission-runner.js";
import type {
  HumanoidControllerSource
} from "../world/humanoid/controller-module.js";
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
  it("requires the exact controller module source when resuming", () => {
    const first = controllerSource("a".repeat(64));
    const second = controllerSource("b".repeat(64));

    expect(() => assertHumanoidControllerSourceCompatible(undefined, undefined))
      .not.toThrow();
    expect(() => assertHumanoidControllerSourceCompatible(first.sourceSha256, first))
      .not.toThrow();
    expect(() => assertHumanoidControllerSourceCompatible(undefined, first))
      .toThrow(/built-in humanoid controller/);
    expect(() => assertHumanoidControllerSourceCompatible(first.sourceSha256, undefined))
      .toThrow(/requires its original external/);
    expect(() => assertHumanoidControllerSourceCompatible(first.sourceSha256, second))
      .toThrow(/does not match/);
  });

  it("persists and passes an external controller source into a new physical world", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-controller-start-"));
    temporaryDirectories.push(runsDir);
    const sourceSha256 = "e".repeat(64);
    const source: HumanoidControllerSource = {
      sourceSha256,
      controllerFactory: async () => {
        throw new Error("external-controller-factory-reached");
      }
    };

    await expect(startHumanoidMission({
      runsDir,
      mission: "Start with a trained whole-body policy",
      scenarioId: "humanoid-controller-test",
      goal: scenario.default_goal,
      catalog,
      provider: provider(),
      controllerSource: source
    })).rejects.toThrow("external-controller-factory-reached");
    const [runId] = (await readdir(runsDir)).filter((entry) => !entry.startsWith("."));
    if (!runId) throw new Error("Controller-backed run definition was not created");
    expect((await RunStore.open(join(runsDir, runId))).definition)
      .toMatchObject({ controller_source_sha256: sourceSha256 });
  }, 60_000);

  it("scopes no-progress receipts to the Agent that produced them", () => {
    const runtime = {
      rootAgentId: HUMANOID_AGENT_IDS.coordinator,
      checkpoint: {
        world: { worldRevision: 42 },
        cycle_index: 3,
        checker: { success: false },
        goal_dag: { state_sha256: "goal-state" },
        committed_actions: {
          motion: progressReceipt("motion", HUMANOID_AGENT_IDS.motion),
          sentry: progressReceipt("sentry", HUMANOID_AGENT_IDS.sentry)
        }
      }
    } as unknown as Parameters<typeof humanoidModelProgressSnapshot>[0];

    expect(humanoidModelProgressSnapshot(
      runtime,
      HUMANOID_AGENT_IDS.motion
    ).receipts.map((receipt) => receipt.transactionId)).toEqual(["motion"]);
    expect(humanoidModelProgressSnapshot(
      runtime,
      HUMANOID_AGENT_IDS.goalManager
    ).receipts).toEqual([]);
    expect(humanoidModelProgressSnapshot(
      runtime,
      HUMANOID_AGENT_IDS.coordinator
    ).receipts.map((receipt) => receipt.transactionId)).toEqual([
      "motion",
      "sentry"
    ]);
  });

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
  }, 60_000);

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
    const config = provider();
    const manifest = createManifest(config);
    await store.writeAgentManifest(manifest);
    const sessionItems = new Map(Object.values(HUMANOID_AGENT_IDS).map((agentId) => (
      [agentId, [{ role: "user" as const, content: `baseline:${agentId}` }]]
    )));
    await Promise.all([...sessionItems.entries()].map(async ([agentId, items]) => {
      await missionSession(store, manifest.epoch_id, agentId).addItems(items);
    }));
    const transportError = Object.assign(new Error("connection reset"), {
      code: "ECONNRESET"
    });
    runnerControl.run.mockImplementation(async (_agent, runInput) => {
      const inputText = typeof runInput === "string" ? runInput : "serialized RunState";
      await Promise.all(Object.values(HUMANOID_AGENT_IDS).map(async (agentId) => {
        await missionSession(store, manifest.epoch_id, agentId).addItems([{
          role: "user",
          content: `failed:${runnerControl.run.mock.calls.length}:${inputText}`
        }]);
      }));
      throw transportError;
    });

    await expect(resume(store, config)).rejects.toBe(transportError);

    const checkpoint = await store.readHumanoidCheckpoint();
    expect(checkpoint.status).toBe("interrupted");
    expect(checkpoint.error).toContain("connection reset");
    expect(runnerControl.run).toHaveBeenCalledTimes(9);
    const providerEvents = await store.readJournal("provider");
    expect(providerEvents.filter((entry) => (
      isRecord(entry) && entry.status === "transport_interrupted"
    ))).toHaveLength(8);
    for (const [agentId, items] of sessionItems) {
      expect(await missionSession(store, manifest.epoch_id, agentId).getItems()).toEqual(items);
    }
    expect(await lifecycleTypes(store)).toEqual(["run_resumed", "run_interrupted"]);
  }, 30_000);

  it("rebases only the failing SDK branch after repeated identical server errors", async () => {
    const store = await createCheckpointedRun();
    const config = provider();
    const manifest = createManifest(config);
    await store.writeAgentManifest(manifest);
    await Promise.all(Object.values(HUMANOID_AGENT_IDS).map(async (agentId) => {
      await missionSession(store, manifest.epoch_id, agentId).addItems([{
        role: "user",
        content: `baseline:${agentId}`
      }]);
    }));
    const serverError = Object.assign(new Error("provider failed request"), {
      name: "ModelTransportError",
      code: 500
    });
    const inspected = new Error("rebased branch inspected");
    runnerControl.run.mockImplementation(async () => {
      const attempt = runnerControl.run.mock.calls.length;
      if (attempt <= 3) throw serverError;
      expect(await missionSession(
        store,
        manifest.epoch_id,
        HUMANOID_AGENT_IDS.coordinator
      ).getItems()).toEqual([{
        role: "user",
        content: `baseline:${HUMANOID_AGENT_IDS.coordinator}`
      }]);
      for (const agentId of Object.values(HUMANOID_AGENT_IDS)) {
        if (agentId === HUMANOID_AGENT_IDS.coordinator) continue;
        expect(await missionSession(store, manifest.epoch_id, agentId).getItems()).toEqual([{
          role: "user",
          content: `baseline:${agentId}`
        }]);
      }
      throw inspected;
    });

    await expect(resume(store, config)).rejects.toBe(inspected);

    expect(runnerControl.run).toHaveBeenCalledTimes(4);
    const recoveries = (await store.readJournal("provider")).filter((entry) => (
      isRecord(entry) && entry.status === "transport_interrupted"
    ));
    expect(recoveries).toHaveLength(3);
    expect(recoveries.at(-1)).toMatchObject({
      recovery_attempt: 3,
      sdk_branch_rebased: true,
      raw_history_preserved: true,
      session_history_preserved: true,
      prompt_cache_prefix_preserved: true
    });
  });

  it("keeps Session histories intact when the checkpoint rejects a RunState", async () => {
    const store = await createCheckpointedRun();
    const config = provider();
    const manifest = createManifest(config);
    await store.writeAgentManifest(manifest);
    const sessions = new Map(Object.values(HUMANOID_AGENT_IDS).map((agentId) => (
      [agentId, missionSession(store, manifest.epoch_id, agentId)]
    )));
    await Promise.all([...sessions.entries()].map(async ([agentId, session]) => {
      await session.addItems([{ role: "user", content: `baseline:${agentId}` }]);
    }));
    const sessionBaseline = await captureHumanoidSessionStateIdentity(sessions);
    await Promise.all([...sessions.entries()].map(async ([agentId, session]) => {
      await session.addItems([{ role: "assistant", content: `durable-suffix:${agentId}` }]);
    }));
    await store.writeAgentState(
      "checkpoint-incompatible-state",
      "a".repeat(64),
      sessionBaseline
    );
    const inspectionComplete = new Error("Session inspection complete");
    runnerControl.run.mockImplementationOnce(async () => {
      for (const [agentId, session] of sessions) {
        expect(await session.getItems()).toEqual([
          { role: "user", content: `baseline:${agentId}` },
          { role: "assistant", content: `durable-suffix:${agentId}` }
        ]);
      }
      throw inspectionComplete;
    });

    await expect(resume(store, config)).rejects.toBe(inspectionComplete);

    expect(await store.readAgentStateRecord()).toBeUndefined();
    for (const [agentId, session] of sessions) {
      expect(await session.getItems()).toEqual([
        { role: "user", content: `baseline:${agentId}` },
        { role: "assistant", content: `durable-suffix:${agentId}` }
      ]);
    }
  });

  it("continues prose-only decisions in the same Agent Session", async () => {
    const store = await createCheckpointedRun();
    const config = provider();
    const manifest = createManifest(config);
    await store.writeAgentManifest(manifest);
    const coordinatorSession = missionSession(
      store,
      manifest.epoch_id,
      HUMANOID_AGENT_IDS.coordinator
    );
    await coordinatorSession.addItems([{ role: "user", content: "durable-prefix" }]);
    const inputs: unknown[] = [];
    let activeSession: FileSession | undefined;
    runnerControl.run.mockImplementation(async (
      agent: { name: string },
      runInput,
      options: { session?: FileSession; maxTurns?: number | null }
    ) => {
      expect(agent.name).toBe("人形自主协调智能体");
      expect(options.maxTurns).toBeNull();
      inputs.push(runInput);
      if (activeSession) expect(options.session).toBe(activeSession);
      else activeSession = options.session;
      const attempt = runnerControl.run.mock.calls.length;
      await options.session?.addItems([{
        role: "assistant",
        content: `prose-only-response:${attempt}`
      }]);
      return {
        state: { toString: () => `completed-prose-state:${attempt}` },
        completed: Promise.resolve(),
        finalOutput: "ordinary text without a business tool result",
        async *[Symbol.asyncIterator]() {}
      };
    });

    await expect(resume(store, config)).rejects.toThrow(
      "Humanoid coordinator did not return a formal tool result"
    );

    expect(runnerControl.run).toHaveBeenCalledTimes(4);
    expect(String(inputs[0])).toContain("继续下一次人形自主闭环");
    for (const input of inputs.slice(1)) {
      expect(String(input)).toContain("上一次模型分支没有产生 Harness 可验收的正式工具决策");
    }
    const followUps = (await store.readJournal("provider")).filter((entry) => (
      isRecord(entry) && entry.status === "model_decision_follow_up"
    ));
    expect(followUps).toEqual([1, 2, 3].map((attempt) => expect.objectContaining({
      agent_id: HUMANOID_AGENT_IDS.coordinator,
      follow_up_attempt: attempt,
      invalid_response_retained: true,
      session_history_preserved: true,
      continuation: "same_agent_model_and_session",
      automatic_actuation: false
    })));
    expect((await missionSession(
      store,
      manifest.epoch_id,
      HUMANOID_AGENT_IDS.coordinator
    ).getItems()).map((item) => item.content)).toEqual([
      "durable-prefix",
      "prose-only-response:1",
      "prose-only-response:2",
      "prose-only-response:3",
      "prose-only-response:4"
    ]);
  }, 30_000);

  it("routes a specialist stall back through the original coordinator without a repair Agent", async () => {
    const store = await createCheckpointedRun();
    const config = provider();
    const manifest = createManifest(config);
    await store.writeAgentManifest(manifest);
    const inspectionComplete = new Error("same-session follow-up inspected");
    let firstAgent: unknown;
    let firstSession: FileSession | undefined;
    runnerControl.run.mockImplementation(async (
      agent,
      runInput,
      options: { session?: FileSession; maxTurns?: number | null }
    ) => {
      const attempt = runnerControl.run.mock.calls.length;
      if (attempt === 1) {
        firstAgent = agent;
        firstSession = options.session;
        return {
          state: { toString: () => "specialist-stall-state" },
          completed: Promise.resolve(),
          finalOutput: undefined,
          async *[Symbol.asyncIterator]() {
            throw new ModelDecisionStallError(
              HUMANOID_AGENT_IDS.motion,
              "motion returned no terminal tool receipt"
            );
          }
        };
      }
      expect(agent).toBe(firstAgent);
      expect(options.session).toBe(firstSession);
      expect(options.maxTurns).toBeNull();
      expect(String(runInput)).toContain("上一次模型分支没有产生 Harness 可验收的正式工具决策");
      throw inspectionComplete;
    });

    await expect(resume(store, config)).rejects.toBe(inspectionComplete);
    expect(runnerControl.run).toHaveBeenCalledTimes(2);
    const followUps = (await store.readJournal("provider")).filter((entry) => (
      isRecord(entry) && entry.status === "model_decision_follow_up"
    ));
    expect(followUps).toEqual([
      expect.objectContaining({
        agent_id: HUMANOID_AGENT_IDS.motion,
        follow_up_attempt: 1,
        continuation: "same_agent_model_and_session"
      })
    ]);
  }, 30_000);

  it("keeps decision follow-up budgets independent for each hierarchy node", async () => {
    const store = await createCheckpointedRun();
    const config = provider();
    const manifest = createManifest(config);
    await store.writeAgentManifest(manifest);
    const inputs: unknown[] = [];
    const stalls = [
      HUMANOID_AGENT_IDS.coordinator,
      HUMANOID_AGENT_IDS.motion,
      HUMANOID_AGENT_IDS.coordinator
    ];
    const inspectionComplete = new Error("per-agent follow-up budgets inspected");
    runnerControl.run.mockImplementation(async (_agent, runInput) => {
      inputs.push(runInput);
      const stalledAgentId = stalls[runnerControl.run.mock.calls.length - 1];
      if (!stalledAgentId) throw inspectionComplete;
      return {
        state: { toString: () => `stall:${stalledAgentId}` },
        completed: Promise.resolve(),
        finalOutput: undefined,
        async *[Symbol.asyncIterator]() {
          throw new ModelDecisionStallError(
            stalledAgentId,
            `${stalledAgentId} returned no formal decision`
          );
        }
      };
    });

    await expect(resume(store, config)).rejects.toBe(inspectionComplete);
    expect(runnerControl.run).toHaveBeenCalledTimes(4);
    expect(String(inputs[1])).toContain(
      `未完成正式决策的节点：${HUMANOID_AGENT_IDS.coordinator}`
    );
    expect(String(inputs[1])).toContain("续行轮次：1");
    expect(String(inputs[2])).toContain(
      `未完成正式决策的节点：${HUMANOID_AGENT_IDS.motion}`
    );
    expect(String(inputs[2])).toContain("续行轮次：1");
    expect(String(inputs[3])).toContain(
      `未完成正式决策的节点：${HUMANOID_AGENT_IDS.coordinator}`
    );
    expect(String(inputs[3])).toContain("续行轮次：2");
    const followUps = (await store.readJournal("provider")).filter((entry) => (
      isRecord(entry) && entry.status === "model_decision_follow_up"
    ));
    expect(followUps).toEqual([
      expect.objectContaining({
        agent_id: HUMANOID_AGENT_IDS.coordinator,
        follow_up_attempt: 1
      }),
      expect.objectContaining({
        agent_id: HUMANOID_AGENT_IDS.motion,
        follow_up_attempt: 1
      }),
      expect.objectContaining({
        agent_id: HUMANOID_AGENT_IDS.coordinator,
        follow_up_attempt: 2
      })
    ]);
  }, 30_000);

  it("scopes decision follow-ups to one authoritative progress state", () => {
    const first = nextModelDecisionFollowUpState(undefined, "authority-a");
    const second = nextModelDecisionFollowUpState(first ?? undefined, "authority-a");
    const third = nextModelDecisionFollowUpState(second ?? undefined, "authority-a");
    expect(first).toEqual({ authorityFingerprint: "authority-a", attempt: 1 });
    expect(second).toEqual({ authorityFingerprint: "authority-a", attempt: 2 });
    expect(third).toEqual({ authorityFingerprint: "authority-a", attempt: 3 });
    expect(nextModelDecisionFollowUpState(
      third ?? undefined,
      "authority-a"
    )).toBeNull();
    expect(nextModelDecisionFollowUpState(
      third ?? undefined,
      "authority-b"
    )).toEqual({ authorityFingerprint: "authority-b", attempt: 1 });
  });

  it("restores Sessions to the last persisted RunState before retrying", async () => {
    const store = await createCheckpointedRun();
    const config = provider();
    const manifest = createManifest(config);
    await store.writeAgentManifest(manifest);
    const baselineItems = new Map(Object.values(HUMANOID_AGENT_IDS).map((agentId) => (
      [agentId, [{ role: "user" as const, content: `baseline:${agentId}` }]]
    )));
    await Promise.all([...baselineItems.entries()].map(async ([agentId, items]) => {
      await missionSession(store, manifest.epoch_id, agentId).addItems(items);
    }));
    const transportError = Object.assign(new Error("nested connection reset"), {
      code: "ECONNRESET"
    });
    runnerControl.run.mockImplementationOnce(async (
      _agent,
      _runInput,
      options: { session: FileSession }
    ) => {
      await options.session.addItems([{
        role: "user",
        content: "aligned:humanoid-coordinator"
      }]);
      return {
        state: { toString: () => "invalid-but-persisted-sdk-state" },
        completed: Promise.resolve(),
        finalOutput: undefined,
        async *[Symbol.asyncIterator]() {
          yield {
            type: "raw_model_stream_event",
            data: { type: "response_started" }
          };
          yield {
            type: "raw_model_stream_event",
            data: { type: "output_text_delta", delta: "partial" }
          };
          await options.session.addItems([{
            role: "user",
            content: "suffix-after-state:humanoid-coordinator"
          }]);
          throw transportError;
        }
      };
    });

    await expect(resume(store, config)).rejects.toThrow();

    expect(runnerControl.run).toHaveBeenCalledTimes(1);
    for (const [agentId, items] of baselineItems) {
      expect(await missionSession(store, manifest.epoch_id, agentId).getItems()).toEqual([
        ...items,
        ...(agentId === HUMANOID_AGENT_IDS.coordinator
          ? [{ role: "user" as const, content: `aligned:${agentId}` }]
          : [])
      ]);
    }
    expect(await store.readAgentStateRecord()).toMatchObject({
      state: "invalid-but-persisted-sdk-state",
      sessionBaseline: Object.fromEntries(Object.values(HUMANOID_AGENT_IDS).map(
        (agentId) => [agentId, {
          item_count: agentId === HUMANOID_AGENT_IDS.coordinator ? 2 : 1
        }]
      ))
    });
  }, 30_000);
});

describe("humanoid Agent state persistence boundaries", () => {
  it("rebases only a disabled tool that remains in the configured Agent", () => {
    expect(recoverableDynamicToolRunStateError(
      new UserError("Tool delegate_motion_reference not found"),
      ["delegate_humanoid_sentry", "delegate_motion_reference"]
    )).toBe("delegate_motion_reference");
    expect(recoverableDynamicToolRunStateError(
      new UserError("Tool removed_tool not found"),
      ["delegate_motion_reference"]
    )).toBeUndefined();
    expect(recoverableDynamicToolRunStateError(
      new Error("Tool delegate_motion_reference not found"),
      ["delegate_motion_reference"]
    )).toBeUndefined();
  });

  it("does not serialize the complete SDK state for token deltas", () => {
    expect(shouldPersistHumanoidAgentState({
      type: "raw_model_stream_event",
      data: { type: "output_text_delta", delta: "一" }
    } as RunStreamEvent)).toBe(false);
  });

  it.each(["response_started", "response_done"] as const)(
    "persists the SDK state at the %s response boundary",
    (type) => {
      expect(shouldPersistHumanoidAgentState({
        type: "raw_model_stream_event",
        data: { type }
      } as RunStreamEvent)).toBe(true);
    }
  );

  it.each(["run_item_stream_event", "agent_updated_stream_event"] as const)(
    "persists the SDK state for %s",
    (type) => {
      expect(shouldPersistHumanoidAgentState({ type } as RunStreamEvent)).toBe(true);
    }
  );
});

function progressReceipt(transactionId: string, agentId: string) {
  return {
    transactionId,
    agentId,
    action: "observe_humanoid" as const,
    accepted: true,
    code: "humanoid_observed",
    worldBeforeRevision: 42,
    worldAfterRevision: 42,
    frameCount: 0
  };
}

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

function controllerSource(sourceSha256: string): HumanoidControllerSource {
  return {
    sourceSha256,
    controllerFactory: async () => {
      throw new Error("Controller construction is outside this compatibility test");
    }
  };
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

function missionSession(
  store: RunStore,
  epochId: string,
  agentId: string
): FileSession {
  return new FileSession(
    agentId === HUMANOID_AGENT_IDS.coordinator
      ? store.sessionPath()
      : store.workerSessionPath(agentId),
    `${store.definition.run_id}:${epochId}:${agentId}`
  );
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
