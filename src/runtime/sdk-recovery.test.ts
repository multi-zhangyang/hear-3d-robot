import {
  Agent,
  Runner,
  tool,
  type Model,
  type RunStreamEvent
} from "@openai/agents";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import type { ProviderConfig } from "../config/load.js";
import { loadRuntimeCatalog } from "../config/load.js";
import type { AgentSpec, RunCheckpoint } from "../domain/schema.js";
import { capabilityCatalog } from "../harness/agents.js";
import { receiptEvidenceRequirement } from "../harness/evidence-contract.js";
import { HierarchyProjection } from "../harness/hierarchy-projection.js";
import { createCheckpoint } from "../harness/runtime-context.js";
import { FileSession } from "../persistence/file-session.js";
import { RunStore } from "../persistence/run-store.js";
import { RapierWorld } from "../world/rapier-world.js";
import { resumeMission } from "./mission-runner.js";
import {
  assertRunStateMatchesOpenRootDelegations,
  pendingFunctionCallIds,
  sdkRecoveryCheckpointFingerprint
} from "./sdk-recovery.js";

const recoveryModel = vi.hoisted(() => ({
  turn: 0,
  requests: [] as string[]
}));

vi.mock("../model/factory.js", async (importOriginal) => {
  const original = await importOriginal<typeof import("../model/factory.js")>();
  return {
    ...original,
    createConfiguredModel: () => ({
      getResponse: async () => {
        throw new Error("Recovery integration must use the SDK streaming lifecycle");
      },
      getStreamedResponse: (request: unknown) => (async function* () {
        recoveryModel.requests.push(JSON.stringify(request));
        const turn = recoveryModel.turn;
        recoveryModel.turn += 1;
        const output = turn === 0
          ? [functionCall("recovery_check", "check_mission", {})]
          : [functionCall("recovery_complete", "complete_mission", {
              summary: "恢复后的真实 Runner 已完成当前目标"
            })];
        yield responseDone(`recovery_${turn}`, output);
      })()
    } as Model)
  };
});

const TEST_PROVIDER: ProviderConfig = {
  protocol: "openai_compatible",
  baseUrl: "https://example.test/v1",
  model: "test-model",
  apiKey: "test-key",
  temperature: 0,
  maxOutputTokens: 512,
  contextWindowTokens: 262_144,
  compactTriggerTokens: 200_000,
  compactRecentModelTurns: 2,
  compactMaxOutputTokens: 2_048
};

afterEach(() => {
  recoveryModel.turn = 0;
  recoveryModel.requests = [];
  vi.restoreAllMocks();
});

describe("SDK recovery state alignment", () => {
  it("observes the original pending call ID before a real SDK tool finishes", async () => {
    const toolStarted = deferred<void>();
    const releaseTool = deferred<void>();
    const callId = "call_open_child";
    let modelTurn = 0;
    const model: Model = {
      getResponse: async () => {
        throw new Error("The regression requires streaming");
      },
      getStreamedResponse: () => (async function* () {
        const output = modelTurn === 0
          ? [functionCall(callId, "delegate_agent", {})]
          : [assistantMessage("tool finished")];
        modelTurn += 1;
        yield responseDone(`alignment_${modelTurn}`, output);
      })()
    };
    const delegate = tool({
      name: "delegate_agent",
      description: "Block until the recovery snapshot is captured.",
      parameters: z.object({}).strict(),
      strict: true,
      execute: async () => {
        toolStarted.resolve();
        await releaseTool.promise;
        return "done";
      }
    });
    const agent = new Agent({
      name: "Mission Coordinator",
      instructions: "Use the provided tool.",
      model,
      tools: [delegate]
    });
    const runner = new Runner({ tracingDisabled: true });
    const stream = await runner.run(agent, "Delegate once.", {
      stream: true,
      maxTurns: 3
    });
    let persistedState: unknown;
    const stateCaptured = deferred<void>();
    const consume = (async () => {
      for await (const event of stream) {
        if (isToolCalled(event, callId)) {
          persistedState = stream.state.toJSON();
          stateCaptured.resolve();
        }
      }
    })();

    await Promise.all([toolStarted.promise, stateCaptured.promise]);
    expect(pendingFunctionCallIds(persistedState)).toEqual(new Set([callId]));

    const hierarchy = HierarchyProjection.create(
      "Exercise SDK recovery",
      ["sense_scene", "read_proprioception"]
    );
    hierarchy.enterChild(null, recoveryChildSpec(), callId);
    const checkpoint = {
      root_id: hierarchy.rootId,
      nodes: hierarchy.snapshot()
    } as RunCheckpoint;
    expect(() => assertRunStateMatchesOpenRootDelegations(checkpoint, persistedState))
      .not.toThrow();

    const mismatched = structuredClone(checkpoint);
    const childId = mismatched.nodes[mismatched.root_id]!.child_ids[0]!;
    mismatched.nodes[childId]!.source_call_id = "different_call";
    expect(() => assertRunStateMatchesOpenRootDelegations(mismatched, persistedState))
      .toThrow("does not own every unfinished root delegation");

    releaseTool.resolve();
    await consume;
    await stream.completed;
    expect(pendingFunctionCallIds(stream.state.toJSON())).toEqual(new Set());
  });

  it("rotates an incompatible RunState and every stale Session before resuming", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-sdk-recovery-"));
    try {
      const catalog = await loadRuntimeCatalog();
      const scenario = catalog.materialize("open_navigation", 0);
      const world = await RapierWorld.create(scenario);
      const initialPosition = world.snapshot().robot.position;
      const goal = {
        summary: "Robot remains at its committed coordinate.",
        predicates: [{
          type: "robot_at" as const,
          target: initialPosition,
          tolerance: 0.01
        }]
      };
      const store = await RunStore.create(runsDir, {
        mission: "Resume from authoritative world state",
        scenarioId: "open_navigation",
        scenario,
        goal
      });
      const hierarchy = HierarchyProjection.create(
        store.definition.mission,
        capabilityCatalog(),
        goal.predicates.length
      );
      const openChild = hierarchy.enterChild(
        null,
        recoveryChildSpec(),
        "checkpoint_open_call"
      ).node;
      const checkpoint = createCheckpoint({
        store,
        hierarchy,
        capabilityCatalog: capabilityCatalog(),
        world
      });
      checkpoint.status = "interrupted";
      checkpoint.error = "process stopped during delegation";
      await store.writeCheckpoint(checkpoint);
      world.dispose();

      const lifecycleOnly = structuredClone(checkpoint);
      lifecycleOnly.status = "running";
      lifecycleOnly.error = null;
      lifecycleOnly.updated_at = new Date(Date.now() + 1_000).toISOString();
      expect(sdkRecoveryCheckpointFingerprint(lifecycleOnly))
        .toBe(sdkRecoveryCheckpointFingerprint(checkpoint));
      const hierarchyChanged = structuredClone(checkpoint);
      hierarchyChanged.nodes[openChild.id]!.status = "completed";
      expect(sdkRecoveryCheckpointFingerprint(hierarchyChanged))
        .not.toBe(sdkRecoveryCheckpointFingerprint(checkpoint));

      const staleState = await completedRunState();
      await store.writeAgentState(
        staleState,
        sdkRecoveryCheckpointFingerprint(checkpoint)
      );
      await new FileSession(store.sessionPath(), checkpoint.run_id).addItems([{
        role: "user",
        content: "STALE_ROOT_SESSION"
      }]);
      await new FileSession(
        store.workerSessionPath(openChild.id),
        `${checkpoint.run_id}:${openChild.id}`
      ).addItems([{ role: "user", content: "STALE_WORKER_SESSION" }]);

      const simulatedExit = new Error("process exited during SDK Session cleanup");
      let mutations = 0;
      await expect(resumeMission({
        runDir: store.runDir,
        catalog,
        provider: TEST_PROVIDER,
        mutationFence: {
          async runMutation<T>(operation: () => Promise<T>): Promise<T> {
            mutations += 1;
            if (mutations === 1) return operation();
            throw simulatedExit;
          }
        }
      })).rejects.toBe(simulatedExit);
      expect(await store.readAgentState()).toBeUndefined();
      expect((await store.readCheckpoint()).nodes[openChild.id]?.status).toBe("active");

      const result = await resumeMission({
        runDir: store.runDir,
        catalog,
        provider: TEST_PROVIDER
      });

      expect(result.output).toContain("恢复后的真实 Runner");
      expect(recoveryModel.requests.join("\n")).not.toContain("STALE_");
      const resumedCheckpoint = await store.readCheckpoint();
      expect(resumedCheckpoint.nodes[openChild.id]).toMatchObject({
        status: "failed",
        last_result: {
          error: expect.stringContaining("RunState call ownership")
        }
      });
      const providerEvents = await store.readJournal("provider");
      expect(providerEvents).toContainEqual(expect.objectContaining({
        status: "context_branch_rotated",
        source: "sdk_state_missing",
        cleared: ["sdk_run_state", "root_sdk_session", "worker_sdk_sessions"]
      }));
      const rootItems = await new FileSession(
        store.sessionPath(),
        checkpoint.run_id
      ).getItems();
      expect(JSON.stringify(rootItems)).not.toContain("STALE_ROOT_SESSION");
      expect(await new FileSession(
        store.workerSessionPath(openChild.id),
        `${checkpoint.run_id}:${openChild.id}`
      ).getItems()).toEqual([]);
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  }, 20_000);
});

async function completedRunState(): Promise<string> {
  const model: Model = {
    getResponse: async () => ({
      usage: usage(),
      output: [assistantMessage("stale completed branch")]
    }),
    getStreamedResponse: () => (async function* () {
      throw new Error("Stale state fixture uses the non-streaming SDK path");
    })()
  };
  const agent = new Agent({
    name: "Mission Coordinator",
    instructions: "Complete this obsolete branch.",
    model
  });
  const result = await new Runner({ tracingDisabled: true }).run(agent, "Old turn.");
  return result.state.toString();
}

function recoveryChildSpec(): AgentSpec {
  return {
    name: "Recovery observer",
    objective: "Observe the committed world once.",
    success_criteria: ["A current scene receipt is returned."],
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
}

function functionCall(callId: string, name: string, input: object) {
  return {
    type: "function_call" as const,
    callId,
    name,
    arguments: JSON.stringify(input),
    status: "completed" as const
  };
}

function assistantMessage(text: string) {
  return {
    type: "message" as const,
    role: "assistant" as const,
    status: "completed" as const,
    content: [{ type: "output_text" as const, text }]
  };
}

function responseDone(id: string, output: ReturnType<typeof functionCall>[] | ReturnType<typeof assistantMessage>[]) {
  return {
    type: "response_done" as const,
    response: { id, usage: usage(), output }
  };
}

function usage() {
  return {
    requests: 1,
    inputTokens: 1,
    outputTokens: 1,
    totalTokens: 2
  };
}

function isToolCalled(event: RunStreamEvent, callId: string): boolean {
  return event.type === "run_item_stream_event"
    && event.name === "tool_called"
    && event.item.rawItem.type === "function_call"
    && event.item.rawItem.callId === callId;
}

function deferred<T>() {
  let resolve: (value: T | PromiseLike<T>) => void = () => undefined;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}
