import { createHash } from "node:crypto";
import {
  Runner,
  RunState,
  setTracingDisabled,
  type RunStreamEvent
} from "@openai/agents";
import type { ProviderConfig, RuntimeCatalog } from "../config/load.js";
import type { Goal } from "../domain/schema.js";
import { LongRunContextManager } from "../harness/context-compaction.js";
import {
  AgentsSdkContextSummaryGenerator,
  isContextCompactionInterruption
} from "../harness/context-summary-agent.js";
import {
  createHumanoidAgentHierarchy,
  HUMANOID_AGENT_IDS
} from "../harness/humanoid/agents.js";
import { createHumanoidRunCheckpoint } from "../harness/humanoid/run-checkpoint.js";
import { HumanoidRunRuntime } from "../harness/humanoid/run-runtime.js";
import {
  ModelDecisionStallError,
  withModelTelemetry
} from "../harness/model-telemetry.js";
import { providerEventJson, sdkEventJson } from "../harness/sdk-events.js";
import { createConfiguredModel, providerIdentity } from "../model/factory.js";
import { FileSession } from "../persistence/file-session.js";
import type { MutationFence } from "../persistence/mutation-fence.js";
import { RunStore } from "../persistence/run-store.js";
import type { RuntimeEventSink } from "./events.js";
import { errorMessage } from "./error-message.js";
import { assertGoalSupported } from "./goal-validation.js";
import { assertHumanoidGoalSupported } from "./humanoid-checker.js";
import {
  ConsecutiveTransportRecovery,
  isTransportInterruption,
  transportRetryPlan
} from "./transport-recovery.js";
import { HumanoidWorld } from "../world/humanoid/world.js";
import { drawSeed } from "../world/world-generator.js";

setTracingDisabled(true);

const MAX_TRANSPORT_RECOVERIES = 8;
const MAX_MODEL_DECISION_RECOVERIES = 3;

export interface HumanoidMissionRunResult {
  runId: string;
  runDir: string;
  output: string;
}

export async function startHumanoidMission(input: {
  runsDir: string;
  mission: string;
  scenarioId: string;
  goal: Goal;
  catalog: RuntimeCatalog;
  provider: ProviderConfig;
  seed?: number;
  eventSink?: RuntimeEventSink;
  signal?: AbortSignal;
  mutationFence?: MutationFence;
}): Promise<HumanoidMissionRunResult> {
  const scenario = input.catalog.materialize(
    input.scenarioId,
    input.seed ?? drawSeed()
  );
  assertGoalSupported(input.goal, scenario);
  assertHumanoidGoalSupported(input.goal, scenario);
  const store = await RunStore.create(input.runsDir, {
    mission: input.mission,
    scenarioId: input.scenarioId,
    scenario,
    goal: input.goal,
    runtime: "humanoid_g1"
  }, input.mutationFence ? { mutationFence: input.mutationFence } : {});
  const world = await HumanoidWorld.create(scenario);
  try {
    const checkpoint = createHumanoidRunCheckpoint({
      store,
      goal: input.goal,
      world
    });
    await store.writeCheckpoint(checkpoint);
    const runtime = new HumanoidRunRuntime({
      store,
      goal: input.goal,
      world,
      checkpoint,
      ...(input.eventSink ? { eventSink: input.eventSink } : {}),
      ...(input.signal ? { signal: input.signal } : {})
    });
    return await executeHumanoidMission({
      runtime,
      provider: input.provider,
      resumed: false,
      ...(input.signal ? { signal: input.signal } : {})
    });
  } finally {
    await world.dispose();
  }
}

export async function resumeHumanoidMission(input: {
  runDir: string;
  catalog: RuntimeCatalog;
  provider: ProviderConfig;
  eventSink?: RuntimeEventSink;
  signal?: AbortSignal;
  mutationFence?: MutationFence;
}): Promise<HumanoidMissionRunResult> {
  const store = await RunStore.open(
    input.runDir,
    input.mutationFence ? { mutationFence: input.mutationFence } : {}
  );
  if (store.definition.runtime !== "humanoid_g1") {
    throw new Error("This run was not created by the humanoid runtime");
  }
  const checkpoint = await store.readHumanoidCheckpoint();
  if (checkpoint.status === "succeeded") throw new Error("A succeeded run cannot be resumed");
  const world = await HumanoidWorld.create(
    store.definition.scenario,
    checkpoint.world_checkpoint
  );
  try {
    const runtime = new HumanoidRunRuntime({
      store,
      goal: store.definition.goal,
      world,
      checkpoint,
      ...(input.eventSink ? { eventSink: input.eventSink } : {}),
      ...(input.signal ? { signal: input.signal } : {})
    });
    return await executeHumanoidMission({
      runtime,
      provider: input.provider,
      resumed: true,
      ...(input.signal ? { signal: input.signal } : {})
    });
  } finally {
    await world.dispose();
  }
}

async function executeHumanoidMission(input: {
  runtime: HumanoidRunRuntime;
  provider: ProviderConfig;
  resumed: boolean;
  signal?: AbortSignal;
}): Promise<HumanoidMissionRunResult> {
  const transportRecovery = new ConsecutiveTransportRecovery(MAX_TRANSPORT_RECOVERIES);
  const sessions = new Map<string, FileSession>();
  const sessionForAgent = (agentId: string): FileSession => {
    const existing = sessions.get(agentId);
    if (existing) return existing;
    const created = new FileSession(
      agentId === input.runtime.rootAgentId
        ? input.runtime.store.sessionPath()
        : input.runtime.store.workerSessionPath(agentId),
      `${input.runtime.runId}:${agentId}`,
      input.runtime.store.mutationFence
    );
    sessions.set(agentId, created);
    return created;
  };
  const onModelResponseCompleted = async (agentId: string): Promise<void> => {
    const recovered = transportRecovery.responseCompleted();
    if (recovered === 0) return;
    await input.runtime.recordProvider({
      status: "transport_recovered",
      consecutive_interruptions: recovered,
      recovery_window_reset: true
    }, agentId);
  };
  const contextManager = new LongRunContextManager({
    runtime: input.runtime,
    provider: input.provider,
    sessionForAgent,
    createGenerator: (agentId) => new AgentsSdkContextSummaryGenerator({
      model: createConfiguredModel(input.provider),
      temperature: input.provider.temperature,
      maxOutputTokens: input.provider.compactMaxOutputTokens,
      onModelResponseCompleted: () => onModelResponseCompleted(agentId)
    })
  });

  const persistAgentEvent = async (
    agentId: string,
    event: RunStreamEvent
  ): Promise<void> => {
    await input.runtime.setActiveAgent(agentId);
    await persistStreamEvent(input.runtime, agentId, event);
  };
  const hierarchy = createHumanoidAgentHierarchy({
    createModel: (agentId) => withModelTelemetry(
      createConfiguredModel(input.provider),
      input.runtime,
      agentId,
      onModelResponseCompleted
    ),
    createSession: sessionForAgent,
    provider: input.provider,
    runtime: input.runtime,
    onAgentStream: persistAgentEvent
  });
  const runner = new Runner({
    tracingDisabled: true,
    traceIncludeSensitiveData: false,
    modelSettings: { parallelToolCalls: false },
    callModelInputFilter: contextManager.filter,
    toolExecution: { maxFunctionToolConcurrency: 1 },
    toolNotFoundBehavior: "return_error_to_model",
    workflowName: "HEAR humanoid autonomy"
  });
  const coordinatorSession = hierarchy.coordinatorSession as FileSession;

  try {
    await input.runtime.start(input.resumed);
    const controller = input.runtime.snapshot().robot.controller;
    await input.runtime.recordProvider({
      status: "configured",
      ...providerIdentity(input.provider),
      hierarchy: "one_model_facade_and_session_per_agent",
      physics: `mujoco_${formatFrequency(controller.physicsStepSeconds)}hz`,
      controller: `${controller.implementation}_${formatFrequency(
        controller.controlStepSeconds
      )}hz`,
      controller_protocol: controller.protocol,
      actuation: controller.actuation
    }, input.runtime.rootAgentId);

    let serializedState = await resumableAgentState(input.runtime);
    let decisionRecoveries = 0;
    for (;;) {
      input.signal?.throwIfAborted();
      const baseline = {
        fingerprint: humanoidCheckpointFingerprint(input.runtime.checkpoint),
        sessionItems: await coordinatorSession.getItems()
      };
      try {
        const runInput = serializedState
          ? await RunState.fromString(hierarchy.coordinator, serializedState)
          : autonomousCycleInput(input.runtime);
        serializedState = undefined;
        const stream = await runner.run(hierarchy.coordinator, runInput, {
          stream: true,
          session: coordinatorSession,
          maxTurns: null,
          toolExecution: { maxFunctionToolConcurrency: 1 },
          toolNotFoundBehavior: "return_error_to_model",
          ...(input.signal ? { signal: input.signal } : {})
        });
        for await (const event of stream) {
          await persistAgentEvent(HUMANOID_AGENT_IDS.coordinator, event);
          await input.runtime.store.writeAgentState(
            stream.state.toString(),
            humanoidCheckpointFingerprint(input.runtime.checkpoint)
          );
        }
        await stream.completed;
        input.signal?.throwIfAborted();
        await contextManager.compactSessionHistories(sessionForAgent);
        const output = typeof stream.finalOutput === "string"
          ? stream.finalOutput
          : JSON.stringify(stream.finalOutput);
        assertCycleOutput(output);
        const succeeded = await input.runtime.completeCycle(output);
        await input.runtime.store.clearAgentState();
        decisionRecoveries = 0;
        transportRecovery.responseCompleted();
        if (succeeded) {
          await input.runtime.succeed(output);
          return {
            runId: input.runtime.runId,
            runDir: input.runtime.store.runDir,
            output
          };
        }
        await input.runtime.setActiveAgent(input.runtime.rootAgentId);
      } catch (error) {
        if (input.signal?.aborted) throw error;
        if (error instanceof ModelDecisionStallError
          && decisionRecoveries < MAX_MODEL_DECISION_RECOVERIES) {
          decisionRecoveries += 1;
          contextManager.startFreshSdkTurn(error.agentId);
          await sessionForAgent(error.agentId).clearSession();
          await input.runtime.store.clearAgentState();
          await input.runtime.recordProvider({
            status: "model_decision_recovery",
            agent_id: error.agentId,
            recovery_attempt: decisionRecoveries,
            maximum_recoveries: MAX_MODEL_DECISION_RECOVERIES,
            error: error.message,
            automatic_actuation: false
          }, error.agentId);
          serializedState = undefined;
          continue;
        }
        if (!isTransportInterruption(error)) throw error;
        const recoveryAttempt = transportRecovery.nextAttempt();
        if (recoveryAttempt === null) throw error;
        const persisted = await resumableAgentState(input.runtime);
        const unchanged = baseline.fingerprint
          === humanoidCheckpointFingerprint(input.runtime.checkpoint);
        if (!persisted && unchanged) {
          await coordinatorSession.replaceItems(baseline.sessionItems);
        }
        serializedState = persisted;
        const retry = transportRetryPlan(error, recoveryAttempt);
        await input.runtime.recordProvider({
          status: "transport_interrupted",
          recovery_attempt: recoveryAttempt,
          maximum_recoveries: MAX_TRANSPORT_RECOVERIES,
          retry_after_ms: retry.waitMs,
          exponential_backoff_ms: retry.backoffMs,
          ...(retry.retryAfterMs === null
            ? {}
            : {
                server_retry_after_ms: retry.retryAfterMs,
                retry_source: retry.retryAfterMs > retry.backoffMs
                  ? "server_retry_after"
                  : "exponential_backoff"
              }),
          resumed_sdk_state: serializedState !== undefined,
          error: errorMessage(error),
          automatic_actuation: false
        });
        await delay(retry.waitMs, input.signal);
      }
    }
  } catch (error) {
    const message = errorMessage(error);
    if (input.signal?.aborted || isContextCompactionInterruption(error)) {
      await input.runtime.interrupt(message);
    } else {
      await input.runtime.fail(message);
    }
    throw error;
  }
}

async function persistStreamEvent(
  runtime: HumanoidRunRuntime,
  agentId: string,
  event: RunStreamEvent
): Promise<void> {
  const framework = sdkEventJson(event);
  const provider = providerEventJson(event);
  if (framework) await runtime.recordFramework(`agent:${agentId}`, framework, agentId);
  if (provider) await runtime.recordProvider(provider, agentId);
}

async function resumableAgentState(runtime: HumanoidRunRuntime): Promise<string | undefined> {
  const record = await runtime.store.readAgentStateRecord();
  if (!record) return undefined;
  if (record.checkpointFingerprint !== humanoidCheckpointFingerprint(runtime.checkpoint)) {
    await runtime.store.clearAgentState();
    return undefined;
  }
  return record.state;
}

function autonomousCycleInput(runtime: HumanoidRunRuntime): string {
  const checkpoint = runtime.checkpoint;
  return [
    "继续下一次人形自主闭环。",
    "必须根据当前传感、目标、历史回执和物理反馈自主决定动作；不得使用固定动作表、预设路径或假执行。",
    `任务：${runtime.store.definition.mission}`,
    `当前循环：${checkpoint.cycle_index + 1}`,
    `当前世界：frame=${checkpoint.world.frame}, revision=${checkpoint.world.worldRevision}`,
    "完成一次真实物理执行后，用 accepted 执行回执调用 complete_autonomous_cycle。"
  ].join("\n");
}

function assertCycleOutput(output: string | undefined): asserts output is string {
  if (!output?.trim()) throw new Error("Humanoid coordinator returned no cycle output");
  let parsed: unknown;
  try {
    parsed = JSON.parse(output);
  } catch {
    throw new Error("Humanoid coordinator did not return the cycle completion tool result");
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)
    || (parsed as Record<string, unknown>).status !== "cycle_completed") {
    throw new Error("Humanoid coordinator did not complete a verified autonomous cycle");
  }
}

function humanoidCheckpointFingerprint(checkpoint: {
  world: { frame: number; worldRevision: number };
  committed_actions: Record<string, unknown>;
  context_memory: { total_compactions: number };
  cycle_index: number;
}): string {
  return createHash("sha256").update(JSON.stringify({
    frame: checkpoint.world.frame,
    worldRevision: checkpoint.world.worldRevision,
    committedActions: Object.keys(checkpoint.committed_actions).sort(),
    totalCompactions: checkpoint.context_memory.total_compactions,
    cycleIndex: checkpoint.cycle_index
  })).digest("hex");
}

function formatFrequency(stepSeconds: number): string {
  const frequency = 1 / stepSeconds;
  return Number.isInteger(frequency)
    ? String(frequency)
    : frequency.toFixed(2).replace(/\.?0+$/u, "");
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
