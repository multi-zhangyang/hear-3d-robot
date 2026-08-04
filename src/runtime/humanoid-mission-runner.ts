import { createHash, randomUUID } from "node:crypto";
import { readdir, stat } from "node:fs/promises";
import { resolve } from "node:path";
import {
  Runner,
  RunState,
  setTracingDisabled,
  type RunStreamEvent
} from "@openai/agents";
import {
  providerConfigForRole,
  type ProviderConfig,
  type RuntimeCatalog
} from "../config/load.js";
import type { Goal } from "../domain/schema.js";
import {
  humanoidRunShouldFinish,
  type HumanoidRunMode
} from "../domain/run-mode.js";
import { LongRunContextManager } from "../harness/context-compaction.js";
import type {
  ModelProgressSnapshot,
  ModelTelemetryRuntime
} from "../harness/context-runtime.js";
import {
  AgentsSdkContextSummaryGenerator,
  isContextCompactionInterruption
} from "../harness/context-summary-agent.js";
import {
  AgentManifestIncompatibleError,
  assertAgentManifestCompatible,
  createHumanoidAgentManifest
} from "../harness/agent-manifest.js";
import {
  createHumanoidAgentHierarchy,
  humanoidAgentRole,
  HUMANOID_AGENT_IDS
} from "../harness/humanoid/agents.js";
import { createHumanoidRunCheckpoint } from "../harness/humanoid/run-checkpoint.js";
import { HumanoidRunRuntime } from "../harness/humanoid/run-runtime.js";
import { assertHumanoidPhysicalWorldDeltaRecovery } from "../harness/humanoid/physical-world-delta.js";
import {
  modelDecisionStallFrom,
  withModelTelemetry
} from "../harness/model-telemetry.js";
import { providerEventJson, sdkEventJson } from "../harness/sdk-events.js";
import { createConfiguredModel, providerIdentity } from "../model/factory.js";
import { FileSession } from "../persistence/file-session.js";
import type { MutationFence } from "../persistence/mutation-fence.js";
import { RunStore } from "../persistence/run-store.js";
import { configuredOutputTokenLimit } from "./context-budget.js";
import type { RuntimeEventSink } from "./events.js";
import { errorMessage } from "./error-message.js";
import { isRunPauseRequested } from "./run-pause.js";
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
  runMode?: HumanoidRunMode;
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
    runtime: "humanoid_g1",
    runMode: input.runMode ?? "mission"
  }, input.mutationFence ? { mutationFence: input.mutationFence } : {});
  const scenarioChunks = await store.readScenarioChunkDeltaState();
  const world = await HumanoidWorld.create(scenario, undefined, { scenarioChunks });
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
  const scenarioChunks = await store.readScenarioChunkDeltaState();
  assertHumanoidPhysicalWorldDeltaRecovery({
    scenario: store.definition.scenario,
    chunks: scenarioChunks,
    world: checkpoint.world
  });
  const world = await HumanoidWorld.create(
    store.definition.scenario,
    checkpoint.world_checkpoint,
    { scenarioChunks }
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
  let initialized = false;
  try {
    const persistedManifest = await persistedManifestForMission(input);
    const manifestEpochId = persistedManifest?.epoch_id ?? randomUUID();
    const sessions = new Map<string, FileSession>();
    const sessionForAgent = (agentId: string): FileSession => {
      const existing = sessions.get(agentId);
      if (existing) return existing;
      const created = new FileSession(
        agentId === input.runtime.rootAgentId
          ? input.runtime.store.sessionPath()
          : input.runtime.store.workerSessionPath(agentId),
        `${input.runtime.runId}:${manifestEpochId}:${agentId}`,
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
    const coordinatorProvider = providerConfigForRole(input.provider, "coordinator");
    const compactorProvider = providerConfigForRole(input.provider, "compactor");
    const compactorOutputLimit = configuredOutputTokenLimit(
      compactorProvider.compactMaxOutputTokens,
      compactorProvider.maxOutputTokens
    );
    const contextManager = new LongRunContextManager({
      runtime: input.runtime,
      provider: coordinatorProvider,
      configForAgent: (agentId) => providerConfigForRole(
        input.provider,
        humanoidAgentRole(agentId)
      ),
      compactorProvider,
      sessionForAgent,
      createGenerator: (agentId) => new AgentsSdkContextSummaryGenerator({
        model: createConfiguredModel(compactorProvider),
        temperature: compactorProvider.temperature,
        ...(compactorOutputLimit === undefined
          ? {}
          : { maxOutputTokens: compactorOutputLimit }),
        onModelResponseCompleted: () => onModelResponseCompleted(agentId)
      })
    });
    const modelTelemetryRuntime: ModelTelemetryRuntime = {
      rootAgentId: input.runtime.rootAgentId,
      activeNode: (agentId) => input.runtime.activeNode(agentId),
      recordModelCallStarted: (agentId) => input.runtime.recordModelCallStarted(agentId),
      recordModelCallCompleted: (record) => input.runtime.recordModelCallCompleted(record),
      recordModelCallFailed: (modelCallId, agentId) => (
        input.runtime.recordModelCallFailed(modelCallId, agentId)
      ),
      modelProgressSnapshot: () => humanoidModelProgressSnapshot(input.runtime)
    };

    const persistAgentEvent = async (
      agentId: string,
      event: RunStreamEvent
    ): Promise<void> => {
      await input.runtime.setActiveAgent(agentId);
      await persistStreamEvent(input.runtime, agentId, event);
    };
    const hierarchy = createHumanoidAgentHierarchy({
      createModel: (agentId, provider) => withModelTelemetry(
        createConfiguredModel(provider),
        modelTelemetryRuntime,
        agentId,
        onModelResponseCompleted
      ),
      createSession: sessionForAgent,
      provider: input.provider,
      runtime: input.runtime,
      onAgentStream: persistAgentEvent
    });
    const currentManifest = createHumanoidAgentManifest({
      hierarchy,
      provider: input.provider,
      epochId: manifestEpochId,
      ...(persistedManifest ? { createdAt: persistedManifest.created_at } : {})
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

    if (persistedManifest) {
      assertAgentManifestCompatible(persistedManifest, currentManifest);
    } else {
      await input.runtime.store.writeAgentManifest(currentManifest);
    }
    await input.runtime.initializeGoalAutonomy(
      persistedManifest ?? currentManifest
    );
    await input.runtime.start(input.resumed);
    const controller = input.runtime.snapshot().robot.controller;
    await input.runtime.recordProvider({
      status: "configured",
      ...providerIdentity(coordinatorProvider),
      hierarchy: "one_model_facade_and_session_per_agent",
      agent_manifest_epoch: currentManifest.epoch_id,
      agent_profiles: Object.fromEntries(Object.entries(currentManifest.agents).map(
        ([role, profile]) => [role, {
          agent_id: profile.agent_id,
          protocol: profile.protocol,
          model: profile.model,
          context_window_tokens: profile.settings.context_window_tokens,
          compact_trigger_tokens: profile.settings.compact_trigger_tokens
        }]
      )),
      physics: `mujoco_${formatFrequency(controller.physicsStepSeconds)}hz`,
      controller: `${controller.implementation}_${formatFrequency(
        controller.controlStepSeconds
      )}hz`,
      controller_protocol: controller.protocol,
      actuation: controller.actuation
    }, input.runtime.rootAgentId);
    initialized = true;

    if (input.runtime.store.definition.run_mode === "mission"
      && input.runtime.missionGoalCompleted()) {
      const lastCycle = input.runtime.checkpoint.last_cycle;
      if (lastCycle === null) {
        throw new Error("Completed Mission Goal has no persisted model cycle output");
      }
      const output = typeof lastCycle === "string"
        ? lastCycle
        : JSON.stringify(lastCycle);
      await input.runtime.succeed(output);
      return {
        runId: input.runtime.runId,
        runDir: input.runtime.store.runDir,
        output
      };
    }

    let serializedState = await resumableAgentState(input.runtime);
    let decisionRecoveries = 0;
    for (;;) {
      input.signal?.throwIfAborted();
      await input.runtime.ensureAutonomousCycle();
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
        const completion = assertCycleOutput(output);
        if (completion.status === "cycle_completed") {
          const activeGoalCompleted = await input.runtime.completeCycle(output);
          if (humanoidRunShouldFinish({
            mode: input.runtime.store.definition.run_mode,
            activeGoalCompleted,
            missionGoalCompleted: input.runtime.missionGoalCompleted()
          })) {
            await input.runtime.succeed(output);
            return {
              runId: input.runtime.runId,
              runDir: input.runtime.store.runDir,
              output
            };
          }
        } else {
          input.runtime.validateGoalTransition();
        }
        await input.runtime.store.clearAgentState();
        decisionRecoveries = 0;
        transportRecovery.responseCompleted();
        await input.runtime.setActiveAgent(input.runtime.rootAgentId);
      } catch (error) {
        if (input.signal?.aborted) throw error;
        const decisionStall = modelDecisionStallFrom(error);
        if (decisionStall
          && decisionRecoveries < MAX_MODEL_DECISION_RECOVERIES) {
          decisionRecoveries += 1;
          contextManager.startFreshSdkTurn(decisionStall.agentId);
          await sessionForAgent(decisionStall.agentId).clearSession();
          await input.runtime.store.clearAgentState();
          await input.runtime.recordProvider({
            status: "model_decision_recovery",
            agent_id: decisionStall.agentId,
            recovery_attempt: decisionRecoveries,
            maximum_recoveries: MAX_MODEL_DECISION_RECOVERIES,
            error: decisionStall.message,
            ...(decisionStall.evidence
              ? { stall_evidence: decisionStall.evidence }
              : {}),
            automatic_actuation: false
          }, decisionStall.agentId);
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
    if (input.signal?.aborted && isRunPauseRequested(input.signal.reason)) {
      await input.runtime.pause(message);
    } else if (!initialized
      || input.signal?.aborted
      || error instanceof AgentManifestRecoveryInterruption
      || error instanceof AgentManifestIncompatibleError
      || isContextCompactionInterruption(error)
      || isTransportInterruption(error)) {
      await input.runtime.interrupt(message);
    } else {
      await input.runtime.fail(message);
    }
    throw error;
  }
}

async function persistedManifestForMission(input: {
  runtime: HumanoidRunRuntime;
  resumed: boolean;
}): Promise<Awaited<ReturnType<RunStore["readAgentManifest"]>> | undefined> {
  if (!input.resumed) return undefined;
  const store = input.runtime.store;
  let manifestExists: boolean;
  try {
    manifestExists = await pathExists(resolve(store.runDir, "agent-manifest.json"));
  } catch (error) {
    throw manifestRecoveryInterruption("Unable to inspect the persisted agent manifest", error);
  }
  if (manifestExists) {
    try {
      return await store.readAgentManifest();
    } catch (error) {
      throw manifestRecoveryInterruption("Unable to read the persisted agent manifest", error);
    }
  }

  let hasRuntimeState: boolean;
  try {
    hasRuntimeState = await hasPersistedAgentRuntimeState(store);
  } catch (error) {
    throw manifestRecoveryInterruption("Unable to verify the persisted Agent runtime state", error);
  }
  if (hasRuntimeState) {
    throw new AgentManifestRecoveryInterruption(
      "Agent manifest is missing while Agent state or Session state exists; refusing to create a replacement epoch"
    );
  }
  return undefined;
}

async function hasPersistedAgentRuntimeState(store: RunStore): Promise<boolean> {
  const [agentStateExists, coordinatorSessionExists] = await Promise.all([
    pathExists(resolve(store.runDir, "agent-state.json")),
    pathExists(store.sessionPath())
  ]);
  if (agentStateExists || coordinatorSessionExists) return true;
  try {
    return (await readdir(resolve(store.runDir, "sessions"))).length > 0;
  } catch (error) {
    if (isMissingFile(error)) return false;
    throw error;
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (isMissingFile(error)) return false;
    throw error;
  }
}

function isMissingFile(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

function manifestRecoveryInterruption(
  prefix: string,
  cause: unknown
): AgentManifestRecoveryInterruption {
  return new AgentManifestRecoveryInterruption(`${prefix}: ${errorMessage(cause)}`, cause);
}

class AgentManifestRecoveryInterruption extends Error {
  constructor(message: string, cause?: unknown) {
    super(message, { cause });
    this.name = "AgentManifestRecoveryInterruption";
  }
}

function humanoidModelProgressSnapshot(
  runtime: HumanoidRunRuntime
): ModelProgressSnapshot {
  const checkpoint = runtime.checkpoint;
  return {
    worldRevision: checkpoint.world.worldRevision,
    cycleIndex: checkpoint.cycle_index,
    checkerSuccess: checkpoint.checker?.success ?? false,
    receipts: Object.values(checkpoint.committed_actions).map((receipt) => ({
      transactionId: receipt.transactionId,
      agentId: receipt.agentId,
      action: receipt.action,
      accepted: receipt.accepted,
      code: receipt.code,
      worldBeforeRevision: receipt.worldBeforeRevision,
      worldAfterRevision: receipt.worldAfterRevision,
      frameCount: receipt.frameCount
    }))
  };
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
  const goalDirection = checkpoint.goal_dag.status === "awaiting_model_selection"
    ? "当前没有 active Goal；先委派目标管理智能体提交 2–3 个真实模型候选并由其另一次响应显式选择，然后再规划物理行动。"
    : "只围绕 goal_dag 当前 epoch 的 active Goal 规划物理行动。";
  return [
    "继续下一次人形自主闭环。",
    "必须根据当前传感、目标、历史回执和物理反馈自主决定动作；不得使用固定动作表、预设路径或假执行。",
    goalDirection,
    runtime.store.definition.run_mode === "mission"
      ? "运行模式：有限任务。只有精确完成 mission_goal 才会结束；阶段子目标完成后若长期条件尚未达成，必须继续选择下一 Goal。"
      : "运行模式：持续自主。完成当前 Goal 后继续从新的物理观察中选择下一 Goal，直到外部明确暂停。",
    `任务：${runtime.store.definition.mission}`,
    `当前循环：${checkpoint.cycle_index + 1}`,
    ...(checkpoint.active_cycle
      ? [`循环身份：${checkpoint.active_cycle.cycle_id}`]
      : []),
    `当前世界：frame=${checkpoint.world.frame}, revision=${checkpoint.world.worldRevision}`,
    "完成一次真实物理执行后，用 accepted 执行回执调用 complete_autonomous_cycle。"
  ].join("\n");
}

function assertCycleOutput(output: string | undefined): {
  status: "cycle_completed" | "goal_transition_completed";
} {
  if (!output?.trim()) throw new Error("Humanoid coordinator returned no cycle output");
  let parsed: unknown;
  try {
    parsed = JSON.parse(output);
  } catch {
    throw new Error("Humanoid coordinator did not return the cycle completion tool result");
  }
  const status = parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
    ? (parsed as Record<string, unknown>).status
    : undefined;
  if (status !== "cycle_completed" && status !== "goal_transition_completed") {
    throw new Error("Humanoid coordinator did not complete a verified runtime transition");
  }
  return { status };
}

function humanoidCheckpointFingerprint(checkpoint: {
  world: { frame: number; worldRevision: number };
  goal_dag: { state_sha256: string };
  goal_progress: {
    goal_sha256: string;
    last_world_frame: number;
    last_world_revision: number;
    predicate_streaks: number[];
  } | null;
  committed_actions: Record<string, unknown>;
  context_memory: { total_compactions: number };
  cycle_index: number;
  active_cycle: { cycle_id: string } | null;
}): string {
  return createHash("sha256").update(JSON.stringify({
    frame: checkpoint.world.frame,
    worldRevision: checkpoint.world.worldRevision,
    goalDAGStateSha256: checkpoint.goal_dag.state_sha256,
    goalProgress: checkpoint.goal_progress,
    committedActions: Object.keys(checkpoint.committed_actions).sort(),
    totalCompactions: checkpoint.context_memory.total_compactions,
    cycleIndex: checkpoint.cycle_index,
    activeCycleId: checkpoint.active_cycle?.cycle_id ?? null
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
