import { randomUUID } from "node:crypto";
import { readdir, stat } from "node:fs/promises";
import { resolve } from "node:path";
import {
  Runner,
  RunState,
  UserError,
  setTracingDisabled,
  type RunStreamEvent
} from "@openai/agents";
import {
  DEFAULT_MODEL_STREAM_EVENT_IDLE_TIMEOUT_MS,
  providerConfigForRole,
  type ProviderConfig,
  type RuntimeCatalog
} from "../config/load.js";
import type { Goal } from "../domain/schema.js";
import { humanoidActionReceiptsInCommitOrder } from "../domain/humanoid-run.js";
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
  contextCompactionInterruptionAgentIdFrom,
  isRetryableContextCompactionInterruption,
  isContextCompactionInterruption
} from "../harness/context-summary-agent.js";
import {
  AgentManifestIncompatibleError,
  assertAgentManifestCompatible,
  createHumanoidAgentManifest
} from "../harness/agent-manifest.js";
import {
  coordinatorInvocationInput,
  createHumanoidAgentHierarchy,
  humanoidAgentRole,
  HUMANOID_AGENT_IDS
} from "../harness/humanoid/agents.js";
import { createHumanoidRunCheckpoint } from "../harness/humanoid/run-checkpoint.js";
import { HumanoidRunRuntime } from "../harness/humanoid/run-runtime.js";
import { DensePolicyRolloutWriter } from
  "../training/dense-policy-rollout-files.js";
import { assertHumanoidPhysicalWorldDeltaRecovery } from "../harness/humanoid/physical-world-delta.js";
import {
  ModelDecisionStallError,
  modelDecisionStallFrom,
  modelTransportInterruptionAgentIdFrom,
  withModelTelemetry
} from "../harness/model-telemetry.js";
import { providerEventJson, sdkEventJson } from "../harness/sdk-events.js";
import {
  createConfiguredModel,
  promptCacheAffinityKey,
  providerIdentity,
  type PromptCacheRequestTrace
} from "../model/factory.js";
import { agentsModelRetrySettings } from "../model/retry.js";
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
  captureHumanoidSessionBaseline,
  captureHumanoidSessionStateIdentity,
  humanoidAgentStateFingerprint,
  restoreHumanoidSessionBaseline,
  restoreHumanoidSessionStateBaselineDetailed,
  type HumanoidSessionBaseline
} from "./humanoid-agent-state.js";
import {
  PerAgentTransportRecovery,
  isTransportInterruption,
  transportStatusCode,
  transportRetryPlan
} from "./transport-recovery.js";
import { HumanoidWorld } from "../world/humanoid/world.js";
import type {
  HumanoidControllerSource
} from "../world/humanoid/controller-module.js";
import { drawSeed } from "../world/world-generator.js";

setTracingDisabled(true);

// The SDK owns retries for one model request. This smaller outer window only
// resumes a durable mission boundary after the SDK could not safely finish it.
const MAX_MISSION_CONTINUITY_RECOVERIES = 2;
const SERVER_ERROR_CONTEXT_RECOVERY_ATTEMPT = 2;
const HUMANOID_PROMPT_CACHE_NAMESPACE = "hear-humanoid-agent-profile-v1";

export interface HumanoidMissionRunResult {
  runId: string;
  runDir: string;
  output: string;
}

export function assertHumanoidControllerSourceCompatible(
  persistedSourceSha256: string | undefined,
  configuredSource: HumanoidControllerSource | undefined
): void {
  resolveHumanoidControllerSourceForRun(
    persistedSourceSha256,
    configuredSource
  );
}

export function resolveHumanoidControllerSourceForRun(
  persistedSourceSha256: string | undefined,
  configuredSource: HumanoidControllerSource | undefined
): HumanoidControllerSource | undefined {
  if (persistedSourceSha256 === undefined) return undefined;
  const configuredSourceSha256 = configuredSource?.sourceSha256;
  if (persistedSourceSha256 === configuredSourceSha256) return configuredSource;
  if (configuredSourceSha256 === undefined) {
    throw new Error(
      "This run requires its original humanoid controller source"
    );
  }
  throw new Error(
    "The configured humanoid controller module does not match the source used by this run"
  );
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
  controllerSource?: HumanoidControllerSource;
  densePolicyRolloutDir?: string;
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
    runMode: input.runMode ?? "mission",
    ...(input.controllerSource
      ? { controllerSourceSha256: input.controllerSource.sourceSha256 }
      : {})
  }, input.mutationFence ? { mutationFence: input.mutationFence } : {});
  const scenarioChunks = await store.readScenarioChunkDeltaState();
  const world = await HumanoidWorld.create(scenario, undefined, {
    scenarioChunks,
    ...(input.controllerSource
      ? { controllerFactory: input.controllerSource.controllerFactory }
      : {})
  });
  const densePolicyWriter = input.densePolicyRolloutDir
    ? new DensePolicyRolloutWriter({
        rootDir: input.densePolicyRolloutDir,
        runId: store.definition.run_id
      })
    : undefined;
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
      ...(densePolicyWriter
        ? { policyFrameSink: densePolicyWriter.recordFrame }
        : {}),
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
    await Promise.all([
      densePolicyWriter?.flush() ?? Promise.resolve(),
      world.dispose()
    ]);
  }
}

export async function resumeHumanoidMission(input: {
  runDir: string;
  catalog: RuntimeCatalog;
  provider: ProviderConfig;
  freshAgentEpoch?: boolean;
  eventSink?: RuntimeEventSink;
  signal?: AbortSignal;
  mutationFence?: MutationFence;
  controllerSource?: HumanoidControllerSource;
  densePolicyRolloutDir?: string;
}): Promise<HumanoidMissionRunResult> {
  const store = await RunStore.open(
    input.runDir,
    input.mutationFence ? { mutationFence: input.mutationFence } : {}
  );
  if (store.definition.runtime !== "humanoid_g1") {
    throw new Error("This run was not created by the humanoid runtime");
  }
  const controllerSource = resolveHumanoidControllerSourceForRun(
    store.definition.controller_source_sha256,
    input.controllerSource
  );
  const checkpoint = await store.readHumanoidCheckpoint();
  if (checkpoint.status === "succeeded") throw new Error("A succeeded run cannot be resumed");
  if (input.freshAgentEpoch) {
    await store.archiveCurrentAgentEpoch();
  }
  const scenarioChunks = await store.readScenarioChunkDeltaState();
  assertHumanoidPhysicalWorldDeltaRecovery({
    scenario: store.definition.scenario,
    chunks: scenarioChunks,
    world: checkpoint.world
  });
  const world = await HumanoidWorld.create(
    store.definition.scenario,
    checkpoint.world_checkpoint,
    {
      scenarioChunks,
      ...(controllerSource
        ? { controllerFactory: controllerSource.controllerFactory }
        : {})
    }
  );
  const densePolicyWriter = input.densePolicyRolloutDir
    ? new DensePolicyRolloutWriter({
        rootDir: input.densePolicyRolloutDir,
        runId: store.definition.run_id
      })
    : undefined;
  try {
    const runtime = new HumanoidRunRuntime({
      store,
      goal: store.definition.goal,
      world,
      checkpoint,
      ...(densePolicyWriter
        ? { policyFrameSink: densePolicyWriter.recordFrame }
        : {}),
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
    await Promise.all([
      densePolicyWriter?.flush() ?? Promise.resolve(),
      world.dispose()
    ]);
  }
}

async function executeHumanoidMission(input: {
  runtime: HumanoidRunRuntime;
  provider: ProviderConfig;
  resumed: boolean;
  signal?: AbortSignal;
}): Promise<HumanoidMissionRunResult> {
  const transportRecovery = new PerAgentTransportRecovery(
    MAX_MISSION_CONTINUITY_RECOVERIES
  );
  let modelProgressRecoveryEpoch = 0;
  let initialized = false;
  try {
    const persistedManifest = await persistedManifestForMission(input);
    const manifestEpochId = persistedManifest?.epoch_id ?? randomUUID();
    const promptCacheKeyFor = (
      agentId: string,
      provider: ReturnType<typeof providerConfigForRole>
    ): string => promptCacheAffinityKey({
      namespace: HUMANOID_PROMPT_CACHE_NAMESPACE,
      agentId,
      provider
    });
    const sessions = new Map<string, FileSession>();
    let modelRequestSessionBaseline: HumanoidSessionBaseline | undefined;
    let activeContextManager: LongRunContextManager | undefined;
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
    const onModelResponseCompleted = async (
      agentId: string,
      usage?: { inputTokens: number }
    ): Promise<void> => {
      if (usage) {
        await activeContextManager?.recordModelInputUsage(agentId, usage.inputTokens);
      }
      const recovered = transportRecovery.responseCompleted(agentId);
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
      createGenerator: (agentId) => new AgentsSdkContextSummaryGenerator({
        model: createConfiguredModel(compactorProvider, {
          promptCacheKey: promptCacheKeyFor(
            `humanoid-context-compactor:${agentId}`,
            compactorProvider
          ),
          onPromptCacheRequest: (event) => recordPromptCacheRequest(
            input.runtime,
            agentId,
            event
          ),
          onPromptCacheStatus: (event) => input.runtime.recordProvider({
            status: "prompt_cache_unsupported",
            source: "protocol_capability_negotiation",
            compatibility_retry: event.compatibilityRetry,
            automatic_actuation: false
          }, agentId)
        }),
        temperature: compactorProvider.temperature,
        ...(compactorProvider.reasoningEffort === undefined
          ? {}
          : { reasoningEffort: compactorProvider.reasoningEffort }),
        ...(compactorOutputLimit === undefined
          ? {}
          : { maxOutputTokens: compactorOutputLimit }),
        onModelResponseCompleted: () => onModelResponseCompleted(agentId),
        onAttemptFailure: (event) => input.runtime.recordProvider({
          status: "context_compaction_attempt_failed",
          source: "agents_sdk",
          attempt: event.attempt,
          error: event.failure,
          automatic_actuation: false
        }, agentId)
      })
    });
    activeContextManager = contextManager;
    const resumedWithFreshAgentEpoch = input.resumed
      && !(await hasPersistedAgentRuntimeState(input.runtime.store))
      && Object.keys(contextManager.snapshot.scopes).length > 0;
    if (resumedWithFreshAgentEpoch) {
      for (const agentId of Object.values(HUMANOID_AGENT_IDS)) {
        contextManager.acceptSdkSessionRollback(agentId);
      }
    }
    const acceptRestoredSessions = (
      agentIds: readonly string[],
      baseline: HumanoidSessionBaseline
    ): void => {
      for (const agentId of agentIds) {
        contextManager.acceptSdkSessionRollback(agentId, baseline.get(agentId));
      }
    };
    const modelTelemetryRuntime: ModelTelemetryRuntime = {
      rootAgentId: input.runtime.rootAgentId,
      activeNode: (agentId) => input.runtime.activeNode(agentId),
      recordModelCallStarted: (agentId) => input.runtime.recordModelCallStarted(agentId),
      recordModelCallCompleted: (record) => input.runtime.recordModelCallCompleted(record),
      recordModelCallFailed: (modelCallId, agentId) => (
        input.runtime.recordModelCallFailed(modelCallId, agentId)
      ),
      modelProgressSnapshot: (agentId) => humanoidModelProgressSnapshot(
        input.runtime,
        agentId
      ),
      modelProgressRecoveryEpoch: () => modelProgressRecoveryEpoch
    };
    const persistAgentEvent = async (
      agentId: string,
      event: RunStreamEvent
    ): Promise<void> => {
      await input.runtime.setActiveAgent(agentId);
      await persistStreamEvent(input.runtime, agentId, event);
      if (isModelResponseStarted(event)) {
        modelRequestSessionBaseline = await captureHumanoidSessionBaseline(sessions);
      }
    };
    const hierarchy = createHumanoidAgentHierarchy({
      createModel: (agentId, provider) => withModelTelemetry(
        createConfiguredModel(provider, {
          promptCacheKey: promptCacheKeyFor(agentId, provider),
          onPromptCacheRequest: (event) => recordPromptCacheRequest(
            input.runtime,
            agentId,
            event
          ),
          onPromptCacheStatus: (event) => input.runtime.recordProvider({
            status: "prompt_cache_unsupported",
            source: "protocol_capability_negotiation",
            compatibility_retry: event.compatibilityRetry,
            automatic_actuation: false
          }, agentId)
        }),
        modelTelemetryRuntime,
        agentId,
        onModelResponseCompleted,
        provider.streamEventIdleTimeoutMs
          ?? DEFAULT_MODEL_STREAM_EVENT_IDLE_TIMEOUT_MS,
        provider.requestTimeoutMs
      ),
      createSession: sessionForAgent,
      callModelInputFilter: contextManager.filter,
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
      modelSettings: {
        parallelToolCalls: false,
        retry: agentsModelRetrySettings()
      },
      callModelInputFilter: contextManager.filter,
      toolExecution: { maxFunctionToolConcurrency: 1 },
      toolNotFoundBehavior: "return_error_to_model",
      // Provider reasoning ids are response-local on OpenAI-compatible chat
      // transports. Omitting them from durable history preserves exact
      // append-only prefixes across coordinator turns and SDK Session replay.
      reasoningItemIdPolicy: "omit",
      workflowName: "HEAR humanoid autonomy"
    });
    const coordinatorSession = hierarchy.coordinatorSession as FileSession;

    if (persistedManifest) {
      assertAgentManifestCompatible(persistedManifest, currentManifest);
    } else {
      await input.runtime.store.writeAgentManifest(currentManifest);
    }
    if (resumedWithFreshAgentEpoch) {
      await input.runtime.recordProvider({
        status: "agent_epoch_started",
        source: "operator_configuration",
        previous_session_history_attached: false,
        context_checkpoint_preserved: true,
        physical_authority_preserved: true,
        agent_manifest_epoch: currentManifest.epoch_id,
        automatic_actuation: false
      }, input.runtime.rootAgentId);
    }
    await input.runtime.initializeGoalAutonomy(
      persistedManifest ?? currentManifest
    );
    const initialResumableState = await resumableAgentState(input.runtime, sessions);
    let serializedState = initialResumableState?.state;
    if (initialResumableState && initialResumableState.restored.size > 0) {
      acceptRestoredSessions(
        [...initialResumableState.restored.keys()],
        initialResumableState.restored
      );
    }
    await input.runtime.start(input.resumed);
    const controller = input.runtime.snapshot().robot.controller;
    await input.runtime.recordProvider({
      status: "configured",
      ...providerIdentity(coordinatorProvider),
      hierarchy: "three_reasoning_agents_plus_deterministic_grounding_and_execution_services",
      prompt_cache_affinity: "stable_per_credential_agent_protocol_native",
      agent_manifest_epoch: currentManifest.epoch_id,
      agent_profiles: Object.fromEntries(Object.entries(currentManifest.agents).map(
        ([role, profile]) => [role, profile.execution_kind === "deterministic_service"
          ? {
              agent_id: profile.agent_id,
              execution_kind: profile.execution_kind,
              implementation_contract: profile.implementation_contract,
              decision_authority_role: profile.decision_authority_role
            }
          : {
              agent_id: profile.agent_id,
              execution_kind: "model",
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
      actuation: controller.actuation,
      ...(input.runtime.store.definition.controller_source_sha256
        ? {
            controller_source_sha256:
              input.runtime.store.definition.controller_source_sha256
          }
        : {})
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

    const acceptVerifiedTransition = async (
      output: string
    ): Promise<HumanoidMissionRunResult | undefined> => {
      const completion = assertCoordinatorStepOutput(output);
      if (completion.status === "step_completed") return undefined;
      await contextManager.commitPendingSessionRewrites(sessionForAgent);
      if (completion.status === "cycle_completed") {
        const activeGoalCompleted = await input.runtime.completeCycle(output);
        if (input.runtime.checkpoint.status === "succeeded") {
          return {
            runId: input.runtime.runId,
            runDir: input.runtime.store.runDir,
            output: input.runtime.checkpoint.final_output ?? output
          };
        }
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
      } else if (completion.status === "satisfied_goal_completed") {
        const activeGoalCompleted = await input.runtime.completeSatisfiedGoal(output);
        if (input.runtime.checkpoint.status === "succeeded") {
          return {
            runId: input.runtime.runId,
            runDir: input.runtime.store.runDir,
            output: input.runtime.checkpoint.final_output ?? output
          };
        }
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
      }
      return undefined;
    };

    const contextCompactionRecoveries = new Map<string, {
      compactionCount: number;
      attempt: number;
    }>();
    for (;;) {
      input.signal?.throwIfAborted();
      await input.runtime.ensureAutonomousCycle();
      const sessionBaseline = await captureHumanoidSessionBaseline(sessions);
      modelRequestSessionBaseline = sessionBaseline;
      try {
        let runInput: string | RunState<unknown, typeof hierarchy.coordinator>;
        if (serializedState) {
          try {
            runInput = await RunState.fromString(
              hierarchy.coordinator,
              serializedState
            );
          } catch (error) {
            const unavailableTool = recoverableDynamicToolRunStateError(
              error,
              hierarchy.coordinator.tools.map((tool) => tool.name)
            );
            if (!unavailableTool) throw error;
            await input.runtime.store.clearAgentState();
            serializedState = undefined;
            await input.runtime.recordProvider({
              status: "sdk_state_rebased",
              source: "dynamic_tool_capability_refresh",
              unavailable_tool: unavailableTool,
              session_history_preserved: true,
              physical_checkpoint_preserved: true,
              automatic_actuation: false
            }, input.runtime.rootAgentId);
            continue;
          }
        } else runInput = autonomousCycleInput(input.runtime);
        serializedState = undefined;
        const stream = await runner.run(hierarchy.coordinator, runInput, {
          stream: true,
          session: coordinatorSession,
          maxTurns: null,
          reasoningItemIdPolicy: "omit",
          toolExecution: { maxFunctionToolConcurrency: 1 },
          toolNotFoundBehavior: "return_error_to_model",
          ...(input.signal ? { signal: input.signal } : {})
        });
        for await (const event of stream) {
          await persistAgentEvent(HUMANOID_AGENT_IDS.coordinator, event);
          if (!shouldPersistHumanoidAgentState(event)) continue;
          const persistedSessionBaseline = await captureHumanoidSessionStateIdentity(
            sessions
          );
          await input.runtime.store.writeAgentState(
            stream.state.toString(),
            humanoidAgentStateFingerprint(input.runtime.checkpoint),
            persistedSessionBaseline
          );
        }
        await stream.completed;
        const finalSessionBaseline = await captureHumanoidSessionStateIdentity(sessions);
        await input.runtime.store.writeAgentState(
          stream.state.toString(),
          humanoidAgentStateFingerprint(input.runtime.checkpoint),
          finalSessionBaseline
        );
        input.signal?.throwIfAborted();
        const output = typeof stream.finalOutput === "string"
          ? stream.finalOutput
          : JSON.stringify(stream.finalOutput);
        const result = await acceptVerifiedTransition(output);
        if (result) return result;
        await input.runtime.store.clearAgentState();
        await input.runtime.setActiveAgent(input.runtime.rootAgentId);
      } catch (error) {
        if (input.signal?.aborted) throw error;
        const decisionStall = modelDecisionStallFrom(error);
        const pendingPhysicalTransactionId = decisionStall
          ? input.runtime.pendingPhysicalExecutionTransactionId()
          : undefined;
        if (decisionStall && pendingPhysicalTransactionId) {
          const receipt = await input.runtime.recoverPendingPhysicalExecution();
          if (!receipt || receipt.transactionId !== pendingPhysicalTransactionId) {
            throw new Error(
              `Physical execution recovery changed transaction identity: ${pendingPhysicalTransactionId}`
            );
          }
          const decisionBaseline = modelRequestSessionBaseline ?? sessionBaseline;
          const restoredAgentIds = await restoreHumanoidSessionBaseline(
            sessions,
            decisionBaseline
          );
          acceptRestoredSessions(restoredAgentIds, decisionBaseline);
          await input.runtime.store.clearAgentState();
          serializedState = undefined;
          await input.runtime.recordProvider({
            status: "physical_execution_recovered_before_model_follow_up",
            transaction_id: receipt.transactionId,
            action: receipt.action,
            accepted: receipt.accepted,
            code: receipt.code,
            world_revision: receipt.worldAfterRevision,
            session_branch_rolled_back: restoredAgentIds.length > 0,
            automatic_actuation: false
          }, receipt.agentId);
          await input.runtime.setActiveAgent(input.runtime.rootAgentId);
          continue;
        }
        if (isRetryableContextCompactionInterruption(error)
          && !isTransportInterruption(error)) {
          const interruptedAgentId = contextCompactionInterruptionAgentIdFrom(error)
            ?? input.runtime.rootAgentId;
          const compactionCount = contextManager.snapshot.scopes[interruptedAgentId]
            ?.compaction_count ?? 0;
          const previousRecovery = contextCompactionRecoveries.get(interruptedAgentId);
          const recoveryAttempt = previousRecovery?.compactionCount === compactionCount
            ? previousRecovery.attempt + 1
            : 1;
          contextCompactionRecoveries.set(interruptedAgentId, {
            compactionCount,
            attempt: recoveryAttempt
          });
          const persisted = await resumableAgentState(input.runtime, sessions);
          let restoredAgentIds: string[] = [];
          if (persisted === undefined) {
            restoredAgentIds = await restoreHumanoidSessionBaseline(
              sessions,
              modelRequestSessionBaseline ?? sessionBaseline
            );
            acceptRestoredSessions(
              restoredAgentIds,
              modelRequestSessionBaseline ?? sessionBaseline
            );
          } else if (persisted.restored.size > 0) {
            acceptRestoredSessions([...persisted.restored.keys()], persisted.restored);
          }
          serializedState = persisted?.state;
          const retry = transportRetryPlan(error, recoveryAttempt);
          await input.runtime.recordProvider({
            status: "context_compaction_recovery_scheduled",
            source: "long_run_context_lifecycle",
            recovery_attempt: recoveryAttempt,
            retry_after_ms: retry.waitMs,
            resumed_sdk_state: serializedState !== undefined,
            raw_history_preserved: true,
            session_history_preserved: true,
            automatic_actuation: false
          }, interruptedAgentId);
          await delay(retry.waitMs, input.signal);
          continue;
        }
        if (!isTransportInterruption(error)) throw error;
        modelProgressRecoveryEpoch += 1;
        const interruptedAgentId = modelTransportInterruptionAgentIdFrom(error)
          ?? input.runtime.rootAgentId;
        const persisted = await resumableAgentState(input.runtime, sessions);
        let restoredAgentIds: string[] = [];
        if (persisted === undefined) {
          restoredAgentIds = await restoreHumanoidSessionBaseline(
            sessions,
            modelRequestSessionBaseline ?? sessionBaseline
          );
        } else if (persisted.restored.size > 0) {
          acceptRestoredSessions([...persisted.restored.keys()], persisted.restored);
        }
        acceptRestoredSessions(
          restoredAgentIds,
          modelRequestSessionBaseline ?? sessionBaseline
        );
        serializedState = persisted?.state;
        const recoveryAttempt = transportRecovery.nextAttempt(interruptedAgentId);
        if (recoveryAttempt === null) throw error;
        const retry = transportRetryPlan(error, recoveryAttempt);
        const statusCode = transportStatusCode(error);
        const rebaseSdkBranch = recoveryAttempt === SERVER_ERROR_CONTEXT_RECOVERY_ATTEMPT
          && statusCode !== undefined
          && statusCode >= 500
          && statusCode <= 599;
        if (rebaseSdkBranch) {
          restoredAgentIds = await restoreHumanoidSessionBaseline(
            sessions,
            modelRequestSessionBaseline ?? sessionBaseline
          );
          acceptRestoredSessions(
            restoredAgentIds,
            modelRequestSessionBaseline ?? sessionBaseline
          );
          await input.runtime.store.clearAgentState();
          serializedState = undefined;
        }
        await input.runtime.recordProvider({
          status: "transport_interrupted",
          recovery_attempt: recoveryAttempt,
          maximum_recoveries: MAX_MISSION_CONTINUITY_RECOVERIES,
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
          ...(rebaseSdkBranch
            ? {
                sdk_branch_rebased: true,
                raw_history_preserved: true,
                session_history_preserved: true,
                prompt_cache_prefix_preserved: true,
                rebase_reason: "repeated_server_error_for_identical_sdk_branch",
                recovery_baseline: "latest_model_request_prefix"
              }
            : {}),
          error: errorMessage(error),
          automatic_actuation: false
        }, interruptedAgentId);
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
      || error instanceof ModelDecisionStallError
      || isTransportInterruption(error)) {
      await input.runtime.interrupt(message);
    } else {
      await input.runtime.fail(message);
    }
    throw error;
  }
}

export function shouldPersistHumanoidAgentState(event: RunStreamEvent): boolean {
  if (event.type === "agent_updated_stream_event"
    || event.type === "run_item_stream_event") return true;
  return event.data.type === "response_started"
    || event.data.type === "response_done";
}

export function recoverableDynamicToolRunStateError(
  error: unknown,
  configuredToolNames: readonly string[]
): string | undefined {
  if (!(error instanceof UserError)) return undefined;
  const match = /^Tool (.+) not found$/u.exec(error.message);
  if (!match?.[1] || !configuredToolNames.includes(match[1])) return undefined;
  return match[1];
}

async function recordPromptCacheRequest(
  runtime: HumanoidRunRuntime,
  agentId: string,
  event: PromptCacheRequestTrace
): Promise<void> {
  await runtime.recordProvider({
    status: "prompt_cache_request",
    source: "protocol_request_fingerprint",
    request_sha256: event.requestSha256,
    message_count: event.messageCount,
    previous_message_count: event.previousMessageCount,
    common_message_prefix_count: event.commonMessagePrefixCount,
    common_message_prefix_bytes: event.commonMessagePrefixBytes,
    append_only_message_prefix: event.appendOnlyMessagePrefix,
    tool_count: event.toolCount,
    tools_stable: event.toolsStable,
    settings_stable: event.settingsStable,
    cache_affinity_present: event.cacheAffinityPresent,
    content_recorded: false,
    automatic_actuation: false
  }, agentId);
}

function isModelResponseStarted(event: RunStreamEvent): boolean {
  return event.type === "raw_model_stream_event"
    && event.data.type === "response_started";
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

export function humanoidModelProgressSnapshot(
  runtime: Pick<HumanoidRunRuntime, "checkpoint" | "rootAgentId">,
  agentId: string
): ModelProgressSnapshot {
  const checkpoint = runtime.checkpoint;
  const receipts = humanoidActionReceiptsInCommitOrder(
    checkpoint.committed_actions
  ).filter((receipt) => (
    agentId === runtime.rootAgentId || receipt.agentId === agentId
  ));
  return {
    worldRevision: checkpoint.world.worldRevision,
    cycleIndex: checkpoint.cycle_index,
    checkerSuccess: checkpoint.checker?.success ?? false,
    goalStateSha256: checkpoint.goal_dag.state_sha256,
    receipts: receipts.map((receipt) => ({
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

async function resumableAgentState(
  runtime: HumanoidRunRuntime,
  sessions: ReadonlyMap<string, FileSession>
): Promise<{
  state: string;
  restored: HumanoidSessionBaseline;
} | undefined> {
  const record = await runtime.store.readAgentStateRecord();
  if (!record) return undefined;
  if (record.sessionBaseline === undefined
    || record.checkpointFingerprint !== humanoidAgentStateFingerprint(runtime.checkpoint)) {
    await runtime.store.clearAgentState();
    return undefined;
  }
  const sessionRestore = await restoreHumanoidSessionStateBaselineDetailed(
    sessions,
    record.sessionBaseline
  );
  if (!sessionRestore.compatible) {
    await runtime.store.clearAgentState();
    return undefined;
  }
  return { state: record.state, restored: sessionRestore.restored };
}

function autonomousCycleInput(runtime: HumanoidRunRuntime): string {
  const checkpoint = runtime.checkpoint;
  const goalDirection = checkpoint.goal_dag.status === "awaiting_model_selection"
    ? "当前没有 active Goal；先委派目标管理智能体提交 2–3 个真实模型候选并由其另一次响应显式选择，然后再规划物理行动。"
    : "只围绕 goal_dag 当前 epoch 的 active Goal 规划物理行动。";
  return [
    "推进人形自主闭环的下一个层级步骤。每次只选择当前 coordinator_phase 对应的一项正式委派或完成动作；工具返回后由 Harness 重新观察阶段，再决定下一步。",
    "必须根据当前传感、目标、历史回执和物理反馈自主决定动作；不得使用固定动作表、预设路径或假执行。",
    goalDirection,
    runtime.store.definition.run_mode === "mission"
      ? "运行模式：有限任务。只有完整满足 mission_goal 的物理 predicates 才会结束；summary 不参与物理语义，阶段子目标完成后若长期条件尚未达成，必须继续选择下一 Goal。"
      : "运行模式：持续自主。完成当前 Goal 后继续从新的物理观察中选择下一 Goal，直到外部明确暂停。",
    `任务：${runtime.store.definition.mission}`,
    `当前循环：${checkpoint.cycle_index + 1}`,
    ...(checkpoint.active_cycle
      ? [`循环身份：${checkpoint.active_cycle.cycle_id}`]
      : []),
    "完成一次真实物理执行后，必须先委派感知哨兵重新观察；只有 cycle_completion.observed_after_execution=true 且 coordinator_phase=complete_cycle 时，才用 Harness 给出的 accepted 因果证据调用 complete_autonomous_cycle。若 coordinator_phase=complete_satisfied_goal，必须直接提交物理 checker 验证，禁止为了制造本周期执行回执而重复移动。",
    coordinatorInvocationInput(runtime.contextAnchor(runtime.rootAgentId))
  ].join("\n");
}

class CompletedResponseDecisionStallError extends ModelDecisionStallError {}

function assertCoordinatorStepOutput(output: string | undefined): {
  status: "cycle_completed" | "satisfied_goal_completed" | "step_completed";
} {
  if (!output?.trim()) {
    throw new CompletedResponseDecisionStallError(
      HUMANOID_AGENT_IDS.coordinator,
      "Humanoid coordinator returned no cycle output"
    );
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(output);
  } catch {
    throw new CompletedResponseDecisionStallError(
      HUMANOID_AGENT_IDS.coordinator,
      "Humanoid coordinator did not return a formal tool result"
    );
  }
  const record = parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
    ? parsed as Record<string, unknown>
    : undefined;
  const status = record?.status;
  if (status === "goal_candidate_selected"
    || status === "goal_epoch_retired"
    || status === "goal_epoch_continued") {
    return { status: "step_completed" };
  }
  if (typeof record?.transactionId === "string"
    && typeof record.action === "string"
    && typeof record.accepted === "boolean") {
    return { status: "step_completed" };
  }
  if (status !== "cycle_completed"
    && status !== "satisfied_goal_completed") {
    throw new CompletedResponseDecisionStallError(
      HUMANOID_AGENT_IDS.coordinator,
      "Humanoid coordinator did not complete a verified runtime transition"
    );
  }
  return { status };
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
