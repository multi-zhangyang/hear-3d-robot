import {
  OutputGuardrailTripwireTriggered,
  RunContext,
  Runner,
  RunState,
  setTracingDisabled,
  type RunStreamEvent
} from "@openai/agents";
import type { ProviderConfig, RuntimeCatalog } from "../config/load.js";
import type { Goal } from "../domain/schema.js";
import {
  capabilityCatalog,
  createAgentHierarchy,
  ModelDecisionStallError,
  type HarnessAgentContext
} from "../harness/agents.js";
import { HierarchyProjection } from "../harness/hierarchy-projection.js";
import { hierarchyNeedsEvidenceContractRotation } from "../harness/evidence-contract.js";
import { LongRunContextManager } from "../harness/context-compaction.js";
import { AgentsSdkContextSummaryGenerator } from "../harness/context-summary-agent.js";
import {
  createCheckpoint,
  HarnessRuntimeContext,
  type RuntimeEventSink
} from "../harness/runtime-context.js";
import { providerEventJson, sdkEventJson } from "../harness/sdk-events.js";
import { createConfiguredModel, providerIdentity } from "../model/factory.js";
import { FileSession } from "../persistence/file-session.js";
import { RunStore } from "../persistence/run-store.js";
import { RapierWorld } from "../world/rapier-world.js";
import { drawSeed } from "../world/world-generator.js";
import { assertGoalSupported } from "./goal-validation.js";
import { errorMessage } from "./error-message.js";
import {
  canReplayInitialModelRequest,
  isTransportInterruption
} from "./transport-recovery.js";

setTracingDisabled(true);

/**
 * How many times a mission may continue through a broken connection. High
 * enough that a flaky link or a throttling window does not decide the outcome,
 * bounded so a provider that is simply down ends the run instead of looping.
 */
const MAX_TRANSPORT_RECOVERIES = 8;
const MAX_MODEL_DECISION_RECOVERIES = 3;

/** Exponential backoff, capped, so a throttled provider is given room to recover. */
function transportBackoffMs(attempt: number): number {
  return Math.min(2_000 * 2 ** (attempt - 1), 30_000);
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

export interface MissionRunResult {
  runId: string;
  runDir: string;
  output: string;
}

export async function startMission(input: {
  runsDir: string;
  mission: string;
  scenarioId: string;
  goal: Goal;
  catalog: RuntimeCatalog;
  provider: ProviderConfig;
  seed?: number;
  eventSink?: RuntimeEventSink;
  signal?: AbortSignal;
}): Promise<MissionRunResult> {
  const worldSeed = input.seed ?? drawSeed();
  // Movement entropy is fresh even when a caller fixes the world seed. It only
  // changes the order of live, reachable choices offered to the model; it never
  // executes a choice. Persisting it in the materialized scenario makes resume
  // continue the same autonomy stream.
  const scenario = input.catalog.materialize(input.scenarioId, worldSeed, drawSeed());
  assertGoalSupported(input.goal, scenario);
  const store = await RunStore.create(input.runsDir, {
    mission: input.mission,
    scenarioId: input.scenarioId,
    scenario,
    goal: input.goal
  });
  const world = await RapierWorld.create(scenario);
  try {
    const capabilities = capabilityCatalog();
    const hierarchy = HierarchyProjection.create(
      input.mission,
      capabilities,
      input.goal.predicates.length
    );
    const checkpoint = createCheckpoint({
      store,
      hierarchy,
      capabilityCatalog: capabilities,
      world
    });
    await store.writeCheckpoint(checkpoint);
    const runtime = new HarnessRuntimeContext({
      store,
      goal: input.goal,
      world,
      hierarchy,
      checkpoint,
      ...(input.eventSink ? { eventSink: input.eventSink } : {}),
      ...(input.signal ? { signal: input.signal } : {})
    });
    return await executeMission({
      runtime,
      provider: input.provider,
      resumed: false,
      ...(input.signal ? { signal: input.signal } : {})
    });
  } finally {
    world.dispose();
  }
}

export async function resumeMission(input: {
  runDir: string;
  catalog: RuntimeCatalog;
  provider: ProviderConfig;
  freshContext?: boolean;
  eventSink?: RuntimeEventSink;
  signal?: AbortSignal;
}): Promise<MissionRunResult> {
  const store = await RunStore.open(input.runDir);
  const checkpoint = await store.readCheckpoint();
  if (checkpoint.status === "succeeded") throw new Error("A succeeded run cannot be resumed");
  const scenario = store.definition.scenario;
  const world = await RapierWorld.create(scenario, checkpoint.world);
  try {
    assertSameCapabilities(checkpoint.capability_catalog, capabilityCatalog());
    const restoredNodes = structuredClone(checkpoint.nodes);
    const restoredRoot = restoredNodes[checkpoint.root_id];
    if (!restoredRoot) throw new Error("Checkpoint hierarchy root is missing");
    // Goal ownership was added after v3 checkpoints existed. The coordinator
    // always owns every final-state predicate, so restoring that invariant is
    // deterministic and does not assign any physical action to the model.
    restoredRoot.goal_predicate_indexes = checkpoint.goal.predicates.map((_, index) => index);
    const hierarchy = new HierarchyProjection(
      restoredNodes,
      checkpoint.root_id,
      checkpoint.active_agent_id,
      checkpoint.active_agent_ids
    );
    const agentState = await store.readAgentState();
    const evidenceContractUpgrade = hierarchyNeedsEvidenceContractRotation(
      restoredNodes,
      checkpoint.root_id
    );
    const freshContext = input.freshContext === true || evidenceContractUpgrade;
    const resumeSerializedState = !freshContext
      && agentState !== undefined
      && (checkpoint.status === "running" || checkpoint.status === "interrupted");
    if (!resumeSerializedState) hierarchy.reactivateRoot();
    const runtime = new HarnessRuntimeContext({
      store,
      goal: store.definition.goal,
      world,
      hierarchy,
      checkpoint: {
        ...checkpoint,
        active_agent_id: hierarchy.activeId,
        active_agent_ids: hierarchy.activeIds,
        nodes: hierarchy.snapshot()
      },
      ...(input.eventSink ? { eventSink: input.eventSink } : {}),
      ...(input.signal ? { signal: input.signal } : {})
    });
    return await executeMission({
      runtime,
      provider: input.provider,
      resumed: true,
      freshContext,
      ...(freshContext
        ? {
            freshContextSource: input.freshContext === true
              ? "operator_requested" as const
              : "evidence_contract_upgrade" as const
          }
        : {}),
      ...(resumeSerializedState ? { serializedState: agentState } : {}),
      ...(input.signal ? { signal: input.signal } : {})
    });
  } finally {
    world.dispose();
  }
}

async function executeMission(input: {
  runtime: HarnessRuntimeContext;
  provider: ProviderConfig;
  resumed: boolean;
  freshContext?: boolean;
  freshContextSource?: "operator_requested" | "evidence_contract_upgrade";
  serializedState?: string;
  signal?: AbortSignal;
}): Promise<MissionRunResult> {
  const sessions = new Map<string, FileSession>();
  const sessionForAgent = (agentId: string): FileSession => {
    const existing = sessions.get(agentId);
    if (existing) return existing;
    const created = new FileSession(
      agentId === input.runtime.rootAgentId
        ? input.runtime.store.sessionPath()
        : input.runtime.store.workerSessionPath(agentId),
      agentId === input.runtime.rootAgentId
        ? input.runtime.runId
        : `${input.runtime.runId}:${agentId}`
    );
    sessions.set(agentId, created);
    return created;
  };
  const contextManager = new LongRunContextManager({
    runtime: input.runtime,
    provider: input.provider,
    sessionForAgent,
    createGenerator: () => new AgentsSdkContextSummaryGenerator({
      model: createConfiguredModel(input.provider),
      temperature: input.provider.temperature,
      maxOutputTokens: input.provider.compactMaxOutputTokens
    })
  });
  const hierarchy = createAgentHierarchy({
    createModel: () => createConfiguredModel(input.provider),
    createSession: sessionForAgent,
    provider: input.provider,
    runtime: input.runtime,
    callModelInputFilter: contextManager.filter
  });
  const runner = new Runner({
    tracingDisabled: true,
    traceIncludeSensitiveData: false,
    modelSettings: { parallelToolCalls: true },
    callModelInputFilter: contextManager.filter,
    toolExecution: { maxFunctionToolConcurrency: 4 },
    toolNotFoundBehavior: "return_error_to_model",
    workflowName: "HEAR mission"
  });
  const session = sessionForAgent(input.runtime.rootAgentId);
  const runContext = new RunContext<HarnessAgentContext>({ runId: input.runtime.runId });

  try {
    if (input.freshContext) {
      contextManager.startFreshSdkTurn(input.runtime.rootAgentId);
      await Promise.all([
        session.clearSession(),
        input.runtime.store.clearWorkerSessions(),
        input.runtime.store.clearAgentState()
      ]);
    }
    await input.runtime.start(input.resumed);
    if (input.freshContext) {
      await input.runtime.recordProvider({
        status: "context_branch_rotated",
        source: input.freshContextSource ?? "operator_requested",
        cleared: ["sdk_run_state", "sdk_session"],
        preserved: ["world_checkpoint", "action_receipts", "context_journal", "compact_memory", "spatial_memory"],
        automatic_actuation: false
      }, input.runtime.rootAgentId);
    }
    await input.runtime.recordProvider({
      status: "configured",
      ...providerIdentity(input.provider)
    });

    let serializedState = input.serializedState;
    const initialRequestBaseline = !input.resumed && serializedState === undefined
      ? {
          checkpoint: input.runtime.checkpoint,
          sessionItems: await session.getItems()
        }
      : undefined;
    let recoveries = 0;
    let decisionRecoveries = 0;
    let decisionRecoveryPrompt: string | undefined;
    // A mission outlives any single HTTP connection, so a dropped socket is a
    // transport event, not a mission outcome. The agent state is written after
    // every stream event, which makes the last committed turn a real resume
    // point: reload it and continue the same conversation. Only failures that
    // would fail identically on replay — a rejected key, a malformed request —
    // fall through to the catch below.
    for (;;) {
      try {
        const runInput = decisionRecoveryPrompt ?? (serializedState
          ? await RunState.fromStringWithContext(
              hierarchy.root,
              serializedState,
              runContext,
              { contextStrategy: "replace" }
            )
          : missionInput(input.runtime, input.runtime.store.definition.mission, input.resumed));
        decisionRecoveryPrompt = undefined;

        const stream = await runner.run(hierarchy.root, runInput, {
          stream: true,
          context: runContext,
          session,
          maxTurns: null,
          toolExecution: { maxFunctionToolConcurrency: 4 },
          toolNotFoundBehavior: "return_error_to_model",
          ...(input.signal ? { signal: input.signal } : {})
        });

        for await (const event of stream) {
          await persistStreamEvent(input.runtime, stream, event);
        }
        await stream.completed;
        await contextManager.compactSessionHistories(sessionForAgent);
        await input.runtime.store.writeAgentState(stream.state.toString());
        // An aborted SDK stream can resolve `completed` without a final model
        // item. Preserve the process-signal reason instead of reading an
        // unavailable finalOutput and overwriting the checkpoint with a
        // misleading "returned no text" failure.
        input.signal?.throwIfAborted();

        const output = typeof stream.finalOutput === "string"
          ? stream.finalOutput
          : JSON.stringify(stream.finalOutput);
        if (!output?.trim()) throw new Error("Mission Coordinator returned no text");
        await input.runtime.succeed(output);
        return {
          runId: input.runtime.runId,
          runDir: input.runtime.store.runDir,
          output
        };
      } catch (error) {
        if (input.signal?.aborted) throw error;
        if ((error instanceof OutputGuardrailTripwireTriggered
            || (error instanceof ModelDecisionStallError
              && error.agentId === input.runtime.rootAgentId))
          && decisionRecoveries < MAX_MODEL_DECISION_RECOVERIES) {
          decisionRecoveries += 1;
          // A RunState reset alone still reuses the FileSession and can feed
          // the same non-decision branch straight back to the model. Rotate
          // that short-term branch while retaining the append-only context
          // journal, compact checkpoint, receipts and authoritative world.
          contextManager.startFreshSdkTurn(input.runtime.rootAgentId);
          await session.clearSession();
          await input.runtime.recordProvider({
            status: "model_decision_recovery",
            error: error.message,
            recovery_attempt: decisionRecoveries,
            maximum_recoveries: MAX_MODEL_DECISION_RECOVERIES,
            sdk_session_rotated: true
          }, input.runtime.rootAgentId);
          // Do not replay the failed RunState: it contains the blank outputs.
          // Start a new real model turn against the same Session, compacted
          // memory and authoritative world state. No action is selected here.
          serializedState = undefined;
          decisionRecoveryPrompt = modelDecisionRecoveryInput(
            input.runtime,
            input.runtime.store.definition.mission,
            decisionRecoveries
          );
          continue;
        }
        if (!isTransportInterruption(error)) throw error;
        const persisted = await input.runtime.store.readAgentState();
        if (recoveries >= MAX_TRANSPORT_RECOVERIES) throw error;

        // The SDK persists a streaming run's input before opening the HTTP
        // response. A brand-new mission can therefore have a durable Session
        // but no RunState when its first request gets no response. Retrying is
        // safe only while every authoritative surface is unchanged. Restore
        // the exact pre-request Session first so the opening prompt is not
        // duplicated. Once any tool, hierarchy transition, memory compaction,
        // model-written compaction, or physics state has landed, only a
        // serialized RunState may resume. The append-only record of the input
        // itself is safe and remains the same logical prefix on the retry.
        const retryingInitialRequest = persisted === undefined;
        if (retryingInitialRequest) {
          if (!initialRequestBaseline || !canReplayInitialModelRequest(
            initialRequestBaseline.checkpoint,
            input.runtime.checkpoint
          )) {
            throw error;
          }
          await session.replaceItems(initialRequestBaseline.sessionItems);
        }

        recoveries += 1;
        const waitMs = transportBackoffMs(recoveries);
        await input.runtime.recordProvider({
          status: "transport_interrupted",
          error: errorMessage(error),
          recovery_attempt: recoveries,
          retry_after_ms: waitMs,
          initial_request_retried: retryingInitialRequest
        });
        await delay(waitMs, input.signal);
        serializedState = persisted;
      }
    }
  } catch (error) {
    const message = errorMessage(error);
    const recoverableInterruption = input.signal?.aborted === true
      || isTransportInterruption(error);
    await input.runtime.recordProvider({
      status: recoverableInterruption ? "interrupted" : "provider_or_runtime_error",
      error: message
    });
    // Exhausting bounded transport retries means the provider is unavailable
    // for now, not that the embodied mission reached a failed state. Persist it
    // as interrupted so a later resume can restore the serialized SDK RunState
    // instead of discarding that real decision branch. Deterministic provider
    // and runtime errors still fail normally.
    if (recoverableInterruption) await input.runtime.interrupt(message);
    else await input.runtime.fail(error);
    throw error;
  }
}

async function persistStreamEvent(
  runtime: HarnessRuntimeContext,
  stream: { state: { toString: () => string } },
  event: RunStreamEvent
): Promise<void> {
  const frameworkEvent = sdkEventJson(event);
  const providerEvent = providerEventJson(event);
  if (!frameworkEvent && !providerEvent) return;
  await runtime.store.writeAgentState(stream.state.toString());
  if (frameworkEvent) {
    await runtime.recordFramework(
      runtime.frameworkScope(runtime.rootAgentId),
      frameworkEvent,
      runtime.rootAgentId
    );
  }
  if (providerEvent) await runtime.recordProvider(providerEvent, runtime.rootAgentId);
}

function missionInput(runtime: HarnessRuntimeContext, mission: string, resumed: boolean): string {
  return [
    resumed ? "Continue this interrupted mission from the current committed state." : "Execute this mission.",
    `Operator mission: ${mission}`,
    `Structured goal: ${JSON.stringify(runtime.goal())}`,
    "Structured goal predicate indexes are zero-based and are the only final-state ownership accepted for supervisors.",
    `Current source-backed world observation: ${JSON.stringify(runtime.worldObservation())}`,
    "Create capability-scoped child agents as needed, use their actual results, and finish only after the current physics state passes Checker."
  ].join("\n");
}

function modelDecisionRecoveryInput(
  runtime: HarnessRuntimeContext,
  mission: string,
  attempt: number
): string {
  return [
    `Begin a fresh model decision turn after ${attempt} sequence(s) of blank or non-tool output.`,
    "Call at least one available tool now; emit no prose before the tool call. Independent delegations may be emitted together, but lifecycle tools and dependent work remain single and sequential.",
    "The harness has not substituted any action and will not choose a target for you.",
    `Operator mission: ${mission}`,
    `Structured goal: ${JSON.stringify(runtime.goal())}`,
    "Structured goal predicate indexes are zero-based; every supervisory delegation must own at least one.",
    `Current source-backed authority: ${JSON.stringify(runtime.contextAnchor(runtime.rootAgentId))}`,
    "If the goal is incomplete, delegate a fresh capability-scoped model node from current evidence. If it may be complete, call Checker first."
  ].join("\n");
}

function assertSameCapabilities(persisted: string[], configured: string[]): void {
  if (persisted.length !== configured.length
    || persisted.some((capability, index) => capability !== configured[index])) {
    throw new Error("The action capability catalog changed; this run cannot be resumed safely");
  }
}
