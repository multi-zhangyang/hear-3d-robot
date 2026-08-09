import { randomUUID } from "node:crypto";
import { Mutex } from "async-mutex";
import type {
  ContextCompactionSummary,
  ContextMemoryState,
  Goal,
  JsonValue,
  Scenario,
  TaskNode
} from "../../domain/schema.js";
import { goalSha256 } from "../../domain/goal-identity.js";
import {
  autonomousCycleRef,
  createActiveAutonomousCycle,
  sameAutonomousCycle,
  type ActiveAutonomousCycle,
  type AutonomousCycleRef
} from "../../domain/autonomous-cycle.js";
import {
  completeGoalEpoch,
  goalCandidateBySequence,
  goalCandidateSequence,
  proposeGoalCandidate,
  restoreGoalDAG,
  selectGoalCandidate as selectDomainGoalCandidate
} from "../../domain/goal-epoch.js";
import {
  retireGoalEpoch as retireDomainGoalEpoch
} from "../../domain/goal-epoch-retirement.js";
import {
  modelPayloadSha256,
  type ModelDecisionRef
} from "../../domain/model-call-authority.js";
import {
  addModelUsage,
  modelUsageDeltaFromProviderEvent,
  ModelUsageStateSchema,
  type ModelUsageState
} from "../../domain/model-usage.js";
import type { AgentManifest } from "../../domain/agent-manifest.js";
import {
  actionCommitPayloadSha256,
  stageActionCommit
} from "../../domain/action-commit-outbox.js";
import {
  acknowledgeTerminalActionExecution,
  actionExecutionFingerprintSha256,
  activeActionExecutions,
  terminalizeActionExecution
} from "../../domain/action-execution-ledger.js";
import {
  actionTransactionFingerprintSha256,
  createActionTransactionIdentity,
  rebuildActionTransactionIdentities,
  type ActionTransactionIdentity
} from "../../domain/action-transaction-identity.js";
import {
  ScenarioBlockRemovalTransactionSchema,
  type ScenarioBlockRemovalTransaction
} from "../../domain/scenario-block-removal.js";
import { materializeScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta.js";
import {
  HumanoidEmbodiedEpisodeSchema,
  HumanoidEmbodiedExperienceSchema,
  PersistedHumanoidActionReceiptSchema,
  type HumanoidEmbodiedEpisode,
  type HumanoidEmbodiedExperience,
  type HumanoidRunCheckpoint
} from "../../domain/humanoid-run.js";
import type { RunStore } from "../../persistence/run-store.js";
import {
  createLifecycleEvent,
  reconcileLifecycleOutbox
} from "../../persistence/lifecycle-outbox.js";
import { reconcileActionCommitOutbox } from "../../persistence/action-commit-reconciler.js";
import type { RuntimeEvent, RuntimeEventSink } from "../../runtime/events.js";
import { assertGoalSupported } from "../../runtime/goal-validation.js";
import {
  advanceHumanoidGoal,
  assertHumanoidGoalProgressIntegrity,
  createHumanoidGoalProgress,
  inspectHumanoidGoal
} from "../../runtime/humanoid-checker.js";
import { HumanoidPhysicsClock } from "../../world/humanoid/physics-clock.js";
import type { HumanoidWorld, HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { LongRunContextRuntime } from "../context-runtime.js";
import {
  HumanoidActionRuntime,
  humanoidActionFingerprint,
  type HumanoidActionInvocationOptions,
  type HumanoidActionToolCallAuthority,
  type HumanoidActionReceipt
} from "./runtime.js";
import {
  appendEmbodiedEpisode,
  rememberEmbodiedActionExperience,
  retainRecentActionReceipts
} from "./embodied-memory.js";
import { createHumanoidContextAnchor } from "./context-anchor.js";
import {
  recallHumanoidEmbodiedHistory,
  type HumanoidEmbodiedRecallRequest
} from "./embodied-recall.js";
import {
  recallGoalHistory as recallDurableGoalHistory,
  type GoalHistoryRecallRequest
} from "./goal-history.js";
import { reconcileAndCompactGoalHistory } from "./goal-history-store.js";
import {
  recoverableBlockedGoalActionReceipt,
  recoverableBlockedGoalEvidence
} from "./goal-retirement-evidence.js";
import {
  assertPendingActionReceipt,
  cycleSummary,
  embodiedActionJournalReceipt,
  json,
  object,
  physicalExecutionReceipt,
  previousCycleEvidence,
  requiresHumanoidClockPause
} from "./run-runtime-persistence.js";
import { reconcileHumanoidHierarchyCapabilities } from "./run-checkpoint.js";
import {
  GoalEvidenceArtifactSchema,
  createActionGoalEvidence,
  createGoalEvaluationEvidence,
  type GoalEvidenceArtifact
} from "./goal-evidence.js";
import { resolveHumanoidMissionCompletion } from "./mission-completion-evidence.js";
import type {
  GoalManagerRuntime,
  GoalToolCallAuthority
} from "./goal-manager-tools.js";
import {
  captureHumanoidPhysicalWorldDelta,
  projectHumanoidPhysicalWorldDelta,
  reconcileHumanoidPhysicalWorldDelta
} from "./physical-world-delta.js";
import {
  prepareAuthorizedBlockRemoval
} from "./block-removal-authority.js";
import {
  projectHumanoidBlockRemoval,
  reconcileHumanoidBlockRemoval
} from "./block-removal-commit.js";
import {
  resolveHumanoidCycleCompletionReadiness,
  validateHumanoidCycleCausalEvidence,
  type HumanoidCycleCausalEvidence,
  type HumanoidCycleCompletionReadiness
} from "./cycle-causal-evidence.js";
import { HumanoidModelAuthority } from "./model-authority.js";
import { HumanoidPhysicalExecutionRuntime } from "./physical-execution-runtime.js";
import {
  loadGoalEvidenceWorkingSet,
  loadModelAuthorityWorkingSet,
  optionalGoalEvidenceRefs,
  optionalModelCallIds,
  requiredGoalEvidenceRefs,
  requiredModelCallIds
} from "./autonomy-history-loader.js";

export type HumanoidCoordinatorPhase =
  | "goal_selection"
  | "goal_transition"
  | "complete_satisfied_goal"
  | "observe_or_plan"
  | "plan"
  | "replan_or_retire"
  | "execute_plan"
  | "post_execution"
  | "complete_cycle";

function observationHasReachableBasePlacements(
  receipt: HumanoidActionReceipt
): boolean {
  const detail = object(receipt.detail);
  const rawGeometry = detail?.manipulation_geometry;
  const geometry = rawGeometry === undefined ? undefined : object(rawGeometry);
  if (!Array.isArray(geometry?.objects)) return false;
  return geometry.objects.some((entry) => {
    const observedObject = object(entry);
    return Array.isArray(observedObject?.reachable_base_placements)
      && observedObject.reachable_base_placements.length > 0;
  });
}

export class HumanoidRunRuntime implements LongRunContextRuntime {
  readonly #store: RunStore;
  readonly #missionGoal: Goal;
  readonly #world: HumanoidWorld;
  readonly #eventSink: RuntimeEventSink;
  readonly #signal: AbortSignal | undefined;
  readonly #physicalExecution: HumanoidPhysicalExecutionRuntime;
  readonly #actions: HumanoidActionRuntime;
  readonly #physicsClock: HumanoidPhysicsClock;
  readonly #actionMutex = new Mutex();
  readonly #goalStateMutex = new Mutex();
  #checkpoint: HumanoidRunCheckpoint;
  #scenario: Scenario;
  #checkpointWriteTail: Promise<void> = Promise.resolve();
  #continuousPhysicsEnabled = false;
  #modelAuthority: HumanoidModelAuthority | undefined;
  #goalEvidence = new Map<string, GoalEvidenceArtifact>();
  #persistedGoalEvidenceRefs = new Set<string>();
  #contextGoalEvidenceRefs = new Map<string, string>();
  #actionTransactionIdentities = new Map<string, ActionTransactionIdentity>();
  #actionTransactionIdentitiesLoaded = false;
  #durableActionReceiptCache = new Map<string, HumanoidActionReceipt>();

  constructor(input: {
    store: RunStore;
    goal: Goal;
    world: HumanoidWorld;
    checkpoint: HumanoidRunCheckpoint;
    eventSink?: RuntimeEventSink;
    signal?: AbortSignal;
  }) {
    this.#store = input.store;
    this.#missionGoal = structuredClone(input.goal);
    this.#world = input.world;
    this.#checkpoint = reconcileHumanoidHierarchyCapabilities(input.checkpoint);
    this.#scenario = structuredClone(input.store.definition.scenario);
    if (goalSha256(this.#missionGoal) !== goalSha256(this.#checkpoint.mission_goal)) {
      throw new Error("Humanoid mission Goal does not match the persisted run constraint");
    }
    this.#assertActiveGoalProgress();
    this.#eventSink = input.eventSink ?? (() => undefined);
    this.#signal = input.signal;
    this.#physicalExecution = new HumanoidPhysicalExecutionRuntime({
      runId: this.runId,
      world: this.#world,
      checkpoint: () => this.#checkpoint,
      scenario: () => this.#scenario,
      activeGoal: () => this.#activeGoal(),
      requiredActiveCycle: () => this.#requiredActiveCycleRef(),
      persist: (refreshWorld) => this.#persist(refreshWorld),
      emitFrame: ({ world, checker, goalProgress, source }) => this.emit(
        "humanoid_world_frame",
        json({
          world,
          checker,
          goal_progress: goalProgress,
          frame_source: source
        }),
        randomUUID(),
        false
      ),
      ...(this.#signal ? { signal: this.#signal } : {})
    });
    this.#actions = new HumanoidActionRuntime(this.#world, {
      receipts: this.#checkpoint.committed_actions,
      state: this.#checkpoint.action_runtime_state,
      frameSink: (frame) => this.#physicalExecution.recordFrame(frame, "execution"),
      physicalFrameSink: (cut) => this.#physicalExecution.recordPhysicalCut(cut),
      receiptSink: (receipt) => this.#commitReceipt(receipt),
      beforePhysicalExecution: (intent) => this.#physicalExecution.admit(intent),
      receiptNormalizer: (receipt) => this.#physicalExecution.normalizeReceipt(receipt),
      prepareBlockRemoval: (request) => this.#prepareBlockRemoval(request),
      realtimeExecution: true,
      retainPhysicalTerminals: true,
      requireSkillBinding: true,
      activeGoal: () => this.#activeGoal(),
      ...(this.#signal ? { signal: this.#signal } : {})
    });
    this.#physicsClock = new HumanoidPhysicsClock({
      world: this.#world,
      frameSink: (frame) => this.#physicalExecution.recordFrame(frame, "stationary"),
      onError: async (error) => {
        await this.recordProvider({
          status: "continuous_physics_stopped",
          error: error instanceof Error ? error.message : String(error),
          automatic_actuation: false
        }, this.rootAgentId);
      }
    });
  }

  get runId(): string {
    return this.#checkpoint.run_id;
  }

  get rootAgentId(): string {
    return this.#checkpoint.root_id;
  }

  get store(): RunStore {
    return this.#store;
  }

  get signal(): AbortSignal | undefined {
    return this.#signal;
  }

  get checkpoint(): HumanoidRunCheckpoint {
    return structuredClone(this.#checkpoint);
  }

  async initializeGoalAutonomy(manifest: AgentManifest): Promise<void> {
    await this.#reconcileActionCommits();
    this.#scenario = materializeScenarioChunkDeltaState(
      this.#store.definition.scenario,
      await this.#store.readScenarioChunkDeltaState()
    );
    const goalEvidenceRefs = requiredGoalEvidenceRefs(this.#checkpoint);
    const optionalEvidenceRefs = optionalGoalEvidenceRefs(this.#checkpoint);
    const modelCallIds = requiredModelCallIds(this.#checkpoint);
    const optionalAuthorityIds = optionalModelCallIds(this.#checkpoint);
    const [
      evidence,
      rawModelCalls,
      rawActionIdentities,
      providerModelUsage,
      archivedManifests
    ] = await Promise.all([
      loadGoalEvidenceWorkingSet(
        this.#store,
        goalEvidenceRefs,
        optionalEvidenceRefs
      ),
      loadModelAuthorityWorkingSet(
        this.#store,
        modelCallIds,
        optionalAuthorityIds
      ),
      this.#store.readJournal("action_identities"),
      latestProviderModelUsage(this.#store),
      this.#store.readArchivedAgentManifests()
    ]);
    const actionTransactionIdentities = rebuildActionTransactionIdentities(
      rawActionIdentities,
      this.runId
    );
    this.#goalEvidence = evidence;
    this.#persistedGoalEvidenceRefs = new Set(evidence.keys());
    this.#modelAuthority = HumanoidModelAuthority.restore({
      manifest,
      archivedManifests,
      nodes: this.#checkpoint.nodes,
      records: rawModelCalls,
      appendRecord: (record) => this.#store.append("model_calls", json(record))
    });
    this.#actionTransactionIdentities = actionTransactionIdentities;
    this.#actionTransactionIdentitiesLoaded = true;
    if (providerModelUsage) {
      this.#checkpoint.model_usage = reconcileModelUsage(
        this.#checkpoint.model_usage,
        providerModelUsage
      );
    }
    this.#checkpoint.goal_dag = restoreGoalDAG(
      this.#checkpoint.goal_dag,
      this.#goalHarness()
    );
    this.#checkpoint.goal_dag = await reconcileAndCompactGoalHistory({
      store: this.#store,
      goalDAG: this.#checkpoint.goal_dag
    });
    for (const receipt of Object.values(this.#checkpoint.committed_actions)) {
      if (physicalExecutionReceipt(receipt) || receipt.action === "remove_world_block") {
        this.#rememberEmbodiedActionExperience(receipt, false);
      }
    }
    this.#assertActiveGoalProgress();
    this.#pruneRuntimeAuthority();
  }

  async submitGoalCandidates(
    input: Parameters<GoalManagerRuntime["submitGoalCandidates"]>[0],
    authority: GoalToolCallAuthority
  ): Promise<JsonValue> {
    return this.#goalStateMutex.runExclusive(async () => {
      this.#assertRunAcceptsDecisions();
      if (this.#checkpoint.goal_dag.status !== "awaiting_model_selection") {
        throw new Error("Goal candidates cannot be submitted while a Goal epoch is active");
      }
      const source = this.#goalModelSource(
        authority,
        "submit_goal_candidates",
        input
      );
      const evidence = this.#requiredContextGoalEvidence(source.agent_id);
      await this.#persistGoalEvidence([evidence.ref]);
      let next = this.#checkpoint.goal_dag;
      const candidateIds: string[] = [];
      const candidateReferences: Array<{
        candidate_sequence: number;
        proposal_id: string;
        candidate_id: string;
      }> = [];
      for (const candidate of input.candidates) {
        const before = new Set(Object.keys(next.candidates));
        try {
          assertGoalSupported(candidate.goal, this.#scenario);
          next = proposeGoalCandidate(next, {
            proposal_id: candidate.proposal_id,
            source,
            goal: candidate.goal,
            mission_link: candidate.mission_link,
            dependency_candidate_ids: candidate.dependency_candidate_ids,
            proposal_evidence_refs: [evidence.ref],
            created_world_revision: evidence.artifact.evidence.world_revision
          }, this.#goalHarness());
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(
            `Goal candidate ${JSON.stringify(candidate.proposal_id)} rejected: ${message}`,
            { cause: error }
          );
        }
        const created = Object.keys(next.candidates).find((id) => !before.has(id));
        if (!created) throw new Error("Goal candidate proposal produced no durable identity");
        candidateIds.push(created);
        candidateReferences.push({
          candidate_sequence: goalCandidateSequence(next, created)!,
          proposal_id: candidate.proposal_id,
          candidate_id: created
        });
      }
      this.#checkpoint.goal_dag = next;
      this.#checkpoint.goal_progress = null;
      this.#checkpoint.checker = null;
      await this.#persist();
      await this.#emitGoalState("candidates_submitted");
      return json({
        status: "goal_candidates_submitted",
        candidates: candidateReferences,
        candidate_ids: candidateIds,
        goal_dag_state_sha256: next.state_sha256
      });
    });
  }

  async selectGoalCandidate(
    input: Parameters<GoalManagerRuntime["selectGoalCandidate"]>[0],
    authority: GoalToolCallAuthority
  ): Promise<JsonValue> {
    return this.#goalStateMutex.runExclusive(() => this.#withPhysicsPaused(async () => {
      this.#assertRunAcceptsDecisions();
      const selectedBy = this.#goalModelSource(
        authority,
        "select_goal_candidate",
        input
      );
      const candidate = goalCandidateBySequence(
        this.#checkpoint.goal_dag,
        input.candidate_sequence
      );
      if (!candidate || candidate.status !== "proposed") {
        throw new Error(
          `Goal candidate sequence is unavailable: ${input.candidate_sequence}`
        );
      }
      if (candidate.source.model_call_id === selectedBy.model_call_id) {
        throw new Error("Goal selection requires a distinct model response after proposal");
      }
      const evidence = this.#requiredContextGoalEvidence(selectedBy.agent_id);
      await this.#persistGoalEvidence([evidence.ref]);
      const next = selectDomainGoalCandidate(this.#checkpoint.goal_dag, {
        candidate_id: candidate.candidate_id,
        selected_by: selectedBy,
        selection_evidence_refs: [evidence.ref],
        created_world_revision: evidence.artifact.evidence.world_revision
      }, this.#goalHarness());
      await this.#refreshWorldPersistenceState();
      this.#checkpoint.goal_dag = next;
      this.#checkpoint.goal_progress = createHumanoidGoalProgress(
        candidate.goal,
        this.#checkpoint.world
      );
      this.#checkpoint.checker = inspectHumanoidGoal(
        candidate.goal,
        this.#scenario,
        this.#checkpoint.world,
        this.#checkpoint.goal_progress
      );
      const cycle = this.#createActiveCycle();
      await this.#persist();
      await this.#emitGoalState("candidate_selected");
      await this.#emitCycleStarted(cycle);
      return json({
        status: "goal_candidate_selected",
        epoch_id: next.current_epoch_id,
        cycle_id: cycle.cycle_id,
        candidate_id: candidate.candidate_id,
        goal: candidate.goal,
        goal_dag_state_sha256: next.state_sha256
      });
    }));
  }

  async retireGoalEpoch(
    input: Parameters<GoalManagerRuntime["retireGoalEpoch"]>[0],
    authority: GoalToolCallAuthority
  ): Promise<JsonValue> {
    return this.#goalStateMutex.runExclusive(() => this.#withPhysicsPaused(async () => {
      this.#assertRunAcceptsDecisions();
      const retiredBy = this.#goalModelSource(
        authority,
        "retire_goal_epoch",
        input
      );
      await this.#persistGoalEvidence(input.evidence_refs);
      const artifacts = input.evidence_refs.map((ref) => this.#requiredGoalEvidence(ref));
      const revisions = new Set(artifacts.map((artifact) => artifact.evidence.world_revision));
      if (revisions.size !== 1) {
        throw new Error("Goal retirement evidence must belong to one world revision");
      }
      const recoverable = input.status === "blocked"
        ? recoverableBlockedGoalEvidence(artifacts)
        : null;
      if (recoverable) {
        throw new Error(
          `Goal remains physically recoverable: ${recoverable.receiptCode}; `
          + `${recoverable.recovery}; continue motion planning instead of marking it blocked`
        );
      }
      const resolvedWorldRevision = artifacts[0]!.evidence.world_revision;
      const activeEpochId = this.#checkpoint.goal_dag.current_epoch_id;
      const retiredGoalDAG = retireDomainGoalEpoch(this.#checkpoint.goal_dag, {
        status: input.status,
        retired_by: retiredBy,
        reason: input.reason,
        resolution_evidence_refs: input.evidence_refs,
        resolved_world_revision: resolvedWorldRevision
      }, this.#goalHarness());
      const next = await reconcileAndCompactGoalHistory({
        store: this.#store,
        goalDAG: retiredGoalDAG
      });
      await this.#refreshWorldPersistenceState();
      const interruptedCycle = this.#checkpoint.active_cycle;
      this.#checkpoint.goal_dag = next;
      this.#checkpoint.goal_progress = null;
      this.#checkpoint.checker = null;
      this.#checkpoint.active_cycle = null;
      this.#pruneRuntimeAuthority();
      await this.#persist();
      await this.#emitGoalState("epoch_retired");
      if (interruptedCycle) {
        await this.emit("autonomous_cycle_interrupted", json({
          cycle: interruptedCycle,
          reason: "goal_epoch_retired",
          retirement_status: input.status
        }));
      }
      return json({
        status: "goal_epoch_retired",
        epoch_id: activeEpochId,
        retirement_status: input.status,
        reason: input.reason,
        resolved_world_revision: resolvedWorldRevision,
        goal_dag_state_sha256: next.state_sha256
      });
    }));
  }

  validateGoalTransition(): JsonValue {
    const latest = this.#checkpoint.goal_dag.epochs.at(-1);
    if (this.#checkpoint.goal_dag.status !== "awaiting_model_selection"
      || this.#checkpoint.goal_dag.current_epoch_id !== null
      || !latest
      || latest.status === "completed"
      || latest.status === "active") {
      throw new Error("No evidence-backed Goal retirement is ready to complete");
    }
    return json({
      epoch_id: latest.epoch_id,
      candidate_id: latest.candidate_id,
      status: latest.status,
      reason: latest.retirement_reason,
      resolved_world_revision: latest.resolved_world_revision,
      evidence_refs: latest.physical_evidence_refs.resolution,
      goal_dag_state_sha256: this.#checkpoint.goal_dag.state_sha256
    });
  }

  validateSatisfiedGoal(): JsonValue {
    if (this.coordinatorPhase() !== "complete_satisfied_goal") {
      throw new Error("The active Goal is not ready for execution-free completion");
    }
    const activeGoal = this.#requiredActiveGoal();
    const activeCycle = this.#requiredActiveCycleRef();
    const world = this.#world.snapshot();
    const progress = this.#checkpoint.goal_progress;
    if (!progress) throw new Error("Active Goal progress is unavailable");
    const projected = world.frame === progress.last_world_frame
      && world.worldRevision === progress.last_world_revision
      ? {
          progress,
          checker: inspectHumanoidGoal(activeGoal, this.#scenario, world, progress)
        }
      : advanceHumanoidGoal(activeGoal, this.#scenario, world, progress);
    if (!projected.checker.success) {
      throw new Error("The active Goal is no longer physically satisfied");
    }
    return json({
      epoch_id: activeCycle.goal_epoch_id,
      cycle_id: activeCycle.cycle_id,
      goal: activeGoal,
      checker: projected.checker,
      physical_execution_required: false
    });
  }

  async ensureAutonomousCycle(): Promise<ActiveAutonomousCycle | undefined> {
    return this.#goalStateMutex.runExclusive(() => this.#withPhysicsPaused(async () => {
      this.#assertRunAcceptsDecisions();
      if (this.#checkpoint.goal_dag.status !== "active") return undefined;
      if (this.#checkpoint.active_cycle) {
        return structuredClone(this.#checkpoint.active_cycle);
      }
      await this.#refreshWorldPersistenceState();
      const cycle = this.#createActiveCycle();
      await this.#persist();
      await this.#emitCycleStarted(cycle);
      return structuredClone(cycle);
    }));
  }

  missionGoalCompleted(): boolean {
    return resolveHumanoidMissionCompletion(
      this.#checkpoint,
      this.#goalEvidence.values()
    ) !== null;
  }

  isActionAvailable(
    action: Parameters<HumanoidActionRuntime["invoke"]>[0],
    agentId: string
  ): boolean {
    return this.#actions.isActionAvailable(action, agentId);
  }

  async invoke(
    action: Parameters<HumanoidActionRuntime["invoke"]>[0],
    rawInput: unknown,
    transactionId: string,
    agentId: string,
    authority: HumanoidActionToolCallAuthority,
    options: HumanoidActionInvocationOptions = {}
  ): ReturnType<HumanoidActionRuntime["invoke"]> {
    return this.#actionMutex.runExclusive(async () => {
      await this.#assertKnownTransactionFingerprint(
        action,
        rawInput,
        transactionId,
        agentId
      );
      const decision = this.#actionModelSource(
        authority,
        action,
        rawInput,
        transactionId,
        agentId
      );
      const durableReceipt = await this.#durableReceiptForInvocation(
        action,
        rawInput,
        transactionId,
        agentId,
        decision
      );
      if (durableReceipt) return durableReceipt;
      this.#assertRunAcceptsDecisions();
      this.#assertDecisionCycleActive(decision);
      this.#physicsClock.throwIfFailed();
      this.#physicalExecution.assertExecutionOwner(transactionId);
      if (!requiresHumanoidClockPause(action)) {
        return this.#actions.invoke(
          action,
          rawInput,
          transactionId,
          agentId,
          decision,
          options
        );
      }
      const resumeClock = this.#continuousPhysicsEnabled;
      await this.#physicsClock.stop();
      try {
        return await this.#actions.invoke(
          action,
          rawInput,
          transactionId,
          agentId,
          decision,
          options
        );
      } finally {
        if (resumeClock
          && this.#checkpoint.status === "running"
          && activeActionExecutions(this.#checkpoint.action_execution_ledger).length === 0) {
          this.#physicsClock.start();
        }
      }
    });
  }

  async stopContinuousPhysics(): Promise<void> {
    this.#continuousPhysicsEnabled = false;
    await this.#physicsClock.stop();
  }

  receipt(transactionId: string): HumanoidActionReceipt | undefined {
    return this.#actions.receipt(transactionId);
  }

  async recallEmbodiedHistory(
    request: HumanoidEmbodiedRecallRequest
  ): Promise<JsonValue> {
    return recallHumanoidEmbodiedHistory({
      store: this.#store,
      memory: this.#checkpoint.embodied_memory,
      currentWorldRevision: this.#world.snapshot().worldRevision,
      request
    });
  }

  async recallGoalHistory(request: GoalHistoryRecallRequest): Promise<JsonValue> {
    this.#assertRunAcceptsDecisions();
    return recallDurableGoalHistory({
      goalDAG: this.#checkpoint.goal_dag,
      journal: this.#store,
      currentWorldRevision: this.#world.snapshot().worldRevision,
      request
    });
  }

  validateCycleEvidence(
    evidenceTransactionIds: readonly string[]
  ): HumanoidActionReceipt {
    return structuredClone(
      this.#validateCycleCausalEvidence(evidenceTransactionIds).execution
    );
  }

  cycleCompletionReadiness(): HumanoidCycleCompletionReadiness {
    return resolveHumanoidCycleCompletionReadiness({
      committedActions: this.#checkpoint.committed_actions,
      previousCycle: this.#checkpoint.last_cycle,
      activeCycle: this.#activeCycleRef(),
      currentWorld: this.#world.snapshot()
    });
  }

  coordinatorPhase(): HumanoidCoordinatorPhase {
    if (this.#checkpoint.goal_dag.status !== "active") return "goal_selection";
    const activeCycle = this.#activeCycleRef();
    if (!activeCycle) return "goal_selection";
    const receipts = Object.values(this.#checkpoint.committed_actions);
    const cycleReceipts = receipts.filter((receipt) => (
      sameAutonomousCycle(receipt.cycle, activeCycle)
    ));
    const completion = this.cycleCompletionReadiness();
    if (completion.status === "ready") {
      return completion.observed_after_execution
        ? "complete_cycle"
        : "post_execution";
    }
    const latestExecutionIndex = cycleReceipts.findLastIndex(physicalExecutionReceipt);
    const latestAcceptedPlanIndex = cycleReceipts.findLastIndex((receipt) => (
      receipt.accepted && isHumanoidPlanningReceipt(receipt)
    ));
    const activeGoal = this.#activeGoal();
    if (this.#store.definition.run_mode === "mission"
      && activeGoal
      && goalSha256(activeGoal) !== goalSha256(this.#missionGoal)
      && this.#missionGoalPhysicallySatisfied()) {
      return "goal_transition";
    }
    if (activeGoal
      && latestAcceptedPlanIndex <= latestExecutionIndex
      && this.#activeGoalPhysicallySatisfied(activeGoal)) {
      return "complete_satisfied_goal";
    }
    if (latestAcceptedPlanIndex > latestExecutionIndex) return "execute_plan";
    const phaseStart = Math.max(0, latestExecutionIndex + 1);
    const observed = cycleReceipts.slice(phaseStart).some((receipt) => (
      receipt.accepted && receipt.action === "observe_humanoid"
    ));
    if (!observed) return "observe_or_plan";
    const rejectedPlan = cycleReceipts.slice(phaseStart).some((receipt) => (
      !receipt.accepted && isHumanoidPlanningReceipt(receipt)
    ));
    return rejectedPlan ? "replan_or_retire" : "plan";
  }

  executorDelegationAvailable(): boolean {
    const phase = this.coordinatorPhase();
    if (phase === "execute_plan") return true;
    if (phase !== "post_execution" && phase !== "complete_cycle") return false;
    const goal = this.#activeGoal();
    return goal?.predicates.some((predicate, index) => (
      predicate.type === "block_removed"
        && this.#checkpoint.checker?.checks[index]?.passed !== true
    )) ?? false;
  }

  goalRetirementDelegationAvailable(): boolean {
    const phase = this.coordinatorPhase();
    if (phase === "goal_transition") return true;
    if (phase !== "replan_or_retire") return false;
    const activeCycle = this.#activeCycleRef();
    if (!activeCycle) return false;
    const latestRejectedPlan = Object.values(this.#checkpoint.committed_actions)
      .findLast((receipt) => (
        sameAutonomousCycle(receipt.cycle, activeCycle)
          && !receipt.accepted
          && isHumanoidPlanningReceipt(receipt)
    ));
    if (!latestRejectedPlan) return false;
    if (recoverableBlockedGoalActionReceipt(latestRejectedPlan)) return false;
    const detail = object(latestRejectedPlan.detail);
    const placements = detail?.reachable_base_placements;
    if (Array.isArray(placements) && placements.length > 0) return false;
    const latestMotionObservation = Object.values(
      this.#checkpoint.committed_actions
    ).findLast((receipt) => (
      sameAutonomousCycle(receipt.cycle, activeCycle)
        && receipt.agentId === "humanoid-motion-reference"
        && receipt.accepted
        && receipt.action === "observe_humanoid"
    ));
    return latestMotionObservation === undefined
      || !observationHasReachableBasePlacements(latestMotionObservation);
  }

  #missionGoalPhysicallySatisfied(): boolean {
    const world = this.#world.snapshot();
    return inspectHumanoidGoal(
      this.#missionGoal,
      this.#scenario,
      world,
      createHumanoidGoalProgress(this.#missionGoal, world)
    ).success;
  }

  #activeGoalPhysicallySatisfied(activeGoal: Goal): boolean {
    const world = this.#world.snapshot();
    const progress = this.#checkpoint.goal_progress;
    if (!progress) return false;
    return (world.frame === progress.last_world_frame
      && world.worldRevision === progress.last_world_revision
      ? inspectHumanoidGoal(activeGoal, this.#scenario, world, progress)
      : advanceHumanoidGoal(activeGoal, this.#scenario, world, progress).checker
    ).success;
  }

  sentryDelegationAvailable(): boolean {
    const phase = this.coordinatorPhase();
    if (phase === "observe_or_plan" || phase === "post_execution") return true;
    if (phase !== "replan_or_retire") return false;
    const activeCycle = this.#activeCycleRef();
    if (!activeCycle) return false;
    const cycleReceipts = Object.values(this.#checkpoint.committed_actions)
      .filter((receipt) => sameAutonomousCycle(receipt.cycle, activeCycle));
    const latestRejectedPlanIndex = cycleReceipts.findLastIndex((receipt) => (
      !receipt.accepted && isHumanoidPlanningReceipt(receipt)
    ));
    if (latestRejectedPlanIndex < 0) return false;
    return !cycleReceipts.slice(latestRejectedPlanIndex + 1).some((receipt) => (
      receipt.agentId === "humanoid-sentry"
        && receipt.accepted
        && receipt.action === "observe_humanoid"
    ));
  }

  #validateCycleCausalEvidence(
    evidenceTransactionIds: readonly string[]
  ): HumanoidCycleCausalEvidence {
    const currentWorld = this.#world.snapshot();
    return validateHumanoidCycleCausalEvidence({
      evidenceTransactionIds,
      committedActions: this.#checkpoint.committed_actions,
      previousCycle: this.#checkpoint.last_cycle,
      activeCycle: this.#activeCycleRef(),
      currentWorld
    });
  }

  snapshot(): HumanoidWorldSnapshot {
    return this.#world.snapshot();
  }

  activeNode(agentId?: string): TaskNode {
    const id = agentId ?? this.#checkpoint.active_agent_id ?? this.rootAgentId;
    const node = this.#checkpoint.nodes[id];
    if (!node) throw new Error(`Unknown humanoid hierarchy node: ${id}`);
    return structuredClone(node);
  }

  contextMemory(): ContextMemoryState {
    return structuredClone(this.#checkpoint.context_memory);
  }

  contextWorldIdentity(): { worldRevision: number } {
    return { worldRevision: this.#world.snapshot().worldRevision };
  }

  contextReceipts(): Record<string, { accepted: boolean; worldRevision: number }> {
    return Object.fromEntries(Object.entries(this.#checkpoint.committed_actions).map(
      ([transactionId, receipt]) => [transactionId, {
        accepted: receipt.accepted,
        worldRevision: receipt.worldAfterRevision
      }]
    ));
  }

  contextAnchor(agentId: string): JsonValue {
    const node = this.activeNode(agentId);
    const observation = this.#world.observe();
    const world = this.#world.snapshot();
    const activeGoal = this.#activeGoal();
    const checkpoint = this.#checkpointForContext(activeGoal, world);
    const result = createHumanoidContextAnchor({
      mission: this.#store.definition.mission,
      runMode: this.#store.definition.run_mode,
      scenarioId: this.#store.definition.scenario_id,
      scenario: this.#scenario,
      missionGoal: this.#missionGoal,
      checkpoint,
      ...(activeGoal ? { activeGoal } : {}),
      node,
      observation,
      world,
      cycleCompletion: this.cycleCompletionReadiness(),
      coordinatorPhase: this.coordinatorPhase()
    });
    this.#rememberGoalEvidence(result.worldEvidence);
    this.#contextGoalEvidenceRefs.set(agentId, result.worldEvidence.evidence.ref);
    this.#pruneGoalEvidence();
    if (agentId !== "humanoid-motion-reference") return result.anchor;
    return json({
      ...(object(result.anchor) ?? {}),
      planning_tool_state: this.#actions.planningToolState(agentId)
    });
  }

  #checkpointForContext(
    activeGoal: Goal | undefined,
    world: HumanoidWorldSnapshot
  ): HumanoidRunCheckpoint {
    const progress = this.#checkpoint.goal_progress;
    if (!activeGoal || !progress
      || (progress.last_world_frame === world.frame
        && progress.last_world_revision === world.worldRevision)) {
      return this.#checkpoint;
    }
    const projected = advanceHumanoidGoal(
      activeGoal,
      this.#scenario,
      world,
      progress
    );
    return {
      ...this.#checkpoint,
      world,
      goal_progress: projected.progress,
      checker: projected.checker
    };
  }

  assertContextSummaryEvidence(summary: ContextCompactionSummary): void {
    for (const item of summary.completed) {
      for (const transactionId of item.transaction_ids) {
        const receipt = this.#checkpoint.committed_actions[transactionId];
        if (!receipt) {
          throw new Error(`Context compaction referenced an unknown transaction: ${transactionId}`);
        }
        if (!receipt.accepted) {
          throw new Error(`Completed context memory referenced a rejected transaction: ${transactionId}`);
        }
      }
    }
    for (const item of summary.blockers) {
      for (const transactionId of item.transaction_ids) {
        if (!this.#checkpoint.committed_actions[transactionId]) {
          throw new Error(`Context compaction referenced an unknown transaction: ${transactionId}`);
        }
      }
    }
  }

  async updateContextMemory(
    state: ContextMemoryState,
    journalRecord?: JsonValue
  ): Promise<void> {
    if (journalRecord !== undefined) await this.#store.append("context", journalRecord);
    this.#checkpoint.context_memory = structuredClone(state);
    await this.#persist();
    await this.emit("context_memory_updated", { context_memory: json(state) });
  }

  async recordModelCallStarted(agentId: string): Promise<string> {
    return this.#goalStateMutex.runExclusive(async () => {
      this.#signal?.throwIfAborted();
      this.#assertRunAcceptsDecisions();
      const cycle = this.#activeCycleRef();
      const record = await this.#requiredModelAuthority().recordStarted(agentId, cycle);
      const node = this.#node(agentId);
      node.model_calls_used += 1;
      node.updated_at = record.at;
      this.#checkpoint.total_model_calls += 1;
      await this.#persist();
      await this.emit("model_request_started", {
        agent_id: agentId,
        agent_name: node.name,
        model_call_id: record.model_call_id,
        ...(cycle ? { cycle } : {}),
        purpose: "agent_decision",
        node_model_calls: node.model_calls_used,
        total_model_calls: this.#checkpoint.total_model_calls
      });
      return record.model_call_id;
    });
  }

  async recordModelCallCompleted(input: {
    modelCallId: string;
    agentId: string;
    responseId: string;
    responseOutputSha256: string;
    toolCalls: Array<{
      toolCallId: string;
      toolName: string;
      argumentsSha256: string;
    }>;
  }): Promise<void> {
    await this.#goalStateMutex.runExclusive(async () => {
      const record = await this.#requiredModelAuthority().recordCompleted(input);
      await this.emit("model_request_completed", {
        agent_id: input.agentId,
        model_call_id: input.modelCallId,
        response_id: input.responseId,
        ...(record.cycle ? { cycle: record.cycle } : {}),
        tool_call_count: input.toolCalls.length
      });
    });
  }

  async recordModelCallFailed(modelCallId: string, agentId: string): Promise<void> {
    await this.#goalStateMutex.runExclusive(async () => {
      const record = await this.#requiredModelAuthority().recordFailed(
        modelCallId,
        agentId
      );
      await this.emit("model_request_failed", {
        agent_id: agentId,
        model_call_id: modelCallId,
        ...(record.cycle ? { cycle: record.cycle } : {})
      });
    });
  }

  async recordCompactionModelCall(agentId: string): Promise<void> {
    await this.#recordCompactionCalls(agentId, 1, "model_request_started");
  }

  async reconcileCompactionModelCalls(agentId: string, additionalCalls: number): Promise<void> {
    if (!Number.isSafeInteger(additionalCalls) || additionalCalls <= 0) return;
    await this.#recordCompactionCalls(agentId, additionalCalls, "model_requests_reconciled");
  }

  async setActiveAgent(agentId: string): Promise<void> {
    if (this.#checkpoint.active_agent_id === agentId) return;
    const at = new Date().toISOString();
    for (const node of Object.values(this.#checkpoint.nodes)) {
      if (node.id === this.rootAgentId) {
        node.status = agentId === node.id ? "active" : "waiting";
      } else {
        node.status = agentId === node.id ? "active" : "ready";
      }
      node.updated_at = at;
    }
    this.#checkpoint.active_agent_id = agentId;
    this.#checkpoint.active_agent_ids = [agentId];
    await this.#persist();
    await this.emit("hierarchy_focus_changed", {
      active_agent_id: agentId,
      nodes: json(this.#checkpoint.nodes)
    });
  }

  async start(resumed: boolean): Promise<void> {
    await this.#reconcileActionCommits();
    const at = new Date().toISOString();
    this.#checkpoint.status = "running";
    this.#checkpoint.error = null;
    this.#checkpoint.active_agent_id = this.rootAgentId;
    this.#checkpoint.active_agent_ids = [this.rootAgentId];
    this.#node(this.rootAgentId).status = "active";
    const type = resumed ? "run_resumed" : "run_started";
    this.#checkpoint.pending_lifecycle_events.push(createLifecycleEvent({
      runId: this.runId,
      type,
      at,
      data: {
        runtime: "humanoid_g1",
        world_frame: this.#checkpoint.world.frame,
        world_revision: this.#checkpoint.world.worldRevision
      }
    }));
    await this.#persist();
    await reconcileLifecycleOutbox({
      store: this.#store,
      checkpoint: this.#checkpoint,
      persistCheckpoint: () => this.#persist(),
      eventSink: this.#eventSink
    });
    await this.recoverPendingPhysicalExecution();
    this.#continuousPhysicsEnabled = true;
    const pendingExecution = activeActionExecutions(
      this.#checkpoint.action_execution_ledger
    )[0];
    if (!pendingExecution) this.#physicsClock.start();
    const controller = this.#world.snapshot().robot.controller;
    await this.emit(
      pendingExecution ? "continuous_physics_deferred" : "continuous_physics_started",
      {
      control_step_seconds: controller.controlStepSeconds,
      physics_step_seconds: controller.physicsStepSeconds,
      planning_policy: "live_authority_isolated_rollout_revalidation",
      ...(pendingExecution
        ? { execution_transaction_id: pendingExecution.transaction_id }
        : {})
      }
    );
  }

  pendingPhysicalExecutionTransactionId(): string | undefined {
    return activeActionExecutions(
      this.#checkpoint.action_execution_ledger
    )[0]?.transaction_id;
  }

  async recoverPendingPhysicalExecution(): Promise<HumanoidActionReceipt | undefined> {
    const pending = activeActionExecutions(
      this.#checkpoint.action_execution_ledger
    );
    if (pending.length === 0) return undefined;
    if (pending.length > 1) {
      throw new Error("Multiple physical executions cannot share one humanoid runtime");
    }
    const execution = pending[0]!;
    const input = {
      planning_transaction_id: execution.admission.planning_transaction_id
    };
    const fingerprint = humanoidActionFingerprint(
      execution.action,
      execution.agent_id,
      input
    );
    if (actionExecutionFingerprintSha256(fingerprint)
      !== execution.action_fingerprint_sha256) {
      throw new Error(
        `Durable physical execution input cannot be reconstructed: ${execution.transaction_id}`
      );
    }
    await this.emit("physical_execution_recovery_started", json({
      transaction_id: execution.transaction_id,
      planning_transaction_id: execution.admission.planning_transaction_id,
      plan_id: execution.admission.plan_id,
      action: execution.action,
      committed_frame_count: execution.progress.committed_frame_count
    }));
    const receipt = await this.invoke(
      execution.action,
      input,
      execution.transaction_id,
      execution.agent_id,
      {
        tool_call_id: execution.transaction_id,
        tool_name: execution.action,
        arguments_sha256: modelPayloadSha256(input)
      }
    );
    await this.emit("physical_execution_recovered", json({
      transaction_id: receipt.transactionId,
      action: receipt.action,
      accepted: receipt.accepted,
      code: receipt.code,
      frame_count: receipt.frameCount,
      world_revision: receipt.worldAfterRevision
    }));
    return receipt;
  }

  async completeCycle(output: string): Promise<boolean> {
    return this.#goalStateMutex.runExclusive(() => this.#withPhysicsPaused(async () => {
      let cycle: JsonValue;
      try {
        cycle = json(JSON.parse(output));
      } catch {
        cycle = output;
      }
      const activeGoal = this.#requiredActiveGoal();
      const captured = await this.#world.capturePersistenceState();
      this.#applyWorldPersistenceState(captured);
      const progress = this.#checkpoint.goal_progress;
      if (!progress) throw new Error("Active Goal progress is unavailable");
      const world = captured.world;
      const checker = inspectHumanoidGoal(
        activeGoal,
        this.#scenario,
        world,
        progress
      );
      const activeCycle = this.#requiredActiveCycleRef();
      const evidence = previousCycleEvidence(cycle);
      const causalEvidence = this.#validateCycleCausalEvidence([...evidence]);
      const execution = causalEvidence.execution;
      const actionEvidenceRef = `action:${execution.transactionId}`;
      if (!this.#goalEvidence.has(actionEvidenceRef)) {
        throw new Error(
          `Autonomous cycle execution has no durable Goal evidence: ${execution.transactionId}`
        );
      }
      const mutationEvidenceRefs = causalEvidence.worldMutations.map((mutation) => (
        `action:${mutation.transactionId}`
      ));
      for (const ref of mutationEvidenceRefs) {
        if (!this.#goalEvidence.has(ref)) {
          throw new Error(`Autonomous cycle world mutation has no durable Goal evidence: ${ref}`);
        }
      }
      const goalEvidenceRefs = [actionEvidenceRef, ...mutationEvidenceRefs];
      let completedGoalDAG = this.#checkpoint.goal_dag;
      if (checker.success) {
        const epochId = this.#checkpoint.goal_dag.current_epoch_id;
        if (!epochId) throw new Error("Active Goal epoch identity is unavailable");
        const evaluation = createGoalEvaluationEvidence({
          epochId,
          goalContentSha256: goalSha256(activeGoal),
          worldFrame: world.frame,
          worldRevision: world.worldRevision,
          evaluation: json(checker)
        });
        this.#rememberGoalEvidence(evaluation);
        await this.#persistGoalEvidence([evaluation.evidence.ref]);
        goalEvidenceRefs.push(evaluation.evidence.ref);
        const resolvedGoalDAG = completeGoalEpoch(this.#checkpoint.goal_dag, {
          resolution_evidence_refs: [evaluation.evidence.ref],
          resolved_world_revision: world.worldRevision
        }, this.#goalHarness());
        completedGoalDAG = await reconcileAndCompactGoalHistory({
          store: this.#store,
          goalDAG: resolvedGoalDAG
        });
      }

      const memory = appendEmbodiedEpisode({
        state: this.#checkpoint.embodied_memory,
        sequence: activeCycle.cycle_index,
        execution,
        modelSummary: cycleSummary(cycle),
        world,
        goalSuccess: checker.success,
        cycle: activeCycle,
        goalEvidenceRefs,
        worldMutations: causalEvidence.worldMutations
      });
      await this.#persistEmbodiedEpisode(memory.episode);

      const actionWindow = retainRecentActionReceipts(
        this.#checkpoint.committed_actions
      );
      // Publish the causal transition as one synchronous checkpoint cut. Other
      // authority publishers may persist while the episode journal is being
      // written; they must observe either the prior active Cycle or the fully
      // committed successor, never a new cycle_index paired with active_cycle.
      this.#checkpoint.goal_dag = completedGoalDAG;
      this.#checkpoint.goal_progress = checker.success
        ? null
        : this.#checkpoint.goal_progress;
      this.#checkpoint.checker = checker.success ? null : checker;
      this.#checkpoint.cycle_index = activeCycle.cycle_index;
      this.#checkpoint.last_cycle = cycle;
      this.#checkpoint.embodied_memory = memory.state;
      this.#checkpoint.active_cycle = null;
      this.#checkpoint.committed_actions = actionWindow.receipts;
      this.#pruneRuntimeAuthority();

      const missionCompleted = checker.success
        && this.#store.definition.run_mode === "mission"
        && this.missionGoalCompleted();
      if (missionCompleted) {
        this.#continuousPhysicsEnabled = false;
        this.#stageFinish(
          "succeeded",
          output,
          null,
          "run_succeeded"
        );
      }

      await this.#persist();
      await this.#store.append("checker", json(checker));
      await this.emit("embodied_episode_recorded", {
        episode: json(memory.episode),
        embodied_memory: json(memory.state),
        retained_episodes: memory.state.recent_episodes.length,
        total_episodes: memory.state.total_episodes,
        pruned_checkpoint_receipts: actionWindow.removed,
        historical_only: false
      });
      await this.emit("autonomous_cycle_completed", {
        cycle_index: this.#checkpoint.cycle_index,
        cycle: json(activeCycle),
        causal_trace: json(memory.episode.causal_trace ?? null),
        output: cycle,
        checker: json(checker),
        goal_epoch_completed: checker.success
      });
      if (checker.success) await this.#emitGoalState("epoch_completed");
      if (missionCompleted) {
        await reconcileLifecycleOutbox({
          store: this.#store,
          checkpoint: this.#checkpoint,
          persistCheckpoint: () => this.#persist(),
          eventSink: this.#eventSink
        });
      }
      return checker.success;
    }));
  }

  async completeSatisfiedGoal(output: string): Promise<boolean> {
    return this.#goalStateMutex.runExclusive(() => this.#withPhysicsPaused(async () => {
      if (this.coordinatorPhase() !== "complete_satisfied_goal") {
        throw new Error("The active Goal is not ready for execution-free completion");
      }
      let cycleOutput: JsonValue;
      try {
        cycleOutput = json(JSON.parse(output));
      } catch {
        cycleOutput = output;
      }
      const activeGoal = this.#requiredActiveGoal();
      const activeCycle = this.#requiredActiveCycleRef();
      const captured = await this.#world.capturePersistenceState();
      this.#applyWorldPersistenceState(captured);
      const progress = this.#checkpoint.goal_progress;
      if (!progress) throw new Error("Active Goal progress is unavailable");
      const checker = inspectHumanoidGoal(
        activeGoal,
        this.#scenario,
        captured.world,
        progress
      );
      if (!checker.success) {
        throw new Error("The active Goal is no longer physically satisfied");
      }
      if (this.cycleCompletionReadiness().status === "ready") {
        throw new Error(
          "A Goal with new physical execution must complete through causal cycle evidence"
        );
      }

      const evaluation = createGoalEvaluationEvidence({
        epochId: activeCycle.goal_epoch_id,
        goalContentSha256: goalSha256(activeGoal),
        worldFrame: captured.world.frame,
        worldRevision: captured.world.worldRevision,
        evaluation: json(checker)
      });
      this.#rememberGoalEvidence(evaluation);
      await this.#persistGoalEvidence([evaluation.evidence.ref]);
      const resolvedGoalDAG = completeGoalEpoch(this.#checkpoint.goal_dag, {
        resolution_evidence_refs: [evaluation.evidence.ref],
        resolved_world_revision: captured.world.worldRevision
      }, this.#goalHarness());
      this.#checkpoint.goal_dag = await reconcileAndCompactGoalHistory({
        store: this.#store,
        goalDAG: resolvedGoalDAG
      });
      this.#checkpoint.goal_progress = null;
      this.#checkpoint.checker = null;
      this.#checkpoint.cycle_index = activeCycle.cycle_index;
      this.#checkpoint.last_cycle = cycleOutput;
      this.#checkpoint.active_cycle = null;
      this.#pruneRuntimeAuthority();

      const missionCompleted = this.#store.definition.run_mode === "mission"
        && this.missionGoalCompleted();
      if (missionCompleted) {
        this.#continuousPhysicsEnabled = false;
        this.#stageFinish("succeeded", output, null, "run_succeeded");
      }

      await this.#persist();
      await this.#store.append("checker", json(checker));
      await this.emit("autonomous_goal_satisfied", {
        cycle: json(activeCycle),
        output: cycleOutput,
        checker: json(checker),
        physical_execution_required: false,
        goal_epoch_completed: true
      });
      await this.#emitGoalState("epoch_completed");
      if (missionCompleted) {
        await reconcileLifecycleOutbox({
          store: this.#store,
          checkpoint: this.#checkpoint,
          persistCheckpoint: () => this.#persist(),
          eventSink: this.#eventSink
        });
      }
      return true;
    }));
  }

  async succeed(output: string): Promise<void> {
    await this.#finish("succeeded", output, null, "run_succeeded");
  }

  async fail(error: string): Promise<void> {
    await this.#finish("failed", null, error, "run_failed");
  }

  async interrupt(error: string): Promise<void> {
    await this.#finish("interrupted", null, error, "run_interrupted");
  }

  async pause(reason: string): Promise<void> {
    await this.#finish("paused", null, null, "run_paused", reason);
  }

  async recordFramework(scope: string, event: JsonValue, agentId?: string): Promise<void> {
    const runtimeEventId = randomUUID();
    const cycle = this.#activeCycleRef();
    const record = {
      scope,
      ...(agentId ? { agent_id: agentId } : {}),
      ...(cycle ? { cycle } : {}),
      event,
      at: new Date().toISOString(),
      runtime_event_id: runtimeEventId
    };
    await this.#store.append("framework", json(record));
    await this.emit("framework_event", json(record), runtimeEventId);
  }

  async recordProvider(event: JsonValue, agentId?: string): Promise<void> {
    const runtimeEventId = randomUUID();
    const cycle = this.#activeCycleRef();
    const at = new Date().toISOString();
    const usageDelta = modelUsageDeltaFromProviderEvent(event, agentId);
    if (usageDelta) {
      this.#checkpoint.model_usage = addModelUsage(
        this.#checkpoint.model_usage,
        usageDelta,
        at
      );
    }
    const record = {
      ...(agentId ? { agent_id: agentId } : {}),
      ...object(event),
      ...(cycle ? { cycle } : {}),
      ...(usageDelta ? { model_usage: this.#checkpoint.model_usage } : {}),
      at,
      runtime_event_id: runtimeEventId
    };
    await this.#store.append("provider", json(record));
    await this.emit("provider_event", json(record), runtimeEventId);
  }

  async emit(
    type: string,
    data: JsonValue,
    eventId = randomUUID(),
    durable = true
  ): Promise<void> {
    const event: RuntimeEvent = {
      event_id: eventId,
      run_id: this.runId,
      type,
      at: new Date().toISOString(),
      data,
      ...(durable ? {} : { durable: false })
    };
    if (!durable) {
      await this.#eventSink(event);
      return;
    }
    const [persisted] = await this.#store.appendRuntimeEvents([event]);
    await this.#eventSink(persisted!);
  }

  async #durableReceiptForInvocation(
    action: Parameters<HumanoidActionRuntime["invoke"]>[0],
    rawInput: unknown,
    rawTransactionId: string,
    rawAgentId: string,
    decision: ModelDecisionRef
  ): Promise<HumanoidActionReceipt | undefined> {
    await this.#ensureActionTransactionIdentities();
    const transactionId = rawTransactionId.trim();
    const agentId = rawAgentId.trim();
    const identity = this.#actionTransactionIdentities.get(transactionId);
    if (!identity) return undefined;
    const fingerprint = humanoidActionFingerprint(action, agentId, rawInput);
    if (identity.agent_id !== agentId
      || identity.action !== action
      || identity.action_fingerprint_sha256
        !== actionTransactionFingerprintSha256(fingerprint)) {
      throw new Error(`Humanoid action transaction conflict: ${transactionId}`);
    }
    const cached = this.#actions.receipt(transactionId)
      ?? this.#durableActionReceiptCache.get(transactionId);
    if (cached) {
      this.#assertDurableReceiptIdentity(identity, cached, decision);
      return structuredClone(cached);
    }
    const receipt = await this.#readDurableActionReceipt(identity, decision);
    this.#durableActionReceiptCache.set(transactionId, receipt);
    return structuredClone(receipt);
  }

  async #assertKnownTransactionFingerprint(
    action: Parameters<HumanoidActionRuntime["invoke"]>[0],
    rawInput: unknown,
    rawTransactionId: string,
    rawAgentId: string
  ): Promise<void> {
    await this.#ensureActionTransactionIdentities();
    const transactionId = rawTransactionId.trim();
    const agentId = rawAgentId.trim();
    const identity = this.#actionTransactionIdentities.get(transactionId);
    if (!identity) return;
    const fingerprint = humanoidActionFingerprint(action, agentId, rawInput);
    if (identity.agent_id !== agentId
      || identity.action !== action
      || identity.action_fingerprint_sha256
        !== actionTransactionFingerprintSha256(fingerprint)) {
      throw new Error(`Humanoid action transaction conflict: ${transactionId}`);
    }
  }

  async #ensureActionTransactionIdentities(): Promise<void> {
    if (this.#actionTransactionIdentitiesLoaded) return;
    const records = await this.#store.readJournal("action_identities");
    this.#actionTransactionIdentities = rebuildActionTransactionIdentities(
      records,
      this.runId
    );
    this.#actionTransactionIdentitiesLoaded = true;
  }

  async #readDurableActionReceipt(
    identity: ActionTransactionIdentity,
    decision: ModelDecisionRef
  ): Promise<HumanoidActionReceipt> {
    const matches = (record: JsonValue): HumanoidActionReceipt | undefined => {
      if (record === null || typeof record !== "object" || Array.isArray(record)
        || record.transactionId !== identity.transaction_id) return undefined;
      if (actionCommitPayloadSha256(record) !== identity.action_record_sha256
        || record.runtime_event_id !== identity.runtime_event_id) {
        throw new Error(
          `Durable action receipt identity conflict: ${identity.transaction_id}`
        );
      }
      const { runtime_event_id: _runtimeEventId, ...rawReceipt } = record;
      const receipt = PersistedHumanoidActionReceiptSchema.parse(rawReceipt);
      this.#assertDurableReceiptIdentity(identity, receipt, decision);
      return receipt;
    };
    const tail = await this.#store.readJournalTail("actions", 256);
    for (let index = tail.entries.length - 1; index >= 0; index -= 1) {
      const receipt = matches(tail.entries[index]!);
      if (receipt) return receipt;
    }
    let end = tail.total - tail.entries.length;
    while (end > 0) {
      const from = Math.max(0, end - 256);
      const page = await this.#store.readJournalPage("actions", from, end - from);
      for (let index = page.entries.length - 1; index >= 0; index -= 1) {
        const receipt = matches(page.entries[index]!);
        if (receipt) return receipt;
      }
      end = from;
    }
    throw new Error(`Durable action receipt is missing: ${identity.transaction_id}`);
  }

  #assertDurableReceiptIdentity(
    identity: ActionTransactionIdentity,
    receipt: HumanoidActionReceipt,
    decision: ModelDecisionRef
  ): void {
    const authorityCycle = this.#requiredModelAuthority().cycleForModelCall(
      decision.model_call_id
    );
    if (receipt.transactionId !== identity.transaction_id
      || receipt.agentId !== identity.agent_id
      || receipt.action !== identity.action
      || actionTransactionFingerprintSha256(receipt.fingerprint)
        !== identity.action_fingerprint_sha256
      || !receipt.decision
      || modelPayloadSha256(receipt.decision) !== modelPayloadSha256(decision)
      || !authorityCycle
      || !sameAutonomousCycle(receipt.cycle, authorityCycle)
      || actionCommitPayloadSha256(json(receipt)) !== identity.receipt_sha256) {
      throw new Error(
        `Durable action receipt conflicts with tombstone: ${identity.transaction_id}`
      );
    }
  }

  async #prepareBlockRemoval(input: {
    transactionId: string;
    agentId: string;
    solidId: string;
    executionTransactionId: string;
  }): Promise<ScenarioBlockRemovalTransaction> {
    const chunks = await this.#store.readScenarioChunkDeltaState();
    return prepareAuthorizedBlockRemoval({
      scenario: this.#store.definition.scenario,
      chunks,
      currentWorld: this.#world.snapshot(),
      activeCycle: this.#requiredActiveCycleRef(),
      removalTransactionId: input.transactionId,
      agentId: input.agentId,
      solidId: input.solidId,
      executionTransactionId: input.executionTransactionId,
      committedActions: this.#checkpoint.committed_actions
    });
  }

  async #commitReceipt(receipt: HumanoidActionReceipt): Promise<void> {
    const activeCycle = this.#requiredActiveCycleRef();
    if (!receipt.decision || !sameAutonomousCycle(receipt.cycle, activeCycle)) {
      throw new Error(
        `Humanoid action has no cycle-bound model decision authority: ${receipt.transactionId}`
      );
    }
    this.#assertActionDecisionRef(
      receipt.decision,
      receipt.action,
      receipt.input,
      receipt.transactionId,
      receipt.agentId,
      activeCycle
    );
    const pending = this.#checkpoint.action_commit_outbox.pending[receipt.transactionId];
    if (pending) {
      assertPendingActionReceipt(pending.action_record, receipt);
      await this.#reconcileActionCommits();
      return;
    }
    const committed = this.#checkpoint.committed_actions[receipt.transactionId];
    if (committed) {
      if (actionCommitPayloadSha256(json(committed))
        !== actionCommitPayloadSha256(json(receipt))) {
        throw new Error(`Committed humanoid action conflicts with retry: ${receipt.transactionId}`);
      }
      await this.#reconcileActionCommits();
      return;
    }
    const physicalExecution = this.#checkpoint.action_execution_ledger.active[
      receipt.transactionId
    ];
    if (physicalExecution) {
      await this.#physicalExecution.synchronizeProgress(receipt.transactionId);
    } else {
      if (physicalExecutionReceipt(receipt)
        && object(receipt.detail).automatic_actuation !== false) {
        throw new Error(
          `Physical receipt cannot commit without an execution ledger: ${receipt.transactionId}`
        );
      }
      await this.#refreshWorldPersistenceState();
    }
    const activeGoal = this.#activeGoal();
    const runtimeEventId = randomUUID();
    const record = {
      ...receipt,
      runtime_event_id: runtimeEventId
    };
    const revisionLag = Math.max(
      0,
      this.#checkpoint.world.worldRevision - receipt.worldAfterRevision
    );
    const actionEvidence = createActionGoalEvidence({
      transactionId: receipt.transactionId,
      worldFrame: Math.max(0, this.#checkpoint.world.frame - revisionLag),
      worldRevision: receipt.worldAfterRevision,
      receipt: json(receipt)
    });
    let physicalWorldDelta: ReturnType<typeof captureHumanoidPhysicalWorldDelta> = undefined;
    let projectedScenarioChunks: ReturnType<
      typeof projectHumanoidPhysicalWorldDelta
    > | undefined = undefined;
    const blockRemoval = receipt.accepted && receipt.action === "remove_world_block"
      ? ScenarioBlockRemovalTransactionSchema.parse(
          object(receipt.detail).removal_transaction
        )
      : undefined;
    if (physicalExecution && blockRemoval) {
      throw new Error("One humanoid action cannot commit physical and block-removal deltas");
    }
    const currentScenarioChunks = physicalExecution || blockRemoval
      ? await this.#store.readScenarioChunkDeltaState()
      : undefined;
    if (physicalExecution && receipt.frameCount > 0 && currentScenarioChunks) {
      physicalWorldDelta = captureHumanoidPhysicalWorldDelta({
        scenario: this.#store.definition.scenario,
        chunks: currentScenarioChunks,
        world: this.#checkpoint.world,
        transactionId: receipt.transactionId
      });
      if (physicalWorldDelta) {
        projectedScenarioChunks = projectHumanoidPhysicalWorldDelta(
          this.#store.definition.scenario,
          currentScenarioChunks,
          physicalWorldDelta
        );
      }
    }
    if (blockRemoval && currentScenarioChunks) {
      projectedScenarioChunks = projectHumanoidBlockRemoval(
        this.#store,
        currentScenarioChunks,
        blockRemoval
      );
    }
    const rememberedExperience = physicalExecutionReceipt(receipt)
      || receipt.action === "remove_world_block"
      ? this.#prepareEmbodiedActionExperience(receipt, true)
      : undefined;
    const experience = rememberedExperience?.experience;
    const checkerScenario = projectedScenarioChunks
      ? materializeScenarioChunkDeltaState(
          this.#store.definition.scenario,
          projectedScenarioChunks
        )
      : this.#scenario;
    const nextChecker = activeGoal && this.#checkpoint.goal_progress
      ? inspectHumanoidGoal(
          activeGoal,
          checkerScenario,
          this.#checkpoint.world,
          this.#checkpoint.goal_progress
        )
      : null;
    const runtimeEvent = {
      event_id: runtimeEventId,
      run_id: this.runId,
      type: "humanoid_action_committed" as const,
      at: new Date().toISOString(),
      data: json({
        receipt: record,
        world: this.#checkpoint.world,
        checker: nextChecker,
        ...(physicalWorldDelta
          ? { physical_world_delta: physicalWorldDelta }
          : {}),
        ...(blockRemoval ? { block_removal: blockRemoval } : {}),
        ...(projectedScenarioChunks
          ? { scenario_chunks: projectedScenarioChunks }
          : {}),
        ...(experience
          ? {
              experience,
              embodied_memory: rememberedExperience?.state
                ?? this.#checkpoint.embodied_memory
            }
          : {})
      })
    };
    const nextOutbox = stageActionCommit(
      this.#checkpoint.action_commit_outbox,
      {
        transactionId: receipt.transactionId,
        runtimeEventId,
        actionRecord: json(record),
        goalEvidenceRef: actionEvidence.evidence.ref,
        goalEvidenceRecord: json(actionEvidence),
        ...(experience
          ? {
              experienceRef: experience.source_ref,
              experienceRecord: json(experience)
            }
          : {}),
        ...(physicalWorldDelta ? { physicalWorldDelta } : {}),
        ...(blockRemoval ? { blockRemoval } : {}),
        runtimeEvent
      }
    );
    let nextExecutionLedger = this.#checkpoint.action_execution_ledger;
    if (physicalExecution) {
      const staged = nextOutbox.pending[receipt.transactionId];
      if (!staged) throw new Error("Physical action commit was not staged");
      nextExecutionLedger = terminalizeActionExecution(
        this.#checkpoint.action_execution_ledger,
        {
          transactionId: receipt.transactionId,
          commit: staged
        }
      );
    }
    const node = this.#node(receipt.agentId);
    node.steps_used += 1;
    node.status = "ready";
    node.updated_at = new Date().toISOString();
    this.#checkpoint.committed_actions[receipt.transactionId] = structuredClone(receipt);
    this.#checkpoint.checker = nextChecker;
    if (rememberedExperience) {
      this.#checkpoint.embodied_memory = rememberedExperience.state;
    }
    this.#checkpoint.action_commit_outbox = nextOutbox;
    this.#checkpoint.action_execution_ledger = nextExecutionLedger;
    await this.#persist();
    await this.#reconcileActionCommits();
  }

  async #reconcileActionCommits(): Promise<void> {
    const staged = Object.values(this.#checkpoint.action_commit_outbox.pending);
    let reconciledScenarioState = false;
    for (const entry of staged) {
      if (entry.physical_world_delta) {
        await reconcileHumanoidPhysicalWorldDelta(
          this.#store,
          entry.physical_world_delta
        );
        reconciledScenarioState = true;
      }
      if (entry.block_removal) {
        await reconcileHumanoidBlockRemoval(
          this.#store,
          entry.block_removal
        );
        reconciledScenarioState = true;
      }
      this.#rememberGoalEvidence(
        GoalEvidenceArtifactSchema.parse(entry.goal_evidence_record)
      );
      const receipt = embodiedActionJournalReceipt(entry.action_record);
      if (receipt) {
        const experience = this.#rememberEmbodiedActionExperience(receipt, true);
        if (entry.experience_record !== undefined) {
          const stagedExperience = HumanoidEmbodiedExperienceSchema.parse(
            entry.experience_record
          );
          if (!experience
            || JSON.stringify(stagedExperience) !== JSON.stringify(experience)) {
            throw new Error(
              `Staged embodied experience conflicts with ${receipt.transactionId}`
            );
          }
        }
      }
    }
    let synchronizedChunks: Awaited<
      ReturnType<RunStore["readScenarioChunkDeltaState"]>
    > | undefined;
    let synchronization: Awaited<
      ReturnType<HumanoidWorld["synchronizeScenarioChunks"]>
    > | undefined;
    if (reconciledScenarioState) {
      synchronizedChunks = await this.#store.readScenarioChunkDeltaState();
      this.#scenario = materializeScenarioChunkDeltaState(
        this.#store.definition.scenario,
        synchronizedChunks
      );
      synchronization = await this.#world.synchronizeScenarioChunks(
        this.#store.definition.scenario,
        synchronizedChunks
      );
      await this.#refreshWorldPersistenceState();
    }
    const reconciled = await reconcileActionCommitOutbox({
      store: this.#store,
      outbox: this.#checkpoint.action_commit_outbox,
      persist: async (outbox) => {
        const previous = this.#checkpoint.action_commit_outbox;
        const previousLedger = this.#checkpoint.action_execution_ledger;
        let nextLedger = previousLedger;
        for (const [transactionId] of Object.entries(previous.pending)) {
          if (outbox.pending[transactionId]) continue;
          const execution = nextLedger.active[transactionId];
          if (!execution) continue;
          if (execution.status !== "terminal" || !execution.terminal) {
            throw new Error(
              `Action commit cannot acknowledge active execution: ${transactionId}`
            );
          }
          nextLedger = acknowledgeTerminalActionExecution(
            nextLedger,
            transactionId,
            execution.terminal
          );
        }
        this.#checkpoint.action_commit_outbox = outbox;
        this.#checkpoint.action_execution_ledger = nextLedger;
        try {
          await this.#persist();
        } catch (error) {
          this.#checkpoint.action_commit_outbox = previous;
          this.#checkpoint.action_execution_ledger = previousLedger;
          throw error;
        }
      },
      publish: async (event) => {
        try {
          await this.#eventSink(event);
        } catch {
          return;
        }
      }
    });
    this.#checkpoint.action_commit_outbox = reconciled;
    for (const entry of staged) {
      this.#persistedGoalEvidenceRefs.add(entry.goal_evidence_ref);
      this.#rememberActionTransactionIdentity(
        createActionTransactionIdentity(entry)
      );
    }
    await this.#physicalExecution.acknowledgeTerminals(staged);
    const actionWindow = retainRecentActionReceipts(
      this.#checkpoint.committed_actions
    );
    if (actionWindow.removed > 0) {
      this.#checkpoint.committed_actions = actionWindow.receipts;
      await this.#persist();
    }
    if (synchronizedChunks && synchronization?.changed) {
      await this.#persist();
      await this.emit("humanoid_scenario_synchronized", json({
        scenario_chunks: synchronizedChunks,
        synchronization
      }));
    }
  }

  async #recordCompactionCalls(
    agentId: string,
    count: number,
    eventType: "model_request_started" | "model_requests_reconciled"
  ): Promise<void> {
    this.#signal?.throwIfAborted();
    const node = this.#node(agentId);
    node.model_calls_used += count;
    node.updated_at = new Date().toISOString();
    this.#checkpoint.total_model_calls += count;
    await this.#persist();
    await this.emit(eventType, {
      agent_id: agentId,
      agent_name: node.name,
      purpose: "context_compaction",
      model_calls: count,
      node_model_calls: node.model_calls_used,
      total_model_calls: this.#checkpoint.total_model_calls
    });
  }

  async #persistEmbodiedEpisode(episode: HumanoidEmbodiedEpisode): Promise<void> {
    const parsed = HumanoidEmbodiedEpisodeSchema.parse(episode);
    const tail = await this.#store.readJournalTail("episodes", 1);
    const last = tail.entries[0];
    if (last !== undefined) {
      const previous = HumanoidEmbodiedEpisodeSchema.parse(last);
      if (previous.source_ref === parsed.source_ref) return;
    }
    await this.#store.append("episodes", json(parsed));
  }

  #rememberEmbodiedActionExperience(
    receipt: HumanoidActionReceipt,
    required: boolean
  ): HumanoidEmbodiedExperience | undefined {
    const remembered = this.#prepareEmbodiedActionExperience(receipt, required);
    if (!remembered) return undefined;
    this.#checkpoint.embodied_memory = remembered.state;
    return remembered.experience;
  }

  #prepareEmbodiedActionExperience(
    receipt: HumanoidActionReceipt,
    required: boolean
  ): ReturnType<typeof rememberEmbodiedActionExperience> | undefined {
    const epochId = receipt.cycle?.goal_epoch_id;
    const epoch = epochId
      ? this.#checkpoint.goal_dag.epochs.find((entry) => entry.epoch_id === epochId)
      : undefined;
    const candidate = epoch
      ? this.#checkpoint.goal_dag.candidates[epoch.candidate_id]
      : undefined;
    if (!receipt.decision || !receipt.cycle || !candidate) {
      if (required) {
        throw new Error(
          `Embodied experience has no model-selected Goal epoch: ${receipt.transactionId}`
        );
      }
      return undefined;
    }
    return rememberEmbodiedActionExperience({
      state: this.#checkpoint.embodied_memory,
      execution: receipt,
      goal: candidate.goal
    });
  }

  async #finish(
    status: "succeeded" | "failed" | "interrupted" | "paused",
    output: string | null,
    error: string | null,
    eventType: "run_succeeded" | "run_failed" | "run_interrupted" | "run_paused",
    reason?: string
  ): Promise<void> {
    if (this.#checkpoint.status === "succeeded" && status !== "succeeded") {
      await reconcileLifecycleOutbox({
        store: this.#store,
        checkpoint: this.#checkpoint,
        persistCheckpoint: () => this.#persist(),
        eventSink: this.#eventSink
      });
      return;
    }
    this.#continuousPhysicsEnabled = false;
    await this.#physicsClock.stop();
    this.#stageFinish(status, output, error, eventType, reason);
    await this.#persist();
    await reconcileLifecycleOutbox({
      store: this.#store,
      checkpoint: this.#checkpoint,
      persistCheckpoint: () => this.#persist(),
      eventSink: this.#eventSink
    });
  }

  #stageFinish(
    status: "succeeded" | "failed" | "interrupted" | "paused",
    output: string | null,
    error: string | null,
    eventType: "run_succeeded" | "run_failed" | "run_interrupted" | "run_paused",
    reason?: string
  ): void {
    const at = new Date().toISOString();
    this.#checkpoint.status = status;
    this.#checkpoint.final_output = output;
    this.#checkpoint.error = error;
    this.#checkpoint.active_agent_id = null;
    this.#checkpoint.active_agent_ids = [];
    this.#node(this.rootAgentId).status = status === "succeeded"
      ? "completed"
      : status === "paused" ? "waiting" : "failed";
    this.#checkpoint.pending_lifecycle_events.push(createLifecycleEvent({
      runId: this.runId,
      type: eventType,
      at,
      data: {
        runtime: "humanoid_g1",
        ...(output ? { output } : {}),
        ...(error ? { error } : {}),
        ...(reason ? { reason } : {})
      }
    }));
  }

  #assertRunAcceptsDecisions(): void {
    if (this.#checkpoint.status !== "starting"
      && this.#checkpoint.status !== "running") {
      throw new Error(
        `Humanoid Run cannot accept new model decisions while ${this.#checkpoint.status}`
      );
    }
  }

  #activeCycleRef(): AutonomousCycleRef | undefined {
    return this.#checkpoint.active_cycle
      ? autonomousCycleRef(this.#checkpoint.active_cycle)
      : undefined;
  }

  #requiredActiveCycleRef(): AutonomousCycleRef {
    const cycle = this.#activeCycleRef();
    if (!cycle) throw new Error("No autonomous cycle is active");
    return cycle;
  }

  #createActiveCycle(): ActiveAutonomousCycle {
    if (this.#checkpoint.active_cycle) {
      throw new Error(
        `Autonomous cycle is already active: ${this.#checkpoint.active_cycle.cycle_id}`
      );
    }
    const goalEpochId = this.#checkpoint.goal_dag.current_epoch_id;
    if (this.#checkpoint.goal_dag.status !== "active" || !goalEpochId) {
      throw new Error("An autonomous cycle requires an active Goal epoch");
    }
    const cycle = createActiveAutonomousCycle({
      cycleIndex: this.#checkpoint.cycle_index + 1,
      goalEpochId,
      worldFrame: this.#checkpoint.world.frame,
      worldRevision: this.#checkpoint.world.worldRevision
    });
    this.#checkpoint.active_cycle = cycle;
    return cycle;
  }

  async #emitCycleStarted(cycle: ActiveAutonomousCycle): Promise<void> {
    await this.emit("autonomous_cycle_started", json({ cycle }));
  }

  #activeGoal(): Goal | undefined {
    const epochId = this.#checkpoint.goal_dag.current_epoch_id;
    if (!epochId) return undefined;
    const epoch = this.#checkpoint.goal_dag.epochs.find(
      (candidate) => candidate.epoch_id === epochId
    );
    if (!epoch || epoch.status !== "active") return undefined;
    const candidate = this.#checkpoint.goal_dag.candidates[epoch.candidate_id];
    return candidate?.status === "active" ? candidate.goal : undefined;
  }

  #requiredActiveGoal(): Goal {
    const goal = this.#activeGoal();
    if (!goal) throw new Error("No model-selected Goal epoch is active");
    return goal;
  }

  #requiredModelAuthority(): HumanoidModelAuthority {
    if (!this.#modelAuthority) {
      throw new Error("Goal autonomy has not been initialized from an Agent manifest");
    }
    return this.#modelAuthority;
  }

  #goalHarness() {
    return this.#requiredModelAuthority().goalHarness({
      evidence: this.#goalEvidence,
      scenario: this.#store.definition.scenario
    });
  }

  #goalModelSource(
    toolAuthority: GoalToolCallAuthority,
    expectedToolName: string,
    normalizedInput: unknown
  ) {
    return this.#requiredModelAuthority().goalModelSource(
      toolAuthority,
      expectedToolName,
      normalizedInput
    );
  }

  #actionModelSource(
    toolAuthority: HumanoidActionToolCallAuthority,
    expectedToolName: Parameters<HumanoidActionRuntime["invoke"]>[0],
    input: unknown,
    transactionId: string,
    agentId: string
  ): ModelDecisionRef {
    return this.#requiredModelAuthority().actionModelSource({
      toolAuthority,
      expectedToolName,
      actionInput: input,
      transactionId,
      agentId
    });
  }

  #assertDecisionCycleActive(decision: ModelDecisionRef): void {
    this.#requiredModelAuthority().assertDecisionCycleActive(
      decision,
      this.#requiredActiveCycleRef()
    );
  }

  #assertActionDecisionRef(
    rawDecision: ModelDecisionRef,
    expectedToolName: Parameters<HumanoidActionRuntime["invoke"]>[0],
    input: unknown,
    transactionId: string,
    agentId: string,
    cycle: AutonomousCycleRef
  ): void {
    this.#requiredModelAuthority().assertActionDecision({
      rawDecision,
      expectedToolName,
      actionInput: input,
      transactionId,
      agentId,
      cycle
    });
  }

  #requiredGoalEvidence(ref: string): GoalEvidenceArtifact {
    const artifact = this.#goalEvidence.get(ref);
    if (!artifact) throw new Error(`Goal evidence is unavailable: ${ref}`);
    return artifact;
  }

  #requiredContextGoalEvidence(agentId: string): {
    ref: string;
    artifact: GoalEvidenceArtifact;
  } {
    const ref = this.#contextGoalEvidenceRefs.get(agentId);
    if (!ref) {
      throw new Error(`Agent has no current Harness evidence binding: ${agentId}`);
    }
    return { ref, artifact: this.#requiredGoalEvidence(ref) };
  }

  #rememberGoalEvidence(rawArtifact: GoalEvidenceArtifact): void {
    const artifact = GoalEvidenceArtifactSchema.parse(rawArtifact);
    const ref = artifact.evidence.ref;
    const existing = this.#goalEvidence.get(ref);
    if (existing && JSON.stringify(existing) !== JSON.stringify(artifact)) {
      throw new Error(`Goal evidence reference was rebound: ${ref}`);
    }
    if (!existing) this.#goalEvidence.set(ref, artifact);
  }

  #pruneRuntimeAuthority(): void {
    this.#pruneGoalEvidence();
    const retainedModelCalls = new Set<string>();
    for (const candidate of Object.values(this.#checkpoint.goal_dag.candidates)) {
      retainedModelCalls.add(candidate.source.model_call_id);
    }
    for (const epoch of this.#checkpoint.goal_dag.epochs) {
      retainedModelCalls.add(epoch.selected_by.model_call_id);
      if (epoch.retired_by) retainedModelCalls.add(epoch.retired_by.model_call_id);
    }
    for (const receipt of Object.values(this.#checkpoint.committed_actions)) {
      if (receipt.decision) retainedModelCalls.add(receipt.decision.model_call_id);
    }
    this.#modelAuthority?.prune(retainedModelCalls);
  }

  #pruneGoalEvidence(): void {
    const retained = new Set(Object.keys(this.#checkpoint.goal_dag.evidence));
    for (const ref of this.#contextGoalEvidenceRefs.values()) retained.add(ref);
    for (const transactionId of Object.keys(this.#checkpoint.committed_actions)) {
      retained.add(`action:${transactionId}`);
    }
    for (const pending of Object.values(this.#checkpoint.action_commit_outbox.pending)) {
      retained.add(pending.goal_evidence_ref);
    }
    for (const ref of this.#goalEvidence.keys()) {
      if (!retained.has(ref)) this.#goalEvidence.delete(ref);
    }
    for (const ref of this.#persistedGoalEvidenceRefs) {
      if (!retained.has(ref)) this.#persistedGoalEvidenceRefs.delete(ref);
    }
  }

  #rememberActionTransactionIdentity(identity: ActionTransactionIdentity): void {
    const existing = this.#actionTransactionIdentities.get(identity.transaction_id);
    if (existing && existing.identity_sha256 !== identity.identity_sha256) {
      throw new Error(
        `Action transaction identity was rebound: ${identity.transaction_id}`
      );
    }
    if (!existing) {
      this.#actionTransactionIdentities.set(
        identity.transaction_id,
        structuredClone(identity)
      );
    }
  }

  async #persistGoalEvidence(refs: readonly string[]): Promise<void> {
    const artifacts = [...new Set(refs)].flatMap((ref) => {
      const artifact = this.#requiredGoalEvidence(ref);
      return this.#persistedGoalEvidenceRefs.has(ref) ? [] : [artifact];
    });
    if (artifacts.length === 0) return;
    await this.#store.appendMany("goal_evidence", artifacts.map((artifact) => json(artifact)));
    for (const artifact of artifacts) {
      this.#persistedGoalEvidenceRefs.add(artifact.evidence.ref);
    }
  }

  async #emitGoalState(
    reason: "candidates_submitted" | "candidate_selected" | "epoch_completed" | "epoch_retired"
  ): Promise<void> {
    await this.emit("humanoid_goal_state_updated", json({
      reason,
      goal_dag: this.#checkpoint.goal_dag,
      active_goal: this.#activeGoal() ?? null,
      active_cycle: this.#checkpoint.active_cycle ?? null,
      goal_progress: this.#checkpoint.goal_progress,
      checker: this.#checkpoint.checker,
      world_frame: this.#checkpoint.world.frame,
      world_revision: this.#checkpoint.world.worldRevision
    }));
  }

  #assertActiveGoalProgress(): void {
    const activeGoal = this.#activeGoal();
    if (!activeGoal) {
      if (this.#checkpoint.goal_dag.status !== "awaiting_model_selection"
        || this.#checkpoint.goal_progress !== null
        || this.#checkpoint.checker !== null) {
        throw new Error(
          "A run without an active Goal epoch cannot retain checker or progress state"
        );
      }
      return;
    }
    if (this.#checkpoint.goal_dag.status !== "active"
      || !this.#checkpoint.goal_progress) {
      throw new Error("An active Goal epoch requires persisted physical progress");
    }
    assertHumanoidGoalProgressIntegrity(
      activeGoal,
      this.#checkpoint.world,
      this.#checkpoint.goal_progress
    );
    if (this.#checkpoint.checker
      && goalSha256(this.#checkpoint.checker.goal) !== goalSha256(activeGoal)) {
      throw new Error("Humanoid checker belongs to another Goal epoch");
    }
  }

  async #withPhysicsPaused<T>(operation: () => Promise<T>): Promise<T> {
    return this.#actionMutex.runExclusive(async () => {
      const resumeClock = this.#continuousPhysicsEnabled;
      await this.#physicsClock.stop();
      try {
        return await operation();
      } finally {
        if (resumeClock
          && this.#checkpoint.status === "running"
          && activeActionExecutions(this.#checkpoint.action_execution_ledger).length === 0) {
          this.#physicsClock.start();
        }
      }
    });
  }

  #node(agentId: string): TaskNode {
    const node = this.#checkpoint.nodes[agentId];
    if (!node) throw new Error(`Unknown humanoid hierarchy node: ${agentId}`);
    return node;
  }

  async #refreshWorldPersistenceState(): Promise<void> {
    const captured = await this.#world.capturePersistenceState();
    this.#applyWorldPersistenceState(captured);
  }

  #applyWorldPersistenceState(
    captured: Awaited<ReturnType<HumanoidWorld["capturePersistenceState"]>>
  ): void {
    const activeGoal = this.#activeGoal();
    if (activeGoal && this.#checkpoint.goal_progress) {
      const advanced = advanceHumanoidGoal(
        activeGoal,
        this.#scenario,
        captured.world,
        this.#checkpoint.goal_progress
      );
      this.#checkpoint.goal_progress = advanced.progress;
      this.#checkpoint.checker = advanced.checker;
    } else if (!activeGoal) {
      this.#checkpoint.goal_progress = null;
      this.#checkpoint.checker = null;
    }
    this.#checkpoint.world = captured.world;
    this.#checkpoint.world_checkpoint = captured.worldCheckpoint;
  }

  async #persist(refreshWorld = true): Promise<void> {
    if (refreshWorld) {
      const captured = await this.#world.capturePersistenceState();
      // Apply and clone in the same continuation. The continuous authority
      // publisher may advance the display snapshot on the next task, but it
      // cannot split this persisted world/world-checkpoint cut.
      this.#applyWorldPersistenceState(captured);
    }
    this.#checkpoint.action_runtime_state = this.#actions.persistenceState();
    this.#assertActiveGoalProgress();
    this.#checkpoint.updated_at = new Date().toISOString();
    const snapshot = structuredClone(this.#checkpoint);
    const write = this.#checkpointWriteTail
      .catch(() => undefined)
      .then(() => this.#store.writeCheckpoint(snapshot));
    this.#checkpointWriteTail = write;
    await write;
  }
}

const PROVIDER_USAGE_SCAN_PAGE = 256;

async function latestProviderModelUsage(
  store: RunStore
): Promise<ModelUsageState | undefined> {
  const tail = await store.readJournalTail("provider", PROVIDER_USAGE_SCAN_PAGE);
  let latest = findLatestProviderModelUsage(tail.entries);
  let before = Math.max(0, tail.total - tail.entries.length);
  while (!latest && before > 0) {
    const from = Math.max(0, before - PROVIDER_USAGE_SCAN_PAGE);
    const page = await store.readJournalPage("provider", from, before - from);
    latest = findLatestProviderModelUsage(page.entries);
    before = from;
  }
  return latest;
}

function findLatestProviderModelUsage(
  records: readonly JsonValue[]
): ModelUsageState | undefined {
  for (let index = records.length - 1; index >= 0; index -= 1) {
    const record = records[index];
    if (record === null || typeof record !== "object" || Array.isArray(record)) continue;
    if (!("model_usage" in record)) continue;
    return ModelUsageStateSchema.parse(record.model_usage);
  }
  return undefined;
}

function reconcileModelUsage(
  checkpoint: ModelUsageState,
  providerJournal: ModelUsageState
): ModelUsageState {
  const journalDominates = modelUsageDominates(providerJournal, checkpoint);
  const checkpointDominates = modelUsageDominates(checkpoint, providerJournal);
  if (!journalDominates && !checkpointDominates) {
    throw new Error("Provider journal model usage conflicts with the humanoid checkpoint");
  }
  if (journalDominates && !checkpointDominates) return structuredClone(providerJournal);
  if (checkpointDominates && !journalDominates) return structuredClone(checkpoint);
  return (providerJournal.updated_at ?? "") > (checkpoint.updated_at ?? "")
    ? structuredClone(providerJournal)
    : structuredClone(checkpoint);
}

function modelUsageDominates(left: ModelUsageState, right: ModelUsageState): boolean {
  if (!usageTotalsDominate(left.total, right.total)) return false;
  for (const [agentId, totals] of Object.entries(right.by_agent)) {
    const candidate = left.by_agent[agentId];
    if (!candidate || !usageTotalsDominate(candidate, totals)) return false;
  }
  return true;
}

function usageTotalsDominate(
  left: ModelUsageState["total"],
  right: ModelUsageState["total"]
): boolean {
  return (Object.keys(right) as Array<keyof ModelUsageState["total"]>).every(
    (key) => left[key] >= right[key]
  );
}

function isHumanoidPlanningReceipt(receipt: HumanoidActionReceipt): boolean {
  return receipt.action === "plan_humanoid_skill"
    || receipt.action === "plan_whole_body_motion"
    || receipt.action === "plan_whole_body_motion_candidates"
    || receipt.action === "plan_humanoid_navigation";
}
