import { randomUUID } from "node:crypto";
import { Mutex } from "async-mutex";
import {
  EmptyContextMemoryState,
  type ContextCompactionSummary,
  type ContextMemoryState,
  type Goal,
  type JsonValue,
  type Scenario,
  type TaskNode
} from "../../domain/schema.js";
import {
  goalConstraintSha256,
  goalSha256
} from "../../domain/goal-identity.js";
import {
  autonomousCycleRef,
  createActiveAutonomousCycle,
  sameAutonomousCycle,
  type ActiveAutonomousCycle,
  type AutonomousCycleRef
} from "../../domain/autonomous-cycle.js";
import {
  beginHumanoidReplanModelCall,
  finishHumanoidReplanModelCall,
  humanoidReplanBudgetAuthority,
  restoreHumanoidReplanModelCall,
  type HumanoidReplanBudget,
  type HumanoidReplanModelCall
} from "../../domain/humanoid-replan-budget.js";
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
  ModelCallLifecycleRecordSchema,
  modelPayloadSha256,
  type ModelCallLifecycleRecord,
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
  restoreActionExecutionLedger,
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
  HumanoidEmbodiedMemoryStateAnchorSchema,
  HumanoidContextMemoryStateAnchorSchema,
  HumanoidExecutionLedgerStateAnchorSchema,
  HumanoidGoalStateAnchorSchema,
  HumanoidPhysicalStateAnchorSchema,
  humanoidEmbodiedMemoryStateSha256,
  humanoidContextMemoryStateSha256,
  humanoidExecutionLedgerStateSha256,
  humanoidGoalControlState,
  humanoidGoalControlStateSha256,
  humanoidPhysicalStateSha256,
  humanoidActionReceiptsInCommitOrder,
  PersistedHumanoidActionReceiptSchema,
  type HumanoidCheckerResult,
  type HumanoidEmbodiedEpisode,
  type HumanoidEmbodiedExperience,
  type HumanoidRunCheckpoint
} from "../../domain/humanoid-run.js";
import type { RunStore } from "../../persistence/run-store.js";
import type { HumanoidPolicyFrameSink } from "../../world/humanoid/simulation.js";
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
import {
  HumanoidEmbodiedSkillEventSchema,
  type HumanoidEmbodiedSkillEvent
} from "../../world/humanoid/embodied-skill-call.js";
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
  recoverableBlockedGoalEvidence
} from "./goal-retirement-evidence.js";
import {
  assertPendingActionReceipt,
  cycleSummary,
  embodiedActionJournalReceipt,
  json,
  object,
  physicalExecutionReceipt,
  completedPhysicalExecution,
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
import {
  currentAgentHarnessInvocation,
  currentAgentHarnessInvocationChain
} from "../agent-scope.js";
import {
  activeNeuralAuthorityLease,
  appendNeuralPredictionError,
  closeNeuralAuthorityLease,
  consumeNeuralRolloutCertificate,
  consumeNeuralSignals,
  establishNeuralSkillCommitment,
  issueNeuralAuthorityLease,
  issueNeuralRolloutCertificate,
  markNeuralPathwayDispatch,
  neuralPathwayDue,
  pendingNeuralSignals,
  publishNeuralSignal,
  transitionNeuralHarnessPhase,
  transitionNeuralSkillCommitment,
  type NeuralAuthorityLease,
  type NeuralHarnessPhase,
  type NeuralHierarchyState,
  type NeuralPathway,
  type NeuralPlanningAction,
  type NeuralPredictionError,
  type NeuralRolloutCertificate,
  type NeuralSignal,
  type NeuralSignalKind,
  type NeuralSkillCommitment
} from "../../domain/neural-hierarchy.js";
import {
  assertHumanoidNeuralSignalRoute,
  HUMANOID_NEURAL_AGENT_IDS,
  HUMANOID_NEURAL_NODE_BY_ID,
  HUMANOID_NEURAL_SIGNAL_CONTRACTS,
  type HumanoidNeuralAgentId
} from "./neural-hierarchy-contract.js";
import type {
  NeuralWakeAuthority
} from "./neural-hierarchy-scheduler.js";
import {
  recoverDurableHumanoidActionRuntimeState,
  verifyDurableHumanoidActionWindow
} from "./durable-action-authority.js";
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
  | "post_failure_observation"
  | "replan_or_retire"
  | "execute_plan"
  | "post_execution"
  | "complete_cycle";

type HumanoidAuthorityDomain =
  | "physical"
  | "goal"
  | "embodied_memory"
  | "context_memory"
  | "execution_ledger";

type HumanoidPersistOptions = {
  refreshWorld?: boolean;
  authorityDomains?: readonly HumanoidAuthorityDomain[] | "all";
  neuralOnly?: boolean;
};

const ALL_HUMANOID_AUTHORITY_DOMAINS: readonly HumanoidAuthorityDomain[] = [
  "physical",
  "goal",
  "embodied_memory",
  "context_memory",
  "execution_ledger"
];

function compactReplanAttemptInProgress(
  budget: HumanoidReplanBudget,
  committedActions: HumanoidRunCheckpoint["committed_actions"],
  activeCycle: ActiveAutonomousCycle
): boolean {
  const latestDecisionIndex = budget.model_calls.findLastIndex(
    (call) => call.role === "replan_decision"
  );
  if (latestDecisionIndex < 0) return false;
  const decision = budget.model_calls[latestDecisionIndex]!;
  const nextFailure = humanoidActionReceiptsInCommitOrder(committedActions)
    .filter((receipt) => sameAutonomousCycle(receipt.cycle, activeCycle))
    .find((receipt) => receipt.committedAt > decision.started_at
      && ((!receipt.accepted && isHumanoidPlanningReceipt(receipt))
        || (physicalExecutionReceipt(receipt)
          && !completedPhysicalExecution(receipt))));
  return nextFailure === undefined;
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
  readonly #neuralStateMutex = new Mutex();
  readonly #persistMutex = new Mutex();
  #checkpoint: HumanoidRunCheckpoint;
  #durableCheckpoint: HumanoidRunCheckpoint;
  #scenario: Scenario;
  #physicalStateAnchorTail: Promise<void> = Promise.resolve();
  #goalStateAnchorTail: Promise<void> = Promise.resolve();
  #embodiedMemoryStateAnchorTail: Promise<void> = Promise.resolve();
  #contextMemoryStateAnchorTail: Promise<void> = Promise.resolve();
  #executionLedgerStateAnchorTail: Promise<void> = Promise.resolve();
  #physicalAnchorOrphanRecoveryPending = true;
  #goalAnchorOrphanRecoveryPending = true;
  #memoryAnchorOrphanRecoveryPending = true;
  #contextAnchorOrphanRecoveryPending = true;
  #executionLedgerAnchorOrphanRecoveryPending = true;
  #recoveredAcknowledgedActionCommits: Array<
    ReturnType<typeof stageActionCommit>["pending"][string]
  > = [];
  #continuousPhysicsEnabled = false;
  #modelAuthority: HumanoidModelAuthority | undefined;
  #goalEvidence = new Map<string, GoalEvidenceArtifact>();
  #persistedGoalEvidenceRefs = new Set<string>();
  #contextGoalEvidenceRefs = new Map<string, string>();
  #actionTransactionIdentities = new Map<string, ActionTransactionIdentity>();
  #actionTransactionIdentitiesLoaded = false;
  #durableActionReceiptCache = new Map<string, HumanoidActionReceipt>();
  readonly #activeModelCallsByAgent = new Map<string, number>();

  constructor(input: {
    store: RunStore;
    goal: Goal;
    world: HumanoidWorld;
    checkpoint: HumanoidRunCheckpoint;
    eventSink?: RuntimeEventSink;
    policyFrameSink?: HumanoidPolicyFrameSink;
    signal?: AbortSignal;
  }) {
    this.#store = input.store;
    this.#missionGoal = structuredClone(input.goal);
    this.#world = input.world;
    this.#checkpoint = reconcileHumanoidHierarchyCapabilities(input.checkpoint);
    this.#durableCheckpoint = structuredClone(this.#checkpoint);
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
      neuralHierarchyEpochId: this.#checkpoint.neural_hierarchy_state.epoch_id,
      frameSink: (frame) => this.#physicalExecution.recordFrame(frame, "execution"),
      ...(input.policyFrameSink
        ? { policyFrameSink: input.policyFrameSink }
        : {}),
      physicalFrameSink: (cut) => this.#physicalExecution.recordPhysicalCut(cut),
      physicalExecutionFrameOffset: (transactionId) => (
        this.#physicalExecution.executionFrameOffset(transactionId)
      ),
      completedPhysicalPlanFrameCount: (transactionId) => (
        this.#physicalExecution.executionCompletedPlanFrameCount(transactionId)
      ),
      completedPhysicalPlanCount: (transactionId) => (
        this.#physicalExecution.executionCompletedPlanCount(transactionId)
      ),
      skillEventSink: (event) => this.#emitHumanoidSkillEvent(event),
      receiptSink: (receipt) => this.#commitReceipt(receipt),
      beforePhysicalExecution: (intent) => this.#physicalExecution.admit(intent),
      receiptNormalizer: async (receipt) => ({
        ...await this.#physicalExecution.normalizeReceipt(receipt),
        commitSequence: this.#nextActionCommitSequence()
      }),
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

  neuralHierarchyState(): NeuralHierarchyState {
    return structuredClone(this.#checkpoint.neural_hierarchy_state);
  }

  neuralHarnessPhase(): NeuralHierarchyState["harness_phase"] {
    return structuredClone(this.#checkpoint.neural_hierarchy_state.harness_phase);
  }

  activeNeuralAuthorityLease(input: {
    targetChildNodeId: string;
    signalKind?: NeuralSignalKind;
  }): NeuralAuthorityLease | undefined {
    const phase = this.#checkpoint.neural_hierarchy_state.harness_phase;
    return activeNeuralAuthorityLease({
      state: this.#checkpoint.neural_hierarchy_state,
      targetChildNodeId: input.targetChildNodeId,
      worldRevision: this.#world.snapshot().worldRevision,
      ...(input.signalKind === undefined ? {} : { signalKind: input.signalKind }),
      goalEpochId: phase.goal_epoch_id,
      commitmentId: phase.commitment_id,
      liveInvocationIds: this.#liveNeuralInvocationIds()
    });
  }

  neuralNodeEnabled(input: {
    nodeId: string;
    phases: readonly NeuralHarnessPhase[];
    signalKinds?: readonly NeuralSignalKind[];
    requireCommitment?: boolean;
  }): boolean {
    const state = this.#checkpoint.neural_hierarchy_state;
    const phase = state.harness_phase;
    if (!input.phases.includes(phase.phase)) return false;
    if (input.requireCommitment && phase.commitment_id === null) return false;
    const descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(input.nodeId);
    if (!descriptor) return false;
    if (descriptor.parentKey === null) return input.nodeId === this.rootAgentId;
    const authorityPath = this.#activeNeuralAuthorityPath(descriptor.id);
    const lease = authorityPath?.at(-1)?.lease;
    if (!lease) return false;
    return input.signalKinds === undefined
      || input.signalKinds.some((kind) => lease.allowed_signal_kinds.includes(kind));
  }

  resolveNeuralWakeAuthority(
    requestedTargetNodeId: string
  ): NeuralWakeAuthority {
    let descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(requestedTargetNodeId);
    if (!descriptor) throw new Error(`Unknown neural wake target: ${requestedTargetNodeId}`);
    while (descriptor.parentKey !== null) {
      if (descriptor.executionKind === "model_agent") {
        const authorityPath = this.#activeNeuralAuthorityPath(descriptor.id);
        const direct = authorityPath?.at(-1);
        if (authorityPath && direct) {
          return {
            targetNodeId: descriptor.id,
            parentNodeId: direct.parentNodeId,
            authorityLeaseId: direct.lease.lease_id,
            authorityPath: authorityPath.map(({ parentNodeId, childNodeId, lease }) => ({
              parentNodeId,
              childNodeId,
              authorityLeaseId: lease.lease_id
            }))
          };
        }
      }
      descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(
        HUMANOID_NEURAL_AGENT_IDS[descriptor.parentKey]
      );
      if (!descriptor) throw new Error("Neural control tree has a missing ancestor");
    }
    return {
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      parentNodeId: null,
      authorityLeaseId: null,
      authorityPath: []
    };
  }

  /**
   * Resolve a live, invocation-linked chain from Executive to one descendant.
   * A direct child lease is insufficient: every intermediate Manager must
   * still be running under the lease issued by its own structural parent.
   */
  #activeNeuralAuthorityPath(targetNodeId: HumanoidNeuralAgentId): Array<{
    parentNodeId: HumanoidNeuralAgentId;
    childNodeId: HumanoidNeuralAgentId;
    lease: NeuralAuthorityLease;
  }> | undefined {
    const reversed: Array<{
      parentNodeId: HumanoidNeuralAgentId;
      childNodeId: HumanoidNeuralAgentId;
      lease: NeuralAuthorityLease;
    }> = [];
    let child = HUMANOID_NEURAL_NODE_BY_ID.get(targetNodeId);
    while (child?.parentKey !== null && child?.parentKey !== undefined) {
      const parentNodeId = HUMANOID_NEURAL_AGENT_IDS[child.parentKey];
      const lease = this.activeNeuralAuthorityLease({ targetChildNodeId: child.id });
      if (!lease || lease.issuing_parent_node_id !== parentNodeId) return undefined;
      reversed.push({ parentNodeId, childNodeId: child.id, lease });
      child = HUMANOID_NEURAL_NODE_BY_ID.get(parentNodeId);
    }
    if (!child || child.id !== HUMANOID_NEURAL_AGENT_IDS.executive) return undefined;
    const path = reversed.reverse();
    for (let index = 1; index < path.length; index += 1) {
      const parentLease = path[index - 1]!.lease;
      const childLease = path[index]!.lease;
      if (childLease.parent_invocation_id !== parentLease.invocation_id
        || childLease.parent_episode_id !== parentLease.invocation_id) {
        return undefined;
      }
    }
    return path;
  }

  async reconcileNeuralHarnessPhase(): Promise<NeuralHierarchyState["harness_phase"]> {
    const checkpoint = this.#checkpoint;
    const state = checkpoint.neural_hierarchy_state;
    const current = state.harness_phase;
    let phase: NeuralHarnessPhase;
    let reason: string;
    if (checkpoint.status === "succeeded" || checkpoint.status === "failed") {
      phase = "terminal";
      reason = "run_terminal";
    } else if (checkpoint.goal_dag.status !== "active") {
      phase = "goal_valuation";
      reason = "no_active_goal_epoch";
    } else {
      const coordinator = this.coordinatorPhase();
      if (coordinator === "complete_satisfied_goal" || coordinator === "complete_cycle") {
        phase = "cycle_completion";
        reason = "runtime_completion_barrier_ready";
      } else if (coordinator === "post_execution") {
        phase = "feedback";
        reason = "post_execution_observation_required";
      } else if (coordinator === "post_failure_observation"
        || coordinator === "replan_or_retire") {
        if (activeRecoveryReplacementCommitment(state)) {
          phase = "motor_assessment";
          reason = "recovery_replacement_commitment_requires_assessment";
        } else if (recoveryEscalationAwaitsGoalValuation(state)) {
          phase = "goal_valuation";
          reason = "recovery_escalation_requires_goal_valuation";
        } else {
          phase = "recovery";
          reason = "runtime_failure_feedback_required";
        }
      } else if (coordinator === "execute_plan") {
        const activeCommitment = state.active_skill_commitment;
        const activeCertificates = Object.values(state.rollout_certificates).filter(
          (certificate) => certificate.status === "active"
            && certificate.commitment_id === activeCommitment?.commitment_id
        );
        if (activeCommitment?.state === "executing"
          && activeCertificates.length === 1) {
          phase = "execution";
          reason = "certified_commitment_requires_serial_execution";
        } else {
          phase = "rollout_review";
          reason = "accepted_plan_requires_action_selection_authorization";
        }
      } else if (current.phase === "bootstrapping"
        || current.phase === "goal_valuation"
        || current.phase === "cycle_completion") {
        phase = "perception";
        reason = "active_goal_requires_current_perception";
      } else {
        return structuredClone(current);
      }
    }
    if (phase === current.phase) return structuredClone(current);
    return this.transitionNeuralHarnessPhase({
      phase,
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason
    });
  }

  neuralPathwayDue(pathway: NeuralPathway, now = Date.now()): boolean {
    return neuralPathwayDue(
      this.#checkpoint.neural_hierarchy_state,
      pathway,
      now
    );
  }

  pendingNeuralSignals(input: {
    targetNodeId?: string;
    kinds?: readonly NeuralSignalKind[];
    invocationId?: string;
  } = {}): NeuralSignal[] {
    return pendingNeuralSignals({
      state: this.#checkpoint.neural_hierarchy_state,
      ...(input.targetNodeId === undefined
        ? {}
        : { targetNodeId: input.targetNodeId }),
      ...(input.kinds === undefined ? {} : { kinds: input.kinds }),
      worldRevision: this.#world.snapshot().worldRevision,
      liveInvocationIds: this.#liveNeuralInvocationIds()
    }).filter((signal) => (
      (input.invocationId === undefined || signal.invocation_id === input.invocationId)
    ));
  }

  currentWorldRevision(): number {
    return this.#world.snapshot().worldRevision;
  }

  neuralExecutionAvailable(): boolean {
    return this.executorDelegationAvailable();
  }

  async publishNeuralSignal(input: {
    kind: NeuralSignalKind;
    pathway: NeuralPathway;
    direction: "descending" | "ascending" | "reentrant";
    sourceNodeId: string;
    targetNodeId: string;
    ttlRevisions: number;
    priority: number;
    causalParentIds?: readonly string[];
    authorityLeaseId?: string | null;
    sourceAuthorityLeaseId?: string | null;
    invocationId?: string;
    parentInvocationId?: string | null;
    parentEpisodeId?: string;
    payload: JsonValue;
  }): Promise<NeuralSignal> {
    return this.#neuralStateMutex.runExclusive(async () => {
      this.#assertRunAcceptsDecisions();
      const route = assertHumanoidNeuralSignalRoute({
        ...input,
        kind: input.kind
      });
      const world = this.#world.snapshot();
      let authorityLeaseId = input.authorityLeaseId ?? null;
      const sourceAuthorityLeaseId = input.sourceAuthorityLeaseId ?? null;
      let provenanceLease: NeuralAuthorityLease | undefined;
      if (input.direction === "descending") {
        if (sourceAuthorityLeaseId !== null) {
          throw new Error("Descending neural signal cannot claim source-lease provenance");
        }
        const candidate = authorityLeaseId
          ? this.#checkpoint.neural_hierarchy_state.authority_leases[authorityLeaseId]
          : this.activeNeuralAuthorityLease({
              targetChildNodeId: route.target.id,
              signalKind: input.kind
            });
        const liveInvocationIds = this.#liveNeuralInvocationIds();
        const invocationIsLive = candidate
          ? liveInvocationIds.includes(candidate.invocation_id)
          : false;
        const lease = candidate
          && candidate.status === "active"
          && world.worldRevision >= candidate.issued_world_revision
          && (invocationIsLive
            || (world.worldRevision <= candidate.expires_world_revision
              && Date.now() <= Date.parse(candidate.expires_at)))
          ? candidate
          : undefined;
        if (!lease || lease.status !== "active"
          || lease.issuing_parent_node_id !== route.source.id
          || lease.target_child_node_id !== route.target.id
          || !lease.allowed_signal_kinds.includes(input.kind)) {
          throw new Error(
            `Descending neural signal lacks active single-parent authority: ${route.source.id} -> ${route.target.id}`
          );
        }
        authorityLeaseId = lease.lease_id;
        provenanceLease = lease;
      } else {
        if (authorityLeaseId !== null) {
          throw new Error("Feedback cannot carry current child authority");
        }
        if (sourceAuthorityLeaseId === null) {
          throw new Error("Feedback must retain source-lease provenance");
        }
        const lease = this.#checkpoint.neural_hierarchy_state
          .authority_leases[sourceAuthorityLeaseId];
        const endpointOwnsLease = lease
          && (lease.issuing_parent_node_id === route.source.id
            || lease.target_child_node_id === route.source.id);
        if (!lease || !endpointOwnsLease
          || (lease.status !== "active" && lease.status !== "suspended")) {
          throw new Error("Ascending or reentrant signal references invalid authority lease");
        }
        provenanceLease = lease;
      }
      if (!provenanceLease) {
        throw new Error("Neural signal has no invocation-scoped authority lease");
      }
      const invocationId = input.invocationId ?? provenanceLease.invocation_id;
      if (invocationId !== provenanceLease.invocation_id) {
        throw new Error("Neural signal invocation does not match its authority lease");
      }
      const parentInvocationId = input.parentInvocationId
        ?? provenanceLease.parent_invocation_id;
      if (parentInvocationId !== provenanceLease.parent_invocation_id) {
        throw new Error(
          "Neural signal parent invocation does not match its authority lease"
        );
      }
      const parentEpisodeId = input.parentEpisodeId
        ?? provenanceLease.parent_episode_id;
      if (parentEpisodeId !== provenanceLease.parent_episode_id) {
        throw new Error("Neural signal parent episode does not match its authority lease");
      }
      const causalParents = (input.causalParentIds ?? []).map(
        (signalId) => this.#checkpoint.neural_hierarchy_state.signals[signalId]
      );
      if (causalParents.some((signal) => signal === undefined)) {
        throw new Error("Neural signal references an unknown causal parent");
      }
      const published = publishNeuralSignal(
        this.#checkpoint.neural_hierarchy_state,
        {
          kind: input.kind,
          pathway: input.pathway,
          direction: input.direction,
          sourceNodeId: route.source.id,
          sourceLayer: route.source.layer,
          targetNodeId: route.target.id,
          targetLayer: route.target.layer,
          worldFrame: world.frame,
          worldRevision: world.worldRevision,
          ttlRevisions: input.ttlRevisions,
          priority: input.priority,
          invocationId,
          parentInvocationId,
          parentEpisodeId,
          ...(input.causalParentIds === undefined
            ? {}
            : { causalParentIds: input.causalParentIds }),
          authorityLeaseId,
          sourceAuthorityLeaseId,
          payload: input.payload,
          liveInvocationIds: this.#liveNeuralInvocationIds()
        }
      );
      this.#checkpoint.neural_hierarchy_state = published.state;
      await this.#persistNeuralState();
      await this.emit("neural_signal_published", json({
        signal: published.signal,
        automatic_actuation: false
      }));
      return published.signal;
    });
  }

  async consumeNeuralSignals(
    consumerNodeId: string,
    signalIds: readonly string[]
  ): Promise<void> {
    await this.#neuralStateMutex.runExclusive(async () => {
      const uniqueSignalIds = [...new Set(signalIds)];
      const pendingById = new Map(this.pendingNeuralSignals({
        targetNodeId: consumerNodeId
      }).map((signal) => [signal.signal_id, signal] as const));
      for (const signalId of uniqueSignalIds) {
        if (!pendingById.has(signalId)) {
          throw new Error(
            `Neural signal is absent, stale, consumed, or owned by another node: ${signalId}`
          );
        }
      }
      this.#checkpoint.neural_hierarchy_state = consumeNeuralSignals(
        this.#checkpoint.neural_hierarchy_state,
        uniqueSignalIds
      );
      await this.#persistNeuralState();
      await this.emit("neural_signals_consumed", json({
        consumer_node_id: consumerNodeId,
        signal_ids: uniqueSignalIds,
        automatic_actuation: false
      }));
    });
  }

  async markNeuralPathway(
    pathway: NeuralPathway,
    phase: "started" | "completed"
  ): Promise<void> {
    await this.#neuralStateMutex.runExclusive(async () => {
      this.#checkpoint.neural_hierarchy_state = markNeuralPathwayDispatch(
        this.#checkpoint.neural_hierarchy_state,
        pathway,
        phase
      );
      await this.#persistNeuralState();
    });
  }

  async transitionNeuralHarnessPhase(input: {
    phase: NeuralHarnessPhase;
    enteredByNodeId: string;
    reason: string;
    goalEpochId?: string | null;
    commitmentId?: string | null;
  }): Promise<NeuralHierarchyState["harness_phase"]> {
    return this.#neuralStateMutex.runExclusive(async () => {
      this.#assertRunAcceptsDecisions();
      const state = this.#checkpoint.neural_hierarchy_state;
      const goalEpochId = input.goalEpochId === undefined
        ? this.#checkpoint.goal_dag.current_epoch_id
        : input.goalEpochId;
      const activeCommitment = state.active_skill_commitment;
      const commitmentId = input.commitmentId === undefined
        ? activeCommitment
          && !["completed", "failed", "released"].includes(activeCommitment.state)
            ? activeCommitment.commitment_id
            : null
        : input.commitmentId;
      this.#checkpoint.neural_hierarchy_state = transitionNeuralHarnessPhase(
        state,
        {
          ...input,
          goalEpochId,
          commitmentId,
          worldRevision: this.#world.snapshot().worldRevision,
          liveInvocationIds: this.#liveNeuralInvocationIds()
        }
      );
      await this.#persistNeuralState();
      await this.emit("neural_harness_phase_transitioned", json({
        phase: this.#checkpoint.neural_hierarchy_state.harness_phase,
        automatic_actuation: false
      }));
      return structuredClone(this.#checkpoint.neural_hierarchy_state.harness_phase);
    });
  }

  async issueNeuralAuthorityLease(input: {
    issuingParentNodeId: string;
    targetChildNodeId: string;
    allowedSignalKinds: readonly NeuralSignalKind[];
    correctionScope?: NeuralAuthorityLease["correction_scope"];
    ttlRevisions?: number;
    ttlMs?: number;
    exclusive?: boolean;
    suspendLeaseIds?: readonly string[];
    invocationId?: string;
    parentInvocationId?: string | null;
    parentEpisodeId: string;
  }): Promise<NeuralAuthorityLease> {
    const child = HUMANOID_NEURAL_NODE_BY_ID.get(input.targetChildNodeId);
    if (!child || child.parentKey === null
      || HUMANOID_NEURAL_AGENT_IDS[child.parentKey] !== input.issuingParentNodeId) {
      throw new Error("Authority lease issuer is not the child's structural parent");
    }
    const invocation = currentAgentHarnessInvocation();
    if (!invocation
      || invocation.agentId !== child.id
      || invocation.parentAgentId !== input.issuingParentNodeId
      || invocation.invocationId !== input.invocationId
      || invocation.parentInvocationId === null
      || invocation.parentInvocationId !== input.parentInvocationId
      || invocation.parentInvocationId !== input.parentEpisodeId) {
      throw new Error(
        `Authority lease must be opened inside the direct child SDK episode: ${input.issuingParentNodeId} -> ${child.id}`
      );
    }
    if (input.issuingParentNodeId !== HUMANOID_NEURAL_AGENT_IDS.executive) {
      const parentAuthority = this.#activeNeuralAuthorityPath(
        input.issuingParentNodeId as HumanoidNeuralAgentId
      );
      const parentLease = parentAuthority?.at(-1)?.lease;
      if (!parentLease
        || parentLease.invocation_id !== invocation.parentInvocationId) {
        throw new Error(
          `Authority lease issuer has no live Executive-owned invocation chain: ${input.issuingParentNodeId}`
        );
      }
    }
    const route = HUMANOID_NEURAL_SIGNAL_CONTRACTS.find((contract) => (
      contract.sourceAgentId === input.issuingParentNodeId
        && contract.targetAgentId === input.targetChildNodeId
        && contract.direction === "descending"
    ));
    if (!route || input.allowedSignalKinds.some(
      (kind) => !route.signalKinds.includes(kind)
    )) {
      throw new Error("Authority lease signal scope exceeds the parent-child contract");
    }
    const correctionScopeRank: Readonly<Record<
      NeuralAuthorityLease["correction_scope"] | "none",
      number
    >> = {
      none: 0,
      ordinary: 0,
      local: 1,
      pathway: 2,
      supervisory: 3
    };
    const requestedScope = input.correctionScope ?? "ordinary";
    if (requestedScope !== "ordinary"
      && correctionScopeRank[requestedScope]
        > correctionScopeRank[child.maximumCorrectionScope]) {
      throw new Error(
        `Authority lease correction scope exceeds the child contract: ${child.id}`
      );
    }
    if ((child.orchestrationKind === "exclusive_lease_episode")
      !== (input.exclusive === true)) {
      throw new Error(
        `Exclusive lease mode does not match the child orchestration contract: ${child.id}`
      );
    }
    return this.#neuralStateMutex.runExclusive(async () => {
      this.#assertRunAcceptsDecisions();
      const worldRevision = this.#world.snapshot().worldRevision;
      const phase = this.#checkpoint.neural_hierarchy_state.harness_phase;
      const issued = issueNeuralAuthorityLease(
        this.#checkpoint.neural_hierarchy_state,
        {
          ...input,
          goalEpochId: phase.goal_epoch_id,
          commitmentId: phase.commitment_id,
          worldRevision,
          liveInvocationIds: this.#liveNeuralInvocationIds(),
          expiresWorldRevision: worldRevision + (
            input.ttlRevisions ?? this.#neuralLeaseRevisionHorizon(input.ttlMs ?? 120_000)
          ),
          expiresAt: new Date(Date.now() + (input.ttlMs ?? 120_000)).toISOString()
        }
      );
      this.#checkpoint.neural_hierarchy_state = issued.state;
      await this.#persistNeuralState();
      await this.emit("neural_authority_lease_issued", json({
        lease: issued.lease,
        automatic_actuation: false
      }));
      return issued.lease;
    });
  }

  async closeNeuralAuthorityLease(input: {
    leaseId: string;
    closedByNodeId: string;
    reason: string;
    status?: "closed" | "revoked" | "expired";
    resumeSuspended?: boolean;
  }): Promise<void> {
    await this.#neuralStateMutex.runExclusive(async () => {
      this.#checkpoint.neural_hierarchy_state = closeNeuralAuthorityLease(
        this.#checkpoint.neural_hierarchy_state,
        {
          ...input,
          worldRevision: this.#world.snapshot().worldRevision,
          liveInvocationIds: this.#liveNeuralInvocationIds()
        }
      );
      await this.#persistNeuralState();
      await this.emit("neural_authority_lease_closed", json({
        lease_id: input.leaseId,
        closed_by_node_id: input.closedByNodeId,
        reason: input.reason,
        automatic_actuation: false
      }));
    });
  }

  #neuralLeaseRevisionHorizon(ttlMs: number): number {
    const controlStepSeconds = this.#world.snapshot().robot.controller.controlStepSeconds;
    const revisions = Math.ceil(ttlMs / (controlStepSeconds * 1_000));
    return Math.max(1, Math.min(revisions + 4, 1_000_000));
  }

  #liveNeuralInvocationIds(): string[] {
    return currentAgentHarnessInvocationChain().map(
      (invocation) => invocation.invocationId
    );
  }

  async establishNeuralSkillCommitment(input: {
    ownerNodeId: string;
    goalEpochId: string;
    skill: string;
    terminationContract: JsonValue;
    sourceSignalIds: readonly string[];
  }): Promise<NeuralSkillCommitment> {
    if (input.ownerNodeId !== HUMANOID_NEURAL_AGENT_IDS.actionSelection) {
      throw new Error("Only Action Selection may establish a neural skill commitment");
    }
    return this.#neuralStateMutex.runExclusive(async () => {
      this.#assertRunAcceptsDecisions();
      const worldRevision = this.#world.snapshot().worldRevision;
      const established = establishNeuralSkillCommitment(
        this.#checkpoint.neural_hierarchy_state,
        {
          ...input,
          worldRevision
        }
      );
      this.#checkpoint.neural_hierarchy_state = established.state;
      await this.#persistNeuralState();
      await this.emit("neural_skill_commitment_established", json({
        commitment: established.commitment,
        automatic_actuation: false
      }));
      return established.commitment;
    });
  }

  async transitionNeuralSkillCommitment(input: {
    ownerNodeId: string;
    commitmentId: string;
    state: NeuralSkillCommitment["state"];
    sourceSignalIds?: readonly string[];
  }): Promise<NeuralSkillCommitment> {
    if (input.ownerNodeId !== HUMANOID_NEURAL_AGENT_IDS.actionSelection) {
      throw new Error("Only Action Selection may transition a neural skill commitment");
    }
    return this.#neuralStateMutex.runExclusive(async () => {
      this.#assertRunAcceptsDecisions();
      const transitioned = transitionNeuralSkillCommitment(
        this.#checkpoint.neural_hierarchy_state,
        {
          ...input,
          worldRevision: this.#world.snapshot().worldRevision
        }
      );
      this.#checkpoint.neural_hierarchy_state = transitioned.state;
      await this.#persistNeuralState();
      await this.emit("neural_skill_commitment_transitioned", json({
        commitment: transitioned.commitment,
        automatic_actuation: false
      }));
      return transitioned.commitment;
    });
  }

  async recordNeuralPredictionError(input: {
    observerNodeId: string;
    sourceSignalId: string;
    magnitude: number;
    tolerance: number;
    correctionScope: NeuralPredictionError["correction_scope"];
    detail: JsonValue;
  }): Promise<NeuralPredictionError> {
    if (input.observerNodeId !== HUMANOID_NEURAL_AGENT_IDS.predictive
      && input.observerNodeId !== HUMANOID_NEURAL_AGENT_IDS.reflex) {
      throw new Error(
        "Only Predictive Critic or the Reflex loop may record prediction error"
      );
    }
    return this.#neuralStateMutex.runExclusive(async () => {
      const recorded = appendNeuralPredictionError(
        this.#checkpoint.neural_hierarchy_state,
        {
          ...input,
          worldRevision: this.#world.snapshot().worldRevision
        }
      );
      this.#checkpoint.neural_hierarchy_state = recorded.state;
      await this.#persistNeuralState();
      await this.emit("neural_prediction_error_recorded", json({
        prediction_error: recorded.error,
        automatic_actuation: false
      }));
      return recorded.error;
    });
  }

  async issueNeuralRolloutCertificate(input: {
    issuedByNodeId: string;
    commitmentId: string;
    goalEpochId: string;
    planningTransactionId: string;
    planningAction: NeuralPlanningAction;
    rolloutSignalId: string;
    predictiveSignalId: string;
    rolloutPayloadSha256: string;
    rolloutInvocationId: string;
    predictiveInvocationId: string;
    ttlRevisions?: number;
  }): Promise<NeuralRolloutCertificate> {
    if (input.issuedByNodeId !== HUMANOID_NEURAL_AGENT_IDS.predictive) {
      throw new Error("Only Predictive Critic may issue a rollout certificate");
    }
    return this.#neuralStateMutex.runExclusive(async () => {
      const worldRevision = this.#world.snapshot().worldRevision;
      const issued = issueNeuralRolloutCertificate(
        this.#checkpoint.neural_hierarchy_state,
        {
          ...input,
          worldRevision,
          expiresWorldRevision: worldRevision + (input.ttlRevisions ?? 10_000)
        }
      );
      this.#checkpoint.neural_hierarchy_state = issued.state;
      await this.#persistNeuralState();
      await this.emit("neural_rollout_certificate_issued", json({
        rollout_certificate: issued.certificate,
        automatic_actuation: false
      }));
      return issued.certificate;
    });
  }

  async initializeGoalAutonomy(manifest: AgentManifest): Promise<void> {
    await this.#verifyExistingPhysicalStateAnchor();
    await this.#verifyExistingGoalStateAnchor();
    await this.#verifyExistingEmbodiedMemoryStateAnchor();
    await this.#verifyExistingContextMemoryStateAnchor();
    await this.#verifyExistingExecutionLedgerStateAnchor();
    await this.#reconcileActionCommits();
    this.#scenario = materializeScenarioChunkDeltaState(
      this.#store.definition.scenario,
      await this.#store.readScenarioChunkDeltaState()
    );
    const goalEvidenceRefs = requiredGoalEvidenceRefs(this.#checkpoint);
    const optionalEvidenceRefs = optionalGoalEvidenceRefs(this.#checkpoint);
    const modelCallIds = requiredModelCallIds(this.#checkpoint);
    const optionalAuthorityIds = optionalModelCallIds(this.#checkpoint);
    const activeExecutionPlanningTransactionIds = new Set(
      Object.values(this.#checkpoint.action_execution_ledger.active).map(
        (entry) => entry.admission.planning_transaction_id
      )
    );
    const skillEventRecoveryCallIds = this.#actions.skillEventRecoveryCallIds(
      activeExecutionPlanningTransactionIds
    );
    const [
      evidence,
      rawModelCalls,
      rawActionIdentities,
      providerModelUsage,
      archivedManifests,
      skillEvents
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
      this.#store.readArchivedAgentManifests(),
      loadDurableHumanoidSkillEvents(
        this.#store,
        skillEventRecoveryCallIds
      )
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
      appendRecord: (record) => appendDurableModelCallLifecycleRecord(
        this.#store,
        record
      )
    });
    const durableActions = await verifyDurableHumanoidActionWindow({
      store: this.#store,
      runId: this.runId,
      receipts: this.#checkpoint.committed_actions,
      identities: actionTransactionIdentities,
      assertDecisionAuthority: (receipt, toolAuthority) => {
        if (!receipt.decision || !receipt.cycle) {
          throw new Error(
            `Committed action has no durable model decision: ${receipt.transactionId}`
          );
        }
        this.#assertActionDecisionRef(
          receipt.decision,
          receipt.action,
          receipt.input,
          receipt.transactionId,
          receipt.agentId,
          receipt.cycle,
          toolAuthority
        );
      }
    });
    const actionRuntimeRecovery = recoverDurableHumanoidActionRuntimeState({
      receipts: this.#checkpoint.committed_actions,
      proofs: durableActions,
      checkpointState: this.#checkpoint.action_runtime_state,
      neuralHierarchyEpochId: this.#checkpoint.neural_hierarchy_state.epoch_id
    });
    if (actionRuntimeRecovery.state !== null) {
      this.#checkpoint.action_runtime_state = actionRuntimeRecovery.state;
      this.#actions.recoverPersistenceState(actionRuntimeRecovery.state);
    }
    this.#durableActionReceiptCache = new Map(
      [...durableActions].map(([transactionId, proof]) => (
        [transactionId, structuredClone(proof.receipt)] as const
      ))
    );
    const replanBudgetReconciliation = this.#reconcileReplanBudgetFromModelLifecycle(
      rawModelCalls
    );
    const interruptedModelCalls = this.#modelAuthority.startedCalls();
    if (interruptedModelCalls.length > 0) {
      const interruptedAt = new Date().toISOString();
      for (const call of interruptedModelCalls) {
        await this.#modelAuthority.recordFailed(
          call.model_call_id,
          call.agent_id,
          interruptedAt < call.at ? call.at : interruptedAt
        );
        if (this.#checkpoint.active_cycle
          && sameAutonomousCycle(call.cycle, this.#checkpoint.active_cycle)) {
          const update = finishHumanoidReplanModelCall(
            this.#checkpoint.active_cycle.replan_budget,
            {
              modelCallId: call.model_call_id,
              status: "failed",
              at: interruptedAt < call.at ? call.at : interruptedAt
            }
          );
          this.#checkpoint.active_cycle.replan_budget = update.budget;
        }
      }
    }
    if (actionRuntimeRecovery.checkpointRecovered
      || replanBudgetReconciliation.changed
      || interruptedModelCalls.length > 0) {
      await this.#persist();
    }
    if (interruptedModelCalls.length > 0) {
      await this.emit("model_requests_interrupted_by_restart", json({
        model_calls: interruptedModelCalls.map((call) => ({
          model_call_id: call.model_call_id,
          agent_id: call.agent_id,
          ...(call.cycle ? { cycle: call.cycle } : {})
        })),
        automatic_actuation: false
      }));
    }
    this.#actionTransactionIdentities = actionTransactionIdentities;
    this.#actionTransactionIdentitiesLoaded = true;
    this.#actions.restoreSkillEventJournal(skillEvents);
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
    if (!this.#checkpoint.physical_state_anchor
      || !this.#checkpoint.goal_state_anchor
      || !this.#checkpoint.embodied_memory_state_anchor
      || !this.#checkpoint.context_memory_state_anchor
      || !this.#checkpoint.execution_ledger_state_anchor) {
      await this.#persist(false);
    }
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
      const expiredCandidates = Object.values(next.candidates).flatMap((entry) => (
        entry.status === "expired"
          && this.#checkpoint.goal_dag.candidates[entry.candidate_id]?.status === "proposed"
          ? [{
              candidate_sequence: goalCandidateSequence(next, entry.candidate_id)!,
              candidate_id: entry.candidate_id
            }]
          : []
      ));
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
        expired_candidates: expiredCandidates,
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

  async continueGoalEpoch(
    input: Parameters<GoalManagerRuntime["continueGoalEpoch"]>[0],
    authority: GoalToolCallAuthority
  ): Promise<JsonValue> {
    return this.#goalStateMutex.runExclusive(() => this.#withPhysicsPaused(async () => {
      this.#assertRunAcceptsDecisions();
      if (this.coordinatorPhase() !== "replan_or_retire") {
        throw new Error("The active Goal is not awaiting recovery re-evaluation");
      }
      const activeCycle = this.#checkpoint.active_cycle;
      if (!activeCycle?.replan_budget.goal_reevaluation_started) {
        throw new Error("Goal continuation requires exhausted compact replan authority");
      }
      const continuedBy = this.#goalModelSource(
        authority,
        "continue_goal_epoch",
        input
      );
      const evidence = this.#requiredContextGoalEvidence(continuedBy.agent_id);
      await this.#persistGoalEvidence([evidence.ref]);
      const activeGoal = this.#requiredActiveGoal();
      const captured = await this.#world.capturePersistenceState();
      this.#applyWorldPersistenceState(captured);
      const progress = this.#checkpoint.goal_progress;
      if (!progress) throw new Error("Active Goal progress is unavailable");
      const projected = captured.world.frame === progress.last_world_frame
        && captured.world.worldRevision === progress.last_world_revision
        ? {
            progress,
            checker: inspectHumanoidGoal(
              activeGoal,
              this.#scenario,
              captured.world,
              progress
            )
          }
        : advanceHumanoidGoal(activeGoal, this.#scenario, captured.world, progress);
      if (projected.checker.success) {
        throw new Error(
          "A physically satisfied Goal must complete instead of opening another recovery cycle"
        );
      }
      const interruptedCycle = autonomousCycleRef(activeCycle);
      this.#checkpoint.goal_progress = projected.progress;
      this.#checkpoint.checker = projected.checker;
      this.#checkpoint.cycle_index = interruptedCycle.cycle_index;
      this.#checkpoint.last_cycle = json({
        status: "goal_epoch_continued",
        cycle: interruptedCycle,
        reason: input.reason,
        recovery_intent: input.recovery_intent,
        evidence_ref: evidence.ref,
        world_revision: captured.world.worldRevision
      });
      // Continuing an active Goal transfers control back to the Coordinator,
      // so publish the successor Cycle in the same locked checkpoint cut. A
      // specialist result must never expose `active Goal + no active Cycle` as
      // if the Goal Manager should select the already-active Goal again.
      this.#checkpoint.active_cycle = null;
      const nextCycle = this.#createActiveCycle();
      this.#pruneRuntimeAuthority();
      await this.#persist();
      await this.emit("autonomous_cycle_interrupted", json({
        cycle: interruptedCycle,
        reason: "goal_epoch_continued_after_re_evaluation",
        recovery_intent: input.recovery_intent,
        evidence_ref: evidence.ref,
        world_revision: captured.world.worldRevision
      }));
      await this.#emitCycleStarted(nextCycle);
      return json({
        status: "goal_epoch_continued",
        epoch_id: interruptedCycle.goal_epoch_id,
        completed_cycle_id: interruptedCycle.cycle_id,
        next_cycle_id: nextCycle.cycle_id,
        next_cycle_index: nextCycle.cycle_index,
        reason: input.reason,
        recovery_intent: input.recovery_intent,
        evidence_ref: evidence.ref,
        world_revision: captured.world.worldRevision
      });
    }));
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
      const decision = options.recoveryDecision
        ? structuredClone(options.recoveryDecision)
        : this.#actionModelSource(
            authority,
            action,
            rawInput,
            transactionId,
            agentId
          );
      if (options.recoveryDecision) {
        this.#assertActionDecisionRef(
          decision,
          action,
          rawInput,
          transactionId,
          agentId,
          this.#requiredActiveCycleRef(),
          authority
        );
      }
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
      this.#assertActionRoleAuthority(action, agentId);
      this.#assertCurrentExecutionAuthority(action, rawInput, transactionId);
      this.#physicsClock.throwIfFailed();
      this.#physicalExecution.assertExecutionOwner(transactionId);
      if (!requiresHumanoidClockPause(action)) {
        return this.#actions.invoke(
          action,
          rawInput,
          transactionId,
          agentId,
          decision,
          { ...options, toolAuthority: authority }
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
          { ...options, toolAuthority: authority }
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
    const activeCycle = this.#checkpoint.active_cycle;
    if (!activeCycle) {
      throw new Error(
        "An active Goal must enter an autonomous Cycle before Coordinator dispatch"
      );
    }
    const receipts = humanoidActionReceiptsInCommitOrder(
      this.#checkpoint.committed_actions
    );
    const cycleReceipts = receipts.filter((receipt) => (
      sameAutonomousCycle(receipt.cycle, activeCycle)
    ));
    const completion = this.cycleCompletionReadiness();
    if (completion.status === "ready") {
      if (this.#pendingBlockRemoval()) {
        const latestRejectedRemovalIndex = cycleReceipts.findLastIndex((receipt) => (
          receipt.action === "remove_world_block"
            && (!receipt.accepted
              || receipt.code !== "world_block_removal_authorized")
        ));
        if (latestRejectedRemovalIndex >= 0) {
          const failureObserved = cycleReceipts.slice(
            latestRejectedRemovalIndex + 1
          ).some((receipt) => isHumanoidSensorFusionActor(receipt.agentId)
            && receipt.accepted
            && receipt.action === "observe_humanoid");
          return failureObserved ? "replan_or_retire" : "post_failure_observation";
        }
        return "post_execution";
      }
      return completion.observed_after_execution
        ? "complete_cycle"
        : "post_execution";
    }
    const latestExecutionIndex = cycleReceipts.findLastIndex(physicalExecutionReceipt);
    const latestExecution = cycleReceipts[latestExecutionIndex];
    const latestAcceptedPlanIndex = cycleReceipts.findLastIndex((receipt) => (
      receipt.accepted && isHumanoidPlanningReceipt(receipt)
    ));
    const latestRejectedPlanIndex = cycleReceipts.findLastIndex((receipt) => (
      !receipt.accepted && isHumanoidPlanningReceipt(receipt)
    ));
    const activeGoal = this.#activeGoal();
    if (this.#store.definition.run_mode === "mission"
      && activeGoal
      // Goal summaries are model-authored descriptions, not physical
      // obligations. Keep the full Goal hash for candidate identity and
      // evidence integrity, but classify a mission Goal by its predicates.
      && goalConstraintSha256(activeGoal)
        !== goalConstraintSha256(this.#missionGoal)
      && this.#missionGoalPhysicallySatisfied()) {
      return "goal_transition";
    }
    if (activeGoal
      && latestAcceptedPlanIndex <= latestExecutionIndex
      && this.#activeGoalPhysicallySatisfied(activeGoal)) {
      return "complete_satisfied_goal";
    }
    if (latestAcceptedPlanIndex > latestExecutionIndex) return "execute_plan";
    const latestFailedExecutionIndex = latestExecution
      && !completedPhysicalExecution(latestExecution)
      ? latestExecutionIndex
      : -1;
    const relevantRejectedPlanIndex = latestRejectedPlanIndex > latestExecutionIndex
      ? latestRejectedPlanIndex
      : -1;
    const failureBarrier = Math.max(
      latestFailedExecutionIndex,
      relevantRejectedPlanIndex
    );
    if (failureBarrier >= 0) {
      const failureObserved = cycleReceipts.slice(failureBarrier + 1).some(
        (receipt) => isHumanoidSensorFusionActor(receipt.agentId)
          && receipt.accepted
          && receipt.action === "observe_humanoid"
      );
      return failureObserved ? "replan_or_retire" : "post_failure_observation";
    }
    const phaseStart = Math.max(0, latestExecutionIndex + 1);
    const observed = cycleReceipts.slice(phaseStart).some((receipt) => (
      receipt.accepted && receipt.action === "observe_humanoid"
    ));
    if (!observed) return "observe_or_plan";
    return "plan";
  }

  executorDelegationAvailable(): boolean {
    const phase = this.coordinatorPhase();
    if (phase === "execute_plan") return true;
    if (phase !== "post_execution") return false;
    return this.#pendingBlockRemoval();
  }

  #pendingBlockRemoval(): boolean {
    const goal = this.#activeGoal();
    const progress = this.#checkpoint.goal_progress;
    const world = this.#world.snapshot();
    const checker = goal && progress
      ? (world.frame === progress.last_world_frame
          && world.worldRevision === progress.last_world_revision
        ? inspectHumanoidGoal(goal, this.#scenario, world, progress)
        : advanceHumanoidGoal(goal, this.#scenario, world, progress).checker)
      : null;
    return goal?.predicates.some((predicate, index) => (
      predicate.type === "block_removed"
        && checker?.checks[index]?.passed !== true
    )) ?? false;
  }

  goalRetirementDelegationAvailable(): boolean {
    const phase = this.coordinatorPhase();
    if (phase === "goal_transition") return true;
    if (phase !== "replan_or_retire") return false;
    const activeCycleState = this.#checkpoint.active_cycle;
    if (!activeCycleState) return false;
    const budget = activeCycleState.replan_budget;
    if (budget.goal_reevaluation_started) return true;
    const compactAttemptInProgress = compactReplanAttemptInProgress(
      budget,
      this.#checkpoint.committed_actions,
      activeCycleState
    );
    if (!compactAttemptInProgress
      && budget.compact_replans_started >= budget.compact_replan_limit) {
      return true;
    }
    return false;
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
    if (phase === "observe_or_plan" || phase === "post_failure_observation") {
      return true;
    }
    if (phase === "post_execution") {
      if (this.#pendingBlockRemoval()) return false;
      return !this.cycleCompletionReadiness().observed_after_execution;
    }
    if (phase !== "replan_or_retire") return false;
    const activeCycle = this.#activeCycleRef();
    if (!activeCycle) return false;
    const cycleReceipts = humanoidActionReceiptsInCommitOrder(
      this.#checkpoint.committed_actions
    )
      .filter((receipt) => sameAutonomousCycle(receipt.cycle, activeCycle));
    const latestRejectedPlanIndex = cycleReceipts.findLastIndex((receipt) => (
      !receipt.accepted && isHumanoidPlanningReceipt(receipt)
    ));
    const latestFailedExecutionIndex = cycleReceipts.findLastIndex((receipt) => (
      physicalExecutionReceipt(receipt) && !completedPhysicalExecution(receipt)
    ));
    const feedbackBarrier = Math.max(
      latestRejectedPlanIndex,
      latestFailedExecutionIndex
    );
    if (feedbackBarrier < 0) return false;
    return !cycleReceipts.slice(feedbackBarrier + 1).some((receipt) => (
      isHumanoidSensorFusionActor(receipt.agentId)
        && receipt.accepted
        && receipt.action === "observe_humanoid"
    ));
  }

  motionDelegationAvailable(): boolean {
    const phase = this.coordinatorPhase();
    if (phase === "observe_or_plan" || phase === "plan") return true;
    if (phase !== "replan_or_retire") return false;
    const budget = this.#checkpoint.active_cycle?.replan_budget;
    if (!budget || budget.goal_reevaluation_started) return false;
    if (compactReplanAttemptInProgress(
      budget,
      this.#checkpoint.committed_actions,
      this.#checkpoint.active_cycle!
    )) return true;
    return budget.compact_replans_started < budget.compact_replan_limit;
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
    const neuralNode = HUMANOID_NEURAL_NODE_BY_ID.get(agentId);
    const directedSignals = this.pendingNeuralSignals({ targetNodeId: agentId });
    const visiblePathways = new Set<NeuralPathway>();
    if (neuralNode) {
      visiblePathways.add(neuralNode.pathway);
      if (neuralNode.parentKey !== null) {
        visiblePathways.add(
          HUMANOID_NEURAL_NODE_BY_ID.get(
            HUMANOID_NEURAL_AGENT_IDS[neuralNode.parentKey]
          )!.pathway
        );
      }
      for (const childKey of neuralNode.childKeys) {
        visiblePathways.add(
          HUMANOID_NEURAL_NODE_BY_ID.get(
            HUMANOID_NEURAL_AGENT_IDS[childKey]
          )!.pathway
        );
      }
    }
    for (const signal of directedSignals) visiblePathways.add(signal.pathway);
    const activeCommitment =
      this.#checkpoint.neural_hierarchy_state.active_skill_commitment;
    const activeAuthorityLeases = Object.values(
      this.#checkpoint.neural_hierarchy_state.authority_leases
    ).filter((lease) => (
      lease.status === "active"
        && (lease.issuing_parent_node_id === agentId
          || lease.target_child_node_id === agentId)
    ));
    const neuralProjection = {
      epoch_id: this.#checkpoint.neural_hierarchy_state.epoch_id,
      harness_phase: this.#checkpoint.neural_hierarchy_state.harness_phase,
      directed_signals: directedSignals,
      endpoint_authority_leases: activeAuthorityLeases,
      owned_skill_commitment: activeCommitment?.owner_node_id === agentId
        ? activeCommitment
        : null,
      active_rollout_certificates: Object.values(
        this.#checkpoint.neural_hierarchy_state.rollout_certificates
      ).filter((certificate) => (
        certificate.status === "active"
          && certificate.commitment_id === activeCommitment?.commitment_id
          && (agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection
            || agentId === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
            || agentId === HUMANOID_NEURAL_AGENT_IDS.predictive)
      )),
      unresolved_prediction_errors:
        this.#checkpoint.neural_hierarchy_state.prediction_errors
          .filter((error) => !error.corrected && error.observer_node_id === agentId)
          .slice(-16),
      pathway_cadence: Object.values(
        this.#checkpoint.neural_hierarchy_state.pathway_cadences
      ).filter((cadence) => visiblePathways.has(cadence.pathway))
    };
    if (agentId !== HUMANOID_NEURAL_AGENT_IDS.motorIntent) return json({
        ...(object(result.anchor) ?? {}),
        neural_hierarchy: neuralProjection
      });
    return json({
      ...(object(result.anchor) ?? {}),
      neural_hierarchy: neuralProjection,
      planning_tool_state: this.#actions.planningToolState(
        HUMANOID_NEURAL_AGENT_IDS.motorIntent
      ),
      grounding_snapshot: this.#actions.planningGroundingState(
        HUMANOID_NEURAL_AGENT_IDS.motorIntent
      )
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

  async resetAgentContextEpoch(): Promise<void> {
    const previous = this.#checkpoint.context_memory;
    this.#checkpoint.context_memory = {
      ...structuredClone(EmptyContextMemoryState),
      context_window_tokens: previous.context_window_tokens,
      compact_trigger_tokens: previous.compact_trigger_tokens,
      compact_recent_model_turns: previous.compact_recent_model_turns,
      compact_max_output_tokens: previous.compact_max_output_tokens
    };
    this.#checkpoint.context_memory_state_anchor = null;
    await this.#persist();
    await this.emit("agent_context_epoch_reset", {
      previous_scope_ids: Object.keys(previous.scopes).sort(),
      preserved_goal_dag: true,
      preserved_embodied_memory: true,
      preserved_physical_state: true,
      automatic_actuation: false
    });
  }

  async recordModelCallStarted(agentId: string): Promise<string> {
    return this.#goalStateMutex.runExclusive(async () => {
      this.#signal?.throwIfAborted();
      this.#assertRunAcceptsDecisions();
      const descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(agentId);
      if (!descriptor || descriptor.executionKind !== "model_agent") {
        throw new Error(
          `Model telemetry cannot originate from a non-model hierarchy node: ${agentId}`
        );
      }
      const cycle = this.#activeCycleRef();
      const modelCallId = randomUUID();
      const at = new Date().toISOString();
      const rawBudgetRole = this.coordinatorPhase() === "replan_or_retire"
        ? replanBudgetRole(agentId)
        : undefined;
      const budgetRole = rawBudgetRole === "coordinator"
        && this.#checkpoint.active_cycle
        && (this.#checkpoint.active_cycle.replan_budget.goal_reevaluation_started
          || compactReplanAttemptInProgress(
            this.#checkpoint.active_cycle.replan_budget,
            this.#checkpoint.committed_actions,
            this.#checkpoint.active_cycle
          ))
        ? undefined
        : rawBudgetRole;
      const budgetUpdate = budgetRole && this.#checkpoint.active_cycle
        ? beginHumanoidReplanModelCall(
            this.#checkpoint.active_cycle.replan_budget,
            { modelCallId, agentId, role: budgetRole, at }
          )
        : undefined;
      const record = await this.#requiredModelAuthority().recordStarted(
        agentId,
        cycle,
        modelCallId,
        at,
        budgetUpdate?.call
      );
      if (budgetUpdate && this.#checkpoint.active_cycle) {
        this.#checkpoint.active_cycle.replan_budget = budgetUpdate.budget;
      }
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
        total_model_calls: this.#checkpoint.total_model_calls,
        ...replanModelCallEvent(
          budgetUpdate?.call,
          this.#checkpoint.active_cycle
        )
      });
      // The provider call begins only after this method returns. Registering
      // activity last prevents a failed lifecycle/checkpoint write from
      // leaving a ghost Agent in the parallel active set.
      this.#beginAgentActivity(agentId);
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
      try {
        const record = await this.#requiredModelAuthority().recordCompleted(input);
        const budgetUpdate = this.#finishReplanModelCall(
          input.modelCallId,
          "completed",
          record.at
        );
        await this.#persist();
        await this.emit("model_request_completed", {
          agent_id: input.agentId,
          model_call_id: input.modelCallId,
          response_id: input.responseId,
          ...(record.cycle ? { cycle: record.cycle } : {}),
          tool_call_count: input.toolCalls.length,
          ...replanModelCallEvent(
            budgetUpdate,
            this.#checkpoint.active_cycle
          )
        });
      } finally {
        this.#endAgentActivity(input.agentId);
      }
    });
  }

  async recordModelCallFailed(modelCallId: string, agentId: string): Promise<void> {
    await this.#goalStateMutex.runExclusive(async () => {
      try {
        const record = await this.#requiredModelAuthority().recordFailed(
          modelCallId,
          agentId
        );
        const budgetUpdate = this.#finishReplanModelCall(
          modelCallId,
          "failed",
          record.at
        );
        await this.#persist();
        await this.emit("model_request_failed", {
          agent_id: agentId,
          model_call_id: modelCallId,
          ...(record.cycle ? { cycle: record.cycle } : {}),
          ...replanModelCallEvent(
            budgetUpdate,
            this.#checkpoint.active_cycle
          )
        });
      } finally {
        this.#endAgentActivity(agentId);
      }
    });
  }

  #finishReplanModelCall(
    modelCallId: string,
    status: "completed" | "failed",
    at: string
  ): HumanoidReplanModelCall | undefined {
    const cycle = this.#checkpoint.active_cycle;
    if (!cycle) return undefined;
    const update = finishHumanoidReplanModelCall(cycle.replan_budget, {
      modelCallId,
      status,
      at
    });
    if (!update.call) return undefined;
    cycle.replan_budget = update.budget;
    return update.call;
  }

  #reconcileReplanBudgetFromModelLifecycle(
    rawRecords: readonly unknown[]
  ): { changed: boolean; restoredModelCallIds: string[] } {
    const cycle = this.#checkpoint.active_cycle;
    if (!cycle) return { changed: false, restoredModelCallIds: [] };
    const records = rawRecords.map((record) => (
      ModelCallLifecycleRecordSchema.parse(record)
    ));
    const restoredModelCallIds: string[] = [];
    let changed = false;
    const replanStarts = new Map<string, Extract<
      ModelCallLifecycleRecord,
      { lifecycle: "started" }
    >>();
    for (const record of records) {
      if (record.lifecycle !== "started"
        || !record.replan_budget_call
        || !sameAutonomousCycle(record.cycle, cycle)) continue;
      const alreadyCheckpointed = cycle.replan_budget.model_calls.some(
        (call) => call.model_call_id === record.model_call_id
      );
      cycle.replan_budget = restoreHumanoidReplanModelCall(
        cycle.replan_budget,
        record.replan_budget_call
      );
      if (!alreadyCheckpointed) {
        changed = true;
        restoredModelCallIds.push(record.model_call_id);
      }
      replanStarts.set(record.model_call_id, record);
    }
    for (const record of records) {
      if (record.lifecycle === "started"
        || !replanStarts.has(record.model_call_id)) continue;
      const existing = cycle.replan_budget.model_calls.find(
        (call) => call.model_call_id === record.model_call_id
      );
      if (!existing) {
        throw new Error(
          `Replan model lifecycle has no durable budget start: ${record.model_call_id}`
        );
      }
      if (existing.status === "started") {
        changed = true;
        cycle.replan_budget = finishHumanoidReplanModelCall(
          cycle.replan_budget,
          {
            modelCallId: record.model_call_id,
            status: record.lifecycle === "completed" ? "completed" : "failed",
            at: record.at
          }
        ).budget;
        continue;
      }
      if (existing.status !== record.lifecycle
        || existing.completed_at !== record.at) {
        throw new Error(
          `Replan model lifecycle conflicts with durable budget: ${record.model_call_id}`
        );
      }
    }
    return { changed, restoredModelCallIds };
  }

  async recordCompactionModelCall(agentId: string): Promise<void> {
    await this.#recordCompactionCalls(agentId, 1, "model_request_started");
  }

  async reconcileCompactionModelCalls(agentId: string, additionalCalls: number): Promise<void> {
    if (!Number.isSafeInteger(additionalCalls) || additionalCalls <= 0) return;
    await this.#recordCompactionCalls(agentId, additionalCalls, "model_requests_reconciled");
  }

  async setActiveAgent(agentId: string): Promise<void> {
    this.#node(agentId);
    if (this.#checkpoint.active_agent_id === agentId
      && this.#checkpoint.active_agent_ids.includes(agentId)) return;
    const at = new Date().toISOString();
    const active = new Set([...this.#activeModelCallsByAgent.entries()]
      .filter(([, count]) => count > 0)
      .map(([activeAgentId]) => activeAgentId));
    active.add(agentId);
    this.#applyActiveAgentSet(active, agentId, at);
    await this.emit("hierarchy_focus_changed", {
      active_agent_id: agentId,
      active_agent_ids: [...active],
      nodes: json(this.#checkpoint.nodes)
    });
  }

  async setActiveAgents(agentIds: readonly string[]): Promise<void> {
    const unique = [...new Set(agentIds)];
    if (unique.length === 0) throw new Error("Hierarchy focus requires an active Agent");
    for (const agentId of unique) this.#node(agentId);
    const at = new Date().toISOString();
    this.#applyActiveAgentSet(new Set(unique), unique[0]!, at);
    await this.emit("hierarchy_focus_changed", {
      active_agent_id: unique[0]!,
      active_agent_ids: unique,
      nodes: json(this.#checkpoint.nodes)
    });
  }

  #applyActiveAgentSet(
    active: ReadonlySet<string>,
    focus: string,
    at: string
  ): void {
    for (const node of Object.values(this.#checkpoint.nodes)) {
      if (active.has(node.id)) node.status = "active";
      else if (node.id === this.rootAgentId) node.status = "waiting";
      else if (node.status !== "completed" && node.status !== "failed") {
        node.status = "ready";
      }
      node.updated_at = at;
    }
    this.#checkpoint.active_agent_id = focus;
    this.#checkpoint.active_agent_ids = [...active];
    // Hierarchy focus is execution telemetry, not physical or Goal authority.
    // Persisting it used to take a fresh MuJoCo/Goal cut for every SDK stream
    // event. In a standard nested Agent loop those events can arrive while a
    // specialist tool is atomically selecting a Goal or committing an action,
    // allowing the telemetry write to pair an older anchor with newer state.
    // The durable focus event below is sufficient; the next authoritative
    // state transition checkpoints the latest focus together with its own cut.
  }

  #beginAgentActivity(agentId: string): void {
    this.#activeModelCallsByAgent.set(
      agentId,
      (this.#activeModelCallsByAgent.get(agentId) ?? 0) + 1
    );
    this.#applyActiveAgentSet(
      new Set([...this.#activeModelCallsByAgent.entries()]
        .filter(([, count]) => count > 0)
        .map(([activeAgentId]) => activeAgentId)),
      agentId,
      new Date().toISOString()
    );
  }

  #endAgentActivity(agentId: string): void {
    const remaining = Math.max(0, (this.#activeModelCallsByAgent.get(agentId) ?? 0) - 1);
    if (remaining === 0) this.#activeModelCallsByAgent.delete(agentId);
    else this.#activeModelCallsByAgent.set(agentId, remaining);
    const active = new Set([...this.#activeModelCallsByAgent.entries()]
      .filter(([, count]) => count > 0)
      .map(([activeAgentId]) => activeAgentId));
    if (active.size === 0) active.add(this.rootAgentId);
    this.#applyActiveAgentSet(active, active.values().next().value!, new Date().toISOString());
  }

  async start(resumed: boolean): Promise<void> {
    await this.#reconcileActionCommits();
    const reconciledPhysicalTail = resumed
      ? await this.#physicalExecution.synchronizePendingExecution()
      : undefined;
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
    if (reconciledPhysicalTail) {
      await this.emit("physical_execution_tail_reconciled", json({
        transaction_id: reconciledPhysicalTail.transactionId,
        previous_committed_frame_count:
          reconciledPhysicalTail.previousCommittedFrameCount,
        committed_frame_count: reconciledPhysicalTail.committedFrameCount,
        complete_trajectory: reconciledPhysicalTail.completeTrajectory,
        automatic_actuation: false
      }));
    }
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
    const authority = execution.admission.tool_call_authority;
    if (!execution.admission.decision || !authority) {
      throw new Error(
        `Durable physical execution has no Coordinator delegation: ${execution.transaction_id}`
      );
    }
    const receipt = await this.invoke(
      execution.action,
      input,
      execution.transaction_id,
      execution.agent_id,
      authority,
      {
        recoveryDecision: execution.admission.decision,
        ...(execution.admission.neural_rollout_certificate
          ? {
              neuralRolloutCertificate:
                execution.admission.neural_rollout_certificate
            }
          : {})
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
      const missionCompleted = checker.success
        && this.#store.definition.run_mode === "mission"
        && this.missionGoalCompleted();
      const committedCycle = missionCompleted
        ? verifiedMissionCompletionOutput({
            missionGoal: this.#missionGoal,
            modelCycle: cycle,
            checker,
            evidenceTransactionIds: [...evidence],
            worldFrame: world.frame,
            worldRevision: world.worldRevision
          })
        : cycle;
      this.#checkpoint.last_cycle = committedCycle;
      this.#checkpoint.embodied_memory = memory.state;
      this.#checkpoint.active_cycle = null;
      this.#checkpoint.committed_actions = actionWindow.receipts;
      this.#pruneRuntimeAuthority();

      if (missionCompleted) {
        this.#continuousPhysicsEnabled = false;
        this.#stageFinish(
          "succeeded",
          JSON.stringify(committedCycle),
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
        output: committedCycle,
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
      const missionCompleted = this.#store.definition.run_mode === "mission"
        && this.missionGoalCompleted();
      const committedCycle = missionCompleted
        ? verifiedMissionCompletionOutput({
            missionGoal: this.#missionGoal,
            modelCycle: cycleOutput,
            checker,
            evidenceTransactionIds: [],
            worldFrame: captured.world.frame,
            worldRevision: captured.world.worldRevision
          })
        : cycleOutput;
      this.#checkpoint.last_cycle = committedCycle;
      this.#checkpoint.active_cycle = null;
      this.#pruneRuntimeAuthority();

      if (missionCompleted) {
        this.#continuousPhysicsEnabled = false;
        this.#stageFinish(
          "succeeded",
          JSON.stringify(committedCycle),
          null,
          "run_succeeded"
        );
      }

      await this.#persist();
      await this.#store.append("checker", json(checker));
      await this.emit("autonomous_goal_satisfied", {
        cycle: json(activeCycle),
        output: committedCycle,
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

  async #emitHumanoidSkillEvent(
    event: HumanoidEmbodiedSkillEvent
  ): Promise<void> {
    const eventId = `humanoid-skill-event:${modelPayloadSha256({
      run_id: this.runId,
      event
    })}`;
    const runtimeEvent: RuntimeEvent = {
      event_id: eventId,
      run_id: this.runId,
      type: "humanoid_skill_event",
      at: new Date().toISOString(),
      data: json(event)
    };
    let persisted: RuntimeEvent | undefined;
    try {
      [persisted] = await this.#store.appendRuntimeEvents([runtimeEvent]);
    } catch (error) {
      const committed = await findDurableRuntimeEventById(
        this.#store,
        eventId
      );
      if (!committed) throw error;
      assertHumanoidSkillRuntimeEvent(committed, this.runId, event);
      persisted = committed;
    }
    try {
      await this.#eventSink(persisted!);
    } catch {
      // The journal is authoritative. A disconnected operator stream can
      // replay this cursor; it must not roll physical Skill event state back.
    }
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
    const toolAuthority = this.#actions.toolCallAuthority(receipt.transactionId)
      ?? this.#checkpoint.action_execution_ledger.active[receipt.transactionId]
        ?.admission.tool_call_authority;
    this.#assertActionDecisionRef(
      receipt.decision,
      receipt.action,
      receipt.input,
      receipt.transactionId,
      receipt.agentId,
      activeCycle,
      toolAuthority
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
        ...(toolAuthority
          ? { action_tool_authority: toolAuthority }
          : {}),
        action_runtime_state: this.#actions.persistenceState(),
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
    const stagedByTransaction = new Map(
      this.#recoveredAcknowledgedActionCommits.map((entry) => (
        [entry.transaction_id, structuredClone(entry)] as const
      ))
    );
    for (const entry of Object.values(this.#checkpoint.action_commit_outbox.pending)) {
      stagedByTransaction.set(entry.transaction_id, structuredClone(entry));
    }
    const staged = [...stagedByTransaction.values()];
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
    const recoveredAcknowledgements = this.#recoveredAcknowledgedActionCommits.length;
    if (recoveredAcknowledgements > 0) {
      let recoveredLedger = this.#checkpoint.action_execution_ledger;
      for (const entry of this.#recoveredAcknowledgedActionCommits) {
        const execution = recoveredLedger.active[entry.transaction_id];
        if (!execution) continue;
        if (execution.status !== "terminal" || !execution.terminal) {
          throw new Error(
            `Recovered action acknowledgement has a nonterminal execution: ${entry.transaction_id}`
          );
        }
        recoveredLedger = acknowledgeTerminalActionExecution(
          recoveredLedger,
          entry.transaction_id,
          execution.terminal
        );
      }
      this.#checkpoint.action_execution_ledger = recoveredLedger;
    }
    await this.#physicalExecution.acknowledgeTerminals(staged);
    this.#recoveredAcknowledgedActionCommits = [];
    if (recoveredAcknowledgements > 0) {
      // The durable outbox head may outrun checkpoint.json at the acknowledgement
      // cut. Close that cut together with any terminal MuJoCo cleanup before
      // accepting another action.
      await this.#persist();
    }
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
    await this.#physicalExecution.synchronizePendingExecution();
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
    if ((status === "succeeded" || status === "failed")
      && this.#checkpoint.neural_hierarchy_state.harness_phase.phase !== "terminal") {
      this.#checkpoint.neural_hierarchy_state = transitionNeuralHarnessPhase(
        this.#checkpoint.neural_hierarchy_state,
        {
          phase: "terminal",
          goalEpochId: null,
          commitmentId: null,
          worldRevision: this.#world.snapshot().worldRevision,
          enteredByNodeId: this.rootAgentId,
          reason: eventType,
          liveInvocationIds: this.#liveNeuralInvocationIds(),
          at
        }
      );
    }
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

  #nextActionCommitSequence(): number {
    const receipts = humanoidActionReceiptsInCommitOrder(
      this.#checkpoint.committed_actions
    );
    return (receipts.at(-1)?.commitSequence ?? 0) + 1;
  }

  #assertActionRoleAuthority(
    action: Parameters<HumanoidActionRuntime["invoke"]>[0],
    agentId: string
  ): void {
    const allowedAgentIds = humanoidActionRoleAuthority(action);
    if (!allowedAgentIds.has(agentId.trim())) {
      throw new Error(
        `Humanoid action ${action} is outside Agent role authority: ${agentId}`
      );
    }
  }

  #assertCurrentExecutionAuthority(
    action: Parameters<HumanoidActionRuntime["invoke"]>[0],
    rawInput: unknown,
    transactionId: string
  ): void {
    if (action !== "execute_humanoid_skill"
      && action !== "execute_whole_body_motion"
      && action !== "execute_humanoid_navigation") {
      return;
    }
    const input = rawInput !== null
      && typeof rawInput === "object"
      && !Array.isArray(rawInput)
      ? rawInput as Record<string, unknown>
      : undefined;
    const planningTransactionId = input?.planning_transaction_id;
    const referencedReceipt = typeof planningTransactionId === "string"
      ? this.#checkpoint.committed_actions[planningTransactionId]
      : undefined;
    // Malformed, missing, rejected, and action-mismatched references cannot
    // actuate in HumanoidActionRuntime. Let that boundary produce its durable
    // rejection receipt; current authority is required for an executable plan.
    if (!referencedReceipt?.accepted
      || !isHumanoidPlanningReceipt(referencedReceipt)
      || action !== humanoidExecutionActionForPlan(referencedReceipt.action)) {
      return;
    }
    const phase = this.coordinatorPhase();
    if (phase !== "execute_plan") {
      throw new Error(
        `Humanoid physical action has no current execution authority in phase ${phase}`
      );
    }
    const activeCycle = this.#requiredActiveCycleRef();
    const planningReceipt = humanoidActionReceiptsInCommitOrder(
      this.#checkpoint.committed_actions
    )
      .findLast((receipt) => (
        receipt.accepted
          && sameAutonomousCycle(receipt.cycle, activeCycle)
          && isHumanoidPlanningReceipt(receipt)
      ));
    if (!planningReceipt) {
      throw new Error("Humanoid physical action has no current planning receipt");
    }
    const expectedAction = humanoidExecutionActionForPlan(planningReceipt.action);
    if (planningTransactionId !== planningReceipt.transactionId
      || action !== expectedAction) {
      throw new Error(
        `Humanoid physical action does not match current execution authority: ${planningReceipt.transactionId}`
      );
    }
    const committedConsumer = Object.values(this.#checkpoint.committed_actions).find(
      (receipt) => (
        sameAutonomousCycle(receipt.cycle, activeCycle)
          && physicalExecutionReceipt(receipt)
          && planningTransactionIdFromReceipt(receipt) === planningReceipt.transactionId
      )
    );
    if (committedConsumer) {
      throw new Error(
        `Humanoid planning authority was already consumed: ${planningReceipt.transactionId}`
      );
    }
    const activeConsumer = activeActionExecutions(
      this.#checkpoint.action_execution_ledger
    ).find((execution) => (
      execution.admission.planning_transaction_id === planningReceipt.transactionId
    ));
    if (activeConsumer && activeConsumer.transaction_id !== transactionId) {
      throw new Error(
        `Humanoid planning authority is already executing: ${planningReceipt.transactionId}`
      );
    }
  }

  #assertActionDecisionRef(
    rawDecision: ModelDecisionRef,
    expectedToolName: Parameters<HumanoidActionRuntime["invoke"]>[0],
    input: unknown,
    transactionId: string,
    agentId: string,
    cycle: AutonomousCycleRef,
    toolAuthority?: HumanoidActionToolCallAuthority
  ): void {
    this.#requiredModelAuthority().assertActionDecision({
      rawDecision,
      expectedToolName,
      actionInput: input,
      transactionId,
      agentId,
      cycle,
      ...(toolAuthority ? { toolAuthority } : {})
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
    for (const execution of Object.values(
      this.#checkpoint.action_execution_ledger.active
    )) {
      if (execution.admission.decision) {
        retainedModelCalls.add(execution.admission.decision.model_call_id);
      }
    }
    for (const call of this.#checkpoint.active_cycle?.replan_budget.model_calls ?? []) {
      retainedModelCalls.add(call.model_call_id);
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

  async #persistNeuralState(): Promise<void> {
    await this.#persist({
      refreshWorld: false,
      authorityDomains: [],
      neuralOnly: true
    });
  }

  async #persist(
    input: boolean | HumanoidPersistOptions = {}
  ): Promise<void> {
    const options: HumanoidPersistOptions = typeof input === "boolean"
      ? { refreshWorld: input }
      : input;
    const refreshWorld = options.refreshWorld ?? true;
    const authorityDomains = new Set(
      options.authorityDomains === "all" || options.authorityDomains === undefined
        ? ALL_HUMANOID_AUTHORITY_DOMAINS
        : options.authorityDomains
    );
    await this.#persistMutex.runExclusive(async () => {
      if (options.neuralOnly) {
        const snapshot = structuredClone(this.#durableCheckpoint);
        snapshot.neural_hierarchy_state = structuredClone(
          this.#checkpoint.neural_hierarchy_state
        );
        snapshot.updated_at = new Date().toISOString();
        await this.#store.writeCheckpoint(snapshot);
        this.#durableCheckpoint = snapshot;
        return;
      }
      if (refreshWorld) {
        const captured = await this.#world.capturePersistenceState();
        this.#applyWorldPersistenceState(captured);
      }
      this.#checkpoint.action_runtime_state = this.#actions.persistenceState();
      this.#assertActiveGoalProgress();
      this.#checkpoint.updated_at = new Date().toISOString();

      // Anchor and validate an immutable authority cut. Parallel hierarchy
      // branches can keep changing the live state while durable events are
      // appended, but those later mutations must not leak into this write.
      const snapshot = structuredClone(this.#checkpoint);
      if (authorityDomains.has("physical")) {
        await this.#anchorCurrentPhysicalState(snapshot);
      }
      if (authorityDomains.has("goal")) {
        await this.#anchorCurrentGoalState(snapshot);
      }
      if (authorityDomains.has("embodied_memory")) {
        await this.#anchorCurrentEmbodiedMemoryState(snapshot);
      }
      if (authorityDomains.has("context_memory")) {
        await this.#anchorCurrentContextMemoryState(snapshot);
      }
      if (authorityDomains.has("execution_ledger")) {
        await this.#anchorCurrentExecutionLedgerState(snapshot);
      }
      await this.#store.writeCheckpoint(snapshot);
      this.#durableCheckpoint = structuredClone(snapshot);
      this.#adoptAuthorityAnchors(snapshot, authorityDomains);
    });
  }

  #adoptAuthorityAnchors(
    snapshot: HumanoidRunCheckpoint,
    domains: ReadonlySet<HumanoidAuthorityDomain>
  ): void {
    if (domains.has("physical")
      && snapshot.world.frame === this.#checkpoint.world.frame
      && snapshot.world.worldRevision === this.#checkpoint.world.worldRevision
      && humanoidPhysicalStateSha256(snapshot.world_checkpoint)
        === humanoidPhysicalStateSha256(this.#checkpoint.world_checkpoint)) {
      this.#checkpoint.physical_state_anchor = snapshot.physical_state_anchor;
    }
    if (domains.has("goal")
      && snapshot.goal_dag.state_sha256 === this.#checkpoint.goal_dag.state_sha256
      && humanoidGoalControlStateSha256(snapshot)
        === humanoidGoalControlStateSha256(this.#checkpoint)) {
      this.#checkpoint.goal_state_anchor = snapshot.goal_state_anchor;
    }
    if (domains.has("embodied_memory")
      && humanoidEmbodiedMemoryStateSha256(snapshot.embodied_memory)
        === humanoidEmbodiedMemoryStateSha256(this.#checkpoint.embodied_memory)) {
      this.#checkpoint.embodied_memory_state_anchor =
        snapshot.embodied_memory_state_anchor;
    }
    if (domains.has("context_memory")
      && humanoidContextMemoryStateSha256(snapshot.context_memory)
        === humanoidContextMemoryStateSha256(this.#checkpoint.context_memory)) {
      this.#checkpoint.context_memory_state_anchor =
        snapshot.context_memory_state_anchor;
    }
    if (domains.has("execution_ledger")
      && humanoidExecutionLedgerStateSha256(snapshot.action_execution_ledger)
        === humanoidExecutionLedgerStateSha256(
          this.#checkpoint.action_execution_ledger
        )) {
      this.#checkpoint.execution_ledger_state_anchor =
        snapshot.execution_ledger_state_anchor;
    }
  }

  async #verifyExistingPhysicalStateAnchor(): Promise<void> {
    const anchor = this.#checkpoint.physical_state_anchor;
    if (!anchor) {
      const worldCheckpointSha256 = actionCommitPayloadSha256(
        json(this.#checkpoint.world_checkpoint)
      );
      const identity = {
        version: 1 as const,
        run_id: this.runId,
        world_frame: this.#checkpoint.world.frame,
        world_revision: this.#checkpoint.world.worldRevision,
        world_checkpoint_sha256: worldCheckpointSha256
      };
      const eventId = `humanoid-physical-state:${actionCommitPayloadSha256(json(identity))}`;
      const orphan = await findDurableRuntimeEventById(this.#store, eventId);
      if (orphan) {
        await assertDurableAnchorIsLatest(
          this.#store,
          "humanoid_physical_state_anchored",
          eventId,
          "Physical state anchor"
        );
        const recovered = HumanoidPhysicalStateAnchorSchema.parse({
          version: 1,
          event_id: eventId,
          world_frame: identity.world_frame,
          world_revision: identity.world_revision,
          world_checkpoint_sha256: worldCheckpointSha256,
          anchored_at: orphan.at
        });
        assertHumanoidPhysicalStateAnchorEvent(orphan, this.runId, recovered);
        this.#checkpoint.physical_state_anchor = recovered;
        this.#physicalAnchorOrphanRecoveryPending = false;
        return;
      }
      const physicalStateSha256 = humanoidPhysicalStateSha256(
        this.#checkpoint.world_checkpoint
      );
      const physicalIdentity = {
        version: 2 as const,
        run_id: this.runId,
        world_frame: this.#checkpoint.world.frame,
        world_revision: this.#checkpoint.world.worldRevision,
        physical_state_sha256: physicalStateSha256
      };
      const physicalEventId = `humanoid-physical-state:${
        actionCommitPayloadSha256(json(physicalIdentity))
      }`;
      const physicalOrphan = await findDurableRuntimeEventById(
        this.#store,
        physicalEventId
      );
      if (physicalOrphan) {
        await assertDurableAnchorIsLatest(
          this.#store,
          "humanoid_physical_state_anchored",
          physicalEventId,
          "Physical state anchor"
        );
        const recovered = HumanoidPhysicalStateAnchorSchema.parse({
          version: 2,
          event_id: physicalEventId,
          world_frame: physicalIdentity.world_frame,
          world_revision: physicalIdentity.world_revision,
          physical_state_sha256: physicalStateSha256,
          anchored_at: physicalOrphan.at
        });
        assertHumanoidPhysicalStateAnchorEvent(
          physicalOrphan,
          this.runId,
          recovered
        );
        this.#checkpoint.physical_state_anchor = recovered;
        this.#physicalAnchorOrphanRecoveryPending = false;
        return;
      }
      await assertNoDurableAnchorDowngrade(
        this.#store,
        "humanoid_physical_state_anchored",
        "Physical state anchor"
      );
      return;
    }
    const cut = await this.#world.capturePersistenceState();
    const persistedPhysicalStateSha256 = humanoidPhysicalStateSha256(
      this.#checkpoint.world_checkpoint
    );
    const restoredPhysicalStateSha256 = humanoidPhysicalStateSha256(
      cut.worldCheckpoint
    );
    if (cut.world.frame !== anchor.world_frame
      || cut.world.worldRevision !== anchor.world_revision
      || restoredPhysicalStateSha256 !== persistedPhysicalStateSha256
      || (anchor.version === 1
        ? actionCommitPayloadSha256(json(this.#checkpoint.world_checkpoint))
          !== anchor.world_checkpoint_sha256
        : persistedPhysicalStateSha256 !== anchor.physical_state_sha256)) {
      throw new Error("Physical state anchor conflicts with the restored MuJoCo checkpoint");
    }
    const event = await findDurableRuntimeEventById(this.#store, anchor.event_id);
    if (!event) {
      throw new Error(`Physical state anchor event is missing: ${anchor.event_id}`);
    }
    assertHumanoidPhysicalStateAnchorEvent(event, this.runId, anchor);
    await assertDurableAnchorIsLatest(
      this.#store,
      "humanoid_physical_state_anchored",
      anchor.event_id,
      "Physical state anchor"
    );
  }

  async #verifyExistingGoalStateAnchor(): Promise<void> {
    const anchor = this.#checkpoint.goal_state_anchor;
    if (!anchor) {
      const identity = {
        version: 1 as const,
        run_id: this.runId,
        goal_dag_state_sha256: this.#checkpoint.goal_dag.state_sha256,
        control_state_sha256: humanoidGoalControlStateSha256(this.#checkpoint)
      };
      const eventId = `humanoid-goal-state:${actionCommitPayloadSha256(json(identity))}`;
      const orphan = await findDurableRuntimeEventById(this.#store, eventId);
      if (orphan) {
        await assertDurableAnchorIsLatest(
          this.#store,
          "humanoid_goal_state_anchored",
          eventId,
          "Goal state anchor"
        );
        const recovered = HumanoidGoalStateAnchorSchema.parse({
          version: 1,
          event_id: eventId,
          goal_dag_state_sha256: identity.goal_dag_state_sha256,
          control_state_sha256: identity.control_state_sha256,
          anchored_at: orphan.at
        });
        assertHumanoidGoalStateAnchorEvent(orphan, this.runId, recovered);
        this.#checkpoint.goal_state_anchor = recovered;
        this.#goalAnchorOrphanRecoveryPending = false;
        return;
      }
      await assertNoDurableAnchorDowngrade(
        this.#store,
        "humanoid_goal_state_anchored",
        "Goal state anchor"
      );
      return;
    }
    if (anchor.goal_dag_state_sha256 !== this.#checkpoint.goal_dag.state_sha256
      || anchor.control_state_sha256
        !== humanoidGoalControlStateSha256(this.#checkpoint)) {
      throw new Error("Goal state anchor conflicts with the persisted Goal DAG");
    }
    const event = await findDurableRuntimeEventById(this.#store, anchor.event_id);
    if (!event) throw new Error(`Goal state anchor event is missing: ${anchor.event_id}`);
    assertHumanoidGoalStateAnchorEvent(event, this.runId, anchor);
    const latest = await latestDurableGoalStateAnchorEvent(this.#store);
    if (!latest) throw new Error("Goal state anchor durable history is missing");
    if (latest.event_id === anchor.event_id) return;
    throw new Error("Goal state anchor is not the latest durable state");
  }

  async #verifyExistingEmbodiedMemoryStateAnchor(): Promise<void> {
    const anchor = this.#checkpoint.embodied_memory_state_anchor;
    if (!anchor) {
      const memorySha256 = humanoidEmbodiedMemoryStateSha256(
        this.#checkpoint.embodied_memory
      );
      const identity = {
        version: 1 as const,
        run_id: this.runId,
        embodied_memory_sha256: memorySha256
      };
      const eventId = `humanoid-embodied-memory:${actionCommitPayloadSha256(json(identity))}`;
      const orphan = await findDurableRuntimeEventById(this.#store, eventId);
      if (orphan) {
        await assertDurableAnchorIsLatest(
          this.#store,
          "humanoid_embodied_memory_state_anchored",
          eventId,
          "Embodied memory state anchor"
        );
        const recovered = HumanoidEmbodiedMemoryStateAnchorSchema.parse({
          version: 1,
          event_id: eventId,
          embodied_memory_sha256: memorySha256,
          anchored_at: orphan.at
        });
        assertHumanoidEmbodiedMemoryStateAnchorEvent(orphan, this.runId, recovered);
        this.#checkpoint.embodied_memory_state_anchor = recovered;
        this.#memoryAnchorOrphanRecoveryPending = false;
        return;
      }
      await assertNoDurableAnchorDowngrade(
        this.#store,
        "humanoid_embodied_memory_state_anchored",
        "Embodied memory state anchor"
      );
      return;
    }
    if (anchor.embodied_memory_sha256
      !== humanoidEmbodiedMemoryStateSha256(this.#checkpoint.embodied_memory)) {
      throw new Error(
        "Embodied memory state anchor conflicts with the persisted memory"
      );
    }
    const event = await findDurableRuntimeEventById(this.#store, anchor.event_id);
    if (!event) {
      throw new Error(`Embodied memory state anchor event is missing: ${anchor.event_id}`);
    }
    assertHumanoidEmbodiedMemoryStateAnchorEvent(event, this.runId, anchor);
    await assertDurableAnchorIsLatest(
      this.#store,
      "humanoid_embodied_memory_state_anchored",
      anchor.event_id,
      "Embodied memory state anchor"
    );
  }

  async #verifyExistingContextMemoryStateAnchor(): Promise<void> {
    const anchor = this.#checkpoint.context_memory_state_anchor;
    if (!anchor) {
      const memorySha256 = humanoidContextMemoryStateSha256(
        this.#checkpoint.context_memory
      );
      const identity = {
        version: 1 as const,
        run_id: this.runId,
        context_memory_sha256: memorySha256
      };
      const eventId = `humanoid-context-memory:${actionCommitPayloadSha256(json(identity))}`;
      const orphan = await findDurableRuntimeEventById(this.#store, eventId);
      if (orphan) {
        await assertDurableAnchorIsLatest(
          this.#store,
          "humanoid_context_memory_state_anchored",
          eventId,
          "Context memory state anchor"
        );
        const recovered = HumanoidContextMemoryStateAnchorSchema.parse({
          version: 1,
          event_id: eventId,
          context_memory_sha256: memorySha256,
          anchored_at: orphan.at
        });
        assertHumanoidContextMemoryStateAnchorEvent(orphan, this.runId, recovered);
        this.#checkpoint.context_memory_state_anchor = recovered;
        this.#contextAnchorOrphanRecoveryPending = false;
        return;
      }
      await assertNoDurableAnchorDowngrade(
        this.#store,
        "humanoid_context_memory_state_anchored",
        "Context memory state anchor"
      );
      return;
    }
    if (anchor.context_memory_sha256
      !== humanoidContextMemoryStateSha256(this.#checkpoint.context_memory)) {
      throw new Error("Context memory state anchor conflicts with the persisted memory");
    }
    const event = await findDurableRuntimeEventById(this.#store, anchor.event_id);
    if (!event) {
      throw new Error(`Context memory state anchor event is missing: ${anchor.event_id}`);
    }
    assertHumanoidContextMemoryStateAnchorEvent(event, this.runId, anchor);
    await assertDurableAnchorIsLatest(
      this.#store,
      "humanoid_context_memory_state_anchored",
      anchor.event_id,
      "Context memory state anchor"
    );
  }

  async #verifyExistingExecutionLedgerStateAnchor(): Promise<void> {
    const anchor = this.#checkpoint.execution_ledger_state_anchor;
    const ledgerSha256 = humanoidExecutionLedgerStateSha256(
      this.#checkpoint.action_execution_ledger
    );
    if (!anchor) {
      const identity = {
        version: 1 as const,
        run_id: this.runId,
        execution_ledger_sha256: ledgerSha256
      };
      const eventId = `humanoid-execution-ledger:${actionCommitPayloadSha256(json(identity))}`;
      const orphan = await findDurableRuntimeEventById(this.#store, eventId);
      if (orphan) {
        await assertDurableAnchorIsLatest(
          this.#store,
          "humanoid_execution_ledger_state_anchored",
          eventId,
          "Execution ledger state anchor"
        );
        const recovered = HumanoidExecutionLedgerStateAnchorSchema.parse({
          version: 1,
          event_id: eventId,
          execution_ledger_sha256: ledgerSha256,
          anchored_at: orphan.at
        });
        assertHumanoidExecutionLedgerStateAnchorEvent(orphan, this.runId, recovered);
        this.#checkpoint.execution_ledger_state_anchor = recovered;
        this.#executionLedgerAnchorOrphanRecoveryPending = false;
        return;
      }
      await assertNoDurableAnchorDowngrade(
        this.#store,
        "humanoid_execution_ledger_state_anchored",
        "Execution ledger state anchor"
      );
      return;
    }
    if (anchor.execution_ledger_sha256 !== ledgerSha256) {
      throw new Error("Execution ledger state anchor conflicts with the persisted ledger");
    }
    const event = await findDurableRuntimeEventById(this.#store, anchor.event_id);
    if (!event) {
      throw new Error(`Execution ledger state anchor event is missing: ${anchor.event_id}`);
    }
    assertHumanoidExecutionLedgerStateAnchorEvent(event, this.runId, anchor);
    const latest = await latestDurableExecutionLedgerAnchorEvent(this.#store);
    if (!latest) {
      throw new Error("Execution ledger state anchor durable history is missing");
    }
    if (latest.event_id === anchor.event_id) return;

    // The only legitimate newer ledger head with an older checkpoint is the
    // acknowledgement/removal cut written immediately before checkpoint.json.
    // Recover it only when it consumes exactly the checkpoint's terminal
    // outbox entries and the durable action journals prove those commits.
    const recovered = await this.#recoverAcknowledgedExecutionLedgerHead(latest);
    if (recovered) {
      this.#checkpoint.action_execution_ledger = recovered;
      this.#checkpoint.execution_ledger_state_anchor = latest.anchor;
      return;
    }
    const recoveredAdmission = this.#recoverCertifiedExecutionAdmissionHead(latest);
    if (!recoveredAdmission) {
      throw new Error("Execution ledger state anchor is not the latest durable state");
    }
    this.#checkpoint.action_execution_ledger = recoveredAdmission.ledger;
    this.#checkpoint.neural_hierarchy_state = recoveredAdmission.neuralState;
    this.#checkpoint.execution_ledger_state_anchor = latest.anchor;
  }

  #recoverCertifiedExecutionAdmissionHead(
    event: RuntimeEvent
  ): {
    ledger: HumanoidRunCheckpoint["action_execution_ledger"];
    neuralState: NeuralHierarchyState;
  } | undefined {
    const data = object(event.data);
    if (data.action_execution_ledger === undefined) return undefined;
    try {
      const ledger = restoreActionExecutionLedger(data.action_execution_ledger);
      const ledgerSha256 = humanoidExecutionLedgerStateSha256(ledger);
      const anchor = HumanoidExecutionLedgerStateAnchorSchema.parse({
        version: 1,
        event_id: event.event_id,
        execution_ledger_sha256: ledgerSha256,
        anchored_at: event.at
      });
      assertHumanoidExecutionLedgerStateAnchorEvent(event, this.runId, anchor);
      const previous = this.#checkpoint.action_execution_ledger;
      for (const [transactionId, entry] of Object.entries(previous.active)) {
        if (JSON.stringify(ledger.active[transactionId]) !== JSON.stringify(entry)) {
          return undefined;
        }
      }
      const added = Object.values(ledger.active).filter(
        (entry) => previous.active[entry.transaction_id] === undefined
      );
      if (added.length !== 1 || Object.keys(ledger.active).length
        !== Object.keys(previous.active).length + 1) return undefined;
      const [admission] = added;
      const neural = admission?.admission.neural_rollout_certificate;
      if (!admission || admission.status !== "admitted" || !neural
        || this.#checkpoint.committed_actions[admission.transaction_id]) {
        return undefined;
      }
      const consumed = consumeNeuralRolloutCertificate(
        this.#checkpoint.neural_hierarchy_state,
        {
          certificateId: neural.certificate_id,
          commitmentId: neural.commitment_id,
          planningTransactionId: neural.planning_transaction_id,
          planningAction: neural.planning_action,
          executionTransactionId: admission.transaction_id,
          worldRevision: admission.admission.world_revision,
          at: admission.admitted_at
        }
      );
      return { ledger, neuralState: consumed.state };
    } catch {
      return undefined;
    }
  }

  async #recoverAcknowledgedExecutionLedgerHead(
    event: RuntimeEvent
  ): Promise<HumanoidRunCheckpoint["action_execution_ledger"] | undefined> {
    const data = object(event.data);
    const rawLedger = data.action_execution_ledger;
    if (rawLedger === undefined) return undefined;
    try {
      const ledger = restoreActionExecutionLedger(rawLedger);
      const ledgerSha256 = humanoidExecutionLedgerStateSha256(ledger);
      const anchor = HumanoidExecutionLedgerStateAnchorSchema.parse({
        version: 1,
        event_id: event.event_id,
        execution_ledger_sha256: ledgerSha256,
        anchored_at: event.at
      });
      assertHumanoidExecutionLedgerStateAnchorEvent(event, this.runId, anchor);
      const expected = structuredClone(this.#checkpoint.action_execution_ledger);
      const pending = Object.values(this.#checkpoint.action_commit_outbox.pending);
      const consumed = Object.values(expected.active).filter((entry) => (
        entry.status === "terminal"
          && this.#checkpoint.action_commit_outbox.pending[entry.transaction_id]
          && ledger.active[entry.transaction_id] === undefined
      ));
      if (pending.length === 0 || consumed.length === 0) {
        return undefined;
      }
      for (const entry of consumed) delete expected.active[entry.transaction_id];
      if (humanoidExecutionLedgerStateSha256(expected) !== ledgerSha256) return undefined;
      const consumedTransactions = new Set(consumed.map((entry) => entry.transaction_id));
      if (!await durableExecutionCommitProofsExist(
        this.#store,
        pending.filter((entry) => consumedTransactions.has(entry.transaction_id)),
        this.#checkpoint.committed_actions
      )) return undefined;
      return ledger;
    } catch {
      return undefined;
    }
  }

  async #anchorCurrentPhysicalState(checkpoint: HumanoidRunCheckpoint): Promise<void> {
    const anchoring = this.#physicalStateAnchorTail
      .catch(() => undefined)
      .then(() => this.#anchorCurrentPhysicalStateOnce(checkpoint));
    this.#physicalStateAnchorTail = anchoring;
    await anchoring;
  }

  async #anchorCurrentPhysicalStateOnce(checkpoint: HumanoidRunCheckpoint): Promise<void> {
    const physicalStateSha256 = humanoidPhysicalStateSha256(
      checkpoint.world_checkpoint
    );
    const current = checkpoint.physical_state_anchor;
    if (current
      && current.world_frame === checkpoint.world.frame
      && current.world_revision === checkpoint.world.worldRevision
      && (current.version === 1
        ? current.world_checkpoint_sha256 === actionCommitPayloadSha256(
            json(checkpoint.world_checkpoint)
          )
        : current.physical_state_sha256 === physicalStateSha256)) {
      return;
    }
    const identity = {
      version: 2 as const,
      run_id: this.runId,
      world_frame: checkpoint.world.frame,
      world_revision: checkpoint.world.worldRevision,
      physical_state_sha256: physicalStateSha256
    };
    const eventId = `humanoid-physical-state:${actionCommitPayloadSha256(json(identity))}`;
    const existing = this.#physicalAnchorOrphanRecoveryPending
      ? await findDurableRuntimeEventById(this.#store, eventId)
      : undefined;
    this.#physicalAnchorOrphanRecoveryPending = false;
    const committed = existing ?? await findDurableRuntimeEventById(this.#store, eventId);
    const anchoredAt = committed?.at ?? new Date().toISOString();
    const anchor = HumanoidPhysicalStateAnchorSchema.options[1].parse({
      version: 2,
      event_id: eventId,
      world_frame: identity.world_frame,
      world_revision: identity.world_revision,
      physical_state_sha256: physicalStateSha256,
      anchored_at: anchoredAt
    });
    const runtimeEvent: RuntimeEvent = {
      event_id: eventId,
      run_id: this.runId,
      type: "humanoid_physical_state_anchored",
      at: anchoredAt,
      data: json({
        version: anchor.version,
        world_frame: anchor.world_frame,
        world_revision: anchor.world_revision,
        physical_state_sha256: anchor.physical_state_sha256
      })
    };
    let persisted: RuntimeEvent | undefined;
    if (committed) {
      persisted = committed;
    } else try {
      [persisted] = await this.#store.appendRuntimeEvents([runtimeEvent]);
    } catch (error) {
      const committed = await findDurableRuntimeEventById(this.#store, eventId);
      if (!committed) throw error;
      persisted = committed;
    }
    assertHumanoidPhysicalStateAnchorEvent(persisted!, this.runId, anchor);
    checkpoint.physical_state_anchor = anchor;
    try {
      await this.#eventSink(persisted!);
    } catch {
      return;
    }
  }

  async #anchorCurrentGoalState(checkpoint: HumanoidRunCheckpoint): Promise<void> {
    const anchoring = this.#goalStateAnchorTail
      .catch(() => undefined)
      .then(() => this.#anchorCurrentGoalStateOnce(checkpoint));
    this.#goalStateAnchorTail = anchoring;
    await anchoring;
  }

  async #anchorCurrentGoalStateOnce(checkpoint: HumanoidRunCheckpoint): Promise<void> {
    const goalDAG = structuredClone(checkpoint.goal_dag);
    const goalControlState = structuredClone(humanoidGoalControlState(checkpoint));
    const controlStateSha256 = actionCommitPayloadSha256(goalControlState);
    const current = checkpoint.goal_state_anchor;
    if (current?.goal_dag_state_sha256 === goalDAG.state_sha256
      && current.control_state_sha256 === controlStateSha256) return;
    const identity = {
      version: 1 as const,
      run_id: this.runId,
      goal_dag_state_sha256: goalDAG.state_sha256,
      control_state_sha256: controlStateSha256
    };
    const eventId = `humanoid-goal-state:${actionCommitPayloadSha256(json(identity))}`;
    const existing = this.#goalAnchorOrphanRecoveryPending
      ? await findDurableRuntimeEventById(this.#store, eventId)
      : undefined;
    this.#goalAnchorOrphanRecoveryPending = false;
    const committed = existing ?? await findDurableRuntimeEventById(this.#store, eventId);
    const anchoredAt = committed?.at ?? new Date().toISOString();
    const anchor = HumanoidGoalStateAnchorSchema.parse({
      version: 1,
      event_id: eventId,
      goal_dag_state_sha256: goalDAG.state_sha256,
      control_state_sha256: controlStateSha256,
      anchored_at: anchoredAt
    });
    const runtimeEvent: RuntimeEvent = {
      event_id: eventId,
      run_id: this.runId,
      type: "humanoid_goal_state_anchored",
      at: anchoredAt,
      data: json({
        version: anchor.version,
        goal_dag_state_sha256: anchor.goal_dag_state_sha256,
        control_state_sha256: anchor.control_state_sha256,
        goal_dag: goalDAG,
        goal_control_state: goalControlState
      })
    };
    let persisted: RuntimeEvent | undefined;
    if (committed) {
      persisted = committed;
    } else try {
      [persisted] = await this.#store.appendRuntimeEvents([runtimeEvent]);
    } catch (error) {
      const committed = await findDurableRuntimeEventById(this.#store, eventId);
      if (!committed) throw error;
      persisted = committed;
    }
    assertHumanoidGoalStateAnchorEvent(persisted!, this.runId, anchor);
    checkpoint.goal_state_anchor = anchor;
    try {
      await this.#eventSink(persisted!);
    } catch {
      return;
    }
  }

  async #anchorCurrentEmbodiedMemoryState(
    checkpoint: HumanoidRunCheckpoint
  ): Promise<void> {
    const anchoring = this.#embodiedMemoryStateAnchorTail
      .catch(() => undefined)
      .then(() => this.#anchorCurrentEmbodiedMemoryStateOnce(checkpoint));
    this.#embodiedMemoryStateAnchorTail = anchoring;
    await anchoring;
  }

  async #anchorCurrentEmbodiedMemoryStateOnce(
    checkpoint: HumanoidRunCheckpoint
  ): Promise<void> {
    const memory = checkpoint.embodied_memory;
    const memorySha256 = humanoidEmbodiedMemoryStateSha256(memory);
    const current = checkpoint.embodied_memory_state_anchor;
    if (current?.embodied_memory_sha256 === memorySha256) return;
    const identity = {
      version: 1 as const,
      run_id: this.runId,
      embodied_memory_sha256: memorySha256
    };
    const eventId = `humanoid-embodied-memory:${actionCommitPayloadSha256(json(identity))}`;
    const existing = this.#memoryAnchorOrphanRecoveryPending
      ? await findDurableRuntimeEventById(this.#store, eventId)
      : undefined;
    this.#memoryAnchorOrphanRecoveryPending = false;
    const committed = existing ?? await findDurableRuntimeEventById(this.#store, eventId);
    const anchoredAt = committed?.at ?? new Date().toISOString();
    const anchor = HumanoidEmbodiedMemoryStateAnchorSchema.parse({
      version: 1,
      event_id: eventId,
      embodied_memory_sha256: memorySha256,
      anchored_at: anchoredAt
    });
    const runtimeEvent: RuntimeEvent = {
      event_id: eventId,
      run_id: this.runId,
      type: "humanoid_embodied_memory_state_anchored",
      at: anchoredAt,
      data: json({
        version: anchor.version,
        embodied_memory_sha256: anchor.embodied_memory_sha256,
        embodied_memory: memory
      })
    };
    let persisted: RuntimeEvent | undefined;
    if (committed) {
      persisted = committed;
    } else try {
      [persisted] = await this.#store.appendRuntimeEvents([runtimeEvent]);
    } catch (error) {
      const committed = await findDurableRuntimeEventById(this.#store, eventId);
      if (!committed) throw error;
      persisted = committed;
    }
    assertHumanoidEmbodiedMemoryStateAnchorEvent(persisted!, this.runId, anchor);
    checkpoint.embodied_memory_state_anchor = anchor;
    try {
      await this.#eventSink(persisted!);
    } catch {
      return;
    }
  }

  async #anchorCurrentContextMemoryState(
    checkpoint: HumanoidRunCheckpoint
  ): Promise<void> {
    const anchoring = this.#contextMemoryStateAnchorTail
      .catch(() => undefined)
      .then(() => this.#anchorCurrentContextMemoryStateOnce(checkpoint));
    this.#contextMemoryStateAnchorTail = anchoring;
    await anchoring;
  }

  async #anchorCurrentContextMemoryStateOnce(
    checkpoint: HumanoidRunCheckpoint
  ): Promise<void> {
    const memory = checkpoint.context_memory;
    const memorySha256 = humanoidContextMemoryStateSha256(memory);
    const current = checkpoint.context_memory_state_anchor;
    if (current?.context_memory_sha256 === memorySha256) return;
    const identity = {
      version: 1 as const,
      run_id: this.runId,
      context_memory_sha256: memorySha256
    };
    const eventId = `humanoid-context-memory:${actionCommitPayloadSha256(json(identity))}`;
    const existing = this.#contextAnchorOrphanRecoveryPending
      ? await findDurableRuntimeEventById(this.#store, eventId)
      : undefined;
    this.#contextAnchorOrphanRecoveryPending = false;
    const committed = existing ?? await findDurableRuntimeEventById(this.#store, eventId);
    const anchoredAt = committed?.at ?? new Date().toISOString();
    const anchor = HumanoidContextMemoryStateAnchorSchema.parse({
      version: 1,
      event_id: eventId,
      context_memory_sha256: memorySha256,
      anchored_at: anchoredAt
    });
    const runtimeEvent: RuntimeEvent = {
      event_id: eventId,
      run_id: this.runId,
      type: "humanoid_context_memory_state_anchored",
      at: anchoredAt,
      data: json({
        version: anchor.version,
        context_memory_sha256: anchor.context_memory_sha256,
        context_memory: memory
      })
    };
    let persisted: RuntimeEvent | undefined;
    if (committed) {
      persisted = committed;
    } else try {
      [persisted] = await this.#store.appendRuntimeEvents([runtimeEvent]);
    } catch (error) {
      const committed = await findDurableRuntimeEventById(this.#store, eventId);
      if (!committed) throw error;
      persisted = committed;
    }
    assertHumanoidContextMemoryStateAnchorEvent(persisted!, this.runId, anchor);
    checkpoint.context_memory_state_anchor = anchor;
    try {
      await this.#eventSink(persisted!);
    } catch {
      return;
    }
  }

  async #anchorCurrentExecutionLedgerState(
    checkpoint: HumanoidRunCheckpoint
  ): Promise<void> {
    const anchoring = this.#executionLedgerStateAnchorTail
      .catch(() => undefined)
      .then(async () => {
        do {
          await this.#anchorCurrentExecutionLedgerStateOnce(checkpoint);
        } while (checkpoint.execution_ledger_state_anchor
          ?.execution_ledger_sha256 !== humanoidExecutionLedgerStateSha256(
            checkpoint.action_execution_ledger
          ));
      });
    this.#executionLedgerStateAnchorTail = anchoring;
    await anchoring;
  }

  async #anchorCurrentExecutionLedgerStateOnce(
    checkpoint: HumanoidRunCheckpoint
  ): Promise<void> {
    const ledger = structuredClone(checkpoint.action_execution_ledger);
    const ledgerSha256 = humanoidExecutionLedgerStateSha256(ledger);
    const current = checkpoint.execution_ledger_state_anchor;
    if (current?.execution_ledger_sha256 === ledgerSha256) {
      const latest = await latestDurableExecutionLedgerAnchorEvent(this.#store);
      if (latest?.event_id === current.event_id) return;
      const event = await findDurableRuntimeEventById(this.#store, current.event_id);
      if (!event) {
        throw new Error(`Execution ledger state anchor event is missing: ${current.event_id}`);
      }
      assertHumanoidExecutionLedgerStateAnchorEvent(event, this.runId, current);
      await this.#store.appendRuntimeEvents([{
        event_id: event.event_id,
        run_id: event.run_id,
        type: event.type,
        at: event.at,
        data: event.data
      }]);
      return;
    }
    const identity = {
      version: 1 as const,
      run_id: this.runId,
      execution_ledger_sha256: ledgerSha256
    };
    const eventId = `humanoid-execution-ledger:${actionCommitPayloadSha256(json(identity))}`;
    const existing = this.#executionLedgerAnchorOrphanRecoveryPending
      ? await findDurableRuntimeEventById(this.#store, eventId)
      : undefined;
    this.#executionLedgerAnchorOrphanRecoveryPending = false;
    const committed = existing ?? await findDurableRuntimeEventById(this.#store, eventId);
    const anchoredAt = committed?.at ?? new Date().toISOString();
    const anchor = HumanoidExecutionLedgerStateAnchorSchema.parse({
      version: 1,
      event_id: eventId,
      execution_ledger_sha256: ledgerSha256,
      anchored_at: anchoredAt
    });
    const runtimeEvent: RuntimeEvent = {
      event_id: eventId,
      run_id: this.runId,
      type: "humanoid_execution_ledger_state_anchored",
      at: anchoredAt,
      data: json({
        version: anchor.version,
        execution_ledger_sha256: anchor.execution_ledger_sha256,
        action_execution_ledger: ledger
      })
    };
    let persisted: RuntimeEvent | undefined;
    if (committed) {
      persisted = committed;
    } else try {
      [persisted] = await this.#store.appendRuntimeEvents([runtimeEvent]);
    } catch (error) {
      const committed = await findDurableRuntimeEventById(this.#store, eventId);
      if (!committed) throw error;
      persisted = committed;
    }
    assertHumanoidExecutionLedgerStateAnchorEvent(persisted!, this.runId, anchor);
    checkpoint.execution_ledger_state_anchor = anchor;
    try {
      await this.#eventSink(persisted!);
    } catch {
      return;
    }
  }

}

function activeRecoveryReplacementCommitment(
  state: NeuralHierarchyState
): NeuralSkillCommitment | undefined {
  const commitment = state.active_skill_commitment;
  if (!commitment || commitment.state !== "committed") return undefined;
  const recoveryId = HUMANOID_NEURAL_AGENT_IDS.recovery;
  const sensorimotorId = HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager;
  const actionSelectionId = HUMANOID_NEURAL_AGENT_IDS.actionSelection;
  const sourceSignals = commitment.source_signal_ids.map(
    (signalId) => state.signals[signalId]
  ).filter((signal): signal is NeuralSignal => signal !== undefined);
  return sourceSignals.some((proposal) => proposal.kind === "skill_proposal"
    && proposal.source_node_id === sensorimotorId
    && proposal.target_node_id === actionSelectionId
    && neuralSignalDescendsFromRecoveryProposal(state, proposal, recoveryId))
    ? commitment
    : undefined;
}

function recoveryEscalationAwaitsGoalValuation(
  state: NeuralHierarchyState
): boolean {
  return Object.values(state.signals).some((signal) => signal.status === "pending"
    && signal.kind === "escalation"
    && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
    && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.executive
    && neuralSignalDescendsFromRecoveryKind(state, signal, "escalation"));
}

function neuralSignalDescendsFromRecoveryProposal(
  state: NeuralHierarchyState,
  signal: NeuralSignal,
  recoveryNodeId: typeof HUMANOID_NEURAL_AGENT_IDS.recovery
): boolean {
  return neuralSignalDescendsFromRecoveryKind(
    state,
    signal,
    "skill_proposal",
    recoveryNodeId
  );
}

function neuralSignalDescendsFromRecoveryKind(
  state: NeuralHierarchyState,
  signal: NeuralSignal,
  kind: "skill_proposal" | "escalation",
  recoveryNodeId = HUMANOID_NEURAL_AGENT_IDS.recovery
): boolean {
  const visited = new Set<string>();
  const pending = [...signal.causal_parent_ids];
  while (pending.length > 0) {
    const signalId = pending.pop()!;
    if (visited.has(signalId)) continue;
    visited.add(signalId);
    const parent = state.signals[signalId];
    if (!parent) continue;
    if (parent.kind === kind
      && parent.source_node_id === recoveryNodeId
      && parent.target_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager) {
      return true;
    }
    pending.push(...parent.causal_parent_ids);
  }
  return false;
}

function verifiedMissionCompletionOutput(input: {
  missionGoal: Goal;
  modelCycle: JsonValue;
  checker: HumanoidCheckerResult;
  evidenceTransactionIds: readonly string[];
  worldFrame: number;
  worldRevision: number;
}): JsonValue {
  if (!input.checker.success
    || input.checker.worldFrame !== input.worldFrame
    || input.checker.worldRevision !== input.worldRevision
    || goalConstraintSha256(input.checker.goal)
      !== goalConstraintSha256(input.missionGoal)) {
    throw new Error("Mission completion output requires a successful current-world checker");
  }
  return {
    status: "mission_completed",
    mission_goal: json(input.missionGoal),
    checker: json(input.checker),
    world_frame: input.worldFrame,
    world_revision: input.worldRevision,
    evidence_transaction_ids: [...input.evidenceTransactionIds],
    model_summary: cycleSummary(input.modelCycle)
  };
}

function assertHumanoidPhysicalStateAnchorEvent(
  event: RuntimeEvent,
  runId: string,
  anchor: ReturnType<typeof HumanoidPhysicalStateAnchorSchema.parse>
): void {
  const data = object(event.data);
  if (event.event_id !== anchor.event_id
    || event.run_id !== runId
    || event.type !== "humanoid_physical_state_anchored"
    || event.at !== anchor.anchored_at
    || data.version !== anchor.version
    || data.world_frame !== anchor.world_frame
    || data.world_revision !== anchor.world_revision
    || (anchor.version === 1
      ? data.world_checkpoint_sha256 !== anchor.world_checkpoint_sha256
      : data.physical_state_sha256 !== anchor.physical_state_sha256)) {
    throw new Error(`Physical state anchor event conflicts with ${anchor.event_id}`);
  }
}

function assertHumanoidGoalStateAnchorEvent(
  event: RuntimeEvent,
  runId: string,
  anchor: ReturnType<typeof HumanoidGoalStateAnchorSchema.parse>
): void {
  const data = object(event.data);
  if (event.event_id !== anchor.event_id
    || event.run_id !== runId
    || event.type !== "humanoid_goal_state_anchored"
    || event.at !== anchor.anchored_at
    || data.version !== anchor.version
    || data.goal_dag_state_sha256 !== anchor.goal_dag_state_sha256
    || data.control_state_sha256 !== anchor.control_state_sha256
    || object(data.goal_dag ?? null).state_sha256
      !== anchor.goal_dag_state_sha256
    || data.goal_control_state === undefined
    || actionCommitPayloadSha256(json(data.goal_control_state))
      !== anchor.control_state_sha256) {
    throw new Error(`Goal state anchor event conflicts with ${anchor.event_id}`);
  }
}

function assertHumanoidEmbodiedMemoryStateAnchorEvent(
  event: RuntimeEvent,
  runId: string,
  anchor: ReturnType<typeof HumanoidEmbodiedMemoryStateAnchorSchema.parse>
): void {
  const data = object(event.data);
  if (event.event_id !== anchor.event_id
    || event.run_id !== runId
    || event.type !== "humanoid_embodied_memory_state_anchored"
    || event.at !== anchor.anchored_at
    || data.version !== anchor.version
    || data.embodied_memory_sha256 !== anchor.embodied_memory_sha256
    || data.embodied_memory === undefined
    || humanoidEmbodiedMemoryStateSha256(
      data.embodied_memory as HumanoidRunCheckpoint["embodied_memory"]
    ) !== anchor.embodied_memory_sha256) {
    throw new Error(
      `Embodied memory state anchor event conflicts with ${anchor.event_id}`
    );
  }
}

function assertHumanoidContextMemoryStateAnchorEvent(
  event: RuntimeEvent,
  runId: string,
  anchor: ReturnType<typeof HumanoidContextMemoryStateAnchorSchema.parse>
): void {
  const data = object(event.data);
  if (event.event_id !== anchor.event_id
    || event.run_id !== runId
    || event.type !== "humanoid_context_memory_state_anchored"
    || event.at !== anchor.anchored_at
    || data.version !== anchor.version
    || data.context_memory_sha256 !== anchor.context_memory_sha256
    || data.context_memory === undefined
    || humanoidContextMemoryStateSha256(
      data.context_memory as HumanoidRunCheckpoint["context_memory"]
    ) !== anchor.context_memory_sha256) {
    throw new Error(
      `Context memory state anchor event conflicts with ${anchor.event_id}`
    );
  }
}

function assertHumanoidExecutionLedgerStateAnchorEvent(
  event: RuntimeEvent,
  runId: string,
  anchor: ReturnType<typeof HumanoidExecutionLedgerStateAnchorSchema.parse>
): void {
  const data = object(event.data);
  if (event.event_id !== anchor.event_id
    || event.run_id !== runId
    || event.type !== "humanoid_execution_ledger_state_anchored"
    || event.at !== anchor.anchored_at
    || data.version !== anchor.version
    || data.execution_ledger_sha256 !== anchor.execution_ledger_sha256
    || data.action_execution_ledger === undefined
    || humanoidExecutionLedgerStateSha256(
      data.action_execution_ledger as HumanoidRunCheckpoint["action_execution_ledger"]
    ) !== anchor.execution_ledger_sha256) {
    throw new Error(
      `Execution ledger state anchor event conflicts with ${anchor.event_id}`
    );
  }
}

async function findDurableRuntimeEventById(
  store: RunStore,
  eventId: string
): Promise<RuntimeEvent | undefined> {
  const pageSize = 256;
  const tail = await store.readJournalTail("events", 1);
  let before = tail.total;
  while (before > 0) {
    const from = Math.max(0, before - pageSize);
    const page = await store.readJournalPage("events", from, before - from);
    if (page.entries.length === 0) {
      throw new Error(`Runtime event journal stopped before offset ${before}`);
    }
    for (let offset = page.entries.length - 1; offset >= 0; offset -= 1) {
      const record = object(page.entries[offset]!);
      if (record.event_id !== eventId) continue;
      if (typeof record.run_id !== "string"
        || typeof record.type !== "string"
        || typeof record.at !== "string") {
        throw new Error(`Durable runtime event is malformed: ${eventId}`);
      }
      return {
        event_id: eventId,
        run_id: record.run_id,
        type: record.type,
        at: record.at,
        data: json(record.data ?? null),
        ...(typeof record.cursor === "string" ? { cursor: record.cursor } : {})
      };
    }
    before = from;
  }
  return undefined;
}

async function assertNoDurableAnchorDowngrade(
  store: RunStore,
  eventType: string,
  label: string
): Promise<void> {
  const pageSize = 256;
  const tail = await store.readJournalTail("events", 1);
  let before = tail.total;
  while (before > 0) {
    const from = Math.max(0, before - pageSize);
    const page = await store.readJournalPage("events", from, before - from);
    if (page.entries.length === 0) {
      throw new Error(`Runtime event journal stopped before offset ${before}`);
    }
    if (page.entries.some((entry) => object(entry).type === eventType)) {
      throw new Error(`${label} is missing while durable anchor history exists`);
    }
    before = from;
  }
}

async function latestDurableExecutionLedgerAnchorEvent(
  store: RunStore
): Promise<(RuntimeEvent & {
  anchor: ReturnType<typeof HumanoidExecutionLedgerStateAnchorSchema.parse>;
}) | undefined> {
  const eventType = "humanoid_execution_ledger_state_anchored";
  const pageSize = 256;
  const tail = await store.readJournalTail("events", 1);
  let before = tail.total;
  while (before > 0) {
    const from = Math.max(0, before - pageSize);
    const page = await store.readJournalPage("events", from, before - from);
    if (page.entries.length === 0) {
      throw new Error(`Runtime event journal stopped before offset ${before}`);
    }
    for (let offset = page.entries.length - 1; offset >= 0; offset -= 1) {
      const record = object(page.entries[offset]!);
      if (record.type !== eventType) continue;
      if (typeof record.event_id !== "string"
        || typeof record.run_id !== "string"
        || typeof record.at !== "string") {
        throw new Error(`Durable ${eventType} event is malformed`);
      }
      const data = object(record.data ?? null);
      const anchor = HumanoidExecutionLedgerStateAnchorSchema.parse({
        version: data.version,
        event_id: record.event_id,
        execution_ledger_sha256: data.execution_ledger_sha256,
        anchored_at: record.at
      });
      return {
        event_id: record.event_id,
        run_id: record.run_id,
        type: eventType,
        at: record.at,
        data: json(record.data ?? null),
        ...(typeof record.cursor === "string" ? { cursor: record.cursor } : {}),
        anchor
      };
    }
    before = from;
  }
  return undefined;
}

async function latestDurableGoalStateAnchorEvent(
  store: RunStore
): Promise<(RuntimeEvent & {
  anchor: ReturnType<typeof HumanoidGoalStateAnchorSchema.parse>;
}) | undefined> {
  const eventType = "humanoid_goal_state_anchored";
  const pageSize = 256;
  const tail = await store.readJournalTail("events", 1);
  let before = tail.total;
  while (before > 0) {
    const from = Math.max(0, before - pageSize);
    const page = await store.readJournalPage("events", from, before - from);
    if (page.entries.length === 0) {
      throw new Error(`Runtime event journal stopped before offset ${before}`);
    }
    for (let offset = page.entries.length - 1; offset >= 0; offset -= 1) {
      const record = object(page.entries[offset]!);
      if (record.type !== eventType) continue;
      if (typeof record.event_id !== "string"
        || typeof record.run_id !== "string"
        || typeof record.at !== "string") {
        throw new Error(`Durable ${eventType} event is malformed`);
      }
      const data = object(record.data ?? null);
      const anchor = HumanoidGoalStateAnchorSchema.parse({
        version: data.version,
        event_id: record.event_id,
        goal_dag_state_sha256: data.goal_dag_state_sha256,
        control_state_sha256: data.control_state_sha256,
        anchored_at: record.at
      });
      return {
        event_id: record.event_id,
        run_id: record.run_id,
        type: eventType,
        at: record.at,
        data: json(record.data ?? null),
        ...(typeof record.cursor === "string" ? { cursor: record.cursor } : {}),
        anchor
      };
    }
    before = from;
  }
  return undefined;
}

async function durableExecutionCommitProofsExist(
  store: RunStore,
  commits: readonly ReturnType<typeof stageActionCommit>["pending"][string][],
  receipts: HumanoidRunCheckpoint["committed_actions"]
): Promise<boolean> {
  const requiredTransactions = new Set(commits.map((entry) => entry.transaction_id));
  const requiredEvidence = new Set(commits.map((entry) => entry.goal_evidence_ref));
  const requiredEvents = new Set(commits.map((entry) => entry.runtime_event_id));
  const actions = new Map<string, Set<string>>();
  const evidence = new Map<string, Set<string>>();
  const identities = new Map<string, Set<string>>();
  const events = new Map<string, Set<string>>();
  await Promise.all([
    store.scanJournal("actions", (record) => {
      const transactionId = object(record).transactionId;
      if (typeof transactionId === "string" && requiredTransactions.has(transactionId)) {
        addDurableProofHash(actions, transactionId, actionCommitPayloadSha256(record));
      }
    }),
    store.scanJournal("goal_evidence", (record) => {
      const ref = object(object(record).evidence ?? null).ref;
      if (typeof ref === "string" && requiredEvidence.has(ref)) {
        addDurableProofHash(evidence, ref, actionCommitPayloadSha256(record));
      }
    }),
    store.scanJournal("action_identities", (record) => {
      const transactionId = object(record).transaction_id;
      if (typeof transactionId === "string" && requiredTransactions.has(transactionId)) {
        addDurableProofHash(identities, transactionId, actionCommitPayloadSha256(record));
      }
    }),
    store.scanJournal("events", (record) => {
      const envelope = object(record);
      const eventId = envelope.event_id;
      if (typeof eventId !== "string" || !requiredEvents.has(eventId)) return;
      const { cursor: _cursor, ...event } = envelope;
      addDurableProofHash(events, eventId, actionCommitPayloadSha256(json(event)));
    })
  ]);
  return commits.every((entry) => {
    const receipt = receipts[entry.transaction_id];
    const expectedIdentity = createActionTransactionIdentity(entry);
    return receipt !== undefined
      && exactlyOneDurableProof(actions, entry.transaction_id, entry.action_record_sha256)
      && exactlyOneDurableProof(evidence, entry.goal_evidence_ref, entry.goal_evidence_sha256)
      && exactlyOneDurableProof(
        identities,
        entry.transaction_id,
        actionCommitPayloadSha256(json(expectedIdentity))
      )
      && exactlyOneDurableProof(events, entry.runtime_event_id, entry.runtime_event_sha256);
  });
}

function addDurableProofHash(
  index: Map<string, Set<string>>,
  id: string,
  sha256: string
): void {
  const hashes = index.get(id) ?? new Set<string>();
  hashes.add(sha256);
  index.set(id, hashes);
}

function exactlyOneDurableProof(
  index: ReadonlyMap<string, ReadonlySet<string>>,
  id: string,
  expectedSha256: string
): boolean {
  const hashes = index.get(id);
  return hashes?.size === 1 && hashes.has(expectedSha256);
}

async function assertDurableAnchorIsLatest(
  store: RunStore,
  eventType: string,
  eventId: string,
  label: string
): Promise<void> {
  const pageSize = 256;
  const tail = await store.readJournalTail("events", 1);
  let before = tail.total;
  while (before > 0) {
    const from = Math.max(0, before - pageSize);
    const page = await store.readJournalPage("events", from, before - from);
    if (page.entries.length === 0) {
      throw new Error(`Runtime event journal stopped before offset ${before}`);
    }
    for (let offset = page.entries.length - 1; offset >= 0; offset -= 1) {
      const record = object(page.entries[offset]!);
      if (record.type !== eventType) continue;
      if (record.event_id !== eventId) {
        throw new Error(`${label} is not the latest durable state`);
      }
      return;
    }
    before = from;
  }
  throw new Error(`${label} has no durable state history`);
}

async function appendDurableModelCallLifecycleRecord(
  store: RunStore,
  record: ModelCallLifecycleRecord
): Promise<void> {
  try {
    await store.append("model_calls", json(record));
  } catch (error) {
    const committed = await findDurableModelCallLifecycleRecord(store, record);
    if (!committed) throw error;
  }
}

async function findDurableModelCallLifecycleRecord(
  store: RunStore,
  record: ModelCallLifecycleRecord
): Promise<boolean> {
  const pageSize = 256;
  const expectedSha256 = modelPayloadSha256(record);
  const tail = await store.readJournalTail("model_calls", 1);
  let before = tail.total;
  while (before > 0) {
    const from = Math.max(0, before - pageSize);
    const page = await store.readJournalPage("model_calls", from, before - from);
    if (page.entries.length === 0) {
      throw new Error(`Model call journal stopped before offset ${before}`);
    }
    for (let offset = page.entries.length - 1; offset >= 0; offset -= 1) {
      const candidate = object(page.entries[offset]!);
      if (candidate.model_call_id !== record.model_call_id
        || candidate.lifecycle !== record.lifecycle) continue;
      if (modelPayloadSha256(candidate) !== expectedSha256) {
        throw new Error(
          `Durable model call lifecycle identity conflict: ${record.model_call_id}`
        );
      }
      return true;
    }
    before = from;
  }
  return false;
}

function assertHumanoidSkillRuntimeEvent(
  runtimeEvent: RuntimeEvent,
  runId: string,
  event: HumanoidEmbodiedSkillEvent
): void {
  if (runtimeEvent.run_id !== runId
    || runtimeEvent.type !== "humanoid_skill_event"
    || modelPayloadSha256(runtimeEvent.data) !== modelPayloadSha256(event)) {
    throw new Error(`Durable Skill event identity conflict: ${runtimeEvent.event_id}`);
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

async function loadDurableHumanoidSkillEvents(
  store: RunStore,
  retainedCallIds: ReadonlySet<string>
): Promise<HumanoidEmbodiedSkillEvent[]> {
  if (retainedCallIds.size === 0) return [];
  const indexed: Array<{ index: number; event: HumanoidEmbodiedSkillEvent }> = [];
  const starts = new Set<string>();
  const seen = new Set<string>();
  const tail = await store.readJournalTail("events", 1);
  let before = tail.total;
  while (before > 0 && starts.size < retainedCallIds.size) {
    const from = Math.max(0, before - 256);
    const page = await store.readJournalPage("events", from, before - from);
    if (page.entries.length === 0) {
      throw new Error(`Skill event journal stopped before offset ${before}`);
    }
    for (let offset = page.entries.length - 1; offset >= 0; offset -= 1) {
      const envelope = object(page.entries[offset]!);
      if (envelope.type !== "humanoid_skill_event"
        || envelope.data === undefined) continue;
      const data = object(envelope.data);
      if (data.status === undefined) continue;
      const status = object(data.status);
      if (typeof status.callId !== "string"
        || !retainedCallIds.has(status.callId)) continue;
      const event = HumanoidEmbodiedSkillEventSchema.parse(envelope.data);
      indexed.push({ index: from + offset, event });
      seen.add(event.status.callId);
      if (event.sequence === 0) starts.add(event.status.callId);
    }
    before = from;
  }
  const missing = [...seen].filter((callId) => !starts.has(callId));
  if (missing.length > 0) {
    throw new Error(`Durable Skill event history is incomplete: ${missing[0]}`);
  }
  return indexed
    .sort((left, right) => left.index - right.index)
    .map(({ event }) => event);
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

function replanBudgetRole(
  agentId: string
): "coordinator" | "motion" | "goal_manager" | undefined {
  if (agentId === HUMANOID_NEURAL_AGENT_IDS.executive
    || agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection) return "coordinator";
  if (agentId === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
    || agentId === HUMANOID_NEURAL_AGENT_IDS.premotor
    || agentId === HUMANOID_NEURAL_AGENT_IDS.motorIntent) return "motion";
  if (agentId === HUMANOID_NEURAL_AGENT_IDS.goalManager) return "goal_manager";
  return undefined;
}

function replanModelCallEvent(
  call: HumanoidReplanModelCall | undefined,
  cycle: ActiveAutonomousCycle | null
): Record<string, JsonValue> {
  if (!call || !cycle) return {};
  return {
    replan_budget_evidence: json({
      call,
      authority: humanoidReplanBudgetAuthority(
        cycle.replan_budget,
        call.completed_at ?? call.started_at
      )
    })
  };
}

function isHumanoidPlanningReceipt(receipt: HumanoidActionReceipt): boolean {
  return receipt.action === "plan_humanoid_skill"
    || receipt.action === "plan_whole_body_motion"
    || receipt.action === "plan_whole_body_motion_candidates"
    || receipt.action === "plan_humanoid_navigation";
}

function humanoidExecutionActionForPlan(
  action: HumanoidActionReceipt["action"]
): "execute_humanoid_skill" | "execute_whole_body_motion" | "execute_humanoid_navigation" {
  if (action === "plan_humanoid_skill") return "execute_humanoid_skill";
  if (action === "plan_humanoid_navigation") return "execute_humanoid_navigation";
  if (action === "plan_whole_body_motion"
    || action === "plan_whole_body_motion_candidates") {
    return "execute_whole_body_motion";
  }
  throw new Error(`Humanoid action is not a planning receipt: ${action}`);
}

function planningTransactionIdFromReceipt(
  receipt: HumanoidActionReceipt
): string | undefined {
  const input = object(receipt.input);
  return typeof input.planning_transaction_id === "string"
    ? input.planning_transaction_id
    : undefined;
}

function humanoidActionRoleAuthority(action: HumanoidActionReceipt["action"]): Set<string> {
  if (action === "observe_humanoid") {
    return new Set([HUMANOID_NEURAL_AGENT_IDS.sensorFusion]);
  }
  if (action === "submit_humanoid_skill_plan"
    || action === "begin_humanoid_skill"
    || isHumanoidPlanningActionName(action)) {
    return new Set([HUMANOID_NEURAL_AGENT_IDS.motorIntent]);
  }
  return new Set([HUMANOID_NEURAL_AGENT_IDS.executor]);
}

function isHumanoidSensorFusionActor(agentId: string): boolean {
  // Read-side checkpoint compatibility only. humanoidActionRoleAuthority()
  // above does not grant the retired V2 Sentry any new observation call.
  return agentId === HUMANOID_NEURAL_AGENT_IDS.sensorFusion
    || agentId === "humanoid-sentry";
}

function isHumanoidPlanningActionName(
  action: HumanoidActionReceipt["action"]
): boolean {
  return action === "plan_humanoid_skill"
    || action === "plan_whole_body_motion"
    || action === "plan_whole_body_motion_candidates"
    || action === "plan_humanoid_navigation";
}
