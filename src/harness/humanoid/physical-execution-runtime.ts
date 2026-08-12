import {
  actionExecutionFingerprintSha256,
  activeActionExecutions,
  recordActionExecutionProgress,
  stageActionExecutionIntent,
  ExecutionGateToolCallAuthoritySchema,
  type ActionExecutionLedgerEntry
} from "../../domain/action-execution-ledger.js";
import type { PendingActionCommit } from "../../domain/action-commit-outbox.js";
import {
  sameAutonomousCycle,
  type AutonomousCycleRef
} from "../../domain/autonomous-cycle.js";
import type { HumanoidRunCheckpoint } from "../../domain/humanoid-run.js";
import type { PhysicalTrajectorySummary } from "../../domain/physical-trajectory.js";
import type { Goal, Scenario } from "../../domain/schema.js";
import type { HumanoidGroundingReceipt } from
  "../../domain/humanoid-grounding.js";
import { advanceHumanoidGoal } from "../../runtime/humanoid-checker.js";
import type {
  HumanoidWorld,
  HumanoidWorldSnapshot
} from "../../world/humanoid/world.js";
import {
  json,
  object,
  physicalCheckpointSha256,
  physicalExecutionCheckpointDue,
  physicalExecutionReceipt
} from "./run-runtime-persistence.js";
import {
  advancePhysicalTrajectory,
  createPhysicalTrajectory
} from "./physical-trajectory-recorder.js";
import type {
  HumanoidActionReceipt,
  HumanoidPhysicalExecutionIntent
} from "./runtime.js";
import { groundHumanoidPhysicalExecution } from "./dispatch-grounding.js";

const EXECUTION_FRAME_CHECKPOINT_INTERVAL = 10;
const STATIONARY_CHECKPOINT_INTERVAL_SECONDS = 5 * 60;

type HumanoidPersistenceCut = Awaited<
  ReturnType<HumanoidWorld["capturePersistenceState"]>
>;

export class HumanoidPhysicalExecutionRuntime {
  readonly #runId: string;
  readonly #world: HumanoidWorld;
  readonly #checkpoint: () => HumanoidRunCheckpoint;
  readonly #scenario: () => Scenario;
  readonly #activeGoal: () => Goal | undefined;
  readonly #requiredActiveCycle: () => AutonomousCycleRef;
  readonly #persist: (refreshWorld?: boolean) => Promise<void>;
  readonly #emitFrame: (input: {
    world: HumanoidWorldSnapshot;
    checker: HumanoidRunCheckpoint["checker"];
    goalProgress: HumanoidRunCheckpoint["goal_progress"];
    source: "execution" | "stationary";
  }) => Promise<void>;
  readonly #signal: AbortSignal | undefined;
  readonly #physicalTrajectories = new Map<string, PhysicalTrajectorySummary>();
  #lastStationaryCheckpointSimulationTime: number;

  constructor(input: {
    runId: string;
    world: HumanoidWorld;
    checkpoint: () => HumanoidRunCheckpoint;
    scenario: () => Scenario;
    activeGoal: () => Goal | undefined;
    requiredActiveCycle: () => AutonomousCycleRef;
    persist: (refreshWorld?: boolean) => Promise<void>;
    emitFrame: (input: {
      world: HumanoidWorldSnapshot;
      checker: HumanoidRunCheckpoint["checker"];
      goalProgress: HumanoidRunCheckpoint["goal_progress"];
      source: "execution" | "stationary";
    }) => Promise<void>;
    signal?: AbortSignal;
  }) {
    this.#runId = input.runId;
    this.#world = input.world;
    this.#checkpoint = input.checkpoint;
    this.#scenario = input.scenario;
    this.#activeGoal = input.activeGoal;
    this.#requiredActiveCycle = input.requiredActiveCycle;
    this.#persist = input.persist;
    this.#emitFrame = input.emitFrame;
    this.#signal = input.signal;
    this.#lastStationaryCheckpointSimulationTime =
      input.checkpoint().world.robot.simulatedTime;
  }

  assertExecutionOwner(transactionId: string): void {
    const active = activeActionExecutions(
      this.#checkpoint().action_execution_ledger
    );
    if (active.length > 1) {
      throw new Error("Multiple physical executions cannot share one humanoid runtime");
    }
    const owner = active[0];
    if (owner && owner.transaction_id !== transactionId.trim()) {
      throw new Error(
        `Physical execution ${owner.transaction_id} must be recovered before ${transactionId}`
      );
    }
  }

  executionFrameOffset(transactionId: string): number {
    const entry = this.#checkpoint().action_execution_ledger.active[
      transactionId.trim()
    ];
    return entry?.progress.committed_frame_count ?? 0;
  }

  executionCompletedPlanFrameCount(transactionId: string): number {
    const entry = this.#checkpoint().action_execution_ledger.active[
      transactionId.trim()
    ];
    const last = entry?.progress.completed_plan_terminals.at(-1);
    return entry && last
      ? last.final_world_revision - entry.admission.world_revision
      : 0;
  }

  executionCompletedPlanCount(transactionId: string): number {
    return this.#checkpoint().action_execution_ledger.active[
      transactionId.trim()
    ]?.progress.completed_plan_terminals.length ?? 0;
  }

  async admit(
    intent: HumanoidPhysicalExecutionIntent
  ): Promise<HumanoidGroundingReceipt | undefined> {
    if (!intent.decision || !intent.toolAuthority) {
      throw new Error(
        `Physical execution has no durable Coordinator delegation: ${intent.transactionId}`
      );
    }
    const toolCallAuthority = ExecutionGateToolCallAuthoritySchema.parse(
      intent.toolAuthority
    );
    const checkpoint = this.#checkpoint();
    const cycle = this.#requiredActiveCycle();
    const planningReceipt = checkpoint.committed_actions[intent.planningTransactionId];
    if (!planningReceipt || !sameAutonomousCycle(planningReceipt.cycle, cycle)) {
      throw new Error(
        `Physical execution plan belongs to another autonomous cycle: ${intent.planningTransactionId}`
      );
    }
    this.assertExecutionOwner(intent.transactionId);
    const existing = checkpoint.action_execution_ledger.active[intent.transactionId];
    if (existing) {
      const cut = await this.#capturePhysicalCut();
      this.#applyPhysicalCut(cut);
      this.#assertExecutionIntent(existing, intent);
      if (existing.status !== "terminal") {
        this.#synchronizeExecutionProgress(existing, cut);
      }
      await this.#persist();
      return existing.admission.grounding_receipt;
    }
    const observation = await this.#world.captureObservation();
    const cut = await this.#capturePhysicalCut();
    if (observation.frame !== cut.world.frame
      || observation.worldRevision !== cut.world.worldRevision) {
      throw new Error("Grounding observation is not aligned with physical authority");
    }
    this.#applyPhysicalCut(cut);
    const activeGoal = this.#activeGoal();
    const grounding = groundHumanoidPhysicalExecution({
      planningReceipt,
      intent,
      observation,
      authorityStateSha256: cut.authority.stateSha256,
      ...(activeGoal ? { activeGoal } : {})
    });
    if (!grounding.accepted) return grounding;
    const trajectory = createPhysicalTrajectory(cut.world);
    this.#physicalTrajectories.set(intent.transactionId, trajectory);
    checkpoint.action_execution_ledger = stageActionExecutionIntent(
      checkpoint.action_execution_ledger,
      {
        runId: this.#runId,
        transactionId: intent.transactionId,
        agentId: intent.agentId,
        action: intent.action,
        actionFingerprint: intent.fingerprint,
        cycle,
        planningTransactionId: intent.planningTransactionId,
        planId: intent.planId,
        worldFrame: cut.world.frame,
        worldRevision: cut.world.worldRevision,
        authorityStateSha256: cut.authority.stateSha256,
        physicalCheckpointSha256: physicalCheckpointSha256(cut),
        decision: intent.decision,
        toolCallAuthority,
        physicalTrajectory: trajectory,
        groundingReceipt: grounding
      }
    );
    await this.#persist();
    return grounding;
  }

  async normalizeReceipt(
    receipt: HumanoidActionReceipt
  ): Promise<HumanoidActionReceipt> {
    const checkpoint = this.#checkpoint();
    const cycle = this.#requiredActiveCycle();
    if (receipt.cycle && !sameAutonomousCycle(receipt.cycle, cycle)) {
      throw new Error(
        `Humanoid action receipt changed autonomous cycle: ${receipt.transactionId}`
      );
    }
    receipt = { ...receipt, cycle };
    if (!physicalExecutionReceipt(receipt)) return receipt;
    const entry = checkpoint.action_execution_ledger.active[receipt.transactionId];
    if (!entry) {
      if (object(receipt.detail).automatic_actuation === false) return receipt;
      throw new Error(
        `Physical receipt has no durable execution intent: ${receipt.transactionId}`
      );
    }
    if (entry.agent_id !== receipt.agentId
      || entry.action !== receipt.action
      || entry.action_fingerprint_sha256
        !== actionExecutionFingerprintSha256(receipt.fingerprint)) {
      throw new Error(
        `Physical receipt conflicts with its durable execution intent: ${receipt.transactionId}`
      );
    }
    await this.synchronizeProgress(receipt.transactionId);
    const current = checkpoint.action_execution_ledger.active[receipt.transactionId]!;
    const frameCount = current.progress.committed_frame_count;
    const detail = object(receipt.detail);
    detail.frames = frameCount;
    const trajectory = current.progress.physical_trajectory;
    if (!trajectory) {
      throw new Error(
        `Physical receipt has no authoritative trajectory: ${receipt.transactionId}`
      );
    }
    detail.physical_trajectory = json(trajectory);
    if (current.admission.grounding_receipt) {
      detail.grounding_receipt = json(current.admission.grounding_receipt);
    }
    if (current.progress.completed_plan_terminals.length > 0) {
      detail.completed_plan_terminals = json(
        current.progress.completed_plan_terminals
      );
    }
    return {
      ...receipt,
      worldBeforeRevision: current.admission.world_revision,
      worldAfterRevision: current.progress.world_revision,
      frameCount,
      detail
    };
  }

  async synchronizeProgress(transactionId: string): Promise<void> {
    const checkpoint = this.#checkpoint();
    const entry = checkpoint.action_execution_ledger.active[transactionId];
    if (!entry) {
      throw new Error(`Physical execution ledger is unavailable: ${transactionId}`);
    }
    const cut = await this.#capturePhysicalCut();
    this.#applyPhysicalCut(cut);
    if (entry.status !== "terminal") {
      this.#synchronizeExecutionProgress(entry, cut);
    }
  }

  async synchronizePendingExecution(): Promise<{
    transactionId: string;
    previousCommittedFrameCount: number;
    committedFrameCount: number;
    completeTrajectory: boolean;
  } | undefined> {
    const pending = activeActionExecutions(
      this.#checkpoint().action_execution_ledger
    );
    if (pending.length > 1) {
      throw new Error("Multiple physical executions cannot share one humanoid runtime");
    }
    const [entry] = pending;
    if (!entry || entry.status === "terminal") return undefined;
    const previousCommittedFrameCount = entry.progress.committed_frame_count;
    await this.synchronizeProgress(entry.transaction_id);
    const current = this.#checkpoint().action_execution_ledger.active[
      entry.transaction_id
    ];
    if (!current
      || current.progress.committed_frame_count === previousCommittedFrameCount) {
      return undefined;
    }
    return {
      transactionId: entry.transaction_id,
      previousCommittedFrameCount,
      committedFrameCount: current.progress.committed_frame_count,
      completeTrajectory:
        current.progress.physical_trajectory?.complete_from_admission === true
    };
  }

  async recordFrame(
    frame: HumanoidWorldSnapshot,
    source: "execution" | "stationary"
  ): Promise<void> {
    const checkpoint = this.#checkpoint();
    const advanced = this.#advanceGoal(frame);
    if (source === "stationary") {
      if (frame.robot.simulatedTime - this.#lastStationaryCheckpointSimulationTime
        >= STATIONARY_CHECKPOINT_INTERVAL_SECONDS) {
        const cut = await this.#capturePhysicalCut();
        const durableAdvance = this.#advanceGoal(cut.world);
        this.#applyPhysicalCut(cut);
        checkpoint.goal_progress = durableAdvance?.progress ?? null;
        checkpoint.checker = durableAdvance?.checker ?? null;
        await this.#persist(false);
        this.#lastStationaryCheckpointSimulationTime =
          cut.world.robot.simulatedTime;
      }
    } else {
      const [entry] = activeActionExecutions(checkpoint.action_execution_ledger);
      if (!entry) {
        throw new Error("Physical execution frame has no durable execution intent");
      }
      this.#recordTrajectoryFrame(entry, frame);
      const cut = await this.#capturePhysicalCut();
      const durableAdvance = this.#advanceGoal(cut.world);
      this.#applyPhysicalCut(cut);
      checkpoint.goal_progress = durableAdvance?.progress ?? null;
      checkpoint.checker = durableAdvance?.checker ?? null;
      await this.#persistExecutionCut(cut);
    }
    if (this.#signal?.aborted) return;
    await this.#publishFrame(frame, advanced, source);
  }

  async recordPhysicalCut(cut: HumanoidPersistenceCut): Promise<void> {
    this.#assertAlignedCut(cut, "Humanoid physical frame cut is not aligned");
    const checkpoint = this.#checkpoint();
    const frame = cut.world;
    const advanced = this.#advanceGoal(frame);
    this.#applyPhysicalCut(cut);
    const [entry] = activeActionExecutions(checkpoint.action_execution_ledger);
    if (!entry) {
      throw new Error("Physical execution cut has no durable execution intent");
    }
    this.#recordTrajectoryFrame(entry, frame);
    checkpoint.goal_progress = advanced?.progress ?? null;
    checkpoint.checker = advanced?.checker ?? null;
    await this.#persistExecutionCut(cut);
    if (this.#signal?.aborted) return;
    await this.#publishFrame(frame, advanced, "execution");
  }

  async acknowledgeTerminals(
    staged: readonly PendingActionCommit[]
  ): Promise<void> {
    const checkpoint = this.#checkpoint();
    let changed = false;
    for (const entry of staged) {
      if (checkpoint.action_commit_outbox.pending[entry.transaction_id]) continue;
      const action = object(entry.action_record);
      const detail = object(action.detail ?? null);
      const durableTerminals = checkpoint.action_execution_ledger.active[
        entry.transaction_id
      ]?.progress.completed_plan_terminals ?? [];
      const terminals = durableTerminals.length > 0
        ? durableTerminals.map((terminal) => ({
            kind: terminal.kind,
            planId: terminal.plan_id,
            resultSha256: terminal.result_sha256
          }))
        : physicalPlanTerminals(action.action, detail);
      for (const terminal of terminals) {
        changed = terminal.kind === "motion"
          ? await this.#world.acknowledgeWholeBodyMotion(
              terminal.planId,
              terminal.resultSha256
            ) || changed
          : await this.#world.acknowledgeNavigation(
              terminal.planId,
              terminal.resultSha256
            ) || changed;
      }
      this.#physicalTrajectories.delete(entry.transaction_id);
    }
    if (!changed) return;
    try {
      await this.#persist();
    } catch {
      return;
    }
  }

  async #persistExecutionCut(cut: HumanoidPersistenceCut): Promise<void> {
    const checkpoint = this.#checkpoint();
    const [entry] = activeActionExecutions(checkpoint.action_execution_ledger);
    if (!entry) {
      throw new Error("Physical frame was committed without a durable execution intent");
    }
    if (!physicalExecutionCheckpointDue(
      entry,
      cut,
      EXECUTION_FRAME_CHECKPOINT_INTERVAL
    )) return;
    if (entry.status !== "terminal") this.#recordExecutionProgress(entry, cut);
    await this.#persist(false);
  }

  #advanceGoal(frame: HumanoidWorldSnapshot) {
    const checkpoint = this.#checkpoint();
    const activeGoal = this.#activeGoal();
    return activeGoal && checkpoint.goal_progress
      ? advanceHumanoidGoal(
          activeGoal,
          this.#scenario(),
          frame,
          checkpoint.goal_progress
        )
      : null;
  }

  async #publishFrame(
    frame: HumanoidWorldSnapshot,
    advanced: ReturnType<typeof advanceHumanoidGoal> | null,
    source: "execution" | "stationary"
  ): Promise<void> {
    try {
      await this.#emitFrame({
        world: frame,
        checker: advanced?.checker ?? null,
        goalProgress: advanced?.progress ?? null,
        source
      });
    } catch {
      return;
    }
  }

  #assertExecutionIntent(
    entry: ActionExecutionLedgerEntry,
    intent: HumanoidPhysicalExecutionIntent
  ): void {
    const cycle = this.#requiredActiveCycle();
    if (entry.run_id !== this.#runId
      || entry.agent_id !== intent.agentId
      || !sameAutonomousCycle(entry.cycle, cycle)
      || entry.action !== intent.action
      || entry.action_fingerprint_sha256
        !== actionExecutionFingerprintSha256(intent.fingerprint)
      || entry.admission.planning_transaction_id !== intent.planningTransactionId
      || entry.admission.plan_id !== intent.planId
      || !intent.decision
      || !intent.toolAuthority
      || JSON.stringify(entry.admission.decision) !== JSON.stringify(intent.decision)
      || JSON.stringify(entry.admission.tool_call_authority)
        !== JSON.stringify(intent.toolAuthority)) {
      throw new Error(
        `Physical execution retry conflicts with durable intent: ${intent.transactionId}`
      );
    }
  }

  #recordExecutionProgress(
    entry: ActionExecutionLedgerEntry,
    cut: HumanoidPersistenceCut
  ): void {
    const checkpoint = this.#checkpoint();
    const committedFrameCount = cut.world.worldRevision - entry.admission.world_revision;
    if (committedFrameCount < entry.progress.committed_frame_count
      || cut.world.frame !== entry.admission.world_frame + committedFrameCount) {
      throw new Error(
        `Physical checkpoint regressed from durable execution: ${entry.transaction_id}`
      );
    }
    if (committedFrameCount === 0
      && entry.progress.committed_frame_count === 0) {
      if (cut.authority.stateSha256 !== entry.progress.authority_state_sha256) {
        throw new Error(
          `Physical admission authority changed without a frame: ${entry.transaction_id}`
        );
      }
      return;
    }
    checkpoint.action_execution_ledger = recordActionExecutionProgress(
      checkpoint.action_execution_ledger,
      {
        transactionId: entry.transaction_id,
        committedFrameCount,
        worldFrame: cut.world.frame,
        worldRevision: cut.world.worldRevision,
        authorityStateSha256: cut.authority.stateSha256,
        physicalCheckpointSha256: physicalCheckpointSha256(cut),
        physicalTrajectory: this.#requiredTrajectory(entry, cut.world),
        completedPlanTerminals: completedExecutionPlanTerminals(entry, cut)
      }
    );
  }

  #synchronizeExecutionProgress(
    entry: ActionExecutionLedgerEntry,
    cut: HumanoidPersistenceCut
  ): void {
    this.#recoverRestoredPhysicalTail(entry, cut);
    this.#recordTrajectoryFrame(entry, cut.world);
    this.#recordExecutionProgress(entry, cut);
  }

  #recoverRestoredPhysicalTail(
    entry: ActionExecutionLedgerEntry,
    cut: HumanoidPersistenceCut
  ): void {
    if (this.#physicalTrajectories.has(entry.transaction_id)
      || cut.world.worldRevision <= entry.progress.world_revision + 1) return;

    const committedFrameCount = cut.world.worldRevision
      - entry.admission.world_revision;
    if (committedFrameCount <= entry.progress.committed_frame_count
      || cut.world.frame !== entry.admission.world_frame + committedFrameCount) {
      throw new Error(
        `Restored physical tail is not aligned with execution ${entry.transaction_id}`
      );
    }
    const planId = entry.admission.plan_id;
    const activeSkillCallId = executionSkillCallId(entry, cut);
    const motion = activeSkillCallId === undefined
      ? cut.worldCheckpoint.motions.find(({ plan }) => plan.id === planId)
      : cut.worldCheckpoint.motions.find((candidate) => (
          candidate.skillCallIdentity?.callId === activeSkillCallId
          && candidate.terminal === null
        )) ?? [...cut.worldCheckpoint.motions].reverse().find((candidate) => (
          candidate.skillCallIdentity?.callId === activeSkillCallId
          && candidate.terminal?.final_world_revision === cut.world.worldRevision
        ));
    const route = activeSkillCallId === undefined
      ? cut.worldCheckpoint.routes.find(({ id }) => id === planId)
      : cut.worldCheckpoint.routes.find((candidate) => (
          candidate.skillCallIdentity?.callId === activeSkillCallId
          && candidate.terminal === null
        )) ?? [...cut.worldCheckpoint.routes].reverse().find((candidate) => (
          candidate.skillCallIdentity?.callId === activeSkillCallId
          && candidate.terminal?.final_world_revision === cut.world.worldRevision
        ));
    const expectsMotion = entry.action === "execute_whole_body_motion";
    const expectsRoute = entry.action === "execute_humanoid_navigation";
    const matchedPlanCount = Number(Boolean(motion)) + Number(Boolean(route));
    if ((expectsMotion && (!motion || route))
      || (expectsRoute && (!route || motion))
      || (!expectsMotion && !expectsRoute && matchedPlanCount !== 1)) {
      throw new Error(
        `Restored physical plan is unavailable for execution ${entry.transaction_id}`
      );
    }
    const planProgressRevision = motion
      ? motion.terminal?.final_world_revision
        ?? (motion.validatedRevision ?? motion.createdRevision)
          + motion.progress.nextFrameIndex
      : route
        ? route.terminal?.final_world_revision
          ?? (route.progress
            ? (route.validatedRevision ?? route.createdRevision)
              + route.progress.committed_frame_count
            : undefined)
        : undefined;
    if (planProgressRevision !== cut.world.worldRevision) {
      throw new Error(
        `Restored physical plan progress is not aligned with execution ${entry.transaction_id}`
      );
    }

    // Older checkpoints could persist the exact MuJoCo state after the last
    // periodic ledger cut. The missing intermediate observations cannot be
    // reconstructed, so resume from the authoritative endpoint and mark the
    // trajectory as incomplete instead of inventing samples.
    this.#physicalTrajectories.set(
      entry.transaction_id,
      createPhysicalTrajectory(cut.world, false)
    );
  }

  #recordTrajectoryFrame(
    entry: ActionExecutionLedgerEntry,
    frame: HumanoidWorldSnapshot
  ): PhysicalTrajectorySummary {
    const existing = this.#physicalTrajectories.get(entry.transaction_id)
      ?? entry.progress.physical_trajectory
      ?? createPhysicalTrajectory(frame, false);
    const advanced = advancePhysicalTrajectory(existing, frame);
    this.#physicalTrajectories.set(entry.transaction_id, advanced);
    return advanced;
  }

  #requiredTrajectory(
    entry: ActionExecutionLedgerEntry,
    frame: HumanoidWorldSnapshot
  ): PhysicalTrajectorySummary {
    const trajectory = this.#recordTrajectoryFrame(entry, frame);
    if (trajectory.end_frame !== frame.frame
      || trajectory.end_world_revision !== frame.worldRevision) {
      throw new Error(
        `Physical trajectory is not aligned with execution ${entry.transaction_id}`
      );
    }
    return trajectory;
  }

  async #capturePhysicalCut(): Promise<HumanoidPersistenceCut> {
    const cut = await this.#world.capturePersistenceState();
    this.#assertAlignedCut(cut, "Humanoid persistence cut is not physically aligned");
    return cut;
  }

  #assertAlignedCut(cut: HumanoidPersistenceCut, message: string): void {
    if (cut.authority.revision !== cut.world.worldRevision
      || cut.world.frame !== cut.worldCheckpoint.frame
      || cut.world.worldRevision !== cut.worldCheckpoint.worldRevision) {
      throw new Error(message);
    }
  }

  #applyPhysicalCut(cut: HumanoidPersistenceCut): void {
    const checkpoint = this.#checkpoint();
    checkpoint.world = structuredClone(cut.world);
    checkpoint.world_checkpoint = structuredClone(cut.worldCheckpoint);
  }
}

function executionSkillCallId(
  entry: ActionExecutionLedgerEntry,
  cut: HumanoidPersistenceCut
): string | undefined {
  const admissionMotion = cut.worldCheckpoint.motions.find(
    ({ plan }) => plan.id === entry.admission.plan_id
  );
  const admissionRoute = cut.worldCheckpoint.routes.find(
    ({ id }) => id === entry.admission.plan_id
  );
  const last = entry.progress.completed_plan_terminals.at(-1);
  const completedMotion = last?.kind === "motion"
    ? cut.worldCheckpoint.motions.find(({ plan }) => plan.id === last.plan_id)
    : undefined;
  const completedRoute = last?.kind === "navigation"
    ? cut.worldCheckpoint.routes.find(({ id }) => id === last.plan_id)
    : undefined;
  return admissionMotion?.skillCallIdentity?.callId
    ?? admissionRoute?.skillCallIdentity?.callId
    ?? completedMotion?.skillCallIdentity?.callId
    ?? completedRoute?.skillCallIdentity?.callId;
}

function completedExecutionPlanTerminals(
  entry: ActionExecutionLedgerEntry,
  cut: HumanoidPersistenceCut
): ActionExecutionLedgerEntry["progress"]["completed_plan_terminals"] {
  if (entry.action !== "execute_humanoid_skill") {
    return entry.progress.completed_plan_terminals;
  }
  const callId = executionSkillCallId(entry, cut);
  if (!callId) return entry.progress.completed_plan_terminals;
  return [
    ...cut.worldCheckpoint.motions.flatMap((motion) => (
      motion.skillCallIdentity?.callId === callId && motion.terminal
        ? [{
            kind: "motion" as const,
            plan_id: motion.plan.id,
            result_sha256: motion.terminal.result_sha256,
            final_frame: motion.terminal.final_frame,
            final_world_revision: motion.terminal.final_world_revision
          }]
        : []
    )),
    ...cut.worldCheckpoint.routes.flatMap((route) => (
      route.skillCallIdentity?.callId === callId && route.terminal
        ? [{
            kind: "navigation" as const,
            plan_id: route.id,
            result_sha256: route.terminal.result_sha256,
            final_frame: route.terminal.final_frame,
            final_world_revision: route.terminal.final_world_revision
          }]
        : []
    ))
  ].sort((left, right) => (
    left.final_world_revision - right.final_world_revision
      || left.plan_id.localeCompare(right.plan_id)
  ));
}

interface PhysicalPlanTerminal {
  kind: "motion" | "navigation";
  planId: string;
  resultSha256: string;
}

function physicalPlanTerminals(
  action: unknown,
  detail: Record<string, unknown>
): PhysicalPlanTerminal[] {
  if (action !== "execute_whole_body_motion"
    && action !== "execute_humanoid_navigation"
    && action !== "execute_humanoid_skill") return [];

  const kind = action === "execute_humanoid_navigation"
    || (action === "execute_humanoid_skill"
      && detail.autonomous_plan_kind === "navigation")
    ? "navigation"
    : "motion";
  const durable = Array.isArray(detail.completed_plan_terminals)
    ? detail.completed_plan_terminals.flatMap((value): PhysicalPlanTerminal[] => {
        const terminal = object(value);
        return (terminal.kind === "motion" || terminal.kind === "navigation")
          && typeof terminal.plan_id === "string"
          && typeof terminal.result_sha256 === "string"
          ? [{
              kind: terminal.kind,
              planId: terminal.plan_id,
              resultSha256: terminal.result_sha256
            }]
          : [];
      })
    : [];
  const result = unknownRecord(detail.result);
  const horizon = unknownRecord(result.articulation_horizon);
  const navigationHorizon = unknownRecord(result.navigation_horizon);
  const articulationSegments = Array.isArray(horizon.segments) ? horizon.segments : [];
  const navigationSegments = Array.isArray(navigationHorizon.segments)
    ? navigationHorizon.segments
    : [];
  const segments = articulationSegments.length > 0
    ? articulationSegments
    : navigationSegments;
  const terminals = durable.length > 0 ? durable : segments.flatMap(
    (value): PhysicalPlanTerminal[] => {
    const segment = object(value);
    return typeof segment.plan_id === "string"
      && typeof segment.terminal_result_sha256 === "string"
      ? [{
          kind: navigationSegments.length > 0 ? "navigation" : "motion",
          planId: segment.plan_id,
          resultSha256: segment.terminal_result_sha256
        }]
      : [];
    });
  if (terminals.length === 0
    && typeof detail.plan_id === "string"
    && typeof detail.terminal_result_sha256 === "string") {
    terminals.push({
      kind,
      planId: detail.plan_id,
      resultSha256: detail.terminal_result_sha256
    });
  }
  const unique = new Map<string, PhysicalPlanTerminal>();
  for (const terminal of terminals) {
    const key = `${terminal.kind}:${terminal.planId}`;
    const previous = unique.get(key);
    if (previous && previous.resultSha256 !== terminal.resultSha256) {
      throw new Error(`Conflicting physical terminal evidence: ${terminal.planId}`);
    }
    unique.set(key, terminal);
  }
  return [...unique.values()];
}

function unknownRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}
