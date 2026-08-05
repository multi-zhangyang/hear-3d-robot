import {
  HUMANOID_END_EFFECTORS,
  type JsonValue
} from "../../domain/schema.js";
import {
  modelPayloadSha256,
  type ModelDecisionRef
} from "../../domain/model-call-authority.js";
import type { AutonomousCycleRef } from "../../domain/autonomous-cycle.js";
import type { ScenarioBlockRemovalTransaction } from "../../domain/scenario-block-removal.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import type { HumanoidBodyChannel } from "../../world/humanoid/motion-plan.js";
import {
  type HumanoidFrameSink,
  type HumanoidPersistenceSink,
  type HumanoidWorld,
  type HumanoidWorldObservation,
  type HumanoidWorldSnapshot
} from "../../world/humanoid/world.js";
import {
  HumanoidActionInputs,
  type HumanoidActionName
} from "./actions.js";
import { MAX_CHECKPOINT_ACTION_RECEIPTS } from "./embodied-memory.js";
import { normalizeHumanoidMotionCandidateBatchInput } from "./motion-candidate-input.js";
import { BlockRemovalAuthorityError } from "./block-removal-authority.js";

type HumanoidPlanningActionName = "plan_whole_body_motion"
  | "plan_whole_body_motion_candidates"
  | "plan_humanoid_navigation";

type HumanoidPhysicalActionName = "execute_whole_body_motion"
  | "execute_humanoid_navigation";

export interface HumanoidPhysicalExecutionIntent {
  transactionId: string;
  agentId: string;
  action: HumanoidPhysicalActionName;
  fingerprint: string;
  planningTransactionId: string;
  planId: string;
}

export interface HumanoidActionToolCallAuthority {
  tool_call_id: string;
  tool_name: string;
  arguments_sha256: string;
}

export interface HumanoidActionInvoker {
  invoke(
    name: HumanoidActionName,
    rawInput: unknown,
    transactionId: string,
    agentId: string,
    authority: HumanoidActionToolCallAuthority
  ): Promise<HumanoidActionReceipt>;
}

export interface HumanoidActionReceipt {
  transactionId: string;
  agentId: string;
  decision?: ModelDecisionRef | undefined;
  cycle?: AutonomousCycleRef | undefined;
  action: HumanoidActionName;
  input: JsonValue;
  fingerprint: string;
  accepted: boolean;
  code: string;
  worldBeforeRevision: number;
  worldAfterRevision: number;
  frameCount: number;
  channels: HumanoidBodyChannel[];
  detail: JsonValue;
  committedAt: string;
}

export interface HumanoidActionRuntimeOptions {
  frameSink?: HumanoidFrameSink;
  physicalFrameSink?: HumanoidPersistenceSink;
  receiptSink?: (receipt: HumanoidActionReceipt) => void | Promise<void>;
  beforePhysicalExecution?: (
    intent: HumanoidPhysicalExecutionIntent
  ) => void | Promise<void>;
  receiptNormalizer?: (
    receipt: HumanoidActionReceipt
  ) => HumanoidActionReceipt | Promise<HumanoidActionReceipt>;
  prepareBlockRemoval?: (input: {
    transactionId: string;
    agentId: string;
    solidId: string;
    executionTransactionId: string;
  }) => ScenarioBlockRemovalTransaction | Promise<ScenarioBlockRemovalTransaction>;
  receipts?: Readonly<Record<string, HumanoidActionReceipt>>;
  realtimeExecution?: boolean;
  retainPhysicalTerminals?: boolean;
  signal?: AbortSignal;
}

export class HumanoidActionRuntime {
  readonly #world: HumanoidWorld;
  readonly #frameSink: HumanoidFrameSink | undefined;
  readonly #physicalFrameSink: HumanoidPersistenceSink | undefined;
  readonly #receiptSink: HumanoidActionRuntimeOptions["receiptSink"];
  readonly #beforePhysicalExecution: HumanoidActionRuntimeOptions[
    "beforePhysicalExecution"
  ];
  readonly #receiptNormalizer: HumanoidActionRuntimeOptions["receiptNormalizer"];
  readonly #prepareBlockRemoval: HumanoidActionRuntimeOptions["prepareBlockRemoval"];
  readonly #realtimeExecution: boolean;
  readonly #retainPhysicalTerminals: boolean;
  readonly #signal: AbortSignal | undefined;
  readonly #receipts = new Map<string, HumanoidActionReceipt>();
  readonly #transactions = new Map<string, {
    fingerprint: string;
    decisionSha256: string | undefined;
    promise: Promise<HumanoidActionReceipt>;
  }>();
  readonly #receiptCommits = new Map<string, Promise<void>>();
  readonly #uncommittedReceiptIds = new Set<string>();
  readonly #planChannels = new Map<string, HumanoidBodyChannel[]>();
  readonly #inFlightTransactions = new Set<string>();

  constructor(world: HumanoidWorld, options: HumanoidActionRuntimeOptions = {}) {
    this.#world = world;
    this.#frameSink = options.frameSink;
    this.#physicalFrameSink = options.physicalFrameSink;
    this.#receiptSink = options.receiptSink;
    this.#beforePhysicalExecution = options.beforePhysicalExecution;
    this.#receiptNormalizer = options.receiptNormalizer;
    this.#prepareBlockRemoval = options.prepareBlockRemoval;
    this.#realtimeExecution = options.realtimeExecution ?? false;
    this.#retainPhysicalTerminals = options.retainPhysicalTerminals ?? false;
    this.#signal = options.signal;
    for (const [transactionId, source] of Object.entries(options.receipts ?? {})) {
      const receipt = structuredClone(source);
      if (receipt.transactionId !== transactionId) {
        throw new Error(`Humanoid receipt identity mismatch: ${transactionId}`);
      }
      const fingerprint = humanoidActionFingerprint(
        receipt.action,
        receipt.agentId,
        receipt.input
      );
      if (receipt.fingerprint !== fingerprint) {
        throw new Error(`Humanoid receipt fingerprint mismatch: ${transactionId}`);
      }
      this.#receipts.set(transactionId, receipt);
      this.#transactions.set(transactionId, {
        fingerprint,
        decisionSha256: receipt.decision
          ? modelPayloadSha256(receipt.decision)
          : undefined,
        promise: Promise.resolve(receipt)
      });
      if (receipt.accepted
        && (receipt.action === "plan_whole_body_motion"
          || receipt.action === "plan_whole_body_motion_candidates"
          || receipt.action === "plan_humanoid_navigation")) {
        const planId = jsonObject(receipt.detail)?.plan_id;
        if (typeof planId === "string" && planId) {
          this.#planChannels.set(planId, [...receipt.channels]);
        }
      }
    }
    this.#pruneTransactionHistory();
  }

  snapshot(): HumanoidWorldSnapshot {
    return this.#world.snapshot();
  }

  receipt(transactionId: string): HumanoidActionReceipt | undefined {
    const receipt = this.#receipts.get(transactionId);
    return receipt ? structuredClone(receipt) : undefined;
  }

  async invoke(
    name: HumanoidActionName,
    rawInput: unknown,
    transactionId: string,
    agentId: string,
    decision?: ModelDecisionRef
  ): Promise<HumanoidActionReceipt> {
    const normalizedTransactionId = transactionId.trim();
    const normalizedAgentId = agentId.trim();
    if (!normalizedTransactionId) throw new Error("Humanoid action transaction id is required");
    if (!normalizedAgentId) throw new Error("Humanoid action agent id is required");
    const fingerprint = humanoidActionFingerprint(name, normalizedAgentId, rawInput);
    const decisionSha256 = decision ? modelPayloadSha256(decision) : undefined;
    const existing = this.#transactions.get(normalizedTransactionId);
    if (existing) {
      if (existing.fingerprint !== fingerprint
        || existing.decisionSha256 !== decisionSha256) {
        throw new Error(
          `Humanoid action transaction conflict: ${normalizedTransactionId}`
        );
      }
      const receipt = await existing.promise;
      await this.#ensureReceiptCommitted(receipt);
      return structuredClone(receipt);
    }
    const unresolved = this.#uncommittedReceiptIds.values().next().value;
    if (unresolved !== undefined) {
      throw new Error(
        `Humanoid action commit is unresolved; retry transaction ${unresolved} before executing another action`
      );
    }
    this.#inFlightTransactions.add(normalizedTransactionId);
    const promise = this.#invokeOnce(
      name,
      rawInput,
      normalizedTransactionId,
      normalizedAgentId,
      fingerprint,
      decision
    );
    this.#transactions.set(normalizedTransactionId, {
      fingerprint,
      decisionSha256,
      promise
    });
    try {
      const receipt = await promise;
      await this.#ensureReceiptCommitted(receipt);
      return structuredClone(receipt);
    } catch (error) {
      if (!this.#receipts.has(normalizedTransactionId)) {
        this.#transactions.delete(normalizedTransactionId);
      }
      throw error;
    } finally {
      this.#inFlightTransactions.delete(normalizedTransactionId);
      this.#pruneTransactionHistory();
    }
  }

  #pruneTransactionHistory(): void {
    const activePlanIds = new Set(this.#world.consumablePlanIds());
    const protectedTransactions = new Set(this.#inFlightTransactions);
    for (const [transactionId, receipt] of this.#receipts) {
      const planId = planIdFromReceipt(receipt);
      if (receipt.accepted
        && isPlanningAction(receipt.action)
        && planId !== undefined
        && activePlanIds.has(planId)) {
        protectedTransactions.add(transactionId);
      }
    }
    for (const transactionId of this.#uncommittedReceiptIds) {
      protectedTransactions.add(transactionId);
    }
    const recentTransactions = new Set(
      [...this.#receipts.keys()].slice(-MAX_CHECKPOINT_ACTION_RECEIPTS)
    );
    for (const transactionId of this.#receipts.keys()) {
      if (!recentTransactions.has(transactionId)
        && !protectedTransactions.has(transactionId)) {
        this.#receipts.delete(transactionId);
      }
    }
    for (const transactionId of this.#transactions.keys()) {
      if (!this.#receipts.has(transactionId)
        && !protectedTransactions.has(transactionId)) {
        this.#transactions.delete(transactionId);
      }
    }
    for (const planId of this.#planChannels.keys()) {
      if (!activePlanIds.has(planId)) this.#planChannels.delete(planId);
    }
  }

  async #invokeOnce(
    name: HumanoidActionName,
    rawInput: unknown,
    transactionId: string,
    agentId: string,
    fingerprint: string,
    decision: ModelDecisionRef | undefined
  ): Promise<HumanoidActionReceipt> {
    const before = this.#world.snapshot();
    const result = await this.#execute(name, rawInput, {
      transactionId,
      agentId,
      fingerprint
    });
    const after = this.#world.snapshot();
    const baseReceipt: HumanoidActionReceipt = {
      transactionId,
      agentId,
      ...(decision ? { decision: structuredClone(decision) } : {}),
      action: name,
      input: jsonValue(rawInput),
      fingerprint,
      accepted: result.accepted,
      code: result.code,
      worldBeforeRevision: before.worldRevision,
      worldAfterRevision: result.causalWorldAfterRevision
        ?? after.worldRevision,
      frameCount: result.causalFrameCount ?? 0,
      channels: result.channels,
      detail: jsonValue(result.detail),
      committedAt: new Date().toISOString()
    };
    const receipt = this.#receiptNormalizer
      ? await this.#receiptNormalizer(structuredClone(baseReceipt))
      : baseReceipt;
    assertNormalizedReceiptIdentity(baseReceipt, receipt);
    this.#receipts.set(transactionId, receipt);
    if (this.#receiptSink) this.#uncommittedReceiptIds.add(transactionId);
    return structuredClone(receipt);
  }

  async #ensureReceiptCommitted(receipt: HumanoidActionReceipt): Promise<void> {
    const receiptSink = this.#receiptSink;
    if (!receiptSink || !this.#uncommittedReceiptIds.has(receipt.transactionId)) return;
    let commit = this.#receiptCommits.get(receipt.transactionId);
    if (!commit) {
      commit = Promise.resolve()
        .then(() => receiptSink(structuredClone(receipt)))
        .then(() => {
          this.#uncommittedReceiptIds.delete(receipt.transactionId);
        })
        .finally(() => {
          this.#receiptCommits.delete(receipt.transactionId);
        });
      this.#receiptCommits.set(receipt.transactionId, commit);
    }
    await commit;
  }

  async #execute(
    name: HumanoidActionName,
    rawInput: unknown,
    invocation: {
      transactionId: string;
      agentId: string;
      fingerprint: string;
    }
  ): Promise<{
    accepted: boolean;
    code: string;
    channels: HumanoidBodyChannel[];
    detail: unknown;
    causalFrameCount?: number;
    causalWorldAfterRevision?: number;
  }> {
    if (name === "observe_humanoid") {
      HumanoidActionInputs.observe_humanoid.parse(rawInput);
      return {
        accepted: true,
        code: "humanoid_observed",
        channels: [],
        detail: modelObservation(this.#world.observe())
      };
    }
    if (name === "plan_whole_body_motion") {
      const plan = HumanoidActionInputs.plan_whole_body_motion.parse(rawInput);
      const result = await this.#world.planWholeBodyMotion(plan);
      if (result.accepted) this.#planChannels.set(result.planId, result.channels);
      return {
        accepted: result.accepted,
        code: result.accepted ? "whole_body_plan_validated" : "whole_body_plan_rejected",
        channels: result.channels,
        detail: {
          plan_id: result.planId,
          created_revision: result.createdRevision,
          expires_revision: result.expiresRevision,
          intent_sha256: result.intentSha256,
          motion: result.motion,
          validation: {
            feasible: result.validation.feasible,
            failures: result.validation.failures,
            evidence: result.validation.evidence,
            predicted_final: conciseRobot(result.validation.finalSnapshot)
          }
        }
      };
    }
    if (name === "plan_whole_body_motion_candidates") {
      const batch = normalizeHumanoidMotionCandidateBatchInput(
        HumanoidActionInputs.plan_whole_body_motion_candidates.parse(rawInput)
      );
      const result = await this.#world.planWholeBodyMotionCandidates(batch);
      if (result.accepted) this.#planChannels.set(result.planId, result.channels);
      return {
        accepted: result.accepted,
        code: result.accepted
          ? "whole_body_candidates_validated"
          : "whole_body_candidates_rejected",
        channels: result.channels,
        detail: {
          plan_id: result.planId,
          objective: batch.objective,
          created_revision: result.createdRevision,
          expires_revision: result.expiresRevision,
          intent_sha256: result.intentSha256,
          selection: result.selection,
          selected_candidate_id: result.selectedCandidateId,
          selected_rank: result.selectedRank,
          candidate_count: result.candidates.length,
          termination: batch.termination,
          option: result.option,
          motion: result.motion,
          candidates: result.candidates.map((candidate) => ({
            rank: candidate.rank,
            plan_id: candidate.planId,
            intent: candidate.intent,
            intent_sha256: candidate.intentSha256,
            selected: candidate.planId === result.selectedCandidateId,
            channels: candidate.channels,
            motion: candidate.motion,
            option_certificate: candidate.optionCertificate,
            validation: {
              feasible: candidate.validation.feasible,
              failures: candidate.validation.failures,
              evidence: candidate.validation.evidence,
              predicted_final: conciseRobot(candidate.validation.finalSnapshot)
            }
          }))
        }
      };
    }
    if (name === "execute_whole_body_motion") {
      const input = HumanoidActionInputs.execute_whole_body_motion.parse(rawInput);
      const reference = this.#planningReference(
        input.planning_transaction_id,
        ["plan_whole_body_motion", "plan_whole_body_motion_candidates"]
      );
      if (!reference.accepted) return reference.result;
      const channels = this.#planChannels.get(reference.planId) ?? [];
      if (!this.#world.consumablePlanIds().includes(reference.planId)) {
        this.#planChannels.delete(reference.planId);
        return {
          accepted: false,
          code: "plan_stale",
          channels,
          detail: {
            planning_transaction_id: input.planning_transaction_id,
            planning_action: reference.planningAction,
            ...reference.candidateSelection,
            plan_id: reference.planId,
            automatic_actuation: false,
            reason: "validated plan is no longer consumable at the current world revision"
          }
        };
      }
      await this.#beforePhysicalExecution?.({
        transactionId: invocation.transactionId,
        agentId: invocation.agentId,
        action: name,
        fingerprint: invocation.fingerprint,
        planningTransactionId: input.planning_transaction_id,
        planId: reference.planId
      });
      const result = await this.#world.executeWholeBodyMotion(
        reference.planId,
        this.#frameSink,
        {
          realtime: this.#realtimeExecution,
          retainTerminal: this.#retainPhysicalTerminals,
          ...(this.#physicalFrameSink
            ? { persistenceSink: this.#physicalFrameSink }
            : {}),
          ...(this.#signal ? { signal: this.#signal } : {})
        }
      );
      if (!this.#retainPhysicalTerminals) this.#planChannels.delete(reference.planId);
      return {
        accepted: result.accepted,
        code: result.code,
        channels,
        causalFrameCount: result.frames,
        causalWorldAfterRevision: result.finalSnapshot.worldRevision,
        detail: {
          planning_transaction_id: input.planning_transaction_id,
          planning_action: reference.planningAction,
          ...reference.candidateSelection,
          plan_id: reference.planId,
          frames: result.frames,
          ...(result.terminalResultSha256
            ? { terminal_result_sha256: result.terminalResultSha256 }
            : {}),
          result: result.detail,
          final: conciseRobot(result.finalSnapshot.robot)
        }
      };
    }
    if (name === "plan_humanoid_navigation") {
      const input = HumanoidActionInputs.plan_humanoid_navigation.parse(rawInput);
      const result = await this.#world.planNavigation(input.target);
      if (result.accepted) this.#planChannels.set(result.planId, ["locomotion"]);
      return {
        accepted: result.accepted,
        code: result.accepted ? "humanoid_route_validated" : "humanoid_route_rejected",
        channels: ["locomotion"],
        detail: {
          plan_id: result.planId,
          created_revision: result.createdRevision,
          expires_revision: result.expiresRevision,
          intent_sha256: result.intentSha256,
          target: result.target,
          chunk_target: result.chunkTarget,
          waypoints: result.waypoints,
          distance: result.distance,
          remaining_distance: result.remainingDistance,
          carry: result.carry,
          ...(result.reason ? { reason: result.reason } : {})
        }
      };
    }
    if (name === "remove_world_block") {
      const input = HumanoidActionInputs.remove_world_block.parse(rawInput);
      if (!this.#prepareBlockRemoval) {
        throw new Error("Block-removal authority is unavailable");
      }
      try {
        const transaction = await this.#prepareBlockRemoval({
          transactionId: invocation.transactionId,
          agentId: invocation.agentId,
          solidId: input.solid_id,
          executionTransactionId: input.execution_transaction_id
        });
        return {
          accepted: true,
          code: "world_block_removal_authorized",
          channels: [],
          detail: {
            solid_id: input.solid_id,
            execution_transaction_id: input.execution_transaction_id,
            automatic_actuation: false,
            removal_transaction: transaction
          }
        };
      } catch (error) {
        if (!(error instanceof BlockRemovalAuthorityError)) throw error;
        return {
          accepted: false,
          code: error.code,
          channels: [],
          detail: {
            solid_id: input.solid_id,
            execution_transaction_id: input.execution_transaction_id,
            automatic_actuation: false,
            reason: error.message
          }
        };
      }
    }
    const input = HumanoidActionInputs.execute_humanoid_navigation.parse(rawInput);
    const reference = this.#planningReference(
      input.planning_transaction_id,
      ["plan_humanoid_navigation"]
    );
    if (!reference.accepted) return reference.result;
    if (!this.#world.consumablePlanIds().includes(reference.planId)) {
      this.#planChannels.delete(reference.planId);
      return {
        accepted: false,
        code: "plan_stale",
        channels: ["locomotion"],
        detail: {
          planning_transaction_id: input.planning_transaction_id,
          planning_action: reference.planningAction,
          plan_id: reference.planId,
          automatic_actuation: false,
          reason: "validated route is no longer consumable at the current world revision"
        }
      };
    }
    await this.#beforePhysicalExecution?.({
      transactionId: invocation.transactionId,
      agentId: invocation.agentId,
      action: name,
      fingerprint: invocation.fingerprint,
      planningTransactionId: input.planning_transaction_id,
      planId: reference.planId
    });
    const result = await this.#world.executeNavigation(
      reference.planId,
      this.#frameSink,
      {
        realtime: this.#realtimeExecution,
        retainTerminal: this.#retainPhysicalTerminals,
        ...(this.#physicalFrameSink
          ? { persistenceSink: this.#physicalFrameSink }
          : {}),
        ...(this.#signal ? { signal: this.#signal } : {})
      }
    );
    if (!this.#retainPhysicalTerminals) this.#planChannels.delete(reference.planId);
    return {
      accepted: result.accepted,
      code: result.code,
      channels: ["locomotion"],
      causalFrameCount: result.frames,
      causalWorldAfterRevision: result.finalSnapshot.worldRevision,
      detail: {
        planning_transaction_id: input.planning_transaction_id,
        planning_action: reference.planningAction,
        plan_id: reference.planId,
        frames: result.frames,
        ...(result.terminalResultSha256
          ? { terminal_result_sha256: result.terminalResultSha256 }
          : {}),
        result: result.detail,
        final: conciseRobot(result.finalSnapshot.robot)
      }
    };
  }

  #planningReference(
    transactionId: string,
    expectedActions: readonly HumanoidPlanningActionName[]
  ): {
    accepted: true;
    planId: string;
    planningAction: HumanoidPlanningActionName;
    candidateSelection: {
      candidate_count: number;
      selected_rank: number;
      selected_candidate_id: string;
    } | undefined;
  } | {
    accepted: false;
    result: {
      accepted: false;
      code: string;
      channels: HumanoidBodyChannel[];
      detail: JsonValue;
    };
  } {
    const receipt = this.#receipts.get(transactionId);
    if (!receipt) {
      return rejectedPlanningReference(
        "planning_receipt_missing",
        transactionId,
        expectedActions
      );
    }
    if (!expectedActions.includes(receipt.action as HumanoidPlanningActionName)) {
      return rejectedPlanningReference(
        "planning_receipt_action_mismatch",
        transactionId,
        expectedActions,
        receipt.action
      );
    }
    if (!receipt.accepted) {
      return rejectedPlanningReference(
        "planning_receipt_rejected",
        transactionId,
        expectedActions,
        receipt.action
      );
    }
    const planId = jsonObject(receipt.detail)?.plan_id;
    if (typeof planId !== "string" || !planId) {
      return rejectedPlanningReference(
        "planning_receipt_missing_plan",
        transactionId,
        expectedActions,
        receipt.action
      );
    }
    const detail = jsonObject(receipt.detail);
    const candidateCount = detail?.candidate_count;
    const selectedRank = detail?.selected_rank;
    const selectedCandidateId = detail?.selected_candidate_id;
    const candidateSelection = receipt.action === "plan_whole_body_motion_candidates"
      && typeof candidateCount === "number"
      && Number.isSafeInteger(candidateCount)
      && typeof selectedRank === "number"
      && Number.isSafeInteger(selectedRank)
      && typeof selectedCandidateId === "string"
      && selectedCandidateId
      ? {
          candidate_count: candidateCount,
          selected_rank: selectedRank,
          selected_candidate_id: selectedCandidateId
        }
      : undefined;
    return {
      accepted: true,
      planId,
      planningAction: receipt.action as HumanoidPlanningActionName,
      candidateSelection
    };
  }
}

function isPlanningAction(action: HumanoidActionName): action is HumanoidPlanningActionName {
  return action === "plan_whole_body_motion"
    || action === "plan_whole_body_motion_candidates"
    || action === "plan_humanoid_navigation";
}

function planIdFromReceipt(receipt: HumanoidActionReceipt): string | undefined {
  const planId = jsonObject(receipt.detail)?.plan_id;
  return typeof planId === "string" && planId ? planId : undefined;
}

function modelObservation(snapshot: HumanoidWorldObservation): unknown {
  const robot = snapshot.robot;
  const embodiedObjectIds = new Set([
    ...snapshot.objectTokens.flatMap((token) => (
      token.status === "visible" && token.observable
        ? [token.id]
        : []
    )),
    ...snapshot.grasp.assessments.flatMap((assessment) => (
      assessment.evidence.contact.status === "missing"
        ? []
        : [assessment.object_id]
    ))
  ]);
  return {
    frame: snapshot.frame,
    world_revision: snapshot.worldRevision,
    controller: robot.controller,
    sensor: {
      id: "head_sensor",
      position: snapshot.sensor.position,
      rotation: snapshot.sensor.rotation,
      maximum_range: snapshot.sensor.maximumRange,
      horizontal_field_of_view: snapshot.sensor.horizontalFieldOfView,
      vertical_field_of_view: snapshot.sensor.verticalFieldOfView
    },
    root: {
      position: robot.rootPosition,
      rotation: robot.rootRotation
    },
    fallen: robot.fallen,
    balance: robot.balance,
    feet: robot.feet,
    joints: robot.joints,
    key_links: Object.fromEntries([
      "pelvis",
      "head_link",
      "torso_link",
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link"
    ].flatMap((name) => robot.links[name as keyof typeof robot.links]
      ? [[name, robot.links[name as keyof typeof robot.links]]]
      : [])),
    end_effectors: Object.fromEntries(HUMANOID_END_EFFECTORS.map((endEffector) => [
      endEffector,
      {
        world_position: humanoidEndEffectorPosition(robot, endEffector, "world"),
        pelvis_relative_position: humanoidEndEffectorPosition(robot, endEffector, "pelvis")
      }
    ])),
    object_tokens: snapshot.objectTokens.map((token) => ({
      id: token.id,
      role: token.role,
      kind: token.kind,
      color: token.color,
      size: token.size,
      portable: token.portable,
      status: token.status,
      state: token.state,
      authority: token.authority,
      exact: token.exact,
      observable: token.observable,
      pose: token.pose,
      observed_frame: token.observedFrame,
      observed_world_revision: token.observedWorldRevision,
      position: token.position,
      rotation: token.rotation,
      linear_velocity: token.linearVelocity,
      angular_velocity: token.angularVelocity,
      first_seen_revision: token.firstSeenRevision,
      last_seen_revision: token.lastSeenRevision,
      last_seen_frame: token.lastSeenFrame,
      observation_count: token.observationCount,
      age_revisions: token.ageRevisions,
      relation: {
        distance_to_robot: token.relation.distanceToRobot,
        bearing_radians: token.relation.bearingRadians,
        vertical_offset: token.relation.verticalOffset,
        distance_to_left_wrist: token.relation.distanceToLeftWrist,
        distance_to_right_wrist: token.relation.distanceToRightWrist
      },
      current_contacts: token.currentContacts.map((contact) => (
        "body" in contact
          ? {
              body: contact.body,
              normal_force: contact.normalForce
            }
          : {
              hand_surface: contact.handSurface,
              normal_force: contact.normalForce
            }
      ))
    })),
    solid_tokens: snapshot.solidTokens.map((token) => ({
      id: token.id,
      source_id: token.sourceId,
      kind: token.kind,
      center: token.center,
      size: token.size,
      current_contacts: token.currentContacts
    })),
    grasp: {
      contractSha256: snapshot.grasp.contractSha256,
      assessments: snapshot.grasp.assessments.filter((assessment) => (
        embodiedObjectIds.has(assessment.object_id)
      ))
    },
    interaction: snapshot.interaction,
    contacts: robot.contacts,
    navigation: snapshot.navigation
  };
}

function conciseRobot(robot: HumanoidWorldSnapshot["robot"]): unknown {
  return {
    simulated_time: robot.simulatedTime,
    root_position: robot.rootPosition,
    fallen: robot.fallen,
    balance: robot.balance,
    feet: robot.feet,
    non_foot_environment_contacts: robot.nonFootEnvironmentContacts
  };
}

function jsonValue(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) throw new Error("Humanoid action detail is not serializable");
  return JSON.parse(serialized) as JsonValue;
}

function jsonObject(value: JsonValue): Record<string, JsonValue> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : undefined;
}

function rejectedPlanningReference(
  code: string,
  transactionId: string,
  expectedActions: readonly HumanoidPlanningActionName[],
  actualAction?: HumanoidActionName
): {
  accepted: false;
  result: {
    accepted: false;
    code: string;
    channels: HumanoidBodyChannel[];
    detail: JsonValue;
  };
} {
  return {
    accepted: false,
    result: {
      accepted: false,
      code,
      channels: [],
      detail: {
        planning_transaction_id: transactionId,
        expected_action: expectedActions.length === 1
          ? expectedActions[0]!
          : [...expectedActions],
        actual_action: actualAction ?? null,
        automatic_actuation: false
      }
    }
  };
}

export function humanoidActionFingerprint(
  action: HumanoidActionName,
  agentId: string,
  input: unknown
): string {
  return `${action}\n${agentId}\n${stableJson(input)}`;
}

function assertNormalizedReceiptIdentity(
  source: HumanoidActionReceipt,
  normalized: HumanoidActionReceipt
): void {
  if (normalized.transactionId !== source.transactionId
    || normalized.agentId !== source.agentId
    || normalized.action !== source.action
    || normalized.fingerprint !== source.fingerprint
    || stableJson(normalized.input) !== stableJson(source.input)) {
    throw new Error(
      `Humanoid receipt normalizer changed action identity: ${source.transactionId}`
    );
  }
}

function stableJson(value: unknown): string {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Humanoid action input must be finite JSON");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableJson(item)).join(",")}]`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, item]) => item !== undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    return `{${entries.map(([key, item]) => (
      `${JSON.stringify(key)}:${stableJson(item)}`
    )).join(",")}}`;
  }
  throw new Error("Humanoid action input must be JSON serializable");
}
