import {
  HUMANOID_END_EFFECTORS,
  type JsonValue
} from "../../domain/schema.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import type { HumanoidBodyChannel } from "../../world/humanoid/motion-plan.js";
import {
  type HumanoidFrameSink,
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

type HumanoidPlanningActionName = "plan_whole_body_motion"
  | "plan_whole_body_motion_candidates"
  | "plan_humanoid_navigation";

export interface HumanoidActionReceipt {
  transactionId: string;
  agentId: string;
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
  receiptSink?: (receipt: HumanoidActionReceipt) => void | Promise<void>;
  receipts?: Readonly<Record<string, HumanoidActionReceipt>>;
}

export class HumanoidActionRuntime {
  readonly #world: HumanoidWorld;
  readonly #frameSink: HumanoidFrameSink | undefined;
  readonly #receiptSink: HumanoidActionRuntimeOptions["receiptSink"];
  readonly #receipts = new Map<string, HumanoidActionReceipt>();
  readonly #transactions = new Map<string, {
    fingerprint: string;
    promise: Promise<HumanoidActionReceipt>;
  }>();
  readonly #planChannels = new Map<string, HumanoidBodyChannel[]>();
  readonly #inFlightTransactions = new Set<string>();

  constructor(world: HumanoidWorld, options: HumanoidActionRuntimeOptions = {}) {
    this.#world = world;
    this.#frameSink = options.frameSink;
    this.#receiptSink = options.receiptSink;
    for (const [transactionId, source] of Object.entries(options.receipts ?? {})) {
      const receipt = structuredClone(source);
      if (receipt.transactionId !== transactionId) {
        throw new Error(`Humanoid receipt identity mismatch: ${transactionId}`);
      }
      const fingerprint = actionFingerprint(receipt.action, receipt.agentId, receipt.input);
      if (receipt.fingerprint !== fingerprint) {
        throw new Error(`Humanoid receipt fingerprint mismatch: ${transactionId}`);
      }
      this.#receipts.set(transactionId, receipt);
      this.#transactions.set(transactionId, {
        fingerprint,
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
    agentId: string
  ): Promise<HumanoidActionReceipt> {
    const normalizedTransactionId = transactionId.trim();
    const normalizedAgentId = agentId.trim();
    if (!normalizedTransactionId) throw new Error("Humanoid action transaction id is required");
    if (!normalizedAgentId) throw new Error("Humanoid action agent id is required");
    const fingerprint = actionFingerprint(name, normalizedAgentId, rawInput);
    const existing = this.#transactions.get(normalizedTransactionId);
    if (existing) {
      if (existing.fingerprint !== fingerprint) {
        throw new Error(
          `Humanoid action transaction conflict: ${normalizedTransactionId}`
        );
      }
      return structuredClone(await existing.promise);
    }
    this.#inFlightTransactions.add(normalizedTransactionId);
    const promise = this.#invokeOnce(
      name,
      rawInput,
      normalizedTransactionId,
      normalizedAgentId,
      fingerprint
    );
    this.#transactions.set(normalizedTransactionId, { fingerprint, promise });
    try {
      return structuredClone(await promise);
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
    fingerprint: string
  ): Promise<HumanoidActionReceipt> {
    const before = this.#world.snapshot();
    const result = await this.#execute(name, rawInput);
    const after = this.#world.snapshot();
    const receipt: HumanoidActionReceipt = {
      transactionId,
      agentId,
      action: name,
      input: jsonValue(rawInput),
      fingerprint,
      accepted: result.accepted,
      code: result.code,
      worldBeforeRevision: before.worldRevision,
      worldAfterRevision: after.worldRevision,
      frameCount: after.frame - before.frame,
      channels: result.channels,
      detail: jsonValue(result.detail),
      committedAt: new Date().toISOString()
    };
    this.#receipts.set(transactionId, receipt);
    await this.#receiptSink?.(structuredClone(receipt));
    return structuredClone(receipt);
  }

  async #execute(
    name: HumanoidActionName,
    rawInput: unknown
  ): Promise<{
    accepted: boolean;
    code: string;
    channels: HumanoidBodyChannel[];
    detail: unknown;
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
      const result = await this.#world.executeWholeBodyMotion(reference.planId, this.#frameSink);
      this.#planChannels.delete(reference.planId);
      return {
        accepted: result.accepted,
        code: result.code,
        channels,
        detail: {
          planning_transaction_id: input.planning_transaction_id,
          planning_action: reference.planningAction,
          ...reference.candidateSelection,
          plan_id: reference.planId,
          frames: result.frames,
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
          target: result.target,
          chunk_target: result.chunkTarget,
          waypoints: result.waypoints,
          distance: result.distance,
          remaining_distance: result.remainingDistance,
          ...(result.reason ? { reason: result.reason } : {})
        }
      };
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
    const result = await this.#world.executeNavigation(reference.planId, this.#frameSink);
    this.#planChannels.delete(reference.planId);
    return {
      accepted: result.accepted,
      code: result.code,
      channels: ["locomotion"],
      detail: {
        planning_transaction_id: input.planning_transaction_id,
        planning_action: reference.planningAction,
        plan_id: reference.planId,
        frames: result.frames,
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
      current_contacts: token.currentContacts.map((contact) => ({
        body: contact.body,
        normal_force: contact.normalForce
      }))
    })),
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

function actionFingerprint(
  action: HumanoidActionName,
  agentId: string,
  input: unknown
): string {
  return `${action}\n${agentId}\n${stableJson(input)}`;
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
