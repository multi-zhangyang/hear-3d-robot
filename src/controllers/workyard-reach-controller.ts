import { createHash } from "node:crypto";
import * as ort from "onnxruntime-web";
import { z } from "zod";
import type {
  HumanoidControllerModuleAsset
} from "../world/humanoid/controller-module.js";
import {
  HUMANOID_JOINT_NAMES
} from "../world/humanoid/model.js";
import type { HumanoidReference } from "../world/humanoid/reference.js";
import type {
  HumanoidControllerExecutionState,
  HumanoidControllerDescriptor,
  HumanoidControllerInferenceOptions,
  HumanoidControllerInferenceTrace,
  HumanoidControllerSkillOutcome,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "../world/humanoid/whole-body-controller.js";
import type {
  MjlabG1VelocityPolicy
} from "./mjlab-g1-velocity-controller.js";
import {
  WORKYARD_REACH_OBSERVATION_PROTOCOL,
  WORKYARD_REACH_OBSERVATION_SIZE,
  encodeWorkyardReachObservation
} from "./workyard-contact-observation.js";

const REACH_POLICY_ASSET_ID = "reach_policy";
const REACH_REPORT_ASSET_ID = "reach_policy_report";
const REACH_ACTION_SIZE = 14;
const FIRST_UPPER_BODY_JOINT_INDEX = 15;
const UPPER_BODY_ACTION_SCALE = 0.5;
const CONTACT_ALIGNMENT_RADIUS_METERS = 0.15;
const CONTACT_POCKET_SLOW_RADIUS_METERS = 0.06;
const CONTACT_SUPPORTED_PLANAR_TOLERANCE_METERS = 0.045;
const CONTACT_SUPPORTED_RIGHT_PLANAR_TOLERANCE_METERS = 0.060;
const CONTACT_SUPPORTED_VERTICAL_TOLERANCE_METERS = 0.045;
const CONTACT_SUPPORTED_FORCE_NEWTONS = 2;
const CONTACT_ALIGNMENT_ACTION_SLEW = 0.015 / UPPER_BODY_ACTION_SCALE;
const CONTACT_POCKET_ACTION_SLEW = 0.0010 / UPPER_BODY_ACTION_SCALE;
const CONTACT_COMMAND_LEAD = 0.10 / UPPER_BODY_ACTION_SCALE;
const CONTACT_POCKET_COMMAND_LEAD = 0.04 / UPPER_BODY_ACTION_SCALE;

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const ArtifactSchema = z.object({
  file: z.string().trim().min(1),
  bytes: z.number().int().positive(),
  sha256: Sha256Schema
}).strict();

const ReachPolicyReportSchema = z.object({
  protocol: z.literal("hear-frozen-reach-policy-export-v1"),
  created_at: z.iso.datetime({ offset: true }),
  source: z.object({
    training_report_file: z.string().trim().min(1),
    training_report_sha256: Sha256Schema,
    checkpoint_file: z.string().trim().min(1),
    checkpoint_bytes: z.number().int().positive(),
    checkpoint_sha256: Sha256Schema,
    selected_source: z.enum(["dagger", "ppo"]),
    phase_one_accepted: z.literal(true),
    hand_checkpoint_expansion_authorized: z.literal(true),
    waist_checkpoint_expansion_authorized: z.literal(false),
    held_out_environment_count: z.number().int().min(500),
    held_out_success_rate: z.number().finite().min(0.99).max(1)
  }).strict(),
  policy: ArtifactSchema.extend({
    runtime: z.literal("torchscript_cuda"),
    input: z.literal(WORKYARD_REACH_OBSERVATION_PROTOCOL),
    input_size: z.literal(WORKYARD_REACH_OBSERVATION_SIZE),
    output: z.literal("bounded-upper-body-residual-mean"),
    output_size: z.literal(REACH_ACTION_SIZE),
    distribution: z.literal("beta_bounded_minus_one_one"),
    deterministic_statistic: z.literal("mean"),
    batch_dynamic: z.literal(true),
    normalizer_epsilon: z.literal(0.01),
    parameter_count: z.number().int().positive(),
    gradient_parameter_count: z.literal(0)
  }),
  onnx: ArtifactSchema.extend({
    runtime: z.literal("onnxruntime"),
    opset: z.literal(17),
    input: z.literal("observation"),
    input_protocol: z.literal(WORKYARD_REACH_OBSERVATION_PROTOCOL),
    input_size: z.literal(WORKYARD_REACH_OBSERVATION_SIZE),
    output: z.literal("reach_action"),
    output_protocol: z.literal("bounded-upper-body-residual-mean"),
    output_size: z.literal(REACH_ACTION_SIZE),
    batch_dynamic: z.literal(true)
  }),
  validation: z.object({
    probe_batch_size: z.number().int().positive(),
    maximum_export_error: z.number().finite().min(0).max(1e-6),
    maximum_onnx_error: z.number().finite().min(0).max(1e-5),
    maximum_absolute_action: z.number().finite().min(0).max(1.000001),
    finite: z.literal(true),
    onnx_checker_full: z.literal(true)
  }).strict()
}).strict();

const ReachControllerStateSchema = z.object({
  protocol: z.literal("workyard-reach-controller-state-v1"),
  policy_sha256: Sha256Schema,
  previous_teacher_action: z.array(z.number().finite()).length(29),
  previous_reach_action: z.array(
    z.number().finite().min(-1).max(1)
  ).length(REACH_ACTION_SIZE),
  terminal_hold: z.object({
    call_id: z.string().trim().min(1),
    active_hand: z.enum(["left", "right"]),
    object_id: z.string().trim().min(1),
    action: z.array(z.number().finite().min(-1).max(1)).length(7)
  }).strict().nullable(),
  guarded_release: z.object({
    call_id: z.string().trim().min(1),
    active_hand: z.enum(["left", "right"]),
    object_id: z.string().trim().min(1),
    action: z.array(z.number().finite().min(-1).max(1)).length(7)
  }).strict().nullable().optional(),
  body: z.object({
    protocol: z.literal("humanoid-controller-state-v1"),
    version: z.literal(1),
    implementation: z.string().trim().min(1),
    payload: z.unknown()
  }).strict()
}).strict();

interface TerminalArmHold {
  callId: string;
  activeHand: "left" | "right";
  objectId: string;
  action: Float64Array;
}

interface GuardedArmRelease extends TerminalArmHold {}

export async function createWorkyardReachController(input: {
  assets: readonly HumanoidControllerModuleAsset[];
  body: HumanoidWholeBodyController;
  bodyPolicy: MjlabG1VelocityPolicy;
  targetZoneId: string;
}): Promise<WorkyardReachController> {
  const policyAsset = requiredAsset(input.assets, REACH_POLICY_ASSET_ID);
  const reportAsset = requiredAsset(input.assets, REACH_REPORT_ASSET_ID);
  parseReachReport(policyAsset, reportAsset);
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(policyAsset.bytes, {
    executionProviders: ["wasm"]
  });
  try {
    assertSessionContract(session);
    return new WorkyardReachController({
      session,
      policySha256: policyAsset.sha256,
      body: input.body,
      bodyPolicy: input.bodyPolicy,
      targetZoneId: input.targetZoneId
    });
  } catch (error) {
    await session.release();
    throw error;
  }
}

/**
 * Preserves the locomotion actor's lower-body/waist authority while applying
 * the accepted 14D reach actor to the arms.  Inside the typed 15 cm terminal
 * shell, the existing MuJoCo task-space reference becomes a deterministic
 * active-arm executor; the learned reach actor never gains contact authority.
 */
export class WorkyardReachController implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor = {
    protocol: "humanoid-controller-v1",
    implementation: "workyard_frozen_reach_onnx",
    actuation: "joint_position_pd",
    controlStepSeconds: 0.02,
    physicsStepSeconds: 0.005,
    commandResponseHorizonSeconds: 0.2,
    minimumEffectivePlanarSpeedMetersPerSecond: 0.15,
    learnedPolicy: {
      protocol: "humanoid-learned-policy-v1",
      runtime: "onnxruntime-web/wasm",
      observationSpace: {
        protocol: WORKYARD_REACH_OBSERVATION_PROTOCOL,
        size: WORKYARD_REACH_OBSERVATION_SIZE
      },
      actionSpace: {
        protocol: "bounded-upper-body-residual-mean",
        size: REACH_ACTION_SIZE
      },
      observationFeatures: [
        "proprioception",
        "command_history",
        "root_kinematics",
        "hand_state",
        "end_effector_state",
        "contact_state",
        "object_state",
        "task_space_command",
        "grasp_command"
      ],
      capabilities: ["balance", "locomotion", "joint_reference_tracking"]
    }
  };
  readonly #session: ort.InferenceSession;
  readonly #policySha256: string;
  readonly #body: HumanoidWholeBodyController;
  readonly #bodyPolicy: MjlabG1VelocityPolicy;
  readonly #targetZoneId: string;
  readonly #upperStiffness: Float64Array;
  readonly #upperDamping: Float64Array;
  #previousTeacherAction = new Float64Array(HUMANOID_JOINT_NAMES.length);
  #previousReachAction = new Float64Array(REACH_ACTION_SIZE);
  #lastTrace: HumanoidControllerInferenceTrace | null = null;
  #terminalExecutorActive = false;
  #terminalHold: TerminalArmHold | null = null;
  #guardedRelease: GuardedArmRelease | null = null;

  constructor(input: {
    session: ort.InferenceSession;
    policySha256: string;
    body: HumanoidWholeBodyController;
    bodyPolicy: MjlabG1VelocityPolicy;
    targetZoneId: string;
  }) {
    this.#session = input.session;
    this.#policySha256 = input.policySha256;
    this.#body = input.body;
    this.#bodyPolicy = input.bodyPolicy;
    this.#targetZoneId = input.targetZoneId;
    this.#upperStiffness = Float64Array.from(
      { length: REACH_ACTION_SIZE },
      (_, offset) => offset % 7 < 4 ? 80 : 40
    );
    this.#upperDamping = Float64Array.from(
      this.#upperStiffness,
      (stiffness, offset) => {
        const index = FIRST_UPPER_BODY_JOINT_INDEX + offset;
        return this.#bodyPolicy.damping[index]!
          * Math.sqrt(stiffness / this.#bodyPolicy.stiffness[index]!);
      }
    );
  }

  executionState(): HumanoidControllerExecutionState {
    return {
      protocol: "humanoid-controller-execution-v1",
      mode: this.#terminalExecutorActive ? "hybrid_control" : "learned_policy",
      activeImplementation: this.#terminalExecutorActive
        ? `${this.descriptor.implementation}+typed_terminal_task_space_executor`
        : this.descriptor.implementation,
      transition: null
    };
  }

  inferenceTrace(): HumanoidControllerInferenceTrace | null {
    return this.#lastTrace ? structuredClone(this.#lastTrace) : null;
  }

  capabilityEvidence() {
    return this.#body.capabilityEvidence?.() ?? [];
  }

  recordSkillOutcome(outcome: HumanoidControllerSkillOutcome): void {
    this.#body.recordSkillOutcome?.(outcome);
  }

  reset(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): void {
    this.#body.reset(state, reference, options);
    this.#previousTeacherAction.fill(0);
    this.#previousReachAction.fill(0);
    this.#lastTrace = null;
    this.#terminalExecutorActive = false;
    this.#terminalHold = null;
    this.#guardedRelease = null;
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): Promise<HumanoidJointPositionCommand> {
    const task = options.taskCommand;
    const oneGrasp = task?.command.grasps.length === 1;
    const reachObservation = oneGrasp
      ? encodeWorkyardReachObservation({
          state,
          options,
          previousTeacherAction: this.#previousTeacherAction,
          previousReachAction: this.#previousReachAction
        }, {
          bodyDefaultJointPositions: this.#bodyPolicy.defaultJointPositions,
          bodyActionScale: this.#bodyPolicy.actionScale,
          targetZoneId: this.#targetZoneId
        })
      : null;
    const [bodyCommand, learnedReachAction] = await Promise.all([
      this.#body.infer(state, reference, options),
      reachObservation ? this.#inferReach(reachObservation) : null
    ]);
    const bodyTrace = this.#body.inferenceTrace?.() ?? null;
    const teacherAction = teacherActionFrom(bodyTrace);
    if (!reachObservation || !learnedReachAction || !task) {
      this.#previousTeacherAction = Float64Array.from(teacherAction);
      this.#previousReachAction.fill(0);
      this.#terminalExecutorActive = false;
      this.#terminalHold = null;
      this.#guardedRelease = null;
      this.#lastTrace = bodyTrace;
      return bodyCommand;
    }
    const appliedReachAction = this.#terminalApproachAction(
      learnedReachAction,
      state,
      reference,
      options
    );
    const command = composeCommand(
      bodyCommand,
      appliedReachAction,
      this.#bodyPolicy.defaultJointPositions,
      this.#upperStiffness,
      this.#upperDamping
    );
    this.#previousTeacherAction = Float64Array.from(teacherAction);
    this.#previousReachAction = Float64Array.from(appliedReachAction);
    this.#lastTrace = reachInferenceTrace(
      this.descriptor.implementation,
      bodyTrace,
      reachObservation,
      appliedReachAction,
      this.#terminalExecutorActive
    );
    return command;
  }

  advanceHistory(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): void {
    this.#body.advanceHistory(state, reference, options);
  }

  captureState(): HumanoidControllerState {
    const body = this.#body.captureState();
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: this.descriptor.implementation,
      payload: {
        protocol: "workyard-reach-controller-state-v1",
        policy_sha256: this.#policySha256,
        previous_teacher_action: [...this.#previousTeacherAction],
        previous_reach_action: [...this.#previousReachAction],
        terminal_hold: this.#terminalHold ? {
          call_id: this.#terminalHold.callId,
          active_hand: this.#terminalHold.activeHand,
          object_id: this.#terminalHold.objectId,
          action: [...this.#terminalHold.action]
        } : null,
        guarded_release: this.#guardedRelease ? {
          call_id: this.#guardedRelease.callId,
          active_hand: this.#guardedRelease.activeHand,
          object_id: this.#guardedRelease.objectId,
          action: [...this.#guardedRelease.action]
        } : null,
        body: {
          protocol: body.protocol,
          version: body.version,
          implementation: body.implementation,
          payload: structuredClone(body.payload)
        }
      }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    if (state.protocol !== "humanoid-controller-state-v1"
      || state.version !== 1
      || state.implementation !== this.descriptor.implementation) {
      throw new Error("Workyard reach controller state is incompatible");
    }
    const parsed = ReachControllerStateSchema.safeParse(state.payload);
    if (!parsed.success
      || parsed.data.policy_sha256 !== this.#policySha256
      || parsed.data.body.implementation !== this.#body.descriptor.implementation) {
      throw new Error("Workyard reach controller state is incompatible");
    }
    this.#body.restoreState(parsed.data.body as HumanoidControllerState);
    this.#previousTeacherAction = Float64Array.from(
      parsed.data.previous_teacher_action
    );
    this.#previousReachAction = Float64Array.from(
      parsed.data.previous_reach_action
    );
    this.#terminalHold = parsed.data.terminal_hold ? {
      callId: parsed.data.terminal_hold.call_id,
      activeHand: parsed.data.terminal_hold.active_hand,
      objectId: parsed.data.terminal_hold.object_id,
      action: Float64Array.from(parsed.data.terminal_hold.action)
    } : null;
    this.#guardedRelease = parsed.data.guarded_release ? {
      callId: parsed.data.guarded_release.call_id,
      activeHand: parsed.data.guarded_release.active_hand,
      objectId: parsed.data.guarded_release.object_id,
      action: Float64Array.from(parsed.data.guarded_release.action)
    } : null;
    this.#lastTrace = null;
    this.#terminalExecutorActive = this.#terminalHold !== null
      || this.#guardedRelease !== null;
  }

  async dispose(): Promise<void> {
    const results = await Promise.allSettled([
      this.#session.release(),
      this.#body.dispose()
    ]);
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }

  async #inferReach(observation: Float32Array): Promise<Float32Array> {
    const input = new ort.Tensor(
      "float32",
      observation,
      [1, WORKYARD_REACH_OBSERVATION_SIZE]
    );
    let result: Awaited<ReturnType<ort.InferenceSession["run"]>> | undefined;
    try {
      result = await this.#session.run({ observation: input });
      const output = result.reach_action;
      if (!output
        || output.type !== "float32"
        || output.dims.length !== 2
        || output.dims[0] !== 1
        || output.dims[1] !== REACH_ACTION_SIZE
        || !(output.data instanceof Float32Array)
        || output.data.length !== REACH_ACTION_SIZE
        || !output.data.every((value) => (
          Number.isFinite(value) && value >= -1.000001 && value <= 1.000001
        ))) {
        throw new Error("Workyard reach ONNX returned an invalid action tensor");
      }
      return Float32Array.from(output.data, clampUnit);
    } finally {
      for (const output of Object.values(result ?? {})) output.dispose();
      input.dispose();
    }
  }

  #terminalApproachAction(
    learned: Float32Array,
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions
  ): Float32Array {
    const assessment = options.handPolicyAuthority;
    const granted = assessment?.granted ? assessment.state : null;
    if (granted) {
      this.#guardedRelease = null;
      if (!sameTerminalAuthority(this.#terminalHold, granted)) {
        const offset = granted.activeHand === "left" ? 0 : 7;
        const bodyOffset = FIRST_UPPER_BODY_JOINT_INDEX + offset;
        const action = Float64Array.from({ length: 7 }, (_, jointOffset) => (
          clampUnit(
            (state.jointPositions[bodyOffset + jointOffset]!
              - this.#bodyPolicy.defaultJointPositions[bodyOffset + jointOffset]!)
              / UPPER_BODY_ACTION_SCALE
          )
        ));
        if (!action.every(Number.isFinite)) {
          throw new Error("Workyard terminal pose hold captured a non-finite arm state");
        }
        this.#terminalHold = {
          callId: granted.callId,
          activeHand: granted.activeHand,
          objectId: granted.objectId,
          action
        };
      }
      const output = Float32Array.from(learned);
      const offset = granted.activeHand === "left" ? 0 : 7;
      output.set(this.#terminalHold!.action, offset);
      this.#terminalExecutorActive = true;
      return output;
    }
    this.#terminalHold = null;
    const planarErrorMeters = assessment?.geometry.planarErrorMeters ?? null;
    const verticalErrorMeters = assessment?.geometry.verticalErrorMeters ?? null;
    const terminalErrorMeters = planarErrorMeters !== null
      && verticalErrorMeters !== null
      ? Math.hypot(planarErrorMeters, verticalErrorMeters)
      : null;
    const pendingInsideShell = assessment?.reason === "closure_geometry_pending"
      && terminalErrorMeters !== null
      && terminalErrorMeters <= CONTACT_ALIGNMENT_RADIUS_METERS;
    const active = assessment?.state?.activeHand
      ?? options.taskCommand?.command.grasps[0]?.hand;
    this.#terminalExecutorActive = Boolean(
      active && (assessment?.granted || pendingInsideShell)
    );
    if (!this.#terminalExecutorActive || !active) {
      this.#guardedRelease = null;
      return learned;
    }
    const output = Float32Array.from(learned);
    const offset = active === "left" ? 0 : 7;
    const grasp = options.taskCommand?.command.grasps[0];
    const contactSupportedPlanarTolerance = active === "right"
      ? CONTACT_SUPPORTED_RIGHT_PLANAR_TOLERANCE_METERS
      : CONTACT_SUPPORTED_PLANAR_TOLERANCE_METERS;
    const actionSlew = terminalErrorMeters !== null
      && terminalErrorMeters <= CONTACT_POCKET_SLOW_RADIUS_METERS
      ? CONTACT_POCKET_ACTION_SLEW
      : CONTACT_ALIGNMENT_ACTION_SLEW;
    const commandLead = terminalErrorMeters !== null
      && terminalErrorMeters <= CONTACT_POCKET_SLOW_RADIUS_METERS
      ? CONTACT_POCKET_COMMAND_LEAD
      : CONTACT_COMMAND_LEAD;
    const guardedEarlyContact = planarErrorMeters !== null
      && verticalErrorMeters !== null
      && (
        planarErrorMeters > contactSupportedPlanarTolerance
        || verticalErrorMeters > CONTACT_SUPPORTED_VERTICAL_TOLERANCE_METERS
      )
      && grasp?.hand === active
      && activeHandObjectForce(state, grasp.objectId, active)
        >= CONTACT_SUPPORTED_FORCE_NEWTONS;
    if (guardedEarlyContact) {
      if (!grasp || !options.taskCommand) {
        throw new Error("Typed Workyard guarded release lost its grasp authority");
      }
      if (!sameTerminalAuthority(this.#guardedRelease, {
        callId: options.taskCommand.identity.callId,
        activeHand: active,
        objectId: grasp.objectId
      })) {
        const action = Float64Array.from({ length: 7 }, (_, armOffset) => {
          const bodyIndex = FIRST_UPPER_BODY_JOINT_INDEX + offset + armOffset;
          if (reference.jointTrackingWeights[bodyIndex]! <= 0) {
            throw new Error(
              "Typed Workyard guarded release requires an active-arm reference"
            );
          }
          const requestedAction = clampUnit(
            (reference.jointPositions[bodyIndex]!
              - this.#bodyPolicy.defaultJointPositions[bodyIndex]!)
              / UPPER_BODY_ACTION_SCALE
          );
          const previousAction = this.#previousReachAction[offset + armOffset]!;
          const approachDirection = clamp(
            requestedAction - previousAction,
            -actionSlew,
            actionSlew
          );
          return clampUnit(previousAction - 4 * approachDirection);
        });
        this.#guardedRelease = {
          callId: options.taskCommand.identity.callId,
          activeHand: active,
          objectId: grasp.objectId,
          action
        };
      }
      if (!this.#guardedRelease) {
        throw new Error("Typed Workyard guarded release failed to latch");
      }
      output.set(this.#guardedRelease.action, offset);
      return output;
    }
    this.#guardedRelease = null;
    for (let armOffset = 0; armOffset < 7; armOffset += 1) {
      const bodyIndex = FIRST_UPPER_BODY_JOINT_INDEX + offset + armOffset;
      if (reference.jointTrackingWeights[bodyIndex]! <= 0) {
        throw new Error(
          "Typed Workyard terminal executor requires an active-arm task-space reference"
        );
      }
      const measuredAction = clampUnit(
        (state.jointPositions[bodyIndex]!
          - this.#bodyPolicy.defaultJointPositions[bodyIndex]!)
          / UPPER_BODY_ACTION_SCALE
      );
      const requestedAction = clampUnit(
        (reference.jointPositions[bodyIndex]!
          - this.#bodyPolicy.defaultJointPositions[bodyIndex]!)
          / UPPER_BODY_ACTION_SCALE
      );
      const leadLimitedAction = clamp(
        requestedAction,
        measuredAction - commandLead,
        measuredAction + commandLead
      );
      const previousAction = this.#previousReachAction[offset + armOffset]!;
      output[offset + armOffset] = clampUnit(clamp(
        leadLimitedAction,
        previousAction - actionSlew,
        previousAction + actionSlew
      ));
    }
    return output;
  }
}

function sameTerminalAuthority(
  hold: TerminalArmHold | null,
  authority: { callId: string; activeHand: "left" | "right"; objectId: string }
): boolean {
  return hold?.callId === authority.callId
    && hold.activeHand === authority.activeHand
    && hold.objectId === authority.objectId;
}

function composeCommand(
  body: HumanoidJointPositionCommand,
  reachAction: Float32Array,
  defaults: readonly number[],
  upperStiffness: Float64Array,
  upperDamping: Float64Array
): HumanoidJointPositionCommand {
  if (body.handSynergy) {
    throw new Error("Workyard locomotion controller cannot own hand actuation");
  }
  const positions = body.positions.slice();
  const stiffness = body.stiffness.slice();
  const damping = body.damping.slice();
  for (let offset = 0; offset < REACH_ACTION_SIZE; offset += 1) {
    const index = FIRST_UPPER_BODY_JOINT_INDEX + offset;
    positions[index] = defaults[index]!
      + reachAction[offset]! * UPPER_BODY_ACTION_SCALE;
    stiffness[index] = upperStiffness[offset]!;
    damping[index] = upperDamping[offset]!;
  }
  return { kind: "joint_position_pd", positions, stiffness, damping };
}

function teacherActionFrom(
  trace: HumanoidControllerInferenceTrace | null
): number[] {
  const component = trace?.components.find(({ role }) => role === "primary")
    ?? trace?.components.find(({ role }) => role === "direct");
  if (!component
    || component.action.values.length !== HUMANOID_JOINT_NAMES.length
    || component.action.values.some((value) => !Number.isFinite(value))) {
    throw new Error("Workyard reach requires the locomotion actor action trace");
  }
  return component.action.values;
}

function reachInferenceTrace(
  implementation: string,
  bodyTrace: HumanoidControllerInferenceTrace | null,
  observation: Float32Array,
  action: Float32Array,
  terminalExecutorActive: boolean
): HumanoidControllerInferenceTrace {
  const body = bodyTrace?.components[0];
  return {
    protocol: "humanoid-controller-inference-trace-v1",
    implementation,
    route: "upper_body_overlay",
    components: [
      ...(body ? [{ ...structuredClone(body), role: "primary" as const }] : []),
      {
        protocol: "humanoid-controller-tensor-trace-v1",
        role: "fallback",
        implementation: terminalExecutorActive
          ? "typed_terminal_task_space_executor"
          : "workyard_frozen_reach_onnx",
        observation: {
          protocol: WORKYARD_REACH_OBSERVATION_PROTOCOL,
          values: [...observation]
        },
        action: {
          protocol: "bounded-upper-body-residual-mean",
          values: [...action]
        }
      }
    ]
  };
}

function parseReachReport(
  policyAsset: HumanoidControllerModuleAsset,
  reportAsset: HumanoidControllerModuleAsset
): void {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(
      reportAsset.bytes
    ));
  } catch (error) {
    throw new Error("Workyard reach report is not valid UTF-8 JSON", {
      cause: error
    });
  }
  const parsed = ReachPolicyReportSchema.safeParse(value);
  if (!parsed.success
    || parsed.data.onnx.file !== "workyard_reach.onnx"
    || parsed.data.onnx.bytes !== policyAsset.bytes.byteLength
    || parsed.data.onnx.sha256 !== policyAsset.sha256
    || sha256(policyAsset.bytes) !== policyAsset.sha256) {
    throw new Error("Workyard reach ONNX is not backed by the accepted v15 policy");
  }
}

function requiredAsset(
  assets: readonly HumanoidControllerModuleAsset[],
  id: string
): HumanoidControllerModuleAsset {
  const asset = assets.find((candidate) => candidate.id === id);
  if (!asset) throw new Error(`Workyard reach controller asset is missing: ${id}`);
  return asset;
}

function assertSessionContract(session: ort.InferenceSession): void {
  const input = session.inputMetadata[0];
  const output = session.outputMetadata[0];
  if (session.inputNames.length !== 1
    || session.inputNames[0] !== "observation"
    || session.outputNames.length !== 1
    || session.outputNames[0] !== "reach_action"
    || !dynamicMetadataMatches(
      input,
      "observation",
      WORKYARD_REACH_OBSERVATION_SIZE
    )
    || !dynamicMetadataMatches(output, "reach_action", REACH_ACTION_SIZE)) {
    throw new Error("Workyard reach ONNX input or output contract is incompatible");
  }
}

function dynamicMetadataMatches(
  metadata: ort.InferenceSession.ValueMetadata | undefined,
  name: string,
  width: number
): boolean {
  if (!metadata?.isTensor) return false;
  const batch = metadata.shape[0];
  return metadata.name === name
    && metadata.type === "float32"
    && metadata.shape.length === 2
    && (batch === 1 || batch === -1 || typeof batch === "string")
    && metadata.shape[1] === width;
}

function sha256(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function clampUnit(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function activeHandObjectForce(
  state: HumanoidPolicyState,
  objectId: string,
  hand: "left" | "right"
): number {
  return state.environment?.contacts.reduce((total, contact) => {
    const handLink = contact.firstHandLink ?? contact.secondHandLink;
    const matchesObject = contact.firstObject === objectId
      || contact.secondObject === objectId;
    return matchesObject && handLink?.startsWith(`${hand}_`)
      ? total + contact.normalForce
      : total;
  }, 0) ?? 0;
}
