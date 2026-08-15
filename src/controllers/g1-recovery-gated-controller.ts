import { z } from "zod";
import type { JsonValue } from "../domain/schema.js";
import {
  HUMANOID_POLICY_OBSERVATION_FEATURES,
  type HumanoidControllerCapabilityEvidenceSummary,
  type HumanoidControllerDescriptor,
  type HumanoidControllerExecutionState,
  type HumanoidControllerInferenceOptions,
  type HumanoidControllerInferenceTrace,
  type HumanoidControllerSkillOutcome,
  type HumanoidControllerState,
  type HumanoidJointPositionCommand,
  type HumanoidPolicyState,
  type HumanoidWholeBodyController
} from "../world/humanoid/whole-body-controller.js";
import type { HumanoidReference } from "../world/humanoid/reference.js";
import {
  HUMANOID_RECOVERY_CONTROL_STEP_SECONDS,
  HUMANOID_RECOVERY_HANDOFF_STEPS,
  HUMANOID_LEARNED_POLICY_CAPABILITIES,
  HUMANOID_RECOVERY_STABLE_STEPS,
  type HumanoidLearnedPolicyCapability
} from "../domain/humanoid-policy.js";
import { g1ProjectedUpright } from "./g1-getup-observation.js";

const STABLE_CONTROL_STEPS = HUMANOID_RECOVERY_STABLE_STEPS;
const HANDOFF_CONTROL_STEPS = HUMANOID_RECOVERY_HANDOFF_STEPS;
const HANDOFF_SECONDS = HANDOFF_CONTROL_STEPS
  * HUMANOID_RECOVERY_CONTROL_STEP_SECONDS;

const EmbeddedControllerStateSchema = z.object({
  protocol: z.literal("humanoid-controller-state-v1"),
  version: z.literal(1),
  implementation: z.string().trim().min(1),
  payload: z.unknown()
}).strict();

const RecoveryGateStateSchema = z.object({
  protocol: z.literal("hear-g1-recovery-gate-state-v1"),
  mode: z.enum(["normal", "recovery", "handoff"]),
  stable_steps: z.number().int().min(0).max(STABLE_CONTROL_STEPS),
  handoff_step: z.number().int().min(0).max(HANDOFF_CONTROL_STEPS),
  handoff_ready: z.boolean(),
  recovery_call_id: z.string().trim().min(1).nullable(),
  body: EmbeddedControllerStateSchema,
  recovery: EmbeddedControllerStateSchema
}).strict();

type GateMode = "normal" | "recovery" | "handoff";
type InvokedControllers = "body" | "recovery" | "both";

/**
 * Low-level expert gate modelled after specialist recovery controllers: the
 * Harness requests the semantic `stabilize` Skill once, then this gate keeps
 * recovery authority below the LLM boundary until measured proprioception says
 * the robot has stood stably.  It never asks a model to choose per-frame poses.
 */
export class G1RecoveryGatedController implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor;
  readonly #body: HumanoidWholeBodyController;
  readonly #recovery: HumanoidWholeBodyController;
  #mode: GateMode = "normal";
  #stableSteps = 0;
  #handoffStep = 0;
  #handoffReady = false;
  #recoveryCallId: string | null = null;
  #lastInvoked: InvokedControllers = "body";
  #lastTrace: HumanoidControllerInferenceTrace | null = null;

  constructor(
    body: HumanoidWholeBodyController,
    recovery: HumanoidWholeBodyController
  ) {
    if (Math.abs(body.descriptor.controlStepSeconds
      - recovery.descriptor.controlStepSeconds) > 1e-12
      || Math.abs(body.descriptor.physicsStepSeconds
        - recovery.descriptor.physicsStepSeconds) > 1e-12) {
      throw new Error("G1 recovery and body experts require identical timing");
    }
    if (Math.abs(body.descriptor.controlStepSeconds
      - HUMANOID_RECOVERY_CONTROL_STEP_SECONDS) > 1e-12) {
      throw new Error(
        `G1 recovery requires a ${HUMANOID_RECOVERY_CONTROL_STEP_SECONDS}s control step`
      );
    }
    this.#body = body;
    this.#recovery = recovery;
    this.descriptor = compositeDescriptor(body.descriptor, recovery.descriptor);
  }

  executionState(): HumanoidControllerExecutionState {
    if (this.#mode === "recovery") {
      return {
        protocol: "humanoid-controller-execution-v1",
        mode: "learned_policy",
        activeImplementation: this.#recovery.descriptor.implementation,
        transition: null
      };
    }
    const body = bodyExecutionState(this.#body);
    if (this.#mode !== "handoff") return body;
    return {
      ...body,
      activeImplementation: body.activeImplementation,
      transition: {
        fromImplementation: this.#recovery.descriptor.implementation,
        toImplementation: body.activeImplementation,
        progress: Math.min(1, this.#handoffStep / HANDOFF_CONTROL_STEPS),
        durationSeconds: HANDOFF_SECONDS
      }
    };
  }

  inferenceTrace(): HumanoidControllerInferenceTrace | null {
    return this.#lastTrace ? structuredClone(this.#lastTrace) : null;
  }

  capabilityEvidence(): readonly HumanoidControllerCapabilityEvidenceSummary[] {
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
    this.#recovery.reset(state, reference, options);
    this.#mode = "normal";
    this.#stableSteps = 0;
    this.#handoffStep = 0;
    this.#handoffReady = false;
    this.#recoveryCallId = null;
    this.#lastInvoked = "body";
    this.#lastTrace = null;
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): Promise<HumanoidJointPositionCommand> {
    const authorizedRecovery = isAuthorizedRecoveryCommand(options);
    if (this.#mode !== "normal" && !authorizedRecovery) {
      this.#exitRecovery(state, reference, options);
    }
    if (this.#mode !== "normal" && authorizedRecovery
      && options.taskCommand?.identity.callId !== this.#recoveryCallId) {
      this.#recoveryCallId = options.taskCommand!.identity.callId;
    }
    if (this.#mode === "normal"
      && authorizedRecovery
      && isFallen(state)) {
      this.#enterRecovery(state, reference, options);
    }
    if (this.#mode === "recovery" && this.#handoffReady) {
      this.#enterHandoff(state, reference, options);
    }
    if (this.#mode === "recovery") {
      const command = await this.#recovery.infer(state, reference, options);
      this.#lastInvoked = "recovery";
      this.#lastTrace = this.#recovery.inferenceTrace?.() ?? null;
      if (isStableStanding(state)) {
        this.#stableSteps = Math.min(
          STABLE_CONTROL_STEPS,
          this.#stableSteps + 1
        );
      } else {
        this.#stableSteps = 0;
      }
      this.#handoffReady = this.#stableSteps >= STABLE_CONTROL_STEPS;
      return command;
    }
    if (this.#mode === "handoff") {
      if (isFallen(state)) {
        this.#mode = "recovery";
        this.#stableSteps = 0;
        this.#handoffStep = 0;
        this.#handoffReady = false;
        return this.infer(state, reference, options);
      }
      const [recoveryCommand, bodyCommand] = await Promise.all([
        this.#recovery.infer(state, reference, options),
        this.#body.infer(state, reference, options)
      ]);
      this.#handoffStep = Math.min(
        HANDOFF_CONTROL_STEPS,
        this.#handoffStep + 1
      );
      const progress = this.#handoffStep / HANDOFF_CONTROL_STEPS;
      this.#lastInvoked = "both";
      this.#lastTrace = handoffTrace(
        this.descriptor.implementation,
        this.#recovery.inferenceTrace?.() ?? null,
        this.#body.inferenceTrace?.() ?? null
      );
      const command = blendCommands(recoveryCommand, bodyCommand, progress);
      if (this.#handoffStep >= HANDOFF_CONTROL_STEPS) {
        this.#mode = "normal";
        this.#stableSteps = 0;
        this.#handoffStep = 0;
        this.#handoffReady = false;
        this.#recoveryCallId = null;
      }
      return command;
    }
    const command = await this.#body.infer(state, reference, options);
    this.#lastInvoked = "body";
    this.#lastTrace = this.#body.inferenceTrace?.() ?? null;
    return command;
  }

  advanceHistory(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): void {
    if (this.#lastInvoked === "body" || this.#lastInvoked === "both") {
      this.#body.advanceHistory(state, reference, options);
    }
    if (this.#lastInvoked === "recovery" || this.#lastInvoked === "both") {
      this.#recovery.advanceHistory(state, reference, options);
    }
  }

  captureState(): HumanoidControllerState {
    const body = this.#body.captureState();
    const recovery = this.#recovery.captureState();
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: this.descriptor.implementation,
      payload: {
        protocol: "hear-g1-recovery-gate-state-v1",
        mode: this.#mode,
        stable_steps: this.#stableSteps,
        handoff_step: this.#handoffStep,
        handoff_ready: this.#handoffReady,
        recovery_call_id: this.#recoveryCallId,
        body: embeddedState(body),
        recovery: embeddedState(recovery)
      }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    if (state.protocol !== "humanoid-controller-state-v1"
      || state.version !== 1
      || state.implementation !== this.descriptor.implementation) {
      throw new Error("G1 recovery gate state is incompatible");
    }
    const parsed = RecoveryGateStateSchema.safeParse(state.payload);
    if (!parsed.success
      || parsed.data.body.implementation !== this.#body.descriptor.implementation
      || parsed.data.recovery.implementation
        !== this.#recovery.descriptor.implementation) {
      throw new Error("G1 recovery gate state is incompatible");
    }
    this.#body.restoreState(parsed.data.body as HumanoidControllerState);
    this.#recovery.restoreState(parsed.data.recovery as HumanoidControllerState);
    this.#mode = parsed.data.mode;
    this.#stableSteps = parsed.data.stable_steps;
    this.#handoffStep = parsed.data.handoff_step;
    this.#handoffReady = parsed.data.handoff_ready;
    this.#recoveryCallId = parsed.data.recovery_call_id;
    this.#lastInvoked = this.#mode === "normal"
      ? "body"
      : this.#mode === "handoff"
        ? "both"
        : "recovery";
    this.#lastTrace = null;
  }

  async dispose(): Promise<void> {
    const results = await Promise.allSettled([
      this.#recovery.dispose(),
      this.#body.dispose()
    ]);
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }

  #enterRecovery(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions
  ): void {
    this.#recovery.reset(state, reference, options);
    this.#mode = "recovery";
    this.#stableSteps = 0;
    this.#handoffStep = 0;
    this.#handoffReady = false;
    this.#recoveryCallId = options.taskCommand?.identity.callId ?? null;
  }

  #enterHandoff(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions
  ): void {
    this.#body.reset(state, reference, options);
    this.#mode = "handoff";
    this.#handoffStep = 0;
    this.#handoffReady = false;
  }

  #exitRecovery(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions
  ): void {
    this.#body.reset(state, reference, options);
    this.#mode = "normal";
    this.#stableSteps = 0;
    this.#handoffStep = 0;
    this.#handoffReady = false;
    this.#recoveryCallId = null;
  }
}

function isAuthorizedRecoveryCommand(
  options: HumanoidControllerInferenceOptions
): boolean {
  const command = options.taskCommand;
  return command?.identity.runtimeKind === "semantic_skill"
    && command.identity.skillId === "stabilize"
    && command.identity.phase === "recover_support"
    && command.contract?.protocol === "humanoid-embodied-recovery-contract-v1"
    && command.contract.safetyInterrupt.status === "acknowledged"
    && command.requestedCapabilities.includes("whole_body_recovery")
    && "recoveryTerrainContact" in command.safety
    && command.safety.recoveryTerrainContact;
}

function isFallen(state: HumanoidPolicyState): boolean {
  const rootHeight = state.environment?.rootPosition?.y;
  return rootHeight !== undefined
    && (rootHeight < 0.45 || g1ProjectedUpright(state.rootQuaternion) < 0.55);
}

function isStableStanding(state: HumanoidPolicyState): boolean {
  const environment = state.environment;
  if (!environment?.rootPosition || !environment.feet) return false;
  const rootSpeed = Math.hypot(...environment.rootLinearVelocity);
  const angularSpeed = Math.hypot(...environment.rootAngularVelocity);
  const maximumJointSpeed = Math.max(
    ...Array.from(state.jointVelocities, Math.abs)
  );
  return environment.rootPosition.y >= 0.70
    && g1ProjectedUpright(state.rootQuaternion) >= 0.90
    && environment.feet.left.touching
    && environment.feet.right.touching
    && rootSpeed <= 0.35
    && angularSpeed <= 0.50
    && maximumJointSpeed <= 1.50;
}

function blendCommands(
  recovery: HumanoidJointPositionCommand,
  body: HumanoidJointPositionCommand,
  progress: number
): HumanoidJointPositionCommand {
  if (recovery.kind !== "joint_position_pd" || body.kind !== "joint_position_pd"
    || recovery.positions.length !== body.positions.length) {
    throw new Error("G1 recovery handoff received incompatible commands");
  }
  const mix = (left: ArrayLike<number>, right: ArrayLike<number>) => (
    Float64Array.from(left, (value, index) => (
      value + (right[index]! - value) * progress
    ))
  );
  return {
    kind: "joint_position_pd",
    positions: mix(recovery.positions, body.positions),
    stiffness: mix(recovery.stiffness, body.stiffness),
    damping: mix(recovery.damping, body.damping),
    ...(progress >= 1 && body.handSynergy
      ? { handSynergy: structuredClone(body.handSynergy) }
      : {})
  };
}

function handoffTrace(
  implementation: string,
  recovery: HumanoidControllerInferenceTrace | null,
  body: HumanoidControllerInferenceTrace | null
): HumanoidControllerInferenceTrace | null {
  const components = [
    ...(recovery?.components ?? []),
    ...(body?.components ?? [])
  ].map((component) => structuredClone(component));
  return components.length === 0 ? null : {
    protocol: "humanoid-controller-inference-trace-v1",
    implementation,
    route: "primary",
    components
  };
}

function bodyExecutionState(
  body: HumanoidWholeBodyController
): HumanoidControllerExecutionState {
  return body.executionState?.() ?? {
    protocol: "humanoid-controller-execution-v1",
    mode: body.descriptor.learnedPolicy ? "learned_policy" : "reference_control",
    activeImplementation: body.descriptor.implementation,
    transition: null
  };
}

function embeddedState(state: HumanoidControllerState): {
  protocol: "humanoid-controller-state-v1";
  version: 1;
  implementation: string;
  payload: JsonValue;
} {
  return {
    protocol: state.protocol,
    version: state.version,
    implementation: state.implementation,
    payload: structuredClone(state.payload)
  };
}

function compositeDescriptor(
  body: HumanoidControllerDescriptor,
  recovery: HumanoidControllerDescriptor
): HumanoidControllerDescriptor {
  const bodyPolicy = body.learnedPolicy;
  const recoveryPolicy = recovery.learnedPolicy;
  const capabilities = orderedCapabilities([
    ...(bodyPolicy?.capabilities ?? []),
    ...(recoveryPolicy?.capabilities ?? [])
  ]);
  const featureSet = new Set([
    ...(bodyPolicy?.observationFeatures ?? []),
    ...(recoveryPolicy?.observationFeatures ?? [])
  ]);
  return {
    ...structuredClone(body),
    implementation: `${body.implementation}+g1_recovery_expert_gate`,
    learnedPolicy: {
      protocol: "humanoid-learned-policy-v1",
      runtime: [bodyPolicy?.runtime, recoveryPolicy?.runtime]
        .filter(Boolean).join("+"),
      observationSpace: {
        protocol: "hear-g1-expert-gated-observation-v1",
        size: Math.max(
          bodyPolicy?.observationSpace.size ?? 0,
          recoveryPolicy?.observationSpace.size ?? 0
        )
      },
      actionSpace: {
        protocol: "hear-g1-expert-gated-action-v1",
        size: Math.max(
          bodyPolicy?.actionSpace.size ?? 0,
          recoveryPolicy?.actionSpace.size ?? 0
        )
      },
      observationFeatures: HUMANOID_POLICY_OBSERVATION_FEATURES.filter(
        (feature) => featureSet.has(feature)
      ),
      capabilities
    }
  };
}

function orderedCapabilities(
  values: readonly HumanoidLearnedPolicyCapability[]
): HumanoidLearnedPolicyCapability[] {
  const set = new Set(values);
  return HUMANOID_LEARNED_POLICY_CAPABILITIES.filter((value) => set.has(value));
}
