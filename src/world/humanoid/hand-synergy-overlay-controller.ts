import { z } from "zod";
import type { JsonValue } from "../../domain/schema.js";
import {
  HUMANOID_LEARNED_POLICY_CAPABILITIES,
  type HumanoidLearnedPolicyCapability
} from "../../domain/humanoid-policy.js";
import {
  G1HandCoordinationSchema,
  type G1HandCoordination
} from "./hand-coordination.js";
import {
  HumanoidHandPolicyAuthorityStateSchema,
  type HumanoidHandPolicyAuthorityAssessment,
  type HumanoidHandPolicyAuthorityState
} from "./hand-policy-authority.js";
import type { HumanoidReference } from "./reference.js";
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
} from "./whole-body-controller.js";

const HAND_ACTION_SIZE = 8;
const CONTROL_STEP_SECONDS = 0.02;

export interface HumanoidHandSynergyPolicyDescriptor {
  protocol: "humanoid-hand-synergy-policy-v1";
  implementation: string;
  runtime: string;
  observation: {
    protocol: string;
    size: number;
  };
  action: {
    protocol: "hear-active-hand-synergy-action-v1";
    size: 8;
    coordinationStep: 0.0075;
    maximumClosingJointLeadRadians: 0.25;
  };
}

export interface HumanoidHandSynergyPolicyInput {
  state: HumanoidPolicyState;
  reference: HumanoidReference;
  options: HumanoidControllerInferenceOptions;
  bodyCommand: HumanoidJointPositionCommand;
  bodyInference: HumanoidControllerInferenceTrace | null;
  authority: HumanoidHandPolicyAuthorityState;
  coordination: G1HandCoordination;
  previousAuthorizedAction: Float64Array;
}

export interface HumanoidHandSynergyPolicyOutput {
  observation: Float32Array;
  action: Float32Array;
}

export interface HumanoidHandSynergyPolicy {
  readonly descriptor: HumanoidHandSynergyPolicyDescriptor;
  reset(): void;
  infer(input: HumanoidHandSynergyPolicyInput):
  Promise<HumanoidHandSynergyPolicyOutput>;
  captureState(): JsonValue;
  restoreState(state: JsonValue): void;
  dispose(): Promise<void>;
}

const OverlayStateSchema = z.object({
  protocol: z.literal("humanoid-hand-synergy-overlay-state-v1"),
  body: z.object({
    protocol: z.literal("humanoid-controller-state-v1"),
    version: z.literal(1),
    implementation: z.string().trim().min(1),
    payload: z.unknown()
  }).strict(),
  coordination: z.array(z.number().finite().min(0).max(1)).length(8),
  previous_authorized_action: z.array(
    z.number().finite().min(-1).max(1)
  ).length(8),
  authority: HumanoidHandPolicyAuthorityStateSchema.nullable(),
  policy: z.unknown()
}).strict();

/**
 * Composes an autonomous body controller with a narrowly authorized 8D hand
 * actor.  The body controller keeps all locomotion, waist and reach authority;
 * this overlay owns only active-hand coordination after the deterministic
 * physical closure latch has been granted by HumanoidSimulation.
 */
export class HumanoidHandSynergyOverlayController
implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor;
  readonly #body: HumanoidWholeBodyController;
  readonly #hand: HumanoidHandSynergyPolicy;
  readonly #coordination = new Float64Array(HAND_ACTION_SIZE);
  #previousAuthorizedAction = new Float64Array(HAND_ACTION_SIZE);
  #authority: HumanoidHandPolicyAuthorityState | null = null;

  constructor(
    body: HumanoidWholeBodyController,
    hand: HumanoidHandSynergyPolicy
  ) {
    assertHandPolicyDescriptor(hand.descriptor);
    if (Math.abs(body.descriptor.controlStepSeconds - CONTROL_STEP_SECONDS) > 1e-12) {
      throw new Error("Hand synergy overlay requires a 50 Hz body controller");
    }
    this.#body = body;
    this.#hand = hand;
    this.descriptor = compositeDescriptor(body.descriptor, hand.descriptor);
  }

  executionState(): HumanoidControllerExecutionState {
    const body = this.#body.executionState?.() ?? {
      protocol: "humanoid-controller-execution-v1",
      mode: this.#body.descriptor.learnedPolicy
        ? "learned_policy"
        : "reference_control",
      activeImplementation: this.#body.descriptor.implementation,
      transition: null
    };
    if (!this.#authority) return body;
    return {
      ...structuredClone(body),
      mode: "hybrid_control",
      activeImplementation:
        `${body.activeImplementation}+${this.#hand.descriptor.implementation}`
    };
  }

  inferenceTrace(): HumanoidControllerInferenceTrace | null {
    return this.#body.inferenceTrace?.() ?? null;
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
    this.#resetHandState();
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): Promise<HumanoidJointPositionCommand> {
    const bodyCommand = await this.#body.infer(state, reference, options);
    if (bodyCommand.handSynergy) {
      throw new Error("Body controller attempted to bypass the hand synergy overlay");
    }
    const assessment = options.handPolicyAuthority;
    if (!assessment?.granted || !assessment.state) {
      this.#resetHandState();
      return bodyCommand;
    }
    this.#admitAuthority(assessment);
    const authority = this.#authority!;
    const output = await this.#hand.infer({
      state,
      reference,
      options,
      bodyCommand,
      bodyInference: this.#body.inferenceTrace?.() ?? null,
      authority,
      coordination: coordinationFrom(this.#coordination),
      previousAuthorizedAction: this.#previousAuthorizedAction.slice()
    });
    assertPolicyOutput(output, this.#hand.descriptor);
    const applied = Float64Array.from(output.action);
    forceInactiveHandZero(applied, authority.activeHand);
    this.#advanceCoordination(applied, authority.activeHand);
    this.#previousAuthorizedAction = applied;
    return {
      ...bodyCommand,
      handSynergy: {
        protocol: "humanoid-authorized-hand-synergy-command-v1",
        authority: structuredClone(authority),
        action: applied.slice(),
        coordination: coordinationFrom(this.#coordination),
        maximumClosingJointLeadRadians:
          this.#hand.descriptor.action.maximumClosingJointLeadRadians
      }
    };
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
        protocol: "humanoid-hand-synergy-overlay-state-v1",
        body: {
          protocol: body.protocol,
          version: body.version,
          implementation: body.implementation,
          payload: structuredClone(body.payload)
        },
        coordination: [...this.#coordination],
        previous_authorized_action: [...this.#previousAuthorizedAction],
        authority: this.#authority ? structuredClone(this.#authority) : null,
        policy: structuredClone(this.#hand.captureState())
      }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    if (state.protocol !== "humanoid-controller-state-v1"
      || state.version !== 1
      || state.implementation !== this.descriptor.implementation) {
      throw new Error("Hand synergy overlay state does not match its controller");
    }
    const parsed = OverlayStateSchema.parse(state.payload);
    const body = parsed.body as HumanoidControllerState;
    if (body.implementation !== this.#body.descriptor.implementation) {
      throw new Error("Hand synergy overlay body state changed implementation");
    }
    this.#body.restoreState(body);
    this.#coordination.set(parsed.coordination);
    this.#previousAuthorizedAction = Float64Array.from(
      parsed.previous_authorized_action
    );
    this.#authority = parsed.authority;
    this.#hand.restoreState(parsed.policy as JsonValue);
  }

  async dispose(): Promise<void> {
    const results = await Promise.allSettled([
      this.#hand.dispose(),
      this.#body.dispose()
    ]);
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }

  #admitAuthority(assessment: HumanoidHandPolicyAuthorityAssessment): void {
    const authority = HumanoidHandPolicyAuthorityStateSchema.parse(
      assessment.state
    );
    if (!sameAuthority(this.#authority, authority)) {
      this.#coordination.fill(0);
      this.#previousAuthorizedAction.fill(0);
      this.#hand.reset();
    }
    this.#authority = authority;
  }

  #advanceCoordination(
    action: Float64Array,
    activeHand: "left" | "right"
  ): void {
    const activeOffset = activeHand === "left" ? 0 : 4;
    const inactiveOffset = activeHand === "left" ? 4 : 0;
    this.#coordination.fill(0, inactiveOffset, inactiveOffset + 4);
    for (let index = activeOffset; index < activeOffset + 4; index += 1) {
      this.#coordination[index] = clamp(
        this.#coordination[index]! + action[index]!
          * this.#hand.descriptor.action.coordinationStep,
        0,
        1
      );
    }
  }

  #resetHandState(): void {
    if (this.#authority || this.#coordination.some((value) => value !== 0)
      || this.#previousAuthorizedAction.some((value) => value !== 0)) {
      this.#hand.reset();
    }
    this.#authority = null;
    this.#coordination.fill(0);
    this.#previousAuthorizedAction.fill(0);
  }
}

function compositeDescriptor(
  body: HumanoidControllerDescriptor,
  hand: HumanoidHandSynergyPolicyDescriptor
): HumanoidControllerDescriptor {
  const bodyPolicy = body.learnedPolicy;
  const capabilities = orderedCapabilities([
    ...(bodyPolicy?.capabilities ?? []),
    "contact_rich_manipulation"
  ]);
  const observationFeatures = HUMANOID_POLICY_OBSERVATION_FEATURES.filter(
    (feature) => new Set([
      ...(bodyPolicy?.observationFeatures ?? []),
      "hand_state",
      "end_effector_state",
      "contact_state",
      "object_state",
      "task_space_command",
      "grasp_command",
      "command_history"
    ]).has(feature)
  );
  return {
    ...structuredClone(body),
    implementation: `${body.implementation}+${hand.implementation}`,
    learnedPolicy: {
      protocol: "humanoid-learned-policy-v1",
      runtime: bodyPolicy
        ? `${bodyPolicy.runtime}+${hand.runtime}`
        : hand.runtime,
      observationSpace: {
        protocol: "humanoid-body-hand-composition-observation-v1",
        size: (bodyPolicy?.observationSpace.size ?? 0) + hand.observation.size
      },
      actionSpace: {
        protocol: "humanoid-body-hand-composition-action-v1",
        size: (bodyPolicy?.actionSpace.size ?? 0) + hand.action.size
      },
      observationFeatures,
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

function assertHandPolicyDescriptor(
  descriptor: HumanoidHandSynergyPolicyDescriptor
): void {
  if (descriptor.protocol !== "humanoid-hand-synergy-policy-v1"
    || descriptor.implementation.trim().length === 0
    || descriptor.runtime.trim().length === 0
    || descriptor.observation.protocol.trim().length === 0
    || !Number.isSafeInteger(descriptor.observation.size)
    || descriptor.observation.size <= 0
    || descriptor.action.protocol !== "hear-active-hand-synergy-action-v1"
    || descriptor.action.size !== HAND_ACTION_SIZE
    || descriptor.action.coordinationStep !== 0.0075
    || descriptor.action.maximumClosingJointLeadRadians !== 0.25) {
    throw new Error("Invalid humanoid hand synergy policy descriptor");
  }
}

function assertPolicyOutput(
  output: HumanoidHandSynergyPolicyOutput,
  descriptor: HumanoidHandSynergyPolicyDescriptor
): void {
  if (!(output.observation instanceof Float32Array)
    || output.observation.length !== descriptor.observation.size
    || !output.observation.every(Number.isFinite)
    || !(output.action instanceof Float32Array)
    || output.action.length !== descriptor.action.size
    || !output.action.every((value) => (
      Number.isFinite(value) && value >= -1 && value <= 1
    ))) {
    throw new Error("Humanoid hand synergy policy returned an invalid tensor");
  }
}

function forceInactiveHandZero(
  action: Float64Array,
  activeHand: "left" | "right"
): void {
  const offset = activeHand === "left" ? 4 : 0;
  action.fill(0, offset, offset + 4);
}

function coordinationFrom(values: Float64Array): G1HandCoordination {
  return G1HandCoordinationSchema.parse({
    left: {
      thumb_opposition: values[0],
      thumb_curl: values[1],
      index_curl: values[2],
      middle_curl: values[3]
    },
    right: {
      thumb_opposition: values[4],
      thumb_curl: values[5],
      index_curl: values[6],
      middle_curl: values[7]
    }
  });
}

function sameAuthority(
  left: HumanoidHandPolicyAuthorityState | null,
  right: HumanoidHandPolicyAuthorityState
): boolean {
  return left?.callId === right.callId
    && left.activeHand === right.activeHand
    && left.objectId === right.objectId;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
