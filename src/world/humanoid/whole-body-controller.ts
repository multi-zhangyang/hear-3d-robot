import type { JsonValue } from "../../domain/schema.js";
import type { HumanoidReference } from "./reference.js";

export interface HumanoidPolicyState {
  jointPositions: ArrayLike<number>;
  jointVelocities: ArrayLike<number>;
  rootQuaternion: readonly [w: number, x: number, y: number, z: number];
  rootAngularVelocity: readonly [x: number, y: number, z: number];
}

export const HUMANOID_LEARNED_POLICY_CAPABILITIES = [
  "balance",
  "locomotion",
  "joint_reference_tracking",
  "contact_rich_manipulation",
  "bimanual_manipulation"
] as const;

type HumanoidLearnedPolicyCapability =
  typeof HUMANOID_LEARNED_POLICY_CAPABILITIES[number];

interface HumanoidLearnedPolicyDescriptor {
  protocol: "humanoid-learned-policy-v1";
  runtime: string;
  observationSpace: {
    protocol: string;
    size: number;
  };
  actionSpace: {
    protocol: string;
    size: number;
  };
  capabilities: HumanoidLearnedPolicyCapability[];
}

export interface HumanoidControllerDescriptor {
  protocol: "humanoid-controller-v1";
  implementation: string;
  actuation: "joint_position_pd";
  controlStepSeconds: number;
  physicsStepSeconds: number;
  commandResponseHorizonSeconds?: number | undefined;
  minimumEffectivePlanarSpeedMetersPerSecond?: number | undefined;
  learnedPolicy?: HumanoidLearnedPolicyDescriptor | undefined;
}

export interface HumanoidJointPositionCommand {
  kind: "joint_position_pd";
  positions: Float64Array;
  stiffness: Float64Array;
  damping: Float64Array;
}

export interface HumanoidControllerState {
  protocol: "humanoid-controller-state-v1";
  version: 1;
  implementation: string;
  payload: JsonValue;
}

export interface HumanoidControllerInferenceOptions {
  trackedJointPolicyCommand?: "measured" | "neutral";
}

export interface HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor;
  reset(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options?: HumanoidControllerInferenceOptions
  ): void;
  infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options?: HumanoidControllerInferenceOptions
  ): Promise<HumanoidJointPositionCommand>;
  advanceHistory(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options?: HumanoidControllerInferenceOptions
  ): void;
  captureState(): HumanoidControllerState;
  restoreState(state: HumanoidControllerState): void;
  dispose(): Promise<void>;
}

export type HumanoidWholeBodyControllerFactory =
  () => Promise<HumanoidWholeBodyController>;
