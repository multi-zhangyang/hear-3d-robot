import type { JsonValue } from "../../domain/schema.js";
import type { HumanoidReference } from "./reference.js";

export interface HumanoidPolicyState {
  jointPositions: ArrayLike<number>;
  jointVelocities: ArrayLike<number>;
  rootQuaternion: readonly [w: number, x: number, y: number, z: number];
  rootAngularVelocity: readonly [x: number, y: number, z: number];
}

export interface HumanoidControllerDescriptor {
  protocol: "humanoid-controller-v1";
  implementation: string;
  actuation: "joint_position_pd";
  controlStepSeconds: number;
  physicsStepSeconds: number;
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

export interface HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor;
  reset(state: HumanoidPolicyState, reference: HumanoidReference): void;
  infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference
  ): Promise<HumanoidJointPositionCommand>;
  advanceHistory(state: HumanoidPolicyState, reference: HumanoidReference): void;
  captureState(): HumanoidControllerState;
  restoreState(state: HumanoidControllerState): void;
  dispose(): Promise<void>;
}
