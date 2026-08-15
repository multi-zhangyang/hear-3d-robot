import {
  HUMANOID_JOINT_NAMES
} from "../world/humanoid/model.js";
import type {
  HumanoidPolicyState
} from "../world/humanoid/whole-body-controller.js";

export const G1_GETUP_OBSERVATION_PROTOCOL =
  "hear-g1-getup-proprioception-v1";
export const G1_GETUP_OBSERVATION_SIZE = 99;

export interface G1GetupObservationConfiguration {
  readonly defaultJointPositions: readonly number[];
}

/**
 * Deployment observation for the recovery expert.  This is intentionally
 * proprioceptive: no reference motion identity, phase clock, scripted pose, or
 * Harness state enters the 50 Hz policy loop.
 */
export function encodeG1GetupObservation(
  state: HumanoidPolicyState,
  previousAction: ArrayLike<number>,
  configuration: G1GetupObservationConfiguration
): Float32Array {
  const environment = state.environment;
  if (state.jointPositions.length !== HUMANOID_JOINT_NAMES.length
    || state.jointVelocities.length !== HUMANOID_JOINT_NAMES.length
    || previousAction.length !== HUMANOID_JOINT_NAMES.length
    || configuration.defaultJointPositions.length
      !== HUMANOID_JOINT_NAMES.length
    || environment?.protocol !== "humanoid-policy-environment-v1"
    || environment.authority !== "mujoco_state"
    || environment.rootVelocityFrame !== "pelvis_imu"
    || environment.rootPosition === undefined
    || environment.feet === undefined) {
    throw new Error("G1 get-up policy state is incomplete");
  }
  const projectedGravity = inverseRotate(
    state.rootQuaternion,
    [0, 0, -1]
  );
  const observation = Float32Array.from([
    ...projectedGravity,
    ...environment.rootAngularVelocity,
    ...environment.rootLinearVelocity,
    environment.rootPosition.y,
    ...Array.from(state.jointPositions, (value, index) => (
      value - configuration.defaultJointPositions[index]!
    )),
    ...Array.from(state.jointVelocities),
    ...Array.from(previousAction),
    environment.feet.left.touching ? 1 : 0,
    environment.feet.right.touching ? 1 : 0
  ]);
  if (observation.length !== G1_GETUP_OBSERVATION_SIZE
    || !observation.every(Number.isFinite)) {
    throw new Error("G1 get-up observation is invalid");
  }
  return observation;
}

export function g1ProjectedUpright(
  quaternion: HumanoidPolicyState["rootQuaternion"]
): number {
  return -inverseRotate(quaternion, [0, 0, -1])[2];
}

function inverseRotate(
  [w, x, y, z]: readonly [number, number, number, number],
  [vx, vy, vz]: readonly [number, number, number]
): [number, number, number] {
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx - w * tx + (y * tz - z * ty),
    vy - w * ty + (z * tx - x * tz),
    vz - w * tz + (x * ty - y * tx)
  ];
}
