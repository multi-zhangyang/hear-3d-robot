import { z } from "zod";
import { yawFromQuaternion } from "../geometry.js";
import { HUMANOID_JOINT_NAMES, YAHMP_POLICY } from "./model.js";
import {
  releaseReferenceTracking,
  stationaryHumanoidReference,
  targetReference,
  type HumanoidReference
} from "./reference.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

const POSITION_GAIN = 1.2;
const LATERAL_POSITION_GAIN = 0.8;
const VELOCITY_DAMPING = 0.65;
const YAW_GAIN = 1.8;
const MAXIMUM_FORWARD_SPEED = 0.3;
const MAXIMUM_LATERAL_SPEED = 0.2;
const MAXIMUM_YAW_SPEED = 1;
const MAXIMUM_TRACKED_POSTURE_DRIFT_METERS = 0.2;
const STANCE_FULL_AUTHORITY_DRIFT_METERS = 0.015;
const STANCE_RELEASE_DRIFT_METERS = 0.08;
const POLICY_MINIMUM_COMMAND_SPEED = 0.15;
const POLICY_LOW_SPEED_SCALE = 0.075;
const POLICY_HIGH_SPEED_OFFSET = 0.075;
const POLICY_LOW_SPEED_COMMAND = POLICY_MINIMUM_COMMAND_SPEED - 1e-3;
const POLICY_LOW_SPEED_MAXIMUM =
  POLICY_LOW_SPEED_COMMAND * POLICY_LOW_SPEED_SCALE;
const POLICY_HIGH_SPEED_MINIMUM =
  POLICY_MINIMUM_COMMAND_SPEED - POLICY_HIGH_SPEED_OFFSET;
const POLICY_RESPONSE_SWITCH_SPEED =
  (POLICY_LOW_SPEED_MAXIMUM + POLICY_HIGH_SPEED_MINIMUM) / 2;
const MINIMUM_CORRECTION_SPEED = 0.02;
const MINIMUM_YAW_CORRECTION_RADIANS = 0.04;
const MAXIMUM_COMMAND_ACCELERATION_METERS_PER_SECOND_SQUARED = 1;

export const HumanoidStationKeepingAnchorSchema = z.object({
  x: z.number().finite(),
  z: z.number().finite(),
  yaw: z.number().finite(),
  sourceFrame: z.number().int().nonnegative(),
  sourceWorldRevision: z.number().int().nonnegative()
}).strict();

export type HumanoidStationKeepingAnchor = z.infer<
  typeof HumanoidStationKeepingAnchorSchema
>;

export function captureHumanoidStationKeepingAnchor(
  snapshot: HumanoidSimulationSnapshot,
  sourceFrame: number,
  sourceWorldRevision: number
): HumanoidStationKeepingAnchor {
  return HumanoidStationKeepingAnchorSchema.parse({
    x: snapshot.rootPosition.x,
    z: snapshot.rootPosition.z,
    yaw: yawFromQuaternion(snapshot.rootRotation),
    sourceFrame,
    sourceWorldRevision
  });
}

export function stationKeepingHumanoidReference(
  reference: HumanoidReference,
  snapshot: HumanoidSimulationSnapshot,
  anchor: HumanoidStationKeepingAnchor,
  options: {
    preserveTrackedLowerBody?: boolean;
    previousPlanarCommand?: readonly [number, number];
    controlStepSeconds?: number;
    minimumEffectiveYawSpeedRadiansPerSecond?: number;
  } = {}
): HumanoidReference {
  const validated = HumanoidStationKeepingAnchorSchema.parse(anchor);
  const yaw = yawFromQuaternion(snapshot.rootRotation);
  const deltaX = validated.x - snapshot.rootPosition.x;
  const deltaZ = validated.z - snapshot.rootPosition.z;
  const localForward = deltaX * Math.sin(yaw) + deltaZ * Math.cos(yaw);
  const localLateral = deltaX * Math.cos(yaw) - deltaZ * Math.sin(yaw);
  const pelvisVelocity = snapshot.links?.pelvis?.linearVelocity
    ?? { x: 0, y: 0, z: 0 };
  const localForwardVelocity = pelvisVelocity.x * Math.sin(yaw)
    + pelvisVelocity.z * Math.cos(yaw);
  const localLateralVelocity = pelvisVelocity.x * Math.cos(yaw)
    - pelvisVelocity.z * Math.sin(yaw);
  const yawError = normalizeAngle(validated.yaw - yaw);
  const minimumYawCommand = options.minimumEffectiveYawSpeedRadiansPerSecond ?? 0;
  if (!Number.isFinite(minimumYawCommand) || minimumYawCommand < 0) {
    throw new Error("Station-keeping minimum yaw command must be finite and nonnegative");
  }
  const planarDrift = Math.hypot(deltaX, deltaZ);
  const postureReference = planarDrift
      > MAXIMUM_TRACKED_POSTURE_DRIFT_METERS
    && reference.jointTrackingWeights.some((weight) => weight > 0)
    ? releaseReferenceTracking(reference)
    : reference;
  let compensated = inversePolicyPlanarVelocity(
    localForward * POSITION_GAIN - localForwardVelocity * VELOCITY_DAMPING,
    localLateral * LATERAL_POSITION_GAIN - localLateralVelocity * VELOCITY_DAMPING
  );
  if (options.previousPlanarCommand) {
    const step = MAXIMUM_COMMAND_ACCELERATION_METERS_PER_SECOND_SQUARED
      * (options.controlStepSeconds ?? 0);
    if (!Number.isFinite(step) || step <= 0) {
      throw new Error("Station-keeping command slew requires a positive control step");
    }
    compensated = {
      forward: approach(
        options.previousPlanarCommand[0],
        compensated.forward,
        step
      ),
      lateral: approach(
        options.previousPlanarCommand[1],
        compensated.lateral,
        step
      )
    };
  }
  const stanceReference = options.preserveTrackedLowerBody
    ? lowerBodyStanceReference(postureReference, stanceAuthority(planarDrift))
    : postureReference;
  const stationary = stationaryHumanoidReference(stanceReference);
  if (options.preserveTrackedLowerBody) {
    HUMANOID_JOINT_NAMES.forEach((name, index) => {
      if (name.startsWith("waist_")
        || name.startsWith("left_hip_")
        || name.startsWith("right_hip_")
        || name.includes("knee")
        || name.includes("ankle")) {
        stationary.jointTrackingWeights[index] =
          stanceReference.jointTrackingWeights[index]!;
      }
    });
  }
  const proportionalYawCommand = Math.abs(yawError)
      <= MINIMUM_YAW_CORRECTION_RADIANS
    ? 0
    : clamp(yawError * YAW_GAIN, -MAXIMUM_YAW_SPEED, MAXIMUM_YAW_SPEED);
  const effectiveYawCommand = proportionalYawCommand !== 0
      && Math.abs(proportionalYawCommand) < minimumYawCommand
    ? Math.sign(proportionalYawCommand) * minimumYawCommand
    : proportionalYawCommand;
  return targetReference(stationary, {
    rootVelocity: [
      compensated.forward,
      compensated.lateral
    ],
    rootYawVelocity: effectiveYawCommand
  });
}

function lowerBodyStanceReference(
  reference: HumanoidReference,
  authority: number
): HumanoidReference {
  const adjusted: HumanoidReference = {
    ...reference,
    jointPositions: reference.jointPositions.slice(),
    jointVelocities: reference.jointVelocities.slice(),
    jointTrackingWeights: reference.jointTrackingWeights.slice(),
    rootVelocity: [...reference.rootVelocity]
  };
  HUMANOID_JOINT_NAMES.forEach((name, index) => {
    if (!isLowerBodyJoint(name)) return;
    adjusted.jointPositions[index] = mix(
      YAHMP_POLICY.defaultJointPositions[index]!,
      reference.jointPositions[index]!,
      authority
    );
    adjusted.jointVelocities[index] = reference.jointVelocities[index]! * authority;
    adjusted.jointTrackingWeights[index] =
      reference.jointTrackingWeights[index]! * authority;
  });
  return adjusted;
}

function stanceAuthority(planarDrift: number): number {
  const progress = clamp(
    (planarDrift - STANCE_FULL_AUTHORITY_DRIFT_METERS)
      / (STANCE_RELEASE_DRIFT_METERS - STANCE_FULL_AUTHORITY_DRIFT_METERS),
    0,
    1
  );
  return 1 - progress * progress * (3 - 2 * progress);
}

function isLowerBodyJoint(name: typeof HUMANOID_JOINT_NAMES[number]): boolean {
  return name.startsWith("waist_")
    || name.startsWith("left_hip_")
    || name.startsWith("right_hip_")
    || name.includes("knee")
    || name.includes("ankle");
}

function inversePolicyPlanarVelocity(
  desiredForward: number,
  desiredLateral: number
): { forward: number; lateral: number } {
  const desiredSpeed = Math.hypot(desiredForward, desiredLateral);
  if (desiredSpeed <= MINIMUM_CORRECTION_SPEED) {
    return { forward: 0, lateral: 0 };
  }
  const commandSpeed = desiredSpeed <= POLICY_LOW_SPEED_MAXIMUM
    ? desiredSpeed / POLICY_LOW_SPEED_SCALE
    : desiredSpeed < POLICY_RESPONSE_SWITCH_SPEED
      ? POLICY_LOW_SPEED_COMMAND
      : Math.max(
          POLICY_MINIMUM_COMMAND_SPEED,
          desiredSpeed + POLICY_HIGH_SPEED_OFFSET
        );
  const scaleToCommand = commandSpeed / desiredSpeed;
  const forward = desiredForward * scaleToCommand;
  const lateral = desiredLateral * scaleToCommand;
  const scaleToLimits = Math.min(
    1,
    Math.abs(forward) > 1e-9 ? MAXIMUM_FORWARD_SPEED / Math.abs(forward) : 1,
    Math.abs(lateral) > 1e-9 ? MAXIMUM_LATERAL_SPEED / Math.abs(lateral) : 1
  );
  return {
    forward: forward * scaleToLimits,
    lateral: lateral * scaleToLimits
  };
}

function normalizeAngle(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function mix(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
}

function approach(current: number, target: number, maximumDelta: number): number {
  if (![current, target, maximumDelta].every(Number.isFinite)
    || maximumDelta < 0) {
    throw new Error("Station-keeping command slew must be finite");
  }
  if (current < target) return Math.min(target, current + maximumDelta);
  return Math.max(target, current - maximumDelta);
}
