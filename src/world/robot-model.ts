export const ROBOT_SPEC = {
  base: {
    centerY: 0.38,
    halfExtents: { x: 0.36, y: 0.36, z: 0.3 },
    cornerRadius: 0.08,
    footprintRadius: 0.46,
    maximumLinearVelocity: 0.8,
    maximumAngularVelocity: 1.6
  },
  wheels: {
    radius: 0.14,
    trackWidth: 0.64
  },
  torso: {
    centerHeight: 0.78,
    halfExtents: { x: 0.25, y: 0.32, z: 0.2 }
  },
  sensorHead: {
    centerHeight: 1.32,
    forwardOffset: -0.24,
    halfExtents: { x: 0.18, y: 0.13, z: 0.16 },
    horizontalFieldOfView: Math.PI * 0.72,
    verticalFieldOfView: Math.PI * 0.5
  },
  arm: {
    shoulderHeight: 0.92,
    shoulderForwardOffset: 0.16,
    upperLength: 0.62,
    forearmLength: 0.55,
    wristLength: 0.18,
    upperHalfExtents: { x: 0.09, y: 0.09, z: 0.29 },
    forearmHalfExtents: { x: 0.085, y: 0.085, z: 0.255 },
    wristHalfExtents: { x: 0.08, y: 0.08, z: 0.08 }
  },
  gripper: {
    fingerHalfExtents: { x: 0.035, y: 0.14, z: 0.08 },
    cornerRadius: 0.02,
    minimumContactFrames: 2,
    minimumStableAttachmentFrames: 3,
    maximumAttachmentPositionDrift: 0.045,
    maximumAttachmentRotationDrift: 0.18,
    slipDetectionFrames: 2,
    defaultMaximumForce: 1000
  },
  joints: {
    head_yaw: { minimum: -Math.PI, maximum: Math.PI, maximumVelocity: 1.8 },
    head_pitch: { minimum: -0.85, maximum: 0.85, maximumVelocity: 1.2 },
    shoulder: { minimum: -1.55, maximum: 1.55, maximumVelocity: 1.15 },
    elbow: { minimum: -2.5, maximum: -0.05, maximumVelocity: 1.3 },
    wrist: { minimum: -1.7, maximum: 1.7, maximumVelocity: 1.5 },
    // The opening clears the diagonal of the largest 0.5m payload plus both
    // finger pads, so a rotated block can fall free instead of balancing on a
    // finger after the physical attachment is released.
    gripper_aperture: { minimum: 0.04, maximum: 0.82, maximumVelocity: 0.38 }
  },
  defaultJoints: {
    head_yaw: 0,
    head_pitch: 0,
    shoulder: 1.25,
    elbow: -2.2,
    wrist: 0.95,
    gripper_aperture: 0.4
  }
} as const;

export type RobotJointName = keyof typeof ROBOT_SPEC.joints;
export type ArmJointName = "shoulder" | "elbow" | "wrist";

export interface RobotJointState {
  head_yaw: number;
  head_pitch: number;
  shoulder: number;
  elbow: number;
  wrist: number;
  gripper_aperture: number;
}

export function jointLimitIssue(
  targets: Partial<RobotJointState>
): { joint: RobotJointName; value: number; minimum: number; maximum: number } | undefined {
  for (const [joint, rawValue] of Object.entries(targets) as [RobotJointName, number][]) {
    const limit = ROBOT_SPEC.joints[joint];
    if (!Number.isFinite(rawValue) || rawValue < limit.minimum || rawValue > limit.maximum) {
      return { joint, value: rawValue, minimum: limit.minimum, maximum: limit.maximum };
    }
  }
  return undefined;
}
