import { createHash } from "node:crypto";
import { z } from "zod";
import { QuaternionSchema, Vec3Schema, type Vec3 } from "../../domain/schema.js";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import { G1_HAND_JOINT_NAMES } from "./morphology.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

const MotionDriftLimitsV1Schema = z.object({
  root_position_m: z.number().finite().positive(),
  root_orientation_rad: z.number().finite().positive(),
  joint_rms_rad: z.number().finite().positive(),
  end_effector_m: z.number().finite().positive(),
  end_effector_orientation_rad: z.number().finite().positive().optional(),
  consecutive_steps: z.number().int().positive()
}).strict();

const MotionDriftLimitsV2Schema = MotionDriftLimitsV1Schema.extend({
  hand_joint_rms_rad: z.number().finite().positive(),
  hand_target_rad: z.number().finite().positive()
}).strict();

const HUMANOID_MOTION_DRIFT_LIMITS_V1 = MotionDriftLimitsV1Schema.parse({
  root_position_m: 0.08,
  root_orientation_rad: 0.18,
  joint_rms_rad: 0.18,
  end_effector_m: 0.12,
  end_effector_orientation_rad: 0.24,
  consecutive_steps: 3
});

const HUMANOID_MOTION_DRIFT_LIMITS_V2 = MotionDriftLimitsV2Schema.parse({
  ...HUMANOID_MOTION_DRIFT_LIMITS_V1,
  hand_joint_rms_rad: 0.12,
  hand_target_rad: 1e-9
});

const EndEffectorPositionsSchema = z.object({
  leftWrist: Vec3Schema,
  rightWrist: Vec3Schema,
  leftAnkle: Vec3Schema,
  rightAnkle: Vec3Schema
}).strict();

const EndEffectorRotationsSchema = z.object({
  leftWrist: QuaternionSchema,
  rightWrist: QuaternionSchema,
  leftAnkle: QuaternionSchema,
  rightAnkle: QuaternionSchema
}).strict();

const HumanoidMotionRolloutFrameV1Schema = z.object({
  atSeconds: z.number().finite().positive(),
  rootPosition: Vec3Schema,
  rootRotation: QuaternionSchema,
  jointPositions: z.array(z.number().finite()).length(HUMANOID_JOINT_NAMES.length),
  endEffectors: EndEffectorPositionsSchema,
  endEffectorRotations: EndEffectorRotationsSchema.optional()
}).strict();

const HumanoidMotionRolloutFrameV2Schema = z.object({
  atSeconds: z.number().finite().positive(),
  rootPosition: Vec3Schema,
  rootRotation: QuaternionSchema,
  jointPositions: z.array(z.number().finite()).length(HUMANOID_JOINT_NAMES.length),
  handJointPositions: z.array(z.number().finite()).length(G1_HAND_JOINT_NAMES.length),
  handJointTargets: z.array(z.number().finite()).length(G1_HAND_JOINT_NAMES.length),
  endEffectors: EndEffectorPositionsSchema,
  endEffectorRotations: EndEffectorRotationsSchema
}).strict();

const HumanoidMotionRolloutV1Schema = z.object({
  version: z.literal(1),
  protocol: z.literal("humanoid-motion-rollout-v1"),
  limits: MotionDriftLimitsV1Schema,
  frames: z.array(HumanoidMotionRolloutFrameV1Schema).min(1)
}).strict();

const HumanoidMotionRolloutV2Schema = z.object({
  version: z.literal(2),
  protocol: z.literal("humanoid-motion-rollout-v2"),
  limits: MotionDriftLimitsV2Schema,
  frames: z.array(HumanoidMotionRolloutFrameV2Schema).min(1)
}).strict();

export const HumanoidMotionRolloutSchema = z.discriminatedUnion("version", [
  HumanoidMotionRolloutV1Schema,
  HumanoidMotionRolloutV2Schema
]).superRefine((rollout, context) => {
  let previous = 0;
  for (let index = 0; index < rollout.frames.length; index += 1) {
    const atSeconds = rollout.frames[index]!.atSeconds;
    if (atSeconds <= previous) {
      context.addIssue({
        code: "custom",
        path: ["frames", index, "atSeconds"],
        message: "Humanoid motion rollout frame times must increase"
      });
    }
    previous = atSeconds;
  }
});

export type HumanoidMotionRolloutFrameV1 = z.infer<
  typeof HumanoidMotionRolloutFrameV1Schema
>;
export type HumanoidMotionRolloutFrameV2 = z.infer<
  typeof HumanoidMotionRolloutFrameV2Schema
>;
export type HumanoidMotionRolloutFrame =
  | HumanoidMotionRolloutFrameV1
  | HumanoidMotionRolloutFrameV2;
export type HumanoidMotionRollout = z.infer<typeof HumanoidMotionRolloutSchema>;
export type HumanoidMotionDriftLimits = HumanoidMotionRollout["limits"];

export interface HumanoidMotionDriftEvidence {
  drifted: boolean;
  rootPositionErrorMeters: number;
  rootOrientationErrorRadians: number;
  jointRmsErrorRadians: number;
  maximumEndEffectorErrorMeters: number;
  maximumEndEffectorOrientationErrorRadians?: number | undefined;
  handJointRmsErrorRadians?: number | undefined;
  maximumHandTargetErrorRadians?: number | undefined;
}

export function createHumanoidMotionRollout(
  frames: readonly HumanoidMotionRolloutFrame[],
  artifactVersion?: 1 | 2
): HumanoidMotionRollout {
  if (frames.length === 0) {
    throw new Error("Humanoid motion rollout requires at least one frame");
  }
  const version2Frames = frames.filter(isRolloutFrameV2);
  if (version2Frames.length !== 0 && version2Frames.length !== frames.length) {
    throw new Error("Humanoid motion rollout cannot mix version 1 and version 2 frames");
  }
  const rolloutVersion = version2Frames.length === frames.length ? 2 : 1;
  if (artifactVersion !== undefined && rolloutVersion !== artifactVersion) {
    throw new Error("Humanoid motion rollout version must match its motion artifact");
  }
  return HumanoidMotionRolloutSchema.parse(rolloutVersion === 2 ? {
    version: 2,
    protocol: "humanoid-motion-rollout-v2",
    limits: HUMANOID_MOTION_DRIFT_LIMITS_V2,
    frames
  } : {
    version: 1,
    protocol: "humanoid-motion-rollout-v1",
    limits: HUMANOID_MOTION_DRIFT_LIMITS_V1,
    frames
  });
}

export function captureHumanoidMotionRolloutFrame(
  atSeconds: number,
  snapshot: HumanoidSimulationSnapshot
): HumanoidMotionRolloutFrameV1;
export function captureHumanoidMotionRolloutFrame(
  atSeconds: number,
  snapshot: HumanoidSimulationSnapshot,
  artifactVersion: 1
): HumanoidMotionRolloutFrameV1;
export function captureHumanoidMotionRolloutFrame(
  atSeconds: number,
  snapshot: HumanoidSimulationSnapshot,
  artifactVersion: 2
): HumanoidMotionRolloutFrameV2;
export function captureHumanoidMotionRolloutFrame(
  atSeconds: number,
  snapshot: HumanoidSimulationSnapshot,
  artifactVersion: 1 | 2
): HumanoidMotionRolloutFrame;
export function captureHumanoidMotionRolloutFrame(
  atSeconds: number,
  snapshot: HumanoidSimulationSnapshot,
  artifactVersion: 1 | 2 = 1
): HumanoidMotionRolloutFrame {
  const frame = {
    atSeconds,
    rootPosition: snapshot.rootPosition,
    rootRotation: snapshot.rootRotation,
    jointPositions: HUMANOID_JOINT_NAMES.map((name) => snapshot.joints[name].position),
    endEffectors: {
      leftWrist: snapshot.links.left_wrist_yaw_link.position,
      rightWrist: snapshot.links.right_wrist_yaw_link.position,
      leftAnkle: snapshot.links.left_ankle_roll_link.position,
      rightAnkle: snapshot.links.right_ankle_roll_link.position
    },
    endEffectorRotations: {
      leftWrist: snapshot.links.left_wrist_yaw_link.rotation,
      rightWrist: snapshot.links.right_wrist_yaw_link.rotation,
      leftAnkle: snapshot.links.left_ankle_roll_link.rotation,
      rightAnkle: snapshot.links.right_ankle_roll_link.rotation
    }
  };
  return artifactVersion === 1
    ? HumanoidMotionRolloutFrameV1Schema.parse(frame)
    : HumanoidMotionRolloutFrameV2Schema.parse({
        ...frame,
        handJointPositions: G1_HAND_JOINT_NAMES.map(
          (name) => snapshot.hands.joints[name].position
        ),
        handJointTargets: G1_HAND_JOINT_NAMES.map(
          (name) => snapshot.hands.joints[name].target
        )
      });
}

export function humanoidMotionRolloutSha256(
  rollout: HumanoidMotionRollout
): string {
  const parsed = HumanoidMotionRolloutSchema.parse(rollout);
  return createHash("sha256").update(JSON.stringify(parsed)).digest("hex");
}

export function detectHumanoidMotionDrift(
  actual: HumanoidSimulationSnapshot,
  expected: HumanoidMotionRolloutFrame,
  limits: HumanoidMotionDriftLimits = isRolloutFrameV2(expected)
    ? HUMANOID_MOTION_DRIFT_LIMITS_V2
    : HUMANOID_MOTION_DRIFT_LIMITS_V1
): HumanoidMotionDriftEvidence {
  const handLimits = isRolloutFrameV2(expected)
    ? MotionDriftLimitsV2Schema.parse(limits)
    : undefined;
  const parsedLimits = handLimits ?? MotionDriftLimitsV1Schema.parse(limits);
  const rootPositionErrorMeters = distance(actual.rootPosition, expected.rootPosition);
  const rootOrientationErrorRadians = quaternionAngularDistance(
    actual.rootRotation,
    expected.rootRotation
  );
  const jointRmsErrorRadians = rms(HUMANOID_JOINT_NAMES.map((name, index) => (
    actual.joints[name].position - expected.jointPositions[index]!
  )));
  const endEffectorErrors = [
    distance(actual.links.left_wrist_yaw_link.position, expected.endEffectors.leftWrist),
    distance(actual.links.right_wrist_yaw_link.position, expected.endEffectors.rightWrist),
    distance(actual.links.left_ankle_roll_link.position, expected.endEffectors.leftAnkle),
    distance(actual.links.right_ankle_roll_link.position, expected.endEffectors.rightAnkle)
  ];
  const maximumEndEffectorErrorMeters = Math.max(...endEffectorErrors);
  const maximumEndEffectorOrientationErrorRadians = expected.endEffectorRotations
    ? Math.max(
        quaternionAngularDistance(
          actual.links.left_wrist_yaw_link.rotation,
          expected.endEffectorRotations.leftWrist
        ),
        quaternionAngularDistance(
          actual.links.right_wrist_yaw_link.rotation,
          expected.endEffectorRotations.rightWrist
        ),
        quaternionAngularDistance(
          actual.links.left_ankle_roll_link.rotation,
          expected.endEffectorRotations.leftAnkle
        ),
        quaternionAngularDistance(
          actual.links.right_ankle_roll_link.rotation,
          expected.endEffectorRotations.rightAnkle
        )
      )
    : 0;
  const handJointRmsErrorRadians = isRolloutFrameV2(expected)
    ? rms(G1_HAND_JOINT_NAMES.map((name, index) => (
        actual.hands.joints[name].position - expected.handJointPositions[index]!
      )))
    : undefined;
  const maximumHandTargetErrorRadians = isRolloutFrameV2(expected)
    ? Math.max(...G1_HAND_JOINT_NAMES.map((name, index) => Math.abs(
        actual.hands.joints[name].target - expected.handJointTargets[index]!
      )))
    : undefined;
  const handDrifted = isRolloutFrameV2(expected)
    && handJointRmsErrorRadians !== undefined
    && maximumHandTargetErrorRadians !== undefined
    && handLimits !== undefined
    && (handJointRmsErrorRadians > handLimits.hand_joint_rms_rad
      || maximumHandTargetErrorRadians > handLimits.hand_target_rad);
  return {
    drifted: rootPositionErrorMeters > parsedLimits.root_position_m
      || rootOrientationErrorRadians > parsedLimits.root_orientation_rad
      || jointRmsErrorRadians > parsedLimits.joint_rms_rad
      || maximumEndEffectorErrorMeters > parsedLimits.end_effector_m
      || maximumEndEffectorOrientationErrorRadians
        > (parsedLimits.end_effector_orientation_rad
          ?? parsedLimits.root_orientation_rad)
      || handDrifted,
    rootPositionErrorMeters,
    rootOrientationErrorRadians,
    jointRmsErrorRadians,
    maximumEndEffectorErrorMeters,
    maximumEndEffectorOrientationErrorRadians,
    ...(handJointRmsErrorRadians === undefined
      ? {}
      : { handJointRmsErrorRadians }),
    ...(maximumHandTargetErrorRadians === undefined
      ? {}
      : { maximumHandTargetErrorRadians })
  };
}

function isRolloutFrameV2(
  frame: HumanoidMotionRolloutFrame
): frame is HumanoidMotionRolloutFrameV2 {
  return "handJointPositions" in frame && "handJointTargets" in frame;
}

function rms(errors: readonly number[]): number {
  return Math.sqrt(errors.reduce((sum, value) => sum + value * value, 0) / errors.length);
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(
    left.x - right.x,
    left.y - right.y,
    left.z - right.z
  );
}

function quaternionAngularDistance(
  left: HumanoidMotionRolloutFrame["rootRotation"],
  right: HumanoidMotionRolloutFrame["rootRotation"]
): number {
  const dot = Math.abs(
    left.w * right.w
      + left.x * right.x
      + left.y * right.y
      + left.z * right.z
  );
  return 2 * Math.acos(Math.min(1, Math.max(-1, dot)));
}
