import { createHash } from "node:crypto";
import { z } from "zod";
import { QuaternionSchema, Vec3Schema, type Vec3 } from "../../domain/schema.js";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

const MotionDriftLimitsSchema = z.object({
  root_position_m: z.number().finite().positive(),
  root_orientation_rad: z.number().finite().positive(),
  joint_rms_rad: z.number().finite().positive(),
  end_effector_m: z.number().finite().positive(),
  consecutive_steps: z.number().int().positive()
}).strict();

const HUMANOID_MOTION_DRIFT_LIMITS = MotionDriftLimitsSchema.parse({
  root_position_m: 0.08,
  root_orientation_rad: 0.18,
  joint_rms_rad: 0.18,
  end_effector_m: 0.12,
  consecutive_steps: 3
});

const HumanoidMotionRolloutFrameSchema = z.object({
  atSeconds: z.number().finite().positive(),
  rootPosition: Vec3Schema,
  rootRotation: QuaternionSchema,
  jointPositions: z.array(z.number().finite()).length(HUMANOID_JOINT_NAMES.length),
  endEffectors: z.object({
    leftWrist: Vec3Schema,
    rightWrist: Vec3Schema,
    leftAnkle: Vec3Schema,
    rightAnkle: Vec3Schema
  }).strict()
}).strict();

export const HumanoidMotionRolloutSchema = z.object({
  version: z.literal(1),
  protocol: z.literal("humanoid-motion-rollout-v1"),
  limits: MotionDriftLimitsSchema,
  frames: z.array(HumanoidMotionRolloutFrameSchema).min(1)
}).strict().superRefine((rollout, context) => {
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

export type HumanoidMotionRolloutFrame = z.infer<
  typeof HumanoidMotionRolloutFrameSchema
>;
export type HumanoidMotionRollout = z.infer<typeof HumanoidMotionRolloutSchema>;

export interface HumanoidMotionDriftEvidence {
  drifted: boolean;
  rootPositionErrorMeters: number;
  rootOrientationErrorRadians: number;
  jointRmsErrorRadians: number;
  maximumEndEffectorErrorMeters: number;
}

export function createHumanoidMotionRollout(
  frames: readonly HumanoidMotionRolloutFrame[]
): HumanoidMotionRollout {
  return HumanoidMotionRolloutSchema.parse({
    version: 1,
    protocol: "humanoid-motion-rollout-v1",
    limits: HUMANOID_MOTION_DRIFT_LIMITS,
    frames
  });
}

export function captureHumanoidMotionRolloutFrame(
  atSeconds: number,
  snapshot: HumanoidSimulationSnapshot
): HumanoidMotionRolloutFrame {
  return HumanoidMotionRolloutFrameSchema.parse({
    atSeconds,
    rootPosition: snapshot.rootPosition,
    rootRotation: snapshot.rootRotation,
    jointPositions: HUMANOID_JOINT_NAMES.map((name) => snapshot.joints[name].position),
    endEffectors: {
      leftWrist: snapshot.links.left_wrist_yaw_link.position,
      rightWrist: snapshot.links.right_wrist_yaw_link.position,
      leftAnkle: snapshot.links.left_ankle_roll_link.position,
      rightAnkle: snapshot.links.right_ankle_roll_link.position
    }
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
  limits = HUMANOID_MOTION_DRIFT_LIMITS
): HumanoidMotionDriftEvidence {
  const parsedLimits = MotionDriftLimitsSchema.parse(limits);
  const rootPositionErrorMeters = distance(actual.rootPosition, expected.rootPosition);
  const rootOrientationErrorRadians = quaternionAngularDistance(
    actual.rootRotation,
    expected.rootRotation
  );
  const jointRmsErrorRadians = Math.sqrt(HUMANOID_JOINT_NAMES.reduce(
    (sum, name, index) => {
      const error = actual.joints[name].position - expected.jointPositions[index]!;
      return sum + error * error;
    },
    0
  ) / HUMANOID_JOINT_NAMES.length);
  const endEffectorErrors = [
    distance(actual.links.left_wrist_yaw_link.position, expected.endEffectors.leftWrist),
    distance(actual.links.right_wrist_yaw_link.position, expected.endEffectors.rightWrist),
    distance(actual.links.left_ankle_roll_link.position, expected.endEffectors.leftAnkle),
    distance(actual.links.right_ankle_roll_link.position, expected.endEffectors.rightAnkle)
  ];
  const maximumEndEffectorErrorMeters = Math.max(...endEffectorErrors);
  return {
    drifted: rootPositionErrorMeters > parsedLimits.root_position_m
      || rootOrientationErrorRadians > parsedLimits.root_orientation_rad
      || jointRmsErrorRadians > parsedLimits.joint_rms_rad
      || maximumEndEffectorErrorMeters > parsedLimits.end_effector_m,
    rootPositionErrorMeters,
    rootOrientationErrorRadians,
    jointRmsErrorRadians,
    maximumEndEffectorErrorMeters
  };
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
