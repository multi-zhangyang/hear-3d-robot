import { createHash } from "node:crypto";
import { z } from "zod";
import { QuaternionSchema, Vec3Schema } from "../../domain/schema.js";
import { normalizeQuaternion } from "../geometry.js";
import { HUMANOID_END_EFFECTOR_BODIES } from "./task-space-targets.js";

export const HUMANOID_TASK_SPACE_SERVO_AUTHORITY = Object.freeze({
  protocol: "humanoid-task-space-servo-v1" as const,
  positionConvergenceMeters: 0.015,
  orientationConvergenceRadians: 0.02,
  maximumIterations: 128,
  damping: 0.018,
  maximumJointDeltaRadians: 0.14,
  maximumReferenceCorrectionRadians: 0.06
});

const HUMANOID_TASK_SPACE_SERVO_AUTHORITY_SHA256 = createHash("sha256")
  .update(JSON.stringify(HUMANOID_TASK_SPACE_SERVO_AUTHORITY))
  .digest("hex");

export const HumanoidTaskSpaceServoDescriptorSchema = z.object({
  protocol: z.literal(HUMANOID_TASK_SPACE_SERVO_AUTHORITY.protocol),
  authoritySha256: z.literal(HUMANOID_TASK_SPACE_SERVO_AUTHORITY_SHA256)
}).strict();

export const HUMANOID_TASK_SPACE_SERVO_DESCRIPTOR = Object.freeze({
  protocol: HUMANOID_TASK_SPACE_SERVO_AUTHORITY.protocol,
  authoritySha256: HUMANOID_TASK_SPACE_SERVO_AUTHORITY_SHA256
});

const HumanoidTaskSpaceServoTargetSchema = z.object({
  body: z.enum(HUMANOID_END_EFFECTOR_BODIES),
  position: Vec3Schema,
  frame: z.enum(["world", "pelvis"]),
  tolerance: z.number().finite().positive(),
  orientation: QuaternionSchema.optional(),
  orientationTolerance: z.number().finite().positive().max(Math.PI).optional()
}).strict().superRefine((target, context) => {
  const hasOrientation = target.orientation !== undefined;
  const hasTolerance = target.orientationTolerance !== undefined;
  if (hasOrientation !== hasTolerance) {
    context.addIssue({
      code: "custom",
      path: [hasOrientation ? "orientationTolerance" : "orientation"],
      message: "Task-space servo orientation and tolerance must be provided together"
    });
  }
  if (target.orientation) {
    try {
      normalizeQuaternion(target.orientation);
    } catch (error) {
      context.addIssue({
        code: "custom",
        path: ["orientation"],
        message: error instanceof Error ? error.message : "Invalid servo quaternion"
      });
    }
  }
});

export const HumanoidTaskSpaceServoTargetsSchema = z.array(
  HumanoidTaskSpaceServoTargetSchema
).min(1).max(HUMANOID_END_EFFECTOR_BODIES.length).superRefine((targets, context) => {
  const bodies = new Set<string>();
  targets.forEach((target, index) => {
    if (bodies.has(target.body)) {
      context.addIssue({
        code: "custom",
        path: [index, "body"],
        message: "A task-space servo frame cannot repeat an end effector"
      });
    }
    bodies.add(target.body);
  });
});

export type HumanoidTaskSpaceServoTarget = z.infer<
  typeof HumanoidTaskSpaceServoTargetSchema
>;
