import { z } from "zod";
import { JsonValueSchema, Vec3Schema } from "../../domain/schema.js";
import {
  HumanoidMotionArtifactSchema,
  HumanoidReferenceStateSchema
} from "./motion-artifact.js";
import { HumanoidMotionPlanSchema } from "./motion-plan.js";
import {
  HumanoidMotionGeneratorDescriptorSchema,
  TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR
} from "./motion-generator-contract.js";
import { HUMANOID_BODY_NAMES } from "./model.js";
import { HumanoidObjectMemoryCheckpointSchema } from "./object-memory.js";

const FiniteArraySchema = z.array(z.number().finite());

const NavigationPlanSchema = z.object({
  waypoints: z.array(Vec3Schema),
  distance: z.number().finite().nonnegative(),
  resolvedTarget: Vec3Schema,
  projectionDistance: z.number().finite().nonnegative()
}).strict();

const NavigationStateSchema = z.object({
  planId: z.string().min(1).nullable(),
  status: z.enum(["idle", "planned", "executing", "completed", "blocked"]),
  target: Vec3Schema.nullable(),
  waypoints: z.array(Vec3Schema),
  waypointIndex: z.number().int().nonnegative().nullable()
}).strict();

const HumanoidMotionExecutionFailureSchema = z.object({
  code: z.enum(["fallen", "environment_contact"]),
  atSeconds: z.number().finite().nonnegative(),
  bodies: z.array(z.enum(HUMANOID_BODY_NAMES)).optional(),
  contacts: z.array(z.object({
    body: z.enum(HUMANOID_BODY_NAMES),
    objectId: z.string().min(1).nullable(),
    normalForce: z.number().finite()
  }).strict()).optional()
}).strict();

const HumanoidMotionExecutionProgressSchema = z.object({
  nextFrameIndex: z.number().int().nonnegative(),
  satisfiedContactKeys: z.array(z.string().min(1)),
  failure: HumanoidMotionExecutionFailureSchema.nullable()
}).strict();

export type HumanoidMotionExecutionProgress = z.infer<
  typeof HumanoidMotionExecutionProgressSchema
>;

const StoredMotionSchema = z.object({
  plan: HumanoidMotionPlanSchema,
  artifact: HumanoidMotionArtifactSchema,
  createdRevision: z.number().int().nonnegative(),
  progress: HumanoidMotionExecutionProgressSchema.default({
    nextFrameIndex: 0,
    satisfiedContactKeys: [],
    failure: null
  })
}).strict().superRefine((motion, context) => {
  if (motion.progress.nextFrameIndex > motion.artifact.frames.length) {
    context.addIssue({
      code: "custom",
      path: ["progress", "nextFrameIndex"],
      message: "Humanoid motion progress exceeds its artifact frame count"
    });
  }
});

export const HumanoidWorldCheckpointSchema = z.object({
  version: z.literal(1),
  motionGenerator: HumanoidMotionGeneratorDescriptorSchema.default(
    TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR
  ),
  frame: z.number().int().nonnegative(),
  worldRevision: z.number().int().nonnegative(),
  routeSequence: z.number().int().nonnegative(),
  simulation: z.object({
    time: z.number().finite().nonnegative(),
    positions: FiniteArraySchema,
    velocities: FiniteArraySchema,
    controls: FiniteArraySchema,
    activations: FiniteArraySchema,
    accelerationWarmstart: FiniteArraySchema,
    controller: z.object({
      protocol: z.literal("humanoid-controller-state-v1"),
      version: z.literal(1),
      implementation: z.string().trim().min(1),
      payload: JsonValueSchema
    }).strict()
  }).strict(),
  reference: HumanoidReferenceStateSchema,
  motions: z.array(StoredMotionSchema),
  routes: z.array(z.object({
    id: z.string().min(1),
    plan: NavigationPlanSchema,
    requestedTarget: Vec3Schema,
    createdRevision: z.number().int().nonnegative()
  }).strict()),
  navigation: NavigationStateSchema,
  objectMemory: HumanoidObjectMemoryCheckpointSchema.default({
    version: 1,
    records: []
  })
}).strict();

export type HumanoidWorldCheckpoint = z.infer<typeof HumanoidWorldCheckpointSchema>;
