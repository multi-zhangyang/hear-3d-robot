import { z } from "zod";
import { QuaternionSchema, Vec3Schema } from "../../domain/schema.js";
import { HUMANOID_BODY_NAMES, HUMANOID_JOINT_NAMES } from "./model.js";
import {
  HumanoidMotionGeneratorDescriptorSchema,
  TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR
} from "./motion-generator-contract.js";

const LinkSchema = z.object({
  position: Vec3Schema,
  rotation: QuaternionSchema,
  linearVelocity: Vec3Schema,
  angularVelocity: Vec3Schema
}).strict();

const JointSchema = z.object({
  position: z.number().finite(),
  velocity: z.number().finite(),
  minimum: z.number().finite(),
  maximum: z.number().finite()
}).strict();

const FootSchema = z.object({
  touching: z.boolean(),
  contactCount: z.number().int().nonnegative(),
  normalForce: z.number().finite().nonnegative(),
  points: z.array(Vec3Schema)
}).strict();

const ContactSchema = z.object({
  position: Vec3Schema,
  normal: Vec3Schema,
  normalForce: z.number().finite().nonnegative(),
  firstBody: z.enum(HUMANOID_BODY_NAMES).nullable(),
  secondBody: z.enum(HUMANOID_BODY_NAMES).nullable(),
  firstObject: z.string().min(1).nullable(),
  secondObject: z.string().min(1).nullable()
}).strict();

const ObjectSchema = LinkSchema.extend({
  id: z.string().min(1)
}).strict();

const HumanoidSimulationSnapshotSchema = z.object({
  simulatedTime: z.number().finite().nonnegative(),
  controller: z.object({
    protocol: z.literal("humanoid-controller-v1"),
    implementation: z.string().trim().min(1),
    actuation: z.literal("joint_position_pd"),
    controlStepSeconds: z.number().finite().positive(),
    physicsStepSeconds: z.number().finite().positive()
  }).strict(),
  rootPosition: Vec3Schema,
  rootRotation: QuaternionSchema,
  joints: z.record(z.enum(HUMANOID_JOINT_NAMES), JointSchema),
  links: z.record(z.enum(HUMANOID_BODY_NAMES), LinkSchema),
  objects: z.record(z.string().min(1), ObjectSchema),
  contactCount: z.number().int().nonnegative(),
  contacts: z.array(ContactSchema),
  feet: z.object({ left: FootSchema, right: FootSchema }).strict(),
  balance: z.object({
    centerOfMass: Vec3Schema,
    support: z.enum(["none", "left", "right", "double"]),
    supportMargin: z.number().finite().nullable(),
    upright: z.number().finite()
  }).strict(),
  nonFootEnvironmentContacts: z.array(z.enum(HUMANOID_BODY_NAMES)),
  fallen: z.boolean()
}).strict();

export const HumanoidWorldSnapshotSchema = z.object({
  frame: z.number().int().nonnegative(),
  worldRevision: z.number().int().nonnegative(),
  motionGenerator: HumanoidMotionGeneratorDescriptorSchema.default(
    TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR
  ),
  robot: HumanoidSimulationSnapshotSchema,
  navigation: z.object({
    planId: z.string().min(1).nullable(),
    status: z.enum(["idle", "planned", "executing", "completed", "blocked"]),
    target: Vec3Schema.nullable(),
    waypoints: z.array(Vec3Schema),
    waypointIndex: z.number().int().nonnegative().nullable()
  }).strict()
}).strict();
