import { z } from "zod";
import { QuaternionSchema, Vec3Schema } from "../../domain/schema.js";
import { HUMANOID_BODY_NAMES, HUMANOID_JOINT_NAMES } from "./model.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  G1_HAND_JOINT_NAMES,
  G1_HAND_LINK_NAMES
} from "./morphology.js";
import {
  HumanoidMotionGeneratorDescriptorSchema,
  TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR
} from "./motion-generator-contract.js";
import { HumanoidPhysicalSafetyEvidenceSchema } from "./physical-safety.js";
import { HumanoidWorldGraspStateSchema } from "./grasp-world-state.js";
import {
  HUMANOID_POLICY_OBSERVATION_FEATURES
} from "./whole-body-controller.js";
import {
  HUMANOID_LEARNED_POLICY_CAPABILITIES
} from "../../domain/humanoid-policy.js";

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
  maximum: z.number().finite(),
  effort: z.object({
    requestedNewtonMeters: z.number().finite(),
    appliedNewtonMeters: z.number().finite(),
    minimumNewtonMeters: z.number().finite(),
    maximumNewtonMeters: z.number().finite(),
    requestedUtilization: z.number().finite().nonnegative(),
    appliedUtilization: z.number().finite().nonnegative(),
    saturated: z.boolean()
  }).strict().optional()
}).strict();

const HandJointSchema = z.object({
  position: z.number().finite(),
  velocity: z.number().finite(),
  target: z.number().finite(),
  minimum: z.number().finite(),
  maximum: z.number().finite(),
  stiffnessNewtonMetersPerRadian: z.number().finite().positive(),
  dampingNewtonMeterSecondsPerRadian: z.number().finite().nonnegative(),
  appliedNewtonMeters: z.number().finite(),
  minimumNewtonMeters: z.number().finite(),
  maximumNewtonMeters: z.number().finite(),
  saturated: z.boolean()
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
  secondObject: z.string().min(1).nullable(),
  firstSolid: z.string().min(1).nullable().optional(),
  secondSolid: z.string().min(1).nullable().optional(),
  firstHandLink: z.enum(G1_HAND_CONTACT_SURFACE_NAMES).nullable().default(null),
  secondHandLink: z.enum(G1_HAND_CONTACT_SURFACE_NAMES).nullable().default(null)
}).strict();

const ObjectSchema = LinkSchema.extend({
  id: z.string().min(1),
  articulation: z.object({
    type: z.enum(["hinge", "slide"]),
    position: z.number().finite(),
    velocity: z.number().finite(),
    minimum: z.number().finite(),
    maximum: z.number().finite(),
    normalized: z.number().finite().min(0).max(1)
  }).strict().optional()
}).strict();

const HumanoidSimulationSnapshotSchema = z.object({
  morphology: z.object({
    id: z.literal("unitree_g1_43dof_with_hands"),
    bodyJointCount: z.literal(29),
    handJointCount: z.literal(14),
    totalJointCount: z.literal(43),
    source: z.object({
      repository: z.literal("google-deepmind/mujoco_menagerie"),
      commit: z.literal("71f066ad0be9cd271f7ed58c030243ef157af9f4"),
      model: z.literal("unitree_g1/g1_with_hands.xml")
    }).strict()
  }).strict(),
  simulatedTime: z.number().finite().nonnegative(),
  controller: z.object({
    protocol: z.literal("humanoid-controller-v1"),
    implementation: z.string().trim().min(1),
    actuation: z.literal("joint_position_pd"),
    controlStepSeconds: z.number().finite().positive(),
    physicsStepSeconds: z.number().finite().positive(),
    commandResponseHorizonSeconds: z.number().finite().positive().optional(),
    minimumEffectivePlanarSpeedMetersPerSecond:
      z.number().finite().positive().optional(),
    learnedPolicy: z.object({
      protocol: z.literal("humanoid-learned-policy-v1"),
      runtime: z.string().trim().min(1),
      observationSpace: z.object({
        protocol: z.string().trim().min(1),
        size: z.number().int().positive()
      }).strict(),
      actionSpace: z.object({
        protocol: z.string().trim().min(1),
        size: z.number().int().positive()
      }).strict(),
      observationFeatures: z.array(
        z.enum(HUMANOID_POLICY_OBSERVATION_FEATURES)
      ).optional(),
      capabilities: z.array(z.enum(HUMANOID_LEARNED_POLICY_CAPABILITIES))
    }).strict().optional(),
    capabilityRouting: z.object({
      protocol: z.literal("humanoid-controller-capability-routing-v1"),
      strategy: z.literal("declared_capabilities"),
      fallback: z.object({
        mode: z.literal("reference_control"),
        implementation: z.string().trim().min(1)
      }).strict()
    }).strict().optional()
  }).strict(),
  rootPosition: Vec3Schema,
  rootRotation: QuaternionSchema,
  joints: z.record(z.enum(HUMANOID_JOINT_NAMES), JointSchema),
  links: z.record(z.enum(HUMANOID_BODY_NAMES), LinkSchema),
  hands: z.object({
    controller: z.object({
      protocol: z.literal("g1-hand-controller-v1"),
      implementation: z.literal("mujoco_continuous_position_pd"),
      actuation: z.literal("joint_position_pd"),
      jointCount: z.literal(14)
    }).strict(),
    joints: z.record(z.enum(G1_HAND_JOINT_NAMES), HandJointSchema),
    links: z.record(z.enum(G1_HAND_LINK_NAMES), LinkSchema)
  }).strict(),
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

const LegacyHumanoidSimulationSnapshotSchema = (
  HumanoidSimulationSnapshotSchema.omit({ morphology: true, hands: true })
);

const HumanoidWorldSnapshotBaseSchema = z.object({
  frame: z.number().int().nonnegative(),
  worldRevision: z.number().int().nonnegative(),
  motionGenerator: HumanoidMotionGeneratorDescriptorSchema.default(
    TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR
  ),
  physicalSafety: z.object({
    planId: z.string().min(1),
    evidence: HumanoidPhysicalSafetyEvidenceSchema
  }).strict().optional(),
  robot: HumanoidSimulationSnapshotSchema,
  grasp: HumanoidWorldGraspStateSchema,
  navigation: z.object({
    planId: z.string().min(1).nullable(),
    status: z.enum(["idle", "planned", "executing", "completed", "blocked"]),
    target: Vec3Schema.nullable(),
    waypoints: z.array(Vec3Schema),
    waypointIndex: z.number().int().nonnegative().nullable()
  }).strict()
}).strict();

export const HumanoidWorldSnapshotSchema = HumanoidWorldSnapshotBaseSchema
  .superRefine((snapshot, context) => {
    snapshot.grasp.assessments.forEach((assessment, index) => {
      if (assessment.frame !== snapshot.frame) {
        context.addIssue({
          code: "custom",
          path: ["grasp", "assessments", index, "frame"],
          message: "World grasp assessment must belong to the snapshot frame"
        });
      }
    });
  });

export const PreGraspHumanoidWorldSnapshotSchema =
  HumanoidWorldSnapshotBaseSchema.omit({ grasp: true });

export const LegacyHumanoidWorldSnapshotSchema =
  PreGraspHumanoidWorldSnapshotSchema.extend({
  robot: LegacyHumanoidSimulationSnapshotSchema
}).strict();
