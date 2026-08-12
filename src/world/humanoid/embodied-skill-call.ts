import { z } from "zod";
import {
  HumanoidSkillIdSchema,
  HumanoidSkillInvocationSchema
} from "../../domain/humanoid-skill.js";
import {
  HUMANOID_LEARNED_POLICY_CAPABILITIES
} from "../../domain/humanoid-policy.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import {
  QuaternionSchema,
  Vec3Schema
} from "../../domain/schema.js";
import { HumanoidContactConstraintSchema } from "./motion-plan-schema.js";
import { HumanoidMotionOptionContractSchema } from "./motion-option-contract.js";

const NonEmptyIdSchema = z.string().trim().min(1);
const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const UnitRateSchema = z.number().finite().min(0).max(1);

export const HumanoidEmbodiedSkillIdentitySchema = z.object({
  protocol: z.literal("humanoid-embodied-skill-identity-v1"),
  callId: NonEmptyIdSchema,
  runtimeKind: z.enum([
    "semantic_skill",
    "navigation",
    "station_keeping",
    "legacy_motion"
  ]),
  agentId: NonEmptyIdSchema.nullable(),
  bindingTransactionId: NonEmptyIdSchema.nullable(),
  skillPlanTransactionId: NonEmptyIdSchema.nullable(),
  skillNodeId: NonEmptyIdSchema.nullable(),
  skillId: HumanoidSkillIdSchema.nullable(),
  phase: NonEmptyIdSchema,
  invocation: HumanoidSkillInvocationSchema.nullable(),
  invocationSha256: Sha256Schema.nullable(),
  skillCatalogSha256: Sha256Schema.nullable(),
  observedFrame: z.number().int().nonnegative(),
  observedWorldRevision: z.number().int().nonnegative()
}).strict().superRefine((identity, context) => {
  if (identity.runtimeKind !== "semantic_skill") {
    if (identity.skillId !== null || identity.invocation !== null
      || identity.invocationSha256 !== null) {
      context.addIssue({
        code: "custom",
        path: ["runtimeKind"],
        message: "Only semantic Skill identities may carry a Skill invocation"
      });
    }
    return;
  }
  if (!identity.agentId || !identity.bindingTransactionId
    || !identity.skillId || !identity.invocation || !identity.invocationSha256
    || !identity.skillCatalogSha256) {
    context.addIssue({
      code: "custom",
      message: "A semantic Skill identity requires complete Harness authority"
    });
    return;
  }
  if (identity.invocation.skill !== identity.skillId) {
    context.addIssue({
      code: "custom",
      path: ["skillId"],
      message: "Skill identity must match its typed invocation"
    });
  }
  if (modelPayloadSha256(identity.invocation) !== identity.invocationSha256) {
    context.addIssue({
      code: "custom",
      path: ["invocationSha256"],
      message: "Skill invocation identity does not match its payload"
    });
  }
});

export type HumanoidEmbodiedSkillIdentity = z.infer<
  typeof HumanoidEmbodiedSkillIdentitySchema
>;

const HumanoidEmbodiedEndEffectorCommandSchema = z.object({
  body: NonEmptyIdSchema,
  frame: z.enum(["world", "pelvis", "torso"]),
  position: Vec3Schema,
  tolerance: z.number().finite().positive(),
  orientation: QuaternionSchema.optional(),
  orientationTolerance: z.number().finite().positive().max(Math.PI).optional()
}).strict().superRefine((target, context) => {
  if ((target.orientation === undefined)
    !== (target.orientationTolerance === undefined)) {
    context.addIssue({
      code: "custom",
      message: "End-effector orientation and tolerance must be supplied together"
    });
  }
});

const HumanoidEmbodiedGraspCommandSchema = z.object({
  objectId: NonEmptyIdSchema,
  hand: z.enum(["left", "right"]),
  minimumNormalForceN: z.number().finite().positive(),
  minimumDistinctContactSurfaces: z.number().int().min(1).max(8)
}).strict();

const HumanoidEmbodiedMotionContractSchema = z.object({
  protocol: z.literal("humanoid-embodied-motion-contract-v1"),
  option: HumanoidMotionOptionContractSchema
}).strict();

const HumanoidEmbodiedNavigationContractSchema = z.object({
  protocol: z.literal("humanoid-embodied-navigation-contract-v1"),
  target: Vec3Schema,
  positionTolerance: z.number().finite().positive(),
  heading: z.union([
    z.object({
      type: z.literal("face_point"),
      target: Vec3Schema,
      toleranceRadians: z.number().finite().positive().max(Math.PI)
    }).strict(),
    z.object({
      type: z.literal("yaw"),
      yawRadians: z.number().finite(),
      toleranceRadians: z.number().finite().positive().max(Math.PI)
    }).strict()
  ]).nullable()
}).strict();

export const HumanoidEmbodiedSkillContractSchema = z.discriminatedUnion(
  "protocol",
  [HumanoidEmbodiedMotionContractSchema, HumanoidEmbodiedNavigationContractSchema]
);

export type HumanoidEmbodiedSkillContract = z.infer<
  typeof HumanoidEmbodiedSkillContractSchema
>;

export const HumanoidEmbodiedSkillCallSchema = z.object({
  protocol: z.literal("humanoid-embodied-skill-call-v2"),
  identity: HumanoidEmbodiedSkillIdentitySchema,
  authority: z.object({
    source: z.enum(["agent_harness", "deterministic_runtime"]),
    worldFrame: z.number().int().nonnegative(),
    worldRevision: z.number().int().nonnegative()
  }).strict(),
  window: z.object({
    mode: z.literal("autonomous_closed_loop"),
    replanPolicy: z.literal("event_driven"),
    controlStepSeconds: z.number().finite().positive(),
    maximumSteps: z.number().int().positive(),
    stepIndex: z.number().int().nonnegative(),
    remainingSteps: z.number().int().nonnegative()
  }).strict(),
  requestedCapabilities: z.array(
    z.enum(HUMANOID_LEARNED_POLICY_CAPABILITIES)
  ),
  command: z.object({
    baseTwist: z.object({
      forwardMetersPerSecond: z.number().finite(),
      lateralMetersPerSecond: z.number().finite(),
      yawRadiansPerSecond: z.number().finite()
    }).strict(),
    rootHeightMeters: z.number().finite().positive(),
    leftWristPositionPelvis: Vec3Schema.nullable(),
    rightWristPositionPelvis: Vec3Schema.nullable(),
    endEffectors: z.array(HumanoidEmbodiedEndEffectorCommandSchema),
    grasps: z.array(HumanoidEmbodiedGraspCommandSchema)
  }).strict(),
  contract: HumanoidEmbodiedSkillContractSchema.nullable(),
  safety: z.object({
    authorizedContacts: z.array(HumanoidContactConstraintSchema),
    stopOnFall: z.literal(true),
    stopOnUnauthorizedContact: z.literal(true),
    stopOnContractViolation: z.literal(true)
  }).strict(),
  feedback: z.object({
    mode: z.literal("event_driven"),
    progressDelta: z.number().finite().positive().max(1),
    events: z.array(z.enum([
      "accepted",
      "progress",
      "succeeded",
      "failed",
      "interrupted",
      "environment_changed"
    ])).min(1)
  }).strict()
}).strict().superRefine((call, context) => {
  if (call.window.stepIndex > call.window.maximumSteps
    || call.window.remainingSteps
      !== Math.max(0, call.window.maximumSteps - call.window.stepIndex)) {
    context.addIssue({
      code: "custom",
      path: ["window"],
      message: "Skill execution window progress is inconsistent"
    });
  }
  if (new Set(call.requestedCapabilities).size
    !== call.requestedCapabilities.length) {
    context.addIssue({
      code: "custom",
      path: ["requestedCapabilities"],
      message: "Requested capabilities must be unique"
    });
  }
  if (new Set(call.feedback.events).size !== call.feedback.events.length) {
    context.addIssue({
      code: "custom",
      path: ["feedback", "events"],
      message: "Requested Skill events must be unique"
    });
  }
});

export type HumanoidEmbodiedSkillCall = z.infer<
  typeof HumanoidEmbodiedSkillCallSchema
>;

const HUMANOID_EMBODIED_SKILL_STATES = [
  "accepted",
  "executing",
  "succeeded",
  "failed",
  "uncertain",
  "interrupted"
] as const;

export const HumanoidEmbodiedSkillStatusSchema = z.object({
  protocol: z.literal("humanoid-embodied-skill-status-v1"),
  callId: NonEmptyIdSchema,
  state: z.enum(HUMANOID_EMBODIED_SKILL_STATES),
  progress: z.object({
    elapsedRatio: UnitRateSchema,
    physicalCompletionRatio: UnitRateSchema.nullable(),
    satisfiedPredicateRatio: UnitRateSchema.nullable(),
    stableSteps: z.number().int().nonnegative(),
    requiredStableSteps: z.number().int().positive().nullable()
  }).strict(),
  confidence: z.object({
    value: UnitRateSchema,
    basis: z.literal("observable_contract_evidence")
  }).strict(),
  failure: z.object({
    code: NonEmptyIdSchema,
    detail: z.string().trim().min(1).nullable()
  }).strict().nullable(),
  recoverability: z.enum([
    "not_applicable",
    "retry_skill",
    "replan",
    "switch_policy",
    "safety_stop"
  ]),
  worldFrame: z.number().int().nonnegative(),
  worldRevision: z.number().int().nonnegative(),
  controller: z.object({
    mode: z.enum(["learned_policy", "reference_control", "hybrid_control"]),
    implementation: NonEmptyIdSchema
  }).strict().nullable()
}).strict().superRefine((status, context) => {
  const terminalFailure = status.state === "failed"
    || status.state === "uncertain" || status.state === "interrupted";
  if (terminalFailure !== (status.failure !== null)) {
    context.addIssue({
      code: "custom",
      path: ["failure"],
      message: "Failure evidence must match the Skill state"
    });
  }
  if ((status.state === "accepted" || status.state === "executing"
      || status.state === "succeeded")
    && status.recoverability !== "not_applicable") {
    context.addIssue({
      code: "custom",
      path: ["recoverability"],
      message: "Non-failing Skill states cannot prescribe recovery"
    });
  }
});

export type HumanoidEmbodiedSkillStatus = z.infer<
  typeof HumanoidEmbodiedSkillStatusSchema
>;

export const HumanoidEmbodiedSkillEventSchema = z.object({
  protocol: z.literal("humanoid-embodied-skill-event-v1"),
  sequence: z.number().int().nonnegative(),
  type: z.enum([
    "accepted",
    "progress",
    "succeeded",
    "failed",
    "interrupted",
    "environment_changed"
  ]),
  status: HumanoidEmbodiedSkillStatusSchema
}).strict().superRefine((event, context) => {
  const matchingState = event.type === "progress"
    ? event.status.state === "executing"
    : event.type === "failed"
      ? event.status.state === "failed" || event.status.state === "uncertain"
    : event.type === "environment_changed"
      ? event.status.state === "executing"
        || event.status.state === "interrupted"
        || event.status.state === "uncertain"
      : event.type === event.status.state;
  if (!matchingState) {
    context.addIssue({
      code: "custom",
      path: ["status", "state"],
      message: "Skill event type must match its status state"
    });
  }
});

export type HumanoidEmbodiedSkillEvent = z.infer<
  typeof HumanoidEmbodiedSkillEventSchema
>;

export function legacyHumanoidEmbodiedSkillIdentity(input: {
  callId: string;
  runtimeKind: "navigation" | "station_keeping" | "legacy_motion";
  phase: string;
  observedFrame: number;
  observedWorldRevision: number;
}): HumanoidEmbodiedSkillIdentity {
  return HumanoidEmbodiedSkillIdentitySchema.parse({
    protocol: "humanoid-embodied-skill-identity-v1",
    callId: input.callId,
    runtimeKind: input.runtimeKind,
    agentId: null,
    bindingTransactionId: null,
    skillPlanTransactionId: null,
    skillNodeId: null,
    skillId: null,
    phase: input.phase,
    invocation: null,
    invocationSha256: null,
    skillCatalogSha256: null,
    observedFrame: input.observedFrame,
    observedWorldRevision: input.observedWorldRevision
  });
}
