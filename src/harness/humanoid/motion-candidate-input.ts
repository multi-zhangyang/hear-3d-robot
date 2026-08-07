import { z } from "zod";
import {
  HumanoidEndEffectorSchema,
  QuaternionSchema,
  Vec3Schema
} from "../../domain/schema.js";
import { G1HandCoordinationSchema } from "../../world/humanoid/hand-coordination.js";
import { HUMANOID_BODY_NAMES } from "../../world/humanoid/model.js";
import { G1_HAND_CONTACT_SURFACE_NAMES } from "../../world/humanoid/morphology.js";
import {
  HumanoidMotionCandidateBatchSchema,
  type HumanoidMotionCandidateBatch
} from "../../world/humanoid/motion-plan.js";

const PositionToleranceSchema = z.number().finite().positive().max(5);
const UnitDirectionSchema = Vec3Schema.refine(
  (value) => Math.abs(Math.hypot(value.x, value.y, value.z) - 1) <= 1e-3,
  "direction must be normalized"
);
const TrackingToleranceSchema = z.number().finite().min(0.01).max(0.12);

const RootNearPointPredicateSchema = z.object({
  type: z.literal("root_near_point"),
  target: Vec3Schema,
  tolerance_m: PositionToleranceSchema
}).strict();

const BodyNearPointPredicateSchema = z.object({
  type: z.literal("body_near_point"),
  body: z.enum(HUMANOID_BODY_NAMES),
  target: Vec3Schema,
  tolerance_m: PositionToleranceSchema
}).strict();

const EndEffectorNearPointPredicateSchema = z.object({
  type: z.literal("end_effector_near_point"),
  end_effector: HumanoidEndEffectorSchema,
  frame: z.enum(["world", "pelvis"]),
  target: Vec3Schema,
  tolerance_m: PositionToleranceSchema
}).strict();

const EndEffectorNearPosePredicateSchema = z.object({
  type: z.literal("end_effector_near_pose"),
  end_effector: HumanoidEndEffectorSchema,
  frame: z.enum(["world", "pelvis"]),
  target: Vec3Schema,
  tolerance_m: PositionToleranceSchema,
  target_orientation: QuaternionSchema,
  orientation_tolerance_rad: z.number().finite().positive().max(Math.PI)
}).strict();

const BodyContactObjectPredicateSchema = z.object({
  type: z.literal("body_contact_object"),
  body: z.enum(HUMANOID_BODY_NAMES),
  object_id: z.string().trim().min(1),
  minimum_normal_force: z.number().finite().positive()
}).strict();

const HandContactObjectPredicateSchema = z.object({
  type: z.literal("hand_contact_object"),
  hand_surface: z.enum(G1_HAND_CONTACT_SURFACE_NAMES),
  object_id: z.string().trim().min(1),
  minimum_normal_force: z.number().finite().positive()
    .describe("该真实掌指碰撞面与物体必须持续达到的最小法向力，单位 N")
}).strict();

const BodyContactSolidPredicateSchema = z.object({
  type: z.literal("body_contact_solid"),
  body: z.enum(HUMANOID_BODY_NAMES),
  solid_id: z.string().trim().min(1),
  minimum_normal_force: z.number().finite().positive()
}).strict();

const HandContactSolidPredicateSchema = z.object({
  type: z.literal("hand_contact_solid"),
  hand_surface: z.enum(G1_HAND_CONTACT_SURFACE_NAMES),
  solid_id: z.string().trim().min(1),
  minimum_normal_force: z.number().finite().positive()
}).strict();

const ObjectNearPointPredicateSchema = z.object({
  type: z.literal("object_near_point"),
  object_id: z.string().trim().min(1),
  target: Vec3Schema,
  tolerance_m: PositionToleranceSchema
}).strict();

const ObjectInZonePredicateSchema = z.object({
  type: z.literal("object_in_zone"),
  object_id: z.string().trim().min(1),
  zone_id: z.string().trim().min(1),
  expected: z.boolean(),
  tolerance_m: z.number().finite().nonnegative().max(5)
}).strict();

const ArticulationStatePredicateSchema = z.object({
  type: z.literal("articulation_state"),
  object_id: z.string().trim().min(1),
  joint_id: z.string().trim().min(1),
  state: z.enum(["open", "closed"]),
  tolerance: z.number().finite().min(0).max(0.49)
}).strict();

const ObjectInsidePredicateSchema = z.object({
  type: z.literal("object_inside"),
  object_id: z.string().trim().min(1),
  container_id: z.string().trim().min(1),
  expected: z.boolean(),
  tolerance_m: z.number().finite().nonnegative().max(1)
}).strict().refine(
  (predicate) => predicate.object_id !== predicate.container_id,
  { path: ["container_id"], message: "object cannot be inside itself" }
);

const ObjectOnPredicateSchema = z.object({
  type: z.literal("object_on"),
  object_id: z.string().trim().min(1),
  support_id: z.string().trim().min(1),
  expected: z.boolean(),
  tolerance_m: z.number().finite().nonnegative().max(1)
}).strict().refine(
  (predicate) => predicate.object_id !== predicate.support_id,
  { path: ["support_id"], message: "object cannot support itself" }
);

const ObjectDisplacedPredicateSchema = z.object({
  type: z.literal("object_displaced"),
  object_id: z.string().trim().min(1),
  origin: Vec3Schema,
  direction_world: UnitDirectionSchema,
  minimum_distance_m: z.number().finite().positive().max(5),
  maximum_lateral_error_m: z.number().finite().nonnegative().max(2)
}).strict();

const ArticulationDisplacedPredicateSchema = z.object({
  type: z.literal("articulation_displaced"),
  object_id: z.string().trim().min(1),
  joint_id: z.string().trim().min(1),
  origin_position: z.number().finite(),
  direction: z.enum(["increasing", "decreasing"]),
  minimum_delta: z.number().finite().positive()
}).strict();

const BalanceStablePredicateSchema = z.object({
  type: z.literal("balance_stable"),
  minimum_support_margin_m: z.number().finite().nonnegative().max(0.3)
}).strict();

const GraspVerifiedPredicateSchema = z.object({
  type: z.literal("grasp_verified"),
  object_id: z.string().trim().min(1),
  hand: z.enum(["left", "right"]),
  grasp_contract_sha256: z.string().regex(/^[a-f0-9]{64}$/)
}).strict();

const ObjectReleasedPredicateSchema = z.object({
  type: z.literal("object_released"),
  object_id: z.string().trim().min(1),
  hand: z.enum(["left", "right"])
}).strict();

const ObjectSettledOnSupportPredicateSchema = z.object({
  type: z.literal("object_settled_on_support"),
  object_id: z.string().trim().min(1)
}).strict();

const ModelMotionPredicateSchema = z.discriminatedUnion("type", [
  RootNearPointPredicateSchema,
  BodyNearPointPredicateSchema,
  EndEffectorNearPointPredicateSchema,
  EndEffectorNearPosePredicateSchema,
  BodyContactObjectPredicateSchema,
  HandContactObjectPredicateSchema,
  BodyContactSolidPredicateSchema,
  HandContactSolidPredicateSchema,
  ObjectNearPointPredicateSchema,
  ObjectInZonePredicateSchema,
  ArticulationStatePredicateSchema,
  ObjectInsidePredicateSchema,
  ObjectOnPredicateSchema,
  ObjectDisplacedPredicateSchema,
  ArticulationDisplacedPredicateSchema,
  BalanceStablePredicateSchema,
  GraspVerifiedPredicateSchema,
  ObjectReleasedPredicateSchema,
  ObjectSettledOnSupportPredicateSchema
]);

const ModelMotionConditionSchema = z.object({
  op: z.enum(["all", "any"]),
  predicate_indexes: z.array(z.number().int().min(0).max(15)).max(16),
  not_predicate_indexes: z.array(z.number().int().min(0).max(15)).max(16)
}).strict().superRefine((condition, context) => {
  const indexes = [
    ...condition.predicate_indexes,
    ...condition.not_predicate_indexes
  ];
  if (indexes.length === 0) {
    context.addIssue({
      code: "custom",
      path: ["predicate_indexes"],
      message: "A motion condition requires at least one predicate index"
    });
  }
  if (new Set(indexes).size !== indexes.length) {
    context.addIssue({
      code: "custom",
      path: ["predicate_indexes"],
      message: "A motion condition cannot repeat a predicate index"
    });
  }
});

const ModelMotionOptionPreconditionSchema = z.object({
    condition: ModelMotionConditionSchema,
    stable_steps: z.number().int().min(1).max(500).nullable()
  }).strict();

const ModelMotionOptionDuringSchema = z.object({
    condition: ModelMotionConditionSchema
  }).strict();

const ModelMotionOptionTerminalSchema = z.object({
    condition: ModelMotionConditionSchema
  }).strict();

const ModelMotionOptionSchema = z.discriminatedUnion("mode", [
  z.object({
    mode: z.literal("all"),
    option_id: z.string().trim().min(1),
    predicates: z.array(ModelMotionPredicateSchema).min(1).max(16),
    stable_steps: z.number().int().min(1).max(500)
  }).strict(),
  z.object({
    mode: z.literal("phased"),
    option_id: z.string().trim().min(1),
    predicates: z.array(ModelMotionPredicateSchema).min(1).max(16),
    stable_steps: z.number().int().min(1).max(500),
    precondition: ModelMotionOptionPreconditionSchema.nullable(),
    during: ModelMotionOptionDuringSchema.nullable(),
    terminal: ModelMotionOptionTerminalSchema
  }).strict()
]);

const ModelBodyObjectContactSchema = z.object({
  body: z.enum(HUMANOID_BODY_NAMES),
  object_id: z.string().trim().min(1),
  required: z.boolean()
}).strict();

const ModelBodySolidContactSchema = z.object({
  body: z.enum(HUMANOID_BODY_NAMES),
  solid_id: z.string().trim().min(1),
  required: z.boolean()
}).strict();

const ModelHandObjectContactSchema = z.object({
  hand_surface: z.enum(G1_HAND_CONTACT_SURFACE_NAMES),
  object_id: z.string().trim().min(1),
  required: z.boolean()
}).strict();

const ModelHandSolidContactSchema = z.object({
  hand_surface: z.enum(G1_HAND_CONTACT_SURFACE_NAMES),
  solid_id: z.string().trim().min(1),
  required: z.boolean()
}).strict();

const ModelContactConstraintSchema = z.discriminatedUnion("type", [
  ModelBodyObjectContactSchema.extend({
    type: z.literal("body_object"),
  }).strict(),
  ModelBodySolidContactSchema.extend({
    type: z.literal("body_solid"),
  }).strict(),
  ModelHandObjectContactSchema.extend({
    type: z.literal("hand_object"),
  }).strict(),
  ModelHandSolidContactSchema.extend({
    type: z.literal("hand_solid"),
  }).strict()
]);

const ModelGroupedContactConstraintsSchema = z.object({
  hand_object: z.array(ModelHandObjectContactSchema).max(16)
}).strict();

const ModelContactConstraintsSchema = z.union([
  z.array(ModelContactConstraintSchema).max(16),
  ModelGroupedContactConstraintsSchema
]);

const ModelMotionChannelSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("root_velocity"),
    forward_mps: z.number().finite(),
    lateral_mps: z.number().finite()
  }).strict(),
  z.object({
    type: z.literal("root_yaw_velocity"),
    radians_per_second: z.number().finite()
  }).strict(),
  z.object({
    type: z.literal("root_height"),
    meters: z.number().finite().min(0.45).max(1.2)
      .describe("骨盆沿世界竖直 Y 轴的绝对高度，单位米；不是世界 X/Z 平面位置")
  }).strict(),
  z.object({
    type: z.literal("root_roll"),
    radians: z.number().finite()
  }).strict(),
  z.object({
    type: z.literal("root_pitch"),
    radians: z.number().finite()
  }).strict(),
  z.object({
    type: z.literal("torso_yaw"),
    radians: z.number().finite().min(-1.2).max(1.2)
  }).strict(),
  z.object({
    type: z.literal("hand_coordination"),
    coordination: G1HandCoordinationSchema
  }).strict(),
  z.object({
    type: z.literal("end_effector_position"),
    end_effector: HumanoidEndEffectorSchema,
    frame: z.enum(["world", "pelvis"]),
    position: Vec3Schema.describe(
      "真实腕或踝 Link 目标，不是物体或掌指接触点；手部接触必须用观察的 surface_from_wrist_world 换算腕目标"
    ),
    tolerance_m: TrackingToleranceSchema
  }).strict(),
  z.object({
    type: z.literal("end_effector_pose"),
    end_effector: HumanoidEndEffectorSchema,
    frame: z.enum(["world", "pelvis"]),
    position: Vec3Schema.describe(
      "真实腕或踝 Link 目标，不是物体或掌指接触点"
    ),
    tolerance_m: TrackingToleranceSchema,
    orientation: QuaternionSchema,
    orientation_tolerance_rad: z.number().finite().positive().max(Math.PI)
  }).strict()
]);

const ModelMotionKeyframeSchema = z.object({
  at_seconds: z.number().finite().nonnegative(),
  channels: z.array(ModelMotionChannelSchema).max(10)
}).strict().superRefine((keyframe, context) => {
  const keys = keyframe.channels.map(channelIdentity);
  if (new Set(keys).size !== keys.length) {
    context.addIssue({
      code: "custom",
      path: ["channels"],
      message: "A keyframe cannot command the same physical channel twice"
    });
  }
});

const ModelMotionCandidateSchema = z.object({
  id: z.string().trim().min(1),
  intent: z.string().trim().min(1),
  duration_seconds: z.number().finite().positive().max(8),
  contacts: ModelContactConstraintsSchema.describe(
    "接触约束可写成带 type 的数组；仅含手-物体约束时也可写成 {hand_object:[...]}，两种形式物理语义相同"
  ),
  keyframes: z.array(ModelMotionKeyframeSchema).min(2).max(32)
}).strict();

const HumanoidMotionCandidateBatchInputShapeSchema = z.object({
  skill_transaction_id: z.string().trim().min(1).nullable().default(null),
  objective: z.string().trim().min(1),
  termination: ModelMotionOptionSchema,
  candidates: z.array(ModelMotionCandidateSchema).min(1).max(3)
}).strict();

export const HumanoidMotionCandidateBatchInputSchema =
  HumanoidMotionCandidateBatchInputShapeSchema.superRefine((input, context) => {
    const result = HumanoidMotionCandidateBatchSchema.safeParse(normalizedBatch(input));
    if (result.success) return;
    for (const issue of result.error.issues.slice(0, 24)) {
      context.addIssue({
        code: "custom",
        path: issue.path,
        message: issue.message
      });
    }
  });

type ModelMotionBatch = z.infer<typeof HumanoidMotionCandidateBatchInputShapeSchema>;
type ModelMotionPredicate = z.infer<typeof ModelMotionPredicateSchema>;
type ModelMotionCondition = z.infer<typeof ModelMotionConditionSchema>;
type ModelMotionChannel = z.infer<typeof ModelMotionChannelSchema>;
type ModelContactConstraint = z.infer<typeof ModelContactConstraintSchema>;

export function normalizeHumanoidMotionCandidateBatchInput(
  input: z.infer<typeof HumanoidMotionCandidateBatchInputSchema>
): HumanoidMotionCandidateBatch {
  return HumanoidMotionCandidateBatchSchema.parse(normalizedBatch(input));
}

function normalizedBatch(input: ModelMotionBatch): unknown {
  return {
    objective: input.objective,
    termination: {
      option_id: input.termination.option_id,
      predicates: input.termination.predicates.map(normalizePredicate),
      stable_steps: input.termination.stable_steps,
      phases: input.termination.mode === "all"
        ? null
        : {
            precondition: input.termination.precondition === null
              ? null
              : {
                  condition: normalizeCondition(
                    input.termination.precondition.condition
                  ),
                  stable_steps: input.termination.precondition.stable_steps
                },
            during: input.termination.during === null
              ? null
              : {
                  condition: normalizeCondition(
                    input.termination.during.condition
                  )
                },
            terminal: {
              condition: normalizeCondition(
                input.termination.terminal.condition
              )
            }
          }
    },
    candidates: input.candidates.map((candidate) => ({
      id: candidate.id,
      intent: candidate.intent,
      duration_seconds: candidate.duration_seconds,
      contact_constraints: contactList(candidate.contacts).map(normalizeContact),
      keyframes: candidate.keyframes.map((keyframe) => {
        const normalized: Record<string, unknown> = {
          at_seconds: keyframe.at_seconds
        };
        for (const channel of keyframe.channels) {
          Object.assign(normalized, normalizeChannel(channel));
        }
        return normalized;
      })
    }))
  };
}

function contactList(
  contacts: ModelMotionBatch["candidates"][number]["contacts"]
): ModelContactConstraint[] {
  if (Array.isArray(contacts)) return contacts;
  return contacts.hand_object.map((contact) => ({
      type: "hand_object" as const,
      ...contact
    }));
}

function normalizePredicate(predicate: ModelMotionPredicate): unknown {
  if (predicate.type !== "end_effector_near_pose") return predicate;
  return {
    type: "end_effector_near_point",
    end_effector: predicate.end_effector,
    frame: predicate.frame,
    target: predicate.target,
    tolerance_m: predicate.tolerance_m,
    target_orientation: predicate.target_orientation,
    orientation_tolerance_rad: predicate.orientation_tolerance_rad
  };
}

function normalizeCondition(condition: ModelMotionCondition): unknown {
  const positive = condition.predicate_indexes.map((predicateIndex) => ({
    op: "predicate" as const,
    predicate_index: predicateIndex
  }));
  const negative = condition.not_predicate_indexes.map((predicateIndex) => ({
    op: "not" as const,
    condition: {
      op: "predicate" as const,
      predicate_index: predicateIndex
    }
  }));
  const conditions = [...positive, ...negative];
  return conditions.length === 1
    ? conditions[0]
    : { op: condition.op, conditions };
}

function normalizeContact(contact: ModelContactConstraint): unknown {
  if (contact.type === "body_object") {
    return {
      body: contact.body,
      object_id: contact.object_id,
      required: contact.required
    };
  }
  if (contact.type === "body_solid") {
    return {
      body: contact.body,
      solid_id: contact.solid_id,
      required: contact.required
    };
  }
  if (contact.type === "hand_object") {
    return {
      hand_surface: contact.hand_surface,
      object_id: contact.object_id,
      required: contact.required
    };
  }
  return {
    hand_surface: contact.hand_surface,
    solid_id: contact.solid_id,
    required: contact.required
  };
}

function normalizeChannel(channel: ModelMotionChannel): Record<string, unknown> {
  if (channel.type === "root_velocity") {
    return {
      root_velocity: {
        forward_mps: channel.forward_mps,
        lateral_mps: channel.lateral_mps
      }
    };
  }
  if (channel.type === "root_yaw_velocity") {
    return { root_yaw_velocity: channel.radians_per_second };
  }
  if (channel.type === "root_height") return { root_height: channel.meters };
  if (channel.type === "root_roll") return { root_roll: channel.radians };
  if (channel.type === "root_pitch") return { root_pitch: channel.radians };
  if (channel.type === "torso_yaw") return { torso_yaw: channel.radians };
  if (channel.type === "hand_coordination") {
    return { hand_coordination: channel.coordination };
  }
  const target = {
    position: channel.position,
    frame: channel.frame,
    tolerance_m: channel.tolerance_m,
    ...(channel.type === "end_effector_pose"
      ? {
          orientation: channel.orientation,
          orientation_tolerance_rad: channel.orientation_tolerance_rad
        }
      : {})
  };
  if (channel.end_effector === "left_wrist") return { left_hand: target };
  if (channel.end_effector === "right_wrist") return { right_hand: target };
  if (channel.end_effector === "left_ankle") return { left_foot: target };
  return { right_foot: target };
}

function channelIdentity(channel: ModelMotionChannel): string {
  return channel.type === "end_effector_position"
    || channel.type === "end_effector_pose"
    ? `end_effector:${channel.end_effector}`
    : channel.type;
}
