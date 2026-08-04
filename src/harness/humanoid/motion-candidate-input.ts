import { z } from "zod";
import {
  HumanoidEndEffectorSchema,
  QuaternionSchema,
  Vec3Schema
} from "../../domain/schema.js";
import { HUMANOID_BODY_NAMES } from "../../world/humanoid/model.js";
import { G1_HAND_CONTACT_SURFACE_NAMES } from "../../world/humanoid/morphology.js";
import {
  duplicateHumanoidMotionCandidateIndexes,
  HumanoidMotionCandidateBatchSchema,
  HumanoidMotionPlanSchema,
  type HumanoidMotionCandidateBatch
} from "../../world/humanoid/motion-plan.js";

const PredicateTypeSchema = z.enum([
  "root_near_point",
  "body_near_point",
  "end_effector_near_point",
  "body_contact_object",
  "body_contact_solid",
  "hand_contact_solid",
  "object_near_point",
  "object_in_zone",
  "grasp_verified",
  "object_settled_on_support"
]);

const ModelMotionPredicateSchema = z.object({
  type: PredicateTypeSchema,
  body: z.enum(HUMANOID_BODY_NAMES).nullable()
    .describe("仅身体谓词使用，否则填 null"),
  end_effector: HumanoidEndEffectorSchema.nullable()
    .describe("仅末端谓词使用，否则填 null"),
  frame: z.enum(["world", "pelvis"]).nullable()
    .describe("仅末端谓词使用，否则填 null"),
  object_id: z.string().trim().min(1).nullable()
    .describe("仅物体谓词使用，否则填 null"),
  solid_id: z.string().trim().min(1).nullable()
    .describe("仅静态实体接触谓词使用，否则填 null"),
  hand_surface: z.enum(G1_HAND_CONTACT_SURFACE_NAMES).nullable()
    .describe("仅精确手部接触谓词使用，否则填 null"),
  hand: z.enum(["left", "right"]).nullable()
    .describe("仅 grasp_verified 使用，否则填 null"),
  grasp_contract_sha256: z.string().regex(/^[a-f0-9]{64}$/).nullable()
    .describe("仅 grasp_verified 使用，逐字复制当前观察中的权威契约哈希，否则填 null"),
  zone_id: z.string().trim().min(1).nullable()
    .describe("仅区域谓词使用，否则填 null"),
  target: Vec3Schema.nullable()
    .describe("仅位置谓词使用，否则填 null"),
  tolerance_m: z.number().finite().nonnegative().max(5).nullable()
    .describe("仅位置或区域谓词使用，否则填 null"),
  target_orientation: QuaternionSchema.nullable()
    .describe("末端姿态目标可选使用，否则填 null"),
  orientation_tolerance_rad: z.number().finite().positive().max(Math.PI).nullable()
    .describe("末端姿态容差可选使用，否则填 null"),
  minimum_normal_force: z.number().finite().positive().nullable()
    .describe("仅接触谓词使用，否则填 null"),
  expected: z.boolean().nullable()
    .describe("仅 object_in_zone 使用，否则填 null")
}).strict().superRefine((predicate, context) => {
  const required = requiredPredicateFields(predicate.type);
  for (const field of PREDICATE_VALUE_FIELDS) {
    if (field === "target_orientation" || field === "orientation_tolerance_rad") {
      continue;
    }
    const value = predicate[field];
    if (required.has(field) ? value === null : value !== null) {
      context.addIssue({
        code: "custom",
        path: [field],
        message: required.has(field)
          ? `${field} is required for ${predicate.type}`
          : `${field} must be null for ${predicate.type}`
      });
    }
  }
  const hasOrientation = predicate.target_orientation !== null;
  const hasOrientationTolerance = predicate.orientation_tolerance_rad !== null;
  if (predicate.type !== "end_effector_near_point"
    && (hasOrientation || hasOrientationTolerance)) {
    context.addIssue({
      code: "custom",
      path: ["target_orientation"],
      message: "Orientation fields are only valid for end_effector_near_point"
    });
  } else if (hasOrientation !== hasOrientationTolerance) {
    context.addIssue({
      code: "custom",
      path: [hasOrientation ? "orientation_tolerance_rad" : "target_orientation"],
      message: "End-effector orientation and tolerance must be provided together"
    });
  }
  if (predicate.tolerance_m === 0 && predicate.type !== "object_in_zone") {
    context.addIssue({
      code: "custom",
      path: ["tolerance_m"],
      message: `${predicate.type} requires a positive tolerance`
    });
  }
});

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

const ModelMotionOptionPhasesSchema = z.object({
  precondition: z.object({
    condition: ModelMotionConditionSchema,
    stable_steps: z.number().int().min(1).max(500).nullable()
  }).strict().nullable(),
  during: z.object({
    condition: ModelMotionConditionSchema
  }).strict().nullable(),
  terminal: z.object({
    condition: ModelMotionConditionSchema
  }).strict()
}).strict();

const ModelMotionOptionSchema = z.object({
  option_id: z.string().trim().min(1),
  predicates: z.array(ModelMotionPredicateSchema).min(1).max(16),
  stable_steps: z.number().int().min(1).max(500),
  phases: ModelMotionOptionPhasesSchema.nullable()
}).strict();

export const HumanoidMotionCandidateBatchInputSchema = z.object({
  objective: z.string().trim().min(1)
    .describe("所有候选共同服务的当前自主目标"),
  termination: ModelMotionOptionSchema
    .describe("可观测物理结果；条件通过谓词索引组合，不直接编写关节角"),
  candidates: z.array(HumanoidMotionPlanSchema).min(2).max(3)
    .describe("按模型偏好排序的不同连续全身动作候选")
}).strict().superRefine((batch, context) => {
  for (const duplicate of duplicateHumanoidMotionCandidateIndexes(
    batch.candidates
  )) {
    context.addIssue({
      code: "custom",
      path: ["candidates", duplicate.candidateIndex],
      message: `Candidate motion content duplicates candidate ${duplicate.originalIndex + 1}; id and intent labels do not make a distinct candidate`
    });
  }
});

type ModelMotionPredicate = z.infer<typeof ModelMotionPredicateSchema>;
type ModelMotionCondition = z.infer<typeof ModelMotionConditionSchema>;
type ModelMotionOption = z.infer<typeof ModelMotionOptionSchema>;

const PREDICATE_VALUE_FIELDS = [
  "body",
  "end_effector",
  "frame",
  "object_id",
  "solid_id",
  "hand_surface",
  "hand",
  "grasp_contract_sha256",
  "zone_id",
  "target",
  "tolerance_m",
  "target_orientation",
  "orientation_tolerance_rad",
  "minimum_normal_force",
  "expected"
] as const satisfies readonly (keyof ModelMotionPredicate)[];

export function normalizeHumanoidMotionCandidateBatchInput(
  input: z.infer<typeof HumanoidMotionCandidateBatchInputSchema>
): HumanoidMotionCandidateBatch {
  return HumanoidMotionCandidateBatchSchema.parse({
    objective: input.objective,
    termination: normalizeOption(input.termination),
    candidates: input.candidates
  });
}

function normalizeOption(option: ModelMotionOption): unknown {
  return {
    option_id: option.option_id,
    predicates: option.predicates.map(normalizePredicate),
    stable_steps: option.stable_steps,
    phases: option.phases === null
      ? null
      : {
          precondition: option.phases.precondition === null
            ? null
            : {
                condition: normalizeCondition(
                  option.phases.precondition.condition
                ),
                stable_steps: option.phases.precondition.stable_steps
              },
          during: option.phases.during === null
            ? null
            : {
                condition: normalizeCondition(option.phases.during.condition)
              },
          terminal: {
            condition: normalizeCondition(option.phases.terminal.condition)
          }
        }
  };
}

function normalizePredicate(predicate: ModelMotionPredicate): unknown {
  if (predicate.type === "root_near_point") {
    return {
      type: predicate.type,
      target: predicate.target,
      tolerance_m: predicate.tolerance_m
    };
  }
  if (predicate.type === "body_near_point") {
    return {
      type: predicate.type,
      body: predicate.body,
      target: predicate.target,
      tolerance_m: predicate.tolerance_m
    };
  }
  if (predicate.type === "end_effector_near_point") {
    return {
      type: predicate.type,
      end_effector: predicate.end_effector,
      frame: predicate.frame,
      target: predicate.target,
      tolerance_m: predicate.tolerance_m,
      ...(predicate.target_orientation !== null
        && predicate.orientation_tolerance_rad !== null
        ? {
            target_orientation: predicate.target_orientation,
            orientation_tolerance_rad: predicate.orientation_tolerance_rad
          }
        : {})
    };
  }
  if (predicate.type === "body_contact_object") {
    return {
      type: predicate.type,
      body: predicate.body,
      object_id: predicate.object_id,
      minimum_normal_force: predicate.minimum_normal_force
    };
  }
  if (predicate.type === "body_contact_solid") {
    return {
      type: predicate.type,
      body: predicate.body,
      solid_id: predicate.solid_id,
      minimum_normal_force: predicate.minimum_normal_force
    };
  }
  if (predicate.type === "hand_contact_solid") {
    return {
      type: predicate.type,
      hand_surface: predicate.hand_surface,
      solid_id: predicate.solid_id,
      minimum_normal_force: predicate.minimum_normal_force
    };
  }
  if (predicate.type === "object_near_point") {
    return {
      type: predicate.type,
      object_id: predicate.object_id,
      target: predicate.target,
      tolerance_m: predicate.tolerance_m
    };
  }
  if (predicate.type === "grasp_verified") {
    return {
      type: predicate.type,
      object_id: predicate.object_id,
      hand: predicate.hand,
      grasp_contract_sha256: predicate.grasp_contract_sha256
    };
  }
  if (predicate.type === "object_settled_on_support") {
    return {
      type: predicate.type,
      object_id: predicate.object_id
    };
  }
  return {
    type: predicate.type,
    object_id: predicate.object_id,
    zone_id: predicate.zone_id,
    expected: predicate.expected,
    tolerance_m: predicate.tolerance_m
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

function requiredPredicateFields(
  type: ModelMotionPredicate["type"]
): ReadonlySet<(typeof PREDICATE_VALUE_FIELDS)[number]> {
  if (type === "root_near_point") return new Set(["target", "tolerance_m"]);
  if (type === "body_near_point") {
    return new Set(["body", "target", "tolerance_m"]);
  }
  if (type === "end_effector_near_point") {
    return new Set([
      "end_effector",
      "frame",
      "target",
      "tolerance_m"
    ]);
  }
  if (type === "body_contact_object") {
    return new Set(["body", "object_id", "minimum_normal_force"]);
  }
  if (type === "body_contact_solid") {
    return new Set(["body", "solid_id", "minimum_normal_force"]);
  }
  if (type === "hand_contact_solid") {
    return new Set(["hand_surface", "solid_id", "minimum_normal_force"]);
  }
  if (type === "grasp_verified") {
    return new Set(["object_id", "hand", "grasp_contract_sha256"]);
  }
  if (type === "object_settled_on_support") {
    return new Set(["object_id"]);
  }
  if (type === "object_near_point") {
    return new Set(["object_id", "target", "tolerance_m"]);
  }
  return new Set(["object_id", "zone_id", "expected", "tolerance_m"]);
}
