import { z } from "zod";
import {
  HumanoidEndEffectorSchema,
  Vec3Schema
} from "../../domain/schema.js";
import { HUMANOID_BODY_NAMES } from "../../world/humanoid/model.js";
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
  "object_near_point",
  "object_in_zone"
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
  zone_id: z.string().trim().min(1).nullable()
    .describe("仅区域谓词使用，否则填 null"),
  target: Vec3Schema.nullable()
    .describe("仅位置谓词使用，否则填 null"),
  tolerance_m: z.number().finite().nonnegative().max(5).nullable()
    .describe("仅位置或区域谓词使用，否则填 null"),
  minimum_normal_force: z.number().finite().positive().nullable()
    .describe("仅接触谓词使用，否则填 null"),
  expected: z.boolean().nullable()
    .describe("仅 object_in_zone 使用，否则填 null")
}).strict().superRefine((predicate, context) => {
  const required = requiredPredicateFields(predicate.type);
  for (const field of PREDICATE_VALUE_FIELDS) {
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
  "zone_id",
  "target",
  "tolerance_m",
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
      tolerance_m: predicate.tolerance_m
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
  if (predicate.type === "object_near_point") {
    return {
      type: predicate.type,
      object_id: predicate.object_id,
      target: predicate.target,
      tolerance_m: predicate.tolerance_m
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
  if (type === "object_near_point") {
    return new Set(["object_id", "target", "tolerance_m"]);
  }
  return new Set(["object_id", "zone_id", "expected", "tolerance_m"]);
}
