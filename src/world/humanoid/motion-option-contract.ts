import { createHash } from "node:crypto";
import { z } from "zod";
import {
  HumanoidEndEffectorSchema,
  QuaternionSchema,
  Vec3Schema
} from "../../domain/schema.js";
import { normalizeQuaternion } from "../geometry.js";
import { HUMANOID_BODY_NAMES } from "./model.js";
import { G1_HAND_CONTACT_SURFACE_NAMES } from "./morphology.js";

const PositionToleranceSchema = z.number().finite().positive().max(5);

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
  tolerance_m: PositionToleranceSchema,
  target_orientation: QuaternionSchema.optional(),
  orientation_tolerance_rad: z.number().finite().positive().max(Math.PI).optional()
}).strict().superRefine((predicate, context) => {
  if ((predicate.target_orientation === undefined)
    !== (predicate.orientation_tolerance_rad === undefined)) {
    context.addIssue({
      code: "custom",
      path: [predicate.target_orientation === undefined
        ? "target_orientation"
        : "orientation_tolerance_rad"],
      message: "End-effector orientation and tolerance must be provided together"
    });
  }
  if (predicate.target_orientation) {
    try {
      normalizeQuaternion(predicate.target_orientation);
    } catch (error) {
      context.addIssue({
        code: "custom",
        path: ["target_orientation"],
        message: error instanceof Error ? error.message : "Invalid quaternion"
      });
    }
  }
});

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

const HumanoidMotionOptionPredicateSchema = z.discriminatedUnion("type", [
  RootNearPointPredicateSchema,
  BodyNearPointPredicateSchema,
  EndEffectorNearPointPredicateSchema,
  BodyContactObjectPredicateSchema,
  HandContactObjectPredicateSchema,
  BodyContactSolidPredicateSchema,
  HandContactSolidPredicateSchema,
  ObjectNearPointPredicateSchema,
  ObjectInZonePredicateSchema,
  GraspVerifiedPredicateSchema,
  ObjectReleasedPredicateSchema,
  ObjectSettledOnSupportPredicateSchema
]);

export type HumanoidMotionOptionPredicate = z.infer<
  typeof HumanoidMotionOptionPredicateSchema
>;

export type HumanoidMotionOptionCondition =
  | {
      op: "predicate";
      predicate_index: number;
    }
  | {
      op: "all" | "any";
      conditions: HumanoidMotionOptionCondition[];
    }
  | {
      op: "not";
      condition: HumanoidMotionOptionCondition;
    };

const HumanoidMotionOptionConditionSchema:
  z.ZodType<HumanoidMotionOptionCondition, HumanoidMotionOptionCondition> =
    z.lazy(() => z.discriminatedUnion("op", [
      z.object({
        op: z.literal("predicate"),
        predicate_index: z.number().int().min(0).max(15)
      }).strict(),
      z.object({
        op: z.literal("all"),
        conditions: z.array(HumanoidMotionOptionConditionSchema).min(1).max(16)
      }).strict(),
      z.object({
        op: z.literal("any"),
        conditions: z.array(HumanoidMotionOptionConditionSchema).min(1).max(16)
      }).strict(),
      z.object({
        op: z.literal("not"),
        condition: HumanoidMotionOptionConditionSchema
      }).strict()
    ]));

const StableStepsSchema = z.number().int().min(1).max(500);

const HumanoidMotionOptionPhasesSchema = z.object({
  precondition: z.object({
    condition: HumanoidMotionOptionConditionSchema,
    stable_steps: StableStepsSchema.nullable().default(null)
  }).strict().nullable().default(null),
  during: z.object({
    condition: HumanoidMotionOptionConditionSchema
  }).strict().nullable().default(null),
  terminal: z.object({
    condition: HumanoidMotionOptionConditionSchema
  }).strict()
}).strict();

const HumanoidMotionOptionContractShapeSchema = z.object({
  option_id: z.string().trim().min(1),
  predicates: z.array(HumanoidMotionOptionPredicateSchema).min(1).max(16),
  stable_steps: StableStepsSchema,
  phases: HumanoidMotionOptionPhasesSchema.nullable().default(null)
}).strict();

export const HumanoidMotionOptionContractSchema =
  HumanoidMotionOptionContractShapeSchema.superRefine((contract, context) => {
    if (!contract.phases) return;
    const phaseConditions: Array<{
      path: Array<string | number>;
      condition: HumanoidMotionOptionCondition;
    }> = [
      ...(contract.phases.precondition
        ? [{
            path: ["phases", "precondition", "condition"],
            condition: contract.phases.precondition.condition
          }]
        : []),
      ...(contract.phases.during
        ? [{
            path: ["phases", "during", "condition"],
            condition: contract.phases.during.condition
          }]
        : []),
      {
        path: ["phases", "terminal", "condition"],
        condition: contract.phases.terminal.condition
      }
    ];
    for (const phase of phaseConditions) {
      const metrics = humanoidMotionOptionConditionMetrics(phase.condition);
      if (metrics.depth > 8) {
        context.addIssue({
          code: "custom",
          path: phase.path,
          message: "A humanoid option condition cannot exceed eight AST levels"
        });
      }
      if (metrics.nodes > 64) {
        context.addIssue({
          code: "custom",
          path: phase.path,
          message: "A humanoid option condition cannot exceed 64 AST nodes"
        });
      }
      for (const predicateIndex of metrics.predicateIndexes) {
        if (predicateIndex >= contract.predicates.length) {
          context.addIssue({
            code: "custom",
            path: phase.path,
            message: `Condition references missing predicate ${predicateIndex}`
          });
        }
      }
    }
  });

export type HumanoidMotionOptionContract = z.input<
  typeof HumanoidMotionOptionContractSchema
>;

export function humanoidMotionOptionConditionMetrics(
  condition: HumanoidMotionOptionCondition
): { depth: number; nodes: number; predicateIndexes: number[] } {
  if (condition.op === "predicate") {
    return {
      depth: 1,
      nodes: 1,
      predicateIndexes: [condition.predicate_index]
    };
  }
  if (condition.op === "not") {
    const child = humanoidMotionOptionConditionMetrics(condition.condition);
    return {
      depth: child.depth + 1,
      nodes: child.nodes + 1,
      predicateIndexes: child.predicateIndexes
    };
  }
  let depth = 0;
  let nodes = 1;
  const predicateIndexes: number[] = [];
  for (const nested of condition.conditions) {
    const child = humanoidMotionOptionConditionMetrics(nested);
    depth = Math.max(depth, child.depth);
    nodes += child.nodes;
    predicateIndexes.push(...child.predicateIndexes);
  }
  return { depth: depth + 1, nodes, predicateIndexes };
}

export function humanoidMotionOptionContractSha256(
  contract: HumanoidMotionOptionContract
): string {
  const parsed = HumanoidMotionOptionContractSchema.parse(contract);
  const canonical = parsed.phases === null
    ? {
        option_id: parsed.option_id,
        predicates: parsed.predicates,
        stable_steps: parsed.stable_steps
      }
    : parsed;
  return createHash("sha256").update(JSON.stringify(canonical)).digest("hex");
}
