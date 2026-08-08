import { z } from "zod";
import {
  QuaternionSchema,
  Vec3Schema,
  type Quaternion
} from "../../domain/schema.js";
import { normalizeQuaternion } from "../geometry.js";
import { HUMANOID_BODY_NAMES } from "./model.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  g1HandContactSurfaceHand,
  type G1HandContactSurfaceName
} from "./morphology.js";
import { G1HandCoordinationSchema } from "./hand-coordination.js";
import { HumanoidMotionOptionContractSchema } from "./motion-option.js";
import { HUMANOID_TASK_SPACE_KINEMATIC_SCOPES } from "./task-space-targets.js";

const HumanoidRootVelocitySchema = z.object({
  forward_mps: z.number().finite().describe("身体局部前向速度，单位米每秒"),
  lateral_mps: z.number().finite().describe("身体局部左向速度，单位米每秒")
}).strict();

const HumanoidEndEffectorTargetSchema = z.object({
  position: Vec3Schema.describe("末端目标位置；world 为世界坐标，pelvis 为相对骨盆坐标"),
  frame: z.enum(["world", "pelvis"]),
  tolerance_m: z.number().finite().min(0.01).max(0.12)
    .describe("真实物理跟踪的位置容差，单位米"),
  kinematic_scope: z.enum(HUMANOID_TASK_SPACE_KINEMATIC_SCOPES)
    .describe("arm_only 保持躯干稳定；whole_body_reach 允许受正则约束的腰部冗余伸手")
    .optional(),
  servo_mode: z.enum(["precision", "task_tolerance"])
    .describe("precision 使用生成级精度；task_tolerance 使用任务声明的物理容差")
    .optional(),
  orientation: QuaternionSchema
    .describe("可选末端朝向；与 position 使用相同坐标系")
    .optional(),
  orientation_tolerance_rad: z.number().finite().positive().max(Math.PI)
    .describe("真实物理跟踪的朝向容差，单位弧度")
    .optional()
}).strict().superRefine((target, context) => {
  if ((target.orientation === undefined)
    !== (target.orientation_tolerance_rad === undefined)) {
    context.addIssue({
      code: "custom",
      path: [target.orientation === undefined
        ? "orientation"
        : "orientation_tolerance_rad"],
      message: "End-effector orientation and tolerance must be provided together"
    });
  }
  if (target.orientation) {
    try {
      normalizeQuaternion(target.orientation);
    } catch (error) {
      context.addIssue({
        code: "custom",
        path: ["orientation"],
        message: error instanceof Error ? error.message : "Invalid quaternion"
      });
    }
  }
});

const HumanoidBodyContactConstraintSchema = z.object({
  body: z.enum(HUMANOID_BODY_NAMES)
    .describe("允许接触指定物体的真实 G1 Link"),
  object_id: z.string().trim().min(1)
    .describe("必须与当前 MuJoCo 动态物体 ID 完全一致"),
  required: z.boolean()
    .describe("为 true 时，完整物理预演和真实执行都必须观测到该接触")
}).strict();

const HumanoidBodySolidContactConstraintSchema = z.object({
  body: z.enum(HUMANOID_BODY_NAMES)
    .describe("允许接触指定静态实体的真实 G1 Link"),
  solid_id: z.string().trim().min(1)
    .describe("必须与当前观察中的 MuJoCo solid ID 完全一致"),
  required: z.boolean()
    .describe("为 true 时，完整物理预演和真实执行都必须观测到该接触")
}).strict();

const HumanoidHandSurfaceContactConstraintSchema = z.object({
  hand_surface: z.enum(G1_HAND_CONTACT_SURFACE_NAMES)
    .describe("允许接触指定物体的精确手掌或手指碰撞面"),
  object_id: z.string().trim().min(1)
    .describe("必须与当前 MuJoCo 动态物体 ID 完全一致"),
  required: z.boolean()
    .describe("为 true 时，完整物理预演和真实执行都必须观测到该接触")
}).strict();

const HumanoidHandSurfaceSolidContactConstraintSchema = z.object({
  hand_surface: z.enum(G1_HAND_CONTACT_SURFACE_NAMES)
    .describe("允许接触指定静态实体的精确手掌或手指碰撞面"),
  solid_id: z.string().trim().min(1)
    .describe("必须与当前观察中的 MuJoCo solid ID 完全一致"),
  required: z.boolean()
    .describe("为 true 时，完整物理预演和真实执行都必须观测到该接触")
}).strict();

export const HumanoidContactConstraintSchema = z.union([
  HumanoidBodyContactConstraintSchema,
  HumanoidBodySolidContactConstraintSchema,
  HumanoidHandSurfaceContactConstraintSchema,
  HumanoidHandSurfaceSolidContactConstraintSchema
]);

export type HumanoidContactConstraint = z.infer<
  typeof HumanoidContactConstraintSchema
>;

const HumanoidKeyframeSchema = z.object({
  at_seconds: z.number().finite().nonnegative(),
  root_velocity: HumanoidRootVelocitySchema.nullable().optional(),
  root_yaw_velocity: z.number().finite().describe("根节点偏航角速度，单位弧度每秒").nullable().optional(),
  root_height: z.number().finite().nullable().optional().refine(
    (value) => value == null || (value >= 0.45 && value <= 1.2),
    "root_height must be null when unused or a world-Y pelvis height from 0.45m to 1.2m"
  ),
  root_roll: z.number().finite().nullable().optional(),
  root_pitch: z.number().finite().nullable().optional(),
  torso_yaw: z.number().finite().min(-1.2).max(1.2).nullable().optional(),
  hand_coordination: G1HandCoordinationSchema
    .describe("双手连续协同控制；八个值均为 [0,1]，缺失或 null 表示保持当前手指目标")
    .nullable()
    .optional(),
  left_hand: HumanoidEndEffectorTargetSchema.nullable().optional(),
  right_hand: HumanoidEndEffectorTargetSchema.nullable().optional(),
  left_foot: HumanoidEndEffectorTargetSchema.nullable().optional(),
  right_foot: HumanoidEndEffectorTargetSchema.nullable().optional()
}).strict();

export type HumanoidMotionKeyframe = z.infer<typeof HumanoidKeyframeSchema>;

export const HumanoidMotionPlanSchema = z.object({
  id: z.string().trim().min(1),
  intent: z.string().trim().min(1),
  duration_seconds: z.number().finite().positive().max(30)
    .describe("本次连续运动分块的总时长，最多 30 秒"),
  contact_constraints: z.array(HumanoidContactConstraintSchema)
    .max(16)
    .describe("只授权列出的身体 Link 或精确手部 surface 接触指定动态物体或静态 solid")
    .nullable()
    .optional(),
  keyframes: z.array(HumanoidKeyframeSchema).min(2).max(128)
}).strict().superRefine((plan, context) => {
  const contactKeys = plan.contact_constraints?.map(contactKey) ?? [];
  if (new Set(contactKeys).size !== contactKeys.length) {
    context.addIssue({
      code: "custom",
      path: ["contact_constraints"],
      message: "A humanoid plan cannot repeat the same physical contact constraint"
    });
  }
  if (plan.keyframes[0]?.at_seconds !== 0) {
    context.addIssue({
      code: "custom",
      path: ["keyframes", 0, "at_seconds"],
      message: "The first humanoid keyframe must start at zero"
    });
  }
  for (let index = 1; index < plan.keyframes.length; index += 1) {
    if (plan.keyframes[index]!.at_seconds <= plan.keyframes[index - 1]!.at_seconds) {
      context.addIssue({
        code: "custom",
        path: ["keyframes", index, "at_seconds"],
        message: "Humanoid keyframe times must increase"
      });
    }
  }
  const finalTime = plan.keyframes.at(-1)?.at_seconds;
  if (finalTime !== plan.duration_seconds) {
    context.addIssue({
      code: "custom",
      path: ["duration_seconds"],
      message: "The final humanoid keyframe must equal the plan duration"
    });
  }
  if (plan.keyframes.some((keyframe) => keyframe.hand_coordination != null)
    && plan.keyframes[0]?.hand_coordination == null) {
    context.addIssue({
      code: "custom",
      path: ["keyframes", 0, "hand_coordination"],
      message: "A hand-coordinated motion must declare its initial coordination at time zero"
    });
  }
});

export type HumanoidMotionPlan = z.infer<typeof HumanoidMotionPlanSchema>;

export function humanoidMotionPlanHasPlanarRootMotion(
  plan: Pick<HumanoidMotionPlan, "keyframes">
): boolean {
  return plan.keyframes.some((keyframe) => (
    keyframe.root_velocity != null || keyframe.root_yaw_velocity != null
  ));
}

function duplicateHumanoidMotionCandidateIndexes(
  candidates: readonly HumanoidMotionPlan[]
): Array<{ candidateIndex: number; originalIndex: number }> {
  const firstIndexByContent = new Map<string, number>();
  const duplicates: Array<{ candidateIndex: number; originalIndex: number }> = [];
  candidates.forEach((candidate, candidateIndex) => {
    const content = humanoidMotionCandidateContent(candidate);
    const originalIndex = firstIndexByContent.get(content);
    if (originalIndex === undefined) {
      firstIndexByContent.set(content, candidateIndex);
      return;
    }
    duplicates.push({ candidateIndex, originalIndex });
  });
  return duplicates;
}

export const HumanoidMotionCandidateBatchSchema = z.object({
  objective: z.string().trim().min(1)
    .describe("所有候选共同服务的当前自主目标"),
  termination: HumanoidMotionOptionContractSchema
    .describe("所有候选必须共同达成的可观测物理结果；时长只是最多八秒的执行上界"),
  candidates: z.array(HumanoidMotionPlanSchema).min(1).max(4)
    .describe("按模型偏好排序的 1 至 3 个连续全身动作候选；每个候选都会从同一物理状态完整预演")
}).strict().superRefine((batch, context) => {
  const ids = batch.candidates.map((candidate) => candidate.id);
  if (new Set(ids).size !== ids.length) {
    context.addIssue({
      code: "custom",
      path: ["candidates"],
      message: "A humanoid motion candidate batch cannot repeat a plan identifier"
    });
  }
  for (const duplicate of duplicateHumanoidMotionCandidateIndexes(
    batch.candidates
  )) {
    context.addIssue({
      code: "custom",
      path: ["candidates", duplicate.candidateIndex],
      message: `Candidate motion content duplicates candidate ${duplicate.originalIndex + 1}; id and intent labels do not make a distinct candidate`
    });
  }
  const contactPredicates = batch.termination.predicates.filter((predicate) => (
    predicate.type === "body_contact_object"
      || predicate.type === "hand_contact_object"
      || predicate.type === "hand_contact_object_any"
      || predicate.type === "hand_contact_object_region"
      || predicate.type === "body_contact_solid"
      || predicate.type === "hand_contact_solid"
  ));
  const graspPredicates = batch.termination.predicates.filter((predicate) => (
    predicate.type === "grasp_verified"
  ));
  batch.candidates.forEach((candidate, candidateIndex) => {
    if (candidate.duration_seconds > 8) {
      context.addIssue({
        code: "custom",
        path: ["candidates", candidateIndex, "duration_seconds"],
        message: "Autonomous humanoid options must return control within eight seconds"
      });
    }
    for (const predicate of contactPredicates) {
      const authorizedCount = new Set(candidate.contact_constraints?.flatMap(
        (constraint) => contactPredicateAuthorized(predicate, constraint)
          ? [contactKey(constraint)] : []
      ) ?? []).size;
      const requiredCount = predicate.type === "hand_contact_object_any"
        || predicate.type === "hand_contact_object_region"
        ? predicate.minimum_distinct_surfaces ?? 1
        : 1;
      const authorized = authorizedCount >= requiredCount;
      if (!authorized) {
        context.addIssue({
          code: "custom",
          path: ["candidates", candidateIndex, "contact_constraints"],
          message: missingContactAuthorizationMessage(predicate)
        });
      }
    }
    for (const predicate of graspPredicates) {
      const surfaces = authorizedGraspSurfaces(
        candidate,
        predicate.hand,
        predicate.object_id
      );
      if (surfaces.size < 2) {
        context.addIssue({
          code: "custom",
          path: ["candidates", candidateIndex, "contact_constraints"],
          message: `The shared grasp_verified predicate binds every candidate to hand=${predicate.hand} and object_id=${predicate.object_id}; authorize at least two distinct required hand_object surfaces for that same hand and object (found: ${[...surfaces].sort().join(", ") || "none"})`
        });
      }
    }
  });
});

function missingContactAuthorizationMessage(
  predicate: Extract<HumanoidMotionCandidateBatch["termination"]["predicates"][number], {
    type: "body_contact_object" | "hand_contact_object" | "hand_contact_object_any"
      | "hand_contact_object_region"
      | "body_contact_solid" | "hand_contact_solid";
  }>
): string {
  const contact = predicate.type === "body_contact_object"
    ? `type=body_object body=${predicate.body} object_id=${predicate.object_id}`
    : predicate.type === "hand_contact_object"
      ? `type=hand_object hand_surface=${predicate.hand_surface} object_id=${predicate.object_id}`
      : predicate.type === "hand_contact_object_any"
        ? `type=hand_object_any hand=${predicate.hand} object_id=${predicate.object_id}`
      : predicate.type === "hand_contact_object_region"
        ? `type=hand_object_region hand=${predicate.hand} object_id=${predicate.object_id}`
      : predicate.type === "body_contact_solid"
        ? `type=body_solid body=${predicate.body} solid_id=${predicate.solid_id}`
        : `type=hand_solid hand_surface=${predicate.hand_surface} solid_id=${predicate.solid_id}`;
  return `Every candidate shares the termination predicates and must authorize the required contact: ${contact}`;
}

export interface HumanoidGraspContactAuthorizationFailure {
  candidateIndex: number;
  predicateIndex: number;
  contractSha256: string;
  hand: "left" | "right";
  objectId: string;
  minimumDistinctContactSurfaces: number;
  authorizedContactSurfaces: G1HandContactSurfaceName[];
}

export function humanoidGraspContactAuthorizationFailures(
  batch: HumanoidMotionCandidateBatch,
  minimumDistinctContactSurfaces: (
    graspContractSha256: string
  ) => number
): HumanoidGraspContactAuthorizationFailure[] {
  const failures: HumanoidGraspContactAuthorizationFailure[] = [];
  batch.termination.predicates.forEach((predicate, predicateIndex) => {
    if (predicate.type !== "grasp_verified") return;
    const minimum = minimumDistinctContactSurfaces(
      predicate.grasp_contract_sha256
    );
    if (!Number.isInteger(minimum) || minimum < 2) {
      throw new Error("Trusted grasp policy must require at least two distinct contact surfaces");
    }
    batch.candidates.forEach((candidate, candidateIndex) => {
      const surfaces = [...authorizedGraspSurfaces(
        candidate,
        predicate.hand,
        predicate.object_id
      )].sort();
      if (surfaces.length >= minimum) return;
      failures.push({
        candidateIndex,
        predicateIndex,
        contractSha256: predicate.grasp_contract_sha256,
        hand: predicate.hand,
        objectId: predicate.object_id,
        minimumDistinctContactSurfaces: minimum,
        authorizedContactSurfaces: surfaces
      });
    });
  });
  return failures;
}

function authorizedGraspSurfaces(
  candidate: HumanoidMotionPlan,
  hand: "left" | "right",
  objectId: string
): Set<G1HandContactSurfaceName> {
  return new Set((candidate.contact_constraints ?? []).flatMap((constraint) => (
    "hand_surface" in constraint
      && "object_id" in constraint
      && constraint.object_id === objectId
      && g1HandContactSurfaceHand(constraint.hand_surface) === hand
      ? [constraint.hand_surface]
      : []
  )));
}

function humanoidMotionCandidateContent(candidate: HumanoidMotionPlan): string {
  const contactConstraints = [...(candidate.contact_constraints ?? [])]
    .sort((left, right) => {
      const leftKey = contactKey(left);
      const rightKey = contactKey(right);
      return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
    })
    .map(canonicalContactConstraint);
  return JSON.stringify({
    duration_seconds: candidate.duration_seconds,
    contact_constraints: contactConstraints,
    keyframes: candidate.keyframes.map((keyframe) => ({
      at_seconds: keyframe.at_seconds,
      root_velocity: keyframe.root_velocity ?? null,
      root_yaw_velocity: keyframe.root_yaw_velocity ?? null,
      root_height: keyframe.root_height ?? null,
      root_roll: keyframe.root_roll ?? null,
      root_pitch: keyframe.root_pitch ?? null,
      torso_yaw: keyframe.torso_yaw ?? null,
      hand_coordination: keyframe.hand_coordination ?? null,
      left_hand: canonicalTaskSpaceTarget(keyframe.left_hand),
      right_hand: canonicalTaskSpaceTarget(keyframe.right_hand),
      left_foot: canonicalTaskSpaceTarget(keyframe.left_foot),
      right_foot: canonicalTaskSpaceTarget(keyframe.right_foot)
    }))
  });
}

function canonicalTaskSpaceTarget(
  target: z.infer<typeof HumanoidEndEffectorTargetSchema> | null | undefined
): unknown {
  if (!target) return null;
  return {
    position: target.position,
    frame: target.frame,
    tolerance_m: target.tolerance_m,
    ...(target.kinematic_scope
      ? { kinematic_scope: target.kinematic_scope }
      : {}),
    ...(target.servo_mode ? { servo_mode: target.servo_mode } : {}),
    ...(target.orientation !== undefined
      && target.orientation_tolerance_rad !== undefined
      ? {
          orientation: canonicalQuaternion(target.orientation),
          orientation_tolerance_rad: target.orientation_tolerance_rad
        }
      : {})
  };
}

function canonicalQuaternion(value: Quaternion): Quaternion {
  const normalized = normalizeQuaternion(value);
  const leading = [normalized.w, normalized.x, normalized.y, normalized.z]
    .find((component) => Math.abs(component) > 1e-12) ?? 1;
  return leading < 0
    ? {
        x: -normalized.x,
        y: -normalized.y,
        z: -normalized.z,
        w: -normalized.w
      }
    : normalized;
}

export type HumanoidMotionCandidateBatch = z.infer<
  typeof HumanoidMotionCandidateBatchSchema
>;

function contactKey(constraint: HumanoidContactConstraint): string {
  const target = "object_id" in constraint
    ? constraint.object_id
    : `solid\u0000${constraint.solid_id}`;
  return "body" in constraint
    ? `${constraint.body}\u0000${target}`
    : `hand_surface\u0000${constraint.hand_surface}\u0000${target}`;
}

function contactPredicateAuthorized(
  predicate: Extract<HumanoidMotionCandidateBatch["termination"]["predicates"][number], {
    type: "body_contact_object" | "hand_contact_object" | "hand_contact_object_any"
      | "hand_contact_object_region"
      | "body_contact_solid" | "hand_contact_solid";
  }>,
  constraint: HumanoidContactConstraint
): boolean {
  if (predicate.type === "hand_contact_object_any"
    || predicate.type === "hand_contact_object_region") {
    return "hand_surface" in constraint
      && "object_id" in constraint
      && constraint.hand_surface.startsWith(`${predicate.hand}_`)
      && constraint.object_id === predicate.object_id;
  }
  if (!constraint.required) return false;
  if (predicate.type === "body_contact_object") {
    return "body" in constraint
      && "object_id" in constraint
      && constraint.body === predicate.body
      && constraint.object_id === predicate.object_id;
  }
  if (predicate.type === "hand_contact_object") {
    return "hand_surface" in constraint
      && "object_id" in constraint
      && constraint.hand_surface === predicate.hand_surface
      && constraint.object_id === predicate.object_id;
  }
  if (predicate.type === "body_contact_solid") {
    return "body" in constraint
      && "solid_id" in constraint
      && constraint.body === predicate.body
      && constraint.solid_id === predicate.solid_id;
  }
  return "hand_surface" in constraint
    && "solid_id" in constraint
    && constraint.hand_surface === predicate.hand_surface
    && constraint.solid_id === predicate.solid_id;
}

function canonicalContactConstraint(
  constraint: HumanoidContactConstraint
): Record<string, string | boolean> {
  return {
    ...("body" in constraint
      ? { body: constraint.body }
      : { hand_surface: constraint.hand_surface }),
    ...("object_id" in constraint
      ? { object_id: constraint.object_id }
      : { solid_id: constraint.solid_id }),
    required: constraint.required
  };
}
