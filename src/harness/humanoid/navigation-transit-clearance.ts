import { z } from "zod";
import {
  Vec3Schema,
  type JsonValue,
  type Vec3
} from "../../domain/schema.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import type { HumanoidMotionPlan } from "../../world/humanoid/motion-plan.js";
import type { HumanoidMotionOptionContract } from "../../world/humanoid/motion-option.js";
import {
  HumanoidNavigationCollisionEvidenceSchema,
  type HumanoidNavigationCollisionEvidence
} from "../../world/humanoid/navigation-collision-evidence.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  g1HandContactSurfaceHand,
  type G1HandContactSurfaceName
} from "../../world/humanoid/morphology.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { HumanoidSolidToken } from "../../world/humanoid/solid-observation.js";

export const NavigationTransitClearanceRequirementSchema = z.object({
  sourceTransactionId: z.string().trim().min(1),
  skillTransactionId: z.string().trim().min(1).nullable().default(null),
  blockedAction: z.enum([
    "plan_humanoid_navigation",
    "plan_humanoid_skill"
  ]),
  observedWorldRevision: z.number().int().nonnegative(),
  handSurface: z.enum(G1_HAND_CONTACT_SURFACE_NAMES),
  hand: z.enum(["left", "right"]),
  endEffector: z.enum(["left_wrist", "right_wrist"]),
  collisionTargetId: z.string().trim().min(1),
  collisionTargetKind: z.enum(["object", "solid", "environment"]),
  currentWristWorld: Vec3Schema,
  currentFeetWorld: z.object({
    left: Vec3Schema,
    right: Vec3Schema
  }).strict(),
  collisionTargetWorld: Vec3Schema.nullable(),
  contactPointWorld: Vec3Schema.nullable(),
  separationNormalWorld: Vec3Schema.nullable(),
  separationNormalRobot: Vec3Schema.nullable(),
  normalForceN: z.number().finite().nonnegative().nullable()
}).strict();

export type NavigationTransitClearanceRequirement = z.infer<
  typeof NavigationTransitClearanceRequirementSchema
>;

const HAND_SURFACES = new Set<string>(G1_HAND_CONTACT_SURFACE_NAMES);
const ENVIRONMENT_CONTACT_REASON = /(?:^|;)environment_contact:([^:;]+):([^;]+)/;
const MINIMUM_WRIST_CLEARANCE_DISPLACEMENT_METERS = 0.05;

export function navigationTransitClearanceFromRejection(input: {
  reason: unknown;
  transactionId: string;
  blockedAction?: "plan_humanoid_navigation" | "plan_humanoid_skill";
  worldRevision: number;
  snapshot: HumanoidWorldSnapshot;
  blockingContacts?: unknown;
  solidTokens?: readonly HumanoidSolidToken[];
  skillTransactionId?: string | null;
}): NavigationTransitClearanceRequirement | null {
  const structured = structuredHandCollision(input.blockingContacts);
  const match = typeof input.reason === "string"
    ? ENVIRONMENT_CONTACT_REASON.exec(input.reason)
    : null;
  const handSurface = structured?.surface.name ?? match?.[1];
  const collisionTargetId = structured?.target.id ?? match?.[2]?.trim();
  if (!handSurface || !collisionTargetId || !HAND_SURFACES.has(handSurface)) {
    return null;
  }
  const typedSurface = handSurface as G1HandContactSurfaceName;
  const hand = g1HandContactSurfaceHand(typedSurface);
  const endEffector = hand === "left" ? "left_wrist" : "right_wrist";
  const currentWristWorld = humanoidEndEffectorPosition(
    input.snapshot.robot,
    endEffector,
    "world"
  );
  const leftFootWorld = humanoidEndEffectorPosition(
    input.snapshot.robot,
    "left_ankle",
    "world"
  );
  const rightFootWorld = humanoidEndEffectorPosition(
    input.snapshot.robot,
    "right_ankle",
    "world"
  );
  if (!currentWristWorld || !leftFootWorld || !rightFootWorld) return null;
  const collisionTargetKind = structured?.target.kind
    ?? inferCollisionTargetKind(
      input.snapshot,
      input.solidTokens ?? [],
      collisionTargetId
    );
  return {
    sourceTransactionId: input.transactionId,
    skillTransactionId: input.skillTransactionId ?? null,
    blockedAction: input.blockedAction ?? "plan_humanoid_navigation",
    observedWorldRevision: input.worldRevision,
    handSurface: typedSurface,
    hand,
    endEffector,
    collisionTargetId,
    collisionTargetKind,
    currentWristWorld,
    currentFeetWorld: {
      left: leftFootWorld,
      right: rightFootWorld
    },
    collisionTargetWorld: collisionTargetPosition(
      input.snapshot,
      input.solidTokens ?? [],
      collisionTargetKind,
      collisionTargetId
    ),
    contactPointWorld: structured
      ? { ...structured.contact_point_world }
      : null,
    separationNormalWorld: structured
      ? { ...structured.separation_normal_world }
      : null,
    separationNormalRobot: structured
      ? { ...structured.separation_normal_robot }
      : null,
    normalForceN: structured?.normal_force_n ?? null
  };
}

export function navigationTransitClearanceContext(
  requirement: NavigationTransitClearanceRequirement
): JsonValue {
  return {
    status: "required",
    blocked_action: requirement.blockedAction,
    source_transaction_id: requirement.sourceTransactionId,
    skill_transaction_id: requirement.skillTransactionId,
    observed_world_revision: requirement.observedWorldRevision,
    collision_hand_surface: requirement.handSurface,
    required_end_effector: requirement.endEffector,
    collision_target: {
      kind: requirement.collisionTargetKind,
      id: requirement.collisionTargetKind === "environment"
        ? null
        : requirement.collisionTargetId,
      center_world: requirement.collisionTargetWorld
    },
    collision_target_id: requirement.collisionTargetId,
    collision_target_world: requirement.collisionTargetWorld,
    collision_contact: {
      point_world: requirement.contactPointWorld,
      separation_normal_world: requirement.separationNormalWorld,
      separation_normal_robot: requirement.separationNormalRobot,
      normal_force_n: requirement.normalForceN
    },
    current_wrist_world: requirement.currentWristWorld,
    fixed_foot_world_targets: requirement.currentFeetWorld,
    recovery_options: {
      strategy_selection: "model",
      alternate_navigation: "available",
      whole_body_clearance: {
        root_translation: "forbidden",
        support_foot_motion: "forbidden",
        collision_target_contact: "forbidden",
        future_wrist_world_target: "required",
        minimum_wrist_displacement_m: MINIMUM_WRIST_CLEARANCE_DISPLACEMENT_METERS,
        matching_end_effector_terminal: "required",
        required_candidate_contract: {
          every_keyframe_channels: [{
            type: "end_effector_position",
            end_effector: "left_ankle",
            frame: "world",
            position: requirement.currentFeetWorld.left,
            tolerance_m: 0.015
          }, {
            type: "end_effector_position",
            end_effector: "right_ankle",
            frame: "world",
            position: requirement.currentFeetWorld.right,
            tolerance_m: 0.015
          }],
          future_collision_side_wrist_channel: {
            type: "end_effector_position",
            end_effector: requirement.endEffector,
            frame: "world",
            position: "model_selected_world_point_at_least_0.05m_from_current_wrist",
            tolerance_m: 0.05
          },
          matching_terminal_predicate: {
            type: "end_effector_near_point",
            end_effector: requirement.endEffector,
            frame: "world",
            target: "exactly_copy_the_model_selected_future_wrist_position",
            tolerance_m: 0.05
          }
        }
      }
    },
    automatic_actuation: false
  };
}

export function refreshNavigationTransitClearanceRequirement(input: {
  requirement: NavigationTransitClearanceRequirement;
  worldRevision: number;
  snapshot: HumanoidWorldSnapshot;
  solidTokens: readonly HumanoidSolidToken[];
}): NavigationTransitClearanceRequirement {
  const currentWristWorld = humanoidEndEffectorPosition(
    input.snapshot.robot,
    input.requirement.endEffector,
    "world"
  );
  const leftFootWorld = humanoidEndEffectorPosition(
    input.snapshot.robot,
    "left_ankle",
    "world"
  );
  const rightFootWorld = humanoidEndEffectorPosition(
    input.snapshot.robot,
    "right_ankle",
    "world"
  );
  if (!currentWristWorld || !leftFootWorld || !rightFootWorld) {
    throw new Error("Navigation collision recovery requires current humanoid geometry");
  }
  return NavigationTransitClearanceRequirementSchema.parse({
    ...input.requirement,
    observedWorldRevision: input.worldRevision,
    currentWristWorld,
    currentFeetWorld: {
      left: leftFootWorld,
      right: rightFootWorld
    },
    collisionTargetWorld: collisionTargetPosition(
      input.snapshot,
      input.solidTokens,
      input.requirement.collisionTargetKind,
      input.requirement.collisionTargetId
    ) ?? input.requirement.collisionTargetWorld
  });
}

export function navigationTransitClearanceMotionRejection(
  plans: readonly HumanoidMotionPlan[],
  requirement: NavigationTransitClearanceRequirement,
  termination?: HumanoidMotionOptionContract
): {
  accepted: false;
  code: "navigation_transit_clearance_required";
  channels: [];
  detail: JsonValue;
} | null {
  const failures = plans.flatMap((plan) => {
    const reasons: string[] = [];
    if (plan.keyframes.some((keyframe) => (
      keyframe.root_velocity != null
        && Math.hypot(
          keyframe.root_velocity.forward_mps,
          keyframe.root_velocity.lateral_mps
        ) > 1e-6
    ))) {
      reasons.push("root_translation_present");
    }
    for (const [side, field] of [
      ["left", "left_foot"],
      ["right", "right_foot"]
    ] as const) {
      const fixedTarget = requirement.currentFeetWorld[side];
      if (!plan.keyframes.every((keyframe) => {
        const target = keyframe[field];
        return target?.frame === "world"
          && pointDistance(target.position, fixedTarget) <= 0.015;
      })) {
        reasons.push(`${side}_support_foot_target_missing_or_changed`);
      }
    }
    const wristField = requirement.hand === "left" ? "left_hand" : "right_hand";
    const worldTargets = plan.keyframes.slice(1).flatMap((keyframe) => {
      const target = keyframe[wristField];
      return target?.frame === "world" ? [target.position] : [];
    });
    const displacedTargets = worldTargets.filter((target) => (
      pointDistance(target, requirement.currentWristWorld)
        >= MINIMUM_WRIST_CLEARANCE_DISPLACEMENT_METERS
    ));
    if (worldTargets.length === 0) {
      reasons.push("future_collision_side_wrist_target_missing");
    } else if (displacedTargets.length === 0) {
      reasons.push("future_collision_side_wrist_target_not_displaced");
    }
    if (termination && !termination.predicates.some((predicate) => (
      predicate.type === "end_effector_near_point"
        && predicate.end_effector === requirement.endEffector
        && predicate.frame === "world"
        && displacedTargets.some((target) => pointDistance(
          target,
          predicate.target
        ) <= 1e-6)
    ))) {
      reasons.push("matching_wrist_terminal_missing");
    }
    if ((plan.contact_constraints ?? []).some((constraint) => (
      "object_id" in constraint
        ? constraint.object_id === requirement.collisionTargetId
        : constraint.solid_id === requirement.collisionTargetId
    ))) {
      reasons.push("collision_target_contact_authorized");
    }
    return reasons.length === 0 ? [] : [{ plan_id: plan.id, reasons }];
  });
  if (failures.length === 0) return null;
  return {
    accepted: false,
    code: "navigation_transit_clearance_required",
    channels: [],
    detail: {
      ...navigationTransitClearanceContext(requirement) as Record<string, JsonValue>,
      rejected_candidates: failures,
      recovery: "For every candidate, copy both fixed ankle world targets into every keyframe, choose a displaced collision-side wrist world target, and copy that exact wrist target into the terminal predicate. The model may instead choose a materially different navigation strategy."
    }
  };
}

function pointDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(
    left.x - right.x,
    left.y - right.y,
    left.z - right.z
  );
}

function collisionTargetPosition(
  snapshot: HumanoidWorldSnapshot,
  solidTokens: readonly HumanoidSolidToken[],
  kind: HumanoidNavigationCollisionEvidence["target"]["kind"],
  targetId: string
): Vec3 | null {
  if (kind === "object") {
    const object = snapshot.robot.objects[targetId];
    return object ? { ...object.position } : null;
  }
  if (kind === "solid") {
    const solid = solidTokens.find((candidate) => candidate.id === targetId);
    return solid ? { ...solid.center } : null;
  }
  return null;
}

function structuredHandCollision(
  value: unknown
): HumanoidNavigationCollisionEvidence | null {
  const parsed = HumanoidNavigationCollisionEvidenceSchema.array().safeParse(value);
  if (!parsed.success) return null;
  const collision = parsed.data.find((candidate) => (
    candidate.surface.kind === "hand_surface"
  ));
  return collision?.surface.kind === "hand_surface" ? collision : null;
}

function inferCollisionTargetKind(
  snapshot: HumanoidWorldSnapshot,
  solidTokens: readonly HumanoidSolidToken[],
  targetId: string
): HumanoidNavigationCollisionEvidence["target"]["kind"] {
  if (snapshot.robot.objects[targetId]) return "object";
  if (solidTokens.some((candidate) => candidate.id === targetId)) return "solid";
  return "environment";
}
