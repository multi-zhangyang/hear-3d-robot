import type { HumanoidEndEffector, JsonValue, Vec3 } from "../../domain/schema.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import type { HumanoidMotionPlan } from "../../world/humanoid/motion-plan.js";
import type { HumanoidMotionOptionContract } from "../../world/humanoid/motion-option.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  g1HandContactSurfaceHand,
  type G1HandContactSurfaceName
} from "../../world/humanoid/morphology.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";

export interface NavigationTransitClearanceRequirement {
  sourceTransactionId: string;
  blockedAction: "plan_humanoid_navigation" | "plan_humanoid_skill";
  observedWorldRevision: number;
  handSurface: G1HandContactSurfaceName;
  hand: "left" | "right";
  endEffector: Extract<HumanoidEndEffector, "left_wrist" | "right_wrist">;
  collisionTargetId: string;
  currentWristWorld: Vec3;
  currentFeetWorld: {
    left: Vec3;
    right: Vec3;
  };
  collisionTargetWorld: Vec3 | null;
}

const HAND_SURFACES = new Set<string>(G1_HAND_CONTACT_SURFACE_NAMES);
const ENVIRONMENT_CONTACT_REASON = /(?:^|;)environment_contact:([^:;]+):([^;]+)/;
const MINIMUM_WRIST_CLEARANCE_DISPLACEMENT_METERS = 0.05;

export function navigationTransitClearanceFromRejection(input: {
  reason: unknown;
  transactionId: string;
  blockedAction?: "plan_humanoid_navigation" | "plan_humanoid_skill";
  worldRevision: number;
  snapshot: HumanoidWorldSnapshot;
}): NavigationTransitClearanceRequirement | null {
  if (typeof input.reason !== "string") return null;
  const match = ENVIRONMENT_CONTACT_REASON.exec(input.reason);
  const handSurface = match?.[1];
  const collisionTargetId = match?.[2]?.trim();
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
  return {
    sourceTransactionId: input.transactionId,
    blockedAction: input.blockedAction ?? "plan_humanoid_navigation",
    observedWorldRevision: input.worldRevision,
    handSurface: typedSurface,
    hand,
    endEffector,
    collisionTargetId,
    currentWristWorld,
    currentFeetWorld: {
      left: leftFootWorld,
      right: rightFootWorld
    },
    collisionTargetWorld: collisionTargetPosition(
      input.snapshot,
      collisionTargetId
    )
  };
}

export function navigationTransitClearanceContext(
  requirement: NavigationTransitClearanceRequirement
): JsonValue {
  return {
    status: "required",
    blocked_action: requirement.blockedAction,
    source_transaction_id: requirement.sourceTransactionId,
    observed_world_revision: requirement.observedWorldRevision,
    collision_hand_surface: requirement.handSurface,
    required_end_effector: requirement.endEffector,
    collision_target_id: requirement.collisionTargetId,
    current_wrist_world: requirement.currentWristWorld,
    fixed_foot_world_targets: requirement.currentFeetWorld,
    collision_target_world: requirement.collisionTargetWorld,
    constraints: {
      root_translation: "forbidden",
      support_foot_motion: "forbidden",
      collision_target_contact: "forbidden",
      future_wrist_world_target: "required",
      minimum_wrist_displacement_m: MINIMUM_WRIST_CLEARANCE_DISPLACEMENT_METERS,
      matching_end_effector_terminal: "required"
    },
    automatic_actuation: false
  };
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
      recovery: "Submit a short arm-clearance posture chosen by the model: keep the base fixed, move the collision-side wrist to a new task-space target, and do not authorize contact with the collision target. Execute and observe that posture before replanning navigation."
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
  targetId: string
): Vec3 | null {
  const object = snapshot.robot.objects[targetId];
  return object ? { ...object.position } : null;
}
