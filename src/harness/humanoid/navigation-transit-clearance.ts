import type { HumanoidEndEffector, JsonValue, Vec3 } from "../../domain/schema.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import type { HumanoidMotionPlan } from "../../world/humanoid/motion-plan.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  g1HandContactSurfaceHand,
  type G1HandContactSurfaceName
} from "../../world/humanoid/morphology.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";

export interface NavigationTransitClearanceRequirement {
  sourceTransactionId: string;
  observedWorldRevision: number;
  handSurface: G1HandContactSurfaceName;
  hand: "left" | "right";
  endEffector: Extract<HumanoidEndEffector, "left_wrist" | "right_wrist">;
  collisionTargetId: string;
  currentWristWorld: Vec3;
  collisionTargetWorld: Vec3 | null;
}

const HAND_SURFACES = new Set<string>(G1_HAND_CONTACT_SURFACE_NAMES);
const ENVIRONMENT_CONTACT_REASON = /(?:^|;)environment_contact:([^:;]+):([^;]+)/;

export function navigationTransitClearanceFromRejection(input: {
  reason: unknown;
  transactionId: string;
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
  if (!currentWristWorld) return null;
  return {
    sourceTransactionId: input.transactionId,
    observedWorldRevision: input.worldRevision,
    handSurface: typedSurface,
    hand,
    endEffector,
    collisionTargetId,
    currentWristWorld,
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
    blocked_action: "plan_humanoid_navigation",
    source_transaction_id: requirement.sourceTransactionId,
    observed_world_revision: requirement.observedWorldRevision,
    collision_hand_surface: requirement.handSurface,
    required_end_effector: requirement.endEffector,
    collision_target_id: requirement.collisionTargetId,
    current_wrist_world: requirement.currentWristWorld,
    collision_target_world: requirement.collisionTargetWorld,
    constraints: {
      root_translation: "forbidden",
      collision_target_contact: "forbidden",
      future_wrist_target: "required"
    },
    automatic_actuation: false
  };
}

export function navigationTransitClearanceMotionRejection(
  plans: readonly HumanoidMotionPlan[],
  requirement: NavigationTransitClearanceRequirement
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
    const wristField = requirement.hand === "left" ? "left_hand" : "right_hand";
    if (!plan.keyframes.slice(1).some((keyframe) => keyframe[wristField] != null)) {
      reasons.push("future_collision_side_wrist_target_missing");
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

function collisionTargetPosition(
  snapshot: HumanoidWorldSnapshot,
  targetId: string
): Vec3 | null {
  const object = snapshot.robot.objects[targetId];
  return object ? { ...object.position } : null;
}
