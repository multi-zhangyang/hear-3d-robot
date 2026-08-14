import type {
  Goal,
  GoalPredicate,
  Vec3
} from "../../domain/schema.js";
import type { HumanoidSkillInvocation } from "../../domain/humanoid-skill.js";
import type { HumanoidWorldObservation } from "../../world/humanoid/world.js";

const MINIMUM_PROGRESS_METERS = 0.02;

export type HumanoidSkillGoalAlignment =
  | {
      accepted: true;
      relation: "direct" | "prerequisite" | "recovery" | "safety";
      predicateIndex: number | null;
    }
  | {
      accepted: false;
      reason: string;
    };

export function alignHumanoidSkillToGoal(input: {
  goal: Goal;
  invocation: HumanoidSkillInvocation;
  observation: HumanoidWorldObservation;
  recoveryAuthorized?: boolean;
}): HumanoidSkillGoalAlignment {
  if (input.invocation.skill === "stabilize") {
    return { accepted: true, relation: "safety", predicateIndex: null };
  }
  if (input.invocation.skill === "retreat" && input.recoveryAuthorized) {
    return { accepted: true, relation: "recovery", predicateIndex: null };
  }
  for (const [predicateIndex, predicate] of input.goal.predicates.entries()) {
    const relation = skillPredicateRelation(
      input.invocation,
      predicate,
      input.observation
    );
    if (relation) return { accepted: true, relation, predicateIndex };
  }
  return {
    accepted: false,
    reason: misalignedSkillReason(input.goal, input.invocation)
  };
}

function misalignedSkillReason(
  goal: Goal,
  invocation: HumanoidSkillInvocation
): string {
  if (invocation.skill === "navigate_to_zone") {
    const objectGoal = goal.predicates.find((predicate) => (
      predicate.type === "object_grasped"
        || predicate.type === "object_at"
        || predicate.type === "object_in_zone"
        || predicate.type === "object_placed"
        || predicate.type === "object_inside"
        || predicate.type === "object_on"
    ));
    if (objectGoal && "object_id" in objectGoal) {
      return `navigate_to_zone(${invocation.zone_id}) moves only the robot root and does not establish a manipulation prerequisite for ${objectGoal.type}(${objectGoal.object_id}). Use an object-targeted approach/reach/grasp/lift entry before carrying, or carry_to_zone/place only after a verified carried-object binding exists`;
    }
  }
  return "The selected Skill neither advances nor establishes a prerequisite for any active Goal predicate";
}

function skillPredicateRelation(
  invocation: HumanoidSkillInvocation,
  predicate: GoalPredicate,
  observation: HumanoidWorldObservation
): "direct" | "prerequisite" | undefined {
  if (predicate.type === "robot_at") {
    const target = navigationTarget(invocation, observation);
    return target && positionAdvances(
      observation.robot.rootPosition,
      target,
      predicate.target,
      predicate.tolerance
    ) ? "direct" : undefined;
  }
  if (predicate.type === "robot_in_zone") {
    if (invocation.skill === "navigate_to_zone"
      && invocation.zone_id === predicate.zone_id) {
      return "direct";
    }
    const zone = observedZone(observation, predicate.zone_id);
    const target = navigationTarget(invocation, observation);
    return zone && target && zoneDistance(target, zone, predicate.tolerance)
      < zoneDistance(observation.robot.rootPosition, zone, predicate.tolerance)
        - MINIMUM_PROGRESS_METERS
      ? "direct"
      : undefined;
  }
  if (predicate.type === "block_removed") {
    return invocation.skill === "break_block"
      && invocation.solid_id === predicate.block_id
      ? "direct"
      : undefined;
  }
  if (predicate.type === "object_grasped") {
    if (!objectPrerequisiteSkill(invocation, predicate.object_id)) return undefined;
    return invocationHandMatches(invocation, predicate.hand)
      ? "prerequisite"
      : undefined;
  }
  if (predicate.type === "object_at") {
    if (objectPrerequisiteSkill(invocation, predicate.object_id)) {
      return "prerequisite";
    }
    const target = objectMotionTarget(invocation, predicate.object_id, observation);
    const current = observedObjectPosition(observation, predicate.object_id);
    return target && current && positionAdvances(
      current,
      target,
      predicate.target,
      predicate.tolerance
    ) ? "direct" : undefined;
  }
  if (predicate.type === "object_in_zone"
    || predicate.type === "object_placed") {
    if (objectPrerequisiteSkill(invocation, predicate.object_id)) {
      return "prerequisite";
    }
    if (invocation.skill === "place"
      && invocation.object_id === predicate.object_id
      && invocation.destination.type === "semantic_zone") {
      const expectedInside = predicate.type === "object_placed" || predicate.expected;
      return expectedInside
        && invocation.destination.zone_id === predicate.zone_id
        && invocation.destination.tolerance_m <= predicate.tolerance
        ? "direct" : undefined;
    }
    const zone = observedZone(observation, predicate.zone_id);
    const current = observedObjectPosition(observation, predicate.object_id);
    const target = objectMotionTarget(invocation, predicate.object_id, observation);
    if (!zone || !current || !target) return undefined;
    const currentDistance = zoneDistance(current, zone, predicate.tolerance);
    const targetDistance = zoneDistance(target, zone, predicate.tolerance);
    const expectedInside = predicate.type === "object_placed" || predicate.expected;
    return expectedInside
      ? targetDistance <= 1e-9
        || targetDistance < currentDistance - MINIMUM_PROGRESS_METERS
          ? "direct"
          : undefined
      : targetDistance > currentDistance + MINIMUM_PROGRESS_METERS
        ? "direct"
        : undefined;
  }
  if (predicate.type === "object_inside") {
    if (objectPrerequisiteSkill(invocation, predicate.object_id)) {
      return "prerequisite";
    }
    if (invocation.skill !== "place"
      || invocation.object_id !== predicate.object_id) {
      return undefined;
    }
    const destinationMatches = invocation.destination.type === "container"
      && invocation.destination.object_id === predicate.container_id;
    return destinationMatches === predicate.expected ? "direct" : undefined;
  }
  if (predicate.type === "object_on") {
    if (objectPrerequisiteSkill(invocation, predicate.object_id)) {
      return "prerequisite";
    }
    if (invocation.skill !== "place"
      || invocation.object_id !== predicate.object_id) {
      return undefined;
    }
    const destinationMatches = invocation.destination.type === "support_surface"
      && invocation.destination.object_id === predicate.support_id;
    return destinationMatches === predicate.expected ? "direct" : undefined;
  }
  if (predicate.type === "articulation_state") {
    if (objectPrerequisiteSkill(invocation, predicate.object_id)) {
      return "prerequisite";
    }
    if (!articulationSkillTargets(invocation, predicate.object_id, predicate.joint_id)) {
      return undefined;
    }
    if (invocation.skill === "open") {
      return predicate.state === "open" ? "direct" : undefined;
    }
    if (invocation.skill === "close") {
      return predicate.state === "closed" ? "direct" : undefined;
    }
    return "direct";
  }
  if (predicate.type === "end_effector_at") {
    if (invocation.skill !== "reach") return undefined;
    const expectedHand = predicate.end_effector === "left_wrist"
      ? "left"
      : predicate.end_effector === "right_wrist"
        ? "right"
        : undefined;
    if (!expectedHand || invocation.hand !== expectedHand || predicate.frame !== "world") {
      return undefined;
    }
    const point = observation.interaction.object_world_model.objects
      .find(({ id }) => id === invocation.object_id)
      ?.interaction_points.find(({ id }) => id === invocation.interaction_point_id);
    return point && planarDistance(point.world_position, predicate.target)
      <= predicate.tolerance
      ? "direct"
      : undefined;
  }
  return undefined;
}

function navigationTarget(
  invocation: HumanoidSkillInvocation,
  observation: HumanoidWorldObservation
): Vec3 | undefined {
  if (invocation.skill === "explore") {
    return observation.spatialBelief.frontiers.find(
      ({ id }) => id === invocation.frontier_id
    )?.target;
  }
  if (invocation.skill === "navigate_to_zone") {
    return observedZone(observation, invocation.zone_id)?.center;
  }
  if (invocation.skill === "carry_to_zone") {
    const zone = observedZone(observation, invocation.zone_id);
    const object = observedObjectPosition(observation, invocation.object_id);
    return zone && object ? {
      x: observation.robot.rootPosition.x + zone.center.x - object.x,
      y: observation.robot.rootPosition.y,
      z: observation.robot.rootPosition.z + zone.center.z - object.z
    } : undefined;
  }
  if (invocation.skill === "carry"
    || invocation.skill === "bimanual_carry"
    || invocation.skill === "retreat") {
    return invocation.target;
  }
  return undefined;
}

function objectPrerequisiteSkill(
  invocation: HumanoidSkillInvocation,
  objectId: string
): boolean {
  if (!("object_id" in invocation) || invocation.object_id !== objectId) return false;
  return invocation.skill === "approach"
    || invocation.skill === "reach"
    || invocation.skill === "grasp"
    || invocation.skill === "lift"
    || invocation.skill === "regrasp"
    || invocation.skill === "bimanual_support";
}

function invocationHandMatches(
  invocation: HumanoidSkillInvocation,
  expected: "left" | "right" | "either"
): boolean {
  if (expected === "either") return true;
  if ("hand" in invocation && invocation.hand) return invocation.hand === expected;
  if (invocation.skill === "regrasp") return invocation.to_hand === expected;
  if (invocation.skill === "bimanual_support") return true;
  return false;
}

function articulationSkillTargets(
  invocation: HumanoidSkillInvocation,
  objectId: string,
  jointId: string
): boolean {
  if (!("object_id" in invocation) || invocation.object_id !== objectId) return false;
  if (invocation.skill === "open"
    || invocation.skill === "close"
    || invocation.skill === "turn") {
    return invocation.joint_id === jointId;
  }
  return invocation.skill === "push"
    || invocation.skill === "pull"
    || invocation.skill === "press";
}

function objectMotionTarget(
  invocation: HumanoidSkillInvocation,
  objectId: string,
  observation: HumanoidWorldObservation
): Vec3 | undefined {
  if (!("object_id" in invocation) || invocation.object_id !== objectId) return undefined;
  if (invocation.skill === "carry" || invocation.skill === "bimanual_carry") {
    return invocation.target;
  }
  if (invocation.skill === "carry_to_zone") {
    return observedZone(observation, invocation.zone_id)?.center;
  }
  if (invocation.skill === "place") {
    if (invocation.destination.type === "semantic_zone") {
      return observedZone(observation, invocation.destination.zone_id)?.center;
    }
    if (invocation.destination.type === "world_pose") {
      return invocation.destination.position;
    }
    return observedObjectPosition(observation, invocation.destination.object_id);
  }
  if (invocation.skill === "push" || invocation.skill === "pull") {
    const current = observedObjectPosition(observation, objectId);
    return current ? {
      x: current.x + invocation.direction_world.x * invocation.distance_m,
      y: current.y + invocation.direction_world.y * invocation.distance_m,
      z: current.z + invocation.direction_world.z * invocation.distance_m
    } : undefined;
  }
  return undefined;
}

function observedObjectPosition(
  observation: HumanoidWorldObservation,
  objectId: string
): Vec3 | undefined {
  return observation.interaction.object_world_model.objects.find(
    ({ id }) => id === objectId
  )?.pose.position;
}

function observedZone(
  observation: HumanoidWorldObservation,
  zoneId: string
): HumanoidWorldObservation["interaction"]["zones"][number] | undefined {
  return observation.interaction.zones.find(({ zone_id: id }) => id === zoneId);
}

function positionAdvances(
  current: Vec3,
  target: Vec3,
  goal: Vec3,
  tolerance: number
): boolean {
  const currentDistance = planarDistance(current, goal);
  const targetDistance = planarDistance(target, goal);
  return targetDistance <= tolerance
    || targetDistance < currentDistance - MINIMUM_PROGRESS_METERS;
}

function zoneDistance(
  point: Vec3,
  zone: HumanoidWorldObservation["interaction"]["zones"][number],
  tolerance: number
): number {
  const dx = Math.max(
    0,
    Math.abs(point.x - zone.center.x) - zone.size.x * 0.5 - tolerance
  );
  const dz = Math.max(
    0,
    Math.abs(point.z - zone.center.z) - zone.size.z * 0.5 - tolerance
  );
  return Math.hypot(dx, dz);
}

function planarDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}
