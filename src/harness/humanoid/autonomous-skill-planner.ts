import type { Vec3 } from "../../domain/schema.js";
import type { HumanoidSkillInvocation } from "../../domain/humanoid-skill.js";
import type { HumanoidMotionCandidateBatch } from "../../world/humanoid/motion-plan.js";
import type { HumanoidMotionOptionContract } from "../../world/humanoid/motion-option.js";
import type { HumanoidWorldObservation } from "../../world/humanoid/world.js";
import type { G1HandCoordination } from "../../world/humanoid/hand-coordination.js";
import type { G1HandContactSurfaceName } from "../../world/humanoid/morphology.js";
import type { ActiveHumanoidSkillBinding } from "./skill-binding.js";
import {
  MINIMUM_BLOCK_REMOVAL_NORMAL_FORCE_N,
  MINIMUM_BLOCK_REMOVAL_STABLE_FRAMES
} from "../../domain/scenario-block-removal.js";

type HumanoidMotionOptionPredicate = HumanoidMotionOptionContract["predicates"][number];

export type AutonomousHumanoidSkillPlan =
  | {
      kind: "navigation";
      targets: Array<{
        target: Vec3;
        arrivalHeading: {
          type: "face_point";
          target: Vec3;
          tolerance_radians: number;
        } | null;
        score: number;
      }>;
    }
  | { kind: "motion"; batch: HumanoidMotionCandidateBatch };

export function planAutonomousHumanoidSkill(input: {
  binding: ActiveHumanoidSkillBinding;
  observation: HumanoidWorldObservation;
}): AutonomousHumanoidSkillPlan {
  if (input.binding.phase_authority === "navigation") {
    return navigationSkillPlan(input.binding, input.observation);
  }
  return {
    kind: "motion",
    batch: motionSkillPlan(input.binding, input.observation)
  };
}

function navigationSkillPlan(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation
): Extract<AutonomousHumanoidSkillPlan, { kind: "navigation" }> {
  const invocation = binding.invocation;
  if (invocation.skill === "explore") {
    const selected = observation.spatialBelief.frontiers.find(
      ({ id }) => id === invocation.frontier_id
    );
    if (!selected) throw new Error("Selected exploration frontier is no longer observable");
    return {
      kind: "navigation",
      targets: [selected].map((frontier) => ({
        target: { ...frontier.target },
        arrivalHeading: null,
        score: explorationScore(frontier, invocation.strategy)
      })).sort((left, right) => right.score - left.score)
    };
  }
  if (invocation.skill === "carry"
    || invocation.skill === "bimanual_carry"
    || invocation.skill === "retreat") {
    return {
      kind: "navigation",
      targets: [{ target: { ...invocation.target }, arrivalHeading: null, score: 1 }]
    };
  }
  if (invocation.skill === "break_block") {
    return blockApproachPlan(binding, observation);
  }
  if (invocation.skill !== "approach") {
    throw new Error(`Skill ${invocation.skill} has no navigation implementation`);
  }
  const point = selectedInteractionPoint(binding);
  const around = point?.world_position ?? requiredTargetPosition(binding);
  const directions = approachDirections(point?.approach_direction_world);
  const current = observation.robot.rootPosition;
  return {
    kind: "navigation",
    targets: directions.map((direction, index) => {
      const target = {
        x: around.x - direction.x * invocation.standoff_m,
        y: current.y,
        z: around.z - direction.z * invocation.standoff_m
      };
      return {
        target,
        arrivalHeading: {
          type: "face_point" as const,
          target: { ...around },
          tolerance_radians: 0.12
        },
        score: (point?.approach_direction_world ? 2 : 1)
          - planarDistance(current, target) * 0.05
          - index * 0.01
      };
    }).sort((left, right) => right.score - left.score)
  };
}

function motionSkillPlan(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation
): HumanoidMotionCandidateBatch {
  const invocation = binding.invocation;
  const phase = binding.phase;
  const motion = skillMotionTarget(binding, observation);
  const predicates = skillPredicates(binding, observation, motion);
  const coordination = handCoordinationForSkill(
    observation.handCoordination,
    invocation,
    phase
  );
  const candidates = motionCandidates({
    binding,
    observation,
    motion,
    coordination,
    contactConstraints: skillContactConstraints(binding, observation, predicates)
  });
  return {
    objective: `Execute ${invocation.skill}.${phase} through the generic closed-loop skill solver`,
    termination: {
      option_id: `skill:${binding.transaction_id}:${phase}`,
      predicates,
      stable_steps: invocation.skill === "break_block"
        ? MINIMUM_BLOCK_REMOVAL_STABLE_FRAMES
        : predicates.some(({ type }) => type === "grasp_verified") ? 8 : 4,
      phases: null
    },
    candidates
  };
}

interface SkillMotionTarget {
  hands: Partial<Record<"left" | "right", Vec3>>;
  handSurfaces?: Partial<Record<"left" | "right", G1HandContactSurfaceName>>;
  objectTarget?: Vec3;
  direction?: Vec3;
}

function skillMotionTarget(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation
): SkillMotionTarget {
  const invocation = binding.invocation;
  if (invocation.skill === "stabilize") return { hands: {} };
  if (invocation.skill === "break_block") {
    const contact = solidContactTarget(binding, observation, invocation.hand);
    return {
      hands: { [invocation.hand]: contact.wristTarget },
      handSurfaces: { [invocation.hand]: contact.handSurface },
      direction: contact.inwardDirection
    };
  }
  if (invocation.skill === "lift") {
    const wrist = wristPosition(observation, invocation.hand);
    return {
      hands: { [invocation.hand]: add(wrist, { x: 0, y: invocation.clearance_m, z: 0 }) },
      objectTarget: add(requiredTargetPosition(binding), {
        x: 0,
        y: invocation.clearance_m,
        z: 0
      })
    };
  }
  if (invocation.skill === "place") {
    const object = objectEntry(observation, invocation.object_id);
    const objectTarget = placementTarget(invocation, observation);
    const delta = subtract(objectTarget, object.pose.position);
    const hands = carriedHands(observation, invocation.object_id, invocation.hands)
      .reduce<SkillMotionTarget["hands"]>((targets, hand) => {
        targets[hand] = add(wristPosition(observation, hand), delta);
        return targets;
      }, {});
    return { hands, objectTarget };
  }
  if (invocation.skill === "bimanual_support") {
    return {
      hands: {
        left: wristTargetForPoint(binding, observation, "left", invocation.left_interaction_point_id),
        right: wristTargetForPoint(binding, observation, "right", invocation.right_interaction_point_id)
      }
    };
  }
  if (invocation.skill === "regrasp") {
    return {
      hands: {
        [invocation.to_hand]: wristTargetForPoint(binding, observation, invocation.to_hand)
      }
    };
  }
  if (!("hand" in invocation)) return { hands: {} };
  const contact = wristTargetForPoint(binding, observation, invocation.hand);
  const stroke = skillStroke(binding, invocation);
  return {
    hands: { [invocation.hand]: stroke ? add(contact, stroke) : contact },
    ...(stroke ? { direction: normalize(stroke) } : {})
  };
}

function skillStroke(
  binding: ActiveHumanoidSkillBinding,
  invocation: Extract<HumanoidSkillInvocation, { hand: "left" | "right" }>
): Vec3 | null {
  if (binding.phase !== "apply_force"
    && binding.phase !== "press_stroke"
    && binding.phase !== "actuate_joint") return null;
  if (invocation.skill === "push" || invocation.skill === "pull") {
    return scale(invocation.direction_world, invocation.distance_m);
  }
  if (invocation.skill === "press") {
    const axis = binding.target_articulation?.axis_world ?? { x: 0, y: -1, z: 0 };
    return scale(normalize(axis), invocation.travel_m);
  }
  if (invocation.skill === "open" || invocation.skill === "close") {
    const articulation = binding.target_articulation;
    const point = selectedInteractionPoint(binding);
    if (!articulation || !point) return null;
    const sign = invocation.skill === "open" ? 1 : -1;
    if (articulation.type === "slide") {
      return scale(normalize(articulation.axis_world), sign * 0.18);
    }
    const radial = subtract(point.world_position, articulation.anchor_world);
    return scale(normalize(cross(articulation.axis_world, radial)), sign * 0.2);
  }
  return null;
}

function skillPredicates(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  motion: SkillMotionTarget
): HumanoidMotionOptionPredicate[] {
  const invocation = binding.invocation;
  const phase = binding.phase;
  if (invocation.skill === "stabilize") {
    return [{
      type: "balance_stable",
      minimum_support_margin_m: invocation.minimum_support_margin_m
    }];
  }
  if (invocation.skill === "break_block") {
    const handSurface = motion.handSurfaces?.[invocation.hand];
    if (!handSurface) throw new Error("Block contact requires an observed hand surface");
    return [{
      type: "hand_contact_solid",
      hand_surface: handSurface,
      solid_id: invocation.solid_id,
      minimum_normal_force: MINIMUM_BLOCK_REMOVAL_NORMAL_FORCE_N
    }];
  }
  if (invocation.skill === "grasp"
    || invocation.skill === "bimanual_support"
    || invocation.skill === "regrasp" && phase === "transfer_grasp") {
    const hands = invocation.skill === "bimanual_support"
      ? ["left", "right"] as const
      : [invocation.skill === "regrasp" ? invocation.to_hand : invocation.hand];
    const predicates: HumanoidMotionOptionPredicate[] = hands.map((hand) => ({
      type: "grasp_verified",
      object_id: invocation.object_id,
      hand,
      grasp_contract_sha256: observation.interaction.grasp_authority.contract_sha256
    }));
    if (invocation.skill === "regrasp") {
      predicates.push({
        type: "object_released",
        object_id: invocation.object_id,
        hand: invocation.from_hand
      });
    }
    return predicates;
  }
  if (invocation.skill === "lift") {
    return [{
      type: "grasp_verified",
      object_id: invocation.object_id,
      hand: invocation.hand,
      grasp_contract_sha256: observation.interaction.grasp_authority.contract_sha256
    }, {
      type: "object_near_point",
      object_id: invocation.object_id,
      target: motion.objectTarget!,
      tolerance_m: Math.max(0.04, invocation.clearance_m * 0.25)
    }];
  }
  if (invocation.skill === "place" && phase === "settle_and_release") {
    return placePredicates(invocation, motion.objectTarget!);
  }
  if ((invocation.skill === "open" || invocation.skill === "close")
    && phase === "actuate_joint") {
    return [{
      type: "articulation_state",
      object_id: invocation.object_id,
      joint_id: invocation.joint_id,
      state: invocation.skill === "open" ? "open" : "closed",
      tolerance: invocation.skill === "open"
        ? 1 - invocation.minimum_open_fraction
        : invocation.maximum_open_fraction
    }];
  }
  if ((invocation.skill === "push" || invocation.skill === "pull")
    && phase === "apply_force") {
    const articulation = binding.target_articulation;
    if (articulation?.position != null) {
      return [articulationDisplacementPredicate(binding, invocation.direction_world, invocation.distance_m)];
    }
    return [{
      type: "object_displaced",
      object_id: invocation.object_id,
      origin: requiredTargetPosition(binding),
      direction_world: invocation.direction_world,
      minimum_distance_m: invocation.distance_m,
      maximum_lateral_error_m: Math.max(0.05, invocation.distance_m * 0.25)
    }];
  }
  if (invocation.skill === "press") {
    return [articulationDisplacementPredicate(
      binding,
      motion.direction ?? { x: 0, y: -1, z: 0 },
      invocation.travel_m
    )];
  }
  const hand = "hand" in invocation ? invocation.hand
    : invocation.skill === "regrasp" ? invocation.to_hand : null;
  if (hand && motion.hands[hand]) {
    return [{
      type: "end_effector_near_point",
      end_effector: hand === "left" ? "left_wrist" : "right_wrist",
      frame: "world",
      target: motion.hands[hand]!,
      tolerance_m: invocation.skill === "reach" ? invocation.tolerance_m : 0.05
    }];
  }
  if (motion.objectTarget && "object_id" in invocation) {
    return [{
      type: "object_near_point",
      object_id: invocation.object_id,
      target: motion.objectTarget,
      tolerance_m: 0.08
    }];
  }
  throw new Error(`Skill ${invocation.skill}.${phase} has no observable terminal`);
}

function articulationDisplacementPredicate(
  binding: ActiveHumanoidSkillBinding,
  direction: Vec3,
  travel: number
): HumanoidMotionOptionPredicate {
  const articulation = binding.target_articulation;
  const point = selectedInteractionPoint(binding);
  const objectId = "object_id" in binding.invocation
    ? binding.invocation.object_id
    : null;
  if (!articulation || articulation.position == null || !point || !objectId) {
    throw new Error("Articulation motion requires live joint and interaction-point geometry");
  }
  if (articulation.type === "slide") {
    return {
      type: "articulation_displaced",
      object_id: objectId,
      joint_id: articulation.joint_id,
      origin_position: articulation.position,
      direction: dot(direction, articulation.axis_world) >= 0 ? "increasing" : "decreasing",
      minimum_delta: travel
    };
  }
  const tangent = cross(
    articulation.axis_world,
    subtract(point.world_position, articulation.anchor_world)
  );
  const radius = Math.hypot(tangent.x, tangent.y, tangent.z);
  if (radius <= 1e-6) throw new Error("Articulation interaction point lies on its hinge axis");
  return {
    type: "articulation_displaced",
    object_id: objectId,
    joint_id: articulation.joint_id,
    origin_position: articulation.position,
    direction: dot(direction, tangent) >= 0 ? "increasing" : "decreasing",
    minimum_delta: travel / radius
  };
}

function motionCandidates(input: {
  binding: ActiveHumanoidSkillBinding;
  observation: HumanoidWorldObservation;
  motion: SkillMotionTarget;
  coordination: G1HandCoordination | null;
  contactConstraints: HumanoidMotionCandidateBatch["candidates"][number]["contact_constraints"];
}): HumanoidMotionCandidateBatch["candidates"] {
  if (input.binding.invocation.skill === "break_block") {
    return blockContactCandidates(input);
  }
  const hands = Object.entries(input.motion.hands) as Array<["left" | "right", Vec3]>;
  const current = Object.fromEntries(hands.map(([hand]) => [
    hand,
    wristPosition(input.observation, hand)
  ])) as Partial<Record<"left" | "right", Vec3>>;
  const variants = hands.length === 0 ? [0] : [0, 0.06, -0.06];
  return variants.map((lateralOffset, index) => {
    const duration = hands.length === 0 ? 1 : 2.4;
    const middleHands = Object.fromEntries(hands.map(([hand, target]) => {
      const start = current[hand]!;
      return [hand, {
        x: (start.x + target.x) / 2 + lateralOffset,
        y: Math.max(start.y, target.y) + (index === 0 ? 0.04 : 0.08),
        z: (start.z + target.z) / 2
      }];
    })) as Partial<Record<"left" | "right", Vec3>>;
    return {
      id: `solver:${input.binding.transaction_id}:${input.binding.phase}:${index + 1}`,
      intent: index === 0 ? "minimum-displacement closed-loop path"
        : index === 1 ? "elevated positive-lateral clearance path"
          : "elevated negative-lateral clearance path",
      duration_seconds: duration,
      contact_constraints: input.contactConstraints ?? [],
      keyframes: [
        motionKeyframe(0, current, input.coordination ? input.observation.handCoordination : null),
        ...(hands.length === 0 ? [] : [
          motionKeyframe(duration * 0.55, middleHands, null)
        ]),
        motionKeyframe(duration, input.motion.hands, input.coordination)
      ]
    };
  });
}

function blockContactCandidates(input: {
  binding: ActiveHumanoidSkillBinding;
  observation: HumanoidWorldObservation;
  motion: SkillMotionTarget;
  coordination: G1HandCoordination | null;
  contactConstraints: HumanoidMotionCandidateBatch["candidates"][number]["contact_constraints"];
}): HumanoidMotionCandidateBatch["candidates"] {
  const invocation = input.binding.invocation;
  if (invocation.skill !== "break_block" || !input.motion.direction) {
    throw new Error("Block contact strategy is unavailable");
  }
  const hand = invocation.hand;
  const start = wristPosition(input.observation, hand);
  const target = input.motion.hands[hand];
  if (!target) throw new Error("Block contact wrist target is unavailable");
  const strokeDistance = distance(start, target);
  const strike = invocation.strategy === "strike";
  const duration = strike
    ? clamp(0.9 + strokeDistance * 1.2, 1.1, 1.8)
    : clamp(1.8 + strokeDistance * 1.6, 2, 3.2);
  const lateralAxes = [0, 0.05, -0.05];
  return lateralAxes.map((lateral, index) => {
    const clearance = { x: lateral, y: index === 0 ? 0.03 : 0.07, z: 0 };
    const preparation = strike
      ? add(add(start, scale(input.motion.direction!, -Math.min(
          0.16,
          Math.max(0.06, strokeDistance * 0.35)
        ))), clearance)
      : add({
          x: (start.x + target.x) / 2,
          y: (start.y + target.y) / 2,
          z: (start.z + target.z) / 2
        }, clearance);
    return {
      id: `solver:${input.binding.transaction_id}:${input.binding.phase}:${index + 1}`,
      intent: strike
        ? index === 0 ? "direct acceleration and sustained block contact"
          : "clearance-biased acceleration and sustained block contact"
        : index === 0 ? "progressive force-controlled block contact"
          : "clearance-biased progressive block contact",
      duration_seconds: duration,
      contact_constraints: input.contactConstraints ?? [],
      keyframes: [
        motionKeyframe(0, { [hand]: start }, null),
        motionKeyframe(duration * (strike ? 0.32 : 0.48), {
          [hand]: preparation
        }, null),
        motionKeyframe(duration * (strike ? 0.68 : 0.78), {
          [hand]: target
        }, null),
        motionKeyframe(duration, { [hand]: target }, null)
      ]
    };
  });
}

function motionKeyframe(
  atSeconds: number,
  hands: SkillMotionTarget["hands"],
  coordination: G1HandCoordination | null
): HumanoidMotionCandidateBatch["candidates"][number]["keyframes"][number] {
  return {
    at_seconds: atSeconds,
    ...(coordination ? { hand_coordination: coordination } : {}),
    ...(hands.left ? {
      left_hand: { position: hands.left, frame: "world", tolerance_m: 0.04 }
    } : {}),
    ...(hands.right ? {
      right_hand: { position: hands.right, frame: "world", tolerance_m: 0.04 }
    } : {})
  };
}

function skillContactConstraints(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  predicates: readonly HumanoidMotionOptionPredicate[]
): NonNullable<HumanoidMotionCandidateBatch["candidates"][number]["contact_constraints"]> {
  if (binding.invocation.skill === "break_block") {
    const solidId = binding.invocation.solid_id;
    const surface = predicates.find((predicate) => (
      predicate.type === "hand_contact_solid"
        && predicate.solid_id === solidId
    ));
    if (!surface || surface.type !== "hand_contact_solid") return [];
    return [{
      hand_surface: surface.hand_surface,
      solid_id: solidId,
      required: true
    }];
  }
  const objectId = "object_id" in binding.invocation
    ? binding.invocation.object_id
    : null;
  if (!objectId) return [];
  const hands = new Set(predicates.flatMap((predicate) => (
    predicate.type === "grasp_verified" ? [predicate.hand] : []
  )));
  if (hands.size === 0 && "hand" in binding.invocation
    && ["push", "pull", "press", "open", "close"].includes(binding.invocation.skill)) {
    hands.add(binding.invocation.hand);
  }
  return [...hands].flatMap((hand) => graspSurfaces(observation, hand)
    .slice(0, 3)
    .map((handSurface) => ({
      hand_surface: handSurface,
      object_id: objectId,
      required: predicates.some((predicate) => predicate.type === "grasp_verified")
    })));
}

function blockApproachPlan(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation
): Extract<AutonomousHumanoidSkillPlan, { kind: "navigation" }> {
  const invocation = binding.invocation;
  if (invocation.skill !== "break_block" || !binding.target_solid) {
    throw new Error("Block approach requires a bound visible block");
  }
  const solid = binding.target_solid;
  const current = observation.robot.rootPosition;
  const directions = approachDirections({
    x: current.x - solid.center.x,
    y: 0,
    z: current.z - solid.center.z
  });
  return {
    kind: "navigation",
    targets: directions.map((direction, index) => {
      const halfExtent = Math.abs(direction.x) * solid.size.x / 2
        + Math.abs(direction.z) * solid.size.z / 2;
      const distance = halfExtent + invocation.approach_clearance_m;
      const target = {
        x: solid.center.x + direction.x * distance,
        y: current.y,
        z: solid.center.z + direction.z * distance
      };
      return {
        target,
        arrivalHeading: {
          type: "face_point" as const,
          target: { ...solid.center },
          tolerance_radians: 0.12
        },
        score: 1 - planarDistance(current, target) * 0.05 - index * 0.01
      };
    }).sort((left, right) => right.score - left.score)
  };
}

function solidContactTarget(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  hand: "left" | "right"
): {
  wristTarget: Vec3;
  handSurface: G1HandContactSurfaceName;
  inwardDirection: Vec3;
} {
  const solid = binding.target_solid;
  if (!solid) throw new Error("Block contact requires a bound visible block");
  const wrist = wristPosition(observation, hand);
  const half = scale(solid.size, 0.5);
  const relative = subtract(wrist, solid.center);
  const normalized = [
    { axis: "x" as const, value: relative.x / Math.max(half.x, 1e-6) },
    { axis: "y" as const, value: relative.y / Math.max(half.y, 1e-6) },
    { axis: "z" as const, value: relative.z / Math.max(half.z, 1e-6) }
  ].sort((left, right) => Math.abs(right.value) - Math.abs(left.value))[0]!;
  const surfacePoint = {
    x: clamp(wrist.x, solid.center.x - half.x, solid.center.x + half.x),
    y: clamp(wrist.y, solid.center.y - half.y, solid.center.y + half.y),
    z: clamp(wrist.z, solid.center.z - half.z, solid.center.z + half.z)
  };
  surfacePoint[normalized.axis] = solid.center[normalized.axis]
    + Math.sign(normalized.value || 1) * half[normalized.axis];
  const inwardDirection = normalize(subtract(solid.center, surfacePoint));
  const penetration = Math.min(0.04, Math.max(0.012, Math.min(
    solid.size.x,
    solid.size.y,
    solid.size.z
  ) * (binding.invocation.skill === "break_block"
    && binding.invocation.strategy === "strike" ? 0.08 : 0.04)));
  const desiredSurface = add(surfacePoint, scale(inwardDirection, penetration));
  const surfaces = observation.handSurfaces.filter((surface) => surface.hand === hand);
  const best = surfaces.sort((left, right) => (
    distance(add(left.wristWorldPosition, left.surfaceFromWristWorld), desiredSurface)
      - distance(add(right.wristWorldPosition, right.surfaceFromWristWorld), desiredSurface)
  ))[0];
  if (!best) throw new Error(`No observed ${hand} hand contact surface`);
  return {
    wristTarget: subtract(desiredSurface, best.surfaceFromWristWorld),
    handSurface: best.handSurface,
    inwardDirection
  };
}

function handCoordinationForSkill(
  current: G1HandCoordination,
  invocation: HumanoidSkillInvocation,
  phase: string
): G1HandCoordination | null {
  const result = structuredClone(current);
  const close = (hand: "left" | "right") => {
    result[hand] = {
      thumb_opposition: 0.85,
      thumb_curl: 0.72,
      index_curl: 0.78,
      middle_curl: 0.8
    };
  };
  const open = (hand: "left" | "right") => {
    result[hand] = {
      thumb_opposition: 0.15,
      thumb_curl: 0.08,
      index_curl: 0.05,
      middle_curl: 0.05
    };
  };
  if (invocation.skill === "grasp") close(invocation.hand);
  else if (invocation.skill === "bimanual_support") {
    close("left");
    close("right");
  } else if (invocation.skill === "regrasp" && phase === "transfer_grasp") {
    close(invocation.to_hand);
    open(invocation.from_hand);
  } else if (invocation.skill === "place" && phase === "settle_and_release") {
    if (invocation.hands === "both") {
      open("left");
      open("right");
    } else open(invocation.hands);
  } else return null;
  return result;
}

function placePredicates(
  invocation: Extract<HumanoidSkillInvocation, { skill: "place" }>,
  objectTarget: Vec3
): HumanoidMotionOptionPredicate[] {
  const hands = invocation.hands === "both"
    ? ["left", "right"] as const : [invocation.hands];
  const released: HumanoidMotionOptionPredicate[] = hands.map((hand) => ({
    type: "object_released",
    object_id: invocation.object_id,
    hand
  }));
  const relation: HumanoidMotionOptionPredicate = invocation.destination.type === "container"
    ? {
        type: "object_inside",
        object_id: invocation.object_id,
        container_id: invocation.destination.object_id,
        expected: true,
        tolerance_m: 0.04
      }
    : invocation.destination.type === "support_surface"
      ? {
          type: "object_on",
          object_id: invocation.object_id,
          support_id: invocation.destination.object_id,
          expected: true,
          tolerance_m: 0.04
        }
      : {
          type: "object_near_point",
          object_id: invocation.object_id,
          target: objectTarget,
          tolerance_m: invocation.destination.type === "world_pose"
            ? invocation.destination.position_tolerance_m : 0.05
        };
  return [...released, {
    type: "object_settled_on_support",
    object_id: invocation.object_id
  }, relation];
}

function placementTarget(
  invocation: Extract<HumanoidSkillInvocation, { skill: "place" }>,
  observation: HumanoidWorldObservation
): Vec3 {
  if (invocation.destination.type === "world_pose") {
    return { ...invocation.destination.position };
  }
  const destination = objectEntry(observation, invocation.destination.object_id);
  if (invocation.destination.type === "slot") {
    const interactionPointId = invocation.destination.interaction_point_id;
    const point = destination.interaction_points.find(
      ({ id }) => id === interactionPointId
    );
    if (!point) throw new Error("Placement slot interaction point is unavailable");
    return { ...point.world_position };
  }
  if ((invocation.destination.type === "support_surface"
      || invocation.destination.type === "container")
    && invocation.destination.local_target) {
    return add(destination.pose.position, invocation.destination.local_target);
  }
  return {
    x: destination.pose.position.x,
    y: invocation.destination.type === "support_surface"
      ? destination.pose.position.y + destination.size.y / 2
      : destination.pose.position.y,
    z: destination.pose.position.z
  };
}

function wristTargetForPoint(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  hand: "left" | "right",
  requestedPointId?: string
): Vec3 {
  const point = requestedPointId
    ? binding.eligible_interaction_points.find(({ id }) => id === requestedPointId)
    : selectedInteractionPoint(binding);
  if (!point) throw new Error("Skill motion requires an eligible interaction point");
  const surfaces = observation.handSurfaces.filter((surface) => surface.hand === hand);
  const best = surfaces.sort((left, right) => (
    distance(add(left.wristWorldPosition, left.surfaceFromWristWorld), point.world_position)
      - distance(add(right.wristWorldPosition, right.surfaceFromWristWorld), point.world_position)
  ))[0];
  if (!best) throw new Error(`No observed ${hand} hand contact surface`);
  return subtract(point.world_position, best.surfaceFromWristWorld);
}

function graspSurfaces(
  observation: HumanoidWorldObservation,
  hand: "left" | "right"
): G1HandContactSurfaceName[] {
  const preferred = ["palm", "index", "middle", "thumb"];
  return observation.handSurfaces
    .filter((surface) => surface.hand === hand)
    .sort((left, right) => surfaceRank(left.handSurface, preferred)
      - surfaceRank(right.handSurface, preferred))
    .map(({ handSurface }) => handSurface);
}

function selectedInteractionPoint(binding: ActiveHumanoidSkillBinding) {
  const requested = "interaction_point_id" in binding.invocation
    ? binding.invocation.interaction_point_id : null;
  return requested
    ? binding.eligible_interaction_points.find(({ id }) => id === requested)
    : binding.eligible_interaction_points[0];
}

function carriedHands(
  observation: HumanoidWorldObservation,
  objectId: string,
  requested: "left" | "right" | "both"
): Array<"left" | "right"> {
  const bound = observation.interaction.carrying.bindings
    .filter(({ object_id }) => object_id === objectId)
    .map(({ hand }) => hand);
  const desired = requested === "both" ? ["left", "right"] as const : [requested];
  const hands = desired.filter((hand) => bound.includes(hand));
  if (hands.length !== desired.length) throw new Error("Requested carrying hands are not bound");
  return [...hands];
}

function objectEntry(observation: HumanoidWorldObservation, objectId: string) {
  const object = observation.interaction.object_world_model.objects.find(
    ({ id }) => id === objectId
  );
  if (!object || object.status !== "visible") {
    throw new Error(`Object is unavailable to the skill solver: ${objectId}`);
  }
  return object;
}

function requiredTargetPosition(binding: ActiveHumanoidSkillBinding): Vec3 {
  if (!binding.target_position) throw new Error("Skill has no bound target position");
  return { ...binding.target_position };
}

function wristPosition(
  observation: HumanoidWorldObservation,
  hand: "left" | "right"
): Vec3 {
  return { ...observation.robot.links[
    hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
  ].position };
}

function approachDirections(preferred?: Vec3): Vec3[] {
  const directions: Vec3[] = preferred
    && Math.hypot(preferred.x, preferred.y, preferred.z) > 1e-9
    ? [normalize(preferred)] : [];
  for (let index = 0; index < 16; index += 1) {
    const angle = index / 16 * Math.PI * 2;
    directions.push({ x: Math.cos(angle), y: 0, z: Math.sin(angle) });
  }
  const unique = new Map(directions.map((direction) => [
    `${direction.x.toFixed(3)}:${direction.z.toFixed(3)}`,
    direction
  ]));
  return [...unique.values()];
}

function explorationScore(
  frontier: HumanoidWorldObservation["spatialBelief"]["frontiers"][number],
  strategy: Extract<HumanoidSkillInvocation, { skill: "explore" }>["strategy"]
): number {
  if (strategy === "information_gain") {
    return frontier.expected_information_gain - frontier.revisit_penalty * 2;
  }
  if (strategy === "coverage") {
    return frontier.expected_information_gain / (1 + frontier.revisit_penalty);
  }
  return frontier.score;
}

function surfaceRank(surface: string, preferred: readonly string[]): number {
  const rank = preferred.findIndex((part) => surface.includes(part));
  return rank < 0 ? preferred.length : rank;
}

function planarDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function add(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x + right.x, y: left.y + right.y, z: left.z + right.z };
}

function subtract(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function scale(value: Vec3, amount: number): Vec3 {
  return { x: value.x * amount, y: value.y * amount, z: value.z * amount };
}

function normalize(value: Vec3): Vec3 {
  const length = Math.hypot(value.x, value.y, value.z);
  if (length <= 1e-9) throw new Error("Skill direction cannot be zero");
  return scale(value, 1 / length);
}

function dot(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function cross(left: Vec3, right: Vec3): Vec3 {
  return {
    x: left.y * right.z - left.z * right.y,
    y: left.z * right.x - left.x * right.z,
    z: left.x * right.y - left.y * right.x
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
