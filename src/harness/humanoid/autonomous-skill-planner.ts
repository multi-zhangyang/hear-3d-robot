import type { Goal, Quaternion, Vec3 } from "../../domain/schema.js";
import type { NeuralSafetyInterrupt } from "../../domain/neural-hierarchy.js";
import type { HumanoidSkillInvocation } from "../../domain/humanoid-skill.js";
import type { HumanoidMotionCandidateBatch } from "../../world/humanoid/motion-plan.js";
import type { HumanoidMotionOptionContract } from "../../world/humanoid/motion-option.js";
import type { HumanoidWorldObservation } from "../../world/humanoid/world.js";
import type { G1HandCoordination } from "../../world/humanoid/hand-coordination.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  type G1HandContactSurfaceName
} from "../../world/humanoid/morphology.js";
import {
  navigableManipulationBasePlacements,
  type ActiveHumanoidSkillBinding
} from "./skill-binding.js";
import {
  MINIMUM_BLOCK_REMOVAL_NORMAL_FORCE_N,
  MINIMUM_BLOCK_REMOVAL_STABLE_FRAMES
} from "../../domain/scenario-block-removal.js";
import {
  HUMANOID_ARTICULATION_HORIZON,
  humanoidArticulationGoal,
  humanoidArticulationSegmentMinimumDelta,
  type HumanoidArticulationGoal
} from "./articulation-control.js";
import {
  solveHumanoidArticulationTrajectory
} from "./articulation-trajectory.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  rotateVector,
  yawFromQuaternion
} from "../../world/geometry.js";
import type { HumanoidNavigationArrivalHeading } from "../../world/humanoid/navigation-arrival.js";
import { solveG1PregraspPose } from "../../world/humanoid/pregrasp-pose.js";
import { minimumHumanoidManipulationRootStandoff } from "../../world/humanoid/manipulation-reachability.js";
import { HUMANOID_NAVIGATION_PROFILE } from "../../world/humanoid/environment.js";
import { navigationObstaclePlanarExpansion } from "../../world/navigation.js";
import { alignHumanoidSkillToGoal } from "./goal-skill-alignment.js";
import {
  HUMANOID_RECOVERY_HANDOFF_STEPS,
  HUMANOID_RECOVERY_MAXIMUM_STEPS,
  HUMANOID_RECOVERY_STABLE_STEPS,
  type HumanoidRecoveryPlan
} from "../../world/humanoid/recovery-execution-contract.js";
import {
  humanoidRecoverySafetyInterruptIsCurrent
} from "./recovery-safety-authority.js";

type HumanoidMotionOptionPredicate = HumanoidMotionOptionContract["predicates"][number];

const MINIMUM_SEMANTIC_NAVIGATION_TOLERANCE_METERS = 0.18;
const MAXIMUM_SEMANTIC_NAVIGATION_TOLERANCE_METERS = 0.5;
const MAXIMUM_GOAL_SETTLING_RESERVE_METERS = 0.05;
const APPROACH_SUPPORT_CLEARANCE_MARGIN_METERS = 0.01;

export type AutonomousHumanoidSkillPlan =
  | {
      kind: "navigation";
      targets: Array<{
        target: Vec3;
        arrivalHeading: HumanoidNavigationArrivalHeading | null;
        acceptedPositionToleranceMeters?: number;
        score: number;
      }>;
    }
  | { kind: "motion"; batch: HumanoidMotionCandidateBatch }
  | { kind: "recovery"; plan: HumanoidRecoveryPlan };

export function planAutonomousHumanoidSkill(input: {
  binding: ActiveHumanoidSkillBinding;
  observation: HumanoidWorldObservation;
  articulationGoal?: HumanoidArticulationGoal;
  activeGoal?: Goal;
  recoveryAuthorized?: boolean;
  recoveryInterrupt?: NeuralSafetyInterrupt;
}): AutonomousHumanoidSkillPlan {
  if (input.activeGoal) {
    const alignment = alignHumanoidSkillToGoal({
      goal: input.activeGoal,
      invocation: input.binding.invocation,
      observation: input.observation,
      ...(input.recoveryAuthorized ? { recoveryAuthorized: true } : {})
    });
    if (!alignment.accepted) {
      throw new Error(`Selected Skill is not causally aligned with the active Goal: ${alignment.reason}`);
    }
  }
  if (input.binding.phase_authority === "navigation") {
    const plan = navigationSkillPlan(input.binding, input.observation);
    return {
      ...plan,
      targets: plan.targets.map((candidate) => ({
        ...candidate,
        ...(candidate.acceptedPositionToleranceMeters === undefined
          ? {}
          : {
              acceptedPositionToleranceMeters: goalConstrainedPositionTolerance(
                candidate.target,
                candidate.acceptedPositionToleranceMeters,
                input.activeGoal
              )
            })
      }))
    };
  }
  if (input.binding.invocation.skill === "stabilize"
    && input.binding.phase === "recover_support"
    && (input.observation.robot.fallen
      || input.recoveryAuthorized && input.recoveryInterrupt !== undefined)) {
    return recoverySkillPlan(input);
  }
  return {
    kind: "motion",
    batch: motionSkillPlan(
      input.binding,
      input.observation,
      input.articulationGoal
    )
  };
}

function recoverySkillPlan(input: {
  binding: ActiveHumanoidSkillBinding;
  observation: HumanoidWorldObservation;
  recoveryAuthorized?: boolean;
  recoveryInterrupt?: NeuralSafetyInterrupt;
}): Extract<AutonomousHumanoidSkillPlan, { kind: "recovery" }> {
  const interrupt = input.recoveryInterrupt;
  if (!input.recoveryAuthorized || !interrupt
    || input.binding.recovery_interrupt_id !== interrupt.interrupt_id
    || !humanoidRecoverySafetyInterruptIsCurrent(interrupt, {
      worldRevision: input.observation.worldRevision,
      interruptId: input.binding.recovery_interrupt_id
    })) {
    throw new Error(
      "Fallen recovery requires the current acknowledged Body-to-Reflex safety interrupt"
    );
  }
  const invocation = input.binding.invocation;
  if (invocation.skill !== "stabilize") {
    throw new Error("Only stabilize.recover_support may plan fallen recovery");
  }
  return {
    kind: "recovery",
    plan: {
      id: `recovery:${input.binding.transaction_id}:${interrupt.interrupt_id}`,
      contract: {
        protocol: "humanoid-embodied-recovery-contract-v1",
        safetyInterrupt: structuredClone(interrupt),
        minimumSupportMarginMeters: invocation.minimum_support_margin_m,
        stableSteps: HUMANOID_RECOVERY_STABLE_STEPS,
        handoffSteps: HUMANOID_RECOVERY_HANDOFF_STEPS,
        maximumSteps: HUMANOID_RECOVERY_MAXIMUM_STEPS,
        authorizedContacts: recoveryCarriedObjectContacts(input.observation),
        standing: {
          minimumRootHeightMeters: 0.7,
          minimumUpright: 0.9,
          maximumRootLinearSpeedMetersPerSecond: 0.35,
          maximumRootAngularSpeedRadiansPerSecond: 0.5,
          maximumJointSpeedRadiansPerSecond: 1.5,
          requireBothFeetContact: true
        },
        safetyLimits: {
          maximumPeakContactNormalForceN: 2500,
          maximumTotalContactNormalForceN: 4000,
          maximumTotalContactForceRiseRateNPerSecond: 100000,
          maximumJointSpeedRadiansPerSecond: 40,
          minimumJointLimitMarginRadians: -0.1
        }
      }
    }
  };
}

function recoveryCarriedObjectContacts(
  observation: HumanoidWorldObservation
): Array<{
  hand_surface: G1HandContactSurfaceName;
  object_id: string;
  required: false;
}> {
  const knownSurfaces = new Set<string>(G1_HAND_CONTACT_SURFACE_NAMES);
  return observation.interaction.carrying.bindings.flatMap((binding) => (
    observation.interaction.grasp_authority.hand_surfaces[binding.hand]
      .filter((surface): surface is G1HandContactSurfaceName => (
        knownSurfaces.has(surface)
      ))
      .map((surface) => ({
        hand_surface: surface,
        object_id: binding.object_id,
        required: false as const
      }))
  ));
}

function goalConstrainedPositionTolerance(
  target: Vec3,
  semanticTolerance: number,
  activeGoal: Goal | undefined
): number {
  const remainingGoalMargins = activeGoal?.predicates.flatMap((predicate) => {
    if (predicate.type !== "robot_at") return [];
    const margin = predicate.tolerance - planarDistance(target, predicate.target);
    if (margin <= 0) return [];
    const settlingReserve = Math.min(
      MAXIMUM_GOAL_SETTLING_RESERVE_METERS,
      margin * 0.5
    );
    return [margin - settlingReserve];
  }) ?? [];
  return remainingGoalMargins.length === 0
    ? semanticTolerance
    : Math.min(semanticTolerance, ...remainingGoalMargins);
}

function navigationSkillPlan(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation
): Extract<AutonomousHumanoidSkillPlan, { kind: "navigation" }> {
  const invocation = binding.invocation;
  if (invocation.skill === "navigate_to_zone") {
    const zone = observation.interaction.zones.find(
      ({ zone_id: zoneId }) => zoneId === invocation.zone_id
    );
    if (!zone) throw new Error("Selected navigation zone is no longer observable");
    return {
      kind: "navigation",
      targets: [{
        target: requiredTargetPosition(binding),
        arrivalHeading: null,
        acceptedPositionToleranceMeters: Math.min(
          1,
          Math.max(0.06, Math.min(zone.size.x, zone.size.z) * 0.25)
        ),
        score: 1
      }]
    };
  }
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
        acceptedPositionToleranceMeters: clamp(
          observation.spatialBelief.resolution_m * 0.5,
          MINIMUM_SEMANTIC_NAVIGATION_TOLERANCE_METERS,
          MAXIMUM_SEMANTIC_NAVIGATION_TOLERANCE_METERS
        ),
        score: explorationScore(frontier, invocation.strategy)
      })).sort((left, right) => right.score - left.score)
    };
  }
  if (invocation.skill === "carry"
    || invocation.skill === "bimanual_carry") {
    return {
      kind: "navigation",
      targets: [{
        target: { ...invocation.target },
        arrivalHeading: null,
        acceptedPositionToleranceMeters: invocation.tolerance_m,
        score: 1
      }]
    };
  }
  if (invocation.skill === "carry_to_zone") {
    const zone = observation.interaction.zones.find(
      ({ zone_id: zoneId }) => zoneId === invocation.zone_id
    );
    if (!zone) throw new Error("Selected carry destination zone is no longer observable");
    const object = objectEntry(observation, invocation.object_id);
    return {
      kind: "navigation",
      targets: [{
        target: requiredTargetPosition(binding),
        arrivalHeading: null,
        acceptedPositionToleranceMeters: carriedObjectZoneArrivalTolerance(
          invocation.tolerance_m,
          zone,
          object
        ),
        score: 1
      }]
    };
  }
  if (invocation.skill === "retreat") {
    return {
      kind: "navigation",
      targets: [{
        target: { ...invocation.target },
        arrivalHeading: null,
        acceptedPositionToleranceMeters: clamp(
          invocation.minimum_obstacle_clearance_m * 0.4,
          MINIMUM_SEMANTIC_NAVIGATION_TOLERANCE_METERS,
          MAXIMUM_SEMANTIC_NAVIGATION_TOLERANCE_METERS
        ),
        score: 1
      }]
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
  const currentYaw = yawFromQuaternion(observation.robot.rootRotation);
  const lateralOffsets = approachLateralOffsets(invocation.hand ?? null);
  const requestedStandoff = Math.max(
    invocation.standoff_m,
    minimumHumanoidManipulationRootStandoff({
      ...(point ? { clearanceMeters: point.clearance_m } : {})
    })
  );
  const contactedSolids = objectContactedSolids(
    observation,
    invocation.object_id
  );
  const objectBasePlacements = observation.manipulationBasePlacements
    .filter((placement) => placement.objectId === invocation.object_id);
  const selectedBasePlacements = navigableManipulationBasePlacements(
    observation,
    invocation.object_id
  )
    .filter((placement) => (invocation.interaction_point_id === null
        || placement.interactionPointId === invocation.interaction_point_id)
      && (!invocation.hand
        || placement.handSurface.startsWith(`${invocation.hand}_`)));
  const reachabilityTargets = selectedBasePlacements
    .filter((placement) => (
      planarDistance(current, placement.rootWorldTarget) > 0.025
        || Math.abs(Math.atan2(
          Math.sin(placement.rootYawRadians - currentYaw),
          Math.cos(placement.rootYawRadians - currentYaw)
        )) > 0.03
    ))
    .map((placement) => ({
      target: { ...placement.rootWorldTarget },
      arrivalHeading: {
        type: "yaw" as const,
        yaw_radians: placement.rootYawRadians,
        tolerance_radians: 0.12
      },
      score: 4 - placement.ikResidualMeters
        - planarDistance(current, placement.rootWorldTarget) * 0.05
    }));
  const geometricTargets = directions.flatMap((direction, directionIndex) => (
      lateralOffsets.map((lateralOffset, lateralIndex) => {
      const lateral = { x: direction.z, y: 0, z: -direction.x };
      const standoff = Math.max(
        requestedStandoff,
        minimumContactedSolidStandoff(around, direction, contactedSolids)
      );
      const target = {
        x: around.x - direction.x * standoff
          + lateral.x * lateralOffset,
        y: current.y,
        z: around.z - direction.z * standoff
          + lateral.z * lateralOffset
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
          - directionIndex * 0.01 - lateralIndex * 0.005
      };
    })));
  return {
    kind: "navigation",
    // Once live IK has produced an exact base-placement contract, a generic
    // face-the-object pose is not an equivalent fallback: it can complete
    // navigation while leaving the committed hand target unreachable.
    targets: (objectBasePlacements.length > 0
      ? reachabilityTargets
      : geometricTargets).sort((left, right) => right.score - left.score)
  };
}

function carriedObjectZoneArrivalTolerance(
  requestedTolerance: number,
  zone: HumanoidWorldObservation["interaction"]["zones"][number],
  object: ReturnType<typeof objectEntry>
): number {
  const horizontalMargin = Math.min(
    zone.size.x / 2 - objectExtentAlong(
      object.size,
      object.pose.rotation,
      { x: 1, y: 0, z: 0 }
    ),
    zone.size.z / 2 - objectExtentAlong(
      object.size,
      object.pose.rotation,
      { x: 0, y: 0, z: 1 }
    )
  );
  if (horizontalMargin <= 0) {
    throw new Error("Carried object cannot fit inside the selected semantic zone");
  }
  const settlingReserve = Math.min(0.05, horizontalMargin * 0.25);
  return Math.min(
    requestedTolerance,
    Math.max(0.02, horizontalMargin - settlingReserve)
  );
}

function approachLateralOffsets(hand: "left" | "right" | null): number[] {
  if (hand === "right") return [0.2, 0.14, 0.26, 0, -0.18];
  if (hand === "left") return [-0.2, -0.14, -0.26, 0, 0.18];
  return [0, 0.18, -0.18];
}

function objectContactedSolids(
  observation: HumanoidWorldObservation,
  objectId: string
): HumanoidWorldObservation["solidTokens"] {
  return observation.solidTokens.filter((solid) => (
    solid.currentContacts.some((contact) => (
      contact.firstObject === objectId && contact.secondSolid === solid.id
    ) || (
      contact.secondObject === objectId && contact.firstSolid === solid.id
    ))
  ));
}

function minimumContactedSolidStandoff(
  interactionPoint: Vec3,
  direction: Vec3,
  solids: HumanoidWorldObservation["solidTokens"]
): number {
  const planarLength = Math.hypot(direction.x, direction.z);
  if (planarLength <= 1e-9 || solids.length === 0) return 0;
  const unit = {
    x: direction.x / planarLength,
    z: direction.z / planarLength
  };
  const navigationExpansion = navigationObstaclePlanarExpansion(
    HUMANOID_NAVIGATION_PROFILE.radius
  ) + APPROACH_SUPPORT_CLEARANCE_MARGIN_METERS;
  return Math.max(0, ...solids.map((solid) => {
    const pointFromCenter = {
      x: interactionPoint.x - solid.center.x,
      z: interactionPoint.z - solid.center.z
    };
    const pointProjection = pointFromCenter.x * unit.x
      + pointFromCenter.z * unit.z;
    const solidProjectedHalfExtent = Math.abs(unit.x) * solid.size.x / 2
      + Math.abs(unit.z) * solid.size.z / 2;
    return pointProjection + solidProjectedHalfExtent + navigationExpansion;
  }));
}

function motionSkillPlan(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  articulationGoal?: HumanoidArticulationGoal
): HumanoidMotionCandidateBatch {
  const invocation = binding.invocation;
  const phase = binding.phase;
  const motion = skillMotionTarget(binding, observation, articulationGoal);
  const coordination = handCoordinationForSkill(
    observation.handCoordination,
    invocation,
    phase
  );
  const predicates = skillPredicates(
    binding,
    observation,
    motion,
    coordination
  );
  const candidates = fixedBaseStanceCandidates(motionCandidates({
    binding,
    observation,
    motion,
    coordination,
    contactConstraints: skillContactConstraints(binding, observation, predicates)
  }), observation);
  return {
    objective: `Execute ${invocation.skill}.${phase} through the generic closed-loop skill solver`,
    termination: {
      option_id: `skill:${binding.transaction_id}:${phase}`,
      predicates,
      stable_steps: invocation.skill === "break_block"
        ? MINIMUM_BLOCK_REMOVAL_STABLE_FRAMES
        : motion.contactEstablishment ? 4
        : predicates.some(({ type }) => type === "grasp_verified") ? 8 : 4,
      phases: skillTerminationPhases(invocation, phase, predicates)
    },
    candidates
  };
}

function skillTerminationPhases(
  invocation: HumanoidSkillInvocation,
  phase: string,
  predicates: HumanoidMotionOptionPredicate[]
): HumanoidMotionCandidateBatch["termination"]["phases"] {
  if (invocation.skill !== "place" || phase !== "settle_and_release"
    || !invocation.release_after_settled) return null;
  const graspIndexes = predicates.flatMap((predicate, index) => (
    predicate.type === "grasp_verified"
      && predicate.object_id === invocation.object_id ? [index] : []
  ));
  if (graspIndexes.length === 0) {
    throw new Error("Place release requires a verified-grasp precondition");
  }
  const predicate = (predicateIndex: number) => ({
    op: "predicate" as const,
    predicate_index: predicateIndex
  });
  return {
    precondition: {
      condition: {
        op: "all",
        conditions: graspIndexes.map(predicate)
      },
      stable_steps: 1
    },
    during: null,
    terminal: {
      condition: {
        op: "all",
        conditions: [
          ...graspIndexes.map((predicateIndex) => ({
            op: "not" as const,
            condition: predicate(predicateIndex)
          })),
          ...predicates.flatMap((_, predicateIndex) => (
            graspIndexes.includes(predicateIndex) ? [] : [predicate(predicateIndex)]
          ))
        ]
      }
    }
  };
}

function fixedBaseStanceCandidates(
  candidates: HumanoidMotionCandidateBatch["candidates"],
  observation: HumanoidWorldObservation
): HumanoidMotionCandidateBatch["candidates"] {
  const feet = {
    left_foot: {
      position: {
        ...observation.robot.links.left_ankle_roll_link.position
      },
      frame: "world" as const,
      tolerance_m: 0.04,
      servo_mode: "task_tolerance" as const
    },
    right_foot: {
      position: {
        ...observation.robot.links.right_ankle_roll_link.position
      },
      frame: "world" as const,
      tolerance_m: 0.04,
      servo_mode: "task_tolerance" as const
    }
  };
  return candidates.map((candidate) => {
    const mobile = candidate.keyframes.some((keyframe) => (
      keyframe.root_velocity != null || keyframe.root_yaw_velocity != null
    ));
    if (mobile) return candidate;
    return {
      ...candidate,
      keyframes: candidate.keyframes.map((keyframe) => ({
        ...keyframe,
        left_foot: feet.left_foot,
        right_foot: feet.right_foot
      }))
    };
  });
}

interface SkillMotionTarget {
  hands: Partial<Record<"left" | "right", Vec3>>;
  handOrientations?: Partial<Record<"left" | "right", Quaternion>>;
  handPaths?: Partial<Record<"left" | "right", Vec3[]>>;
  handPathOrientations?: Partial<Record<"left" | "right", Quaternion[]>>;
  toleranceM?: number;
  handSurfaces?: Partial<Record<"left" | "right", G1HandContactSurfaceName>>;
  objectTarget?: Vec3;
  objectOrientationTarget?: Quaternion;
  direction?: Vec3;
  contactApproach?: {
    hand: "left" | "right";
    directionWorld: Vec3;
    standoffDistanceM: number;
  };
  contactEstablishment?: {
    hand: "left" | "right";
    approachDirection: Vec3;
  };
  articulationSegment?: {
    originPosition: number;
    targetPosition: number;
    finalTargetPosition: number;
    direction: "increasing" | "decreasing";
    minimumDelta: number;
    horizonComplete: boolean;
  };
}

function skillMotionTarget(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  articulationGoal?: HumanoidArticulationGoal
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
    const placement = placementTarget(invocation, observation);
    const objectTarget = binding.phase === "align_destination"
      ? placement.stagingPosition
      : placement.finalPosition;
    const rotationDelta = placement.targetRotation
      ? normalizeQuaternion(multiplyQuaternion(
          placement.targetRotation,
          inverseQuaternion(object.pose.rotation)
        ))
      : null;
    const hands = carriedHands(observation, invocation.object_id, invocation.hands)
      .reduce<SkillMotionTarget["hands"]>((targets, hand) => {
        const wrist = wristPosition(observation, hand);
        targets[hand] = rotationDelta
          ? add(objectTarget, rotateVector(
              rotationDelta,
              subtract(wrist, object.pose.position)
            ))
          : add(wrist, subtract(objectTarget, object.pose.position));
        return targets;
      }, {});
    const handOrientations = rotationDelta
      ? Object.fromEntries(carriedHands(
          observation,
          invocation.object_id,
          invocation.hands
        ).map((hand) => [
          hand,
          normalizeQuaternion(multiplyQuaternion(
            rotationDelta,
            wristPose(observation, hand).rotation
          ))
        ])) as SkillMotionTarget["handOrientations"]
      : undefined;
    return {
      hands,
      ...(handOrientations ? { handOrientations } : {}),
      objectTarget,
      ...(placement.targetRotation
        ? { objectOrientationTarget: placement.targetRotation }
        : {})
    };
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
  if ((invocation.skill === "open" || invocation.skill === "close"
      || invocation.skill === "turn")
    && (binding.phase === "reach_handle"
      || binding.phase === "reach_interaction"
      || binding.phase === "establish_grasp")) {
    const point = selectedInteractionPoint(binding);
    if (!point) throw new Error("Articulation grasp requires an interaction point");
    const wrist = wristPose(observation, invocation.hand);
    const approachDirection = normalize(point.approach_direction_world
      ?? subtract(point.world_position, wrist.position));
    const pregrasp = solveG1PregraspPose({
      hand: invocation.hand,
      wristRotation: wrist.rotation,
      handSurfaces: observation.handSurfaces,
      interactionPoint: point.world_position,
      approachDirection,
      ...(binding.target_articulation?.axis_world
        ? { preferredGraspAxis: binding.target_articulation.axis_world }
        : {})
    });
    return {
      hands: { [invocation.hand]: pregrasp.position },
      handOrientations: { [invocation.hand]: pregrasp.rotation },
      contactApproach: {
        hand: invocation.hand,
        directionWorld: approachDirection,
        standoffDistanceM: maximumHandSweepRadius(observation, invocation.hand)
          + point.clearance_m
      },
      handSurfaces: {
        [invocation.hand]: selectedHandSurfaceForPoint(
          binding,
          observation,
          invocation.hand,
          point
        )
      },
      ...(binding.phase === "establish_grasp"
        ? {
            contactEstablishment: {
              hand: invocation.hand,
              approachDirection
            }
          }
        : {}),
      toleranceM: point.clearance_m
    };
  }
  if ((invocation.skill === "open" || invocation.skill === "close"
      || invocation.skill === "turn")
    && binding.phase === "actuate_joint") {
    return articulationMotionTarget(
      binding,
      observation,
      invocation,
      articulationGoal
    );
  }
  if (invocation.skill === "approach" || !("hand" in invocation)) {
    return { hands: {} };
  }
  const contact = wristTargetForPoint(binding, observation, invocation.hand);
  const stroke = skillStroke(binding, invocation);
  const point = selectedInteractionPoint(binding);
  const handSurface = point
    ? selectedHandSurfaceForPoint(binding, observation, invocation.hand, point)
    : undefined;
  return {
    hands: { [invocation.hand]: stroke ? add(contact, stroke) : contact },
    ...(handSurface
      ? { handSurfaces: { [invocation.hand]: handSurface } }
      : {}),
    toleranceM: invocation.skill === "reach"
      ? invocation.tolerance_m
      : clamp((point?.clearance_m ?? 0.06) * 0.75, 0.04, 0.08),
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
  return null;
}

function articulationMotionTarget(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  invocation: Extract<HumanoidSkillInvocation, {
    skill: "open" | "close" | "turn";
  }>,
  continuedGoal?: HumanoidArticulationGoal
): SkillMotionTarget {
  const articulation = binding.target_articulation;
  const point = selectedInteractionPoint(binding);
  if (!articulation || articulation.position === null || !point) {
    throw new Error("Articulation Skill requires live joint and interaction-point geometry");
  }
  const currentJointPosition = articulation.position;
  const goal = continuedGoal ?? humanoidArticulationGoal({
    invocation,
    articulation
  });
  const trajectory = solveHumanoidArticulationTrajectory({
    articulation,
    interactionPoint: point.world_position,
    targetPosition: goal.target_position,
    maximumPathLengthM: HUMANOID_ARTICULATION_HORIZON.maximum_task_path_m
  });
  const handSurface = selectedHandSurfaceForPoint(
    binding,
    observation,
    invocation.hand,
    point
  );
  const surface = observation.handSurfaces.find((candidate) => (
    candidate.handSurface === handSurface
  ));
  if (!surface) {
    throw new Error("Articulation Skill requires an observed hand contact surface");
  }
  const currentWrist = wristPosition(observation, invocation.hand);
  const hasLiveContact = handContactsObject(
    observation,
    invocation.hand,
    invocation.object_id
  );
  const graspOffset = hasLiveContact
    ? subtract(point.world_position, currentWrist)
    : articulatedGraspOffset(
        observation,
        invocation.hand,
        surface.surfaceFromWristWorld
      );
  const contactWrist = hasLiveContact
    ? currentWrist
    : subtract(point.world_position, graspOffset);
  const wristRotation = wristPose(observation, invocation.hand).rotation;
  const rotationDeltas = trajectory.joint_waypoints.map((jointPosition) => (
    articulation.type === "hinge"
      ? axisAngleQuaternion(
          articulation.axis_world,
          jointPosition - currentJointPosition
        )
      : IDENTITY_QUATERNION
  ));
  const wristPath = [
    contactWrist,
    ...trajectory.interaction_waypoints.map((waypoint, index) => subtract(
      waypoint,
      rotateVector(
        rotationDeltas[index]!,
        graspOffset
      )
    ))
  ];
  const wristOrientations = [
    wristRotation,
    ...rotationDeltas.map((delta) => normalizeQuaternion(
      multiplyQuaternion(delta, wristRotation)
    ))
  ];
  return {
    hands: { [invocation.hand]: wristPath.at(-1)! },
    handPaths: { [invocation.hand]: wristPath },
    handPathOrientations: { [invocation.hand]: wristOrientations },
    handSurfaces: { [invocation.hand]: surface.handSurface },
    direction: trajectory.initial_direction_world,
    articulationSegment: {
      originPosition: currentJointPosition,
      targetPosition: trajectory.joint_target_position,
      finalTargetPosition: trajectory.final_target_position,
      direction: trajectory.joint_delta > 0 ? "increasing" : "decreasing",
      minimumDelta: humanoidArticulationSegmentMinimumDelta({
        articulation,
        segmentDelta: trajectory.joint_delta
      }),
      horizonComplete: trajectory.horizon_complete
    }
  };
}

function skillPredicates(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  motion: SkillMotionTarget,
  coordination: G1HandCoordination | null
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
  const contactHand = "hand" in invocation ? invocation.hand : null;
  const contactSurface = contactHand
    ? motion.handSurfaces?.[contactHand]
    : undefined;
  if (contactHand && contactSurface
    && "object_id" in invocation
    && [
      "reach_handle",
      "reach_interaction",
      "solve_whole_body_reach",
      "establish_contact"
    ].includes(phase)) {
    return [interactionRegionContactPredicate(binding, contactHand), {
      type: "root_near_point",
      target: { ...observation.robot.rootPosition },
      tolerance_m: 0.08
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
  if ((invocation.skill === "open" || invocation.skill === "close"
      || invocation.skill === "turn")
    && phase === "establish_grasp") {
    const origin = observation.handCoordination[invocation.hand];
    const target = coordination?.[invocation.hand];
    if (!target) {
      throw new Error("Articulation grasp requires a hand-coordination target");
    }
    const closureDistance = Math.hypot(
      target.thumb_opposition - origin.thumb_opposition,
      target.thumb_curl - origin.thumb_curl,
      target.index_curl - origin.index_curl,
      target.middle_curl - origin.middle_curl
    );
    return [interactionRegionContactPredicate(binding, invocation.hand, 2),
      ...(closureDistance > 1e-6 ? [{
        type: "hand_coordination_displaced" as const,
        hand: invocation.hand,
        origin: { ...origin },
        minimum_distance: closureDistance * 0.2
      }] : []), {
      type: "root_near_point",
      target: { ...observation.robot.rootPosition },
      tolerance_m: 0.08
    }];
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
    return placePredicates(invocation, motion, observation);
  }
  if ((invocation.skill === "open" || invocation.skill === "close")
    && phase === "actuate_joint") {
    const segment = motion.articulationSegment;
    if (!segment) throw new Error("Articulation Skill segment is unavailable");
    return [articulationContactPredicate(invocation, 2), {
      type: "articulation_displaced",
      object_id: invocation.object_id,
      joint_id: invocation.joint_id,
      origin_position: segment.originPosition,
      direction: segment.direction,
      minimum_delta: segment.minimumDelta
      }];
  }
  if (invocation.skill === "turn" && phase === "actuate_joint") {
    const segment = motion.articulationSegment;
    if (!segment) throw new Error("Turn Skill segment is unavailable");
    return [articulationContactPredicate(invocation, 2), {
      type: "articulation_displaced",
      object_id: invocation.object_id,
      joint_id: invocation.joint_id,
      origin_position: segment.originPosition,
      direction: segment.direction,
      minimum_delta: segment.minimumDelta
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
      tolerance_m: motion.toleranceM ?? 0.05
    }];
  }
  if (motion.objectTarget && "object_id" in invocation) {
    return [{
      type: "object_near_point",
      object_id: invocation.object_id,
      target: motion.objectTarget,
      tolerance_m: 0.08,
      ...(motion.objectOrientationTarget
        ? {
            target_orientation: motion.objectOrientationTarget,
            orientation_tolerance_rad: 0.18
          }
        : {})
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

function articulationContactPredicate(
  invocation: Extract<HumanoidSkillInvocation, {
    skill: "open" | "close" | "turn";
  }>,
  minimumDistinctSurfaces = 1
): HumanoidMotionOptionPredicate {
  return {
    type: "hand_contact_object_any",
    hand: invocation.hand,
    object_id: invocation.object_id,
    minimum_normal_force: 1,
    minimum_distinct_surfaces: minimumDistinctSurfaces
  };
}

function interactionRegionContactPredicate(
  binding: ActiveHumanoidSkillBinding,
  hand: "left" | "right",
  minimumDistinctSurfaces = 1
): HumanoidMotionOptionPredicate {
  if (!("object_id" in binding.invocation)) {
    throw new Error("Interaction-region contact requires a bound object");
  }
  const point = selectedInteractionPoint(binding);
  if (!point) {
    throw new Error("Interaction-region contact requires a live interaction point");
  }
  return {
    type: "hand_contact_object_region",
    hand,
    object_id: binding.invocation.object_id,
    center_world: { ...point.world_position },
    maximum_distance_m: point.clearance_m,
    minimum_normal_force: 1,
    minimum_distinct_surfaces: minimumDistinctSurfaces
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
  if (input.motion.contactEstablishment) {
    return articulationGraspCandidates(input);
  }
  if (input.motion.handPaths) return articulationCandidates(input);
  const hands = Object.entries(input.motion.hands) as Array<["left" | "right", Vec3]>;
  const current = Object.fromEntries(hands.map(([hand]) => [
    hand,
    wristPosition(input.observation, hand)
  ])) as Partial<Record<"left" | "right", Vec3>>;
  const currentOrientations = Object.fromEntries(hands.map(([hand]) => [
    hand,
    wristPose(input.observation, hand).rotation
  ])) as Partial<Record<"left" | "right", Quaternion>>;
  const variants = hands.length === 0
    ? [{
        lateral: 0,
        durationScale: 1,
        kinematicScope: "arm_only" as const,
        clearOppositeHand: false
      }]
    : [
        {
          lateral: 0,
          durationScale: 1,
          kinematicScope: "arm_only" as const,
          clearOppositeHand: false
        },
        {
          lateral: 0,
          durationScale: 1.15,
          kinematicScope: "whole_body_reach" as const,
          clearOppositeHand: false
        },
        {
          lateral: 0,
          durationScale: 1.35,
          kinematicScope: "arm_only" as const,
          clearOppositeHand: true
        },
        {
          lateral: 0.06,
          durationScale: 1.7,
          kinematicScope: "whole_body_reach" as const,
          clearOppositeHand: true
        }
      ];
  const maximumTravel = hands.reduce((maximum, [hand, target]) => Math.max(
    maximum,
    distance(current[hand]!, target)
  ), 0);
  const directStepCount = Math.max(1, Math.ceil(maximumTravel / 0.08));
  const servoMode = input.motion.handSurfaces
    ? "task_tolerance" as const
    : "precision" as const;
  const baseDuration = hands.length === 0
    ? 1 : clamp(directStepCount * 1.25, 3.2, 8);
  return variants.map(({
    lateral: lateralOffset,
    durationScale,
    kinematicScope,
    clearOppositeHand
  }, index) => {
    const duration = clamp(baseDuration * durationScale, 1, 8);
    const oppositeClearance = clearOppositeHand && hands.length === 1
      ? nonOperatingHandWorldClearance(
          input.binding,
          input.observation,
          hands[0]![0]
        )
      : null;
    const variantCurrent = oppositeClearance
      ? {
          ...current,
          [oppositeClearance.hand]: wristPosition(
            input.observation,
            oppositeClearance.hand
          )
        }
      : current;
    const variantFinal = oppositeClearance
      ? { ...input.motion.hands, [oppositeClearance.hand]: oppositeClearance.target }
      : input.motion.hands;
    const variantHands = Object.entries(variantFinal) as Array<
      ["left" | "right", Vec3]
    >;
    const middleHands = Object.fromEntries(variantHands.map(([hand, target]) => {
      const operating = hands.some(([candidate]) => candidate === hand);
      const start = variantCurrent[hand]!;
      return [hand, {
        x: (start.x + target.x) / 2 + (operating ? lateralOffset : 0),
        y: index === 0 || !operating
          ? (start.y + target.y) / 2
          : Math.max(start.y, target.y) + 0.08,
        z: (start.z + target.z) / 2
      }];
    })) as Partial<Record<"left" | "right", Vec3>>;
    const contactStage = input.motion.contactApproach
      ? {
          ...variantFinal,
          [input.motion.contactApproach.hand]: add(
            variantFinal[input.motion.contactApproach.hand]!,
            scale(
              input.motion.contactApproach.directionWorld,
              -input.motion.contactApproach.standoffDistanceM
            )
          )
        }
      : null;
    const contactTransferFrames = contactStage
      ? [0.25, 0.5, 0.75, 1].map((progress) => motionKeyframe(
          duration * 0.55 * progress,
          Object.fromEntries(Object.entries(contactStage).map(([hand, target]) => [
            hand,
            add(
              variantCurrent[hand as "left" | "right"]!,
              scale(subtract(
                target,
                variantCurrent[hand as "left" | "right"]!
              ), progress)
            )
          ])) as SkillMotionTarget["hands"],
          null,
          Math.min(0.06, input.motion.toleranceM ?? 0.04),
          kinematicScope,
          undefined,
          servoMode,
          currentOrientations
        ))
      : [];
    return {
      id: motionCandidateId(input, index),
      intent: index === 0 ? "minimum-displacement arm path"
        : index === 1 ? "fixed-base whole-body reach path"
          : index === 2 ? "opposite-hand clearance arm path"
            : "opposite-hand clearance whole-body path",
      duration_seconds: duration,
      contact_constraints: input.contactConstraints ?? [],
      keyframes: [
        motionKeyframe(
          0,
          variantCurrent,
          input.coordination ? input.observation.handCoordination : null,
          0.04,
          kinematicScope,
          undefined,
          servoMode,
          currentOrientations
        ),
        ...(contactStage ? contactTransferFrames : hands.length === 0
          || lateralOffset === 0 && !clearOppositeHand ? [] : [
          motionKeyframe(
            duration * 0.55,
            middleHands,
            null,
            servoMode === "task_tolerance"
              ? Math.min(0.06, input.motion.toleranceM ?? 0.04)
              : Math.max(0.05, input.motion.toleranceM ?? 0.04),
            kinematicScope,
            undefined,
            servoMode,
            input.motion.handOrientations
          )
        ]),
        ...[motionKeyframe(
            duration,
            variantFinal,
            input.coordination,
            input.motion.toleranceM ?? 0.04,
            kinematicScope,
            undefined,
            servoMode,
            input.motion.handOrientations
          )]
      ]
    };
  });
}

function articulationGraspCandidates(input: {
  binding: ActiveHumanoidSkillBinding;
  observation: HumanoidWorldObservation;
  motion: SkillMotionTarget;
  coordination: G1HandCoordination | null;
  contactConstraints: HumanoidMotionCandidateBatch["candidates"][number]["contact_constraints"];
}): HumanoidMotionCandidateBatch["candidates"] {
  const establishment = input.motion.contactEstablishment;
  if (!establishment) throw new Error("Articulation grasp authority is missing");
  const { hand, approachDirection } = establishment;
  const start = wristPosition(input.observation, hand);
  const graspTarget = input.motion.hands[hand];
  if (!graspTarget) throw new Error("Articulation grasp target is missing");
  const orientation = input.motion.handOrientations?.[hand]
    ?? wristPose(input.observation, hand).rotation;
  const startOrientation = wristPose(input.observation, hand).rotation;
  const lateral = normalize(cross({ x: 0, y: 1, z: 0 }, approachDirection));
  const towardGrasp = subtract(graspTarget, start);
  const targets = [
    start,
    add(add(start, scale(towardGrasp, 0.35)), scale(approachDirection, 0.005)),
    add(add(add(
      start,
      scale(towardGrasp, 0.6)
    ), scale(approachDirection, -0.005)), scale(lateral, 0.008))
  ];
  return targets.map((target, index) => {
    const closureSeconds = 0.32 + index * 0.04;
    const duration = 1.2 + index * 0.25;
    return {
      id: motionCandidateId(input, index),
      intent: index === 0
        ? "hold contact frame while establishing a multi-surface grasp"
        : index === 1
          ? "approach-biased tactile grasp acquisition"
          : "retraction-lateral tactile grasp acquisition",
      duration_seconds: duration,
      contact_constraints: input.contactConstraints ?? [],
      keyframes: [
        motionKeyframe(
          0,
          { [hand]: start },
          input.observation.handCoordination,
          0.08,
          "arm_only",
          undefined,
          "task_tolerance",
          { [hand]: startOrientation },
          0.45
        ),
        motionKeyframe(
          closureSeconds,
          { [hand]: target },
          input.coordination,
          0.08,
          "arm_only",
          undefined,
          "task_tolerance",
          { [hand]: orientation },
          0.45
        ),
        motionKeyframe(
          duration,
          { [hand]: target },
          input.coordination,
          0.08,
          "arm_only",
          undefined,
          "task_tolerance",
          { [hand]: orientation },
          0.45
        )
      ]
    };
  });
}

function nonOperatingHandWorldClearance(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  operatingHand: "left" | "right"
): { hand: "left" | "right"; target: Vec3 } {
  const hand = operatingHand === "left" ? "right" : "left";
  const current = wristPosition(observation, hand);
  const point = selectedInteractionPoint(binding);
  const towardContact = point?.approach_direction_world
    ? normalize(point.approach_direction_world)
    : normalize(subtract(point?.world_position ?? requiredTargetPosition(binding), current));
  return {
    hand,
    target: add(current, {
      x: -towardContact.x * 0.14,
      y: 0.12,
      z: -towardContact.z * 0.14
    })
  };
}

function articulationCandidates(input: {
  binding: ActiveHumanoidSkillBinding;
  observation: HumanoidWorldObservation;
  motion: SkillMotionTarget;
  coordination: G1HandCoordination | null;
  contactConstraints: HumanoidMotionCandidateBatch["candidates"][number]["contact_constraints"];
}): HumanoidMotionCandidateBatch["candidates"] {
  const pathEntries = Object.entries(input.motion.handPaths ?? {}) as Array<
    ["left" | "right", Vec3[]]
  >;
  if (pathEntries.length !== 1 || pathEntries[0]![1].length === 0) {
    throw new Error("Articulation trajectory requires one non-empty hand path");
  }
  const [hand, path] = pathEntries[0]!;
  const pathOrientations = input.motion.handPathOrientations?.[hand];
  if (pathOrientations && pathOrientations.length !== path.length) {
    throw new Error("Articulation hand position and orientation paths must align");
  }
  const start = wristPosition(input.observation, hand);
  const pathLength = [start, ...path].slice(1).reduce((total, point, index, all) => (
    total + distance(index === 0 ? start : all[index - 1]!, point)
  ), 0);
  const contactEstablishmentSeconds = 0.8;
  const baseMotionDuration = clamp(1.8 + pathLength * 3.2, 2.2, 7.2);
  return [0.9, 1.1, 1.35].map((durationScale, candidateIndex) => {
    const motionDuration = clamp(baseMotionDuration * durationScale, 2.2, 7.2);
    const duration = contactEstablishmentSeconds + motionDuration;
    const kinematicScope = candidateIndex === 0
      ? "arm_only" as const
      : "whole_body_reach" as const;
    const taskToleranceMeters = [0.035, 0.05, 0.09][candidateIndex]!;
    const rootVelocities = candidateIndex === 2
      ? contactFollowingRootVelocities({
          points: path.slice(1),
          durationSeconds: motionDuration,
          rootYawRadians: yawFromQuaternion(input.observation.robot.rootRotation)
        })
      : null;
    const servoMode = candidateIndex === 2
      ? "task_tolerance" as const
      : "precision" as const;
    const initial = motionKeyframe(
      0,
      { [hand]: start },
      input.observation.handCoordination,
      taskToleranceMeters,
      kinematicScope,
      undefined,
      servoMode,
      pathOrientations ? { [hand]: wristPose(input.observation, hand).rotation } : undefined
    );
    const establishContact = motionKeyframe(
      contactEstablishmentSeconds,
      { [hand]: path[0]! },
      input.coordination,
      taskToleranceMeters,
      kinematicScope,
      undefined,
      servoMode,
      pathOrientations ? { [hand]: pathOrientations[0]! } : undefined
    );
    const trajectoryFrames = path.slice(1).map((target, index, movingPath) => motionKeyframe(
      index === movingPath.length - 1
        ? duration
        : contactEstablishmentSeconds
          + motionDuration * (index + 1) / movingPath.length,
      { [hand]: target },
      input.coordination,
      taskToleranceMeters,
      kinematicScope,
      rootVelocities?.[index],
      servoMode,
      pathOrientations ? { [hand]: pathOrientations[index + 1]! } : undefined
    ));
    const keyframes = [initial, establishContact, ...trajectoryFrames];
    return {
      id: motionCandidateId(input, candidateIndex),
      intent: candidateIndex === 0
        ? "direct contact-preserving articulation trajectory"
        : candidateIndex === 1
          ? "balanced contact-preserving articulation trajectory"
          : "contact-guided mobile articulation trajectory",
      duration_seconds: duration,
      contact_constraints: input.contactConstraints ?? [],
      keyframes
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
      id: motionCandidateId(input, index),
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
  coordination: G1HandCoordination | null,
  toleranceMeters = 0.04,
  kinematicScope: "arm_only" | "whole_body_reach" = "arm_only",
  rootVelocity?: { forward_mps: number; lateral_mps: number },
  servoMode: "precision" | "task_tolerance" = "precision",
  orientations?: Partial<Record<"left" | "right", Quaternion>>,
  orientationToleranceRadians = 0.22
): HumanoidMotionCandidateBatch["candidates"][number]["keyframes"][number] {
  return {
    at_seconds: atSeconds,
    ...(rootVelocity ? { root_velocity: rootVelocity } : {}),
    ...(coordination ? { hand_coordination: coordination } : {}),
    ...(hands.left ? {
      left_hand: {
        position: hands.left,
        frame: "world",
        tolerance_m: toleranceMeters,
        ...(kinematicScope === "whole_body_reach"
          ? { kinematic_scope: kinematicScope }
          : {}),
        ...(servoMode === "task_tolerance" ? { servo_mode: servoMode } : {}),
        ...(orientations?.left ? {
          orientation: orientations.left,
          orientation_tolerance_rad: orientationToleranceRadians
        } : {})
      }
    } : {}),
    ...(hands.right ? {
      right_hand: {
        position: hands.right,
        frame: "world",
        tolerance_m: toleranceMeters,
        ...(kinematicScope === "whole_body_reach"
          ? { kinematic_scope: kinematicScope }
          : {}),
        ...(servoMode === "task_tolerance" ? { servo_mode: servoMode } : {}),
        ...(orientations?.right ? {
          orientation: orientations.right,
          orientation_tolerance_rad: orientationToleranceRadians
        } : {})
      }
    } : {})
  };
}

function contactFollowingRootVelocities(input: {
  points: readonly Vec3[];
  durationSeconds: number;
  rootYawRadians: number;
}): Array<{ forward_mps: number; lateral_mps: number }> {
  if (input.points.length < 2) {
    return [{ forward_mps: 0, lateral_mps: 0 }];
  }
  const segmentSeconds = input.durationSeconds / (input.points.length - 1);
  return input.points.map((_, index) => {
    const previous = input.points[Math.max(0, index - 1)]!;
    const next = input.points[Math.min(input.points.length - 1, index + 1)]!;
    const divisor = index === 0 || index === input.points.length - 1
      ? segmentSeconds
      : segmentSeconds * 2;
    const worldX = (next.x - previous.x) * 1.05 / divisor;
    const worldZ = (next.z - previous.z) * 1.05 / divisor;
    const forward = worldX * Math.sin(input.rootYawRadians)
      + worldZ * Math.cos(input.rootYawRadians);
    const lateral = worldX * Math.cos(input.rootYawRadians)
      - worldZ * Math.sin(input.rootYawRadians);
    return inversePolicyPlanarVelocity(forward, lateral);
  });
}

function inversePolicyPlanarVelocity(
  desiredForward: number,
  desiredLateral: number
): { forward_mps: number; lateral_mps: number } {
  const desiredSpeed = Math.hypot(desiredForward, desiredLateral);
  if (desiredSpeed <= 0.01) return { forward_mps: 0, lateral_mps: 0 };
  const commandSpeed = Math.min(0.3, Math.max(0.15, desiredSpeed + 0.075));
  const scale = commandSpeed / desiredSpeed;
  return {
    forward_mps: desiredForward * scale,
    lateral_mps: desiredLateral * scale
  };
}

function motionCandidateId(
  input: { binding: ActiveHumanoidSkillBinding; observation: HumanoidWorldObservation },
  candidateIndex: number
): string {
  return `solver:${input.binding.transaction_id}:${input.binding.phase}`
    + `:${input.observation.worldRevision}:${candidateIndex + 1}`;
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
  const hands = new Set(predicates.flatMap((predicate) => {
    if (predicate.type === "grasp_verified"
      || predicate.type === "hand_contact_object_any"
      || predicate.type === "hand_contact_object_region") {
      return [predicate.hand];
    }
    if (predicate.type === "hand_contact_object") {
      return [predicate.hand_surface.startsWith("left_") ? "left" as const : "right" as const];
    }
    return [];
  }));
  if (hands.size === 0 && "hand" in binding.invocation
    && binding.invocation.hand
    && ["push", "pull", "press", "open", "close", "turn"].includes(
      binding.invocation.skill
    )) {
    hands.add(binding.invocation.hand);
  }
  const forceContactRequired = binding.phase === "apply_force"
    || binding.phase === "press_stroke" || binding.phase === "actuate_joint";
  const point = selectedInteractionPoint(binding);
  return [...hands].flatMap((hand) => {
    const anyHandContactRequired = predicates.some((predicate) => (
      (predicate.type === "hand_contact_object_any"
        || predicate.type === "hand_contact_object_region")
        && predicate.hand === hand
        && predicate.object_id === objectId
    ));
    const surfaces = graspSurfaces(observation, hand);
    const reachableSurface = point
      ? preferredReachability(binding, observation, hand, point.id)?.handSurface
      : undefined;
    const nearest = reachableSurface ?? (point
      ? nearestHandSurface(observation, hand, point.world_position)
      : surfaces[0]);
    return surfaces.map((handSurface) => ({
      hand_surface: handSurface,
      object_id: objectId,
      required: predicates.some((predicate) => (
        predicate.type === "grasp_verified" && predicate.hand === hand
          || predicate.type === "hand_contact_object"
            && predicate.hand_surface === handSurface
            && predicate.object_id === objectId
      )) || forceContactRequired && !anyHandContactRequired
        && handSurface === nearest
    }));
  });
}

function articulatedGraspOffset(
  observation: HumanoidWorldObservation,
  hand: "left" | "right",
  fallback: Vec3
): Vec3 {
  const distalNames = [
    `${hand}_hand_thumb_2_link`,
    `${hand}_hand_index_1_link`,
    `${hand}_hand_middle_1_link`
  ];
  const distal = observation.handSurfaces.filter(({ handSurface }) => (
    distalNames.includes(handSurface)
  ));
  if (distal.length !== distalNames.length) return { ...fallback };
  return scale(distal.reduce((sum, surface) => add(
    sum,
    surface.surfaceFromWristWorld
  ), { x: 0, y: 0, z: 0 }), 1 / distal.length);
}

function handContactsObject(
  observation: HumanoidWorldObservation,
  hand: "left" | "right",
  objectId: string
): boolean {
  const prefix = `${hand}_hand_`;
  return observation.robot.contacts.some((contact) => (
    contact.firstHandLink?.startsWith(prefix) === true
      && contact.secondObject === objectId
    || contact.secondHandLink?.startsWith(prefix) === true
      && contact.firstObject === objectId
  ));
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
  const close = (
    hand: "left" | "right",
    mode: "firm" | "compliant" = "firm"
  ) => {
    result[hand] = mode === "firm"
      ? {
          thumb_opposition: 0.85,
          thumb_curl: 0.72,
          index_curl: 0.78,
          middle_curl: 0.8
        }
      : {
          thumb_opposition: 0.78,
          thumb_curl: 0.66,
          index_curl: 0.74,
          middle_curl: 0.76
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
  else if ((invocation.skill === "open" || invocation.skill === "turn"
      || invocation.skill === "pull")
    && (phase === "establish_grasp" || phase === "actuate_joint"
      || phase === "apply_force")) {
    close(invocation.hand, "compliant");
  }
  else if (invocation.skill === "bimanual_support") {
    close("left");
    close("right");
  } else if (invocation.skill === "regrasp" && phase === "transfer_grasp") {
    close(invocation.to_hand);
    open(invocation.from_hand);
  } else if (invocation.skill === "place" && phase === "settle_and_release") {
    if (!invocation.release_after_settled) return null;
    if (invocation.hands === "both") {
      open("left");
      open("right");
    } else open(invocation.hands);
  } else return null;
  return result;
}

function placePredicates(
  invocation: Extract<HumanoidSkillInvocation, { skill: "place" }>,
  motion: SkillMotionTarget,
  observation: HumanoidWorldObservation
): HumanoidMotionOptionPredicate[] {
  if (!motion.objectTarget) throw new Error("Place Skill has no object target");
  const hands = invocation.hands === "both"
    ? ["left", "right"] as const : [invocation.hands];
  const graspState: HumanoidMotionOptionPredicate[] = hands.map((hand) => ({
    type: "grasp_verified" as const,
    object_id: invocation.object_id,
    hand,
    grasp_contract_sha256:
      observation.interaction.grasp_authority.contract_sha256
  }));
  const releaseState: HumanoidMotionOptionPredicate[] = invocation.release_after_settled
    ? hands.map((hand) => ({
        type: "object_released" as const,
        object_id: invocation.object_id,
        hand
      }))
    : [];
  const relation: HumanoidMotionOptionPredicate =
    invocation.destination.type === "semantic_zone"
      ? {
          type: "object_in_zone",
          object_id: invocation.object_id,
          zone_id: invocation.destination.zone_id,
          expected: true,
          tolerance_m: invocation.destination.tolerance_m
        }
      : invocation.destination.type === "container"
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
          target: motion.objectTarget,
          tolerance_m: invocation.destination.type === "world_pose"
            ? invocation.destination.position_tolerance_m : 0.05,
          ...(motion.objectOrientationTarget
            ? {
                target_orientation: motion.objectOrientationTarget,
                orientation_tolerance_rad: 0.18
              }
            : {})
        };
  const pose: HumanoidMotionOptionPredicate = {
    type: "object_near_point",
    object_id: invocation.object_id,
    target: motion.objectTarget,
    tolerance_m: invocation.destination.type === "world_pose"
      ? invocation.destination.position_tolerance_m : 0.05,
    ...(motion.objectOrientationTarget
      ? {
          target_orientation: motion.objectOrientationTarget,
          orientation_tolerance_rad: 0.18
        }
      : {})
  };
  return [
    ...graspState,
    ...releaseState,
    ...(invocation.release_after_settled
      ? [{
          type: "object_settled_on_support" as const,
          object_id: invocation.object_id
        }]
      : []),
    relation,
    ...(relation.type === "object_near_point" ? [] : [pose])
  ];
}

interface PlacementTarget {
  finalPosition: Vec3;
  stagingPosition: Vec3;
  targetRotation?: Quaternion;
}

function placementTarget(
  invocation: Extract<HumanoidSkillInvocation, { skill: "place" }>,
  observation: HumanoidWorldObservation
): PlacementTarget {
  const object = objectEntry(observation, invocation.object_id);
  if (invocation.destination.type === "semantic_zone") {
    const destination = invocation.destination;
    const zone = observation.interaction.zones.find(
      ({ zone_id: zoneId }) => zoneId === destination.zone_id
    );
    if (!zone) throw new Error("Place destination semantic zone is unavailable");
    const worldUp = { x: 0, y: 1, z: 0 };
    const finalPosition = {
      x: zone.center.x,
      y: zone.center.y + zone.size.y / 2
        + objectExtentAlong(object.size, object.pose.rotation, worldUp),
      z: zone.center.z
    };
    return {
      finalPosition,
      stagingPosition: add(finalPosition, {
        x: 0,
        y: placementClearance(object),
        z: 0
      })
    };
  }
  if (invocation.destination.type === "world_pose") {
    const finalPosition = { ...invocation.destination.position };
    return {
      finalPosition,
      stagingPosition: add(finalPosition, { x: 0, y: placementClearance(object), z: 0 })
    };
  }
  const destination = objectEntry(observation, invocation.destination.object_id);
  if (invocation.destination.type === "slot") {
    const interactionPointId = invocation.destination.interaction_point_id;
    const point = destination.interaction_points.find(
      ({ id }) => id === interactionPointId
    );
    if (!point) throw new Error("Placement slot interaction point is unavailable");
    if (!point.approach_direction_world) {
      throw new Error("Placement slot requires an observed insertion direction");
    }
    const insertionDirection = normalize(point.approach_direction_world);
    const finalPosition = add(
      point.world_position,
      scale(insertionDirection, invocation.destination.insertion_depth_m)
    );
    const targetRotation = insertionRotation(object, insertionDirection);
    return {
      finalPosition,
      stagingPosition: add(
        finalPosition,
        scale(insertionDirection, -placementClearance(object, insertionDirection))
      ),
      targetRotation
    };
  }
  if ((invocation.destination.type === "support_surface"
      || invocation.destination.type === "container")
    && invocation.destination.local_target) {
    const finalPosition = add(
      destination.pose.position,
      rotateVector(destination.pose.rotation, invocation.destination.local_target)
    );
    const insertionDirection = invocation.destination.type === "container"
      ? destination.container
        ? scale(normalize(destination.container.opening_direction_world), -1)
        : { x: 0, y: -1, z: 0 }
      : destination.support_surface
        ? scale(normalize(destination.support_surface.normal_world), -1)
        : { x: 0, y: -1, z: 0 };
    return {
      finalPosition,
      stagingPosition: add(
        finalPosition,
        scale(insertionDirection, -placementClearance(object, insertionDirection))
      )
    };
  }
  if (invocation.destination.type === "container") {
    const container = destination.container;
    if (!container) throw new Error("Place destination has no container geometry");
    const insertionDirection = scale(
      normalize(container.opening_direction_world),
      -1
    );
    const finalPosition = { ...container.interior_center_world };
    return {
      finalPosition,
      stagingPosition: add(
        finalPosition,
        scale(insertionDirection, -placementClearance(object, insertionDirection))
      )
    };
  }
  const support = destination.support_surface;
  if (!support) throw new Error("Place destination has no support-surface geometry");
  const normal = normalize(support.normal_world);
  const finalPosition = add(
    support.center_world,
    scale(normal, objectExtentAlong(object.size, object.pose.rotation, normal))
  );
  const insertionDirection = scale(normal, -1);
  return {
    finalPosition,
    stagingPosition: add(
      finalPosition,
      scale(insertionDirection, -placementClearance(object, insertionDirection))
    )
  };
}

function placementClearance(
  object: ReturnType<typeof objectEntry>,
  direction: Vec3 = { x: 0, y: -1, z: 0 }
): number {
  return clamp(
    objectExtentAlong(object.size, object.pose.rotation, normalize(direction)) + 0.08,
    0.12,
    0.45
  );
}

function objectExtentAlong(size: Vec3, rotation: Quaternion, direction: Vec3): number {
  const axes = [
    rotateVector(rotation, { x: size.x / 2, y: 0, z: 0 }),
    rotateVector(rotation, { x: 0, y: size.y / 2, z: 0 }),
    rotateVector(rotation, { x: 0, y: 0, z: size.z / 2 })
  ];
  return axes.reduce((extent, axis) => extent + Math.abs(dot(axis, direction)), 0);
}

function insertionRotation(
  object: ReturnType<typeof objectEntry>,
  insertionDirection: Vec3
): Quaternion {
  const dimensions = [
    { axis: { x: 1, y: 0, z: 0 }, extent: object.size.x },
    { axis: { x: 0, y: 1, z: 0 }, extent: object.size.y },
    { axis: { x: 0, y: 0, z: 1 }, extent: object.size.z }
  ].sort((left, right) => right.extent - left.extent);
  let currentAxis = normalize(rotateVector(object.pose.rotation, dimensions[0]!.axis));
  const targetAxis = normalize(insertionDirection);
  if (dot(currentAxis, targetAxis) < 0) currentAxis = scale(currentAxis, -1);
  const delta = quaternionBetweenDirections(currentAxis, targetAxis);
  return normalizeQuaternion(multiplyQuaternion(delta, object.pose.rotation));
}

function quaternionBetweenDirections(from: Vec3, to: Vec3): Quaternion {
  const cosine = clamp(dot(from, to), -1, 1);
  if (cosine >= 1 - 1e-9) return IDENTITY_QUATERNION;
  if (cosine <= -1 + 1e-9) {
    const candidate = Math.abs(from.x) < 0.8
      ? { x: 1, y: 0, z: 0 }
      : { x: 0, y: 1, z: 0 };
    return axisAngleQuaternion(normalize(cross(from, candidate)), Math.PI);
  }
  const axis = cross(from, to);
  return normalizeQuaternion({ x: axis.x, y: axis.y, z: axis.z, w: 1 + cosine });
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
  const selected = selectedHandSurfaceForPoint(
    binding,
    observation,
    hand,
    point
  );
  const reachable = preferredReachability(binding, observation, hand, point.id);
  if (reachable && reachable.handSurface === selected) {
    return { ...reachable.wristWorldTarget };
  }
  const best = observation.handSurfaces.find(({ handSurface }) => (
    handSurface === selected
  ));
  if (!best) throw new Error(`No observed ${hand} hand contact surface`);
  return subtract(point.world_position, best.surfaceFromWristWorld);
}

function selectedHandSurfaceForPoint(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  hand: "left" | "right",
  point: ActiveHumanoidSkillBinding["eligible_interaction_points"][number]
): G1HandContactSurfaceName | undefined {
  const nearest = observation.handSurfaces
    .filter((surface) => surface.hand === hand)
    .sort((left, right) => (
      distance(left.worldPosition, point.world_position)
        - distance(right.worldPosition, point.world_position)
    ))[0];
  if (nearest && distance(nearest.worldPosition, point.world_position) <= 0.12) {
    return nearest.handSurface;
  }
  return preferredReachability(binding, observation, hand, point.id)?.handSurface
    ?? nearest?.handSurface;
}

function preferredReachability(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  hand: "left" | "right",
  interactionPointId: string
): HumanoidWorldObservation["manipulationReachability"][number] | undefined {
  const objectId = "object_id" in binding.invocation
    ? binding.invocation.object_id : null;
  if (!objectId) return undefined;
  return observation.manipulationReachability
    .filter((entry) => entry.objectId === objectId
      && (entry.interactionPointId === undefined
        || entry.interactionPointId === interactionPointId)
      && entry.handSurface.startsWith(`${hand}_`)
      && (entry.ikReferenceReachable
        || entry.ikResidualMeters !== null && entry.ikResidualMeters <= 0.12))
    .sort((left, right) => (
      distance(left.wristWorldTarget, wristPosition(observation, hand))
        + (left.ikResidualMeters ?? 1) * 0.2
        - distance(right.wristWorldTarget, wristPosition(observation, hand))
        - (right.ikResidualMeters ?? 1) * 0.2
    ))[0];
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

function nearestHandSurface(
  observation: HumanoidWorldObservation,
  hand: "left" | "right",
  target: Vec3
): G1HandContactSurfaceName | undefined {
  return observation.handSurfaces
    .filter((surface) => surface.hand === hand)
    .sort((left, right) => (
      distance(left.worldPosition, target) - distance(right.worldPosition, target)
    ))[0]?.handSurface;
}

function maximumHandSweepRadius(
  observation: HumanoidWorldObservation,
  hand: "left" | "right"
): number {
  const wrist = wristPosition(observation, hand);
  const surfaces = observation.handSurfaces.filter((surface) => (
    surface.hand === hand
  ));
  if (surfaces.length === 0) {
    throw new Error(`Observed ${hand} hand has no contact geometry`);
  }
  return Math.max(...surfaces.map((surface) => distance(
    wrist,
    surface.worldPosition
  )));
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

function wristPose(
  observation: HumanoidWorldObservation,
  hand: "left" | "right"
): { position: Vec3; rotation: Quaternion } {
  const wrist = observation.robot.links[
    hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
  ];
  return {
    position: { ...wrist.position },
    rotation: { ...wrist.rotation }
  };
}

const IDENTITY_QUATERNION: Quaternion = { x: 0, y: 0, z: 0, w: 1 };

function axisAngleQuaternion(axis: Vec3, radians: number): Quaternion {
  const normalized = normalize(axis);
  const sine = Math.sin(radians / 2);
  return {
    x: normalized.x * sine,
    y: normalized.y * sine,
    z: normalized.z * sine,
    w: Math.cos(radians / 2)
  };
}

function approachDirections(preferred?: Vec3): Vec3[] {
  const directions: Vec3[] = preferred
    && Math.hypot(preferred.x, preferred.z) > 1e-9
    ? [{
        x: preferred.x / Math.hypot(preferred.x, preferred.z),
        y: 0,
        z: preferred.z / Math.hypot(preferred.x, preferred.z)
      }] : [];
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
