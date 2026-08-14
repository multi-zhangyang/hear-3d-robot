import type { Quaternion, Vec3 } from "../../domain/schema.js";
import { yawFromQuaternion } from "../geometry.js";
import type { HumanoidObjectToken } from "./object-memory.js";
import type { HumanoidReference } from "./reference.js";
import type { HumanoidSolidToken } from "./solid-observation.js";
import {
  HumanoidTaskSpaceIkError,
  type HumanoidHandSurfaceObservation,
  type HumanoidSimulation,
  type HumanoidSimulationSnapshot,
  type HumanoidSimulationState
} from "./simulation.js";
import type {
  HumanoidManipulationBasePlacementObservation,
  HumanoidManipulationReachabilityObservation
} from "./world-contract.js";
import { solveG1PregraspPose } from "./pregrasp-pose.js";
import {
  HUMANOID_MANIPULATION_SUPPORT_CLEARANCE_MARGIN_METERS,
  humanoidManipulationBaseNavigationBlockerIds,
  humanoidManipulationSupportSolids
} from "./manipulation-base-navigation.js";
import { HUMANOID_NAVIGATION_PROFILE } from "./environment.js";
import { navigationObstaclePlanarExpansion } from "../navigation.js";

const SURFACES_WITH_BASE_PLACEMENT_PROBES_PER_HAND = 1;
const ROOT_TRANSLATION_DISTANCES_METERS = [0, 0.06, 0.12, 0.18] as const;
const ARM_WORKSPACE_RADII_METERS = [0.28, 0.36, 0.44, 0.52] as const;
const BASE_PLACEMENT_LATERAL_OFFSETS_METERS = [-0.24, -0.12, 0, 0.12, 0.24] as const;
const BASE_PLACEMENT_BOUNDARY_TANGENT_OFFSETS_METERS = [
  -0.18, -0.12, -0.06, 0, 0.06, 0.12, 0.18
] as const;
const BASE_PLACEMENT_BOUNDARY_OUTSIDE_MARGIN_METERS = 0.005;
const NAVIGABLE_BASE_PLACEMENT_YAW_OFFSETS_RADIANS = [
  0, -0.12, 0.12, -0.24, 0.24, -0.36, 0.36
] as const;
const NAVIGABLE_BASE_PLACEMENTS_PER_ALIGNMENT = 8;
const NAVIGABLE_BASE_ROOTS_PER_ALIGNMENT = 4;
const NAVIGABLE_BASE_YAWS_PER_ROOT = 2;
const HUMANOID_PLANAR_BODY_CLEARANCE_METERS = 0.18;

export interface HumanoidManipulationReachabilityMap {
  alignments: HumanoidManipulationReachabilityObservation[];
  basePlacements: HumanoidManipulationBasePlacementObservation[];
}

export function probeHumanoidManipulationReachability(input: {
  simulation: HumanoidSimulation;
  authoritativeState: HumanoidSimulationState;
  reference: HumanoidReference;
  robot: HumanoidSimulationSnapshot;
  objectTokens: readonly HumanoidObjectToken[];
  solidTokens: readonly HumanoidSolidToken[];
  handSurfaces: readonly HumanoidHandSurfaceObservation[];
  interactionTargets?: readonly {
    objectId: string;
    interactionPointId: string;
    worldPosition: Vec3;
    approachDirection?: Vec3;
    preferredGraspAxis?: Vec3;
    clearanceMeters?: number;
  }[];
}): HumanoidManipulationReachabilityMap {
  const visibleObjects = input.objectTokens.filter((object) => (
    object.portable
      && object.status === "visible"
      && object.observable
  ));
  const targets: Array<{
    objectId: string;
    interactionPointId?: string | undefined;
    worldPosition: Vec3;
    approachDirection?: Vec3 | undefined;
    preferredGraspAxis?: Vec3 | undefined;
    clearanceMeters?: number | undefined;
  }> = input.interactionTargets && input.interactionTargets.length > 0
    ? input.interactionTargets.map((target) => ({
        objectId: target.objectId,
        interactionPointId: target.interactionPointId,
        worldPosition: { ...target.worldPosition },
        ...(target.approachDirection
          ? { approachDirection: { ...target.approachDirection } }
          : {}),
        ...(target.preferredGraspAxis
          ? { preferredGraspAxis: { ...target.preferredGraspAxis } }
          : {}),
        ...(target.clearanceMeters !== undefined
          ? { clearanceMeters: target.clearanceMeters }
          : {})
      }))
    : visibleObjects.map((object) => ({
        objectId: object.id,
        worldPosition: { ...object.position }
      }));
  const alignments = targets.flatMap((target) => (
    input.handSurfaces.map((surface) => probeAlignment(input, target, surface))
  ));
  const basePlacements = targets.flatMap((target) => {
    const objectAlignments = alignments.filter((entry) => (
      entry.objectId === target.objectId
        && entry.interactionPointId === target.interactionPointId
    ));
    const current = objectAlignments
      .filter((entry) => entry.ikReferenceReachable
        && manipulationBasePreservesBodyClearance(
          input.robot.rootPosition,
          target
        ))
      .map((entry) => currentRootPlacement(input.robot, entry));
    if (current.length > 0) return current;
    return basePlacementProbeAlignments(objectAlignments)
      .flatMap((alignment) => {
        const surface = input.handSurfaces.find((entry) => (
          entry.handSurface === alignment.handSurface
        ));
        if (!surface) return [];
        const placements = reachableRootPlacements(
          input,
          target,
          alignment,
          surface
        );
        return placements;
      });
  });
  input.simulation.restoreState(input.authoritativeState);
  return { alignments, basePlacements };
}

export function basePlacementProbeAlignments(
  alignments: readonly HumanoidManipulationReachabilityObservation[]
): HumanoidManipulationReachabilityObservation[] {
  return (["left", "right"] as const).flatMap((hand) => (
    alignments
      .filter((entry) => (
        entry.ikResidualMeters !== null
          && entry.handSurface.startsWith(`${hand}_`)
      ))
      .sort((left, right) => left.ikResidualMeters! - right.ikResidualMeters!)
      .slice(0, SURFACES_WITH_BASE_PLACEMENT_PROBES_PER_HAND)
  ));
}

function probeAlignment(
  input: Parameters<typeof probeHumanoidManipulationReachability>[0],
  target: {
    objectId: string;
    interactionPointId?: string | undefined;
    worldPosition: Vec3;
    approachDirection?: Vec3 | undefined;
    preferredGraspAxis?: Vec3 | undefined;
    clearanceMeters?: number | undefined;
  },
  surface: HumanoidHandSurfaceObservation
): HumanoidManipulationReachabilityObservation {
  const wrist = input.robot.links[
    surface.hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
  ];
  const pregrasp = target.approachDirection
    ? solveG1PregraspPose({
        hand: surface.hand,
        wristRotation: wrist.rotation,
        handSurfaces: input.handSurfaces,
        interactionPoint: target.worldPosition,
        approachDirection: target.approachDirection,
        ...(target.preferredGraspAxis
          ? { preferredGraspAxis: target.preferredGraspAxis }
          : {})
      })
    : null;
  const wristWorldTarget = pregrasp?.position
    ?? subtract(target.worldPosition, surface.surfaceFromWristWorld);
  try {
    const solution = solveWrist(
      input.simulation,
      input.reference,
      surface,
      wristWorldTarget,
      undefined,
      pregrasp?.rotation
    );
    return {
      objectId: target.objectId,
      ...(target.interactionPointId
        ? { interactionPointId: target.interactionPointId }
        : {}),
      handSurface: surface.handSurface,
      wristWorldTarget,
      ...(pregrasp ? { wristWorldOrientation: pregrasp.rotation } : {}),
      ikReferenceReachable: true,
      ikResidualMeters: solution
    };
  } catch (error) {
    return {
      objectId: target.objectId,
      ...(target.interactionPointId
        ? { interactionPointId: target.interactionPointId }
        : {}),
      handSurface: surface.handSurface,
      wristWorldTarget,
      ...(pregrasp ? { wristWorldOrientation: pregrasp.rotation } : {}),
      ikReferenceReachable: false,
      ikResidualMeters: ikResidual(error)
    };
  }
}

function reachableRootPlacements(
  input: Parameters<typeof probeHumanoidManipulationReachability>[0],
  target: {
    objectId: string;
    interactionPointId?: string | undefined;
    worldPosition: Vec3;
    approachDirection?: Vec3 | undefined;
    preferredGraspAxis?: Vec3 | undefined;
    clearanceMeters?: number | undefined;
  },
  alignment: HumanoidManipulationReachabilityObservation,
  surface: HumanoidHandSurfaceObservation
): HumanoidManipulationBasePlacementObservation[] {
  const wristDelta = subtract(alignment.wristWorldTarget, surface.wristWorldPosition);
  const placements: HumanoidManipulationBasePlacementObservation[] = [];
  const translations = uniqueTranslations([
    ...humanoidManipulationRootTranslationProbes(input.robot, wristDelta),
    ...humanoidManipulationNavigationBoundaryProbes(input, target)
  ]);
  for (const translation of translations) {
    const rootWorldTarget = add(input.robot.rootPosition, translation);
    if (!manipulationBasePreservesBodyClearance(rootWorldTarget, target)) {
      continue;
    }
    const currentYaw = yawFromQuaternion(input.robot.rootRotation);
    const alignedRootYaw = handAlignedRootYaw(
      input.robot,
      rootWorldTarget,
      target.worldPosition,
      surface
    );
    const navigationClear = humanoidManipulationBaseNavigationBlockerIds({
      solidTokens: input.solidTokens,
      objectId: alignment.objectId,
      rootWorldTarget
    }).length === 0;
    const yawOffsets = navigationClear
      ? NAVIGABLE_BASE_PLACEMENT_YAW_OFFSETS_RADIANS
      : [0] as const;
    for (const yawOffset of yawOffsets) {
      input.simulation.restoreState(input.authoritativeState);
      const rootYawRadians = normalizeRadians(alignedRootYaw + yawOffset);
      const wristWorldTarget = alignment.wristWorldOrientation
        ? alignment.wristWorldTarget
        : subtract(target.worldPosition, rotatePlanar(
            surface.surfaceFromWristWorld,
            rootYawRadians - currentYaw
          ));
      try {
        const residual = solveWrist(
          input.simulation,
          input.reference,
          surface,
          wristWorldTarget,
          {
            position: rootWorldTarget,
            yawRadians: rootYawRadians
          },
          alignment.wristWorldOrientation
        );
        placements.push({
          objectId: alignment.objectId,
          ...(alignment.interactionPointId
            ? { interactionPointId: alignment.interactionPointId }
            : {}),
          handSurface: alignment.handSurface,
          rootWorldTarget,
          rootTranslationWorld: translation,
          rootYawRadians,
          wristWorldTarget,
          ikResidualMeters: residual
        });
      } catch (error) {
        const residual = ikResidual(error);
        const orientationResidual = ikOrientationResidual(error);
        if (residual !== null && residual <= 0.12
          && (orientationResidual === null || orientationResidual <= 0.12)) {
          placements.push({
            objectId: alignment.objectId,
            ...(alignment.interactionPointId
              ? { interactionPointId: alignment.interactionPointId }
              : {}),
            handSurface: alignment.handSurface,
            rootWorldTarget,
            rootTranslationWorld: translation,
            rootYawRadians,
            wristWorldTarget,
            ikResidualMeters: residual
          });
        }
      }
    }
  }
  const ranked = placements.sort((left, right) => (
    left.ikResidualMeters - right.ikResidualMeters
  ));
  const best = ranked[0];
  const navigable = ranked.filter((placement) => (
    humanoidManipulationBaseNavigationBlockerIds({
      solidTokens: input.solidTokens,
      objectId: placement.objectId,
      rootWorldTarget: placement.rootWorldTarget
    }).length === 0
  ));
  return uniquePlacements([
    ...(best ? [best] : []),
    ...diverseNavigableBasePlacements(navigable)
  ]);
}

function diverseNavigableBasePlacements(
  placements: readonly HumanoidManipulationBasePlacementObservation[]
): HumanoidManipulationBasePlacementObservation[] {
  const byRoot = new Map<
    string,
    HumanoidManipulationBasePlacementObservation[]
  >();
  for (const placement of placements) {
    const key = `${placement.rootWorldTarget.x.toFixed(4)}:`
      + `${placement.rootWorldTarget.y.toFixed(4)}:`
      + placement.rootWorldTarget.z.toFixed(4);
    const variants = byRoot.get(key) ?? [];
    variants.push(placement);
    byRoot.set(key, variants);
  }
  const rootGroups = [...byRoot.values()]
    .sort((left, right) => (
      left[0]!.ikResidualMeters - right[0]!.ikResidualMeters
    ));
  const selected = uniquePlacements(
    rootGroups
      .slice(0, NAVIGABLE_BASE_ROOTS_PER_ALIGNMENT)
      .flatMap((variants) => (
        variants.slice(0, NAVIGABLE_BASE_YAWS_PER_ROOT)
      ))
  );
  if (selected.length >= NAVIGABLE_BASE_PLACEMENTS_PER_ALIGNMENT) {
    return selected.slice(0, NAVIGABLE_BASE_PLACEMENTS_PER_ALIGNMENT);
  }
  return uniquePlacements([
    ...selected,
    ...placements
  ]).slice(0, NAVIGABLE_BASE_PLACEMENTS_PER_ALIGNMENT);
}

function humanoidManipulationNavigationBoundaryProbes(
  input: Parameters<typeof probeHumanoidManipulationReachability>[0],
  target: {
    objectId: string;
    worldPosition: Vec3;
    approachDirection?: Vec3 | undefined;
    clearanceMeters?: number | undefined;
  }
): Vec3[] {
  const supports = humanoidManipulationSupportSolids(
    input.solidTokens,
    target.objectId
  );
  if (supports.length === 0) return [];
  const expansion = navigationObstaclePlanarExpansion(
    HUMANOID_NAVIGATION_PROFILE.radius
  ) + HUMANOID_MANIPULATION_SUPPORT_CLEARANCE_MARGIN_METERS
    + BASE_PLACEMENT_BOUNDARY_OUTSIDE_MARGIN_METERS;
  return supports.flatMap((support) => {
    const sides = manipulationBoundarySides(
      input.robot.rootPosition,
      support.center,
      target.approachDirection
    );
    return sides.flatMap((side) => {
      const normalCoordinate = side.axis === "x"
        ? support.center.x + side.sign * (support.size.x / 2 + expansion)
        : support.center.z + side.sign * (support.size.z / 2 + expansion);
      const tangentCenter = side.axis === "x"
        ? input.robot.rootPosition.z
        : input.robot.rootPosition.x;
      return BASE_PLACEMENT_BOUNDARY_TANGENT_OFFSETS_METERS.flatMap((offset) => {
        const rootWorldTarget = side.axis === "x"
          ? {
              x: normalCoordinate,
              y: input.robot.rootPosition.y,
              z: tangentCenter + offset
            }
          : {
              x: tangentCenter + offset,
              y: input.robot.rootPosition.y,
              z: normalCoordinate
            };
        return manipulationBasePreservesBodyClearance(rootWorldTarget, target)
          && humanoidManipulationBaseNavigationBlockerIds({
            solidTokens: input.solidTokens,
            objectId: target.objectId,
            rootWorldTarget
          }).length === 0
          ? [subtract(rootWorldTarget, input.robot.rootPosition)]
          : [];
      });
    });
  });
}

function manipulationBoundarySides(
  rootPosition: Vec3,
  supportCenter: Vec3,
  approachDirection?: Vec3
): Array<{ axis: "x" | "z"; sign: -1 | 1 }> {
  const planarX = Math.abs(approachDirection?.x ?? 0);
  const planarZ = Math.abs(approachDirection?.z ?? 0);
  if (planarX > 1e-6 || planarZ > 1e-6) {
    if (planarX >= planarZ) {
      return [{ axis: "x", sign: (approachDirection!.x > 0 ? -1 : 1) }];
    }
    return [{ axis: "z", sign: (approachDirection!.z > 0 ? -1 : 1) }];
  }
  const radial = subtract(rootPosition, supportCenter);
  const radialAxis = Math.abs(radial.x) >= Math.abs(radial.z)
    ? "x"
    : "z";
  const radialCoordinate = radialAxis === "x" ? radial.x : radial.z;
  return [{ axis: radialAxis, sign: radialCoordinate < 0 ? -1 : 1 }];
}

function uniquePlacements(
  placements: readonly HumanoidManipulationBasePlacementObservation[]
): HumanoidManipulationBasePlacementObservation[] {
  const unique = new Map(placements.map((placement) => [
    `${placement.objectId}:${placement.interactionPointId ?? ""}:${placement.handSurface}:`
      + `${placement.rootWorldTarget.x.toFixed(4)}:${placement.rootWorldTarget.z.toFixed(4)}:`
      + `${placement.rootYawRadians.toFixed(4)}`,
    placement
  ]));
  return [...unique.values()];
}

export function manipulationBasePreservesBodyClearance(
  rootPosition: Vec3,
  target: {
    worldPosition: Vec3;
    approachDirection?: Vec3 | undefined;
    clearanceMeters?: number | undefined;
  }
): boolean {
  if (!target.approachDirection) return true;
  const requiredStandoff = minimumHumanoidManipulationRootStandoff(target);
  const rootToTarget = subtract(target.worldPosition, rootPosition);
  const planarApproachLength = Math.hypot(
    target.approachDirection.x,
    target.approachDirection.z
  );
  if (planarApproachLength <= 1e-9) {
    return Math.hypot(rootToTarget.x, rootToTarget.z) >= requiredStandoff;
  }
  return (
    rootToTarget.x * target.approachDirection.x
      + rootToTarget.z * target.approachDirection.z
  ) / planarApproachLength >= requiredStandoff;
}

export function minimumHumanoidManipulationRootStandoff(target: {
  clearanceMeters?: number | undefined;
}): number {
  return HUMANOID_PLANAR_BODY_CLEARANCE_METERS
    + (target.clearanceMeters ?? 0);
}

function handAlignedRootYaw(
  robot: HumanoidSimulationSnapshot,
  rootWorldTarget: Vec3,
  interactionPoint: Vec3,
  surface: HumanoidHandSurfaceObservation
): number {
  const delta = subtract(interactionPoint, rootWorldTarget);
  const distance = Math.hypot(delta.x, delta.z);
  const bearing = Math.atan2(delta.x, delta.z);
  if (distance <= 1e-6) return bearing;
  const currentYaw = yawFromQuaternion(robot.rootRotation);
  const currentLeft = { x: Math.cos(currentYaw), y: 0, z: -Math.sin(currentYaw) };
  const observedWristLateral = dot(
    subtract(surface.wristWorldPosition, robot.rootPosition),
    currentLeft
  );
  const desiredLateral = clamp(
    observedWristLateral,
    -Math.min(0.2, distance * 0.72),
    Math.min(0.2, distance * 0.72)
  );
  return bearing - Math.asin(clamp(desiredLateral / distance, -0.72, 0.72));
}

function rotatePlanar(vector: Vec3, radians: number): Vec3 {
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  return {
    x: vector.x * cosine + vector.z * sine,
    y: vector.y,
    z: -vector.x * sine + vector.z * cosine
  };
}

function humanoidManipulationRootTranslationProbes(
  robot: HumanoidSimulationSnapshot,
  wristDelta: Vec3
): Vec3[] {
  const yaw = yawFromQuaternion(robot.rootRotation);
  const forward = { x: Math.sin(yaw), y: 0, z: Math.cos(yaw) };
  const left = { x: Math.cos(yaw), y: 0, z: -Math.sin(yaw) };
  const lateralDirection = signedDirection(left, dot(wristDelta, left));
  const forwardDirection = signedDirection(forward, dot(wristDelta, forward));
  const diagonalDirection = normalize(add(lateralDirection, forwardDirection));
  const directDirection = normalize(wristDelta);
  const directLateral = { x: directDirection.z, y: 0, z: -directDirection.x };
  const planarDistance = Math.hypot(wristDelta.x, wristDelta.z);
  const adaptiveDistances = ARM_WORKSPACE_RADII_METERS
    .map((radius) => planarDistance - radius)
    .filter((distance) => distance > 0.18);
  const distances = [...new Set([
    ...ROOT_TRANSLATION_DISTANCES_METERS,
    ...adaptiveDistances
  ].map((distance) => Number(distance.toFixed(6))))];
  return uniqueTranslations(distances.flatMap((distance) => [
    ...BASE_PLACEMENT_LATERAL_OFFSETS_METERS.map((offset) => add(
      scale(directDirection, distance),
      scale(directLateral, offset)
    )),
    scale(diagonalDirection, distance),
    scale(forwardDirection, distance),
    scale(lateralDirection, distance)
  ]));
}

function uniqueTranslations(translations: readonly Vec3[]): Vec3[] {
  const unique = new Map(translations.map((translation) => [
    `${translation.x.toFixed(4)}:${translation.z.toFixed(4)}`,
    translation
  ]));
  return [...unique.values()];
}

function solveWrist(
  simulation: HumanoidSimulation,
  reference: HumanoidReference,
  surface: HumanoidHandSurfaceObservation,
  wristWorldTarget: Vec3,
  planningRootPose?: {
    position: Vec3;
    yawRadians: number;
  },
  wristWorldOrientation?: Quaternion
): number {
  const solution = simulation.solveEndEffectorTargets(reference, [{
    body: surface.hand === "left"
      ? "left_wrist_yaw_link"
      : "right_wrist_yaw_link",
    position: wristWorldTarget,
    frame: "world",
    tolerance: 0.02,
    ...(wristWorldOrientation
      ? {
          orientation: wristWorldOrientation,
          orientationTolerance: 0.08
        }
      : {}),
    ...(planningRootPose ? { kinematicScope: "whole_body_reach" as const } : {})
  }], planningRootPose ? { planningRootPose } : {});
  return solution.residuals[0]?.error ?? 0;
}

function ikResidual(error: unknown): number | null {
  return error instanceof HumanoidTaskSpaceIkError
    ? error.residuals[0]?.error ?? null
    : null;
}

function ikOrientationResidual(error: unknown): number | null {
  return error instanceof HumanoidTaskSpaceIkError
    ? error.residuals[0]?.orientationError ?? null
    : null;
}

function currentRootPlacement(
  robot: HumanoidSimulationSnapshot,
  alignment: HumanoidManipulationReachabilityObservation
): HumanoidManipulationBasePlacementObservation {
  return {
    objectId: alignment.objectId,
    ...(alignment.interactionPointId
      ? { interactionPointId: alignment.interactionPointId }
      : {}),
    handSurface: alignment.handSurface,
    rootWorldTarget: { ...robot.rootPosition },
    rootTranslationWorld: { x: 0, y: 0, z: 0 },
    rootYawRadians: yawFromQuaternion(robot.rootRotation),
    wristWorldTarget: { ...alignment.wristWorldTarget },
    ikResidualMeters: alignment.ikResidualMeters ?? 0
  };
}

function dot(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalizeRadians(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function add(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x + right.x, y: left.y + right.y, z: left.z + right.z };
}

function subtract(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function scale(value: Vec3, multiplier: number): Vec3 {
  return { x: value.x * multiplier, y: 0, z: value.z * multiplier };
}

function signedDirection(value: Vec3, signSource: number): Vec3 {
  return scale(value, signSource < 0 ? -1 : 1);
}

function normalize(value: Vec3): Vec3 {
  const length = Math.hypot(value.x, value.z);
  return length <= 1e-9
    ? { x: 0, y: 0, z: 0 }
    : { x: value.x / length, y: 0, z: value.z / length };
}
