import type { Quaternion, Vec3 } from "../../domain/schema.js";
import { yawFromQuaternion } from "../geometry.js";
import type { HumanoidObjectToken } from "./object-memory.js";
import type { HumanoidReference } from "./reference.js";
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

const SURFACES_WITH_BASE_PLACEMENT_PROBES_PER_HAND = 1;
const ROOT_TRANSLATION_DISTANCES_METERS = [0.06, 0.12, 0.18] as const;
const ARM_WORKSPACE_RADII_METERS = [0.28, 0.36, 0.44, 0.52] as const;
const BASE_PLACEMENT_LATERAL_OFFSETS_METERS = [-0.24, -0.12, 0, 0.12, 0.24] as const;
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
        const placement = bestReachableRootPlacement(
          input,
          target,
          alignment,
          surface
        );
        return placement ? [placement] : [];
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

function bestReachableRootPlacement(
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
): HumanoidManipulationBasePlacementObservation | null {
  const wristDelta = subtract(alignment.wristWorldTarget, surface.wristWorldPosition);
  const placements: HumanoidManipulationBasePlacementObservation[] = [];
  for (const translation of humanoidManipulationRootTranslationProbes(
    input.robot,
    wristDelta
  )) {
    input.simulation.restoreState(input.authoritativeState);
    const rootWorldTarget = add(input.robot.rootPosition, translation);
    if (!manipulationBasePreservesBodyClearance(rootWorldTarget, target)) {
      continue;
    }
    const currentYaw = yawFromQuaternion(input.robot.rootRotation);
    const rootYawRadians = handAlignedRootYaw(
      input.robot,
      rootWorldTarget,
      target.worldPosition,
      surface
    );
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
      if (residual <= 0.005) return placements.at(-1)!;
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
  return placements.sort((left, right) => (
    left.ikResidualMeters - right.ikResidualMeters
  ))[0] ?? null;
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
