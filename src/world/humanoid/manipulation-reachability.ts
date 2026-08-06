import type { Vec3 } from "../../domain/schema.js";
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

const SURFACES_WITH_BASE_PLACEMENT_PROBES_PER_HAND = 1;
const ROOT_TRANSLATION_DISTANCES_METERS = [0.06, 0.12, 0.18] as const;

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
}): HumanoidManipulationReachabilityMap {
  const visibleObjects = input.objectTokens.filter((object) => (
    object.portable
      && object.status === "visible"
      && object.observable
  ));
  const alignments = visibleObjects.flatMap((object) => (
    input.handSurfaces.map((surface) => probeAlignment(input, object, surface))
  ));
  const basePlacements = visibleObjects.flatMap((object) => {
    const objectAlignments = alignments.filter((entry) => entry.objectId === object.id);
    const current = objectAlignments
      .filter((entry) => entry.ikReferenceReachable)
      .map((entry) => currentRootPlacement(input.robot, entry));
    if (current.length > 0) return current;
    return basePlacementProbeAlignments(objectAlignments)
      .flatMap((alignment) => {
        const surface = input.handSurfaces.find((entry) => (
          entry.handSurface === alignment.handSurface
        ));
        if (!surface) return [];
        const placement = bestReachableRootPlacement(input, alignment, surface);
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
  object: HumanoidObjectToken,
  surface: HumanoidHandSurfaceObservation
): HumanoidManipulationReachabilityObservation {
  const wristWorldTarget = subtract(object.position, surface.surfaceFromWristWorld);
  try {
    const solution = solveWrist(input.simulation, input.reference, surface, wristWorldTarget);
    return {
      objectId: object.id,
      handSurface: surface.handSurface,
      wristWorldTarget,
      ikReferenceReachable: true,
      ikResidualMeters: solution
    };
  } catch (error) {
    return {
      objectId: object.id,
      handSurface: surface.handSurface,
      wristWorldTarget,
      ikReferenceReachable: false,
      ikResidualMeters: ikResidual(error)
    };
  }
}

function bestReachableRootPlacement(
  input: Parameters<typeof probeHumanoidManipulationReachability>[0],
  alignment: HumanoidManipulationReachabilityObservation,
  surface: HumanoidHandSurfaceObservation
): HumanoidManipulationBasePlacementObservation | null {
  const wristDelta = subtract(alignment.wristWorldTarget, surface.wristWorldPosition);
  const placements: HumanoidManipulationBasePlacementObservation[] = [];
  for (const translation of rootTranslationProbes(input.robot, wristDelta)) {
    input.simulation.restoreState(translatedRootState(
      input.authoritativeState,
      translation
    ));
    try {
      const residual = solveWrist(
        input.simulation,
        input.reference,
        surface,
        alignment.wristWorldTarget
      );
      placements.push({
        objectId: alignment.objectId,
        handSurface: alignment.handSurface,
        rootWorldTarget: add(input.robot.rootPosition, translation),
        rootTranslationWorld: translation,
        rootYawRadians: yawFromQuaternion(input.robot.rootRotation),
        wristWorldTarget: alignment.wristWorldTarget,
        ikResidualMeters: residual
      });
    } catch {
      // The next model-visible probe remains grounded in an independent IK solve.
    }
  }
  return placements.sort((left, right) => (
    left.ikResidualMeters - right.ikResidualMeters
  ))[0] ?? null;
}

function rootTranslationProbes(
  robot: HumanoidSimulationSnapshot,
  wristDelta: Vec3
): Vec3[] {
  const yaw = yawFromQuaternion(robot.rootRotation);
  const forward = { x: Math.sin(yaw), y: 0, z: Math.cos(yaw) };
  const left = { x: Math.cos(yaw), y: 0, z: -Math.sin(yaw) };
  const lateralDirection = signedDirection(left, dot(wristDelta, left));
  const forwardDirection = signedDirection(forward, dot(wristDelta, forward));
  const diagonalDirection = normalize(add(lateralDirection, forwardDirection));
  return ROOT_TRANSLATION_DISTANCES_METERS.flatMap((distance) => [
    scale(lateralDirection, distance),
    scale(forwardDirection, distance),
    scale(diagonalDirection, distance)
  ]);
}

function solveWrist(
  simulation: HumanoidSimulation,
  reference: HumanoidReference,
  surface: HumanoidHandSurfaceObservation,
  wristWorldTarget: Vec3
): number {
  const solution = simulation.solveEndEffectorTargets(reference, [{
    body: surface.hand === "left"
      ? "left_wrist_yaw_link"
      : "right_wrist_yaw_link",
    position: wristWorldTarget,
    frame: "world",
    tolerance: 0.02
  }]);
  return solution.residuals[0]?.error ?? 0;
}

function ikResidual(error: unknown): number | null {
  return error instanceof HumanoidTaskSpaceIkError
    ? error.residuals[0]?.error ?? null
    : null;
}

function translatedRootState(
  state: HumanoidSimulationState,
  translation: Vec3
): HumanoidSimulationState {
  const positions = state.positions.slice();
  positions[0] = (positions[0] ?? 0) + translation.z;
  positions[1] = (positions[1] ?? 0) + translation.x;
  return {
    ...state,
    positions,
    velocities: state.velocities.slice(),
    controls: state.controls.slice(),
    activations: state.activations.slice(),
    accelerationWarmstart: state.accelerationWarmstart.slice(),
    ...(state.requestedActuatorTorques
      ? { requestedActuatorTorques: state.requestedActuatorTorques.slice() }
      : {}),
    ...(state.handCommandTargets
      ? { handCommandTargets: state.handCommandTargets.slice() }
      : {}),
    controller: structuredClone(state.controller)
  };
}

function currentRootPlacement(
  robot: HumanoidSimulationSnapshot,
  alignment: HumanoidManipulationReachabilityObservation
): HumanoidManipulationBasePlacementObservation {
  return {
    objectId: alignment.objectId,
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
