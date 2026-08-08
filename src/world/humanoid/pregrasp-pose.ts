import type { Quaternion, Vec3 } from "../../domain/schema.js";
import {
  add,
  inverseQuaternion,
  quaternionAngularDistance,
  quaternionFromRotationMatrix,
  rotateVector,
  scale,
  subtract,
  vectorLength
} from "../geometry.js";
import type { HumanoidHandSurfaceObservation } from "./simulation.js";

export interface G1PregraspPose {
  position: Vec3;
  rotation: Quaternion;
  graspAxisWorld: Vec3;
}

export function solveG1PregraspPose(input: {
  hand: "left" | "right";
  wristRotation: Quaternion;
  handSurfaces: readonly HumanoidHandSurfaceObservation[];
  interactionPoint: Vec3;
  approachDirection: Vec3;
  preferredGraspAxis?: Vec3;
}): G1PregraspPose {
  const forward = normalize(input.approachDirection, "approach direction");
  const geometricAxis = graspAxis(forward, input.preferredGraspAxis);
  const candidates = [geometricAxis, scale(geometricAxis, -1)].map((axis) => ({
    axis,
    rotation: pregraspRotation(axis, forward)
  }));
  const selected = candidates.sort((left, right) => (
    quaternionAngularDistance(left.rotation, input.wristRotation)
      - quaternionAngularDistance(right.rotation, input.wristRotation)
  ))[0]!;
  const localContactOffset = distalContactOffsetLocal(input);
  return {
    position: subtract(
      input.interactionPoint,
      rotateVector(selected.rotation, localContactOffset)
    ),
    rotation: selected.rotation,
    graspAxisWorld: selected.axis
  };
}

function pregraspRotation(axis: Vec3, forward: Vec3): Quaternion {
  const lateral = normalize(cross(axis, forward), "pregrasp lateral axis");
  return quaternionFromRotationMatrix([
    lateral.x, axis.x, forward.x,
    lateral.y, axis.y, forward.y,
    lateral.z, axis.z, forward.z
  ]);
}

function distalContactOffsetLocal(input: Parameters<
  typeof solveG1PregraspPose
>[0]): Vec3 {
  const names = [
    `${input.hand}_hand_thumb_2_link`,
    `${input.hand}_hand_index_1_link`,
    `${input.hand}_hand_middle_1_link`
  ];
  const surfaces = input.handSurfaces.filter(({ handSurface }) => (
    names.includes(handSurface)
  ));
  if (surfaces.length !== names.length) {
    throw new Error(`Observed ${input.hand} hand is missing distal contact geometry`);
  }
  const inverseWrist = inverseQuaternion(input.wristRotation);
  return scale(surfaces.reduce((sum, surface) => add(
    sum,
    rotateVector(inverseWrist, surface.surfaceFromWristWorld)
  ), { x: 0, y: 0, z: 0 }), 1 / surfaces.length);
}

function graspAxis(forward: Vec3, preferred?: Vec3): Vec3 {
  const candidates = [
    ...(preferred ? [preferred] : []),
    { x: 0, y: 1, z: 0 },
    { x: 1, y: 0, z: 0 },
    { x: 0, y: 0, z: 1 }
  ];
  for (const candidate of candidates) {
    const projected = subtract(candidate, scale(forward, dot(candidate, forward)));
    if (vectorLength(projected) >= 0.2) {
      return normalize(projected, "grasp axis");
    }
  }
  throw new Error("Cannot resolve a grasp axis orthogonal to the approach direction");
}

function normalize(value: Vec3, label: string): Vec3 {
  const length = vectorLength(value);
  if (!Number.isFinite(length) || length <= 1e-9) {
    throw new Error(`${label} must be a finite non-zero vector`);
  }
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
