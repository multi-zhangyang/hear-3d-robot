import type { Quaternion, Vec3 } from "../../domain/schema.js";
import { orientedBoxWorldHalfExtents } from "../geometry.js";

export interface HumanoidObjectZoneRelation {
  inside: boolean;
  centerOffset: Vec3;
  worldHalfExtents: Vec3;
  horizontalClearance: {
    x: number;
    z: number;
    minimum: number;
  };
  objectBottomY: number;
  zoneSurfaceY: number;
  supportHeightError: number;
  tolerance: number;
}

export function humanoidObjectZoneRelation(input: {
  object: {
    position: Vec3;
    rotation: Quaternion;
    size: Vec3;
  };
  zone: {
    center: Vec3;
    size: Vec3;
  };
  tolerance: number;
}): HumanoidObjectZoneRelation {
  if (!Number.isFinite(input.tolerance) || input.tolerance < 0) {
    throw new Error("Humanoid object-zone tolerance must be finite and non-negative");
  }
  const halfExtents = orientedBoxWorldHalfExtents(
    input.object.size,
    input.object.rotation
  );
  const centerOffset = {
    x: input.object.position.x - input.zone.center.x,
    y: input.object.position.y - input.zone.center.y,
    z: input.object.position.z - input.zone.center.z
  };
  const horizontalClearance = {
    x: input.zone.size.x / 2 + input.tolerance
      - Math.abs(centerOffset.x) - halfExtents.x,
    z: input.zone.size.z / 2 + input.tolerance
      - Math.abs(centerOffset.z) - halfExtents.z
  };
  const objectBottomY = input.object.position.y - halfExtents.y;
  const zoneSurfaceY = input.zone.center.y + input.zone.size.y / 2;
  const supportHeightError = objectBottomY - zoneSurfaceY;
  return {
    inside: horizontalClearance.x >= 0
      && horizontalClearance.z >= 0
      && Math.abs(supportHeightError) <= Math.max(input.tolerance, 0.025),
    centerOffset,
    worldHalfExtents: halfExtents,
    horizontalClearance: {
      ...horizontalClearance,
      minimum: Math.min(horizontalClearance.x, horizontalClearance.z)
    },
    objectBottomY,
    zoneSurfaceY,
    supportHeightError,
    tolerance: input.tolerance
  };
}

export function humanoidObjectInsideZone(input: Parameters<
  typeof humanoidObjectZoneRelation
>[0]): boolean {
  return humanoidObjectZoneRelation(input).inside;
}
