import type { Vec3 } from "../domain/schema.js";
import type { StandoffPose } from "./navigation.js";

export interface SelectedStandoffPose {
  pose: StandoffPose;
  distanceToRobot: number;
  axisAlignmentError: number;
}

/**
 * Keep several navmesh-validated distance bands instead of allowing the
 * closest ring to consume the whole result limit.
 *
 * A voxel arm moves in the base-forward vertical plane, so axis-aligned voxel
 * faces are served best by axis-aligned approaches. This function only ranks
 * already walkable candidates; it neither plans nor moves the base.
 */
export function selectDiverseStandoffs(input: {
  poses: readonly StandoffPose[];
  radii: readonly number[];
  around: Vec3;
  robotPosition: Vec3;
  preferAxisAligned: boolean;
  limit: number;
}): SelectedStandoffPose[] {
  const limit = Math.max(0, Math.trunc(input.limit));
  if (limit === 0) return [];

  const rings = input.radii.map((radius) => input.poses
    .filter((pose) => pose.radius === radius)
    .map((pose): SelectedStandoffPose => ({
      pose,
      distanceToRobot: planarDistance(pose.position, input.robotPosition),
      axisAlignmentError: axisAlignmentError(pose.position, input.around)
    }))
    .sort((left, right) =>
      (input.preferAxisAligned
        ? left.axisAlignmentError - right.axisAlignmentError
        : 0)
      || left.distanceToRobot - right.distanceToRobot
      || left.pose.position.x - right.pose.position.x
      || left.pose.position.z - right.pose.position.z
    ));

  const selected: SelectedStandoffPose[] = [];
  for (let offset = 0; selected.length < limit; offset += 1) {
    let added = false;
    for (const ring of rings) {
      const candidate = ring[offset];
      if (!candidate) continue;
      selected.push(candidate);
      added = true;
      if (selected.length >= limit) break;
    }
    if (!added) break;
  }
  return selected;
}

function axisAlignmentError(position: Vec3, around: Vec3): number {
  const dx = Math.abs(position.x - around.x);
  const dz = Math.abs(position.z - around.z);
  const radius = Math.hypot(dx, dz);
  return radius <= 1e-9 ? Number.POSITIVE_INFINITY : Math.min(dx, dz) / radius;
}

function planarDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}
