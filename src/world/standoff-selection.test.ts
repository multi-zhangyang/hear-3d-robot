import { describe, expect, it } from "vitest";
import type { StandoffPose } from "./navigation.js";
import { selectDiverseStandoffs } from "./standoff-selection.js";

describe("standoff candidate selection", () => {
  it("keeps distance bands and prioritizes axis-aligned voxel approaches", () => {
    const radii = [1.6, 1.1, 2.0, 2.4];
    const poses: StandoffPose[] = radii.flatMap((radius) => [
      pose(radius, radius, 0),
      pose(radius, radius / Math.SQRT2, radius / Math.SQRT2),
      pose(radius, 0, radius)
    ]);
    // Reproduce the navigation layer's closest-first output. Selection must
    // still preserve the caller's preferred ring order.
    poses.sort((left, right) => left.distance - right.distance);

    const selected = selectDiverseStandoffs({
      poses,
      radii,
      around: { x: 0, y: 0, z: 0 },
      robotPosition: { x: 3, y: 0, z: 0 },
      preferAxisAligned: true,
      limit: 8
    });

    expect(selected[0]?.pose.radius).toBe(1.6);
    expect(selected[0]?.axisAlignmentError).toBe(0);
    expect(new Set(selected.map((candidate) => candidate.pose.radius))).toEqual(
      new Set(radii)
    );
    expect(selected.slice(0, 4).map((candidate) => candidate.pose.radius)).toEqual(radii);
  });
});

function pose(radius: number, x: number, z: number): StandoffPose {
  return { position: { x, y: 0, z }, radius, distance: Math.hypot(x, z) };
}
