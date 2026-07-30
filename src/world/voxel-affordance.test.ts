import { describe, expect, it } from "vitest";
import {
  rankVoxelStandoffs,
  voxelAffordanceContractStale,
  withoutVoxelDynamicAffordances,
  type VoxelStandoffCandidate
} from "./voxel-affordance.js";

function candidate(z: number, radius: number): VoxelStandoffCandidate {
  return {
    target: { x: 0, y: 0.38, z },
    radius,
    distanceToEntity: radius,
    distanceToRobot: Math.abs(z),
    axisAlignmentError: 0
  };
}

describe("voxel manipulation affordance ranking", () => {
  it("removes executable voxel geometry from a superseded observation", () => {
    const old = {
      material: null,
      placement_interaction_points: [{ interaction_point: { x: 1, y: 2, z: 3 } }],
      reachable_standoff_poses: [{ target: { x: 4, y: 5, z: 6 } }]
    };
    expect(voxelAffordanceContractStale("inspect_voxel", old)).toBe(true);
    expect(withoutVoxelDynamicAffordances(old)).toMatchObject({
      material: null,
      affordance_contract_stale: true,
      omitted_dynamic_affordances: expect.arrayContaining([
        "placement_interaction_points",
        "reachable_standoff_poses"
      ])
    });
    expect(withoutVoxelDynamicAffordances(old)).not.toHaveProperty(
      "placement_interaction_points"
    );
  });

  it("prefers a working ring for an elevated side placement over a folded minimum ring", () => {
    const ranked = rankVoxelStandoffs({
      candidates: [candidate(1.167, 1.167), candidate(1.617, 1.617)],
      interactions: [{
        normal: { x: 0, y: 0, z: 1 },
        interaction_point: { x: 0, y: 1.35, z: 0.685 }
      }]
    });

    expect(ranked[0]).toMatchObject({
      candidate: { radius: 1.617 },
      fit: "preferred"
    });
    expect(ranked[1]).toMatchObject({
      candidate: { radius: 1.167 },
      fit: "folded"
    });
  });

  it("prefers the minimum ring when a low top face is inside the arm workspace", () => {
    const candidates: VoxelStandoffCandidate[] = [
      {
        ...candidate(0, 1.167),
        target: { x: 1.167, y: 0.38, z: 0 }
      },
      {
        ...candidate(0, 1.617),
        target: { x: 1.617, y: 0.38, z: 0 }
      }
    ];
    const ranked = rankVoxelStandoffs({
      candidates,
      interactions: [{
        normal: { x: 0, y: 1, z: 0 },
        interaction_point: { x: 0, y: 1.085, z: 0 }
      }]
    });

    expect(ranked[0]).toMatchObject({
      candidate: { radius: 1.167 },
      fit: "preferred"
    });
    expect(ranked[1]?.fit).toBe("out_of_span");
  });
});
