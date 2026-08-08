import { describe, expect, it } from "vitest";
import { rotateVector } from "../geometry.js";
import { solveG1PregraspPose } from "./pregrasp-pose.js";
import type { HumanoidHandSurfaceObservation } from "./simulation.js";

describe("G1 pregrasp pose", () => {
  it("aligns hand forward and curl axes while preserving distal contact geometry", () => {
    const surfaces = [
      surface("right_hand_thumb_2_link", { x: 0.03, y: 0, z: 0.15 }),
      surface("right_hand_index_1_link", { x: 0, y: 0.03, z: 0.18 }),
      surface("right_hand_middle_1_link", { x: 0, y: -0.03, z: 0.18 })
    ];
    const point = { x: 2, y: 1, z: 3 };
    const pose = solveG1PregraspPose({
      hand: "right",
      wristRotation: { x: 0, y: 0, z: 0, w: 1 },
      handSurfaces: surfaces,
      interactionPoint: point,
      approachDirection: { x: 0, y: 0, z: 1 },
      preferredGraspAxis: { x: 0, y: 1, z: 0 }
    });

    expect(rotateVector(pose.rotation, { x: 0, y: 0, z: 1 })).toEqual({
      x: 0,
      y: 0,
      z: 1
    });
    expect(pose.graspAxisWorld.y).toBeCloseTo(1, 8);
    const averageLocal = { x: 0.01, y: 0, z: 0.17 };
    const achieved = add(pose.position, rotateVector(pose.rotation, averageLocal));
    expect(achieved.x).toBeCloseTo(point.x, 8);
    expect(achieved.y).toBeCloseTo(point.y, 8);
    expect(achieved.z).toBeCloseTo(point.z, 8);
  });

  it("falls back to gravity when the articulation axis is parallel to approach", () => {
    const pose = solveG1PregraspPose({
      hand: "right",
      wristRotation: { x: 0, y: 0, z: 0, w: 1 },
      handSurfaces: [
        surface("right_hand_thumb_2_link", { x: 0, y: 0, z: 0.1 }),
        surface("right_hand_index_1_link", { x: 0, y: 0, z: 0.1 }),
        surface("right_hand_middle_1_link", { x: 0, y: 0, z: 0.1 })
      ],
      interactionPoint: { x: 0, y: 0, z: 1 },
      approachDirection: { x: 0, y: 0, z: 1 },
      preferredGraspAxis: { x: 0, y: 0, z: -1 }
    });

    expect(pose.graspAxisWorld.y).toBeCloseTo(1, 8);
  });

  it("selects the equivalent grasp roll closest to the observed wrist", () => {
    const pose = solveG1PregraspPose({
      hand: "right",
      wristRotation: { x: 0, y: 0, z: 1, w: 0 },
      handSurfaces: [
        surface("right_hand_thumb_2_link", { x: 0, y: 0, z: 0.1 }),
        surface("right_hand_index_1_link", { x: 0, y: 0, z: 0.1 }),
        surface("right_hand_middle_1_link", { x: 0, y: 0, z: 0.1 })
      ],
      interactionPoint: { x: 0, y: 0, z: 1 },
      approachDirection: { x: 0, y: 0, z: 1 },
      preferredGraspAxis: { x: 0, y: 1, z: 0 }
    });

    expect(pose.graspAxisWorld.y).toBeCloseTo(-1, 8);
  });
});

function surface(
  handSurface: HumanoidHandSurfaceObservation["handSurface"],
  offset: { x: number; y: number; z: number }
): HumanoidHandSurfaceObservation {
  return {
    handSurface,
    hand: "right",
    worldPosition: { ...offset },
    worldRotation: { x: 0, y: 0, z: 0, w: 1 },
    wristWorldPosition: { x: 0, y: 0, z: 0 },
    surfaceFromWristWorld: { ...offset }
  };
}

function add(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number }
) {
  return { x: left.x + right.x, y: left.y + right.y, z: left.z + right.z };
}
