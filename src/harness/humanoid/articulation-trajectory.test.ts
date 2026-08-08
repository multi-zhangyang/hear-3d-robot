import { describe, expect, it } from "vitest";
import { solveHumanoidArticulationTrajectory } from "./articulation-trajectory.js";

describe("humanoid articulation trajectory", () => {
  it("follows the observed hinge arc to the requested joint position", () => {
    const trajectory = solveHumanoidArticulationTrajectory({
      articulation: articulation("hinge", 0, {
        axis_world: { x: 0, y: 1, z: 0 },
        anchor_world: { x: 0, y: 1, z: 0 }
      }),
      interactionPoint: { x: 1, y: 1, z: 0 },
      targetPosition: Math.PI / 2
    });
    expect(trajectory.joint_delta).toBeCloseTo(Math.PI / 2);
    expect(trajectory.interaction_waypoints.length).toBeGreaterThan(2);
    expect(trajectory.joint_waypoints.at(-1)).toBeCloseTo(Math.PI / 2);
    expect(trajectory.interaction_waypoints.at(-1)).toMatchObject({
      x: expect.closeTo(0, 6),
      y: 1,
      z: expect.closeTo(-1, 6)
    });
    expect(trajectory.path_length_m).toBeGreaterThan(1.5);
  });

  it("follows the live slide axis instead of a fixed stroke", () => {
    const trajectory = solveHumanoidArticulationTrajectory({
      articulation: articulation("slide", 0.1, {
        axis_world: { x: 0, y: 0, z: -1 },
        anchor_world: { x: 2, y: 1, z: 3 }
      }),
      interactionPoint: { x: 2, y: 1, z: 3 },
      targetPosition: 0.52
    });
    expect(trajectory.joint_delta).toBeCloseTo(0.42);
    expect(trajectory.joint_waypoints.at(-1)).toBeCloseTo(0.52);
    expect(trajectory.interaction_waypoints.at(-1)).toEqual({
      x: 2,
      y: 1,
      z: 2.58
    });
  });

  it("bounds each horizon by task-space travel while preserving the final goal", () => {
    const trajectory = solveHumanoidArticulationTrajectory({
      articulation: articulation("hinge", 0, {
        axis_world: { x: 0, y: 1, z: 0 },
        anchor_world: { x: 0, y: 1, z: 0 }
      }),
      interactionPoint: { x: 1, y: 1, z: 0 },
      targetPosition: Math.PI / 2,
      maximumPathLengthM: 0.14
    });
    expect(trajectory.final_target_position).toBeCloseTo(Math.PI / 2);
    expect(trajectory.joint_target_position).toBeCloseTo(0.14);
    expect(trajectory.joint_delta).toBeCloseTo(0.14);
    expect(trajectory.horizon_complete).toBe(false);
    expect(trajectory.path_length_m).toBeLessThanOrEqual(0.141);
  });
});

function articulation(
  type: "hinge" | "slide",
  position: number,
  geometry: {
    axis_world: { x: number; y: number; z: number };
    anchor_world: { x: number; y: number; z: number };
  }
) {
  return {
    joint_id: "joint",
    parent_object_id: null,
    type,
    semantic: type === "hinge" ? "valve" : "drawer",
    ...geometry,
    position,
    velocity: 0,
    range: { minimum: 0, maximum: 2 },
    closed_position: 0,
    open_position: 1,
    open_fraction: position,
    state: "intermediate" as const
  };
}
