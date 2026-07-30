import { describe, expect, it } from "vitest";
import { planArmTrajectory, type ArmPose } from "./arm-trajectory.js";

const bounds = {
  shoulder: { minimum: -1.5, maximum: 1.5 },
  elbow: { minimum: -2.5, maximum: -0.05 },
  wrist: { minimum: -1.7, maximum: 1.7 }
};

describe("arm trajectory planner", () => {
  it("keeps a collision-free direct joint edge", () => {
    const target: ArmPose = { shoulder: 0.4, elbow: -1.3, wrist: 0.2 };
    const result = planArmTrajectory({
      start: { shoulder: 0, elbow: -1.5, wrist: 0 },
      target,
      bounds,
      isPoseValid: () => true
    });
    expect(result).toMatchObject({ direct: true, waypoints: [target] });
  });

  it("finds a multi-segment route around blocked joint space", () => {
    const start: ArmPose = { shoulder: 0, elbow: -1, wrist: 0 };
    const target: ArmPose = { shoulder: 1, elbow: -1, wrist: 0 };
    const valid = (pose: ArmPose): boolean => !(
      pose.shoulder > 0.32
      && pose.shoulder < 0.68
      && pose.elbow > -1.22
      && pose.elbow < -0.78
    );
    const result = planArmTrajectory({ start, target, bounds, isPoseValid: valid });
    expect("waypoints" in result).toBe(true);
    if (!("waypoints" in result)) return;
    expect(result.direct).toBe(false);
    expect(result.waypoints.length).toBeGreaterThan(1);
    expect(result.waypoints.at(-1)).toEqual(target);
    expect(result.waypoints.some((pose) => Math.abs(pose.elbow + 1) > 0.22)).toBe(true);
    for (const pose of result.waypoints) expect(valid(pose)).toBe(true);
  });
});
