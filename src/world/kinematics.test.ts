import { describe, expect, it } from "vitest";

import { ROBOT_SPEC } from "./robot-model.js";
import { solveEndEffectorTarget } from "./kinematics.js";

const RESTING_JOINTS = {
  head_yaw: 0,
  head_pitch: 0,
  shoulder: 1.25,
  elbow: -2.2,
  wrist: 0.95,
  gripper_aperture: 0.4
};

describe("inverse kinematics residuals", () => {
  it("names orientation as the failing term when only orientation is out of tolerance", () => {
    // From a live run: an attempt converged in position at 0.0248m — inside the
    // 0.025m tolerance — and was refused purely on orientation. Both numbers
    // were reported and neither was labelled, so the agent spent nine calls
    // nudging a target position that was already correct.
    // An identity orientation asks the gripper to stay world-axis-aligned while
    // the arm swings in its vertical plane. At this point the arm reaches the
    // position and cannot hold the rotation — the live run's exact shape.
    const result = solveEndEffectorTarget({
      basePosition: { x: 0, y: ROBOT_SPEC.base.height / 2, z: 0 },
      baseYaw: 0,
      currentJoints: RESTING_JOINTS,
      target: {
        position: { x: 0, y: 0.6, z: 0.8 },
        orientation: { x: 0, y: 0, z: 0, w: 1 }
      }
    });

    expect("code" in result).toBe(true);
    if (!("code" in result)) return;
    expect(result.code).toBe("ik_residual_too_large");
    const detail = result.detail as Record<string, unknown>;
    expect(detail.failing_residual).toBe("orientation");
    expect(detail.position_error as number).toBeLessThanOrEqual(
      detail.position_tolerance as number
    );
    expect(detail.orientation_error as number).toBeGreaterThan(
      detail.orientation_tolerance as number
    );
    // The remedy is the orientation, not the position the agent kept changing.
    expect(String(detail.recovery)).toContain("solve_end_effector_position");
    expect(String(detail.recovery)).not.toContain("plan_base_path");
  });

  it("names position as the failing term and points at base motion", () => {
    const result = solveEndEffectorTarget({
      basePosition: { x: 0, y: ROBOT_SPEC.base.height / 2, z: 0 },
      baseYaw: 0,
      currentJoints: RESTING_JOINTS,
      // Far enough that no joint angles reach it, with no orientation asked for.
      target: { position: { x: 0, y: 0.3, z: 2.4 } }
    });

    expect("code" in result).toBe(true);
    if (!("code" in result)) return;
    const detail = result.detail as Record<string, unknown>;
    if (result.code !== "ik_residual_too_large") {
      // A solver that refuses earlier is fine; it must still not be silent.
      expect(Object.keys(detail).length).toBeGreaterThan(0);
      return;
    }
    expect(detail.failing_residual).toBe("position");
    expect(String(detail.recovery)).toContain("execute_base_plan");
  });
});
