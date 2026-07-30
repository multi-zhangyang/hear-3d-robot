import { describe, expect, it } from "vitest";
import { armReachDiagnosis } from "./arm-reach-diagnosis.js";

describe("arm reach diagnosis", () => {
  it("distinguishes a folded too-close target from an out-of-range target", () => {
    const tooClose = armReachDiagnosis({
      base: { x: 71.77354431152344, y: 0.38, z: 25.490863800048828 },
      yaw: -1.5733146953144324,
      target: { x: 71.185, y: 1.35, z: 25.5 }
    });
    expect(tooClose.target_within_reach).toBe(true);
    expect(tooClose.recovery).toContain("farther");
    expect(tooClose.recovery).not.toContain("Drive the base closer");

    const tooFar = armReachDiagnosis({
      base: { x: 0, y: 0.38, z: 0 },
      yaw: 0,
      target: { x: 0, y: 0.9, z: 3 }
    });
    expect(tooFar.target_within_reach).toBe(false);
    expect(tooFar.recovery).toContain("closer");
  });

  it("reports lateral error before suggesting a distance change", () => {
    const diagnosis = armReachDiagnosis({
      base: { x: 0, y: 0.38, z: 0 },
      yaw: 0,
      target: { x: 0.4, y: 1, z: 0.8 }
    });
    expect(Number(diagnosis.arm_motion_plane_lateral_error)).toBeCloseTo(0.4, 5);
    expect(diagnosis.recovery).toContain("sideways");
    expect(diagnosis.recovery).toContain("face_point");
  });
});
