import { ROBOT_SPEC, type RobotJointState } from "./robot-model.js";

export type ArmIkSeed = Pick<RobotJointState, "shoulder" | "elbow" | "wrist">;

/**
 * Low-discrepancy numerical starts for the IK solver.
 *
 * These are not poses the robot executes. They only prevent a valid Cartesian
 * target from being rejected because a local solver happened to start in a
 * poor basin. Every resulting joint solution still has to satisfy limits and
 * the world's continuous Rapier trajectory check before a plan is returned.
 */
export function armIkSeeds(limit = 48): ArmIkSeed[] {
  const count = Math.max(0, Math.trunc(limit));
  return Array.from({ length: count }, (_, offset) => {
    const index = offset + 1;
    return {
      shoulder: sampleJoint("shoulder", halton(index, 2)),
      elbow: sampleJoint("elbow", halton(index, 3)),
      wrist: sampleJoint("wrist", halton(index, 5))
    };
  });
}

function sampleJoint(joint: keyof ArmIkSeed, fraction: number): number {
  const bounds = ROBOT_SPEC.joints[joint];
  return bounds.minimum + (bounds.maximum - bounds.minimum) * fraction;
}

function halton(index: number, base: number): number {
  let value = 0;
  let denominator = 1;
  let remaining = index;
  while (remaining > 0) {
    denominator *= base;
    value += (remaining % base) / denominator;
    remaining = Math.floor(remaining / base);
  }
  return value;
}
