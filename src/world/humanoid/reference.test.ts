import { describe, expect, it } from "vitest";
import {
  HUMANOID_JOINT_INDEX,
  HUMANOID_JOINT_NAMES
} from "./model.js";
import {
  assertHumanoidReference,
  interpolateReference,
  neutralHumanoidReference,
  releaseReferenceTracking,
  stationaryHumanoidReference,
  targetReference
} from "./reference.js";

describe("humanoid joint tracking references", () => {
  it("leaves the neural policy autonomous at neutral", () => {
    const reference = neutralHumanoidReference();

    expect(reference.jointTrackingWeights).toHaveLength(HUMANOID_JOINT_NAMES.length);
    expect([...reference.jointTrackingWeights]).toEqual(
      Array.from({ length: HUMANOID_JOINT_NAMES.length }, () => 0)
    );
    expect(() => assertHumanoidReference(reference)).not.toThrow();
  });

  it("tracks only explicit joint targets and interpolates their authority", () => {
    const neutral = neutralHumanoidReference();
    const joint = "left_shoulder_pitch_joint" as const;
    const index = HUMANOID_JOINT_INDEX.get(joint)!;
    const targeted = targetReference(neutral, {
      joints: { [joint]: -0.8 }
    });

    expect(targeted.jointTrackingWeights[index]).toBe(1);
    expect([...targeted.jointTrackingWeights].reduce((sum, value) => sum + value, 0)).toBe(1);

    const interpolated = interpolateReference(neutral, targeted, 0.5, 2);
    expect(interpolated.jointTrackingWeights[index]).toBeCloseTo(0.5, 12);
    expect([...interpolated.jointTrackingWeights].every((value, candidate) => (
      candidate === index ? value === 0.5 : value === 0
    ))).toBe(true);
  });

  it("rejects malformed dimensions, weights, and interpolation inputs", () => {
    const wrongLength = neutralHumanoidReference();
    wrongLength.jointTrackingWeights = new Float64Array(HUMANOID_JOINT_NAMES.length - 1);
    expect(() => assertHumanoidReference(wrongLength)).toThrow("invalid joint count");

    const outOfRange = neutralHumanoidReference();
    outOfRange.jointTrackingWeights[0] = 1.01;
    expect(() => assertHumanoidReference(outOfRange)).toThrow("tracking weights");

    const neutral = neutralHumanoidReference();
    expect(() => interpolateReference(neutral, neutral, Number.NaN, 1)).toThrow(
      "progress must be finite"
    );
    expect(() => interpolateReference(neutral, neutral, 0.5, 0)).toThrow(
      "duration must be positive"
    );
  });

  it("returns motion-scoped joints to the learned policy reference", () => {
    const tracked = targetReference(neutralHumanoidReference(), {
      joints: { right_elbow_joint: 0.25 },
      rootVelocity: [0.1, -0.05]
    });

    const released = releaseReferenceTracking(tracked);

    expect([...released.jointPositions]).toEqual([
      ...neutralHumanoidReference().jointPositions
    ]);
    expect([...released.jointVelocities]).toEqual(
      Array.from({ length: HUMANOID_JOINT_NAMES.length }, () => 0)
    );
    expect(released.rootVelocity).toEqual(tracked.rootVelocity);
    expect([...released.jointTrackingWeights]).toEqual(
      Array.from({ length: HUMANOID_JOINT_NAMES.length }, () => 0)
    );
    expect(released.jointPositions).not.toBe(tracked.jointPositions);
    expect(released.jointTrackingWeights).not.toBe(tracked.jointTrackingWeights);
  });

  it("stops root motion while retaining only the last commanded arm posture", () => {
    const moving = targetReference(neutralHumanoidReference(), {
      joints: { left_elbow_joint: 0.2 },
      rootVelocity: [0.32, -0.08],
      rootYawVelocity: 0.45,
      rootHeight: 0.81
    });

    const stationary = stationaryHumanoidReference(moving);

    expect(stationary.rootVelocity).toEqual([0, 0]);
    expect(stationary.rootYawVelocity).toBe(0);
    expect(stationary.rootHeight).toBe(0.81);
    expect([...stationary.jointPositions]).toEqual([...moving.jointPositions]);
    expect(stationary.jointTrackingWeights[
      HUMANOID_JOINT_INDEX.get("left_elbow_joint")!
    ]).toBe(1);
    expect(stationary.jointTrackingWeights[
      HUMANOID_JOINT_INDEX.get("left_knee_joint")!
    ]).toBe(0);
  });

  it("does not turn uncommanded measured joints into persistent actuation", () => {
    const stationary = stationaryHumanoidReference(neutralHumanoidReference());

    const shoulder = HUMANOID_JOINT_INDEX.get("right_shoulder_pitch_joint")!;
    const elbow = HUMANOID_JOINT_INDEX.get("right_elbow_joint")!;
    const knee = HUMANOID_JOINT_INDEX.get("right_knee_joint")!;
    expect(stationary.jointTrackingWeights[shoulder]).toBe(0);
    expect(stationary.jointTrackingWeights[elbow]).toBe(0);
    expect(stationary.jointTrackingWeights[knee]).toBe(0);
  });
});
