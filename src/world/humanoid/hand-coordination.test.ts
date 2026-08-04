import { describe, expect, it } from "vitest";
import {
  G1HandArtifactCommandSchema,
  createG1HandArtifactCommand,
  interpolateG1HandCoordination,
  resolveG1HandCoordination,
  type G1HandCoordination
} from "./hand-coordination.js";
import { G1_HAND_JOINT_LIMITS, G1_HAND_JOINT_NAMES } from "./morphology.js";

const OPEN_HAND: G1HandCoordination = {
  left: {
    thumb_opposition: 0,
    thumb_curl: 0,
    index_curl: 0,
    middle_curl: 0
  },
  right: {
    thumb_opposition: 0,
    thumb_curl: 0,
    index_curl: 0,
    middle_curl: 0
  }
};

const FULL_COORDINATION: G1HandCoordination = {
  left: {
    thumb_opposition: 1,
    thumb_curl: 1,
    index_curl: 1,
    middle_curl: 1
  },
  right: {
    thumb_opposition: 1,
    thumb_curl: 1,
    index_curl: 1,
    middle_curl: 1
  }
};

describe("G1 hand coordination", () => {
  it("resolves mirrored left and right curls inside every physical joint limit", () => {
    const targets = resolveG1HandCoordination(FULL_COORDINATION);

    expect(targets.left_hand_index_0_joint).toBeLessThan(0);
    expect(targets.left_hand_middle_1_joint).toBeLessThan(0);
    expect(targets.right_hand_index_0_joint).toBeGreaterThan(0);
    expect(targets.right_hand_middle_1_joint).toBeGreaterThan(0);
    expect(targets.left_hand_thumb_2_joint).toBeGreaterThan(0);
    expect(targets.right_hand_thumb_2_joint).toBeLessThan(0);
    for (const name of G1_HAND_JOINT_NAMES) {
      const [minimum, maximum] = G1_HAND_JOINT_LIMITS[name];
      expect(targets[name]).toBeGreaterThanOrEqual(minimum);
      expect(targets[name]).toBeLessThanOrEqual(maximum);
    }
  });

  it("uses bounded smooth interpolation with exact endpoints", () => {
    expect(interpolateG1HandCoordination(OPEN_HAND, FULL_COORDINATION, 0))
      .toEqual(OPEN_HAND);
    expect(interpolateG1HandCoordination(OPEN_HAND, FULL_COORDINATION, 1))
      .toEqual(FULL_COORDINATION);
    const quarter = interpolateG1HandCoordination(
      OPEN_HAND,
      FULL_COORDINATION,
      0.25
    );
    expect(quarter.left.index_curl).toBeCloseTo(0.15625, 12);
    expect(quarter.right.thumb_opposition).toBeCloseTo(0.15625, 12);
    expect(() => interpolateG1HandCoordination(OPEN_HAND, FULL_COORDINATION, 1.01))
      .toThrow(/within \[0, 1\]/);
  });

  it("rejects resolved targets that do not match their coordination source", () => {
    const command = createG1HandArtifactCommand(FULL_COORDINATION);
    const tampered = structuredClone(command);
    tampered.jointTargets.left_hand_index_0_joint += 0.1;

    expect(() => G1HandArtifactCommandSchema.parse(tampered))
      .toThrow(/does not match coordination input/);
  });
});
