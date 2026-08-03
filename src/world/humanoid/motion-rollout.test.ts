import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import {
  captureHumanoidMotionRolloutFrame,
  createHumanoidMotionRollout,
  detectHumanoidMotionDrift,
  humanoidMotionRolloutSha256
} from "./motion-rollout.js";
import { HumanoidWorld } from "./world.js";

const scenario = ScenarioSchema.parse({
  title: "rollout probe",
  seed: 11,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 5,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [],
  zones: [],
  default_goal: {
    summary: "保持站立",
    predicates: [{
      type: "robot_at",
      target: { x: 2, y: 0, z: 2 },
      tolerance: 1
    }]
  }
});

describe("humanoid physical rollout", () => {
  it("hashes immutable prediction frames and detects whole-body divergence", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot().robot;
      const expected = captureHumanoidMotionRolloutFrame(0.02, snapshot);
      const rollout = createHumanoidMotionRollout([expected]);

      expect(humanoidMotionRolloutSha256(structuredClone(rollout))).toBe(
        humanoidMotionRolloutSha256(rollout)
      );
      expect(detectHumanoidMotionDrift(snapshot, expected).drifted).toBe(false);

      const displaced = structuredClone(expected);
      displaced.rootPosition.x += 0.5;
      const evidence = detectHumanoidMotionDrift(snapshot, displaced);
      expect(evidence).toMatchObject({
        drifted: true,
        rootPositionErrorMeters: 0.5
      });
    } finally {
      await world.dispose();
    }
  });
});
