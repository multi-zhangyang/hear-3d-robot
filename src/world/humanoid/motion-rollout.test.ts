import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { ScenarioSchema } from "../../domain/schema.js";
import { multiplyQuaternion, normalizeQuaternion } from "../geometry.js";
import {
  HumanoidMotionRolloutSchema,
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
      const expected = captureHumanoidMotionRolloutFrame(0.02, snapshot, 2);
      const rollout = createHumanoidMotionRollout([expected], 2);

      expect(rollout).toMatchObject({
        version: 2,
        protocol: "humanoid-motion-rollout-v2"
      });
      expect(expected.handJointPositions).toHaveLength(14);
      expect(expected.handJointTargets).toHaveLength(14);

      expect(humanoidMotionRolloutSha256(structuredClone(rollout))).toBe(
        humanoidMotionRolloutSha256(rollout)
      );
      expect(detectHumanoidMotionDrift(snapshot, expected).drifted).toBe(false);

      const displaced = structuredClone(expected);
      displaced.rootPosition.x += 0.5;
      const evidence = detectHumanoidMotionDrift(snapshot, displaced);
      expect(evidence).toMatchObject({ drifted: true });
      expect(evidence.rootPositionErrorMeters).toBeCloseTo(0.5, 12);

      const orientationOnly = structuredClone(expected);
      orientationOnly.endEffectorRotations!.leftWrist = normalizeQuaternion(
        multiplyQuaternion(
          orientationOnly.endEffectorRotations!.leftWrist,
          { x: 0, y: 0, z: Math.sin(0.3), w: Math.cos(0.3) }
        )
      );
      expect(detectHumanoidMotionDrift(snapshot, orientationOnly)).toMatchObject({
        drifted: true,
        rootPositionErrorMeters: 0,
        maximumEndEffectorErrorMeters: 0,
        maximumEndEffectorOrientationErrorRadians: expect.closeTo(0.6, 10)
      });

      const handPositionOnly = structuredClone(expected);
      handPositionOnly.handJointPositions[0] += 0.5;
      expect(detectHumanoidMotionDrift(snapshot, handPositionOnly)).toMatchObject({
        drifted: true,
        handJointRmsErrorRadians: expect.any(Number),
        maximumHandTargetErrorRadians: 0
      });

      const handTargetOnly = structuredClone(expected);
      handTargetOnly.handJointTargets[0] += 0.1;
      expect(detectHumanoidMotionDrift(snapshot, handTargetOnly)).toMatchObject({
        drifted: true,
        handJointRmsErrorRadians: 0,
        maximumHandTargetErrorRadians: expect.closeTo(0.1, 12)
      });

      const {
        handJointPositions: _handJointPositions,
        handJointTargets: _handJointTargets,
        endEffectorRotations: _endEffectorRotations,
        ...legacyFrame
      } = expected;
      const legacy = HumanoidMotionRolloutSchema.parse({
        version: 1,
        protocol: "humanoid-motion-rollout-v1",
        limits: {
          root_position_m: rollout.limits.root_position_m,
          root_orientation_rad: rollout.limits.root_orientation_rad,
          joint_rms_rad: rollout.limits.joint_rms_rad,
          end_effector_m: rollout.limits.end_effector_m,
          consecutive_steps: rollout.limits.consecutive_steps
        },
        frames: [legacyFrame]
      });
      expect(humanoidMotionRolloutSha256(legacy)).toBe(
        createHash("sha256")
          .update(JSON.stringify(HumanoidMotionRolloutSchema.parse(legacy)))
          .digest("hex")
      );
      expect(detectHumanoidMotionDrift(snapshot, legacy.frames[0]!).drifted)
        .toBe(false);
      const capturedLegacy = captureHumanoidMotionRolloutFrame(0.02, snapshot, 1);
      expect(capturedLegacy).not.toHaveProperty("handJointPositions");
      expect(createHumanoidMotionRollout([capturedLegacy], 1)).toMatchObject({
        version: 1,
        protocol: "humanoid-motion-rollout-v1"
      });
      expect(() => createHumanoidMotionRollout([capturedLegacy], 2))
        .toThrow(/version must match/);
    } finally {
      await world.dispose();
    }
  }, 30_000);
});
