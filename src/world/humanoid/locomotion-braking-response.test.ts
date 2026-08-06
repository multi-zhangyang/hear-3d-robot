import { describe, expect, it } from "vitest";
import {
  neutralHumanoidReference,
  targetReference
} from "./reference.js";
import { HumanoidSimulation } from "./simulation.js";

describe("YAHMP locomotion braking response", () => {
  it("stops while explicit arm tracking remains active", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      for (let frame = 0; frame < 100; frame += 1) {
        await simulation.step(neutral);
      }

      const walking = targetReference(neutral, { rootVelocity: [0.22, 0] });
      for (let frame = 0; frame < 250; frame += 1) {
        await simulation.step(walking);
      }
      const steady = simulation.snapshot();
      const stoppedWithTrackedArm = targetReference(walking, {
        rootVelocity: [0, 0],
        joints: {
          left_shoulder_pitch_joint: steady.joints.left_shoulder_pitch_joint.position,
          left_shoulder_roll_joint: steady.joints.left_shoulder_roll_joint.position,
          left_shoulder_yaw_joint: steady.joints.left_shoulder_yaw_joint.position,
          left_elbow_joint: steady.joints.left_elbow_joint.position,
          left_wrist_roll_joint: steady.joints.left_wrist_roll_joint.position,
          left_wrist_pitch_joint: steady.joints.left_wrist_pitch_joint.position,
          left_wrist_yaw_joint: steady.joints.left_wrist_yaw_joint.position
        }
      });

      let final = steady;
      for (let frame = 0; frame < 50; frame += 1) {
        final = await simulation.step(stoppedWithTrackedArm, {
          trackedJointPolicyCommand: "neutral"
        });
      }
      const travelledMeters = Math.hypot(
        final.rootPosition.x - steady.rootPosition.x,
        final.rootPosition.z - steady.rootPosition.z
      );
      const pelvisVelocity = final.links.pelvis.linearVelocity;
      expect(steady.fallen).toBe(false);
      expect(final.fallen).toBe(false);
      expect(Math.hypot(pelvisVelocity.x, pelvisVelocity.z)).toBeLessThan(0.08);
      expect(travelledMeters).toBeLessThan(0.08);
    } finally {
      await simulation.dispose();
    }
  }, 15_000);
});
