import { describe, expect, it } from "vitest";
import { HUMANOID_JOINT_INDEX } from "./model.js";
import { neutralHumanoidReference } from "./reference.js";
import { HumanoidSimulation } from "./simulation.js";
import { contactAwareTaskSpaceCompliance } from "./contact-aware-task-space-compliance.js";

describe("contact-aware task-space compliance", () => {
  it("reduces only the overloaded hand chain tracking authority", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const snapshot = structuredClone(simulation.snapshot());
      snapshot.contacts = [{
        position: { x: 0, y: 1, z: 0 },
        normal: { x: 0, y: 0, z: 1 },
        normalForce: 40,
        firstBody: null,
        secondBody: null,
        firstObject: null,
        secondObject: "fixture",
        firstHandLink: "right_hand_palm_link",
        secondHandLink: null
      }];
      const reference = neutralHumanoidReference();
      reference.jointTrackingWeights.fill(1);
      const compliant = contactAwareTaskSpaceCompliance({
        reference,
        snapshot,
        taskSpaceTargets: [{
          body: "right_wrist_yaw_link",
          position: { x: 1, y: 1, z: 2 },
          frame: "world",
          tolerance: 0.05
        }],
        graspTargets: [{
          objectId: "fixture",
          hand: "right",
          minimumNormalForceN: 5
        }]
      });

      expect(compliant.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get("right_elbow_joint")!
      ]).toBeCloseTo(0.3);
      expect(compliant.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get("left_elbow_joint")!
      ]).toBe(1);
      expect(reference.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get("right_elbow_joint")!
      ]).toBe(1);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("does not soften task tracking for same-object contact outside the target region", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const snapshot = structuredClone(simulation.snapshot());
      snapshot.contacts = [{
        position: { x: 0, y: 0.5, z: 0 },
        normal: { x: 0, y: 0, z: 1 },
        normalForce: 40,
        firstBody: null,
        secondBody: null,
        firstObject: null,
        secondObject: "door",
        firstHandLink: "right_hand_palm_link",
        secondHandLink: null
      }];
      const reference = neutralHumanoidReference();
      reference.jointTrackingWeights.fill(1);
      const compliant = contactAwareTaskSpaceCompliance({
        reference,
        snapshot,
        taskSpaceTargets: [{
          body: "right_wrist_yaw_link",
          position: { x: 1, y: 1, z: 2 },
          frame: "world",
          tolerance: 0.05
        }],
        graspTargets: [{
          objectId: "door",
          hand: "right",
          minimumNormalForceN: 5,
          contactRegion: {
            centerWorld: { x: 0, y: 1, z: 0 },
            maximumDistanceM: 0.08
          }
        }]
      });

      expect(compliant.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get("right_elbow_joint")!
      ]).toBe(1);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);
});
