import { describe, expect, it } from "vitest";
import { createG1HandArtifactCommand } from "./hand-coordination.js";
import { contactAwareG1GraspTargets } from "./contact-aware-grasp-servo.js";
import { neutralHumanoidReference } from "./reference.js";
import { HumanoidSimulation } from "./simulation.js";

describe("contact-aware G1 grasp servo", () => {
  it("keeps opening commands exact and bounds free closure per physical frame", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const snapshot = await simulation.step(neutralHumanoidReference());
      const closing = createG1HandArtifactCommand({
        left: {
          thumb_opposition: 1,
          thumb_curl: 1,
          index_curl: 1,
          middle_curl: 1
        },
        right: {
          thumb_opposition: 0,
          thumb_curl: 0,
          index_curl: 0,
          middle_curl: 0
        }
      });
      const result = contactAwareG1GraspTargets({
        command: closing,
        snapshot,
        targets: [{ objectId: "crate", hand: "left", minimumNormalForceN: 5 }]
      });
      expect(Math.abs(
        result.jointTargets.left_hand_index_1_joint
          - snapshot.hands.joints.left_hand_index_1_joint.position
      ))
        .toBeLessThanOrEqual(0.025 + 1e-9);
      expect(result.jointTargets.right_hand_index_1_joint).toBe(0);
      expect(result.evidence).toMatchObject({
        targetObjectIds: ["crate"],
        limitedDigits: [],
        maximumObservedNormalForceN: 0
      });
    } finally {
      await simulation.dispose();
    }
  });

  it("rejects ambiguous targets for one hand", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const snapshot = simulation.snapshot();
      const command = createG1HandArtifactCommand({
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
      });
      expect(() => contactAwareG1GraspTargets({
        command,
        snapshot,
        targets: [
          { objectId: "first", hand: "left", minimumNormalForceN: 5 },
          { objectId: "second", hand: "left", minimumNormalForceN: 5 }
        ]
      })).toThrow("two left targets");
    } finally {
      await simulation.dispose();
    }
  });

  it("holds the existing preload after a digit reaches qualified contact", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const snapshot = structuredClone(simulation.snapshot());
      const closing = createG1HandArtifactCommand({
        left: {
          thumb_opposition: 1,
          thumb_curl: 1,
          index_curl: 1,
          middle_curl: 1
        },
        right: {
          thumb_opposition: 0,
          thumb_curl: 0,
          index_curl: 0,
          middle_curl: 0
        }
      });
      const firstIndex = snapshot.hands.joints.left_hand_index_0_joint;
      const secondIndex = snapshot.hands.joints.left_hand_index_1_joint;
      firstIndex.target = firstIndex.position - 0.02;
      secondIndex.target = secondIndex.position - 0.02;
      snapshot.contacts = [{
        position: { x: 0, y: 0, z: 0 },
        normal: { x: 1, y: 0, z: 0 },
        normalForce: 12,
        firstBody: "left_wrist_yaw_link",
        secondBody: null,
        firstObject: null,
        secondObject: "crate",
        firstHandLink: "left_hand_index_1_link",
        secondHandLink: null
      }];
      snapshot.contactCount = 1;

      const result = contactAwareG1GraspTargets({
        command: closing,
        snapshot,
        targets: [{ objectId: "crate", hand: "left", minimumNormalForceN: 5 }]
      });

      expect(result.jointTargets.left_hand_index_0_joint).toBe(firstIndex.target);
      expect(result.jointTargets.left_hand_index_1_joint).toBe(secondIndex.target);
      expect(result.evidence.limitedDigits).toContain("left:index");
    } finally {
      await simulation.dispose();
    }
  });

  it("uses contact moments to oppose carried-object rotation without tightening saturation", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const snapshot = structuredClone(simulation.snapshot());
      const closing = createG1HandArtifactCommand({
        left: {
          thumb_opposition: 1,
          thumb_curl: 1,
          index_curl: 1,
          middle_curl: 1
        },
        right: {
          thumb_opposition: 0,
          thumb_curl: 0,
          index_curl: 0,
          middle_curl: 0
        }
      });
      const wrist = snapshot.links.left_wrist_yaw_link;
      wrist.position = { x: 0, y: 0, z: 0 };
      wrist.rotation = { x: 0, y: 0, z: 0, w: 1 };
      wrist.angularVelocity = { x: 0, y: 0, z: 0 };
      snapshot.objects.crate = {
        id: "crate",
        position: { x: 0, y: 0, z: 0 },
        rotation: {
          x: 0,
          y: -Math.sin(0.03),
          z: 0,
          w: Math.cos(0.03)
        },
        linearVelocity: { x: 0, y: 0, z: 0 },
        angularVelocity: { x: 0, y: 0, z: 0 }
      };
      const positions = {
        left_hand_thumb_1_joint: 0.4,
        left_hand_thumb_2_joint: 0.6,
        left_hand_middle_0_joint: -0.4,
        left_hand_middle_1_joint: -0.4,
        left_hand_index_0_joint: -0.4,
        left_hand_index_1_joint: -0.4
      } as const;
      for (const [joint, position] of Object.entries(positions)) {
        snapshot.hands.joints[joint as keyof typeof snapshot.hands.joints].position = position;
        snapshot.hands.joints[joint as keyof typeof snapshot.hands.joints].target = position;
      }
      snapshot.contacts = [{
        position: { x: -0.03, y: 0, z: -0.03 },
        normal: { x: -1, y: 0, z: 0 },
        normalForce: 12,
        firstBody: null,
        secondBody: null,
        firstObject: "crate",
        secondObject: null,
        firstHandLink: null,
        secondHandLink: "left_hand_thumb_2_link"
      }, {
        position: { x: 0.03, y: 0, z: -0.03 },
        normal: { x: 1, y: 0, z: 0 },
        normalForce: 12,
        firstBody: null,
        secondBody: null,
        firstObject: "crate",
        secondObject: null,
        firstHandLink: null,
        secondHandLink: "left_hand_middle_1_link"
      }, {
        position: { x: 0.03, y: 0, z: 0.03 },
        normal: { x: 1, y: 0, z: 0 },
        normalForce: 12,
        firstBody: null,
        secondBody: null,
        firstObject: "crate",
        secondObject: null,
        firstHandLink: null,
        secondHandLink: "left_hand_index_1_link"
      }];
      snapshot.contactCount = snapshot.contacts.length;

      const result = contactAwareG1GraspTargets({
        command: closing,
        snapshot,
        targets: [{
          objectId: "crate",
          hand: "left",
          minimumNormalForceN: 5,
          referenceRelativePose: {
            translation: { x: 0, y: 0, z: 0 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          }
        }]
      });

      expect(result.jointTargets.left_hand_thumb_2_joint)
        .toBeLessThan(positions.left_hand_thumb_2_joint);
      expect(result.jointTargets.left_hand_middle_1_joint)
        .toBeLessThan(positions.left_hand_middle_1_joint);
      expect(result.jointTargets.left_hand_index_1_joint)
        .toBeGreaterThan(positions.left_hand_index_1_joint);
      expect(result.evidence).toMatchObject({
        poseRegulatedHands: ["left"],
        saturationLimitedDigits: []
      });
      expect(result.evidence.maximumRotationErrorRadians).toBeCloseTo(0.06, 12);
    } finally {
      await simulation.dispose();
    }
  });

});
