import { describe, expect, it } from "vitest";
import { createG1HandArtifactCommand } from "./hand-coordination.js";
import {
  contactAwareG1GraspTargets,
  contactAwareG1WristAdmittanceTargets,
  contactAwareG1GraspTargetsForOption,
  g1GraspAcquisitionContactEvidence
} from "./contact-aware-grasp-servo.js";
import { DEFAULT_HUMANOID_GRASP_CONTRACT } from "./grasp-registry.js";
import { neutralHumanoidReference } from "./reference.js";
import { HumanoidSimulation } from "./simulation.js";

describe("contact-aware G1 grasp servo", () => {
  it("unloads an over-force wrist target along the measured contact normal", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const snapshot = structuredClone(simulation.snapshot());
      snapshot.contacts = [{
        position: { x: 0, y: 1, z: 0 },
        normal: { x: 0, y: 0, z: 1 },
        normalForce: 57.5,
        firstBody: null,
        secondBody: null,
        firstObject: null,
        secondObject: "fixture",
        firstHandLink: "right_hand_palm_link",
        secondHandLink: null
      }];
      const target = {
        body: "right_wrist_yaw_link" as const,
        position: { x: 1, y: 1, z: 2 },
        frame: "world" as const,
        tolerance: 0.05
      };
      const [adjusted] = contactAwareG1WristAdmittanceTargets({
        snapshot,
        taskSpaceTargets: [target],
        graspTargets: [{
          objectId: "fixture",
          hand: "right",
          minimumNormalForceN: 5,
          minimumDistinctContactSurfaces: 2
        }]
      });

      expect(adjusted?.position).toEqual({ x: 1, y: 1, z: 1.9975 });
      expect(target.position.z).toBe(2);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("force-limits an articulated interaction contact without portable-object authority", () => {
    const targets = contactAwareG1GraspTargetsForOption({
      option: {
        option_id: "actuate-articulation",
        predicates: [{
          type: "hand_contact_object_any",
          hand: "right",
          object_id: "articulated-object",
          minimum_normal_force: 1
        }],
        stable_steps: 4,
        phases: null
      },
      graspContract: DEFAULT_HUMANOID_GRASP_CONTRACT
    });

    expect(targets).toEqual([{
      objectId: "articulated-object",
      hand: "right",
      minimumNormalForceN:
        DEFAULT_HUMANOID_GRASP_CONTRACT.minimum_contact_normal_force_n,
      acquisitionNormalForceN: 1,
      minimumDistinctContactSurfaces: 1,
      holdOnAcquisition: false
    }]);
  });

  it("carries an option interaction region into the tactile servo target", () => {
    const targets = contactAwareG1GraspTargetsForOption({
      option: {
        option_id: "reach-handle",
        predicates: [{
          type: "hand_contact_object_region",
          hand: "right",
          object_id: "door",
          center_world: { x: 1, y: 1.2, z: 2 },
          maximum_distance_m: 0.07,
          minimum_normal_force: 1,
          minimum_distinct_surfaces: 2
        }],
        stable_steps: 4,
        phases: null
      },
      graspContract: DEFAULT_HUMANOID_GRASP_CONTRACT
    });

    expect(targets).toEqual([expect.objectContaining({
      objectId: "door",
      hand: "right",
      minimumDistinctContactSurfaces: 2,
      holdOnAcquisition: true,
      contactRegion: {
        centerWorld: { x: 1, y: 1.2, z: 2 },
        maximumDistanceM: 0.07
      }
    })]);
  });

  it("holds only a contact-establishment option, not a moving articulation", () => {
    const contact = {
      type: "hand_contact_object_any" as const,
      hand: "right" as const,
      object_id: "door",
      minimum_normal_force: 1,
      minimum_distinct_surfaces: 2
    };
    const establish = contactAwareG1GraspTargetsForOption({
      option: {
        option_id: "establish",
        predicates: [contact, {
          type: "root_near_point",
          target: { x: 0, y: 0, z: 0 },
          tolerance_m: 0.1
        }],
        stable_steps: 4,
        phases: null
      },
      graspContract: DEFAULT_HUMANOID_GRASP_CONTRACT
    });
    const actuate = contactAwareG1GraspTargetsForOption({
      option: {
        option_id: "actuate",
        predicates: [contact, {
          type: "articulation_displaced",
          object_id: "door",
          joint_id: "hinge",
          origin_position: 0,
          direction: "increasing",
          minimum_delta: 0.1
        }],
        stable_steps: 4,
        phases: null
      },
      graspContract: DEFAULT_HUMANOID_GRASP_CONTRACT
    });

    expect(establish[0]?.holdOnAcquisition).toBe(true);
    expect(actuate[0]?.holdOnAcquisition).toBe(false);
  });

  it("holds the measured wrist pose at first contact while the manifold closes", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const snapshot = structuredClone(simulation.snapshot());
      const wrist = snapshot.links.right_wrist_yaw_link;
      wrist.position = { x: 0.4, y: 1.1, z: -0.2 };
      snapshot.contacts = [{
          position: { x: 0.45, y: 1.1, z: -0.2 },
          normal: { x: 1, y: 0, z: 0 },
          normalForce: 2,
          firstBody: null,
          secondBody: null,
          firstObject: null,
          secondObject: "handle",
          firstHandLink: "right_hand_index_1_link",
          secondHandLink: null
      }];
      const graspTarget = {
        objectId: "handle",
        hand: "right" as const,
        minimumNormalForceN: 5,
        acquisitionNormalForceN: 1,
        minimumDistinctContactSurfaces: 2,
        holdOnAcquisition: true
      };
      expect(g1GraspAcquisitionContactEvidence({
        snapshot,
        target: graspTarget
      })).toMatchObject({
        contactEstablished: true,
        acquired: false,
        distinctContactSurfaces: 1
      });
      const [held] = contactAwareG1WristAdmittanceTargets({
        snapshot,
        taskSpaceTargets: [{
          body: "right_wrist_yaw_link",
          position: { x: 0.6, y: 1.1, z: -0.2 },
          frame: "world",
          tolerance: 0.05
        }],
        graspTargets: [graspTarget]
      });
      expect(held?.position).toEqual(wrist.position);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("ignores same-object contacts outside the declared interaction region", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const snapshot = structuredClone(simulation.snapshot());
      const wrist = snapshot.links.right_wrist_yaw_link;
      wrist.position = { x: 0.4, y: 1.1, z: -0.2 };
      snapshot.contacts = [{
        position: { x: 0.45, y: 0.55, z: -0.2 },
        normal: { x: 1, y: 0, z: 0 },
        normalForce: 40,
        firstBody: null,
        secondBody: null,
        firstObject: null,
        secondObject: "door",
        firstHandLink: "right_hand_index_1_link",
        secondHandLink: null
      }];
      const graspTarget = {
        objectId: "door",
        hand: "right" as const,
        minimumNormalForceN: 5,
        acquisitionNormalForceN: 1,
        minimumDistinctContactSurfaces: 2,
        holdOnAcquisition: true,
        contactRegion: {
          centerWorld: { x: 0.45, y: 1.1, z: -0.2 },
          maximumDistanceM: 0.08
        }
      };
      expect(g1GraspAcquisitionContactEvidence({
        snapshot,
        target: graspTarget
      })).toEqual({
        contactEstablished: false,
        acquired: false,
        distinctContactSurfaces: 0,
        maximumNormalForceN: 0
      });
      const commanded = {
        body: "right_wrist_yaw_link" as const,
        position: { x: 0.6, y: 1.1, z: -0.2 },
        frame: "world" as const,
        tolerance: 0.05
      };
      const [unheld] = contactAwareG1WristAdmittanceTargets({
        snapshot,
        taskSpaceTargets: [commanded],
        graspTargets: [graspTarget]
      });
      expect(unheld?.position).toEqual(commanded.position);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

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
  }, 30_000);

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
        normalForce: 6,
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

  it("redistributes digit closure to oppose carried-object translation", async () => {
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
      wrist.linearVelocity = { x: 0, y: 0, z: 0 };
      snapshot.objects.crate = {
        id: "crate",
        position: { x: 0, y: -0.013, z: 0 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        linearVelocity: { x: 0, y: 0, z: 0 },
        angularVelocity: { x: 0, y: 0, z: 0 }
      };
      const positions = {
        left_hand_thumb_2_joint: 0.6,
        left_hand_middle_1_joint: -0.6,
        left_hand_index_1_joint: -0.6
      } as const;
      for (const [joint, position] of Object.entries(positions)) {
        snapshot.hands.joints[joint as keyof typeof snapshot.hands.joints].position = position;
        snapshot.hands.joints[joint as keyof typeof snapshot.hands.joints].target = position;
      }
      snapshot.contacts = [{
        position: { x: -0.03, y: 0, z: 0 },
        normal: { x: 0, y: 1, z: 0 },
        normalForce: 20,
        firstBody: null,
        secondBody: null,
        firstObject: "crate",
        secondObject: null,
        firstHandLink: null,
        secondHandLink: "left_hand_thumb_2_link"
      }, {
        position: { x: 0.03, y: 0, z: 0 },
        normal: { x: 0, y: -1, z: 0 },
        normalForce: 5,
        firstBody: null,
        secondBody: null,
        firstObject: "crate",
        secondObject: null,
        firstHandLink: null,
        secondHandLink: "left_hand_middle_1_link"
      }, {
        position: { x: 0, y: 0, z: 0.03 },
        normal: { x: 0, y: -1, z: 0 },
        normalForce: 10,
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
        .toBe(positions.left_hand_thumb_2_joint);
      expect(result.jointTargets.left_hand_middle_1_joint)
        .toBeLessThan(positions.left_hand_middle_1_joint);
      expect(result.jointTargets.left_hand_index_1_joint)
        .toBeLessThan(positions.left_hand_index_1_joint);
      expect(result.evidence).toMatchObject({
        poseRegulatedHands: ["left"],
        maximumTranslationErrorMeters: 0.013
      });
    } finally {
      await simulation.dispose();
    }
  });

});
