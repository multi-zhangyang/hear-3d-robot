import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import {
  HumanoidMotionOptionContractSchema,
  advanceHumanoidMotionOptionMonitor,
  createHumanoidMotionOptionMonitorState,
  detectHumanoidMotionOption,
  humanoidMotionOptionContractSha256,
  type HumanoidMotionOptionCondition,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionDetectorInput,
  type HumanoidMotionOptionRobotSnapshot
} from "./motion-option.js";
import type {
  HumanoidGraspAssessment,
  HumanoidGraspContract
} from "./grasp-tracker.js";
import { humanoidGraspContractSha256 } from "./grasp-tracker.js";

const robot: HumanoidMotionOptionRobotSnapshot = {
  rootPosition: { x: 1, y: 0.8, z: 2 },
  links: {
    pelvis: {
      position: { x: 1, y: 0.8, z: 2 },
      rotation: {
        x: 0,
        y: Math.SQRT1_2,
        z: 0,
        w: Math.SQRT1_2
      }
    },
    left_wrist_yaw_link: {
      position: { x: 1.2, y: 1.1, z: 2.4 }
    },
    right_wrist_yaw_link: {
      position: { x: 1.4, y: 1.1, z: 1.8 }
    }
  },
  contacts: [{
    normalForce: 14,
    firstBody: "left_wrist_yaw_link",
    secondBody: null,
    firstObject: null,
    secondObject: "crate"
  }]
};

const observableCrate = {
  id: "crate",
  position: { x: 1, y: 0.3, z: 3 },
  rotation: { x: 0, y: 0, z: 0, w: 1 },
  size: { x: 0.5, y: 0.5, z: 0.5 }
};

const destination = {
  id: "destination",
  center: { x: 1, y: 0, z: 3 },
  size: { x: 2, y: 0.1, z: 2 }
};

const contract: HumanoidMotionOptionContract = {
  option_id: "reach-and-place",
  stable_steps: 5,
  predicates: [
    {
      type: "root_near_point",
      target: { x: 1, y: 0.8, z: 2 },
      tolerance_m: 0.05
    },
    {
      type: "body_near_point",
      body: "left_wrist_yaw_link",
      target: { x: 1.2, y: 1.1, z: 2.4 },
      tolerance_m: 0.05
    },
    {
      type: "body_contact_object",
      body: "left_wrist_yaw_link",
      object_id: "crate",
      minimum_normal_force: 10
    },
    {
      type: "object_near_point",
      object_id: "crate",
      target: { x: 1, y: 0.3, z: 3 },
      tolerance_m: 0.05
    },
    {
      type: "object_in_zone",
      object_id: "crate",
      zone_id: "destination",
      expected: true,
      tolerance_m: 0.01
    }
  ]
};

const graspContract = {
  protocol: "humanoid-grasp-contract-v1",
  world_up: { x: 0, y: 1, z: 0 },
  minimum_distinct_contact_links: 2,
  minimum_contact_normal_force_n: 5,
  maximum_opposing_normal_dot: -0.5,
  maximum_opposing_position_dot: -0.5,
  minimum_opposing_contact_separation_m: 0.02,
  minimum_contact_radial_distance_m: 0.005,
  maximum_relative_translation_drift_m: 0.01,
  maximum_relative_rotation_drift_rad: 0.05,
  minimum_relative_pose_stable_frames: 3,
  minimum_lift_m: 0.05,
  minimum_lifted_hold_frames: 2,
  minimum_support_normal_force_n: 2,
  minimum_support_up_dot: 0.7
} as const satisfies HumanoidGraspContract;

const graspOptionContract: HumanoidMotionOptionContract = {
  option_id: "lift-crate",
  stable_steps: 1,
  predicates: [{
    type: "grasp_verified",
    object_id: "crate",
    hand: "left",
    grasp_contract_sha256: humanoidGraspContractSha256(graspContract)
  }]
};

function graspAssessment(overrides: {
  objectId?: string;
  hand?: "left" | "right";
  verified?: boolean;
  reason?: HumanoidGraspAssessment["reason"];
  contactStatus?: HumanoidGraspAssessment["evidence"]["contact"]["status"];
  supportStatus?: HumanoidGraspAssessment["evidence"]["support"]["status"];
  baselineProjection?: number | null;
} = {}): HumanoidGraspAssessment {
  const verified = overrides.verified ?? true;
  const contactStatus = overrides.contactStatus ?? "opposed";
  const supportStatus = overrides.supportStatus ?? "unsupported";
  return {
    protocol: "humanoid-grasp-assessment-v1",
    frame: 42,
    object_id: overrides.objectId ?? "crate",
    hand: overrides.hand ?? "left",
    phase: verified ? "verified" : "idle",
    grasp_verified: verified,
    reason: overrides.reason ?? (verified ? "grasp_verified" : "contact_missing"),
    reset_reason: null,
    evidence: {
      contact: {
        status: contactStatus,
        observed_contact_count: contactStatus === "missing" ? 0 : 2,
        force_qualified_contact_count: contactStatus === "missing" ? 0 : 2,
        distinct_force_qualified_links: contactStatus === "missing"
          ? []
          : ["left_hand_index_1_link", "left_hand_thumb_2_link"],
        distinct_normal_qualified_links: contactStatus === "opposed"
          ? ["left_hand_index_1_link", "left_hand_thumb_2_link"]
          : [],
        opposing_pair: contactStatus === "opposed" ? {
          first_link: "left_hand_index_1_link",
          second_link: "left_hand_thumb_2_link",
          first_position: { x: -0.03, y: 0.5, z: 0 },
          second_position: { x: 0.03, y: 0.5, z: 0 },
          first_normal_from_hand: { x: 1, y: 0, z: 0 },
          second_normal_from_hand: { x: -1, y: 0, z: 0 },
          first_normal_force_n: 10,
          second_normal_force_n: 11,
          separation_m: 0.06,
          normal_dot: -1,
          position_dot: -1
        } : null
      },
      support: {
        status: supportStatus,
        candidate_contact_count: supportStatus === "unsupported" ? 0 : 1,
        force_qualified_contact_count: supportStatus === "unsupported" ? 0 : 1,
        upward_contact_count: supportStatus === "supported" ? 1 : 0,
        baseline_projection_m: overrides.baselineProjection === undefined
          ? 0.5
          : overrides.baselineProjection,
        current_projection_m: 0.56,
        lift_m: overrides.baselineProjection === null ? null : 0.06
      },
      relative_pose: {
        stable_frames: verified ? 5 : 0,
        translation_drift_m: verified ? 0.001 : null,
        rotation_drift_rad: verified ? 0.002 : null
      },
      lifted_hold_frames: verified ? 2 : 0
    }
  };
}

function graspDetectorInput(
  assessment: HumanoidGraspAssessment | null = graspAssessment()
): HumanoidMotionOptionDetectorInput {
  return {
    snapshot: robot,
    observableObjects: [],
    zones: [],
    ...(assessment ? {
      graspAssessments: [{
        predicate_index: 0,
        contract_sha256: humanoidGraspContractSha256(graspContract),
        assessment
      }]
    } : {})
  };
}

describe("humanoid motion option detector", () => {
  it("validates physical predicate contracts and stable steps", () => {
    expect(HumanoidMotionOptionContractSchema.parse(contract)).toEqual({
      ...contract,
      phases: null
    });
    expect(humanoidMotionOptionContractSha256(contract)).toBe(
      createHash("sha256").update(JSON.stringify({
        option_id: contract.option_id,
        predicates: contract.predicates,
        stable_steps: contract.stable_steps
      })).digest("hex")
    );
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      stable_steps: 0
    })).toThrow();
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      predicates: [{
        type: "body_near_point",
        body: "invented_link",
        target: { x: 0, y: 0, z: 0 },
        tolerance_m: 0.1
      }]
    })).toThrow();
  });

  it("validates grasp policy references without exposing mutable thresholds", () => {
    const parsed = HumanoidMotionOptionContractSchema.parse(graspOptionContract);
    expect(parsed.predicates[0]).toEqual(graspOptionContract.predicates[0]);
    expect(humanoidGraspContractSha256(graspContract)).toBe(
      createHash("sha256").update(JSON.stringify(graspContract)).digest("hex")
    );

    const stricter = structuredClone(graspOptionContract);
    const predicate = stricter.predicates[0];
    if (predicate?.type !== "grasp_verified") throw new Error("Missing grasp predicate");
    predicate.grasp_contract_sha256 = humanoidGraspContractSha256({
      ...graspContract,
      minimum_lift_m: 0.08
    });
    expect(humanoidMotionOptionContractSha256(stricter)).not.toBe(
      humanoidMotionOptionContractSha256(graspOptionContract)
    );
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...graspOptionContract,
      predicates: [{
        ...graspOptionContract.predicates[0],
        grasp_contract_sha256: "not-a-policy-hash"
      }]
    })).toThrow();
  });

  it("accepts only a runtime-verified grasp and preserves its complete evidence", () => {
    const assessment = graspAssessment();
    const detection = detectHumanoidMotionOption(
      graspOptionContract,
      graspDetectorInput(assessment)
    );

    expect(detection).toMatchObject({
      status: "satisfied",
      allSatisfied: true,
      hasUncertain: false,
      evidence: [{
        predicateIndex: 0,
        type: "grasp_verified",
        status: "satisfied",
        objectId: "crate",
        hand: "left",
        contractSha256: humanoidGraspContractSha256(graspContract),
        reason: "grasp_verified",
        assessment
      }]
    });
  });

  it("distinguishes a measured failed grasp from missing physical evidence", () => {
    const failed = graspAssessment({
      verified: false,
      reason: "contacts_not_opposed",
      contactStatus: "not_opposed"
    });
    expect(detectHumanoidMotionOption(
      graspOptionContract,
      graspDetectorInput(failed)
    )).toMatchObject({
      status: "unsatisfied",
      hasUncertain: false,
      evidence: [{
        status: "unsatisfied",
        reason: "contacts_not_opposed",
        assessment: failed
      }]
    });

    const uncertainAssessments = [
      graspAssessment({
        verified: false,
        reason: "contact_normal_insufficient",
        contactStatus: "insufficient_normal"
      }),
      graspAssessment({
        verified: false,
        reason: "support_evidence_insufficient",
        supportStatus: "insufficient_normal"
      }),
      graspAssessment({
        verified: false,
        reason: "support_baseline_missing",
        baselineProjection: null
      })
    ];
    for (const assessment of uncertainAssessments) {
      expect(detectHumanoidMotionOption(
        graspOptionContract,
        graspDetectorInput(assessment)
      )).toMatchObject({
        status: "uncertain",
        allSatisfied: false,
        hasUncertain: true,
        evidence: [{ status: "uncertain", reason: assessment.reason, assessment }]
      });
    }
  });

  it("returns uncertain when the current grasp assessment is absent", () => {
    expect(detectHumanoidMotionOption(
      graspOptionContract,
      graspDetectorInput(null)
    )).toMatchObject({
      status: "uncertain",
      allSatisfied: false,
      hasUncertain: true,
      evidence: [{
        type: "grasp_verified",
        status: "uncertain",
        assessment: null,
        reason: "grasp_assessment_missing"
      }]
    });
  });

  it("rejects grasp assessments with duplicate or mismatched authority bindings", () => {
    const valid = graspDetectorInput().graspAssessments![0]!;
    expect(() => detectHumanoidMotionOption(graspOptionContract, {
      ...graspDetectorInput(),
      graspAssessments: [valid, valid]
    })).toThrow(/Duplicate.*predicate index/);
    expect(() => detectHumanoidMotionOption(graspOptionContract, {
      ...graspDetectorInput(),
      graspAssessments: [{
        ...valid,
        assessment: graspAssessment({ objectId: "parcel" })
      }]
    })).toThrow(/object does not match/);
    expect(() => detectHumanoidMotionOption(graspOptionContract, {
      ...graspDetectorInput(),
      graspAssessments: [{
        ...valid,
        assessment: graspAssessment({ hand: "right" })
      }]
    })).toThrow(/hand does not match/);
    expect(() => detectHumanoidMotionOption(graspOptionContract, {
      ...graspDetectorInput(),
      graspAssessments: [{ ...valid, contract_sha256: "0".repeat(64) }]
    })).toThrow(/contract does not match/);

    const duplicateObjectContract: HumanoidMotionOptionContract = {
      ...graspOptionContract,
      predicates: [
        graspOptionContract.predicates[0]!,
        graspOptionContract.predicates[0]!
      ]
    };
    expect(() => detectHumanoidMotionOption(duplicateObjectContract, {
      ...graspDetectorInput(),
      graspAssessments: [valid, { ...valid, predicate_index: 1 }]
    })).toThrow(/Duplicate.*object and hand/);

    expect(() => detectHumanoidMotionOption(contract, {
      snapshot: robot,
      observableObjects: [observableCrate],
      zones: [destination],
      graspAssessments: [valid]
    })).toThrow(/does not reference a grasp predicate/);
  });

  it("evaluates grasp evidence through explicit condition phases", () => {
    const phased = HumanoidMotionOptionContractSchema.parse({
      ...graspOptionContract,
      phases: {
        precondition: null,
        during: {
          condition: { op: "predicate", predicate_index: 0 }
        },
        terminal: {
          condition: { op: "predicate", predicate_index: 0 }
        }
      }
    });
    const detection = detectHumanoidMotionOption(phased, graspDetectorInput());
    expect(detection.phases).toMatchObject({
      precondition: null,
      during: { status: "satisfied", predicateIndexes: [0] },
      terminal: { status: "satisfied", predicateIndexes: [0] }
    });
    expect(advanceHumanoidMotionOptionMonitor(
      phased,
      createHumanoidMotionOptionMonitorState(phased),
      graspDetectorInput()
    ).state).toMatchObject({ phase: "succeeded", terminalStableSteps: 1 });
  });

  it("evaluates named end effectors in world and pelvis frames", () => {
    const endEffectorContract: HumanoidMotionOptionContract = {
      option_id: "right-hand-frame-check",
      stable_steps: 2,
      predicates: [
        {
          type: "end_effector_near_point",
          end_effector: "right_wrist",
          frame: "world",
          target: { x: 1.4, y: 1.1, z: 1.8 },
          tolerance_m: 0.001
        },
        {
          type: "end_effector_near_point",
          end_effector: "right_wrist",
          frame: "pelvis",
          target: { x: 0.2, y: 0.3, z: 0.4 },
          tolerance_m: 0.001
        }
      ]
    };

    const detection = detectHumanoidMotionOption(endEffectorContract, {
      snapshot: robot,
      observableObjects: [],
      zones: []
    });

    expect(detection.allSatisfied).toBe(true);
    expect(detection.evidence).toEqual([
      expect.objectContaining({
        type: "end_effector_near_point",
        frame: "world",
        endEffector: "right_wrist",
        actualPosition: { x: 1.4, y: 1.1, z: 1.8 },
        status: "satisfied"
      }),
      expect.objectContaining({
        type: "end_effector_near_point",
        frame: "pelvis",
        endEffector: "right_wrist",
        actualPosition: {
          x: expect.closeTo(0.2, 10),
          y: expect.closeTo(0.3, 10),
          z: expect.closeTo(0.4, 10)
        },
        status: "satisfied"
      })
    ]);
  });

  it("requires a stable end-effector pose and treats quaternion signs as equivalent", () => {
    const wristRotation = {
      x: 0,
      y: 0,
      z: Math.sin(0.2),
      w: Math.cos(0.2)
    };
    const poseRobot: HumanoidMotionOptionRobotSnapshot = {
      ...robot,
      links: {
        ...robot.links,
        right_wrist_yaw_link: {
          ...robot.links.right_wrist_yaw_link!,
          rotation: wristRotation
        }
      }
    };
    const poseContract: HumanoidMotionOptionContract = {
      option_id: "right-wrist-pose",
      stable_steps: 2,
      predicates: [{
        type: "end_effector_near_point",
        end_effector: "right_wrist",
        frame: "world",
        target: { x: 1.4, y: 1.1, z: 1.8 },
        tolerance_m: 0.001,
        target_orientation: {
          x: -wristRotation.x,
          y: -wristRotation.y,
          z: -wristRotation.z,
          w: -wristRotation.w
        },
        orientation_tolerance_rad: 0.01
      }]
    };

    const satisfied = detectHumanoidMotionOption(poseContract, {
      snapshot: poseRobot,
      observableObjects: [],
      zones: []
    });
    expect(satisfied).toMatchObject({
      allSatisfied: true,
      evidence: [{
        status: "satisfied",
        orientationErrorRadians: expect.closeTo(0, 12),
        orientationToleranceRadians: 0.01
      }]
    });

    const wrongOrientation = structuredClone(poseContract);
    const predicate = wrongOrientation.predicates[0];
    if (predicate?.type !== "end_effector_near_point") throw new Error("Missing pose");
    predicate.target_orientation = { x: 0, y: 0, z: 0, w: 1 };
    predicate.orientation_tolerance_rad = 0.1;
    const wrongInput = {
      snapshot: poseRobot,
      observableObjects: [],
      zones: []
    };
    expect(detectHumanoidMotionOption(wrongOrientation, wrongInput)).toMatchObject({
      allSatisfied: false,
      evidence: [{ status: "unsatisfied", orientationErrorRadians: expect.closeTo(0.4, 10) }]
    });
    expect(advanceHumanoidMotionOptionMonitor(
      wrongOrientation,
      createHumanoidMotionOptionMonitorState(wrongOrientation),
      wrongInput
    ).state.terminalStableSteps).toBe(0);

    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...poseContract,
      predicates: [{
        ...poseContract.predicates[0],
        orientation_tolerance_rad: undefined
      }]
    })).toThrow(/provided together/);
  });

  it("fails closed when a pelvis-relative frame cannot be observed", () => {
    const endEffectorContract: HumanoidMotionOptionContract = {
      option_id: "missing-pelvis-frame",
      stable_steps: 1,
      predicates: [{
        type: "end_effector_near_point",
        end_effector: "left_wrist",
        frame: "pelvis",
        target: { x: 0, y: 0, z: 0 },
        tolerance_m: 0.1
      }]
    };
    const detection = detectHumanoidMotionOption(endEffectorContract, {
      snapshot: { ...robot, links: { left_wrist_yaw_link: robot.links.left_wrist_yaw_link! } },
      observableObjects: [],
      zones: []
    });

    expect(detection.status).toBe("uncertain");
    expect(detection.evidence[0]).toMatchObject({
      actualPosition: null,
      reason: "end_effector_snapshot_missing"
    });
  });

  it("accepts only bounded predicate-index condition ASTs", () => {
    const terminal = {
      precondition: null,
      during: null,
      terminal: {
        condition: {
          op: "all" as const,
          conditions: [
            { op: "predicate" as const, predicate_index: 0 },
            {
              op: "not" as const,
              condition: { op: "predicate" as const, predicate_index: 1 }
            }
          ]
        }
      }
    };
    expect(HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: terminal
    }).phases).toEqual(terminal);

    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: {
        ...terminal,
        terminal: {
          condition: { op: "predicate", predicate_index: 15 }
        }
      }
    })).toThrow(/missing predicate 15/);
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: {
        ...terminal,
        terminal: {
          condition: {
            op: "predicate",
            predicate_index: 0,
            javascript: "return true"
          }
        }
      }
    })).toThrow();

    let tooDeep: HumanoidMotionOptionCondition = {
      op: "predicate",
      predicate_index: 0
    };
    for (let level = 0; level < 8; level += 1) {
      tooDeep = { op: "not", condition: tooDeep };
    }
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: {
        ...terminal,
        terminal: { condition: tooDeep }
      }
    })).toThrow(/eight AST levels/);

    const tooManyNodes: HumanoidMotionOptionCondition = {
      op: "all",
      conditions: Array.from({ length: 16 }, () => ({
        op: "all" as const,
        conditions: Array.from({ length: 4 }, () => ({
          op: "predicate" as const,
          predicate_index: 0
        }))
      }))
    };
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: {
        ...terminal,
        terminal: { condition: tooManyNodes }
      }
    })).toThrow(/64 AST nodes/);
  });

  it("returns ordered evidence when every observable physical relation holds", () => {
    const detection = detectHumanoidMotionOption(contract, {
      snapshot: robot,
      observableObjects: [observableCrate],
      zones: [destination]
    });

    expect(detection.allSatisfied).toBe(true);
    expect(detection.hasUncertain).toBe(false);
    expect(detection.evidence.map((entry) => ({
      index: entry.predicateIndex,
      type: entry.type,
      status: entry.status
    }))).toEqual([
      { index: 0, type: "root_near_point", status: "satisfied" },
      { index: 1, type: "body_near_point", status: "satisfied" },
      { index: 2, type: "body_contact_object", status: "satisfied" },
      { index: 3, type: "object_near_point", status: "satisfied" },
      { index: 4, type: "object_in_zone", status: "satisfied" }
    ]);
    expect(detection.evidence[2]).toMatchObject({
      maximumNormalForce: 14,
      minimumNormalForce: 10,
      objectObservable: true
    });
  });

  it("binds body and exact hand-surface predicates to the same observed solid", () => {
    const solidContract = HumanoidMotionOptionContractSchema.parse({
      option_id: "touch-observed-block",
      stable_steps: 2,
      predicates: [{
        type: "body_contact_solid",
        body: "left_wrist_yaw_link",
        solid_id: "block-a",
        minimum_normal_force: 5
      }, {
        type: "hand_contact_solid",
        hand_surface: "left_hand_palm_link",
        solid_id: "block-a",
        minimum_normal_force: 5
      }]
    });
    const solidContactRobot: HumanoidMotionOptionRobotSnapshot = {
      ...robot,
      contacts: [{
        normalForce: 12,
        firstBody: null,
        secondBody: "left_wrist_yaw_link",
        firstObject: null,
        secondObject: null,
        firstSolid: "block-a",
        secondSolid: null,
        firstHandLink: null,
        secondHandLink: "left_hand_palm_link"
      }]
    };

    expect(detectHumanoidMotionOption(solidContract, {
      snapshot: solidContactRobot,
      observableObjects: [],
      observableSolidIds: ["block-a"],
      zones: []
    })).toMatchObject({
      status: "satisfied",
      evidence: [{
        type: "body_contact_solid",
        solidId: "block-a",
        solidObservable: true,
        maximumNormalForce: 12
      }, {
        type: "hand_contact_solid",
        handSurface: "left_hand_palm_link",
        solidId: "block-a",
        solidObservable: true,
        maximumNormalForce: 12
      }]
    });
    expect(detectHumanoidMotionOption(solidContract, {
      snapshot: solidContactRobot,
      observableObjects: [],
      observableSolidIds: [],
      zones: []
    })).toMatchObject({
      status: "uncertain",
      hasUncertain: true,
      evidence: [{ reason: "solid_not_observable" }, {
        reason: "solid_not_observable"
      }]
    });
  });

  it("measures an exact hand surface contacting an observable dynamic object", () => {
    const handObjectContract = HumanoidMotionOptionContractSchema.parse({
      option_id: "touch-observed-object",
      stable_steps: 2,
      predicates: [{
        type: "hand_contact_object",
        hand_surface: "left_hand_palm_link",
        object_id: "crate",
        minimum_normal_force: 5
      }]
    });
    const handObjectRobot: HumanoidMotionOptionRobotSnapshot = {
      ...robot,
      contacts: [{
        normalForce: 12,
        firstBody: null,
        secondBody: null,
        firstObject: null,
        secondObject: "crate",
        firstHandLink: "left_hand_palm_link",
        secondHandLink: null
      }]
    };

    expect(detectHumanoidMotionOption(handObjectContract, {
      snapshot: handObjectRobot,
      observableObjects: [observableCrate],
      zones: []
    })).toMatchObject({
      status: "satisfied",
      evidence: [{
        type: "hand_contact_object",
        handSurface: "left_hand_palm_link",
        objectId: "crate",
        objectObservable: true,
        maximumNormalForce: 12,
        minimumNormalForce: 5
      }]
    });
    expect(detectHumanoidMotionOption(handObjectContract, {
      snapshot: handObjectRobot,
      observableObjects: [],
      zones: []
    })).toMatchObject({
      status: "uncertain",
      hasUncertain: true,
      evidence: [{
        type: "hand_contact_object",
        objectObservable: false,
        maximumNormalForce: null,
        reason: "object_not_observable"
      }]
    });
  });

  it("never uses hidden snapshot objects or contacts as observable success", () => {
    const snapshotWithHiddenObject = {
      ...robot,
      objects: {
        hidden: {
          id: "hidden",
          position: { x: 4, y: 0.25, z: 4 }
        }
      },
      contacts: [{
        normalForce: 50,
        firstBody: "left_wrist_yaw_link" as const,
        secondBody: null,
        firstObject: null,
        secondObject: "hidden"
      }]
    };
    const hiddenContract = HumanoidMotionOptionContractSchema.parse({
      option_id: "hidden-object-must-not-pass",
      stable_steps: 2,
      predicates: [
        {
          type: "body_contact_object",
          body: "left_wrist_yaw_link",
          object_id: "hidden",
          minimum_normal_force: 1
        },
        {
          type: "object_near_point",
          object_id: "hidden",
          target: { x: 4, y: 0.25, z: 4 },
          tolerance_m: 0.01
        },
        {
          type: "object_in_zone",
          object_id: "hidden",
          zone_id: "destination",
          expected: true,
          tolerance_m: 0.01
        }
      ]
    });

    const detection = detectHumanoidMotionOption(hiddenContract, {
      snapshot: snapshotWithHiddenObject,
      observableObjects: [],
      zones: [destination]
    });

    expect(detection.allSatisfied).toBe(false);
    expect(detection.hasUncertain).toBe(true);
    expect(detection.evidence).toHaveLength(3);
    expect(detection.evidence.every((entry) => entry.status === "uncertain")).toBe(true);
    expect(detection.evidence.every((entry) => (
      "reason" in entry && entry.reason === "object_not_observable"
    ))).toBe(true);
  });

  it("distinguishes measured failure from unavailable body or zone evidence", () => {
    const input: HumanoidMotionOptionDetectorInput = {
      snapshot: {
        ...robot,
        rootPosition: { x: 4, y: 0.8, z: 4 },
        links: {},
        contacts: []
      },
      observableObjects: [observableCrate],
      zones: []
    };
    const detection = detectHumanoidMotionOption(contract, input);

    expect(detection.allSatisfied).toBe(false);
    expect(detection.hasUncertain).toBe(true);
    expect(detection.evidence[0]).toMatchObject({ status: "unsatisfied" });
    expect(detection.evidence[1]).toMatchObject({
      status: "uncertain",
      reason: "body_snapshot_missing"
    });
    expect(detection.evidence[2]).toMatchObject({
      status: "unsatisfied",
      maximumNormalForce: 0
    });
    expect(detection.evidence[4]).toMatchObject({
      status: "uncertain",
      reason: "zone_not_found"
    });
  });

  it("supports an explicit expectation that an observed object is outside a zone", () => {
    const outside = { ...observableCrate, position: { x: 5, y: 0.3, z: 5 } };
    const outsideContract = HumanoidMotionOptionContractSchema.parse({
      option_id: "outside-zone",
      stable_steps: 1,
      predicates: [{
        type: "object_in_zone",
        object_id: "crate",
        zone_id: "destination",
        expected: false,
        tolerance_m: 0
      }]
    });
    const detection = detectHumanoidMotionOption(outsideContract, {
      snapshot: robot,
      observableObjects: [outside],
      zones: [destination]
    });

    expect(detection).toMatchObject({
      allSatisfied: true,
      hasUncertain: false,
      evidence: [{ status: "satisfied", inside: false, expected: false }]
    });
  });

  it("uses the physical object orientation when testing zone support and containment", () => {
    const placedContract = HumanoidMotionOptionContractSchema.parse({
      option_id: "rotated-object-in-zone",
      stable_steps: 1,
      predicates: [{
        type: "object_in_zone",
        object_id: "rod",
        zone_id: "drop-zone",
        expected: true,
        tolerance_m: 0.01
      }]
    });
    const detection = detectHumanoidMotionOption(placedContract, {
      snapshot: robot,
      observableObjects: [{
        id: "rod",
        position: { x: 1, y: 0.015, z: 3 },
        rotation: {
          x: 0,
          y: 0,
          z: Math.SQRT1_2,
          w: Math.SQRT1_2
        },
        size: { x: 0.03, y: 0.27, z: 0.03 }
      }],
      zones: [{
        id: "drop-zone",
        center: { x: 1, y: -0.025, z: 3 },
        size: { x: 1, y: 0.05, z: 1 }
      }]
    });

    expect(detection).toMatchObject({
      allSatisfied: true,
      hasUncertain: false,
      evidence: [{ status: "satisfied", inside: true }]
    });
  });

  it("evaluates all, any, and not with three-valued physical logic", () => {
    const predicates = [
      {
        type: "root_near_point" as const,
        target: { ...robot.rootPosition },
        tolerance_m: 0.01
      },
      {
        type: "object_near_point" as const,
        object_id: "hidden",
        target: { x: 0, y: 0, z: 0 },
        tolerance_m: 0.01
      },
      {
        type: "root_near_point" as const,
        target: { x: 9, y: 0.8, z: 9 },
        tolerance_m: 0.01
      }
    ];
    const input = {
      snapshot: robot,
      observableObjects: [observableCrate],
      zones: [destination]
    };
    const detectionFor = (condition: unknown) => detectHumanoidMotionOption(
      HumanoidMotionOptionContractSchema.parse({
        option_id: "three-value-logic",
        predicates,
        stable_steps: 1,
        phases: {
          precondition: null,
          during: null,
          terminal: { condition }
        }
      }),
      input
    );

    const resolvedAny = detectionFor({
      op: "any",
      conditions: [
        { op: "predicate", predicate_index: 0 },
        { op: "predicate", predicate_index: 1 }
      ]
    });
    expect(resolvedAny).toMatchObject({
      status: "satisfied",
      allSatisfied: true,
      hasUncertain: false,
      phases: { terminal: { status: "satisfied", predicateIndexes: [0, 1] } }
    });

    const resolvedAll = detectionFor({
      op: "all",
      conditions: [
        { op: "predicate", predicate_index: 2 },
        { op: "predicate", predicate_index: 1 }
      ]
    });
    expect(resolvedAll).toMatchObject({
      status: "unsatisfied",
      allSatisfied: false,
      hasUncertain: false
    });

    const unresolvedNot = detectionFor({
      op: "not",
      condition: { op: "predicate", predicate_index: 1 }
    });
    expect(unresolvedNot).toMatchObject({
      status: "uncertain",
      allSatisfied: false,
      hasUncertain: true
    });
  });

  it("monitors precondition, during invariant, and terminal stability windows", () => {
    const phasedContract = HumanoidMotionOptionContractSchema.parse({
      option_id: "phased-reach",
      predicates: [
        {
          type: "root_near_point",
          target: { ...robot.rootPosition },
          tolerance_m: 0.01
        },
        {
          type: "body_near_point",
          body: "left_wrist_yaw_link",
          target: { x: 1.2, y: 1.1, z: 2.4 },
          tolerance_m: 0.05
        },
        {
          type: "object_near_point",
          object_id: "crate",
          target: { x: 4, y: 0.3, z: 4 },
          tolerance_m: 0.05
        }
      ],
      stable_steps: 2,
      phases: {
        precondition: {
          condition: { op: "predicate", predicate_index: 0 },
          stable_steps: 2
        },
        during: {
          condition: { op: "predicate", predicate_index: 1 }
        },
        terminal: {
          condition: { op: "predicate", predicate_index: 2 }
        }
      }
    });
    const inputWith = (
      snapshot: HumanoidMotionOptionRobotSnapshot,
      observableObjects = [observableCrate]
    ): HumanoidMotionOptionDetectorInput => ({
      snapshot,
      observableObjects,
      zones: [destination]
    });
    const targetCrate = {
      ...observableCrate,
      position: { x: 4, y: 0.3, z: 4 }
    };

    let state = createHumanoidMotionOptionMonitorState(phasedContract);
    let update = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      state,
      inputWith(robot, [targetCrate])
    );
    expect(update.state).toMatchObject({
      phase: "awaiting_precondition",
      preconditionStableSteps: 1,
      terminalStableSteps: 0
    });
    state = update.state;
    update = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      state,
      inputWith(robot, [targetCrate])
    );
    expect(update.state).toMatchObject({
      phase: "running",
      preconditionStableSteps: 2,
      terminalStableSteps: 0
    });

    const afterPrecondition = {
      ...robot,
      rootPosition: { x: 8, y: 0.8, z: 8 }
    };
    state = update.state;
    update = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      state,
      inputWith(afterPrecondition, [targetCrate])
    );
    expect(update.state).toMatchObject({
      phase: "running",
      terminalStableSteps: 1
    });

    state = update.state;
    update = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      state,
      inputWith(afterPrecondition, [])
    );
    expect(update).toMatchObject({
      observationStatus: "uncertain",
      state: { phase: "running", terminalStableSteps: 0 }
    });

    for (let step = 0; step < 2; step += 1) {
      update = advanceHumanoidMotionOptionMonitor(
        phasedContract,
        update.state,
        inputWith(afterPrecondition, [targetCrate])
      );
    }
    expect(update.state).toMatchObject({
      phase: "succeeded",
      terminalStableSteps: 2
    });

    let invariantState = createHumanoidMotionOptionMonitorState(phasedContract);
    invariantState = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      invariantState,
      inputWith(robot)
    ).state;
    invariantState = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      invariantState,
      inputWith(robot)
    ).state;
    const missingBody = {
      ...robot,
      links: {}
    };
    const uncertainInvariant = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      invariantState,
      inputWith(missingBody)
    );
    expect(uncertainInvariant).toMatchObject({
      observationStatus: "uncertain",
      state: { phase: "indeterminate", terminalStableSteps: 0 }
    });
    const stillIndeterminate = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      uncertainInvariant.state,
      inputWith(robot, [targetCrate])
    );
    expect(stillIndeterminate).toMatchObject({
      observationStatus: "uncertain",
      state: { phase: "indeterminate", terminalStableSteps: 0 }
    });
    const violatedInvariant = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      invariantState,
      inputWith({
        ...robot,
        links: {
          left_wrist_yaw_link: { position: { x: 9, y: 9, z: 9 } }
        }
      })
    );
    expect(violatedInvariant.state.phase).toBe("violated");
  });
});
