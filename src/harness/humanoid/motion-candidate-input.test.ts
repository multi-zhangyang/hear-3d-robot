import { describe, expect, it } from "vitest";
import {
  HumanoidMotionCandidateBatchInputSchema,
  normalizeHumanoidMotionCandidateBatchInput
} from "./motion-candidate-input.js";

describe("model-facing humanoid motion candidate input", () => {
  it("accepts one fully simulated local candidate", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "move one wrist",
      termination: allTermination([{
        type: "end_effector_near_point",
        end_effector: "left_wrist",
        frame: "world",
        target: { x: 0.2, y: 0.8, z: 0.3 },
        tolerance_m: 0.05
      }]),
      candidates: [candidates()[0]]
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input).candidates)
      .toHaveLength(1);
  });

  it("normalizes bounded all/any/not clauses into the physical option tree", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "抬起左脚同时保持根节点位置",
      termination: {
        mode: "phased",
        option_id: "balanced-left-foot",
        predicates: [
          {
            type: "root_near_point",
            target: { x: 2.5, y: 0.76, z: 2.5 },
            tolerance_m: 0.08
          },
          {
            type: "body_near_point",
            body: "left_ankle_roll_link",
            target: { x: 2.64, y: 0.12, z: 2.55 },
            tolerance_m: 0.05
          }
        ],
        stable_steps: 3,
        precondition: null,
        during: {
          condition: {
            op: "all",
            predicate_indexes: [0],
            not_predicate_indexes: []
          }
        },
        terminal: {
          condition: {
            op: "any",
            predicate_indexes: [1],
            not_predicate_indexes: [0]
          }
        }
      },
      candidates: candidates()
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input).termination.phases)
      .toEqual({
        precondition: null,
        during: {
          condition: { op: "predicate", predicate_index: 0 }
        },
        terminal: {
          condition: {
            op: "any",
            conditions: [
              { op: "predicate", predicate_index: 1 },
              {
                op: "not",
                condition: { op: "predicate", predicate_index: 0 }
              }
            ]
          }
        }
      });
  });

  it("rejects fields belonging to a different predicate variant", () => {
    const result = batchResult({
      type: "root_near_point",
      body: "pelvis",
      target: { x: 0, y: 0.76, z: 0 },
      tolerance_m: 0.1
    });

    expect(result.success).toBe(false);
  });

  it("normalizes a pelvis-relative end-effector position", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "将右手移动到骨盆前方",
      termination: allTermination([{
        type: "end_effector_near_point",
        end_effector: "right_wrist",
        frame: "pelvis",
        target: { x: 0.25, y: 0.3, z: 0.15 },
        tolerance_m: 0.04
      }]),
      candidates: candidates()
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input)
      .termination.predicates[0]).toEqual({
        type: "end_effector_near_point",
        end_effector: "right_wrist",
        frame: "pelvis",
        target: { x: 0.25, y: 0.3, z: 0.15 },
        tolerance_m: 0.04
      });
  });

  it("normalizes the explicit end-effector pose variant", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "调整右腕位姿",
      termination: allTermination([{
        type: "end_effector_near_pose",
        end_effector: "right_wrist",
        frame: "world",
        target: { x: 0.2, y: 1.1, z: 0.4 },
        tolerance_m: 0.04,
        target_orientation: { x: 0, y: 0, z: 0, w: 1 },
        orientation_tolerance_rad: 0.12
      }]),
      candidates: candidates()
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input)
      .termination.predicates[0]).toEqual({
        type: "end_effector_near_point",
        end_effector: "right_wrist",
        frame: "world",
        target: { x: 0.2, y: 1.1, z: 0.4 },
        tolerance_m: 0.04,
        target_orientation: { x: 0, y: 0, z: 0, w: 1 },
        orientation_tolerance_rad: 0.12
      });
  });

  it("normalizes compact typed channels and contacts", () => {
    const contractSha256 = "a".repeat(64);
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "用左手真实抓住箱体",
      termination: allTermination([{
        type: "grasp_verified",
        object_id: "crate",
        hand: "left",
        grasp_contract_sha256: contractSha256
      }]),
      candidates: graspCandidates("crate")
    });

    const normalized = normalizeHumanoidMotionCandidateBatchInput(input);
    expect(normalized.termination.predicates[0]).toEqual({
      type: "grasp_verified",
      object_id: "crate",
      hand: "left",
      grasp_contract_sha256: contractSha256
    });
    expect(normalized.candidates[0]).toMatchObject({
      contact_constraints: [{
        hand_surface: "left_hand_palm_link",
        object_id: "crate",
        required: false
      }, {
        hand_surface: "left_hand_index_1_link",
        object_id: "crate",
        required: false
      }],
      keyframes: [{
        at_seconds: 0,
        hand_coordination: openHands()
      }, {
        at_seconds: 2,
        left_hand: {
          frame: "pelvis",
          position: { x: 0.25, y: 0.2, z: 0.2 },
          tolerance_m: 0.08
        }
      }]
    });
  });

  it("rejects a planar world coordinate used as pelvis height", () => {
    const candidate = candidates()[0]!;
    const parsed = HumanoidMotionCandidateBatchInputSchema.safeParse({
      objective: "保持有效骨盆高度",
      termination: allTermination([{
        type: "root_near_point",
        target: { x: 4.2, y: 0.76, z: 4.8 },
        tolerance_m: 0.1
      }]),
      candidates: [{
        ...candidate,
        keyframes: candidate.keyframes.map((keyframe) => ({
          ...keyframe,
          channels: [{ type: "root_height", meters: 4.8 }]
        }))
      }]
    });

    expect(parsed.success).toBe(false);
    if (parsed.success) throw new Error("Planar world coordinate was accepted as root height");
    expect(parsed.error.issues).toContainEqual(expect.objectContaining({
      path: ["candidates", 0, "keyframes", 0, "channels", 0, "meters"]
    }));
  });

  it("normalizes grouped contact constraints without changing their semantics", () => {
    const contractSha256 = "c".repeat(64);
    const grouped = graspCandidates("assembly_rod").map((candidate) => ({
      ...candidate,
      contacts: {
        hand_object: [{
          hand_surface: "right_hand_middle_1_link" as const,
          object_id: "assembly_rod",
          required: true
        }, {
          hand_surface: "right_hand_palm_link" as const,
          object_id: "assembly_rod",
          required: true
        }]
      }
    }));
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "用右手抓取装配杆",
      termination: allTermination([{
        type: "grasp_verified",
        object_id: "assembly_rod",
        hand: "right",
        grasp_contract_sha256: contractSha256
      }]),
      candidates: grouped
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input).candidates[0])
      .toMatchObject({
        contact_constraints: [{
          hand_surface: "right_hand_middle_1_link",
          object_id: "assembly_rod",
          required: true
        }, {
          hand_surface: "right_hand_palm_link",
          object_id: "assembly_rod",
          required: true
        }]
      });
  });

  it("binds a dynamic-object hand-contact predicate to exact required authority", () => {
    const handContactCandidates = candidates().map((candidate) => ({
      ...candidate,
      contacts: [{
        type: "hand_object",
        hand_surface: "left_hand_palm_link",
        object_id: "crate",
        required: true
      }]
    }));
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "用左掌真实接触箱体",
      termination: allTermination([{
        type: "hand_contact_object",
        hand_surface: "left_hand_palm_link",
        object_id: "crate",
        minimum_normal_force: 3
      }]),
      candidates: handContactCandidates
    });

    const normalized = normalizeHumanoidMotionCandidateBatchInput(input);
    expect(normalized.termination.predicates).toEqual([{
      type: "hand_contact_object",
      hand_surface: "left_hand_palm_link",
      object_id: "crate",
      minimum_normal_force: 3
    }]);
    expect(normalized.candidates[0]).toMatchObject({
      contact_constraints: [{
        hand_surface: "left_hand_palm_link",
        object_id: "crate",
        required: true
      }]
    });

    const rejected = HumanoidMotionCandidateBatchInputSchema.safeParse({
      objective: "缺少必需接触授权",
      termination: allTermination([{
        type: "hand_contact_object",
        hand_surface: "left_hand_palm_link",
        object_id: "crate",
        minimum_normal_force: 3
      }]),
      candidates: handContactCandidates.map((candidate) => ({
        ...candidate,
        contacts: candidate.contacts.map((contact) => ({
          ...contact,
          required: false
        }))
      }))
    });
    expect(rejected.success).toBe(false);
  });

  it("requires the observed grasp contract and rejects model-owned thresholds", () => {
    expect(batchResult({
      type: "grasp_verified",
      object_id: "crate",
      hand: "right"
    }).success).toBe(false);

    expect(batchResult({
      type: "grasp_verified",
      object_id: "crate",
      hand: "right",
      grasp_contract_sha256: "b".repeat(64),
      minimum_lift_m: 0
    }).success).toBe(false);
  });

  it("exposes released and settled evidence for physical placement", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "将箱体放入目标区并确认已经放稳脱手",
      termination: allTermination([{
        type: "object_in_zone",
        object_id: "crate",
        zone_id: "destination",
        tolerance_m: 0.02,
        expected: true
      }, {
        type: "object_released",
        object_id: "crate",
        hand: "left"
      }, {
        type: "object_settled_on_support",
        object_id: "crate"
      }]),
      candidates: candidates()
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input).termination)
      .toMatchObject({
        predicates: [{
          type: "object_in_zone",
          object_id: "crate",
          zone_id: "destination",
          expected: true,
          tolerance_m: 0.02
        }, {
          type: "object_released",
          object_id: "crate",
          hand: "left"
        }, {
          type: "object_settled_on_support",
          object_id: "crate"
        }],
        phases: null
      });

    expect(batchResult({
      type: "object_settled_on_support",
      object_id: "crate",
      maximum_linear_speed_mps: 100
    }).success).toBe(false);
  });

  it("normalizes exact static-solid contact authority", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "用左掌持续接触当前可见方块",
      termination: allTermination([{
        type: "body_contact_solid",
        body: "left_wrist_yaw_link",
        solid_id: "block-a",
        minimum_normal_force: 5
      }, {
        type: "hand_contact_solid",
        hand_surface: "left_hand_palm_link",
        solid_id: "block-a",
        minimum_normal_force: 5
      }]),
      candidates: candidates().map((candidate) => ({
        ...candidate,
        contacts: [{
          type: "body_solid",
          body: "left_wrist_yaw_link",
          solid_id: "block-a",
          required: true
        }, {
          type: "hand_solid",
          hand_surface: "left_hand_palm_link",
          solid_id: "block-a",
          required: true
        }]
      }))
    });

    const normalized = normalizeHumanoidMotionCandidateBatchInput(input);
    expect(normalized.termination.predicates).toEqual([{
      type: "body_contact_solid",
      body: "left_wrist_yaw_link",
      solid_id: "block-a",
      minimum_normal_force: 5
    }, {
      type: "hand_contact_solid",
      hand_surface: "left_hand_palm_link",
      solid_id: "block-a",
      minimum_normal_force: 5
    }]);
    expect(normalized.candidates[0]?.contact_constraints).toEqual([{
      body: "left_wrist_yaw_link",
      solid_id: "block-a",
      required: true
    }, {
      hand_surface: "left_hand_palm_link",
      solid_id: "block-a",
      required: true
    }]);
  });

  it("rejects candidates that differ only by labels", () => {
    const first = candidates()[0]!;
    const parsed = HumanoidMotionCandidateBatchInputSchema.safeParse({
      objective: "比较真正不同的动作候选",
      termination: allTermination([{
        type: "root_near_point",
        target: { x: 0, y: 0.76, z: 0.1 },
        tolerance_m: 0.05
      }]),
      candidates: [
        first,
        {
          ...first,
          id: "candidate-renamed",
          intent: "只改了显示名称"
        }
      ]
    });

    expect(parsed.success).toBe(false);
    if (parsed.success) throw new Error("Duplicate motion candidates were accepted");
    expect(parsed.error.issues).toContainEqual(expect.objectContaining({
      path: ["candidates", 1],
      message: expect.stringContaining("id and intent labels")
    }));
  });
});

function allTermination(predicates: unknown[]) {
  return {
    mode: "all",
    option_id: "physical-result",
    predicates,
    stable_steps: 3
  };
}

function batchResult(predicate: unknown) {
  return HumanoidMotionCandidateBatchInputSchema.safeParse({
    objective: "验证谓词",
    termination: allTermination([predicate]),
    candidates: candidates()
  });
}

function candidates() {
  return [
    {
      id: "candidate-a",
      intent: "候选 A",
      duration_seconds: 0.4,
      contacts: [],
      keyframes: [
        { at_seconds: 0, channels: [] },
        { at_seconds: 0.4, channels: [] }
      ]
    },
    {
      id: "candidate-b",
      intent: "候选 B",
      duration_seconds: 0.5,
      contacts: [],
      keyframes: [
        { at_seconds: 0, channels: [] },
        { at_seconds: 0.5, channels: [] }
      ]
    }
  ];
}

function graspCandidates(objectId: string) {
  return candidates().map((candidate, index) => ({
    ...candidate,
    duration_seconds: index === 0 ? 2 : 3,
    contacts: [{
      type: "hand_object",
      hand_surface: "left_hand_palm_link",
      object_id: objectId,
      required: false
    }, {
      type: "hand_object",
      hand_surface: "left_hand_index_1_link",
      object_id: objectId,
      required: false
    }],
    keyframes: [{
      at_seconds: 0,
      channels: [{
        type: "hand_coordination",
        coordination: openHands()
      }]
    }, {
      at_seconds: index === 0 ? 2 : 3,
      channels: [{
        type: "hand_coordination",
        coordination: {
          ...openHands(),
          left: {
            thumb_opposition: 0.8,
            thumb_curl: 0.8,
            index_curl: 0.8,
            middle_curl: 0.8
          }
        }
      }, {
        type: "end_effector_position",
        end_effector: "left_wrist",
        frame: "pelvis",
        position: { x: 0.25, y: 0.2, z: 0.2 },
        tolerance_m: 0.08
      }]
    }]
  }));
}

function openHands() {
  return {
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
}
