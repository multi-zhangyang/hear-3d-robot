import { describe, expect, it } from "vitest";
import {
  HumanoidMotionCandidateBatchInputSchema,
  normalizeHumanoidMotionCandidateBatchInput
} from "./motion-candidate-input.js";

describe("model-facing humanoid motion candidate input", () => {
  it("normalizes bounded all/any/not clauses into the physical option tree", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "抬起左脚同时保持根节点位置",
      termination: {
        option_id: "balanced-left-foot",
        predicates: [
          predicate("root_near_point", {
            target: { x: 2.5, y: 0.76, z: 2.5 },
            tolerance_m: 0.08
          }),
          predicate("body_near_point", {
            body: "left_ankle_roll_link",
            target: { x: 2.64, y: 0.12, z: 2.55 },
            tolerance_m: 0.05
          })
        ],
        stable_steps: 3,
        phases: {
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

  it("rejects ambiguous predicate fields before physics", () => {
    expect(() => HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "无效谓词",
      termination: {
        option_id: "ambiguous",
        predicates: [predicate("root_near_point", {
          body: "pelvis",
          target: { x: 0, y: 0, z: 0 },
          tolerance_m: 0.1
        })],
        stable_steps: 2,
        phases: null
      },
      candidates: candidates()
    })).toThrow();
  });

  it("normalizes explicit pelvis-relative end-effector predicates", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "将右手移动到骨盆前方",
      termination: {
        option_id: "right-hand-relative-reach",
        predicates: [predicate("end_effector_near_point", {
          end_effector: "right_wrist",
          frame: "pelvis",
          target: { x: 0.25, y: 0.3, z: 0.15 },
          tolerance_m: 0.04
        })],
        stable_steps: 4,
        phases: null
      },
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

  it("normalizes an optional end-effector orientation as physical termination", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "调整右腕位姿",
      termination: {
        option_id: "right-wrist-pose",
        predicates: [predicate("end_effector_near_point", {
          end_effector: "right_wrist",
          frame: "world",
          target: { x: 0.2, y: 1.1, z: 0.4 },
          tolerance_m: 0.04,
          target_orientation: { x: 0, y: 0, z: 0, w: 1 },
          orientation_tolerance_rad: 0.12
        })],
        stable_steps: 3,
        phases: null
      },
      candidates: candidates()
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input)
      .termination.predicates[0]).toMatchObject({
        type: "end_effector_near_point",
        target_orientation: { x: 0, y: 0, z: 0, w: 1 },
        orientation_tolerance_rad: 0.12
      });
  });

  it("normalizes only an authority-bound grasp reference into the physical option", () => {
    const contractSha256 = "a".repeat(64);
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "用左手真实抓住箱体",
      termination: {
        option_id: "left-grasp",
        predicates: [graspPredicate({
          object_id: "crate",
          hand: "left",
          grasp_contract_sha256: contractSha256
        })],
        stable_steps: 3,
        phases: null
      },
      candidates: candidates().map((candidate) => ({
        ...candidate,
        contact_constraints: [
          {
            hand_surface: "left_hand_palm_link",
            object_id: "crate",
            required: false
          },
          {
            hand_surface: "left_hand_index_1_link",
            object_id: "crate",
            required: false
          }
        ]
      }))
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input)
      .termination.predicates[0]).toEqual({
        type: "grasp_verified",
        object_id: "crate",
        hand: "left",
        grasp_contract_sha256: contractSha256
      });
  });

  it("requires the observed grasp contract hash and rejects model-owned thresholds", () => {
    const missingHash = graspPredicate({
      object_id: "crate",
      hand: "right",
      grasp_contract_sha256: null
    });
    expect(modelGraspPredicateResult(missingHash).success).toBe(false);

    expect(modelGraspPredicateResult({
      ...graspPredicate({
        object_id: "crate",
        hand: "right",
        grasp_contract_sha256: "b".repeat(64)
      }),
      minimum_lift_m: 0
    }).success).toBe(false);
  });

  it("normalizes a model-selected physical placement without exposing settle thresholds", () => {
    const contractSha256 = "c".repeat(64);
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "将箱体放入目标区并确认已经放稳脱手",
      termination: {
        option_id: "place-crate",
        predicates: [
          {
            ...emptyPredicateFields(),
            type: "object_in_zone",
            object_id: "crate",
            zone_id: "destination",
            tolerance_m: 0.02,
            expected: true
          },
          graspPredicate({
            object_id: "crate",
            hand: "left",
            grasp_contract_sha256: contractSha256
          }),
          settledPredicate("crate")
        ],
        stable_steps: 8,
        phases: {
          precondition: null,
          during: null,
          terminal: {
            condition: {
              op: "all",
              predicate_indexes: [0, 2],
              not_predicate_indexes: [1]
            }
          }
        }
      },
      candidates: candidates().map((candidate) => ({
        ...candidate,
        contact_constraints: [
          {
            hand_surface: "left_hand_palm_link",
            object_id: "crate",
            required: false
          },
          {
            hand_surface: "left_hand_index_1_link",
            object_id: "crate",
            required: false
          }
        ]
      }))
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input).termination)
      .toMatchObject({
        predicates: [
          {
            type: "object_in_zone",
            object_id: "crate",
            zone_id: "destination",
            expected: true,
            tolerance_m: 0.02
          },
          {
            type: "grasp_verified",
            object_id: "crate",
            hand: "left",
            grasp_contract_sha256: contractSha256
          },
          {
            type: "object_settled_on_support",
            object_id: "crate"
          }
        ],
        phases: {
          terminal: {
            condition: {
              op: "all",
              conditions: [
                { op: "predicate", predicate_index: 0 },
                { op: "predicate", predicate_index: 2 },
                {
                  op: "not",
                  condition: { op: "predicate", predicate_index: 1 }
                }
              ]
            }
          }
        }
      });

    expect(modelPredicateResult({
      ...settledPredicate("crate"),
      maximum_linear_speed_mps: 100
    }).success).toBe(false);
  });

  it("normalizes exact static-solid contact authority without object substitution", () => {
    const input = HumanoidMotionCandidateBatchInputSchema.parse({
      objective: "用左掌持续接触当前可见方块",
      termination: {
        option_id: "touch-visible-block",
        predicates: [{
          ...emptyPredicateFields(),
          type: "body_contact_solid",
          body: "left_wrist_yaw_link",
          solid_id: "block-a",
          minimum_normal_force: 5
        }, {
          ...emptyPredicateFields(),
          type: "hand_contact_solid",
          hand_surface: "left_hand_palm_link",
          solid_id: "block-a",
          minimum_normal_force: 5
        }],
        stable_steps: 3,
        phases: null
      },
      candidates: candidates().map((candidate) => ({
        ...candidate,
        contact_constraints: [{
          body: "left_wrist_yaw_link",
          solid_id: "block-a",
          required: true
        }, {
          hand_surface: "left_hand_palm_link",
          solid_id: "block-a",
          required: true
        }]
      }))
    });

    expect(normalizeHumanoidMotionCandidateBatchInput(input).termination.predicates)
      .toEqual([{
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
  });

  it("rejects candidates that differ only by labels before normalization", () => {
    const first = candidates()[0]!;
    const parsed = HumanoidMotionCandidateBatchInputSchema.safeParse({
      objective: "比较真正不同的动作候选",
      termination: {
        option_id: "distinct-motion",
        predicates: [predicate("root_near_point", {
          target: { x: 0, y: 0.76, z: 0.1 },
          tolerance_m: 0.05
        })],
        stable_steps: 2,
        phases: null
      },
      candidates: [
        { ...first, contact_constraints: null },
        {
          ...first,
          id: "candidate-renamed",
          intent: "只改了显示名称",
          contact_constraints: [],
          keyframes: [
            { at_seconds: 0, root_pitch: null },
            { at_seconds: 0.4, root_pitch: null }
          ]
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

function predicate(
  type: "root_near_point" | "body_near_point" | "end_effector_near_point",
  values: Partial<{
    body: "pelvis" | "left_ankle_roll_link";
    end_effector: "left_wrist" | "right_wrist" | "left_ankle" | "right_ankle";
    frame: "world" | "pelvis";
    target: { x: number; y: number; z: number };
    tolerance_m: number;
    target_orientation: { x: number; y: number; z: number; w: number };
    orientation_tolerance_rad: number;
  }>
) {
  return {
    type,
    body: values.body ?? null,
    end_effector: values.end_effector ?? null,
    frame: values.frame ?? null,
    object_id: null,
    solid_id: null,
    hand_surface: null,
    hand: null,
    grasp_contract_sha256: null,
    zone_id: null,
    target: values.target ?? null,
    tolerance_m: values.tolerance_m ?? null,
    target_orientation: values.target_orientation ?? null,
    orientation_tolerance_rad: values.orientation_tolerance_rad ?? null,
    minimum_normal_force: null,
    expected: null
  };
}

function graspPredicate(values: {
  object_id: string;
  hand: "left" | "right";
  grasp_contract_sha256: string | null;
}) {
  return {
    type: "grasp_verified" as const,
    body: null,
    end_effector: null,
    frame: null,
    object_id: values.object_id,
    solid_id: null,
    hand_surface: null,
    hand: values.hand,
    grasp_contract_sha256: values.grasp_contract_sha256,
    zone_id: null,
    target: null,
    tolerance_m: null,
    target_orientation: null,
    orientation_tolerance_rad: null,
    minimum_normal_force: null,
    expected: null
  };
}

function settledPredicate(objectId: string) {
  return {
    ...emptyPredicateFields(),
    type: "object_settled_on_support" as const,
    object_id: objectId
  };
}

function emptyPredicateFields() {
  return {
    body: null,
    end_effector: null,
    frame: null,
    object_id: null,
    solid_id: null,
    hand_surface: null,
    hand: null,
    grasp_contract_sha256: null,
    zone_id: null,
    target: null,
    tolerance_m: null,
    target_orientation: null,
    orientation_tolerance_rad: null,
    minimum_normal_force: null,
    expected: null
  };
}

function modelGraspPredicateResult(predicateValue: unknown) {
  return modelPredicateResult(predicateValue);
}

function modelPredicateResult(predicateValue: unknown) {
  return HumanoidMotionCandidateBatchInputSchema.shape.termination.shape.predicates
    .element.safeParse(predicateValue);
}

function candidates() {
  return [
    {
      id: "candidate-a",
      intent: "候选 A",
      duration_seconds: 0.4,
      keyframes: [{ at_seconds: 0 }, { at_seconds: 0.4 }]
    },
    {
      id: "candidate-b",
      intent: "候选 B",
      duration_seconds: 0.5,
      keyframes: [{ at_seconds: 0 }, { at_seconds: 0.5 }]
    }
  ];
}
