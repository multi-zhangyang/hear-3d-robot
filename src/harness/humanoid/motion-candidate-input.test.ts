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
  }>
) {
  return {
    type,
    body: values.body ?? null,
    end_effector: values.end_effector ?? null,
    frame: values.frame ?? null,
    object_id: null,
    zone_id: null,
    target: values.target ?? null,
    tolerance_m: values.tolerance_m ?? null,
    minimum_normal_force: null,
    expected: null
  };
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
