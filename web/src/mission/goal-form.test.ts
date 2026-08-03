import { describe, expect, it } from "vitest";
import { emptyPredicate, validGoal } from "./goal-form";

describe("末端任务目标", () => {
  it("创建完整的骨盆相对目标并校验稳定窗", () => {
    expect(emptyPredicate("end_effector_at")).toEqual({
      type: "end_effector_at",
      end_effector: "left_wrist",
      frame: "pelvis",
      target: { x: 0, y: 0, z: 0 },
      tolerance: 0.05,
      stable_frames: 5
    });
    expect(validGoal({
      summary: "抬起左手",
      predicates: [{
        type: "end_effector_at",
        end_effector: "left_wrist",
        frame: "pelvis",
        target: { x: 0.25, y: 0.3, z: 0.1 },
        tolerance: 0.05,
        stable_frames: 4
      }]
    })).toBe(true);
    expect(validGoal({
      summary: "无效稳定窗",
      predicates: [{
        type: "end_effector_at",
        end_effector: "left_wrist",
        frame: "pelvis",
        target: { x: 0.25, y: 0.3, z: 0.1 },
        tolerance: 0.05,
        stable_frames: 0
      }]
    })).toBe(false);
  });
});
