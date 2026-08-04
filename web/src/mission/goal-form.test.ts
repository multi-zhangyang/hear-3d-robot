import { describe, expect, it } from "vitest";
import type { GoalPredicate } from "../types";
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

  it("兼容纯位置目标并完整校验可选姿态", () => {
    const positionPredicate = {
      type: "end_effector_at",
      end_effector: "right_wrist",
      frame: "world",
      target: { x: 2, y: 1.1, z: 3 },
      tolerance: 0.05,
      stable_frames: 3
    } satisfies Extract<GoalPredicate, { type: "end_effector_at" }>;
    const positionOnly = {
      summary: "保持右腕位置",
      predicates: [positionPredicate]
    };
    expect(validGoal(positionOnly)).toBe(true);
    expect(validGoal({
      ...positionOnly,
      predicates: [{
        ...positionPredicate,
        orientation: { x: 0, y: 0, z: 0, w: 1 },
        orientation_tolerance_rad: 0.15
      }]
    })).toBe(true);
    expect(validGoal({
      ...positionOnly,
      predicates: [{
        ...positionPredicate,
        orientation: { x: 0, y: 0, z: 0, w: 0 },
        orientation_tolerance_rad: 0.15
      }]
    })).toBe(false);
    expect(validGoal({
      ...positionOnly,
      predicates: [{
        ...positionPredicate,
        orientation_tolerance_rad: 0.15
      }]
    })).toBe(false);
  });
});

describe("真实抓取任务目标", () => {
  it("创建不包含权威阈值的任意手抓取条件", () => {
    expect(emptyPredicate("object_grasped")).toEqual({
      type: "object_grasped",
      object_id: "",
      hand: "either"
    });
    expect(validGoal({
      summary: "抓住可搬动物体",
      predicates: [{
        type: "object_grasped",
        object_id: "crate",
        hand: "right"
      }]
    })).toBe(true);
    expect(validGoal({
      summary: "缺少物体",
      predicates: [{
        type: "object_grasped",
        object_id: "",
        hand: "either"
      }]
    })).toBe(false);
  });
});

describe("真实放置任务目标", () => {
  it("只收集对象、区域和几何容差", () => {
    expect(emptyPredicate("object_placed")).toEqual({
      type: "object_placed",
      object_id: "",
      zone_id: "",
      tolerance: 0.05
    });
    expect(validGoal({
      summary: "将箱体稳放到目标区域",
      predicates: [{
        type: "object_placed",
        object_id: "crate",
        zone_id: "arrival",
        tolerance: 0.04
      }]
    })).toBe(true);
    expect(validGoal({
      summary: "缺少放置区域",
      predicates: [{
        type: "object_placed",
        object_id: "crate",
        zone_id: "",
        tolerance: 0.04
      }]
    })).toBe(false);
  });
});
