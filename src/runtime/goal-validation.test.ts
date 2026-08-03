import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { GoalValidationError, assertGoalSupported } from "./goal-validation.js";

describe("goal validation", () => {
  it("accepts humanoid position and object predicates", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 0);

    expect(() => assertGoalSupported({
      summary: "庭院中的人形目标",
      predicates: [
        { type: "robot_at", target: { x: 3, y: 0, z: 3 }, tolerance: 0.25 },
        {
          type: "object_in_zone",
          object_id: "courtyard_crate",
          zone_id: "courtyard_beacon",
          expected: true,
          tolerance: 0.05
        },
        {
          type: "object_at",
          object_id: "courtyard_crate",
          target: { x: 4.5, y: 0.25, z: 5.5 },
          tolerance: 0.1
        },
        {
          type: "end_effector_at",
          end_effector: "left_wrist",
          frame: "world",
          target: { x: 3, y: 1, z: 3 },
          tolerance: 0.05,
          stable_frames: 4
        },
        {
          type: "end_effector_at",
          end_effector: "right_ankle",
          frame: "pelvis",
          target: { x: -0.1, y: -0.7, z: 0.15 },
          tolerance: 0.04,
          stable_frames: 3
        }
      ]
    }, scenario)).not.toThrow();
  });

  it("rejects coordinates and entity references outside the selected world", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 0);

    expect(() => assertGoalSupported({
      summary: "越界目标",
      predicates: [
        { type: "robot_at", target: { x: 100, y: 0, z: 100 }, tolerance: 0.25 }
      ]
    }, scenario)).toThrow(GoalValidationError);

    expect(() => assertGoalSupported({
      summary: "未知物体",
      predicates: [{
        type: "object_at",
        object_id: "missing_object",
        target: { x: 2, y: 0.2, z: 2 },
        tolerance: 0.1
      }]
    }, scenario)).toThrow("Unknown object: missing_object");

    expect(() => assertGoalSupported({
      summary: "未知区域",
      predicates: [{
        type: "object_in_zone",
        object_id: "courtyard_crate",
        zone_id: "missing_zone",
        expected: true,
        tolerance: 0
      }]
    }, scenario)).toThrow("Unknown zone: missing_zone");

    expect(() => assertGoalSupported({
      summary: "越界末端目标",
      predicates: [{
        type: "end_effector_at",
        end_effector: "right_wrist",
        frame: "world",
        target: { x: -0.1, y: 1, z: 2 },
        tolerance: 0.05,
        stable_frames: 3
      }]
    }, scenario)).toThrow("End-effector world target is outside the world bounds");

    expect(() => assertGoalSupported({
      summary: "骨盆相对坐标允许负值",
      predicates: [{
        type: "end_effector_at",
        end_effector: "right_wrist",
        frame: "pelvis",
        target: { x: -0.25, y: -0.1, z: 0.1 },
        tolerance: 0.05,
        stable_frames: 3
      }]
    }, scenario)).not.toThrow();
  });
});
