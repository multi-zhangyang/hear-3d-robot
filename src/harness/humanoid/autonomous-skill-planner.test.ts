import { describe, expect, it } from "vitest";
import type { HumanoidWorldObservation } from "../../world/humanoid/world.js";
import { planAutonomousHumanoidSkill } from "./autonomous-skill-planner.js";
import type { ActiveHumanoidSkillBinding } from "./skill-binding.js";

describe("autonomous humanoid navigation Goal constraints", () => {
  it("does not let a semantic frontier tolerance exceed the active robot_at Goal", () => {
    const target = { x: 9.25, y: 0, z: 14.75 };
    const plan = planAutonomousHumanoidSkill({
      binding: explorationBinding(),
      observation: explorationObservation(target),
      activeGoal: {
        summary: "到达模型选择的前沿",
        predicates: [{ type: "robot_at", target, tolerance: 0.2 }]
      }
    });

    expect(plan).toMatchObject({
      kind: "navigation",
      targets: [{ target }]
    });
    if (plan.kind !== "navigation") return;
    expect(plan.targets[0]?.acceptedPositionToleranceMeters).toBeCloseTo(0.15, 12);
  });

  it("rejects a frontier that moves away from the active robot_at Goal", () => {
    const target = { x: 9.25, y: 0, z: 14.75 };
    expect(() => planAutonomousHumanoidSkill({
      binding: explorationBinding(),
      observation: explorationObservation(target),
      activeGoal: {
        summary: "到达相反方向的模型前沿",
        predicates: [{
          type: "robot_at",
          target: { x: -15, y: 0, z: -13 },
          tolerance: 0.2
        }]
      }
    })).toThrow("not causally aligned");
  });

  it("keeps a positive settling reserve for a narrow Goal", () => {
    const target = { x: 9.25, y: 0, z: 14.75 };
    const plan = planAutonomousHumanoidSkill({
      binding: explorationBinding(),
      observation: explorationObservation(target),
      activeGoal: {
        summary: "精确到达前沿",
        predicates: [{ type: "robot_at", target, tolerance: 0.04 }]
      }
    });

    expect(plan.kind).toBe("navigation");
    if (plan.kind !== "navigation") return;
    expect(plan.targets[0]?.acceptedPositionToleranceMeters).toBeCloseTo(0.02, 12);
  });
});

function explorationBinding(): ActiveHumanoidSkillBinding {
  return {
    phase_authority: "navigation",
    invocation: {
      skill: "explore",
      frontier_id: "frontier:18:29",
      strategy: "balanced",
      maximum_travel_m: 1
    }
  } as ActiveHumanoidSkillBinding;
}

function explorationObservation(target: {
  x: number;
  y: number;
  z: number;
}): HumanoidWorldObservation {
  return {
    robot: { rootPosition: { x: 0, y: 0.75, z: 0 } },
    spatialBelief: {
      resolution_m: 0.5,
      frontiers: [{
        id: "frontier:18:29",
        target,
        expected_information_gain: 8,
        travel_distance_m: 0.34,
        revisit_penalty: 0,
        score: 8
      }]
    }
  } as HumanoidWorldObservation;
}
