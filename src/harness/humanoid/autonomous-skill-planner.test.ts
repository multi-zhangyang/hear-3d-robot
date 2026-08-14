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

  it("does not replace an obstructed IK base contract with geometric approach", () => {
    const binding = {
      phase_authority: "navigation",
      invocation: {
        skill: "approach",
        object_id: "assembly_rod",
        interaction_point_id: "geometry-z-negative",
        hand: "left",
        standoff_m: 0.2
      },
      target_position: { x: 4.2, y: 0.67, z: 4.8 },
      eligible_interaction_points: [{
        id: "geometry-z-negative",
        kind: "grasp",
        compatible_hands: "either",
        world_position: { x: 4.2, y: 0.67, z: 4.77 },
        approach_direction_world: { x: 0, y: 0, z: 1 },
        clearance_m: 0.025,
        source: "geometry"
      }]
    } as ActiveHumanoidSkillBinding;
    const rejectedIkTarget = { x: 4, y: 0.7655, z: 4.565 };
    const plan = planAutonomousHumanoidSkill({
      binding,
      observation: {
        robot: { rootPosition: { x: 3.98, y: 0.7655, z: 4.02 } },
        manipulationBasePlacements: [{
          objectId: "assembly_rod",
          interactionPointId: "geometry-z-negative",
          handSurface: "left_hand_palm_link",
          rootWorldTarget: rejectedIkTarget,
          rootTranslationWorld: { x: 0, y: 0, z: 0 },
          rootYawRadians: 0,
          wristWorldTarget: { x: 4.2, y: 0.67, z: 4.77 },
          ikResidualMeters: 0.01
        }],
        solidTokens: [{
          id: "object-pickup_stand",
          sourceId: "pickup_stand",
          kind: "fixed_object",
          center: { x: 4.2, y: 0.555, z: 4.8 },
          size: { x: 0.12, y: 0.01, z: 0.12 },
          currentContacts: [{
            position: { x: 4.2, y: 0.56, z: 4.8 },
            normal: { x: 0, y: -1, z: 0 },
            normalForce: 1,
            firstBody: null,
            secondBody: null,
            firstObject: "assembly_rod",
            secondObject: null,
            firstSolid: null,
            secondSolid: "object-pickup_stand",
            firstHandLink: null,
            secondHandLink: null
          }]
        }]
      } as HumanoidWorldObservation
    });

    expect(plan.kind).toBe("navigation");
    if (plan.kind !== "navigation") return;
    expect(plan.targets).toEqual([]);
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
