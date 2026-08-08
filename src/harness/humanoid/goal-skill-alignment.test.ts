import { describe, expect, it } from "vitest";
import type { HumanoidWorldObservation } from "../../world/humanoid/world.js";
import { alignHumanoidSkillToGoal } from "./goal-skill-alignment.js";

describe("humanoid Skill to active Goal alignment", () => {
  it("rejects a valid observed frontier when it moves away from robot_at", () => {
    const observation = spatialObservation({
      id: "frontier:32:14",
      target: { x: 16.25, y: 0, z: 7.25 }
    });

    expect(alignHumanoidSkillToGoal({
      goal: {
        summary: "前往模型选择的目标前沿",
        predicates: [{
          type: "robot_at",
          target: { x: 8.75, y: 0, z: 14.75 },
          tolerance: 0.2
        }]
      },
      invocation: {
        skill: "explore",
        frontier_id: "frontier:32:14",
        strategy: "balanced",
        maximum_travel_m: 8
      },
      observation
    })).toMatchObject({ accepted: false });
  });

  it("allows a model-selected intermediate frontier only when it advances the Goal", () => {
    const observation = spatialObservation({
      id: "frontier:20:27",
      target: { x: 10.25, y: 0, z: 13.75 }
    });

    expect(alignHumanoidSkillToGoal({
      goal: {
        summary: "前往模型选择的目标前沿",
        predicates: [{
          type: "robot_at",
          target: { x: 8.75, y: 0, z: 14.75 },
          tolerance: 0.2
        }]
      },
      invocation: {
        skill: "explore",
        frontier_id: "frontier:20:27",
        strategy: "information_gain",
        maximum_travel_m: 8
      },
      observation
    })).toEqual({
      accepted: true,
      relation: "direct",
      predicateIndex: 0
    });
  });

  it("accepts matching object prerequisites but rejects work on an unrelated object", () => {
    const goal = {
      summary: "抓取目标箱体",
      predicates: [{
        type: "object_grasped" as const,
        object_id: "target_crate",
        hand: "right" as const
      }]
    };
    const observation = spatialObservation({
      id: "frontier:1:1",
      target: { x: 1, y: 0, z: 1 }
    });

    expect(alignHumanoidSkillToGoal({
      goal,
      invocation: {
        skill: "approach",
        object_id: "target_crate",
        interaction_point_id: null,
        hand: "right",
        standoff_m: 0.6
      },
      observation
    })).toMatchObject({ accepted: true, relation: "prerequisite" });
    expect(alignHumanoidSkillToGoal({
      goal,
      invocation: {
        skill: "approach",
        object_id: "unrelated_crate",
        interaction_point_id: null,
        hand: "right",
        standoff_m: 0.6
      },
      observation
    })).toMatchObject({ accepted: false });
  });
});

function spatialObservation(frontier: {
  id: string;
  target: { x: number; y: number; z: number };
}): HumanoidWorldObservation {
  return {
    robot: { rootPosition: { x: 14.65, y: 0.75, z: 12.52 } },
    spatialBelief: { frontiers: [frontier] },
    interaction: {
      zones: [],
      object_world_model: { objects: [] }
    }
  } as HumanoidWorldObservation;
}
