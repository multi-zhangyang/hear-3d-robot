import { describe, expect, it } from "vitest";
import { GoalSchema, ScenarioSchema } from "../domain/schema.js";
import { HumanoidWorld } from "../world/humanoid/world.js";
import {
  assertHumanoidGoalSupported,
  checkHumanoidGoal
} from "./humanoid-checker.js";

const scenario = ScenarioSchema.parse({
  title: "人形目标检查场",
  seed: 41,
  bounds: { width: 12, depth: 12 },
  visibility_radius: 7,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [{
    id: "crate",
    kind: "crate",
    color: "#b87942",
    position: { x: 8, y: 0.15, z: 8 },
    size: { x: 0.3, y: 0.3, z: 0.3 },
    portable: true
  }],
  zones: [{
    id: "arrival",
    color: "#55a88b",
    center: { x: 8, y: 0.01, z: 8 },
    size: { x: 2, y: 0.02, z: 2 }
  }],
  default_goal: {
    summary: "进入区域",
    predicates: [{ type: "robot_in_zone", zone_id: "arrival", tolerance: 0.1 }]
  }
});

describe("humanoid goal checker", () => {
  it("checks robot and physical object positions against one authoritative frame", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      snapshot.robot.rootPosition = { x: 8, y: snapshot.robot.rootPosition.y, z: 8 };
      snapshot.robot.objects.crate!.position = { x: 8, y: 0.15, z: 8 };
      const goal = GoalSchema.parse({
        summary: "机器人与箱体到达目标",
        predicates: [
          { type: "robot_at", target: { x: 8, y: 0, z: 8 }, tolerance: 0.1 },
          { type: "robot_in_zone", zone_id: "arrival", tolerance: 0.05 },
          {
            type: "object_at",
            object_id: "crate",
            target: { x: 8, y: 0.15, z: 8 },
            tolerance: 0.05
          },
          {
            type: "object_in_zone",
            object_id: "crate",
            zone_id: "arrival",
            expected: true,
            tolerance: 0.03
          }
        ]
      });

      assertHumanoidGoalSupported(goal, scenario);
      const result = checkHumanoidGoal(goal, scenario, snapshot);
      expect(result.success).toBe(true);
      expect(result.worldFrame).toBe(snapshot.frame);
      expect(result.worldRevision).toBe(snapshot.worldRevision);
      expect(result.checks).toHaveLength(4);
      expect(result.checks.every((check) => check.passed)).toBe(true);

      snapshot.robot.objects.crate!.position = { x: 5, y: 0.15, z: 5 };
      const moved = checkHumanoidGoal(goal, scenario, snapshot);
      expect(moved.success).toBe(false);
      expect(moved.checks.slice(2).every((check) => !check.passed)).toBe(true);
    } finally {
      await world.dispose();
    }
  }, 30_000);
});
