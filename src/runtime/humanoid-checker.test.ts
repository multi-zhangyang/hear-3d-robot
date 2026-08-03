import { describe, expect, it } from "vitest";
import { GoalSchema, ScenarioSchema } from "../domain/schema.js";
import { HumanoidWorld } from "../world/humanoid/world.js";
import {
  advanceHumanoidGoal,
  assertHumanoidGoalProgressIntegrity,
  assertHumanoidGoalSupported,
  checkHumanoidGoal,
  createHumanoidGoalProgress,
  inspectHumanoidGoal
} from "./humanoid-checker.js";
import { add, rotateVector } from "../world/geometry.js";

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

  it("requires consecutive authoritative frames for world and pelvis end-effector goals", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      const pelvisPosition = { x: 6, y: 0.8, z: 6 };
      const halfAngle = Math.PI / 4;
      const pelvisRotation = {
        x: 0,
        y: Math.sin(halfAngle),
        z: 0,
        w: Math.cos(halfAngle)
      };
      const rightAnkleInPelvis = { x: -0.12, y: -0.74, z: 0.16 };
      const leftWristInWorld = { x: 6.35, y: 0.92, z: 6.18 };
      snapshot.robot.links.pelvis.position = pelvisPosition;
      snapshot.robot.links.pelvis.rotation = pelvisRotation;
      snapshot.robot.links.left_wrist_yaw_link.position = { ...leftWristInWorld };
      snapshot.robot.links.right_ankle_roll_link.position = add(
        pelvisPosition,
        rotateVector(pelvisRotation, rightAnkleInPelvis)
      );
      const goal = GoalSchema.parse({
        summary: "保持左腕和右踝末端位置",
        predicates: [
          {
            type: "end_effector_at",
            end_effector: "left_wrist",
            frame: "world",
            target: leftWristInWorld,
            tolerance: 0.01,
            stable_frames: 3
          },
          {
            type: "end_effector_at",
            end_effector: "right_ankle",
            frame: "pelvis",
            target: rightAnkleInPelvis,
            tolerance: 0.01,
            stable_frames: 3
          }
        ]
      });
      let progress = createHumanoidGoalProgress(goal, snapshot);
      expect(inspectHumanoidGoal(goal, scenario, snapshot, progress)).toMatchObject({
        success: false,
        checks: [
          { passed: false, actual: { current_stable_frames: 0 } },
          { passed: false, actual: { current_stable_frames: 0 } }
        ]
      });

      for (let expected = 1; expected <= 3; expected += 1) {
        snapshot.frame += 1;
        snapshot.worldRevision += 1;
        const advanced = advanceHumanoidGoal(goal, scenario, snapshot, progress);
        progress = advanced.progress;
        expect(advanced.checker.checks).toEqual(expect.arrayContaining([
          expect.objectContaining({
            passed: expected === 3,
            actual: expect.objectContaining({
              satisfied: true,
              current_stable_frames: expected,
              required_stable_frames: 3
            })
          })
        ]));
        expect(advanced.checker.success).toBe(expected === 3);
      }

      const duplicate = advanceHumanoidGoal(goal, scenario, snapshot, progress);
      expect(duplicate.progress).toEqual(progress);
      expect(duplicate.checker.success).toBe(true);

      snapshot.robot.links.left_wrist_yaw_link.position.x += 0.1;
      snapshot.frame += 1;
      snapshot.worldRevision += 1;
      const broken = advanceHumanoidGoal(goal, scenario, snapshot, progress);
      expect(broken.checker.success).toBe(false);
      expect(broken.progress.predicate_streaks).toEqual([0, 3]);

      snapshot.robot.links.left_wrist_yaw_link.position = { ...leftWristInWorld };
      snapshot.frame += 2;
      snapshot.worldRevision += 2;
      const afterGap = advanceHumanoidGoal(goal, scenario, snapshot, broken.progress);
      expect(afterGap.progress.predicate_streaks).toEqual([1, 1]);
      expect(afterGap.checker.success).toBe(false);
      assertHumanoidGoalProgressIntegrity(goal, snapshot, afterGap.progress);
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("rejects tampered, stale and partially advanced goal progress", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      const goal = GoalSchema.parse({
        summary: "保持左腕位置",
        predicates: [{
          type: "end_effector_at",
          end_effector: "left_wrist",
          frame: "world",
          target: snapshot.robot.links.left_wrist_yaw_link.position,
          tolerance: 0.05,
          stable_frames: 2
        }]
      });
      const progress = createHumanoidGoalProgress(goal, snapshot);
      expect(() => inspectHumanoidGoal(goal, scenario, snapshot, {
        ...progress,
        goal_sha256: "0".repeat(64)
      })).toThrow("another goal");
      expect(() => inspectHumanoidGoal(goal, scenario, snapshot, {
        ...progress,
        predicate_count: 2,
        predicate_streaks: [0, 0]
      })).toThrow("predicate count");
      expect(() => inspectHumanoidGoal(goal, scenario, snapshot, {
        ...progress,
        predicate_streaks: [3]
      })).toThrow("impossible stability streak");

      snapshot.frame += 1;
      expect(() => advanceHumanoidGoal(goal, scenario, snapshot, progress)).toThrow(
        "advance partially"
      );
    } finally {
      await world.dispose();
    }
  }, 30_000);
});
