import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { executeSkill, executeTool } from "../runtime/actions.js";
import { RapierWorld, type SourceCommand } from "./rapier-world.js";

describe("model-facing terrain frontier", () => {
  it("returns a reachable target plus facing point that expands the observed map", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("voxel_survey", 11);
    const world = await RapierWorld.create(scenario);
    try {
      const before = world.snapshot();
      const surveyed = world.surveyTerrain(12);
      expect(surveyed).toMatchObject({ accepted: true, code: "terrain_survey" });
      const detail = record(surveyed.detail);
      expect(detail.movement_sampling).toMatchObject({
        decision_owner: "model",
        automatic_actuation: false,
        strategy: "entropy_ordered_reachable_frontiers"
      });
      const frontier = detail.frontier;
      expect(Array.isArray(frontier)).toBe(true);
      if (!Array.isArray(frontier) || frontier.length === 0) throw new Error("Expected a frontier");
      const choice = record(frontier[0]);
      expect(choice.choice_id).toEqual(expect.any(String));
      const target = point(choice.target);
      const facePoint = point(choice.face_point);
      expect(Math.hypot(target.x - facePoint.x, target.z - facePoint.z)).toBeGreaterThan(0.05);
      expect(choice.travel_distance).toEqual(expect.any(Number));
      expect(choice.unseen_neighbours).toEqual(expect.any(Number));
      expect(choice.turn_degrees).toEqual(expect.any(Number));

      // These values are offered to the model as choices; the world does not
      // execute one itself. This test follows one choice through the same public
      // planning and physical execution tools an agent has to call.
      const planned = await executeTool(world, "plan_base_path", {
        target,
        face_point: facePoint
      });
      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
      const planId = record(planned.detail).plan_id;
      if (typeof planId !== "string") throw new Error("Planner did not return a plan ID");
      const command: SourceCommand = {
        id: "frontier_command",
        agentId: "frontier_agent",
        agentName: "Frontier agent",
        skill: "execute_base_plan",
        channels: ["base"]
      };
      const moved = await executeSkill(world, command, "execute_base_plan", { plan_id: planId });
      expect(moved).toMatchObject({ accepted: true, code: "base_plan_completed" });

      const after = world.snapshot();
      expect(after.explored.seen).toBeGreaterThan(before.explored.seen);
      expect(Math.hypot(
        after.robot.position.x - before.robot.position.x,
        after.robot.position.z - before.robot.position.z
      )).toBeGreaterThan(0.5);
      expect(after.navigation.face).toMatchObject({ x: facePoint.x, z: facePoint.z });
    } finally {
      world.dispose();
    }
  });

  it("rechecks the survey revision at the instant an atomic frontier command starts", async () => {
    const catalog = await loadRuntimeCatalog();
    const world = await RapierWorld.create(catalog.materialize("voxel_survey", 23));
    try {
      const surveyed = world.surveyTerrain(12);
      const frontier = record(surveyed.detail).frontier;
      if (!Array.isArray(frontier) || frontier.length === 0) throw new Error("Expected a frontier");
      const selected = record(frontier[0]);
      const surveyedRevision = world.snapshot().world_revision;
      const headMoved = await executeSkill(world, {
        id: "intervening_head_command",
        agentId: "sensor_agent",
        agentName: "Sensor agent",
        skill: "set_head_target",
        channels: ["head"]
      }, "set_head_target", { yaw: 0.25, pitch: -0.1 });
      expect(headMoved).toMatchObject({ accepted: true, code: "head_target_reached" });
      const beforeAttempt = world.snapshot();

      const result = await executeSkill(world, {
        id: "stale_frontier_command",
        agentId: "frontier_agent",
        agentName: "Frontier agent",
        skill: "navigate_frontier",
        channels: ["base"]
      }, "navigate_frontier", {
        survey_transaction_id: "frontier_agent:survey",
        survey_world_revision: surveyedRevision,
        choice_id: selected.choice_id,
        target: point(selected.target),
        face_point: point(selected.face_point)
      });

      expect(result).toMatchObject({
        accepted: false,
        code: "stale_survey_revision",
        detail: {
          surveyed_world_revision: surveyedRevision,
          current_world_revision: beforeAttempt.world_revision
        }
      });
      expect(world.snapshot().robot.position).toEqual(beforeAttempt.robot.position);
    } finally {
      world.dispose();
    }
  });
});

function record(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Expected a record");
  }
  return value as Record<string, unknown>;
}

function point(value: unknown): { x: number; y: number; z: number } {
  const candidate = record(value);
  if (typeof candidate.x !== "number" || typeof candidate.y !== "number"
    || typeof candidate.z !== "number") throw new Error("Expected a point");
  return { x: candidate.x, y: candidate.y, z: candidate.z };
}
