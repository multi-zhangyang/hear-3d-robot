import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import type { BodyChannel } from "../domain/schema.js";
import { executeSkill, executeTool } from "../runtime/actions.js";
import { armIkSeeds } from "./arm-ik-seeds.js";
import { RapierWorld, type SourceCommand } from "./rapier-world.js";
import { ROBOT_SPEC } from "./robot-model.js";
import { VOXEL_INTERACTION_CLEARANCE } from "./voxel-interaction.js";

function source(id: string, skill: string, channels: BodyChannel[]): SourceCommand {
  return {
    id,
    agentId: "position_ik_test_agent",
    agentName: "Position IK test agent",
    skill,
    channels
  };
}

function detail(value: unknown): Record<string, unknown> {
  expect(value).not.toBeNull();
  expect(typeof value).toBe("object");
  expect(Array.isArray(value)).toBe(false);
  return value as Record<string, unknown>;
}

describe("position-only arm IK", () => {
  it("spans joint limits with bounded low-discrepancy numerical starts", () => {
    const seeds = armIkSeeds();
    expect(seeds).toHaveLength(48);
    expect(new Set(seeds.map((seed) => JSON.stringify(seed))).size).toBe(seeds.length);
    for (const seed of seeds) {
      for (const joint of ["shoulder", "elbow", "wrist"] as const) {
        expect(seed[joint]).toBeGreaterThanOrEqual(ROBOT_SPEC.joints[joint].minimum);
        expect(seed[joint]).toBeLessThanOrEqual(ROBOT_SPEC.joints[joint].maximum);
      }
    }
  });

  it("escapes a real retracted-pose solver stall without moving during planning", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("voxel_survey", 73);
    scenario.robot = {
      x: 70.59767150878906,
      z: 26.5341739654541,
      yaw: -1.6117512856271317,
      joints: {
        ...scenario.robot.joints,
        shoulder: 1.5424816954009641,
        elbow: -0.3017270977576265,
        wrist: 1.6916752024507278
      }
    };
    const world = await RapierWorld.create(scenario);

    try {
      const before = world.snapshot();
      const planned = await executeTool(world, "solve_end_effector_position", {
        position: {
          x: 69.5,
          y: 0.9 + VOXEL_INTERACTION_CLEARANCE,
          z: 26.5
        }
      });
      expect(planned).toMatchObject({ accepted: true, code: "end_effector_solution" });
      expect(world.snapshot()).toMatchObject({
        frame: before.frame,
        world_revision: before.world_revision,
        robot: { joints: before.robot.joints }
      });
      const plannedDetail = detail(planned.detail);
      expect(plannedDetail.orientation_error).toBeNull();
      expect(plannedDetail.numerical_search).toMatchObject({
        strategy: "bounded_low_discrepancy_ik_restarts",
        automatic_actuation: false
      });
      const planId = plannedDetail.plan_id;
      expect(typeof planId).toBe("string");
      if (typeof planId !== "string") return;

      const executed = await executeSkill(
        world,
        source("execute_position_ik", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: planId }
      );
      expect(executed).toMatchObject({ accepted: true, code: "joint_targets_reached" });
      expect(world.snapshot().world_revision).toBe(before.world_revision + 1);
    } finally {
      world.dispose();
    }
  }, 30_000);
});
