import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import type { BodyChannel } from "../domain/schema.js";
import { executeSkill, executeTool } from "../runtime/actions.js";
import { RapierWorld, type SourceCommand } from "./rapier-world.js";
import { ROBOT_SPEC } from "./robot-model.js";

function source(id: string, skill: string, channels: BodyChannel[]): SourceCommand {
  return {
    id,
    agentId: "retraction_test_agent",
    agentName: "Retraction test agent",
    skill,
    channels
  };
}

function detailRecord(value: unknown): Record<string, unknown> {
  expect(value).not.toBeNull();
  expect(typeof value).toBe("object");
  expect(Array.isArray(value)).toBe(false);
  return value as Record<string, unknown>;
}

describe("route-specific arm retraction", () => {
  it("offers model-selectable candidates without actuating and clears the rejected route", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("voxel_survey", 73);

    // This pose is taken from a real model run that repeatedly hit the same
    // finger/forearm voxel sweep while approaching the source block.
    scenario.robot = {
      x: 69.53379821777344,
      z: 27.50035285949707,
      yaw: 0.7325195183615504,
      joints: {
        ...scenario.robot.joints,
        shoulder: 0.3128958897281366,
        elbow: -1.9976304935616684,
        wrist: 0.03224354860962475
      }
    };
    const target = { x: 69.5, y: 0.45, z: 27.6671 };
    const facePoint = { x: 69.5, y: 0.45, z: 26.5 };
    const world = await RapierWorld.create(scenario);

    try {
      const blocked = await executeTool(world, "plan_base_path", {
        target,
        face_point: facePoint
      });
      expect(blocked).toMatchObject({ accepted: false, code: "base_path_collision" });

      const before = world.snapshot();
      const planned = await executeTool(world, "plan_arm_retraction", {
        target,
        face_point: facePoint
      });
      expect(planned).toMatchObject({ accepted: true, code: "arm_retraction_options" });
      expect(world.snapshot()).toMatchObject({
        frame: before.frame,
        world_revision: before.world_revision,
        robot: { joints: before.robot.joints }
      });

      const detail = detailRecord(planned.detail);
      expect(detail).toMatchObject({
        automatic_actuation: false,
        decision_owner: "model"
      });
      const candidates = detail.candidates as Array<{
        choice_id: string;
        targets: { shoulder: number; elbow: number; wrist: number };
      }>;
      expect(candidates.length).toBeGreaterThan(1);
      expect(candidates.length).toBeLessThanOrEqual(6);
      for (const candidate of candidates) {
        expect(candidate.choice_id).toMatch(/^arm_retraction_\d+$/);
        for (const joint of ["shoulder", "elbow", "wrist"] as const) {
          expect(candidate.targets[joint]).toBeGreaterThanOrEqual(
            ROBOT_SPEC.joints[joint].minimum
          );
          expect(candidate.targets[joint]).toBeLessThanOrEqual(
            ROBOT_SPEC.joints[joint].maximum
          );
        }
      }

      const chosen = candidates[0]!;
      const moved = await executeSkill(
        world,
        source("model_selected_retraction", "set_joint_targets", ["arm"]),
        "set_joint_targets",
        { targets: chosen.targets }
      );
      expect(moved).toMatchObject({ accepted: true, code: "joint_targets_reached" });
      expect(world.snapshot().world_revision).toBe(before.world_revision + 1);

      const replanned = await executeTool(world, "plan_base_path", {
        target,
        face_point: facePoint
      });
      expect(replanned).toMatchObject({ accepted: true, code: "base_path_planned" });
    } finally {
      world.dispose();
    }
  }, 30_000);
});
