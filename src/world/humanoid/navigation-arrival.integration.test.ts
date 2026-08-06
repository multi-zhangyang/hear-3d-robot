import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { yawFromQuaternion } from "../geometry.js";
import { HumanoidWorld } from "./world.js";

const scenario = ScenarioSchema.parse({
  title: "Arrival heading field",
  seed: 41,
  bounds: { width: 6, depth: 6 },
  visibility_radius: 4,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [],
  zones: [],
  default_goal: {
    summary: "到达并面向目标",
    predicates: [{
      type: "robot_at",
      target: { x: 2, y: 0, z: 2.35 },
      tolerance: 0.2
    }]
  }
});

describe("humanoid navigation arrival heading", () => {
  it("previews and executes a model-selected final facing constraint in MuJoCo", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const target = { x: 2, y: 0, z: 2.35 };
      const arrivalHeading = {
        type: "face_point" as const,
        target: { x: 3, y: 0.7, z: 2.35 },
        tolerance_radians: 0.15
      };
      const planned = await world.planNavigation(target, arrivalHeading);

      expect(planned.accepted, planned.reason).toBe(true);
      expect(planned.requestedArrivalHeading).toEqual(arrivalHeading);
      expect(planned.arrivalHeading).toEqual(arrivalHeading);

      const executed = await world.executeNavigation(planned.planId);
      const final = executed.finalSnapshot.robot;
      const desiredYaw = Math.atan2(
        arrivalHeading.target.x - final.rootPosition.x,
        arrivalHeading.target.z - final.rootPosition.z
      );
      const yawError = Math.atan2(
        Math.sin(desiredYaw - yawFromQuaternion(final.rootRotation)),
        Math.cos(desiredYaw - yawFromQuaternion(final.rootRotation))
      );

      expect(executed).toMatchObject({
        accepted: true,
        code: "navigation_completed"
      });
      expect(Math.hypot(
        target.x - final.rootPosition.x,
        target.z - final.rootPosition.z
      )).toBeLessThanOrEqual(0.18);
      expect(Math.abs(yawError)).toBeLessThanOrEqual(
        arrivalHeading.tolerance_radians
      );
      expect(final.fallen).toBe(false);
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("brakes a short learned-locomotion approach without passing its precision target", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const target = { x: 2, y: 0, z: 2.18 };
      const planned = await world.planNavigation(target, {
        type: "yaw",
        yaw_radians: 0,
        tolerance_radians: 0.15
      });

      expect(planned.accepted, planned.reason).toBe(true);
      const executed = await world.executeNavigation(planned.planId);
      const final = executed.finalSnapshot.robot.rootPosition;

      expect(executed).toMatchObject({
        accepted: true,
        code: "navigation_completed"
      });
      expect(Math.hypot(target.x - final.x, target.z - final.z)).toBeLessThanOrEqual(0.04);
      expect(final.z).toBeLessThanOrEqual(target.z + 0.04);
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("holds a short arrival position while aligning a distinct final heading", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const target = { x: 2.155, y: 0, z: 2.09 };
      const arrivalHeading = {
        type: "yaw" as const,
        yaw_radians: 0.24,
        tolerance_radians: 0.15
      };
      const planned = await world.planNavigation(target, arrivalHeading);

      expect(planned.accepted, planned.reason).toBe(true);
      const executed = await world.executeNavigation(planned.planId);
      const final = executed.finalSnapshot.robot;
      const yawError = Math.atan2(
        Math.sin(arrivalHeading.yaw_radians - yawFromQuaternion(final.rootRotation)),
        Math.cos(arrivalHeading.yaw_radians - yawFromQuaternion(final.rootRotation))
      );

      expect(executed).toMatchObject({ accepted: true, code: "navigation_completed" });
      expect(Math.hypot(
        target.x - final.rootPosition.x,
        target.z - final.rootPosition.z
      )).toBeLessThanOrEqual(0.061);
      expect(Math.abs(yawError)).toBeLessThanOrEqual(arrivalHeading.tolerance_radians);
    } finally {
      await world.dispose();
    }
  }, 30_000);
});
