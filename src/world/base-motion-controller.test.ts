import { describe, expect, it } from "vitest";
import type { Vec3 } from "../domain/schema.js";
import { normalizeAngle, planarDistance } from "./geometry.js";
import { controlBaseTowardTarget } from "./base-motion-controller.js";

describe("base motion controller", () => {
  it("recovers a reverse segment after its live pose drifts past the planned line", () => {
    const target: Vec3 = { x: 3.2, y: 0.38, z: 0.1 };
    let current: Vec3 = { x: 0.04, y: 0.38, z: -0.04 };
    let yaw = normalizeAngle(Math.atan2(3.2, 0.1) + Math.PI);
    const timestep = 1 / 60;

    for (let frame = 0; frame < 1_200 && planarDistance(current, target) > 0.015; frame += 1) {
      const control = controlBaseTowardTarget({
        current,
        target,
        yaw,
        linearSign: -1,
        maximumLinearVelocity: 0.55,
        maximumAngularVelocity: 1.6,
        timestep
      });
      const midpointYaw = normalizeAngle(yaw + control.angularVelocity * timestep / 2);
      yaw = normalizeAngle(yaw + control.angularVelocity * timestep);
      current = {
        x: current.x + Math.sin(midpointYaw) * control.linearVelocity * timestep,
        y: current.y,
        z: current.z + Math.cos(midpointYaw) * control.linearVelocity * timestep
      };
    }

    expect(planarDistance(current, target)).toBeLessThanOrEqual(0.015);
  });

  it("turns back toward a waypoint instead of accelerating away after an overshoot", () => {
    const control = controlBaseTowardTarget({
      current: { x: 3.4, y: 0.38, z: 0.1 },
      target: { x: 3.2, y: 0.38, z: 0.1 },
      yaw: -Math.PI / 2,
      linearSign: -1,
      maximumLinearVelocity: 0.55,
      maximumAngularVelocity: 1.6,
      timestep: 1 / 60
    });

    expect(Math.abs(control.headingError)).toBeGreaterThan(0.32);
    expect(control.linearVelocity).toBe(0);
    expect(control.angularVelocity).not.toBe(0);
  });
});
