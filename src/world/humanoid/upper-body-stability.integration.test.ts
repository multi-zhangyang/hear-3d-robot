import { describe, expect, it } from "vitest";
import { prepareHumanoidMotion } from "./motion-plan.js";
import { neutralHumanoidReference } from "./reference.js";
import { HumanoidSimulation } from "./simulation.js";
import { yawFromQuaternion } from "../geometry.js";

describe("hierarchical humanoid upper-body control", () => {
  it("reaches through the arm while keeping the learned lower-body policy stable", async () => {
    const simulation = await HumanoidSimulation.create({
      spawn: {
        position: { x: 7.5, y: 0, z: 5.07 },
        yaw: 0.18
      }
    });
    try {
      const reference = neutralHumanoidReference();
      for (let index = 0; index < 100; index += 1) {
        await simulation.step(reference);
      }
      const before = simulation.snapshot();
      const yaw = yawFromQuaternion(before.rootRotation);
      const forward = 0.38;
      const lateral = -0.16;
      const interactionPoint = {
        x: before.rootPosition.x + forward * Math.sin(yaw) + lateral * Math.cos(yaw),
        y: 1,
        z: before.rootPosition.z + forward * Math.cos(yaw) - lateral * Math.sin(yaw)
      };
      const palm = simulation.handSurfaceObservations(before).find(
        ({ handSurface }) => handSurface === "right_hand_palm_link"
      );
      if (!palm) throw new Error("Right palm contact surface is unavailable");
      const target = {
        x: interactionPoint.x - palm.surfaceFromWristWorld.x,
        y: interactionPoint.y - palm.surfaceFromWristWorld.y,
        z: interactionPoint.z - palm.surfaceFromWristWorld.z
      };

      const prepared = await prepareHumanoidMotion(simulation, {
        id: "stable-upper-body-reach",
        intent: "reach a forward interaction point without moving the support base",
        duration_seconds: 6,
        keyframes: [{
          at_seconds: 0,
          right_hand: {
            position: { ...before.links.right_wrist_yaw_link.position },
            frame: "world",
            tolerance_m: 0.05
          }
        }, {
          at_seconds: 6,
          right_hand: {
            position: target,
            frame: "world",
            tolerance_m: 0.08
          }
        }]
      }, reference);

      expect(
        prepared.validation.feasible,
        JSON.stringify({
          failures: prepared.validation.failures,
          evidence: prepared.validation.evidence,
          root: prepared.validation.finalSnapshot.rootPosition,
          wrist: prepared.validation.finalSnapshot.links.right_wrist_yaw_link.position
        })
      ).toBe(true);
      expect(prepared.validation.finalSnapshot.fallen).toBe(false);
      expect(prepared.validation.evidence.travelledDistance).toBeLessThan(0.12);
      expect(prepared.validation.finalSnapshot.balance.upright).toBeGreaterThan(0.9);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);
});
