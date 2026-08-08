import { describe, expect, it } from "vitest";
import { yawFromQuaternion } from "../geometry.js";
import { HUMANOID_JOINT_INDEX } from "./model.js";
import {
  neutralHumanoidReference,
  targetReference
} from "./reference.js";
import {
  HumanoidSimulation,
  type HumanoidSimulationSnapshot
} from "./simulation.js";
import {
  captureHumanoidStationKeepingAnchor,
  stationKeepingHumanoidReference
} from "./station-keeping.js";

describe("humanoid station keeping", () => {
  it("converts world position and heading error into bounded local feedback", () => {
    const anchor = captureHumanoidStationKeepingAnchor(
      snapshot(4, 4, 0),
      12,
      14
    );
    const correction = stationKeepingHumanoidReference(
      neutralHumanoidReference(),
      snapshot(4.5, 3.5, Math.PI / 2),
      anchor
    );

    expect(anchor).toEqual({
      x: 4,
      z: 4,
      yaw: 0,
      sourceFrame: 12,
      sourceWorldRevision: 14
    });
    expect(correction.rootVelocity[0]).toBeCloseTo(-0.3, 12);
    expect(correction.rootVelocity[1]).toBeCloseTo(-0.2, 12);
    expect(correction.rootYawVelocity).toBeCloseTo(-1, 12);
  });

  it("does not add a command when the authoritative pose is at its anchor", () => {
    const state = snapshot(2, 3, -0.4);
    const reference = stationKeepingHumanoidReference(
      neutralHumanoidReference(),
      state,
      captureHumanoidStationKeepingAnchor(state, 0, 0)
    );

    expect(reference.rootVelocity).toEqual([0, 0]);
    expect(reference.rootYawVelocity).toBeCloseTo(0, 12);
  });

  it("overrides inherited planar velocity when the plan has no root authority", () => {
    const state = snapshot(2, 3, -0.4);
    const inherited = targetReference(neutralHumanoidReference(), {
      rootVelocity: [0.12, -0.08],
      rootYawVelocity: 0.3
    });
    const reference = stationKeepingHumanoidReference(
      inherited,
      state,
      captureHumanoidStationKeepingAnchor(state, 0, 0)
    );

    expect(reference.rootVelocity).toEqual([0, 0]);
    expect(reference.rootYawVelocity).toBeCloseTo(0, 12);
  });

  it("inverts the locomotion deadzone before drift becomes large", () => {
    const anchorState = snapshot(2, 3, 0);
    const reference = stationKeepingHumanoidReference(
      neutralHumanoidReference(),
      snapshot(2, 2.97, 0),
      captureHumanoidStationKeepingAnchor(anchorState, 0, 0)
    );

    expect(reference.rootVelocity[0]).toBeGreaterThan(0.08);
    expect(reference.rootVelocity[0]).toBeLessThan(0.15);
    expect(reference.rootVelocity[1]).toBe(0);
  });

  it("damps measured pelvis motion before it can cross the anchor", () => {
    const anchorState = snapshot(2, 3, 0);
    const moving = snapshot(2, 3.03, 0);
    moving.links = {
      pelvis: {
        linearVelocity: { x: 0, y: 0, z: 0.12 }
      }
    } as HumanoidSimulationSnapshot["links"];
    const reference = stationKeepingHumanoidReference(
      neutralHumanoidReference(),
      moving,
      captureHumanoidStationKeepingAnchor(anchorState, 0, 0)
    );

    expect(reference.rootVelocity[0]).toBeLessThan(0);
  });

  it("slews policy commands at the controller acceleration boundary", () => {
    const anchorState = snapshot(2, 3, 0);
    const reference = stationKeepingHumanoidReference(
      neutralHumanoidReference(),
      snapshot(2, 2.8, 0),
      captureHumanoidStationKeepingAnchor(anchorState, 0, 0),
      {
        previousPlanarCommand: [0, 0],
        controlStepSeconds: 0.02
      }
    );

    expect(reference.rootVelocity).toEqual([0.02, 0]);
  });

  it("releases a tracked posture before base drift can become a runaway", () => {
    const anchorState = snapshot(2, 3, 0);
    const tracked = targetReference(neutralHumanoidReference(), {
      joints: { left_elbow_joint: 0.6 }
    });
    const reference = stationKeepingHumanoidReference(
      tracked,
      snapshot(2.21, 3, 0),
      captureHumanoidStationKeepingAnchor(anchorState, 0, 0)
    );

    expect(reference.jointTrackingWeights.every((weight) => weight === 0)).toBe(true);
    expect(reference.rootVelocity[1]).toBeLessThan(0);
  });

  it("preserves explicit lower-body authority only for a stance task", () => {
    const state = snapshot(2, 3, 0);
    const tracked = targetReference(neutralHumanoidReference(), {
      joints: {
        left_hip_pitch_joint: -0.2,
        left_knee_joint: 0.5,
        waist_pitch_joint: 0.1
      }
    });
    const reference = stationKeepingHumanoidReference(
      tracked,
      state,
      captureHumanoidStationKeepingAnchor(state, 0, 0),
      { preserveTrackedLowerBody: true }
    );

    for (const joint of [
      "left_hip_pitch_joint",
      "left_knee_joint",
      "waist_pitch_joint"
    ] as const) {
      expect(reference.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get(joint)!
      ]).toBe(1);
    }
  });

  it("hands lower-body authority back to the balance policy at the root boundary", () => {
    const anchorState = snapshot(2, 3, 0);
    const tracked = targetReference(neutralHumanoidReference(), {
      joints: {
        left_hip_pitch_joint: -0.2,
        left_knee_joint: 0.5,
        right_ankle_pitch_joint: -0.25
      }
    });
    const reference = stationKeepingHumanoidReference(
      tracked,
      snapshot(2, 3.08, 0),
      captureHumanoidStationKeepingAnchor(anchorState, 0, 0),
      { preserveTrackedLowerBody: true }
    );

    for (const joint of [
      "left_hip_pitch_joint",
      "left_knee_joint",
      "right_ankle_pitch_joint"
    ] as const) {
      const index = HUMANOID_JOINT_INDEX.get(joint)!;
      expect(reference.jointTrackingWeights[index]).toBeCloseTo(0, 12);
      expect(reference.jointPositions[index]).toBeCloseTo(
        neutralHumanoidReference().jointPositions[index]!,
        12
      );
    }
  });

  it("holds a real YAHMP humanoid in place across a long MuJoCo run", async () => {
    const simulation = await HumanoidSimulation.create({
      spawn: { position: { x: 4, y: 0, z: 4 }, yaw: 0 }
    });
    try {
      let reference = neutralHumanoidReference();
      for (let index = 0; index < 80; index += 1) {
        await simulation.step(reference);
      }
      const initial = simulation.snapshot();
      const anchor = captureHumanoidStationKeepingAnchor(initial, 0, 0);
      let maximumPlanarDrift = 0;
      for (let index = 0; index < 4_050; index += 1) {
        const current = simulation.snapshot();
        reference = stationKeepingHumanoidReference(reference, current, anchor);
        const advanced = await simulation.step(reference);
        maximumPlanarDrift = Math.max(
          maximumPlanarDrift,
          Math.hypot(
            advanced.rootPosition.x - anchor.x,
            advanced.rootPosition.z - anchor.z
          )
        );
      }
      const final = simulation.snapshot();
      const yawError = Math.atan2(
        Math.sin(yawFromQuaternion(final.rootRotation) - anchor.yaw),
        Math.cos(yawFromQuaternion(final.rootRotation) - anchor.yaw)
      );

      expect(final.simulatedTime - initial.simulatedTime).toBeCloseTo(81, 6);
      expect(maximumPlanarDrift).toBeLessThan(0.04);
      expect(Math.abs(yawError)).toBeLessThan(0.04);
      expect(final.fallen).toBe(false);
      expect(final.balance.support).toBe("double");
      expect(final.rootPosition.y).toBeGreaterThan(0.7);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);
});

function snapshot(x: number, z: number, yaw: number): HumanoidSimulationSnapshot {
  return {
    rootPosition: { x, y: 0.79, z },
    rootRotation: {
      x: 0,
      y: Math.sin(yaw / 2),
      z: 0,
      w: Math.cos(yaw / 2)
    }
  } as HumanoidSimulationSnapshot;
}
