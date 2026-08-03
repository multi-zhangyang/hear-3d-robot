import { describe, expect, it } from "vitest";
import type { HumanoidObjectSnapshot } from "./simulation.js";
import {
  historicalHumanoidObjectState,
  HumanoidAuthoritativeObjectFrame
} from "./object-state.js";

const descriptors = [
  {
    id: "crate",
    kind: "crate",
    size: { x: 0.4, y: 0.3, z: 0.5 },
    portable: true
  },
  {
    id: "landmark",
    kind: "column",
    size: { x: 1, y: 2, z: 1 },
    portable: false
  }
] as const;

describe("HumanoidAuthoritativeObjectFrame", () => {
  it("separates current MuJoCo authority from historical sensor slots", () => {
    const frame = new HumanoidAuthoritativeObjectFrame(descriptors);
    const initial = {
      crate: object("crate", 1),
      landmark: object("landmark", 3)
    };
    frame.refresh(12, 7, initial, new Set(["crate"]));

    expect(frame.activeStates()).toEqual([
      expect.objectContaining({
        id: "crate",
        role: "manipulable",
        state: "active",
        authority: "mujoco_exact",
        exact: true,
        observable: true,
        frame: 12,
        worldRevision: 7,
        observedFrame: 12,
        observedWorldRevision: 7,
        pose: expect.objectContaining({ position: initial.crate.position })
      }),
      expect.objectContaining({
        id: "landmark",
        role: "fixture",
        state: "active",
        authority: "mujoco_exact",
        exact: true,
        observable: false,
        observedFrame: null,
        observedWorldRevision: null
      })
    ]);
    expect(frame.observableStates(12, 7).map((state) => state.id)).toEqual(["crate"]);

    frame.refresh(12, 7, {
      crate: object("crate", 2),
      landmark: initial.landmark
    }, new Set());
    expect(frame.observableStates(12, 7)).toEqual([]);
    expect(frame.activeStates()[0]).toMatchObject({
      id: "crate",
      observable: false,
      exact: true,
      pose: { position: { x: 2, y: 0.2, z: 0 } }
    });

    const historical = historicalHumanoidObjectState(descriptors[0], {
      id: "crate",
      position: initial.crate.position,
      rotation: initial.crate.rotation,
      linearVelocity: initial.crate.linearVelocity,
      angularVelocity: initial.crate.angularVelocity,
      lastSeenFrame: 12,
      lastSeenRevision: 7
    });
    expect(historical).toMatchObject({
      id: "crate",
      state: "historical",
      authority: "sensor_history",
      exact: false,
      observable: false,
      frame: 12,
      worldRevision: 7,
      observedFrame: 12,
      observedWorldRevision: 7,
      pose: { position: initial.crate.position }
    });
  });

  it("never serves a current observation across a frame or revision boundary", () => {
    const frame = new HumanoidAuthoritativeObjectFrame(descriptors);
    frame.refresh(4, 9, { crate: object("crate", 1) }, new Set(["crate"]));

    expect(frame.observableStates(4, 9)).toHaveLength(1);
    expect(frame.observableStates(5, 9)).toEqual([]);
    expect(frame.observableStates(4, 10)).toEqual([]);
  });

  it("rejects sensor identities without a matching authoritative object", () => {
    const frame = new HumanoidAuthoritativeObjectFrame(descriptors);
    expect(() => frame.refresh(1, 1, {}, new Set(["crate"]))).toThrow(
      "Observable humanoid object has no authoritative state: crate"
    );
  });
});

function object(id: string, x: number): HumanoidObjectSnapshot {
  return {
    id,
    position: { x, y: 0.2, z: 0 },
    rotation: { x: 0, y: 0, z: 0, w: 1 },
    linearVelocity: { x: 0, y: 0, z: 0 },
    angularVelocity: { x: 0, y: 0, z: 0 }
  };
}
