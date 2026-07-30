import { describe, expect, it, vi } from "vitest";
import type { WorldSnapshot } from "../types";
import {
  AuthoritativeFrameBuffer,
  interpolateWorldFrame
} from "./authoritative-frame-buffer";

describe("AuthoritativeFrameBuffer", () => {
  it("orders an SSE batch and presents every authoritative physics frame", () => {
    const buffer = new AuthoritativeFrameBuffer(worldFrame(0));
    buffer.sample(0, true);

    expect(buffer.push([worldFrame(3), worldFrame(1), worldFrame(2)])).toBe(3);
    expect(buffer.snapshots().map((entry) => entry.frame)).toEqual([0, 1, 2, 3]);

    expect(buffer.sample(0, true)?.frame).toBe(0);
    expect(buffer.sample(1000 / 60, true)?.frame).toBeCloseTo(1, 5);
    expect(buffer.sample(2000 / 60, true)?.frame).toBeCloseTo(2, 5);
    expect(buffer.sample(3000 / 60, true)).toBe(buffer.latest);
  });

  it("ignores late frames and replaces a duplicate terminal frame in place", () => {
    const buffer = new AuthoritativeFrameBuffer(worldFrame(4));
    buffer.push([worldFrame(5), worldFrame(6)]);
    const terminal = worldFrame(6, { x: 42, active: false });

    expect(buffer.push([worldFrame(3), worldFrame(5), terminal])).toBe(1);
    expect(buffer.snapshots().map((entry) => entry.frame)).toEqual([4, 5, 6]);
    expect(buffer.latest).toBe(terminal);
    expect(buffer.latest?.robot.position.x).toBe(42);
    expect(buffer.latest?.active_commands).toEqual([]);
  });

  it("never extrapolates past the newest received snapshot", () => {
    const buffer = new AuthoritativeFrameBuffer(worldFrame(0));
    buffer.sample(0, true);
    const latest = worldFrame(1);
    buffer.push([latest]);
    buffer.sample(0, true);

    expect(buffer.sample(60_000, true)).toBe(latest);
    expect(buffer.sample(120_000, true)).toBe(latest);
    expect(buffer.pending).toBe(false);
  });

  it("snaps a stopped run to its exact terminal snapshot", () => {
    const buffer = new AuthoritativeFrameBuffer(worldFrame(0));
    buffer.sample(0, true);
    const terminal = worldFrame(1, { x: 1, active: false });
    buffer.push([terminal]);
    buffer.sample(0, true);

    const movingPose = buffer.sample(1000 / 120, true);
    expect(movingPose?.robot.position.x).toBeCloseTo(0.5, 5);
    expect(buffer.sample(1000 / 120, false)).toBe(terminal);
    expect(buffer.pending).toBe(false);
  });

  it("notifies consumers once for a changed batch and not for stale input", () => {
    const buffer = new AuthoritativeFrameBuffer(worldFrame(0));
    const listener = vi.fn();
    const unsubscribe = buffer.subscribe(listener);

    buffer.push([worldFrame(1), worldFrame(2)]);
    buffer.push([worldFrame(1)]);
    unsubscribe();
    buffer.push([worldFrame(3)]);

    expect(listener).toHaveBeenCalledTimes(1);
  });
});

describe("interpolateWorldFrame", () => {
  it("interpolates link, object and wheel poses without mutating authoritative inputs", () => {
    const left = worldFrame(0);
    const right = worldFrame(1, { x: 1, yaw: Math.PI / 2 });
    right.robot.links.base!.rotation = { x: 0, y: Math.SQRT1_2, z: 0, w: Math.SQRT1_2 };
    right.objects[0]!.position.x = 2;
    const visual = interpolateWorldFrame(left, right, 0.5);

    expect(visual.robot.position.x).toBeCloseTo(0.5, 6);
    expect(visual.robot.links.base!.position.x).toBeCloseTo(0.5, 6);
    expect(visual.robot.links.base!.rotation.y).toBeCloseTo(Math.sin(Math.PI / 8), 6);
    expect(visual.objects[0]!.position.x).toBeCloseTo(1, 6);
    expect(left.robot.position.x).toBe(0);
    expect(right.robot.position.x).toBe(1);
  });
});

function worldFrame(
  index: number,
  options: { x?: number; yaw?: number; active?: boolean } = {}
): WorldSnapshot {
  const x = options.x ?? index;
  const yaw = options.yaw ?? 0;
  const active = options.active ?? true;
  const position = { x, y: 0.38, z: 0 };
  const rotation = { x: 0, y: 0, z: 0, w: 1 };
  const velocity = { x: active ? 1 : 0, y: 0, z: 0 };
  const command = active ? {
    id: "command",
    agent_id: "agent",
    agent_name: "Agent",
    skill: "drive_base",
    phase: "executing",
    channels: ["base" as const]
  } : null;
  return {
    frame: index,
    simulated_time: index / 60,
    world_revision: index,
    robot: {
      position,
      yaw,
      joints: {
        head_yaw: 0,
        head_pitch: 0,
        shoulder: 0,
        elbow: 0,
        wrist: 0,
        gripper_aperture: 0.4
      },
      contacts: {
        left_object_id: null,
        right_object_id: null,
        left_force: 0,
        right_force: 0
      },
      attachment: null,
      odometry: {
        left_wheel: { position: x, velocity: active ? 1 : 0 },
        right_wheel: { position: x, velocity: active ? 1 : 0 }
      },
      joint_status: {
        shoulder: {
          position: x,
          velocity: active ? 1 : 0,
          target: 1,
          minimum: -1,
          maximum: 1,
          maximum_velocity: 1
        }
      },
      links: {
        base: { position, rotation, linear_velocity: velocity, angular_velocity: { x: 0, y: 0, z: 0 } }
      },
      gripper: {
        aperture: 0.4,
        target_aperture: 0.4,
        maximum_force: 1000,
        left_contact_object_id: null,
        right_contact_object_id: null,
        left_contact_force: 0,
        right_contact_force: 0
      }
    },
    objects: [{
      id: "block",
      kind: "block",
      color: "#fff",
      position: { x, y: 0.25, z: 1 },
      rotation,
      linear_velocity: velocity,
      angular_velocity: { x: 0, y: 0, z: 0 },
      size: { x: 0.5, y: 0.5, z: 0.5 },
      portable: true,
      locked: false,
      container_id: null,
      enabled: true,
      visible: true
    }],
    zones: [],
    obstacles: [],
    explored: { cells: "", seen: 0, total: 0 },
    voxels: null,
    active_command: command,
    active_commands: command ? [command] : [],
    last_command: null,
    navigation: {
      plan_id: null,
      status: "idle",
      target: null,
      face: null,
      waypoints: [],
      waypoint_index: null,
      distance: null,
      planned_at_frame: null,
      actual_path: []
    },
    plans: { base: [], arm: [] },
    affordance_events: []
  };
}
