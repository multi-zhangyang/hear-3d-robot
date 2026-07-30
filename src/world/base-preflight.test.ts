import { describe, expect, it } from "vitest";
import { resolveBasePlan } from "./base-preflight.js";
import type { CollisionIssue } from "./collision.js";

describe("base path preflight", () => {
  it("adds a collision-checked egress before turning from an existing penetration", () => {
    const collisionAt = (position: { z: number }): CollisionIssue[] => {
      const depth = 0.2 - position.z;
      return depth > 0
        ? [{
            segment: "base",
            collider_kind: "voxel",
            collider_id: "voxel_a",
            penetration_depth: depth
          }]
        : [];
    };
    const result = resolveBasePlan({
      waypoints: [
        { x: 0, y: 0.38, z: 0 },
        { x: 0, y: 0.38, z: 1 }
      ],
      start: { x: 0, y: 0.38, z: 0 },
      yaw: 0,
      collisionsAt: (position) => collisionAt(position)
    });

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.segments[0]).toMatchObject({
      waypointIndex: 0,
      bodyYaw: 0,
      linearSign: 1
    });
    expect(result.segments[0]?.target.z).toBeGreaterThanOrEqual(0.2);
    expect(result.segments.at(-1)?.target).toMatchObject({ x: 0, z: 1 });
  });

  it("does not invent an egress through a collision that only deepens", () => {
    const result = resolveBasePlan({
      waypoints: [
        { x: 0, y: 0.38, z: 0 },
        { x: 0, y: 0.38, z: 1 }
      ],
      start: { x: 0, y: 0.38, z: 0 },
      yaw: 0,
      collisionsAt: (position) => [{
        segment: "base",
        collider_kind: "voxel",
        collider_id: "sealed",
        penetration_depth: 0.1 + Math.abs(position.z)
      }]
    });

    expect(result.ok).toBe(false);
  });

  it("creates clearance before a turn whose articulated sweep is initially blocked", () => {
    const result = resolveBasePlan({
      waypoints: [
        { x: 0, y: 0.38, z: 0 },
        { x: 1, y: 0.38, z: 1 }
      ],
      start: { x: 0, y: 0.38, z: 0 },
      yaw: 0,
      collisionsAt: (position, yaw) =>
        position.z < 0.16 && Math.abs(yaw) > 0.01
          ? [{
              segment: "right_finger",
              collider_kind: "voxel",
              collider_id: "turn_sweep",
              penetration_depth: 0.01
            }]
          : []
    });

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.segments).toHaveLength(2);
    expect(result.segments[0]).toMatchObject({ waypointIndex: 0, linearSign: 1 });
    expect(result.segments[0]?.target.z).toBeGreaterThanOrEqual(0.16);
    expect(result.segments[1]?.target).toMatchObject({ x: 1, z: 1 });
  });
});
