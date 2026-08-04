import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import type { HumanoidObjectSnapshot } from "./simulation.js";
import { humanoidDynamicNavigationObstacles } from "./navigation-obstacles.js";

describe("humanoid dynamic navigation obstacles", () => {
  it("keeps baked solids out of the dynamic tile cache and tracks live objects", () => {
    const scenario = ScenarioSchema.parse({
      title: "Navigation obstacle ownership",
      seed: 9,
      bounds: { width: 12, depth: 12 },
      visibility_radius: 6,
      robot: { x: 1, z: 1, yaw: 0 },
      obstacles: [{
        id: "baked-wall",
        center: { x: 4, y: 0.5, z: 4 },
        size: { x: 2, y: 1, z: 0.5 }
      }],
      objects: [{
        id: "fixed-crate",
        kind: "crate",
        color: "#555555",
        position: { x: 5, y: 0.25, z: 5 },
        size: { x: 0.5, y: 0.5, z: 0.5 },
        portable: false
      }, {
        id: "moving-crate",
        kind: "crate",
        color: "#885533",
        position: { x: 2, y: 0.2, z: 2 },
        size: { x: 0.4, y: 0.4, z: 0.4 },
        portable: true
      }],
      zones: [],
      default_goal: {
        summary: "保持站立",
        predicates: [{
          type: "robot_at",
          target: { x: 1, y: 0, z: 1 },
          tolerance: 0.25
        }]
      }
    });
    const objectSnapshots = {
      "moving-crate": {
        id: "moving-crate",
        position: { x: 7, y: 0.2, z: 8 },
        rotation: { x: 0, y: 0, z: 0, w: 1 }
      }
    } satisfies Record<string, HumanoidObjectSnapshot>;

    expect(humanoidDynamicNavigationObstacles({
      scenario,
      objectSnapshots
    })).toEqual([{
      id: "fixed-object-fixed-crate",
      center: { x: 5, y: 0.25, z: 5 },
      halfExtents: { x: 0.25, y: 0.25, z: 0.25 },
      yaw: 0
    }, {
      id: "portable-object-moving-crate",
      center: { x: 7, y: 0.2, z: 8 },
      halfExtents: { x: 0.2, y: 0.2, z: 0.2 },
      yaw: 0
    }]);
  });

  it("removes only carried portable objects from navigation", () => {
    const scenario = ScenarioSchema.parse({
      title: "Carried obstacle exclusion",
      seed: 10,
      bounds: { width: 8, depth: 8 },
      visibility_radius: 4,
      robot: { x: 1, z: 1, yaw: 0 },
      obstacles: [],
      objects: [{
        id: "carried",
        kind: "crate",
        color: "#885533",
        position: { x: 2, y: 0.2, z: 2 },
        size: { x: 0.4, y: 0.4, z: 0.4 },
        portable: true
      }],
      zones: [],
      default_goal: {
        summary: "保持站立",
        predicates: [{
          type: "robot_at",
          target: { x: 1, y: 0, z: 1 },
          tolerance: 0.25
        }]
      }
    });

    expect(humanoidDynamicNavigationObstacles({
      scenario,
      objectSnapshots: {},
      excludedPortableObjectIds: new Set(["carried"])
    })).toEqual([]);
  });
});
