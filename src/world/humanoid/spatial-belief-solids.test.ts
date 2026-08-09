import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";
import { humanoidSpatialBeliefSolids } from "./spatial-belief-solids.js";

const scenario = ScenarioSchema.parse({
  title: "空间占据投影",
  seed: 41,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 6,
  robot: { x: 1, z: 1, yaw: 0 },
  obstacles: [{
    id: "wall",
    center: { x: 4, y: 1, z: 4 },
    size: { x: 1, y: 2, z: 1 }
  }],
  objects: [{
    id: "carried",
    kind: "crate",
    color: "#aa7744",
    position: { x: 1.5, y: 1.1, z: 1.4 },
    size: { x: 0.4, y: 0.5, z: 0.6 },
    portable: true
  }, {
    id: "contacted",
    kind: "crate",
    color: "#6688aa",
    position: { x: 1.4, y: 0.3, z: 1.8 },
    size: { x: 0.5, y: 0.6, z: 0.5 },
    portable: true
  }, {
    id: "door",
    kind: "door",
    color: "#887766",
    position: { x: 3, y: 1, z: 3 },
    size: { x: 1.2, y: 2, z: 0.2 },
    portable: false
  }],
  zones: [],
  default_goal: {
    summary: "移动",
    predicates: [{
      type: "robot_at",
      target: { x: 2, y: 0, z: 2 },
      tolerance: 0.3
    }]
  }
});

describe("humanoidSpatialBeliefSolids", () => {
  it("uses current object poses, tactile objects and carried-object semantics", () => {
    const projected = humanoidSpatialBeliefSolids({
      scenario,
      robot: robotSnapshot(),
      visibleObjectIds: ["door", "carried"],
      solidTokens: [{
        id: "wall",
        sourceId: "wall",
        kind: "block",
        center: { x: 4, y: 1, z: 4 },
        size: { x: 1, y: 2, z: 1 },
        currentContacts: []
      }, {
        id: "fixed_object:door",
        sourceId: "door",
        kind: "fixed_object",
        center: { x: 3, y: 1, z: 3 },
        size: { x: 1.2, y: 2, z: 0.2 },
        currentContacts: []
      }],
      carriedObjectIds: new Set(["carried"])
    });

    expect(projected).toHaveLength(4);
    expect(projected[0]).toEqual({
      center: { x: 4, y: 1, z: 4 },
      size: { x: 1, y: 2, z: 1 }
    });
    expect(projected[1]).toMatchObject({
      center: { x: 1.6, y: 1.15, z: 1.45 },
      occupiesNavigationSpace: false
    });
    expect(projected[2]).toMatchObject({
      center: { x: 1.45, y: 0.3, z: 1.85 },
      occupiesNavigationSpace: true
    });
    expect(projected[3]).toMatchObject({
      center: { x: 3.2, y: 1, z: 3.1 },
      occupiesNavigationSpace: true
    });
    expect(projected[3]!.size.x).toBeCloseTo(0.2, 12);
    expect(projected[3]!.size.y).toBeCloseTo(2, 12);
    expect(projected[3]!.size.z).toBeCloseTo(1.2, 12);
  });

  it("rejects a sensed object without an authoritative physical descriptor", () => {
    expect(() => humanoidSpatialBeliefSolids({
      scenario,
      robot: robotSnapshot(),
      visibleObjectIds: ["unknown"],
      solidTokens: [],
      carriedObjectIds: new Set()
    })).toThrow("Observed object has no spatial descriptor: unknown");
  });
});

function robotSnapshot(): HumanoidSimulationSnapshot {
  const rotation = {
    x: 0,
    y: Math.sin(Math.PI / 4),
    z: 0,
    w: Math.cos(Math.PI / 4)
  };
  return {
    objects: {
      carried: {
        position: { x: 1.6, y: 1.15, z: 1.45 },
        rotation: { x: 0, y: 0, z: 0, w: 1 }
      },
      contacted: {
        position: { x: 1.45, y: 0.3, z: 1.85 },
        rotation: { x: 0, y: 0, z: 0, w: 1 }
      },
      door: {
        position: { x: 3.2, y: 1, z: 3.1 },
        rotation
      }
    },
    contacts: [{ firstObject: "contacted", secondObject: null }]
  } as unknown as HumanoidSimulationSnapshot;
}
