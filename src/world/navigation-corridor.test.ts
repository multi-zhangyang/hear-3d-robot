import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../domain/schema.js";
import { rebuildScenarioChunkManifest } from "../domain/scenario-chunk.js";
import { HUMANOID_NAVIGATION_PROFILE } from "./humanoid/environment.js";
import { navigationCorridorBuildStages } from "./navigation-corridor.js";
import { NavigationMesh } from "./navigation.js";

const SCENARIO = ScenarioSchema.parse({
  title: "Navigation corridor field",
  seed: 7,
  bounds: { width: 64, depth: 64 },
  visibility_radius: 8,
  robot: { x: 1, z: 1, yaw: 0 },
  obstacles: [],
  objects: [],
  zones: [],
  default_goal: {
    summary: "Cross the field",
    predicates: [{
      type: "robot_at",
      target: { x: 63, y: 0, z: 63 },
      tolerance: 0.4
    }]
  }
});

describe("navigation corridor build stages", () => {
  it("builds a narrow connected tile band before a full-world final stage", () => {
    const stages = navigationCorridorBuildStages(
      SCENARIO,
      { x: 1, z: 1 },
      { x: 63, z: 63 }
    );

    expect(stages.length).toBeGreaterThan(1);
    expect(stages[0]).toMatchObject({
      expansionTiles: 1,
      fullWorld: false
    });
    expect(stages[0]!.selectedTileCount).toBeLessThan(
      stages[0]!.totalTileCount
    );
    expect(stages.at(-1)).toMatchObject({
      expansionTiles: null,
      fullWorld: true
    });
    for (const stage of stages) {
      expect(stage.scope.walkableRegions).toHaveLength(stage.selectedTileCount);
      expect(stage.scope.region.minimum.x).toBeGreaterThanOrEqual(0);
      expect(stage.scope.region.minimum.z).toBeGreaterThanOrEqual(0);
      expect(stage.scope.region.maximum.x).toBeLessThanOrEqual(64);
      expect(stage.scope.region.maximum.z).toBeLessThanOrEqual(64);
    }
  });

  it("plans across the first diagonal band without filling its bounding rectangle", async () => {
    const start = { x: 1, y: 0, z: 1 };
    const target = { x: 63, y: 0, z: 63 };
    const stage = navigationCorridorBuildStages(SCENARIO, start, target)[0]!;
    const navigation = await NavigationMesh.create(
      SCENARIO,
      stage.scope,
      HUMANOID_NAVIGATION_PROFILE
    );
    try {
      const plan = navigation.plan(start, target, [{
        id: "outside-corridor",
        center: { x: 60, y: 0.5, z: 2 },
        halfExtents: { x: 0.5, y: 0.5, z: 0.5 },
        yaw: 0
      }]);
      expect(plan.waypoints.length).toBeGreaterThanOrEqual(2);
      expect(plan.resolvedTarget.x).toBeCloseTo(target.x, 1);
      expect(plan.resolvedTarget.z).toBeCloseTo(target.z, 1);
    } finally {
      navigation.dispose();
    }
  }, 30_000);

  it("never appends an unreachable target to a partial Recast path", async () => {
    const scenario = ScenarioSchema.parse(rebuildScenarioChunkManifest({
      ...SCENARIO,
      bounds: { width: 10, depth: 10 },
      robot: { x: 1, z: 5, yaw: 0 },
      default_goal: {
        summary: "Cross the wall",
        predicates: [{
          type: "robot_at",
          target: { x: 9, y: 0, z: 5 },
          tolerance: 0.4
        }]
      }
    }));
    const start = { x: 1, y: 0, z: 5 };
    const target = { x: 9, y: 0, z: 5 };
    const stage = navigationCorridorBuildStages(scenario, start, target)[0]!;
    const navigation = await NavigationMesh.create(
      scenario,
      stage.scope,
      HUMANOID_NAVIGATION_PROFILE
    );
    try {
      expect(() => navigation.plan(start, target, [{
        id: "sealed-wall",
        center: { x: 5, y: 1, z: 5 },
        halfExtents: { x: 0.5, y: 1, z: 4.8 },
        yaw: 0
      }])).toThrow(/No complete navigation path/);
    } finally {
      navigation.dispose();
    }
  }, 30_000);
});
