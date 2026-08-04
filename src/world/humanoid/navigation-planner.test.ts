import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { HUMANOID_NAVIGATION_PROFILE } from "./environment.js";
import { HumanoidNavigationPlanner } from "./navigation-planner.js";

const OPEN_FIELD = ScenarioSchema.parse({
  title: "Scoped humanoid navigation",
  seed: 41,
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

describe("HumanoidNavigationPlanner", () => {
  it("builds lazily and reuses a successful route corridor", async () => {
    const planner = new HumanoidNavigationPlanner(
      OPEN_FIELD,
      HUMANOID_NAVIGATION_PROFILE
    );
    const start = { x: 1, y: 0, z: 1 };
    const target = { x: 63, y: 0, z: 63 };
    try {
      expect(planner.state()).toMatchObject({
        cachedScopeCount: 0,
        buildCount: 0,
        lastSelectedTileCount: null
      });

      const first = await planner.plan(start, target, []);
      const firstState = planner.state();
      expect(first.resolvedTarget.x).toBeCloseTo(target.x, 1);
      expect(first.resolvedTarget.z).toBeCloseTo(target.z, 1);
      expect(firstState.cachedScopeCount).toBe(1);
      expect(firstState.buildCount).toBe(1);
      expect(firstState.lastExpansionTiles).toBe(1);
      expect(firstState.lastSelectedTileCount).toBeLessThan(
        firstState.lastTotalTileCount!
      );

      await planner.plan(start, target, []);
      expect(planner.state()).toMatchObject({
        cachedScopeCount: 1,
        buildCount: 1
      });
    } finally {
      await planner.dispose();
    }

    await expect(planner.plan(start, target, [])).rejects.toThrow(/disposed/i);
  }, 30_000);

  it("widens a blocked local corridor before considering the full world", async () => {
    const scenario = ScenarioSchema.parse({
      ...OPEN_FIELD,
      robot: { x: 1, z: 32, yaw: 0 }
    });
    const planner = new HumanoidNavigationPlanner(
      scenario,
      HUMANOID_NAVIGATION_PROFILE
    );
    try {
      const result = await planner.plan(
        { x: 1, y: 0, z: 32 },
        { x: 63, y: 0, z: 32 },
        [29.9, 31.8, 33.7, 35.6, 37.3].map((z, index) => ({
          id: `local-cross-wall-${String(index)}`,
          center: { x: 32, y: 1, z },
          halfExtents: { x: 0.5, y: 1, z: 0.6 },
          yaw: 0
        }))
      );
      const state = planner.state();
      expect(result.distance).toBeGreaterThan(62);
      expect(state.buildCount).toBeGreaterThan(1);
      expect(state.lastExpansionTiles).not.toBe(1);
      expect(state.lastSelectedTileCount).toBeLessThan(state.lastTotalTileCount!);
      expect(state.cachedScopeCount).toBeLessThanOrEqual(2);
    } finally {
      await planner.dispose();
    }
  }, 30_000);
});
