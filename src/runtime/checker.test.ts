import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { RapierWorld } from "../world/rapier-world.js";
import { checkGoal, unmetGoalRecovery } from "./checker.js";

describe("final-state checker", () => {
  it("requires an object footprint to fit in a zone and rest on its surface", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);
    const object = scenario.objects.find((candidate) => candidate.id === "red_block");
    if (!object) throw new Error("red_block is required");
    const world = await RapierWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      const goal = {
        summary: "The red block rests fully inside the target zone.",
        predicates: [{
          type: "object_in_zone" as const,
          object_id: object.id,
          zone_id: "test_zone",
          expected: true,
          tolerance: 0.01
        }]
      };
      const containingWorld = {
        ...snapshot,
        zones: [{
          id: "test_zone",
          color: "#3d9b68",
          center: { x: object.position.x, y: 0.01, z: object.position.z },
          size: { x: 0.8, y: 0.02, z: 0.8 }
        }]
      };
      expect(checkGoal(goal, containingWorld).success).toBe(true);

      const overhangingWorld = {
        ...containingWorld,
        zones: [{
          ...containingWorld.zones[0]!,
          size: { x: 0.3, y: 0.02, z: 0.3 }
        }]
      };
      expect(checkGoal(goal, overhangingWorld).success).toBe(false);

      const airborneWorld = {
        ...containingWorld,
        objects: containingWorld.objects.map((candidate) => candidate.id === object.id
          ? {
              ...candidate,
              position: { ...candidate.position, y: candidate.position.y + 0.5 }
            }
          : candidate)
      };
      expect(checkGoal(goal, airborneWorld).success).toBe(false);
    } finally {
      world.dispose();
    }
  });

  it("names the physical change each unmet predicate needs", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);
    const world = await RapierWorld.create(scenario);
    try {
      // The scenario's own opening state: the block sits where it starts, far
      // from the zone, held by nobody. Both predicates are open, and this is
      // exactly the state a coordinator re-checked three times in a live run.
      const result = checkGoal({
        summary: "The red block rests in the green zone and is not attached.",
        predicates: [
          {
            type: "object_in_zone" as const,
            object_id: "red_block",
            zone_id: "green_zone",
            expected: true,
            tolerance: 0.05
          },
          {
            type: "object_attached" as const,
            object_id: "red_block",
            expected: true
          }
        ]
      }, world.snapshot());
      expect(result.success).toBe(false);

      const recovery = unmetGoalRecovery(result);
      expect(recovery.unmet_predicates).toEqual(["1:object_in_zone", "2:object_attached"]);
      const text = String(recovery.recovery);
      expect(text).toContain("checking again returns this same answer");
      expect(text).toContain("away from its center");
      expect(text).toContain("close the gripper");
    } finally {
      world.dispose();
    }
  });

  it("says nothing when every predicate is satisfied", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);
    const world = await RapierWorld.create(scenario);
    try {
      const result = checkGoal({
        summary: "The red block is not attached to the robot.",
        predicates: [{ type: "object_attached" as const, object_id: "red_block", expected: false }]
      }, world.snapshot());
      expect(result.success).toBe(true);
      expect(unmetGoalRecovery(result)).toEqual({});
    } finally {
      world.dispose();
    }
  });

  it("measures generated-world exploration as a real final-state predicate", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("voxel_survey", 11);
    const world = await RapierWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      const goal = {
        summary: "Map at least five percent of the terrain.",
        predicates: [{ type: "terrain_explored" as const, minimum_fraction: 0.05 }]
      };
      const initial = checkGoal(goal, snapshot);
      expect(initial.checks[0]?.actual).toMatchObject({
        cells_seen: snapshot.explored.seen,
        cells_total: snapshot.explored.total,
        minimum_fraction: 0.05
      });

      const complete = checkGoal(goal, {
        ...snapshot,
        explored: { ...snapshot.explored, seen: Math.ceil(snapshot.explored.total * 0.05) }
      });
      expect(complete.success).toBe(true);

      const recovery = String(unmetGoalRecovery(initial).recovery);
      expect(recovery).toContain("Survey the current frontier");
      expect(recovery).toContain("5.0%");
    } finally {
      world.dispose();
    }
  });
});
