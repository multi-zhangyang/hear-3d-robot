import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { GoalValidationError, assertGoalSupported } from "./goal-validation.js";

describe("goal validation", () => {
  it("accepts supported final-state predicates", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    expect(() => assertGoalSupported({
      summary: "Supported final state",
      predicates: [
        { type: "robot_at", target: { x: 1, y: 0, z: 1 }, tolerance: 0.25 },
        { type: "object_in_zone", object_id: "red_block", zone_id: "green_zone", expected: true, tolerance: 0.05 },
        { type: "object_at", object_id: "blue_block", target: { x: 1, y: 0.25, z: 5 }, tolerance: 0.1 },
        { type: "object_property", object_id: "red_block", property: "enabled", expected: true },
        { type: "object_attached", object_id: "red_block", expected: false }
      ]
    }, scenario)).not.toThrow();
  });

  it("rejects coordinates and entity references outside the selected world", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    expect(() => assertGoalSupported({
      summary: "Outside bounds",
      predicates: [
        { type: "robot_at", target: { x: 100, y: 0, z: 100 }, tolerance: 0.25 }
      ]
    }, scenario)).toThrow(GoalValidationError);

    expect(() => assertGoalSupported({
      summary: "Unknown object",
      predicates: [
        { type: "object_attached", object_id: "missing_object", expected: false }
      ]
    }, scenario)).toThrow("Unknown object: missing_object");

    expect(() => assertGoalSupported({
      summary: "Unknown zone",
      predicates: [
        { type: "object_in_zone", object_id: "red_block", zone_id: "missing_zone", expected: true, tolerance: 0 }
      ]
    }, scenario)).toThrow("Unknown zone: missing_zone");
  });

  it("does not allow attachment goals for non-portable objects", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("locked_container", 0);

    expect(() => assertGoalSupported({
      summary: "Container attached",
      predicates: [
        { type: "object_attached", object_id: "locked_box", expected: true }
      ]
    }, scenario)).toThrow("Object is not portable: locked_box");
  });

  it("allows exploration goals only in voxel terrain", async () => {
    const catalog = await loadRuntimeCatalog();
    const voxel = catalog.materialize("voxel_survey", 5);
    const authored = catalog.materialize("open_navigation", 5);
    const goal = {
      summary: "Explore terrain",
      predicates: [{ type: "terrain_explored" as const, minimum_fraction: 0.05 }]
    };

    expect(() => assertGoalSupported(goal, voxel)).not.toThrow();
    expect(() => assertGoalSupported(goal, authored))
      .toThrow("terrain_explored requires a generated voxel world");
  });
});
