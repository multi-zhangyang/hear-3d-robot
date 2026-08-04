import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import {
  applyScenarioChunkDeltaMutation
} from "../../domain/scenario-chunk-delta.js";
import {
  createScenarioChunkDeltaState
} from "../../domain/scenario-chunk-delta-schema.js";
import { analyzeHumanoidScenarioSynchronization } from "./scenario-synchronization.js";

const scenario = ScenarioSchema.parse({
  title: "Scenario synchronization",
  seed: 19,
  bounds: { width: 12, depth: 12 },
  visibility_radius: 6,
  robot: { x: 1.5, z: 1.5, yaw: 0 },
  obstacles: [],
  objects: [{
    id: "crate",
    kind: "crate",
    color: "#8b6b45",
    position: { x: 2, y: 0.2, z: 2 },
    size: { x: 0.4, y: 0.4, z: 0.4 },
    portable: true
  }],
  zones: [{
    id: "beacon",
    color: "#55b38b",
    center: { x: 9, y: 0.01, z: 9 },
    size: { x: 1.5, y: 0.02, z: 1.5 }
  }],
  default_goal: {
    summary: "Reach the beacon",
    predicates: [{ type: "robot_in_zone", zone_id: "beacon", tolerance: 0.2 }]
  }
});

describe("humanoid scenario synchronization", () => {
  it("ignores portable poses already owned by MuJoCo", () => {
    const state = applyScenarioChunkDeltaMutation(
      scenario,
      createScenarioChunkDeltaState(scenario),
      {
        type: "put_dynamic_entity",
        entity: {
          id: "crate",
          kind: "crate",
          color: "#8b6b45",
          position: { x: 4, y: 0.2, z: 5 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
          linear_velocity: { x: 0, y: 0, z: 0 },
          angular_velocity: { x: 0, y: 0, z: 0 },
          size: { x: 0.4, y: 0.4, z: 0.4 },
          portable: true,
          properties: {}
        }
      }
    );
    expect(analyzeHumanoidScenarioSynchronization({
      current: scenario,
      baseline: scenario,
      chunks: state
    })).toMatchObject({
      changed: false,
      requiresResourceRebuild: false,
      changedDomains: []
    });
  });

  it("requires a resource rebuild for a created collision block", () => {
    const state = applyScenarioChunkDeltaMutation(
      scenario,
      createScenarioChunkDeltaState(scenario),
      {
        type: "create_block",
        block: {
          id: "wall",
          center: { x: 6, y: 1, z: 6 },
          size: { x: 1, y: 2, z: 4 },
          material: "stone",
          properties: {}
        }
      }
    );
    const result = analyzeHumanoidScenarioSynchronization({
      current: scenario,
      baseline: scenario,
      chunks: state
    });
    expect(result).toMatchObject({
      changed: true,
      requiresResourceRebuild: true,
      changedDomains: ["geometry"]
    });
    expect(result.scenario.obstacles.map(({ id }) => id)).toEqual(["wall"]);
  });

  it("rejects portable topology changes without state migration", () => {
    const state = applyScenarioChunkDeltaMutation(
      scenario,
      createScenarioChunkDeltaState(scenario),
      { type: "remove_dynamic_entity", entity_id: "crate" }
    );
    expect(() => analyzeHumanoidScenarioSynchronization({
      current: scenario,
      baseline: scenario,
      chunks: state
    })).toThrow(/checkpoint migration/i);
  });
});
