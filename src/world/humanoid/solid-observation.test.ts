import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { visibleHumanoidSolidTokens } from "./solid-observation.js";

const scenario = ScenarioSchema.parse({
  title: "Solid observation",
  seed: 1,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 5,
  robot: { x: 1, z: 1, yaw: 0 },
  obstacles: [{
    id: "stone",
    center: { x: 2, y: 0.5, z: 2 },
    size: { x: 1, y: 1, z: 1 }
  }],
  objects: [],
  zones: [],
  default_goal: {
    summary: "Stay",
    predicates: [{
      type: "robot_at",
      target: { x: 1, y: 0, z: 1 },
      tolerance: 0.2
    }]
  }
});

describe("humanoid solid observation", () => {
  it("preserves the exact physical solid contact identity", () => {
    const contact = {
      position: { x: 1.5, y: 0.7, z: 2 },
      normal: { x: 1, y: 0, z: 0 },
      normalForce: 8,
      firstBody: "left_wrist_yaw_link" as const,
      secondBody: null,
      firstObject: null,
      secondObject: null,
      firstSolid: null,
      secondSolid: "stone",
      firstHandLink: null,
      secondHandLink: null
    };
    expect(visibleHumanoidSolidTokens({
      scenario,
      sensed: {
        sensor: {
          position: { x: 1, y: 1.5, z: 1 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
          maximumRange: 5,
          horizontalFieldOfView: 1,
          verticalFieldOfView: 1
        },
        solids: {}
      },
      contacts: [contact]
    })).toEqual([{
      id: "stone",
      sourceId: "stone",
      kind: "block",
      center: { x: 2, y: 0.5, z: 2 },
      size: { x: 1, y: 1, z: 1 },
      currentContacts: [contact]
    }]);
  });
});
