import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { humanoidObjectCapability } from "./object-capability.js";

describe("humanoid object capabilities", () => {
  it("derives shape-aware grasp points without object identity rules", () => {
    const scenario = scenarioWith({
      id: "any-cylinder",
      kind: "tool",
      color: "#999999",
      position: { x: 2, y: 0.5, z: 2 },
      size: { x: 0.12, y: 0.6, z: 0.12 },
      portable: true,
      capability: {
        shape: "cylinder",
        mass_kg: 1.4,
        friction: { sliding: 0.55, torsional: 0.02, rolling: 0.004 },
        affordances: [],
        interaction_points: []
      }
    });
    const capability = humanoidObjectCapability(scenario.objects[0]!);

    expect(capability).toMatchObject({
      shape: "cylinder",
      massKg: 1.4,
      mobility: "free",
      friction: { sliding: 0.55, torsional: 0.02, rolling: 0.004 }
    });
    expect(capability.affordances).toEqual(expect.arrayContaining([
      "graspable",
      "movable",
      "pushable",
      "pullable"
    ]));
    expect(capability.interactionPoints.filter(({ kind }) => kind === "grasp"))
      .toHaveLength(5);
    expect(capability.interactionPoints.every(({ source }) => source === "geometry"))
      .toBe(true);
  });

  it("derives articulated affordances from joint semantics", () => {
    const scenario = scenarioWith({
      id: "panel",
      kind: "panel",
      color: "#8a765e",
      position: { x: 2, y: 1, z: 2 },
      size: { x: 0.8, y: 1.2, z: 0.06 },
      portable: false,
      capability: {
        shape: "box",
        mass_kg: 4,
        affordances: [],
        interaction_points: [{
          id: "handle",
          kind: "pull",
          local_position: { x: 0.3, y: 0, z: 0.06 },
          approach_direction: { x: 0, y: 0, z: -1 },
          compatible_hands: "either",
          clearance_m: 0.08
        }],
        articulation: {
          joint_id: "panel-hinge",
          type: "hinge",
          semantic: "cabinet_door",
          axis: { x: 0, y: 1, z: 0 },
          anchor_world: { x: 1.6, y: 1, z: 2 },
          range: { minimum: 0, maximum: 1.6 },
          initial_position: 0,
          closed_position: 0,
          open_position: 1.4,
          damping: 0.4,
          friction_loss: 0.08
        }
      }
    });
    const capability = humanoidObjectCapability(scenario.objects[0]!);

    expect(capability.mobility).toBe("articulated");
    expect(capability.affordances).toEqual(expect.arrayContaining([
      "openable",
      "closeable",
      "pullable",
      "pushable"
    ]));
    expect(capability.interactionPoints).toEqual([
      expect.objectContaining({ id: "handle", kind: "pull", source: "authored" })
    ]);
  });
});

function scenarioWith(object: Record<string, unknown>) {
  return ScenarioSchema.parse({
    title: "Object capability field",
    seed: 1,
    bounds: { width: 8, depth: 8 },
    visibility_radius: 6,
    robot: { x: 1, z: 1, yaw: 0 },
    obstacles: [],
    objects: [object],
    zones: [],
    default_goal: {
      summary: "抵达目标",
      predicates: [{
        type: "robot_at",
        target: { x: 6, y: 0, z: 6 },
        tolerance: 0.3
      }]
    }
  });
}
