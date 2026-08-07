import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { humanoidEnvironment } from "./environment.js";
import { HumanoidSimulation } from "./simulation.js";

describe("MuJoCo articulated object world", () => {
  it("loads free-form shapes, an articulated door and a physical container", async () => {
    const scenario = ScenarioSchema.parse({
      title: "Interactive fixtures",
      seed: 4,
      bounds: { width: 10, depth: 10 },
      visibility_radius: 8,
      robot: { x: 1.5, z: 1.5, yaw: 0 },
      obstacles: [],
      objects: [
        {
          id: "cabinet-door",
          kind: "cabinet_door",
          color: "#806c52",
          position: { x: 3, y: 1, z: 3 },
          size: { x: 0.8, y: 1.2, z: 0.06 },
          portable: false,
          capability: {
            shape: "box",
            mass_kg: 3.5,
            affordances: [],
            interaction_points: [],
            articulation: {
              joint_id: "cabinet-door-hinge",
              type: "hinge",
              semantic: "cabinet_door",
              axis: { x: 0, y: 1, z: 0 },
              anchor_world: { x: 2.6, y: 1, z: 3 },
              range: { minimum: 0, maximum: 1.6 },
              initial_position: 0.25,
              closed_position: 0,
              open_position: 1.5,
              damping: 0.7,
              friction_loss: 0.1
            }
          }
        },
        {
          id: "receiving-bin",
          kind: "container",
          color: "#50785f",
          position: { x: 5, y: 0.45, z: 3 },
          size: { x: 0.9, y: 0.8, z: 0.9 },
          portable: false,
          capability: {
            shape: "box",
            mass_kg: 6,
            affordances: ["container"],
            interaction_points: [],
            container: {
              interior_center: { x: 0, y: 0, z: 0 },
              interior_size: { x: 0.72, y: 0.65, z: 0.72 },
              opening_direction: { x: 0, y: 1, z: 0 },
              wall_thickness_m: 0.04
            }
          }
        },
        {
          id: "round-workpiece",
          kind: "workpiece",
          color: "#c78848",
          position: { x: 4, y: 0.12, z: 2 },
          size: { x: 0.18, y: 0.24, z: 0.18 },
          portable: true,
          capability: {
            shape: "cylinder",
            mass_kg: 0.7,
            friction: { sliding: 0.6, torsional: 0.015, rolling: 0.002 },
            affordances: [],
            interaction_points: []
          }
        }
      ],
      zones: [],
      default_goal: {
        summary: "抵达目标",
        predicates: [{
          type: "robot_at",
          target: { x: 8, y: 0, z: 8 },
          tolerance: 0.3
        }]
      }
    });
    const simulation = await HumanoidSimulation.create(humanoidEnvironment(scenario));
    try {
      const snapshot = simulation.snapshot();
      expect(Object.keys(snapshot.objects).sort()).toEqual([
        "cabinet-door",
        "receiving-bin",
        "round-workpiece"
      ]);
      expect(snapshot.objects["cabinet-door"]?.articulation).toMatchObject({
        type: "hinge",
        position: 0.25,
        minimum: 0,
        maximum: 1.6
      });
      const doorPosition = snapshot.objects["cabinet-door"]!.position;
      expect(Math.hypot(
        doorPosition.x - 2.6,
        doorPosition.z - 3
      )).toBeCloseTo(0.4, 5);
      expect(doorPosition.x).not.toBeCloseTo(2.6, 5);
      expect(snapshot.objects["receiving-bin"]?.articulation).toBeUndefined();
      expect(snapshot.objects["round-workpiece"]?.position).toMatchObject({
        x: 4,
        z: 2
      });
    } finally {
      await simulation.dispose();
    }
  }, 30_000);
});
