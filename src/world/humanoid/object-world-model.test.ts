import { describe, expect, it } from "vitest";
import { ScenarioSchema, type Quaternion, type Vec3 } from "../../domain/schema.js";
import { createHumanoidObjectWorldModel } from "./object-world-model.js";
import type { HumanoidObjectToken } from "./object-memory.js";
import { createHumanoidSkillCatalog } from "./skill-catalog.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

describe("object-centered humanoid world model", () => {
  it("publishes affordances, interaction points, joint state and container relations", () => {
    const scenario = ScenarioSchema.parse({
      title: "Object-centered work cell",
      seed: 8,
      bounds: { width: 10, depth: 10 },
      visibility_radius: 8,
      robot: { x: 2, z: 2, yaw: 0 },
      obstacles: [],
      objects: [
        {
          id: "parcel",
          kind: "workpiece",
          color: "#c88646",
          position: { x: 4, y: 0.45, z: 4 },
          size: { x: 0.2, y: 0.2, z: 0.2 },
          portable: true,
          capability: {
            shape: "sphere",
            mass_kg: 0.6,
            affordances: [],
            interaction_points: []
          }
        },
        {
          id: "bin",
          kind: "container",
          color: "#557b64",
          position: { x: 4, y: 0.45, z: 4 },
          size: { x: 1, y: 0.8, z: 1 },
          portable: false,
          capability: {
            shape: "box",
            mass_kg: 5,
            affordances: ["container"],
            interaction_points: [],
            container: {
              interior_center: { x: 0, y: 0, z: 0 },
              interior_size: { x: 0.8, y: 0.6, z: 0.8 },
              opening_direction: { x: 0, y: 1, z: 0 },
              wall_thickness_m: 0.04
            }
          }
        },
        {
          id: "door",
          kind: "cabinet_door",
          color: "#7b674f",
          position: { x: 6, y: 1, z: 4 },
          size: { x: 0.8, y: 1.2, z: 0.06 },
          portable: false,
          capability: {
            shape: "box",
            mass_kg: 3,
            affordances: [],
            interaction_points: [{
              id: "handle",
              kind: "pull",
              local_position: { x: 0.3, y: 0, z: 0.05 },
              compatible_hands: "either",
              clearance_m: 0.08
            }],
            articulation: {
              joint_id: "door-hinge",
              parent_object_id: "bin",
              type: "hinge",
              semantic: "cabinet_door",
              axis: { x: 0, y: 1, z: 0 },
              anchor_world: { x: 5.6, y: 1, z: 4 },
              range: { minimum: 0, maximum: 1.2 },
              initial_position: 0,
              closed_position: 0,
              open_position: 1,
              damping: 0.5,
              friction_loss: 0.05
            }
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
    const objects = {
      parcel: objectSnapshot("parcel", { x: 4, y: 0.45, z: 4 }),
      bin: objectSnapshot("bin", { x: 4, y: 0.45, z: 4 }),
      door: {
        ...objectSnapshot("door", { x: 6, y: 1, z: 4 }),
        articulation: {
          type: "hinge" as const,
          position: 0.95,
          velocity: 0,
          minimum: 0,
          maximum: 1.2,
          normalized: 0.95 / 1.2
        }
      }
    };
    const world = createHumanoidObjectWorldModel({
      frame: 20,
      worldRevision: 20,
      scenario,
      robot: { objects } as unknown as HumanoidSimulationSnapshot,
      objectTokens: scenario.objects.map((descriptor) => token(
        descriptor.id,
        objects[descriptor.id as keyof typeof objects]!.position,
        descriptor.portable
      ))
    });

    const parcel = world.objects.find(({ id }) => id === "parcel")!;
    const bin = world.objects.find(({ id }) => id === "bin")!;
    const door = world.objects.find(({ id }) => id === "door")!;
    expect(parcel).toMatchObject({
      shape: "sphere",
      authority: "sensor_observation",
      physical: { mass_kg: null, mobility: "free" },
      belief: {
        size: { confidence: 0.82, source: "visual_geometry" },
        mass_kg: { estimate: null, confidence: 0, source: "unobserved" }
      },
      relations: { contained_by: ["bin"] }
    });
    expect(parcel.affordances).toEqual(expect.arrayContaining(["graspable", "movable"]));
    expect(parcel.interaction_points.length).toBeGreaterThanOrEqual(4);
    expect(bin.relations.contains).toEqual(["parcel"]);
    expect(door).toMatchObject({
      articulation: { joint_id: "door-hinge", state: "open", open_fraction: 0.95 },
      relations: { connected_to: ["bin"] }
    });

    const catalog = createHumanoidSkillCatalog(world, [], [
      "balance",
      "locomotion",
      "joint_reference_tracking"
    ], ["assembly-zone"]);
    expect(catalog.entries.find(({ id }) => id === "navigate_to_zone")).toMatchObject({
      available: true,
      observable_zone_ids: ["assembly-zone"],
      learned_policy_ready: true,
      learned_policy_missing_capabilities: []
    });
    expect(catalog.entries.find(({ id }) => id === "grasp")).toMatchObject({
      available: true,
      observable_target_ids: ["parcel"],
      recovery_entry: ["regrasp", "reach"],
      learned_policy_ready: false,
      learned_policy_missing_capabilities: ["contact_rich_manipulation"]
    });
    expect(catalog.entries.find(({ id }) => id === "open")).toMatchObject({
      available: true,
      observable_target_ids: ["door"],
      learned_policy_ready: false,
      learned_policy_missing_capabilities: ["contact_rich_manipulation"]
    });
    expect(catalog.entries.find(({ id }) => id === "place")).toMatchObject({
      destination_ids: ["bin"]
    });
    expect(catalog.entries.find(({ id }) => id === "explore")).toMatchObject({
      available: true,
      observable_target_ids: [],
      learned_policy_ready: true,
      learned_policy_missing_capabilities: []
    });
  });
});

function token(id: string, position: Vec3, portable: boolean): HumanoidObjectToken {
  const rotation: Quaternion = { x: 0, y: 0, z: 0, w: 1 };
  return {
    id,
    role: portable ? "manipulable" : "fixture",
    kind: id,
    color: "#777777",
    size: { x: 0.2, y: 0.2, z: 0.2 },
    portable,
    status: "visible",
    state: "active",
    authority: "mujoco_exact",
    exact: true,
    observable: true,
    pose: { position: { ...position }, rotation },
    observedFrame: 20,
    observedWorldRevision: 20,
    position: { ...position },
    rotation,
    linearVelocity: { x: 0, y: 0, z: 0 },
    angularVelocity: { x: 0, y: 0, z: 0 },
    firstSeenRevision: 20,
    lastSeenRevision: 20,
    lastSeenFrame: 20,
    observationCount: 1,
    ageRevisions: 0,
    relation: {
      distanceToRobot: 1,
      bearingRadians: 0,
      verticalOffset: 0,
      distanceToLeftWrist: 1,
      distanceToRightWrist: 1
    },
    currentContacts: []
  };
}

function objectSnapshot(id: string, position: Vec3) {
  return {
    id,
    position,
    rotation: { x: 0, y: 0, z: 0, w: 1 },
    linearVelocity: { x: 0, y: 0, z: 0 },
    angularVelocity: { x: 0, y: 0, z: 0 }
  };
}
