import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { ROBOT_LINK_IDS } from "../robot-rig";
import { disposeObject } from "../three-kit";
import type { ScenarioDefinition, WorldSnapshot } from "../types";
import { WorldScene } from "./WorldScene";

const WORLD_EDGE = 384;
const WORLD_CELL_COUNT = WORLD_EDGE * WORLD_EDGE;
const TERRAIN_BLOCK_HEIGHT = 0.5;
const HIGHEST_TERRAIN_LEVELS = 23;

describe("WorldScene maximum world bounds", () => {
  it("constructs a 384 by 384 world and scans long entity arrays without argument overflow", () => {
    const scene = new THREE.Scene();
    const scenario = maximumWorldScenario();
    const world = new WorldScene(scene, scenario, () => undefined);
    let disposed = false;

    try {
      const snapshot = worldSnapshot();
      expect(world.worldBounds(snapshot).max.y).toBe(
        HIGHEST_TERRAIN_LEVELS * TERRAIN_BLOCK_HEIGHT
      );

      const ordinaryObstacle: WorldSnapshot["obstacles"][number] = {
        id: "ordinary-obstacle",
        center: { x: 1, y: 1, z: 1 },
        size: { x: 1, y: 2, z: 1 }
      };
      const obstacles = new Array<WorldSnapshot["obstacles"][number]>(WORLD_CELL_COUNT)
        .fill(ordinaryObstacle);
      obstacles[WORLD_CELL_COUNT - 2] = {
        ...ordinaryObstacle,
        id: "invalid-obstacle",
        center: { x: 1, y: Number.POSITIVE_INFINITY, z: 1 }
      };
      obstacles[WORLD_CELL_COUNT - 1] = {
        ...ordinaryObstacle,
        id: "highest-obstacle",
        center: { x: 1, y: 28, z: 1 },
        size: { x: 1, y: 6, z: 1 }
      };
      expect(world.worldBounds({ ...snapshot, obstacles }).max.y).toBe(31);

      const ordinaryObject: WorldSnapshot["objects"][number] = {
        id: "ordinary-object",
        kind: "block",
        color: "#ffffff",
        position: { x: 1, y: 1, z: 1 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        linear_velocity: { x: 0, y: 0, z: 0 },
        angular_velocity: { x: 0, y: 0, z: 0 },
        size: { x: 1, y: 2, z: 1 },
        portable: true,
        locked: false,
        container_id: null,
        enabled: true,
        visible: true
      };
      const objects = new Array<WorldSnapshot["objects"][number]>(WORLD_CELL_COUNT)
        .fill(ordinaryObject);
      objects[WORLD_CELL_COUNT - 3] = {
        ...ordinaryObject,
        id: "invalid-object",
        position: { x: 1, y: Number.NaN, z: 1 }
      };
      objects[WORLD_CELL_COUNT - 2] = {
        ...ordinaryObject,
        id: "disabled-object",
        position: { x: 1, y: 900, z: 1 },
        enabled: false
      };
      objects[WORLD_CELL_COUNT - 1] = {
        ...ordinaryObject,
        id: "highest-object",
        position: { x: 1, y: 42, z: 1 },
        size: { x: 1, y: 8, z: 1 }
      };
      expect(world.worldBounds({ ...snapshot, objects }).max.y).toBe(46);

      world.update(snapshot);
      const resources = sceneResources(scene);
      expect(resources.instancedMeshes.length).toBeGreaterThan(0);
      expect(resources.geometries.size).toBeGreaterThan(0);
      expect(resources.materials.size).toBeGreaterThan(0);
      const instancedDisposals = resources.instancedMeshes.map((mesh) =>
        vi.spyOn(mesh, "dispose"));
      const geometryDisposals = [...resources.geometries].map((geometry) =>
        vi.spyOn(geometry, "dispose"));
      const materialDisposals = [...resources.materials].map((material) =>
        vi.spyOn(material, "dispose"));

      disposeObject(scene);
      disposed = true;

      for (const dispose of instancedDisposals) expect(dispose).toHaveBeenCalledOnce();
      for (const dispose of geometryDisposals) expect(dispose).toHaveBeenCalledOnce();
      for (const dispose of materialDisposals) expect(dispose).toHaveBeenCalledOnce();
    } finally {
      if (!disposed) disposeScene(scene);
    }
  });
});

function maximumWorldScenario(): ScenarioDefinition {
  const heights = new Array<number>(WORLD_CELL_COUNT).fill(2);
  heights[WORLD_CELL_COUNT - 3] = Number.NaN;
  heights[WORLD_CELL_COUNT - 2] = Number.POSITIVE_INFINITY;
  heights[WORLD_CELL_COUNT - 1] = HIGHEST_TERRAIN_LEVELS;
  return {
    title: "Maximum world",
    seed: 17,
    motion_seed: 31,
    bounds: { width: WORLD_EDGE, depth: WORLD_EDGE },
    terrain: {
      cell: 1,
      columns: WORLD_EDGE,
      rows: WORLD_EDGE,
      block: TERRAIN_BLOCK_HEIGHT,
      chunk_size: 16,
      maximum_height: 32,
      heights
    },
    visibility_radius: 24,
    robot: { x: 1, z: 1, yaw: 0 },
    obstacles: [],
    objects: [],
    zones: [],
    affordances: [],
    default_goal: { summary: "Explore", predicates: [] }
  };
}

function worldSnapshot(): WorldSnapshot {
  const still = { x: 0, y: 0, z: 0 };
  const rotation = { x: 0, y: 0, z: 0, w: 1 };
  const links = Object.fromEntries(ROBOT_LINK_IDS.map((id) => [id, {
    position: { x: 1, y: 1, z: 1 },
    rotation: { ...rotation },
    linear_velocity: { ...still },
    angular_velocity: { ...still }
  }])) as WorldSnapshot["robot"]["links"];
  return {
    frame: 1,
    simulated_time: 0,
    world_revision: 0,
    robot: {
      position: { x: 1, y: 1, z: 1 },
      yaw: 0,
      joints: {
        head_yaw: 0,
        head_pitch: 0,
        shoulder: 0,
        elbow: 0,
        wrist: 0,
        gripper_aperture: 0.1
      },
      contacts: {
        left_object_id: "contact-probe",
        right_object_id: null,
        left_force: 1,
        right_force: 0
      },
      attachment: null,
      odometry: {
        left_wheel: { position: 0, velocity: 0 },
        right_wheel: { position: 0, velocity: 0 }
      },
      joint_status: {},
      links,
      gripper: {
        aperture: 0.1,
        target_aperture: 0.1,
        maximum_force: 10,
        left_contact_object_id: "contact-probe",
        right_contact_object_id: null,
        left_contact_force: 1,
        right_contact_force: 0
      }
    },
    objects: [],
    zones: [],
    obstacles: [],
    explored: { cells: "", seen: 0, total: WORLD_CELL_COUNT },
    voxels: {
      version: 1,
      revision: 0,
      chunk_size: 16,
      load_radius_chunks: 1,
      loaded_chunks: [{ column: 0, row: 0 }],
      mutations: [],
      inventory: { grass: 0, dirt: 0, stone: 0, sand: 0, placed: 0 }
    },
    active_command: null,
    last_command: null,
    navigation: {
      plan_id: "plan-1",
      status: "planned",
      target: { x: 4, y: 0, z: 4 },
      face: { x: 5, y: 0, z: 4 },
      waypoints: [{ x: 1, y: 0, z: 1 }, { x: 4, y: 0, z: 4 }],
      waypoint_index: 0,
      distance: 4.24,
      planned_at_frame: 1,
      actual_path: [{ x: 1, y: 0, z: 1 }, { x: 1.1, y: 0, z: 1.1 }]
    },
    plans: { base: [], arm: [] },
    affordance_events: []
  };
}

function sceneResources(scene: THREE.Scene): {
  instancedMeshes: THREE.InstancedMesh[];
  geometries: Set<THREE.BufferGeometry>;
  materials: Set<THREE.Material>;
} {
  const instancedMeshes: THREE.InstancedMesh[] = [];
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  scene.traverse((object) => {
    if (object instanceof THREE.InstancedMesh) instancedMeshes.push(object);
    if (!(object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.Sprite)) {
      return;
    }
    if (!(object instanceof THREE.Sprite)) geometries.add(object.geometry);
    const material = object.material;
    if (Array.isArray(material)) material.forEach((entry) => materials.add(entry));
    else materials.add(material);
  });
  return { instancedMeshes, geometries, materials };
}

function disposeScene(scene: THREE.Scene): void {
  disposeObject(scene);
}
