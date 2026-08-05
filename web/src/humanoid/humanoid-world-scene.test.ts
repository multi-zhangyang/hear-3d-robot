import * as THREE from "three";
import { describe, expect, it } from "vitest";
import type {
  HumanoidWorldSnapshot,
  ScenarioChunkDeltaState,
  ScenarioDefinition
} from "../types";
import type { G1Rig } from "./g1-rig";
import {
  HumanoidWorldScene,
  humanoidObjectInteractionState
} from "./humanoid-world-scene";

describe("HumanoidWorldScene chunk residency", () => {
  it("creates only nearby chunk meshes and releases them after migration", () => {
    const scene = new THREE.Scene();
    const world = createWorldScene(scene, streamingScenario());

    world.update(snapshotAt(2));
    const firstMovingMesh = world.root.getObjectByName("moving_object");
    expect(world.root.getObjectByName("left_wall")).toBeDefined();
    expect(world.root.getObjectByName("right_wall")).toBeUndefined();
    expect(firstMovingMesh).toBeDefined();

    world.update(snapshotAt(18));
    const migratedMesh = world.root.getObjectByName("moving_object");
    expect(world.root.getObjectByName("left_wall")).toBeUndefined();
    expect(world.root.getObjectByName("right_wall")).toBeDefined();
    expect(migratedMesh).toBeDefined();
    expect(migratedMesh).not.toBe(firstMovingMesh);
    expect(migratedMesh?.position.x).toBe(18);

    world.updateScenarioChunks(changedRightChunk());
    world.update(snapshotAt(18));
    expect(world.root.getObjectByName("right_wall")).toBeUndefined();
    expect(world.root.getObjectByName("placed_step")).toBeDefined();

    world.dispose();
  });

  it("derives object highlighting only from current physical grasp evidence", () => {
    const snapshot = snapshotAt(2);
    snapshot.frame = 9;
    snapshot.grasp = {
      contractSha256: "a".repeat(64),
      assessments: [{
        frame: 9,
        object_id: "moving_object",
        hand: "left",
        grasp_verified: false,
        evidence: { contact: { status: "opposed" } }
      }] as HumanoidWorldSnapshot["grasp"]["assessments"]
    };
    expect(humanoidObjectInteractionState(snapshot, "moving_object")).toBe("contact");
    snapshot.grasp.assessments[0]!.grasp_verified = true;
    expect(humanoidObjectInteractionState(snapshot, "moving_object")).toBe("verified");
    snapshot.grasp.assessments[0]!.frame = 8;
    expect(humanoidObjectInteractionState(snapshot, "moving_object")).toBe("idle");
  });
});

function createWorldScene(
  scene: THREE.Scene,
  scenario: ScenarioDefinition
): HumanoidWorldScene {
  const rig = {
    root: new THREE.Group(),
    update: () => undefined,
    bounds: () => new THREE.Box3(
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(1, 1, 1)
    )
  } as unknown as G1Rig;
  const Constructor = HumanoidWorldScene as unknown as new (
    target: THREE.Scene,
    definition: ScenarioDefinition,
    chunks: ScenarioChunkDeltaState,
    humanoidRig: G1Rig
  ) => HumanoidWorldScene;
  return new Constructor(scene, scenario, emptyChunks(scenario.seed), rig);
}

function emptyChunks(seed: number): ScenarioChunkDeltaState {
  return {
    version: 1,
    scenario_seed: seed,
    scenario_sha256: "a".repeat(64),
    manifest_version: 1,
    revision: 0,
    changed_chunk_ids: [],
    chunks: []
  };
}

function changedRightChunk(): ScenarioChunkDeltaState {
  return {
    ...emptyChunks(7),
    revision: 1,
    changed_chunk_ids: ["chunk_1_0"],
    chunks: [{
      chunk_id: "chunk_1_0",
      revision: 1,
      blocks: [{
        id: "placed_step",
        origin: "created",
        present: true,
        center: { x: 16, y: 0.25, z: 4 },
        size: { x: 1, y: 0.5, z: 1 },
        material: "stone",
        properties: {}
      }, {
        id: "right_wall",
        origin: "scenario",
        present: false,
        center: { x: 18, y: 0.5, z: 3 },
        size: { x: 1, y: 1, z: 1 },
        material: "solid",
        properties: {}
      }],
      zones: [],
      dynamic_entities: []
    }]
  };
}

function snapshotAt(x: number): HumanoidWorldSnapshot {
  return {
    robot: {
      rootPosition: { x, y: 0.8, z: 5 },
      objects: {
        moving_object: {
          id: "moving_object",
          position: { x, y: 0.2, z: 5 },
          rotation: { x: 0, y: 0, z: 0, w: 1 }
        }
      },
      contacts: [],
      feet: {
        left: { points: [] },
        right: { points: [] }
      },
      balance: { centerOfMass: { x, y: 0.8, z: 5 } },
      fallen: false
    },
    navigation: { waypoints: [] }
  } as unknown as HumanoidWorldSnapshot;
}

function streamingScenario(): ScenarioDefinition {
  return {
    title: "Streaming chunks",
    seed: 7,
    bounds: { width: 20, depth: 10 },
    visibility_radius: 1,
    robot: { x: 2, z: 5, yaw: 0 },
    obstacles: [
      {
        id: "left_wall",
        center: { x: 2, y: 0.5, z: 3 },
        size: { x: 1, y: 1, z: 1 }
      },
      {
        id: "right_wall",
        center: { x: 18, y: 0.5, z: 3 },
        size: { x: 1, y: 1, z: 1 }
      }
    ],
    objects: [{
      id: "moving_object",
      kind: "cube",
      color: "#55ddaa",
      position: { x: 2, y: 0.2, z: 5 },
      size: { x: 0.4, y: 0.4, z: 0.4 },
      portable: true
    }],
    zones: [],
    default_goal: {
      summary: "Move through both chunks",
      predicates: [{
        type: "robot_at",
        target: { x: 18, y: 0, z: 5 },
        tolerance: 0.5
      }]
    },
    chunk_manifest: {
      version: 1,
      chunk_size: 10,
      grid: { columns: 2, rows: 1 },
      chunks: [
        {
          id: "chunk_0_0",
          coordinate: { column: 0, row: 0 },
          bounds: {
            minimum: { x: 0, z: 0 },
            maximum: { x: 10, z: 10 }
          },
          entity_ids: {
            obstacles: ["left_wall"],
            objects: ["moving_object"],
            zones: []
          }
        },
        {
          id: "chunk_1_0",
          coordinate: { column: 1, row: 0 },
          bounds: {
            minimum: { x: 10, z: 0 },
            maximum: { x: 20, z: 10 }
          },
          entity_ids: {
            obstacles: ["right_wall"],
            objects: [],
            zones: []
          }
        }
      ]
    }
  };
}
