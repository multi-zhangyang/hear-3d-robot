import RAPIER from "@dimforge/rapier3d-compat";
import { beforeAll, describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import type { BodyChannel, Scenario, VoxelCoordinate } from "../domain/schema.js";
import { executeSkill, executeTool } from "../runtime/actions.js";
import { checkGoal } from "../runtime/checker.js";
import { RapierWorld, type SourceCommand } from "./rapier-world.js";
import { ROBOT_SPEC } from "./robot-model.js";
import {
  VoxelInteraction,
  VOXEL_INTERACTION_CLEARANCE,
  VOXEL_INTERACTION_DISTANCE
} from "./voxel-interaction.js";
import { VoxelStore } from "./voxel-store.js";

const TARGET: VoxelCoordinate = { column: 8, level: 0, row: 9 };
const PLACEMENT_TRAP: VoxelCoordinate = { column: 8, level: 1, row: 9 };

beforeAll(async () => {
  await RAPIER.init();
});

function source(id: string, skill: string, channels: BodyChannel[]): SourceCommand {
  return {
    id,
    agentId: "voxel_leaf",
    agentName: "Voxel manipulation leaf",
    skill,
    channels
  };
}

function detail(value: unknown): Record<string, unknown> {
  expect(value).not.toBeNull();
  expect(typeof value).toBe("object");
  expect(Array.isArray(value)).toBe(false);
  return value as Record<string, unknown>;
}

async function interactionScenario(): Promise<Scenario> {
  const catalog = await loadRuntimeCatalog();
  const scenario = catalog.materialize("voxel_survey", 11);
  const size = 32;
  const heights = new Array<number>(size * size).fill(0);
  for (let index = 0; index < size; index += 1) {
    heights[index] = 3;
    heights[(size - 1) * size + index] = 3;
    heights[index * size] = 3;
    heights[index * size + size - 1] = 3;
  }
  heights[TARGET.row * size + TARGET.column] = 1;
  scenario.bounds = { width: size, depth: size };
  scenario.terrain = {
    cell: 1,
    block: 0.9,
    columns: size,
    rows: size,
    chunk_size: 16,
    maximum_height: 12,
    heights
  };
  scenario.robot = {
    x: 8.5,
    z: 8,
    yaw: 0,
    // A collision-free pose just above the target. The test still asks the IK
    // solver for a different endpoint and executes the returned physical plan;
    // this merely avoids making an unrelated long joint-space sweep the setup.
    joints: {
      head_yaw: 0,
      head_pitch: -0.15,
      shoulder: 0.2097693681716919,
      elbow: -0.2218160331249237,
      wrist: 0.44839078187942505,
      gripper_aperture: 0.4
    }
  };
  scenario.objects = [];
  scenario.zones = [];
  return scenario;
}

async function streamingScenario(): Promise<Scenario> {
  const catalog = await loadRuntimeCatalog();
  const scenario = catalog.materialize("voxel_realm", 29);
  const size = 192;
  const heights = new Array<number>(size * size).fill(0);
  for (let index = 0; index < size; index += 1) {
    heights[index] = 3;
    heights[(size - 1) * size + index] = 3;
    heights[index * size] = 3;
    heights[index * size + size - 1] = 3;
  }
  scenario.terrain = {
    cell: 1,
    block: 0.9,
    columns: size,
    rows: size,
    chunk_size: 16,
    maximum_height: 24,
    heights
  };
  scenario.bounds = { width: size, depth: size };
  scenario.robot = { x: 63, z: 63, yaw: 0 };
  scenario.objects = [];
  scenario.zones = [];
  return scenario;
}

async function placementTrapScenario(): Promise<Scenario> {
  const scenario = await interactionScenario();
  const size = scenario.terrain?.columns ?? 32;
  if (!scenario.terrain) throw new Error("Expected voxel terrain");
  scenario.terrain.heights = new Array<number>(size * size).fill(0);
  for (let index = 0; index < size; index += 1) {
    scenario.terrain.heights[index] = 3;
    scenario.terrain.heights[(size - 1) * size + index] = 3;
    scenario.terrain.heights[index * size] = 3;
    scenario.terrain.heights[index * size + size - 1] = 3;
  }
  // One block below the empty target supplies legitimate placement support.
  scenario.terrain.heights[PLACEMENT_TRAP.row * size + PLACEMENT_TRAP.column] = 1;
  // This is the translated form of the real pose that exposed the original
  // bug: the physical gripper is inside the future block volume, while an IK
  // solution to the reported side point has only been planned, not executed.
  scenario.robot = {
    x: 10.08390808105469,
    z: 9.502439498901367,
    yaw: -1.5646644695811267,
    joints: {
      head_yaw: -0.08784015827803149,
      head_pitch: -0.08353373495629057,
      shoulder: 0.8038892149925232,
      elbow: -1.0138753652572632,
      wrist: -0.9518631100654602,
      gripper_aperture: 0.38398728417605166
    }
  };
  return scenario;
}

describe("voxel manipulation through the agent tool surface", () => {
  it("scans, reaches, edits physics/navigation, checkpoints, and resumes", async () => {
    const scenario = await interactionScenario();
    const world = await RapierWorld.create(scenario);
    let resumed: RapierWorld | undefined;
    let restoredPlacement: RapierWorld | undefined;
    try {
      const blockedBeforeBreak = await executeTool(world, "plan_base_path", {
        target: { x: 8.5, y: 0, z: 9.5 }
      });
      expect(blockedBeforeBreak).toMatchObject({
        accepted: false,
        code: "base_path_unavailable"
      });

      const scan = await executeTool(world, "scan_voxels", { radius: 4, limit: 8 });
      expect(scan).toMatchObject({ accepted: true, code: "voxel_scan" });
      const blocks = detail(scan.detail).blocks as Array<Record<string, unknown>>;
      expect(blocks).toEqual(expect.arrayContaining([
        expect.objectContaining({
          coordinate: TARGET,
          material: "grass",
          visible: true,
          reachable_by_gripper: true
        })
      ]));

      const boundedScan = await executeTool(world, "scan_voxels", {
        radius: 64,
        limit: 64
      });
      expect(boundedScan).toMatchObject({ accepted: true, code: "voxel_scan" });
      const boundedBlocks = detail(boundedScan.detail).blocks as unknown[];
      expect(boundedBlocks.length).toBeLessThanOrEqual(48);

      const inspected = await executeTool(world, "inspect_voxel", { coordinate: TARGET });
      expect(inspected).toMatchObject({ accepted: true, code: "voxel_state" });
      const standoffs = detail(inspected.detail).reachable_standoff_poses as Array<{
        target: { x: number; y: number; z: number };
        face_point: { x: number; y: number; z: number };
        standoff_radius: number;
        approach_axis_error: number;
        recommended: boolean;
      }>;
      expect(standoffs.length).toBeGreaterThan(0);
      expect(standoffs.every((pose) =>
        Math.hypot(
          pose.target.x - pose.face_point.x,
          pose.target.z - pose.face_point.z
        ) > 0.02
      )).toBe(true);
      expect(standoffs[0]).toMatchObject({ recommended: true, approach_axis_error: 0 });
      expect(new Set(standoffs.map((pose) => pose.standoff_radius)).size)
        .toBeGreaterThan(1);

      const voxelRadius = (scenario.terrain?.cell ?? 1) * Math.SQRT1_2;
      const minimumSafeRadius = ROBOT_SPEC.base.footprintRadius + voxelRadius;
      expect(standoffs[0]?.standoff_radius).toBeCloseTo(minimumSafeRadius, 6);

      const faces = detail(inspected.detail).exposed_faces as Array<{
        normal: { x: number; y: number; z: number };
        interaction_point: { x: number; y: number; z: number };
      }>;
      const topFace = faces.find((face) => face.normal.y === 1);
      expect(topFace).toBeDefined();
      expect(VOXEL_INTERACTION_CLEARANCE).toBeLessThan(VOXEL_INTERACTION_DISTANCE);
      expect(topFace?.interaction_point.y).toBeCloseTo(
        0.9 + VOXEL_INTERACTION_CLEARANCE,
        6
      );

      const approach = standoffs[0];
      if (!approach || !topFace) throw new Error("Expected a voxel approach and top face");
      const facingYaw = Math.atan2(
        approach.face_point.x - approach.target.x,
        approach.face_point.z - approach.target.z
      );
      const shoulder = {
        x: approach.target.x
          + Math.sin(facingYaw) * ROBOT_SPEC.arm.shoulderForwardOffset,
        y: ROBOT_SPEC.arm.shoulderHeight,
        z: approach.target.z
          + Math.cos(facingYaw) * ROBOT_SPEC.arm.shoulderForwardOffset
      };
      const armReach = ROBOT_SPEC.arm.upperLength
        + ROBOT_SPEC.arm.forearmLength
        + ROBOT_SPEC.arm.wristLength;
      expect(Math.hypot(
        topFace.interaction_point.x - shoulder.x,
        topFace.interaction_point.y - shoulder.y,
        topFace.interaction_point.z - shoulder.z
      )).toBeLessThan(armReach);

      const solved = await executeTool(world, "solve_end_effector_position", {
        position: topFace?.interaction_point
      });
      expect(solved).toMatchObject({ accepted: true, code: "end_effector_solution" });
      const planId = detail(solved.detail).plan_id;
      expect(typeof planId).toBe("string");
      const reached = await executeSkill(
        world,
        source("reach_voxel", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: planId }
      );
      expect(reached).toMatchObject({ accepted: true, code: "joint_targets_reached" });

      const broken = await executeSkill(
        world,
        source("break_voxel", "break_voxel", ["arm"]),
        "break_voxel",
        { coordinate: TARGET }
      );
      expect(broken).toMatchObject({
        accepted: true,
        code: "voxel_broken",
        detail: {
          mutation: { coordinate: TARGET, before: "grass", after: null, revision: 1 },
          physics_updated: true,
          navigation_updated: true
        }
      });
      expect(checkGoal({
        summary: "The voxel is empty.",
        predicates: [{ type: "voxel_at", coordinate: TARGET, material: null }]
      }, world.snapshot()).success).toBe(true);

      const pathThroughRemovedVoxel = await executeTool(world, "plan_base_path", {
        target: { x: 8.5, y: 0, z: 9.5 }
      });
      expect(pathThroughRemovedVoxel).toMatchObject({
        accepted: true,
        code: "base_path_planned"
      });

      const brokenSnapshot = world.snapshot();
      expect(brokenSnapshot.voxels?.mutations).toHaveLength(1);
      resumed = await RapierWorld.create(scenario, brokenSnapshot);
      const emptyInspection = await executeTool(resumed, "inspect_voxel", { coordinate: TARGET });
      expect(emptyInspection).toMatchObject({
        accepted: true,
        code: "voxel_state",
        detail: { coordinate: TARGET, material: null }
      });
      const placementPoints = detail(emptyInspection.detail).placement_interaction_points as Array<{
        normal: { x: number; y: number; z: number };
        interaction_point: { x: number; y: number; z: number };
      }>;
      const topPlacement = placementPoints.find((point) => point.normal.y === 1);
      expect(topPlacement?.interaction_point.y).toBeCloseTo(
        0.9 + VOXEL_INTERACTION_CLEARANCE,
        6
      );
      expect(await executeTool(resumed, "plan_base_path", {
        target: { x: 8.5, y: 0, z: 9.5 }
      })).toMatchObject({ accepted: true, code: "base_path_planned" });

      const placed = await executeSkill(
        resumed,
        source("place_voxel", "place_voxel", ["arm"]),
        "place_voxel",
        { coordinate: TARGET, material: "grass" }
      );
      expect(placed).toMatchObject({
        accepted: true,
        code: "voxel_placed",
        detail: {
          mutation: { coordinate: TARGET, before: null, after: "grass", revision: 2 },
          physics_updated: true,
          navigation_updated: true
        }
      });
      expect(await executeTool(resumed, "plan_base_path", {
        target: { x: 8.5, y: 0, z: 9.5 }
      })).toMatchObject({ accepted: false, code: "base_path_unavailable" });

      const physicalContact = await executeSkill(
        resumed,
        source("drive_into_restored_voxel", "drive_base", ["base"]),
        "drive_base",
        {
          linear_meters_per_second: 0.5,
          angular_radians_per_second: 0,
          duration_seconds: 3
        }
      );
      expect(physicalContact).toMatchObject({ accepted: false, code: "base_motion_blocked" });
      expect(JSON.stringify(physicalContact.detail)).toContain('"collider_kind":"voxel"');

      const placedSnapshot = resumed.snapshot();
      restoredPlacement = await RapierWorld.create(scenario, placedSnapshot);
      expect(await executeTool(restoredPlacement, "inspect_voxel", {
        coordinate: TARGET
      })).toMatchObject({
        accepted: true,
        code: "voxel_state",
        detail: { coordinate: TARGET, material: "grass", voxel_revision: 2 }
      });
      expect(checkGoal({
        summary: "The voxel was rebuilt.",
        predicates: [{ type: "voxel_at", coordinate: TARGET, material: "grass" }]
      }, restoredPlacement.snapshot()).success).toBe(true);
    } finally {
      restoredPlacement?.dispose();
      resumed?.dispose();
      world.dispose();
    }
  }, 30_000);

  it("requires the physical gripper to reach an interaction point after IK planning", async () => {
    const scenario = await placementTrapScenario();
    const seedWorld = await RapierWorld.create(scenario);
    const snapshot = seedWorld.snapshot();
    seedWorld.dispose();
    if (!snapshot.voxels) throw new Error("Expected voxel snapshot");
    snapshot.voxels.inventory.grass = 1;

    const world = await RapierWorld.create(scenario, snapshot);
    try {
      const inspected = await executeTool(world, "inspect_voxel", {
        coordinate: PLACEMENT_TRAP
      });
      expect(inspected).toMatchObject({
        accepted: true,
        code: "voxel_state",
        detail: { material: null, placement_supported: true }
      });
      const points = detail(inspected.detail).placement_interaction_points as Array<{
        interaction_point: { x: number; y: number; z: number };
        gripper_distance: number;
        arm_workspace_fit: string;
        recommended: boolean;
      }>;
      const recommended = points.find((point) => point.recommended);
      const nearest = [...points].sort((left, right) =>
        left.gripper_distance - right.gripper_distance
      )[0];
      expect(recommended).toBeDefined();
      expect(recommended?.gripper_distance).toBeGreaterThan(VOXEL_INTERACTION_DISTANCE);
      expect(recommended?.arm_workspace_fit).toBe("preferred");

      const beforePlan = world.snapshot();
      const solved = await executeTool(world, "solve_end_effector_position", {
        position: recommended?.interaction_point
      });
      expect(solved).toMatchObject({ accepted: true, code: "end_effector_solution" });
      expect(world.snapshot().world_revision).toBe(beforePlan.world_revision);
      expect(world.snapshot().robot.links.gripper.position)
        .toEqual(beforePlan.robot.links.gripper.position);

      const placedWithoutExecution = await executeSkill(
        world,
        source("planned_only_place", "place_voxel", ["arm", "gripper"]),
        "place_voxel",
        { coordinate: PLACEMENT_TRAP, material: "grass" }
      );
      expect(placedWithoutExecution).toMatchObject({
        accepted: false,
        code: "voxel_out_of_reach",
        detail: {
          coordinate: PLACEMENT_TRAP,
          maximum_distance: VOXEL_INTERACTION_DISTANCE,
          nearest_interaction_point: {
            interaction_point: nearest?.interaction_point,
            gripper_distance: nearest?.gripper_distance
          }
        }
      });
      expect(String(detail(placedWithoutExecution.detail).recovery))
        .toContain("execute that arm plan");
      const after = world.snapshot();
      expect(after.world_revision).toBe(beforePlan.world_revision);
      expect(after.voxels?.revision).toBe(0);
      expect(after.voxels?.inventory.grass).toBe(1);
      expect(world.voxelMaterialAt(PLACEMENT_TRAP)).toBeNull();
    } finally {
      world.dispose();
    }
  }, 30_000);

  it("detects a robot collider in a future voxel without a broad-phase refresh", async () => {
    const scenario = await placementTrapScenario();
    if (!scenario.terrain) throw new Error("Expected voxel terrain");
    const store = new VoxelStore(scenario.terrain);
    const center = store.centerOf(PLACEMENT_TRAP);
    store.setLoadedChunks(store.desiredChunksAround(center, 2));
    const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    try {
      const finger = world.createRigidBody(
        RAPIER.RigidBodyDesc.kinematicPositionBased()
          .setTranslation(center.x, center.y, center.z)
          .setUserData({ kind: "robot", link_id: "test_finger" })
      );
      world.createCollider(RAPIER.ColliderDesc.cuboid(0.04, 0.14, 0.08), finger);
      const interaction = new VoxelInteraction({
        world,
        store,
        links: new Map(),
        visibilityRadius: scenario.visibility_radius
      });
      expect(interaction.placementObstructions(PLACEMENT_TRAP)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            collider_kind: "robot",
            collider_id: "test_finger"
          })
        ])
      );
    } finally {
      world.free();
    }
  });

  it("streams a bounded chunk neighbourhood while crossing a large world", async () => {
    const scenario = await streamingScenario();
    const world = await RapierWorld.create(scenario);
    let restored: RapierWorld | undefined;
    try {
      const before = world.snapshot();
      expect(scenario.bounds).toEqual({ width: 192, depth: 192 });
      expect(before.voxels?.loaded_chunks).toHaveLength(25);
      const beforeKeys = new Set(before.voxels?.loaded_chunks.map((chunk) =>
        `${chunk.column}:${chunk.row}`
      ));
      expect(beforeKeys.has("1:1")).toBe(true);
      expect(beforeKeys.has("5:5")).toBe(true);

      const crossed = await executeSkill(
        world,
        source("cross_chunk_boundary", "drive_base", ["base"]),
        "drive_base",
        {
          linear_meters_per_second: 0.6,
          angular_radians_per_second: 0,
          duration_seconds: 5
        }
      );
      expect(crossed).toMatchObject({ accepted: true, code: "base_motion_completed" });
      const after = world.snapshot();
      expect(after.robot.position.z).toBeGreaterThan(65.5);
      expect(after.voxels?.loaded_chunks).toHaveLength(25);
      const afterKeys = new Set(after.voxels?.loaded_chunks.map((chunk) =>
        `${chunk.column}:${chunk.row}`
      ));
      expect(afterKeys.has("1:1")).toBe(false);
      expect(afterKeys.has("1:6")).toBe(true);

      const nearbyPlan = await executeTool(world, "plan_base_path", {
        target: { x: after.robot.position.x, y: 0, z: after.robot.position.z + 4 }
      });
      expect(nearbyPlan).toMatchObject({ accepted: true, code: "base_path_planned" });

      restored = await RapierWorld.create(scenario, after);
      expect(restored.snapshot().voxels?.loaded_chunks).toEqual(after.voxels?.loaded_chunks);
      expect(restored.snapshot().robot.position.z).toBeCloseTo(after.robot.position.z, 5);
    } finally {
      restored?.dispose();
      world.dispose();
    }
  }, 30_000);
});
