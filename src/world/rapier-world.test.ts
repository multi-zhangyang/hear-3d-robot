import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { WorldSnapshotSchema, type BodyChannel } from "../domain/schema.js";
import { executeSkill, executeTool } from "../runtime/actions.js";
import { checkGoal } from "../runtime/checker.js";
import { RapierWorld, type SourceCommand } from "./rapier-world.js";
import { ROBOT_SPEC } from "./robot-model.js";

function source(id: string, skill: string, channels: BodyChannel[]): SourceCommand {
  return {
    id,
    agentId: "agent_node_1",
    agentName: "Motion controller",
    skill,
    channels
  };
}

function detailRecord(value: unknown): Record<string, unknown> {
  expect(value).not.toBeNull();
  expect(typeof value).toBe("object");
  expect(Array.isArray(value)).toBe(false);
  return value as Record<string, unknown>;
}

describe("RapierWorld atomic controls", () => {
  it("plans and executes an arbitrary navmesh path from an atomic base command", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const before = world.snapshot();
      const target = { x: 3, y: 0, z: 1 };
      const planned = await executeTool(world, "plan_base_path", { target });
      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });

      const planId = detailRecord(planned.detail).plan_id;
      expect(typeof planId).toBe("string");
      if (typeof planId !== "string") return;

      const moved = await executeSkill(
        world,
        source("base_command_1", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: planId }
      );
      expect(moved).toMatchObject({ accepted: true, code: "base_plan_completed" });

      const after = world.snapshot();
      expect(after.frame - before.frame).toBeGreaterThan(1);
      expect(after.robot.position.x).toBeCloseTo(target.x, 1);
      expect(after.robot.position.z).toBeCloseTo(target.z, 1);
      expect(after.robot.position.x).not.toBe(before.robot.position.x);
      expect(after.navigation).toMatchObject({
        plan_id: planId,
        status: "completed"
      });
      expect(after.navigation.target?.x).toBeCloseTo(target.x, 5);
      expect(after.navigation.target?.y).toBeCloseTo(ROBOT_SPEC.base.centerY, 5);
      expect(after.navigation.target?.z).toBeCloseTo(target.z, 5);
      expect(after.navigation.actual_path.length).toBeGreaterThan(1);
      expect(Math.abs(after.robot.odometry.left_wheel.position)
        + Math.abs(after.robot.odometry.right_wheel.position)).toBeGreaterThan(0);
      expect(after.last_command).toMatchObject({
        id: "base_command_1",
        agent_id: "agent_node_1",
        agent_name: "Motion controller",
        skill: "execute_base_plan",
        channels: ["base"],
        accepted: true,
        result_code: "base_plan_completed"
      });
      expect(WorldSnapshotSchema.safeParse(after).success).toBe(true);

      expect(checkGoal({
        summary: "Robot reaches the requested coordinate.",
        predicates: [{ type: "robot_at", target, tolerance: 0.15 }]
      }, after).success).toBe(true);
    } finally {
      world.dispose();
    }
  });

  it("expands enabled portable objects by the mobile-base footprint", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);
    const redBlock = scenario.objects.find((object) => object.id === "red_block");

    const world = await RapierWorld.create(scenario);
    try {
      const planned = await executeTool(world, "plan_base_path", {
        target: {
          x: redBlock.position.x,
          y: 0,
          z: redBlock.position.z - 0.5
        },
        face_point: redBlock.position
      });
      expect(planned).toMatchObject({ accepted: false, code: "base_path_unavailable" });
      const detail = detailRecord(planned.detail);
      expect(detail.error).toContain("projection");
      expect(world.snapshot().plans.base).toHaveLength(0);
    } finally {
      world.dispose();
    }
  });

  it("accepts a clear manipulation standoff outside an expanded portable object", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);
    const redBlock = scenario.objects.find((object) => object.id === "red_block");

    const world = await RapierWorld.create(scenario);
    try {
      const standoff = Math.max(
        ROBOT_SPEC.base.footprintRadius + redBlock.size.z / 2 + 0.16,
        ROBOT_SPEC.base.halfExtents.z
          + redBlock.size.z / 2
          + ROBOT_SPEC.arm.wristLength
          + ROBOT_SPEC.arm.wristHalfExtents.z
          + 0.05
      );
      const planned = await executeTool(world, "plan_base_path", {
        target: {
          x: redBlock.position.x,
          y: 0,
          z: redBlock.position.z - standoff
        },
        face_point: redBlock.position
      });

      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
      const detail = detailRecord(planned.detail);
      expect(typeof detail.plan_id).toBe("string");
      expect(Number(detail.projection_distance)).toBeLessThanOrEqual(0.15);
      expect(world.snapshot().plans.base).toHaveLength(1);
    } finally {
      world.dispose();
    }
  });

  it("executes a reverse path when turning in place would hit a nearby object", async () => {
    const catalog = await loadRuntimeCatalog();
    const baseScenario = catalog.materialize("open_navigation", 0);

    const scenario = structuredClone(baseScenario);
    scenario.robot = { x: 2, z: 2, yaw: 0 };
    scenario.objects = [{
      id: "nearby_block",
      kind: "block",
      color: "#777777",
      position: { x: 2, y: 0.25, z: 2.6 },
      size: { x: 0.5, y: 0.5, z: 0.5 },
      portable: true
    }];
    const world = await RapierWorld.create(scenario);
    try {
      const target = { x: 2, y: 0, z: 1.2 };
      const planned = await executeTool(world, "plan_base_path", { target });
      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
      const detail = detailRecord(planned.detail);
      const planId = detail.plan_id;
      expect(typeof planId).toBe("string");
      if (typeof planId !== "string") return;

      const moved = await executeSkill(
        world,
        source("reverse_base_plan", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: planId, options: { tolerance: 0.015 } }
      );
      expect(moved).toMatchObject({ accepted: true, code: "base_plan_completed" });
      const after = world.snapshot();
      expect(after.robot.position.x).toBeCloseTo(target.x, 1);
      expect(after.robot.position.z).toBeCloseTo(target.z, 1);
      expect(Math.abs(after.robot.yaw)).toBeLessThan(0.08);
      expect(after.robot.odometry.left_wheel.position).toBeLessThan(0);
      expect(after.robot.odometry.right_wheel.position).toBeLessThan(0);
    } finally {
      world.dispose();
    }
  });

  it("executes explicit head, arm, and gripper targets", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const head = await executeSkill(
        world,
        source("head_command_1", "set_head_target", ["head"]),
        "set_head_target",
        { yaw: 0.35, pitch: -0.2 }
      );
      expect(head).toMatchObject({ accepted: true, code: "head_target_reached" });
      expect(world.snapshot().robot.joints).toMatchObject({ head_yaw: 0.35, head_pitch: -0.2 });

      const arm = await executeSkill(
        world,
        source("arm_command_1", "set_joint_targets", ["arm"]),
        "set_joint_targets",
        { targets: { shoulder: 1.2, elbow: -2.1, wrist: 0.9 } }
      );
      expect(arm).toMatchObject({ accepted: true, code: "joint_targets_reached" });
      expect(world.snapshot().robot.joints).toMatchObject({ shoulder: 1.2, elbow: -2.1, wrist: 0.9 });

      const gripper = await executeSkill(
        world,
        source("gripper_command_1", "set_gripper_target", ["gripper"]),
        "set_gripper_target",
        { aperture: 0.5, max_force: 80 }
      );
      expect(gripper).toMatchObject({ accepted: true, code: "gripper_target_reached" });
      expect(world.snapshot().robot.gripper).toMatchObject({
        aperture: 0.5,
        target_aperture: 0.5,
        maximum_force: 80
      });

    } finally {
      world.dispose();
    }
  });

  it("does not invent a world revision for an already-reached joint target", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);
    const world = await RapierWorld.create(scenario);

    try {
      const planned = await executeTool(world, "plan_base_path", {
        target: { x: 3, y: 0, z: 1 }
      });
      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
      const planId = detailRecord(planned.detail).plan_id;
      expect(typeof planId).toBe("string");
      if (typeof planId !== "string") return;

      const before = world.snapshot();
      const noOp = await executeSkill(
        world,
        source("already_reached_arm", "set_joint_targets", ["arm"]),
        "set_joint_targets",
        {
          targets: {
            shoulder: before.robot.joints.shoulder,
            elbow: before.robot.joints.elbow,
            wrist: before.robot.joints.wrist
          }
        }
      );
      expect(noOp).toMatchObject({ accepted: true, code: "joint_targets_reached" });

      const after = world.snapshot();
      expect(after.frame).toBe(before.frame);
      expect(after.world_revision).toBe(before.world_revision);
      expect(after.last_command).toMatchObject({
        id: "already_reached_arm",
        accepted: true,
        result_code: "joint_targets_reached"
      });

      const executed = await executeSkill(
        world,
        source("plan_after_noop", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: planId }
      );
      expect(executed).toMatchObject({ accepted: true, code: "base_plan_completed" });
    } finally {
      world.dispose();
    }
  });

  it("steps non-conflicting base, head, and arm commands in shared physics frames", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);
    const world = await RapierWorld.create(scenario);
    const frames: ReturnType<RapierWorld["snapshot"]>[] = [];
    world.setFrameSink((batch) => {
      frames.push(...batch);
    });

    try {
      const before = world.snapshot();
      const armPlan = await executeTool(world, "plan_joint_targets", {
        targets: { shoulder: 1.2, elbow: -2.1, wrist: 0.9 }
      });
      expect(armPlan).toMatchObject({ accepted: true, code: "joint_target_plan" });
      const armPlanId = detailRecord(armPlan.detail).plan_id;
      expect(typeof armPlanId).toBe("string");
      if (typeof armPlanId !== "string") return;
      const afterPlanning = world.snapshot();
      expect(afterPlanning.frame).toBe(before.frame);
      expect(afterPlanning.world_revision).toBe(before.world_revision);
      expect(afterPlanning.plans.arm).toContainEqual(expect.objectContaining({
        id: armPlanId,
        kind: "joint_targets",
        target: null
      }));

      const [base, head, arm] = await Promise.all([
        world.driveBase(
          source("parallel_base", "drive_base", ["base"]),
          0.12,
          0,
          0.35
        ),
        world.setHeadTarget(
          source("parallel_head", "set_head_target", ["head"]),
          0.25,
          -0.12
        ),
        world.executeJointPlan(
          source("parallel_arm", "execute_joint_plan", ["arm"]),
          armPlanId
        )
      ]);

      expect([base, head, arm].every((result) => result.accepted)).toBe(true);
      expect(frames.some((frame) => {
        const channels = new Set(frame.active_commands.flatMap((command) => command.channels));
        return channels.has("base") && channels.has("head") && channels.has("arm");
      })).toBe(true);
      const after = world.snapshot();
      expect(after.frame).toBeGreaterThan(before.frame);
      expect(after.world_revision - before.world_revision).toBe(3);
      expect(after.active_commands).toEqual([]);
      expect(after.robot.position.z).toBeGreaterThan(before.robot.position.z);
      expect(after.robot.joints).toMatchObject({
        head_yaw: 0.25,
        head_pitch: -0.12,
        shoulder: 1.2,
        elbow: -2.1,
        wrist: 0.9
      });
    } finally {
      world.dispose();
    }
  });

  it("generates and consumes an inverse-kinematics plan for the current end effector pose", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const position = world.snapshot().robot.links.gripper?.position;

      const solved = await executeTool(world, "solve_end_effector_position", { position });
      expect(solved).toMatchObject({ accepted: true, code: "end_effector_solution" });
      const planId = detailRecord(solved.detail).plan_id;
      expect(typeof planId).toBe("string");
      if (typeof planId !== "string") return;

      const executed = await executeSkill(
        world,
        source("arm_plan_command_1", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: planId }
      );
      expect(executed).toMatchObject({ accepted: true, code: "joint_targets_reached" });

      const repeated = await executeSkill(
        world,
        source("arm_plan_command_2", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: planId }
      );
      expect(repeated).toMatchObject({ accepted: false, code: "plan_already_consumed" });
      expect(detailRecord(repeated.detail)).toMatchObject({ plan_id: planId });

      const overconstrained = await executeTool(world, "solve_end_effector_pose", {
        position: { x: 0.5, y: 0.5, z: 1 },
        orientation: { x: 0, y: 0, z: Math.SQRT1_2, w: Math.SQRT1_2 }
      });
      expect(overconstrained.accepted).toBe(false);
      expect(overconstrained.code).not.toBe("ik_solver_error");
      expect(String(detailRecord(overconstrained.detail).recovery))
        .toContain("solve_end_effector_position");
    } finally {
      world.dispose();
    }
  });

  it("tells a denied solve that the target is out of reach and the base must move", async () => {
    // A live run stalled here: the agent solved for the block from across the
    // room, got a 1.20m residual with no explanation, and re-read
    // proprioception instead of driving. The number alone is not actionable.
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const block = scenario.objects.find((object) => object.id === "red_block");

      const denial = await executeTool(world, "solve_end_effector_pose", {
        position: { x: block.position.x, y: block.position.y + block.size.y / 2, z: block.position.z }
      });
      expect(denial.accepted).toBe(false);

      const detail = detailRecord(denial.detail);
      expect(detail.target_within_reach).toBe(false);
      expect(detail.maximum_arm_reach).toBeCloseTo(1.35, 5);
      expect(detail.target_distance_from_shoulder).toBeGreaterThan(
        detail.maximum_arm_reach as number
      );
      // The recovery must name the tools that actually close the gap.
      expect(detail.recovery).toContain("reachable_standoff_poses");
      expect(detail.recovery).toContain("execute_base_plan");
    } finally {
      world.dispose();
    }
  });

  it("tells a blocked solve exactly how much higher the target must be", async () => {
    // Also from a live run: standing next to the block, the agent aimed at the
    // block's own center and the wrist ended up inside it. The denial listed
    // the colliding segment but not the distance, so the agent retried three
    // more guesses instead of one correction.
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const block = scenario.objects.find((object) => object.id === "red_block");

    const world = await RapierWorld.create(scenario);
    try {
      const inspected = await executeTool(world, "inspect_entity", { entity_id: "red_block" });
      const standoffs = detailRecord(inspected.detail).reachable_standoff_poses;
      expect(Array.isArray(standoffs) && standoffs.length > 0).toBe(true);
      if (!Array.isArray(standoffs) || standoffs.length === 0) return;

      // The closest legal pose is the one a grasp would actually be attempted from.
      const closest = [...standoffs].map(detailRecord).sort(
        (a, b) => (a.distance_to_entity as number) - (b.distance_to_entity as number)
      )[0];
      const planned = await executeTool(world, "plan_base_path", {
        target: closest.target,
        face_point: closest.face_point
      });
      expect(planned).toMatchObject({ accepted: true });
      const planId = detailRecord(planned.detail).plan_id;
      if (typeof planId !== "string") return;
      const drove = await executeSkill(
        world,
        source("approach_for_clearance", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: planId }
      );
      expect(drove).toMatchObject({ accepted: true });

      // Aiming at the block's own center puts the wrist inside it.
      const blocked = await executeTool(world, "solve_end_effector_pose", {
        position: { ...block.position }
      });
      expect(blocked).toMatchObject({ accepted: false, code: "ik_trajectory_endpoint_blocked" });

      const recovery = detailRecord(blocked.detail).recovery;
      expect(typeof recovery).toBe("string");
      expect(recovery).toContain("red_block");
      // The correction must be a concrete height, not just a complaint.
      expect(recovery).toMatch(/y=\d+\.\d+/);
    } finally {
      world.dispose();
    }
  });

  it("does not claim a target is unreachable when the arm can span it", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const position = world.snapshot().robot.links.gripper?.position;

      // Its own current pose is by definition reachable, so a solve there must
      // succeed and carry no reach complaint at all.
      const solved = await executeTool(world, "solve_end_effector_pose", { position });
      expect(solved).toMatchObject({ accepted: true });
      expect(detailRecord(solved.detail).recovery).toBeUndefined();
    } finally {
      world.dispose();
    }
  });

  it("restores unconsumed base and arm plans that remain executable after restart", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);

    const planningWorld = await RapierWorld.create(scenario);
    const { snapshot, basePlanId, armPlanId, jointPlanId } = await (async () => {
      try {
        const basePlan = await executeTool(planningWorld, "plan_base_path", {
          target: { x: 3, y: 0, z: 1 }
        });
        expect(basePlan).toMatchObject({ accepted: true, code: "base_path_planned" });
        const basePlanId = detailRecord(basePlan.detail).plan_id;
        expect(typeof basePlanId).toBe("string");
        if (typeof basePlanId !== "string") {
          throw new Error("Base planner did not return a plan identifier");
        }

        const gripperPosition = planningWorld.snapshot().robot.links.gripper?.position;
        expect(gripperPosition).toBeDefined();
        if (!gripperPosition) throw new Error("World snapshot is missing the gripper link");
        const armPlan = await executeTool(planningWorld, "solve_end_effector_pose", {
          position: gripperPosition
        });
        expect(armPlan).toMatchObject({ accepted: true, code: "end_effector_solution" });
        const armPlanId = detailRecord(armPlan.detail).plan_id;
        expect(typeof armPlanId).toBe("string");
        if (typeof armPlanId !== "string") {
          throw new Error("Arm planner did not return a plan identifier");
        }

        const jointPlan = await executeTool(planningWorld, "plan_joint_targets", {
          targets: { shoulder: 1.2, elbow: -2.1, wrist: 0.9 }
        });
        expect(jointPlan).toMatchObject({ accepted: true, code: "joint_target_plan" });
        const jointPlanId = detailRecord(jointPlan.detail).plan_id;
        expect(typeof jointPlanId).toBe("string");
        if (typeof jointPlanId !== "string") {
          throw new Error("Joint planner did not return a plan identifier");
        }

        const snapshot = planningWorld.snapshot();
        expect(snapshot.plans.base.map((plan) => plan.id)).toContain(basePlanId);
        expect(snapshot.plans.arm.map((plan) => plan.id)).toContain(armPlanId);
        expect(snapshot.plans.arm).toContainEqual(expect.objectContaining({
          id: jointPlanId,
          kind: "joint_targets",
          target: null
        }));
        return { snapshot, basePlanId, armPlanId, jointPlanId };
      } finally {
        planningWorld.dispose();
      }
    })();

    const baseWorld = await RapierWorld.create(scenario, snapshot);
    try {
      expect(baseWorld.snapshot().plans.base.map((plan) => plan.id)).toContain(basePlanId);
      const executedBase = await executeSkill(
        baseWorld,
        source("restored_base_command", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: basePlanId }
      );
      expect(executedBase).toMatchObject({ accepted: true, code: "base_plan_completed" });
    } finally {
      baseWorld.dispose();
    }

    const armWorld = await RapierWorld.create(scenario, snapshot);
    try {
      expect(armWorld.snapshot().plans.arm.map((plan) => plan.id)).toContain(armPlanId);
      const executedArm = await executeSkill(
        armWorld,
        source("restored_arm_command", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: armPlanId }
      );
      expect(executedArm).toMatchObject({ accepted: true, code: "joint_targets_reached" });
    } finally {
      armWorld.dispose();
    }

    const jointWorld = await RapierWorld.create(scenario, snapshot);
    try {
      expect(jointWorld.snapshot().plans.arm.map((plan) => plan.id)).toContain(jointPlanId);
      const executedJointPlan = await executeSkill(
        jointWorld,
        source("restored_joint_command", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: jointPlanId }
      );
      expect(executedJointPlan).toMatchObject({ accepted: true, code: "joint_targets_reached" });
      expect(jointWorld.snapshot().robot.joints).toMatchObject({
        shoulder: 1.2,
        elbow: -2.1,
        wrist: 0.9
      });
    } finally {
      jointWorld.dispose();
    }
  });

  it("restores an end-effector plan from a checkpoint created before arm plan kinds", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);
    const planningWorld = await RapierWorld.create(scenario);
    let legacySnapshot: ReturnType<RapierWorld["snapshot"]> | undefined;
    let planId: string | undefined;
    try {
      const position = planningWorld.snapshot().robot.links.gripper?.position;
      expect(position).toBeDefined();
      const planned = await executeTool(planningWorld, "solve_end_effector_position", { position });
      expect(planned).toMatchObject({ accepted: true, code: "end_effector_solution" });
      const rawPlanId = detailRecord(planned.detail).plan_id;
      expect(typeof rawPlanId).toBe("string");
      if (typeof rawPlanId !== "string") throw new Error("Planner returned no plan id");
      planId = rawPlanId;

      const legacy = structuredClone(planningWorld.snapshot()) as unknown as Record<string, unknown>;
      const plans = (legacy.plans as { arm: Array<Record<string, unknown>> }).arm;
      delete plans[0]?.kind;
      legacySnapshot = WorldSnapshotSchema.parse(legacy);
      expect(legacySnapshot.plans.arm[0]?.kind).toBe("end_effector");
    } finally {
      planningWorld.dispose();
    }

    if (!legacySnapshot || !planId) throw new Error("Legacy checkpoint fixture was not built");
    const restored = await RapierWorld.create(scenario, legacySnapshot);
    try {
      const executed = await executeSkill(
        restored,
        source("legacy_arm_plan", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: planId }
      );
      expect(executed).toMatchObject({ accepted: true, code: "joint_targets_reached" });
    } finally {
      restored.dispose();
    }
  });

  it("allows base retreat when each step reduces an existing link penetration", async () => {
    const catalog = await loadRuntimeCatalog();
    const baseScenario = catalog.materialize("open_navigation", 0);

    const probe = await RapierWorld.create(baseScenario);
    const wristPosition = probe.snapshot().robot.links.wrist?.position;
    probe.dispose();

    const scenario = structuredClone(baseScenario);
    scenario.objects = [{
      id: "wrist_blocker",
      kind: "block",
      color: "#777777",
      position: wristPosition,
      size: { x: 0.14, y: 0.14, z: 0.14 },
      portable: false
    }];
    const world = await RapierWorld.create(scenario);
    try {
      const before = world.snapshot().robot.position;
      const retreated = await executeSkill(
        world,
        source("collision_retreat", "drive_base", ["base"]),
        "drive_base",
        {
          linear_meters_per_second: -0.3,
          angular_radians_per_second: 0,
          duration_seconds: 0.25
        }
      );

      expect(retreated).toMatchObject({ accepted: true, code: "base_motion_completed" });
      const after = world.snapshot().robot.position;
      expect(Math.hypot(after.x - before.x, after.z - before.z)).toBeGreaterThan(0.05);
    } finally {
      world.dispose();
    }
  });

  it("blocks base motion when either the torso or sensor head reaches an obstacle", async () => {
    const catalog = await loadRuntimeCatalog();
    const baseScenario = catalog.materialize("open_navigation", 0);

    const probe = await RapierWorld.create(baseScenario);
    const links = probe.snapshot().robot.links;
    probe.dispose();
    const cases = [
      {
        segment: "torso",
        center: {
          x: links.torso!.position.x - 0.28,
          y: 0.92,
          z: links.torso!.position.z + 0.36
        },
        size: { x: 0.1, y: 0.2, z: 0.1 }
      },
      {
        segment: "sensor_head",
        center: {
          x: links.sensor_head!.position.x + 0.2,
          y: links.sensor_head!.position.y,
          z: links.sensor_head!.position.z + 0.32
        },
        size: { x: 0.08, y: 0.1, z: 0.1 }
      }
    ];

    for (const candidate of cases) {
      const scenario = structuredClone(baseScenario);
      scenario.obstacles.push({
        id: `${candidate.segment}_barrier`,
        center: candidate.center,
        size: candidate.size
      });
      const world = await RapierWorld.create(scenario);
      try {
        const result = await executeSkill(
          world,
          source(`${candidate.segment}_base_collision`, "drive_base", ["base"]),
          "drive_base",
          {
            linear_meters_per_second: 0.3,
            angular_radians_per_second: 0,
            duration_seconds: 1
          }
        );
        expect(result).toMatchObject({ accepted: false, code: "base_motion_blocked" });
        const issue = detailRecord(detailRecord(result.detail).issue);
        const collisions = issue.collisions as Array<Record<string, unknown>>;
        expect(collisions.some((collision) => collision.segment === candidate.segment)).toBe(true);
      } finally {
        world.dispose();
      }
    }
  });

  it("rejects sensor-head and gripper targets that would enter obstacles", async () => {
    const catalog = await loadRuntimeCatalog();
    const baseScenario = catalog.materialize("open_navigation", 0);

    const probe = await RapierWorld.create(baseScenario);
    const probeLinks = probe.snapshot().robot.links;
    probe.dispose();

    const headScenario = structuredClone(baseScenario);
    headScenario.obstacles.push({
      id: "head_sweep_barrier",
      center: {
        x: probeLinks.sensor_head!.position.x + 0.24,
        y: probeLinks.sensor_head!.position.y,
        z: probeLinks.sensor_head!.position.z
      },
      size: { x: 0.05, y: 0.12, z: 0.05 }
    });
    const headWorld = await RapierWorld.create(headScenario);
    try {
      const result = await executeSkill(
        headWorld,
        source("blocked_head", "set_head_target", ["head"]),
        "set_head_target",
        { yaw: 0.8, pitch: 0 }
      );
      expect(result).toMatchObject({ accepted: false, code: "head_motion_blocked" });
      const collisions = detailRecord(result.detail).collisions as Array<Record<string, unknown>>;
      expect(collisions.some((collision) => collision.segment === "sensor_head"
        && collision.collider_id === "head_sweep_barrier")).toBe(true);
    } finally {
      headWorld.dispose();
    }

    const gripperScenario = structuredClone(baseScenario);
    gripperScenario.obstacles.push({
      id: "finger_sweep_barrier",
      center: {
        x: probeLinks.left_finger!.position.x + 0.11,
        y: probeLinks.left_finger!.position.y,
        z: probeLinks.left_finger!.position.z
      },
      size: { x: 0.06, y: 0.1, z: 0.1 }
    });
    const gripperWorld = await RapierWorld.create(gripperScenario);
    try {
      const result = await executeSkill(
        gripperWorld,
        source("blocked_gripper", "set_gripper_target", ["gripper"]),
        "set_gripper_target",
        { aperture: ROBOT_SPEC.joints.gripper_aperture.maximum, max_force: 1000 }
      );
      expect(result).toMatchObject({ accepted: false, code: "gripper_motion_blocked" });
      const collisions = detailRecord(result.detail).collisions as Array<Record<string, unknown>>;
      expect(collisions.some((collision) => collision.segment === "left_finger"
        && collision.collider_id === "finger_sweep_barrier")).toBe(true);
    } finally {
      gripperWorld.dispose();
    }
  });

  it("rejects arm motion that creates non-adjacent robot self-collision", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);

    const world = await RapierWorld.create(scenario);
    try {
      // Elbow at its fold limit swings the fingers back into the torso, which
      // is a non-adjacent pair. Dropping the shoulder instead would drive the
      // fingers into the ground, and that is a different rejection.
      const result = await executeSkill(
        world,
        source("self_collision", "set_joint_targets", ["arm"]),
        "set_joint_targets",
        { targets: { shoulder: 0, elbow: -2.5, wrist: -1.7 } }
      );
      expect(result).toMatchObject({ accepted: false, code: "joint_motion_blocked" });
      const collisions = detailRecord(result.detail).collisions as Array<Record<string, unknown>>;
      expect(collisions.some((collision) => collision.collider_kind === "robot")).toBe(true);
    } finally {
      world.dispose();
    }
  });

  it("does not retreat through one penetration by deepening another collision", async () => {
    const catalog = await loadRuntimeCatalog();
    const baseScenario = catalog.materialize("open_navigation", 0);

    const probe = await RapierWorld.create(baseScenario);
    const wrist = probe.snapshot().robot.links.wrist?.position;
    probe.dispose();

    const scenario = structuredClone(baseScenario);
    scenario.objects = [-0.06, 0.06].map((offset, index) => ({
      id: `wrist_blocker_${index}`,
      kind: "block",
      color: "#777777",
      position: { x: wrist.x, y: wrist.y, z: wrist.z + offset },
      size: { x: 0.12, y: 0.12, z: 0.12 },
      portable: false
    }));
    const world = await RapierWorld.create(scenario);
    try {
      const before = world.snapshot().robot.position;
      const result = await executeSkill(
        world,
        source("multi_collision_retreat", "drive_base", ["base"]),
        "drive_base",
        {
          linear_meters_per_second: -0.3,
          angular_radians_per_second: 0,
          duration_seconds: 0.25
        }
      );
      expect(result).toMatchObject({ accepted: false, code: "base_motion_blocked" });
      const issue = detailRecord(detailRecord(result.detail).issue);
      const collisions = issue.collisions as Array<Record<string, unknown>>;
      const colliderIds = new Set(collisions.map((collision) => collision.collider_id));
      expect(colliderIds.has("wrist_blocker_0")).toBe(true);
      expect(colliderIds.has("wrist_blocker_1")).toBe(true);
      expect(world.snapshot().robot.position).toMatchObject(before);
    } finally {
      world.dispose();
    }
  });

  it("creates a bilateral-contact attachment and physically releases it by opening", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);
    const object = scenario.objects.find((candidate) => candidate.id === "red_block");

    const world = await RapierWorld.create(scenario);
    try {
      const compact = await executeSkill(
        world,
        source("grasp_compact_arm", "set_joint_targets", ["arm"]),
        "set_joint_targets",
        { targets: { shoulder: 1.55, elbow: -2.5, wrist: -0.6 } }
      );
      expect(compact).toMatchObject({ accepted: true, code: "joint_targets_reached" });

      const standoff = Math.max(
        ROBOT_SPEC.base.footprintRadius + object.size.z / 2 + 0.16,
        ROBOT_SPEC.base.halfExtents.z
          + object.size.z / 2
          + ROBOT_SPEC.arm.wristLength
          + ROBOT_SPEC.arm.wristHalfExtents.z
          + 0.05
      );
      const baseTarget = {
        x: object.position.x,
        y: 0,
        z: object.position.z - standoff
      };
      const basePlan = await executeTool(world, "plan_base_path", {
        target: baseTarget,
        face_point: object.position
      });
      expect(basePlan).toMatchObject({ accepted: true, code: "base_path_planned" });
      const basePlanId = detailRecord(basePlan.detail).plan_id;
      expect(typeof basePlanId).toBe("string");
      if (typeof basePlanId !== "string") return;
      const positioned = await executeSkill(
        world,
        source("grasp_base_plan", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: basePlanId, options: { tolerance: 0.015 } }
      );
      expect(positioned).toMatchObject({ accepted: true, code: "base_plan_completed" });

      const opened = await executeSkill(
        world,
        source("grasp_open", "set_gripper_target", ["gripper"]),
        "set_gripper_target",
        { aperture: ROBOT_SPEC.joints.gripper_aperture.maximum, max_force: 1000 }
      );
      expect(opened).toMatchObject({ accepted: true, code: "gripper_target_reached" });

      // The fingers straddle the block laterally, so the tool centre belongs
      // over the block's centre at the height of its top face — not backed off
      // towards the near face, which would leave one finger short of it.
      const endEffectorTarget = {
        x: object.position.x,
        y: object.position.y + object.size.y / 2,
        z: object.position.z
      };
      const armPlan = await executeTool(world, "solve_end_effector_pose", {
        position: endEffectorTarget
      });
      expect(armPlan).toMatchObject({ accepted: true, code: "end_effector_solution" });
      const armPlanId = detailRecord(armPlan.detail).plan_id;
      expect(typeof armPlanId).toBe("string");
      if (typeof armPlanId !== "string") return;
      const reached = await executeSkill(
        world,
        source("grasp_arm_plan", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: armPlanId }
      );
      expect(reached).toMatchObject({ accepted: true, code: "joint_targets_reached" });
      expect(detailRecord(reached.detail).end_effector_verification).toMatchObject({
        position_tolerance: 0.025
      });

      const contactAperture = object.size.x
        + ROBOT_SPEC.gripper.fingerHalfExtents.x * 2
        - 0.01;
      const closed = await executeSkill(
        world,
        source("grasp_close", "set_gripper_target", ["gripper"]),
        "set_gripper_target",
        { aperture: contactAperture, max_force: 1000 }
      );
      expect(closed).toMatchObject({ accepted: true, code: "gripper_target_reached" });
      expect(detailRecord(closed.detail).attachment_stable_frames).toBe(
        ROBOT_SPEC.gripper.minimumStableAttachmentFrames
      );
      const attached = world.snapshot();
      expect(attached.robot.contacts).toMatchObject({
        left_object_id: object.id,
        right_object_id: object.id
      });
      expect(attached.robot.attachment).toMatchObject({
        object_id: object.id,
        source_command_id: "grasp_close"
      });
      const contacts = await executeTool(world, "query_contacts", {});
      expect(contacts).toMatchObject({ accepted: true, code: "contact_state" });
      const pairs = detailRecord(contacts.detail).pairs as Array<Record<string, unknown>>;
      expect(pairs.some((pair) => pair.link_id === "left_finger"
        && pair.collider_id === object.id && Number(pair.force) >= 0)).toBe(true);
      expect(pairs.some((pair) => pair.link_id === "right_finger"
        && pair.collider_id === object.id && Number(pair.force) >= 0)).toBe(true);
      const attachedObject = attached.objects.find((candidate) => candidate.id === object.id);

      const released = await executeSkill(
        world,
        source("grasp_release", "set_gripper_target", ["gripper"]),
        "set_gripper_target",
        { aperture: ROBOT_SPEC.joints.gripper_aperture.maximum, max_force: 1000 }
      );
      expect(released).toMatchObject({ accepted: true, code: "gripper_target_reached" });
      const afterRelease = world.snapshot();
      expect(afterRelease.robot.attachment).toBeNull();
      const releasedObject = afterRelease.objects.find((candidate) => candidate.id === object.id);
      expect(Math.hypot(
        releasedObject.position.x - attachedObject.position.x,
        releasedObject.position.y - attachedObject.position.y,
        releasedObject.position.z - attachedObject.position.z
      )).toBeLessThan(0.5);
      expect(WorldSnapshotSchema.safeParse(afterRelease).success).toBe(true);
    } finally {
      world.dispose();
    }
  });
});

describe("reachable standoff envelope", () => {
  it("offers standoff poses that plan_base_path accepts", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const inspected = await executeTool(world, "inspect_entity", { entity_id: "red_block" });
      expect(inspected).toMatchObject({ accepted: true, code: "entity_state" });
      const poses = detailRecord(inspected.detail).reachable_standoff_poses;
      expect(Array.isArray(poses)).toBe(true);
      if (!Array.isArray(poses) || poses.length === 0) {
        throw new Error("expected at least one reachable standoff pose");
      }

      // Every offered pose must actually plan, or the field is worse than useless.
      for (const pose of poses) {
        const entry = detailRecord(pose);
        const planned = await executeTool(world, "plan_base_path", {
          target: entry.target,
          face_point: entry.face_point
        });
        expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
      }
    } finally {
      world.dispose();
    }
  });

  it("offers a grasp pose the solver accepts from a standoff it also offered", async () => {
    // A live run derived its own grasp point from the block's size and asked for
    // a gripper centre a metre off in z, twice, burning five solver calls on
    // targets no joint angles could reach. Both halves of the answer are things
    // the world measures, so it reports them instead.
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const inspected = await executeTool(world, "inspect_entity", { entity_id: "red_block" });
      const detail = detailRecord(inspected.detail);
      const poses = detail.reachable_standoff_poses;
      const grasp = detailRecord(detail.grasp_pose);
      const position = detailRecord(grasp.position);
      if (!Array.isArray(poses) || poses.length === 0) {
        throw new Error("expected at least one reachable standoff pose");
      }

      // The gripper descends onto the top face, so the grasp point sits above
      // the block's centre and directly over it on the ground plane.
      const block = detailRecord(detail.position);
      const size = detailRecord(detail.size);
      expect(position.x).toBeCloseTo(block.x as number, 5);
      expect(position.z).toBeCloseTo(block.z as number, 5);
      expect(position.y as number).toBeGreaterThan((block.y as number) + (size.y as number) / 2);
      expect(grasp.aperture_to_hold as number)
        .toBeLessThan(grasp.aperture_before_descent as number);

      // Drive to an offered standoff, then the offered grasp pose must solve —
      // the two fields have to agree or neither is worth publishing.
      const standoff = detailRecord(poses[0]);
      const planned = await executeTool(world, "plan_base_path", {
        target: standoff.target,
        face_point: standoff.face_point
      });
      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
      const executed = await executeSkill(
        world,
        source("grasp_standoff_drive", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: detailRecord(planned.detail).plan_id }
      );
      expect(executed).toMatchObject({ accepted: true, code: "base_plan_completed" });

      const solved = await executeTool(world, "solve_end_effector_pose", { position });
      expect(solved).toMatchObject({ accepted: true, code: "end_effector_solution" });

      // The same point with an orientation the arm cannot hold. The position
      // now solves exactly, so only the unnecessary orientation constraint has
      // to go; moving a base that is already parked correctly would not help.
      const overconstrained = await executeTool(world, "solve_end_effector_pose", {
        position,
        orientation: { x: 0, y: 0, z: 0, w: 1 }
      });
      expect(overconstrained).toMatchObject({ accepted: false, code: "ik_residual_too_large" });
      const residual = detailRecord(overconstrained.detail);
      expect(residual.failing_residual).toBe("orientation");
      expect(String(residual.recovery)).toContain("solve_end_effector_position");
      expect(String(residual.recovery)).not.toContain("drive the base closer");
    } finally {
      world.dispose();
    }
  });

  it("answers an unreachable target with alternatives that do plan", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      // The red block's own centre: inside the footprint-eroded navmesh hole.
      const denied = await executeTool(world, "plan_base_path", { target: { x: 2, y: 0, z: 1.5 } });
      expect(denied).toMatchObject({ accepted: false, code: "base_path_unavailable" });
      const alternatives = detailRecord(denied.detail).nearest_reachable_alternatives;
      if (!Array.isArray(alternatives) || alternatives.length === 0) {
        throw new Error("expected nearest_reachable_alternatives for an unreachable target");
      }

      const first = detailRecord(alternatives[0]);
      const planned = await executeTool(world, "plan_base_path", {
        target: first.target,
        face_point: first.face_point
      });
      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
    } finally {
      world.dispose();
    }
  });

  it("explains what a face_point is for when it duplicates the target", async () => {
    // A live run passed the green zone's centre as both target and face_point,
    // which reads as "stand on the zone facing the zone". The denial stated the
    // rule and left the agent to infer the roles of the two fields.
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const zone = scenario.zones.find((candidate) => candidate.id === "green_zone");

      const refused = await executeTool(world, "plan_base_path", {
        target: zone.center,
        face_point: zone.center
      });
      expect(refused).toMatchObject({ accepted: false, code: "invalid_base_face_point" });
      const recovery = String(detailRecord(refused.detail).recovery);
      expect(recovery).toContain("reachable_standoff_pose");

      const verticalOnlyDifference = await executeTool(world, "plan_base_path", {
        target: { ...zone.center, y: 0 },
        face_point: zone.center
      });
      expect(verticalOnlyDifference).toMatchObject({
        accepted: false,
        code: "invalid_base_face_point"
      });
      expect(String(detailRecord(verticalOnlyDifference.detail).error)).toContain("x/z plane");

      // And the field it points at has to answer the question it raises.
      const inspected = await executeTool(world, "inspect_entity", { entity_id: "green_zone" });
      const poses = detailRecord(inspected.detail).reachable_standoff_poses;
      if (!Array.isArray(poses) || poses.length === 0) {
        throw new Error("recovery points at reachable_standoff_poses but none are offered");
      }
      const pose = detailRecord(poses[0]);
      expect(await executeTool(world, "plan_base_path", {
        target: pose.target,
        face_point: pose.face_point
      })).toMatchObject({ accepted: true, code: "base_path_planned" });
    } finally {
      world.dispose();
    }
  });
});

describe("visibility denials", () => {
  it("turns an out-of-view entity into a head aim that actually reveals it", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      // Look away from the block so it leaves the sensor cone.
      const away = await executeSkill(
        world,
        source("head_away", "set_head_target", ["head"]),
        "set_head_target",
        { yaw: -2.6, pitch: 0 }
      );
      expect(away).toMatchObject({ accepted: true, code: "head_target_reached" });

      const blind = await executeTool(world, "inspect_entity", { entity_id: "red_block" });
      expect(blind).toMatchObject({ accepted: false, code: "entity_not_visible" });
      const detail = detailRecord(blind.detail);
      expect(detail.reason).toBe("outside_field_of_view");

      // The denial has to carry numbers the agent can pass straight back.
      const recovery = String(detail.recovery);
      const yaw = Number(/yaw=(-?\d+\.\d+)/.exec(recovery)?.[1]);
      const pitch = Number(/pitch=(-?\d+\.\d+)/.exec(recovery)?.[1]);
      expect(Number.isFinite(yaw)).toBe(true);
      expect(Number.isFinite(pitch)).toBe(true);

      const aimed = await executeSkill(
        world,
        source("head_aim", "set_head_target", ["head"]),
        "set_head_target",
        { yaw, pitch }
      );
      expect(aimed).toMatchObject({ accepted: true, code: "head_target_reached" });

      // One corrected attempt, not a retry loop: the world's own advice works.
      const seen = await executeTool(world, "inspect_entity", { entity_id: "red_block" });
      expect(seen).toMatchObject({ accepted: true, code: "entity_state" });
    } finally {
      world.dispose();
    }
  });

  it("answers an unknown entity id with the ids that do exist", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      // A live run spent seven calls guessing at the robot's own id here.
      const missing = await executeTool(world, "inspect_entity", { entity_id: "robot_base" });
      expect(missing).toMatchObject({ accepted: false, code: "unknown_entity" });
      const detail = detailRecord(missing.detail);
      expect(detail.known_objects).toContain("red_block");
      expect(detail.known_zones).toContain("green_zone");
      expect(String(detail.recovery)).toContain("read_proprioception");

      // Every id it names has to be one this tool actually accepts.
      for (const id of detail.known_objects as string[]) {
        const named = await executeTool(world, "inspect_entity", { entity_id: id });
        expect(named.code).not.toBe("unknown_entity");
      }
    } finally {
      world.dispose();
    }
  });
});

describe("plan lifetime", () => {
  it("keeps a plan valid across observations and only expires it on a body command", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const planned = await executeTool(world, "plan_base_path", { target: { x: 1.2, y: 0, z: 1.6 } });
      expect(planned).toMatchObject({ accepted: true, code: "base_path_planned" });
      const planId = detailRecord(planned.detail).plan_id;
      expect(typeof planId).toBe("string");
      if (typeof planId !== "string") return;

      const plannedRevision = world.snapshot().world_revision;
      for (let index = 0; index < 5; index += 1) {
        await executeTool(world, "sense_scene", {});
        await executeTool(world, "read_proprioception", {});
      }
      // Reading the world must never invalidate a plan.
      expect(world.snapshot().world_revision).toBe(plannedRevision);

      const executed = await executeSkill(
        world,
        source("lifetime_execute", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: planId }
      );
      expect(executed).toMatchObject({ accepted: true, code: "base_plan_completed" });
      // The committed body command is what advances the revision.
      expect(world.snapshot().world_revision).toBeGreaterThan(plannedRevision);
    } finally {
      world.dispose();
    }
  });

  it("distinguishes a consumed plan from one invalidated by a later body command", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const planned = await executeTool(world, "plan_base_path", { target: { x: 1.2, y: 0, z: 1.6 } });
      const planId = detailRecord(planned.detail).plan_id;
      if (typeof planId !== "string") return;
      await executeSkill(
        world,
        source("consume_execute", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: planId }
      );

      const repeated = await executeSkill(
        world,
        source("consume_repeat", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: planId }
      );
      expect(repeated).toMatchObject({ accepted: false, code: "plan_already_consumed" });
      expect(detailRecord(repeated.detail).recovery).toContain("plan_base_path");

      const second = await executeTool(world, "plan_base_path", { target: { x: 2, y: 0, z: 2.4 } });
      expect(second).toMatchObject({ accepted: true, code: "base_path_planned" });
      const secondId = detailRecord(second.detail).plan_id;
      if (typeof secondId !== "string") return;
      const plannedRevision = world.snapshot().world_revision;

      await executeSkill(
        world,
        source("invalidating_head", "set_head_target", ["head"]),
        "set_head_target",
        { yaw: 0.3, pitch: -0.1 }
      );
      const stale = await executeSkill(
        world,
        source("stale_execute", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: secondId }
      );
      expect(stale).toMatchObject({ accepted: false, code: "stale_plan_revision" });
      expect(detailRecord(stale.detail)).toMatchObject({
        plan_id: secondId,
        planned_world_revision: plannedRevision,
        current_world_revision: world.snapshot().world_revision
      });
      expect(detailRecord(stale.detail).recovery).toContain("plan_base_path");

      const unknown = await executeSkill(
        world,
        source("unknown_execute", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: "base_plan_never_created" }
      );
      expect(unknown).toMatchObject({ accepted: false, code: "unknown_base_plan" });
    } finally {
      world.dispose();
    }
  });
});

describe("end-to-end manipulation", () => {
  /**
   * The mission the whole system exists to accomplish, driven through the same
   * tool surface the agents use, with no shortcuts into world internals. It is
   * the only test that proves the scenario is physically achievable at all:
   * every geometric constant it depends on — the standoff envelope, base
   * facing accuracy, the arm's reachable height band, the finger span, the
   * payload lift clearance — has to hold simultaneously for it to pass.
   */
  it("navigates, grasps, carries and releases the red block into the green zone", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("fetch_red_block", 0);

    const world = await RapierWorld.create(scenario);
    try {
      const planId = (result: Awaited<ReturnType<typeof executeTool>>): string => {
        expect(result.accepted).toBe(true);
        const id = detailRecord(result.detail).plan_id;
        expect(typeof id).toBe("string");
        return id as string;
      };

      // 1. The world offers standoff poses; the first one must be drivable.
      const inspected = await executeTool(world, "inspect_entity", { entity_id: "red_block" });
      expect(inspected).toMatchObject({ accepted: true });
      const standoffs = detailRecord(inspected.detail).reachable_standoff_poses as Array<{
        target: { x: number; y: number; z: number };
        face_point: { x: number; y: number; z: number };
      }>;
      expect(standoffs.length).toBeGreaterThan(0);

      const approach = standoffs[0]!;
      const approachPlan = await executeTool(world, "plan_base_path", {
        target: approach.target,
        face_point: approach.face_point
      });
      const drove = await executeSkill(
        world,
        source("approach", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: planId(approachPlan) }
      );
      expect(drove).toMatchObject({ accepted: true, code: "base_plan_completed" });

      const block = world.snapshot().objects.find((object) => object.id === "red_block");

      // 2. Grasp from above: the fingers must straddle the block's upper half,
      //    and must be opened wider than the block before descending onto it.
      expect(await executeSkill(
        world,
        source("open", "set_gripper_target", ["gripper"]),
        "set_gripper_target",
        { aperture: ROBOT_SPEC.joints.gripper_aperture.maximum, max_force: 1000 }
      )).toMatchObject({ accepted: true });

      const graspHeight = block.position.y + block.size.y / 2 + 0.08;
      const grasp = await executeTool(world, "solve_end_effector_pose", {
        position: { x: block.position.x, y: graspHeight, z: block.position.z }
      });
      expect(grasp).toMatchObject({ accepted: true, code: "end_effector_solution" });
      expect(await executeSkill(
        world,
        source("reach", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: planId(grasp) }
      )).toMatchObject({ accepted: true });

      const closed = await executeSkill(
        world,
        source("close", "set_gripper_target", ["gripper"]),
        "set_gripper_target",
        { aperture: Math.min(block.size.x, block.size.z) + 0.06, max_force: 1000 }
      );
      expect(closed).toMatchObject({ accepted: true, code: "gripper_target_reached" });
      expect(detailRecord(closed.detail).attachment_object_id).toBe("red_block");

      // Over-closing on a held object is denied, and the denial must name the
      // width that would have worked instead of leaving the agent to search.
      const crushed = await executeSkill(
        world,
        source("crush", "set_gripper_target", ["gripper"]),
        "set_gripper_target",
        { aperture: ROBOT_SPEC.joints.gripper_aperture.minimum, max_force: 20 }
      );
      expect(crushed).toMatchObject({ accepted: false, code: "gripper_force_limit" });
      expect(detailRecord(crushed.detail)).toMatchObject({
        contacted_object_id: "red_block",
        contacted_object_width: Math.min(block.size.x, block.size.z)
      });
      expect(detailRecord(crushed.detail).recovery).toContain("red_block");

      // 3. Before lifting, the payload is still at floor height, so no base
      //    plan is drivable. A live run burned its whole remaining budget
      //    retrying different targets here, because the denial named the
      //    collision but not the remedy. It must now state the lift.
      const zoneBeforeLift = await executeTool(world, "inspect_entity", { entity_id: "green_zone" });
      const beforeLiftStandoffs = detailRecord(zoneBeforeLift.detail)
        .reachable_standoff_poses as Array<{
          target: { x: number; y: number; z: number };
          face_point: { x: number; y: number; z: number };
        }>;
      expect(beforeLiftStandoffs.length).toBeGreaterThan(0);
      const blocked = await executeTool(world, "plan_base_path", {
        target: beforeLiftStandoffs[0]!.target,
        face_point: beforeLiftStandoffs[0]!.face_point
      });
      expect(blocked).toMatchObject({ accepted: false, code: "base_path_collision" });
      expect(detailRecord(blocked.detail).collision_segments).toContain("attached_payload");
      const carryClearance = detailRecord(blocked.detail).carry_clearance as {
        payload_id: string;
        required_lift: number;
      };
      expect(carryClearance.payload_id).toBe("red_block");
      expect(carryClearance.required_lift).toBeGreaterThan(0);
      expect(detailRecord(blocked.detail).recovery).toContain("solve_end_effector_position");

      // 4. Lift clear of the ground; a payload dragging on the floor blocks
      //    every base plan, which is what made carrying impossible before.
      const lift = await executeTool(world, "solve_end_effector_position", {
        position: { x: block.position.x, y: graspHeight + 0.27, z: block.position.z }
      });
      expect(lift).toMatchObject({ accepted: true });
      expect(await executeSkill(
        world,
        source("lift", "execute_joint_plan", ["arm"]),
        "execute_joint_plan",
        { plan_id: planId(lift) }
      )).toMatchObject({ accepted: true });
      const lifted = world.snapshot().objects.find((object) => object.id === "red_block");
      expect(lifted?.position.y).toBeGreaterThan(block.position.y + 0.2);

      // 5. Carry to the zone using its own advertised standoff.
      const zone = await executeTool(world, "inspect_entity", { entity_id: "green_zone" });
      const zoneStandoffs = detailRecord(zone.detail).reachable_standoff_poses as Array<{
        target: { x: number; y: number; z: number };
        face_point: { x: number; y: number; z: number };
      }>;
      expect(zoneStandoffs.length).toBeGreaterThan(0);
      const dropoff = zoneStandoffs[0]!;
      const carryPlan = await executeTool(world, "plan_base_path", {
        target: dropoff.target,
        face_point: dropoff.face_point
      });
      expect(await executeSkill(
        world,
        source("carry", "execute_base_plan", ["base"]),
        "execute_base_plan",
        { plan_id: planId(carryPlan) }
      )).toMatchObject({ accepted: true, code: "base_plan_completed" });

      // 6. Release and let the block settle under gravity.
      expect(await executeSkill(
        world,
        source("release", "set_gripper_target", ["gripper"]),
        "set_gripper_target",
        { aperture: ROBOT_SPEC.joints.gripper_aperture.maximum, max_force: 1000 }
      )).toMatchObject({ accepted: true });
      for (let step = 0; step < 150; step += 1) await world.observe();

      const final = world.snapshot();
      const goal = checkGoal(scenario.default_goal, final);
      expect(goal.checks.map((check) => [check.name, check.passed])).toEqual([
        ["1:object_in_zone", true],
        ["2:object_attached", true]
      ]);
      expect(goal.success).toBe(true);
    } finally {
      world.dispose();
    }
  }, 120_000);
});
