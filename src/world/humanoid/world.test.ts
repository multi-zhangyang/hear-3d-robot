import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../../config/load.js";
import { ScenarioSchema } from "../../domain/schema.js";
import { serializeHumanoidReference } from "./motion-artifact.js";
import type {
  HumanoidMotionGenerator,
  HumanoidMotionGeneratorInput
} from "./motion-plan.js";
import { HumanoidWorld } from "./world.js";

const scenario = ScenarioSchema.parse({
  title: "Humanoid field",
  seed: 7,
  bounds: { width: 10, depth: 10 },
  visibility_radius: 6,
  robot: { x: 1.5, z: 1.5, yaw: 0 },
  obstacles: [{
    id: "column",
    center: { x: 6, y: 1, z: 6 },
    size: { x: 1, y: 2, z: 1 }
  }],
  objects: [],
  zones: [],
  default_goal: {
    summary: "到达开放区域",
    predicates: [{
      type: "robot_at",
      target: { x: 1.5, y: 0, z: 2.2 },
      tolerance: 0.25
    }]
  }
});

const perceptionScenario = ScenarioSchema.parse({
  ...scenario,
  title: "Humanoid perception field",
  obstacles: [],
  objects: [{
    id: "crate",
    kind: "crate",
    color: "#8b6b45",
    position: { x: 1.5, y: 0.15, z: 3.2 },
    size: { x: 0.3, y: 0.3, z: 0.3 },
    portable: true
  }]
});

describe("HumanoidWorld", () => {
  it("executes long routes as physically previewed closed-loop chunks", async () => {
    const catalog = await loadRuntimeCatalog();
    const generated = catalog.materialize("humanoid_frontier", 0);
    const target = generated.zones.find((zone) => zone.id === "frontier_beacon")!.center;
    const world = await HumanoidWorld.create(generated);
    try {
      const first = await world.planNavigation(target);
      expect(first.accepted, first.reason).toBe(true);
      expect(first.distance).toBeCloseTo(3, 6);
      expect(first.remainingDistance).toBeGreaterThan(3);
      expect(first.chunkTarget).not.toEqual(target);

      const executed = await world.executeNavigation(first.planId);
      expect(executed.accepted, executed.detail.reason).toBe(true);
      expect(executed.detail.travelledDistance).toBeGreaterThan(2.5);
      expect(executed.finalSnapshot.robot.fallen).toBe(false);

      const second = await world.planNavigation(target);
      expect(second.accepted, second.reason).toBe(true);
      expect(second.remainingDistance).toBeLessThan(first.remainingDistance);
    } finally {
      await world.dispose();
    }
  }, 45_000);

  it("plans and executes model-authored full-body motion and physical navigation", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const initial = world.snapshot();
      expect(initial.motionGenerator).toEqual({
        protocol: "humanoid-motion-generator-v1",
        implementation: "task_space_constraints",
        motionClass: "constraint_solver",
        sampling: "deterministic"
      });
      const motion = await world.planWholeBodyMotion({
        id: "look-and-balance",
        intent: "保持站立并改变双臂姿态",
        duration_seconds: 1,
        keyframes: [
          { at_seconds: 0 },
          {
            at_seconds: 1,
            torso_yaw: 0.05,
            left_hand: {
              frame: "world",
              position: {
                ...initial.robot.links.left_wrist_yaw_link.position,
                y: initial.robot.links.left_wrist_yaw_link.position.y + 0.01
              }
            },
            right_hand: {
              frame: "world",
              position: {
                ...initial.robot.links.right_wrist_yaw_link.position,
                y: initial.robot.links.right_wrist_yaw_link.position.y + 0.01
              }
            }
          }
        ]
      });
      expect(motion.accepted).toBe(true);
      const afterMotionPreview = world.snapshot();
      expect(afterMotionPreview.frame).toBe(initial.frame);
      expect(afterMotionPreview.worldRevision).toBe(initial.worldRevision);
      expect(afterMotionPreview.robot.simulatedTime).toBe(initial.robot.simulatedTime);
      expect(afterMotionPreview.robot.rootPosition.x).toBeCloseTo(initial.robot.rootPosition.x, 4);
      expect(afterMotionPreview.robot.rootPosition.z).toBeCloseTo(initial.robot.rootPosition.z, 4);

      const motionResult = await world.executeWholeBodyMotion(motion.planId);
      expect(motionResult.accepted).toBe(true);
      expect(motionResult.code).toBe("motion_completed");
      expect(motionResult.finalSnapshot.robot.fallen).toBe(false);
      expect(motionResult.finalSnapshot.worldRevision).toBeGreaterThan(0);

      const beforeNavigation = world.snapshot();
      const target = {
        x: beforeNavigation.robot.rootPosition.x,
        y: 0,
        z: beforeNavigation.robot.rootPosition.z + 0.65
      };
      const route = await world.planNavigation(target);
      expect(route.accepted, route.reason).toBe(true);
      const afterRoutePreview = world.snapshot();
      expect(afterRoutePreview.frame).toBe(beforeNavigation.frame);
      expect(afterRoutePreview.worldRevision).toBe(beforeNavigation.worldRevision);
      expect(afterRoutePreview.robot.simulatedTime).toBe(beforeNavigation.robot.simulatedTime);
      expect(afterRoutePreview.robot.rootPosition.x).toBeCloseTo(
        beforeNavigation.robot.rootPosition.x,
        4
      );
      expect(afterRoutePreview.robot.rootPosition.z).toBeCloseTo(
        beforeNavigation.robot.rootPosition.z,
        4
      );

      const navigationResult = await world.executeNavigation(route.planId);
      expect(navigationResult.accepted).toBe(true);
      expect(navigationResult.code).toBe("navigation_completed");
      expect(navigationResult.detail.travelledDistance).toBeGreaterThan(0.3);
      expect(navigationResult.finalSnapshot.robot.fallen).toBe(false);
      expect(navigationResult.finalSnapshot.robot.balance.support).not.toBe("none");

      const checkpoint = world.checkpoint();
      resumed = await HumanoidWorld.create(scenario, checkpoint);
      const restored = resumed.snapshot();
      expect(restored.frame).toBe(navigationResult.finalSnapshot.frame);
      expect(restored.worldRevision).toBe(navigationResult.finalSnapshot.worldRevision);
      expect(restored.robot.simulatedTime).toBe(
        navigationResult.finalSnapshot.robot.simulatedTime
      );
      expect(restored.robot.rootPosition.x).toBeCloseTo(
        navigationResult.finalSnapshot.robot.rootPosition.x,
        5
      );
      expect(restored.robot.rootPosition.z).toBeCloseTo(
        navigationResult.finalSnapshot.robot.rootPosition.z,
        5
      );
      expect(restored.robot.fallen).toBe(false);
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it("persists one validated motion artifact and executes that exact artifact after resume", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const planned = await world.planWholeBodyMotion({
        id: "resumable-motion",
        intent: "保持平衡并轻微转动躯干",
        duration_seconds: 0.4,
        keyframes: [
          { at_seconds: 0 },
          { at_seconds: 0.4, torso_yaw: 0.04 }
        ]
      });
      expect(planned.accepted).toBe(true);
      expect(planned.motion).toMatchObject({
        protocol: "humanoid-motion-v1",
        generator: "task_space_constraints",
        frame_count: 20
      });

      const checkpoint = world.checkpoint();
      const artifact = checkpoint.motions[0]!.artifact;
      expect(artifact.frames).toHaveLength(planned.motion!.frame_count);
      expect(artifact.frames.at(-1)?.atSeconds).toBe(artifact.durationSeconds);

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      const executed = await resumed.executeWholeBodyMotion(planned.planId);
      expect(executed.accepted).toBe(true);
      expect(executed.frames).toBe(artifact.frames.length);
      expect(executed.detail.motion).toEqual(planned.motion);
      expect(executed.finalSnapshot.robot.fallen).toBe(false);
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("resumes an interrupted motion from the exact next artifact frame", async () => {
    const uninterrupted = await HumanoidWorld.create(scenario);
    const interrupted = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const plan = {
        id: "mid-motion-resume",
        intent: "保持平衡并连续改变躯干姿态",
        duration_seconds: 0.4,
        keyframes: [
          { at_seconds: 0 },
          { at_seconds: 0.4, torso_yaw: 0.04 }
        ]
      };
      const completePlan = await uninterrupted.planWholeBodyMotion(plan);
      const interruptedPlan = await interrupted.planWholeBodyMotion(plan);
      expect(completePlan.accepted).toBe(true);
      expect(interruptedPlan.accepted).toBe(true);

      const complete = await uninterrupted.executeWholeBodyMotion(completePlan.planId);
      let recoveryCheckpoint;
      await expect(interrupted.executeWholeBodyMotion(
        interruptedPlan.planId,
        () => {
          const checkpoint = interrupted.checkpoint();
          if (checkpoint.worldRevision === 10) {
            recoveryCheckpoint = checkpoint;
            throw new Error("simulated process interruption");
          }
        }
      )).rejects.toThrow("simulated process interruption");

      expect(recoveryCheckpoint).toBeDefined();
      expect(recoveryCheckpoint!.motions[0]).toMatchObject({
        createdRevision: 0,
        progress: { nextFrameIndex: 10 }
      });
      resumed = await HumanoidWorld.create(scenario, recoveryCheckpoint);
      const recovered = await resumed.executeWholeBodyMotion(interruptedPlan.planId);
      expect(recovered.accepted).toBe(true);
      expect(recovered.frames).toBe(10);
      expect(recovered.finalSnapshot.frame).toBe(complete.finalSnapshot.frame);
      expect(recovered.finalSnapshot.worldRevision).toBe(complete.finalSnapshot.worldRevision);
      expect(recovered.finalSnapshot.robot.simulatedTime).toBeCloseTo(
        complete.finalSnapshot.robot.simulatedTime,
        10
      );
      expect(recovered.finalSnapshot.robot.rootPosition).toEqual(
        complete.finalSnapshot.robot.rootPosition
      );
      expect(recovered.finalSnapshot.robot.joints).toEqual(
        complete.finalSnapshot.robot.joints
      );
    } finally {
      await resumed?.dispose();
      await interrupted.dispose();
      await uninterrupted.dispose();
    }
  }, 45_000);

  it("blocks restored navigation state when its route is no longer restorable", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const planned = await world.planNavigation({ x: 1.5, y: 0, z: 2.2 });
      expect(planned.accepted, planned.reason).toBe(true);
      const checkpoint = world.checkpoint();
      checkpoint.worldRevision += 1;

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      expect(resumed.snapshot().navigation).toMatchObject({
        planId: null,
        status: "blocked",
        waypointIndex: null
      });
      expect(resumed.checkpoint().routes).toEqual([]);
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("executes an injected motion generator only through the validated artifact protocol", async () => {
    const generator = new HoldingMotionGenerator();
    const world = await HumanoidWorld.create(scenario, undefined, {
      motionGeneratorFactory: async () => generator
    });
    try {
      const planned = await world.planWholeBodyMotion({
        id: "injected-generator-motion",
        intent: "保持当前全身姿态",
        duration_seconds: 0.1,
        keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
      });
      expect(generator.calls).toBe(1);
      expect(planned.accepted).toBe(true);
      expect(planned.motion).toEqual({
        protocol: "humanoid-motion-v1",
        generator: "contract_test_generator",
        control_step_seconds: 0.02,
        duration_seconds: 0.1,
        frame_count: 5
      });
      const persistedArtifact = world.checkpoint().motions[0]!.artifact;
      const executed = await world.executeWholeBodyMotion(planned.planId);
      expect(executed.accepted).toBe(true);
      expect(executed.frames).toBe(persistedArtifact.frames.length);
      expect(world.checkpoint().motions).toEqual([]);
    } finally {
      await world.dispose();
    }
    expect(generator.disposed).toBe(true);
  }, 30_000);

  it("rejects a generator artifact whose cadence disagrees with the controller", async () => {
    const generator = new HoldingMotionGenerator(true);
    const world = await HumanoidWorld.create(scenario, undefined, {
      motionGeneratorFactory: async () => generator
    });
    try {
      const planned = await world.planWholeBodyMotion({
        id: "invalid-generator-cadence",
        intent: "验证运动制品时序",
        duration_seconds: 0.1,
        keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
      });
      expect(planned.accepted).toBe(false);
      expect(planned.motion).toBeNull();
      expect(planned.validation.failures).toContainEqual(expect.objectContaining({
        code: "invalid_reference",
        message: "Humanoid motion artifact frame cadence mismatch"
      }));
      expect(world.checkpoint().motions).toEqual([]);
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("refuses to resume with a different motion generator contract", async () => {
    const generator = new HoldingMotionGenerator();
    const world = await HumanoidWorld.create(scenario, undefined, {
      motionGeneratorFactory: async () => generator
    });
    const checkpoint = world.checkpoint();
    await world.dispose();

    await expect(HumanoidWorld.create(scenario, checkpoint)).rejects.toThrow(
      "Humanoid motion generator mismatch"
    );
  }, 30_000);

  it("persists sensor-grounded object tokens and rejects contact plans before observation", async () => {
    const world = await HumanoidWorld.create(perceptionScenario);
    try {
      const unobserved = await world.planWholeBodyMotion(contactPlan("before-observation"));
      expect(unobserved.accepted).toBe(false);
      expect(unobserved.validation.failures).toContainEqual(expect.objectContaining({
        code: "contact_object_not_currently_visible"
      }));

      const observation = world.observe();
      expect(observation.objectTokens).toHaveLength(1);
      expect(observation.objectTokens[0]).toMatchObject({
        id: "crate",
        status: "visible",
        ageRevisions: 0
      });

      const grounded = await world.planWholeBodyMotion(contactPlan("after-observation"));
      expect(grounded.accepted).toBe(false);
      expect(grounded.validation.failures).toContainEqual(expect.objectContaining({
        code: "required_contact_missing"
      }));
      expect(grounded.validation.failures.some((failure) => (
        failure.code === "contact_object_not_currently_visible"
      ))).toBe(false);

      const checkpoint = world.checkpoint();
      expect(checkpoint.objectMemory.records).toHaveLength(1);
      expect(checkpoint.objectMemory.records[0]).toMatchObject({
        id: "crate",
        lastSeenRevision: observation.worldRevision
      });
    } finally {
      await world.dispose();
    }
  }, 30_000);
});

function contactPlan(id: string) {
  return {
    id,
    intent: "保持站立并验证左手与箱体接触",
    duration_seconds: 0.1,
    contact_constraints: [{
      body: "left_wrist_yaw_link" as const,
      object_id: "crate",
      required: true
    }],
    keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
  };
}

class HoldingMotionGenerator implements HumanoidMotionGenerator {
  readonly descriptor = {
    protocol: "humanoid-motion-generator-v1" as const,
    implementation: "contract_test_generator",
    motionClass: "constraint_solver" as const,
    sampling: "deterministic" as const
  };
  calls = 0;
  disposed = false;

  constructor(private readonly invalidCadence = false) {}

  async generate(input: HumanoidMotionGeneratorInput) {
    this.calls += 1;
    const count = Math.ceil(input.plan.duration_seconds / input.controlStepSeconds);
    return {
      version: 1 as const,
      protocol: "humanoid-motion-v1" as const,
      generator: this.descriptor.implementation,
      controlStepSeconds: input.controlStepSeconds,
      durationSeconds: input.plan.duration_seconds,
      frames: Array.from({ length: count }, (_, index) => ({
        atSeconds: this.invalidCadence && index === 0
          ? input.controlStepSeconds / 2
          : Math.min(
              (index + 1) * input.controlStepSeconds,
              input.plan.duration_seconds
            ),
        reference: serializeHumanoidReference(input.baseline)
      }))
    };
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}
