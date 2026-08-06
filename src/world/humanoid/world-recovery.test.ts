import { describe, expect, it } from "vitest";
import { HoldingMotionGenerator } from "./world-motion-generator.test-support.js";
import {
  humanoidWorldPerceptionTestScenario as perceptionScenario,
  humanoidWorldTestScenario as scenario
} from "./world-scenarios.test-support.js";
import { HumanoidWorld } from "./world.js";

describe("HumanoidWorld recovery and generator contracts", () => {
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
        progress: {
          nextFrameIndex: 10,
          physicalSafety: {
            frame_count: 10,
            first_frame: 1,
            last_frame: 10
          }
        }
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
      expect(recovered.detail.physical_safety).toEqual(complete.detail.physical_safety);
      expect(recovered.finalSnapshot.physicalSafety).toEqual(
        complete.finalSnapshot.physicalSafety
      );
    } finally {
      await resumed?.dispose();
      await interrupted.dispose();
      await uninterrupted.dispose();
    }
  }, 45_000);

  it("restores a navigation intent and revalidates it against the current revision", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const planned = await world.planNavigation({ x: 1.5, y: 0, z: 2.2 });
      expect(planned.accepted, planned.reason).toBe(true);
      const checkpoint = world.checkpoint();
      checkpoint.worldRevision += 1;

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      expect(resumed.snapshot().navigation).toMatchObject({
        planId: planned.planId,
        status: "planned",
        waypointIndex: 1
      });
      expect(resumed.checkpoint().routes).toHaveLength(1);
      const executed = await resumed.executeNavigation(planned.planId);
      expect(executed.accepted, executed.detail.reason).toBe(true);
      expect(executed.detail.revalidation).toMatchObject({
        performed: true,
        accepted: true,
        intent_sha256: planned.intentSha256,
        revalidation_count: 1
      });
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
      expect(planned.motion).toMatchObject({
        protocol: "humanoid-motion-v1",
        generator: "contract_test_generator",
        control_step_seconds: 0.02,
        duration_seconds: 0.1,
        frame_count: 5
      });
      expect(planned.motion?.sha256).toMatch(/^[a-f0-9]{64}$/);
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
    let restored: HumanoidWorld | undefined;
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
        role: "manipulable",
        status: "visible",
        state: "active",
        authority: "mujoco_exact",
        exact: true,
        observable: true,
        observedFrame: observation.frame,
        observedWorldRevision: observation.worldRevision,
        ageRevisions: 0
      });

      await world.advanceStationary();
      const afterPassivePhysics = await world.planWholeBodyMotion(
        contactPlan("after-passive-physics")
      );
      expect(afterPassivePhysics.validation.failures.some((failure) => (
        failure.code === "contact_object_not_currently_visible"
      ))).toBe(false);

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
      restored = await HumanoidWorld.create(perceptionScenario, checkpoint);
      const historicalOnly = await restored.planWholeBodyMotion(
        contactPlan("restored-before-observation")
      );
      expect(historicalOnly.accepted).toBe(false);
      expect(historicalOnly.validation.failures).toContainEqual(expect.objectContaining({
        code: "contact_object_not_currently_visible"
      }));

      const restoredObservation = restored.observe();
      expect(restoredObservation.objectTokens[0]).toMatchObject({
        id: "crate",
        state: "active",
        authority: "mujoco_exact",
        exact: true,
        observable: true
      });
      const observedAgain = await restored.planWholeBodyMotion(
        contactPlan("restored-after-observation")
      );
      expect(observedAgain.validation.failures.some((failure) => (
        failure.code === "contact_object_not_currently_visible"
      ))).toBe(false);
    } finally {
      await Promise.all([world.dispose(), restored?.dispose()]);
    }
  }, 45_000);
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
