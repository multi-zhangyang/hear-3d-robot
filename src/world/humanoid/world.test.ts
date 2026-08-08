import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../../config/load.js";
import { applyScenarioChunkDeltaMutation } from "../../domain/scenario-chunk-delta.js";
import { createScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
import { humanoidMotionArtifactSha256 } from "./motion-artifact.js";
import type { HumanoidMotionCandidateBatch } from "./motion-plan.js";
import {
  createHumanoidMotionOptionMonitorState,
  humanoidMotionOptionContractSha256
} from "./motion-option.js";
import { humanoidMotionRolloutSha256 } from "./motion-rollout.js";
import { humanoidMotionIntentSha256 } from "./plan-lifecycle.js";
import { humanoidMotionContactEvidenceSha256 } from "./motion-contact-evidence.js";
import { BlockingFirstMotionGenerator } from "./world-motion-generator.test-support.js";
import {
  humanoidWorldPerceptionTestScenario as perceptionScenario,
  humanoidWorldTestScenario as scenario
} from "./world-scenarios.test-support.js";
import { HumanoidWorld } from "./world.js";
import { YahmpController } from "./yahmp-controller.js";

describe("HumanoidWorld", () => {
  it("rebuilds live collision and navigation resources from chunk geometry", async () => {
    let controllerInstances = 0;
    const world = await HumanoidWorld.create(scenario, undefined, {
      controllerFactory: async () => {
        controllerInstances += 1;
        return YahmpController.create();
      }
    });
    try {
      expect(controllerInstances).toBe(2);
      const target = { x: 1.5, y: 0, z: 2.2 };
      const plannedBefore = await world.planNavigation(target);
      expect(plannedBefore.accepted, plannedBefore.reason).toBe(true);
      const before = world.snapshot();
      const chunks = applyScenarioChunkDeltaMutation(
        scenario,
        createScenarioChunkDeltaState(scenario),
        {
          type: "create_block",
          block: {
            id: "live-target-block",
            center: { x: target.x, y: 0.5, z: target.z },
            size: { x: 0.9, y: 1, z: 0.9 },
            material: "stone",
            properties: {}
          }
        }
      );

      const synchronized = await world.synchronizeScenarioChunks(scenario, chunks);
      expect(controllerInstances).toBe(4);
      expect(synchronized).toMatchObject({
        changed: true,
        chunkRevision: 1,
        resourceRebuilt: true,
        changedDomains: ["geometry"],
        invalidatedPlanIds: [plannedBefore.planId]
      });
      const after = world.snapshot();
      expect(after.frame).toBe(before.frame + 1);
      expect(after.worldRevision).toBe(before.worldRevision + 1);
      expect(after.robot.rootPosition).toEqual(before.robot.rootPosition);
      expect(world.checkpoint().routes).toEqual([]);

      const plannedAfter = await world.planNavigation(target);
      expect(plannedAfter.accepted).toBe(false);
      expect(plannedAfter.reason).toMatch(/projection|path|mesh/i);
      if (/projection/i.test(plannedAfter.reason ?? "")) {
        expect(plannedAfter.chunkTarget).not.toEqual(after.robot.rootPosition);
        expect(plannedAfter.chunkTarget).not.toEqual(target);
      }
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("persists navigation carry authority and committed continuation evidence", async () => {
    const world = await HumanoidWorld.create(scenario);
    let restored: HumanoidWorld | undefined;
    try {
      const planned = await world.planNavigation({ x: 1.5, y: 0, z: 2.2 });
      expect(planned.accepted, planned.reason).toBe(true);
      const plannedRoute = world.checkpoint().routes.find(({ id }) => (
        id === planned.planId
      ));
      expect(plannedRoute?.carriedObjectBindings).toMatchObject({
        protocol: "humanoid-carried-object-binding-set-v1",
        bindings: []
      });
      expect(plannedRoute?.carriedObjectContinuation).toBeNull();

      const executed = await world.executeNavigation(
        planned.planId,
        undefined,
        { retainTerminal: true }
      );
      expect(executed.accepted, executed.detail.reason).toBe(true);
      expect(executed.detail.carry).toMatchObject({
        binding_set: { bindings: [] },
        continuation: { continued: true, bindings: [] },
        unauthorized_contacts: []
      });
      const checkpoint = world.checkpoint();
      const terminalRoute = checkpoint.routes.find(({ id }) => id === planned.planId);
      expect(terminalRoute?.carriedObjectContinuation).toEqual(
        executed.detail.carry?.continuation
      );

      restored = await HumanoidWorld.create(scenario, checkpoint);
      expect(restored.checkpoint().routes[0]?.carriedObjectBindings).toEqual(
        terminalRoute?.carriedObjectBindings
      );
      expect(restored.checkpoint().routes[0]?.carriedObjectContinuation).toEqual(
        terminalRoute?.carriedObjectContinuation
      );

      const tampered = structuredClone(checkpoint);
      tampered.routes[0]!.carriedObjectContinuation!.binding_set_sha256 = "0".repeat(64);
      await expect(HumanoidWorld.create(scenario, tampered)).rejects.toThrow(
        /carry continuation|authority binding/i
      );
    } finally {
      await restored?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("publishes and restores frame-aligned grasp authority state", async () => {
    const world = await HumanoidWorld.create(perceptionScenario);
    let restored: HumanoidWorld | undefined;
    try {
      const initial = world.snapshot();
      const observed = world.observe();
      expect(initial.frame).toBe(0);
      expect(initial.grasp.contractSha256).toMatch(/^[a-f0-9]{64}$/);
      expect(initial.grasp.assessments.map(({ object_id, hand, frame }) => (
        [object_id, hand, frame]
      ))).toEqual([
        ["crate", "left", 0],
        ["crate", "right", 0]
      ]);
      expect(observed.grasp).toEqual(initial.grasp);
      expect(world.checkpoint().graspRegistry.last_frame).toBe(initial.frame);

      const advanced = await world.advanceStationary();
      expect(advanced?.grasp.assessments.every((assessment) => (
        assessment.frame === advanced.frame
      ))).toBe(true);
      const checkpoint = world.checkpoint();
      expect(checkpoint.graspRegistry.last_frame).toBe(checkpoint.frame);

      restored = await HumanoidWorld.create(perceptionScenario, checkpoint);
      expect(restored.snapshot().grasp).toEqual(advanced?.grasp);
      expect(restored.checkpoint().graspRegistry).toEqual(
        checkpoint.graspRegistry
      );

      const wrongFrame = structuredClone(checkpoint);
      wrongFrame.graspRegistry.last_frame = checkpoint.frame + 1;
      await expect(HumanoidWorld.create(perceptionScenario, wrongFrame))
        .rejects.toThrow(/registry frame|registry assessment/);
    } finally {
      await restored?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("rejects untrusted grasp contracts before rollout and isolates preview tracking", async () => {
    const world = await HumanoidWorld.create(perceptionScenario);
    try {
      world.observe();
      const before = world.checkpoint().graspRegistry;
      await expect(world.planWholeBodyMotionCandidates(
        graspCandidateBatch("0".repeat(64))
      )).rejects.toThrow("contract hash does not match authority");
      expect(world.checkpoint().graspRegistry).toEqual(before);

      const planned = await world.planWholeBodyMotionCandidates(
        graspCandidateBatch(world.snapshot().grasp.contractSha256)
      );
      expect(planned.accepted).toBe(false);
      expect(world.checkpoint().graspRegistry).toEqual(before);
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("keeps authoritative physics live while a plan waits and revalidates its intent", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const initial = world.snapshot();
      const streamed: number[] = [];
      const advanced = await world.advanceStationary((frame) => {
        streamed.push(frame.frame);
      });

      expect(advanced).not.toBeNull();
      expect(advanced!.frame).toBe(initial.frame + 1);
      expect(advanced!.worldRevision).toBe(initial.worldRevision + 1);
      expect(advanced!.robot.simulatedTime - initial.robot.simulatedTime).toBeCloseTo(
        initial.robot.controller.controlStepSeconds,
        9
      );
      expect(advanced!.robot.fallen).toBe(false);
      expect(streamed).toEqual([advanced!.frame]);

      const plan = await world.planNavigation({ x: 1.5, y: 0, z: 2.2 });
      expect(plan.accepted, plan.reason).toBe(true);
      const planned = world.snapshot();
      const live = await world.advanceStationary();
      expect(live).not.toBeNull();
      expect(live!.frame).toBe(planned.frame + 1);
      expect(live!.worldRevision).toBe(planned.worldRevision + 1);
      expect(world.consumablePlanIds()).toContain(plan.planId);

      const executed = await world.executeNavigation(plan.planId);
      expect(executed.accepted, executed.detail.reason).toBe(true);
      expect(executed.detail.revalidation).toMatchObject({
        performed: true,
        accepted: true,
        intent_sha256: plan.intentSha256,
        planning_revision: plan.createdRevision,
        previous_validation_revision: plan.createdRevision,
        validation_revision: live!.worldRevision,
        revalidation_count: 1
      });
      expect(world.checkpoint().graspRegistry.last_frame).toBe(
        executed.finalSnapshot.frame
      );
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("advances authority while an isolated motion rollout is still planning", async () => {
    const generator = new BlockingFirstMotionGenerator();
    const world = await HumanoidWorld.create(scenario, undefined, {
      motionGeneratorFactory: async () => generator
    });
    try {
      const planning = world.planWholeBodyMotion({
        id: "slow-isolated-rollout",
        intent: "保持平衡并验证规划期间物理继续推进",
        duration_seconds: 0.1,
        keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
      });
      await generator.firstCallEntered;
      const before = world.snapshot();
      const first = await world.advanceStationary();
      const second = await world.advanceStationary();
      expect(first?.worldRevision).toBe(before.worldRevision + 1);
      expect(second?.worldRevision).toBe(before.worldRevision + 2);

      generator.releaseFirstCall();
      const planned = await planning;
      expect(planned.accepted).toBe(true);
      expect(planned.createdRevision).toBe(before.worldRevision);
      expect(world.snapshot().worldRevision).toBeGreaterThan(planned.createdRevision);

      const executed = await world.executeWholeBodyMotion(planned.planId);
      expect(executed.accepted).toBe(true);
      expect(executed.detail.revalidation).toMatchObject({
        performed: true,
        accepted: true,
        intent_sha256: planned.intentSha256,
        revalidation_count: 1
      });
    } finally {
      generator.releaseFirstCall();
      await world.dispose();
    }
  }, 30_000);

  it("retains and replays a durable motion terminal until its hash is acknowledged", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const planned = await world.planWholeBodyMotion({
        id: "durable-motion-terminal",
        intent: "保持平衡并完成一次短全身姿态调整",
        duration_seconds: 0.1,
        keyframes: [
          { at_seconds: 0 },
          { at_seconds: 0.1, torso_yaw: 0.01 }
        ]
      });
      expect(planned.accepted).toBe(true);
      const cuts: number[] = [];
      const executed = await world.executeWholeBodyMotion(
        planned.planId,
        undefined,
        {
          retainTerminal: true,
          persistenceSink: (cut) => {
            expect(cut.authority.revision).toBe(cut.world.worldRevision);
            expect(cut.worldCheckpoint.worldRevision).toBe(cut.world.worldRevision);
            cuts.push(cut.world.worldRevision);
          }
        }
      );
      expect(executed.terminalResultSha256).toMatch(/^[a-f0-9]{64}$/);
      expect(cuts).toHaveLength(executed.frames);
      const checkpoint = world.checkpoint();
      expect(checkpoint.motions[0]!.terminal).toMatchObject({
        plan_id: planned.planId,
        total_frames: executed.frames,
        final_frame: executed.finalSnapshot.frame,
        final_world_revision: executed.finalSnapshot.worldRevision,
        result_sha256: executed.terminalResultSha256
      });

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      const beforeReplay = resumed.snapshot();
      const replayed = await resumed.executeWholeBodyMotion(
        planned.planId,
        undefined,
        { retainTerminal: true }
      );
      expect(replayed).toMatchObject({
        accepted: executed.accepted,
        code: executed.code,
        frames: 0,
        terminalResultSha256: executed.terminalResultSha256
      });
      expect(resumed.snapshot()).toEqual(beforeReplay);
      await expect(resumed.acknowledgeWholeBodyMotion(
        planned.planId,
        "0".repeat(64)
      )).rejects.toThrow("acknowledgement mismatch");
      expect(await resumed.acknowledgeWholeBodyMotion(
        planned.planId,
        executed.terminalResultSha256!
      )).toBe(true);
      expect(resumed.checkpoint().motions).toEqual([]);
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("resumes navigation after its last durable physical frame", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const start = world.snapshot().robot.rootPosition;
      const planned = await world.planNavigation({
        x: start.x,
        y: 0,
        z: start.z + 0.8
      });
      expect(planned.accepted, planned.reason).toBe(true);
      let recoveryCheckpoint: ReturnType<HumanoidWorld["checkpoint"]> | undefined;
      let committedCuts = 0;
      await expect(world.executeNavigation(
        planned.planId,
        undefined,
        {
          retainTerminal: true,
          persistenceSink: (cut) => {
            committedCuts += 1;
            if (committedCuts !== 5) return;
            recoveryCheckpoint = structuredClone(cut.worldCheckpoint);
            throw new Error("simulated_navigation_process_loss");
          }
        }
      )).rejects.toThrow("simulated_navigation_process_loss");
      expect(recoveryCheckpoint).toBeDefined();
      const durableProgress = recoveryCheckpoint!.routes[0]!.progress!;
      expect(durableProgress.committed_frame_count).toBe(5);

      resumed = await HumanoidWorld.create(scenario, recoveryCheckpoint);
      expect(resumed.snapshot().navigation).toMatchObject({
        planId: planned.planId,
        status: "planned",
        waypointIndex: durableProgress.waypoint_index
      });
      const recoveryCuts: number[] = [];
      const executed = await resumed.executeNavigation(
        planned.planId,
        undefined,
        {
          retainTerminal: true,
          persistenceSink: (cut) => {
            recoveryCuts.push(cut.world.worldRevision);
          }
        }
      );
      expect(executed.accepted, executed.detail.reason).toBe(true);
      expect(recoveryCuts[0]).toBe(recoveryCheckpoint!.worldRevision + 1);
      expect(executed.frames).toBe(
        durableProgress.committed_frame_count
          + executed.finalSnapshot.worldRevision
          - recoveryCheckpoint!.worldRevision
      );
      expect(executed.terminalResultSha256).toMatch(/^[a-f0-9]{64}$/);
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it("replays a durable navigation terminal without stepping physics", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const start = world.snapshot().robot.rootPosition;
      const planned = await world.planNavigation({
        x: start.x,
        y: 0,
        z: start.z + 0.55
      });
      expect(planned.accepted, planned.reason).toBe(true);
      const executed = await world.executeNavigation(
        planned.planId,
        undefined,
        { retainTerminal: true, persistenceSink: () => undefined }
      );
      expect(executed.accepted, executed.detail.reason).toBe(true);
      expect(executed.terminalResultSha256).toMatch(/^[a-f0-9]{64}$/);
      const checkpoint = world.checkpoint();
      expect(checkpoint.routes[0]!.terminal).toMatchObject({
        plan_id: planned.planId,
        total_frames: executed.frames,
        final_frame: executed.finalSnapshot.frame,
        final_world_revision: executed.finalSnapshot.worldRevision,
        result_sha256: executed.terminalResultSha256
      });

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      const beforeReplay = resumed.snapshot();
      const replayed = await resumed.executeNavigation(
        planned.planId,
        undefined,
        { retainTerminal: true }
      );
      expect(replayed).toMatchObject({
        accepted: executed.accepted,
        code: executed.code,
        frames: 0,
        terminalResultSha256: executed.terminalResultSha256
      });
      expect(resumed.snapshot()).toEqual(beforeReplay);
      await expect(resumed.acknowledgeNavigation(
        planned.planId,
        "0".repeat(64)
      )).rejects.toThrow("acknowledgement mismatch");
      expect(await resumed.acknowledgeNavigation(
        planned.planId,
        executed.terminalResultSha256!
      )).toBe(true);
      expect(resumed.checkpoint().routes).toEqual([]);
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it("persists a rejected navigation terminal after admission state changes", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const start = world.snapshot().robot.rootPosition;
      const planned = await world.planNavigation({
        x: start.x,
        y: 0,
        z: start.z + 0.55
      });
      expect(planned.accepted, planned.reason).toBe(true);
      await world.advanceStationary();
      const execution = world.executeNavigation(
        planned.planId,
        undefined,
        { retainTerminal: true }
      );
      await Promise.all(Array.from({ length: 4 }, () => world.advanceStationary()));
      const rejected = await execution;
      expect(rejected).toMatchObject({
        accepted: false,
        code: "plan_stale",
        frames: 0
      });
      expect(() => world.checkpoint()).not.toThrow();
      expect(world.checkpoint().routes[0]!.terminal).toMatchObject({
        plan_id: planned.planId,
        code: "plan_stale",
        total_frames: 0,
        final_world_revision: rejected.finalSnapshot.worldRevision
      });
    } finally {
      await world.dispose();
    }
  }, 30_000);

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
              },
              tolerance_m: 0.045
            },
            right_hand: {
              frame: "world",
              position: {
                ...initial.robot.links.right_wrist_yaw_link.position,
                y: initial.robot.links.right_wrist_yaw_link.position.y + 0.01
              },
              tolerance_m: 0.045
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
      expect(motionResult.detail.physical_safety).toMatchObject({
        protocol: "humanoid-physical-safety-evidence-v1",
        frame_count: motion.motion!.frame_count,
        first_frame: 1,
        last_frame: motion.motion!.frame_count
      });
      expect(motionResult.finalSnapshot.physicalSafety).toEqual({
        planId: motion.planId,
        evidence: motionResult.detail.physical_safety
      });

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
      expect(restored.physicalSafety).toEqual(motionResult.finalSnapshot.physicalSafety);
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

  it("prunes old-revision orphan motions before checkpoint recovery", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const orphan = await world.planWholeBodyMotion({
        id: "old-revision-orphan",
        intent: "保持平衡并轻微转动躯干",
        duration_seconds: 0.4,
        keyframes: [
          { at_seconds: 0 },
          { at_seconds: 0.4, torso_yaw: 0.02 }
        ]
      });
      const selected = await world.planWholeBodyMotion({
        id: "selected-current-motion",
        intent: "保持平衡并完成当前躯干动作",
        duration_seconds: 0.4,
        keyframes: [
          { at_seconds: 0 },
          { at_seconds: 0.4, torso_yaw: 0.04 }
        ]
      });
      expect(orphan.accepted).toBe(true);
      expect(selected.accepted).toBe(true);
      expect(world.consumablePlanIds()).toEqual([
        orphan.planId,
        selected.planId
      ]);

      const executed = await world.executeWholeBodyMotion(selected.planId);
      expect(executed.accepted).toBe(true);
      expect(world.consumablePlanIds()).toEqual([]);
      const checkpoint = world.checkpoint();
      expect(checkpoint.motions).toEqual([]);

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      expect(resumed.consumablePlanIds()).toEqual([]);
      expect(resumed.checkpoint().motions).toEqual([]);
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("selects the first physically feasible model-ranked whole-body candidate", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const before = world.snapshot();
      const planned = await world.planWholeBodyMotionCandidates(
        forwardCandidateBatch(before, "selection")
      );

      expect(planned.accepted, JSON.stringify(planned.candidates)).toBe(true);
      expect(planned).toMatchObject({
        accepted: true,
        planId: "selection-forward",
        selectedCandidateId: "selection-forward",
        selectedRank: 2,
        selection: "model_rank_then_physics"
      });
      expect(planned.candidates).toHaveLength(2);
      expect(planned.candidates[0]!.validation).toMatchObject({
        feasible: false,
        failures: [expect.objectContaining({ code: "motion_goal_unmet" })]
      });
      expect(planned.candidates[1]!.validation.feasible).toBe(true);
      expect(planned.option?.certificate.artifact_sha256).toBe(planned.motion?.sha256);
      expect(planned.option?.certificate.physical_safety).toMatchObject({
        protocol: "humanoid-physical-safety-evidence-v1",
        frame_count: planned.option!.certificate.validated_frame_limit,
        first_frame: 1,
        last_frame: planned.option!.certificate.validated_frame_limit,
        maximum_actuator_effort_utilization: {
          joint: expect.any(String),
          requested_utilization: expect.any(Number),
          applied_utilization: expect.any(Number),
          saturated: expect.any(Boolean)
        }
      });
      expect(world.snapshot().frame).toBe(before.frame);
      expect(world.snapshot().worldRevision).toBe(before.worldRevision);
      expect(world.checkpoint().motions).toHaveLength(1);
      expect(world.checkpoint().motions[0]!.plan.id).toBe("selection-forward");

      const executed = await world.executeWholeBodyMotion(planned.planId);
      expect(executed.accepted).toBe(true);
      expect(executed.code).toBe("motion_option_succeeded");
      expect(executed.detail.option).toMatchObject({
        option_id: "selection-forward-option",
        status: "succeeded",
        termination_reason: "physical_success",
        full_frame_count: planned.motion!.frame_count,
        artifact_sha256: planned.motion!.sha256
      });
      expect(executed.detail.option!.executed_prefix_frame_count).toBeLessThan(
        executed.detail.option!.full_frame_count
      );
      expect(executed.detail.option!.actual_termination_frame).toBe(
        executed.detail.option!.predicted_termination_frame
      );
      expect(executed.detail.physical_safety).toMatchObject({
        frame_count: executed.detail.option!.executed_prefix_frame_count,
        first_frame: 1,
        last_frame: executed.detail.option!.executed_prefix_frame_count,
        maximum_actuator_effort_utilization: {
          joint: expect.any(String),
          requested_utilization: expect.any(Number),
          applied_utilization: expect.any(Number),
          saturated: expect.any(Boolean)
        }
      });
      expect(executed.finalSnapshot.physicalSafety).toEqual({
        planId: planned.planId,
        evidence: executed.detail.physical_safety
      });
      expect(executed.finalSnapshot.robot.fallen).toBe(false);
      expect(world.checkpoint().motions).toEqual([]);
      expect(world.checkpoint().physicalSafety).toEqual(
        executed.finalSnapshot.physicalSafety
      );
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("rejects a candidate as soon as its bounded during constraint is violated", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const before = world.snapshot();
      const batch = forwardCandidateBatch(before, "phase-violation");
      batch.termination.predicates.push({
        type: "root_near_point",
        target: { ...before.robot.rootPosition },
        tolerance_m: 0.015
      });
      batch.termination.phases = {
        precondition: null,
        during: {
          condition: { op: "predicate", predicate_index: 1 }
        },
        terminal: {
          condition: { op: "predicate", predicate_index: 0 }
        }
      };

      const planned = await world.planWholeBodyMotionCandidates(batch);

      expect(planned.accepted).toBe(false);
      expect(planned.candidates).toHaveLength(2);
      expect(planned.candidates.some((candidate) => (
        candidate.validation.failures.some((failure) => (
          failure.code === "motion_constraint_violated"
        ))
      ))).toBe(true);
      expect(world.snapshot().worldRevision).toBe(before.worldRevision);
      expect(world.checkpoint().motions).toEqual([]);
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("resumes a physical option with its detector streak and never replays committed frames", async () => {
    const uninterrupted = await HumanoidWorld.create(scenario);
    const interrupted = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const completePlan = await uninterrupted.planWholeBodyMotionCandidates(
        forwardCandidateBatch(uninterrupted.snapshot(), "option-resume")
      );
      const interruptedPlan = await interrupted.planWholeBodyMotionCandidates(
        forwardCandidateBatch(interrupted.snapshot(), "option-resume")
      );
      expect(completePlan.accepted).toBe(true);
      expect(interruptedPlan.accepted).toBe(true);
      expect(interruptedPlan.motion?.sha256).toBe(completePlan.motion?.sha256);

      const complete = await uninterrupted.executeWholeBodyMotion(completePlan.planId);
      const predictedFrame = completePlan.option!.certificate.predicted_termination_frame;
      expect(complete.detail.option?.actual_termination_frame).toBe(predictedFrame);
      const interruptionFrame = predictedFrame - 1;
      let recoveryCheckpoint: ReturnType<HumanoidWorld["checkpoint"]> | undefined;
      await expect(interrupted.executeWholeBodyMotion(
        interruptedPlan.planId,
        () => {
          if (interrupted.snapshot().worldRevision !== interruptionFrame) return;
          recoveryCheckpoint = interrupted.checkpoint();
          throw new Error("simulated option interruption");
        }
      )).rejects.toThrow("simulated option interruption");

      expect(recoveryCheckpoint).toBeDefined();
      expect(recoveryCheckpoint!.motions[0]).toMatchObject({
        progress: {
          nextFrameIndex: interruptionFrame,
          physicalSafety: {
            frame_count: interruptionFrame,
            first_frame: 1,
            last_frame: interruptionFrame
          }
        },
        option: {
          status: "executing",
          successStreak: 1,
          actualTerminationFrame: null,
          terminationReason: null
        }
      });
      resumed = await HumanoidWorld.create(scenario, recoveryCheckpoint);
      const recovered = await resumed.executeWholeBodyMotion(interruptedPlan.planId);
      expect(recovered).toMatchObject({
        accepted: true,
        code: "motion_option_succeeded",
        frames: 1
      });
      expect(recovered.detail.option?.executed_prefix_frame_count).toBe(predictedFrame);
      expect(recovered.detail.physical_safety).toEqual(complete.detail.physical_safety);
      expect(recovered.finalSnapshot.physicalSafety).toEqual(
        complete.finalSnapshot.physicalSafety
      );
      expect(recovered.finalSnapshot.frame).toBe(complete.finalSnapshot.frame);
      expect(recovered.finalSnapshot.worldRevision).toBe(
        complete.finalSnapshot.worldRevision
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

  it("rejects restored options when their artifact or termination contract was altered", async () => {
    const world = await HumanoidWorld.create(scenario);
    let legacyWorld: HumanoidWorld | undefined;
    try {
      const planned = await world.planWholeBodyMotionCandidates(
        forwardCandidateBatch(world.snapshot(), "tamper")
      );
      expect(planned.accepted).toBe(true);
      const checkpoint = world.checkpoint();

      const artifactChanged = structuredClone(checkpoint);
      artifactChanged.motions[0]!.artifact.frames[0]!.reference.jointPositions[0]! += 0.001;
      await expect(HumanoidWorld.create(scenario, artifactChanged)).rejects.toThrow(
        "certificate does not match"
      );

      const contractChanged = structuredClone(checkpoint);
      const predicate = contractChanged.motions[0]!.option!.contract.predicates[0]!;
      if (predicate.type !== "root_near_point") throw new Error("Unexpected test predicate");
      predicate.target.x += 0.01;
      await expect(HumanoidWorld.create(scenario, contractChanged)).rejects.toThrow(
        "certificate does not match"
      );

      const legacy = structuredClone(checkpoint);
      delete legacy.motions[0]!.option!.certificate.physical_safety;
      delete legacy.physicalSafety;
      delete legacy.simulation.requestedActuatorTorques;
      legacyWorld = await HumanoidWorld.create(scenario, legacy);
      expect(
        legacyWorld.snapshot().robot.joints.left_hip_pitch_joint.effort
      ).toBeUndefined();
      const executed = await legacyWorld.executeWholeBodyMotion(planned.planId);
      expect(executed.accepted).toBe(true);
      expect(executed.detail.physical_safety).toMatchObject({
        protocol: "humanoid-physical-safety-evidence-v1",
        first_frame: 1,
        maximum_actuator_effort_utilization: {
          requested_utilization: expect.any(Number),
          applied_utilization: expect.any(Number)
        }
      });
    } finally {
      await legacyWorld?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it("rejects a restored success that did not complete its physical stability window", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const planned = await world.planWholeBodyMotionCandidates(
        forwardCandidateBatch(world.snapshot(), "forged-success-window")
      );
      expect(planned.accepted).toBe(true);
      const checkpoint = world.checkpoint();
      const stored = checkpoint.motions[0]!;
      const terminationFrame = stored.option!.certificate.predicted_termination_frame;
      stored.progress.nextFrameIndex = terminationFrame;
      stored.option!.status = "succeeded";
      stored.option!.successStreak = 0;
      stored.option!.monitor.phase = "succeeded";
      stored.option!.monitor.terminalStableSteps = 0;
      stored.option!.actualTerminationFrame = terminationFrame;
      stored.option!.terminationReason = "physical_success";
      checkpoint.frame = terminationFrame;
      checkpoint.worldRevision = terminationFrame;

      await expect(HumanoidWorld.create(scenario, checkpoint)).rejects.toThrow(
        "full physical stability window"
      );
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("rejects invalid contact history and preserves executed-prefix evidence", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const planned = await world.planWholeBodyMotionCandidates(
        forwardCandidateBatch(world.snapshot(), "contact-history")
      );
      expect(planned.accepted).toBe(true);
      const initialCheckpoint = world.checkpoint();
      const contact = {
        body: "left_wrist_yaw_link" as const,
        object_id: "unobserved-crate",
        required: true
      };
      const contactKey = `${contact.body}\u0000${contact.object_id}`;

      const outsideContract = structuredClone(initialCheckpoint);
      outsideContract.motions[0]!.progress.satisfiedContactKeys = [contactKey];
      await expect(HumanoidWorld.create(scenario, outsideContract)).rejects.toThrow(
        "outside its current contact contract"
      );

      const prefilledPlan = structuredClone(initialCheckpoint);
      prefilledPlan.motions[0]!.plan.contact_constraints = [contact];
      prefilledPlan.motions[0]!.intentSha256 = humanoidMotionIntentSha256(
        prefilledPlan.motions[0]!.plan
      );
      prefilledPlan.motions[0]!.progress.satisfiedContactKeys = [contactKey];
      await expect(HumanoidWorld.create(scenario, prefilledPlan)).rejects.toThrow(
        "unstarted humanoid motion"
      );

      let interruptedCheckpoint: ReturnType<HumanoidWorld["checkpoint"]> | undefined;
      await expect(world.executeWholeBodyMotion(planned.planId, () => {
        if (world.snapshot().worldRevision !== 1) return;
        interruptedCheckpoint = world.checkpoint();
        throw new Error("simulated contact-history interruption");
      })).rejects.toThrow("simulated contact-history interruption");
      expect(interruptedCheckpoint).toBeDefined();
      interruptedCheckpoint!.motions[0]!.plan.contact_constraints = [contact];
      interruptedCheckpoint!.motions[0]!.intentSha256 = humanoidMotionIntentSha256(
        interruptedCheckpoint!.motions[0]!.plan
      );
      interruptedCheckpoint!.motions[0]!.progress.satisfiedContactKeys = [contactKey];

      await expect(HumanoidWorld.create(
        scenario,
        structuredClone(interruptedCheckpoint)
      )).rejects.toThrow("contact evidence identity does not match");
      const interruptedMotion = interruptedCheckpoint!.motions[0]!;
      interruptedMotion.progress.satisfiedContactEvidenceSha256 =
        humanoidMotionContactEvidenceSha256({
          planId: interruptedMotion.plan.id,
          intentSha256: interruptedMotion.intentSha256,
          artifactSha256: humanoidMotionArtifactSha256(
            interruptedMotion.artifact
          ),
          nextFrameIndex: interruptedMotion.progress.nextFrameIndex,
          satisfiedContactKeys: interruptedMotion.progress.satisfiedContactKeys
        });

      resumed = await HumanoidWorld.create(scenario, interruptedCheckpoint);
      expect(resumed.checkpoint().motions[0]!.progress.satisfiedContactKeys).toEqual([
        contactKey
      ]);
      const executed = await resumed.executeWholeBodyMotion(planned.planId);
      expect(executed).toMatchObject({
        accepted: true,
        code: "motion_option_succeeded",
        detail: {
          option: {
            status: "succeeded",
            termination_reason: "physical_success"
          }
        }
      });
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it("does not accept artifact exhaustion as physical option success", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const planned = await world.planWholeBodyMotionCandidates(
        forwardCandidateBatch(world.snapshot(), "forged-certificate-probe")
      );
      expect(planned.accepted).toBe(true);
      const checkpoint = world.checkpoint();
      const stored = checkpoint.motions[0]!;
      const contract = {
        option_id: "unreachable-option",
        predicates: [{
          type: "root_near_point" as const,
          target: {
            ...world.snapshot().robot.rootPosition,
            z: world.snapshot().robot.rootPosition.z + 0.5
          },
          tolerance_m: 0.01
        }],
        stable_steps: 2
      };
      stored.option!.contract = contract;
      stored.option!.certificate.contract_sha256 =
        humanoidMotionOptionContractSha256(contract);
      stored.option!.monitor = createHumanoidMotionOptionMonitorState(contract);
      stored.option!.certificate.stable_steps = contract.stable_steps;
      stored.option!.certificate.evidence = [];

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      const executed = await resumed.executeWholeBodyMotion(planned.planId);
      expect(executed).toMatchObject({
        accepted: false,
        code: "motion_goal_unmet",
        frames: stored.option!.certificate.validated_frame_limit,
        detail: {
          option: {
            status: "goal_unmet",
            termination_reason: "motion_goal_unmet",
            executed_prefix_frame_count:
              stored.option!.certificate.validated_frame_limit
          }
        }
      });
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("cuts off a persistently diverged rollout instead of playing its remaining frames", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const planned = await world.planWholeBodyMotionCandidates(
        forwardCandidateBatch(world.snapshot(), "execution-drift")
      );
      expect(planned.accepted).toBe(true);
      const checkpoint = world.checkpoint();
      const stored = checkpoint.motions[0]!;
      expect(stored.rollout).not.toBeNull();
      for (const frame of stored.rollout!.frames) frame.rootPosition.x += 0.5;
      stored.option!.certificate.rollout_sha256 = humanoidMotionRolloutSha256(
        stored.rollout!
      );

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      const executed = await resumed.executeWholeBodyMotion(planned.planId);

      expect(executed).toMatchObject({
        accepted: false,
        code: "motion_execution_drifted",
        frames: stored.rollout!.limits.consecutive_steps,
        detail: {
          option: {
            status: "failed",
            termination_reason: "execution_drift",
            drift_streak: stored.rollout!.limits.consecutive_steps,
            drift_evidence: { drifted: true }
          }
        }
      });
      expect(executed.detail.option!.executed_prefix_frame_count).toBeLessThan(
        executed.detail.option!.full_frame_count
      );
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("terminates real execution when a certified during constraint is violated", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const before = world.snapshot();
      const planned = await world.planWholeBodyMotionCandidates(
        forwardCandidateBatch(before, "execution-constraint")
      );
      expect(planned.accepted).toBe(true);
      const checkpoint = world.checkpoint();
      const stored = checkpoint.motions[0]!;
      const contract = {
        ...stored.option!.contract,
        predicates: [
          ...stored.option!.contract.predicates,
          {
            type: "root_near_point" as const,
            target: { ...before.robot.rootPosition },
            tolerance_m: 0.015
          }
        ],
        phases: {
          precondition: null,
          during: {
            condition: { op: "predicate" as const, predicate_index: 1 }
          },
          terminal: {
            condition: { op: "predicate" as const, predicate_index: 0 }
          }
        }
      };
      stored.option!.contract = contract;
      stored.option!.certificate.contract_sha256 =
        humanoidMotionOptionContractSha256(contract);
      stored.option!.monitor = createHumanoidMotionOptionMonitorState(contract);

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      const executed = await resumed.executeWholeBodyMotion(planned.planId);

      expect(executed).toMatchObject({
        accepted: false,
        code: "motion_constraint_violated",
        detail: {
          option: {
            status: "failed",
            termination_reason: "motion_constraint_violated"
          }
        }
      });
      expect(executed.frames).toBeLessThan(stored.artifact.frames.length);
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("rejects a restored option precondition before the first real physics step", async () => {
    const world = await HumanoidWorld.create(scenario);
    let resumed: HumanoidWorld | undefined;
    try {
      const initial = world.snapshot();
      const planned = await world.planWholeBodyMotionCandidates(
        forwardCandidateBatch(initial, "execution-precondition")
      );
      expect(planned.accepted).toBe(true);
      const checkpoint = world.checkpoint();
      const stored = checkpoint.motions[0]!;
      const contract = {
        ...stored.option!.contract,
        predicates: [
          ...stored.option!.contract.predicates,
          {
            type: "root_near_point" as const,
            target: { ...initial.robot.rootPosition },
            tolerance_m: 0.03
          }
        ],
        phases: {
          precondition: {
            condition: { op: "predicate" as const, predicate_index: 1 },
            stable_steps: 1
          },
          during: null,
          terminal: {
            condition: { op: "predicate" as const, predicate_index: 0 }
          }
        }
      };
      stored.option!.contract = contract;
      stored.option!.certificate.contract_sha256 =
        humanoidMotionOptionContractSha256(contract);
      stored.option!.monitor = createHumanoidMotionOptionMonitorState(contract);
      checkpoint.simulation.positions[1]! += 0.5;

      resumed = await HumanoidWorld.create(scenario, checkpoint);
      const beforeExecution = resumed.snapshot();
      expect(beforeExecution.robot.rootPosition.x).toBeGreaterThan(
        initial.robot.rootPosition.x + 0.4
      );
      const executed = await resumed.executeWholeBodyMotion(planned.planId);

      expect(executed).toMatchObject({
        accepted: false,
        code: "motion_constraint_violated",
        frames: 0,
        detail: {
          failures: [expect.objectContaining({
            code: "motion_constraint_violated",
            atSeconds: 0,
            message: "Motion option precondition is not satisfied before execution"
          })],
          option: {
            status: "failed",
            termination_reason: "motion_constraint_violated",
            executed_prefix_frame_count: 0,
            actual_termination_frame: 0
          }
        }
      });
      expect(executed.finalSnapshot.frame).toBe(beforeExecution.frame);
      expect(executed.finalSnapshot.worldRevision).toBe(beforeExecution.worldRevision);
      expect(executed.finalSnapshot.robot.simulatedTime).toBe(
        beforeExecution.robot.simulatedTime
      );
      expect(executed.finalSnapshot.robot.rootPosition).toEqual(
        beforeExecution.robot.rootPosition
      );
      expect(resumed.checkpoint().motions).toEqual([]);
    } finally {
      await resumed?.dispose();
      await world.dispose();
    }
  }, 45_000);

});

function graspCandidateBatch(
  graspContractSha256: string
): HumanoidMotionCandidateBatch {
  const contactConstraints = [{
    hand_surface: "left_hand_thumb_2_link" as const,
    object_id: "crate",
    required: false
  }, {
    hand_surface: "left_hand_index_1_link" as const,
    object_id: "crate",
    required: false
  }];
  return {
    objective: "验证左手真实抓取",
    termination: {
      option_id: "grasp-crate-option",
      predicates: [{
        type: "grasp_verified",
        object_id: "crate",
        hand: "left",
        grasp_contract_sha256: graspContractSha256
      }],
      stable_steps: 2
    },
    candidates: [{
      id: "grasp-crate-hold",
      intent: "保持当前姿态并闭合左手",
      duration_seconds: 0.1,
      contact_constraints: contactConstraints,
      keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
    }, {
      id: "grasp-crate-turn",
      intent: "轻微转身并闭合左手",
      duration_seconds: 0.1,
      contact_constraints: contactConstraints,
      keyframes: [{
        at_seconds: 0,
        root_yaw_velocity: 0.05
      }, {
        at_seconds: 0.1,
        root_yaw_velocity: 0.05
      }]
    }]
  };
}

function forwardCandidateBatch(
  snapshot: ReturnType<HumanoidWorld["snapshot"]>,
  prefix: string
): HumanoidMotionCandidateBatch {
  const target = {
    ...snapshot.robot.rootPosition,
    z: snapshot.robot.rootPosition.z + 0.08
  };
  return {
    objective: "比较连续全身候选并真实前进",
    termination: {
      option_id: `${prefix}-forward-option`,
      predicates: [{
        type: "root_near_point",
        target,
        tolerance_m: 0.035
      }],
      stable_steps: 2
    },
    candidates: [
      {
        id: `${prefix}-noop`,
        intent: "没有实现当前前进目标",
        duration_seconds: 0.8,
        keyframes: [{ at_seconds: 0 }, { at_seconds: 0.8 }]
      },
      {
        id: `${prefix}-forward`,
        intent: "保持双足支撑并连续向前移动",
        duration_seconds: 0.8,
        keyframes: [
          {
            at_seconds: 0,
            root_velocity: { forward_mps: 0.2, lateral_mps: 0 }
          },
          {
            at_seconds: 0.8,
            root_velocity: { forward_mps: 0.2, lateral_mps: 0 }
          }
        ]
      }
    ]
  };
}
