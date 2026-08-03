import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../../config/load.js";
import { ScenarioSchema } from "../../domain/schema.js";
import {
  humanoidMotionArtifactSha256,
  serializeHumanoidReference
} from "./motion-artifact.js";
import type {
  HumanoidMotionCandidateBatch,
  HumanoidMotionGenerator,
  HumanoidMotionGeneratorInput
} from "./motion-plan.js";
import {
  createHumanoidMotionOptionMonitorState,
  humanoidMotionOptionContractSha256
} from "./motion-option.js";
import { humanoidMotionRolloutSha256 } from "./motion-rollout.js";
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
      expect(executed.finalSnapshot.robot.fallen).toBe(false);
      expect(world.checkpoint().motions).toEqual([]);
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
        progress: { nextFrameIndex: interruptionFrame },
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
    } finally {
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

  it("rejects invalid contact history and requires fresh contact after restoration", async () => {
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
      interruptedCheckpoint!.motions[0]!.progress.satisfiedContactKeys = [contactKey];

      resumed = await HumanoidWorld.create(scenario, interruptedCheckpoint);
      expect(resumed.checkpoint().motions[0]!.progress.satisfiedContactKeys).toEqual([]);
      const executed = await resumed.executeWholeBodyMotion(planned.planId);
      expect(executed).toMatchObject({
        accepted: false,
        code: "motion_goal_unmet",
        detail: {
          option: {
            status: "goal_unmet",
            termination_reason: "motion_goal_unmet"
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
        frames: stored.artifact.frames.length,
        detail: {
          option: {
            status: "goal_unmet",
            termination_reason: "motion_goal_unmet",
            executed_prefix_frame_count: stored.artifact.frames.length
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
