import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { HumanoidWorld } from "../../world/humanoid/world.js";
import {
  advancePhysicalTrajectory,
  createPhysicalTrajectory
} from "./physical-trajectory-recorder.js";

const scenario = ScenarioSchema.parse({
  title: "物理轨迹证据场",
  seed: 73,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 6,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [{
    id: "trace-crate",
    kind: "crate",
    color: "copper",
    position: { x: 4, y: 0.25, z: 4 },
    size: { x: 0.5, y: 0.5, z: 0.5 },
    portable: true
  }],
  zones: [],
  default_goal: {
    summary: "保持直立",
    predicates: [{
      type: "robot_at",
      target: { x: 2, y: 0, z: 2 },
      tolerance: 0.5
    }]
  }
});

describe("authoritative physical trajectory recorder", () => {
  it("accumulates body, joint, end-effector, contact and object motion", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const first = world.snapshot();
      const second = structuredClone(first);
      second.frame += 1;
      second.worldRevision += 1;
      second.robot.rootPosition.x += 0.25;
      second.robot.joints.left_hip_pitch_joint.position += 0.1;
      second.robot.links.left_wrist_yaw_link.position.z += 0.2;
      second.robot.objects["trace-crate"]!.position.z += 0.3;
      second.robot.contacts = [];

      const initial = createPhysicalTrajectory(first);
      const advanced = advancePhysicalTrajectory(initial, second);

      expect(advanced).toMatchObject({
        complete_from_admission: true,
        start_frame: first.frame,
        end_frame: second.frame,
        observed_frame_count: 2,
        root_path_length_m: 0.25,
        root_planar_path_length_m: 0.25,
        joint_total_variation_rad: 0.1,
        object_path_length_m: { "trace-crate": 0.3 },
        controller_usage: {
          complete_from_admission: true,
          observed_frame_count: 2
        }
      });
      expect(Object.values(
        advanced.controller_usage!.mode_frame_counts
      ).reduce((total, count) => total + count, 0)).toBe(2);
      expect(advanced.end_effector_path_length_m.left_wrist).toBe(0.2);
      expect(advanced.trajectory_sha256).not.toBe(initial.trajectory_sha256);
      expect(advancePhysicalTrajectory(advanced, second)).toEqual(advanced);
    } finally {
      await world.dispose();
    }
  }, 15_000);

  it("bounds retained samples without losing complete-frame metrics or endpoints", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      let frame = world.snapshot();
      let summary = createPhysicalTrajectory(frame);
      for (let index = 0; index < 140; index += 1) {
        frame = structuredClone(frame);
        frame.frame += 1;
        frame.worldRevision += 1;
        frame.robot.rootPosition.z += 0.01;
        summary = advancePhysicalTrajectory(summary, frame);
      }

      expect(summary.samples.length).toBeLessThanOrEqual(64);
      expect(summary.samples[0]!.frame).toBe(summary.start_frame);
      expect(summary.samples.at(-1)!.frame).toBe(summary.end_frame);
      expect(summary.observed_frame_count).toBe(141);
      expect(summary.root_planar_path_length_m).toBeCloseTo(1.4, 6);
      expect(summary.sample_stride).toBeGreaterThan(1);

      const skipped = structuredClone(frame);
      skipped.frame += 2;
      skipped.worldRevision += 2;
      expect(() => advancePhysicalTrajectory(summary, skipped)).toThrow(/not contiguous/);
    } finally {
      await world.dispose();
    }
  }, 15_000);

  it("preserves a partial controller-usage boundary when resuming legacy evidence", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const first = world.snapshot();
      const legacy = createPhysicalTrajectory(first);
      delete legacy.controller_usage;
      const second = structuredClone(first);
      second.frame += 1;
      second.worldRevision += 1;

      const advanced = advancePhysicalTrajectory(legacy, second);

      expect(advanced.controller_usage).toMatchObject({
        complete_from_admission: false,
        observed_frame_count: 1
      });
    } finally {
      await world.dispose();
    }
  }, 15_000);

  it("records a Memory Bridge transition once instead of counting every frame", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const first = world.snapshot();
      first.robot.controllerExecution = memoryBridgeExecution("guiding", false);
      const initial = createPhysicalTrajectory(first);
      const second = structuredClone(first);
      second.frame += 1;
      second.worldRevision += 1;
      second.robot.controllerExecution = memoryBridgeExecution("completed", true);
      const completed = advancePhysicalTrajectory(initial, second);

      expect(completed.samples[0]?.controller_execution?.routing).toMatchObject({
        reason: "entry_state_ood",
        memory_bridge: { phase: "guiding" }
      });
      expect(completed.samples.at(-1)?.controller_execution?.routing).toMatchObject({
        reason: "memory_bridge_completed",
        memory_bridge: { phase: "completed", progress: 1 }
      });
      expect(completed.controller_usage?.routing).toMatchObject({
        decision_count: 1,
        memory_bridge_attempt_count: 1,
        memory_bridge_completed_count: 1,
        memory_bridge_timeout_count: 0
      });
    } finally {
      await world.dispose();
    }
  }, 15_000);
});

function memoryBridgeExecution(
  phase: "guiding" | "completed",
  admitted: boolean
): NonNullable<ReturnType<HumanoidWorld["snapshot"]>["robot"]["controllerExecution"]> {
  return {
    protocol: "humanoid-controller-execution-v1",
    mode: admitted ? "learned_policy" : "reference_control",
    activeImplementation: admitted ? "trained-policy" : "yahmp",
    transition: null,
    routing: {
      callId: "memory-bridge-call",
      route: admitted ? "primary" : "fallback",
      assessment: {
        protocol: "humanoid-policy-admission-v1",
        implementation: "trained-policy",
        skillFamily: "navigation",
        admitted,
        reason: admitted ? "memory_bridge_completed" : "entry_state_ood",
        coldStart: false,
        entryStateOodScore: admitted ? 1 : 6,
        commandOodScore: 0,
        posterior: {
          outcomes: 8,
          successes: 8,
          failures: 0,
          posteriorMean: 0.9,
          lowerBound: 0.65,
          upperBound: 0.98,
          recentSuccessRate: 1,
          transitionAttempts: 0,
          transitionSuccesses: 0
        },
        successfulEntryPrototype: Array.from({ length: 29 }, () => 0)
      },
      attribution: {
        primarySteps: admitted ? 1 : 0,
        fallbackSteps: 0,
        upperBodyOverlaySteps: 0,
        memoryBridgeSteps: admitted ? 5 : 1
      },
      memoryBridge: {
        protocol: "humanoid-policy-memory-bridge-v1",
        phase,
        trigger: "entry_state_ood",
        completedSteps: admitted ? 5 : 0,
        maximumSteps: 200,
        stableSteps: admitted ? 5 : 0,
        requiredStableSteps: 5,
        progress: admitted ? 1 : 0,
        entryStateOodScore: admitted ? 1 : 6,
        jointPrototypeRmsError: admitted ? 0.05 : 0.8,
        maximumJointVelocity: admitted ? 0.1 : 3
      }
    }
  };
}
