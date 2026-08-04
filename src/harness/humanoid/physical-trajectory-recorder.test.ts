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
        object_path_length_m: { "trace-crate": 0.3 }
      });
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
});
