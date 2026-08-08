import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { HumanoidFrameBuffer } from "../stage/humanoid-frame-buffer";
import type { HumanoidRunDetails, HumanoidWorldSnapshot } from "../types";
import {
  HUMANOID_BODY_CHANNELS,
  HumanoidMissionWorkspace,
  activeHumanoidGrasps,
  humanoidManipulationTelemetry,
  movingHumanoidChannels
} from "./HumanoidMissionWorkspace";

describe("人形身体通道", () => {
  it("按服务端契约展示六个通道", () => {
    expect(HUMANOID_BODY_CHANNELS).toEqual([
      "locomotion",
      "left_leg",
      "right_leg",
      "torso",
      "left_arm",
      "right_arm"
    ]);
  });

  it("分别识别根运动、双腿、躯干和双臂的实时活动", () => {
    const frame = {
      robot: {
        links: {
          pelvis: {
            linearVelocity: { x: 0.06, y: 0, z: 0 },
            angularVelocity: { x: 0, y: 0, z: 0 }
          }
        },
        joints: {
          left_ankle_pitch_joint: { velocity: 0.1 },
          right_knee_joint: { velocity: -0.11 },
          waist_yaw_joint: { velocity: 0.12 },
          left_elbow_joint: { velocity: 0.13 },
          right_shoulder_pitch_joint: { velocity: -0.14 }
        }
      }
    } as unknown as HumanoidWorldSnapshot;

    expect(movingHumanoidChannels(frame)).toEqual(HUMANOID_BODY_CHANNELS);
  });

  it("按当前工作智能体的恢复预算显示上下文压力", () => {
    const world = worldSnapshot();
    const frameBuffer = new HumanoidFrameBuffer();
    frameBuffer.reset(world);
    const html = renderToStaticMarkup(createElement(HumanoidMissionWorkspace, {
      details: runDetails(world),
      frameBuffer,
      streamState: "connected"
    }));

    expect(html).toContain("width:50%");
    expect(html).toContain(
      "当前上下文估算为 4000 个令牌，上下文窗口为 32768 个令牌，压缩触发线为 8000 个令牌"
    );
    expect(html).toContain("4000 / 3.3万");
    expect(html).toContain("压缩线 8000");
  });

  it("成功终态不再显示仍在选择 Goal", () => {
    const world = worldSnapshot();
    const frameBuffer = new HumanoidFrameBuffer();
    frameBuffer.reset(world);
    const details = runDetails(world);
    details.checkpoint.version = 6;
    details.checkpoint.status = "succeeded";
    details.checkpoint.goal_dag = {
      status: "awaiting_model_selection",
      candidates: {
        first: { status: "proposed" }
      }
    } as unknown as NonNullable<HumanoidRunDetails["checkpoint"]["goal_dag"]>;

    const html = renderToStaticMarkup(createElement(HumanoidMissionWorkspace, {
      details,
      frameBuffer,
      streamState: "inactive"
    }));

    expect(html).toContain("任务目标已完成");
    expect(html).toContain("已完成");
    expect(html).not.toContain("等待目标管理智能体选择");
  });

  it("只显示当前物理帧的真实抓取与保持进度", () => {
    const world = worldSnapshot();
    world.grasp.assessments = [
      graspAssessment(0, true),
      graspAssessment(1, true),
      graspAssessment(0, false)
    ];
    expect(activeHumanoidGrasps(world)).toHaveLength(1);

    const frameBuffer = new HumanoidFrameBuffer();
    frameBuffer.reset(world);
    const details = runDetails(world);
    details.checkpoint.version = 5;
    details.checkpoint.goal = {
      summary: "抓住测试箱体",
      predicates: [{
        type: "object_grasped",
        object_id: "test-crate",
        hand: "left"
      }]
    };
    const html = renderToStaticMarkup(createElement(HumanoidMissionWorkspace, {
      details,
      frameBuffer,
      streamState: "connected"
    }));
    expect(html).toContain("实时抓取状态");
    expect(html).toContain("左手 · 场景实体");
    expect(html).toContain("已抓稳 · 接触 8 帧 · 抬离 6 帧");
    expect(html).toContain("抓取保持进度");
    expect(html).toContain(">8/6<");
  });

  it("以当前物理证据呈现全身交互闭环", () => {
    const world = worldSnapshot();
    world.robot.objects["test-crate"] = {
      id: "test-crate",
      position: { x: 0.3, y: 0.2, z: 0.4 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      linearVelocity: { x: 0, y: 0, z: 0 },
      angularVelocity: { x: 0, y: 0, z: 0 }
    };
    world.grasp.assessments = [graspAssessment(0, true)];
    const details = runDetails(world);
    details.checkpoint.version = 5;
    details.checkpoint.goal = {
      summary: "稳放箱体",
      predicates: [{
        type: "object_placed",
        object_id: "test-crate",
        zone_id: "arrival",
        tolerance: 0.05
      }]
    };
    details.checkpoint.checker = {
      success: false,
      checks: [{ name: "1:object_placed", passed: false, actual: {} }]
    } as HumanoidRunDetails["checkpoint"]["checker"];

    expect(humanoidManipulationTelemetry(
      world,
      details.checkpoint.goal,
      details.checkpoint.checker
    )).toEqual({
      objectId: "test-crate",
      present: true,
      contact: true,
      grasped: true,
      placed: false
    });

    const frameBuffer = new HumanoidFrameBuffer();
    frameBuffer.reset(world);
    const html = renderToStaticMarkup(createElement(HumanoidMissionWorkspace, {
      details,
      frameBuffer,
      streamState: "connected"
    }));
    expect(html).toContain("实时全身交互闭环");
    expect(html).toContain("全身交互");
    expect(html).toContain("目标");
    expect(html).toContain("持握");
    expect(html).toContain("落位");
  });
});

function runDetails(world: HumanoidWorldSnapshot): HumanoidRunDetails {
  return {
    definition: { run_id: "run-worker", scenario: {} },
    actions: [],
    scenario_chunks: {
      version: 1,
      scenario_seed: 0,
      scenario_sha256: "a".repeat(64),
      manifest_version: 1,
      revision: 0,
      changed_chunk_ids: [],
      chunks: []
    },
    checkpoint: {
      status: "running",
      root_id: "worker",
      active_agent_id: "worker",
      nodes: {
        worker: {
          id: "worker",
          name: "工作智能体",
          status: "active",
          model_calls_used: 2,
          created_at: "2026-08-03T00:00:00.000Z"
        }
      },
      world,
      goal: { predicates: [] },
      checker: null,
      embodied_memory: { total_episodes: 0 },
      context_memory: {
        version: 1,
        context_window_tokens: 65_536,
        compact_trigger_tokens: 40_000,
        compact_recent_model_turns: 4,
        compact_max_output_tokens: 2_048,
        active_scope_id: "worker",
        active_estimated_tokens: 100,
        total_compactions: 0,
        last_compacted_at: null,
        scopes: {
          worker: {
            scope_id: "worker",
            agent_id: "worker",
            agent_name: "工作智能体",
            raw_item_count: 0,
            raw_chain_hash: null,
            compacted_item_count: 0,
            retained_item_count: 0,
            retained_chain_hash: null,
            active_estimated_tokens: 4_000,
            context_window_tokens: 32_768,
            compact_trigger_tokens: 8_000,
            compact_recent_model_turns: 2,
            compact_max_output_tokens: 768,
            compaction_count: 0,
            summary: null,
            summary_origin: null,
            summary_world_revision: null,
            last_compacted_at: null
          }
        }
      }
    }
  } as unknown as HumanoidRunDetails;
}

function worldSnapshot(): HumanoidWorldSnapshot {
  return {
    frame: 0,
    worldRevision: 0,
    motionGenerator: {
      protocol: "humanoid-motion-generator-v1",
      implementation: "task_space_constraints",
      motionClass: "constraint_solver",
      sampling: "deterministic"
    },
    robot: {
      simulatedTime: 0,
      controller: {
        protocol: "humanoid-controller-v1",
        implementation: "yahmp_onnx",
        actuation: "joint_position_pd",
        controlStepSeconds: 0.02,
        physicsStepSeconds: 0.002
      },
      rootPosition: { x: 0, y: 0.8, z: 0 },
      rootRotation: { x: 0, y: 0, z: 0, w: 1 },
      joints: {},
      links: {
        pelvis: {
          position: { x: 0, y: 0.8, z: 0 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
          linearVelocity: { x: 0, y: 0, z: 0 },
          angularVelocity: { x: 0, y: 0, z: 0 }
        }
      },
      objects: {},
      contactCount: 0,
      contacts: [],
      feet: {
        left: { touching: true, contactCount: 1, normalForce: 300, points: [] },
        right: { touching: true, contactCount: 1, normalForce: 300, points: [] }
      },
      balance: {
        centerOfMass: { x: 0, y: 0.8, z: 0 },
        support: "double",
        supportMargin: 0.1,
        upright: 1
      },
      nonFootEnvironmentContacts: [],
      fallen: false
    },
    grasp: {
      contractSha256: "fc1e2d113bb5e5f5f8a75f0faa3efc8bd97ecc18eb41463da09d26bb52cfc193",
      assessments: []
    },
    navigation: {
      planId: null,
      status: "idle",
      target: null,
      waypoints: [],
      waypointIndex: null
    }
  };
}

function graspAssessment(frame: number, verified: boolean) {
  return {
    frame,
    object_id: verified ? "test-crate" : "idle-crate",
    hand: "left" as const,
    phase: verified ? "verified" as const : "idle" as const,
    grasp_verified: verified,
    evidence: {
      contact: { status: verified ? "opposed" as const : "missing" as const },
      relative_pose: { stable_frames: verified ? 8 : 0 },
      lifted_hold_frames: verified ? 6 : 0
    }
  } as HumanoidWorldSnapshot["grasp"]["assessments"][number];
}
