import { describe, expect, it } from "vitest";
import {
  actionLabel,
  agentNameLabel,
  bodyChannelLabel,
  entityLabel,
  humanoidControllerExecutionLabel,
  humanoidControllerLabel,
  modelOutputLabel,
  motionGeneratorLabel,
  nodeResultLabel,
  predicateLabel,
  resultCodeLabel
} from "./ui-text";
import type { TaskNode } from "./types";

describe("中文界面文案", () => {
  it("覆盖全部六个人形身体通道", () => {
    expect([
      bodyChannelLabel("locomotion"),
      bodyChannelLabel("left_leg"),
      bodyChannelLabel("right_leg"),
      bodyChannelLabel("torso"),
      bodyChannelLabel("left_arm"),
      bodyChannelLabel("right_arm")
    ]).toEqual(["双足运动", "左腿", "右腿", "躯干", "左臂", "右臂"]);
  });

  it("为模型生成的带编号角色保留身份并显示中文名称", () => {
    expect(agentNameLabel("humanoid-coordinator")).toBe("自主协调智能体");
    expect(agentNameLabel("人形自主协调智能体")).toBe("自主协调智能体");
    expect(agentNameLabel("humanoid_executor_7")).toBe("人形物理执行智能体 7");
    expect(agentNameLabel("unrecognized_specialist_3")).toBe("专项智能体 3");
  });

  it("不会把内部实体 ID 和回执码直接暴露为英文界面文案", () => {
    expect(entityLabel("courtyard_crate")).toBe("庭院木箱");
    expect(entityLabel("unknown_prop")).toBe("场景实体");
    expect(resultCodeLabel("whole_body_plan_validated")).toBe("全身动作已通过物理预演");
    expect(resultCodeLabel("planning_receipt_missing")).toBe("缺少规划回执");
    expect(resultCodeLabel("required_contact_missing")).toBe("动作缺少要求的物理接触");
    expect(actionLabel("execute_humanoid_skill")).toBe("执行自主技能");
    expect(actionLabel("submit_coordinator_decision")).toBe("提交协调决策");
    expect(resultCodeLabel("autonomous_skill_route_validated"))
      .toBe("技能路线已通过物理预演");
    expect(resultCodeLabel("plan_revalidation_failed"))
      .toBe("执行前物理复验未通过");
    expect(resultCodeLabel("future_runtime_code")).toBe("未识别的运行回执");
  });

  it("根据实时能力描述显示运动生成器和全身控制器", () => {
    expect(motionGeneratorLabel("task_space_constraints")).toBe("任务约束");
    expect(motionGeneratorLabel("ardy_g1")).toBe("ARDY");
    expect(motionGeneratorLabel("unknown_backend")).toBe("运动生成器");
    expect(humanoidControllerLabel("yahmp_onnx")).toBe("YAHMP");
    expect(humanoidControllerLabel("mjlab_g1_velocity_onnx")).toBe("mjlab G1");
    expect(humanoidControllerLabel("sonic_onnx")).toBe("SONIC");
    expect(humanoidControllerExecutionLabel({
      protocol: "humanoid-controller-execution-v1",
      mode: "reference_control",
      activeImplementation: "yahmp_onnx",
      transition: null
    }, "mjlab_g1_velocity_onnx")).toBe("YAHMP · 参考控制");
    expect(humanoidControllerExecutionLabel({
      protocol: "humanoid-controller-execution-v1",
      mode: "reference_control",
      activeImplementation: "yahmp_onnx",
      transition: {
        fromImplementation: "mjlab_g1_velocity_onnx",
        toImplementation: "yahmp_onnx",
        progress: 0.37,
        durationSeconds: 0.2
      }
    }, "mjlab_g1_velocity_onnx")).toBe("YAHMP · 交接 37%");
    expect(humanoidControllerExecutionLabel({
      protocol: "humanoid-controller-execution-v1",
      mode: "hybrid_control",
      activeImplementation: "mjlab_g1_velocity_onnx+yahmp_onnx",
      transition: null
    }, "mjlab_g1_velocity_onnx")).toBe("mjlab G1 + YAHMP · 混合控制");
  });

  it("完整显示具名末端的三维目标", () => {
    expect(predicateLabel({
      type: "end_effector_at",
      end_effector: "right_wrist",
      frame: "pelvis",
      target: { x: -0.25, y: 0.3, z: 0.15 },
      tolerance: 0.05,
      stable_frames: 5
    })).toBe("右手腕到达骨盆相对 [-0.25, 0.30, 0.15]");
    expect(predicateLabel({
      type: "end_effector_at",
      end_effector: "left_wrist",
      frame: "world",
      target: { x: 2, y: 1.1, z: 3 },
      tolerance: 0.05,
      stable_frames: 5,
      orientation: { x: 0, y: 0, z: 0, w: 1 },
      orientation_tolerance_rad: 0.15
    })).toBe("左手腕到达世界 [2.00, 1.10, 3.00] · 姿态");
  });

  it("用中文显示抓取手和目标物体", () => {
    expect(predicateLabel({
      type: "object_grasped",
      object_id: "courtyard_crate",
      hand: "either"
    })).toBe("任意手抓住庭院木箱");
  });

  it("用中文显示真实稳放目标", () => {
    expect(predicateLabel({
      type: "object_placed",
      object_id: "courtyard_crate",
      zone_id: "courtyard_beacon",
      tolerance: 0.05
    })).toBe("将庭院木箱稳放在区域 庭院信标区");
  });

  it("从结构化模型结果中只展示真实摘要", () => {
    const node = {
      status: "completed",
      last_result: {
        output: JSON.stringify({
          status: "completed",
          summary: "机器人已完成当前全身动作。",
          endpoint: "https://provider.example.invalid/v1",
          model: "vendor/example-model",
          response_id: "provider-response-id",
          evidence: [{ transaction_id: "internal-transaction-id" }]
        })
      }
    } as TaskNode;

    expect(nodeResultLabel(node)).toBe("机器人已完成当前全身动作。");
  });

  it("限制模型文本长度并移除配置、链接和原始 JSON", () => {
    const visible = modelOutputLabel(
      `已完成庭院双足导航。 AI_API_KEY=not-for-display endpoint=https://service.example.invalid model=vendor/example-model {"response_id":"hidden"}`
    );
    expect(visible).toBe("已完成庭院双足导航。");

    const long = modelOutputLabel("物理状态".repeat(80), 80);
    expect(Array.from(long ?? "")).toHaveLength(80);
    expect(long?.endsWith("…")).toBe(true);

    expect(modelOutputLabel({
      provider: "sample-provider",
      endpoint: "https://provider.example.invalid/v1",
      model: "vendor/example-model"
    })).toBeNull();
  });
});
