import { describe, expect, it } from "vitest";
import {
  agentNameLabel,
  entityLabel,
  modelOutputLabel,
  nodeResultLabel,
  resultCodeLabel
} from "./ui-text";
import type { TaskNode } from "./types";

describe("中文界面文案", () => {
  it("为模型生成的带编号角色保留身份并显示中文名称", () => {
    expect(agentNameLabel("mission_supervisor")).toBe("任务监督智能体");
    expect(agentNameLabel("movement_executor_7")).toBe("运动执行智能体 7");
    expect(agentNameLabel("frontier_mover_1")).toBe("探索边界智能体 1");
    expect(agentNameLabel("unrecognized_specialist_3")).toBe("专项智能体 3");
  });

  it("不会把内部实体 ID 和回执码直接暴露为英文界面文案", () => {
    expect(entityLabel("brass_key")).toBe("黄铜钥匙");
    expect(entityLabel("unknown_prop")).toBe("场景实体");
    expect(resultCodeLabel("base_path_collision")).toBe("底盘路线存在碰撞");
    expect(resultCodeLabel("unknown_arm_plan")).toBe("机械臂规划不存在");
    expect(resultCodeLabel("spatial_memory_context_unavailable")).toBe("当前智能体无可用空间记忆上下文");
    expect(resultCodeLabel("future_runtime_code")).toBe("未识别的运行回执");
  });

  it("从结构化模型结果中只展示真实摘要", () => {
    const node = {
      status: "completed",
      last_result: {
        output: JSON.stringify({
          status: "completed",
          summary: "机器人已完成当前勘察区域的路线执行。",
          endpoint: "https://provider.example.invalid/v1",
          model: "vendor/example-model",
          response_id: "provider-response-id",
          evidence: [{ transaction_id: "internal-transaction-id" }]
        })
      }
    } as TaskNode;

    expect(nodeResultLabel(node)).toBe("机器人已完成当前勘察区域的路线执行。");
  });

  it("限制模型文本长度并移除配置、链接和原始 JSON", () => {
    const visible = modelOutputLabel(
      `已完成北侧地形勘察。 AI_API_KEY=not-for-display endpoint=https://service.example.invalid model=vendor/example-model {"response_id":"hidden"}`
    );
    expect(visible).toBe("已完成北侧地形勘察。");

    const long = modelOutputLabel("地形信息".repeat(80), 80);
    expect(Array.from(long ?? "")).toHaveLength(80);
    expect(long?.endsWith("…")).toBe(true);

    expect(modelOutputLabel({
      provider: "sample-provider",
      endpoint: "https://provider.example.invalid/v1",
      model: "vendor/example-model"
    })).toBeNull();
  });
});
