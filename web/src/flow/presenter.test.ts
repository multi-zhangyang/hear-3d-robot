import { describe, expect, it } from "vitest";
import { presentEmbodiedEpisode, presentFramework } from "./presenter";

describe("agent timeline presenter", () => {
  it("derives stable keys from the durable runtime event identity", () => {
    const entry = {
      runtime_event_id: "framework-event-1",
      at: "2026-07-30T00:00:00.000Z",
      event: {
        type: "run_item_stream_event",
        name: "message_output_created",
        item: { rawItem: { content: [{ type: "output_text", text: "继续观察。" }] } }
      }
    };

    expect(presentFramework([entry])[0]?.id).toBe("message-framework-event-1");
    expect(presentFramework([{ ignored: true }, entry])[0]?.id)
      .toBe("message-framework-event-1");
  });

  it("presents model messages, tool results, usage, and active-agent transitions", () => {
    const at = "2026-07-30T00:00:00.000Z";
    const entries = [
      {
        agent_name: "humanoid-coordinator",
        at,
        event: { type: "agent_updated_stream_event", agent: "humanoid-coordinator" }
      },
      {
        agent_name: "humanoid-coordinator",
        at,
        event: {
          type: "run_item_stream_event",
          name: "message_output_created",
          item: {
            rawItem: {
              id: "message-1",
              content: [{ type: "output_text", text: "已确认当前姿态，准备规划全身动作。" }]
            }
          }
        }
      },
      {
        agent_name: "humanoid-coordinator",
        at,
        event: {
          type: "run_item_stream_event",
          name: "tool_output",
          item: {
            rawItem: {
              name: "plan_whole_body_motion",
              callId: "call-1",
              output: {
                type: "text",
                text: JSON.stringify({ accepted: true, code: "whole_body_plan_validated" })
              }
            }
          }
        }
      },
      {
        agent_name: "humanoid-coordinator",
        at,
        event: {
          type: "raw_model_stream_event",
          data: {
            type: "response_done",
            response_id: "response-1",
            usage: { inputTokens: 120, outputTokens: 30, totalTokens: 150 }
          }
        }
      }
    ];

    const moments = presentFramework(entries);
    expect(moments).toEqual([
      expect.objectContaining({ title: "接管当前执行流", agent: "自主协调智能体" }),
      expect.objectContaining({ title: "模型输出", detail: "已确认当前姿态，准备规划全身动作。" }),
      expect.objectContaining({
        title: "规划全身动作回执",
        detail: "动作已接受 · 全身动作已通过物理预演",
        tone: "success"
      }),
      expect.objectContaining({
        title: "模型调用已完成",
        detail: "150 个令牌 · 输入 120 · 输出 30"
      })
    ]);
    expect(JSON.stringify(moments)).not.toContain("response-1");
  });

  it("omits empty message placeholders and marks rejected tool results", () => {
    const moments = presentFramework([
      {
        at: "2026-07-30T00:00:00.000Z",
        event: {
          type: "run_item_stream_event",
          name: "reasoning_item_created",
          item: { rawItem: { content: [{ type: "input_text", text: "private reasoning" }] } }
        }
      },
      {
        at: "2026-07-30T00:00:00.000Z",
        event: {
          type: "run_item_stream_event",
          name: "message_output_created",
          item: { rawItem: { content: [{ type: "output_text", text: "\n" }] } }
        }
      },
      {
        at: "2026-07-30T00:00:01.000Z",
        event: {
          type: "run_item_stream_event",
          name: "tool_output",
          item: {
            rawItem: {
              name: "execute_whole_body_motion",
              callId: "call-2",
              output: {
                type: "text",
                text: JSON.stringify({
                  accepted: false,
                  code: "planning_receipt_missing",
                  detail: { recovery: "A matching planning receipt is required." }
                })
              }
            }
          }
        }
      }
    ]);
    expect(moments).toEqual([
      expect.objectContaining({
        tone: "warning",
        detail: "动作已拒绝 · 缺少规划回执"
      })
    ]);
    expect(JSON.stringify(moments)).not.toContain("A matching planning receipt is required");
  });

  it("shows bounded real output without exposing structured provider fields", () => {
    const at = "2026-07-30T00:00:00.000Z";
    const moments = presentFramework([
      {
        agent_name: "humanoid-motion-reference",
        at,
        event: {
          type: "run_item_stream_event",
          name: "message_output_created",
          item: {
            rawItem: {
              id: "provider-message-id",
              content: [{
                type: "output_text",
                text: JSON.stringify({
                  summary: "已选定连续全身参考，下一步将进行物理预演。",
                  provider: "sample-provider",
                  endpoint: "https://provider.example.invalid/v1",
                  model: "vendor/example-model",
                  response_id: "provider-response-id",
                  instructions: "internal instructions"
                })
              }]
            }
          }
        }
      },
      {
        agent_name: "humanoid-executor",
        at,
        event: {
          type: "run_item_stream_event",
          name: "message_output_created",
          item: {
            rawItem: {
              content: [{
                type: "output_text",
                text: "机器人已完成连续双足动作。 endpoint=https://service.example.invalid model=vendor/example-model {\"response_id\":\"hidden\"}"
              }]
            }
          }
        }
      }
    ]);

    expect(moments.map((moment) => moment.detail)).toEqual([
      "已选定连续全身参考，下一步将进行物理预演。",
      "机器人已完成连续双足动作。"
    ]);
    const visible = JSON.stringify(moments);
    expect(visible).not.toContain("sample-provider");
    expect(visible).not.toContain("provider.example.invalid");
    expect(visible).not.toContain("vendor/example-model");
    expect(visible).not.toContain("provider-response-id");
    expect(visible).not.toContain("internal instructions");
    expect(visible).not.toContain("response_id");
  });

  it("extracts concrete tool facts and uses a tool-specific neutral fallback", () => {
    const moments = presentFramework([
      {
        at: "2026-07-30T00:00:00.000Z",
        event: {
          type: "run_item_stream_event",
          name: "tool_output",
          item: {
            rawItem: {
              name: "execute_humanoid_navigation",
              output: {
                type: "text",
                text: JSON.stringify({
                  accepted: true,
                  code: "navigation_completed",
                  frame_count: 12
                })
              }
            }
          }
        }
      },
      {
        at: "2026-07-30T00:00:01.000Z",
        event: {
          type: "run_item_stream_event",
          name: "tool_output",
          item: {
            rawItem: {
              name: "observe_humanoid",
              output: { type: "text", text: "unstructured internal payload" }
            }
          }
        }
      }
    ]);

    expect(moments).toEqual([
      expect.objectContaining({
        title: "执行双足导航回执",
        detail: "动作已接受 · 双足导航分块已完成 · 12 个物理帧",
        tone: "success"
      }),
      expect.objectContaining({
        title: "感知人形世界回执",
        detail: "感知人形世界已返回。",
        tone: "neutral"
      })
    ]);
    expect(JSON.stringify(moments)).not.toContain("unstructured internal payload");
  });

  it("presents an embodied episode as a bounded physical-memory event", () => {
    const presented = presentEmbodiedEpisode({
      sequence: 3,
      transaction_id: "execution-3",
      action: "execute_whole_body_motion",
      code: "motion_completed",
      model_summary: "完成连续全身动作并保持双脚支撑。 endpoint=https://hidden.invalid",
      world_before_revision: 20,
      world_after_revision: 45,
      frame_count: 25,
      result_frame: 125,
      result_root_position: { x: 1, y: 0.8, z: 2 },
      fallen: false,
      support: "double",
      upright: 0.99,
      goal_success: false,
      recorded_at: "2026-08-02T00:00:03.000Z"
    });

    expect(presented).toMatchObject({
      title: "物理经历已记住",
      detail: "完成连续全身动作并保持双脚支撑。",
      meta: "25 个物理帧 · 世界版本 45",
      category: "verify",
      tone: "neutral"
    });

    expect(presentEmbodiedEpisode({
      sequence: 4,
      transaction_id: "execution-4",
      action: "execute_whole_body_motion",
      code: "motion_failed",
      model_summary: "动作执行后失去平衡。",
      world_before_revision: 45,
      world_after_revision: 50,
      frame_count: 5,
      result_frame: 130,
      result_root_position: { x: 1, y: 0.2, z: 2 },
      fallen: true,
      support: "none",
      upright: 0.2,
      goal_success: false,
      recorded_at: "2026-08-02T00:00:04.000Z"
    }).tone).toBe("warning");
  });
});
