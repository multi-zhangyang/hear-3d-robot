import { describe, expect, it } from "vitest";
import { presentFramework } from "./presenter";

describe("agent timeline presenter", () => {
  it("derives stable keys from the durable runtime event identity", () => {
    const entry = {
      runtime_event_id: "framework-event-1",
      at: "2026-07-30T00:00:00.000Z",
      event: {
        type: "run_item_stream_event",
        name: "message_output_created",
        item: { rawItem: { content: [{ type: "output_text", text: "继续探索。" }] } }
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
        agent_name: "Navigator",
        at,
        event: { type: "agent_updated_stream_event", agent: "Navigator" }
      },
      {
        agent_name: "Navigator",
        at,
        event: {
          type: "run_item_stream_event",
          name: "message_output_created",
          item: {
            rawItem: {
              id: "message-1",
              content: [{ type: "output_text", text: "已找到可达探索边界，准备继续勘察。" }]
            }
          }
        }
      },
      {
        agent_name: "Navigator",
        at,
        event: {
          type: "run_item_stream_event",
          name: "tool_output",
          item: {
            rawItem: {
              name: "plan_base_path",
              callId: "call-1",
              output: {
                type: "text",
                text: JSON.stringify({ accepted: true, code: "base_path_planned" })
              }
            }
          }
        }
      },
      {
        agent_name: "Navigator",
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
      expect.objectContaining({ title: "接管当前执行流", agent: "导航智能体" }),
      expect.objectContaining({ title: "模型输出", detail: "已找到可达探索边界，准备继续勘察。" }),
      expect.objectContaining({
        title: "规划底盘路线回执",
        detail: "动作已接受 · 底盘路线已规划",
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
              name: "drive_base",
              callId: "call-2",
              output: {
                type: "text",
                text: JSON.stringify({
                  accepted: false,
                  code: "body_channel_busy",
                  detail: { recovery: "Wait for the active body lease." }
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
        detail: "动作已拒绝 · 身体通道正忙"
      })
    ]);
    expect(JSON.stringify(moments)).not.toContain("Wait for the active body lease");
  });

  it("shows bounded real output without exposing structured provider fields", () => {
    const at = "2026-07-30T00:00:00.000Z";
    const moments = presentFramework([
      {
        agent_name: "Navigator",
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
                  summary: "已选定北侧可达边界，下一步将规划路线。",
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
        agent_name: "Navigator",
        at,
        event: {
          type: "run_item_stream_event",
          name: "message_output_created",
          item: {
            rawItem: {
              content: [{
                type: "output_text",
                text: "机器人已到达新的勘察位置。 endpoint=https://service.example.invalid model=vendor/example-model {\"response_id\":\"hidden\"}"
              }]
            }
          }
        }
      }
    ]);

    expect(moments.map((moment) => moment.detail)).toEqual([
      "已选定北侧可达边界，下一步将规划路线。",
      "机器人已到达新的勘察位置。"
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
              name: "survey_terrain",
              output: {
                type: "text",
                text: JSON.stringify({
                  accepted: true,
                  code: "terrain_survey",
                  frame_count: 12,
                  detail: { movement_sampling: { choice_count: 7 } }
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
              name: "sense_scene",
              output: { type: "text", text: "unstructured internal payload" }
            }
          }
        }
      }
    ]);

    expect(moments).toEqual([
      expect.objectContaining({
        title: "勘察附近地形回执",
        detail: "动作已接受 · 地形勘察已完成 · 12 个物理帧 · 7 个可达候选",
        tone: "success"
      }),
      expect.objectContaining({
        title: "观察可见世界回执",
        detail: "观察可见世界已返回。",
        tone: "neutral"
      })
    ]);
    expect(JSON.stringify(moments)).not.toContain("unstructured internal payload");
  });
});
