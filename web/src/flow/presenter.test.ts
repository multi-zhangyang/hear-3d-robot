import { describe, expect, it } from "vitest";
import { presentAction, presentEmbodiedEpisode, presentFramework } from "./presenter";

describe("agent timeline presenter", () => {
  it("presents the semantic Skill pipeline without unknown-action placeholders", () => {
    const planned = presentAction({
      transactionId: "skill-route",
      agentId: "humanoid-motion-reference",
      action: "plan_humanoid_skill",
      input: { skill_transaction_id: "bound-skill" },
      fingerprint: "fingerprint",
      accepted: true,
      code: "autonomous_skill_route_validated",
      worldBeforeRevision: 20,
      worldAfterRevision: 20,
      frameCount: 0,
      channels: [],
      detail: {},
      committedAt: "2026-08-08T00:00:00.000Z"
    });
    const executed = presentAction({
      transactionId: "skill-execution",
      agentId: "humanoid-executor",
      action: "execute_humanoid_skill",
      input: { planning_transaction_id: "skill-route" },
      fingerprint: "fingerprint",
      accepted: true,
      code: "navigation_completed",
      worldBeforeRevision: 20,
      worldAfterRevision: 534,
      frameCount: 514,
      channels: ["locomotion"],
      detail: {},
      committedAt: "2026-08-08T00:00:01.000Z"
    });

    expect(planned).toMatchObject({
      title: "验证技能路线",
      detail: "技能路线已完成导航、全身控制与 MuJoCo 物理预演。",
      meta: "技能路线已通过物理预演",
      category: "plan"
    });
    expect(executed).toMatchObject({
      title: "执行自主技能",
      detail: "机器人已执行模型选择并通过预演的技能路线。",
      meta: "514 个物理帧",
      category: "move"
    });
  });

  it("shows the real candidate count and selected physical rank", () => {
    const presented = presentAction({
      transactionId: "candidate-plan",
      agentId: "humanoid-motion-reference",
      action: "plan_whole_body_motion_candidates",
      input: { objective: "比较全身候选", candidates: [] },
      fingerprint: "fingerprint",
      accepted: true,
      code: "whole_body_candidates_validated",
      worldBeforeRevision: 4,
      worldAfterRevision: 4,
      frameCount: 0,
      channels: ["torso"],
      detail: { candidate_count: 3, selected_rank: 2 },
      committedAt: "2026-08-03T00:00:00.000Z"
    });

    expect(presented).toMatchObject({
      title: "筛选全身候选",
      detail: "3 个模型候选已分别完成 MuJoCo 预演，选择第 2 个可行动作。",
      meta: "全身候选已通过物理筛选",
      tone: "success",
      category: "plan"
    });
  });

  it("presents physical option success with predicted and actual termination", () => {
    const presented = presentAction({
      transactionId: "execute-option",
      agentId: "humanoid-executor",
      action: "execute_whole_body_motion",
      input: { planning_transaction_id: "candidate-plan" },
      fingerprint: "fingerprint",
      accepted: true,
      code: "motion_option_succeeded",
      worldBeforeRevision: 4,
      worldAfterRevision: 31,
      frameCount: 27,
      channels: ["locomotion"],
      detail: {
        result: {
          motion: { control_step_seconds: 0.02 },
          option: {
            status: "succeeded",
            termination_reason: "physical_success",
            predicted_termination_frame: 29,
            actual_termination_frame: 27
          }
        }
      },
      committedAt: "2026-08-03T00:00:01.000Z"
    });

    expect(presented).toMatchObject({
      detail: "物理目标稳定达成 · 预测 0.58 秒 · 实际 0.54 秒",
      meta: "27 个物理帧",
      tone: "active"
    });
  });

  it("presents rollout drift as an early physical interruption", () => {
    const presented = presentAction({
      transactionId: "execute-drifted-option",
      agentId: "humanoid-executor",
      action: "execute_whole_body_motion",
      input: { planning_transaction_id: "candidate-plan" },
      fingerprint: "fingerprint",
      accepted: false,
      code: "motion_execution_drifted",
      worldBeforeRevision: 4,
      worldAfterRevision: 7,
      frameCount: 3,
      channels: ["locomotion"],
      detail: {
        result: {
          option: {
            status: "failed",
            termination_reason: "execution_drift",
            drift_streak: 3
          }
        }
      },
      committedAt: "2026-08-03T00:00:01.000Z"
    });

    expect(presented).toMatchObject({
      detail: "执行连续 3 帧偏离物理预演，已截断并交回重规划。",
      meta: "3 个物理帧",
      tone: "warning"
    });
  });

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

  it("turns internal lifecycle assignments into a concise status message", () => {
    const moments = presentFramework([{
      runtime_event_id: "framework-internal-state",
      at: "2026-08-08T00:00:00.000Z",
      event: {
        type: "run_item_stream_event",
        name: "message_output_created",
        item: {
          rawItem: {
            content: [{
              type: "output_text",
              text: "当前状态 coordinator_phase=complete_cycle，cycle_completion.status=ready。"
            }]
          }
        }
      }
    }]);

    expect(moments).toEqual([
      expect.objectContaining({
        title: "模型输出",
        detail: "当前物理执行、执行后感知与目标验收证据已经齐备。"
      })
    ]);
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
