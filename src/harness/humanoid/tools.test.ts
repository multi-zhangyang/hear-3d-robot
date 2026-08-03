import { RunContext } from "@openai/agents";
import { describe, expect, it, vi } from "vitest";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  createHumanoidActionTools,
  type HumanoidActionInvoker
} from "./tools.js";

describe("humanoid Agents SDK tools", () => {
  it("preserves action grants, agent identity and SDK transaction identity", async () => {
    const invoke = vi.fn(async (
      action,
      _input,
      transactionId,
      agentId
    ): Promise<HumanoidActionReceipt> => ({
      transactionId,
      agentId,
      action,
      accepted: true,
      code: "humanoid_observed",
      worldBeforeRevision: 4,
      worldAfterRevision: 4,
      frameCount: 0,
      channels: [],
      detail: { frame: 120 },
      committedAt: "2026-08-02T00:00:00.000Z"
    }));
    const tools = createHumanoidActionTools(
      { invoke } as HumanoidActionInvoker,
      "perception-agent",
      ["observe_humanoid"]
    );
    expect(tools.map((entry) => entry.name)).toEqual(["observe_humanoid"]);
    const observe = tools[0];
    if (!observe || observe.type !== "function") throw new Error("Observe tool is missing");
    expect(observe.description).toContain("29 个关节");

    const output = await observe.invoke(
      new RunContext({ runId: "run-humanoid-tools" }),
      "{}",
      {
        toolCall: {
          type: "function_call",
          callId: "observe-call-1",
          name: "observe_humanoid",
          arguments: "{}",
          status: "completed"
        }
      }
    );

    expect(JSON.parse(String(output))).toMatchObject({
      transactionId: "observe-call-1",
      agentId: "perception-agent",
      action: "observe_humanoid",
      accepted: true
    });
    expect(invoke).toHaveBeenCalledWith(
      "observe_humanoid",
      {},
      "observe-call-1",
      "perception-agent"
    );
  });

  it("rejects duplicate action grants instead of exposing ambiguous tools", () => {
    const runtime = {
      invoke: vi.fn()
    } as unknown as HumanoidActionInvoker;
    expect(() => createHumanoidActionTools(runtime, "motion-agent", [
      "plan_whole_body_motion",
      "plan_whole_body_motion"
    ])).toThrow("Duplicate humanoid action grant");
  });

  it("publishes a strict task-space schema without exposing joint-angle authoring", () => {
    const runtime = {
      invoke: vi.fn()
    } as unknown as HumanoidActionInvoker;
    const plan = createHumanoidActionTools(
      runtime,
      "motion-agent",
      ["plan_whole_body_motion"]
    )[0];
    if (!plan || plan.type !== "function") throw new Error("Motion tool is missing");
    const parameters = JSON.stringify(plan.parameters);
    expect(parameters).toContain('"left_hand"');
    expect(parameters).toContain('"right_hand"');
    expect(parameters).toContain('"root_velocity"');
    expect(parameters).not.toContain("left_shoulder_pitch_joint");
    expect(parameters).not.toContain("right_wrist_yaw_joint");
    expect(parameters).toContain('"position"');
    expect(parameters).not.toContain('"additionalProperties":{"type":"number"}');
    expect(parameters).not.toContain('"items":[');
  });

  it("returns exact validation issues to the model without invoking physics", async () => {
    const invoke = vi.fn();
    const plan = createHumanoidActionTools(
      { invoke } as unknown as HumanoidActionInvoker,
      "motion-agent",
      ["plan_whole_body_motion"]
    )[0];
    if (!plan || plan.type !== "function") throw new Error("Motion tool is missing");
    const output = await plan.invoke(
      new RunContext({ runId: "invalid-motion-input" }),
      "{}",
      {
        toolCall: {
          type: "function_call",
          callId: "invalid-plan-1",
          name: "plan_whole_body_motion",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(output))).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      tool: "plan_whole_body_motion",
      automatic_actuation: false,
      validation_issues: expect.arrayContaining([
        expect.objectContaining({ path: "id" }),
        expect.objectContaining({ path: "keyframes" })
      ])
    });
    expect(invoke).not.toHaveBeenCalled();
  });
});
