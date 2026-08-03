import { RunContext } from "@openai/agents";
import { describe, expect, it, vi } from "vitest";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  createHumanoidActionTools,
  createHumanoidEmbodiedRecallTool,
  type HumanoidActionInvoker,
  type HumanoidEmbodiedRecallInvoker
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

  it("exposes bounded read-only embodied recall and forces historical provenance", async () => {
    const recallEmbodiedHistory = vi.fn(async () => ({
      historical_only: false,
      current_world_revision: 19,
      episodes: [{ source_ref: "episode:12", sequence: 12 }],
      actions: [{
        source_ref: "action:execute-failed",
        transactionId: "execute-failed",
        action: "execute_whole_body_motion",
        accepted: false,
        code: "motion_goal_unmet"
      }],
      missing_source_refs: [],
      next_before_sequence: null
    }));
    const recall = createHumanoidEmbodiedRecallTool({
      recallEmbodiedHistory
    } as HumanoidEmbodiedRecallInvoker);
    const parameters = JSON.stringify(recall.parameters);
    expect(parameters).toContain('"source_refs"');
    expect(parameters).toContain('"before_sequence"');
    expect(parameters).toContain('"limit"');
    expect(parameters).toContain("action:");
    expect(recall.description).toContain("不代表当前传感");

    const output = await recall.invoke(
      new RunContext({ runId: "bounded-embodied-recall" }),
      JSON.stringify({
        source_refs: ["episode:12", "action:execute-failed"],
        before_sequence: 20,
        limit: 2
      }),
      {
        toolCall: {
          type: "function_call",
          callId: "recall-episode-12",
          name: "recall_embodied_history",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(output))).toMatchObject({
      historical_only: true,
      current_world_revision: 19,
      episodes: [{ source_ref: "episode:12", sequence: 12 }],
      actions: [{
        source_ref: "action:execute-failed",
        transactionId: "execute-failed",
        code: "motion_goal_unmet"
      }]
    });
    expect(recallEmbodiedHistory).toHaveBeenCalledWith({
      source_refs: ["episode:12", "action:execute-failed"],
      before_sequence: 20,
      limit: 2
    });

    const invalid = await recall.invoke(
      new RunContext({ runId: "invalid-embodied-recall" }),
      JSON.stringify({
        source_refs: ["episode:12", "episode:12"],
        limit: 2
      }),
      {
        toolCall: {
          type: "function_call",
          callId: "invalid-recall-source-refs",
          name: "recall_embodied_history",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(invalid))).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      tool: "recall_embodied_history",
      historical_only: true
    });
    expect(recallEmbodiedHistory).toHaveBeenCalledTimes(1);
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
    expect(parameters).toContain('"left_foot"');
    expect(parameters).toContain('"right_foot"');
    expect(parameters).toContain('"root_velocity"');
    expect(parameters).not.toContain("left_shoulder_pitch_joint");
    expect(parameters).not.toContain('"default"');
    expect(parameters).not.toContain("right_wrist_yaw_joint");
    expect(parameters).toContain('"position"');
    expect(parameters).not.toContain('"additionalProperties":{"type":"number"}');
    expect(parameters).not.toContain('"items":[');
    expect(plan.description).toContain("双脚踝");
  });

  it("exposes and enforces a bounded model-ranked candidate schema", async () => {
    const invoke = vi.fn();
    const runtime = { invoke } as unknown as HumanoidActionInvoker;
    const candidates = createHumanoidActionTools(
      runtime,
      "motion-agent",
      ["plan_whole_body_motion_candidates"]
    )[0];
    if (!candidates || candidates.type !== "function") {
      throw new Error("Motion candidate tool is missing");
    }
    const parameters = JSON.stringify(candidates.parameters);
    expect(parameters).toContain('"objective"');
    expect(parameters).toContain('"candidates"');
    expect(parameters).toContain('"root_velocity"');
    expect(parameters).toContain('"predicate_indexes"');
    expect(parameters).toContain('"not_predicate_indexes"');
    expect(parameters).not.toContain("left_shoulder_pitch_joint");
    expect(parameters).not.toContain('"default"');

    const output = await candidates.invoke(
      new RunContext({ runId: "bounded-motion-candidates" }),
      JSON.stringify({
        objective: "比较候选",
        candidates: [{
          id: "only-one",
          intent: "只有一个候选",
          duration_seconds: 0.2,
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.2 }]
        }]
      }),
      {
        toolCall: {
          type: "function_call",
          callId: "invalid-candidate-count",
          name: "plan_whole_body_motion_candidates",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(output))).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      tool: "plan_whole_body_motion_candidates",
      automatic_actuation: false
    });

    const duplicateOutput = await candidates.invoke(
      new RunContext({ runId: "duplicate-motion-candidates" }),
      JSON.stringify({
        objective: "拒绝重复动作候选",
        termination: {
          option_id: "duplicate-motion-option",
          predicates: [{
            type: "root_near_point",
            body: null,
            object_id: null,
            zone_id: null,
            target: { x: 0, y: 0.76, z: 0.1 },
            tolerance_m: 0.05,
            minimum_normal_force: null,
            expected: null
          }],
          stable_steps: 2,
          phases: null
        },
        candidates: [
          {
            id: "same-motion-a",
            intent: "候选 A",
            duration_seconds: 0.2,
            keyframes: [{ at_seconds: 0 }, { at_seconds: 0.2 }]
          },
          {
            id: "same-motion-b",
            intent: "候选 B",
            duration_seconds: 0.2,
            keyframes: [{ at_seconds: 0 }, { at_seconds: 0.2 }]
          }
        ]
      }),
      {
        toolCall: {
          type: "function_call",
          callId: "duplicate-candidate-content",
          name: "plan_whole_body_motion_candidates",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(duplicateOutput))).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      tool: "plan_whole_body_motion_candidates",
      validation_issues: [expect.objectContaining({
        path: "candidates.1",
        message: expect.stringContaining("id and intent labels")
      })],
      automatic_actuation: false
    });
    expect(invoke).not.toHaveBeenCalled();
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
