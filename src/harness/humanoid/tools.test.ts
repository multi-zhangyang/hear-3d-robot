import { RunContext } from "@openai/agents";
import { describe, expect, it, vi } from "vitest";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import { HumanoidActionInputs } from "./actions.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  createHumanoidActionTools,
  createHumanoidEmbodiedRecallTool,
  type HumanoidActionInvoker,
  type HumanoidEmbodiedRecallInvoker
} from "./tools.js";

describe("humanoid Agents SDK tools", () => {
  it("keeps navigation heading guidance consistent with IK base placements", () => {
    const description = HumanoidActionInputs.plan_humanoid_navigation
      .shape.arrival_heading.description;

    expect(description).toContain("reachable_base_placements");
    expect(description).toContain("type=yaw");
    expect(description).toContain("root_yaw_radians");
  });

  it("filters stale Motion planning tools until its own observation is fresh", async () => {
    let fresh = false;
    const runtime = {
      isActionAvailable: (name: string, agentId: string) => (
        agentId !== "humanoid-motion-reference"
          || name === "observe_humanoid"
          || fresh
      ),
      invoke: async () => {
        throw new Error("Tool invocation is outside this availability test");
      }
    } as unknown as HumanoidActionInvoker;
    const tools = createHumanoidActionTools(
      runtime,
      "humanoid-motion-reference",
      ["observe_humanoid", "plan_humanoid_navigation"]
    );
    const context = new RunContext({});
    const agent = {} as never;

    expect(await tools[0]!.isEnabled(context, agent)).toBe(true);
    expect(await tools[1]!.isEnabled(context, agent)).toBe(false);
    fresh = true;
    expect(await tools[1]!.isEnabled(context, agent)).toBe(true);
  });

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
      detail: {
        frame: 120,
        physical_trajectory: { samples: Array.from({ length: 200 }, (_, frame) => ({ frame })) },
        root: { position: { x: 1, y: 0.76, z: 2 } },
        fallen: false,
        end_effectors: {
          left_wrist: { world_position: { x: 1.2, y: 0.75, z: 2.1 } }
        },
        joints: Object.fromEntries(Array.from(
          { length: 200 },
          (_, index) => [`joint-${index}`, { position: index }]
        ))
      },
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
    expect(observe.description).toContain("根姿态");
    expect(observe.description).toContain("掌指碰撞面真实几何");

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

    const parsedOutput = JSON.parse(String(output));
    expect(parsedOutput).toMatchObject({
      transactionId: "observe-call-1",
      agentId: "perception-agent",
      action: "observe_humanoid",
      accepted: true,
      detail: {
        frame: 120
      },
      durable_evidence: {
        source_ref: "action:observe-call-1",
        full_receipt_persisted: true
      }
    });
    expect(parsedOutput.detail).not.toHaveProperty("physical_trajectory");
    expect(parsedOutput.detail).not.toHaveProperty("joints");
    expect(String(output).length).toBeLessThan(2_000);
    expect(invoke).toHaveBeenCalledWith(
      "observe_humanoid",
      {},
      "observe-call-1",
      "perception-agent",
      {
        tool_call_id: "observe-call-1",
        tool_name: "observe_humanoid",
        arguments_sha256: modelPayloadSha256({})
      }
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
    expect(parameters).toContain('"before_experience_sequence"');
    expect(parameters).toContain('"outcomes"');
    expect(parameters).toContain('"predicate_types"');
    expect(parameters).toContain('"object_ids"');
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

    await recall.invoke(
      new RunContext({ runId: "semantic-embodied-recall" }),
      JSON.stringify({
        outcomes: ["physically_failed"],
        object_ids: ["crate"],
        before_sequence: 20,
        before_experience_sequence: 30,
        limit: 4
      }),
      {
        toolCall: {
          type: "function_call",
          callId: "recall-failed-crate",
          name: "recall_embodied_history",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(recallEmbodiedHistory).toHaveBeenLastCalledWith({
      outcomes: ["physically_failed"],
      object_ids: ["crate"],
      before_sequence: 20,
      before_experience_sequence: 30,
      limit: 4
    });

    const invalid = await recall.invoke(
      new RunContext({ runId: "invalid-embodied-recall" }),
      JSON.stringify({
        source_refs: ["episode:12"],
        outcomes: ["rejected"],
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
    expect(recallEmbodiedHistory).toHaveBeenCalledTimes(2);
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
          mode: "all",
          option_id: "duplicate-motion-option",
          predicates: [{
            type: "root_near_point",
            target: { x: 0, y: 0.76, z: 0.1 },
            tolerance_m: 0.05
          }],
          stable_steps: 2
        },
        candidates: [
          {
            id: "same-motion-a",
            intent: "候选 A",
            duration_seconds: 0.2,
            contacts: [],
            keyframes: [
              { at_seconds: 0, channels: [] },
              { at_seconds: 0.2, channels: [] }
            ]
          },
          {
            id: "same-motion-b",
            intent: "候选 B",
            duration_seconds: 0.2,
            contacts: [],
            keyframes: [
              { at_seconds: 0, channels: [] },
              { at_seconds: 0.2, channels: [] }
            ]
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

  it("reports repeated invalid physical values on the same action tool", async () => {
    const invoke = vi.fn();
    const plan = createHumanoidActionTools(
      { invoke } as unknown as HumanoidActionInvoker,
      "motion-agent",
      ["plan_whole_body_motion"]
    )[0];
    if (!plan || plan.type !== "function") throw new Error("Motion tool is missing");
    const invalid = JSON.stringify({
      id: "bad-height",
      intent: "lower root",
      duration_seconds: 0.2,
      keyframes: [
        { at_seconds: 0, root_height: 0 },
        { at_seconds: 0.2, root_height: 0 }
      ]
    });
    const details = {
      toolCall: {
        type: "function_call" as const,
        callId: "repeated-invalid-plan",
        name: "plan_whole_body_motion",
        arguments: invalid,
        status: "completed" as const
      }
    };

    await plan.invoke(
      new RunContext({ runId: "repeated-invalid-motion-input" }),
      invalid,
      details
    );
    const output = await plan.invoke(
      new RunContext({ runId: "repeated-invalid-motion-input" }),
      invalid,
      details
    );

    expect(JSON.parse(String(output))).toMatchObject({
      code: "repeated_invalid_tool_input",
      repeated_attempt: {
        count: 2,
        invalid_fields: expect.arrayContaining([{
          path: "keyframes.0.root_height",
          value: 0
        }])
      }
    });
    expect(invoke).not.toHaveBeenCalled();
  });

  it("returns structured recovery for malformed action JSON", async () => {
    const invoke = vi.fn();
    const plan = createHumanoidActionTools(
      { invoke } as unknown as HumanoidActionInvoker,
      "motion-agent",
      ["plan_whole_body_motion_candidates"]
    )[0];
    if (!plan || plan.type !== "function") throw new Error("Motion tool is missing");
    const parameterDefinitions = (
      plan.parameters as Record<string, unknown>
    ).$defs as Record<string, unknown> | undefined;
    expect(JSON.stringify(plan.parameters).length).toBeLessThan(24_000);
    expect(Object.keys(parameterDefinitions ?? {})).not.toHaveLength(0);

    const output = await plan.invoke(
      new RunContext({ runId: "malformed-motion-input" }),
      "{\"objective\":",
      {
        toolCall: {
          type: "function_call",
          callId: "malformed-plan-1",
          name: "plan_whole_body_motion_candidates",
          arguments: "{\"objective\":",
          status: "completed"
        }
      }
    );

    expect(JSON.parse(String(output))).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      tool: "plan_whole_body_motion_candidates",
      validation_issues: [{
        path: "",
        code: "invalid_json"
      }],
      automatic_actuation: false
    });
    expect(invoke).not.toHaveBeenCalled();
  });
});
