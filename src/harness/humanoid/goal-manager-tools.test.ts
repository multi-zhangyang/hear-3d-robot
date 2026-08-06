import { RunContext } from "@openai/agents";
import { describe, expect, it, vi } from "vitest";
import {
  modelPayloadSha256,
  modelToolArgumentsSha256
} from "../../domain/model-call-authority.js";
import {
  createGoalManagerTools,
  type GoalManagerRuntime
} from "./goal-manager-tools.js";

describe("Goal Manager SDK tools", () => {
  it("rejects duplicate predicates before a candidate reaches the Goal runtime", async () => {
    const submitGoalCandidates = vi.fn();
    const runtime = {
      recallGoalHistory: vi.fn(),
      submitGoalCandidates,
      selectGoalCandidate: vi.fn(),
      retireGoalEpoch: vi.fn()
    } as unknown as GoalManagerRuntime;
    const submit = createGoalManagerTools(runtime).find((entry) => (
      entry.name === "submit_goal_candidates"
    ));
    if (!submit) throw new Error("Goal candidate tool is missing");
    const repeated = {
      type: "object_at",
      object_id: "assembly_rod",
      target: { x: 4.2, y: 0.67, z: 4.8 },
      tolerance: 0.05
    };
    const input = JSON.stringify({
      candidates: [{
        proposal_id: "duplicate-predicate",
        mission_link: "测试重复谓词拒绝",
        goal: {
          summary: "重复谓词候选",
          predicates: [repeated, repeated]
        },
        dependency_candidate_ids: []
      }, {
        proposal_id: "distinct-candidate",
        mission_link: "测试替代候选",
        goal: {
          summary: "不同候选",
          predicates: [{
            type: "object_grasped",
            object_id: "assembly_rod",
            hand: "left"
          }]
        },
        dependency_candidate_ids: []
      }]
    });

    const result = await submit.invoke(
      new RunContext({ runId: "goal-duplicate-predicate" }),
      input,
      {
        toolCall: {
          type: "function_call",
          callId: "goal-call-duplicate",
          name: "submit_goal_candidates",
          arguments: input,
          status: "completed"
        }
      }
    );

    expect(result).toContain("cannot repeat an identical predicate");
    expect(submitGoalCandidates).not.toHaveBeenCalled();
  });

  it("preserves raw response authority when the SDK normalizes nullable optionals", async () => {
    const submitGoalCandidates = vi.fn(async () => ({
      status: "goal_candidates_submitted"
    }));
    const runtime = {
      recallGoalHistory: vi.fn(),
      submitGoalCandidates,
      selectGoalCandidate: vi.fn(),
      retireGoalEpoch: vi.fn()
    } as unknown as GoalManagerRuntime;
    const submit = createGoalManagerTools(runtime).find((entry) => (
      entry.name === "submit_goal_candidates"
    ));
    if (!submit) throw new Error("Goal candidate tool is missing");
    const rawInput = {
      candidates: [{
        proposal_id: "approach-left",
        mission_link: "建立左腕接近条件",
        goal: {
          summary: "左腕接近目标",
          predicates: [{
            type: "end_effector_at",
            end_effector: "left_wrist",
            frame: "world",
            target: { x: 4.2, y: 0.67, z: 4.8 },
            tolerance: 0.15,
            stable_frames: 8,
            orientation: null,
            orientation_tolerance_rad: null
          }]
        },
        dependency_candidate_ids: []
      }, {
        proposal_id: "approach-root",
        mission_link: "建立全身接近条件",
        goal: {
          summary: "机器人接近目标",
          predicates: [{
            type: "robot_at",
            target: { x: 4.2, y: 0.67, z: 4.8 },
            tolerance: 0.2
          }]
        },
        dependency_candidate_ids: []
      }]
    };
    const rawArguments = JSON.stringify(rawInput);

    await submit.invoke(
      new RunContext({ runId: "goal-raw-authority" }),
      rawArguments,
      {
        toolCall: {
          type: "function_call",
          callId: "goal-call-normalized",
          name: "submit_goal_candidates",
          arguments: rawArguments,
          status: "completed"
        }
      }
    );

    expect(submitGoalCandidates).toHaveBeenCalledTimes(1);
    const [normalizedInput, authority] = submitGoalCandidates.mock.calls[0]!;
    expect(normalizedInput.candidates[0]!.goal.predicates[0]).not.toHaveProperty(
      "orientation"
    );
    expect(authority).toEqual({
      tool_call_id: "goal-call-normalized",
      tool_name: "submit_goal_candidates",
      arguments_sha256: modelToolArgumentsSha256(rawArguments),
      normalized_arguments_sha256: modelPayloadSha256(normalizedInput)
    });
    expect(authority.normalized_arguments_sha256).not.toBe(
      authority.arguments_sha256
    );
  });
});
