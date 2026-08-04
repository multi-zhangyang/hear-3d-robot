import { describe, expect, it } from "vitest";
import type { Goal, HumanoidRunCheckpoint } from "./types";
import {
  activeCheckpointGoal,
  goalSelectionLabel,
  goalSelectionPhase,
  missionCheckpointGoal
} from "./goal-state";

const goal: Goal = {
  summary: "抵达目标",
  predicates: [{ type: "robot_at", target: { x: 1, y: 0, z: 2 }, tolerance: 0.2 }]
};

describe("checkpoint goal projection", () => {
  it("reads legacy active goals without changing them", () => {
    const checkpoint = { version: 5, goal } as HumanoidRunCheckpoint;
    expect(activeCheckpointGoal(checkpoint)).toBe(goal);
    expect(missionCheckpointGoal(checkpoint)).toBe(goal);
  });

  it("does not present a mission constraint as an active model-selected goal", () => {
    const checkpoint = {
      version: 6,
      mission_goal: goal,
      goal_dag: {
        version: 1,
        status: "awaiting_model_selection",
        candidates: {},
        epochs: [],
        current_epoch_id: null,
        next_epoch_index: 0,
        evidence: {},
        state_sha256: "a".repeat(64)
      }
    } as unknown as HumanoidRunCheckpoint;
    expect(activeCheckpointGoal(checkpoint)).toBeNull();
    expect(missionCheckpointGoal(checkpoint)).toBe(goal);
    expect(goalSelectionPhase(checkpoint)).toEqual({
      kind: "candidate_generation",
      candidateCount: 0
    });
    expect(goalSelectionLabel(checkpoint)).toBe("目标管理智能体正在生成候选");
  });

  it("distinguishes submitted candidates from candidate generation", () => {
    const checkpoint = {
      version: 6,
      goal_dag: {
        status: "awaiting_model_selection",
        candidates: {
          first: { status: "proposed" },
          second: { status: "proposed" },
          retired: { status: "completed" }
        }
      }
    } as unknown as HumanoidRunCheckpoint;

    expect(goalSelectionPhase(checkpoint)).toEqual({
      kind: "candidate_selection",
      candidateCount: 2
    });
    expect(goalSelectionLabel(checkpoint)).toBe("目标管理智能体正在选择 · 2 个候选");
  });

  it("resolves the active goal only through the selected epoch and candidate", () => {
    const source = {
      agent_id: "goal-manager",
      agent_manifest_sha256: "a".repeat(64),
      agent_manifest_epoch_id: "00000000-0000-4000-8000-000000000000",
      model_call_id: "00000000-0000-4000-8000-000000000001",
      response_id: "response",
      response_output_sha256: "b".repeat(64),
      tool_call_id: "tool-call",
      tool_arguments_sha256: "c".repeat(64)
    };
    const checkpoint = {
      version: 6,
      mission_goal: goal,
      goal_dag: {
        version: 1,
        status: "active",
        candidates: {
          selected: {
            candidate_id: "selected",
            proposal_id: "proposal",
            source,
            goal,
            mission_link: "mission",
            identity_sha256: "d".repeat(64),
            content_sha256: "e".repeat(64),
            integrity_sha256: "f".repeat(64),
            dependency_candidate_ids: [],
            status: "active",
            physical_evidence_refs: { proposal: ["world:1"], resolution: [] },
            created_world_revision: 1,
            resolved_world_revision: null
          }
        },
        epochs: [{
          epoch_id: "epoch",
          epoch_index: 0,
          previous_epoch_id: null,
          candidate_id: "selected",
          candidate_source: source,
          selected_by: source,
          candidate_identity_sha256: "d".repeat(64),
          candidate_content_sha256: "e".repeat(64),
          dependency_candidate_ids: [],
          identity_sha256: "f".repeat(64),
          status: "active",
          retired_by: null,
          retirement_reason: null,
          physical_evidence_refs: { selection: ["world:1"], resolution: [] },
          created_world_revision: 1,
          resolved_world_revision: null
        }],
        current_epoch_id: "epoch",
        next_epoch_index: 1,
        evidence: {},
        state_sha256: "0".repeat(64)
      }
    } as unknown as HumanoidRunCheckpoint;
    expect(activeCheckpointGoal(checkpoint)).toBe(goal);
    expect(goalSelectionPhase(checkpoint)).toBeNull();
  });
});
