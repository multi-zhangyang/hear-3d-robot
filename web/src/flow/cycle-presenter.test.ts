import { describe, expect, it } from "vitest";
import type {
  HumanoidActionReceipt,
  HumanoidEmbodiedEpisode,
  HumanoidRunCheckpoint
} from "../types";
import { presentAutonomousCycles } from "./cycle-presenter";

const CYCLE = {
  cycle_id: "autonomous-cycle:00000000-0000-4000-8000-000000000003",
  cycle_index: 3,
  goal_epoch_id: `goal-epoch:${"a".repeat(64)}`
} as const;

describe("autonomous cycle presenter", () => {
  it("links perception, selected planning receipt, execution, verification and memory", () => {
    const observation = receipt({
      transactionId: "observe-3",
      action: "observe_humanoid",
      at: "2026-08-03T00:00:00.000Z",
      before: 150,
      after: 150
    });
    const rejected = receipt({
      transactionId: "plan-rejected",
      action: "plan_whole_body_motion",
      accepted: false,
      at: "2026-08-03T00:00:01.000Z",
      before: 150,
      after: 150
    });
    const planning = receipt({
      transactionId: "plan-3",
      action: "plan_whole_body_motion_candidates",
      at: "2026-08-03T00:00:02.000Z",
      before: 150,
      after: 150,
      detail: { candidate_count: 3, selected_rank: 2 }
    });
    const execution = receipt({
      transactionId: "execute-3",
      action: "execute_whole_body_motion",
      at: "2026-08-03T00:00:03.000Z",
      before: 150,
      after: 225,
      input: { planning_transaction_id: "plan-3" },
      frames: 75
    });
    const mutation = receipt({
      transactionId: "remove-3",
      action: "remove_world_block",
      at: "2026-08-03T00:00:03.500Z",
      before: 225,
      after: 225,
      detail: {
        removal_transaction: {
          projected_chunk_revision: 1
        }
      }
    });
    const episode = embodiedEpisode();
    episode.causal_trace!.world_mutation_transaction_ids = ["remove-3"];
    episode.causal_trace!.goal_evidence_refs.push("action:remove-3");
    episode.world_mutations = [{
      transaction_id: "remove-3",
      action: "remove_world_block",
      decision: {
        ...episode.causal_trace!.execution_decision,
        tool_call_id: "remove-3"
      },
      code: "world_block_removal_authorized",
      execution_transaction_id: "execute-3",
      solid_id: "block-a",
      world_before_revision: 225,
      world_after_revision: 225,
      chunk_before_revision: 0,
      chunk_after_revision: 1
    }];
    const checkpoint = checkpointWith([episode], 3, "succeeded", 225);

    const [cycle] = presentAutonomousCycles({
      checkpoint,
      actions: [planning, execution, mutation, observation, rejected],
      framework: []
    });

    expect(cycle).toMatchObject({
      index: 3,
      state: "completed",
      worldBeforeRevision: 150,
      worldAfterRevision: 225,
      goalReached: true
    });
    expect(cycle?.stages.map((stage) => stage.kind)).toEqual([
      "sense", "plan", "execute", "mutate", "verify", "memory"
    ]);
    expect(cycle?.stages[1]).toMatchObject({
      state: "success",
      detail: "3 个模型候选已分别完成 MuJoCo 预演，选择第 2 个可行动作。"
    });
    expect(cycle?.stages[1]?.meta).toContain("1 个候选未通过约束");
    expect(cycle?.stages[3]).toMatchObject({
      state: "success",
      detail: "目标方块已从权威世界移除 · 区块 R1。"
    });
    expect(cycle?.stages[4]?.detail).toContain("满足本轮 Goal");
    expect(JSON.stringify(cycle)).not.toContain("planning_transaction_id");
  });

  it("keeps absent live stages waiting and only shows sanitized model output", () => {
    const checkpoint = checkpointWith([], 0, "running", 8);
    const [cycle] = presentAutonomousCycles({
      checkpoint,
      actions: [],
      framework: [{
        at: "2026-08-03T00:00:01.000Z",
        agent_name: "humanoid-coordinator",
        cycle: {
          ...CYCLE,
          cycle_id: "autonomous-cycle:00000000-0000-4000-8000-000000000001",
          cycle_index: 1
        },
        event: {
          type: "run_item_stream_event",
          name: "message_output_created",
          item: { rawItem: { content: [{
            type: "output_text",
            text: "正在选择下一步动作。 endpoint=https://hidden.invalid model=vendor/private"
          }] } }
        }
      }]
    });

    expect(cycle).toMatchObject({ index: 1, state: "active" });
    expect(cycle?.stages.every((stage) => stage.state === "waiting")).toBe(true);
    expect(cycle?.liveModelOutput?.detail).toBe("正在选择下一步动作。");
    expect(JSON.stringify(cycle)).not.toContain("hidden.invalid");
    expect(JSON.stringify(cycle)).not.toContain("vendor/private");
  });

  it("presents the real Goal selection phase before an active Goal exists", () => {
    const checkpoint = {
      ...checkpointWith([], 0, "running", 8),
      version: 6,
      active_cycle: null,
      goal_dag: {
        status: "awaiting_model_selection",
        candidates: {
          first: { status: "proposed" },
          second: { status: "proposed" }
        }
      }
    } as unknown as HumanoidRunCheckpoint;

    const [cycle] = presentAutonomousCycles({ checkpoint, actions: [], framework: [] });

    expect(cycle?.phaseLabel).toBe("目标管理智能体正在选择 · 2 个候选");
    expect(cycle?.stages.every((stage) => stage.state === "waiting")).toBe(true);
  });

  it("does not invent a new cycle for an already completed run", () => {
    const episode = embodiedEpisode();
    const checkpoint = checkpointWith([episode], 3, "succeeded", 225);

    const cycles = presentAutonomousCycles({
      checkpoint,
      actions: [],
      framework: [{
        at: "2026-08-03T00:00:05.000Z",
        agent_name: "humanoid-coordinator",
        event: {
          type: "run_item_stream_event",
          name: "message_output_created",
          item: { rawItem: { content: [{ type: "output_text", text: "运行已经完成。" }] } }
        }
      }]
    });

    expect(cycles).toHaveLength(1);
    expect(cycles[0]).toMatchObject({ index: 3, state: "completed" });
  });

  it("recognizes autonomous skill planning and execution receipts", () => {
    const planning = receipt({
      transactionId: "skill-plan-3",
      action: "plan_humanoid_skill",
      at: "2026-08-03T00:00:02.000Z",
      before: 150,
      after: 150
    });
    const execution = receipt({
      transactionId: "skill-execute-3",
      action: "execute_humanoid_skill",
      at: "2026-08-03T00:00:03.000Z",
      before: 150,
      after: 225,
      input: { planning_transaction_id: "skill-plan-3" },
      frames: 75
    });
    const episode = {
      ...embodiedEpisode(),
      transaction_id: "skill-execute-3",
      action: "execute_humanoid_skill" as const,
      planning_action: "plan_humanoid_skill" as const,
      causal_trace: {
        ...embodiedEpisode().causal_trace!,
        planning_transaction_id: "skill-plan-3",
        execution_transaction_id: "skill-execute-3"
      }
    };

    const [cycle] = presentAutonomousCycles({
      checkpoint: checkpointWith([episode], 3, "succeeded", 225),
      actions: [planning, execution],
      framework: []
    });

    expect(cycle?.stages[1]).toMatchObject({ state: "success" });
    expect(cycle?.stages[2]).toMatchObject({ state: "success" });
  });
});

function receipt(input: {
  transactionId: string;
  action: HumanoidActionReceipt["action"];
  at: string;
  before: number;
  after: number;
  accepted?: boolean;
  input?: unknown;
  detail?: unknown;
  frames?: number;
}): HumanoidActionReceipt {
  return {
    transactionId: input.transactionId,
    agentId: input.action === "observe_humanoid" ? "humanoid-sentry" : "humanoid-motion-reference",
    cycle: CYCLE,
    action: input.action,
    input: input.input ?? {},
    fingerprint: `${input.transactionId}-fingerprint`,
    accepted: input.accepted ?? true,
    code: input.accepted === false ? "whole_body_plan_rejected" : input.action === "observe_humanoid"
      ? "humanoid_observed" : input.action === "remove_world_block"
        ? "world_block_removal_authorized" : input.action.startsWith("execute")
          ? "motion_completed" : "whole_body_candidates_validated",
    worldBeforeRevision: input.before,
    worldAfterRevision: input.after,
    frameCount: input.frames ?? 0,
    channels: [],
    detail: input.detail ?? {},
    committedAt: input.at
  };
}

function embodiedEpisode(): HumanoidEmbodiedEpisode {
  return {
    sequence: 3,
    causal_trace: {
      cycle: CYCLE,
      planning_transaction_id: "plan-3",
      execution_transaction_id: "execute-3",
      execution_decision: {
        agent_id: "humanoid-executor",
        agent_manifest_sha256: "b".repeat(64),
        agent_manifest_epoch_id: "00000000-0000-4000-8000-000000000001",
        model_call_id: "00000000-0000-4000-8000-000000000002",
        response_id: "response-3",
        response_output_sha256: "c".repeat(64),
        tool_call_id: "execute-3",
        tool_arguments_sha256: "d".repeat(64)
      },
      goal_evidence_refs: ["action:execute-3"],
      memory_id: "embodied-memory:00000000-0000-4000-8000-000000000003"
    },
    transaction_id: "execute-3",
    action: "execute_whole_body_motion",
    planning_action: "plan_whole_body_motion_candidates",
    candidate_count: 3,
    selected_rank: 2,
    code: "motion_completed",
    model_summary: "抵达目标并保持直立。",
    world_before_revision: 150,
    world_after_revision: 225,
    frame_count: 75,
    result_frame: 300,
    result_root_position: { x: 2.5, y: 0.76, z: 4.5 },
    fallen: false,
    support: "double",
    upright: 0.99,
    goal_success: true,
    recorded_at: "2026-08-03T00:00:04.000Z"
  };
}

function checkpointWith(
  episodes: HumanoidEmbodiedEpisode[],
  cycleIndex: number,
  status: HumanoidRunCheckpoint["status"],
  worldRevision: number
): HumanoidRunCheckpoint {
  return {
    status,
    cycle_index: cycleIndex,
    active_cycle: status === "running" ? {
      ...CYCLE,
      cycle_index: cycleIndex + 1,
      cycle_id: `autonomous-cycle:00000000-0000-4000-8000-${String(cycleIndex + 1).padStart(12, "0")}`,
      started_world_frame: 0,
      started_world_revision: worldRevision,
      started_at: "2026-08-03T00:00:00.000Z"
    } : null,
    world: { worldRevision },
    embodied_memory: { recent_episodes: episodes }
  } as unknown as HumanoidRunCheckpoint;
}
