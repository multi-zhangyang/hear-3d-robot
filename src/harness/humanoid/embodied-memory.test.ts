import { describe, expect, it } from "vitest";
import { EmptyHumanoidEmbodiedMemoryState } from "../../domain/humanoid-run.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  appendEmbodiedEpisode,
  MAX_CHECKPOINT_ACTION_RECEIPTS,
  MAX_RECENT_EMBODIED_EPISODES,
  MAX_RECENT_EMBODIED_EXPERIENCES,
  recentEmbodiedEpisodes,
  recentEmbodiedExperiences,
  rememberEmbodiedActionExperience,
  retainRecentActionReceipts
} from "./embodied-memory.js";

describe("humanoid multi-scale embodied memory", () => {
  it("keeps a bounded physical episode window while retaining lifetime counts", () => {
    let state = structuredClone(EmptyHumanoidEmbodiedMemoryState);
    for (let sequence = 1; sequence <= 70; sequence += 1) {
      const input = episodeInput(sequence);
      input.state = state;
      state = appendEmbodiedEpisode(input).state;
    }

    expect(state.total_episodes).toBe(70);
    expect(state.pruned_episodes).toBe(70 - MAX_RECENT_EMBODIED_EPISODES);
    expect(state.recent_episodes).toHaveLength(MAX_RECENT_EMBODIED_EPISODES);
    expect(state.recent_episodes[0]!.sequence).toBe(7);
    expect(state.recent_episodes.at(-1)!.sequence).toBe(70);
    expect(recentEmbodiedEpisodes(state)).toHaveLength(12);
    expect(recentEmbodiedEpisodes(state)[0]!.sequence).toBe(59);
  });

  it("refuses rejected or future receipts as embodied memory", () => {
    const input = episodeInput(1);
    input.execution.accepted = false;
    expect(() => appendEmbodiedEpisode(input)).toThrow(
      "current accepted execution evidence"
    );

    input.execution.accepted = true;
    input.execution.worldAfterRevision = input.world.worldRevision + 1;
    expect(() => appendEmbodiedEpisode(input)).toThrow(
      "current accepted execution evidence"
    );
  });

  it("binds an evidence-authorized world mutation to its physical episode", () => {
    const input = episodeInput(1);
    const transactionId = "remove-block-1";
    const mutation = {
      transactionId,
      agentId: input.execution.agentId,
      decision: {
        ...input.execution.decision!,
        tool_call_id: transactionId,
        tool_arguments_sha256: "e".repeat(64)
      },
      cycle: input.cycle,
      accepted: true,
      action: "remove_world_block",
      input: {
        solid_id: "block-a",
        execution_transaction_id: input.execution.transactionId
      },
      fingerprint: "remove-block-1-fingerprint",
      code: "world_block_removal_authorized",
      worldBeforeRevision: input.execution.worldAfterRevision,
      worldAfterRevision: input.execution.worldAfterRevision,
      frameCount: 0,
      channels: [],
      detail: {
        solid_id: "block-a",
        execution_transaction_id: input.execution.transactionId,
        removal_transaction: {
          transaction_id: transactionId,
          execution_transaction_id: input.execution.transactionId,
          solid_id: "block-a",
          base_chunk_revision: 3,
          projected_chunk_revision: 4
        }
      },
      committedAt: "2026-01-01T00:01:01.000Z"
    } satisfies HumanoidActionReceipt;
    input.worldMutations = [mutation];
    input.goalEvidenceRefs.push(`action:${transactionId}`);

    expect(appendEmbodiedEpisode(input).episode).toMatchObject({
      causal_trace: {
        execution_transaction_id: input.execution.transactionId,
        world_mutation_transaction_ids: [transactionId],
        goal_evidence_refs: [
          `action:${input.execution.transactionId}`,
          `action:${transactionId}`
        ]
      },
      result_world_revision: input.world.worldRevision,
      world_mutations: [{
        transaction_id: transactionId,
        action: "remove_world_block",
        execution_transaction_id: input.execution.transactionId,
        solid_id: "block-a",
        chunk_before_revision: 3,
        chunk_after_revision: 4
      }]
    });

    input.goalEvidenceRefs.pop();
    expect(() => appendEmbodiedEpisode(input)).toThrow(
      "durable evidence for every world mutation"
    );
  });

  it("retains physically selected candidate evidence across context compaction", () => {
    const input = episodeInput(1);
    input.execution.code = "motion_option_succeeded";
    input.execution.detail = {
      planning_action: "plan_whole_body_motion_candidates",
      candidate_count: 1,
      selected_rank: 1,
      selected_candidate_id: "precise-candidate",
      result: {
        option: {
          option_id: "reach-target",
          status: "succeeded",
          termination_reason: "physical_success",
          full_frame_count: 100,
          executed_prefix_frame_count: 31,
          predicted_termination_frame: 33,
          actual_termination_frame: 31,
          artifact_sha256: "a".repeat(64)
        }
      }
    };

    expect(appendEmbodiedEpisode(input).episode).toMatchObject({
      planning_action: "plan_whole_body_motion_candidates",
      candidate_count: 1,
      selected_rank: 1,
      selected_candidate_id: "precise-candidate",
      motion_option: {
        option_id: "reach-target",
        status: "succeeded",
        actual_termination_frame: 31,
        predicted_termination_frame: 33,
        artifact_sha256: "a".repeat(64)
      }
    });
  });

  it("bounds the hot checkpoint receipt window without changing append-only evidence", () => {
    const receipts = Object.fromEntries(Array.from({ length: 40 }, (_, index) => (
      [`receipt-${index + 1}`, { sequence: index + 1 }]
    )));
    const retained = retainRecentActionReceipts(receipts);

    expect(retained.removed).toBe(40 - MAX_CHECKPOINT_ACTION_RECEIPTS);
    expect(Object.keys(retained.receipts)).toHaveLength(MAX_CHECKPOINT_ACTION_RECEIPTS);
    expect(Object.keys(retained.receipts)[0]).toBe(
      `receipt-${40 - MAX_CHECKPOINT_ACTION_RECEIPTS + 1}`
    );
    expect(Object.keys(receipts)).toHaveLength(40);
  });

  it("indexes successful, rejected and physically failed model actions by Goal semantics", () => {
    let state = structuredClone(EmptyHumanoidEmbodiedMemoryState);
    const goal = {
      summary: "把可动物体放入目标区域",
      predicates: [{
        type: "object_placed" as const,
        object_id: "crate",
        zone_id: "green-zone",
        tolerance: 0.05
      }]
    };
    const succeeded = episodeInput(1).execution;
    let remembered = rememberEmbodiedActionExperience({
      state,
      execution: succeeded,
      goal
    });
    state = remembered.state;
    expect(remembered.created).toBe(true);

    const duplicate = rememberEmbodiedActionExperience({
      state,
      execution: succeeded,
      goal
    });
    expect(duplicate.created).toBe(false);
    expect(duplicate.state.total_experiences).toBe(1);

    const rejected = episodeInput(2).execution;
    rejected.accepted = false;
    rejected.frameCount = 0;
    rejected.code = "plan_stale";
    state = rememberEmbodiedActionExperience({ state, execution: rejected, goal }).state;

    const failed = episodeInput(3).execution;
    failed.accepted = false;
    failed.frameCount = 7;
    failed.code = "motion_goal_unmet";
    state = rememberEmbodiedActionExperience({ state, execution: failed, goal }).state;

    expect(state).toMatchObject({
      version: 2,
      total_experiences: 3,
      outcome_counts: {
        succeeded: 1,
        rejected: 1,
        physically_failed: 1
      },
      object_outcome_counts: {
        crate: {
          succeeded: 1,
          rejected: 1,
          physically_failed: 1
        }
      },
      zone_outcome_counts: {
        "green-zone": {
          succeeded: 1,
          rejected: 1,
          physically_failed: 1
        }
      }
    });
    expect(recentEmbodiedExperiences(state)).toHaveLength(3);
    expect(MAX_RECENT_EMBODIED_EXPERIENCES).toBe(128);
  });

  it("indexes a world mutation by its sensed solid identity", () => {
    const mutation = episodeInput(1).execution;
    mutation.action = "remove_world_block";
    mutation.input = {
      solid_id: "block-a",
      execution_transaction_id: "execution-0"
    };
    mutation.code = "world_block_removal_authorized";
    mutation.frameCount = 0;
    mutation.detail = {
      solid_id: "block-a",
      execution_transaction_id: "execution-0"
    };
    const remembered = rememberEmbodiedActionExperience({
      state: structuredClone(EmptyHumanoidEmbodiedMemoryState),
      execution: mutation,
      goal: {
        summary: "拆除当前观察到的方块",
        predicates: [{ type: "block_removed", block_id: "block-a" }]
      }
    });

    expect(remembered.experience).toMatchObject({
      action: "remove_world_block",
      outcome: "succeeded",
      predicate_types: ["block_removed"],
      object_ids: [],
      solid_ids: ["block-a"]
    });
  });

  it("indexes successful collision recovery by the contacted solid", () => {
    const execution = episodeInput(1).execution;
    execution.detail = {
      planning_action: "plan_whole_body_motion_candidates",
      planning_transaction_id: "planning-1",
      recovery_kind: "navigation_transit_clearance",
      recovery_collision: {
        hand_surface: "right_hand_index_1_link",
        target_kind: "solid",
        target_id: "stone_column",
        contact_point_world: { x: 1, y: 0.8, z: 1 },
        separation_normal_world: { x: -1, y: 0, z: 0 }
      }
    };
    const remembered = rememberEmbodiedActionExperience({
      state: structuredClone(EmptyHumanoidEmbodiedMemoryState),
      execution,
      goal: {
        summary: "继续探索当前区域",
        predicates: [{
          type: "robot_at",
          target: { x: 4, y: 0, z: 5 },
          tolerance: 0.2
        }]
      }
    });

    expect(remembered.experience.solid_ids).toEqual(["stone_column"]);
    expect(remembered.state.solid_outcome_counts).toEqual({
      stone_column: {
        succeeded: 1,
        rejected: 0,
        physically_failed: 0
      }
    });
  });
});

function episodeInput(sequence: number): Parameters<typeof appendEmbodiedEpisode>[0] {
  const at = new Date(Date.UTC(2026, 0, 1, 0, sequence)).toISOString();
  const transactionId = `execution-${sequence}`;
  const cycle = {
    cycle_id: `autonomous-cycle:00000000-0000-4000-8000-${String(sequence).padStart(12, "0")}`,
    cycle_index: sequence,
    goal_epoch_id: `goal-epoch:${"a".repeat(64)}`
  } as const;
  const execution = {
    transactionId,
    agentId: "humanoid-executor",
    decision: {
      agent_id: "humanoid-executor",
      agent_manifest_sha256: "b".repeat(64),
      agent_manifest_epoch_id: "00000000-0000-4000-8000-000000000001",
      model_call_id: `10000000-0000-4000-8000-${String(sequence).padStart(12, "0")}`,
      response_id: `response-${sequence}`,
      response_output_sha256: "c".repeat(64),
      tool_call_id: transactionId,
      tool_arguments_sha256: "d".repeat(64)
    },
    cycle,
    accepted: true,
    action: "execute_whole_body_motion",
    input: { planning_transaction_id: `planning-${sequence}` },
    fingerprint: `execution-${sequence}-fingerprint`,
    code: "motion_completed",
    worldBeforeRevision: sequence - 1,
    worldAfterRevision: sequence,
    frameCount: 10,
    channels: ["locomotion"],
    detail: {
      planning_action: "plan_whole_body_motion",
      planning_transaction_id: `planning-${sequence}`
    },
    committedAt: at
  } satisfies HumanoidActionReceipt;
  const world = {
    frame: sequence * 10,
    worldRevision: sequence,
    robot: {
      rootPosition: { x: 0, y: 0.76, z: sequence / 10 },
      fallen: false,
      balance: { support: "double", upright: 1 }
    }
  } as HumanoidWorldSnapshot;
  return {
    state: structuredClone(EmptyHumanoidEmbodiedMemoryState),
    sequence,
    execution,
    modelSummary: `自主循环 ${sequence}`,
    world,
    goalSuccess: sequence === 70,
    cycle,
    goalEvidenceRefs: [`action:${transactionId}`]
  };
}
