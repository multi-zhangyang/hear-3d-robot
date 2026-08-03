import { describe, expect, it } from "vitest";
import { EmptyHumanoidEmbodiedMemoryState } from "../../domain/humanoid-run.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  appendEmbodiedEpisode,
  MAX_CHECKPOINT_ACTION_RECEIPTS,
  MAX_RECENT_EMBODIED_EPISODES,
  recentEmbodiedEpisodes,
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

  it("refuses rejected or stale receipts as embodied memory", () => {
    const input = episodeInput(1);
    input.execution.accepted = false;
    expect(() => appendEmbodiedEpisode(input)).toThrow(
      "current accepted execution evidence"
    );

    input.execution.accepted = true;
    input.execution.worldAfterRevision = 0;
    expect(() => appendEmbodiedEpisode(input)).toThrow(
      "current accepted execution evidence"
    );
  });

  it("retains physically selected candidate evidence across context compaction", () => {
    const input = episodeInput(1);
    input.execution.code = "motion_option_succeeded";
    input.execution.detail = {
      planning_action: "plan_whole_body_motion_candidates",
      candidate_count: 3,
      selected_rank: 2,
      selected_candidate_id: "balanced-candidate",
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
      candidate_count: 3,
      selected_rank: 2,
      selected_candidate_id: "balanced-candidate",
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
    expect(Object.keys(retained.receipts)[0]).toBe("receipt-9");
    expect(Object.keys(receipts)).toHaveLength(40);
  });
});

function episodeInput(sequence: number): Parameters<typeof appendEmbodiedEpisode>[0] {
  const at = new Date(Date.UTC(2026, 0, 1, 0, sequence)).toISOString();
  const execution = {
    transactionId: `execution-${sequence}`,
    accepted: true,
    action: "execute_whole_body_motion",
    code: "motion_completed",
    worldBeforeRevision: sequence - 1,
    worldAfterRevision: sequence,
    frameCount: 10,
    committedAt: at
  } as HumanoidActionReceipt;
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
    goalSuccess: sequence === 70
  };
}
