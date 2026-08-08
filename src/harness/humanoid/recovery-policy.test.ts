import { describe, expect, it } from "vitest";
import {
  createHumanoidRecoveryPolicy,
  humanoidRecoverySelectionAccepted
} from "./recovery-policy.js";
import type { ActiveHumanoidSkillBinding } from "./skill-binding.js";

const binding = {
  protocol: "humanoid-active-skill-v1",
  transaction_id: "skill-1",
  agent_id: "humanoid-motion-reference",
  skill_plan_transaction_id: null,
  skill_node_id: null,
  invocation: {
    skill: "open",
    object_id: "door",
    interaction_point_id: "handle",
    joint_id: "hinge",
    hand: "right",
    minimum_open_fraction: 0.85
  },
  invocation_sha256: "a".repeat(64),
  phase: "actuate_joint",
  phase_authority: "whole_body",
  planning_action: "plan_whole_body_motion_candidates",
  observed_frame: 2,
  observed_world_revision: 3,
  skill_catalog_sha256: "b".repeat(64),
  target_position: { x: 1, y: 1, z: 1 },
  target_articulation: null,
  eligible_interaction_points: [],
  eligible_interaction_point_ids: ["handle"]
} as const satisfies ActiveHumanoidSkillBinding;

describe("humanoid recovery policy", () => {
  it("prioritizes failure-aware recovery without selecting or actuating one", () => {
    const policy = createHumanoidRecoveryPolicy({
      executionTransactionId: "execution-1",
      planningTransactionId: "planning-1",
      physicalFailureCode: "articulation_stalled",
      worldRevision: 4,
      binding
    });
    expect(policy).toMatchObject({
      failure_reason: "articulation_stalled",
      candidate_skills: ["regrasp", "approach", "stabilize", "pull", "push", "retreat"],
      requires_model_selection: true,
      automatic_actuation: false
    });
    expect(humanoidRecoverySelectionAccepted(policy, {
      skill: "stabilize",
      minimum_support_margin_m: 0.02
    })).toBe(true);
    expect(humanoidRecoverySelectionAccepted(policy, {
      skill: "press",
      object_id: "door",
      interaction_point_id: null,
      hand: "right",
      travel_m: 0.05
    })).toBe(false);
  });
});
