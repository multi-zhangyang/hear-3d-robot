import { describe, expect, it } from "vitest";
import { modelPayloadSha256 } from "../domain/model-call-authority.js";
import {
  ActiveHumanoidSkillBindingSchema,
  humanoidEmbodiedSkillIdentity
} from "../harness/humanoid/skill-binding.js";
import {
  createHarnessSkillRolloutRecords
} from "./harness-rollout-dataset.js";

describe("Harness Skill rollout dataset", () => {
  it("joins a durable semantic Skill Call across planning, execution and events", () => {
    const invocation = {
      skill: "stabilize" as const,
      minimum_support_margin_m: 0.03
    };
    const binding = ActiveHumanoidSkillBindingSchema.parse({
      protocol: "humanoid-active-skill-v1",
      transaction_id: "binding-1",
      agent_id: "humanoid-motion-reference",
      skill_plan_transaction_id: "skill-plan-1",
      skill_node_id: "stabilize-now",
      invocation,
      invocation_sha256: modelPayloadSha256(invocation),
      phase: "recover_support",
      phase_authority: "whole_body",
      planning_action: "plan_humanoid_skill",
      observed_frame: 12,
      observed_world_revision: 12,
      skill_catalog_sha256: "a".repeat(64),
      target_position: null,
      target_solid: null,
      target_articulation: null,
      eligible_interaction_points: [],
      eligible_interaction_point_ids: [],
      learned_policy_required_capabilities: ["balance"],
      learned_policy_missing_capabilities: [],
      control_mode: "learned_policy"
    });
    const identity = humanoidEmbodiedSkillIdentity(binding);
    const acceptedStatus = {
      protocol: "humanoid-embodied-skill-status-v1" as const,
      callId: identity.callId,
      state: "accepted" as const,
      progress: {
        elapsedRatio: 0,
        physicalCompletionRatio: 0,
        satisfiedPredicateRatio: 0,
        stableSteps: 0,
        requiredStableSteps: 4
      },
      confidence: {
        value: 1,
        basis: "observable_contract_evidence" as const
      },
      failure: null,
      recoverability: "not_applicable" as const,
      worldFrame: 12,
      worldRevision: 12,
      controller: {
        mode: "learned_policy" as const,
        implementation: "teacher-policy"
      }
    };
    const succeededStatus = {
      ...acceptedStatus,
      state: "succeeded" as const,
      progress: {
        ...acceptedStatus.progress,
        elapsedRatio: 1,
        physicalCompletionRatio: 1,
        satisfiedPredicateRatio: 1,
        stableSteps: 4
      },
      worldFrame: 20,
      worldRevision: 20
    };
    const actions = [
      receipt(0, {
        transactionId: "planning-1",
        agentId: binding.agent_id,
        action: "plan_humanoid_skill",
        input: { skill_transaction_id: binding.transaction_id },
        accepted: true,
        code: "autonomous_skill_motion_validated",
        worldBeforeRevision: 12,
        worldAfterRevision: 12,
        frameCount: 0,
        channels: ["left_leg", "right_leg", "torso"],
        detail: {
          skill_binding: binding,
          autonomous_plan_kind: "motion",
          plan_id: "motion-plan-1",
          intent_sha256: "b".repeat(64)
        }
      }),
      receipt(1, {
        transactionId: "execution-1",
        agentId: "humanoid-physics-executor",
        action: "execute_humanoid_skill",
        input: { planning_transaction_id: "planning-1" },
        accepted: true,
        code: "motion_option_succeeded",
        worldBeforeRevision: 12,
        worldAfterRevision: 20,
        frameCount: 8,
        channels: ["left_leg", "right_leg", "torso"],
        detail: {
          planning_transaction_id: "planning-1",
          plan_id: "motion-plan-1",
          result: {
            skill_status: succeededStatus,
            controller_routing: null
          }
        }
      })
    ];
    const records = createHarnessSkillRolloutRecords({
      run: {
        version: 1,
        run_id: "run-1",
        mission: "recover stable support",
        scenario_id: "workyard",
        created_at: "2026-08-11T00:00:00.000Z"
      },
      actions,
      events: [
        skillEvent(0, "accepted-event", {
          protocol: "humanoid-embodied-skill-event-v1",
          sequence: 0,
          type: "accepted",
          status: acceptedStatus
        }),
        skillEvent(1, "succeeded-event", {
          protocol: "humanoid-embodied-skill-event-v1",
          sequence: 1,
          type: "succeeded",
          status: succeededStatus
        })
      ]
    });

    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({
      identity: { callId: identity.callId },
      semantic_command: invocation,
      declared_capabilities: ["balance"],
      planning_attempts: [{ transaction_id: "planning-1", accepted: true }],
      execution: {
        transaction_id: "execution-1",
        terminal_status: { state: "succeeded" }
      },
      outcome: {
        category: "success",
        base: "success",
        is_recovery: false
      },
      dense_policy_rollout: { available: false }
    });
  });
});

function receipt(index: number, input: Record<string, unknown>) {
  return {
    index,
    value: {
      fingerprint: `fingerprint-${index}`,
      committedAt: `2026-08-11T00:00:0${index}.000Z`,
      ...input
    }
  };
}

function skillEvent(index: number, eventId: string, event: unknown) {
  return {
    index,
    value: {
      event_id: eventId,
      run_id: "run-1",
      type: "humanoid_skill_event",
      at: `2026-08-11T00:00:1${index}.000Z`,
      data: event
    }
  };
}
