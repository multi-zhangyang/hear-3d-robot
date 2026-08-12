import { describe, expect, it, vi } from "vitest";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import type {
  HumanoidExecutionReceipt,
  HumanoidWorld
} from "../../world/humanoid/world.js";
import { HumanoidSkillEventStream } from
  "../../world/humanoid/skill-event-stream.js";
import {
  HUMANOID_ARTICULATION_HORIZON,
  humanoidArticulationSegmentBudgetExhausted
} from "./articulation-control.js";
import { executeHumanoidArticulationHorizon } from
  "./articulation-horizon-executor.js";
import { ActiveHumanoidSkillBindingSchema, humanoidEmbodiedSkillIdentity } from
  "./skill-binding.js";

describe("humanoid articulation horizon", () => {
  it("counts the durable prefix in the bounded segment budget", () => {
    expect(humanoidArticulationSegmentBudgetExhausted(31, 0)).toBe(false);
    expect(humanoidArticulationSegmentBudgetExhausted(31, 1)).toBe(true);
    expect(humanoidArticulationSegmentBudgetExhausted(32, 0)).toBe(true);
    expect(() => humanoidArticulationSegmentBudgetExhausted(-1, 0)).toThrow(
      "non-negative integer"
    );
  });

  it("does not exceed the durable segment budget after a process restart", async () => {
    const invocation = {
      skill: "open" as const,
      object_id: "cabinet-door",
      interaction_point_id: "door-handle",
      joint_id: "cabinet-hinge",
      hand: "right" as const,
      minimum_open_fraction: 0.8
    };
    const articulation = {
      joint_id: "cabinet-hinge",
      parent_object_id: "cabinet",
      type: "hinge" as const,
      semantic: "cabinet_door",
      axis_world: { x: 0, y: 1, z: 0 },
      anchor_world: { x: 0, y: 1, z: 0 },
      position: 0.4,
      velocity: 0,
      range: { minimum: 0, maximum: 1.5 },
      closed_position: 0,
      open_position: 1.5,
      open_fraction: 0.4 / 1.5,
      state: "intermediate" as const
    };
    const observation = {
      frame: 12,
      worldRevision: 7,
      interaction: {
        object_world_model: {
          objects: [{ id: "cabinet-door", articulation }]
        }
      }
    } as unknown as ReturnType<HumanoidWorld["observe"]>;
    const binding = ActiveHumanoidSkillBindingSchema.parse({
      protocol: "humanoid-active-skill-v1",
      transaction_id: "durable-articulation-call",
      agent_id: "humanoid-motion-reference",
      skill_plan_transaction_id: "durable-skill-plan",
      skill_node_id: "durable-skill-node",
      invocation,
      invocation_sha256: modelPayloadSha256(invocation),
      phase: "actuate_joint",
      phase_authority: "whole_body",
      planning_action: "plan_humanoid_skill",
      observed_frame: observation.frame,
      observed_world_revision: observation.worldRevision,
      skill_catalog_sha256: "a".repeat(64),
      target_position: null,
      target_solid: null,
      target_articulation: articulation,
      eligible_interaction_points: [],
      eligible_interaction_point_ids: [],
      learned_policy_required_capabilities: [],
      learned_policy_missing_capabilities: [],
      control_mode: "reference_control_fallback"
    });

    const finalSnapshot = {
      frame: observation.frame,
      worldRevision: observation.worldRevision,
      robot: { controllerExecution: null }
    } as unknown as HumanoidExecutionReceipt["finalSnapshot"];
    const executeWholeBodyMotion = vi.fn();
    const recordSkillOutcome = vi.fn();
    const events: unknown[] = [];
    const result = await executeHumanoidArticulationHorizon({
      world: {
        observe: () => observation,
        snapshot: () => finalSnapshot,
        pendingWholeBodyMotionPlanIdForSkillCall: () => undefined,
        planWholeBodyMotionCandidates: vi.fn(),
        executeWholeBodyMotion,
        recordSkillOutcome
      } as unknown as HumanoidWorld,
      binding,
      initialPlanId: null,
      initialExecution: {
        accepted: true,
        code: "motion_option_succeeded",
        frames: 0,
        finalSnapshot,
        detail: {}
      },
      skillEventStream: new HumanoidSkillEventStream(
        humanoidEmbodiedSkillIdentity(binding),
        (event) => events.push(event),
        {
          nextSequence: 1,
          accepted: true,
          lastProgress: 0,
          lastStableSteps: 0,
          terminal: false
        }
      ),
      initialCommittedFrames: 640,
      initialCompletedSegments: HUMANOID_ARTICULATION_HORIZON.maximum_segments
    });

    expect(result).toMatchObject({
      accepted: false,
      code: "articulation_horizon_exhausted",
      frames: 640,
      detail: {
        articulation_horizon: {
          segment_count: HUMANOID_ARTICULATION_HORIZON.maximum_segments,
          completed_segment_prefix_count:
            HUMANOID_ARTICULATION_HORIZON.maximum_segments,
          segments: []
        }
      }
    });
    expect(executeWholeBodyMotion).not.toHaveBeenCalled();
    expect(recordSkillOutcome).toHaveBeenCalledTimes(1);
    expect(events).toMatchObject([{ sequence: 1, type: "failed" }]);
  });
});
