import { describe, expect, it } from "vitest";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  modelToolReceiptDetail,
  recentReceiptContext
} from "./receipt-context.js";

describe("humanoid receipt context", () => {
  it("returns current embodied geometry to the Agent that requested observation", () => {
    const receipt: HumanoidActionReceipt = {
      transactionId: "observe-motion-1",
      agentId: "humanoid-motion-reference",
      action: "observe_humanoid",
      input: {},
      fingerprint: "fingerprint",
      accepted: true,
      code: "humanoid_observed",
      worldBeforeRevision: 22,
      worldAfterRevision: 22,
      frameCount: 0,
      channels: [],
      detail: {
        frame: 22,
        world_revision: 22,
        root: { position: { x: 4, y: 0.76, z: 4.2 } },
        end_effectors: {
          left_wrist: { world_position: { x: 4.2, y: 0.75, z: 4.3 } }
        },
        manipulation_geometry: {
          objects: [{
            object_id: "assembly_rod",
            hands: {
              left: {
                interaction_alignments: [{
                  interaction_point_id: "assembly_rod:grasp:0",
                  hand_surface: "left_hand_palm_link",
                  wrist_world_target: {
                    x: 4.18,
                    y: 0.72,
                    z: 4.74
                  }
                }]
              }
            }
          }]
        },
        joints: { hidden: true }
      },
      committedAt: "2026-08-06T00:00:00.000Z"
    };

    const projected = modelToolReceiptDetail(receipt);

    expect(projected).toMatchObject({
      root: { position: { y: 0.76 } },
      manipulation_geometry: {
        objects: [{ object_id: "assembly_rod" }]
      }
    });
    expect(projected.joints).toBeUndefined();
  });

  it("keeps Sentry observation output bounded for the Coordinator", () => {
    const receipt: HumanoidActionReceipt = {
      transactionId: "observe-sentry-1",
      agentId: "humanoid-sentry",
      action: "observe_humanoid",
      input: {},
      fingerprint: "fingerprint",
      accepted: true,
      code: "humanoid_observed",
      worldBeforeRevision: 22,
      worldAfterRevision: 22,
      frameCount: 0,
      channels: [],
      detail: {
        frame: 22,
        world_revision: 22,
        manipulation_geometry: { objects: [{ object_id: "assembly_rod" }] }
      },
      committedAt: "2026-08-06T00:00:00.000Z"
    };

    expect(modelToolReceiptDetail(receipt)).toEqual({
      frame: 22,
      world_revision: 22
    });
  });

  it("retains bounded physical rejection feedback for a recovered model branch", () => {
    const receipt: HumanoidActionReceipt = {
      transactionId: "route-rejected-1",
      agentId: "humanoid-motion-reference",
      action: "plan_humanoid_navigation",
      input: {
        target: { x: 4, y: 0.75, z: 4.5 },
        arrival_heading: null
      },
      fingerprint: "fingerprint",
      accepted: false,
      code: "humanoid_route_rejected",
      worldBeforeRevision: 22,
      worldAfterRevision: 22,
      frameCount: 0,
      channels: ["locomotion"],
      detail: {
        target: { x: 4, y: 0.75, z: 4.5 },
        chunk_target: { x: 4, y: 0.757, z: 4.3 },
        partial_endpoint: { x: 4.01, y: 0.753, z: 4.105 },
        preview_frames: 667,
        preview_travelled_m: 0.072,
        reason: "Navigation target projection exceeds 0.15m",
        physical_trajectory: Array.from({ length: 500 }, (_, frame) => ({ frame }))
      },
      committedAt: "2026-08-06T00:00:00.000Z"
    };

    const projected = recentReceiptContext(receipt);

    expect(projected).toMatchObject({
      transaction_id: "route-rejected-1",
      accepted: false,
      code: "humanoid_route_rejected",
      detail: {
        target: { x: 4, y: 0.75, z: 4.5 },
        chunk_target: { x: 4, y: 0.757, z: 4.3 },
        partial_endpoint: { x: 4.01, y: 0.753, z: 4.105 },
        preview_frames: 667,
        preview_travelled_m: 0.072,
        reason: "Navigation target projection exceeds 0.15m"
      }
    });
    expect(JSON.stringify(projected)).not.toContain("physical_trajectory");
    expect(JSON.stringify(projected).length).toBeLessThan(1_000);
  });

  it("preserves IK-validated base placements without exposing motion trajectories", () => {
    const receipt: HumanoidActionReceipt = {
      transactionId: "base-placement-required-1",
      agentId: "humanoid-motion-reference",
      action: "plan_whole_body_motion_candidates",
      input: {},
      fingerprint: "fingerprint",
      accepted: false,
      code: "manipulation_base_placement_required",
      worldBeforeRevision: 31,
      worldAfterRevision: 31,
      frameCount: 0,
      channels: [],
      detail: {
        reachable_base_placements: [{
          object_id: "assembly_rod",
          hand_surface: "right_hand_middle_1_link",
          root_world_target: { x: 4.19, y: 0.75, z: 4.35 },
          root_yaw_radians: 0.24,
          ik_residual_m: 0.01,
          navigation_validation_required: true
        }],
        physical_trajectory: Array.from({ length: 500 }, (_, frame) => ({ frame }))
      },
      committedAt: "2026-08-06T00:00:00.000Z"
    };

    const projected = recentReceiptContext(receipt);

    expect(projected).toMatchObject({
      detail: {
        reachable_base_placements: [{
          object_id: "assembly_rod",
          root_yaw_radians: 0.24,
          navigation_validation_required: true
        }]
      }
    });
    expect(JSON.stringify(projected)).not.toContain("physical_trajectory");
  });

  it("keeps repeated-planning evidence visible to the deciding model", () => {
    const receipt: HumanoidActionReceipt = {
      transactionId: "repeated-plan-1",
      agentId: "humanoid-motion-reference",
      action: "plan_whole_body_motion_candidates",
      input: {},
      fingerprint: "fingerprint",
      accepted: false,
      code: "repeated_planning_failure",
      worldBeforeRevision: 42,
      worldAfterRevision: 42,
      frameCount: 0,
      channels: [],
      detail: {
        repeated_action: "plan_whole_body_motion_candidates",
        repeated_failure_count: 2,
        previous_code: "manipulation_base_placement_required",
        physical_execution_revision: 9,
        automatic_actuation: false,
        recovery: "Choose materially different physical parameters."
      },
      committedAt: "2026-08-06T00:00:00.000Z"
    };

    expect(modelToolReceiptDetail(receipt)).toEqual({
      repeated_action: "plan_whole_body_motion_candidates",
      repeated_failure_count: 2,
      previous_code: "manipulation_base_placement_required",
      physical_execution_revision: 9,
      automatic_actuation: false,
      recovery: "Choose materially different physical parameters."
    });
  });

  it("keeps the exact active Skill identity visible without its geometry payload", () => {
    const receipt: HumanoidActionReceipt = {
      transactionId: "skill-binding-call",
      agentId: "humanoid-motion-reference",
      action: "begin_humanoid_skill",
      input: {},
      fingerprint: "fingerprint",
      accepted: true,
      code: "humanoid_skill_bound",
      worldBeforeRevision: 52,
      worldAfterRevision: 52,
      frameCount: 0,
      channels: [],
      detail: {
        automatic_actuation: false,
        binding: {
          transaction_id: "skill-binding-call",
          skill_plan_transaction_id: "skill-plan-call",
          skill_node_id: "approach-object",
          invocation: {
            skill: "approach",
            object_id: "workpiece",
            interaction_point_id: "geometry-x-negative",
            standoff_m: 0.45
          },
          phase: "route",
          phase_authority: "navigation",
          planning_action: "plan_humanoid_navigation",
          observed_world_revision: 52,
          eligible_interaction_points: Array.from(
            { length: 100 },
            (_, index) => ({ id: `point-${index}` })
          )
        }
      },
      committedAt: "2026-08-06T00:00:00.000Z"
    };

    const projected = modelToolReceiptDetail(receipt);

    expect(projected).toMatchObject({
      automatic_actuation: false,
      binding: {
        transaction_id: "skill-binding-call",
        skill_plan_transaction_id: "skill-plan-call",
        skill_node_id: "approach-object",
        phase: "route",
        planning_action: "plan_humanoid_navigation"
      }
    });
    expect(JSON.stringify(projected)).not.toContain("eligible_interaction_points");
  });
});
