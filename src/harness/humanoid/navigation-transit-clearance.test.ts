import { describe, expect, it } from "vitest";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { HumanoidMotionPlan } from "../../world/humanoid/motion-plan.js";
import {
  navigationTransitClearanceFromRejection,
  navigationTransitClearanceMotionRejection,
  type NavigationTransitClearanceRequirement
} from "./navigation-transit-clearance.js";

const requirement: NavigationTransitClearanceRequirement = {
  sourceTransactionId: "nav-rejected",
  observedWorldRevision: 42,
  handSurface: "right_hand_index_1_link",
  hand: "right",
  endEffector: "right_wrist",
  collisionTargetId: "workpiece",
  currentWristWorld: { x: 1, y: 0.8, z: 1 },
  collisionTargetWorld: { x: 1.2, y: 0.7, z: 1.3 }
};

function plan(overrides: Partial<HumanoidMotionPlan> = {}): HumanoidMotionPlan {
  return {
    id: "clear-right-arm",
    intent: "clear the right arm before walking",
    duration_seconds: 2,
    contact_constraints: [],
    keyframes: [
      { at_seconds: 0 },
      {
        at_seconds: 2,
        right_hand: {
          frame: "world",
          position: { x: 0.9, y: 0.85, z: 0.8 },
          tolerance_m: 0.04
        }
      }
    ],
    ...overrides
  };
}

describe("navigation transit clearance", () => {
  it("extracts a hand collision without inventing a recovery target", () => {
    const snapshot = {
      robot: {
        links: {
          right_wrist_yaw_link: {
            position: { x: 1, y: 0.8, z: 1 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          }
        },
        objects: {
          workpiece: {
            id: "workpiece",
            position: { x: 1.2, y: 0.7, z: 1.3 }
          }
        }
      }
    } as unknown as HumanoidWorldSnapshot;

    expect(navigationTransitClearanceFromRejection({
      reason: "environment_contact:right_hand_index_1_link:workpiece; preview_frames=45",
      transactionId: "nav-rejected",
      worldRevision: 42,
      snapshot
    })).toEqual(requirement);
  });

  it("accepts only a fixed-base candidate that moves the collision-side wrist", () => {
    expect(navigationTransitClearanceMotionRejection(
      [plan()],
      requirement
    )).toBeNull();

    const rejected = navigationTransitClearanceMotionRejection([
      plan({
        contact_constraints: [{
          hand_surface: "right_hand_index_1_link",
          object_id: "workpiece",
          required: true
        }],
        keyframes: [
          { at_seconds: 0 },
          {
            at_seconds: 2,
            root_velocity: { forward_mps: 0.1, lateral_mps: 0 }
          }
        ]
      })
    ], requirement);

    expect(rejected?.code).toBe("navigation_transit_clearance_required");
    expect(rejected?.detail).toMatchObject({
      blocked_action: "plan_humanoid_navigation",
      collision_hand_surface: "right_hand_index_1_link",
      rejected_candidates: [{
        reasons: [
          "root_translation_present",
          "future_collision_side_wrist_target_missing",
          "collision_target_contact_authorized"
        ]
      }]
    });
  });
});
