import { describe, expect, it } from "vitest";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { HumanoidMotionPlan } from "../../world/humanoid/motion-plan.js";
import type { HumanoidMotionOptionContract } from "../../world/humanoid/motion-option.js";
import {
  navigationTransitClearanceFromRejection,
  navigationTransitClearanceMotionRejection,
  type NavigationTransitClearanceRequirement
} from "./navigation-transit-clearance.js";

const requirement: NavigationTransitClearanceRequirement = {
  sourceTransactionId: "nav-rejected",
  skillTransactionId: null,
  blockedAction: "plan_humanoid_navigation",
  observedWorldRevision: 42,
  handSurface: "right_hand_index_1_link",
  hand: "right",
  endEffector: "right_wrist",
  collisionTargetId: "workpiece",
  collisionTargetKind: "object",
  currentWristWorld: { x: 1, y: 0.8, z: 1 },
  currentFeetWorld: {
    left: { x: 0.9, y: 0.05, z: 1 },
    right: { x: 1.1, y: 0.05, z: 1 }
  },
  collisionTargetWorld: { x: 1.2, y: 0.7, z: 1.3 },
  contactPointWorld: null,
  separationNormalWorld: null,
  separationNormalRobot: null,
  normalForceN: null
};

function plan(overrides: Partial<HumanoidMotionPlan> = {}): HumanoidMotionPlan {
  const feet = {
    left_foot: {
      frame: "world" as const,
      position: { ...requirement.currentFeetWorld.left },
      tolerance_m: 0.04
    },
    right_foot: {
      frame: "world" as const,
      position: { ...requirement.currentFeetWorld.right },
      tolerance_m: 0.04
    }
  };
  return {
    id: "clear-right-arm",
    intent: "clear the right arm before walking",
    duration_seconds: 2,
    contact_constraints: [],
    keyframes: [
      { at_seconds: 0, ...feet },
      {
        at_seconds: 2,
        ...feet,
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
          },
          left_ankle_roll_link: {
            position: { x: 0.9, y: 0.05, z: 1 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          },
          right_ankle_roll_link: {
            position: { x: 1.1, y: 0.05, z: 1 },
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

    expect(navigationTransitClearanceMotionRejection([
      plan({
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
        ]
      })
    ], requirement)?.detail).toMatchObject({
      rejected_candidates: [{ reasons: [
        "left_support_foot_target_missing_or_changed",
        "right_support_foot_target_missing_or_changed"
      ] }]
    });

    const rejected = navigationTransitClearanceMotionRejection([
      plan({
        contact_constraints: [{
          hand_surface: "right_hand_index_1_link",
          object_id: "workpiece",
          required: true
        }],
        keyframes: [
          {
            at_seconds: 0,
            left_foot: {
              frame: "world",
              position: { ...requirement.currentFeetWorld.left },
              tolerance_m: 0.04
            },
            right_foot: {
              frame: "world",
              position: { ...requirement.currentFeetWorld.right },
              tolerance_m: 0.04
            }
          },
          {
            at_seconds: 2,
            left_foot: {
              frame: "world",
              position: { ...requirement.currentFeetWorld.left },
              tolerance_m: 0.04
            },
            right_foot: {
              frame: "world",
              position: { ...requirement.currentFeetWorld.right },
              tolerance_m: 0.04
            },
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

  it("uses structured MuJoCo solid geometry before legacy reason parsing", () => {
    const snapshot = {
      robot: {
        links: {
          right_wrist_yaw_link: {
            position: { x: 1, y: 0.8, z: 1 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          },
          left_ankle_roll_link: {
            position: { x: 0.9, y: 0.05, z: 1 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          },
          right_ankle_roll_link: {
            position: { x: 1.1, y: 0.05, z: 1 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          }
        },
        objects: {}
      }
    } as unknown as HumanoidWorldSnapshot;
    const result = navigationTransitClearanceFromRejection({
      reason: "environment_contact:right_hand_index_1_link:environment",
      blockingContacts: [{
        surface: {
          kind: "hand_surface",
          name: "right_hand_index_1_link"
        },
        target: { kind: "solid", id: "stone_column" },
        contact_point_world: { x: 1.12, y: 0.79, z: 1.18 },
        separation_normal_world: { x: -1, y: 0, z: 0 },
        separation_normal_robot: { x: 0, y: 0, z: -1 },
        normal_force_n: 18.5,
        simulated_time: 4.2
      }],
      solidTokens: [{
        id: "stone_column",
        sourceId: "stone_column",
        kind: "fixed_object",
        center: { x: 1.3, y: 0.8, z: 1.2 },
        size: { x: 0.5, y: 1.6, z: 0.5 },
        currentContacts: []
      }],
      transactionId: "structured-collision",
      worldRevision: 43,
      snapshot
    });

    expect(result).toMatchObject({
      collisionTargetId: "stone_column",
      collisionTargetKind: "solid",
      collisionTargetWorld: { x: 1.3, y: 0.8, z: 1.2 },
      contactPointWorld: { x: 1.12, y: 0.79, z: 1.18 },
      separationNormalWorld: { x: -1, y: 0, z: 0 },
      separationNormalRobot: { x: 0, y: 0, z: -1 },
      normalForceN: 18.5
    });
  });

  it("requires an observable terminal for the model-selected new wrist target", () => {
    const target = { x: 0.9, y: 0.85, z: 0.8 };
    const termination: HumanoidMotionOptionContract = {
      option_id: "clear-right-wrist",
      predicates: [{
        type: "end_effector_near_point",
        end_effector: "right_wrist",
        frame: "world",
        target,
        tolerance_m: 0.04
      }],
      stable_steps: 2,
      phases: null
    };
    expect(navigationTransitClearanceMotionRejection(
      [plan()],
      requirement,
      termination
    )).toBeNull();

    expect(navigationTransitClearanceMotionRejection(
      [plan()],
      requirement,
      {
        ...termination,
        predicates: [{
          ...termination.predicates[0]!,
          target: { x: 1.5, y: 1, z: 1.5 }
        }]
      }
    )?.detail).toMatchObject({
      rejected_candidates: [{ reasons: ["matching_wrist_terminal_missing"] }]
    });

    expect(navigationTransitClearanceMotionRejection([
      plan({
        keyframes: [
          {
            at_seconds: 0,
            left_foot: {
              frame: "world",
              position: { ...requirement.currentFeetWorld.left },
              tolerance_m: 0.04
            },
            right_foot: {
              frame: "world",
              position: { ...requirement.currentFeetWorld.right },
              tolerance_m: 0.04
            }
          },
          {
            at_seconds: 2,
            left_foot: {
              frame: "world",
              position: { ...requirement.currentFeetWorld.left },
              tolerance_m: 0.04
            },
            right_foot: {
              frame: "world",
              position: { ...requirement.currentFeetWorld.right },
              tolerance_m: 0.04
            },
            right_hand: {
              frame: "world",
              position: { ...requirement.currentWristWorld },
              tolerance_m: 0.04
            }
          }
        ]
      })
    ], requirement)?.detail).toMatchObject({
      rejected_candidates: [{
        reasons: ["future_collision_side_wrist_target_not_displaced"]
      }]
    });
  });
});
