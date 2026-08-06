import { describe, expect, it } from "vitest";
import { createActionGoalEvidence } from "./goal-evidence.js";
import { recoverableBlockedGoalEvidence } from "./goal-retirement-evidence.js";

describe("Goal retirement recovery evidence", () => {
  it("recognizes a rejected manipulation plan with live IK base placements", () => {
    const artifact = createActionGoalEvidence({
      transactionId: "plan-grasp-1",
      worldFrame: 42,
      worldRevision: 42,
      receipt: {
        transactionId: "plan-grasp-1",
        accepted: false,
        code: "manipulation_base_placement_required",
        worldAfterRevision: 42,
        detail: {
          reachable_base_placements: [{
            object_id: "assembly_rod",
            hand_surface: "right_hand_middle_1_link",
            root_world_target: { x: 4.19, y: 0.75, z: 4.35 },
            root_yaw_radians: 0.24,
            ik_residual_m: 0.01
          }]
        }
      }
    });

    expect(recoverableBlockedGoalEvidence([artifact])).toEqual({
      evidenceRef: "action:plan-grasp-1",
      transactionId: "plan-grasp-1",
      code: "manipulation_base_placement_required",
      reachableBasePlacementCount: 1
    });
  });

  it("does not reinterpret an exhausted rejection as a recovery path", () => {
    const artifact = createActionGoalEvidence({
      transactionId: "plan-grasp-2",
      worldFrame: 43,
      worldRevision: 43,
      receipt: {
        transactionId: "plan-grasp-2",
        accepted: false,
        code: "manipulation_base_placement_required",
        worldAfterRevision: 43,
        detail: { reachable_base_placements: [] }
      }
    });

    expect(recoverableBlockedGoalEvidence([artifact])).toBeNull();
  });
});
