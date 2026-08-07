import { describe, expect, it } from "vitest";
import type { HumanoidWorldObservation } from "../../world/humanoid/world.js";
import {
  bindHumanoidSkill,
  validateSkillPlanningReference
} from "./skill-binding.js";
import { planAutonomousHumanoidSkill } from "./autonomous-skill-planner.js";

const door = {
  id: "cabinet-door",
  kind: "door",
  role: "fixture",
  status: "visible",
  authority: "mujoco_exact",
  pose: {
    position: { x: 2, y: 1, z: 3 },
    rotation: { x: 0, y: 0, z: 0, w: 1 }
  },
  size: { x: 0.8, y: 1, z: 0.05 },
  shape: "box",
  physical: {
    mass_kg: 4,
    friction: { sliding: 0.8, torsional: 0.01, rolling: 0.001 },
    mobility: "articulated"
  },
  affordances: ["openable", "closeable", "pullable", "pushable"],
  interaction_points: [{
    id: "door-handle",
    kind: "pull",
    compatible_hands: "either",
    world_position: { x: 2.35, y: 1, z: 3 },
    clearance_m: 0.06,
    source: "authored"
  }],
  articulation: {
    joint_id: "cabinet-hinge",
    parent_object_id: "cabinet",
    type: "hinge",
    semantic: "cabinet_door",
    axis_world: { x: 0, y: 1, z: 0 },
    anchor_world: { x: 1.6, y: 1, z: 3 },
    position: 0.4,
    velocity: 0,
    range: { minimum: 0, maximum: 1.5 },
    open_fraction: 0.4 / 1.5,
    state: "intermediate"
  },
  relations: {
    contained_by: [],
    contains: [],
    supported_by: [],
    supports: [],
    connected_to: ["cabinet"]
  },
  current_contact_count: 0
} as const;

function observation(): HumanoidWorldObservation {
  return {
    frame: 12,
    worldRevision: 7,
    spatialBelief: {
      protocol: "humanoid-spatial-belief-v1",
      resolution_m: 0.5,
      observed_cell_count: 16,
      free_cell_count: 15,
      occupied_cell_count: 1,
      visited_cell_count: 2,
      total_cell_count: 100,
      coverage_ratio: 0.16,
      frontiers: [{
        id: "frontier:4:6",
        target: { x: 2.25, y: 0, z: 3.25 },
        expected_information_gain: 19,
        travel_distance_m: 1.4,
        revisit_penalty: 0,
        score: 7.9
      }]
    },
    robot: {
      rootPosition: { x: 1, y: 0.76, z: 1 },
      links: {
        left_wrist_yaw_link: { position: { x: 0.75, y: 1, z: 1.15 } },
        right_wrist_yaw_link: { position: { x: 1.25, y: 1, z: 1.15 } }
      }
    },
    handSurfaces: [{
      hand: "right",
      handSurface: "right_hand_palm_link",
      wristWorldPosition: { x: 1.25, y: 1, z: 1.15 },
      surfaceFromWristWorld: { x: 0, y: 0, z: 0.08 }
    }],
    solidTokens: [{
      id: "stone:4:6",
      sourceId: "stone:4:6",
      kind: "block",
      center: { x: 2, y: 0.5, z: 2 },
      size: { x: 1, y: 1, z: 1 },
      currentContacts: []
    }],
    interaction: {
      object_world_model: {
        frame: 12,
        world_revision: 7,
        objects: [structuredClone(door)]
      },
      skill_catalog: {
        contract_sha256: "a".repeat(64)
      },
      carrying: { bindings: [] },
      manipulable_objects: []
    }
  } as unknown as HumanoidWorldObservation;
}

describe("humanoid skill binding", () => {
  it("binds one model-selected information frontier without substituting it", () => {
    const current = observation();
    const result = bindHumanoidSkill({
      transactionId: "explore-skill-call",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "explore",
          frontier_id: "frontier:4:6",
          strategy: "information_gain",
          maximum_travel_m: 4
        },
        phase: "route_to_frontier"
      },
      observation: current
    });
    if (!result.accepted) throw new Error("Expected exploration binding");
    expect(result.binding).toMatchObject({
      planning_action: "plan_humanoid_skill",
      target_position: { x: 2.25, y: 0, z: 3.25 }
    });
    expect(planAutonomousHumanoidSkill({
      binding: result.binding,
      observation: current
    })).toEqual({
      kind: "navigation",
      targets: [{
        target: { x: 2.25, y: 0, z: 3.25 },
        arrivalHeading: null,
        score: 19
      }]
    });
  });

  it("derives a removable-block route and certified hand contact from live geometry", () => {
    const current = observation();
    const approach = bindHumanoidSkill({
      transactionId: "block-approach",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "break_block",
          solid_id: "stone:4:6",
          hand: "right",
          strategy: "strike",
          approach_clearance_m: 0.55
        },
        phase: "approach"
      },
      observation: current
    });
    if (!approach.accepted) throw new Error("Expected block approach binding");
    const route = planAutonomousHumanoidSkill({
      binding: approach.binding,
      observation: current
    });
    expect(route).toMatchObject({
      kind: "navigation",
      targets: expect.arrayContaining([expect.objectContaining({
        arrivalHeading: expect.objectContaining({
          type: "face_point",
          target: { x: 2, y: 0.5, z: 2 }
        })
      })])
    });
    if (route.kind !== "navigation") throw new Error("Expected block route");
    expect(route.targets[0]!.target).not.toEqual(current.robot.rootPosition);

    const contact = bindHumanoidSkill({
      transactionId: "block-contact",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: approach.binding.invocation,
        phase: "contact"
      },
      observation: current
    });
    if (!contact.accepted) throw new Error("Expected block contact binding");
    const motion = planAutonomousHumanoidSkill({
      binding: contact.binding,
      observation: current
    });
    if (motion.kind !== "motion") throw new Error("Expected block contact motion");
    expect(motion.batch).toMatchObject({
      termination: {
        stable_steps: 8,
        predicates: [{
          type: "hand_contact_solid",
          hand_surface: "right_hand_palm_link",
          solid_id: "stone:4:6",
          minimum_normal_force: 5
        }]
      },
      candidates: expect.arrayContaining([expect.objectContaining({
        contact_constraints: [{
          hand_surface: "right_hand_palm_link",
          solid_id: "stone:4:6",
          required: true
        }]
      })])
    });
  });

  it("binds a model-selected open phase to live affordance and articulation evidence", () => {
    const result = bindHumanoidSkill({
      transactionId: "skill-call-1",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "open",
          object_id: "cabinet-door",
          interaction_point_id: "door-handle",
          joint_id: "cabinet-hinge",
          hand: "right",
          minimum_open_fraction: 0.8
        },
        phase: "actuate_joint"
      },
      observation: observation()
    });
    expect(result).toMatchObject({
      accepted: true,
      binding: {
        transaction_id: "skill-call-1",
        phase_authority: "whole_body",
        planning_action: "plan_humanoid_skill",
        observed_world_revision: 7,
        eligible_interaction_point_ids: ["door-handle"]
      }
    });
    if (!result.accepted) throw new Error("Expected skill binding");

    expect(validateSkillPlanningReference({
      binding: result.binding,
      action: "plan_humanoid_skill",
      currentWorldRevision: 7,
      rawInput: {
        skill_transaction_id: "skill-call-1",
        termination: {
          predicates: [{
            type: "articulation_state",
            object_id: "cabinet-door",
            joint_id: "cabinet-hinge",
            state: "open",
            tolerance: 0.1
          }]
        }
      }
    })).toEqual({ accepted: true });
  });

  it("rejects stale, mismatched, and bypassed solver authority", () => {
    const result = bindHumanoidSkill({
      transactionId: "skill-call-2",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "open",
          object_id: "cabinet-door",
          interaction_point_id: "door-handle",
          joint_id: "cabinet-hinge",
          hand: "left",
          minimum_open_fraction: 0.9
        },
        phase: "actuate_joint"
      },
      observation: observation()
    });
    if (!result.accepted) throw new Error("Expected skill binding");
    expect(validateSkillPlanningReference({
      binding: result.binding,
      action: "plan_humanoid_skill",
      currentWorldRevision: 8,
      rawInput: { skill_transaction_id: "skill-call-2" }
    })).toMatchObject({ accepted: false, code: "skill_world_revision_stale" });
    expect(validateSkillPlanningReference({
      binding: result.binding,
      action: "plan_humanoid_skill",
      currentWorldRevision: 7,
      rawInput: {
        skill_transaction_id: "other-call",
        termination: { predicates: [] }
      }
    })).toMatchObject({ accepted: false, code: "skill_reference_mismatch" });
    expect(validateSkillPlanningReference({
      binding: result.binding,
      action: "plan_whole_body_motion_candidates",
      currentWorldRevision: 7,
      rawInput: {
        skill_transaction_id: "skill-call-2",
        termination: {
          predicates: [{
            type: "articulation_state",
            object_id: "cabinet-door",
            joint_id: "cabinet-hinge",
            state: "open",
            tolerance: 0.2
          }]
        }
      }
    })).toMatchObject({ accepted: false, code: "skill_phase_authority_mismatch" });
  });

  it("does not authorize sensor or checker phases as actuation", () => {
    const result = bindHumanoidSkill({
      transactionId: "skill-call-3",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "open",
          object_id: "cabinet-door",
          interaction_point_id: "door-handle",
          joint_id: "cabinet-hinge",
          hand: "right",
          minimum_open_fraction: 0.8
        },
        phase: "verify_open"
      },
      observation: observation()
    });
    expect(result).toMatchObject({
      accepted: false,
      code: "skill_phase_not_actionable"
    });
  });

  it("binds approach geometry to the autonomous solver", () => {
    const result = bindHumanoidSkill({
      transactionId: "approach-skill-call",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "approach",
          object_id: "cabinet-door",
          interaction_point_id: "door-handle",
          standoff_m: 0.45
        },
        phase: "route"
      },
      observation: observation()
    });
    if (!result.accepted) throw new Error("Expected approach binding");

    expect(validateSkillPlanningReference({
      binding: result.binding,
      action: "plan_humanoid_skill",
      currentWorldRevision: 7,
      rawInput: {
        skill_transaction_id: "approach-skill-call"
      }
    })).toEqual({ accepted: true });

    expect(validateSkillPlanningReference({
      binding: result.binding,
      action: "plan_humanoid_navigation",
      currentWorldRevision: 7,
      rawInput: {
        skill_transaction_id: "approach-skill-call",
        target: { x: 2, y: 0, z: 2.33 }
      }
    })).toMatchObject({
      accepted: false,
      code: "skill_phase_authority_mismatch"
    });
  });

  it("routes articulated pull through the generic solver", () => {
    const result = bindHumanoidSkill({
      transactionId: "skill-call-4",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "pull",
          object_id: "cabinet-door",
          interaction_point_id: "door-handle",
          hand: "right",
          direction_world: { x: 0, y: 0, z: -1 },
          distance_m: 0.15
        },
        phase: "apply_force"
      },
      observation: observation()
    });
    if (!result.accepted) throw new Error("Expected pull binding");
    expect(validateSkillPlanningReference({
      binding: result.binding,
      action: "plan_humanoid_skill",
      currentWorldRevision: 7,
      rawInput: {
        skill_transaction_id: "skill-call-4"
      }
    })).toEqual({ accepted: true });
  });
});
