import { describe, expect, it } from "vitest";
import type { HumanoidWorldObservation } from "../../world/humanoid/world.js";
import type {
  HumanoidObjectWorldModelEntry
} from "../../world/humanoid/object-world-model.js";
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
    closed_position: 0,
    open_position: 1.5,
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
      rootRotation: { x: 0, y: 0, z: 0, w: 1 },
      contacts: [],
      links: {
        left_ankle_roll_link: {
          position: { x: 0.88, y: 0.08, z: 1 },
          rotation: { x: 0, y: 0, z: 0, w: 1 }
        },
        right_ankle_roll_link: {
          position: { x: 1.12, y: 0.08, z: 1 },
          rotation: { x: 0, y: 0, z: 0, w: 1 }
        },
        left_wrist_yaw_link: {
          position: { x: 0.75, y: 1, z: 1.15 },
          rotation: { x: 0, y: 0, z: 0, w: 1 }
        },
        right_wrist_yaw_link: {
          position: { x: 1.25, y: 1, z: 1.15 },
          rotation: { x: 0, y: 0, z: 0, w: 1 }
        }
      }
    },
    handSurfaces: [{
      hand: "right",
      handSurface: "right_hand_palm_link",
      worldPosition: { x: 1.25, y: 1, z: 1.23 },
      worldRotation: { x: 0, y: 0, z: 0, w: 1 },
      wristWorldPosition: { x: 1.25, y: 1, z: 1.15 },
      surfaceFromWristWorld: { x: 0, y: 0, z: 0.08 }
    }],
    manipulationReachability: [],
    manipulationBasePlacements: [],
    handCoordination: {
      left: {
        thumb_opposition: 0,
        thumb_curl: 0,
        index_curl: 0,
        middle_curl: 0
      },
      right: {
        thumb_opposition: 0,
        thumb_curl: 0,
        index_curl: 0,
        middle_curl: 0
      }
    },
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
      zones: [{
        zone_id: "assembly-zone",
        center: { x: 6, y: 0.01, z: 7 },
        size: { x: 2, y: 0.02, z: 2 },
        robot_planar_distance_m: Math.hypot(5, 6),
        robot_inside_horizontal: false
      }],
      manipulable_objects: []
    }
  } as unknown as HumanoidWorldObservation;
}

function dynamicWorkpiece(): HumanoidObjectWorldModelEntry {
  const workpiece = structuredClone(door) as unknown as HumanoidObjectWorldModelEntry;
  workpiece.id = "workpiece";
  workpiece.kind = "workpiece";
  workpiece.physical.mobility = "dynamic";
  workpiece.affordances = ["graspable", "movable"];
  workpiece.interaction_points = [{
    id: "grasp-a",
    kind: "grasp",
    compatible_hands: "either",
    world_position: { x: 2.2, y: 1, z: 3 },
    clearance_m: 0.04,
    source: "authored"
  }, {
    id: "grasp-b",
    kind: "grasp",
    compatible_hands: "either",
    world_position: { x: 2.5, y: 1, z: 3 },
    clearance_m: 0.04,
    source: "authored"
  }];
  workpiece.articulation = null;
  return workpiece;
}

describe("humanoid skill binding", () => {
  it("grounds a model-selected semantic zone in its live world geometry", () => {
    const current = observation();
    const result = bindHumanoidSkill({
      transactionId: "zone-navigation-call",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "navigate_to_zone",
          zone_id: "assembly-zone"
        },
        phase: "enter_zone"
      },
      observation: current
    });
    if (!result.accepted) throw new Error("Expected zone-navigation binding");
    expect(result.binding).toMatchObject({
      planning_action: "plan_humanoid_skill",
      invocation: {
        skill: "navigate_to_zone",
        zone_id: "assembly-zone"
      },
      target_position: { x: 6, y: 0.01, z: 7 },
      learned_policy_required_capabilities: ["locomotion"],
      control_mode: "reference_control_fallback"
    });
    expect(planAutonomousHumanoidSkill({
      binding: result.binding,
      observation: current
    })).toEqual({
      kind: "navigation",
      targets: [{
        target: { x: 6, y: 0.01, z: 7 },
        arrivalHeading: null,
        acceptedPositionToleranceMeters: 0.5,
        score: 1
      }]
    });
  });

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
        acceptedPositionToleranceMeters: 0.25,
        score: 19
      }]
    });
  });

  it("uses spatial tolerance for a recovery stance", () => {
    const current = observation();
    const result = bindHumanoidSkill({
      transactionId: "retreat-skill-call",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "retreat",
          target: { x: 2.5, y: 0.75, z: 3.5 },
          minimum_obstacle_clearance_m: 0.5
        },
        phase: "route"
      },
      observation: current
    });
    if (!result.accepted) throw new Error("Expected retreat binding");

    expect(planAutonomousHumanoidSkill({
      binding: result.binding,
      observation: current
    })).toEqual({
      kind: "navigation",
      targets: [{
        target: { x: 2.5, y: 0.75, z: 3.5 },
        arrivalHeading: null,
        acceptedPositionToleranceMeters: 0.2,
        score: 1
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
    const reachObservation = observation();
    const wristWorldPosition = { x: 1.25, y: 1, z: 1.15 };
    reachObservation.handSurfaces.push(...[
      ["right_hand_thumb_2_link", { x: -0.025, y: 0.01, z: 0.1 }],
      ["right_hand_index_1_link", { x: 0.02, y: 0, z: 0.11 }],
      ["right_hand_middle_1_link", { x: 0, y: 0, z: 0.115 }]
    ].map(([handSurface, offset]) => ({
      hand: "right" as const,
      handSurface: handSurface as "right_hand_thumb_2_link"
        | "right_hand_index_1_link" | "right_hand_middle_1_link",
      worldPosition: {
        x: wristWorldPosition.x + (offset as { x: number }).x,
        y: wristWorldPosition.y + (offset as { y: number }).y,
        z: wristWorldPosition.z + (offset as { z: number }).z
      },
      worldRotation: { x: 0, y: 0, z: 0, w: 1 },
      wristWorldPosition,
      surfaceFromWristWorld: offset as { x: number; y: number; z: number }
    })));
    const reach = bindHumanoidSkill({
      transactionId: "skill-call-reach",
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
        phase: "reach_handle"
      },
      observation: reachObservation
    });
    if (!reach.accepted) throw new Error("Expected reach-handle binding");
    const reachPlan = planAutonomousHumanoidSkill({
      binding: reach.binding,
      observation: reachObservation
    });
    if (reachPlan.kind !== "motion") throw new Error("Expected reach motion");
    expect(reachPlan.batch.termination.predicates).toEqual([{
      type: "hand_contact_object_region",
      hand: "right",
      object_id: "cabinet-door",
      center_world: { x: 2.35, y: 1, z: 3 },
      maximum_distance_m: 0.06,
      minimum_normal_force: 1,
      minimum_distinct_surfaces: 1
    }, {
      type: "root_near_point",
      target: reachObservation.robot.rootPosition,
      tolerance_m: 0.08
    }]);
    for (const candidate of reachPlan.batch.candidates) {
      const stage = candidate.keyframes.at(-2)?.right_hand;
      const contact = candidate.keyframes.at(-1)?.right_hand;
      expect(candidate.keyframes).toHaveLength(6);
      expect(stage?.orientation).toEqual(
        reachObservation.robot.links.right_wrist_yaw_link.rotation
      );
      expect(contact?.orientation).not.toEqual(stage?.orientation);
      expect(Math.hypot(
        (contact?.position.x ?? 0) - (stage?.position.x ?? 0),
        (contact?.position.y ?? 0) - (stage?.position.y ?? 0),
        (contact?.position.z ?? 0) - (stage?.position.z ?? 0)
      ))
        .toBeGreaterThan(0.15);
    }

    const establish = bindHumanoidSkill({
      transactionId: "skill-call-grasp",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: reach.binding.invocation,
        phase: "establish_grasp"
      },
      observation: reachObservation
    });
    if (!establish.accepted) throw new Error("Expected establish-grasp binding");
    const establishPlan = planAutonomousHumanoidSkill({
      binding: establish.binding,
      observation: reachObservation
    });
    if (establishPlan.kind !== "motion") throw new Error("Expected grasp motion");
    expect(establishPlan.batch.termination.predicates).toEqual([
      expect.objectContaining({
        type: "hand_contact_object_region",
        minimum_distinct_surfaces: 2
      }),
      expect.objectContaining({
        type: "hand_coordination_displaced",
        hand: "right",
        origin: reachObservation.handCoordination.right
      }),
      expect.objectContaining({ type: "root_near_point" })
    ]);

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
        eligible_interaction_point_ids: ["door-handle"],
        learned_policy_required_capabilities: [
          "joint_reference_tracking",
          "contact_rich_manipulation"
        ],
        learned_policy_missing_capabilities: [
          "joint_reference_tracking",
          "contact_rich_manipulation"
        ],
        control_mode: "reference_control_fallback"
      }
    });
    if (!result.accepted) throw new Error("Expected skill binding");

    const planned = planAutonomousHumanoidSkill({
      binding: result.binding,
      observation: observation()
    });
    if (planned.kind !== "motion") throw new Error("Expected articulation motion");
    expect(planned.batch.termination.predicates).toEqual([
      expect.objectContaining({
        type: "hand_contact_object_any",
        hand: "right",
        object_id: "cabinet-door"
      }),
      expect.objectContaining({
        type: "articulation_displaced",
        object_id: "cabinet-door",
        joint_id: "cabinet-hinge",
        origin_position: 0.4,
        direction: "increasing"
      })
    ]);
    const segment = planned.batch.termination.predicates[1];
    expect(segment?.type === "articulation_displaced"
      ? segment.minimum_delta : 0).toBeGreaterThan(0);
    expect(segment?.type === "articulation_displaced"
      ? segment.minimum_delta : Infinity).toBeLessThan(0.4);

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

  it("binds regrasp to the model-selected alternative point and excludes failed points", () => {
    const current = observation();
    current.interaction.object_world_model.objects = [dynamicWorkpiece()];
    current.interaction.carrying.bindings = [{
      object_id: "workpiece",
      hand: "left",
      source_frame: current.frame,
      source_world_revision: current.worldRevision
    }];

    const result = bindHumanoidSkill({
      transactionId: "regrasp-skill-call",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "regrasp",
          object_id: "workpiece",
          interaction_point_id: "grasp-b",
          from_hand: "left",
          to_hand: "right",
          excluded_interaction_point_ids: ["grasp-a"]
        },
        phase: "support_object"
      },
      observation: current
    });
    expect(result).toMatchObject({
      accepted: true,
      binding: {
        eligible_interaction_point_ids: ["grasp-a", "grasp-b"],
        invocation: {
          interaction_point_id: "grasp-b",
          excluded_interaction_point_ids: ["grasp-a"]
        }
      }
    });

    expect(() => bindHumanoidSkill({
      transactionId: "regrasp-excluded-point",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "regrasp",
          object_id: "workpiece",
          interaction_point_id: "grasp-a",
          from_hand: "left",
          to_hand: "right",
          excluded_interaction_point_ids: ["grasp-a"]
        },
        phase: "support_object"
      },
      observation: current
    })).toThrow("regrasp target cannot be one of the excluded grasp points");
  });

  it("places an object center above a live support surface without releasing unrequested grasps", () => {
    const current = observation();
    const workpiece = dynamicWorkpiece();
    workpiece.size = { x: 0.2, y: 0.2, z: 0.4 };
    workpiece.pose.rotation = { x: 0, y: 0, z: 0, w: 1 };
    const support = structuredClone(door) as unknown as HumanoidObjectWorldModelEntry;
    support.id = "table";
    support.kind = "table";
    support.affordances = ["support_surface"];
    support.interaction_points = [];
    support.articulation = null;
    support.support_surface = {
      center_world: { x: 2.4, y: 0.8, z: 3.2 },
      size: { x: 1.2, y: 0.02, z: 0.8 },
      normal_world: { x: 0, y: 1, z: 0 }
    };
    current.interaction.object_world_model.objects = [workpiece, support];
    current.interaction.carrying.bindings = (["left", "right"] as const).map(
      (hand) => ({
        object_id: workpiece.id,
        hand,
        source_frame: current.frame,
        source_world_revision: current.worldRevision
      })
    );
    current.interaction.grasp_authority = {
      contract_sha256: "c".repeat(64),
      minimum_distinct_contact_surfaces: 2,
      minimum_lift_m: 0.05,
      minimum_lifted_hold_frames: 2,
      minimum_relative_pose_stable_frames: 3,
      hand_surfaces: { left: [], right: [] }
    };

    const binding = bindHumanoidSkill({
      transactionId: "place-on-support",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "place",
          object_id: workpiece.id,
          hands: "both",
          destination: {
            type: "support_surface",
            object_id: support.id,
            local_target: null
          },
          release_after_settled: false
        },
        phase: "settle_and_release"
      },
      observation: current
    });
    if (!binding.accepted) throw new Error("Expected support placement binding");
    const plan = planAutonomousHumanoidSkill({
      binding: binding.binding,
      observation: current
    });
    if (plan.kind !== "motion") throw new Error("Expected placement motion");

    expect(plan.batch.termination.predicates).toEqual(expect.arrayContaining([
      expect.objectContaining({
        type: "grasp_verified",
        object_id: workpiece.id,
        hand: "left"
      }),
      expect.objectContaining({
        type: "grasp_verified",
        object_id: workpiece.id,
        hand: "right"
      }),
      expect.objectContaining({
        type: "object_near_point",
        object_id: workpiece.id,
        target: { x: 2.4, y: 0.9, z: 3.2 }
      })
    ]));
    expect(plan.batch.termination.predicates.some(
      ({ type }) => type === "object_released"
        || type === "object_settled_on_support"
    )).toBe(false);
    expect(plan.batch.candidates.every(
      (candidate) => candidate.keyframes.at(-1)?.hand_coordination === undefined
    )).toBe(true);
  });

  it("aligns the object's longest axis with a live slot insertion direction", () => {
    const current = observation();
    const workpiece = dynamicWorkpiece();
    workpiece.size = { x: 0.1, y: 0.12, z: 0.5 };
    workpiece.pose.rotation = {
      x: 0,
      y: Math.SQRT1_2,
      z: 0,
      w: Math.SQRT1_2
    };
    const slot = structuredClone(door) as unknown as HumanoidObjectWorldModelEntry;
    slot.id = "assembly-slot";
    slot.kind = "fixture";
    slot.affordances = ["insertable"];
    slot.interaction_points = [{
      id: "slot-entry",
      kind: "insert",
      compatible_hands: "either",
      world_position: { x: 2.6, y: 1, z: 3.4 },
      approach_direction_world: { x: 0, y: 0, z: 1 },
      clearance_m: 0.08,
      source: "authored"
    }];
    slot.articulation = null;
    current.interaction.object_world_model.objects = [workpiece, slot];
    current.interaction.carrying.bindings = [{
      object_id: workpiece.id,
      hand: "right",
      source_frame: current.frame,
      source_world_revision: current.worldRevision
    }];
    current.interaction.grasp_authority = {
      contract_sha256: "d".repeat(64),
      minimum_distinct_contact_surfaces: 2,
      minimum_lift_m: 0.05,
      minimum_lifted_hold_frames: 2,
      minimum_relative_pose_stable_frames: 3,
      hand_surfaces: { left: [], right: [] }
    };

    const binding = bindHumanoidSkill({
      transactionId: "place-in-slot",
      agentId: "humanoid-motion-reference",
      request: {
        invocation: {
          skill: "place",
          object_id: workpiece.id,
          hands: "right",
          destination: {
            type: "slot",
            object_id: slot.id,
            interaction_point_id: "slot-entry",
            insertion_depth_m: 0.18
          },
          release_after_settled: false
        },
        phase: "settle_and_release"
      },
      observation: current
    });
    if (!binding.accepted) throw new Error("Expected slot placement binding");
    const plan = planAutonomousHumanoidSkill({
      binding: binding.binding,
      observation: current
    });
    if (plan.kind !== "motion") throw new Error("Expected slot placement motion");
    const placement = plan.batch.termination.predicates.find(
      ({ type }) => type === "object_near_point"
    );
    expect(placement).toMatchObject({
      type: "object_near_point",
      object_id: workpiece.id,
      target: { x: 2.6, y: 1, z: 3.58 },
      target_orientation: expect.any(Object),
      orientation_tolerance_rad: 0.18
    });
    if (placement?.type !== "object_near_point") {
      throw new Error("Expected slot placement pose predicate");
    }
    expect(placement.target_orientation).not.toEqual(workpiece.pose.rotation);
  });
});
