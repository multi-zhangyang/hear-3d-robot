import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import {
  composeWorkyardResidualAction,
  dryRunWorkyardResidualTrainingContract,
  loadWorkyardResidualTrainingContract
} from "./workyard-residual-contract.js";

const EMPTY_HANDS = {
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
} as const;

describe("Workyard frozen-teacher residual contract", () => {
  it("binds the 14D DAgger reach policy to the real Workyard and G1", async () => {
    const [catalog, contract] = await Promise.all([
      loadRuntimeCatalog(),
      loadWorkyardResidualTrainingContract()
    ]);
    const report = dryRunWorkyardResidualTrainingContract(
      contract,
      catalog.materialize("humanoid_workyard", 0),
      {
        ready: true,
        jit_sha256: "a".repeat(64),
        report_sha256: "b".repeat(64),
        batch_dynamic: true,
        blockers: []
      }
    );

    expect(report).toMatchObject({
      teacher: {
        observation_size: 99,
        action_size: 29,
        frozen_joint_count: 15,
        gradient_authority: "none"
      },
      student: {
        observation_size: 231,
        action_size: 14,
        upper_body_residual_count: 14,
        hand_synergy_count: 0,
        next_phase_action_size: 22,
        later_phase_action_size: 25
      },
      colab_smoke_ready: true,
      blockers: []
    });
    expect(contract.teacher.actuation).toEqual({
      protocol: "hear-frozen-locomotion-residual-task-tracking-actuation-v1",
      authority: "partitioned_by_joint_ownership",
      runtime_body_model: "frozen_teacher_aligned_upper_body_harness_task_tracking",
      body_joint_count: 29,
      frozen_source_joint_count: 15,
      residual_task_tracking_joint_count: 14,
      frozen_protocol: "mjlab-unitree-g1-source-actuation-v1",
      residual_protocol: "hear-harness-task-tracking-pd-v1",
      task_tracking_stiffness: {
        arm: 80,
        wrist: 40
      },
      damping_scaling: "source_damping_sqrt_stiffness_ratio",
      joint_effort_limits: "unitree_g1_joint_actuatorfrcrange_unchanged",
      generic_xml_position_gains_permitted: false
    });
    expect(contract.student.entry_state).toEqual({
      protocol: "hear-workyard-reach-entry-v1",
      authority: "harness_prepositioned_stance",
      root_position_world: [0.63, 0, 0.79],
      desired_base_twist: [0, 0, 0],
      nominal_object_distance_m: 0.17
    });
    expect(contract.reach_teacher).toMatchObject({
      protocol: "hear-batched-adaptive-reach-teacher-v15",
      solver: "target_conditioned_feasible_posture_servo_with_dls_diagnostics",
      target_protocol: "shoulder-ray-side-clearance-pregrasp-v1",
      pregrasp_shell_radius_m: 0.1,
      pregrasp_lateral_clearance_m: 0.1,
      active_hand_allocation: "nearest_lateral_hand_centerline_balanced-v1",
      contact_target_activation: "contact_authority_only",
      success_metric: "active_wrist_to_command_within_tolerance",
      target_memory: "per_environment_measured_joint_anchored_anti_windup",
      feasible_posture_protocol:
        "offline-collision-aware-side-pregrasp-quadratic-map-v3",
      feasible_posture_feature_protocol:
        "normalized-target-pelvis-xy-quadratic-v1",
      feasible_posture_feature_order: ["bias", "x", "y", "x2", "xy", "y2"],
      feasible_posture_center_xy_m: [
        [0.13263583228233528, 0.14000000000000004],
        [0.13263572415421565, -0.14000000000000004]
      ],
      feasible_posture_feature_scale_m: 0.08,
      feasible_posture_feature_clamp: 1.25,
      feasible_posture_target_memory:
        "per_environment_episode_initial_typed_target",
      feasible_posture_offline_validation: {
        command_jitter_m: 0.08,
        fit_grid_per_arm: 25,
        dense_grid_per_arm: 289,
        tolerance_m: 0.06,
        collision_clearance_m: 0.005,
        success_rate: 0.8875432525951558,
        kinematic_tolerance_rate: 1,
        collision_clear_rate: 0.8875432525951558
      },
      posture_attractor_gain: 1,
      task_space_feedback_gain: 0,
      max_joint_correction_rad: 0.2,
      max_solver_target_slew_rad: 0.03,
      max_command_lead_rad: 0.16,
      hold_enter_error_m: 0.05,
      hold_release_error_m: 0.075,
      actor_observation_exposure: false,
      execution_authority: "none"
    });
    expect(contract.reach_teacher.feasible_posture_normalized_action_coefficients)
      .toHaveLength(2);
    expect(contract.reach_teacher.feasible_posture_normalized_action_coefficients[0])
      .toHaveLength(7);
  });

  it("cannot alter the teacher-owned lower body or waist", async () => {
    const contract = await loadWorkyardResidualTrainingContract();
    const teacher = Array.from({ length: 29 }, (_, index) => index / 10);
    const action = Array.from({ length: 14 }, () => 1);
    const composed = composeWorkyardResidualAction(
      contract,
      teacher,
      action,
      EMPTY_HANDS
    );

    expect(composed.body_joint_targets.slice(0, 15)).toEqual(teacher.slice(0, 15));
    expect(composed.body_joint_targets.slice(15)).toEqual(
      [
        0.7, 0.7, 0.5, 1.1, 0.5, 0.5, 0.5,
        0.7, 0.3, 0.5, 1.1, 0.5, 0.5, 0.5
      ]
    );
    expect(composed.upper_body_residuals).toEqual(
      Array.from({ length: 14 }, () => 0.5)
    );
    expect(composed.hand_coordination).toEqual(EMPTY_HANDS);
  });
});
