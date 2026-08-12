import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { z } from "zod";
import type { Scenario } from "../domain/schema.js";
import {
  createG1HandArtifactCommand,
  G1HandCoordinationSchema,
  type G1HandCoordination
} from "../world/humanoid/hand-coordination.js";
import {
  G1_HAND_JOINT_NAMES,
  G1_MORPHOLOGY
} from "../world/humanoid/morphology.js";
import {
  HUMANOID_JOINT_NAMES,
  YAHMP_POLICY
} from "../world/humanoid/model.js";

const UnitRateSchema = z.number().finite().min(0).max(1);
const SeedRangeSchema = z.object({
  first: z.number().int().nonnegative(),
  last: z.number().int().nonnegative()
}).strict().refine((value) => value.last >= value.first, {
  message: "Residual training seed range must not be reversed"
});
const SizedTermSchema = z.object({
  name: z.string().trim().min(1),
  size: z.number().int().positive()
}).strict();
const ActionTermSchema = SizedTermSchema.extend({
  minimum: z.literal(-1),
  maximum: z.literal(1),
  scale: z.number().finite().positive()
}).strict();

const WorkyardResidualTrainingContractSchema = z.object({
  protocol: z.literal("hear-workyard-residual-training-contract-v4"),
  scenario_id: z.literal("humanoid_workyard"),
  environment: z.object({
    framework: z.literal("mjlab"),
    task_id: z.literal(
      "Hear-Workyard-Frozen-Locomotion-Upper-Body-Residual-G1-v4"
    ),
    module: z.literal("training/workyard_residual_mjlab_env.py"),
    implementation_status: z.enum(["contract_only", "implemented"])
  }).strict(),
  morphology: z.object({
    id: z.literal("unitree_g1_43dof_with_hands"),
    body_joint_names: z.array(z.string().trim().min(1)).length(29),
    hand_joint_names: z.array(z.string().trim().min(1)).length(14)
  }).strict(),
  timing: z.object({
    physics_step_seconds: z.number().finite().positive(),
    control_step_seconds: z.number().finite().positive(),
    episode_seconds: z.number().finite().positive()
  }).strict(),
  teacher: z.object({
    protocol: z.literal("hear-frozen-locomotion-teacher-v1"),
    source_task: z.literal("Mjlab-Velocity-Flat-Unitree-G1"),
    runtime: z.literal("torchscript_cuda"),
    artifact: z.object({
      file: z.literal("g1_velocity_teacher.jit.pt"),
      identity_report: z.literal("training-report.json"),
      hash_authority: z.literal("identity_report")
    }).strict(),
    observation: z.object({
      protocol: z.literal("mjlab-g1-velocity-observation-v1"),
      size: z.literal(99),
      terms: z.array(SizedTermSchema).length(7)
    }).strict(),
    action: z.object({
      protocol: z.literal("mjlab-g1-joint-position-residual-v1"),
      size: z.literal(29)
    }).strict(),
    actuation: z.object({
      protocol: z.literal(
        "hear-frozen-locomotion-residual-task-tracking-actuation-v1"
      ),
      authority: z.literal("partitioned_by_joint_ownership"),
      runtime_body_model: z.literal(
        "frozen_teacher_aligned_upper_body_harness_task_tracking"
      ),
      body_joint_count: z.literal(29),
      frozen_source_joint_count: z.literal(15),
      residual_task_tracking_joint_count: z.literal(14),
      frozen_protocol: z.literal("mjlab-unitree-g1-source-actuation-v1"),
      residual_protocol: z.literal("hear-harness-task-tracking-pd-v1"),
      task_tracking_stiffness: z.object({
        arm: z.literal(80),
        wrist: z.literal(40)
      }).strict(),
      damping_scaling: z.literal("source_damping_sqrt_stiffness_ratio"),
      joint_effort_limits: z.literal(
        "unitree_g1_joint_actuatorfrcrange_unchanged"
      ),
      generic_xml_position_gains_permitted: z.literal(false)
    }).strict(),
    frozen_joint_names: z.array(z.string().trim().min(1)).length(15),
    gradient_authority: z.literal("none"),
    inference_device: z.literal("same_cuda_device_as_environment"),
    cpu_round_trip_permitted: z.literal(false)
  }).strict(),
  reach_teacher: z.object({
    protocol: z.literal("hear-batched-adaptive-reach-teacher-v15"),
    runtime: z.literal("mujoco_warp_torch_cuda"),
    solver: z.literal(
      "target_conditioned_feasible_posture_servo_with_dls_diagnostics"
    ),
    target_protocol: z.literal("shoulder-ray-side-clearance-pregrasp-v1"),
    pregrasp_shell_radius_m: z.literal(0.1),
    pregrasp_lateral_clearance_m: z.literal(0.1),
    active_hand_allocation: z.literal(
      "nearest_lateral_hand_centerline_balanced-v1"
    ),
    contact_target_activation: z.literal("contact_authority_only"),
    success_metric: z.literal("active_wrist_to_command_within_tolerance"),
    controlled_joint_count: z.literal(14),
    task_dimension_per_arm: z.literal(3),
    target_memory: z.literal(
      "per_environment_measured_joint_anchored_anti_windup"
    ),
    base_damping: z.literal(0.015),
    singularity_damping: z.literal(0.12),
    singularity_threshold: z.literal(0.05),
    feasible_posture_protocol: z.literal(
      "offline-collision-aware-side-pregrasp-quadratic-map-v3"
    ),
    feasible_posture_feature_protocol: z.literal(
      "normalized-target-pelvis-xy-quadratic-v1"
    ),
    feasible_posture_feature_order: z.tuple([
      z.literal("bias"),
      z.literal("x"),
      z.literal("y"),
      z.literal("x2"),
      z.literal("xy"),
      z.literal("y2")
    ]),
    feasible_posture_center_xy_m: z.array(
      z.array(z.number().finite()).length(2)
    ).length(2),
    feasible_posture_feature_scale_m: z.literal(0.08),
    feasible_posture_feature_clamp: z.literal(1.25),
    feasible_posture_target_memory: z.literal(
      "per_environment_episode_initial_typed_target"
    ),
    feasible_posture_normalized_action_coefficients: z.array(
      z.array(z.array(z.number().finite()).length(6)).length(7)
    ).length(2),
    feasible_posture_offline_validation: z.object({
      command_jitter_m: z.literal(0.08),
      fit_grid_per_arm: z.literal(25),
      dense_grid_per_arm: z.literal(289),
      tolerance_m: z.literal(0.06),
      collision_clearance_m: z.literal(0.005),
      success_rate: z.literal(0.8875432525951558),
      kinematic_tolerance_rate: z.literal(1),
      collision_clear_rate: z.literal(0.8875432525951558),
      mean_error_m: z.literal(0.008503026325955211),
      p90_error_m: z.literal(0.018600412903803952),
      maximum_error_m: z.literal(0.034634432043045935),
      minimum_clearance_m: z.literal(0)
    }).strict(),
    posture_attractor_gain: z.literal(1),
    task_space_feedback_gain: z.literal(0),
    max_cartesian_step_m: z.literal(0.08),
    max_joint_correction_rad: z.literal(0.2),
    max_solver_target_slew_rad: z.literal(0.03),
    max_command_lead_rad: z.literal(0.16),
    hold_enter_error_m: z.literal(0.05),
    hold_release_error_m: z.literal(0.075),
    supervision: z.literal("online_dagger_and_ppo_rollout_labels_only"),
    actor_observation_exposure: z.literal(false),
    execution_authority: z.literal("none"),
    cpu_round_trip_per_label: z.literal(false),
    diagnostics_protocol: z.literal(
      "hear-reach-teacher-collision-aware-diagnostics-v15"
    )
  }).strict(),
  warm_start: z.object({
    protocol: z.literal("hear-online-dagger-warm-start-v1"),
    algorithm: z.literal("online_dagger"),
    loss: z.literal("smooth_l1_plus_excess_std_penalty"),
    actor_distribution: z.literal("beta_bounded_minus_one_one"),
    maximum_action_std: z.literal(0.15),
    dispersion_coefficient: z.literal(1),
    teacher_beta: z.object({
      initial: z.literal(1),
      final: z.number().finite().min(0).max(1)
    }).strict().refine((value) => value.final <= value.initial, {
      message: "DAgger teacher beta must not increase"
    }),
    actor_normalizer_updates: z.literal(true),
    checkpoint_before_ppo: z.literal(true),
    ppo_handoff: z.literal("same_actor_after_supervised_warm_start")
  }).strict(),
  ppo_retention: z.object({
    protocol: z.literal("hear-ppo-retention-v2"),
    default_mode: z.literal("critic_warmup_rollout_teacher"),
    critic_only_warmup: z.literal(true),
    actor_normalizer_after_dagger: z.literal("frozen"),
    optimizer: z.literal("separate_actor_critic_learning_rates"),
    actor_distribution: z.literal("beta_bounded_minus_one_one"),
    teacher_supervision: z.literal("every_stored_learner_rollout_state"),
    teacher_action_scope: z.literal("authorized_14d_actor_action"),
    loss: z.literal("smooth_l1_plus_excess_std_penalty"),
    loss_coupling: z.literal("same_ppo_minibatch"),
    default_teacher_loss_coefficient: z.literal(1),
    maximum_action_std: z.literal(0.15),
    dispersion_coefficient: z.literal(1),
    entropy_coefficient: z.literal(0),
    acceptance_comparison: z.literal("identical_held_out_seed")
  }).strict(),
  student: z.object({
    phase: z.literal("task_space_reach_dagger_warm_start"),
    role: z.literal("autonomous_skill_window_executor"),
    conditioned_by: z.literal("humanoid-embodied-skill-call-v2"),
    entry_state: z.object({
      protocol: z.literal("hear-workyard-reach-entry-v1"),
      authority: z.literal("harness_prepositioned_stance"),
      root_position_world: z.tuple([
        z.literal(0.63), z.literal(0), z.literal(0.79)
      ]),
      desired_base_twist: z.tuple([
        z.literal(0), z.literal(0), z.literal(0)
      ]),
      nominal_object_distance_m: z.literal(0.17)
    }).strict(),
    trainable_joint_names: z.array(z.string().trim().min(1)).length(14),
    forbidden_observations: z.array(z.string().trim().min(1)).min(1),
    next_phase: z.object({
      name: z.literal("hand_synergy_contact_grasp"),
      action_size: z.literal(22),
      additional_joint_names: z.array(z.string().trim().min(1)).length(8),
      requires_checkpoint_expansion: z.literal(true),
      activation_gate: z.literal("held_out_reach_and_dynamic_com_acceptance")
    }).strict(),
    later_phase: z.object({
      name: z.literal("low_amplitude_waist_residual"),
      action_size: z.literal(25),
      additional_joint_names: z.array(z.string().trim().min(1)).length(3),
      requires_checkpoint_expansion: z.literal(true),
      activation_gate: z.literal(
        "held_out_contact_grasp_and_dynamic_com_acceptance"
      )
    }).strict()
  }).strict(),
  composition: z.object({
    protocol: z.literal("hear-teacher-residual-composition-v1"),
    teacher_reference: z.literal(
      "default_joint_position_plus_teacher_action_times_teacher_scale"
    ),
    frozen_joint_command: z.literal("teacher_reference"),
    trainable_joint_command: z.literal("neutral_joint_position_plus_student_residual"),
    hand_command: z.literal("fixed_neutral_open_pose"),
    attribution: z.object({
      teacher_action_count: z.literal(29),
      frozen_teacher_joint_count: z.literal(15),
      upper_body_residual_joint_count: z.literal(14),
      hand_synergy_count: z.literal(0),
      segments_reported_separately: z.literal(true)
    }).strict()
  }).strict(),
  observation: z.object({
    protocol: z.literal("hear-workyard-residual-observation-v4"),
    size: z.literal(231),
    history_steps: z.literal(1),
    terms: z.array(SizedTermSchema).min(1)
  }).strict(),
  action: z.object({
    protocol: z.literal("hear-workyard-upper-body-residual-action-v4"),
    size: z.literal(14),
    terms: z.array(ActionTermSchema).length(1)
  }).strict(),
  rewards: z.object({
    phase_scope: z.literal("balance_and_reach_only"),
    terms: z.array(z.object({
      id: z.string().trim().min(1),
      objective: z.enum(["maximize", "minimize"]),
      weight: z.number().finite().positive(),
      authority: z.enum([
        "humanoid_physical_safety",
        "locomotion_teacher",
        "analytic_teacher",
        "mujoco_state",
        "policy_action"
      ])
    }).strict()).min(1),
    excluded_until_later_phase: z.array(z.string().trim().min(1)).min(1)
  }).strict(),
  randomization: z.object({
    object_position_jitter_m: z.number().finite().nonnegative(),
    target_position_jitter_m: z.number().finite().nonnegative(),
    object_mass_scale: z.tuple([
      z.number().finite().positive(), z.number().finite().positive()
    ]),
    friction_scale: z.tuple([
      z.number().finite().positive(), z.number().finite().positive()
    ]),
    actuator_strength_scale: z.tuple([
      z.number().finite().positive(), z.number().finite().positive()
    ]),
    observation_latency_control_steps: z.tuple([
      z.number().int().nonnegative(), z.number().int().nonnegative()
    ])
  }).strict(),
  evaluation: z.object({
    seeds: z.object({
      training: SeedRangeSchema,
      validation: SeedRangeSchema,
      held_out: SeedRangeSchema
    }).strict(),
    episodes: z.number().int().positive(),
    dynamic_com_settling_control_steps: z.number().int().nonnegative(),
    analytic_teacher_preflight_success_rate_minimum: UnitRateSchema,
    analytic_teacher_preflight_minimum_mean_error_maximum_m: (
      z.number().finite().positive()
    ),
    wrist_target_success_rate_minimum: UnitRateSchema,
    fall_rate_maximum: UnitRateSchema,
    teacher_frozen_joint_rms_error_maximum_rad: z.number().finite().positive(),
    maximum_joint_velocity_rad_s: z.number().finite().positive(),
    maximum_requested_effort_utilization: z.number().finite().positive(),
    teacher_frame_ratio_minimum: UnitRateSchema,
    upper_body_residual_frame_ratio_minimum: UnitRateSchema,
    teacher_gradient_parameter_count: z.literal(0),
    reach_teacher_label_coverage_minimum: UnitRateSchema,
    ppo_rollout_teacher_label_coverage_minimum: UnitRateSchema,
    ppo_rollout_mean_action_std_maximum: UnitRateSchema,
    ppo_success_rate_delta_minimum: z.number().finite(),
    ppo_minimum_mean_wrist_error_delta_maximum_m: z.number().finite(),
    ppo_minimum_support_margin_delta_minimum_m: z.number().finite(),
    ppo_maximum_foot_slip_delta_maximum_m_s: z.number().finite(),
    ppo_action_clipped_element_rate_maximum: UnitRateSchema,
    minimum_support_margin_m: z.number().finite(),
    maximum_capture_point_norm_m: z.number().finite().positive(),
    maximum_foot_planar_displacement_m: z.number().finite().positive(),
    maximum_foot_slip_speed_m_s: z.number().finite().positive(),
    double_support_loss_rate_maximum: UnitRateSchema,
    no_foot_contact_rate_maximum: UnitRateSchema
  }).strict()
}).strict().superRefine((contract, context) => {
  validateDeclaredSize(contract.teacher.observation, context, ["teacher", "observation"]);
  validateDeclaredSize(contract.observation, context, ["observation"]);
  validateDeclaredSize(contract.action, context, ["action"]);
  for (const [name, range] of Object.entries(contract.randomization).filter(
    ([, value]) => Array.isArray(value)
  ) as Array<[string, readonly [number, number]]>) {
    if (range[1] < range[0]) {
      context.addIssue({
        code: "custom",
        path: ["randomization", name],
        message: "Residual randomization range must not be reversed"
      });
    }
  }
  for (const [name, values] of [
    ["observation terms", contract.observation.terms.map((term) => term.name)],
    ["reward terms", contract.rewards.terms.map((term) => term.id)],
    ["forbidden observations", contract.student.forbidden_observations]
  ] as const) {
    if (new Set(values).size !== values.length) {
      context.addIssue({
        code: "custom",
        path: [],
        message: `Residual ${name} must be unique`
      });
    }
  }
});

export type WorkyardResidualTrainingContract = z.infer<
  typeof WorkyardResidualTrainingContractSchema
>;

const ExpectedTeacherObservation = [
  ["base_lin_vel", 3],
  ["base_ang_vel", 3],
  ["projected_gravity", 3],
  ["joint_pos", 29],
  ["joint_vel", 29],
  ["previous_actions", 29],
  ["twist_command", 3]
] as const;
const ExpectedObservation = [
  ["body_joint_position_offset", 29],
  ["body_joint_velocity", 29],
  ["previous_upper_body_residual", 14],
  ["hand_joint_position_fraction", 14],
  ["hand_joint_velocity", 14],
  ["support_relative_dynamic_com", 4],
  ["locomotion_teacher_action", 29],
  ["root_linear_velocity_pelvis", 3],
  ["root_angular_velocity_pelvis", 3],
  ["projected_gravity_pelvis", 3],
  ["end_effector_pose_pelvis", 28],
  ["support_contact_features", 6],
  ["hand_object_contact_features", 6],
  ["target_object_pose_pelvis", 7],
  ["target_object_twist_pelvis", 6],
  ["target_zone_relation", 5],
  ["requested_capabilities_multi_hot", 5],
  ["skill_window_progress", 1],
  ["active_hand_one_hot", 2],
  ["desired_base_twist", 3],
  ["wrist_pose_targets_pelvis", 14],
  ["wrist_position_tolerances", 2],
  ["grasp_requirements", 4]
] as const;
const ExpectedAction = [
  ["upper_body_joint_position_residual", 14]
] as const;
const PhaseOneRewardIds = [
  "upright_support",
  "teacher_lower_body_tracking",
  "teacher_waist_tracking",
  "wrist_position_tracking",
  "wrist_distance_progress",
  "wrist_orientation_tracking",
  "reach_teacher_action_tracking",
  "dynamic_com_support",
  "upper_body_residual_l2",
  "action_rate",
  "joint_limit_proximity",
  "actuator_saturation",
  "fall",
  "non_foot_collision",
  "foot_slip"
] as const;

const PhaseTwoHandSynergies = [
  "left_hand_synergy_0",
  "left_hand_synergy_1",
  "left_hand_synergy_2",
  "left_hand_synergy_3",
  "right_hand_synergy_0",
  "right_hand_synergy_1",
  "right_hand_synergy_2",
  "right_hand_synergy_3"
] as const;

export interface WorkyardResidualTeacherEvidence {
  ready: boolean;
  jit_sha256: string | null;
  report_sha256: string | null;
  batch_dynamic: boolean;
  blockers: string[];
}

interface TeacherJitIdentity {
  file: string;
  bytes: number;
  sha256: string;
  input: string;
  input_size: number;
  output: string;
  output_size: number;
  batch_dynamic: boolean;
  runtime: string;
}

export interface WorkyardResidualDryRunReport {
  protocol: "hear-workyard-residual-training-dry-run-v4";
  contract_sha256: string;
  scenario_id: "humanoid_workyard";
  morphology: {
    id: typeof G1_MORPHOLOGY.id;
    body_joint_count: 29;
    hand_joint_count: 14;
  };
  teacher: WorkyardResidualTeacherEvidence & {
    observation_size: 99;
    action_size: 29;
    frozen_joint_count: 15;
    gradient_authority: "none";
  };
  reach_teacher: {
    protocol: "hear-batched-adaptive-reach-teacher-v15";
    controlled_joint_count: 14;
    actor_observation_exposure: false;
    execution_authority: "none";
  };
  ppo_retention: {
    protocol: "hear-ppo-retention-v2";
    default_mode: "critic_warmup_rollout_teacher";
    actor_normalizer_after_dagger: "frozen";
    actor_distribution: "beta_bounded_minus_one_one";
    teacher_supervision: "every_stored_learner_rollout_state";
    loss_coupling: "same_ppo_minibatch";
    checkpoint_comparison: "identical_held_out_seed";
  };
  student: {
    phase: "task_space_reach_dagger_warm_start";
    observation_size: 231;
    action_size: 14;
    upper_body_residual_count: 14;
    hand_synergy_count: 0;
    next_phase_action_size: 22;
    later_phase_action_size: 25;
  };
  contract_ready: true;
  colab_smoke_ready: boolean;
  blockers: string[];
}

export async function loadWorkyardResidualTrainingContract(
  path = resolve("training/workyard-task-v4.json")
): Promise<WorkyardResidualTrainingContract> {
  const value = JSON.parse(await readFile(resolve(path), "utf8")) as unknown;
  return WorkyardResidualTrainingContractSchema.parse(value);
}

export async function inspectWorkyardResidualTeacherArtifacts(
  root = resolve("artifacts/training/g1-residual-teacher")
): Promise<WorkyardResidualTeacherEvidence> {
  const jitPath = resolve(root, "g1_velocity_teacher.jit.pt");
  const reportPath = resolve(root, "training-report.json");
  const blockers: string[] = [];
  let jit: Buffer | null = null;
  let reportBytes: Buffer | null = null;
  try {
    [jit, reportBytes] = await Promise.all([
      readFile(jitPath),
      readFile(reportPath)
    ]);
    const identity = teacherJitIdentity(JSON.parse(reportBytes.toString("utf8")));
    if (identity.file !== "g1_velocity_teacher.jit.pt"
      || identity.bytes !== jit.byteLength
      || identity.sha256 !== sha256(jit)
      || identity.input !== "obs"
      || identity.input_size !== 99
      || identity.output !== "actions"
      || identity.output_size !== 29
      || identity.batch_dynamic !== true
      || identity.runtime !== "torchscript_cuda") {
      blockers.push("Teacher JIT identity does not match its training report");
    }
  } catch (error) {
    blockers.push(error instanceof Error ? error.message : String(error));
  }
  return {
    ready: blockers.length === 0,
    jit_sha256: jit ? sha256(jit) : null,
    report_sha256: reportBytes ? sha256(reportBytes) : null,
    batch_dynamic: blockers.length === 0,
    blockers
  };
}

export function dryRunWorkyardResidualTrainingContract(
  contractInput: WorkyardResidualTrainingContract,
  scenario: Scenario,
  teacherEvidence: WorkyardResidualTeacherEvidence
): WorkyardResidualDryRunReport {
  const contract = WorkyardResidualTrainingContractSchema.parse(contractInput);
  assertExactValues(contract.morphology.body_joint_names, HUMANOID_JOINT_NAMES, "body joints");
  assertExactValues(contract.morphology.hand_joint_names, G1_HAND_JOINT_NAMES, "hand joints");
  assertExactValues(contract.teacher.frozen_joint_names, HUMANOID_JOINT_NAMES.slice(0, 15), "frozen teacher joints");
  assertExactValues(contract.student.trainable_joint_names, HUMANOID_JOINT_NAMES.slice(15), "trainable upper-body joints");
  assertExactValues(
    contract.student.next_phase.additional_joint_names,
    PhaseTwoHandSynergies,
    "next-phase hand synergies"
  );
  assertExactValues(
    contract.student.later_phase.additional_joint_names,
    HUMANOID_JOINT_NAMES.slice(12, 15),
    "later-phase waist joints"
  );
  assertLayout(contract.teacher.observation.terms, ExpectedTeacherObservation, "teacher observation");
  assertLayout(contract.observation.terms, ExpectedObservation, "student observation");
  assertLayout(contract.action.terms, ExpectedAction, "student action");
  assertExactSet(contract.rewards.terms.map((term) => term.id), PhaseOneRewardIds, "phase-one rewards");
  assertSeedSplits(contract.evaluation.seeds);
  assertScenarioTarget(scenario);

  const neutralTeacherTargets = HUMANOID_JOINT_NAMES.map((_, index) => index / 100);
  const composed = composeWorkyardResidualAction(
    contract,
    neutralTeacherTargets,
    Array.from({ length: 14 }, () => 0),
    emptyHands()
  );
  if (composed.body_joint_targets.slice(0, 15).some(
    (value, index) => value !== neutralTeacherTargets[index]
  ) || composed.body_joint_targets.slice(15).some(
    (value, index) => value !== YAHMP_POLICY.defaultJointPositions[15 + index]
  )) {
    throw new Error(
      "A neutral residual action must preserve teacher-owned joints and the upper neutral pose"
    );
  }

  const blockers = [
    ...(contract.environment.implementation_status === "implemented"
      ? []
      : ["Residual mjlab environment is not implemented"]),
    ...teacherEvidence.blockers
  ];
  return {
    protocol: "hear-workyard-residual-training-dry-run-v4",
    contract_sha256: sha256(Buffer.from(JSON.stringify(contract))),
    scenario_id: contract.scenario_id,
    morphology: {
      id: G1_MORPHOLOGY.id,
      body_joint_count: 29,
      hand_joint_count: 14
    },
    teacher: {
      ...teacherEvidence,
      observation_size: 99,
      action_size: 29,
      frozen_joint_count: 15,
      gradient_authority: "none"
    },
    reach_teacher: {
      protocol: contract.reach_teacher.protocol,
      controlled_joint_count: contract.reach_teacher.controlled_joint_count,
      actor_observation_exposure: false,
      execution_authority: "none"
    },
    ppo_retention: {
      protocol: contract.ppo_retention.protocol,
      default_mode: contract.ppo_retention.default_mode,
      actor_normalizer_after_dagger: contract.ppo_retention.actor_normalizer_after_dagger,
      actor_distribution: contract.ppo_retention.actor_distribution,
      teacher_supervision: contract.ppo_retention.teacher_supervision,
      loss_coupling: contract.ppo_retention.loss_coupling,
      checkpoint_comparison: contract.ppo_retention.acceptance_comparison
    },
    student: {
      phase: contract.student.phase,
      observation_size: 231,
      action_size: 14,
      upper_body_residual_count: 14,
      hand_synergy_count: 0,
      next_phase_action_size: 22,
      later_phase_action_size: 25
    },
    contract_ready: true,
    colab_smoke_ready: blockers.length === 0,
    blockers
  };
}

export function composeWorkyardResidualAction(
  contractInput: WorkyardResidualTrainingContract,
  teacherBodyTargets: readonly number[],
  studentAction: readonly number[],
  currentHands: G1HandCoordination
): {
  body_joint_targets: number[];
  frozen_teacher_joint_targets: number[];
  upper_body_residuals: number[];
  hand_coordination: G1HandCoordination;
  hand_joint_targets: ReturnType<typeof createG1HandArtifactCommand>["jointTargets"];
} {
  const contract = WorkyardResidualTrainingContractSchema.parse(contractInput);
  assertLayout(contract.action.terms, ExpectedAction, "student action");
  if (teacherBodyTargets.length !== 29 || !teacherBodyTargets.every(Number.isFinite)) {
    throw new Error("Residual composition requires 29 finite teacher targets");
  }
  if (studentAction.length !== 14 || !studentAction.every(
    (value) => Number.isFinite(value) && value >= -1 && value <= 1
  )) {
    throw new Error("Residual student action must contain 14 values inside [-1, 1]");
  }
  const upperScale = contract.action.terms[0]!.scale;
  const upper = studentAction.map((value) => value * upperScale);
  const body: number[] = Array.from(YAHMP_POLICY.defaultJointPositions);
  body.splice(0, 15, ...teacherBodyTargets.slice(0, 15));
  for (const [offset, residual] of upper.entries()) {
    body[15 + offset] = body[15 + offset]! + residual;
  }
  G1HandCoordinationSchema.parse(currentHands);
  const handCoordination = emptyHands();
  return {
    body_joint_targets: body,
    frozen_teacher_joint_targets: body.slice(0, 15),
    upper_body_residuals: upper,
    hand_coordination: handCoordination,
    hand_joint_targets: createG1HandArtifactCommand(handCoordination).jointTargets
  };
}

function teacherJitIdentity(value: unknown): TeacherJitIdentity {
  if (!value || typeof value !== "object" || !("teacher_jit" in value)) {
    throw new Error("Teacher report has no JIT identity");
  }
  const identity = (value as { teacher_jit?: unknown }).teacher_jit;
  if (!identity || typeof identity !== "object") {
    throw new Error("Teacher report has no JIT identity");
  }
  return identity as TeacherJitIdentity;
}

function assertScenarioTarget(scenario: Scenario): void {
  if (!scenario.objects.some((object) => object.id === "assembly_rod" && object.portable)
    || !scenario.objects.some((object) => object.id === "pickup_stand" && !object.portable)
    || !scenario.zones.some((zone) => zone.id === "assembly_bay")) {
    throw new Error("Residual Workyard target does not exist in the real scenario");
  }
}

function assertSeedSplits(
  splits: WorkyardResidualTrainingContract["evaluation"]["seeds"]
): void {
  const values = Object.entries(splits);
  for (let left = 0; left < values.length; left += 1) {
    for (let right = left + 1; right < values.length; right += 1) {
      const [leftName, leftRange] = values[left]!;
      const [rightName, rightRange] = values[right]!;
      if (leftRange.first <= rightRange.last && rightRange.first <= leftRange.last) {
        throw new Error(`Residual seed splits overlap: ${leftName} and ${rightName}`);
      }
    }
  }
}

function assertLayout(
  actual: readonly { name: string; size: number }[],
  expected: readonly (readonly [string, number])[],
  label: string
): void {
  if (actual.length !== expected.length || actual.some((term, index) => (
    term.name !== expected[index]?.[0] || term.size !== expected[index]?.[1]
  ))) {
    throw new Error(`Residual Workyard ${label} layout does not match v4`);
  }
}

function assertExactValues(
  actual: readonly string[],
  expected: readonly string[],
  label: string
): void {
  if (actual.length !== expected.length || actual.some(
    (value, index) => value !== expected[index]
  )) {
    throw new Error(`Residual Workyard ${label} do not match the runtime`);
  }
}

function assertExactSet(
  actual: readonly string[],
  expected: readonly string[],
  label: string
): void {
  if (new Set(actual).size !== actual.length
    || actual.length !== expected.length
    || expected.some((value) => !actual.includes(value))) {
    throw new Error(`Residual Workyard ${label} do not match v4`);
  }
}

function validateDeclaredSize(
  value: { size: number; terms: readonly { size: number }[] },
  context: z.core.$RefinementCtx,
  path: PropertyKey[]
): void {
  const measured = value.terms.reduce((sum, term) => sum + term.size, 0);
  if (measured !== value.size) {
    context.addIssue({
      code: "custom",
      path,
      message: `Declared size ${value.size} does not match term size ${measured}`
    });
  }
}

function emptyHands(): G1HandCoordination {
  return G1HandCoordinationSchema.parse({
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
  });
}

function sha256(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}
