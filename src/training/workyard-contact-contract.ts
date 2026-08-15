import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { z } from "zod";
import type { Scenario } from "../domain/schema.js";

const UnitRateSchema = z.number().finite().min(0).max(1);
const SizedTermSchema = z.object({
  name: z.string().trim().min(1),
  size: z.number().int().positive()
}).strict();

const HandSynergyNames = [
  "left_opposition",
  "left_thumb",
  "left_index",
  "left_middle",
  "right_opposition",
  "right_thumb",
  "right_index",
  "right_middle"
] as const;

const WorkyardContactTrainingContractSchema = z.object({
  protocol: z.literal("hear-workyard-contact-training-contract-v2"),
  scenario_id: z.literal("humanoid_workyard"),
  parent_contract: z.literal("training/workyard-task-v4.json"),
  environment: z.object({
    framework: z.literal("mjlab"),
    task_id: z.literal("Hear-Workyard-Whole-Body-Reach-Hand-Synergy-G1-v2"),
    module: z.literal("training/workyard_contact_mjlab_env.py"),
    implementation_status: z.literal("implemented"),
    terminal_stage: z.literal("grasp"),
    excluded_stages: z.tuple([
      z.literal("lift"), z.literal("carry"), z.literal("place")
    ])
  }).strict(),
  timing: z.object({
    physics_step_seconds: z.literal(0.005),
    control_step_seconds: z.literal(0.02),
    episode_seconds: z.literal(8)
  }).strict(),
  qualified_inputs: z.object({
    locomotion_teacher: z.object({
      protocol: z.literal("hear-frozen-locomotion-teacher-v1"),
      root: z.string().min(1),
      jit: z.literal("g1_velocity_teacher.jit.pt"),
      report: z.literal("training-report.json"),
      gradient_parameter_count: z.literal(0),
      authority: z.literal("whole_body_reference_only")
    }).strict(),
    reach_policy: z.object({
      protocol: z.literal("hear-whole-body-reach-policy-deployment-v3"),
      root: z.string().min(1),
      jit: z.literal("workyard_reach.jit.pt"),
      report: z.literal("reach-policy-report.json"),
      jit_sha256: z.string().regex(/^[a-f0-9]{64}$/u).nullable(),
      source_checkpoint_sha256: z.string().regex(/^[a-f0-9]{64}$/u).nullable(),
      gradient_parameter_count: z.literal(0),
      action_size: z.literal(29),
      authority: z.literal("frozen_whole_body_reach")
    }).strict(),
    analytic_teacher_preflight: z.object({
      protocol: z.literal("hear-workyard-contact-analytic-teacher-preflight-v1"),
      report: z.string().min(1),
      must_pass: z.literal(true),
      minimum_environment_count: z.number().int().min(16),
      minimum_success_rate: UnitRateSchema,
      minimum_success_rate_per_active_hand: UnitRateSchema,
      maximum_contact_force_n: z.number().finite().positive(),
      minimum_opposing_normal_dot_maximum: z.number().finite().min(-1).max(0),
      both_active_hands_must_succeed: z.literal(true)
    }).strict()
  }).strict(),
  harness_executor: z.object({
    protocol: z.literal("hear-contact-pocket-dls-executor-v1"),
    runtime: z.literal("deterministic_no_gradient"),
    authority: z.literal("active_arm_terminal_alignment_only"),
    learner_can_change_executor_state: z.literal(false),
    base_assist_enabled: z.literal(false),
    sequence: z.tuple([
      z.literal("open_hand"),
      z.literal("retreat"),
      z.literal("near_pocket_alignment"),
      z.literal("open_hand_insertion"),
      z.literal("closure_geometry_latch"),
      z.literal("finger_closure"),
      z.literal("thumb_flexion"),
      z.literal("verified_opposing_contact")
    ]),
    closure_authority_latch: z.literal(
      "episode_local_after_first_geometry_gate"
    ),
    pose_hold: z.literal("closure_gate_then_measured_contact_pose"),
    terminal_pocket_max_joint_target_slew_rad_per_control_step: z.literal(
      0.001
    ),
    hand_contact_solref_time_constant_s: z.literal(0.04),
    force_release_threshold_n: z.number().finite().positive(),
    emergency_force_release_threshold_n: z.number().finite().positive(),
    maximum_closing_joint_lead_rad: z.literal(0.25),
    opposing_support_coordination: z.literal(0.4)
  }).strict(),
  learner: z.object({
    phase: z.literal("hand_synergy_contact_grasp"),
    role: z.literal("contact_conditioned_8d_hand_policy"),
    observation: z.object({
      protocol: z.literal("hear-workyard-contact-observation-v2"),
      size: z.literal(262),
      terms: z.array(SizedTermSchema).length(3),
      forbidden_terms: z.tuple([
        z.literal("teacher_stage"),
        z.literal("teacher_target_stage"),
        z.literal("analytic_teacher_action"),
        z.literal("closure_authority_latch_private_state")
      ])
    }).strict(),
    action: z.object({
      protocol: z.literal("hear-active-hand-synergy-action-v1"),
      size: z.literal(8),
      names: z.tuple(HandSynergyNames.map((name) => z.literal(name)) as [
        z.ZodLiteral<"left_opposition">,
        z.ZodLiteral<"left_thumb">,
        z.ZodLiteral<"left_index">,
        z.ZodLiteral<"left_middle">,
        z.ZodLiteral<"right_opposition">,
        z.ZodLiteral<"right_thumb">,
        z.ZodLiteral<"right_index">,
        z.ZodLiteral<"right_middle">
      ]),
      range: z.tuple([z.literal(-1), z.literal(1)]),
      coordination_step: z.literal(0.0075)
    }).strict(),
    authority: z.object({
      active_hand_only: z.literal(true),
      requires_typed_contact_capability: z.literal(true),
      requires_closure_geometry_latch: z.literal(true),
      inactive_hand_forced_zero: z.literal(true),
      locomotion_gradient: z.literal(false),
      waist_gradient: z.literal(false),
      reach_gradient: z.literal(false),
      checkpoint_mutation_outside_hand_actor: z.literal(false)
    }).strict()
  }).strict(),
  composition: z.object({
    protocol: z.literal(
      "hear-frozen-whole-body-reach-hand-synergy-composition-v2"
    ),
    learned_action_size: z.literal(8),
    frozen_reach_action_size: z.literal(29),
    logical_composed_action_size: z.literal(37),
    body_joint_command: z.literal(
      "frozen_whole_body_reach_plus_harness_terminal_active_arm_executor"
    ),
    hand_joint_command: z.literal("authorized_active_hand_synergy_only")
  }).strict(),
  training: z.object({
    protocol: z.literal("hear-contact-dagger-ppo-retention-v1"),
    accelerator: z.literal("colab_pro_gpu"),
    local_training_permitted: z.literal(false),
    numerical_guard: z.object({
      protocol: z.literal("hear-vector-env-numerical-recovery-v1"),
      scope: z.literal("single_environment_reset"),
      maximum_recovery_count: z.number().int().positive(),
      maximum_recovery_rate: z.number().finite().positive().max(0.001),
      evaluation_recovery_permitted: z.literal(false)
    }).strict(),
    dagger: z.object({
      steps: z.number().int().positive(),
      learning_rate: z.number().finite().positive(),
      teacher_beta_initial: UnitRateSchema,
      teacher_beta_final: UnitRateSchema,
      loss: z.literal(
        "authority_weighted_smooth_l1_plus_excess_std_penalty"
      ),
      maximum_action_std: UnitRateSchema,
      label_scope: z.literal("authorized_8d_hand_action"),
      minimum_label_coverage: z.literal(1)
    }).strict(),
    ppo: z.object({
      iterations: z.number().int().positive(),
      environments: z.number().int().positive(),
      steps_per_environment: z.number().int().positive(),
      critic_warmup_iterations: z.number().int().nonnegative(),
      actor_learning_rate: z.number().finite().positive(),
      critic_learning_rate: z.number().finite().positive(),
      rollout_teacher_coefficient: z.number().finite().positive(),
      teacher_supervision: z.literal("every_stored_learner_rollout_state"),
      checkpoint_selection: z.literal(
        "rollback_to_dagger_if_retention_gate_fails"
      )
    }).strict(),
    checkpoint_selection: z.object({
      comparison_episodes: z.number().int().positive(),
      control_steps_per_episode: z.number().int().positive(),
      held_out_seed_first: z.number().int().nonnegative(),
      held_out_seed_last: z.number().int().nonnegative(),
      ppo_success_rate_delta_minimum: z.number().finite().min(-1).max(0),
      ppo_success_count_per_active_hand_minimum: z.number().int().positive(),
      ppo_success_rate_per_active_hand_minimum: UnitRateSchema,
      ppo_maximum_contact_force_n: z.number().finite().positive(),
      ppo_object_lost_count_maximum: z.literal(0),
      ppo_fall_count_maximum: z.literal(0),
      ppo_non_foot_ground_count_maximum: z.literal(0),
      ppo_numerical_instability_count_maximum: z.literal(0),
      ppo_precontact_object_contact_count_maximum: z.literal(0),
      ppo_unauthorized_hand_action_maximum: z.literal(0),
      ppo_inactive_hand_coordination_maximum: z.literal(0),
      ppo_base_assist_environment_count_maximum: z.literal(0),
      ppo_rollout_teacher_label_coverage_minimum: z.literal(1),
      ppo_mean_action_std_maximum: UnitRateSchema
    }).strict()
  }).strict(),
  evaluation: z.object({
    episodes: z.literal(500),
    control_steps_per_episode: z.literal(400),
    held_out_seed_protocol: z.literal("per_environment_deterministic_v1"),
    held_out_seed_first: z.number().int().nonnegative(),
    held_out_seed_last: z.number().int().nonnegative(),
    verified_grasp_success_rate_minimum: UnitRateSchema,
    verified_grasp_success_rate_per_active_hand_minimum: UnitRateSchema,
    success_count_per_active_hand_minimum: z.number().int().positive(),
    maximum_contact_force_n: z.number().finite().positive(),
    minimum_opposing_normal_dot_maximum: z.number().finite().min(-1).max(0),
    object_lost_count_maximum: z.literal(0),
    fall_count_maximum: z.literal(0),
    non_foot_ground_count_maximum: z.literal(0),
    numerical_instability_count_maximum: z.literal(0),
    precontact_object_contact_count_maximum: z.literal(0),
    unauthorized_hand_action_maximum: z.literal(0),
    inactive_hand_coordination_maximum: z.literal(0),
    frozen_gradient_parameter_count: z.literal(0),
    base_assist_environment_count_maximum: z.literal(0),
    finite_required: z.literal(true),
    environment_closed_required: z.literal(true)
  }).strict()
}).strict().superRefine((contract, context) => {
  const observationSize = contract.learner.observation.terms.reduce(
    (sum, term) => sum + term.size, 0
  );
  if (observationSize !== contract.learner.observation.size) {
    context.addIssue({
      code: "custom",
      path: ["learner", "observation", "terms"],
      message: "Contact observation terms do not sum to 262"
    });
  }
  if (contract.training.dagger.teacher_beta_final
    > contract.training.dagger.teacher_beta_initial) {
    context.addIssue({
      code: "custom",
      path: ["training", "dagger"],
      message: "DAgger teacher beta must not increase"
    });
  }
  if (contract.training.ppo.critic_warmup_iterations
    >= contract.training.ppo.iterations) {
    context.addIssue({
      code: "custom",
      path: ["training", "ppo", "critic_warmup_iterations"],
      message: "PPO needs an actor-update iteration after critic warm-up"
    });
  }
  if (contract.evaluation.held_out_seed_last
    - contract.evaluation.held_out_seed_first + 1
    !== contract.evaluation.episodes) {
    context.addIssue({
      code: "custom",
      path: ["evaluation"],
      message: "Held-out seed range must contain exactly 500 episodes"
    });
  }
  if (contract.training.checkpoint_selection.held_out_seed_last
    - contract.training.checkpoint_selection.held_out_seed_first + 1
    !== contract.training.checkpoint_selection.comparison_episodes) {
    context.addIssue({
      code: "custom",
      path: ["training", "checkpoint_selection"],
      message: "Checkpoint comparison seed range must match its episode count"
    });
  }
});

export type WorkyardContactTrainingContract = z.infer<
  typeof WorkyardContactTrainingContractSchema
>;

export interface WorkyardContactArtifactEvidence {
  ready: boolean;
  locomotion_jit_sha256: string | null;
  reach_jit_sha256: string | null;
  preflight_report_sha256: string | null;
  preflight_success_rate: number | null;
  blockers: string[];
}

export interface WorkyardContactDryRunReport {
  protocol: "hear-workyard-contact-training-dry-run-v2";
  contract_sha256: string;
  scenario_id: "humanoid_workyard";
  learner: {
    observation_size: 262;
    action_size: 8;
    logical_composed_action_size: 37;
    active_hand_only: true;
  };
  terminal_stage: "grasp";
  held_out_episode_count: 500;
  artifacts: WorkyardContactArtifactEvidence;
  contract_ready: true;
  colab_training_ready: boolean;
  blockers: string[];
}

export async function loadWorkyardContactTrainingContract(
  path = resolve("training/workyard-contact-task-v1.json")
): Promise<WorkyardContactTrainingContract> {
  const value = JSON.parse(await readFile(resolve(path), "utf8")) as unknown;
  return WorkyardContactTrainingContractSchema.parse(value);
}

export async function inspectWorkyardContactArtifacts(
  contractInput: WorkyardContactTrainingContract,
  contractPath = resolve("training/workyard-contact-task-v1.json")
): Promise<WorkyardContactArtifactEvidence> {
  const contract = WorkyardContactTrainingContractSchema.parse(contractInput);
  const repositoryRoot = resolve(dirname(contractPath), "..");
  const locomotionRoot = resolve(
    repositoryRoot, contract.qualified_inputs.locomotion_teacher.root
  );
  const reachRoot = resolve(
    repositoryRoot, contract.qualified_inputs.reach_policy.root
  );
  const preflightPath = resolve(
    repositoryRoot, contract.qualified_inputs.analytic_teacher_preflight.report
  );
  const blockers: string[] = [];
  let locomotionJit: Buffer | null = null;
  let reachJit: Buffer | null = null;
  let preflightBytes: Buffer | null = null;
  let preflightSuccessRate: number | null = null;
  try {
    const [locomotionJitBytes, locomotionReportBytes, reachJitBytes,
      reachReportBytes, teacherGateBytes] = await Promise.all([
      readFile(resolve(locomotionRoot, contract.qualified_inputs.locomotion_teacher.jit)),
      readFile(resolve(locomotionRoot, contract.qualified_inputs.locomotion_teacher.report)),
      readFile(resolve(reachRoot, contract.qualified_inputs.reach_policy.jit)),
      readFile(resolve(reachRoot, contract.qualified_inputs.reach_policy.report)),
      readFile(preflightPath)
    ]);
    locomotionJit = locomotionJitBytes;
    reachJit = reachJitBytes;
    preflightBytes = teacherGateBytes;
    const locomotionReport = JSON.parse(locomotionReportBytes.toString("utf8")) as {
      teacher_jit?: { file?: string; bytes?: number; sha256?: string;
        input_size?: number; output_size?: number; batch_dynamic?: boolean };
    };
    const locomotionIdentity = locomotionReport.teacher_jit;
    if (!locomotionIdentity
      || locomotionIdentity.file !== contract.qualified_inputs.locomotion_teacher.jit
      || locomotionIdentity.bytes !== locomotionJit.byteLength
      || locomotionIdentity.sha256 !== sha256(locomotionJit)
      || locomotionIdentity.input_size !== 99
      || locomotionIdentity.output_size !== 29
      || locomotionIdentity.batch_dynamic !== true) {
      blockers.push("Frozen locomotion teacher identity is invalid");
    }
    const reachReport = JSON.parse(reachReportBytes.toString("utf8")) as {
      protocol?: string;
      deployment?: { protocol?: string; accepted?: boolean;
        controller_mode?: string; terminal_assistance_step_count?: number;
        minimum_support_margin_m?: number;
        maximum_foot_planar_displacement_m?: number;
        maximum_foot_slip_speed_m_s?: number;
        double_support_loss_rate_maximum?: number;
        no_foot_contact_rate_maximum?: number };
      source?: { checkpoint_sha256?: string; phase_one_accepted?: boolean;
        hand_checkpoint_expansion_authorized?: boolean };
      policy?: { file?: string; bytes?: number; sha256?: string;
        input?: string; input_size?: number; output?: string;
        output_size?: number; batch_dynamic?: boolean;
        gradient_parameter_count?: number };
    };
    if (contract.qualified_inputs.reach_policy.source_checkpoint_sha256 === null
      || contract.qualified_inputs.reach_policy.jit_sha256 === null
      || reachReport.protocol !== contract.qualified_inputs.reach_policy.protocol
      || reachReport.deployment?.protocol
        !== "hear-typescript-mujoco-reach-deployment-gate-v1"
      || reachReport.deployment?.accepted !== true
      || reachReport.deployment?.controller_mode !== "learned_policy_only"
      || reachReport.deployment?.terminal_assistance_step_count !== 0
      || !((reachReport.deployment?.minimum_support_margin_m
        ?? Number.NEGATIVE_INFINITY) >= -0.04)
      || !((reachReport.deployment?.maximum_foot_planar_displacement_m
        ?? Number.POSITIVE_INFINITY) <= 0.08)
      || !((reachReport.deployment?.maximum_foot_slip_speed_m_s
        ?? Number.POSITIVE_INFINITY) <= 0.20)
      || !((reachReport.deployment?.double_support_loss_rate_maximum
        ?? Number.POSITIVE_INFINITY) <= 0.10)
      || !((reachReport.deployment?.no_foot_contact_rate_maximum
        ?? Number.POSITIVE_INFINITY) <= 0.01)
      || reachReport.source?.checkpoint_sha256
        !== contract.qualified_inputs.reach_policy.source_checkpoint_sha256
      || reachReport.source?.phase_one_accepted !== true
      || reachReport.source?.hand_checkpoint_expansion_authorized !== true
      || reachReport.policy?.file !== contract.qualified_inputs.reach_policy.jit
      || reachReport.policy?.bytes !== reachJit.byteLength
      || reachReport.policy?.sha256 !== sha256(reachJit)
      || reachReport.policy.sha256 !== contract.qualified_inputs.reach_policy.jit_sha256
      || reachReport.policy.input
        !== "hear-workyard-whole-body-reach-observation-v5"
      || reachReport.policy.input_size !== 246
      || reachReport.policy.output !== "bounded-whole-body-reach-mean"
      || reachReport.policy.output_size !== 29
      || reachReport.policy.batch_dynamic !== true
      || reachReport.policy.gradient_parameter_count !== 0) {
      blockers.push("Qualified whole-body reach policy identity is invalid or unpinned");
    }
    const preflight = JSON.parse(teacherGateBytes.toString("utf8")) as {
      gate?: { protocol?: string; passed?: boolean; checks?: Record<string, boolean> };
      evaluation?: { environment_count?: number; success_rate?: number;
        success_rate_by_active_hand?: Record<string, number>;
        maximum_active_hand_force_n?: number; minimum_opposing_normal_dot?: number;
        success_count_by_active_hand?: Record<string, number>;
        frozen_locomotion?: { gradient_parameter_count?: number };
        frozen_reach?: { gradient_parameter_count?: number; jit_sha256?: string } };
      contract?: { frozen_locomotion?: { gradient_parameter_count?: number };
        frozen_reach?: { gradient_parameter_count?: number; jit_sha256?: string } };
    };
    preflightSuccessRate = preflight.evaluation?.success_rate ?? null;
    const gateContract = contract.qualified_inputs.analytic_teacher_preflight;
    if (preflight.gate?.protocol !== gateContract.protocol
      || preflight.gate.passed !== true
      || !preflight.gate.checks
      || !Object.values(preflight.gate.checks).every(Boolean)
      || (preflight.evaluation?.environment_count ?? 0)
        < gateContract.minimum_environment_count
      || (preflight.evaluation?.success_rate ?? 0) < gateContract.minimum_success_rate
      || (preflight.evaluation?.success_rate_by_active_hand?.left ?? 0)
        < gateContract.minimum_success_rate_per_active_hand
      || (preflight.evaluation?.success_rate_by_active_hand?.right ?? 0)
        < gateContract.minimum_success_rate_per_active_hand
      || (preflight.evaluation?.maximum_active_hand_force_n ?? Infinity)
        > gateContract.maximum_contact_force_n
      || (preflight.evaluation?.minimum_opposing_normal_dot ?? 1)
        > gateContract.minimum_opposing_normal_dot_maximum
      || (preflight.evaluation?.success_count_by_active_hand?.left ?? 0) < 1
      || (preflight.evaluation?.success_count_by_active_hand?.right ?? 0) < 1
      || (preflight.evaluation?.frozen_locomotion
        ?? preflight.contract?.frozen_locomotion)?.gradient_parameter_count !== 0
      || (preflight.evaluation?.frozen_reach
        ?? preflight.contract?.frozen_reach)?.gradient_parameter_count !== 0
      || (preflight.evaluation?.frozen_reach
        ?? preflight.contract?.frozen_reach)?.jit_sha256 !== sha256(reachJit)) {
      blockers.push("Analytic contact teacher preflight is not qualified");
    }
  } catch (error) {
    blockers.push(error instanceof Error ? error.message : String(error));
  }
  return {
    ready: blockers.length === 0,
    locomotion_jit_sha256: locomotionJit ? sha256(locomotionJit) : null,
    reach_jit_sha256: reachJit ? sha256(reachJit) : null,
    preflight_report_sha256: preflightBytes ? sha256(preflightBytes) : null,
    preflight_success_rate: preflightSuccessRate,
    blockers
  };
}

export function dryRunWorkyardContactTrainingContract(
  contractInput: WorkyardContactTrainingContract,
  scenario: Scenario,
  artifacts: WorkyardContactArtifactEvidence
): WorkyardContactDryRunReport {
  const contract = WorkyardContactTrainingContractSchema.parse(contractInput);
  const rod = scenario.objects.find((object) => object.id === "assembly_rod");
  const stand = scenario.objects.find((object) => object.id === "pickup_stand");
  if (!rod?.portable || !stand || stand.portable) {
    throw new Error("Contact Workyard target does not exist in the real scenario");
  }
  if (rod.size.x !== 0.06 || rod.size.y !== 0.22 || rod.size.z !== 0.06
    || rod.capability?.shape !== "cylinder"
    || rod.capability.mass_kg !== 0.35
    || rod.capability.friction?.sliding !== 0.8
    || rod.capability.friction.torsional !== 0.012
    || rod.capability.friction.rolling !== 0.002) {
    throw new Error("Contact Workyard rod geometry drifted from the trained MuJoCo plant");
  }
  if (stand.size.x !== 0.12 || stand.size.y !== 0.01 || stand.size.z !== 0.12) {
    throw new Error("Contact Workyard pickup stand drifted from the trained MuJoCo plant");
  }
  const expectedTerms = [
    ["frozen_reach_observation", 246],
    ["hand_coordination", 8],
    ["previous_authorized_hand_action", 8]
  ] as const;
  if (contract.learner.observation.terms.some((term, index) => (
    term.name !== expectedTerms[index]?.[0] || term.size !== expectedTerms[index]?.[1]
  ))) {
    throw new Error("Contact learner observation layout drifted");
  }
  if (contract.learner.action.names.some(
    (name, index) => name !== HandSynergyNames[index]
  )) {
    throw new Error("Contact learner hand synergy order drifted");
  }
  const blockers = [...artifacts.blockers];
  return {
    protocol: "hear-workyard-contact-training-dry-run-v2",
    contract_sha256: sha256(Buffer.from(JSON.stringify(contract))),
    scenario_id: contract.scenario_id,
    learner: {
      observation_size: 262,
      action_size: 8,
      logical_composed_action_size: 37,
      active_hand_only: true
    },
    terminal_stage: "grasp",
    held_out_episode_count: contract.evaluation.episodes,
    artifacts,
    contract_ready: true,
    colab_training_ready: blockers.length === 0,
    blockers
  };
}

function sha256(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}
