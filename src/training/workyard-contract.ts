import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { z } from "zod";
import type { Scenario } from "../domain/schema.js";
import {
  HUMANOID_LEARNED_POLICY_CAPABILITIES
} from "../domain/humanoid-policy.js";
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
  HUMANOID_JOINT_NAMES
} from "../world/humanoid/model.js";
import {
  HUMANOID_POLICY_OBSERVATION_FEATURES
} from "../world/humanoid/whole-body-controller.js";

const WORKYARD_TRAINING_STAGES = [
  "reach",
  "contact",
  "grasp",
  "lift",
  "carry",
  "place"
] as const;

const TrainingStageSchema = z.enum(WORKYARD_TRAINING_STAGES);
const UnitRateSchema = z.number().finite().min(0).max(1);
const SeedRangeSchema = z.object({
  first: z.number().int().nonnegative(),
  last: z.number().int().nonnegative()
}).strict().refine((range) => range.last >= range.first, {
  message: "Training seed range must not be reversed"
});

const ObservationTermSchema = z.object({
  name: z.string().trim().min(1),
  size: z.number().int().positive(),
  source: z.enum(["policy_state", "mujoco_state", "task_command"]),
  frame: z.enum(["joint", "pelvis", "task", "none"]),
  normalization: z.enum(["identity", "standardized", "unit_interval", "one_hot"])
}).strict();

const ActionTermSchema = z.object({
  name: z.string().trim().min(1),
  size: z.number().int().positive(),
  representation: z.enum(["reference_position_residual", "synergy_delta"]),
  minimum: z.number().finite(),
  maximum: z.number().finite(),
  scale: z.number().finite().positive()
}).strict().refine((term) => term.maximum > term.minimum, {
  message: "Training action maximum must exceed its minimum"
});

const RewardAuthoritySchema = z.enum([
  "mujoco_state",
  "humanoid_grasp_tracker",
  "humanoid_object_zone_relation",
  "humanoid_object_settled_support",
  "humanoid_physical_safety",
  "policy_action"
]);

export const WorkyardTrainingContractSchema = z.object({
  protocol: z.literal("hear-workyard-training-contract-v2"),
  scenario_id: z.literal("humanoid_workyard"),
  environment: z.object({
    framework: z.literal("mjlab"),
    task_id: z.literal("Hear-Workyard-Skill-Conditioned-G1-v2"),
    module: z.string().trim().min(1),
    implementation_status: z.enum(["contract_only", "implemented"])
  }).strict(),
  morphology: z.object({
    id: z.literal(G1_MORPHOLOGY.id),
    body_joint_names: z.array(z.enum(HUMANOID_JOINT_NAMES)),
    hand_joint_names: z.array(z.enum(G1_HAND_JOINT_NAMES))
  }).strict(),
  timing: z.object({
    physics_step_seconds: z.number().finite().positive(),
    control_step_seconds: z.number().finite().positive(),
    episode_seconds: z.number().finite().positive()
  }).strict(),
  policy: z.object({
    role: z.literal("skill_call_conditioned_student"),
    conditioning_protocol: z.literal("humanoid-embodied-skill-call-v2"),
    capabilities: z.array(z.enum(HUMANOID_LEARNED_POLICY_CAPABILITIES)).min(1),
    observation_features: z.array(z.enum(HUMANOID_POLICY_OBSERVATION_FEATURES)).min(1)
  }).strict(),
  task: z.object({
    object_id: z.string().trim().min(1),
    source_support_id: z.string().trim().min(1),
    target_zone_id: z.string().trim().min(1),
    teacher_stages: z.array(TrainingStageSchema)
      .length(WORKYARD_TRAINING_STAGES.length),
    deployment_authority: z.literal("agent_harness_skill_call"),
    student_forbidden_observations: z.array(z.enum([
      "teacher_stage",
      "teacher_target_stage",
      "full_task_automaton_state"
    ])).length(3)
  }).strict(),
  training_strategy: z.object({
    protocol: z.literal("hear-teacher-student-skill-training-v1"),
    teacher_role: z.literal("curriculum_command_and_label_generator"),
    teacher_deployed: z.literal(false),
    student_role: z.literal("autonomous_skill_window_executor"),
    student_deployed: z.literal(true),
    distillation: z.literal("context_gated_mixture_of_experts"),
    trajectory_sources: z.array(z.enum([
      "teacher_success",
      "student_success",
      "student_failure",
      "harness_recovery"
    ])).length(4),
    outcome_attribution: z.literal(
      "policy_segment_and_intervention_segment_separated"
    )
  }).strict(),
  observation: z.object({
    protocol: z.literal("hear-workyard-observation-v2"),
    size: z.number().int().positive(),
    history_steps: z.number().int().positive(),
    terms: z.array(ObservationTermSchema).min(1)
  }).strict(),
  action: z.object({
    protocol: z.literal("hear-workyard-action-v1"),
    size: z.number().int().positive(),
    terms: z.array(ActionTermSchema).min(1)
  }).strict(),
  rewards: z.object({
    terms: z.array(z.object({
      id: z.string().trim().min(1),
      stage: z.union([TrainingStageSchema, z.literal("all")]),
      authority: RewardAuthoritySchema,
      objective: z.enum(["maximize", "minimize"]),
      weight: z.number().finite().positive()
    }).strict()).min(1)
  }).strict(),
  curriculum: z.array(z.object({
    stage: TrainingStageSchema,
    minimum_episodes: z.number().int().positive(),
    promotion_success_rate: UnitRateSchema,
    previous_stage_replay_fraction: UnitRateSchema
  }).strict()).length(WORKYARD_TRAINING_STAGES.length),
  randomization: z.object({
    object_position_jitter_m: z.number().finite().nonnegative(),
    target_position_jitter_m: z.number().finite().nonnegative(),
    object_mass_scale: z.tuple([z.number().finite().positive(), z.number().finite().positive()]),
    friction_scale: z.tuple([z.number().finite().positive(), z.number().finite().positive()]),
    actuator_strength_scale: z.tuple([
      z.number().finite().positive(),
      z.number().finite().positive()
    ]),
    observation_latency_control_steps: z.tuple([
      z.number().int().nonnegative(),
      z.number().int().nonnegative()
    ])
  }).strict(),
  evaluation: z.object({
    seeds: z.object({
      training: SeedRangeSchema,
      validation: SeedRangeSchema,
      held_out: SeedRangeSchema
    }).strict(),
    episodes_per_stage: z.number().int().positive(),
    full_task_episodes: z.number().int().positive(),
    stage_success_rate_minimum: z.record(TrainingStageSchema, UnitRateSchema),
    full_task_success_rate_minimum: UnitRateSchema,
    fall_rate_maximum: UnitRateSchema,
    dropped_object_rate_maximum: UnitRateSchema,
    minimum_support_margin_m: z.number().finite(),
    maximum_foot_slip_mps: z.number().finite().nonnegative(),
    minimum_joint_limit_margin_rad: z.number().finite().nonnegative(),
    maximum_requested_effort_utilization: z.number().finite().positive(),
    learned_policy_frame_ratio_minimum: UnitRateSchema,
    reference_control_frame_ratio_maximum: UnitRateSchema
  }).strict()
}).strict().superRefine((contract, context) => {
  const ratio = contract.timing.control_step_seconds
    / contract.timing.physics_step_seconds;
  if (ratio < 1 || Math.abs(ratio - Math.round(ratio)) > 1e-9) {
    context.addIssue({
      code: "custom",
      path: ["timing"],
      message: "Control timing must be an integer multiple of physics timing"
    });
  }
  validateDeclaredSize(contract.observation, context, "observation");
  validateDeclaredSize(contract.action, context, "action");
  validateUnique(contract.observation.terms.map((term) => term.name), context, [
    "observation",
    "terms"
  ]);
  validateUnique(contract.action.terms.map((term) => term.name), context, [
    "action",
    "terms"
  ]);
  validateUnique(contract.rewards.terms.map((term) => term.id), context, [
    "rewards",
    "terms"
  ]);
  validateUnique(contract.task.student_forbidden_observations, context, [
    "task",
    "student_forbidden_observations"
  ]);
  validateUnique(contract.training_strategy.trajectory_sources, context, [
    "training_strategy",
    "trajectory_sources"
  ]);
  for (const [name, range] of Object.entries(contract.randomization).filter(
    ([, value]) => Array.isArray(value)
  ) as Array<[string, readonly [number, number]]>) {
    if (range[1] < range[0]) {
      context.addIssue({
        code: "custom",
        path: ["randomization", name],
        message: "Randomization range must not be reversed"
      });
    }
  }
});

export type WorkyardTrainingContract = z.infer<
  typeof WorkyardTrainingContractSchema
>;

export interface WorkyardTrainingDryRunReport {
  protocol: "hear-workyard-training-dry-run-v2";
  contract_sha256: string;
  scenario_id: "humanoid_workyard";
  target: {
    object_id: string;
    source_support_id: string;
    target_zone_id: string;
  };
  morphology: {
    id: typeof G1_MORPHOLOGY.id;
    body_joint_count: number;
    hand_joint_count: number;
  };
  observation: {
    size: number;
    term_offsets: Record<string, { offset: number; size: number }>;
  };
  action: {
    size: number;
    body_residual_count: number;
    hand_synergy_count: number;
  };
  curriculum: typeof WORKYARD_TRAINING_STAGES;
  evidence_authorities: string[];
  contract_ready: true;
  colab_smoke_ready: boolean;
  blockers: string[];
}

const ExpectedObservationLayout = [
  ["body_joint_position_offset", HUMANOID_JOINT_NAMES.length],
  ["body_joint_velocity", HUMANOID_JOINT_NAMES.length],
  ["previous_body_action", HUMANOID_JOINT_NAMES.length],
  ["hand_joint_position_fraction", G1_HAND_JOINT_NAMES.length],
  ["hand_joint_velocity", G1_HAND_JOINT_NAMES.length],
  ["previous_hand_action", 8],
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

const ExpectedActionLayout = [
  ["body_joint_position_residual", HUMANOID_JOINT_NAMES.length],
  ["hand_synergy_delta", 8]
] as const;

const RequiredObservationFeatures = [
  "proprioception",
  "command_history",
  "root_kinematics",
  "hand_state",
  "end_effector_state",
  "contact_state",
  "object_state",
  "task_space_command",
  "grasp_command"
] as const;

const RequiredCapabilities = [
  "balance",
  "locomotion",
  "joint_reference_tracking",
  "contact_rich_manipulation"
] as const;

export async function loadWorkyardTrainingContract(
  path = resolve("training/workyard-task-v2.json")
): Promise<WorkyardTrainingContract> {
  const value = JSON.parse(await readFile(resolve(path), "utf8")) as unknown;
  return WorkyardTrainingContractSchema.parse(value);
}

export function dryRunWorkyardTrainingContract(
  contractInput: WorkyardTrainingContract,
  scenario: Scenario
): WorkyardTrainingDryRunReport {
  const contract = WorkyardTrainingContractSchema.parse(contractInput);
  assertExactValues(
    contract.morphology.body_joint_names,
    HUMANOID_JOINT_NAMES,
    "body joint layout"
  );
  assertExactValues(
    contract.morphology.hand_joint_names,
    G1_HAND_JOINT_NAMES,
    "hand joint layout"
  );
  assertLayout(contract.observation.terms, ExpectedObservationLayout, "observation");
  assertLayout(contract.action.terms, ExpectedActionLayout, "action");
  assertExactValues(
    contract.task.teacher_stages,
    WORKYARD_TRAINING_STAGES,
    "teacher curriculum stages"
  );
  assertExactValues(
    contract.curriculum.map((stage) => stage.stage),
    WORKYARD_TRAINING_STAGES,
    "curriculum stages"
  );
  assertExactSet(
    contract.policy.observation_features,
    RequiredObservationFeatures,
    "policy observation features"
  );
  assertExactSet(contract.policy.capabilities, RequiredCapabilities, "policy capabilities");
  assertScenarioTarget(contract, scenario);
  assertSeedSplits(contract.evaluation.seeds);
  for (const stage of WORKYARD_TRAINING_STAGES) {
    if (!contract.rewards.terms.some((reward) => (
      reward.stage === stage || reward.stage === "all"
    ))) {
      throw new Error(`Workyard reward contract has no signal for stage: ${stage}`);
    }
  }
  const neutralAction = Array.from({ length: contract.action.size }, () => 0);
  const decoded = decodeWorkyardPolicyAction(
    contract,
    neutralAction,
    G1HandCoordinationSchema.parse({
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
    })
  );
  if (decoded.body_joint_position_residuals.some((value) => value !== 0)
    || Object.values(decoded.hand_joint_targets).some((value) => value !== 0)) {
    throw new Error("A neutral Workyard action must preserve the neutral targets");
  }
  const blockers = contract.environment.implementation_status === "implemented"
    ? []
    : [
        `Mjlab task ${contract.environment.task_id} is still contract-only; `
          + `${contract.environment.module} must be implemented before Colab GPU use`
      ];
  return {
    protocol: "hear-workyard-training-dry-run-v2",
    contract_sha256: createHash("sha256")
      .update(JSON.stringify(contract))
      .digest("hex"),
    scenario_id: contract.scenario_id,
    target: {
      object_id: contract.task.object_id,
      source_support_id: contract.task.source_support_id,
      target_zone_id: contract.task.target_zone_id
    },
    morphology: {
      id: G1_MORPHOLOGY.id,
      body_joint_count: HUMANOID_JOINT_NAMES.length,
      hand_joint_count: G1_HAND_JOINT_NAMES.length
    },
    observation: {
      size: contract.observation.size,
      term_offsets: termOffsets(contract.observation.terms)
    },
    action: {
      size: contract.action.size,
      body_residual_count: decoded.body_joint_position_residuals.length,
      hand_synergy_count: 8
    },
    curriculum: WORKYARD_TRAINING_STAGES,
    evidence_authorities: [...new Set(
      contract.rewards.terms.map((reward) => reward.authority)
    )].sort(compareCodePoints),
    contract_ready: true,
    colab_smoke_ready: blockers.length === 0,
    blockers
  };
}

export function decodeWorkyardPolicyAction(
  contractInput: WorkyardTrainingContract,
  action: readonly number[],
  currentHands: G1HandCoordination
): {
  body_joint_position_residuals: number[];
  hand_coordination: G1HandCoordination;
  hand_joint_targets: ReturnType<typeof createG1HandArtifactCommand>["jointTargets"];
} {
  const contract = WorkyardTrainingContractSchema.parse(contractInput);
  assertLayout(contract.action.terms, ExpectedActionLayout, "action");
  if (action.length !== contract.action.size || !action.every(Number.isFinite)) {
    throw new Error(`Workyard policy action must contain ${contract.action.size} finite values`);
  }
  for (const [index, value] of action.entries()) {
    if (value < -1 || value > 1) {
      throw new Error(`Workyard policy action ${index} is outside [-1, 1]`);
    }
  }
  const current = G1HandCoordinationSchema.parse(currentHands);
  const bodyScale = contract.action.terms[0]!.scale;
  const handScale = contract.action.terms[1]!.scale;
  const handOffset = HUMANOID_JOINT_NAMES.length;
  const amount = (index: number, existing: number): number => Math.max(
    0,
    Math.min(1, existing + action[handOffset + index]! * handScale)
  );
  const handCoordination = G1HandCoordinationSchema.parse({
    left: {
      thumb_opposition: amount(0, current.left.thumb_opposition),
      thumb_curl: amount(1, current.left.thumb_curl),
      index_curl: amount(2, current.left.index_curl),
      middle_curl: amount(3, current.left.middle_curl)
    },
    right: {
      thumb_opposition: amount(4, current.right.thumb_opposition),
      thumb_curl: amount(5, current.right.thumb_curl),
      index_curl: amount(6, current.right.index_curl),
      middle_curl: amount(7, current.right.middle_curl)
    }
  });
  const handCommand = createG1HandArtifactCommand(handCoordination);
  return {
    body_joint_position_residuals: action.slice(0, handOffset).map(
      (value) => value * bodyScale
    ),
    hand_coordination: handCoordination,
    hand_joint_targets: handCommand.jointTargets
  };
}

function assertScenarioTarget(
  contract: WorkyardTrainingContract,
  scenario: Scenario
): void {
  const object = scenario.objects.find((candidate) => (
    candidate.id === contract.task.object_id
  ));
  if (!object?.portable) {
    throw new Error(`Workyard training object must be portable: ${contract.task.object_id}`);
  }
  const support = scenario.objects.find((candidate) => (
    candidate.id === contract.task.source_support_id
  ));
  if (!support || support.portable) {
    throw new Error(
      `Workyard source support must exist and be static: ${contract.task.source_support_id}`
    );
  }
  if (!scenario.zones.some((zone) => zone.id === contract.task.target_zone_id)) {
    throw new Error(`Workyard target zone is missing: ${contract.task.target_zone_id}`);
  }
  const goalMatches = scenario.default_goal.predicates.some((predicate) => (
    predicate.type === "object_placed"
      && predicate.object_id === contract.task.object_id
      && predicate.zone_id === contract.task.target_zone_id
  ));
  if (!goalMatches) {
    throw new Error("Workyard contract target does not match the scenario default goal");
  }
}

function assertSeedSplits(
  splits: WorkyardTrainingContract["evaluation"]["seeds"]
): void {
  const entries = Object.entries(splits);
  for (let left = 0; left < entries.length; left += 1) {
    for (let right = left + 1; right < entries.length; right += 1) {
      const [leftName, leftRange] = entries[left]!;
      const [rightName, rightRange] = entries[right]!;
      if (leftRange.first <= rightRange.last && rightRange.first <= leftRange.last) {
        throw new Error(`Workyard seed splits overlap: ${leftName} and ${rightName}`);
      }
    }
  }
}

function termOffsets(
  terms: readonly { name: string; size: number }[]
): Record<string, { offset: number; size: number }> {
  let offset = 0;
  return Object.fromEntries(terms.map((term) => {
    const entry = [term.name, { offset, size: term.size }] as const;
    offset += term.size;
    return entry;
  }));
}

function assertLayout(
  actual: readonly { name: string; size: number }[],
  expected: readonly (readonly [name: string, size: number])[],
  label: string
): void {
  if (actual.length !== expected.length || actual.some((term, index) => (
    term.name !== expected[index]?.[0] || term.size !== expected[index]?.[1]
  ))) {
    throw new Error(`Workyard ${label} layout does not match protocol v1`);
  }
}

function assertExactValues(
  actual: readonly string[],
  expected: readonly string[],
  label: string
): void {
  if (actual.length !== expected.length
    || actual.some((value, index) => value !== expected[index])) {
    throw new Error(`Workyard ${label} does not match the runtime`);
  }
}

function assertExactSet(
  actual: readonly string[],
  expected: readonly string[],
  label: string
): void {
  const actualSet = new Set(actual);
  if (actualSet.size !== actual.length
    || actualSet.size !== expected.length
    || expected.some((value) => !actualSet.has(value))) {
    throw new Error(`Workyard ${label} does not match protocol v1`);
  }
}

function validateDeclaredSize(
  space: { size: number; terms: readonly { size: number }[] },
  context: z.core.$RefinementCtx,
  path: "observation" | "action"
): void {
  const measured = space.terms.reduce((total, term) => total + term.size, 0);
  if (space.size !== measured) {
    context.addIssue({
      code: "custom",
      path: [path, "size"],
      message: `Declared size ${space.size} does not match term size ${measured}`
    });
  }
}

function validateUnique(
  values: readonly string[],
  context: z.core.$RefinementCtx,
  path: PropertyKey[]
): void {
  if (new Set(values).size !== values.length) {
    context.addIssue({
      code: "custom",
      path,
      message: "Training contract identifiers must be unique"
    });
  }
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
