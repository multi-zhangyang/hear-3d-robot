import { z } from "zod";
import {
  JsonValueSchema,
  Vec3Schema,
  type JsonValue,
  type Scenario,
  type Vec3
} from "../../domain/schema.js";
import {
  inverseQuaternion,
  rotateVector,
  subtract
} from "../geometry.js";
import {
  HUMANOID_BODY_NAMES,
  type HumanoidBodyName
} from "./model.js";
import {
  interpolateReference,
  targetReference,
  type HumanoidReference,
  type HumanoidReferenceTarget
} from "./reference.js";
import {
  HumanoidMotionArtifactSchema,
  hydrateHumanoidReference,
  humanoidMotionArtifactSha256,
  serializeHumanoidReference,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import {
  HumanoidMotionOptionContractSchema,
  advanceHumanoidMotionOptionMonitor,
  createHumanoidMotionOptionMonitorState,
  detectHumanoidMotionOption,
  humanoidMotionOptionContractSha256,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionDetection,
  type HumanoidMotionOptionDetectorInput,
  type HumanoidMotionOptionMonitorState,
  type HumanoidMotionOptionObservableObject
} from "./motion-option.js";
import {
  HumanoidMotionGeneratorDescriptorSchema,
  TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR,
  type HumanoidMotionGeneratorDescriptor
} from "./motion-generator-contract.js";
import {
  captureHumanoidMotionRolloutFrame,
  createHumanoidMotionRollout,
  humanoidMotionRolloutSha256,
  type HumanoidMotionDriftEvidence,
  type HumanoidMotionRollout,
  type HumanoidMotionRolloutFrame
} from "./motion-rollout.js";
import {
  type HumanoidEndEffectorTarget,
  type HumanoidSimulation,
  type HumanoidSimulationSnapshot
} from "./simulation.js";

const HumanoidRootVelocitySchema = z.object({
  forward_mps: z.number().finite().describe("身体局部前向速度，单位米每秒"),
  lateral_mps: z.number().finite().describe("身体局部左向速度，单位米每秒")
}).strict();

const HumanoidEndEffectorTargetSchema = z.object({
  position: Vec3Schema.describe("末端目标位置；world 为世界坐标，pelvis 为相对骨盆坐标"),
  frame: z.enum(["world", "pelvis"]),
  tolerance_m: z.number().finite().min(0.01).max(0.12)
    .describe("该关键帧在真实 MuJoCo 跟踪后的物理验收容差，单位米；应依据任务精度和已有跟踪误差选择，不是 IK 数值求解误差")
}).strict();

const HumanoidContactConstraintSchema = z.object({
  body: z.enum(HUMANOID_BODY_NAMES)
    .describe("允许接触指定物体的真实 G1 Link"),
  object_id: z.string().trim().min(1)
    .describe("必须与当前 MuJoCo 动态物体 ID 完全一致"),
  required: z.boolean()
    .describe("为 true 时，完整物理预演和真实执行都必须观测到该接触")
}).strict();

type HumanoidContactConstraint = z.infer<typeof HumanoidContactConstraintSchema>;

const HumanoidKeyframeSchema = z.object({
  at_seconds: z.number().finite().nonnegative(),
  root_velocity: HumanoidRootVelocitySchema.nullable().optional(),
  root_yaw_velocity: z.number().finite().describe("根节点偏航角速度，单位弧度每秒").nullable().optional(),
  root_height: z.number().finite().nullable().optional().refine(
    (value) => value == null || value > 0,
    "root_height must be null when unused or a positive pelvis height in meters"
  ),
  root_roll: z.number().finite().nullable().optional(),
  root_pitch: z.number().finite().nullable().optional(),
  torso_yaw: z.number().finite().min(-1.2).max(1.2).nullable().optional(),
  left_hand: HumanoidEndEffectorTargetSchema.nullable().optional(),
  right_hand: HumanoidEndEffectorTargetSchema.nullable().optional(),
  left_foot: HumanoidEndEffectorTargetSchema.nullable().optional(),
  right_foot: HumanoidEndEffectorTargetSchema.nullable().optional()
}).strict();

export const HumanoidMotionPlanSchema = z.object({
  id: z.string().trim().min(1),
  intent: z.string().trim().min(1),
  duration_seconds: z.number().finite().positive().max(30)
    .describe("本次连续运动分块的总时长，最多 30 秒"),
  contact_constraints: z.array(HumanoidContactConstraintSchema)
    .max(16)
    .describe("只授权列出的 Link 接触列出的物体；未列出的身体-环境接触仍会拒绝计划")
    .nullable()
    .optional(),
  keyframes: z.array(HumanoidKeyframeSchema).min(2).max(128)
}).strict().superRefine((plan, context) => {
  const contactKeys = plan.contact_constraints?.map(contactKey) ?? [];
  if (new Set(contactKeys).size !== contactKeys.length) {
    context.addIssue({
      code: "custom",
      path: ["contact_constraints"],
      message: "A humanoid plan cannot repeat the same body-object contact constraint"
    });
  }
  if (plan.keyframes[0]?.at_seconds !== 0) {
    context.addIssue({
      code: "custom",
      path: ["keyframes", 0, "at_seconds"],
      message: "The first humanoid keyframe must start at zero"
    });
  }
  for (let index = 1; index < plan.keyframes.length; index += 1) {
    if (plan.keyframes[index]!.at_seconds <= plan.keyframes[index - 1]!.at_seconds) {
      context.addIssue({
        code: "custom",
        path: ["keyframes", index, "at_seconds"],
        message: "Humanoid keyframe times must increase"
      });
    }
  }
  const finalTime = plan.keyframes.at(-1)?.at_seconds;
  if (finalTime !== plan.duration_seconds) {
    context.addIssue({
      code: "custom",
      path: ["duration_seconds"],
      message: "The final humanoid keyframe must equal the plan duration"
    });
  }
});

export type HumanoidMotionPlan = z.infer<typeof HumanoidMotionPlanSchema>;

export function duplicateHumanoidMotionCandidateIndexes(
  candidates: readonly HumanoidMotionPlan[]
): Array<{ candidateIndex: number; originalIndex: number }> {
  const firstIndexByContent = new Map<string, number>();
  const duplicates: Array<{ candidateIndex: number; originalIndex: number }> = [];
  candidates.forEach((candidate, candidateIndex) => {
    const content = humanoidMotionCandidateContent(candidate);
    const originalIndex = firstIndexByContent.get(content);
    if (originalIndex === undefined) {
      firstIndexByContent.set(content, candidateIndex);
      return;
    }
    duplicates.push({ candidateIndex, originalIndex });
  });
  return duplicates;
}

export const HumanoidMotionCandidateBatchSchema = z.object({
  objective: z.string().trim().min(1)
    .describe("所有候选共同服务的当前自主目标"),
  termination: HumanoidMotionOptionContractSchema
    .describe("所有候选必须共同达成的可观测物理结果；时长只是最多八秒的执行上界"),
  candidates: z.array(HumanoidMotionPlanSchema).min(2).max(3)
    .describe("按模型偏好排序的不同连续全身动作候选；每个候选都会从同一物理状态完整预演")
}).strict().superRefine((batch, context) => {
  const ids = batch.candidates.map((candidate) => candidate.id);
  if (new Set(ids).size !== ids.length) {
    context.addIssue({
      code: "custom",
      path: ["candidates"],
      message: "A humanoid motion candidate batch cannot repeat a plan identifier"
    });
  }
  for (const duplicate of duplicateHumanoidMotionCandidateIndexes(
    batch.candidates
  )) {
    context.addIssue({
      code: "custom",
      path: ["candidates", duplicate.candidateIndex],
      message: `Candidate motion content duplicates candidate ${duplicate.originalIndex + 1}; id and intent labels do not make a distinct candidate`
    });
  }
  const contactPredicates = batch.termination.predicates.filter((predicate) => (
    predicate.type === "body_contact_object"
  ));
  batch.candidates.forEach((candidate, candidateIndex) => {
    if (candidate.duration_seconds > 8) {
      context.addIssue({
        code: "custom",
        path: ["candidates", candidateIndex, "duration_seconds"],
        message: "Autonomous humanoid options must return control within eight seconds"
      });
    }
    for (const predicate of contactPredicates) {
      const authorized = candidate.contact_constraints?.some((constraint) => (
        constraint.required
        && constraint.body === predicate.body
        && constraint.object_id === predicate.object_id
      ));
      if (!authorized) {
        context.addIssue({
          code: "custom",
          path: ["candidates", candidateIndex, "contact_constraints"],
          message: "A contact termination predicate requires the same required body-object contact constraint"
        });
      }
    }
  });
});

function humanoidMotionCandidateContent(candidate: HumanoidMotionPlan): string {
  const contactConstraints = (candidate.contact_constraints ?? [])
    .map((constraint) => ({
      body: constraint.body,
      object_id: constraint.object_id,
      required: constraint.required
    }))
    .sort((left, right) => {
      const leftKey = contactKey(left);
      const rightKey = contactKey(right);
      return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
    });
  return JSON.stringify({
    duration_seconds: candidate.duration_seconds,
    contact_constraints: contactConstraints,
    keyframes: candidate.keyframes.map((keyframe) => ({
      at_seconds: keyframe.at_seconds,
      root_velocity: keyframe.root_velocity ?? null,
      root_yaw_velocity: keyframe.root_yaw_velocity ?? null,
      root_height: keyframe.root_height ?? null,
      root_roll: keyframe.root_roll ?? null,
      root_pitch: keyframe.root_pitch ?? null,
      torso_yaw: keyframe.torso_yaw ?? null,
      left_hand: keyframe.left_hand ?? null,
      right_hand: keyframe.right_hand ?? null,
      left_foot: keyframe.left_foot ?? null,
      right_foot: keyframe.right_foot ?? null
    }))
  });
}

export type HumanoidMotionCandidateBatch = z.infer<
  typeof HumanoidMotionCandidateBatchSchema
>;

export const HumanoidMotionOptionCertificateSchema = z.object({
  artifact_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  contract_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  rollout_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  rollout_frame_count: z.number().int().positive(),
  drift_consecutive_steps: z.number().int().positive(),
  validated_frame_limit: z.number().int().positive(),
  predicted_termination_frame: z.number().int().positive(),
  predicted_at_seconds: z.number().finite().positive(),
  stable_steps: z.number().int().positive(),
  evidence: JsonValueSchema
}).strict();

export type HumanoidMotionOptionCertificate = z.infer<
  typeof HumanoidMotionOptionCertificateSchema
>;

export type HumanoidBodyChannel =
  | "locomotion"
  | "left_leg"
  | "right_leg"
  | "torso"
  | "left_arm"
  | "right_arm";
const HUMANOID_CHANNEL_ORDER: readonly HumanoidBodyChannel[] = [
  "locomotion",
  "left_leg",
  "right_leg",
  "torso",
  "left_arm",
  "right_arm"
];

export interface HumanoidMotionValidationOptions {
  requireFinalSupport?: boolean;
  contactObjectIds?: ReadonlySet<string>;
  motionOption?: {
    contract: HumanoidMotionOptionContract;
    scenario: Scenario;
  };
}

export interface HumanoidObjectContact {
  body: HumanoidBodyName;
  objectId: string | null;
  normalForce: number;
}

export interface HumanoidMotionValidation {
  feasible: boolean;
  failures: Array<{
    code: "fallen" | "environment_contact" | "required_contact_missing"
      | "unknown_contact_object" | "contact_object_not_currently_visible"
      | "unsupported_finish" | "invalid_reference"
      | "task_space_target_unmet"
      | "motion_goal_already_satisfied" | "motion_goal_unmet"
      | "motion_goal_uncertain" | "motion_constraint_violated"
      | "execution_drift";
    atSeconds: number;
    bodies?: HumanoidBodyName[];
    contacts?: HumanoidObjectContact[];
    constraints?: HumanoidContactConstraint[];
    drift?: HumanoidMotionDriftEvidence;
    taskSpaceTarget?: {
      body: HumanoidEndEffectorTarget["body"];
      frame: HumanoidEndEffectorTarget["frame"];
      target: Vec3;
      achieved: Vec3;
      errorMeters: number;
      toleranceMeters: number;
      requestedAtSeconds: number;
      observedAtSeconds: number;
    };
    message?: string;
  }>;
  evidence: {
    simulatedSteps: number;
    minimumRootHeight: number;
    minimumUpright: number;
    minimumSupportMargin: number | null;
    travelledDistance: number;
    environmentContactBodies: HumanoidBodyName[];
    environmentContacts: HumanoidObjectContact[];
    satisfiedRequiredContacts: HumanoidContactConstraint[];
  };
  finalSnapshot: HumanoidSimulationSnapshot;
}

export interface PreparedHumanoidMotion {
  artifact: HumanoidMotionArtifact | null;
  rollout: HumanoidMotionRollout | null;
  optionCertificate: HumanoidMotionOptionCertificate | null;
  validation: HumanoidMotionValidation;
}

export interface HumanoidMotionGeneratorInput {
  simulation: HumanoidSimulation;
  plan: HumanoidMotionPlan;
  baseline: HumanoidReference;
  controlStepSeconds: number;
}

export interface HumanoidMotionGenerator {
  readonly descriptor: HumanoidMotionGeneratorDescriptor;
  generate(input: HumanoidMotionGeneratorInput): Promise<HumanoidMotionArtifact>;
  dispose(): Promise<void>;
}

export class TaskSpaceHumanoidMotionGenerator implements HumanoidMotionGenerator {
  readonly descriptor: HumanoidMotionGeneratorDescriptor =
    TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR;

  async generate(input: HumanoidMotionGeneratorInput): Promise<HumanoidMotionArtifact> {
    return generateTaskSpaceMotion(input, this.descriptor.implementation);
  }

  async dispose(): Promise<void> {}
}

export function occupiedHumanoidChannels(
  rawPlan: HumanoidMotionPlan
): HumanoidBodyChannel[] {
  const plan = HumanoidMotionPlanSchema.parse(rawPlan);
  const channels = new Set<HumanoidBodyChannel>();
  for (const keyframe of plan.keyframes) {
    if (keyframe.root_velocity
      || keyframe.root_yaw_velocity != null
      || keyframe.root_height != null
      || keyframe.root_roll != null
      || keyframe.root_pitch != null) {
      channels.add("locomotion");
    }
    if (keyframe.torso_yaw != null) channels.add("torso");
    if (keyframe.left_foot) channels.add("left_leg");
    if (keyframe.right_foot) channels.add("right_leg");
    if (keyframe.left_hand) channels.add("left_arm");
    if (keyframe.right_hand) channels.add("right_arm");
  }
  return HUMANOID_CHANNEL_ORDER.filter((channel) => channels.has(channel));
}

async function generateTaskSpaceMotion(
  input: HumanoidMotionGeneratorInput,
  implementation: string
): Promise<HumanoidMotionArtifact> {
  const plan = HumanoidMotionPlanSchema.parse(input.plan);
  const controlStep = input.controlStepSeconds;
  if (!Number.isFinite(controlStep) || controlStep <= 0) {
    throw new Error("Humanoid motion control step must be positive");
  }
  const targets: HumanoidReference[] = [];
  let previous = input.baseline;
  for (const keyframe of plan.keyframes) {
    previous = taskSpaceReference(input.simulation, previous, keyframe);
    targets.push(previous);
  }

  const frames: HumanoidMotionArtifact["frames"] = [];
  const steps = Math.ceil(plan.duration_seconds / controlStep);
  let segment = 1;
  for (let index = 1; index <= steps; index += 1) {
    const atSeconds = Math.min(index * controlStep, plan.duration_seconds);
    while (segment < plan.keyframes.length - 1
      && atSeconds > plan.keyframes[segment]!.at_seconds) {
      segment += 1;
    }
    const startKeyframe = plan.keyframes[segment - 1]!;
    const endKeyframe = plan.keyframes[segment]!;
    const duration = endKeyframe.at_seconds - startKeyframe.at_seconds;
    frames.push({
      atSeconds,
      reference: serializeHumanoidReference(interpolateReference(
        targets[segment - 1]!,
        targets[segment]!,
        (atSeconds - startKeyframe.at_seconds) / duration,
        duration
      ))
    });
  }
  return HumanoidMotionArtifactSchema.parse({
    version: 1,
    protocol: "humanoid-motion-v1",
    generator: implementation,
    controlStepSeconds: controlStep,
    durationSeconds: plan.duration_seconds,
    frames
  });
}

export async function prepareHumanoidMotion(
  simulation: HumanoidSimulation,
  plan: HumanoidMotionPlan,
  baseline: HumanoidReference,
  options: HumanoidMotionValidationOptions = {},
  generator: HumanoidMotionGenerator = new TaskSpaceHumanoidMotionGenerator()
): Promise<PreparedHumanoidMotion> {
  const generationState = simulation.captureState();
  let artifact: HumanoidMotionArtifact | undefined;
  let generationError: unknown;
  try {
    const descriptor = HumanoidMotionGeneratorDescriptorSchema.parse(generator.descriptor);
    const controlStepSeconds = simulation.controllerDescriptor().controlStepSeconds;
    artifact = assertMotionArtifactContract(
      await generator.generate({
        simulation,
        plan,
        baseline,
        controlStepSeconds
      }),
      descriptor,
      plan,
      controlStepSeconds
    );
  } catch (error) {
    generationError = error;
  } finally {
    simulation.restoreState(generationState);
  }
  if (generationError !== undefined || artifact === undefined) {
    const snapshot = simulation.snapshot();
    return {
      artifact: null,
      rollout: null,
      optionCertificate: null,
      validation: validationResult(
        [{
          code: "invalid_reference",
          atSeconds: 0,
          message: generationError instanceof Error
            ? generationError.message
            : String(generationError ?? "Motion generator returned no artifact")
        }],
        snapshot,
        snapshot,
        0,
        snapshot.rootPosition.y,
        snapshot.balance.upright,
        snapshot.balance.supportMargin,
        new Set(),
        new Map(),
        [],
        new Set()
      )
    };
  }
  const validated = await validateHumanoidMotionArtifact(
    simulation,
    plan,
    artifact,
    options
  );
  return { artifact, ...validated };
}

function assertMotionArtifactContract(
  rawArtifact: HumanoidMotionArtifact,
  descriptor: HumanoidMotionGeneratorDescriptor,
  rawPlan: HumanoidMotionPlan,
  controlStepSeconds: number
): HumanoidMotionArtifact {
  const artifact = HumanoidMotionArtifactSchema.parse(rawArtifact);
  const plan = HumanoidMotionPlanSchema.parse(rawPlan);
  const expectedFrames = Math.ceil(plan.duration_seconds / controlStepSeconds);
  if (artifact.generator !== descriptor.implementation) {
    throw new Error("Humanoid motion artifact generator identity mismatch");
  }
  if (Math.abs(artifact.controlStepSeconds - controlStepSeconds) > 1e-9) {
    throw new Error("Humanoid motion artifact control step mismatch");
  }
  if (Math.abs(artifact.durationSeconds - plan.duration_seconds) > 1e-9
    || artifact.frames.length !== expectedFrames) {
    throw new Error("Humanoid motion artifact duration or frame count mismatch");
  }
  for (let index = 0; index < artifact.frames.length; index += 1) {
    const expectedTime = Math.min((index + 1) * controlStepSeconds, plan.duration_seconds);
    if (Math.abs(artifact.frames[index]!.atSeconds - expectedTime) > 1e-9) {
      throw new Error("Humanoid motion artifact frame cadence mismatch");
    }
  }
  return artifact;
}

async function validateHumanoidMotionArtifact(
  simulation: HumanoidSimulation,
  plan: HumanoidMotionPlan,
  artifact: HumanoidMotionArtifact,
  options: HumanoidMotionValidationOptions
): Promise<{
  validation: HumanoidMotionValidation;
  rollout: HumanoidMotionRollout | null;
  optionCertificate: HumanoidMotionOptionCertificate | null;
}> {
  const saved = simulation.captureState();
  const start = simulation.snapshot();
  const constraints = plan.contact_constraints ?? [];
  const allowed = new Set(constraints.map(contactKey));
  const required = constraints.filter((constraint) => constraint.required);
  const knownObjects = new Set(Object.keys(start.objects));
  const physicalTargets = scheduledTaskSpaceTargets(plan);
  let nextPhysicalTargetIndex = 0;
  const failures: HumanoidMotionValidation["failures"] = [];
  const contacted = new Set<HumanoidBodyName>();
  const environmentContacts = new Map<string, HumanoidObjectContact>();
  const satisfiedRequired = new Set<string>();
  let minimumRootHeight = start.rootPosition.y;
  let minimumUpright = start.balance.upright;
  let minimumSupportMargin = start.balance.supportMargin;
  let finalSnapshot = start;
  let simulatedSteps = 0;
  let optionDetection: HumanoidMotionOptionDetection | null = null;
  let optionMonitor: HumanoidMotionOptionMonitorState | null = options.motionOption
    ? createHumanoidMotionOptionMonitorState(options.motionOption.contract)
    : null;
  let optionObservationStatus: "satisfied" | "unsatisfied" | "uncertain" | null = null;
  let predictedTerminationFrame: number | null = null;
  let predictedAtSeconds: number | null = null;
  let predictedEvidence: JsonValue | null = null;
  const rolloutFrames: HumanoidMotionRolloutFrame[] = [];
  try {
    const unknownObjects = constraints.filter((constraint) => (
      !knownObjects.has(constraint.object_id)
    ));
    if (unknownObjects.length > 0) {
      failures.push({
        code: "unknown_contact_object",
        atSeconds: 0,
        constraints: unknownObjects
      });
      return {
        validation: validationResult(
          failures,
          start,
          start,
          0,
          minimumRootHeight,
          minimumUpright,
          minimumSupportMargin,
          contacted,
          environmentContacts,
          required,
          satisfiedRequired
        ),
        rollout: null,
        optionCertificate: null
      };
    }
    const unseenObjects = options.contactObjectIds
      ? constraints.filter((constraint) => (
          !options.contactObjectIds?.has(constraint.object_id)
        ))
      : [];
    if (unseenObjects.length > 0) {
      failures.push({
        code: "contact_object_not_currently_visible",
        atSeconds: 0,
        constraints: unseenObjects
      });
      return {
        validation: validationResult(
          failures,
          start,
          start,
          0,
          minimumRootHeight,
          minimumUpright,
          minimumSupportMargin,
          contacted,
          environmentContacts,
          required,
          satisfiedRequired
        ),
        rollout: null,
        optionCertificate: null
      };
    }
    while (physicalTargets[nextPhysicalTargetIndex]?.atSeconds === 0) {
      const failure = physicalTaskSpaceTargetFailure(
        physicalTargets[nextPhysicalTargetIndex]!,
        start,
        0
      );
      if (failure) failures.push(failure);
      nextPhysicalTargetIndex += 1;
    }
    if (failures.length > 0) {
      return {
        validation: validationResult(
          failures,
          start,
          start,
          0,
          minimumRootHeight,
          minimumUpright,
          minimumSupportMargin,
          contacted,
          environmentContacts,
          required,
          satisfiedRequired
        ),
        rollout: null,
        optionCertificate: null
      };
    }
    if (options.motionOption) {
      optionDetection = detectOptionFromSimulation(
        simulation,
        start,
        options.motionOption,
        options.contactObjectIds
      );
      if (optionDetection.hasUncertain) {
        failures.push({
          code: "motion_goal_uncertain",
          atSeconds: 0,
          message: "Motion option references state that is not currently observable"
        });
      } else if (optionDetection.allSatisfied && required.length === 0) {
        failures.push({
          code: "motion_goal_already_satisfied",
          atSeconds: 0,
          message: "Motion option predicates are already satisfied before execution"
        });
      }
      if (failures.length > 0) {
        return {
          validation: validationResult(
            failures,
            start,
            start,
            0,
            minimumRootHeight,
            minimumUpright,
            minimumSupportMargin,
            contacted,
            environmentContacts,
            required,
            satisfiedRequired
          ),
          rollout: null,
          optionCertificate: null
        };
      }
    }
    for (const frame of artifact.frames) {
      if (options.motionOption && optionMonitor?.phase === "awaiting_precondition") {
        const update = advanceHumanoidMotionOptionMonitor(
          options.motionOption.contract,
          optionMonitor,
          motionOptionDetectorInputFromSimulation(
            simulation,
            finalSnapshot,
            options.motionOption,
            options.contactObjectIds
          )
        );
        optionMonitor = update.state;
        optionDetection = update.detection;
        optionObservationStatus = update.observationStatus;
        if (update.observationStatus !== "satisfied") {
          const atSeconds = simulatedSteps === 0
            ? 0
            : artifact.frames[simulatedSteps - 1]!.atSeconds;
          failures.push({
            code: update.observationStatus === "uncertain"
              ? "motion_goal_uncertain"
              : "motion_constraint_violated",
            atSeconds,
            message: update.observationStatus === "uncertain"
              ? "Motion option precondition is not observable before execution"
              : "Motion option precondition is not satisfied before execution"
          });
          break;
        }
      }
      try {
        finalSnapshot = await simulation.step(hydrateHumanoidReference(frame.reference));
      } catch (error) {
        failures.push({
          code: "invalid_reference",
          atSeconds: frame.atSeconds,
          message: error instanceof Error ? error.message : String(error)
        });
        break;
      }
      simulatedSteps += 1;
      if (options.motionOption) {
        rolloutFrames.push(captureHumanoidMotionRolloutFrame(
          frame.atSeconds,
          finalSnapshot
        ));
      }
      minimumRootHeight = Math.min(minimumRootHeight, finalSnapshot.rootPosition.y);
      minimumUpright = Math.min(minimumUpright, finalSnapshot.balance.upright);
      if (finalSnapshot.balance.supportMargin !== null) {
        minimumSupportMargin = minimumSupportMargin === null
          ? finalSnapshot.balance.supportMargin
          : Math.min(minimumSupportMargin, finalSnapshot.balance.supportMargin);
      }
      const observedContacts = humanoidObjectContacts(finalSnapshot);
      for (const contact of observedContacts) {
        contacted.add(contact.body);
        const key = contactPairKey(contact.body, contact.objectId);
        const previous = environmentContacts.get(key);
        if (!previous || contact.normalForce > previous.normalForce) {
          environmentContacts.set(key, contact);
        }
        if (contact.objectId !== null && allowed.has(contactPairKey(
          contact.body,
          contact.objectId
        ))) {
          satisfiedRequired.add(contactPairKey(contact.body, contact.objectId));
        }
      }
      const blockedContacts = observedContacts.filter((contact) => (
        contact.objectId === null
        || !allowed.has(contactPairKey(contact.body, contact.objectId))
      ));
      if (blockedContacts.length > 0) {
        failures.push({
          code: "environment_contact",
          atSeconds: frame.atSeconds,
          bodies: [...new Set(blockedContacts.map((contact) => contact.body))],
          contacts: blockedContacts
        });
        break;
      }
      if (finalSnapshot.fallen) {
        failures.push({ code: "fallen", atSeconds: frame.atSeconds });
        break;
      }
      while ((physicalTargets[nextPhysicalTargetIndex]?.atSeconds ?? Infinity)
        <= frame.atSeconds + 1e-9) {
        const failure = physicalTaskSpaceTargetFailure(
          physicalTargets[nextPhysicalTargetIndex]!,
          finalSnapshot,
          frame.atSeconds
        );
        if (failure) failures.push(failure);
        nextPhysicalTargetIndex += 1;
      }
      if (failures.length > 0) break;
      if (options.motionOption && optionMonitor?.phase !== "awaiting_precondition") {
        if (!optionMonitor) {
          throw new Error("Humanoid motion option monitor is missing");
        }
        const update = advanceHumanoidMotionOptionMonitor(
          options.motionOption.contract,
          optionMonitor,
          motionOptionDetectorInputFromSimulation(
            simulation,
            finalSnapshot,
            options.motionOption,
            options.contactObjectIds
          )
        );
        optionMonitor = update.state;
        optionDetection = update.detection;
        optionObservationStatus = update.observationStatus;
        if (optionMonitor.phase === "violated") {
          failures.push({
            code: "motion_constraint_violated",
            atSeconds: frame.atSeconds,
            message: "Motion option violated its during constraint"
          });
          break;
        }
        if (optionMonitor.phase === "indeterminate") {
          failures.push({
            code: "motion_goal_uncertain",
            atSeconds: frame.atSeconds,
            message: "Motion option lost observable evidence for its during constraint"
          });
          break;
        }
        if (optionMonitor.phase === "succeeded"
          && missingRequiredHumanoidContacts(
            required,
            satisfiedRequired
          ).length === 0
          && finalSnapshot.balance.support !== "none"
          && predictedTerminationFrame === null) {
          predictedTerminationFrame = simulatedSteps;
          predictedAtSeconds = frame.atSeconds;
          predictedEvidence = asJson({
            predicates: optionDetection.evidence,
            phases: optionDetection.phases,
            monitor: optionMonitor
          });
        }
      }
    }
    const missingRequired = required.filter((constraint) => (
      !satisfiedRequired.has(contactKey(constraint))
    ));
    if (failures.length === 0 && missingRequired.length > 0) {
      failures.push({
        code: "required_contact_missing",
        atSeconds: plan.duration_seconds,
        constraints: missingRequired
      });
    }
    if (failures.length === 0
      && (options.requireFinalSupport ?? true)
      && finalSnapshot.balance.support === "none") {
      failures.push({
        code: "unsupported_finish",
        atSeconds: plan.duration_seconds
      });
    }
    if (failures.length === 0
      && options.motionOption
      && predictedTerminationFrame === null) {
      failures.push({
        code: optionObservationStatus === "uncertain" || optionDetection?.hasUncertain
          ? "motion_goal_uncertain"
          : "motion_goal_unmet",
        atSeconds: plan.duration_seconds,
        message: optionObservationStatus === "uncertain" || optionDetection?.hasUncertain
          ? "Motion option ended without observable success evidence"
          : "Motion option exhausted its verified horizon before physical success"
      });
    }
    const rollout = failures.length === 0
      && options.motionOption
      && rolloutFrames.length === artifact.frames.length
      ? createHumanoidMotionRollout(rolloutFrames)
      : null;
    const optionCertificate = failures.length === 0
      && options.motionOption
      && rollout !== null
      && predictedTerminationFrame !== null
      && predictedAtSeconds !== null
      && predictedEvidence !== null
      ? HumanoidMotionOptionCertificateSchema.parse({
          artifact_sha256: humanoidMotionArtifactSha256(artifact),
          contract_sha256: humanoidMotionOptionContractSha256(
            options.motionOption.contract
          ),
          rollout_sha256: humanoidMotionRolloutSha256(rollout),
          rollout_frame_count: rollout.frames.length,
          drift_consecutive_steps: rollout.limits.consecutive_steps,
          validated_frame_limit: artifact.frames.length,
          predicted_termination_frame: predictedTerminationFrame,
          predicted_at_seconds: predictedAtSeconds,
          stable_steps: options.motionOption.contract.stable_steps,
          evidence: predictedEvidence
        })
      : null;
    return {
      rollout,
      validation: validationResult(
        failures,
        start,
        finalSnapshot,
        simulatedSteps,
        minimumRootHeight,
        minimumUpright,
        minimumSupportMargin,
        contacted,
        environmentContacts,
        required,
        satisfiedRequired
      ),
      optionCertificate
    };
  } finally {
    simulation.restoreState(saved);
  }
}

function detectOptionFromSimulation(
  simulation: HumanoidSimulation,
  snapshot: HumanoidSimulationSnapshot,
  option: NonNullable<HumanoidMotionValidationOptions["motionOption"]>,
  boundObjectIds?: ReadonlySet<string>
): HumanoidMotionOptionDetection {
  return detectHumanoidMotionOption(
    option.contract,
    motionOptionDetectorInputFromSimulation(
      simulation,
      snapshot,
      option,
      boundObjectIds
    )
  );
}

function motionOptionDetectorInputFromSimulation(
  simulation: HumanoidSimulation,
  snapshot: HumanoidSimulationSnapshot,
  option: NonNullable<HumanoidMotionValidationOptions["motionOption"]>,
  boundObjectIds?: ReadonlySet<string>
): HumanoidMotionOptionDetectorInput {
  const visible = simulation.senseObjects(option.scenario.visibility_radius).objects;
  const observableObjects: HumanoidMotionOptionObservableObject[] = [];
  for (const descriptor of option.scenario.objects) {
    const object = visible[descriptor.id];
    if (!object || boundObjectIds && !boundObjectIds.has(descriptor.id)) continue;
    observableObjects.push({
      id: descriptor.id,
      position: { ...object.position },
      size: { ...descriptor.size }
    });
  }
  return {
    snapshot,
    observableObjects,
    zones: option.scenario.zones
  };
}

function asJson(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}

function taskSpaceReference(
  simulation: HumanoidSimulation,
  baseline: HumanoidReference,
  keyframe: z.infer<typeof HumanoidKeyframeSchema>
): HumanoidReference {
  const rootTarget: HumanoidReferenceTarget = {
    ...(keyframe.root_velocity
      ? {
          rootVelocity: [
            keyframe.root_velocity.forward_mps,
            keyframe.root_velocity.lateral_mps
          ]
        }
      : {}),
    ...(keyframe.root_yaw_velocity != null
      ? { rootYawVelocity: keyframe.root_yaw_velocity }
      : {}),
    ...(keyframe.root_height != null ? { rootHeight: keyframe.root_height } : {}),
    ...(keyframe.root_roll != null ? { rootRoll: keyframe.root_roll } : {}),
    ...(keyframe.root_pitch != null ? { rootPitch: keyframe.root_pitch } : {}),
    ...(keyframe.torso_yaw != null
      ? { joints: { waist_yaw_joint: keyframe.torso_yaw } }
      : {})
  };
  const rooted = targetReference(baseline, rootTarget);
  return simulation.solveEndEffectorTargets(rooted, taskSpaceTargets(keyframe)).reference;
}

function taskSpaceTargets(
  keyframe: z.infer<typeof HumanoidKeyframeSchema>
): HumanoidEndEffectorTarget[] {
  return [
    ...(keyframe.left_hand ? [{
      body: "left_wrist_yaw_link" as const,
      position: keyframe.left_hand.position,
      frame: keyframe.left_hand.frame,
      tolerance: keyframe.left_hand.tolerance_m
    }] : []),
    ...(keyframe.right_hand ? [{
      body: "right_wrist_yaw_link" as const,
      position: keyframe.right_hand.position,
      frame: keyframe.right_hand.frame,
      tolerance: keyframe.right_hand.tolerance_m
    }] : []),
    ...(keyframe.left_foot ? [{
      body: "left_ankle_roll_link" as const,
      position: keyframe.left_foot.position,
      frame: keyframe.left_foot.frame,
      tolerance: keyframe.left_foot.tolerance_m
    }] : []),
    ...(keyframe.right_foot ? [{
      body: "right_ankle_roll_link" as const,
      position: keyframe.right_foot.position,
      frame: keyframe.right_foot.frame,
      tolerance: keyframe.right_foot.tolerance_m
    }] : [])
  ];
}

interface ScheduledTaskSpaceTarget {
  atSeconds: number;
  target: HumanoidEndEffectorTarget;
}

function scheduledTaskSpaceTargets(
  plan: HumanoidMotionPlan
): ScheduledTaskSpaceTarget[] {
  return plan.keyframes.flatMap((keyframe) => (
    taskSpaceTargets(keyframe).map((target) => ({
      atSeconds: keyframe.at_seconds,
      target
    }))
  ));
}

function physicalTaskSpaceTargetFailure(
  scheduled: ScheduledTaskSpaceTarget,
  snapshot: HumanoidSimulationSnapshot,
  observedAtSeconds: number
): HumanoidMotionValidation["failures"][number] | null {
  const target = scheduled.target;
  const achievedWorld = snapshot.links[target.body].position;
  const achieved = target.frame === "world"
    ? { ...achievedWorld }
    : rotateVector(
        inverseQuaternion(snapshot.links.pelvis.rotation),
        subtract(achievedWorld, snapshot.links.pelvis.position)
      );
  const errorMeters = Math.hypot(
    achieved.x - target.position.x,
    achieved.y - target.position.y,
    achieved.z - target.position.z
  );
  if (errorMeters <= target.tolerance + 1e-9) return null;
  const evidence = {
    body: target.body,
    frame: target.frame,
    target: { ...target.position },
    achieved,
    errorMeters,
    toleranceMeters: target.tolerance,
    requestedAtSeconds: scheduled.atSeconds,
    observedAtSeconds
  };
  return {
    code: "task_space_target_unmet",
    atSeconds: observedAtSeconds,
    taskSpaceTarget: evidence,
    message: `Physical task-space target missed: ${target.body} `
      + `error=${errorMeters.toFixed(3)}m tolerance=${target.tolerance.toFixed(3)}m`
  };
}

function validationResult(
  failures: HumanoidMotionValidation["failures"],
  start: HumanoidSimulationSnapshot,
  finalSnapshot: HumanoidSimulationSnapshot,
  simulatedSteps: number,
  minimumRootHeight: number,
  minimumUpright: number,
  minimumSupportMargin: number | null,
  contacted: ReadonlySet<HumanoidBodyName>,
  environmentContacts: ReadonlyMap<string, HumanoidObjectContact>,
  required: readonly HumanoidContactConstraint[],
  satisfiedRequired: ReadonlySet<string>
): HumanoidMotionValidation {
  return {
    feasible: failures.length === 0,
    failures,
    evidence: {
      simulatedSteps,
      minimumRootHeight,
      minimumUpright,
      minimumSupportMargin,
      travelledDistance: Math.hypot(
        finalSnapshot.rootPosition.x - start.rootPosition.x,
        finalSnapshot.rootPosition.z - start.rootPosition.z
      ),
      environmentContactBodies: [...contacted],
      environmentContacts: [...environmentContacts.values()],
      satisfiedRequiredContacts: required.filter((constraint) => (
        satisfiedRequired.has(contactKey(constraint))
      ))
    },
    finalSnapshot
  };
}

export function humanoidObjectContacts(
  snapshot: HumanoidSimulationSnapshot
): HumanoidObjectContact[] {
  const contacts = new Map<string, HumanoidObjectContact>();
  for (const contact of snapshot.contacts) {
    if ((contact.firstBody === null) === (contact.secondBody === null)) continue;
    const body = contact.firstBody ?? contact.secondBody;
    if (!body) continue;
    const objectId = contact.firstObject ?? contact.secondObject;
    const foot = body === "left_ankle_roll_link" || body === "right_ankle_roll_link";
    if (foot && objectId === null && Math.abs(contact.normal.y) >= 0.55) continue;
    const key = contactPairKey(body, objectId);
    const candidate = { body, objectId, normalForce: contact.normalForce };
    const previous = contacts.get(key);
    if (!previous || candidate.normalForce > previous.normalForce) contacts.set(key, candidate);
  }
  return [...contacts.values()];
}

export function blockedHumanoidContacts(
  snapshot: HumanoidSimulationSnapshot,
  constraints: readonly HumanoidContactConstraint[]
): HumanoidObjectContact[] {
  const allowed = new Set(constraints.map(contactKey));
  return humanoidObjectContacts(snapshot).filter((contact) => (
    contact.objectId === null
    || !allowed.has(contactPairKey(contact.body, contact.objectId))
  ));
}

export function missingRequiredHumanoidContacts(
  constraints: readonly HumanoidContactConstraint[],
  satisfied: ReadonlySet<string>
): HumanoidContactConstraint[] {
  return constraints.filter((constraint) => (
    constraint.required && !satisfied.has(contactKey(constraint))
  ));
}

export function humanoidContactKey(
  body: HumanoidBodyName,
  objectId: string
): string {
  return contactPairKey(body, objectId);
}

function contactKey(constraint: HumanoidContactConstraint): string {
  return contactPairKey(constraint.body, constraint.object_id);
}

function contactPairKey(body: HumanoidBodyName, objectId: string | null): string {
  return `${body}\u0000${objectId ?? ""}`;
}
