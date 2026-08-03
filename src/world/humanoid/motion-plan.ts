import { z } from "zod";
import { Vec3Schema } from "../../domain/schema.js";
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
  serializeHumanoidReference,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import {
  HumanoidMotionGeneratorDescriptorSchema,
  TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR,
  type HumanoidMotionGeneratorDescriptor
} from "./motion-generator-contract.js";
import {
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
  tolerance_m: z.number().finite().min(0.01).max(0.12).default(0.045)
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
  root_velocity: HumanoidRootVelocitySchema.optional(),
  root_yaw_velocity: z.number().finite().describe("根节点偏航角速度，单位弧度每秒").optional(),
  root_height: z.number().finite().positive().optional(),
  root_roll: z.number().finite().optional(),
  root_pitch: z.number().finite().optional(),
  torso_yaw: z.number().finite().min(-1.2).max(1.2).optional(),
  left_hand: HumanoidEndEffectorTargetSchema.optional(),
  right_hand: HumanoidEndEffectorTargetSchema.optional()
}).strict();

export const HumanoidMotionPlanSchema = z.object({
  id: z.string().trim().min(1),
  intent: z.string().trim().min(1),
  duration_seconds: z.number().finite().positive().max(30)
    .describe("本次连续运动分块的总时长，最多 30 秒"),
  contact_constraints: z.array(HumanoidContactConstraintSchema)
    .max(16)
    .describe("只授权列出的 Link 接触列出的物体；未列出的身体-环境接触仍会拒绝计划")
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

export type HumanoidBodyChannel = "locomotion" | "torso" | "left_arm" | "right_arm";
const HUMANOID_CHANNEL_ORDER: readonly HumanoidBodyChannel[] = [
  "locomotion",
  "torso",
  "left_arm",
  "right_arm"
];

export interface HumanoidMotionValidationOptions {
  requireFinalSupport?: boolean;
  contactObjectIds?: ReadonlySet<string>;
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
      | "unsupported_finish" | "invalid_reference";
    atSeconds: number;
    bodies?: HumanoidBodyName[];
    contacts?: HumanoidObjectContact[];
    constraints?: HumanoidContactConstraint[];
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
      || keyframe.root_yaw_velocity !== undefined
      || keyframe.root_height !== undefined
      || keyframe.root_roll !== undefined
      || keyframe.root_pitch !== undefined) {
      channels.add("locomotion");
    }
    if (keyframe.torso_yaw !== undefined) channels.add("torso");
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
  let artifact: HumanoidMotionArtifact;
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
    const snapshot = simulation.snapshot();
    return {
      artifact: null,
      validation: validationResult(
        [{
          code: "invalid_reference",
          atSeconds: 0,
          message: error instanceof Error ? error.message : String(error)
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
  return {
    artifact,
    validation: await validateHumanoidMotionArtifact(
      simulation,
      plan,
      artifact,
      options
    )
  };
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
): Promise<HumanoidMotionValidation> {
  const saved = simulation.captureState();
  const start = simulation.snapshot();
  const constraints = plan.contact_constraints ?? [];
  const allowed = new Set(constraints.map(contactKey));
  const required = constraints.filter((constraint) => constraint.required);
  const knownObjects = new Set(Object.keys(start.objects));
  const failures: HumanoidMotionValidation["failures"] = [];
  const contacted = new Set<HumanoidBodyName>();
  const environmentContacts = new Map<string, HumanoidObjectContact>();
  const satisfiedRequired = new Set<string>();
  let minimumRootHeight = start.rootPosition.y;
  let minimumUpright = start.balance.upright;
  let minimumSupportMargin = start.balance.supportMargin;
  let finalSnapshot = start;
  let simulatedSteps = 0;
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
      return validationResult(
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
      );
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
      return validationResult(
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
      );
    }
    for (const frame of artifact.frames) {
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
    return validationResult(
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
    );
  } finally {
    simulation.restoreState(saved);
  }
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
    ...(keyframe.root_yaw_velocity !== undefined
      ? { rootYawVelocity: keyframe.root_yaw_velocity }
      : {}),
    ...(keyframe.root_height !== undefined ? { rootHeight: keyframe.root_height } : {}),
    ...(keyframe.root_roll !== undefined ? { rootRoll: keyframe.root_roll } : {}),
    ...(keyframe.root_pitch !== undefined ? { rootPitch: keyframe.root_pitch } : {}),
    ...(keyframe.torso_yaw !== undefined
      ? { joints: { waist_yaw_joint: keyframe.torso_yaw } }
      : {})
  };
  const rooted = targetReference(baseline, rootTarget);
  const targets = [
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
    }] : [])
  ];
  return simulation.solveEndEffectorTargets(rooted, targets).reference;
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
