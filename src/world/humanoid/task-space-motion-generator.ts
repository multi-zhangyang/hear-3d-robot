import {
  HumanoidMotionArtifactSchema,
  serializeHumanoidReference,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import {
  TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR,
  type HumanoidMotionGeneratorDescriptor
} from "./motion-generator-contract.js";
import {
  HumanoidMotionPlanSchema,
  type HumanoidMotionKeyframe,
  type HumanoidMotionPlan
} from "./motion-plan-schema.js";
import {
  HUMANOID_TASK_SPACE_SERVO_DESCRIPTOR,
  type HumanoidTaskSpaceServoTarget
} from "./task-space-servo.js";
import {
  createG1HandArtifactCommand,
  interpolateG1HandCoordination,
  type G1HandArtifactCommand
} from "./hand-coordination.js";
import {
  interpolateReference,
  type HumanoidReference
} from "./reference.js";
import type { HumanoidSimulation } from "./simulation.js";
import {
  taskSpaceReference,
  taskSpaceTargets
} from "./task-space-motion-targets.js";

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

export class HumanoidMotionGenerationError extends Error {
  readonly atSeconds: number;

  constructor(atSeconds: number, cause: unknown) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    super(`Humanoid motion keyframe at ${atSeconds}s is invalid: ${detail}`, { cause });
    this.name = "HumanoidMotionGenerationError";
    this.atSeconds = atSeconds;
  }
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
    if (keyframe.hand_coordination != null) {
      channels.add("left_arm");
      channels.add("right_arm");
    }
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
    try {
      previous = taskSpaceReference(input.simulation, previous, keyframe);
    } catch (error) {
      throw new HumanoidMotionGenerationError(keyframe.at_seconds, error);
    }
    targets.push(previous);
  }

  const frames: Array<{
    atSeconds: number;
    reference: ReturnType<typeof serializeHumanoidReference>;
    handCommand?: G1HandArtifactCommand;
    taskSpaceTargets?: HumanoidTaskSpaceServoTarget[];
  }> = [];
  const coordinatedHands = plan.keyframes.some(
    (keyframe) => keyframe.hand_coordination != null
  );
  const steps = Math.ceil(plan.duration_seconds / controlStep);
  let segment = 1;
  for (let index = 1; index <= steps; index += 1) {
    const atSeconds = Math.min(index * controlStep, plan.duration_seconds);
    while (segment < plan.keyframes.length - 1
      && atSeconds > plan.keyframes[segment]!.at_seconds + 1e-9) {
      segment += 1;
    }
    const startKeyframe = plan.keyframes[segment - 1]!;
    const endKeyframe = plan.keyframes[segment]!;
    const duration = endKeyframe.at_seconds - startKeyframe.at_seconds;
    const interpolated = interpolateReference(
      targets[segment - 1]!,
      targets[segment]!,
      (atSeconds - startKeyframe.at_seconds) / duration,
      duration
    );
    const endpointTargets = taskSpaceTargets(endKeyframe);
    const taskSpaceServoTargets = endpointTargets.length === 0
      ? undefined
      : Math.abs(atSeconds - endKeyframe.at_seconds) <= 1e-9
        ? endpointTargets
        : input.simulation.measureEndEffectorTargets(
            interpolated,
            endpointTargets
          );
    const handCommand = plannedHandCommandAtTime(plan.keyframes, atSeconds);
    frames.push({
      atSeconds,
      reference: serializeHumanoidReference(interpolated),
      ...(handCommand ? { handCommand } : {}),
      ...(taskSpaceServoTargets
        ? { taskSpaceTargets: taskSpaceServoTargets }
        : {})
    });
  }
  const hasTaskSpaceServo = frames.some(
    (frame) => frame.taskSpaceTargets !== undefined
  );
  return HumanoidMotionArtifactSchema.parse({
    version: coordinatedHands ? 2 : 1,
    protocol: coordinatedHands ? "humanoid-motion-v2" : "humanoid-motion-v1",
    generator: implementation,
    controlStepSeconds: controlStep,
    durationSeconds: plan.duration_seconds,
    ...(hasTaskSpaceServo
      ? { taskSpaceServo: HUMANOID_TASK_SPACE_SERVO_DESCRIPTOR }
      : {}),
    frames
  });
}

export function plannedHandCommandAtTime(
  keyframes: readonly HumanoidMotionKeyframe[],
  atSeconds: number
): G1HandArtifactCommand | undefined {
  let previous: HumanoidMotionKeyframe | undefined;
  let next: HumanoidMotionKeyframe | undefined;
  for (const keyframe of keyframes) {
    if (keyframe.hand_coordination == null) continue;
    if (keyframe.at_seconds <= atSeconds + 1e-12) previous = keyframe;
    else {
      next = keyframe;
      break;
    }
  }
  if (!previous?.hand_coordination) return undefined;
  if (!next?.hand_coordination) {
    return createG1HandArtifactCommand(previous.hand_coordination);
  }
  const duration = next.at_seconds - previous.at_seconds;
  const progress = Math.max(0, Math.min(1, (
    atSeconds - previous.at_seconds
  ) / duration));
  return createG1HandArtifactCommand(interpolateG1HandCoordination(
    previous.hand_coordination,
    next.hand_coordination,
    progress
  ));
}
