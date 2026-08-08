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
import {
  type HumanoidPlanningRootPose,
  type HumanoidSimulation
} from "./simulation.js";
import {
  taskSpaceReference,
  taskSpaceRootReference,
  taskSpaceTargets
} from "./task-space-motion-targets.js";
import { yawFromQuaternion } from "../geometry.js";

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

interface PredictedPlanningRootState extends HumanoidPlanningRootPose {
  worldVelocity: { x: number; z: number };
  yawVelocity: number;
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
  const planningRootPoses: PredictedPlanningRootState[] = [];
  const predictsMobileTaskSpace = plan.keyframes.some((keyframe) => (
    keyframe.left_hand?.kinematic_scope === "whole_body_reach"
      || keyframe.right_hand?.kinematic_scope === "whole_body_reach"
  )) && plan.keyframes.some((keyframe) => keyframe.root_velocity != null);
  let previous = input.baseline;
  const initial = input.simulation.snapshot();
  let planningRootPose: PredictedPlanningRootState = {
    position: { ...initial.rootPosition },
    yawRadians: yawFromQuaternion(initial.rootRotation),
    worldVelocity: { x: 0, z: 0 },
    yawVelocity: 0
  };
  let previousAtSeconds = 0;
  for (const keyframe of plan.keyframes) {
    try {
      const rooted = taskSpaceRootReference(previous, keyframe);
      const segmentDuration = keyframe.at_seconds - previousAtSeconds;
      planningRootPose = predictPlanningRootPose(
        planningRootPose,
        previous,
        rooted,
        segmentDuration,
        segmentDuration
      );
      previous = taskSpaceReferenceWithFallbackSeed({
        simulation: input.simulation,
        primary: previous,
        fallback: input.baseline,
        keyframe,
        ...(predictsMobileTaskSpace ? { planningRootPose } : {})
      });
    } catch (error) {
      throw new HumanoidMotionGenerationError(keyframe.at_seconds, error);
    }
    targets.push(previous);
    planningRootPoses.push(planningRootPose);
    previousAtSeconds = keyframe.at_seconds;
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
    const endpointTargets = taskSpaceTargets(endKeyframe);
    const interpolated = interpolateReference(
      targets[segment - 1]!,
      targets[segment]!,
      (atSeconds - startKeyframe.at_seconds) / duration,
      duration
    );
    const taskSpaceServoTargets = endpointTargets.length === 0
      ? undefined
      : Math.abs(atSeconds - endKeyframe.at_seconds) <= 1e-9
        ? endpointTargets
        : input.simulation.measureEndEffectorTargets(
            interpolated,
            endpointTargets,
            predictsMobileTaskSpace ? {
              planningRootPose: predictPlanningRootPose(
                planningRootPoses[segment - 1]!,
                targets[segment - 1]!,
                targets[segment]!,
                atSeconds - startKeyframe.at_seconds,
                duration
              )
            } : {}
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

function taskSpaceReferenceWithFallbackSeed(input: {
  simulation: HumanoidSimulation;
  primary: HumanoidReference;
  fallback: HumanoidReference;
  keyframe: HumanoidMotionKeyframe;
  planningRootPose?: HumanoidPlanningRootPose;
}): HumanoidReference {
  const solve = (seed: HumanoidReference) => taskSpaceReference(
    input.simulation,
    seed,
    input.keyframe,
    input.planningRootPose
  );
  try {
    return solve(input.primary);
  } catch (primaryError) {
    if (input.primary === input.fallback) throw primaryError;
    try {
      return solve(input.fallback);
    } catch (fallbackError) {
      throw new AggregateError(
        [primaryError, fallbackError],
        `Task-space IK failed from continuous and action-start seeds: ${errorMessage(primaryError)}; ${errorMessage(fallbackError)}`
      );
    }
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function predictPlanningRootPose(
  start: PredictedPlanningRootState,
  startReference: HumanoidReference,
  endReference: HumanoidReference,
  elapsedSeconds: number,
  segmentDurationSeconds: number
): PredictedPlanningRootState {
  if (elapsedSeconds <= 0 || segmentDurationSeconds <= 0) {
    return {
      position: { ...start.position },
      yawRadians: start.yawRadians,
      worldVelocity: { ...start.worldVelocity },
      yawVelocity: start.yawVelocity
    };
  }
  const elapsed = Math.min(elapsedSeconds, segmentDurationSeconds);
  const steps = Math.max(1, Math.ceil(elapsed / 0.04));
  const stepSeconds = elapsed / steps;
  const position = { ...start.position };
  let yaw = start.yawRadians;
  let worldVelocity = { ...start.worldVelocity };
  let yawVelocity = start.yawVelocity;
  const response = 1 - Math.exp(-stepSeconds / 1.1);
  for (let index = 0; index < steps; index += 1) {
    const progress = (index + 0.5) * stepSeconds / segmentDurationSeconds;
    const blend = progress * progress * (3 - 2 * progress);
    const commandedForward = mix(
      startReference.rootVelocity[0],
      endReference.rootVelocity[0],
      blend
    );
    const commandedLateral = mix(
      startReference.rootVelocity[1],
      endReference.rootVelocity[1],
      blend
    );
    const commandedYawVelocity = mix(
      startReference.rootYawVelocity,
      endReference.rootYawVelocity,
      blend
    );
    const targetLocalVelocity = effectivePolicyPlanarVelocity(
      commandedForward,
      commandedLateral
    );
    const targetWorldVelocity = {
      x: targetLocalVelocity.forward * Math.sin(yaw)
        + targetLocalVelocity.lateral * Math.cos(yaw),
      z: targetLocalVelocity.forward * Math.cos(yaw)
        - targetLocalVelocity.lateral * Math.sin(yaw)
    };
    worldVelocity = {
      x: mix(worldVelocity.x, targetWorldVelocity.x, response),
      z: mix(worldVelocity.z, targetWorldVelocity.z, response)
    };
    yawVelocity = mix(yawVelocity, commandedYawVelocity, response);
    position.x += worldVelocity.x * stepSeconds;
    position.z += worldVelocity.z * stepSeconds;
    yaw += yawVelocity * stepSeconds;
  }
  return { position, yawRadians: yaw, worldVelocity, yawVelocity };
}

function effectivePolicyPlanarVelocity(
  forward: number,
  lateral: number
): { forward: number; lateral: number } {
  const speed = Math.hypot(forward, lateral);
  if (speed <= 1e-9) return { forward: 0, lateral: 0 };
  const effectiveSpeed = speed < 0.15
    ? speed * 0.075
    : Math.max(0, speed - 0.075);
  const scale = effectiveSpeed / speed;
  return { forward: forward * scale, lateral: lateral * scale };
}

function mix(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
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
