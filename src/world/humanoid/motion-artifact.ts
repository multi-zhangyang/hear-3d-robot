import { z } from "zod";
import { createHash } from "node:crypto";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import {
  assertHumanoidReference,
  type HumanoidReference
} from "./reference.js";

const FiniteJointArraySchema = z.array(z.number().finite())
  .length(HUMANOID_JOINT_NAMES.length);
const JointTrackingWeightsSchema = z.array(z.number().finite().min(0).max(1))
  .length(HUMANOID_JOINT_NAMES.length);

export const HumanoidReferenceStateSchema = z.object({
  jointPositions: FiniteJointArraySchema,
  jointVelocities: FiniteJointArraySchema,
  jointTrackingWeights: JointTrackingWeightsSchema.optional(),
  rootVelocity: z.tuple([z.number().finite(), z.number().finite()]),
  rootYawVelocity: z.number().finite(),
  rootHeight: z.number().finite(),
  rootRoll: z.number().finite(),
  rootPitch: z.number().finite()
}).strict();

export const HumanoidMotionArtifactSchema = z.object({
  version: z.literal(1),
  protocol: z.literal("humanoid-motion-v1"),
  generator: z.string().trim().min(1),
  controlStepSeconds: z.number().finite().positive(),
  durationSeconds: z.number().finite().positive(),
  frames: z.array(z.object({
    atSeconds: z.number().finite().positive(),
    reference: HumanoidReferenceStateSchema
  }).strict()).min(1)
}).strict().superRefine((artifact, context) => {
  let previous = 0;
  for (let index = 0; index < artifact.frames.length; index += 1) {
    const atSeconds = artifact.frames[index]!.atSeconds;
    if (atSeconds <= previous) {
      context.addIssue({
        code: "custom",
        path: ["frames", index, "atSeconds"],
        message: "Humanoid motion artifact frame times must increase"
      });
    }
    previous = atSeconds;
  }
  const finalTime = artifact.frames.at(-1)?.atSeconds;
  if (finalTime !== artifact.durationSeconds) {
    context.addIssue({
      code: "custom",
      path: ["durationSeconds"],
      message: "Humanoid motion artifact must end at its declared duration"
    });
  }
});

export type HumanoidReferenceState = z.infer<typeof HumanoidReferenceStateSchema>;
export type HumanoidMotionArtifact = z.infer<typeof HumanoidMotionArtifactSchema>;

export function serializeHumanoidReference(
  reference: HumanoidReference
): HumanoidReferenceState {
  assertHumanoidReference(reference);
  return HumanoidReferenceStateSchema.parse({
    jointPositions: [...reference.jointPositions],
    jointVelocities: [...reference.jointVelocities],
    jointTrackingWeights: [...reference.jointTrackingWeights],
    rootVelocity: [...reference.rootVelocity],
    rootYawVelocity: reference.rootYawVelocity,
    rootHeight: reference.rootHeight,
    rootRoll: reference.rootRoll,
    rootPitch: reference.rootPitch
  });
}

export function hydrateHumanoidReference(
  state: HumanoidReferenceState
): HumanoidReference {
  const parsed = HumanoidReferenceStateSchema.parse(state);
  const reference: HumanoidReference = {
    jointPositions: Float64Array.from(parsed.jointPositions),
    jointVelocities: Float64Array.from(parsed.jointVelocities),
    jointTrackingWeights: parsed.jointTrackingWeights
      ? Float64Array.from(parsed.jointTrackingWeights)
      : new Float64Array(HUMANOID_JOINT_NAMES.length),
    rootVelocity: parsed.rootVelocity,
    rootYawVelocity: parsed.rootYawVelocity,
    rootHeight: parsed.rootHeight,
    rootRoll: parsed.rootRoll,
    rootPitch: parsed.rootPitch
  };
  assertHumanoidReference(reference);
  return reference;
}

export function humanoidMotionArtifactSummary(
  artifact: HumanoidMotionArtifact
): {
  protocol: HumanoidMotionArtifact["protocol"];
  generator: string;
  control_step_seconds: number;
  duration_seconds: number;
  frame_count: number;
  sha256: string;
} {
  const parsed = HumanoidMotionArtifactSchema.parse(artifact);
  return {
    protocol: parsed.protocol,
    generator: parsed.generator,
    control_step_seconds: parsed.controlStepSeconds,
    duration_seconds: parsed.durationSeconds,
    frame_count: parsed.frames.length,
    sha256: humanoidMotionArtifactSha256(parsed)
  };
}

export function humanoidMotionArtifactSha256(
  artifact: HumanoidMotionArtifact
): string {
  const parsed = HumanoidMotionArtifactSchema.parse(artifact);
  return createHash("sha256").update(JSON.stringify(parsed)).digest("hex");
}
