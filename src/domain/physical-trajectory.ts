import { createHash } from "node:crypto";
import { z } from "zod";
import {
  HUMANOID_END_EFFECTORS,
  QuaternionSchema,
  Vec3Schema
} from "./schema.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

const PhysicalContactSampleSchema = z.object({
  key: z.string().trim().min(1),
  normal_force_n: z.number().finite().nonnegative()
}).strict();

const PhysicalObjectSampleSchema = z.object({
  id: z.string().trim().min(1),
  position: Vec3Schema
}).strict();

const EndEffectorPositionSchema = z.object(Object.fromEntries(
  HUMANOID_END_EFFECTORS.map((name) => [name, Vec3Schema])
) as Record<typeof HUMANOID_END_EFFECTORS[number], typeof Vec3Schema>).strict();

export const PhysicalTrajectoryFrameSchema = z.object({
  frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  root_position: Vec3Schema,
  root_rotation: QuaternionSchema,
  joint_positions: z.array(z.number().finite()).length(43),
  end_effectors: EndEffectorPositionSchema,
  contacts: z.array(PhysicalContactSampleSchema),
  objects: z.array(PhysicalObjectSampleSchema),
  support: z.enum(["none", "left", "right", "double"]),
  fallen: z.boolean(),
  frame_sha256: z.string().regex(SHA256_PATTERN)
}).strict().superRefine((sample, context) => {
  if (sample.frame_sha256 !== physicalTrajectoryFrameSha256(sample)) {
    context.addIssue({
      code: "custom",
      path: ["frame_sha256"],
      message: "Physical trajectory frame hash does not match its authoritative state"
    });
  }
  assertSortedUnique(sample.contacts.map((contact) => contact.key), "contacts", context);
  assertSortedUnique(sample.objects.map((object) => object.id), "objects", context);
});

const EndEffectorDistanceSchema = z.object(Object.fromEntries(
  HUMANOID_END_EFFECTORS.map((name) => [name, z.number().finite().nonnegative()])
) as Record<typeof HUMANOID_END_EFFECTORS[number], z.ZodNumber>).strict();

export const PhysicalTrajectorySummarySchema = z.object({
  version: z.literal(1),
  complete_from_admission: z.boolean(),
  start_frame: z.number().int().nonnegative(),
  end_frame: z.number().int().nonnegative(),
  start_world_revision: z.number().int().nonnegative(),
  end_world_revision: z.number().int().nonnegative(),
  observed_frame_count: z.number().int().positive(),
  sample_stride: z.number().int().positive(),
  joint_names: z.array(z.string().trim().min(1)).length(43),
  samples: z.array(PhysicalTrajectoryFrameSchema).min(1).max(64),
  root_path_length_m: z.number().finite().nonnegative(),
  root_planar_path_length_m: z.number().finite().nonnegative(),
  joint_total_variation_rad: z.number().finite().nonnegative(),
  end_effector_path_length_m: EndEffectorDistanceSchema,
  object_path_length_m: z.record(
    z.string().trim().min(1),
    z.number().finite().nonnegative()
  ),
  contact_transition_count: z.number().int().nonnegative(),
  trajectory_sha256: z.string().regex(SHA256_PATTERN)
}).strict().superRefine((summary, context) => {
  const first = summary.samples[0];
  const last = summary.samples.at(-1);
  if (!first || !last
    || first.frame !== summary.start_frame
    || first.world_revision !== summary.start_world_revision
    || last.frame !== summary.end_frame
    || last.world_revision !== summary.end_world_revision) {
    context.addIssue({
      code: "custom",
      path: ["samples"],
      message: "Physical trajectory samples must retain their exact endpoints"
    });
  }
  if (summary.end_frame < summary.start_frame
    || summary.end_world_revision < summary.start_world_revision
    || summary.end_frame - summary.start_frame
      !== summary.end_world_revision - summary.start_world_revision
    || summary.observed_frame_count !== summary.end_frame - summary.start_frame + 1) {
    context.addIssue({
      code: "custom",
      path: ["observed_frame_count"],
      message: "Physical trajectory coverage is not contiguous"
    });
  }
  if (!powerOfTwo(summary.sample_stride)) {
    context.addIssue({
      code: "custom",
      path: ["sample_stride"],
      message: "Physical trajectory sampling stride must be a power of two"
    });
  }
  for (let index = 1; index < summary.samples.length; index += 1) {
    const previous = summary.samples[index - 1]!;
    const current = summary.samples[index]!;
    if (current.frame <= previous.frame
      || current.world_revision <= previous.world_revision
      || current.frame - previous.frame
        !== current.world_revision - previous.world_revision) {
      context.addIssue({
        code: "custom",
        path: ["samples", index],
        message: "Physical trajectory samples must be strictly ordered"
      });
    }
  }
  assertSortedUnique(summary.joint_names, "joint_names", context, false);
});

export type PhysicalTrajectoryFrame = z.infer<typeof PhysicalTrajectoryFrameSchema>;
export type PhysicalTrajectorySummary = z.infer<typeof PhysicalTrajectorySummarySchema>;

export function physicalTrajectoryFrameSha256(
  frame: Omit<PhysicalTrajectoryFrame, "frame_sha256"> | PhysicalTrajectoryFrame
): string {
  const { frame_sha256: _frameSha256, ...identity } = frame as PhysicalTrajectoryFrame;
  return createHash("sha256").update(canonicalJson(identity)).digest("hex");
}

export function advancePhysicalTrajectorySha256(
  previousSha256: string | null,
  frameSha256: string
): string {
  if (previousSha256 !== null && !SHA256_PATTERN.test(previousSha256)) {
    throw new Error("Previous physical trajectory hash is invalid");
  }
  if (!SHA256_PATTERN.test(frameSha256)) {
    throw new Error("Physical trajectory frame hash is invalid");
  }
  return createHash("sha256")
    .update(previousSha256 === null ? "physical-trajectory-v1\0" : previousSha256)
    .update("\0")
    .update(frameSha256)
    .digest("hex");
}

function assertSortedUnique(
  values: readonly string[],
  path: string,
  context: z.RefinementCtx,
  requireLexicalOrder = true
): void {
  if (new Set(values).size !== values.length
    || requireLexicalOrder && values.some((value, index) => (
      index > 0 && values[index - 1]! > value
    ))) {
    context.addIssue({
      code: "custom",
      path: [path],
      message: `Physical trajectory ${path} must be unique${
        requireLexicalOrder ? " and sorted" : ""
      }`
    });
  }
}

function powerOfTwo(value: number): boolean {
  return value > 0 && (value & (value - 1)) === 0;
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .filter(([, entry]) => entry !== undefined)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => [key, canonicalValue(entry)]));
}
