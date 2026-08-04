import { z } from "zod";
import {
  G1_HAND_JOINT_LIMITS,
  G1_HAND_JOINT_NAMES,
  type G1HandJointName
} from "./morphology.js";

const UnitIntervalSchema = z.number().finite().min(0).max(1);

const G1HandCoordinationSideSchema = z.object({
  thumb_opposition: UnitIntervalSchema,
  thumb_curl: UnitIntervalSchema,
  index_curl: UnitIntervalSchema,
  middle_curl: UnitIntervalSchema
}).strict();

export const G1HandCoordinationSchema = z.object({
  left: G1HandCoordinationSideSchema,
  right: G1HandCoordinationSideSchema
}).strict();

const G1HandJointTargetsSchema = z.record(
  z.enum(G1_HAND_JOINT_NAMES),
  z.number().finite()
).superRefine((targets, context) => {
  for (const name of G1_HAND_JOINT_NAMES) {
    const value = targets[name];
    const [minimum, maximum] = G1_HAND_JOINT_LIMITS[name];
    if (value < minimum || value > maximum) {
      context.addIssue({
        code: "custom",
        path: [name],
        message: `G1 hand target exceeds ${name} limits`
      });
    }
  }
});

export const G1HandArtifactCommandSchema = z.object({
  coordination: G1HandCoordinationSchema,
  jointTargets: G1HandJointTargetsSchema
}).strict().superRefine((command, context) => {
  const resolved = resolveG1HandCoordination(command.coordination);
  for (const name of G1_HAND_JOINT_NAMES) {
    if (Math.abs(command.jointTargets[name] - resolved[name]) > 1e-9) {
      context.addIssue({
        code: "custom",
        path: ["jointTargets", name],
        message: `G1 hand target does not match coordination input: ${name}`
      });
    }
  }
});

export type G1HandCoordination = z.infer<typeof G1HandCoordinationSchema>;
export type G1HandArtifactCommand = z.infer<typeof G1HandArtifactCommandSchema>;

export function interpolateG1HandCoordination(
  startInput: G1HandCoordination,
  endInput: G1HandCoordination,
  progress: number
): G1HandCoordination {
  const start = G1HandCoordinationSchema.parse(startInput);
  const end = G1HandCoordinationSchema.parse(endInput);
  if (!Number.isFinite(progress) || progress < 0 || progress > 1) {
    throw new Error("G1 hand coordination interpolation progress must be within [0, 1]");
  }
  const amount = progress * progress * (3 - 2 * progress);
  return G1HandCoordinationSchema.parse({
    left: interpolateSide(start.left, end.left, amount),
    right: interpolateSide(start.right, end.right, amount)
  });
}

export function createG1HandArtifactCommand(
  input: G1HandCoordination
): G1HandArtifactCommand {
  const coordination = G1HandCoordinationSchema.parse(input);
  return G1HandArtifactCommandSchema.parse({
    coordination,
    jointTargets: resolveG1HandCoordination(coordination)
  });
}

export function resolveG1HandCoordination(
  input: G1HandCoordination
): Record<G1HandJointName, number> {
  const coordination = G1HandCoordinationSchema.parse(input);
  return G1HandJointTargetsSchema.parse({
    left_hand_thumb_0_joint: endpoint(
      "left_hand_thumb_0_joint",
      "minimum",
      coordination.left.thumb_opposition
    ),
    left_hand_thumb_1_joint: endpoint(
      "left_hand_thumb_1_joint",
      "maximum",
      coordination.left.thumb_curl
    ),
    left_hand_thumb_2_joint: endpoint(
      "left_hand_thumb_2_joint",
      "maximum",
      coordination.left.thumb_curl
    ),
    left_hand_index_0_joint: endpoint(
      "left_hand_index_0_joint",
      "minimum",
      coordination.left.index_curl
    ),
    left_hand_index_1_joint: endpoint(
      "left_hand_index_1_joint",
      "minimum",
      coordination.left.index_curl
    ),
    left_hand_middle_0_joint: endpoint(
      "left_hand_middle_0_joint",
      "minimum",
      coordination.left.middle_curl
    ),
    left_hand_middle_1_joint: endpoint(
      "left_hand_middle_1_joint",
      "minimum",
      coordination.left.middle_curl
    ),
    right_hand_thumb_0_joint: endpoint(
      "right_hand_thumb_0_joint",
      "minimum",
      coordination.right.thumb_opposition
    ),
    right_hand_thumb_1_joint: endpoint(
      "right_hand_thumb_1_joint",
      "minimum",
      coordination.right.thumb_curl
    ),
    right_hand_thumb_2_joint: endpoint(
      "right_hand_thumb_2_joint",
      "minimum",
      coordination.right.thumb_curl
    ),
    right_hand_index_0_joint: endpoint(
      "right_hand_index_0_joint",
      "maximum",
      coordination.right.index_curl
    ),
    right_hand_index_1_joint: endpoint(
      "right_hand_index_1_joint",
      "maximum",
      coordination.right.index_curl
    ),
    right_hand_middle_0_joint: endpoint(
      "right_hand_middle_0_joint",
      "maximum",
      coordination.right.middle_curl
    ),
    right_hand_middle_1_joint: endpoint(
      "right_hand_middle_1_joint",
      "maximum",
      coordination.right.middle_curl
    )
  });
}

function endpoint(
  joint: G1HandJointName,
  side: "minimum" | "maximum",
  amount: number
): number {
  const range = G1_HAND_JOINT_LIMITS[joint];
  const target = side === "minimum" ? range[0] : range[1];
  return target * amount;
}

function interpolateSide(
  start: G1HandCoordination["left"],
  end: G1HandCoordination["left"],
  amount: number
): G1HandCoordination["left"] {
  return {
    thumb_opposition: mix(start.thumb_opposition, end.thumb_opposition, amount),
    thumb_curl: mix(start.thumb_curl, end.thumb_curl, amount),
    index_curl: mix(start.index_curl, end.index_curl, amount),
    middle_curl: mix(start.middle_curl, end.middle_curl, amount)
  };
}

function mix(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
}
