import { z } from "zod";
import { QuaternionSchema, Vec3Schema } from "../../domain/schema.js";
import {
  add,
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  rotateVector,
  subtract
} from "../geometry.js";
import type { HumanoidCarriedObjectBindingSet } from "./carried-object-binding.js";
import type { HumanoidEndEffectorBody } from "./task-space-targets.js";
import type { HumanoidReference } from "./reference.js";
import type {
  HumanoidSimulation,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import type { HumanoidTaskSpaceServoTarget } from "./task-space-servo.js";

const CARRY_WRIST_BODIES = [
  "left_wrist_yaw_link",
  "right_wrist_yaw_link"
] as const;
const MAXIMUM_CARRY_REFERENCE_CORRECTION_RADIANS = 0.012;

const HumanoidCarryTaskSpaceTargetSchema = z.object({
  body: z.enum(CARRY_WRIST_BODIES),
  frame: z.literal("torso"),
  position: Vec3Schema,
  orientation: QuaternionSchema,
  tolerance: z.number().finite().positive(),
  orientationTolerance: z.number().finite().positive().max(Math.PI)
}).strict().superRefine((target, context) => {
  try {
    normalizeQuaternion(target.orientation);
  } catch (error) {
    context.addIssue({
      code: "custom",
      path: ["orientation"],
      message: error instanceof Error ? error.message : "Invalid carry quaternion"
    });
  }
});

export const HumanoidCarryTaskSpaceTargetsSchema = z.array(
  HumanoidCarryTaskSpaceTargetSchema
).max(2).superRefine((targets, context) => {
  const bodies = new Set<string>();
  targets.forEach((target, index) => {
    if (!bodies.has(target.body)) {
      bodies.add(target.body);
      return;
    }
    context.addIssue({
      code: "custom",
      path: [index, "body"],
      message: "Carry targets cannot repeat a wrist"
    });
  });
});

export type HumanoidCarryTaskSpaceTarget = z.infer<
  typeof HumanoidCarryTaskSpaceTargetSchema
>;

export function captureHumanoidCarryTaskSpaceTargets(input: {
  snapshot: HumanoidSimulationSnapshot;
  bindings: HumanoidCarriedObjectBindingSet;
}): HumanoidCarryTaskSpaceTarget[] {
  const torso = input.snapshot.links.torso_link;
  const inverseTorso = inverseQuaternion(torso.rotation);
  return HumanoidCarryTaskSpaceTargetsSchema.parse(
    input.bindings.bindings.map((binding) => {
      const body = binding.hand === "left"
        ? "left_wrist_yaw_link" as const
        : "right_wrist_yaw_link" as const;
      const wrist = input.snapshot.links[body];
      return {
        body,
        frame: "torso" as const,
        position: rotateVector(
          inverseTorso,
          subtract(wrist.position, torso.position)
        ),
        orientation: normalizeQuaternion(multiplyQuaternion(
          inverseTorso,
          wrist.rotation
        )),
        tolerance: 0.015,
        orientationTolerance: 0.04
      };
    })
  );
}

export function applyHumanoidCarryTaskSpaceServo(input: {
  simulation: HumanoidSimulation;
  reference: HumanoidReference;
  targets: readonly HumanoidCarryTaskSpaceTarget[];
  modelControlledBodies?: ReadonlySet<HumanoidEndEffectorBody>;
  maximumReferenceCorrectionRadians?: number;
}): HumanoidReference {
  if (input.targets.length === 0) return input.reference;
  const targets = HumanoidCarryTaskSpaceTargetsSchema.parse(input.targets)
    .filter((target) => !input.modelControlledBodies?.has(target.body));
  if (targets.length === 0) return input.reference;
  const torso = input.simulation.snapshot().links.torso_link;
  const worldTargets: HumanoidTaskSpaceServoTarget[] = targets.map((target) => ({
    body: target.body,
    frame: "world",
    position: add(
      torso.position,
      rotateVector(torso.rotation, target.position)
    ),
    orientation: normalizeQuaternion(multiplyQuaternion(
      torso.rotation,
      target.orientation
    )),
    tolerance: target.tolerance,
    orientationTolerance: target.orientationTolerance
  }));
  return input.simulation.solveEndEffectorTargets(
    input.reference,
    worldTargets,
    {
      initialConfiguration: "current",
      maximumReferenceCorrectionRadians: input.maximumReferenceCorrectionRadians
        ?? MAXIMUM_CARRY_REFERENCE_CORRECTION_RADIANS
    }
  ).reference;
}

export function humanoidCarryTaskSpaceTargetsMatchBindings(
  targets: readonly HumanoidCarryTaskSpaceTarget[],
  bindings: HumanoidCarriedObjectBindingSet
): boolean {
  const actual = HumanoidCarryTaskSpaceTargetsSchema.parse(targets)
    .map((target) => target.body)
    .sort();
  const expected = bindings.bindings.map((binding) => (
    binding.hand === "left"
      ? "left_wrist_yaw_link" as const
      : "right_wrist_yaw_link" as const
  )).sort();
  return actual.length === expected.length
    && actual.every((body, index) => body === expected[index]);
}
