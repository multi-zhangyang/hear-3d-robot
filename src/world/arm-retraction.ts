import { planarDistance } from "./geometry.js";
import { ROBOT_SPEC, type ArmJointName, type RobotJointState } from "./robot-model.js";
import { rigTransforms } from "./rig.js";

export interface RankedArmRetraction {
  targets: Pick<RobotJointState, "shoulder" | "elbow" | "wrist">;
  compactness: number;
  estimatedJointTravel: number;
  gripperHeight: number;
  gripperRadius: number;
}

/**
 * Produces a bounded, geometry-ranked joint-space search set.
 *
 * These are not commands or fixed recovery poses. The world still rejects
 * candidates that cannot be reached by its continuous arm trajectory or that
 * fail the requested full-rig base sweep. Keeping enumeration pure makes the
 * expensive Rapier/Recast checks route-specific and testable in RapierWorld.
 */
export function rankedArmRetractions(
  current: RobotJointState,
  limit = 192
): RankedArmRetraction[] {
  const values = {
    shoulder: jointSamples("shoulder", 13),
    elbow: jointSamples("elbow", 12),
    wrist: jointSamples("wrist", 15)
  };
  const base = { x: 0, y: ROBOT_SPEC.base.centerY, z: 0 };
  const candidates: RankedArmRetraction[] = [];

  for (const shoulder of values.shoulder) {
    for (const elbow of values.elbow) {
      for (const wrist of values.wrist) {
        const targets = { shoulder, elbow, wrist };
        const joints: RobotJointState = { ...current, ...targets };
        const transforms = rigTransforms(joints, base, 0);
        const armTransforms = [
          transforms.upperArm,
          transforms.forearm,
          transforms.wrist,
          transforms.leftFinger,
          transforms.rightFinger
        ];
        const envelopeRadius = Math.max(...armTransforms.map((transform) =>
          planarDistance(base, transform.position)
        ));
        const lowestCenter = Math.min(...armTransforms.map((transform) => transform.position.y));
        const gripperRadius = planarDistance(base, transforms.gripper.position);
        const groundPenalty = Math.max(0, ROBOT_SPEC.base.centerY - lowestCenter) * 8;
        const travel = normalizedTravel(current, targets);
        candidates.push({
          targets,
          compactness: envelopeRadius + gripperRadius * 0.35 + groundPenalty + travel * 0.08,
          estimatedJointTravel: travel,
          gripperHeight: transforms.gripper.position.y,
          gripperRadius
        });
      }
    }
  }

  return candidates
    .sort((left, right) => left.compactness - right.compactness
      || left.estimatedJointTravel - right.estimatedJointTravel
      || left.targets.shoulder - right.targets.shoulder
      || left.targets.elbow - right.targets.elbow
      || left.targets.wrist - right.targets.wrist)
    .slice(0, limit);
}

function jointSamples(joint: ArmJointName, count: number): number[] {
  const bounds = ROBOT_SPEC.joints[joint];
  return Array.from({ length: count }, (_, index) => {
    const fraction = count === 1 ? 0.5 : index / (count - 1);
    return bounds.minimum + (bounds.maximum - bounds.minimum) * fraction;
  });
}

function normalizedTravel(
  current: RobotJointState,
  targets: Pick<RobotJointState, "shoulder" | "elbow" | "wrist">
): number {
  return (["shoulder", "elbow", "wrist"] as const).reduce((total, joint) => {
    const bounds = ROBOT_SPEC.joints[joint];
    return total + Math.abs(targets[joint] - current[joint]) / (bounds.maximum - bounds.minimum);
  }, 0);
}
