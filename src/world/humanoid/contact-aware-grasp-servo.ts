import type { Quaternion, Vec3 } from "../../domain/schema.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  quaternionRotationVector,
  rotateVector,
  scale,
  subtract,
  vectorLength
} from "../geometry.js";
import type { G1HandArtifactCommand } from "./hand-coordination.js";
import { g1HandObjectContacts } from "./hand-contact-evidence.js";
import {
  humanoidGraspContractSha256,
  type HumanoidGraspContract
} from "./grasp-tracker.js";
import type { HumanoidMotionOptionContract } from "./motion-option.js";
import type { HumanoidGraspRegistry } from "./grasp-registry.js";
import {
  G1_HAND_JOINT_LIMITS,
  G1_HAND_JOINT_NAMES,
  type G1HandJointName
} from "./morphology.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

export interface G1ContactAwareGraspTarget {
  objectId: string;
  hand: "left" | "right";
  minimumNormalForceN: number;
  referenceRelativePose?: {
    translation: Vec3;
    rotation: Quaternion;
  };
}

export interface G1ContactAwareGraspServoEvidence {
  protocol: "g1-contact-aware-grasp-servo-v1";
  targetObjectIds: string[];
  limitedDigits: string[];
  maximumObservedNormalForceN: number;
  poseRegulatedHands: Array<"left" | "right">;
  saturationLimitedDigits: string[];
  maximumRotationErrorRadians: number;
  maximumTranslationErrorMeters: number;
}

export interface G1ContactAwareGraspServoResult {
  jointTargets: Record<G1HandJointName, number>;
  evidence: G1ContactAwareGraspServoEvidence;
}

const MAXIMUM_FREE_CLOSURE_STEP_RAD = 0.025;
const CONTACT_APPROACH_STEP_RAD = 0.006;
const QUALIFIED_CONTACT_PRELOAD_RAD = 0.0005;
const CARRY_CONTACT_RECOVERY_STEP_RAD = 0.001;
const CARRY_FREE_RECOVERY_STEP_RAD = 0.003;
const MAXIMUM_DIFFERENTIAL_CLOSURE_STEP_RAD = 0.001;
const ROTATION_REGULATION_FULL_AUTHORITY_RADIANS = 0.08;
const TRANSLATION_REGULATION_FULL_AUTHORITY_METERS = 0.012;
const RELATIVE_ANGULAR_VELOCITY_DAMPING_SECONDS = 0.04;
const RELATIVE_LINEAR_VELOCITY_DAMPING_SECONDS = 0.02;
const TRANSLATION_REGULATION_WEIGHT = 0.5;

export function contactAwareG1GraspTargets(input: {
  command: G1HandArtifactCommand;
  snapshot: HumanoidSimulationSnapshot;
  targets: readonly G1ContactAwareGraspTarget[];
}): G1ContactAwareGraspServoResult {
  return contactAwareG1GraspJointTargets({
    requestedJointTargets: input.command.jointTargets,
    snapshot: input.snapshot,
    targets: input.targets
  });
}

export function contactAwareG1GraspJointTargets(input: {
  requestedJointTargets: Readonly<Record<G1HandJointName, number>>;
  snapshot: HumanoidSimulationSnapshot;
  targets: readonly G1ContactAwareGraspTarget[];
}): G1ContactAwareGraspServoResult {
  const targets = validateTargets(input.targets);
  const forceByDigit = new Map<string, number>();
  for (const target of targets) {
    for (const contact of g1HandObjectContacts(
      input.snapshot.contacts,
      target.objectId
    )) {
      if (contact.hand !== target.hand) continue;
      const digit = handDigit(contact.handLink);
      forceByDigit.set(
        digit,
        Math.max(forceByDigit.get(digit) ?? 0, contact.normalForce)
      );
    }
  }

  const targetByHand = new Map(targets.map((target) => [target.hand, target]));
  const poseRegulation = carriedPoseRegulation(input.snapshot, targets);
  const limitedDigits = new Set<string>();
  const saturationLimitedDigits = new Set<string>();
  const jointTargets = Object.fromEntries(G1_HAND_JOINT_NAMES.map((joint) => {
    const requested = input.requestedJointTargets[joint];
    const hand = joint.startsWith("left_") ? "left" : "right";
    const graspTarget = targetByHand.get(hand);
    if (!graspTarget) return [joint, requested];

    const jointState = input.snapshot.hands.joints[joint];
    const current = jointState.position;
    const direction = Math.sign(requested);
    if (direction === 0 || current * direction > Math.abs(requested) + 1e-9) {
      return [joint, requested];
    }
    const requestedClosure = Math.abs(requested);
    const currentClosure = Math.max(0, current * direction);
    if (requestedClosure + 1e-9 < currentClosure) {
      return [joint, requested];
    }

    const digit = jointDigit(joint);
    const force = forceByDigit.get(`${hand}:${digit}`) ?? 0;
    const applied = jointState.target;
    const appliedClosure = applied * direction > 0
      ? Math.min(requestedClosure, applied * direction)
      : 0;
    const heldClosure = Math.max(currentClosure, appliedClosure);
    if (force > 0) limitedDigits.add(`${hand}:${digit}`);
    let closure = graspTarget.referenceRelativePose
      ? force >= graspTarget.minimumNormalForceN
        ? heldClosure
        : Math.min(
            requestedClosure,
            heldClosure + (force > 0
              ? CARRY_CONTACT_RECOVERY_STEP_RAD
              : CARRY_FREE_RECOVERY_STEP_RAD)
          )
      : force >= graspTarget.minimumNormalForceN * 2
        ? heldClosure
        : Math.min(
            requestedClosure,
            heldClosure + (force >= graspTarget.minimumNormalForceN
              ? QUALIFIED_CONTACT_PRELOAD_RAD
              : force > 0
                ? CONTACT_APPROACH_STEP_RAD
                : MAXIMUM_FREE_CLOSURE_STEP_RAD)
          );
    let differential = poseRegulation.correctionByDigit.get(
      `${hand}:${digit}`
    ) ?? 0;
    if (joint.includes("_thumb_0_joint")) differential = 0;
    if (differential > 0
      && jointState.saturated
      && jointState.appliedNewtonMeters * direction > 0) {
      saturationLimitedDigits.add(`${hand}:${digit}`);
      differential = 0;
    }
    if (differential < 0
      && force < graspTarget.minimumNormalForceN * 2) {
      differential = 0;
    }
    if (differential > 0) {
      closure = Math.min(
        requestedClosure,
        Math.max(closure, appliedClosure + differential)
      );
    } else if (differential < 0) {
      closure = Math.max(
        0,
        Math.min(closure, Math.max(0, appliedClosure + differential))
      );
    }
    const [minimum, maximum] = G1_HAND_JOINT_LIMITS[joint];
    return [joint, clamp(direction * closure, minimum, maximum)];
  })) as Record<G1HandJointName, number>;

  return {
    jointTargets,
    evidence: {
      protocol: "g1-contact-aware-grasp-servo-v1",
      targetObjectIds: [...new Set(targets.map((target) => target.objectId))].sort(),
      limitedDigits: [...limitedDigits].sort(),
      maximumObservedNormalForceN: Math.max(0, ...forceByDigit.values()),
      poseRegulatedHands: poseRegulation.poseRegulatedHands,
      saturationLimitedDigits: [...saturationLimitedDigits].sort(),
      maximumRotationErrorRadians:
        poseRegulation.maximumRotationErrorRadians,
      maximumTranslationErrorMeters:
        poseRegulation.maximumTranslationErrorMeters
    }
  };
}

function carriedPoseRegulation(
  snapshot: HumanoidSimulationSnapshot,
  targets: readonly G1ContactAwareGraspTarget[]
): {
  correctionByDigit: ReadonlyMap<string, number>;
  poseRegulatedHands: Array<"left" | "right">;
  maximumRotationErrorRadians: number;
  maximumTranslationErrorMeters: number;
} {
  const correctionByDigit = new Map<string, number>();
  const poseRegulatedHands: Array<"left" | "right"> = [];
  let maximumRotationErrorRadians = 0;
  let maximumTranslationErrorMeters = 0;
  for (const target of targets) {
    if (!target.referenceRelativePose) continue;
    const object = snapshot.objects[target.objectId];
    if (!object) {
      throw new Error(`Carried grasp object is missing: ${target.objectId}`);
    }
    const wrist = snapshot.links[
      target.hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
    ];
    const inverseWrist = inverseQuaternion(wrist.rotation);
    const currentRelativeTranslation = rotateVector(
      inverseWrist,
      subtract(object.position, wrist.position)
    );
    const translationError = subtract(
      target.referenceRelativePose.translation,
      currentRelativeTranslation
    );
    const translationErrorMeters = vectorLength(translationError);
    maximumTranslationErrorMeters = Math.max(
      maximumTranslationErrorMeters,
      translationErrorMeters
    );
    const currentRelativeRotation = normalizeQuaternion(multiplyQuaternion(
      inverseWrist,
      object.rotation
    ));
    const localRotationError = quaternionRotationVector(
      target.referenceRelativePose.rotation,
      currentRelativeRotation
    );
    const rotationErrorRadians = vectorLength(localRotationError);
    maximumRotationErrorRadians = Math.max(
      maximumRotationErrorRadians,
      rotationErrorRadians
    );
    const relativeAngularVelocity = subtract(
      object.angularVelocity,
      wrist.angularVelocity
    );
    const desiredMoment = subtract(
      rotateVector(wrist.rotation, localRotationError),
      scale(
        relativeAngularVelocity,
        RELATIVE_ANGULAR_VELOCITY_DAMPING_SECONDS
      )
    );
    const desiredMagnitude = vectorLength(desiredMoment);
    const worldTranslationError = rotateVector(
      wrist.rotation,
      translationError
    );
    const desiredForce = subtract(
      worldTranslationError,
      scale(
        subtract(object.linearVelocity, wrist.linearVelocity),
        RELATIVE_LINEAR_VELOCITY_DAMPING_SECONDS
      )
    );
    const desiredForceMagnitude = vectorLength(desiredForce);
    if (desiredMagnitude <= 1e-9 && desiredForceMagnitude <= 1e-9) continue;
    const moments = new Map<string, {
      momentSum: Vec3;
      normalSum: Vec3;
      normalForceSum: number;
    }>();
    for (const contact of g1HandObjectContacts(
      snapshot.contacts,
      target.objectId
    )) {
      if (contact.hand !== target.hand
        || !contact.normalFromHand
        || contact.normalForce <= 1e-9
        || contact.handLink.includes("_palm_")) {
        continue;
      }
      const digit = `${target.hand}:${handDigitName(contact.handLink)}`;
      const moment = cross(
        subtract(contact.position, object.position),
        contact.normalFromHand
      );
      const previous = moments.get(digit) ?? {
        momentSum: { x: 0, y: 0, z: 0 },
        normalSum: { x: 0, y: 0, z: 0 },
        normalForceSum: 0
      };
      const weight = Math.sqrt(contact.normalForce);
      previous.momentSum = {
        x: previous.momentSum.x + moment.x * weight,
        y: previous.momentSum.y + moment.y * weight,
        z: previous.momentSum.z + moment.z * weight
      };
      previous.normalSum = {
        x: previous.normalSum.x + contact.normalFromHand.x * weight,
        y: previous.normalSum.y + contact.normalFromHand.y * weight,
        z: previous.normalSum.z + contact.normalFromHand.z * weight
      };
      previous.normalForceSum += weight;
      moments.set(digit, previous);
    }
    if (moments.size < 2) continue;
    poseRegulatedHands.push(target.hand);
    const rotationAuthority = Math.min(
      1,
      rotationErrorRadians / ROTATION_REGULATION_FULL_AUTHORITY_RADIANS
    );
    const translationAuthority = Math.min(
      1,
      translationErrorMeters / TRANSLATION_REGULATION_FULL_AUTHORITY_METERS
    );
    const authority = Math.max(rotationAuthority, translationAuthority);
    const alignments = [...moments].flatMap(([digit, moment]) => {
      const averageMoment = scale(
        moment.momentSum,
        1 / moment.normalForceSum
      );
      const momentMagnitude = vectorLength(averageMoment);
      const averageNormal = scale(
        moment.normalSum,
        1 / moment.normalForceSum
      );
      const normalMagnitude = vectorLength(averageNormal);
      const rotationAlignment = desiredMagnitude > 1e-9
        && momentMagnitude > 1e-9
        ? dot(averageMoment, desiredMoment)
          / (momentMagnitude * desiredMagnitude)
        : 0;
      const translationAlignment = desiredForceMagnitude > 1e-9
        && normalMagnitude > 1e-9
        ? dot(averageNormal, desiredForce)
          / (normalMagnitude * desiredForceMagnitude)
        : 0;
      const alignment = rotationAuthority * rotationAlignment
        + TRANSLATION_REGULATION_WEIGHT
          * translationAuthority
          * translationAlignment;
      return Math.abs(alignment) <= 1e-9
        ? []
        : [{
            digit,
            alignment
          }];
    });
    if (alignments.length < 2) continue;
    const meanAlignment = alignments.reduce(
      (sum, entry) => sum + entry.alignment,
      0
    ) / alignments.length;
    const maximumCenteredAlignment = Math.max(
      1e-9,
      ...alignments.map((entry) => Math.abs(
        entry.alignment - meanAlignment
      ))
    );
    for (const { digit, alignment } of alignments) {
      correctionByDigit.set(
        digit,
        clamp(
          (alignment - meanAlignment) / maximumCenteredAlignment,
          -1,
          1
        ) * authority * MAXIMUM_DIFFERENTIAL_CLOSURE_STEP_RAD
      );
    }
  }
  return {
    correctionByDigit,
    poseRegulatedHands: [...new Set(poseRegulatedHands)].sort(),
    maximumRotationErrorRadians,
    maximumTranslationErrorMeters
  };
}

export function contactAwareG1GraspTargetsForBindings(input: {
  bindings: readonly { object_id: string; hand: "left" | "right" }[];
  graspRegistry: HumanoidGraspRegistry;
}): G1ContactAwareGraspTarget[] {
  const checkpoint = input.graspRegistry.checkpoint();
  return input.bindings.map((binding) => {
    const track = checkpoint.tracker.tracks.find((candidate) => (
      candidate.object_id === binding.object_id && candidate.hand === binding.hand
    ));
    if (!track?.attempt) {
      throw new Error(
        `Carried grasp target has no certified relative pose: ${binding.object_id}/${binding.hand}`
      );
    }
    return {
      objectId: binding.object_id,
      hand: binding.hand,
      minimumNormalForceN:
        input.graspRegistry.contract.minimum_contact_normal_force_n,
      referenceRelativePose: structuredClone(
        track.attempt.reference_relative_pose
      )
    };
  });
}

export function contactAwareG1GraspTargetsForOption(input: {
  option: HumanoidMotionOptionContract;
  graspContract: HumanoidGraspContract;
}): G1ContactAwareGraspTarget[] {
  const contractSha256 = humanoidGraspContractSha256(input.graspContract);
  return input.option.predicates.flatMap((predicate) => {
    if (predicate.type !== "grasp_verified") return [];
    if (predicate.grasp_contract_sha256 !== contractSha256) {
      throw new Error("Contact-aware grasp option contract does not match authority");
    }
    return [{
      objectId: predicate.object_id,
      hand: predicate.hand,
      minimumNormalForceN: input.graspContract.minimum_contact_normal_force_n
    }];
  });
}

export function mergeG1ContactAwareGraspTargets(
  ...groups: ReadonlyArray<readonly G1ContactAwareGraspTarget[]>
): G1ContactAwareGraspTarget[] {
  const targets = new Map<"left" | "right", G1ContactAwareGraspTarget>();
  for (const target of groups.flat()) {
    const previous = targets.get(target.hand);
    if (previous && (previous.objectId !== target.objectId
      || previous.minimumNormalForceN !== target.minimumNormalForceN
      || (previous.referenceRelativePose && target.referenceRelativePose
        && JSON.stringify(previous.referenceRelativePose)
          !== JSON.stringify(target.referenceRelativePose)))) {
      throw new Error(
        `Cannot grasp ${target.objectId} with occupied ${target.hand} hand`
      );
    }
    const referenceRelativePose =
      target.referenceRelativePose ?? previous?.referenceRelativePose;
    targets.set(target.hand, {
      ...previous,
      ...target,
      ...(referenceRelativePose
        ? { referenceRelativePose: structuredClone(referenceRelativePose) }
        : {})
    });
  }
  return validateTargets([...targets.values()]).sort((left, right) => (
    left.hand.localeCompare(right.hand)
  ));
}

function validateTargets(
  targets: readonly G1ContactAwareGraspTarget[]
): G1ContactAwareGraspTarget[] {
  const seenHands = new Set<string>();
  return targets.map((target) => {
    if (target.objectId.trim().length === 0) {
      throw new Error("Contact-aware grasp target object id cannot be empty");
    }
    if (!Number.isFinite(target.minimumNormalForceN)
      || target.minimumNormalForceN <= 0) {
      throw new Error("Contact-aware grasp force threshold must be positive");
    }
    if (seenHands.has(target.hand)) {
      throw new Error(`Contact-aware grasp servo received two ${target.hand} targets`);
    }
    seenHands.add(target.hand);
    const referenceRelativePose = target.referenceRelativePose;
    if (referenceRelativePose
      && ![
        referenceRelativePose.translation.x,
        referenceRelativePose.translation.y,
        referenceRelativePose.translation.z,
        referenceRelativePose.rotation.x,
        referenceRelativePose.rotation.y,
        referenceRelativePose.rotation.z,
        referenceRelativePose.rotation.w
      ].every(Number.isFinite)) {
      throw new Error("Contact-aware grasp reference pose must be finite");
    }
    return {
      ...target,
      objectId: target.objectId.trim(),
      ...(referenceRelativePose
        ? { referenceRelativePose: structuredClone(referenceRelativePose) }
        : {})
    };
  });
}

function handDigit(surface: string): string {
  const hand = surface.startsWith("left_") ? "left" : "right";
  if (surface.includes("_thumb_")) return `${hand}:thumb`;
  if (surface.includes("_index_")) return `${hand}:index`;
  if (surface.includes("_middle_")) return `${hand}:middle`;
  return `${hand}:palm`;
}

function handDigitName(surface: string): "thumb" | "index" | "middle" {
  if (surface.includes("_thumb_")) return "thumb";
  if (surface.includes("_index_")) return "index";
  return "middle";
}

function jointDigit(joint: G1HandJointName): "thumb" | "index" | "middle" {
  if (joint.includes("_thumb_")) return "thumb";
  if (joint.includes("_index_")) return "index";
  return "middle";
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function cross(left: Vec3, right: Vec3): Vec3 {
  return {
    x: left.y * right.z - left.z * right.y,
    y: left.z * right.x - left.x * right.z,
    z: left.x * right.y - left.y * right.x
  };
}

function dot(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}
