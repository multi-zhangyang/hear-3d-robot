import {
  HUMANOID_JOINT_INDEX,
  HUMANOID_JOINT_NAMES,
  YAHMP_POLICY,
  type HumanoidJointName
} from "./model.js";

export interface HumanoidReference {
  jointPositions: Float64Array;
  jointVelocities: Float64Array;
  jointTrackingWeights: Float64Array;
  rootVelocity: readonly [forward: number, lateral: number];
  rootYawVelocity: number;
  rootHeight: number;
  rootRoll: number;
  rootPitch: number;
}

export interface HumanoidReferenceTarget {
  joints?: Partial<Record<HumanoidJointName, number>>;
  rootVelocity?: readonly [forward: number, lateral: number];
  rootYawVelocity?: number;
  rootHeight?: number;
  rootRoll?: number;
  rootPitch?: number;
}

export function neutralHumanoidReference(): HumanoidReference {
  return {
    jointPositions: Float64Array.from(YAHMP_POLICY.defaultJointPositions),
    jointVelocities: new Float64Array(HUMANOID_JOINT_NAMES.length),
    jointTrackingWeights: new Float64Array(HUMANOID_JOINT_NAMES.length),
    rootVelocity: [0, 0],
    rootYawVelocity: 0,
    rootHeight: 0.793,
    rootRoll: 0,
    rootPitch: 0
  };
}

export function targetReference(
  baseline: HumanoidReference,
  target: HumanoidReferenceTarget
): HumanoidReference {
  assertHumanoidReference(baseline);
  const jointPositions = baseline.jointPositions.slice();
  const jointTrackingWeights = baseline.jointTrackingWeights.slice();
  for (const [name, value] of Object.entries(target.joints ?? {}) as [HumanoidJointName, number][]) {
    const index = HUMANOID_JOINT_INDEX.get(name);
    if (index === undefined || !Number.isFinite(value)) {
      throw new Error(`Invalid humanoid joint target: ${name}`);
    }
    jointPositions[index] = value;
    jointTrackingWeights[index] = 1;
  }
  const reference: HumanoidReference = {
    jointPositions,
    jointVelocities: new Float64Array(HUMANOID_JOINT_NAMES.length),
    jointTrackingWeights,
    rootVelocity: target.rootVelocity ?? baseline.rootVelocity,
    rootYawVelocity: target.rootYawVelocity ?? baseline.rootYawVelocity,
    rootHeight: target.rootHeight ?? baseline.rootHeight,
    rootRoll: target.rootRoll ?? baseline.rootRoll,
    rootPitch: target.rootPitch ?? baseline.rootPitch
  };
  assertHumanoidReference(reference);
  return reference;
}

export function interpolateReference(
  start: HumanoidReference,
  end: HumanoidReference,
  progress: number,
  durationSeconds: number
): HumanoidReference {
  assertHumanoidReference(start);
  assertHumanoidReference(end);
  if (!Number.isFinite(progress)) {
    throw new Error("Humanoid reference interpolation progress must be finite");
  }
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    throw new Error("Humanoid reference interpolation duration must be positive");
  }
  const amount = smoothstep(clamp01(progress));
  const safeDuration = Math.max(durationSeconds, YAHMP_POLICY.controlDt);
  const derivative = 6 * clamp01(progress) * (1 - clamp01(progress)) / safeDuration;
  const jointPositions = new Float64Array(HUMANOID_JOINT_NAMES.length);
  const jointVelocities = new Float64Array(HUMANOID_JOINT_NAMES.length);
  const jointTrackingWeights = new Float64Array(HUMANOID_JOINT_NAMES.length);
  for (let index = 0; index < jointPositions.length; index += 1) {
    const delta = end.jointPositions[index]! - start.jointPositions[index]!;
    jointPositions[index] = start.jointPositions[index]! + delta * amount;
    jointVelocities[index] = delta * derivative;
    jointTrackingWeights[index] = mix(
      start.jointTrackingWeights[index]!,
      end.jointTrackingWeights[index]!,
      amount
    );
  }
  return {
    jointPositions,
    jointVelocities,
    jointTrackingWeights,
    rootVelocity: [
      mix(start.rootVelocity[0], end.rootVelocity[0], amount),
      mix(start.rootVelocity[1], end.rootVelocity[1], amount)
    ],
    rootYawVelocity: mix(start.rootYawVelocity, end.rootYawVelocity, amount),
    rootHeight: mix(start.rootHeight, end.rootHeight, amount),
    rootRoll: mix(start.rootRoll, end.rootRoll, amount),
    rootPitch: mix(start.rootPitch, end.rootPitch, amount)
  };
}

export function releaseReferenceTracking(
  reference: HumanoidReference
): HumanoidReference {
  assertHumanoidReference(reference);
  return {
    ...reference,
    jointPositions: reference.jointPositions.slice(),
    jointVelocities: reference.jointVelocities.slice(),
    jointTrackingWeights: new Float64Array(HUMANOID_JOINT_NAMES.length),
    rootVelocity: [...reference.rootVelocity]
  };
}

export function stationaryHumanoidReference(
  reference: HumanoidReference
): HumanoidReference {
  const released = releaseReferenceTracking(reference);
  return {
    ...released,
    rootVelocity: [0, 0],
    rootYawVelocity: 0
  };
}

export function assertHumanoidReference(reference: HumanoidReference): void {
  const jointCount = HUMANOID_JOINT_NAMES.length;
  if (reference.jointPositions.length !== jointCount
    || reference.jointVelocities.length !== jointCount
    || reference.jointTrackingWeights.length !== jointCount) {
    throw new Error("Humanoid reference has an invalid joint count");
  }
  if (![...reference.jointPositions, ...reference.jointVelocities].every(Number.isFinite)) {
    throw new Error("Humanoid reference joint state must be finite");
  }
  if (![...reference.jointTrackingWeights].every((weight) => (
    Number.isFinite(weight) && weight >= 0 && weight <= 1
  ))) {
    throw new Error("Humanoid reference tracking weights must be finite values from zero to one");
  }
  if (reference.rootVelocity.length !== 2
    || !reference.rootVelocity.every(Number.isFinite)
    || ![
      reference.rootYawVelocity,
      reference.rootHeight,
      reference.rootRoll,
      reference.rootPitch
    ].every(Number.isFinite)) {
    throw new Error("Humanoid reference root state must be finite");
  }
}

function smoothstep(value: number): number {
  return value * value * (3 - 2 * value);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function mix(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
}
