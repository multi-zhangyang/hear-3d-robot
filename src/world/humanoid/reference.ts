import {
  HUMANOID_JOINT_INDEX,
  HUMANOID_JOINT_NAMES,
  YAHMP_POLICY,
  type HumanoidJointName
} from "./model.js";

export interface HumanoidReference {
  jointPositions: Float64Array;
  jointVelocities: Float64Array;
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
  const jointPositions = baseline.jointPositions.slice();
  for (const [name, value] of Object.entries(target.joints ?? {}) as [HumanoidJointName, number][]) {
    const index = HUMANOID_JOINT_INDEX.get(name);
    if (index === undefined || !Number.isFinite(value)) {
      throw new Error(`Invalid humanoid joint target: ${name}`);
    }
    jointPositions[index] = value;
  }
  return {
    jointPositions,
    jointVelocities: new Float64Array(HUMANOID_JOINT_NAMES.length),
    rootVelocity: target.rootVelocity ?? baseline.rootVelocity,
    rootYawVelocity: target.rootYawVelocity ?? baseline.rootYawVelocity,
    rootHeight: target.rootHeight ?? baseline.rootHeight,
    rootRoll: target.rootRoll ?? baseline.rootRoll,
    rootPitch: target.rootPitch ?? baseline.rootPitch
  };
}

export function interpolateReference(
  start: HumanoidReference,
  end: HumanoidReference,
  progress: number,
  durationSeconds: number
): HumanoidReference {
  const amount = smoothstep(clamp01(progress));
  const safeDuration = Math.max(durationSeconds, YAHMP_POLICY.controlDt);
  const derivative = 6 * clamp01(progress) * (1 - clamp01(progress)) / safeDuration;
  const jointPositions = new Float64Array(HUMANOID_JOINT_NAMES.length);
  const jointVelocities = new Float64Array(HUMANOID_JOINT_NAMES.length);
  for (let index = 0; index < jointPositions.length; index += 1) {
    const delta = end.jointPositions[index]! - start.jointPositions[index]!;
    jointPositions[index] = start.jointPositions[index]! + delta * amount;
    jointVelocities[index] = delta * derivative;
  }
  return {
    jointPositions,
    jointVelocities,
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

function smoothstep(value: number): number {
  return value * value * (3 - 2 * value);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function mix(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
}
