import { HUMANOID_JOINT_NAMES, type HumanoidJointName } from "./model.js";
import type {
  MujocoActuatedJointBinding,
  MujocoJointBinding
} from "./mujoco-joints.js";

const FREE_JOINT_POSITION_COUNT = 7;
const FREE_JOINT_VELOCITY_COUNT = 6;

export interface LegacyG129DoFStateVectors {
  positions: ArrayLike<number>;
  velocities: ArrayLike<number>;
  controls: ArrayLike<number>;
  activations: ArrayLike<number>;
  accelerationWarmstart: ArrayLike<number>;
}

export interface CurrentG143DoFStateVectors {
  positions: Float64Array;
  velocities: Float64Array;
  controls: Float64Array;
  activations: Float64Array;
  accelerationWarmstart: Float64Array;
}

export function looksLikeLegacyG129DoFState(input: {
  source: LegacyG129DoFStateVectors;
  target: CurrentG143DoFStateVectors;
}): boolean {
  return input.source.positions.length === input.target.positions.length - 14
    || input.source.velocities.length === input.target.velocities.length - 14
    || input.source.controls.length === HUMANOID_JOINT_NAMES.length;
}

export function migrateLegacyG129DoFState(input: {
  source: LegacyG129DoFStateVectors;
  target: CurrentG143DoFStateVectors;
  bodyBindings: readonly MujocoActuatedJointBinding<HumanoidJointName>[];
  objectBindings: readonly MujocoJointBinding[];
}): CurrentG143DoFStateVectors {
  assertLegacyShape(input);
  const migrated: CurrentG143DoFStateVectors = {
    positions: input.target.positions.slice(),
    velocities: input.target.velocities.slice(),
    controls: input.target.controls.slice(),
    activations: finiteArray(input.source.activations, "activations"),
    accelerationWarmstart: input.target.accelerationWarmstart.slice()
  };
  copySegment(
    migrated.positions,
    0,
    input.source.positions,
    0,
    FREE_JOINT_POSITION_COUNT,
    "floating base positions"
  );
  copySegment(
    migrated.velocities,
    0,
    input.source.velocities,
    0,
    FREE_JOINT_VELOCITY_COUNT,
    "floating base velocities"
  );
  copySegment(
    migrated.accelerationWarmstart,
    0,
    input.source.accelerationWarmstart,
    0,
    FREE_JOINT_VELOCITY_COUNT,
    "floating base acceleration warmstart"
  );
  input.bodyBindings.forEach((binding) => {
    const sourceIndex = legacyBodyJointIndex(binding.name);
    migrated.positions[binding.positionAddress] = finiteValue(
      input.source.positions[FREE_JOINT_POSITION_COUNT + sourceIndex],
      `${binding.name} position`
    );
    migrated.velocities[binding.velocityAddress] = finiteValue(
      input.source.velocities[FREE_JOINT_VELOCITY_COUNT + sourceIndex],
      `${binding.name} velocity`
    );
    migrated.accelerationWarmstart[binding.velocityAddress] = finiteValue(
      input.source.accelerationWarmstart[FREE_JOINT_VELOCITY_COUNT + sourceIndex],
      `${binding.name} acceleration warmstart`
    );
    migrated.controls[binding.actuatorId] = finiteValue(
      input.source.controls[sourceIndex],
      `${binding.name} control`
    );
  });
  input.objectBindings.forEach((binding) => {
    const objectIndex = legacyObjectIndex(binding.name, input.objectBindings.length);
    const sourcePositionAddress = FREE_JOINT_POSITION_COUNT
      + HUMANOID_JOINT_NAMES.length
      + objectIndex * FREE_JOINT_POSITION_COUNT;
    const sourceVelocityAddress = FREE_JOINT_VELOCITY_COUNT
      + HUMANOID_JOINT_NAMES.length
      + objectIndex * FREE_JOINT_VELOCITY_COUNT;
    copySegment(
      migrated.positions,
      binding.positionAddress,
      input.source.positions,
      sourcePositionAddress,
      FREE_JOINT_POSITION_COUNT,
      `${binding.name} positions`
    );
    copySegment(
      migrated.velocities,
      binding.velocityAddress,
      input.source.velocities,
      sourceVelocityAddress,
      FREE_JOINT_VELOCITY_COUNT,
      `${binding.name} velocities`
    );
    copySegment(
      migrated.accelerationWarmstart,
      binding.velocityAddress,
      input.source.accelerationWarmstart,
      sourceVelocityAddress,
      FREE_JOINT_VELOCITY_COUNT,
      `${binding.name} acceleration warmstart`
    );
  });
  return migrated;
}

function assertLegacyShape(input: {
  source: LegacyG129DoFStateVectors;
  target: CurrentG143DoFStateVectors;
  bodyBindings: readonly MujocoActuatedJointBinding<HumanoidJointName>[];
  objectBindings: readonly MujocoJointBinding[];
}): void {
  const objectCount = input.objectBindings.length;
  const expectedPositions = FREE_JOINT_POSITION_COUNT
    + HUMANOID_JOINT_NAMES.length
    + objectCount * FREE_JOINT_POSITION_COUNT;
  const expectedVelocities = FREE_JOINT_VELOCITY_COUNT
    + HUMANOID_JOINT_NAMES.length
    + objectCount * FREE_JOINT_VELOCITY_COUNT;
  const bodyNames = input.bodyBindings.map((binding) => binding.name);
  const objectIndexes = input.objectBindings.map((binding) => (
    legacyObjectIndex(binding.name, objectCount)
  ));
  if (input.bodyBindings.length !== HUMANOID_JOINT_NAMES.length
    || new Set(bodyNames).size !== HUMANOID_JOINT_NAMES.length
    || HUMANOID_JOINT_NAMES.some((name) => !bodyNames.includes(name))
    || new Set(objectIndexes).size !== objectCount
    || input.source.positions.length !== expectedPositions
    || input.source.velocities.length !== expectedVelocities
    || input.source.accelerationWarmstart.length !== expectedVelocities
    || input.source.controls.length !== HUMANOID_JOINT_NAMES.length
    || input.source.activations.length !== input.target.activations.length) {
    throw new Error("Legacy G1 29DoF state has an invalid named vector layout");
  }
}

function legacyBodyJointIndex(name: HumanoidJointName): number {
  const index = HUMANOID_JOINT_NAMES.indexOf(name);
  if (index < 0) {
    throw new Error(`Legacy G1 body joint is not part of the 29DoF layout: ${name}`);
  }
  return index;
}

function legacyObjectIndex(name: string, objectCount: number): number {
  const match = /^world-object-joint-(\d+)$/.exec(name);
  const index = match ? Number(match[1]) : Number.NaN;
  if (!Number.isSafeInteger(index) || index < 0 || index >= objectCount) {
    throw new Error(`Legacy G1 object joint has an invalid named layout: ${name}`);
  }
  return index;
}

function finiteArray(values: ArrayLike<number>, label: string): Float64Array {
  return Float64Array.from(values, (value, index) => finiteValue(value, `${label}[${index}]`));
}

function copySegment(
  target: Float64Array,
  targetOffset: number,
  source: ArrayLike<number>,
  sourceOffset: number,
  length: number,
  label: string
): void {
  if (targetOffset < 0 || targetOffset + length > target.length
    || sourceOffset < 0 || sourceOffset + length > source.length) {
    throw new Error(`Legacy G1 ${label} exceed the named vector layout`);
  }
  for (let index = 0; index < length; index += 1) {
    target[targetOffset + index] = finiteValue(
      source[sourceOffset + index],
      `${label}[${index}]`
    );
  }
}

function finiteValue(value: number | undefined, label: string): number {
  if (value === undefined || !Number.isFinite(value)) {
    throw new Error(`Legacy G1 ${label} must be finite`);
  }
  return value;
}
