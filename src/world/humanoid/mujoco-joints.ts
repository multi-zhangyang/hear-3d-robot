import type {
  MujocoModel,
  MujocoModule
} from "./mujoco-runtime.js";

export interface MujocoJointBinding<Name extends string = string> {
  name: Name;
  jointId: number;
  positionAddress: number;
  velocityAddress: number;
}

export interface MujocoActuatedJointBinding<Name extends string = string>
  extends MujocoJointBinding<Name> {
  actuatorId: number;
}

export function resolveMujocoJoints<const Name extends string>(
  runtime: MujocoModule,
  model: MujocoModel,
  names: readonly Name[]
): MujocoJointBinding<Name>[] {
  const bindings = names.map((name) => {
    const jointId = runtime.mj_name2id(
      model,
      runtime.mjtObj.mjOBJ_JOINT.value,
      name
    );
    if (jointId < 0) throw new Error(`MuJoCo joint is missing: ${name}`);
    const positionAddress = requiredInteger(model.jnt_qposadr[jointId], `${name} qpos`);
    const velocityAddress = requiredInteger(model.jnt_dofadr[jointId], `${name} qvel`);
    return { name, jointId, positionAddress, velocityAddress };
  });
  if (new Set(bindings.map((binding) => binding.jointId)).size !== bindings.length) {
    throw new Error("MuJoCo humanoid joint mapping contains duplicate identities");
  }
  return bindings;
}

export function resolveMujocoActuatedJoints<const Name extends string>(
  runtime: MujocoModule,
  model: MujocoModel,
  names: readonly Name[]
): MujocoActuatedJointBinding<Name>[] {
  const bindings = resolveMujocoJoints(runtime, model, names).map((binding) => {
    const { name, jointId, positionAddress, velocityAddress } = binding;
    const actuatorId = actuatorForJoint(model, jointId);
    return { name, jointId, positionAddress, velocityAddress, actuatorId };
  });
  if (new Set(bindings.map((binding) => binding.actuatorId)).size !== bindings.length) {
    throw new Error("MuJoCo humanoid joint mapping contains duplicate identities");
  }
  return bindings;
}

export function configureTorqueControlledBodyActuators(
  model: MujocoModel,
  bindings: readonly MujocoActuatedJointBinding[]
): void {
  const gainWidth = actuatorParameterWidth(
    model.actuator_gainprm,
    model.nu,
    "gain"
  );
  const biasWidth = actuatorParameterWidth(
    model.actuator_biasprm,
    model.nu,
    "bias"
  );
  for (const binding of bindings) {
    fillParameters(model.actuator_gainprm, binding.actuatorId, gainWidth, 0);
    model.actuator_gainprm[binding.actuatorId * gainWidth] = 1;
    fillParameters(model.actuator_biasprm, binding.actuatorId, biasWidth, 0);
    const rangeOffset = binding.jointId * 2;
    const minimum = requiredFinite(
      model.jnt_actfrcrange[rangeOffset],
      `${binding.name} minimum actuator force`
    );
    const maximum = requiredFinite(
      model.jnt_actfrcrange[rangeOffset + 1],
      `${binding.name} maximum actuator force`
    );
    if (minimum >= maximum || minimum >= 0 || maximum <= 0) {
      throw new Error(`MuJoCo body joint has an invalid effort range: ${binding.name}`);
    }
    const controlOffset = binding.actuatorId * 2;
    model.actuator_ctrlrange[controlOffset] = minimum;
    model.actuator_ctrlrange[controlOffset + 1] = maximum;
  }
}

function actuatorForJoint(model: MujocoModel, jointId: number): number {
  for (let actuatorId = 0; actuatorId < model.nu; actuatorId += 1) {
    if (model.actuator_trnid[actuatorId * 2] === jointId) return actuatorId;
  }
  throw new Error(`No actuator found for MuJoCo joint ${jointId}`);
}

function actuatorParameterWidth(
  values: ArrayLike<number>,
  actuatorCount: number,
  label: string
): number {
  const width = values.length / actuatorCount;
  if (!Number.isSafeInteger(width) || width <= 0) {
    throw new Error(`MuJoCo actuator ${label} parameters have an invalid shape`);
  }
  return width;
}

function fillParameters(
  values: { [index: number]: number },
  actuatorId: number,
  width: number,
  value: number
): void {
  const start = actuatorId * width;
  for (let index = start; index < start + width; index += 1) values[index] = value;
}

function requiredFinite(value: number | undefined, label: string): number {
  if (value === undefined || !Number.isFinite(value)) {
    throw new Error(`MuJoCo ${label} must be finite`);
  }
  return value;
}

function requiredInteger(value: number | undefined, label: string): number {
  if (value === undefined || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`MuJoCo ${label} must be a nonnegative integer`);
  }
  return value;
}
