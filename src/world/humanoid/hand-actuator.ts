import {
  G1_HAND_JOINT_LIMITS,
  G1_HAND_JOINT_NAMES,
  type G1HandJointName
} from "./morphology.js";
import {
  resolveMujocoActuatedJoints,
  type MujocoActuatedJointBinding
} from "./mujoco-joints.js";
import type {
  MujocoData,
  MujocoModel,
  MujocoModule
} from "./mujoco-runtime.js";

interface G1HandJointSnapshot {
  position: number;
  velocity: number;
  target: number;
  minimum: number;
  maximum: number;
  stiffnessNewtonMetersPerRadian: number;
  dampingNewtonMeterSecondsPerRadian: number;
  appliedNewtonMeters: number;
  minimumNewtonMeters: number;
  maximumNewtonMeters: number;
  saturated: boolean;
}

export interface G1HandActuatorSnapshot {
  controller: typeof G1_HAND_CONTROLLER_DESCRIPTOR;
  joints: Record<G1HandJointName, G1HandJointSnapshot>;
}

const G1_HAND_CONTROLLER_DESCRIPTOR = {
  protocol: "g1-hand-controller-v1",
  implementation: "mujoco_continuous_position_pd",
  actuation: "joint_position_pd",
  jointCount: G1_HAND_JOINT_NAMES.length
} as const;

export class G1HandActuator {
  readonly #model: MujocoModel;
  readonly #data: MujocoData;
  readonly #bindings: readonly MujocoActuatedJointBinding<G1HandJointName>[];
  readonly #gainWidth: number;
  readonly #biasWidth: number;

  constructor(runtime: MujocoModule, model: MujocoModel, data: MujocoData) {
    this.#model = model;
    this.#data = data;
    this.#bindings = resolveMujocoActuatedJoints(runtime, model, G1_HAND_JOINT_NAMES);
    this.#gainWidth = actuatorParameterWidth(model.actuator_gainprm, model.nu, "gain");
    this.#biasWidth = actuatorParameterWidth(model.actuator_biasprm, model.nu, "bias");
    this.#assertContinuousPositionPdContract();
  }

  holdCurrentPositions(): void {
    for (const binding of this.#bindings) {
      this.#data.ctrl[binding.actuatorId] = this.#data.qpos[binding.positionAddress]!;
    }
  }

  setTargets(targets: Readonly<Partial<Record<G1HandJointName, number>>>): void {
    const entries = Object.entries(targets);
    if (entries.length === 0) throw new Error("G1 hand target command cannot be empty");
    const resolved = entries.map(([name, value]) => {
      const binding = this.#bindings.find((candidate) => candidate.name === name);
      if (!binding) throw new Error(`Unknown G1 hand joint target: ${name}`);
      const target = requiredFinite(value, `${name} target`);
      const [minimum, maximum] = this.#jointRange(binding);
      if (target < minimum || target > maximum) {
        throw new Error(`G1 hand target exceeds ${name} limits`);
      }
      return { binding, target };
    });
    for (const { binding, target } of resolved) {
      this.#data.ctrl[binding.actuatorId] = target;
    }
  }

  validateCurrentTargets(): void {
    for (const binding of this.#bindings) {
      const target = requiredFinite(
        this.#data.ctrl[binding.actuatorId],
        `${binding.name} restored target`
      );
      const [minimum, maximum] = this.#jointRange(binding);
      if (target < minimum || target > maximum) {
        throw new Error(`Restored G1 hand target exceeds ${binding.name} limits`);
      }
    }
  }

  snapshot(): G1HandActuatorSnapshot {
    const joints = Object.fromEntries(this.#bindings.map((binding) => {
      const [minimum, maximum] = this.#jointRange(binding);
      const [minimumEffort, maximumEffort] = this.#jointEffortRange(binding);
      const gainOffset = binding.actuatorId * this.#gainWidth;
      const biasOffset = binding.actuatorId * this.#biasWidth;
      const force = requiredFinite(
        this.#data.actuator_force[binding.actuatorId],
        `${binding.name} applied effort`
      );
      return [binding.name, {
        position: requiredFinite(
          this.#data.qpos[binding.positionAddress],
          `${binding.name} position`
        ),
        velocity: requiredFinite(
          this.#data.qvel[binding.velocityAddress],
          `${binding.name} velocity`
        ),
        target: requiredFinite(
          this.#data.ctrl[binding.actuatorId],
          `${binding.name} target`
        ),
        minimum,
        maximum,
        stiffnessNewtonMetersPerRadian: requiredFinite(
          this.#model.actuator_gainprm[gainOffset],
          `${binding.name} stiffness`
        ),
        dampingNewtonMeterSecondsPerRadian: -requiredFinite(
          this.#model.actuator_biasprm[biasOffset + 2],
          `${binding.name} damping`
        ),
        appliedNewtonMeters: force,
        minimumNewtonMeters: minimumEffort,
        maximumNewtonMeters: maximumEffort,
        saturated: force <= minimumEffort + 1e-6 || force >= maximumEffort - 1e-6
      }];
    })) as Record<G1HandJointName, G1HandJointSnapshot>;
    return { controller: G1_HAND_CONTROLLER_DESCRIPTOR, joints };
  }

  bindings(): readonly MujocoActuatedJointBinding<G1HandJointName>[] {
    return this.#bindings.map((binding) => ({ ...binding }));
  }

  #assertContinuousPositionPdContract(): void {
    if (this.#bindings.length !== G1_HAND_CONTROLLER_DESCRIPTOR.jointCount) {
      throw new Error("G1 hand actuator mapping does not contain 14 joints");
    }
    for (const binding of this.#bindings) {
      const gainOffset = binding.actuatorId * this.#gainWidth;
      const biasOffset = binding.actuatorId * this.#biasWidth;
      const stiffness = requiredFinite(
        this.#model.actuator_gainprm[gainOffset],
        `${binding.name} stiffness`
      );
      const positionBias = requiredFinite(
        this.#model.actuator_biasprm[biasOffset + 1],
        `${binding.name} position bias`
      );
      const velocityBias = requiredFinite(
        this.#model.actuator_biasprm[biasOffset + 2],
        `${binding.name} velocity bias`
      );
      if (stiffness <= 0
        || Math.abs(positionBias + stiffness) > 1e-6
        || velocityBias >= 0) {
        throw new Error(`G1 hand joint is not a continuous position PD actuator: ${binding.name}`);
      }
      const [minimum, maximum] = this.#jointRange(binding);
      const controlOffset = binding.actuatorId * 2;
      if (Math.abs(this.#model.actuator_ctrlrange[controlOffset]! - minimum) > 1e-6
        || Math.abs(this.#model.actuator_ctrlrange[controlOffset + 1]! - maximum) > 1e-6) {
        throw new Error(`G1 hand actuator target range does not match ${binding.name}`);
      }
    }
  }

  #jointRange(binding: MujocoActuatedJointBinding): readonly [number, number] {
    const offset = binding.jointId * 2;
    const minimum = requiredFinite(
      this.#model.jnt_range[offset],
      `${binding.name} minimum position`
    );
    const maximum = requiredFinite(
      this.#model.jnt_range[offset + 1],
      `${binding.name} maximum position`
    );
    if (minimum > maximum) throw new Error(`G1 hand joint has an invalid range: ${binding.name}`);
    const expected = G1_HAND_JOINT_LIMITS[binding.name as G1HandJointName];
    if (Math.abs(minimum - expected[0]) > 1e-6
      || Math.abs(maximum - expected[1]) > 1e-6) {
      throw new Error(`G1 hand morphology limits do not match ${binding.name}`);
    }
    return [minimum, maximum];
  }

  #jointEffortRange(binding: MujocoActuatedJointBinding): readonly [number, number] {
    const offset = binding.jointId * 2;
    const minimum = requiredFinite(
      this.#model.jnt_actfrcrange[offset],
      `${binding.name} minimum effort`
    );
    const maximum = requiredFinite(
      this.#model.jnt_actfrcrange[offset + 1],
      `${binding.name} maximum effort`
    );
    if (minimum >= maximum) {
      throw new Error(`G1 hand joint has an invalid effort range: ${binding.name}`);
    }
    return [minimum, maximum];
  }
}

function actuatorParameterWidth(
  values: ArrayLike<number>,
  actuatorCount: number,
  label: string
): number {
  const width = values.length / actuatorCount;
  if (!Number.isSafeInteger(width) || width <= 2) {
    throw new Error(`MuJoCo actuator ${label} parameters have an invalid shape`);
  }
  return width;
}

function requiredFinite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`G1 hand ${label} must be finite`);
  }
  return value;
}
