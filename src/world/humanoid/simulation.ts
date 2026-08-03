import type { Quaternion, Vec3 } from "../../domain/schema.js";
import {
  add,
  inverseQuaternion,
  rotateVector,
  scale,
  subtract,
  vectorLength
} from "../geometry.js";
import {
  HUMANOID_HEAD_SENSOR,
  HUMANOID_BODY_NAMES,
  HUMANOID_JOINT_NAMES,
  YAHMP_POLICY,
  type HumanoidBodyName,
  type HumanoidJointName
} from "./model.js";
import {
  createHumanoidScenePath,
  humanoidModelPath,
  loadHumanoidMujoco,
  removeHumanoidScene,
  type HumanoidSceneObject,
  type HumanoidSceneSolid,
  type MujocoModule
} from "./mujoco-runtime.js";
import {
  neutralHumanoidReference,
  type HumanoidReference
} from "./reference.js";
import {
  YahmpController
} from "./yahmp-controller.js";
import {
  humanoidEndEffectorJointIndexes,
  type HumanoidEndEffectorBody
} from "./task-space-targets.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

type MujocoModel = InstanceType<MujocoModule["MjModel"]>;
type MujocoData = InstanceType<MujocoModule["MjData"]>;

interface HumanoidLinkSnapshot {
  position: Vec3;
  rotation: Quaternion;
  linearVelocity: Vec3;
  angularVelocity: Vec3;
}

export interface HumanoidSpawn {
  position: Vec3;
  yaw: number;
}

export interface HumanoidSimulationOptions {
  spawn?: HumanoidSpawn;
  solids?: readonly HumanoidSceneSolid[];
  objects?: readonly HumanoidSceneObject[];
  controllerFactory?: () => Promise<HumanoidWholeBodyController>;
}

interface HumanoidFootContactSnapshot {
  touching: boolean;
  contactCount: number;
  normalForce: number;
  points: Vec3[];
}

export interface HumanoidContactSnapshot {
  position: Vec3;
  normal: Vec3;
  normalForce: number;
  firstBody: HumanoidBodyName | null;
  secondBody: HumanoidBodyName | null;
  firstObject: string | null;
  secondObject: string | null;
}

export interface HumanoidObjectSnapshot extends HumanoidLinkSnapshot {
  id: string;
}

export interface HumanoidObjectSensorSnapshot {
  sensor: {
    position: Vec3;
    rotation: Quaternion;
    maximumRange: number;
    horizontalFieldOfView: number;
    verticalFieldOfView: number;
  };
  objects: Record<string, HumanoidObjectSnapshot>;
}

interface HumanoidBalanceSnapshot {
  centerOfMass: Vec3;
  support: "none" | "left" | "right" | "double";
  supportMargin: number | null;
  upright: number;
}

export interface HumanoidSimulationState {
  time: number;
  positions: Float64Array;
  velocities: Float64Array;
  controls: Float64Array;
  activations: Float64Array;
  accelerationWarmstart: Float64Array;
  controller: HumanoidControllerState;
}

export interface HumanoidSimulationSnapshot {
  simulatedTime: number;
  controller: HumanoidWholeBodyController["descriptor"];
  rootPosition: Vec3;
  rootRotation: Quaternion;
  joints: Record<HumanoidJointName, {
    position: number;
    velocity: number;
    minimum: number;
    maximum: number;
  }>;
  links: Record<HumanoidBodyName, HumanoidLinkSnapshot>;
  objects: Record<string, HumanoidObjectSnapshot>;
  contactCount: number;
  contacts: HumanoidContactSnapshot[];
  feet: {
    left: HumanoidFootContactSnapshot;
    right: HumanoidFootContactSnapshot;
  };
  balance: HumanoidBalanceSnapshot;
  nonFootEnvironmentContacts: HumanoidBodyName[];
  fallen: boolean;
}

export interface HumanoidEndEffectorTarget {
  body: HumanoidEndEffectorBody;
  position: Vec3;
  frame: "world" | "pelvis";
  tolerance: number;
}

export interface HumanoidTaskSpaceSolution {
  reference: HumanoidReference;
  residuals: Array<{
    body: HumanoidEndEffectorTarget["body"];
    target: Vec3;
    achieved: Vec3;
    error: number;
  }>;
}

export class HumanoidSimulation {
  readonly #runtime: MujocoModule;
  readonly #model: MujocoModel;
  readonly #data: MujocoData;
  readonly #controller: HumanoidWholeBodyController;
  readonly #jointIds: number[];
  readonly #jointPositionAddresses: number[];
  readonly #jointVelocityAddresses: number[];
  readonly #actuatorIds: number[];
  readonly #bodyIds: number[];
  readonly #bodyNamesById = new Map<number, HumanoidBodyName>();
  readonly #objectNamesByBodyId = new Map<number, string>();
  readonly #pelvisBodyId: number;
  readonly #headBodyId: number;
  readonly #leftFootBodyId: number;
  readonly #rightFootBodyId: number;
  readonly #spawn: HumanoidSpawn;

  static async create(options: HumanoidSimulationOptions = {}): Promise<HumanoidSimulation> {
    const [runtime, controller] = await Promise.all([
      loadHumanoidMujoco(),
      options.controllerFactory
        ? options.controllerFactory()
        : YahmpController.create()
    ]);
    try {
      assertControllerTiming(controller);
      const generatedPath = options.solids || options.objects
        ? createHumanoidScenePath(runtime, options.solids ?? [], options.objects)
        : undefined;
      let model: MujocoModel;
      try {
        model = runtime.MjModel.from_xml_path(generatedPath ?? humanoidModelPath());
      } finally {
        if (generatedPath) removeHumanoidScene(runtime, generatedPath);
      }
      model.opt.timestep = controller.descriptor.physicsStepSeconds;
      const data = new runtime.MjData(model);
      try {
        const instance = new HumanoidSimulation(runtime, model, data, controller, options);
        instance.#initialize();
        return instance;
      } catch (error) {
        data.delete();
        model.delete();
        throw error;
      }
    } catch (error) {
      await controller.dispose();
      throw error;
    }
  }

  private constructor(
    runtime: MujocoModule,
    model: MujocoModel,
    data: MujocoData,
    controller: HumanoidWholeBodyController,
    options: HumanoidSimulationOptions
  ) {
    this.#runtime = runtime;
    this.#model = model;
    this.#data = data;
    this.#controller = controller;
    this.#jointIds = HUMANOID_JOINT_NAMES.map((name) => this.#id("joint", name));
    this.#jointPositionAddresses = this.#jointIds.map((id) => model.jnt_qposadr[id]!);
    this.#jointVelocityAddresses = this.#jointIds.map((id) => model.jnt_dofadr[id]!);
    this.#actuatorIds = this.#jointIds.map((jointId) => this.#actuatorForJoint(jointId));
    this.#bodyIds = HUMANOID_BODY_NAMES.map((name) => this.#id("body", name));
    HUMANOID_BODY_NAMES.forEach((name, index) => {
      this.#bodyNamesById.set(this.#bodyIds[index]!, name);
    });
    (options.objects ?? []).forEach((object, index) => {
      this.#objectNamesByBodyId.set(this.#id("body", `world-object-${index}`), object.id);
    });
    this.#pelvisBodyId = this.#id("body", "pelvis");
    this.#headBodyId = this.#id("body", "head_link");
    this.#leftFootBodyId = this.#id("body", "left_ankle_roll_link");
    this.#rightFootBodyId = this.#id("body", "right_ankle_roll_link");
    this.#spawn = options.spawn ?? {
      position: { x: 0, y: 0, z: 0 },
      yaw: 0
    };
    assertSpawn(this.#spawn);
  }

  async step(reference: HumanoidReference): Promise<HumanoidSimulationSnapshot> {
    this.#validateReference(reference);
    const command = await this.#controller.infer(this.#policyState(), reference);
    const substeps = Math.round(
      this.#controller.descriptor.controlStepSeconds
      / this.#controller.descriptor.physicsStepSeconds
    );
    for (let step = 0; step < substeps; step += 1) {
      this.#applyCommand(command);
      this.#runtime.mj_step(this.#model, this.#data);
    }
    this.#controller.advanceHistory(this.#policyState(), reference);
    return this.snapshot();
  }

  controllerDescriptor(): HumanoidControllerDescriptor {
    return { ...this.#controller.descriptor };
  }

  snapshot(): HumanoidSimulationSnapshot {
    const rootQuaternion = this.#rootQuaternion();
    const links = Object.fromEntries(HUMANOID_BODY_NAMES.map((name, index) => {
      const bodyId = this.#bodyIds[index]!;
      const positionOffset = bodyId * 3;
      const rotationOffset = bodyId * 4;
      return [name, {
        position: worldVector(this.#data.xpos, positionOffset),
        rotation: worldQuaternion(this.#data.xquat, rotationOffset),
        linearVelocity: worldVector(this.#data.cvel, bodyId * 6 + 3),
        angularVelocity: worldVector(this.#data.cvel, bodyId * 6)
      }];
    })) as HumanoidSimulationSnapshot["links"];
    const joints = Object.fromEntries(HUMANOID_JOINT_NAMES.map((name, index) => {
      const rangeOffset = this.#jointIds[index]! * 2;
      return [name, {
        position: this.#data.qpos[this.#jointPositionAddresses[index]!]!,
        velocity: this.#data.qvel[this.#jointVelocityAddresses[index]!]!,
        minimum: this.#model.jnt_range[rangeOffset]!,
        maximum: this.#model.jnt_range[rangeOffset + 1]!
      }];
    })) as HumanoidSimulationSnapshot["joints"];
    const objects = Object.fromEntries([...this.#objectNamesByBodyId].map(([bodyId, id]) => {
      const positionOffset = bodyId * 3;
      const rotationOffset = bodyId * 4;
      return [id, {
        id,
        position: worldVector(this.#data.xpos, positionOffset),
        rotation: worldQuaternion(this.#data.xquat, rotationOffset),
        linearVelocity: worldVector(this.#data.cvel, bodyId * 6 + 3),
        angularVelocity: worldVector(this.#data.cvel, bodyId * 6)
      }];
    })) as HumanoidSimulationSnapshot["objects"];
    const up = rotate(rootQuaternion, [0, 0, 1]);
    const rootHeight = this.#data.qpos[2]!;
    const contacts = this.#contacts();
    const left = footContact(contacts, this.#bodyNamesById.get(this.#leftFootBodyId)!);
    const right = footContact(contacts, this.#bodyNamesById.get(this.#rightFootBodyId)!);
    const centerOfMass = worldVector(this.#data.subtree_com, this.#pelvisBodyId * 3);
    const supportPoints = [...left.points, ...right.points];
    const nonFootEnvironmentContacts = this.#unsafeEnvironmentContacts(contacts);
    return {
      simulatedTime: this.#data.time,
      controller: { ...this.#controller.descriptor },
      rootPosition: worldVector(this.#data.qpos, 0),
      rootRotation: worldQuaternion(this.#data.qpos, 3),
      joints,
      links,
      objects,
      contactCount: this.#data.ncon,
      contacts,
      feet: { left, right },
      balance: {
        centerOfMass,
        support: left.touching && right.touching
          ? "double"
          : left.touching ? "left" : right.touching ? "right" : "none",
        supportMargin: supportMargin(centerOfMass, supportPoints),
        upright: up[2]
      },
      nonFootEnvironmentContacts,
      fallen: rootHeight < 0.45 || up[2] < 0.55
    };
  }

  senseObjects(maximumRange: number): HumanoidObjectSensorSnapshot {
    if (!Number.isFinite(maximumRange) || maximumRange <= 0) {
      throw new Error("Humanoid sensor range must be positive");
    }
    const snapshot = this.snapshot();
    const head = snapshot.links.head_link;
    const origin = add(
      head.position,
      rotateVector(head.rotation, HUMANOID_HEAD_SENSOR.localPosition)
    );
    const visible = Object.fromEntries(Object.entries(snapshot.objects).filter(([id, object]) => {
      const delta = subtract(object.position, origin);
      const distance = vectorLength(delta);
      if (distance <= 0.001 || distance > maximumRange) return false;
      const direction = scale(delta, 1 / distance);
      const local = rotateVector(inverseQuaternion(head.rotation), direction);
      const horizontal = Math.atan2(local.x, local.z);
      const vertical = Math.atan2(local.y, Math.hypot(local.x, local.z));
      if (Math.abs(horizontal) > HUMANOID_HEAD_SENSOR.horizontalFieldOfView / 2
        || Math.abs(vertical) > HUMANOID_HEAD_SENSOR.verticalFieldOfView / 2) {
        return false;
      }
      return this.#rayObject(origin, direction, distance + 0.02) === id;
    }));
    return {
      sensor: {
        position: origin,
        rotation: { ...head.rotation },
        maximumRange,
        horizontalFieldOfView: HUMANOID_HEAD_SENSOR.horizontalFieldOfView,
        verticalFieldOfView: HUMANOID_HEAD_SENSOR.verticalFieldOfView
      },
      objects: visible
    };
  }

  captureState(): HumanoidSimulationState {
    return {
      time: this.#data.time,
      positions: Float64Array.from(this.#data.qpos),
      velocities: Float64Array.from(this.#data.qvel),
      controls: Float64Array.from(this.#data.ctrl),
      activations: Float64Array.from(this.#data.act),
      accelerationWarmstart: Float64Array.from(this.#data.qacc_warmstart),
      controller: this.#controller.captureState()
    };
  }

  restoreState(state: HumanoidSimulationState): void {
    if (state.controller.protocol !== "humanoid-controller-state-v1"
      || state.controller.implementation !== this.#controller.descriptor.implementation) {
      throw new Error("Humanoid controller state does not match the active implementation");
    }
    copyState(this.#data.qpos, state.positions, "positions");
    copyState(this.#data.qvel, state.velocities, "velocities");
    copyState(this.#data.ctrl, state.controls, "controls");
    copyState(this.#data.act, state.activations, "activations");
    copyState(
      this.#data.qacc_warmstart,
      state.accelerationWarmstart,
      "acceleration warmstart"
    );
    this.#data.time = state.time;
    this.#controller.restoreState(state.controller);
    this.#runtime.mj_forward(this.#model, this.#data);
  }

  solveEndEffectorTargets(
    reference: HumanoidReference,
    targets: readonly HumanoidEndEffectorTarget[]
  ): HumanoidTaskSpaceSolution {
    if (targets.length === 0) return { reference, residuals: [] };
    if (new Set(targets.map((target) => target.body)).size !== targets.length) {
      throw new Error("A task-space keyframe cannot repeat an end effector");
    }
    const saved = this.captureState();
    try {
      this.#setReferenceConfiguration(reference);
      const pelvis = this.#linkPosition("pelvis");
      const pelvisRotation = worldQuaternion(this.#data.xquat, this.#pelvisBodyId * 4);
      const resolved = targets.map((target) => ({
        ...target,
        position: target.frame === "world"
          ? { ...target.position }
          : add(pelvis, rotateVector(pelvisRotation, target.position))
      }));
      const jointIndexes = [...new Set(resolved.flatMap((target) => (
        humanoidEndEffectorJointIndexes(target.body)
      )))];
      const epsilon = 1e-4;
      for (let iteration = 0; iteration < 96; iteration += 1) {
        const positions = resolved.map((target) => this.#linkPosition(target.body));
        const error = resolved.flatMap((target, index) => vectorDifference(
          target.position,
          positions[index]!
        ));
        if (resolved.every((target, index) => (
          vectorDistance(target.position, positions[index]!) <= target.tolerance
        ))) {
          return this.#taskSpaceSolution(reference, resolved);
        }
        const jacobian = Array.from({ length: error.length }, () => (
          new Array<number>(jointIndexes.length).fill(0)
        ));
        for (let column = 0; column < jointIndexes.length; column += 1) {
          const jointIndex = jointIndexes[column]!;
          const address = this.#jointPositionAddresses[jointIndex]!;
          const original = this.#data.qpos[address]!;
          this.#data.qpos[address] = original + epsilon;
          this.#runtime.mj_forward(this.#model, this.#data);
          resolved.forEach((target, targetIndex) => {
            const perturbed = this.#linkPosition(target.body);
            const current = positions[targetIndex]!;
            jacobian[targetIndex * 3]![column] = (perturbed.x - current.x) / epsilon;
            jacobian[targetIndex * 3 + 1]![column] = (perturbed.y - current.y) / epsilon;
            jacobian[targetIndex * 3 + 2]![column] = (perturbed.z - current.z) / epsilon;
          });
          this.#data.qpos[address] = original;
          this.#runtime.mj_forward(this.#model, this.#data);
        }
        const delta = dampedLeastSquares(jacobian, error, 0.018);
        for (let column = 0; column < jointIndexes.length; column += 1) {
          const jointIndex = jointIndexes[column]!;
          const jointId = this.#jointIds[jointIndex]!;
          const rangeOffset = jointId * 2;
          const address = this.#jointPositionAddresses[jointIndex]!;
          this.#data.qpos[address] = clamp(
            this.#data.qpos[address]! + clamp(delta[column] ?? 0, -0.14, 0.14),
            this.#model.jnt_range[rangeOffset]!,
            this.#model.jnt_range[rangeOffset + 1]!
          );
        }
        this.#runtime.mj_forward(this.#model, this.#data);
      }
      const residuals = resolved.map((target) => ({
        body: target.body,
        target: { ...target.position },
        achieved: this.#linkPosition(target.body),
        error: vectorDistance(target.position, this.#linkPosition(target.body))
      }));
      throw new Error(`Task-space IK did not converge: ${residuals.map((entry) => (
        `${entry.body}=${entry.error.toFixed(3)}m`
      )).join(", ")}`);
    } finally {
      this.restoreState(saved);
    }
  }

  async dispose(): Promise<void> {
    await this.#controller.dispose();
    this.#data.delete();
    this.#model.delete();
  }

  #initialize(): void {
    this.#data.qvel.fill(0);
    this.#data.ctrl.fill(0);
    this.#data.qpos[0] = this.#spawn.position.z;
    this.#data.qpos[1] = this.#spawn.position.x;
    this.#data.qpos[2] = this.#spawn.position.y + 0.793;
    this.#data.qpos[3] = Math.cos(this.#spawn.yaw / 2);
    this.#data.qpos[6] = Math.sin(this.#spawn.yaw / 2);
    for (let index = 0; index < HUMANOID_JOINT_NAMES.length; index += 1) {
      this.#data.qpos[this.#jointPositionAddresses[index]!] = (
        YAHMP_POLICY.defaultJointPositions[index]!
      );
    }
    this.#runtime.mj_forward(this.#model, this.#data);
    const neutral = neutralHumanoidReference();
    this.#controller.reset(this.#policyState(), neutral);
  }

  #setReferenceConfiguration(reference: HumanoidReference): void {
    this.#validateReference(reference);
    for (let index = 0; index < HUMANOID_JOINT_NAMES.length; index += 1) {
      this.#data.qpos[this.#jointPositionAddresses[index]!] = reference.jointPositions[index]!;
      this.#data.qvel[this.#jointVelocityAddresses[index]!] = 0;
    }
    this.#runtime.mj_forward(this.#model, this.#data);
  }

  #linkPosition(body: HumanoidBodyName): Vec3 {
    const bodyIndex = HUMANOID_BODY_NAMES.indexOf(body);
    if (bodyIndex < 0) throw new Error(`Unknown humanoid task-space body: ${body}`);
    return worldVector(this.#data.xpos, this.#bodyIds[bodyIndex]! * 3);
  }

  #taskSpaceSolution(
    baseline: HumanoidReference,
    targets: readonly HumanoidEndEffectorTarget[]
  ): HumanoidTaskSpaceSolution {
    const jointPositions = baseline.jointPositions.slice();
    for (let index = 0; index < jointPositions.length; index += 1) {
      jointPositions[index] = this.#data.qpos[this.#jointPositionAddresses[index]!]!;
    }
    return {
      reference: {
        ...baseline,
        jointPositions,
        jointVelocities: new Float64Array(jointPositions.length)
      },
      residuals: targets.map((target) => {
        const achieved = this.#linkPosition(target.body);
        return {
          body: target.body,
          target: { ...target.position },
          achieved,
          error: vectorDistance(target.position, achieved)
        };
      })
    };
  }

  #policyState(): HumanoidPolicyState {
    return {
      jointPositions: Float64Array.from(
        this.#jointPositionAddresses,
        (address) => this.#data.qpos[address]
      ),
      jointVelocities: Float64Array.from(
        this.#jointVelocityAddresses,
        (address) => this.#data.qvel[address]
      ),
      rootQuaternion: this.#rootQuaternion(),
      rootAngularVelocity: [this.#data.qvel[3]!, this.#data.qvel[4]!, this.#data.qvel[5]!]
    };
  }

  #rootQuaternion(): [number, number, number, number] {
    return [this.#data.qpos[3]!, this.#data.qpos[4]!, this.#data.qpos[5]!, this.#data.qpos[6]!];
  }

  #applyCommand(command: HumanoidJointPositionCommand): void {
    if (command.kind !== "joint_position_pd"
      || command.positions.length !== HUMANOID_JOINT_NAMES.length
      || command.stiffness.length !== HUMANOID_JOINT_NAMES.length
      || command.damping.length !== HUMANOID_JOINT_NAMES.length) {
      throw new Error("Humanoid controller returned an invalid actuation command");
    }
    if (![...command.positions, ...command.stiffness, ...command.damping]
      .every(Number.isFinite)
      || [...command.stiffness, ...command.damping].some((value) => value < 0)) {
      throw new Error("Humanoid controller returned non-finite or negative PD parameters");
    }
    for (let index = 0; index < HUMANOID_JOINT_NAMES.length; index += 1) {
      const actuator = this.#actuatorIds[index]!;
      const torque = command.stiffness[index]!
        * (command.positions[index]! - this.#data.qpos[this.#jointPositionAddresses[index]!]!)
        - command.damping[index]!
        * this.#data.qvel[this.#jointVelocityAddresses[index]!]!;
      const rangeOffset = actuator * 2;
      this.#data.ctrl[actuator] = Math.max(
        this.#model.actuator_ctrlrange[rangeOffset]!,
        Math.min(this.#model.actuator_ctrlrange[rangeOffset + 1]!, torque)
      );
    }
  }

  #validateReference(reference: HumanoidReference): void {
    if (reference.jointPositions.length !== HUMANOID_JOINT_NAMES.length
      || reference.jointVelocities.length !== HUMANOID_JOINT_NAMES.length) {
      throw new Error("Humanoid reference has an invalid joint count");
    }
    for (let index = 0; index < HUMANOID_JOINT_NAMES.length; index += 1) {
      const rangeOffset = this.#jointIds[index]! * 2;
      const value = reference.jointPositions[index]!;
      if (!Number.isFinite(value)
        || value < this.#model.jnt_range[rangeOffset]!
        || value > this.#model.jnt_range[rangeOffset + 1]!) {
        throw new Error(`Humanoid reference exceeds ${HUMANOID_JOINT_NAMES[index]} limits`);
      }
    }
  }

  #actuatorForJoint(jointId: number): number {
    for (let actuator = 0; actuator < this.#model.nu; actuator += 1) {
      if (this.#model.actuator_trnid[actuator * 2] === jointId) return actuator;
    }
    throw new Error(`No actuator found for humanoid joint ${jointId}`);
  }

  #rayObject(origin: Vec3, direction: Vec3, maximumDistance: number): string | null {
    const geometryId = new this.#runtime.IntBuffer(1);
    const normal = new this.#runtime.DoubleBuffer(3);
    try {
      const distance = this.#runtime.mj_ray(
        this.#model,
        this.#data,
        mujocoVector(origin),
        mujocoVector(direction),
        [1, 1, 1, 1, 1, 1],
        true,
        this.#headBodyId,
        geometryId,
        normal
      );
      if (distance < 0 || distance > maximumDistance) return null;
      const hitGeometry = Number(geometryId.GetView()[0] ?? -1);
      if (hitGeometry < 0) return null;
      const hitBody = this.#model.geom_bodyid[hitGeometry];
      return hitBody === undefined ? null : this.#objectNamesByBodyId.get(hitBody) ?? null;
    } finally {
      normal.delete();
      geometryId.delete();
    }
  }

  #id(kind: "body" | "joint", name: string): number {
    const type = kind === "body"
      ? this.#runtime.mjtObj.mjOBJ_BODY.value
      : this.#runtime.mjtObj.mjOBJ_JOINT.value;
    const id = this.#runtime.mj_name2id(this.#model, type, name);
    if (id < 0) throw new Error(`MuJoCo ${kind} is missing: ${name}`);
    return id;
  }

  #contacts(): HumanoidContactSnapshot[] {
    const contacts: HumanoidContactSnapshot[] = [];
    const force = new this.#runtime.DoubleBuffer(6);
    try {
      for (let index = 0; index < this.#data.ncon; index += 1) {
        const contact = this.#data.contact.get(index);
        if (!contact) continue;
        this.#runtime.mj_contactForce(this.#model, this.#data, index, force);
        contacts.push({
          position: worldVector(contact.pos, 0),
          normal: worldVector(contact.frame, 0),
          normalForce: Math.max(0, Number(force.GetView()[0] ?? 0)),
          firstBody: this.#bodyNamesById.get(this.#model.geom_bodyid[contact.geom1]!) ?? null,
          secondBody: this.#bodyNamesById.get(this.#model.geom_bodyid[contact.geom2]!) ?? null,
          firstObject: this.#objectNamesByBodyId.get(
            this.#model.geom_bodyid[contact.geom1]!
          ) ?? null,
          secondObject: this.#objectNamesByBodyId.get(
            this.#model.geom_bodyid[contact.geom2]!
          ) ?? null
        });
      }
    } finally {
      force.delete();
    }
    return contacts;
  }

  #unsafeEnvironmentContacts(
    contacts: readonly HumanoidContactSnapshot[]
  ): HumanoidBodyName[] {
    const bodies = new Set<HumanoidBodyName>();
    for (const contact of contacts) {
      if ((contact.firstBody === null) === (contact.secondBody === null)) continue;
      const name = contact.firstBody ?? contact.secondBody;
      if (!name) continue;
      const foot = name === "left_ankle_roll_link" || name === "right_ankle_roll_link";
      if (foot && Math.abs(contact.normal.y) >= 0.55) continue;
      bodies.add(name);
    }
    return [...bodies];
  }
}

function worldVector(values: ArrayLike<number>, offset: number): Vec3 {
  return {
    x: requiredValue(values, offset + 1),
    y: requiredValue(values, offset + 2),
    z: requiredValue(values, offset)
  };
}

function mujocoVector(value: Vec3): number[] {
  return [value.z, value.x, value.y];
}

function worldQuaternion(values: ArrayLike<number>, offset: number): Quaternion {
  return {
    x: requiredValue(values, offset + 2),
    y: requiredValue(values, offset + 3),
    z: requiredValue(values, offset + 1),
    w: requiredValue(values, offset)
  };
}

function requiredValue(values: ArrayLike<number>, index: number): number {
  const value = values[index];
  if (value === undefined) throw new Error(`MuJoCo state is missing value ${index}`);
  return value;
}

function footContact(
  contacts: readonly HumanoidContactSnapshot[],
  foot: HumanoidBodyName
): HumanoidFootContactSnapshot {
  const matching = contacts.filter((contact) => (
    (contact.firstBody === foot && contact.secondBody === null)
      || (contact.secondBody === foot && contact.firstBody === null)
  ) && Math.abs(contact.normal.y) >= 0.55);
  return {
    touching: matching.length > 0,
    contactCount: matching.length,
    normalForce: matching.reduce((sum, contact) => sum + contact.normalForce, 0),
    points: matching.map((contact) => ({ ...contact.position }))
  };
}

function supportMargin(center: Vec3, points: readonly Vec3[]): number | null {
  if (points.length === 0) return null;
  const minimumX = Math.min(...points.map((point) => point.x));
  const maximumX = Math.max(...points.map((point) => point.x));
  const minimumZ = Math.min(...points.map((point) => point.z));
  const maximumZ = Math.max(...points.map((point) => point.z));
  return Math.min(
    center.x - minimumX,
    maximumX - center.x,
    center.z - minimumZ,
    maximumZ - center.z
  );
}

function copyState(
  target: { length: number; set(values: ArrayLike<number>): void },
  source: ArrayLike<number>,
  label: string
): void {
  if (target.length !== source.length) {
    throw new Error(`Humanoid state has an invalid ${label} length`);
  }
  target.set(source);
}

function assertSpawn(spawn: HumanoidSpawn): void {
  if (![spawn.position.x, spawn.position.y, spawn.position.z, spawn.yaw]
    .every(Number.isFinite)) {
    throw new Error("Invalid humanoid spawn");
  }
}

function assertControllerTiming(controller: HumanoidWholeBodyController): void {
  const descriptor = controller.descriptor;
  const ratio = descriptor.controlStepSeconds / descriptor.physicsStepSeconds;
  if (descriptor.protocol !== "humanoid-controller-v1"
    || descriptor.actuation !== "joint_position_pd"
    || descriptor.implementation.trim().length === 0
    || !Number.isFinite(descriptor.controlStepSeconds)
    || !Number.isFinite(descriptor.physicsStepSeconds)
    || descriptor.controlStepSeconds <= 0
    || descriptor.physicsStepSeconds <= 0
    || ratio < 1
    || Math.abs(ratio - Math.round(ratio)) > 1e-9) {
    throw new Error("Humanoid controller declares an invalid timing or actuation contract");
  }
}

function dampedLeastSquares(
  jacobian: readonly (readonly number[])[],
  error: readonly number[],
  damping: number
): number[] {
  const rows = jacobian.length;
  const columns = jacobian[0]?.length ?? 0;
  const normal = Array.from({ length: rows }, (_, row) => (
    Array.from({ length: rows }, (_, column) => (
      dot(jacobian[row]!, jacobian[column]!) + (row === column ? damping * damping : 0)
    ))
  ));
  const solved = solveLinearSystem(normal, [...error]);
  return Array.from({ length: columns }, (_, column) => (
    jacobian.reduce((sum, row, index) => sum + row[column]! * solved[index]!, 0)
  ));
}

function solveLinearSystem(matrix: number[][], values: number[]): number[] {
  const size = values.length;
  for (let pivot = 0; pivot < size; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < size; row += 1) {
      if (Math.abs(matrix[row]![pivot]!) > Math.abs(matrix[best]![pivot]!)) best = row;
    }
    [matrix[pivot], matrix[best]] = [matrix[best]!, matrix[pivot]!];
    [values[pivot], values[best]] = [values[best]!, values[pivot]!];
    const pivotRow = matrix[pivot]!;
    const divisor = pivotRow[pivot]!;
    if (Math.abs(divisor) < 1e-12) throw new Error("Task-space IK matrix is singular");
    for (let column = pivot; column < size; column += 1) {
      pivotRow[column] = pivotRow[column]! / divisor;
    }
    values[pivot] = values[pivot]! / divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === pivot) continue;
      const targetRow = matrix[row]!;
      const factor = targetRow[pivot]!;
      for (let column = pivot; column < size; column += 1) {
        targetRow[column] = targetRow[column]! - factor * pivotRow[column]!;
      }
      values[row] = values[row]! - factor * values[pivot]!;
    }
  }
  return values;
}

function dot(left: readonly number[], right: readonly number[]): number {
  return left.reduce((sum, value, index) => sum + value * right[index]!, 0);
}

function vectorDifference(target: Vec3, current: Vec3): number[] {
  return [target.x - current.x, target.y - current.y, target.z - current.z];
}

function vectorDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function rotate(
  [w, x, y, z]: readonly [number, number, number, number],
  [vx, vy, vz]: readonly [number, number, number]
): [number, number, number] {
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx)
  ];
}
