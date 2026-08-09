import type { Quaternion, Vec3 } from "../../domain/schema.js";
import {
  add,
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  quaternionAngularDistance,
  quaternionFromRotationMatrix,
  quaternionRotationVector,
  rotateVector,
  scale,
  subtract,
  vectorLength
} from "../geometry.js";
import {
  HUMANOID_HEAD_SENSOR,
  HUMANOID_BODY_NAMES,
  HUMANOID_JOINT_INDEX,
  HUMANOID_JOINT_NAMES,
  YAHMP_POLICY,
  type HumanoidBodyName,
  type HumanoidJointName
} from "./model.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  G1_HAND_JOINT_NAMES,
  G1_HAND_LINK_NAMES,
  G1_MORPHOLOGY,
  g1MujocoBodyName,
  type G1HandContactSurfaceName,
  type G1HandJointName,
  type G1HandLinkName
} from "./morphology.js";
import { g1HandContactGeomName } from "./hand-collision-geometry.js";
import {
  createHumanoidScenePath,
  humanoidModelPath,
  humanoidSceneSolidGeomName,
  loadHumanoidMujoco,
  removeHumanoidScene,
  resolveHumanoidSceneObject,
  type HumanoidSceneObject,
  type ResolvedHumanoidSceneObject,
  type HumanoidSceneSolid,
  type MujocoData,
  type MujocoModel,
  type MujocoModule
} from "./mujoco-runtime.js";
import {
  configureTorqueControlledBodyActuators,
  resolveMujocoActuatedJoints,
  resolveMujocoJoints,
  type MujocoActuatedJointBinding,
  type MujocoJointBinding
} from "./mujoco-joints.js";
import {
  G1HandActuator,
  type G1HandActuatorSnapshot
} from "./hand-actuator.js";
import {
  looksLikeLegacyG129DoFState,
  migrateLegacyG129DoFState
} from "./legacy-state-migration.js";
import {
  neutralHumanoidReference,
  type HumanoidReference
} from "./reference.js";
import {
  YahmpController
} from "./yahmp-controller.js";
import {
  CapabilityRoutingHumanoidController,
  humanoidControllerNeedsReferenceFallback
} from "./capability-routing-controller.js";
import {
  humanoidEndEffectorJointIndexes,
  humanoidEndEffectorPoseJointIndexes,
  humanoidEndEffectorTrackingJointIndexes
} from "./task-space-targets.js";
import {
  HUMANOID_TASK_SPACE_SERVO_AUTHORITY,
  type HumanoidTaskSpaceServoTarget
} from "./task-space-servo.js";
import type {
  HumanoidControllerInferenceOptions,
  HumanoidControllerDescriptor,
  HumanoidControllerExecutionState,
  HumanoidControllerTaskCommand,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController,
  HumanoidWholeBodyControllerFactory
} from "./whole-body-controller.js";

const ORIENTATION_IK_WEIGHT = 0.25;

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
  controllerFactory?: HumanoidWholeBodyControllerFactory;
}

export interface HumanoidPlanningRootPose {
  position: Vec3;
  yawRadians: number;
}

interface ResolvedHumanoidSimulationOptions
  extends Omit<HumanoidSimulationOptions, "objects"> {
  objects?: readonly ResolvedHumanoidSceneObject[];
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
  firstSolid?: string | null | undefined;
  secondSolid?: string | null | undefined;
  firstHandLink: G1HandContactSurfaceName | null;
  secondHandLink: G1HandContactSurfaceName | null;
}

export interface HumanoidObjectSnapshot extends HumanoidLinkSnapshot {
  id: string;
  articulation?: {
    type: "hinge" | "slide";
    position: number;
    velocity: number;
    minimum: number;
    maximum: number;
    normalized: number;
  } | undefined;
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

export interface HumanoidSolidSensorSnapshot {
  sensor: HumanoidObjectSensorSnapshot["sensor"];
  solids: Record<string, HumanoidSceneSolid>;
}

export interface HumanoidHandSurfaceObservation {
  handSurface: G1HandContactSurfaceName;
  hand: "left" | "right";
  worldPosition: Vec3;
  worldRotation: Quaternion;
  wristWorldPosition: Vec3;
  surfaceFromWristWorld: Vec3;
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
  requestedActuatorTorques?: Float64Array;
  handCommandTargets?: Float64Array;
  controller: HumanoidControllerState;
}

export interface HumanoidSimulationSnapshot {
  morphology: typeof G1_MORPHOLOGY;
  simulatedTime: number;
  controller: HumanoidWholeBodyController["descriptor"];
  controllerExecution?: HumanoidControllerExecutionState | undefined;
  rootPosition: Vec3;
  rootRotation: Quaternion;
  joints: Record<HumanoidJointName, {
    position: number;
    velocity: number;
    minimum: number;
    maximum: number;
    effort?: {
      requestedNewtonMeters: number;
      appliedNewtonMeters: number;
      minimumNewtonMeters: number;
      maximumNewtonMeters: number;
      requestedUtilization: number;
      appliedUtilization: number;
      saturated: boolean;
    } | undefined;
  }>;
  links: Record<HumanoidBodyName, HumanoidLinkSnapshot>;
  hands: G1HandActuatorSnapshot & {
    links: Record<G1HandLinkName, HumanoidLinkSnapshot>;
  };
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

export type HumanoidEndEffectorTarget = HumanoidTaskSpaceServoTarget;

export interface HumanoidTaskSpaceSolution {
  reference: HumanoidReference;
  residuals: Array<{
    body: HumanoidEndEffectorTarget["body"];
    target: Vec3;
    achieved: Vec3;
    error: number;
    orientationTarget?: Quaternion;
    orientationAchieved?: Quaternion;
    orientationError?: number;
  }>;
}

export class HumanoidTaskSpaceIkError extends Error {
  readonly residuals: HumanoidTaskSpaceSolution["residuals"];

  constructor(residuals: HumanoidTaskSpaceSolution["residuals"]) {
    super(`Task-space IK did not converge: ${residuals.map((entry) => (
      `${entry.body}=${entry.error.toFixed(3)}m`
        + (entry.orientationError === undefined
          ? ""
          : `/${entry.orientationError.toFixed(3)}rad`)
    )).join(", ")}`);
    this.name = "HumanoidTaskSpaceIkError";
    this.residuals = structuredClone(residuals);
  }
}

export class HumanoidSimulation {
  readonly #runtime: MujocoModule;
  readonly #model: MujocoModel;
  readonly #data: MujocoData;
  readonly #controller: HumanoidWholeBodyController;
  readonly #bodyJointBindings: readonly MujocoActuatedJointBinding<HumanoidJointName>[];
  readonly #jointIds: number[];
  readonly #jointPositionAddresses: number[];
  readonly #jointVelocityAddresses: number[];
  readonly #actuatorIds: number[];
  readonly #requestedActuatorTorques: Float64Array;
  readonly #handCommandTargets = new Float64Array(G1_HAND_JOINT_NAMES.length);
  #hasRequestedActuatorEvidence = false;
  readonly #bodyIds: number[];
  readonly #bodyNamesById = new Map<number, HumanoidBodyName>();
  readonly #handBodyIds: number[];
  readonly #handBodyNamesById = new Map<number, G1HandLinkName>();
  readonly #handSurfaceNamesByGeomId = new Map<number, G1HandContactSurfaceName>();
  readonly #handSurfaceGeometryIds = new Map<G1HandContactSurfaceName, number>();
  readonly #handActuator: G1HandActuator;
  readonly #objectJointBindings: readonly MujocoJointBinding[];
  readonly #objectJointBindingsById = new Map<string, MujocoJointBinding>();
  readonly #objectNamesByBodyId = new Map<number, string>();
  readonly #objectSizesById = new Map<string, Vec3>();
  readonly #objectDescriptorsById = new Map<string, ResolvedHumanoidSceneObject>();
  readonly #solidNamesByGeomId = new Map<number, string>();
  readonly #solidDescriptorsById = new Map<string, HumanoidSceneSolid>();
  readonly #pelvisBodyId: number;
  readonly #pelvisImuSiteId: number;
  readonly #rootVelocityBuffer: InstanceType<MujocoModule["DoubleBuffer"]>;
  readonly #headBodyId: number;
  readonly #leftFootBodyId: number;
  readonly #rightFootBodyId: number;
  readonly #spawn: HumanoidSpawn;

  static async create(options: HumanoidSimulationOptions = {}): Promise<HumanoidSimulation> {
    const { objects, ...baseOptions } = options;
    const resolvedOptions: ResolvedHumanoidSimulationOptions = {
      ...baseOptions,
      ...(objects
        ? { objects: objects.map(resolveHumanoidSceneObject) }
        : {})
    };
    const [runtime, primaryController] = await Promise.all([
      loadHumanoidMujoco(),
      resolvedOptions.controllerFactory
        ? resolvedOptions.controllerFactory()
        : YahmpController.create()
    ]);
    let controller = primaryController;
    try {
      if (humanoidControllerNeedsReferenceFallback(primaryController.descriptor)) {
        const fallback = await YahmpController.create();
        try {
          controller = new CapabilityRoutingHumanoidController(
            primaryController,
            fallback
          );
        } catch (error) {
          await fallback.dispose();
          throw error;
        }
      }
      assertControllerTiming(controller);
      const generatedPath = resolvedOptions.solids || resolvedOptions.objects
        ? createHumanoidScenePath(
            runtime,
            resolvedOptions.solids ?? [],
            resolvedOptions.objects
          )
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
        const instance = new HumanoidSimulation(
          runtime,
          model,
          data,
          controller,
          resolvedOptions
        );
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
    options: ResolvedHumanoidSimulationOptions
  ) {
    this.#runtime = runtime;
    this.#model = model;
    this.#data = data;
    this.#controller = controller;
    this.#bodyJointBindings = resolveMujocoActuatedJoints(
      runtime,
      model,
      HUMANOID_JOINT_NAMES
    );
    configureTorqueControlledBodyActuators(model, this.#bodyJointBindings);
    this.#jointIds = this.#bodyJointBindings.map((binding) => binding.jointId);
    this.#jointPositionAddresses = this.#bodyJointBindings.map(
      (binding) => binding.positionAddress
    );
    this.#jointVelocityAddresses = this.#bodyJointBindings.map(
      (binding) => binding.velocityAddress
    );
    this.#actuatorIds = this.#bodyJointBindings.map((binding) => binding.actuatorId);
    this.#requestedActuatorTorques = new Float64Array(HUMANOID_JOINT_NAMES.length);
    this.#handActuator = new G1HandActuator(runtime, model, data);
    this.#bodyIds = HUMANOID_BODY_NAMES.map((name) => (
      this.#id("body", g1MujocoBodyName(name))
    ));
    HUMANOID_BODY_NAMES.forEach((name, index) => {
      if (name !== "head_link") this.#bodyNamesById.set(this.#bodyIds[index]!, name);
    });
    this.#handBodyIds = G1_HAND_LINK_NAMES.map((name) => this.#id("body", name));
    G1_HAND_LINK_NAMES.forEach((name, index) => {
      this.#handBodyNamesById.set(this.#handBodyIds[index]!, name);
    });
    for (const surface of G1_HAND_CONTACT_SURFACE_NAMES) {
      const namedGeometryId = runtime.mj_name2id(
        model,
        runtime.mjtObj.mjOBJ_GEOM.value,
        g1HandContactGeomName(surface)
      );
      const geometryId = namedGeometryId >= 0
        ? namedGeometryId
        : this.#soleHandSurfaceGeometry(surface);
      this.#assertHandSurfaceGeometry(surface, geometryId);
      this.#handSurfaceNamesByGeomId.set(geometryId, surface);
      this.#handSurfaceGeometryIds.set(surface, geometryId);
    }
    (options.objects ?? []).forEach((object, index) => {
      if (this.#objectDescriptorsById.has(object.id)) {
        throw new Error(`Duplicate MuJoCo scene object: ${object.id}`);
      }
      this.#objectNamesByBodyId.set(this.#id("body", `world-object-${index}`), object.id);
      this.#objectSizesById.set(object.id, { ...object.size });
      this.#objectDescriptorsById.set(object.id, structuredClone(object));
    });
    (options.solids ?? []).forEach((solid, index) => {
      if (this.#solidDescriptorsById.has(solid.id)) {
        throw new Error(`Duplicate MuJoCo scene solid: ${solid.id}`);
      }
      const geometryId = runtime.mj_name2id(
        model,
        runtime.mjtObj.mjOBJ_GEOM.value,
        humanoidSceneSolidGeomName(index, solid.id)
      );
      if (geometryId < 0) {
        throw new Error(`MuJoCo scene solid is missing: ${solid.id}`);
      }
      this.#solidNamesByGeomId.set(geometryId, solid.id);
      this.#solidDescriptorsById.set(solid.id, {
        id: solid.id,
        center: { ...solid.center },
        size: { ...solid.size }
      });
    });
    const jointObjects = (options.objects ?? []).flatMap((object, index) => (
      object.mobility.type === "fixed"
        ? []
        : [{ object, name: `world-object-joint-${index}` as const }]
    ));
    this.#objectJointBindings = resolveMujocoJoints(
      runtime,
      model,
      jointObjects.map(({ name }) => name)
    );
    jointObjects.forEach(({ object }, index) => {
      this.#objectJointBindingsById.set(object.id, this.#objectJointBindings[index]!);
    });
    this.#pelvisBodyId = this.#id("body", "pelvis");
    this.#headBodyId = this.#id("body", g1MujocoBodyName("head_link"));
    this.#leftFootBodyId = this.#id("body", "left_ankle_roll_link");
    this.#rightFootBodyId = this.#id("body", "right_ankle_roll_link");
    this.#spawn = options.spawn ?? {
      position: { x: 0, y: 0, z: 0 },
      yaw: 0
    };
    assertSpawn(this.#spawn);
    this.#pelvisImuSiteId = runtime.mj_name2id(
      model,
      runtime.mjtObj.mjOBJ_SITE.value,
      "imu_in_pelvis"
    );
    if (this.#pelvisImuSiteId < 0) {
      throw new Error("MuJoCo pelvis IMU site is missing");
    }
    this.#rootVelocityBuffer = new runtime.DoubleBuffer(6);
  }

  async step(
    reference: HumanoidReference,
    options: {
      noslipIterations?: number;
      trackedJointPolicyCommand?: HumanoidControllerInferenceOptions[
        "trackedJointPolicyCommand"
      ];
      taskCommand?: HumanoidControllerTaskCommand;
    } = {}
  ): Promise<HumanoidSimulationSnapshot> {
    this.#validateReference(reference);
    const noslipIterations = options.noslipIterations ?? 0;
    if (!Number.isSafeInteger(noslipIterations)
      || noslipIterations < 0
      || noslipIterations > 3) {
      throw new Error("MuJoCo NoSlip iterations must be an integer from 0 to 3");
    }
    if (options.trackedJointPolicyCommand !== undefined
      && options.trackedJointPolicyCommand !== "measured"
      && options.trackedJointPolicyCommand !== "neutral") {
      throw new Error("Tracked-joint policy command must be measured or neutral");
    }
    const controllerOptions: HumanoidControllerInferenceOptions = {
      ...(options.trackedJointPolicyCommand === undefined
        ? {}
        : { trackedJointPolicyCommand: options.trackedJointPolicyCommand }),
      ...(options.taskCommand === undefined
        ? {}
        : { taskCommand: structuredClone(options.taskCommand) })
    };
    const command = await this.#controller.infer(
      this.#policyState(),
      reference,
      controllerOptions
    );
    const substeps = Math.round(
      this.#controller.descriptor.controlStepSeconds
      / this.#controller.descriptor.physicsStepSeconds
    );
    const previousNoslipIterations = this.#model.opt.noslip_iterations;
    this.#model.opt.noslip_iterations = noslipIterations;
    try {
      for (let step = 0; step < substeps; step += 1) {
        this.#applyCommand(command);
        this.#runtime.mj_step(this.#model, this.#data);
      }
    } finally {
      this.#model.opt.noslip_iterations = previousNoslipIterations;
    }
    this.#controller.advanceHistory(
      this.#policyState(),
      reference,
      controllerOptions
    );
    return this.snapshot();
  }

  controllerDescriptor(): HumanoidControllerDescriptor {
    return { ...this.#controller.descriptor };
  }

  solidIds(): string[] {
    return [...this.#solidDescriptorsById.keys()].sort();
  }

  resetController(reference: HumanoidReference): void {
    this.#validateReference(reference);
    this.#controller.reset(this.#policyState(), reference);
  }

  setHandJointTargets(
    targets: Readonly<Partial<Record<G1HandJointName, number>>>
  ): HumanoidSimulationSnapshot {
    this.#handActuator.setTargets(targets);
    for (const [name, target] of Object.entries(targets)) {
      const index = G1_HAND_JOINT_NAMES.indexOf(name as G1HandJointName);
      if (index >= 0 && target !== undefined) this.#handCommandTargets[index] = target;
    }
    return this.snapshot();
  }

  applyHandServoJointTargets(
    targets: Readonly<Partial<Record<G1HandJointName, number>>>
  ): HumanoidSimulationSnapshot {
    this.#handActuator.setTargets(targets);
    return this.snapshot();
  }

  handJointCommandTargets(): Record<G1HandJointName, number> {
    return Object.fromEntries(G1_HAND_JOINT_NAMES.map((name, index) => (
      [name, this.#handCommandTargets[index]!]
    ))) as Record<G1HandJointName, number>;
  }

  handSurfaceObservations(
    snapshot: HumanoidSimulationSnapshot
  ): HumanoidHandSurfaceObservation[] {
    return G1_HAND_CONTACT_SURFACE_NAMES.map((handSurface) => {
      const geometryId = this.#handSurfaceGeometryIds.get(handSurface);
      if (geometryId === undefined) {
        throw new Error(`G1 hand contact surface geometry is missing: ${handSurface}`);
      }
      const hand = handSurface.startsWith("left_") ? "left" as const : "right" as const;
      const wrist = snapshot.links[
        hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
      ];
      const worldPosition = worldVector(this.#data.geom_xpos, geometryId * 3);
      return {
        handSurface,
        hand,
        worldPosition,
        worldRotation: worldRotationMatrix(this.#data.geom_xmat, geometryId * 9),
        wristWorldPosition: { ...wrist.position },
        surfaceFromWristWorld: subtract(worldPosition, wrist.position)
      };
    });
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
    const handLinks = Object.fromEntries(G1_HAND_LINK_NAMES.map((name, index) => {
      const bodyId = this.#handBodyIds[index]!;
      return [name, {
        position: worldVector(this.#data.xpos, bodyId * 3),
        rotation: worldQuaternion(this.#data.xquat, bodyId * 4),
        linearVelocity: worldVector(this.#data.cvel, bodyId * 6 + 3),
        angularVelocity: worldVector(this.#data.cvel, bodyId * 6)
      }];
    })) as Record<G1HandLinkName, HumanoidLinkSnapshot>;
    const joints = Object.fromEntries(HUMANOID_JOINT_NAMES.map((name, index) => {
      const rangeOffset = this.#jointIds[index]! * 2;
      const actuator = this.#actuatorIds[index]!;
      const controlRangeOffset = actuator * 2;
      const effortMinimum = this.#model.actuator_ctrlrange[controlRangeOffset]!;
      const effortMaximum = this.#model.actuator_ctrlrange[controlRangeOffset + 1]!;
      const effortLimit = Math.max(Math.abs(effortMinimum), Math.abs(effortMaximum));
      const requestedEffort = this.#requestedActuatorTorques[index]!;
      const appliedEffort = this.#data.ctrl[actuator]!;
      return [name, {
        position: this.#data.qpos[this.#jointPositionAddresses[index]!]!,
        velocity: this.#data.qvel[this.#jointVelocityAddresses[index]!]!,
        minimum: this.#model.jnt_range[rangeOffset]!,
        maximum: this.#model.jnt_range[rangeOffset + 1]!,
        ...(this.#hasRequestedActuatorEvidence
          ? {
              effort: {
                requestedNewtonMeters: requestedEffort,
                appliedNewtonMeters: appliedEffort,
                minimumNewtonMeters: effortMinimum,
                maximumNewtonMeters: effortMaximum,
                requestedUtilization: Math.abs(requestedEffort) / effortLimit,
                appliedUtilization: Math.abs(appliedEffort) / effortLimit,
                saturated: requestedEffort < effortMinimum || requestedEffort > effortMaximum
              }
            }
          : {})
      }];
    })) as HumanoidSimulationSnapshot["joints"];
    const objects = Object.fromEntries([...this.#objectNamesByBodyId].map(([bodyId, id]) => {
      const positionOffset = bodyId * 3;
      const rotationOffset = bodyId * 4;
      const descriptor = this.#objectDescriptorsById.get(id);
      if (!descriptor) throw new Error(`Missing humanoid scene object descriptor: ${id}`);
      const binding = this.#objectJointBindingsById.get(id);
      const mobility = descriptor.mobility;
      const articulated = mobility.type === "hinge" || mobility.type === "slide";
      const bodyPosition = worldVector(this.#data.xpos, positionOffset);
      const rotation = worldQuaternion(this.#data.xquat, rotationOffset);
      const geometryOffset = articulated
        ? rotateVector(rotation, subtract(descriptor.center, mobility.anchor))
        : { x: 0, y: 0, z: 0 };
      const angularVelocity = worldVector(this.#data.cvel, bodyId * 6);
      const bodyLinearVelocity = worldVector(this.#data.cvel, bodyId * 6 + 3);
      const position = articulated && binding
        ? this.#data.qpos[binding.positionAddress]!
        : null;
      return [id, {
        id,
        position: add(bodyPosition, geometryOffset),
        rotation,
        linearVelocity: add(
          bodyLinearVelocity,
          crossProduct(angularVelocity, geometryOffset)
        ),
        angularVelocity,
        ...(articulated && binding && position !== null
          ? {
              articulation: {
                type: mobility.type,
                position,
                velocity: this.#data.qvel[binding.velocityAddress]!,
                minimum: mobility.range.minimum,
                maximum: mobility.range.maximum,
                normalized: clamp(
                  (position - mobility.range.minimum)
                    / (mobility.range.maximum - mobility.range.minimum),
                  0,
                  1
                )
              }
            }
          : {})
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
    const hands = this.#handActuator.snapshot();
    return {
      morphology: {
        ...G1_MORPHOLOGY,
        source: { ...G1_MORPHOLOGY.source }
      },
      simulatedTime: this.#data.time,
      controller: { ...this.#controller.descriptor },
      controllerExecution: humanoidControllerExecutionState(this.#controller),
      rootPosition: worldVector(this.#data.qpos, 0),
      rootRotation: worldQuaternion(this.#data.qpos, 3),
      joints,
      links,
      hands: { ...hands, links: handLinks },
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
    const sensorRotation = normalizeQuaternion(multiplyQuaternion(
      head.rotation,
      HUMANOID_HEAD_SENSOR.localRotation
    ));
    const visible = Object.fromEntries(Object.entries(snapshot.objects).filter(([id, object]) => (
      objectVisibilityPoints(object, this.#objectSizesById.get(id)).some((point) => {
        const delta = subtract(point, origin);
        const distance = vectorLength(delta);
        if (distance <= 0.001 || distance > maximumRange) return false;
        const direction = scale(delta, 1 / distance);
        const local = rotateVector(inverseQuaternion(sensorRotation), direction);
        const horizontal = Math.atan2(local.x, local.z);
        const vertical = Math.atan2(local.y, Math.hypot(local.x, local.z));
        if (Math.abs(horizontal) > HUMANOID_HEAD_SENSOR.horizontalFieldOfView / 2
          || Math.abs(vertical) > HUMANOID_HEAD_SENSOR.verticalFieldOfView / 2) {
          return false;
        }
        return this.#rayObject(origin, direction, distance + 0.02) === id;
      })
    )));
    return {
      sensor: {
        position: origin,
        rotation: sensorRotation,
        maximumRange,
        horizontalFieldOfView: HUMANOID_HEAD_SENSOR.horizontalFieldOfView,
        verticalFieldOfView: HUMANOID_HEAD_SENSOR.verticalFieldOfView
      },
      objects: visible
    };
  }

  senseSolids(maximumRange: number): HumanoidSolidSensorSnapshot {
    if (!Number.isFinite(maximumRange) || maximumRange <= 0) {
      throw new Error("Humanoid sensor range must be positive");
    }
    const snapshot = this.snapshot();
    const head = snapshot.links.head_link;
    const origin = add(
      head.position,
      rotateVector(head.rotation, HUMANOID_HEAD_SENSOR.localPosition)
    );
    const sensorRotation = normalizeQuaternion(multiplyQuaternion(
      head.rotation,
      HUMANOID_HEAD_SENSOR.localRotation
    ));
    const solids = Object.fromEntries([...this.#solidDescriptorsById].filter(([id, solid]) => (
      solidVisibilityPoints(solid).some((point) => {
        const delta = subtract(point, origin);
        const distance = vectorLength(delta);
        if (distance <= 0.001 || distance > maximumRange) return false;
        const direction = scale(delta, 1 / distance);
        const local = rotateVector(inverseQuaternion(sensorRotation), direction);
        const horizontal = Math.atan2(local.x, local.z);
        const vertical = Math.atan2(local.y, Math.hypot(local.x, local.z));
        if (Math.abs(horizontal) > HUMANOID_HEAD_SENSOR.horizontalFieldOfView / 2
          || Math.abs(vertical) > HUMANOID_HEAD_SENSOR.verticalFieldOfView / 2) {
          return false;
        }
        return this.#raySolid(origin, direction, distance + 0.02) === id;
      })
    )).map(([id, solid]) => [id, {
      id,
      center: { ...solid.center },
      size: { ...solid.size }
    }]));
    return {
      sensor: {
        position: origin,
        rotation: sensorRotation,
        maximumRange,
        horizontalFieldOfView: HUMANOID_HEAD_SENSOR.horizontalFieldOfView,
        verticalFieldOfView: HUMANOID_HEAD_SENSOR.verticalFieldOfView
      },
      solids
    };
  }

  scenePointVisibility(origin: Vec3, points: readonly Vec3[]): boolean[] {
    if (![origin.x, origin.y, origin.z].every(Number.isFinite)) {
      throw new Error("Humanoid visibility origin must be finite");
    }
    const geometryId = new this.#runtime.IntBuffer(1);
    const normal = new this.#runtime.DoubleBuffer(3);
    try {
      return points.map((point) => {
        if (![point.x, point.y, point.z].every(Number.isFinite)) {
          throw new Error("Humanoid visibility target must be finite");
        }
        const delta = subtract(point, origin);
        const distance = vectorLength(delta);
        if (distance <= 1e-9) return true;
        const hitDistance = this.#runtime.mj_ray(
          this.#model,
          this.#data,
          mujocoVector(origin),
          mujocoVector(scale(delta, 1 / distance)),
          [1, 1, 1, 1, 1, 1],
          true,
          this.#headBodyId,
          geometryId,
          normal
        );
        return hitDistance < 0 || hitDistance >= distance - 0.01;
      });
    } finally {
      normal.delete();
      geometryId.delete();
    }
  }

  captureState(): HumanoidSimulationState {
    return {
      time: this.#data.time,
      positions: Float64Array.from(this.#data.qpos),
      velocities: Float64Array.from(this.#data.qvel),
      controls: Float64Array.from(this.#data.ctrl),
      activations: Float64Array.from(this.#data.act),
      accelerationWarmstart: Float64Array.from(this.#data.qacc_warmstart),
      ...(this.#hasRequestedActuatorEvidence
        ? { requestedActuatorTorques: this.#requestedActuatorTorques.slice() }
        : {}),
      handCommandTargets: this.#handCommandTargets.slice(),
      controller: this.#controller.captureState()
    };
  }

  restoreState(state: HumanoidSimulationState): void {
    if (state.controller.protocol !== "humanoid-controller-state-v1"
      || state.controller.implementation !== this.#controller.descriptor.implementation) {
      throw new Error("Humanoid controller state does not match the active implementation");
    }
    if (state.requestedActuatorTorques
      && state.requestedActuatorTorques.length !== 0
      && state.requestedActuatorTorques.length !== HUMANOID_JOINT_NAMES.length) {
      throw new Error(
        "Requested actuator torque evidence must contain all 29 body joints"
      );
    }
    if (state.handCommandTargets
      && state.handCommandTargets.length !== 0
      && state.handCommandTargets.length !== G1_HAND_JOINT_NAMES.length) {
      throw new Error("Hand command authority must contain all 14 hand joints");
    }
    const current = {
      positions: Float64Array.from(this.#data.qpos),
      velocities: Float64Array.from(this.#data.qvel),
      controls: Float64Array.from(this.#data.ctrl),
      activations: Float64Array.from(this.#data.act),
      accelerationWarmstart: Float64Array.from(this.#data.qacc_warmstart)
    };
    const restored = looksLikeLegacyG129DoFState({ source: state, target: current })
      ? migrateLegacyG129DoFState({
          source: state,
          target: current,
          bodyBindings: this.#bodyJointBindings,
          objectBindings: this.#objectJointBindings
        })
      : {
          positions: state.positions,
          velocities: state.velocities,
          controls: state.controls,
          activations: state.activations,
          accelerationWarmstart: state.accelerationWarmstart
        };
    copyState(this.#data.qpos, restored.positions, "positions");
    copyState(this.#data.qvel, restored.velocities, "velocities");
    copyState(this.#data.ctrl, restored.controls, "controls");
    copyState(this.#data.act, restored.activations, "activations");
    copyState(this.#data.qacc_warmstart, restored.accelerationWarmstart, "acceleration warmstart");
    if (state.requestedActuatorTorques
      && state.requestedActuatorTorques.length > 0) {
      copyState(
        this.#requestedActuatorTorques,
        state.requestedActuatorTorques,
        "requested actuator torques"
      );
      this.#hasRequestedActuatorEvidence = true;
    } else {
      this.#requestedActuatorTorques.fill(0);
      this.#hasRequestedActuatorEvidence = false;
    }
    this.#data.time = state.time;
    this.#controller.restoreState(state.controller);
    this.#handActuator.validateCurrentTargets();
    const restoredHandTargets = state.handCommandTargets
      && state.handCommandTargets.length > 0
      ? state.handCommandTargets
      : Float64Array.from(G1_HAND_JOINT_NAMES, (name) => (
          this.#handActuator.snapshot().joints[name].target
        ));
    this.#handCommandTargets.set(restoredHandTargets);
    this.#runtime.mj_forward(this.#model, this.#data);
  }

  solveEndEffectorTargets(
    reference: HumanoidReference,
    targets: readonly HumanoidEndEffectorTarget[],
    options: {
      initialConfiguration?: "reference" | "current";
      preserveTrackingWeights?: boolean;
      maximumReferenceCorrectionRadians?: number;
      planningRootPose?: HumanoidPlanningRootPose;
      allowBestEffort?: boolean;
    } = {}
  ): HumanoidTaskSpaceSolution {
    if (targets.length === 0) return { reference, residuals: [] };
    if (options.allowBestEffort && options.initialConfiguration !== "current") {
      throw new Error("Best-effort task-space servo requires current-state initialization");
    }
    if (new Set(targets.map((target) => target.body)).size !== targets.length) {
      throw new Error("A task-space keyframe cannot repeat an end effector");
    }
    const maximumReferenceCorrectionRadians =
      options.maximumReferenceCorrectionRadians
        ?? HUMANOID_TASK_SPACE_SERVO_AUTHORITY.maximumReferenceCorrectionRadians;
    if (!Number.isFinite(maximumReferenceCorrectionRadians)
      || maximumReferenceCorrectionRadians <= 0
      || maximumReferenceCorrectionRadians
        > HUMANOID_TASK_SPACE_SERVO_AUTHORITY.maximumReferenceCorrectionRadians) {
      throw new Error("Task-space reference correction exceeds servo authority");
    }
    targets.forEach(assertEndEffectorTarget);
    const saved = this.captureState();
    try {
      if (options.initialConfiguration === "current") {
        if (options.planningRootPose) {
          throw new Error("A current-state task-space solve cannot override the planning root pose");
        }
        this.#validateReference(reference);
        this.#runtime.mj_forward(this.#model, this.#data);
      } else {
        this.#setReferenceConfiguration(reference);
        if (options.planningRootPose) {
          this.#setPlanningRootPose(options.planningRootPose);
        }
      }
      const pelvis = this.#linkPosition("pelvis");
      const pelvisRotation = worldQuaternion(this.#data.xquat, this.#pelvisBodyId * 4);
      const resolved = targets.map((target) => ({
        ...target,
        position: target.frame === "world"
          ? { ...target.position }
          : add(pelvis, rotateVector(pelvisRotation, target.position)),
        ...(target.orientation
          ? {
              orientation: target.frame === "world"
                ? normalizeQuaternion(target.orientation)
                : normalizeQuaternion(multiplyQuaternion(
                    pelvisRotation,
                    target.orientation
                  ))
            }
          : {})
      }));
      const jointIndexes = [...new Set(resolved.flatMap((target) => (
        target.orientation
          ? humanoidEndEffectorPoseJointIndexes(
              target.body,
              target.kinematicScope ?? "arm_only"
            )
          : humanoidEndEffectorJointIndexes(
              target.body,
              target.kinematicScope ?? "arm_only"
            )
      )))];
      const regularizedWaistJointIndexes = new Set(resolved
        .filter((target) => target.kinematicScope === "whole_body_reach")
        .flatMap(() => [
          HUMANOID_JOINT_INDEX.get("waist_yaw_joint")!,
          HUMANOID_JOINT_INDEX.get("waist_roll_joint")!,
          HUMANOID_JOINT_INDEX.get("waist_pitch_joint")!
        ]));
      const positionJacobian = new this.#runtime.DoubleBuffer(3 * this.#model.nv);
      const rotationJacobian = new this.#runtime.DoubleBuffer(3 * this.#model.nv);
      try {
        for (let iteration = 0;
          iteration < HUMANOID_TASK_SPACE_SERVO_AUTHORITY.maximumIterations;
          iteration += 1) {
          const states = resolved.map((target) => ({
            position: this.#linkPosition(target.body),
            ...(target.orientation ? { rotation: this.#linkRotation(target.body) } : {})
          }));
          if (resolved.every((target, index) => (
            endEffectorTargetSatisfied(
              target,
              states[index]!,
              target.servoMode === "task_tolerance"
                ? null
                : {
                    position: HUMANOID_TASK_SPACE_SERVO_AUTHORITY
                      .generationPositionConvergenceMeters,
                    orientation: HUMANOID_TASK_SPACE_SERVO_AUTHORITY
                      .generationOrientationConvergenceRadians
                  }
            )
          ))) {
            return this.#taskSpaceSolution(
              reference,
              resolved,
              jointIndexes,
              !options.preserveTrackingWeights,
              options.initialConfiguration === "current"
                ? maximumReferenceCorrectionRadians
                : null
            );
          }
          const error: number[] = [];
          for (let index = 0; index < resolved.length; index += 1) {
            const target = resolved[index]!;
            const state = states[index]!;
            error.push(...vectorDifference(target.position, state.position));
            if (target.orientation && state.rotation) {
              error.push(...vectorValues(
                quaternionRotationVector(target.orientation, state.rotation),
                ORIENTATION_IK_WEIGHT
              ));
            }
          }
          const jacobian = Array.from({ length: error.length }, () => (
            new Array<number>(jointIndexes.length).fill(0)
          ));
          let row = 0;
          for (const target of resolved) {
            const bodyIndex = HUMANOID_BODY_NAMES.indexOf(target.body);
            const bodyId = this.#bodyIds[bodyIndex];
            if (bodyId === undefined) {
              throw new Error(`Unknown humanoid task-space body: ${target.body}`);
            }
            this.#runtime.mj_jacBody(
              this.#model,
              this.#data,
              positionJacobian,
              rotationJacobian,
              bodyId
            );
            const positionView = positionJacobian.GetView();
            const rotationView = rotationJacobian.GetView();
            for (let axis = 0; axis < 3; axis += 1) {
              for (let column = 0; column < jointIndexes.length; column += 1) {
                const dof = this.#jointVelocityAddresses[jointIndexes[column]!]!;
                jacobian[row + axis]![column] = worldJacobianValue(
                  positionView,
                  this.#model.nv,
                  axis,
                  dof
                );
              }
            }
            row += 3;
            if (target.orientation) {
              for (let axis = 0; axis < 3; axis += 1) {
                for (let column = 0; column < jointIndexes.length; column += 1) {
                  const dof = this.#jointVelocityAddresses[jointIndexes[column]!]!;
                  jacobian[row + axis]![column] = ORIENTATION_IK_WEIGHT
                    * worldJacobianValue(rotationView, this.#model.nv, axis, dof);
                }
              }
              row += 3;
            }
          }
          const postureWeight = Math.sqrt(
            HUMANOID_TASK_SPACE_SERVO_AUTHORITY.wholeBodyPostureWeight
          );
          for (const jointIndex of regularizedWaistJointIndexes) {
            const column = jointIndexes.indexOf(jointIndex);
            if (column < 0) continue;
            const postureRow = new Array<number>(jointIndexes.length).fill(0);
            postureRow[column] = postureWeight;
            jacobian.push(postureRow);
            const address = this.#jointPositionAddresses[jointIndex]!;
            error.push(postureWeight * (
              reference.jointPositions[jointIndex]! - this.#data.qpos[address]!
            ));
          }
          const delta = dampedLeastSquares(
            jacobian,
            error,
            HUMANOID_TASK_SPACE_SERVO_AUTHORITY.damping
          );
          for (let column = 0; column < jointIndexes.length; column += 1) {
            const jointIndex = jointIndexes[column]!;
            const jointId = this.#jointIds[jointIndex]!;
            const rangeOffset = jointId * 2;
            const address = this.#jointPositionAddresses[jointIndex]!;
            const postureMinimum = regularizedWaistJointIndexes.has(jointIndex)
              ? reference.jointPositions[jointIndex]!
                - HUMANOID_TASK_SPACE_SERVO_AUTHORITY.maximumWaistDisplacementRadians
              : Number.NEGATIVE_INFINITY;
            const postureMaximum = regularizedWaistJointIndexes.has(jointIndex)
              ? reference.jointPositions[jointIndex]!
                + HUMANOID_TASK_SPACE_SERVO_AUTHORITY.maximumWaistDisplacementRadians
              : Number.POSITIVE_INFINITY;
            this.#data.qpos[address] = clamp(
              this.#data.qpos[address]! + clamp(
                delta[column] ?? 0,
                -HUMANOID_TASK_SPACE_SERVO_AUTHORITY.maximumJointDeltaRadians,
                HUMANOID_TASK_SPACE_SERVO_AUTHORITY.maximumJointDeltaRadians
              ),
              Math.max(this.#model.jnt_range[rangeOffset]!, postureMinimum),
              Math.min(this.#model.jnt_range[rangeOffset + 1]!, postureMaximum)
            );
          }
          this.#runtime.mj_forward(this.#model, this.#data);
        }
      } finally {
        positionJacobian.delete();
        rotationJacobian.delete();
      }
      const residuals = resolved.map((target) => ({
        body: target.body,
        target: { ...target.position },
        achieved: this.#linkPosition(target.body),
        error: vectorDistance(target.position, this.#linkPosition(target.body)),
        ...(target.orientation
          ? {
              orientationTarget: { ...target.orientation },
              orientationAchieved: this.#linkRotation(target.body),
              orientationError: quaternionAngularDistance(
                target.orientation,
                this.#linkRotation(target.body)
              )
            }
          : {})
      }));
      if (options.allowBestEffort) {
        return this.#taskSpaceSolution(
          reference,
          resolved,
          jointIndexes,
          !options.preserveTrackingWeights,
          maximumReferenceCorrectionRadians
        );
      }
      throw new HumanoidTaskSpaceIkError(residuals);
    } finally {
      this.restoreState(saved);
    }
  }

  measureEndEffectorTargets(
    reference: HumanoidReference,
    templates: readonly HumanoidEndEffectorTarget[],
    options: { planningRootPose?: HumanoidPlanningRootPose } = {}
  ): HumanoidEndEffectorTarget[] {
    if (templates.length === 0) return [];
    if (new Set(templates.map((target) => target.body)).size !== templates.length) {
      throw new Error("A task-space servo frame cannot repeat an end effector");
    }
    templates.forEach(assertEndEffectorTarget);
    const saved = this.captureState();
    try {
      this.#setReferenceConfiguration(reference);
      if (options.planningRootPose) {
        this.#setPlanningRootPose(options.planningRootPose);
      }
      const pelvisPosition = this.#linkPosition("pelvis");
      const pelvisRotation = this.#linkRotation("pelvis");
      const inversePelvis = inverseQuaternion(pelvisRotation);
      return templates.map((template) => {
        const worldPosition = this.#linkPosition(template.body);
        const worldOrientation = template.orientation
          ? this.#linkRotation(template.body)
          : undefined;
        return {
          ...template,
          position: template.frame === "world"
            ? worldPosition
            : rotateVector(inversePelvis, subtract(worldPosition, pelvisPosition)),
          ...(worldOrientation
            ? {
                orientation: template.frame === "world"
                  ? normalizeQuaternion(worldOrientation)
                  : normalizeQuaternion(multiplyQuaternion(
                      inversePelvis,
                      worldOrientation
                    ))
              }
            : {})
        };
      });
    } finally {
      this.restoreState(saved);
    }
  }

  async dispose(): Promise<void> {
    await this.#controller.dispose();
    this.#rootVelocityBuffer.delete();
    this.#data.delete();
    this.#model.delete();
  }

  #initialize(): void {
    this.#data.qvel.fill(0);
    this.#data.ctrl.fill(0);
    this.#requestedActuatorTorques.fill(0);
    this.#hasRequestedActuatorEvidence = false;
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
    for (const [id, descriptor] of this.#objectDescriptorsById) {
      if (descriptor.mobility.type !== "hinge" && descriptor.mobility.type !== "slide") {
        continue;
      }
      const binding = this.#objectJointBindingsById.get(id);
      if (!binding) throw new Error(`Articulated humanoid object has no joint: ${id}`);
      this.#data.qpos[binding.positionAddress] = descriptor.mobility.initialPosition;
      this.#data.qvel[binding.velocityAddress] = 0;
    }
    this.#handActuator.holdCurrentPositions();
    const initialHandState = this.#handActuator.snapshot();
    G1_HAND_JOINT_NAMES.forEach((name, index) => {
      this.#handCommandTargets[index] = initialHandState.joints[name].target;
    });
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

  #setPlanningRootPose(pose: HumanoidPlanningRootPose): void {
    if (![pose.position.x, pose.position.y, pose.position.z, pose.yawRadians]
      .every(Number.isFinite)) {
      throw new Error("Humanoid planning root pose must be finite");
    }
    this.#data.qpos[0] = pose.position.z;
    this.#data.qpos[1] = pose.position.x;
    this.#data.qpos[2] = pose.position.y;
    this.#data.qpos[3] = Math.cos(pose.yawRadians / 2);
    this.#data.qpos[4] = 0;
    this.#data.qpos[5] = 0;
    this.#data.qpos[6] = Math.sin(pose.yawRadians / 2);
    this.#runtime.mj_forward(this.#model, this.#data);
  }

  #linkPosition(body: HumanoidBodyName): Vec3 {
    const bodyIndex = HUMANOID_BODY_NAMES.indexOf(body);
    if (bodyIndex < 0) throw new Error(`Unknown humanoid task-space body: ${body}`);
    return worldVector(this.#data.xpos, this.#bodyIds[bodyIndex]! * 3);
  }

  #linkRotation(body: HumanoidBodyName): Quaternion {
    const bodyIndex = HUMANOID_BODY_NAMES.indexOf(body);
    if (bodyIndex < 0) throw new Error(`Unknown humanoid task-space body: ${body}`);
    return worldQuaternion(this.#data.xquat, this.#bodyIds[bodyIndex]! * 4);
  }

  #taskSpaceSolution(
    baseline: HumanoidReference,
    targets: readonly HumanoidEndEffectorTarget[],
    solvedJointIndexes: readonly number[],
    activateTracking: boolean,
    correctionLimitRadians: number | null
  ): HumanoidTaskSpaceSolution {
    const jointPositions = baseline.jointPositions.slice();
    const jointTrackingWeights = baseline.jointTrackingWeights.slice();
    for (const index of solvedJointIndexes) {
      const solved = this.#data.qpos[this.#jointPositionAddresses[index]!]!;
      jointPositions[index] = correctionLimitRadians === null
        ? solved
        : clamp(
            solved,
            baseline.jointPositions[index]! - correctionLimitRadians,
            baseline.jointPositions[index]! + correctionLimitRadians
          );
    }
    if (activateTracking) {
      for (const target of targets) {
        for (const index of humanoidEndEffectorTrackingJointIndexes(
          target.body,
          target.orientation !== undefined,
          target.kinematicScope ?? "arm_only"
        )) {
          jointTrackingWeights[index] = 1;
        }
      }
    }
    return {
      reference: {
        ...baseline,
        jointPositions,
        jointVelocities: new Float64Array(jointPositions.length),
        jointTrackingWeights
      },
      residuals: targets.map((target) => {
        const achieved = this.#linkPosition(target.body);
        const orientationAchieved = target.orientation
          ? this.#linkRotation(target.body)
          : undefined;
        return {
          body: target.body,
          target: { ...target.position },
          achieved,
          error: vectorDistance(target.position, achieved),
          ...(target.orientation && orientationAchieved
            ? {
                orientationTarget: { ...target.orientation },
                orientationAchieved,
                orientationError: quaternionAngularDistance(
                  target.orientation,
                  orientationAchieved
                )
              }
            : {})
        };
      })
    };
  }

  #policyState(): HumanoidPolicyState {
    this.#runtime.mj_objectVelocity(
      this.#model,
      this.#data,
      this.#runtime.mjtObj.mjOBJ_SITE.value,
      this.#pelvisImuSiteId,
      this.#rootVelocityBuffer,
      1
    );
    const rootVelocity = this.#rootVelocityBuffer.GetView();
    const state: HumanoidPolicyState = {
      jointPositions: Float64Array.from(
        this.#jointPositionAddresses,
        (address) => this.#data.qpos[address]
      ),
      jointVelocities: Float64Array.from(
        this.#jointVelocityAddresses,
        (address) => this.#data.qvel[address]
      ),
      rootQuaternion: this.#rootQuaternion(),
      rootAngularVelocity: [
        this.#data.qvel[3]!,
        this.#data.qvel[4]!,
        this.#data.qvel[5]!
      ]
    };
    const features = this.#controller.descriptor.learnedPolicy
      ?.observationFeatures ?? [];
    if (!features.some((feature) => [
      "root_kinematics",
      "hand_state",
      "end_effector_state",
      "contact_state",
      "object_state",
      "articulation_state"
    ].includes(feature))) return state;
    const snapshot = this.snapshot();
    const endEffectorNames = [
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link"
    ] as const;
    state.environment = {
      protocol: "humanoid-policy-environment-v1",
      authority: "mujoco_state",
      rootVelocityFrame: "pelvis_imu",
      rootLinearVelocity: [
        requiredValue(rootVelocity, 3),
        requiredValue(rootVelocity, 4),
        requiredValue(rootVelocity, 5)
      ],
      rootAngularVelocity: [
        requiredValue(rootVelocity, 0),
        requiredValue(rootVelocity, 1),
        requiredValue(rootVelocity, 2)
      ],
      endEffectors: Object.fromEntries(endEffectorNames.map((name) => [name, {
        position: { ...snapshot.links[name].position },
        rotation: { ...snapshot.links[name].rotation }
      }])),
      hands: Object.fromEntries(Object.entries(snapshot.hands.joints).map(
        ([name, joint]) => [name, {
          position: joint.position,
          velocity: joint.velocity,
          target: joint.target
        }]
      )),
      contacts: snapshot.contacts.map((contact) => ({
        position: { ...contact.position },
        normal: { ...contact.normal },
        normalForce: contact.normalForce,
        firstBody: contact.firstBody,
        secondBody: contact.secondBody,
        firstObject: contact.firstObject,
        secondObject: contact.secondObject,
        firstSolid: contact.firstSolid ?? null,
        secondSolid: contact.secondSolid ?? null,
        firstHandLink: contact.firstHandLink,
        secondHandLink: contact.secondHandLink
      })),
      objects: Object.values(snapshot.objects).map((object) => ({
        id: object.id,
        position: { ...object.position },
        rotation: { ...object.rotation },
        linearVelocity: { ...object.linearVelocity },
        angularVelocity: { ...object.angularVelocity },
        ...(object.articulation
          ? {
              articulation: {
                type: object.articulation.type,
                position: object.articulation.position,
                velocity: object.articulation.velocity,
                minimum: object.articulation.minimum,
                maximum: object.articulation.maximum
              }
            }
          : {})
      }))
    };
    return state;
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
    this.#hasRequestedActuatorEvidence = true;
    for (let index = 0; index < HUMANOID_JOINT_NAMES.length; index += 1) {
      const actuator = this.#actuatorIds[index]!;
      const torque = command.stiffness[index]!
        * (command.positions[index]! - this.#data.qpos[this.#jointPositionAddresses[index]!]!)
        - command.damping[index]!
        * this.#data.qvel[this.#jointVelocityAddresses[index]!]!;
      this.#requestedActuatorTorques[index] = torque;
      const rangeOffset = actuator * 2;
      this.#data.ctrl[actuator] = Math.max(
        this.#model.actuator_ctrlrange[rangeOffset]!,
        Math.min(this.#model.actuator_ctrlrange[rangeOffset + 1]!, torque)
      );
    }
  }

  #validateReference(reference: HumanoidReference): void {
    if (reference.jointPositions.length !== HUMANOID_JOINT_NAMES.length
      || reference.jointVelocities.length !== HUMANOID_JOINT_NAMES.length
      || reference.jointTrackingWeights.length !== HUMANOID_JOINT_NAMES.length) {
      throw new Error("Humanoid reference has an invalid joint count");
    }
    for (let index = 0; index < HUMANOID_JOINT_NAMES.length; index += 1) {
      const rangeOffset = this.#jointIds[index]! * 2;
      const value = reference.jointPositions[index]!;
      const velocity = reference.jointVelocities[index]!;
      const trackingWeight = reference.jointTrackingWeights[index]!;
      if (!Number.isFinite(value) || !Number.isFinite(velocity)
        || value < this.#model.jnt_range[rangeOffset]!
        || value > this.#model.jnt_range[rangeOffset + 1]!) {
        throw new Error(`Humanoid reference exceeds ${HUMANOID_JOINT_NAMES[index]} limits`);
      }
      if (!Number.isFinite(trackingWeight) || trackingWeight < 0 || trackingWeight > 1) {
        throw new Error(
          `Humanoid reference has an invalid ${HUMANOID_JOINT_NAMES[index]} tracking weight`
        );
      }
    }
  }

  #rayObject(origin: Vec3, direction: Vec3, maximumDistance: number): string | null {
    return this.#raySceneEntity(origin, direction, maximumDistance).objectId;
  }

  #raySolid(origin: Vec3, direction: Vec3, maximumDistance: number): string | null {
    return this.#raySceneEntity(origin, direction, maximumDistance).solidId;
  }

  #raySceneEntity(
    origin: Vec3,
    direction: Vec3,
    maximumDistance: number
  ): { objectId: string | null; solidId: string | null } {
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
      if (distance < 0 || distance > maximumDistance) {
        return { objectId: null, solidId: null };
      }
      const hitGeometry = Number(geometryId.GetView()[0] ?? -1);
      if (hitGeometry < 0) return { objectId: null, solidId: null };
      const hitBody = this.#model.geom_bodyid[hitGeometry];
      return {
        objectId: hitBody === undefined
          ? null
          : this.#objectNamesByBodyId.get(hitBody) ?? null,
        solidId: this.#solidNamesByGeomId.get(hitGeometry) ?? null
      };
    } finally {
      normal.delete();
      geometryId.delete();
    }
  }

  #id(kind: "body", name: string): number {
    const id = this.#runtime.mj_name2id(
      this.#model,
      this.#runtime.mjtObj.mjOBJ_BODY.value,
      name
    );
    if (id < 0) throw new Error(`MuJoCo ${kind} is missing: ${name}`);
    return id;
  }

  #contacts(): HumanoidContactSnapshot[] {
    const contacts: HumanoidContactSnapshot[] = [];
    const force = new this.#runtime.DoubleBuffer(6);
    const nativeContacts = this.#data.contact;
    try {
      for (let index = 0; index < this.#data.ncon; index += 1) {
        const contact = nativeContacts.get(index);
        if (!contact) continue;
        try {
          this.#runtime.mj_contactForce(this.#model, this.#data, index, force);
          const firstBodyId = this.#model.geom_bodyid[contact.geom1]!;
          const secondBodyId = this.#model.geom_bodyid[contact.geom2]!;
          contacts.push({
            position: worldVector(contact.pos, 0),
            normal: worldVector(contact.frame, 0),
            normalForce: Math.max(0, Number(force.GetView()[0] ?? 0)),
            firstBody: this.#bodyNamesById.get(firstBodyId) ?? null,
            secondBody: this.#bodyNamesById.get(secondBodyId) ?? null,
            firstObject: this.#objectNamesByBodyId.get(firstBodyId) ?? null,
            secondObject: this.#objectNamesByBodyId.get(secondBodyId) ?? null,
            firstSolid: this.#solidNamesByGeomId.get(contact.geom1) ?? null,
            secondSolid: this.#solidNamesByGeomId.get(contact.geom2) ?? null,
            firstHandLink: this.#handContactSurface(contact.geom1, firstBodyId),
            secondHandLink: this.#handContactSurface(contact.geom2, secondBodyId)
          });
        } finally {
          contact.delete();
        }
      }
    } finally {
      nativeContacts.delete();
      force.delete();
    }
    return contacts;
  }

  #handContactSurface(
    geometryId: number,
    bodyId: number
  ): G1HandContactSurfaceName | null {
    return this.#handSurfaceNamesByGeomId.get(geometryId)
      ?? this.#handBodyNamesById.get(bodyId)
      ?? null;
  }

  #soleHandSurfaceGeometry(surface: G1HandContactSurfaceName): number {
    const bodyId = this.#id("body", g1HandSurfaceBodyName(surface));
    const geometryIds = Array.from({ length: this.#model.ngeom }, (_, index) => index)
      .filter((geometryId) => this.#model.geom_bodyid[geometryId] === bodyId);
    if (geometryIds.length !== 1) {
      throw new Error(
        `MuJoCo hand contact surface ${surface} has ${geometryIds.length} collision geoms`
      );
    }
    return geometryIds[0]!;
  }

  #assertHandSurfaceGeometry(
    surface: G1HandContactSurfaceName,
    geometryId: number
  ): void {
    const expectedBody = g1HandSurfaceBodyName(surface);
    const expectedBodyId = this.#runtime.mj_name2id(
      this.#model,
      this.#runtime.mjtObj.mjOBJ_BODY.value,
      expectedBody
    );
    if (expectedBodyId < 0 || this.#model.geom_bodyid[geometryId] !== expectedBodyId) {
      throw new Error(
        `MuJoCo hand contact geom ${g1HandContactGeomName(surface)} has invalid source body`
      );
    }
  }

  #unsafeEnvironmentContacts(
    contacts: readonly HumanoidContactSnapshot[]
  ): HumanoidBodyName[] {
    const bodies = new Set<HumanoidBodyName>();
    for (const contact of contacts) {
      const firstRobot = contact.firstBody !== null || contact.firstHandLink !== null;
      const secondRobot = contact.secondBody !== null || contact.secondHandLink !== null;
      if (firstRobot === secondRobot) continue;
      const name = contact.firstBody ?? contact.secondBody;
      if (!name) continue;
      const foot = name === "left_ankle_roll_link" || name === "right_ankle_roll_link";
      if (foot && Math.abs(contact.normal.y) >= 0.55) continue;
      bodies.add(name);
    }
    return [...bodies];
  }
}

function g1HandSurfaceBodyName(surface: G1HandContactSurfaceName): string {
  return surface.endsWith("_hand_palm_link")
    ? surface.replace("_hand_palm_link", "_wrist_yaw_link")
    : surface;
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

function worldRotationMatrix(values: ArrayLike<number>, offset: number): Quaternion {
  return quaternionFromRotationMatrix([
    requiredValue(values, offset + 4),
    requiredValue(values, offset + 5),
    requiredValue(values, offset + 3),
    requiredValue(values, offset + 7),
    requiredValue(values, offset + 8),
    requiredValue(values, offset + 6),
    requiredValue(values, offset + 1),
    requiredValue(values, offset + 2),
    requiredValue(values, offset)
  ]);
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

function humanoidControllerExecutionState(
  controller: HumanoidWholeBodyController
): HumanoidControllerExecutionState {
  const state = controller.executionState?.() ?? {
    protocol: "humanoid-controller-execution-v1" as const,
    mode: controller.descriptor.learnedPolicy
      ? "learned_policy" as const
      : "reference_control" as const,
    activeImplementation: controller.descriptor.implementation,
    transition: null
  };
  if (state.protocol !== "humanoid-controller-execution-v1"
    || (state.mode !== "learned_policy" && state.mode !== "reference_control")
    || state.activeImplementation.trim().length === 0
    || (state.transition !== null
      && (state.transition.fromImplementation.trim().length === 0
        || state.transition.toImplementation !== state.activeImplementation
        || state.transition.fromImplementation === state.transition.toImplementation
        || !Number.isFinite(state.transition.progress)
        || state.transition.progress < 0
        || state.transition.progress > 1
        || !Number.isFinite(state.transition.durationSeconds)
        || state.transition.durationSeconds <= 0))) {
    throw new Error("Humanoid controller returned invalid execution state");
  }
  return structuredClone(state);
}

function assertControllerTiming(controller: HumanoidWholeBodyController): void {
  const descriptor = controller.descriptor;
  const ratio = descriptor.controlStepSeconds / descriptor.physicsStepSeconds;
  if (descriptor.protocol !== "humanoid-controller-v1"
    || descriptor.actuation !== "joint_position_pd"
    || descriptor.implementation.trim().length === 0
    || !Number.isFinite(descriptor.controlStepSeconds)
    || !Number.isFinite(descriptor.physicsStepSeconds)
    || (descriptor.commandResponseHorizonSeconds !== undefined
      && (!Number.isFinite(descriptor.commandResponseHorizonSeconds)
        || descriptor.commandResponseHorizonSeconds <= 0))
    || (descriptor.minimumEffectivePlanarSpeedMetersPerSecond !== undefined
      && (!Number.isFinite(descriptor.minimumEffectivePlanarSpeedMetersPerSecond)
        || descriptor.minimumEffectivePlanarSpeedMetersPerSecond <= 0))
    || descriptor.controlStepSeconds <= 0
    || descriptor.physicsStepSeconds <= 0
    || ratio < 1
    || Math.abs(ratio - Math.round(ratio)) > 1e-9) {
    throw new Error("Humanoid controller declares an invalid timing or actuation contract");
  }
}

function assertEndEffectorTarget(target: HumanoidEndEffectorTarget): void {
  if (![target.position.x, target.position.y, target.position.z, target.tolerance]
    .every(Number.isFinite)
    || target.tolerance <= 0) {
    throw new Error("Task-space target must have a finite position and positive tolerance");
  }
  const hasOrientation = target.orientation !== undefined;
  const hasOrientationTolerance = target.orientationTolerance !== undefined;
  if (hasOrientation !== hasOrientationTolerance) {
    throw new Error(
      "Task-space orientation and orientation tolerance must be provided together"
    );
  }
  if (target.orientation) normalizeQuaternion(target.orientation);
  if (target.orientationTolerance !== undefined
    && (!Number.isFinite(target.orientationTolerance)
      || target.orientationTolerance <= 0
      || target.orientationTolerance > Math.PI)) {
    throw new Error("Task-space orientation tolerance must be within (0, pi]");
  }
}

function objectVisibilityPoints(
  object: HumanoidObjectSnapshot,
  size: Vec3 | undefined
): Vec3[] {
  if (!size) throw new Error(`Humanoid sensor is missing object size: ${object.id}`);
  const offsets = [-0.45, 0, 0.45];
  return offsets.flatMap((x) => offsets.flatMap((y) => offsets.map((z) => add(
    object.position,
    rotateVector(object.rotation, {
      x: x * size.x,
      y: y * size.y,
      z: z * size.z
    })
  ))));
}

function solidVisibilityPoints(solid: HumanoidSceneSolid): Vec3[] {
  const offsets = [-0.45, 0, 0.45];
  return offsets.flatMap((x) => offsets.flatMap((y) => offsets.map((z) => ({
    x: solid.center.x + x * solid.size.x,
    y: solid.center.y + y * solid.size.y,
    z: solid.center.z + z * solid.size.z
  }))));
}

function endEffectorTargetSatisfied(
  target: HumanoidEndEffectorTarget,
  state: { position: Vec3; rotation?: Quaternion },
  convergenceCap: { position: number; orientation: number } | null
): boolean {
  if (vectorDistance(target.position, state.position)
    > Math.min(target.tolerance, convergenceCap?.position ?? target.tolerance)) {
    return false;
  }
  if (!target.orientation) return true;
  return state.rotation !== undefined
    && quaternionAngularDistance(target.orientation, state.rotation)
      <= Math.min(
        target.orientationTolerance!,
        convergenceCap?.orientation ?? target.orientationTolerance!
      );
}

function vectorValues(value: Vec3, multiplier = 1): number[] {
  return [value.x * multiplier, value.y * multiplier, value.z * multiplier];
}

function worldJacobianValue(
  values: ArrayLike<number>,
  degreesOfFreedom: number,
  worldAxis: number,
  degreeOfFreedom: number
): number {
  const mujocoAxis = [1, 2, 0][worldAxis];
  if (mujocoAxis === undefined) throw new Error(`Invalid world Jacobian axis: ${worldAxis}`);
  return requiredValue(values, mujocoAxis * degreesOfFreedom + degreeOfFreedom);
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

function crossProduct(left: Vec3, right: Vec3): Vec3 {
  return {
    x: left.y * right.z - left.z * right.y,
    y: left.z * right.x - left.x * right.z,
    z: left.x * right.y - left.y * right.x
  };
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
