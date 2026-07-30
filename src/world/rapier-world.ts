import { randomUUID } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";
import RAPIER from "@dimforge/rapier3d-compat";
import type {
  JsonValue,
  Scenario,
  Vec3,
  VoxelCoordinate,
  VoxelMaterial,
  WorldSnapshot,
  BodyChannel
} from "../domain/schema.js";
import {
  ORIENTATION_TOLERANCE,
  POSITION_TOLERANCE,
  normalizeQuaternion,
  quaternionDistance,
  solveEndEffectorTarget,
  type EndEffectorTarget
} from "./kinematics.js";
import {
  NavigationMesh,
  type NavigationObstacle,
  type NavigationPlan
} from "./navigation.js";
import { selectDiverseStandoffs } from "./standoff-selection.js";
import {
  armTargetsReached,
  boundedTolerance,
  clamp,
  finiteQuaternion,
  finiteVector,
  moveTowards,
  normalizeAngle,
  planarDistance,
  quaternion,
  quaternionYaw,
  sameVector,
  scale,
  subtract,
  vector,
  vectorLength,
  yawRotation
} from "./geometry.js";
import {
  collisionKey,
  collisionSetJson,
  collisionTransitionAllowed,
  colliderData,
  colliderIdentity,
  type CollisionIssue
} from "./collision.js";
import {
  resolveBasePlan,
  type BasePlanResolution,
  type ResolvedBaseSegment
} from "./base-preflight.js";
import { evaluateAffordances, type AffordanceEvent } from "./affordances.js";
import { ExplorationMap } from "./exploration.js";
import { buildScene, type SimObject } from "./scene-builder.js";
import { buildTerrainSurvey, visibleTerrainCells } from "./terrain-observation.js";
import {
  PlanRegistry,
  planDenialDetail,
  type PlanKind
} from "./plan-registry.js";
import { rigTransforms, type RigTransform } from "./rig.js";
import {
  ROBOT_SPEC,
  jointLimitIssue,
  type RobotJointName,
  type RobotJointState
} from "./robot-model.js";
import {
  VoxelChunkPhysics,
  type ChunkResidentRegion
} from "./voxel-chunk-physics.js";
import { VoxelEditError, VoxelStore } from "./voxel-store.js";
import { VoxelInteraction } from "./voxel-interaction.js";
import {
  planArmTrajectory,
  type ArmPose,
  type ArmTrajectory
} from "./arm-trajectory.js";
import {
  WorldCommandScheduler,
  type ActiveWorldCommand,
  type ScheduledCommandSource
} from "./world-command-scheduler.js";
import { GraspAttachment } from "./grasp-attachment.js";
import { controlBaseTowardTarget } from "./base-motion-controller.js";
import { rankedArmRetractions } from "./arm-retraction.js";
import { armIkSeeds, type ArmIkSeed } from "./arm-ik-seeds.js";
import { armReachDiagnosis } from "./arm-reach-diagnosis.js";
import {
  rankVoxelStandoffs,
  type VoxelInteractionTarget,
  type VoxelStandoffCandidate
} from "./voxel-affordance.js";
import {
  armPlanFocus,
  endEffectorVerificationTarget,
  restoreArmPlan,
  snapshotArmPlan,
  type StoredArmPlan
} from "./arm-plan.js";
import { EntityVisibility } from "./entity-visibility.js";
import { ArticulatedCollisionQuery } from "./articulated-collision-query.js";

type Collider = InstanceType<typeof RAPIER.Collider>;
type ContactManifold = InstanceType<typeof RAPIER.TempContactManifold>;
type RigidBody = InstanceType<typeof RAPIER.RigidBody>;
type World = InstanceType<typeof RAPIER.World>;

export interface SourceCommand extends ScheduledCommandSource {
  signal?: AbortSignal;
}

export type WorldFrameSink = (
  frames: WorldSnapshot[]
) => void | Promise<void>;

export interface MotionOptions {
  maxVelocity?: number;
  maxDurationSeconds?: number;
  tolerance?: number;
}

export type ArmJointTargets = ArmPose;

export type ArmJointTargetInput = {
  [Joint in keyof ArmJointTargets]?: number | undefined;
};

export type { EndEffectorTarget } from "./kinematics.js";

export interface CommandResult {
  accepted: boolean;
  code: string;
  detail: JsonValue;
}

interface StoredNavigationPlan extends NavigationPlan {
  id: string;
  createdRevision: number;
  target: Vec3;
  face: Vec3 | null;
  segments: ResolvedBaseSegment[] | null;
}

type NavigationStatus = WorldSnapshot["navigation"];

interface FingerContact {
  objectId: string | null;
  force: number;
}

interface RobotContactPair {
  link_id: string;
  collider_kind: string;
  collider_id: string | null;
  contact_count: number;
  force: number;
}

type WheelOdometry = WorldSnapshot["robot"]["odometry"]["left_wheel"];

const ZERO_VECTOR = { x: 0, y: 0, z: 0 } as const;
const MAX_AFFORDANCE_EVENTS = 32;
/**
 * How far the face point may sit out of the arm's sagittal plane once base
 * facing has converged. Half the IK position tolerance, so alignment error
 * alone can never be what pushes a reachable grasp past the solver's residual
 * limit. Tighter costs only a fraction of a simulated second of rotation.
 */
const FACING_LATERAL_TOLERANCE = POSITION_TOLERANCE / 2;
let rapierInitialization: Promise<void> | undefined;

export class RapierWorld {
  readonly #scenario: Scenario;
  readonly #world: World;
  #navigation: NavigationMesh;
  readonly #robot: RigidBody;
  readonly #linkBodies: Map<string, RigidBody>;
  readonly #linkColliders: Map<string, Collider>;
  readonly #gripperAnchor: RigidBody;
  readonly #leftFingerCollider: Collider;
  readonly #rightFingerCollider: Collider;
  readonly #objects: Map<string, SimObject>;
  readonly #voxelStore: VoxelStore | null;
  readonly #voxelChunks: VoxelChunkPhysics | null;
  readonly #voxelInteraction: VoxelInteraction | null;
  readonly #entityVisibility: EntityVisibility;
  readonly #articulatedCollisions: ArticulatedCollisionQuery;
  readonly #navigationPlans = new PlanRegistry<StoredNavigationPlan>();
  readonly #armPlans = new PlanRegistry<StoredArmPlan>();
  readonly #begunCommands = new Set<string>();
  readonly #advancedCommands = new Set<string>();
  #frame = 0;
  /**
   * Increments once per body command that actually enters a physics frame,
   * never for an already-satisfied no-op target and never per individual step.
   * A plan computed from the current pose stays valid until physical state can
   * have changed, so reads and repeated reached targets do not invent a new
   * world identity.
   */
  #worldRevision = 0;
  #simulatedTime = 0;
  #yaw: number;
  #pendingBasePosition: Vec3 | null = null;
  readonly #commands: WorldCommandScheduler;
  #frameSink: WorldFrameSink | null = null;
  #frameSinkBatch: WorldSnapshot[] = [];
  #joints: RobotJointState;
  #jointTargets: RobotJointState;
  #jointVelocities: RobotJointState = {
    head_yaw: 0,
    head_pitch: 0,
    shoulder: 0,
    elbow: 0,
    wrist: 0,
    gripper_aperture: 0
  };
  #leftWheel: WheelOdometry = { position: 0, velocity: 0 };
  #rightWheel: WheelOdometry = { position: 0, velocity: 0 };
  #leftContact: FingerContact = { objectId: null, force: 0 };
  #rightContact: FingerContact = { objectId: null, force: 0 };
  #contactCandidate: string | null = null;
  #contactFrames = 0;
  #gripperDirection: -1 | 0 | 1 = 0;
  #gripperMaximumForce: number = ROBOT_SPEC.gripper.defaultMaximumForce;
  readonly #attachment: GraspAttachment;
  #navigationStatus: NavigationStatus = idleNavigation();
  #affordanceEvents: AffordanceEvent[] = [];
  #unsupportedAffordancesReported = new Set<string>();
  readonly #explored: ExplorationMap;
  #navigationVoxelSignature = "";

  static async create(scenario: Scenario, restore?: WorldSnapshot): Promise<RapierWorld> {
    rapierInitialization ??= RAPIER.init();
    await rapierInitialization;
    const voxelStore = scenario.terrain
      ? new VoxelStore(scenario.terrain, restore?.voxels)
      : null;
    if (voxelStore) {
      const position = restore?.robot.position ?? {
        x: scenario.robot.x,
        y: ROBOT_SPEC.base.centerY,
        z: scenario.robot.z
      };
      voxelStore.setLoadedChunks(voxelStore.desiredChunksAround(position, 2));
    }
    const navigation = await NavigationMesh.create(
      scenario,
      voxelStore
        ? {
            region: voxelStore.loadedRegion(),
            terrainSolids: voxelStore.loadedChunks().flatMap((chunk) => voxelStore.boxesInChunk(chunk))
          }
        : undefined
    );
    let instance: RapierWorld | undefined;
    try {
      instance = new RapierWorld(scenario, navigation, voxelStore);
      if (restore) instance.#restore(restore);
      await instance.#synchronizeVoxelRuntime();
      return instance;
    } catch (error) {
      if (instance) instance.dispose();
      else navigation.dispose();
      throw error;
    }
  }

  private constructor(
    scenario: Scenario,
    navigation: NavigationMesh,
    voxelStore: VoxelStore | null
  ) {
    this.#scenario = structuredClone(scenario);
    this.#navigation = navigation;
    this.#commands = new WorldCommandScheduler((commandIds) => this.#stepCommands(commandIds));
    this.#yaw = scenario.robot.yaw;
    // The stance the scenario starts in, not a constant. A generated world
    // draws one from its seed so the arm does not begin every run folded the
    // same way; an authored scenario may omit it and get the neutral stance.
    this.#joints = { ...ROBOT_SPEC.defaultJoints, ...scenario.robot.joints };
    this.#jointTargets = { ...this.#joints };
    this.#explored = ExplorationMap.forTerrain(this.#scenario.terrain);

    const scene = buildScene(this.#scenario);
    this.#world = scene.world;
    this.#robot = scene.robot;
    this.#linkBodies = scene.linkBodies;
    this.#linkColliders = scene.linkColliders;
    this.#gripperAnchor = scene.gripperAnchor;
    this.#attachment = new GraspAttachment(this.#world, this.#gripperAnchor);
    this.#leftFingerCollider = scene.leftFingerCollider;
    this.#rightFingerCollider = scene.rightFingerCollider;
    this.#objects = scene.objects;
    this.#entityVisibility = new EntityVisibility({
      world: this.#world,
      linkBodies: this.#linkBodies,
      maximumRange: this.#scenario.visibility_radius
    });
    this.#articulatedCollisions = new ArticulatedCollisionQuery({
      world: this.#world,
      objects: this.#objects,
      attachment: this.#attachment,
      currentBase: () => vector(this.#robot.translation()),
      currentYaw: () => this.#yaw
    });
    this.#voxelStore = voxelStore;
    this.#voxelChunks = voxelStore ? new VoxelChunkPhysics(this.#world, voxelStore, 2) : null;
    this.#voxelInteraction = voxelStore
      ? new VoxelInteraction({
          world: this.#world,
          store: voxelStore,
          links: this.#linkBodies,
          visibilityRadius: this.#scenario.visibility_radius
        })
      : null;
    this.#voxelChunks?.synchronize(
      { x: scenario.robot.x, z: scenario.robot.z },
      this.#dynamicVoxelResidents()
    );
    this.#navigationVoxelSignature = this.#voxelNavigationSignature();

    // The rig is built at the base pose with a straight chain, so pose it by
    // the default joint angles and let the objects settle before anything is
    // reported: the first snapshot must describe a scene at rest.
    this.#syncRig(true);
    for (let index = 0; index < 8; index += 1) this.#world.step();
    this.#refreshContacts();
    this.#observeTerrain();
  }

  dispose(): void {
    this.#voxelChunks?.dispose();
    this.#navigation.dispose();
    this.#world.free();
  }

  async #synchronizeVoxelRuntime(): Promise<void> {
    if (!this.#voxelStore || !this.#voxelChunks) return;
    this.#voxelChunks.synchronize(
      this.#robot.translation(),
      this.#dynamicVoxelResidents()
    );
    const signature = this.#voxelNavigationSignature();
    if (signature === this.#navigationVoxelSignature) return;
    const next = await NavigationMesh.create(this.#scenario, {
      region: this.#voxelChunks.region(),
      terrainSolids: this.#voxelChunks.activeSolids()
    });
    const previous = this.#navigation;
    this.#navigation = next;
    this.#navigationVoxelSignature = signature;
    previous.dispose();
  }

  #voxelNavigationSignature(): string {
    if (!this.#voxelStore) return "";
    return `${this.#voxelStore.revision}|${this.#voxelChunks?.navigationChunks()
      .map((chunk) => `${chunk.column}:${chunk.row}`)
      .join(",") ?? ""}`;
  }

  /**
   * Keeps real dynamic entities and the terrain they can touch in the same
   * Rapier residency set. The navigation window intentionally remains owned by
   * the robot: these swept bounds must not turn one distant payload into a
   * world-spanning Recast rebuild.
   */
  #dynamicVoxelResidents(): ChunkResidentRegion[] {
    if (!this.#voxelStore) return [];
    const timestep = this.#world.timestep;
    const skin = Math.max(0.02, this.#voxelStore.terrain.cell * 0.02);
    return [...this.#objects.values()].flatMap((object) => {
      if (!object.config.portable || !object.body.isEnabled()) return [];
      const position = object.body.translation();
      const velocity = object.body.linvel();
      // A bounding sphere covers every yaw/pitch/roll of a carried or released
      // box. Sweeping it through the next frame prevents a chunk-boundary step
      // from dropping the collider beneath a moving payload.
      const radius = Math.hypot(
        object.config.size.x,
        object.config.size.y,
        object.config.size.z
      ) / 2 + skin;
      const nextX = position.x + velocity.x * timestep;
      const nextZ = position.z + velocity.z * timestep;
      return [{
        minimum: {
          x: Math.min(position.x, nextX) - radius,
          z: Math.min(position.z, nextZ) - radius
        },
        maximum: {
          x: Math.max(position.x, nextX) + radius,
          z: Math.max(position.z, nextZ) + radius
        }
      }];
    });
  }

  setFrameSink(sink: WorldFrameSink | null): void {
    if (this.#frameSinkBatch.length > 0) {
      throw new Error("Cannot replace the world frame sink while frames are pending");
    }
    this.#frameSink = sink;
  }

  recoverInterruptedCommands(): WorldSnapshot | null {
    if (this.#commands.size === 0) return null;
    this.#pendingBasePosition = null;
    this.#leftWheel.velocity = 0;
    this.#rightWheel.velocity = 0;
    this.#gripperDirection = 0;
    this.#jointTargets = { ...this.#joints };
    this.#zeroJointVelocities([
      "head_yaw",
      "head_pitch",
      "shoulder",
      "elbow",
      "wrist",
      "gripper_aperture"
    ]);
    if (this.#navigationStatus.status === "executing") {
      this.#recordActualPath(true);
      this.#navigationStatus.status = "stopped";
    }
    for (const commandId of this.#commands.ids()) {
      this.#completeCommand(commandId, "command_interrupted", false);
    }
    return this.snapshot();
  }

  /** Backward-compatible singular API for checkpoints written before v3 multi-command state. */
  recoverInterruptedCommand(): WorldSnapshot | null {
    return this.recoverInterruptedCommands();
  }

  snapshot(): WorldSnapshot {
    const robotPosition = vector(this.#robot.translation());
    const links = Object.fromEntries([...this.#linkBodies.entries()].map(([id, body]) => [
      id,
      {
        position: vector(body.translation()),
        rotation: quaternion(body.rotation()),
        linear_velocity: vector(body.linvel()),
        angular_velocity: vector(body.angvel())
      }
    ]));
    const jointStatus = Object.fromEntries((Object.keys(this.#joints) as RobotJointName[]).map((joint) => [
      joint,
      {
        position: this.#joints[joint],
        velocity: this.#jointVelocities[joint],
        target: this.#jointTargets[joint],
        minimum: ROBOT_SPEC.joints[joint].minimum,
        maximum: ROBOT_SPEC.joints[joint].maximum,
        maximum_velocity: ROBOT_SPEC.joints[joint].maximumVelocity
      }
    ]));
    const snapshot: WorldSnapshot = {
      frame: this.#frame,
      simulated_time: this.#simulatedTime,
      world_revision: this.#worldRevision,
      robot: {
        position: robotPosition,
        yaw: this.#yaw,
        joints: { ...this.#joints },
        contacts: {
          left_object_id: this.#leftContact.objectId,
          right_object_id: this.#rightContact.objectId,
          left_force: this.#leftContact.force,
          right_force: this.#rightContact.force
        },
        attachment: this.#attachment.snapshot(),
        odometry: {
          left_wheel: { ...this.#leftWheel },
          right_wheel: { ...this.#rightWheel }
        },
        links,
        joint_status: jointStatus,
        gripper: {
          aperture: this.#joints.gripper_aperture,
          target_aperture: this.#jointTargets.gripper_aperture,
          maximum_force: this.#gripperMaximumForce,
          left_contact_object_id: this.#leftContact.objectId,
          right_contact_object_id: this.#rightContact.objectId,
          left_contact_force: this.#leftContact.force,
          right_contact_force: this.#rightContact.force
        }
      },
      objects: [...this.#objects.values()].map((object) => ({
        id: object.config.id,
        kind: object.config.kind,
        color: object.config.color,
        position: vector(object.body.translation()),
        rotation: quaternion(object.body.rotation()),
        linear_velocity: vector(object.body.linvel()),
        angular_velocity: vector(object.body.angvel()),
        size: { ...object.config.size },
        portable: object.config.portable,
        locked: object.locked,
        container_id: object.config.container_id ?? null,
        enabled: object.body.isEnabled(),
        visible: this.#entityVisibility.isVisible(object)
      })),
      zones: structuredClone(this.#scenario.zones),
      obstacles: structuredClone(this.#scenario.obstacles),
      explored: this.#explored.state(),
      voxels: this.#voxelStore && this.#voxelChunks
        ? this.#voxelStore.snapshot(this.#voxelChunks.loadRadiusChunks)
        : null,
      navigation: structuredClone(this.#navigationStatus),
      plans: {
        base: this.#navigationPlans.valid(this.#worldRevision).map((plan) => ({
          id: plan.id,
          created_revision: plan.createdRevision,
          target: structuredClone(plan.target),
          face: plan.face ? structuredClone(plan.face) : null,
          waypoints: structuredClone(plan.waypoints),
          distance: plan.distance
        })),
        arm: this.#armPlans.valid(this.#worldRevision).map(snapshotArmPlan)
      },
      affordance_events: structuredClone(this.#affordanceEvents),
      last_command: this.#commands.last,
      active_command: this.#focusedCommand(),
      active_commands: this.#commands.snapshot()
    };
    return snapshot;
  }

  observe(): JsonValue {
    const state = this.snapshot() as unknown as {
      frame: number;
      robot: JsonValue;
      objects: Array<{ visible: boolean } & Record<string, JsonValue>>;
      zones: JsonValue;
      obstacles: JsonValue;
    };
    const head = this.#linkBodies.get("sensor_head");
    return {
      frame: state.frame,
      sensor: {
        id: "head_sensor",
        position: head ? vector(head.translation()) : null,
        rotation: head ? quaternion(head.rotation()) : null,
        maximum_range: this.#scenario.visibility_radius,
        horizontal_field_of_view: ROBOT_SPEC.sensorHead.horizontalFieldOfView,
        vertical_field_of_view: ROBOT_SPEC.sensorHead.verticalFieldOfView
      },
      robot: state.robot,
      visible_objects: state.objects.filter((object) => object.visible),
      known_zones: state.zones,
      known_static_obstacles: state.obstacles,
      ...(this.#scenario.terrain
        ? {
            exploration: {
              cells_seen: this.#explored.seen,
              cells_total: this.#explored.total,
              fraction: this.#explored.total === 0
                ? 1
                : Number((this.#explored.seen / this.#explored.total).toFixed(4))
            }
          }
        : {})
    } as JsonValue;
  }

  /**
   * The terrain around the robot as a local height map, plus where the unseen
   * parts of the world lie.
   *
   * A world wider than the sensor cannot be navigated from `sense_scene` alone:
   * that reports the entities in view, and in a mostly-unseen world the answer
   * is usually none, which tells an agent nothing about which way to go. This
   * answers the question exploration actually poses — what is the ground like
   * here, and where has nobody looked yet — using only what the robot has in
   * fact seen, so it is a memory of the world rather than a map of it.
   */
  surveyTerrain(radiusCells: number): CommandResult {
    const terrain = this.#voxelStore?.projectedTerrain() ?? this.#scenario.terrain;
    if (!terrain) {
      return denied("terrain_unavailable", {
        recovery: "This world is not voxel terrain. Use sense_scene for its entities and "
          + "static obstacles."
      } as JsonValue);
    }
    try {
      return accepted("terrain_survey", buildTerrainSurvey({
        terrain,
        robotPosition: vector(this.#robot.translation()),
        radiusCells,
        isExplored: (index) => this.#explored.has(index),
        exploredCount: this.#explored.seen,
        exploredTotal: this.#explored.total,
        motionSeed: this.#scenario.motion_seed,
        worldRevision: this.#worldRevision,
        robotYaw: this.#yaw,
        projectWalkable: (candidates) => this.#navigation.walkableProjections(
          candidates,
          this.#navigationObstacles(),
          12
        )
      }) as unknown as JsonValue);
    } catch (error) {
      return denied("navigation_projection_failed", {
        error: errorMessage(error),
        recovery: "The navigation authority could not validate frontier targets. Re-observe the "
          + "world before planning; no frontier choice was inferred or executed."
      });
    }
  }

  voxelMaterialAt(coordinate: VoxelCoordinate): VoxelMaterial | null {
    return this.#voxelStore?.materialAt(coordinate) ?? null;
  }

  /** Read-only lifetime check used when rebuilding model context from stored
   * planning receipts. It exposes no internal plan payload and never consumes
   * or executes a plan. */
  planStatus(
    kind: PlanKind,
    planId: string
  ): "valid" | "unknown" | "consumed" | "stale" {
    return (kind === "base" ? this.#navigationPlans : this.#armPlans)
      .lookup(planId, this.#worldRevision).status;
  }

  scanVoxels(radius: number, limit: number): CommandResult {
    return this.#voxelInteraction?.scan(radius, limit, this.#frame, this.#worldRevision)
      ?? denied("voxel_world_unavailable", {});
  }

  inspectVoxel(coordinate: VoxelCoordinate): CommandResult {
    if (!this.#voxelInteraction || !this.#voxelStore) {
      return denied("voxel_world_unavailable", {});
    }
    const inspected = this.#voxelInteraction.inspect(
      coordinate,
      this.#frame,
      this.#worldRevision
    );
    if (!inspected.accepted
      || typeof inspected.detail !== "object"
      || inspected.detail === null
      || Array.isArray(inspected.detail)) return inspected;
    const voxelRadius = this.#voxelStore.terrain.cell * Math.SQRT1_2;
    const interactions = voxelInteractionTargets(inspected.detail);
    return accepted(inspected.code, {
      ...inspected.detail,
      reachable_standoff_poses: this.#reachableStandoffPoses(
        this.#voxelStore.centerOf(coordinate),
        voxelRadius,
        true,
        // A voxel interaction point can sit well above its centre. The
        // smallest navmesh-safe ring keeps the shoulder inside the arm's full
        // 3D reach; the generic folded-arm working ring can be too far away
        // even though it remains convenient for ordinary object inspection.
        false,
        interactions
      )
    } as JsonValue);
  }

  async breakVoxel(command: SourceCommand, coordinate: VoxelCoordinate): Promise<CommandResult> {
    if (!this.#voxelStore || !this.#voxelChunks || !this.#voxelInteraction) {
      return denied("voxel_world_unavailable", {});
    }
    const interaction = this.#voxelInteraction.validate(coordinate, true);
    if (!interaction.accepted) return interaction;
    this.#begin(command, {
      coordinate,
      position: this.#voxelStore.centerOf(coordinate),
      operation: "break"
    } as unknown as JsonValue);
    await this.#advance(command.id, "contacting_voxel");
    command.signal?.throwIfAborted();
    try {
      const result = this.#voxelStore.breakBlock(coordinate, {
        commandId: command.id,
        agentId: command.agentId
      });
      this.#voxelChunks.rebuild(result.chunk);
      await this.#synchronizeVoxelRuntime();
      await this.#advance(command.id, "voxel_removed");
      await this.#flushFrameSink();
      command.signal?.throwIfAborted();
      this.#completeCommand(command.id, "voxel_broken", true);
      return accepted("voxel_broken", {
        mutation: result.mutation,
        chunk: result.chunk,
        inventory: this.#voxelStore.inventory(),
        physics_updated: true,
        navigation_updated: true
      } as unknown as JsonValue);
    } catch (error) {
      return this.#finishVoxelFailure(command, error);
    }
  }

  async placeVoxel(
    command: SourceCommand,
    coordinate: VoxelCoordinate,
    material: VoxelMaterial
  ): Promise<CommandResult> {
    if (!this.#voxelStore || !this.#voxelChunks || !this.#voxelInteraction) {
      return denied("voxel_world_unavailable", {});
    }
    const interaction = this.#voxelInteraction.validate(coordinate, false);
    if (!interaction.accepted) return interaction;
    const obstruction = this.#voxelInteraction.placementObstructions(coordinate);
    if (obstruction.length > 0) {
      return denied("voxel_placement_blocked", {
        coordinate,
        collisions: obstruction,
        recovery: "Move the robot, arm, payload, or object clear of the requested block volume."
      } as unknown as JsonValue);
    }
    this.#begin(command, {
      coordinate,
      position: this.#voxelStore.centerOf(coordinate),
      operation: "place",
      material
    } as unknown as JsonValue);
    await this.#advance(command.id, "positioning_voxel");
    command.signal?.throwIfAborted();
    try {
      const result = this.#voxelStore.placeBlock(coordinate, material, {
        commandId: command.id,
        agentId: command.agentId
      });
      this.#voxelChunks.rebuild(result.chunk);
      await this.#synchronizeVoxelRuntime();
      await this.#advance(command.id, "voxel_placed");
      await this.#flushFrameSink();
      command.signal?.throwIfAborted();
      this.#completeCommand(command.id, "voxel_placed", true);
      return accepted("voxel_placed", {
        mutation: result.mutation,
        chunk: result.chunk,
        inventory: this.#voxelStore.inventory(),
        physics_updated: true,
        navigation_updated: true
      } as unknown as JsonValue);
    } catch (error) {
      return this.#finishVoxelFailure(command, error);
    }
  }

  async #finishVoxelFailure(command: SourceCommand, error: unknown): Promise<CommandResult> {
    await this.#flushFrameSink();
    command.signal?.throwIfAborted();
    const code = error instanceof VoxelEditError ? error.code : "voxel_edit_failed";
    const detail = error instanceof VoxelEditError
      ? error.detail
      : { error: errorMessage(error) };
    this.#completeCommand(command.id, code, false);
    return denied(code, detail as JsonValue);
  }

  inspectEntity(entityId: string): CommandResult {
    const object = this.#objects.get(entityId);
    if (object) {
      if (!this.#entityVisibility.isVisible(object)) {
        return denied(
          "entity_not_visible",
          this.#entityVisibility.failure(object, this.#yaw)
        );
      }
      const snapshot = this.snapshot().objects.find((candidate) => candidate.id === entityId);
      if (!snapshot) return denied("unknown_entity", this.#unknownEntity(entityId));
      return accepted("entity_state", {
        ...snapshot,
        ...(snapshot.portable ? { grasp_pose: graspPose(snapshot) } : {}),
        reachable_standoff_poses: this.#reachableStandoffPoses(
          snapshot.position,
          Math.max(snapshot.size.x, snapshot.size.z) / 2
        )
      } as unknown as JsonValue);
    }
    const zone = this.#scenario.zones.find((candidate) => candidate.id === entityId);
    if (zone) {
      return accepted("entity_state", {
        entity_type: "zone",
        ...structuredClone(zone),
        reachable_standoff_poses: this.#reachableStandoffPoses(
          zone.center,
          Math.max(zone.size.x, zone.size.z) / 2
        )
      } as unknown as JsonValue);
    }
    const obstacle = this.#scenario.obstacles.find((candidate) => candidate.id === entityId);
    return obstacle
      ? accepted("entity_state", { entity_type: "static_obstacle", ...structuredClone(obstacle) } as JsonValue)
      : denied("unknown_entity", this.#unknownEntity(entityId));
  }

  /**
   * Lists what this tool can actually be asked about.
   *
   * Because the id is the only input, an unknown-id denial must enumerate the
   * inspectable ids and direct robot-state queries to proprioception. Repeating
   * the rejected id alone provides no recovery information.
   */
  #unknownEntity(entityId: string): JsonValue {
    return {
      entity_id: entityId,
      known_objects: [...this.#objects.keys()],
      known_zones: this.#scenario.zones.map((zone) => zone.id),
      known_static_obstacles: this.#scenario.obstacles.map((obstacle) => obstacle.id),
      recovery: "inspect_entity only accepts the object, zone and obstacle ids listed here. "
        + "The robot's own pose, joints, contacts and attachment are not entities — "
        + "read them with read_proprioception."
    } as unknown as JsonValue;
  }

  queryContacts(): CommandResult {
    return accepted("contact_state", {
      frame: this.#frame,
      pairs: this.#robotContactPairs(),
      left: { object_id: this.#leftContact.objectId, force: this.#leftContact.force },
      right: { object_id: this.#rightContact.objectId, force: this.#rightContact.force },
      attachment: {
        object_id: this.#attachment.objectId,
        constraint_id: this.#attachment.constraintId,
        source_command_id: this.#attachment.sourceCommandId
      },
      affordance_events: this.#affordanceEvents
    } as unknown as JsonValue);
  }

  inspectCommand(): CommandResult {
    return accepted("command_state", {
      active: this.#focusedCommand(),
      active_commands: this.#commands.snapshot(),
      last: this.#commands.last,
      navigation: this.#navigationStatus,
      frame: this.#frame
    } as JsonValue);
  }

  /**
   * Standoff poses the base can actually reach around a point, with the
   * face_point to pass straight back to plan_base_path. Radii run from the
   * eroded-navmesh minimum (base footprint clear of the object) out to the
   * arm's planar reach, so every returned pose is both walkable and workable.
   */
  #reachableStandoffPoses(
    around: Vec3,
    objectRadius: number,
    preferAxisAligned = false,
    preferWorkingDistance = true,
    voxelInteractions: readonly VoxelInteractionTarget[] = []
  ): JsonValue {
    const minimum = ROBOT_SPEC.base.footprintRadius + Math.max(objectRadius, 0);
    const armReach = ROBOT_SPEC.arm.upperLength
      + ROBOT_SPEC.arm.forearmLength
      + ROBOT_SPEC.arm.wristLength;
    // The first band leaves room for a folded arm. The navmesh method returns
    // candidates sorted by distance to the entity, so selection below must
    // explicitly retain all bands or the nearest one consumes the whole list.
    const radii = preferWorkingDistance
      ? [
          minimum + armReach / 3,
          minimum,
          minimum + (armReach * 2) / 3,
          minimum + armReach
        ]
      : [
          minimum,
          minimum + armReach / 3,
          minimum + (armReach * 2) / 3,
          minimum + armReach
        ];
    let poses;
    try {
      poses = this.#navigation.reachableStandoffs(around, radii, this.#navigationObstacles());
    } catch {
      return [];
    }
    const selected = selectDiverseStandoffs({
      poses,
      radii,
      around,
      robotPosition: vector(this.#robot.translation()),
      preferAxisAligned,
      limit: 8
    });
    if (voxelInteractions.length > 0) {
      const candidates: VoxelStandoffCandidate[] = selected.map((entry) => ({
        target: entry.pose.position,
        radius: entry.pose.radius,
        distanceToEntity: entry.pose.distance,
        distanceToRobot: entry.distanceToRobot,
        axisAlignmentError: entry.axisAlignmentError
      }));
      return rankVoxelStandoffs({ candidates, interactions: voxelInteractions })
        .map(({ candidate, interaction, metrics, fit }, index) => ({
          target: candidate.target,
          // Facing the exact arm target keeps the planar manipulator aligned;
          // the voxel centre can point the base at a different face.
          face_point: interaction?.interaction_point ?? { ...around },
          interaction_point: interaction?.interaction_point ?? null,
          interaction_normal: interaction?.normal ?? null,
          distance_to_entity: candidate.distanceToEntity,
          distance_to_robot: candidate.distanceToRobot,
          standoff_radius: candidate.radius,
          approach_axis_error: candidate.axisAlignmentError,
          arm_workspace_fit: fit,
          arm_target_distance: metrics?.targetDistanceFromShoulder ?? null,
          maximum_arm_reach: metrics?.maximumArmReach ?? null,
          recommended: index === 0
        })) as unknown as JsonValue;
    }
    return selected.map(({ pose, distanceToRobot, axisAlignmentError }, index) => ({
      target: pose.position,
      face_point: { ...around },
      distance_to_entity: pose.distance,
      distance_to_robot: distanceToRobot,
      standoff_radius: pose.radius,
      approach_axis_error: axisAlignmentError,
      recommended: index === 0
    })) as unknown as JsonValue;
  }

  planBasePath(target: Vec3, face?: Vec3): CommandResult {
    try {
      if (face && planarDistance(target, face) <= 0.02) {
        return denied("invalid_base_face_point", {
          requested_target: target,
          face_point: face,
          error: "target and face_point must differ on the x/z plane; changing only y does not create a facing direction",
          recovery: "Do not use an entity or voxel center as the base target. Re-observe it and "
            + "use one reachable_standoff_pose target/face_point pair exactly as returned."
        });
      }
      const start = vector(this.#robot.translation());
      const plan = this.#navigation.plan(start, target, this.#navigationObstacles());
      if (face && planarDistance(plan.resolvedTarget, face) <= 0.02) {
        // The pair defines a place to stand and a distinct point to face. Equal
        // points have no facing direction; inspection supplies a valid pair.
        return denied("invalid_base_face_point", {
          requested_target: target,
          resolved_target: plan.resolvedTarget,
          face_point: face,
          error: "face_point must differ from the resolved base target on the x/z plane",
          recovery: "target is where the base parks and face_point is what it turns to face, "
            + "so passing the same point for both leaves no facing direction. Use a "
            + "reachable_standoff_pose from inspect_entity: its target is the standoff and "
            + "its face_point is the entity. To face this entity, pass its center as "
            + "face_point and a standoff position as target."
        });
      }
      const resolution = this.#resolveBasePlan(plan.waypoints, face);
      if (!resolution.ok) {
        return denied("base_path_collision", {
          requested_target: target,
          resolved_target: plan.resolvedTarget,
          face: face ?? null,
          issue: resolution.issue,
          ...baseCollisionRecovery(resolution.issue),
          ...this.#carryRecovery(resolution.issue)
        });
      }
      const id = `base_plan_${randomUUID()}`;
      const stored: StoredNavigationPlan = {
        ...plan,
        id,
        createdRevision: this.#worldRevision,
        target: { ...plan.resolvedTarget },
        face: face ? { ...face } : null,
        segments: resolution.segments
      };
      this.#navigationPlans.set(stored);
      this.#navigationStatus = {
        plan_id: id,
        status: "planned",
        target: { ...plan.resolvedTarget },
        face: face ? { ...face } : null,
        waypoints: structuredClone(plan.waypoints),
        waypoint_index: 0,
        distance: plan.distance,
        planned_at_frame: this.#frame,
        actual_path: []
      };
      return accepted("base_path_planned", {
        plan_id: id,
        world_revision: this.#worldRevision,
        requested_target: target,
        resolved_target: plan.resolvedTarget,
        projection_distance: plan.projectionDistance,
        face: face ?? null,
        distance: plan.distance,
        waypoints: plan.waypoints,
        segments: resolution.segments.map((segment) => ({
          waypoint_index: segment.waypointIndex,
          target: segment.target,
          body_yaw: segment.bodyYaw,
          linear_direction: segment.linearSign === 1 ? "forward" : "reverse",
          distance: segment.distance
        }))
      } as JsonValue);
    } catch (error) {
      return denied("base_path_unavailable", {
        requested_target: target,
        face: face ?? null,
        error: errorMessage(error),
        // The navmesh is eroded by the base footprint, so a target near an
        // object is unreachable. Offer the poses that are.
        nearest_reachable_alternatives: this.#reachableStandoffPoses(target, 0, false, false),
        recovery: "Pick one nearest_reachable_alternatives entry and call plan_base_path again with its target and face_point."
      });
    }
  }

  /**
   * Finds arm poses that make one model-selected base route collision-free.
   *
   * This is a planner only: it neither changes joints nor stores a command.
   * Every returned candidate survives the current world's full articulated
   * base sweep and has a continuous collision-checked arm trajectory from the
   * current pose. The model still chooses a candidate and executes it through
   * set_joint_targets before asking for a fresh revision-local base plan.
   */
  planArmRetraction(target: Vec3, face?: Vec3): CommandResult {
    try {
      const start = vector(this.#robot.translation());
      const plan = this.#navigation.plan(start, target, this.#navigationObstacles());
      if (face && planarDistance(plan.resolvedTarget, face) <= 0.02) {
        return denied("invalid_base_face_point", {
          requested_target: target,
          resolved_target: plan.resolvedTarget,
          face_point: face,
          recovery: "Use a reachable_standoff_pose whose target differs from its face_point."
        });
      }

      const currentResolution = this.#resolveBasePlan(plan.waypoints, face);
      if (currentResolution.ok) {
        return accepted("arm_retraction_not_required", {
          requested_target: target,
          resolved_target: plan.resolvedTarget,
          face: face ?? null,
          automatic_actuation: false,
          recovery: "The current full rig already clears this route. Call plan_base_path with the same target and face_point, then execute its accepted receipt."
        } as JsonValue);
      }

      const candidates: JsonValue[] = [];
      for (const candidate of rankedArmRetractions(this.#joints)) {
        const targetState: RobotJointState = { ...this.#joints, ...candidate.targets };
        const baseResolution = this.#resolveBasePlan(plan.waypoints, face, targetState);
        if (!baseResolution.ok) continue;
        const armTrajectory = this.#planArmTrajectory(targetState);
        if (!("waypoints" in armTrajectory)) continue;
        candidates.push({
          choice_id: `arm_retraction_${candidates.length + 1}`,
          targets: candidate.targets,
          estimated_joint_travel: candidate.estimatedJointTravel,
          gripper_height: candidate.gripperHeight,
          gripper_planar_radius: candidate.gripperRadius,
          arm_trajectory_waypoints: armTrajectory.waypoints.length,
          base_route_segments: baseResolution.segments.length
        });
        if (candidates.length >= 6) break;
      }

      if (candidates.length === 0) {
        return denied("arm_retraction_unavailable", {
          requested_target: target,
          resolved_target: plan.resolvedTarget,
          face: face ?? null,
          original_issue: currentResolution.issue,
          automatic_actuation: false,
          recovery: "No collision-free arm trajectory clears this exact base route. Select a different reachable_standoff_pose from current inspection evidence, or report_blocked so the parent can grant a different recovery planner."
        } as JsonValue);
      }

      return accepted("arm_retraction_options", {
        requested_target: target,
        resolved_target: plan.resolvedTarget,
        face: face ?? null,
        original_issue: currentResolution.issue,
        candidates,
        automatic_actuation: false,
        decision_owner: "model",
        recovery: "Choose one returned candidate, call set_joint_targets with its exact targets, then call plan_base_path again from the changed world revision. This planner has not moved the robot."
      } as JsonValue);
    } catch (error) {
      return denied("base_path_unavailable", {
        requested_target: target,
        face: face ?? null,
        error: errorMessage(error),
        nearest_reachable_alternatives: this.#reachableStandoffPoses(target, 0, false, false),
        automatic_actuation: false,
        recovery: "Choose a nearest_reachable_alternatives target/face_point pair before requesting arm retraction options."
      });
    }
  }

  #navigationObstacles(): NavigationObstacle[] {
    return [...this.#objects.values()]
      .filter((object) => object.config.portable
        && object.body.isEnabled()
        && object.config.id !== this.#attachment.objectId)
      .map((object) => ({
        id: object.config.id,
        center: vector(object.body.translation()),
        halfExtents: scale(object.config.size, 0.5),
        yaw: quaternionYaw(quaternion(object.body.rotation()))
      }));
  }

  #resolveBasePlan(
    waypoints: Vec3[],
    face?: Vec3,
    joints: RobotJointState = this.#joints
  ): BasePlanResolution {
    return resolveBasePlan({
      waypoints,
      face,
      start: vector(this.#robot.translation()),
      yaw: this.#yaw,
      collisionsAt: (position, yaw) => this.#articulatedCollisions.robot(position, yaw, joints)
    });
  }

  async executeBasePlan(
    command: SourceCommand,
    planId: string,
    options: MotionOptions = {}
  ): Promise<CommandResult> {
    const lookup = this.#navigationPlans.lookup(planId, this.#worldRevision);
    if (lookup.status !== "valid") {
      const denial = planDenialDetail(lookup, planId, "base", this.#worldRevision);
      return denied(denial.code, denial.detail);
    }
    const plan = lookup.plan;
    if (plan.segments === null) {
      const resolution = this.#resolveBasePlan(plan.waypoints, plan.face ?? undefined);
      if (!resolution.ok) {
        return denied("base_path_collision", {
          plan_id: planId,
          requested_target: plan.target,
          resolved_target: plan.resolvedTarget,
          face: plan.face,
          issue: resolution.issue,
          ...baseCollisionRecovery(resolution.issue),
          ...this.#carryRecovery(resolution.issue)
        });
      }
      plan.segments = resolution.segments;
    }
    const segments = plan.segments;
    const requestedVelocity = options.maxVelocity ?? 0.55;
    if (!Number.isFinite(requestedVelocity) || requestedVelocity <= 0) {
      return denied("invalid_base_velocity", { max_velocity: requestedVelocity });
    }
    const maxLinear = Math.min(requestedVelocity, ROBOT_SPEC.base.maximumLinearVelocity);
    const tolerance = boundedTolerance(options.tolerance, 0.04, 0.015, 0.15);
    const maxDuration = options.maxDurationSeconds
      ?? Math.max(8, plan.distance / maxLinear * 3 + Math.PI / ROBOT_SPEC.base.maximumAngularVelocity * 2);
    if (!Number.isFinite(maxDuration) || maxDuration <= 0) {
      return denied("invalid_motion_duration", { max_duration_seconds: maxDuration });
    }

    this.#navigationPlans.consume(planId, this.#worldRevision);
    this.#begin({
      ...command,
      focus: {
        position: plan.target,
        kind: "navigation_target",
        id: planId,
        label: "Base path target"
      }
    }, {
      plan_id: planId,
      target: plan.target,
      face: plan.face,
      waypoints: plan.waypoints
    } as JsonValue);
    this.#navigationStatus = {
      ...this.#navigationStatus,
      plan_id: planId,
      status: "executing",
      waypoint_index: 0,
      actual_path: [vector(this.#robot.translation())]
    };
    command.signal?.throwIfAborted();
    let elapsed = 0;

    for (const segment of segments) {
      this.#navigationStatus.waypoint_index = segment.waypointIndex;
      if (planarDistance(this.#robot.translation(), segment.target) <= tolerance) continue;

      while (Math.abs(normalizeAngle(segment.bodyYaw - this.#yaw)) > 0.02) {
        command.signal?.throwIfAborted();
        if (elapsed >= maxDuration) {
          return this.#finishBaseFailure(
            command,
            "base_plan_timeout",
            {
              plan_id: planId,
              waypoint_index: segment.waypointIndex,
              phase: "turn_to_waypoint",
              elapsed
            }
          );
        }
        const angular = clamp(
          normalizeAngle(segment.bodyYaw - this.#yaw) * 3,
          -ROBOT_SPEC.base.maximumAngularVelocity,
          ROBOT_SPEC.base.maximumAngularVelocity
        );
        const issue = this.#queueBaseVelocity(0, angular);
        if (issue) {
          await this.#advance(command.id, "blocked");
          command.signal?.throwIfAborted();
          return this.#finishBaseFailure(
            command,
            "base_rotation_blocked",
            {
              plan_id: planId,
              waypoint_index: segment.waypointIndex,
              phase: "turn_to_waypoint",
              issue
            }
          );
        }
        await this.#advance(command.id, "turning_to_waypoint");
        elapsed += this.#world.timestep;
      }

      while (planarDistance(this.#robot.translation(), segment.target) > tolerance) {
        command.signal?.throwIfAborted();
        if (elapsed >= maxDuration) {
          return this.#finishBaseFailure(
            command,
            "base_plan_timeout",
            { plan_id: planId, waypoint_index: segment.waypointIndex, elapsed }
          );
        }
        const current = this.#robot.translation();
        const control = controlBaseTowardTarget({
          current: vector(current),
          target: segment.target,
          yaw: this.#yaw,
          linearSign: segment.linearSign,
          maximumLinearVelocity: maxLinear,
          maximumAngularVelocity: ROBOT_SPEC.base.maximumAngularVelocity,
          timestep: this.#world.timestep
        });
        const issue = this.#queueBaseVelocity(
          control.linearVelocity,
          control.angularVelocity
        );
        if (issue) {
          this.#leftWheel.velocity = 0;
          this.#rightWheel.velocity = 0;
          await this.#advance(command.id, "blocked");
          command.signal?.throwIfAborted();
          return this.#finishBaseFailure(
            command,
            "base_plan_blocked",
            { plan_id: planId, waypoint_index: segment.waypointIndex, issue }
          );
        }
        await this.#advance(command.id, "following_path");
        elapsed += this.#world.timestep;
      }
    }

    if (plan.face) {
      // The arm is a planar chain in the base-forward vertical plane, so what
      // facing must achieve is that the face point lies in that plane. Yaw
      // alone does not express it: the same angular error is a larger lateral
      // offset the further away the face point is, and the base never stops
      // exactly on the planned point. So converge on the lateral offset itself,
      // measured live from the actual pose, to below the IK position tolerance.
      while (true) {
        const here = vector(this.#robot.translation());
        const desiredYaw = Math.atan2(plan.face.x - here.x, plan.face.z - here.z);
        const yawError = normalizeAngle(desiredYaw - this.#yaw);
        const lateralOffset = Math.abs(Math.sin(yawError)) * planarDistance(here, plan.face);
        if (lateralOffset <= FACING_LATERAL_TOLERANCE && Math.abs(yawError) < Math.PI / 2) break;
        command.signal?.throwIfAborted();
        if (elapsed >= maxDuration) {
          return this.#finishBaseFailure(
            command,
            "base_plan_timeout",
            { plan_id: planId, phase: "facing", elapsed }
          );
        }
        const angular = clamp(
          yawError * 3,
          -ROBOT_SPEC.base.maximumAngularVelocity,
          ROBOT_SPEC.base.maximumAngularVelocity
        );
        const issue = this.#queueBaseVelocity(0, angular);
        if (issue) {
          await this.#advance(command.id, "blocked");
          command.signal?.throwIfAborted();
          return this.#finishBaseFailure(
            command,
            "base_rotation_blocked",
            { plan_id: planId, issue }
          );
        }
        await this.#advance(command.id, "facing_target");
        elapsed += this.#world.timestep;
      }
    }

    command.signal?.throwIfAborted();
    this.#leftWheel.velocity = 0;
    this.#rightWheel.velocity = 0;
    await this.#advance(command.id, "completed");
    command.signal?.throwIfAborted();
    this.#navigationStatus.status = "completed";
    await this.#flushFrameSink();
    command.signal?.throwIfAborted();
    this.#completeCommand(command.id, "base_plan_completed", true);
    await this.#synchronizeVoxelRuntime();
    command.signal?.throwIfAborted();
    return accepted("base_plan_completed", {
      plan_id: planId,
      final_position: vector(this.#robot.translation()),
      final_yaw: this.#yaw,
      elapsed_simulated_seconds: elapsed + this.#world.timestep
    } as JsonValue);
  }

  async driveBase(
    command: SourceCommand,
    linearMetersPerSecond: number,
    angularRadiansPerSecond: number,
    durationSeconds: number
  ): Promise<CommandResult> {
    if (!Number.isFinite(linearMetersPerSecond)
      || Math.abs(linearMetersPerSecond) > ROBOT_SPEC.base.maximumLinearVelocity) {
      return denied("base_linear_velocity_limit", {
        requested: linearMetersPerSecond,
        maximum: ROBOT_SPEC.base.maximumLinearVelocity
      });
    }
    if (!Number.isFinite(angularRadiansPerSecond)
      || Math.abs(angularRadiansPerSecond) > ROBOT_SPEC.base.maximumAngularVelocity) {
      return denied("base_angular_velocity_limit", {
        requested: angularRadiansPerSecond,
        maximum: ROBOT_SPEC.base.maximumAngularVelocity
      });
    }
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0 || durationSeconds > 5) {
      return denied("base_duration_limit", { requested: durationSeconds, maximum: 5 });
    }

    this.#begin(command, {
      linear_meters_per_second: linearMetersPerSecond,
      angular_radians_per_second: angularRadiansPerSecond,
      duration_seconds: durationSeconds
    });
    this.#navigationStatus = {
      ...idleNavigation(),
      status: "executing",
      planned_at_frame: this.#frame,
      actual_path: [vector(this.#robot.translation())]
    };
    command.signal?.throwIfAborted();
    const steps = Math.ceil(durationSeconds / this.#world.timestep);
    for (let step = 0; step < steps; step += 1) {
      command.signal?.throwIfAborted();
      const issue = this.#queueBaseVelocity(linearMetersPerSecond, angularRadiansPerSecond);
      if (issue) {
        this.#leftWheel.velocity = 0;
        this.#rightWheel.velocity = 0;
        await this.#advance(command.id, "blocked");
        command.signal?.throwIfAborted();
        return this.#finishBaseFailure(command, "base_motion_blocked", { issue });
      }
      await this.#advance(command.id, step === steps - 1 ? "completed" : "driving");
    }
    command.signal?.throwIfAborted();
    this.#leftWheel.velocity = 0;
    this.#rightWheel.velocity = 0;
    this.#navigationStatus.status = "completed";
    await this.#flushFrameSink();
    command.signal?.throwIfAborted();
    this.#completeCommand(command.id, "base_motion_completed", true);
    await this.#synchronizeVoxelRuntime();
    command.signal?.throwIfAborted();
    return accepted("base_motion_completed", {
      final_position: vector(this.#robot.translation()),
      final_yaw: this.#yaw,
      elapsed_simulated_seconds: steps * this.#world.timestep
    } as JsonValue);
  }

  solveEndEffector(target: EndEffectorTarget): CommandResult {
    if (!finiteVector(target.position) || (target.orientation && !finiteQuaternion(target.orientation))) {
      return denied("invalid_end_effector_target", target as unknown as JsonValue);
    }
    const result = solveEndEffectorTarget({
      basePosition: vector(this.#robot.translation()),
      baseYaw: this.#yaw,
      currentJoints: this.#joints,
      target
    });
    if ("code" in result) {
      // A residual is a number, not an explanation. The world knows where the
      // shoulder is and how long the arm is, so it can say whether the target
      // was simply out of reach from this base pose — which is the difference
      // between "move the base first" and "pick a different grasp point".
      return denied(result.code, {
        ...(result.detail as Record<string, JsonValue>),
        ...armReachDiagnosis({
          base: vector(this.#robot.translation()),
          yaw: this.#yaw,
          target: target.position
        })
      } as JsonValue);
    }
    const trajectory = this.#planArmTrajectory(result.joints);
    if (!("waypoints" in trajectory)) {
      const collisions = this.#articulatedCollisions.arm({ ...this.#joints, ...result.joints }, true);
      const endpointBlocked = !collisionTransitionAllowed(
        this.#articulatedCollisions.arm(this.#joints, true),
        collisions
      );
      return denied(endpointBlocked ? "ik_trajectory_endpoint_blocked" : "ik_trajectory_blocked", {
        target: target as unknown as JsonValue,
        joints: result.joints,
        collisions: collisionSetJson(collisions),
        planner: trajectory,
        ...this.#clearanceRecovery(target.position, collisions)
      } as unknown as JsonValue);
    }
    const planId = `arm_plan_${randomUUID()}`;
    this.#armPlans.set({
      id: planId,
      createdRevision: this.#worldRevision,
      kind: "end_effector",
      target: structuredClone(target),
      joints: { ...result.joints },
      waypoints: structuredClone(trajectory.waypoints)
    });
    return accepted("end_effector_solution", {
      plan_id: planId,
      world_revision: this.#worldRevision,
      target: target as unknown as JsonValue,
      joint_targets: result.joints,
      trajectory_waypoints: trajectory.waypoints,
      trajectory_direct: trajectory.direct,
      trajectory_checked_states: trajectory.checked_states,
      achieved_position: result.achievedPosition,
      achieved_orientation: result.achievedOrientation,
      position_error: result.positionError,
      orientation_error: result.orientationError,
      solver_status: result.status
    } as unknown as JsonValue);
  }

  /**
   * Plan a model-selected posture in robot joint space without actuating it.
   *
   * Unlike an IK plan, this contract deliberately owns no fixed world-space
   * gripper point. The resulting trajectory can therefore execute while the
   * base moves: the arm's relative posture remains meaningful even though the
   * whole robot changes world coordinates.
   */
  planJointTargets(targets: ArmJointTargetInput): CommandResult {
    const normalizedTargets: Partial<ArmJointTargets> = {};
    for (const joint of ["shoulder", "elbow", "wrist"] as const) {
      const value = targets[joint];
      if (value !== undefined) normalizedTargets[joint] = value;
    }
    if (Object.keys(normalizedTargets).length === 0) return denied("empty_joint_target", {});
    const issue = jointLimitIssue(normalizedTargets);
    if (issue) return denied("joint_limit", issue as unknown as JsonValue);

    const targetState: RobotJointState = { ...this.#joints, ...normalizedTargets };
    const trajectory = this.#planArmTrajectory(targetState);
    if (!("waypoints" in trajectory)) {
      const endpointCollisions = this.#articulatedCollisions.arm(targetState, true);
      return denied("joint_trajectory_blocked", {
        targets: normalizedTargets as unknown as JsonValue,
        planner: trajectory,
        collisions: collisionSetJson(endpointCollisions),
        ...this.#jointBlockRecovery(endpointCollisions),
        automatic_actuation: false
      } as unknown as JsonValue);
    }

    const planId = `arm_plan_${randomUUID()}`;
    const joints = {
      shoulder: targetState.shoulder,
      elbow: targetState.elbow,
      wrist: targetState.wrist
    };
    this.#armPlans.set({
      id: planId,
      createdRevision: this.#worldRevision,
      kind: "joint_targets",
      target: null,
      joints,
      waypoints: structuredClone(trajectory.waypoints)
    });
    return accepted("joint_target_plan", {
      plan_id: planId,
      world_revision: this.#worldRevision,
      joint_targets: joints,
      trajectory_waypoints: trajectory.waypoints,
      trajectory_direct: trajectory.direct,
      trajectory_checked_states: trajectory.checked_states,
      trajectory_expanded_states: trajectory.expanded_states,
      coordinate_frame: "robot_joint_space",
      automatic_actuation: false
    } as unknown as JsonValue);
  }

  /**
   * Position-only IK with bounded numerical restarts.
   *
   * A local IK stall is not evidence that a point is unreachable. The current
   * joint posture is tried first, followed by low-discrepancy starts spanning
   * the configured limits. No seed is an action: only a converged solution
   * whose full trajectory passes Rapier is stored, and this method never
   * actuates the arm.
   */
  solveEndEffectorPosition(position: Vec3, preferredSeed?: Partial<ArmIkSeed>): CommandResult {
    const failures = new Map<string, number>();
    const starts: Array<Partial<ArmIkSeed> | undefined> = [preferredSeed];
    starts.push(...armIkSeeds());
    let firstFailure: CommandResult | undefined;

    for (let index = 0; index < starts.length; index += 1) {
      const seed = starts[index];
      const result = this.solveEndEffector({
        position,
        ...(seed ? { seed } : {})
      });
      if (result.accepted) {
        return {
          ...result,
          detail: appendPlannerSearch(result.detail, {
            strategy: "bounded_low_discrepancy_ik_restarts",
            attempts: index + 1,
            automatic_actuation: false
          })
        };
      }
      firstFailure ??= result;
      failures.set(result.code, (failures.get(result.code) ?? 0) + 1);
    }

    const failure = firstFailure ?? denied("ik_not_converged", {});
    return {
      ...failure,
      detail: appendPlannerSearch(failure.detail, {
        strategy: "bounded_low_discrepancy_ik_restarts",
        attempts: starts.length,
        failure_codes: Object.fromEntries(failures),
        automatic_actuation: false,
        recovery: "No collision-free position-only IK solution was found from this base pose. "
          + "Choose a different reachable_standoff_pose from current observation, move the base, "
          + "then inspect the target and solve again."
      })
    };
  }

  /**
   * Says which body the arm is wedged against, and against what.
   *
   * A joint move is refused when the swept pose would drive some link into
   * something. A nearby object can require a different base pose, while ground
   * contact means the requested posture itself lowers a link too far — moving
   * horizontally never moves the ground away. Naming both cases keeps the
   * recovery source-backed without choosing replacement joint angles.
   */
  #jointBlockRecovery(collisions: CollisionIssue[]): Record<string, JsonValue> {
    let worst: CollisionIssue | undefined;
    for (const issue of collisions) {
      if (issue.penetration_depth === null) continue;
      if (!worst || issue.penetration_depth > (worst.penetration_depth ?? 0)) worst = issue;
    }
    if (!worst) return {};
    const blocker = worst.collider_id ?? `the ${worst.collider_kind}`;
    const carried = worst.segment === "attached_payload" && this.#attachment.objectId
      ? `the carried ${this.#attachment.objectId}`
      : `the ${worst.segment}`;
    if (worst.collider_kind === "ground") {
      return {
        recovery: `These joint angles would lower ${carried} into the ground. The ground follows `
          + "every horizontal base pose, so driving elsewhere cannot make this posture valid. "
          + "Choose different in-limit joint targets that keep the named link higher, using "
          + "current proprioception/link transforms to verify the change; the harness will sweep "
          + "the new trajectory before executing it."
      };
    }
    return {
      recovery: `These joint angles would drive ${carried} into ${blocker}, so the arm cannot `
        + `reach them from where the base is standing. Drive the base away from ${blocker} `
        + `first — plan_base_path and execute_base_plan to a pose with more clearance — and `
        + `command the joints again from there. Changing the angles alone will not clear it.`
    };
  }

  /**
   * Turns "the route is blocked" into "the thing you are carrying is too low"
   * when that is what it means.
   *
   * A payload-versus-ground collision is independent of the base destination.
   * Recovery must therefore report the required lift rather than encourage
   * retries with different targets or facing points.
   */
  #carryRecovery(issue: JsonValue): Record<string, JsonValue> {
    const collision = deepestGroundCollision(issue);
    if (!collision || !this.#attachment.objectId) return {};
    const lift = collision.depth + POSITION_TOLERANCE;
    return {
      carry_clearance: {
        payload_id: this.#attachment.objectId,
        ground_penetration_depth: collision.depth,
        required_lift: lift
      } as unknown as JsonValue,
      recovery: `The carried ${this.#attachment.objectId} would scrape the ground by `
        + `${collision.depth.toFixed(3)}m while the base turns, so no base target or face_point `
        + `can make this route drivable. Lift the payload at least ${lift.toFixed(3)}m first — `
        + `solve_end_effector_position for the current grasp point raised by that much and `
        + `execute_joint_plan — then plan the base path again while holding it clear.`
    };
  }

  /**
   * Turns "the arm would hit something" into a distance to move.
   *
   * A surface target can place the wrist inside an object. Penetration depth
   * supplies the minimum correction required to clear the overlap.
   */
  #clearanceRecovery(
    target: Vec3,
    collisions: CollisionIssue[]
  ): Record<string, JsonValue> {
    let deepest: CollisionIssue | undefined;
    for (const issue of collisions) {
      if (issue.collider_kind !== "object" || issue.penetration_depth === null) continue;
      if (!deepest || issue.penetration_depth > (deepest.penetration_depth ?? 0)) deepest = issue;
    }
    if (!deepest || deepest.penetration_depth === null) return {};

    // Clear the overlap and then some: stopping exactly at zero penetration
    // leaves the arm grazing the object, which the same check would reject.
    const clearance = deepest.penetration_depth + POSITION_TOLERANCE;
    return {
      recovery: `The ${deepest.segment} would enter ${deepest.collider_id ?? "an object"} by `
        + `${deepest.penetration_depth.toFixed(3)}m at this target. Raise the requested position.y by `
        + `at least ${clearance.toFixed(3)}m — try y=${(target.y + clearance).toFixed(3)} — so the arm `
        + `stops above the surface instead of inside it, then close the gripper to descend onto it.`
    };
  }

  #planArmTrajectory(target: ArmJointTargets): ReturnType<typeof planArmTrajectory> {
    const baseline = this.#articulatedCollisions.arm(this.#joints, true);
    return planArmTrajectory({
      start: {
        shoulder: this.#joints.shoulder,
        elbow: this.#joints.elbow,
        wrist: this.#joints.wrist
      },
      target,
      bounds: {
        shoulder: ROBOT_SPEC.joints.shoulder,
        elbow: ROBOT_SPEC.joints.elbow,
        wrist: ROBOT_SPEC.joints.wrist
      },
      isPoseValid: (pose) => collisionTransitionAllowed(
        baseline,
        this.#articulatedCollisions.arm({ ...this.#joints, ...pose }, true)
      )
    });
  }

  #verifyEndEffector(target: EndEffectorTarget): CommandResult {
    const achieved = rigTransforms(
      this.#joints,
      vector(this.#robot.translation()),
      this.#yaw
    ).gripper;
    const positionError = vectorLength(subtract(achieved.position, target.position));
    const orientationError = target.orientation
      ? quaternionDistance(achieved.rotation, normalizeQuaternion(target.orientation))
      : null;
    const detail = {
      requested_position: target.position,
      achieved_position: achieved.position,
      position_error: positionError,
      position_tolerance: POSITION_TOLERANCE,
      requested_orientation: target.orientation ?? null,
      achieved_orientation: achieved.rotation,
      orientation_error: orientationError,
      orientation_tolerance: ORIENTATION_TOLERANCE
    } as unknown as JsonValue;
    return positionError <= POSITION_TOLERANCE
      && (orientationError === null || orientationError <= ORIENTATION_TOLERANCE)
      ? accepted("end_effector_verified", detail)
      : denied("end_effector_verification_failed", {
          ...(detail as Record<string, JsonValue>),
          recovery: "The measured tool pose did not match the planned target after physics settled. Re-observe the base and object, then solve a fresh end-effector plan from the current revision."
        });
  }

  async executeJointPlan(
    command: SourceCommand,
    planId: string,
    options: MotionOptions = {}
  ): Promise<CommandResult> {
    const lookup = this.#armPlans.lookup(planId, this.#worldRevision);
    if (lookup.status !== "valid") {
      const denial = planDenialDetail(lookup, planId, "arm", this.#worldRevision);
      return denied(denial.code, denial.detail);
    }
    const plan = lookup.plan;
    this.#armPlans.consume(planId, this.#worldRevision);
    return this.#executeJointTargets(
      { ...command, focus: armPlanFocus(plan) },
      plan.joints,
      options,
      plan.waypoints,
      endEffectorVerificationTarget(plan)
    );
  }

  async executeJointTargets(
    command: SourceCommand,
    targets: ArmJointTargetInput,
    options: MotionOptions = {}
  ): Promise<CommandResult> {
    return this.#executeJointTargets(command, targets, options);
  }

  async #executeJointTargets(
    command: SourceCommand,
    targets: ArmJointTargetInput,
    options: MotionOptions,
    plannedWaypoints?: ArmJointTargets[],
    endEffectorTarget?: EndEffectorTarget
  ): Promise<CommandResult> {
    const normalizedTargets: Partial<ArmJointTargets> = {};
    for (const joint of ["shoulder", "elbow", "wrist"] as const) {
      const value = targets[joint];
      if (value !== undefined) normalizedTargets[joint] = value;
    }
    if (Object.keys(normalizedTargets).length === 0) return denied("empty_joint_target", {});
    const issue = jointLimitIssue(normalizedTargets);
    if (issue) return denied("joint_limit", issue as unknown as JsonValue);
    const targetState: RobotJointState = { ...this.#joints, ...normalizedTargets };
    const planned = plannedWaypoints
      ? {
          waypoints: structuredClone(plannedWaypoints),
          direct: plannedWaypoints.length <= 1,
          expanded_states: 0,
          checked_states: 0
        } satisfies ArmTrajectory
      : this.#planArmTrajectory(targetState);
    if (!("waypoints" in planned)) {
      const endpointCollisions = this.#articulatedCollisions.arm(targetState, true);
      return denied("joint_motion_blocked", {
        targets: normalizedTargets as unknown as JsonValue,
        planner: planned,
        collisions: collisionSetJson(endpointCollisions),
        ...this.#jointBlockRecovery(endpointCollisions)
      } as unknown as JsonValue);
    }
    const maxDuration = options.maxDurationSeconds ?? Math.max(12, planned.waypoints.length * 4);
    if (!Number.isFinite(maxDuration) || maxDuration <= 0) {
      return denied("invalid_motion_duration", { max_duration_seconds: maxDuration });
    }
    const tolerance = boundedTolerance(options.tolerance, 0.006, 0.001, 0.04);
    const requestedVelocity = options.maxVelocity;
    if (requestedVelocity !== undefined && (!Number.isFinite(requestedVelocity) || requestedVelocity <= 0)) {
      return denied("invalid_joint_velocity", { max_velocity: requestedVelocity });
    }

    this.#begin(command, {
      targets: normalizedTargets,
      options,
      trajectory_waypoints: planned.waypoints.length
    } as unknown as JsonValue);
    const armJoints = ["shoulder", "elbow", "wrist"] as const;
    for (const joint of armJoints) this.#jointTargets[joint] = targetState[joint];
    command.signal?.throwIfAborted();
    let elapsed = 0;
    for (const [waypointIndex, waypoint] of planned.waypoints.entries()) {
      const waypointState: RobotJointState = { ...this.#joints, ...waypoint };
      while (!armTargetsReached(this.#joints, waypointState, tolerance)) {
        command.signal?.throwIfAborted();
        if (elapsed >= maxDuration) {
          return this.#finishJointFailure(
            command,
            "joint_motion_timeout",
            { targets: normalizedTargets, waypoint_index: waypointIndex, elapsed }
          );
        }
        const durations = armJoints.map((joint) => {
          const velocity = Math.min(
            requestedVelocity ?? ROBOT_SPEC.joints[joint].maximumVelocity,
            ROBOT_SPEC.joints[joint].maximumVelocity
          );
          return Math.abs(waypointState[joint] - this.#joints[joint]) / velocity;
        });
        const remaining = Math.max(...durations, this.#world.timestep);
        const fraction = Math.min(1, this.#world.timestep / remaining);
        const next = { ...this.#joints };
        for (const joint of armJoints) {
          next[joint] = this.#joints[joint]
            + (waypointState[joint] - this.#joints[joint]) * fraction;
          this.#jointVelocities[joint] = (next[joint] - this.#joints[joint]) / this.#world.timestep;
        }
        const collisions = this.#articulatedCollisions.arm(next, true);
        if (!collisionTransitionAllowed(
          this.#articulatedCollisions.arm(this.#joints, true),
          collisions
        )) {
          this.#zeroJointVelocities(["shoulder", "elbow", "wrist"]);
          await this.#advance(command.id, "blocked");
          command.signal?.throwIfAborted();
          return this.#finishJointFailure(
            command,
            "joint_motion_blocked",
            {
              targets: normalizedTargets as unknown as JsonValue,
              waypoint_index: waypointIndex,
              collisions: collisionSetJson(collisions),
              ...this.#jointBlockRecovery(collisions)
            }
          );
        }
        this.#joints = next;
        elapsed += this.#world.timestep;
        const final = waypointIndex === planned.waypoints.length - 1
          && armTargetsReached(this.#joints, waypointState, tolerance);
        await this.#advance(
          command.id,
          final ? "completed" : `trajectory_${waypointIndex + 1}_of_${planned.waypoints.length}`
        );
      }
    }
    command.signal?.throwIfAborted();
    this.#zeroJointVelocities(["shoulder", "elbow", "wrist"]);
    await this.#flushFrameSink();
    command.signal?.throwIfAborted();
    const verification = endEffectorTarget
      ? this.#verifyEndEffector(endEffectorTarget)
      : null;
    if (verification && !verification.accepted) {
      this.#completeCommand(command.id, "end_effector_verification_failed", false);
      return denied("end_effector_verification_failed", verification.detail);
    }
    this.#completeCommand(command.id, "joint_targets_reached", true);
    return accepted("joint_targets_reached", {
      joints: {
        shoulder: this.#joints.shoulder,
        elbow: this.#joints.elbow,
        wrist: this.#joints.wrist
      },
      elapsed_simulated_seconds: elapsed,
      trajectory: {
        direct: planned.direct,
        waypoint_count: planned.waypoints.length,
        expanded_states: planned.expanded_states,
        checked_states: planned.checked_states
      },
      ...(verification ? { end_effector_verification: verification.detail } : {})
    });
  }

  async setHeadTarget(
    command: SourceCommand,
    yaw: number,
    pitch: number,
    options: MotionOptions = {}
  ): Promise<CommandResult> {
    const issue = jointLimitIssue({ head_yaw: yaw, head_pitch: pitch });
    if (issue) return denied("head_joint_limit", issue as unknown as JsonValue);
    const maxDuration = options.maxDurationSeconds ?? 6;
    const tolerance = boundedTolerance(options.tolerance, 0.004, 0.001, 0.03);
    const requestedVelocity = options.maxVelocity;
    if (!Number.isFinite(maxDuration) || maxDuration <= 0
      || (requestedVelocity !== undefined && (!Number.isFinite(requestedVelocity) || requestedVelocity <= 0))) {
      return denied("invalid_head_motion_options", options as unknown as JsonValue);
    }
    this.#begin(command, { yaw, pitch, options } as unknown as JsonValue);
    this.#jointTargets.head_yaw = yaw;
    this.#jointTargets.head_pitch = pitch;
    command.signal?.throwIfAborted();
    let elapsed = 0;
    while (Math.abs(this.#joints.head_yaw - yaw) > tolerance
      || Math.abs(this.#joints.head_pitch - pitch) > tolerance) {
      command.signal?.throwIfAborted();
      if (elapsed >= maxDuration) {
        this.#zeroJointVelocities(["head_yaw", "head_pitch"]);
        this.#jointTargets.head_yaw = this.#joints.head_yaw;
        this.#jointTargets.head_pitch = this.#joints.head_pitch;
        await this.#flushFrameSink();
        command.signal?.throwIfAborted();
        this.#completeCommand(command.id, "head_motion_timeout", false);
        return denied("head_motion_timeout", { yaw, pitch, elapsed });
      }
      const previousYaw = this.#joints.head_yaw;
      const previousPitch = this.#joints.head_pitch;
      const yawVelocity = Math.min(
        requestedVelocity ?? ROBOT_SPEC.joints.head_yaw.maximumVelocity,
        ROBOT_SPEC.joints.head_yaw.maximumVelocity
      );
      const pitchVelocity = Math.min(
        requestedVelocity ?? ROBOT_SPEC.joints.head_pitch.maximumVelocity,
        ROBOT_SPEC.joints.head_pitch.maximumVelocity
      );
      const next: RobotJointState = {
        ...this.#joints,
        head_yaw: moveTowards(previousYaw, yaw, yawVelocity * this.#world.timestep),
        head_pitch: moveTowards(previousPitch, pitch, pitchVelocity * this.#world.timestep)
      };
      const collisions = this.#articulatedCollisions.head(next);
      if (!collisionTransitionAllowed(
        this.#articulatedCollisions.head(this.#joints),
        collisions
      )) {
        this.#zeroJointVelocities(["head_yaw", "head_pitch"]);
        this.#jointTargets.head_yaw = this.#joints.head_yaw;
        this.#jointTargets.head_pitch = this.#joints.head_pitch;
        await this.#advance(command.id, "blocked");
        command.signal?.throwIfAborted();
        await this.#flushFrameSink();
        command.signal?.throwIfAborted();
        this.#completeCommand(command.id, "head_motion_blocked", false);
        return denied("head_motion_blocked", {
          yaw,
          pitch,
          collisions: collisionSetJson(collisions)
        });
      }
      this.#jointVelocities.head_yaw = (next.head_yaw - previousYaw) / this.#world.timestep;
      this.#jointVelocities.head_pitch = (next.head_pitch - previousPitch) / this.#world.timestep;
      this.#joints = next;
      elapsed += this.#world.timestep;
      const complete = Math.abs(this.#joints.head_yaw - yaw) <= tolerance
        && Math.abs(this.#joints.head_pitch - pitch) <= tolerance;
      await this.#advance(command.id, complete ? "completed" : "orienting_sensor");
    }
    command.signal?.throwIfAborted();
    this.#zeroJointVelocities(["head_yaw", "head_pitch"]);
    await this.#flushFrameSink();
    command.signal?.throwIfAborted();
    this.#completeCommand(command.id, "head_target_reached", true);
    return accepted("head_target_reached", { yaw, pitch, elapsed_simulated_seconds: elapsed });
  }

  async setGripperTarget(
    command: SourceCommand,
    aperture: number,
    maxForce: number = ROBOT_SPEC.gripper.defaultMaximumForce,
    options: MotionOptions = {}
  ): Promise<CommandResult> {
    const issue = jointLimitIssue({ gripper_aperture: aperture });
    if (issue) return denied("gripper_joint_limit", issue as unknown as JsonValue);
    if (!Number.isFinite(maxForce) || maxForce <= 0) {
      return denied("invalid_gripper_force", { max_force: maxForce });
    }
    const maxDuration = options.maxDurationSeconds ?? 5;
    const tolerance = boundedTolerance(options.tolerance, 0.0005, 0.0001, 0.02);
    const velocity = Math.min(
      options.maxVelocity ?? ROBOT_SPEC.joints.gripper_aperture.maximumVelocity,
      ROBOT_SPEC.joints.gripper_aperture.maximumVelocity
    );
    if (!Number.isFinite(maxDuration) || maxDuration <= 0 || !Number.isFinite(velocity) || velocity <= 0) {
      return denied("invalid_gripper_motion_options", options as unknown as JsonValue);
    }

    this.#begin(command, { aperture, max_force: maxForce, options } as unknown as JsonValue);
    this.#gripperMaximumForce = maxForce;
    this.#gripperDirection = aperture < this.#joints.gripper_aperture ? -1
      : aperture > this.#joints.gripper_aperture ? 1
        : 0;
    this.#jointTargets.gripper_aperture = aperture;
    if (this.#gripperDirection > 0 && this.#attachment.attached) this.#removeAttachment();
    command.signal?.throwIfAborted();
    let elapsed = 0;
    while (Math.abs(this.#joints.gripper_aperture - aperture) > tolerance) {
      command.signal?.throwIfAborted();
      if (elapsed >= maxDuration) {
        this.#gripperDirection = 0;
        this.#jointVelocities.gripper_aperture = 0;
        this.#jointTargets.gripper_aperture = this.#joints.gripper_aperture;
        await this.#flushFrameSink();
        command.signal?.throwIfAborted();
        this.#completeCommand(command.id, "gripper_motion_timeout", false);
        return denied("gripper_motion_timeout", { aperture, elapsed });
      }
      const previous = this.#joints.gripper_aperture;
      const next: RobotJointState = {
        ...this.#joints,
        gripper_aperture: moveTowards(previous, aperture, velocity * this.#world.timestep)
      };
      const collisions = this.#articulatedCollisions.gripper(next);
      if (!collisionTransitionAllowed(
        this.#articulatedCollisions.gripper(this.#joints),
        collisions
      )) {
        this.#gripperDirection = 0;
        this.#jointVelocities.gripper_aperture = 0;
        this.#jointTargets.gripper_aperture = this.#joints.gripper_aperture;
        await this.#advance(command.id, "blocked");
        command.signal?.throwIfAborted();
        await this.#flushFrameSink();
        command.signal?.throwIfAborted();
        // A grasped object is what stops the fingers, so the commanded aperture
        // is an intent, not a contract: once both fingers hold the same object
        // the constraint exists and the command achieved what it was for.
        // Reporting that as a failure would hide a real attachment.
        if (this.#attachment.attached) return this.#finishGripperGrasp(command, elapsed);
        this.#completeCommand(command.id, "gripper_motion_blocked", false);
        return denied("gripper_motion_blocked", {
          aperture,
          collisions: collisionSetJson(collisions)
        });
      }
      this.#jointVelocities.gripper_aperture = (
        next.gripper_aperture - previous
      ) / this.#world.timestep;
      this.#joints = next;
      elapsed += this.#world.timestep;
      const complete = Math.abs(this.#joints.gripper_aperture - aperture) <= tolerance;
      await this.#advance(
        command.id,
        complete ? "completed" : this.#gripperDirection < 0 ? "closing" : "opening"
      );
      command.signal?.throwIfAborted();
      if (this.#leftContact.force > maxForce || this.#rightContact.force > maxForce) {
        this.#gripperDirection = 0;
        this.#jointVelocities.gripper_aperture = 0;
        this.#jointTargets.gripper_aperture = this.#joints.gripper_aperture;
        if (this.#attachment.sourceCommandId === command.id) this.#removeAttachment();
        await this.#flushFrameSink();
        command.signal?.throwIfAborted();
        this.#completeCommand(command.id, "gripper_force_limit", false);
        return denied("gripper_force_limit", this.#gripperForceDetail(maxForce));
      }
    }
    if (this.#gripperDirection < 0 && !this.#attachment.attached) {
      for (let step = 0;
        step < ROBOT_SPEC.gripper.minimumContactFrames && !this.#attachment.attached;
        step += 1) {
        command.signal?.throwIfAborted();
        await this.#advance(command.id, "holding_target");
        elapsed += this.#world.timestep;
        if (this.#leftContact.force > maxForce || this.#rightContact.force > maxForce) {
          this.#gripperDirection = 0;
          this.#jointVelocities.gripper_aperture = 0;
          this.#jointTargets.gripper_aperture = this.#joints.gripper_aperture;
          this.#removeAttachment();
          await this.#flushFrameSink();
          command.signal?.throwIfAborted();
          this.#completeCommand(command.id, "gripper_force_limit", false);
          return denied("gripper_force_limit", this.#gripperForceDetail(maxForce));
        }
      }
    }
    command.signal?.throwIfAborted();
    return this.#finishGripperGrasp(command, elapsed);
  }

  /**
   * The fingers pressed harder than the force limit, which happens when the
   * commanded aperture is narrower than the object between them. The world
   * knows both numbers, so it names the aperture that would have held instead
   * of leaving the agent to search blindly.
   */
  #gripperForceDetail(maxForce: number): JsonValue {
    const held = this.#leftContact.objectId ?? this.#rightContact.objectId;
    const object = held ? this.#objects.get(held) : undefined;
    const detail: Record<string, JsonValue> = {
      max_force: maxForce,
      left_force: this.#leftContact.force,
      right_force: this.#rightContact.force,
      contacted_object_id: held ?? null
    };
    if (object) {
      const width = Math.min(object.config.size.x, object.config.size.z);
      detail.contacted_object_width = width;
      detail.recovery = `The fingers closed onto ${held}, which is ${width.toFixed(2)}m across. `
        + `Reopen and close to an aperture near that width instead of to the gripper minimum; `
        + `closing further crushes the object rather than holding it.`;
    }
    return detail as JsonValue;
  }

  /** Settles the gripper where it is and reports the resulting contact and
   * attachment state. Shared by aperture-reached and object-stopped closes. */
  async #finishGripperGrasp(command: SourceCommand, elapsed: number): Promise<CommandResult> {
    this.#gripperDirection = 0;
    this.#jointVelocities.gripper_aperture = 0;
    this.#jointTargets.gripper_aperture = this.#joints.gripper_aperture;
    const attachedObject = this.#attachment.sourceCommandId === command.id
      ? this.#attachment.objectId
      : null;
    let stableFrames = 0;
    while (attachedObject && stableFrames < ROBOT_SPEC.gripper.minimumStableAttachmentFrames) {
      command.signal?.throwIfAborted();
      await this.#advance(command.id, "verifying_grasp");
      elapsed += this.#world.timestep;
      if (this.#attachment.objectId !== attachedObject) {
        await this.#flushFrameSink();
        command.signal?.throwIfAborted();
        this.#completeCommand(command.id, "grasp_unstable", false);
        return denied("grasp_unstable", {
          object_id: attachedObject,
          stable_frames: stableFrames,
          required_stable_frames: ROBOT_SPEC.gripper.minimumStableAttachmentFrames,
          recovery: "The object did not remain constrained through the physical stability window. Re-observe its pose and make a new bilateral grasp."
        });
      }
      stableFrames += 1;
    }
    await this.#flushFrameSink();
    this.#completeCommand(command.id, "gripper_target_reached", true);
    return accepted("gripper_target_reached", {
      aperture: this.#joints.gripper_aperture,
      left_object_id: this.#leftContact.objectId,
      right_object_id: this.#rightContact.objectId,
      attachment_object_id: this.#attachment.objectId,
      constraint_id: this.#attachment.constraintId,
      attachment_stable_frames: stableFrames,
      elapsed_simulated_seconds: elapsed
    });
  }

  #begin(command: SourceCommand, target?: JsonValue): void {
    this.#advancedCommands.delete(command.id);
    this.#commands.begin(command, this.#simulatedTime, target);
    this.#begunCommands.add(command.id);
  }

  async #advance(commandId: string, phase: string): Promise<void> {
    this.#advancedCommands.add(commandId);
    return this.#commands.advance(commandId, phase);
  }

  async #stepCommands(commandIds: string[]): Promise<void> {
    this.#syncRig(false);
    this.#voxelChunks?.synchronize(
      this.#robot.translation(),
      this.#dynamicVoxelResidents()
    );
    this.#world.step();
    this.#pendingBasePosition = null;
    this.#frame += 1;
    this.#simulatedTime += this.#world.timestep;
    this.#refreshContacts();
    this.#updateAttachmentFromContacts();
    this.#validateAttachmentConstraint();
    this.#evaluateAffordances(commandIds);
    if (this.#navigationStatus.status === "executing") {
      const base = commandIds
        .map((commandId) => this.#commands.get(commandId))
        .find((command) => command?.channels.includes("base"));
      const terminalPosition = base?.phase === "completed" || base?.phase === "blocked";
      this.#recordActualPath(terminalPosition);
    }
    await this.#paceCommand();
    if (this.#frameSink) {
      const snapshot = this.snapshot();
      this.#frameSinkBatch.push(snapshot);
      if (this.#frameSinkBatch.length >= 3) await this.#flushFrameSink();
    }
  }

  async #flushFrameSink(): Promise<void> {
    if (!this.#frameSink || this.#frameSinkBatch.length === 0) return;
    const batch = this.#frameSinkBatch;
    this.#frameSinkBatch = [];
    await this.#frameSink(batch);
  }

  async #paceCommand(): Promise<void> {
    if (!this.#frameSink) return;
    const remaining = this.#commands.paceDelayMs(this.#simulatedTime);
    if (remaining > 1) await delay(remaining);
  }

  #completeCommand(commandId: string, resultCode: string, acceptedResult: boolean): void {
    // Restored active commands predate this process-local set. Their checkpoint
    // may already contain partial motion at the old revision, so completion on
    // recovery must conservatively publish a new identity. Commands begun in
    // this process are marked only when they actually request a physics frame.
    const physicalStateMayHaveChanged = this.#advancedCommands.has(commandId)
      || !this.#begunCommands.has(commandId);
    const command = this.#commands.complete(
      commandId,
      resultCode,
      acceptedResult,
      this.#frame,
      this.#simulatedTime
    );
    if (!command) return;
    this.#begunCommands.delete(commandId);
    this.#advancedCommands.delete(commandId);
    this.#zeroCommandVelocities(command.channels);
    if (this.#commands.size === 0) {
      this.#zeroRigVelocities();
    }
    // A no-op command still becomes the last visible command, but it did not
    // change robot/world geometry and therefore cannot invalidate plans or
    // reset revision-scoped repetition guards.
    if (physicalStateMayHaveChanged) {
      this.#worldRevision += 1;
      this.#observeTerrain();
    }
  }

  #zeroRigVelocities(): void {
    for (const body of this.#linkBodies.values()) {
      body.setLinvel(ZERO_VECTOR, true);
      body.setAngvel(ZERO_VECTOR, true);
    }
  }

  #zeroCommandVelocities(channels: BodyChannel[]): void {
    if (channels.includes("base")) {
      this.#leftWheel.velocity = 0;
      this.#rightWheel.velocity = 0;
    }
    if (channels.includes("head")) this.#zeroJointVelocities(["head_yaw", "head_pitch"]);
    if (channels.includes("arm")) this.#zeroJointVelocities(["shoulder", "elbow", "wrist"]);
    if (channels.includes("gripper")) {
      this.#zeroJointVelocities(["gripper_aperture"]);
      this.#gripperDirection = 0;
    }
  }

  #focusedCommand(): ActiveWorldCommand | null {
    return this.#commands.focused();
  }

  #activeCommandForChannel(channel: BodyChannel): ActiveWorldCommand | undefined {
    return this.#commands.forChannel(channel);
  }

  #recordActualPath(force: boolean): void {
    const position = vector(this.#robot.translation());
    const last = this.#navigationStatus.actual_path.at(-1);
    if (!last
      || planarDistance(last, position) >= 0.04
      || (force && planarDistance(last, position) > 1e-6)) {
      this.#navigationStatus.actual_path.push(position);
    }
  }

  #queueBaseVelocity(linear: number, angular: number): JsonValue | undefined {
    const current = this.#robot.translation();
    const dt = this.#world.timestep;
    const nextYaw = normalizeAngle(this.#yaw + angular * dt);
    const midpointYaw = normalizeAngle(this.#yaw + angular * dt / 2);
    const next = {
      x: current.x + Math.sin(midpointYaw) * linear * dt,
      y: ROBOT_SPEC.base.centerY,
      z: current.z + Math.cos(midpointYaw) * linear * dt
    };
    if (!this.#insideBounds(next)) return { code: "target_out_of_bounds", target: next };
    const collisions = this.#articulatedCollisions.robot(next, nextYaw, this.#joints);
    if (!collisionTransitionAllowed(
      this.#articulatedCollisions.robot(vector(current), this.#yaw, this.#joints),
      collisions
    )) return { collisions: collisionSetJson(collisions) };

    this.#yaw = nextYaw;
    this.#pendingBasePosition = next;
    this.#robot.setNextKinematicTranslation(next);
    this.#robot.setNextKinematicRotation(yawRotation(nextYaw));
    const leftLinear = linear - angular * ROBOT_SPEC.wheels.trackWidth / 2;
    const rightLinear = linear + angular * ROBOT_SPEC.wheels.trackWidth / 2;
    this.#leftWheel.velocity = leftLinear / ROBOT_SPEC.wheels.radius;
    this.#rightWheel.velocity = rightLinear / ROBOT_SPEC.wheels.radius;
    this.#leftWheel.position += this.#leftWheel.velocity * dt;
    this.#rightWheel.position += this.#rightWheel.velocity * dt;
    return undefined;
  }

  async #finishBaseFailure(
    command: SourceCommand,
    code: string,
    detail: JsonValue
  ): Promise<CommandResult> {
    this.#leftWheel.velocity = 0;
    this.#rightWheel.velocity = 0;
    if (this.#navigationStatus.status === "executing") this.#recordActualPath(true);
    this.#navigationStatus.status = "blocked";
    await this.#flushFrameSink();
    command.signal?.throwIfAborted();
    this.#completeCommand(command.id, code, false);
    await this.#synchronizeVoxelRuntime();
    command.signal?.throwIfAborted();
    return denied(code, detail);
  }

  async #finishJointFailure(
    command: SourceCommand,
    code: string,
    detail: JsonValue
  ): Promise<CommandResult> {
    this.#zeroJointVelocities(["shoulder", "elbow", "wrist"]);
    this.#jointTargets.shoulder = this.#joints.shoulder;
    this.#jointTargets.elbow = this.#joints.elbow;
    this.#jointTargets.wrist = this.#joints.wrist;
    await this.#flushFrameSink();
    command.signal?.throwIfAborted();
    this.#completeCommand(command.id, code, false);
    return denied(code, detail);
  }

  #zeroJointVelocities(joints: RobotJointName[]): void {
    for (const joint of joints) this.#jointVelocities[joint] = 0;
  }

  #syncRig(immediate: boolean): void {
    const base = this.#pendingBasePosition ?? vector(this.#robot.translation());
    const transforms = rigTransforms(this.#joints, base, this.#yaw);
    this.#setLinkTransform("torso", transforms.torso, immediate);
    this.#setLinkTransform("sensor_head", transforms.sensorHead, immediate);
    this.#setLinkTransform("upper_arm", transforms.upperArm, immediate);
    this.#setLinkTransform("forearm", transforms.forearm, immediate);
    this.#setLinkTransform("wrist", transforms.wrist, immediate);
    this.#setLinkTransform("gripper", transforms.gripper, immediate);
    this.#setLinkTransform("left_finger", transforms.leftFinger, immediate);
    this.#setLinkTransform("right_finger", transforms.rightFinger, immediate);
  }

  #setLinkTransform(id: string, transform: RigTransform, immediate: boolean): void {
    const body = this.#linkBodies.get(id);
    if (!body) throw new Error(`Robot link is missing: ${id}`);
    if (immediate) {
      body.setTranslation(transform.position, true);
      body.setRotation(transform.rotation, true);
    } else {
      body.setNextKinematicTranslation(transform.position);
      body.setNextKinematicRotation(transform.rotation);
    }
  }

  #refreshContacts(): void {
    this.#leftContact = this.#fingerContact(this.#leftFingerCollider);
    this.#rightContact = this.#fingerContact(this.#rightFingerCollider);
  }

  #robotContactPairs(): RobotContactPair[] {
    const pairs: RobotContactPair[] = [];
    for (const [linkId, linkCollider] of this.#linkColliders) {
      this.#world.contactPairsWith(linkCollider, (otherCollider: Collider) => {
        const otherData = colliderData(otherCollider);
        if (otherData.kind === "robot" && otherData.link_id
          && linkId.localeCompare(otherData.link_id) > 0) return;
        let contactCount = 0;
        let impulse = 0;
        this.#world.contactPair(
          linkCollider,
          otherCollider,
          (manifold: ContactManifold) => {
            contactCount += manifold.numContacts();
            for (let index = 0; index < manifold.numContacts(); index += 1) {
              impulse += Math.abs(manifold.contactImpulse(index));
            }
          }
        );
        if (contactCount === 0) return;
        pairs.push({
          link_id: linkId,
          collider_kind: otherData.kind ?? "unknown",
          collider_id: colliderIdentity(otherData),
          contact_count: contactCount,
          force: impulse / this.#world.timestep
        });
      });
    }
    return pairs.sort((left, right) => collisionKey({
      segment: left.link_id,
      collider_kind: left.collider_kind,
      collider_id: left.collider_id
    }).localeCompare(collisionKey({
      segment: right.link_id,
      collider_kind: right.collider_kind,
      collider_id: right.collider_id
    })));
  }

  #fingerContact(finger: Collider): FingerContact {
    let result: FingerContact = { objectId: null, force: 0 };
    for (const object of this.#objects.values()) {
      let contacts = 0;
      let impulse = 0;
      this.#world.contactPair(finger, object.collider, (manifold: ContactManifold) => {
        contacts += manifold.numContacts();
        for (let index = 0; index < manifold.numContacts(); index += 1) {
          impulse += Math.abs(manifold.contactImpulse(index));
        }
      });
      if (contacts === 0) continue;
      const force = impulse / this.#world.timestep;
      if (result.objectId === null || force > result.force) {
        result = { objectId: object.config.id, force };
      }
    }
    return result;
  }

  #updateAttachmentFromContacts(): void {
    const bilateral = this.#leftContact.objectId !== null
      && this.#leftContact.objectId === this.#rightContact.objectId
      ? this.#leftContact.objectId
      : null;
    if (this.#attachment.attached) {
      if (this.#gripperDirection > 0 && bilateral !== this.#attachment.objectId) {
        this.#removeAttachment();
      }
      return;
    }
    if (this.#gripperDirection >= 0 || !bilateral) {
      this.#contactCandidate = null;
      this.#contactFrames = 0;
      return;
    }
    if (this.#leftContact.force > this.#gripperMaximumForce
      || this.#rightContact.force > this.#gripperMaximumForce) {
      this.#contactCandidate = null;
      this.#contactFrames = 0;
      return;
    }
    if (this.#contactCandidate === bilateral) this.#contactFrames += 1;
    else {
      this.#contactCandidate = bilateral;
      this.#contactFrames = 1;
    }
    if (this.#contactFrames < ROBOT_SPEC.gripper.minimumContactFrames) return;
    const object = this.#objects.get(bilateral);
    const gripperCommand = this.#activeCommandForChannel("gripper");
    if (!object?.config.portable || !gripperCommand) return;
    this.#installAttachment(
      object,
      `${gripperCommand.id}:constraint:${this.#frame}`,
      gripperCommand.id
    );
  }

  #installAttachment(object: SimObject, constraintId: string, sourceCommandId: string): void {
    this.#attachment.install(object.config.id, object.body, constraintId, sourceCommandId);
    this.#contactCandidate = null;
    this.#contactFrames = 0;
  }

  #removeAttachment(): void {
    if (!this.#attachment.remove()) return;
    this.#contactCandidate = null;
    this.#contactFrames = 0;
  }

  #validateAttachmentConstraint(): void {
    const bilateral = this.#leftContact.objectId !== null
      && this.#leftContact.objectId === this.#rightContact.objectId
      ? this.#leftContact.objectId
      : null;
    const slip = this.#attachment.validate(bilateral);
    if (!slip) return;
    this.#contactCandidate = null;
    this.#contactFrames = 0;
    this.#recordAffordanceEvent({
      frame: this.#frame,
      affordance_id: null,
      code: "grasp_slipped",
      entity_id: slip.objectId,
      source_command_id: slip.sourceCommandId,
      detail: {
        position_drift: slip.positionDrift,
        rotation_drift: slip.rotationDrift,
        detection_frames: slip.detectionFrames
      }
    });
  }

  #evaluateAffordances(commandIds: string[]): void {
    const source = ["gripper", "arm", "base", "head"]
      .flatMap((channel) => commandIds
        .map((commandId) => this.#commands.get(commandId))
        .filter((command): command is ActiveWorldCommand =>
          command?.channels.includes(channel as BodyChannel) === true
        ))
      .at(0);
    const outcome = evaluateAffordances({
      affordances: this.#scenario.affordances ?? [],
      objects: new Map([...this.#objects].map(([id, object]) => [id, {
        id,
        position: vector(object.body.translation()),
        rotation: quaternion(object.body.rotation()),
        locked: object.locked,
        keyId: object.config.key_id ?? null
      }])),
      inContact: (first, second) => {
        const left = this.#objects.get(first);
        const right = this.#objects.get(second);
        return left !== undefined
          && right !== undefined
          && this.#collidersContact(left.collider, right.collider);
      },
      frame: this.#frame,
      activeCommandId: source?.id ?? null,
      reportedUnsupported: this.#unsupportedAffordancesReported
    });
    for (const id of outcome.unlocked) {
      const object = this.#objects.get(id);
      if (object) object.locked = false;
    }
    for (const event of outcome.events) this.#recordAffordanceEvent(event);
  }

  #recordAffordanceEvent(event: AffordanceEvent): void {
    this.#affordanceEvents.push(event);
    if (this.#affordanceEvents.length > MAX_AFFORDANCE_EVENTS) this.#affordanceEvents.shift();
  }

  #collidersContact(first: Collider, second: Collider): boolean {
    let contact = false;
    this.#world.contactPair(first, second, (manifold: ContactManifold) => {
      if (manifold.numContacts() > 0) contact = true;
    });
    return contact;
  }

  #insideBounds(target: Vec3): boolean {
    const margin = ROBOT_SPEC.base.footprintRadius;
    return target.x >= margin
      && target.z >= margin
      && target.x <= this.#scenario.bounds.width - margin
      && target.z <= this.#scenario.bounds.depth - margin;
  }

  /**
   * Marks every terrain column the sensor can currently see.
   *
   * Run once per committed command rather than once per physics step: the head
   * sweeps continuously, but a single command's worth of motion changes the
   * visible set by a handful of columns, and casting a ray per column sixty
   * times a second would cost more than the physics does. Marking at command
   * boundaries records the same frontier for a fraction of the work.
   */
  #observeTerrain(): void {
    const terrain = this.#scenario.terrain;
    if (!terrain) return;
    const head = this.#linkBodies.get("sensor_head");
    if (!head) return;
    const origin = vector(head.translation());
    const visible = visibleTerrainCells({
      terrain,
      sensorPosition: origin,
      sensorRotation: quaternion(head.rotation()),
      maximumRange: this.#scenario.visibility_radius,
      horizontalFieldOfView: ROBOT_SPEC.sensorHead.horizontalFieldOfView,
      verticalFieldOfView: ROBOT_SPEC.sensorHead.verticalFieldOfView,
      isExplored: (index) => this.#explored.has(index),
      heightAt: (column, row) => this.#voxelStore?.heightAt(column, row)
        ?? terrain.heights[row * terrain.columns + column]
        ?? 0,
      isOccluded: (direction, maximumDistance) => Boolean(this.#world.castRay(
        new RAPIER.Ray(origin, direction),
        maximumDistance,
        true,
        undefined,
        undefined,
        undefined,
        undefined,
        (collider: Collider) => {
          const data = colliderData(collider);
          return data.kind !== "robot" && data.kind !== "ground";
        }
      ))
    });
    for (const index of visible) {
      this.#explored.mark(index);
    }
  }

  #restore(snapshot: WorldSnapshot): void {
    this.#frame = snapshot.frame;
    this.#worldRevision = snapshot.world_revision;
    this.#simulatedTime = snapshot.simulated_time;
    this.#yaw = snapshot.robot.yaw;
    this.#joints = { ...snapshot.robot.joints };
    for (const joint of Object.keys(this.#joints) as RobotJointName[]) {
      const status = snapshot.robot.joint_status[joint];
      if (!status) throw new Error(`Checkpoint is missing joint status: ${joint}`);
      if (status.position !== this.#joints[joint]) {
        throw new Error(`Checkpoint joint state is inconsistent: ${joint}`);
      }
      this.#jointTargets[joint] = status.target;
      this.#jointVelocities[joint] = status.velocity;
    }
    const gripper = snapshot.robot.gripper;
    const contacts = snapshot.robot.contacts;
    if (gripper.aperture !== this.#joints.gripper_aperture
      || gripper.target_aperture !== this.#jointTargets.gripper_aperture
      || gripper.left_contact_object_id !== contacts.left_object_id
      || gripper.right_contact_object_id !== contacts.right_object_id
      || gripper.left_contact_force !== contacts.left_force
      || gripper.right_contact_force !== contacts.right_force) {
      throw new Error("Checkpoint gripper state is inconsistent");
    }
    this.#leftWheel = { ...snapshot.robot.odometry.left_wheel };
    this.#rightWheel = { ...snapshot.robot.odometry.right_wheel };
    this.#leftContact = {
      objectId: snapshot.robot.contacts.left_object_id,
      force: snapshot.robot.contacts.left_force
    };
    this.#rightContact = {
      objectId: snapshot.robot.contacts.right_object_id,
      force: snapshot.robot.contacts.right_force
    };
    this.#gripperMaximumForce = gripper.maximum_force;
    this.#explored.restore(snapshot.explored);
    this.#navigationStatus = restoredNavigation(snapshot.navigation);
    for (const plan of snapshot.plans.base) {
      if (plan.created_revision !== snapshot.world_revision) {
        throw new Error(`Checkpoint contains a stale base plan: ${plan.id}`);
      }
      this.#navigationPlans.set({
        id: plan.id,
        createdRevision: plan.created_revision,
        target: structuredClone(plan.target),
        face: plan.face ? structuredClone(plan.face) : null,
        waypoints: structuredClone(plan.waypoints),
        distance: plan.distance,
        resolvedTarget: structuredClone(plan.target),
        projectionDistance: 0,
        segments: null
      });
    }
    for (const plan of snapshot.plans.arm) {
      this.#armPlans.set(restoreArmPlan(plan, snapshot.world_revision));
    }
    this.#affordanceEvents = structuredClone(snapshot.affordance_events);
    const activeCommands = snapshot.active_commands.length > 0
      ? snapshot.active_commands
      : snapshot.active_command
        ? [snapshot.active_command]
        : [];
    this.#commands.restore(activeCommands, snapshot.last_command, snapshot.simulated_time);
    this.#robot.setTranslation(snapshot.robot.position, true);
    this.#robot.setRotation(yawRotation(this.#yaw), true);
    if (snapshot.objects.length !== this.#objects.size) {
      throw new Error("Checkpoint object set does not match the scenario");
    }
    const restoredObjectIds = new Set<string>();
    for (const state of snapshot.objects) {
      const object = this.#objects.get(state.id);
      if (!object) throw new Error(`Checkpoint contains unknown object: ${state.id}`);
      if (restoredObjectIds.has(state.id)) {
        throw new Error(`Checkpoint contains duplicate object: ${state.id}`);
      }
      restoredObjectIds.add(state.id);
      if (state.kind !== object.config.kind
        || state.color !== object.config.color
        || state.portable !== object.config.portable
        || state.container_id !== (object.config.container_id ?? null)
        || !sameVector(state.size, object.config.size)) {
        throw new Error(`Checkpoint object definition does not match the scenario: ${state.id}`);
      }
      object.locked = state.locked;
      object.body.setTranslation(state.position, true);
      object.body.setRotation(state.rotation, true);
      object.body.setLinvel(state.linear_velocity, true);
      object.body.setAngvel(state.angular_velocity, true);
      object.body.setEnabled(state.enabled);
    }
    this.#syncRig(true);
    const attachment = snapshot.robot.attachment;
    if (attachment) {
      const object = this.#objects.get(attachment.object_id);
      if (!object) throw new Error(`Checkpoint attachment contains unknown object: ${attachment.object_id}`);
      if (!object.config.portable || !object.body.isEnabled()) {
        throw new Error(`Checkpoint attachment is not a portable enabled object: ${attachment.object_id}`);
      }
      this.#installAttachment(
        object,
        attachment.constraint_id,
        attachment.source_command_id
      );
    }
  }
}

function voxelInteractionTargets(detail: JsonValue): VoxelInteractionTarget[] {
  if (typeof detail !== "object" || detail === null || Array.isArray(detail)) return [];
  const record = detail as Record<string, JsonValue>;
  const raw = Array.isArray(record.exposed_faces)
    ? record.exposed_faces
    : Array.isArray(record.placement_interaction_points)
      ? record.placement_interaction_points
      : [];
  return raw.flatMap((entry): VoxelInteractionTarget[] => {
    const target = voxelInteractionTarget(entry);
    return target ? [target] : [];
  });
}

function voxelInteractionTarget(value: JsonValue): VoxelInteractionTarget | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const record = value as Record<string, JsonValue>;
  const normal = jsonVec3(record.normal);
  const interactionPoint = jsonVec3(record.interaction_point);
  return normal && interactionPoint
    ? { normal, interaction_point: interactionPoint }
    : null;
}

function jsonVec3(value: JsonValue | undefined): Vec3 | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const record = value as Record<string, JsonValue>;
  return typeof record.x === "number"
    && Number.isFinite(record.x)
    && typeof record.y === "number"
    && Number.isFinite(record.y)
    && typeof record.z === "number"
    && Number.isFinite(record.z)
    ? { x: record.x, y: record.y, z: record.z }
    : null;
}

/**
 * Finds the worst payload-versus-ground contact anywhere in a preflight issue.
 *
 * The issue is a tree, not a list: a blocked segment reports the drive
 * candidates it tried, and each candidate carries its own collisions. Searching
 * it generically means the shape of that report can change without this
 * quietly returning nothing.
 */
function deepestGroundCollision(issue: JsonValue): { depth: number } | null {
  let deepest: number | null = null;
  const visit = (node: JsonValue): void => {
    if (Array.isArray(node)) {
      for (const item of node) visit(item);
      return;
    }
    if (typeof node !== "object" || node === null) return;
    const record = node as Record<string, JsonValue>;
    if (
      record.segment === "attached_payload"
      && record.collider_kind === "ground"
      && typeof record.penetration_depth === "number"
      && (deepest === null || record.penetration_depth > deepest)
    ) {
      deepest = record.penetration_depth;
    }
    for (const value of Object.values(record)) visit(value);
  };
  visit(issue);
  return deepest === null ? null : { depth: deepest };
}

/**
 * Turns an articulated preflight rejection into a hierarchy-level recovery.
 *
 * Recast can offer many different base targets, but a route swept with the
 * same extended arm can keep hitting terrain through every one of them. The
 * harness still must not pick a posture for the model; it can, however, name
 * the robot links that made the current posture non-drivable and make clear
 * when the planning leaf needs different authority.
 */
function baseCollisionRecovery(issue: JsonValue): Record<string, JsonValue> {
  const segments = new Set<string>();
  const visit = (node: JsonValue): void => {
    if (Array.isArray(node)) {
      for (const item of node) visit(item);
      return;
    }
    if (typeof node !== "object" || node === null) return;
    const record = node as Record<string, JsonValue>;
    if (typeof record.segment === "string" && typeof record.collider_kind === "string") {
      segments.add(record.segment);
    }
    for (const value of Object.values(record)) visit(value);
  };
  visit(issue);
  if (segments.size === 0) return {};
  const collisionSegments = [...segments].sort();
  const articulated = collisionSegments.filter((segment) =>
    segment !== "base" && segment !== "attached_payload"
  );
  return {
    collision_segments: collisionSegments,
    recovery: articulated.length > 0
      ? `The current articulated posture makes this base route collide through ${articulated.join(", ")}. `
        + "A different route may avoid it; if several targets report the same robot link, stop sampling "
        + "base targets and reconfigure that body channel from current proprioception before planning again. "
        + "For an arm or finger, call plan_arm_retraction for this exact target and face point, let the model "
        + "choose one returned posture, and execute it with set_joint_targets; planning alone does not move "
        + "the link. A leaf without both capabilities must report_blocked so its parent can delegate them."
      : "The base footprint itself intersects this swept route. Choose a collision-free target or use a "
        + "reachable alternative reported by the current world; changing arm joints cannot shrink the base."
  };
}

/**
 * Where the gripper has to be to pick this object up.
 *
 * The arm cannot fold below roughly base height, so a floor-standing object is
 * taken from above: the gripper descends onto its top face with the fingers
 * straddling it. The world derives this pose directly from rig geometry and the
 * measured object size so downstream planning uses the authoritative target.
 *
 * The descent height is half the object plus the part of the finger that has to
 * clear its top face, so the fingers end up alongside the object rather than
 * resting on it.
 */
function graspPose(object: { position: Vec3; size: Vec3 }): JsonValue {
  const approachClearance = Math.max(
    ROBOT_SPEC.gripper.fingerHalfExtents.y * 0.5,
    ROBOT_SPEC.arm.wristHalfExtents.y
  );
  return {
    position: {
      x: object.position.x,
      y: object.position.y + object.size.y / 2 + approachClearance,
      z: object.position.z
    },
    aperture_before_descent: Math.max(object.size.x, object.size.z) * 1.4,
    aperture_to_hold: Math.min(object.size.x, object.size.z) * 0.9,
    note: "Drive the base to one of the reachable_standoff_poses facing this object first, "
      + "then solve this position with no orientation constraint, open to "
      + "aperture_before_descent, and close to aperture_to_hold."
  } as unknown as JsonValue;
}

function appendPlannerSearch(
  detail: JsonValue,
  search: Record<string, unknown>
): JsonValue {
  const numericalSearch = search as JsonValue;
  if (typeof detail === "object" && detail !== null && !Array.isArray(detail)) {
    return { ...detail, numerical_search: numericalSearch };
  }
  return { result: detail, numerical_search: numericalSearch };
}

function accepted(code: string, detail: JsonValue): CommandResult {
  return { accepted: true, code, detail };
}

function denied(code: string, detail: JsonValue): CommandResult {
  return { accepted: false, code, detail };
}

function idleNavigation(): NavigationStatus {
  return {
    plan_id: null,
    status: "idle",
    target: null,
    face: null,
    waypoints: [],
    waypoint_index: null,
    distance: null,
    planned_at_frame: null,
    actual_path: []
  };
}

function restoredNavigation(navigation: NavigationStatus): NavigationStatus {
  const restored = structuredClone(navigation);
  if (restored.status === "executing") {
    restored.status = "stopped";
    restored.plan_id = null;
    restored.waypoint_index = null;
  }
  return restored;
}


function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
