import RAPIER from "@dimforge/rapier3d-compat";
import type { JsonValue, Vec3, VoxelCoordinate } from "../domain/schema.js";
import type { CommandResult } from "./rapier-world.js";
import { colliderData, colliderIdentity } from "./collision.js";
import {
  inverseQuaternion,
  quaternion,
  rotateVector,
  scale,
  subtract,
  vector,
  vectorLength
} from "./geometry.js";
import { ROBOT_SPEC } from "./robot-model.js";
import { VoxelStore } from "./voxel-store.js";
import { armReachMetrics } from "./arm-reach-diagnosis.js";
import {
  armWorkspaceFit,
  VOXEL_AFFORDANCE_CONTRACT_VERSION,
  type ArmWorkspaceFit
} from "./voxel-affordance.js";

type Collider = InstanceType<typeof RAPIER.Collider>;
type RigidBody = InstanceType<typeof RAPIER.RigidBody>;
type World = InstanceType<typeof RAPIER.World>;

export interface VoxelInteractionPoint {
  normal: Vec3;
  interaction_point: Vec3;
  gripper_distance: number | null;
  shoulder_distance: number | null;
  maximum_arm_reach: number;
  arm_motion_plane_error: number | null;
  arm_workspace_fit: ArmWorkspaceFit;
  recommended: boolean;
  ranking_basis: "arm_workspace_then_gripper_distance";
  placement_coordinate?: VoxelCoordinate | null;
  outside_coordinate?: VoxelCoordinate;
}

type UnrankedInteractionPoint = Omit<
  VoxelInteractionPoint,
  | "shoulder_distance"
  | "maximum_arm_reach"
  | "arm_motion_plane_error"
  | "arm_workspace_fit"
  | "recommended"
  | "ranking_basis"
>;

interface ArmOrigin {
  base: Vec3;
  yaw: number;
}

const IDENTITY_ROTATION = { x: 0, y: 0, z: 0, w: 1 } as const;
export const VOXEL_INTERACTION_DISTANCE = 0.24;
export const VOXEL_INTERACTION_CLEARANCE = Math.min(
  VOXEL_INTERACTION_DISTANCE - 0.03,
  Math.hypot(
    ROBOT_SPEC.gripper.fingerHalfExtents.x,
    ROBOT_SPEC.gripper.fingerHalfExtents.y,
    ROBOT_SPEC.gripper.fingerHalfExtents.z
  ) + ROBOT_SPEC.gripper.cornerRadius
);

/**
 * Sensor- and gripper-grounded voxel interaction queries.
 *
 * Mutation and chunk ownership stay in VoxelStore/VoxelChunkPhysics. This
 * class answers only what the embodied robot can currently see and touch, so
 * RapierWorld does not have to carry another independent subsystem inline.
 */
export class VoxelInteraction {
  readonly #world: World;
  readonly #store: VoxelStore;
  readonly #links: Map<string, RigidBody>;
  readonly #visibilityRadius: number;

  constructor(input: {
    world: World;
    store: VoxelStore;
    links: Map<string, RigidBody>;
    visibilityRadius: number;
  }) {
    this.#world = input.world;
    this.#store = input.store;
    this.#links = input.links;
    this.#visibilityRadius = input.visibilityRadius;
  }

  scan(radius: number, limit: number, frame: number, worldRevision: number): CommandResult {
    const head = this.#links.get("sensor_head");
    const gripper = this.#links.get("gripper");
    if (!head || !gripper) return denied("robot_link_state_unavailable", {});
    const origin = vector(head.translation());
    const gripperPosition = vector(gripper.translation());
    const requestedRadius = Math.min(
      this.#visibilityRadius,
      Math.max(this.#store.terrain.cell, radius)
    );
    const requestedLimit = Math.min(48, Math.max(1, Math.trunc(limit)));
    const blocks = this.#store
      .visibleSurfaceBlocks(origin, requestedRadius, requestedLimit * 6)
      .filter((block) => this.#store.editability(block.coordinate).editable)
      .filter((block) => this.#voxelVisible(block.coordinate))
      .slice(0, requestedLimit)
      .map((block) => this.#observation(block.coordinate, gripperPosition));
    return accepted("voxel_scan", {
      affordance_contract_version: VOXEL_AFFORDANCE_CONTRACT_VERSION,
      frame,
      world_revision: worldRevision,
      voxel_revision: this.#store.revision,
      chunk_size: this.#store.terrain.chunk_size,
      loaded_chunks: this.#store.loadedChunks(),
      inventory: this.#store.inventory(),
      blocks
    } as unknown as JsonValue);
  }

  inspect(coordinate: VoxelCoordinate, frame: number, worldRevision: number): CommandResult {
    if (!this.#store.contains(coordinate)) {
      return denied("voxel_out_of_bounds", { coordinate });
    }
    if (!this.#store.isLoaded(coordinate)) {
      return denied("voxel_chunk_unloaded", {
        coordinate,
        chunk: this.#store.chunkAt(coordinate),
        loaded_chunks: this.#store.loadedChunks(),
        recovery: "Move the base toward this chunk before inspecting or editing it."
      } as unknown as JsonValue);
    }
    const gripper = this.#links.get("gripper");
    const gripperPosition = gripper ? vector(gripper.translation()) : null;
    const material = this.#store.materialAt(coordinate);
    const editability = this.#store.editability(coordinate);
    const state = material
      ? this.#observation(coordinate, gripperPosition)
      : {
          coordinate,
          material: null,
          center: this.#store.centerOf(coordinate),
          placement_supported: this.#store.placementSupported(coordinate),
          inventory: this.#store.inventory(),
          interaction_distance: VOXEL_INTERACTION_DISTANCE,
          placement_interaction_points: this.#placementInteractionPoints(
            coordinate,
            gripperPosition
          ),
          editable: editability.editable,
          ...(!editability.editable
            ? { edit_denial: { code: editability.code, ...editability.detail } }
            : {})
        };
    return accepted("voxel_state", {
      ...state,
      affordance_contract_version: VOXEL_AFFORDANCE_CONTRACT_VERSION,
      frame,
      world_revision: worldRevision,
      voxel_revision: this.#store.revision
    } as unknown as JsonValue);
  }

  validate(coordinate: VoxelCoordinate, breaking: boolean): CommandResult {
    const editability = this.#store.editability(coordinate);
    if (!editability.editable) {
      const recovery = editability.code === "voxel_chunk_unloaded"
        ? "Move the base toward the target so its backend chunk enters the active physics set."
        : editability.code === "voxel_boundary_protected"
          ? "Choose a non-boundary block returned by scan_voxels."
          : undefined;
      return denied(editability.code, {
        ...editability.detail,
        ...(recovery ? { recovery } : {})
      } as unknown as JsonValue);
    }
    const material = this.#store.materialAt(coordinate);
    if (breaking && !material) return denied("voxel_empty", { coordinate });
    if (!breaking && material) {
      return denied("voxel_occupied", { coordinate, material } as unknown as JsonValue);
    }
    const visible = breaking
      ? this.#voxelVisible(coordinate)
      : this.#pointVisibleToSensor(this.#store.centerOf(coordinate));
    if (!visible) {
      return denied("voxel_not_visible", {
        coordinate,
        recovery: "Move the base or sensor head until scan_voxels reports this coordinate."
      } as unknown as JsonValue);
    }
    const gripper = this.#links.get("gripper");
    if (!gripper) return denied("robot_link_state_unavailable", { link: "gripper" });
    const gripperPosition = vector(gripper.translation());
    const interactionPoints = breaking
      ? this.#breakingInteractionPoints(coordinate, gripperPosition)
      : this.#placementInteractionPoints(coordinate, gripperPosition);
    // Mutation is a contact check, not a planning recommendation. Always use
    // the physically nearest surface even if another point has a more
    // comfortable future IK workspace.
    const nearest = nearestByGripper(interactionPoints);
    if (!nearest || nearest.gripper_distance === null) {
      return denied("voxel_interaction_surface_unavailable", {
        coordinate,
        recovery: breaking
          ? "Scan or inspect an exposed voxel face before attempting to break it."
          : "Inspect an empty supported coordinate with a clear adjacent approach before placing."
      } as unknown as JsonValue);
    }
    const distance = nearest.gripper_distance;
    if (distance > VOXEL_INTERACTION_DISTANCE) {
      return denied("voxel_out_of_reach", {
        coordinate,
        distance,
        maximum_distance: VOXEL_INTERACTION_DISTANCE,
        nearest_interaction_point: nearest,
        ...(breaking
          ? { exposed_faces: interactionPoints }
          : { placement_interaction_points: interactionPoints }),
        recovery: breaking
          ? "Use a reported exposed_faces interaction_point as solve_end_effector_position.position, execute that arm plan, then retry the voxel edit from the new physical gripper pose."
          : "Inspect the empty target and use a reported placement_interaction_points interaction_point as solve_end_effector_position.position, execute that arm plan, then retry placement from the new physical gripper pose."
      } as unknown as JsonValue);
    }
    if (!breaking && !this.#store.placementSupported(coordinate)) {
      return denied("voxel_unsupported", {
        coordinate,
        recovery: "Choose a placement_coordinate reported on an exposed face of scan_voxels."
      } as unknown as JsonValue);
    }
    return accepted("voxel_interaction_ready", {
      coordinate,
      distance,
      interaction_point: nearest.interaction_point,
      normal: nearest.normal
    });
  }

  placementObstructions(coordinate: VoxelCoordinate): JsonValue[] {
    const collisions: JsonValue[] = [];
    const center = this.#store.centerOf(coordinate);
    const shape = new RAPIER.Cuboid(
      this.#store.terrain.cell * 0.46,
      this.#store.terrain.block * 0.46,
      this.#store.terrain.cell * 0.46
    );
    // Shape queries depend on Rapier's broad-phase query pipeline. That
    // pipeline can legitimately lag an immediate checkpoint restore or a
    // just-written kinematic pose, even though collider.contactShape already
    // sees the authoritative collider transform. Placement is rare and the
    // active chunk set is bounded, so walk colliders and run the narrow-phase
    // contact directly. This prevents a future block from being created around
    // the fingers merely because the broad phase still held their old AABB.
    this.#world.forEachCollider((collider: Collider) => {
      const data = colliderData(collider);
      if (data.kind === "ground" || data.kind === "voxel") return;
      const contact = collider.contactShape(shape, center, IDENTITY_ROTATION, 0);
      if (!contact) return;
      collisions.push({
        collider_kind: data.kind ?? "unknown",
        collider_id: colliderIdentity(data),
        penetration_depth: Number.isFinite(contact.distance)
          ? Math.max(0, -contact.distance)
          : null
      });
    });
    return collisions;
  }

  #placementInteractionPoints(
    coordinate: VoxelCoordinate,
    gripperPosition: Vec3 | null
  ): VoxelInteractionPoint[] {
    const center = this.#store.centerOf(coordinate);
    const halfCell = this.#store.terrain.cell / 2;
    const halfBlock = this.#store.terrain.block / 2;
    const points: UnrankedInteractionPoint[] = PLACEMENT_FACE_NORMALS.flatMap((normal) => {
      const outside = {
        column: coordinate.column + normal.x,
        level: coordinate.level + normal.y,
        row: coordinate.row + normal.z
      };
      if (!this.#store.contains(outside) || this.#store.materialAt(outside) !== null) return [];
      const interactionPoint = {
        x: center.x + normal.x * (halfCell + VOXEL_INTERACTION_CLEARANCE),
        y: center.y + normal.y * (halfBlock + VOXEL_INTERACTION_CLEARANCE),
        z: center.z + normal.z * (halfCell + VOXEL_INTERACTION_CLEARANCE)
      };
      return [{
        normal,
        interaction_point: interactionPoint,
        outside_coordinate: outside,
        gripper_distance: gripperPosition
          ? vectorLength(subtract(interactionPoint, gripperPosition))
          : null
      }];
    });
    return rankInteractionPoints(points, this.#armOrigin());
  }

  #breakingInteractionPoints(
    coordinate: VoxelCoordinate,
    gripperPosition: Vec3 | null
  ): VoxelInteractionPoint[] {
    const points: UnrankedInteractionPoint[] = this.#store.exposedFaces(coordinate).map((face) => {
      const interactionPoint = {
        x: face.point.x + face.normal.x * VOXEL_INTERACTION_CLEARANCE,
        y: face.point.y + face.normal.y * VOXEL_INTERACTION_CLEARANCE,
        z: face.point.z + face.normal.z * VOXEL_INTERACTION_CLEARANCE
      };
      const placement = {
        column: coordinate.column + face.normal.x,
        level: coordinate.level + face.normal.y,
        row: coordinate.row + face.normal.z
      };
      return {
        normal: face.normal,
        interaction_point: interactionPoint,
        gripper_distance: gripperPosition
          ? vectorLength(subtract(interactionPoint, gripperPosition))
          : null,
        placement_coordinate: this.#store.contains(placement)
          && this.#store.materialAt(placement) === null
          && this.#store.editability(placement).editable
          ? placement
          : null
      };
    });
    return rankInteractionPoints(points, this.#armOrigin());
  }

  #observation(
    coordinate: VoxelCoordinate,
    gripperPosition: Vec3 | null
  ): Record<string, unknown> {
    const center = this.#store.centerOf(coordinate);
    const faces = this.#breakingInteractionPoints(coordinate, gripperPosition);
    const editability = this.#store.editability(coordinate);
    return {
      coordinate,
      material: this.#store.materialAt(coordinate),
      center,
      chunk: this.#store.chunkAt(coordinate),
      visible: this.#voxelVisible(coordinate),
      editable: editability.editable,
      ...(!editability.editable
        ? { edit_denial: { code: editability.code, ...editability.detail } }
        : {}),
      interaction_distance: VOXEL_INTERACTION_DISTANCE,
      reachable_by_gripper: faces.some((face) =>
        face.gripper_distance !== null && face.gripper_distance <= VOXEL_INTERACTION_DISTANCE
      ),
      exposed_faces: faces
    };
  }

  #voxelVisible(coordinate: VoxelCoordinate): boolean {
    if (!this.#store.materialAt(coordinate)) return false;
    const head = this.#links.get("sensor_head");
    if (!head) return false;
    const origin = vector(head.translation());
    const face = this.#store.exposedFaces(coordinate)
      .sort((left, right) =>
        vectorLength(subtract(left.point, origin)) - vectorLength(subtract(right.point, origin))
      )[0];
    return face ? this.#pointVisibleToSensor(face.point) : false;
  }

  #pointVisibleToSensor(point: Vec3): boolean {
    const head = this.#links.get("sensor_head");
    if (!head) return false;
    const origin = vector(head.translation());
    const delta = subtract(point, origin);
    const distance = vectorLength(delta);
    if (distance > this.#visibilityRadius || distance <= 1e-6) return false;
    const direction = scale(delta, 1 / distance);
    const local = rotateVector(inverseQuaternion(quaternion(head.rotation())), direction);
    if (Math.abs(Math.atan2(local.x, local.z)) > ROBOT_SPEC.sensorHead.horizontalFieldOfView / 2
      || Math.abs(Math.atan2(local.y, Math.hypot(local.x, local.z)))
        > ROBOT_SPEC.sensorHead.verticalFieldOfView / 2) return false;
    const hit = this.#world.castRay(
      new RAPIER.Ray(origin, direction),
      Math.max(0, distance - 0.035),
      true,
      undefined,
      undefined,
      undefined,
      undefined,
      (collider: Collider) => {
        const data = colliderData(collider);
        return data.kind !== "robot" && data.kind !== "ground";
      }
    );
    return hit === null;
  }

  #armOrigin(): ArmOrigin | null {
    const base = this.#links.get("base");
    if (!base) return null;
    const forward = rotateVector(quaternion(base.rotation()), { x: 0, y: 0, z: 1 });
    return {
      base: vector(base.translation()),
      yaw: Math.atan2(forward.x, forward.z)
    };
  }
}

const PLACEMENT_FACE_NORMALS = [
  { x: 0, y: 1, z: 0 },
  { x: 1, y: 0, z: 0 },
  { x: -1, y: 0, z: 0 },
  { x: 0, y: 0, z: 1 },
  { x: 0, y: 0, z: -1 },
  { x: 0, y: -1, z: 0 }
] as const;

function rankInteractionPoints(
  points: UnrankedInteractionPoint[],
  origin: ArmOrigin | null
): VoxelInteractionPoint[] {
  return points.map((point): Omit<VoxelInteractionPoint, "recommended"> => {
    const metrics = origin
      ? armReachMetrics({ ...origin, target: point.interaction_point })
      : null;
    return {
      ...point,
      shoulder_distance: metrics?.targetDistanceFromShoulder ?? null,
      maximum_arm_reach: metrics?.maximumArmReach ?? (
        ROBOT_SPEC.arm.upperLength
        + ROBOT_SPEC.arm.forearmLength
        + ROBOT_SPEC.arm.wristLength
      ),
      arm_motion_plane_error: metrics?.armMotionPlaneLateralError ?? null,
      arm_workspace_fit: armWorkspaceFit(metrics),
      ranking_basis: "arm_workspace_then_gripper_distance"
    };
  }).sort((left, right) =>
      workspaceRank(left.arm_workspace_fit) - workspaceRank(right.arm_workspace_fit)
      || (left.gripper_distance ?? Number.POSITIVE_INFINITY)
        - (right.gripper_distance ?? Number.POSITIVE_INFINITY)
      || left.interaction_point.x - right.interaction_point.x
      || left.interaction_point.y - right.interaction_point.y
      || left.interaction_point.z - right.interaction_point.z
    )
    .map((point, index) => ({
      ...point,
      recommended: index === 0
    }));
}

function nearestByGripper(
  points: readonly VoxelInteractionPoint[]
): VoxelInteractionPoint | undefined {
  return [...points].sort((left, right) =>
    (left.gripper_distance ?? Number.POSITIVE_INFINITY)
      - (right.gripper_distance ?? Number.POSITIVE_INFINITY)
  )[0];
}

function workspaceRank(fit: ArmWorkspaceFit): number {
  if (fit === "preferred") return 0;
  if (fit === "folded") return 1;
  if (fit === "off_plane") return 2;
  if (fit === "out_of_span") return 3;
  return 4;
}

function accepted(code: string, detail: CommandResult["detail"]): CommandResult {
  return { accepted: true, code, detail };
}

function denied(code: string, detail: CommandResult["detail"]): CommandResult {
  return { accepted: false, code, detail };
}
