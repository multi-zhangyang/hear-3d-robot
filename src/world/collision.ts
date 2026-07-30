import RAPIER from "@dimforge/rapier3d-compat";
import type { JsonValue, Vec3 } from "../domain/schema.js";
import { add, multiplyQuaternion, rotateVector, scale, yawRotation } from "./geometry.js";
import { rigTransforms, type RigTransform } from "./rig.js";
import { ROBOT_SPEC, type RobotJointState } from "./robot-model.js";

type Collider = InstanceType<typeof RAPIER.Collider>;
type Shape = InstanceType<typeof RAPIER.Cuboid>;

/** One swept robot volume tested against the world for a candidate pose. */
export interface CollisionCandidate {
  segment: string;
  transform: RigTransform;
  shape: Shape;
  allowPortableContacts: boolean;
}

/** The payload currently held, as far as collision testing is concerned. */
export interface AttachedPayload {
  size: Vec3;
  anchorPosition: Vec3;
  anchorRotation: { x: number; y: number; z: number; w: number };
}

/**
 * Every robot volume that must be collision-free at the given configuration,
 * derived from ROBOT_SPEC and the rig's forward kinematics alone. A held
 * payload is included as a segment of the robot, because while attached it
 * sweeps through the world exactly like a link does.
 */
export function collisionCandidates(input: {
  joints: RobotJointState;
  base: Vec3;
  yaw: number;
  allowPortableFingerContacts: boolean;
  payload?: AttachedPayload | undefined;
}): CollisionCandidate[] {
  const transforms = rigTransforms(input.joints, input.base, input.yaw);
  const cuboid = (halfExtents: Vec3): Shape =>
    new RAPIER.Cuboid(halfExtents.x, halfExtents.y, halfExtents.z);
  const candidates: CollisionCandidate[] = [
    {
      segment: "base",
      transform: { position: input.base, rotation: yawRotation(input.yaw) },
      shape: roundCuboidShape(ROBOT_SPEC.base.halfExtents, ROBOT_SPEC.base.cornerRadius),
      allowPortableContacts: false
    },
    {
      segment: "torso",
      transform: transforms.torso,
      shape: roundCuboidShape(ROBOT_SPEC.torso.halfExtents, 0.05),
      allowPortableContacts: false
    },
    {
      segment: "sensor_head",
      transform: transforms.sensorHead,
      shape: roundCuboidShape(ROBOT_SPEC.sensorHead.halfExtents, 0.05),
      allowPortableContacts: false
    },
    {
      segment: "upper_arm",
      transform: transforms.upperArm,
      shape: cuboid(ROBOT_SPEC.arm.upperHalfExtents),
      allowPortableContacts: false
    },
    {
      segment: "forearm",
      transform: transforms.forearm,
      shape: cuboid(ROBOT_SPEC.arm.forearmHalfExtents),
      allowPortableContacts: false
    },
    {
      segment: "wrist",
      transform: transforms.wrist,
      shape: cuboid(ROBOT_SPEC.arm.wristHalfExtents),
      allowPortableContacts: false
    },
    {
      segment: "left_finger",
      transform: transforms.leftFinger,
      shape: roundCuboidShape(ROBOT_SPEC.gripper.fingerHalfExtents, ROBOT_SPEC.gripper.cornerRadius),
      allowPortableContacts: input.allowPortableFingerContacts
    },
    {
      segment: "right_finger",
      transform: transforms.rightFinger,
      shape: roundCuboidShape(ROBOT_SPEC.gripper.fingerHalfExtents, ROBOT_SPEC.gripper.cornerRadius),
      allowPortableContacts: input.allowPortableFingerContacts
    }
  ];

  if (input.payload) {
    candidates.push({
      segment: "attached_payload",
      transform: {
        position: add(
          transforms.gripper.position,
          rotateVector(transforms.gripper.rotation, input.payload.anchorPosition)
        ),
        rotation: multiplyQuaternion(transforms.gripper.rotation, input.payload.anchorRotation)
      },
      shape: roundCuboidShape(
        scale(input.payload.size, 0.5),
        Math.min(0.04, input.payload.size.x / 8, input.payload.size.z / 8)
      ),
      allowPortableContacts: false
    });
  }
  return candidates;
}

function roundCuboidShape(halfExtents: Vec3, radius: number): Shape {
  const inner = roundedInnerHalfExtents(halfExtents, radius);
  return new RAPIER.RoundCuboid(inner.x, inner.y, inner.z, radius);
}

export function roundCuboidDesc(
  halfExtents: Vec3,
  radius: number
): InstanceType<typeof RAPIER.ColliderDesc> {
  const inner = roundedInnerHalfExtents(halfExtents, radius);
  return RAPIER.ColliderDesc.roundCuboid(inner.x, inner.y, inner.z, radius);
}

function roundedInnerHalfExtents(halfExtents: Vec3, radius: number): Vec3 {
  if (!Number.isFinite(radius) || radius <= 0
    || radius >= Math.min(halfExtents.x, halfExtents.y, halfExtents.z)) {
    throw new Error("Rounded cuboid radius must be smaller than every half extent");
  }
  return {
    x: halfExtents.x - radius,
    y: halfExtents.y - radius,
    z: halfExtents.z - radius
  };
}

export interface CollisionIssue {
  segment: string;
  collider_kind: string;
  collider_id: string | null;
  penetration_depth: number | null;
  code?: string;
}

export interface ColliderData {
  kind?: string;
  id?: string;
  link_id?: string;
}

export function collisionTransitionAllowed(
  current: CollisionIssue[],
  next: CollisionIssue[]
): boolean {
  const contactTolerance = 1e-5;
  if (next.length === 0) return true;
  const currentByKey = new Map(current.map((collision) => [collisionKey(collision), collision]));
  return next.every((collision) => {
    const previous = currentByKey.get(collisionKey(collision));
    return previous !== undefined
      && previous.penetration_depth !== null
      && collision.penetration_depth !== null
      && (
        // Rapier reports a touching pair as depth 0. Requiring it to become
        // negative would pin a robot that is merely sliding along or turning
        // away from an existing contact. Any measurable penetration increase
        // still fails on the following branch.
        (previous.penetration_depth <= contactTolerance
          && collision.penetration_depth <= contactTolerance)
        || collision.penetration_depth < previous.penetration_depth - 1e-6
      );
  });
}

function collisionJson(collision: CollisionIssue): JsonValue {
  return {
    segment: collision.segment,
    collider_kind: collision.collider_kind,
    collider_id: collision.collider_id,
    penetration_depth: collision.penetration_depth,
    ...(collision.code === undefined ? {} : { code: collision.code })
  };
}

export function collisionSetJson(collisions: CollisionIssue[]): JsonValue[] {
  return collisions.map(collisionJson);
}

export function uniqueCollisions(collisions: CollisionIssue[]): CollisionIssue[] {
  const unique = new Map<string, CollisionIssue>();
  for (const collision of collisions) {
    const key = collisionKey(collision);
    const previous = unique.get(key);
    if (!previous) {
      unique.set(key, collision);
      continue;
    }
    unique.set(key, {
      ...previous,
      penetration_depth: previous.penetration_depth === null || collision.penetration_depth === null
        ? null
        : Math.max(previous.penetration_depth, collision.penetration_depth),
      ...(previous.code === undefined && collision.code !== undefined ? { code: collision.code } : {})
    });
  }
  return [...unique.values()].sort((left, right) =>
    collisionKey(left).localeCompare(collisionKey(right))
  );
}

export function collisionKey(collision: Pick<
  CollisionIssue,
  "segment" | "collider_kind" | "collider_id"
>): string {
  return `${collision.segment}\u0000${collision.collider_kind}\u0000${collision.collider_id ?? ""}`;
}

export function colliderData(collider: Collider): ColliderData {
  const value = collider.parent()?.userData;
  return typeof value === "object" && value !== null
    ? value as ColliderData
    : {};
}

export function colliderIdentity(data: ColliderData): string | null {
  return data.id ?? data.link_id ?? null;
}

export function linkPairKey(left: string, right: string): string {
  return left < right ? `${left}\u0000${right}` : `${right}\u0000${left}`;
}
