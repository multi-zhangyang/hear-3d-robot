import RAPIER from "@dimforge/rapier3d-compat";
import type { Vec3 } from "../domain/schema.js";
import {
  collisionCandidates,
  colliderData,
  colliderIdentity,
  linkPairKey,
  uniqueCollisions,
  type CollisionCandidate,
  type CollisionIssue
} from "./collision.js";
import type { Quaternion } from "./kinematics.js";
import type { RobotJointState } from "./robot-model.js";
import type { SimObject } from "./scene-builder.js";

type Collider = InstanceType<typeof RAPIER.Collider>;
type World = InstanceType<typeof RAPIER.World>;

interface CollisionAttachment {
  readonly attached: boolean;
  readonly objectId: string | null;
  readonly anchorPosition: Vec3 | null;
  readonly anchorRotation: Quaternion | null;
}

export interface ArticulatedCollisionQueryOptions {
  world: World;
  objects: ReadonlyMap<string, SimObject>;
  attachment: CollisionAttachment;
  currentBase: () => Vec3;
  currentYaw: () => number;
}

const ADJACENT_SEGMENTS = new Set([
  linkPairKey("base", "torso"),
  linkPairKey("torso", "sensor_head"),
  linkPairKey("torso", "upper_arm"),
  linkPairKey("upper_arm", "forearm"),
  linkPairKey("forearm", "wrist"),
  linkPairKey("wrist", "left_finger"),
  linkPairKey("wrist", "right_finger"),
  linkPairKey("left_finger", "right_finger"),
  linkPairKey("wrist", "attached_payload"),
  linkPairKey("left_finger", "attached_payload"),
  linkPairKey("right_finger", "attached_payload")
]);

const ROBOT_SEGMENTS = new Set([
  "base",
  "torso",
  "sensor_head",
  "upper_arm",
  "forearm",
  "wrist",
  "left_finger",
  "right_finger",
  "attached_payload"
]);

const ARM_SEGMENTS = new Set([
  "upper_arm",
  "forearm",
  "wrist",
  "left_finger",
  "right_finger",
  "attached_payload"
]);

export class ArticulatedCollisionQuery {
  readonly #world: World;
  readonly #objects: ReadonlyMap<string, SimObject>;
  readonly #attachment: CollisionAttachment;
  readonly #currentBase: () => Vec3;
  readonly #currentYaw: () => number;

  constructor(options: ArticulatedCollisionQueryOptions) {
    this.#world = options.world;
    this.#objects = options.objects;
    this.#attachment = options.attachment;
    this.#currentBase = options.currentBase;
    this.#currentYaw = options.currentYaw;
  }

  robot(base: Vec3, yaw: number, joints: RobotJointState): CollisionIssue[] {
    return this.#issues(joints, base, yaw, ROBOT_SEGMENTS, false);
  }

  arm(
    joints: RobotJointState,
    allowPortableContacts: boolean,
    baseOverride?: Vec3,
    yawOverride?: number
  ): CollisionIssue[] {
    return this.#issues(
      joints,
      baseOverride ?? this.#currentBase(),
      yawOverride ?? this.#currentYaw(),
      ARM_SEGMENTS,
      allowPortableContacts
    );
  }

  head(joints: RobotJointState): CollisionIssue[] {
    return this.#issues(
      joints,
      this.#currentBase(),
      this.#currentYaw(),
      new Set(["sensor_head"]),
      false
    );
  }

  gripper(joints: RobotJointState): CollisionIssue[] {
    return this.#issues(
      joints,
      this.#currentBase(),
      this.#currentYaw(),
      new Set(["left_finger", "right_finger"]),
      true
    );
  }

  #issues(
    joints: RobotJointState,
    base: Vec3,
    yaw: number,
    movedSegments: ReadonlySet<string>,
    allowPortableFingerContacts: boolean
  ): CollisionIssue[] {
    const candidates = this.#candidates(joints, base, yaw, allowPortableFingerContacts);
    const issues: CollisionIssue[] = [];
    if (this.#attachment.attached
      && !candidates.some((candidate) => candidate.segment === "attached_payload")) {
      issues.push({
        segment: "attached_payload",
        code: "attached_object_missing",
        collider_kind: "state",
        collider_id: null,
        penetration_depth: null
      });
    }

    for (const candidate of candidates) {
      if (!movedSegments.has(candidate.segment)) continue;
      issues.push(...this.#intersections(candidate, this.#attachment.objectId ?? undefined));
    }

    for (let leftIndex = 0; leftIndex < candidates.length; leftIndex += 1) {
      const left = candidates[leftIndex]!;
      for (let rightIndex = leftIndex + 1; rightIndex < candidates.length; rightIndex += 1) {
        const right = candidates[rightIndex]!;
        const leftMoved = movedSegments.has(left.segment);
        const rightMoved = movedSegments.has(right.segment);
        if ((!leftMoved && !rightMoved)
          || ADJACENT_SEGMENTS.has(linkPairKey(left.segment, right.segment))) continue;
        if (!left.shape.intersectsShape(
          left.transform.position,
          left.transform.rotation,
          right.shape,
          right.transform.position,
          right.transform.rotation
        )) continue;
        const contact = left.shape.contactShape(
          left.transform.position,
          left.transform.rotation,
          right.shape,
          right.transform.position,
          right.transform.rotation,
          0
        );
        const primary = leftMoved ? left : right;
        const other = primary === left ? right : left;
        issues.push({
          segment: primary.segment,
          collider_kind: "robot",
          collider_id: other.segment,
          penetration_depth: contact && Number.isFinite(contact.distance)
            ? Math.max(0, -contact.distance)
            : null
        });
      }
    }
    return uniqueCollisions(issues);
  }

  #candidates(
    joints: RobotJointState,
    base: Vec3,
    yaw: number,
    allowPortableFingerContacts: boolean
  ): CollisionCandidate[] {
    const attached = this.#attachment.objectId
      ? this.#objects.get(this.#attachment.objectId)
      : undefined;
    const attachmentPosition = this.#attachment.anchorPosition;
    const attachmentRotation = this.#attachment.anchorRotation;
    return collisionCandidates({
      joints,
      base,
      yaw,
      allowPortableFingerContacts,
      payload: attached && attachmentPosition && attachmentRotation
        ? {
          size: attached.config.size,
          anchorPosition: attachmentPosition,
          anchorRotation: attachmentRotation
        }
        : undefined
    });
  }

  #intersections(
    candidate: CollisionCandidate,
    ignoreObjectId?: string
  ): CollisionIssue[] {
    const collisions: CollisionIssue[] = [];
    this.#world.intersectionsWithShape(
      candidate.transform.position,
      candidate.transform.rotation,
      candidate.shape,
      (collider: Collider) => {
        const data = colliderData(collider);
        const contact = collider.contactShape(
          candidate.shape,
          candidate.transform.position,
          candidate.transform.rotation,
          0
        );
        collisions.push({
          segment: candidate.segment,
          collider_kind: data.kind ?? "unknown",
          collider_id: colliderIdentity(data),
          penetration_depth: contact && Number.isFinite(contact.distance)
            ? Math.max(0, -contact.distance)
            : null
        });
        return true;
      },
      undefined,
      undefined,
      undefined,
      undefined,
      (collider: Collider) => {
        const data = colliderData(collider);
        if (data.kind === "robot") return false;
        if (data.kind === "object" && data.id === ignoreObjectId) return false;
        if (candidate.allowPortableContacts && data.kind === "object" && data.id) {
          return this.#objects.get(data.id)?.config.portable !== true;
        }
        return true;
      }
    );
    return collisions;
  }
}
