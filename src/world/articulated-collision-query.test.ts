import RAPIER from "@dimforge/rapier3d-compat";
import { beforeAll, describe, expect, it } from "vitest";
import type { Vec3 } from "../domain/schema.js";
import { ArticulatedCollisionQuery } from "./articulated-collision-query.js";
import { collisionKey } from "./collision.js";
import { rigTransforms } from "./rig.js";
import { ROBOT_SPEC, type RobotJointState } from "./robot-model.js";
import type { SimObject } from "./scene-builder.js";

type World = InstanceType<typeof RAPIER.World>;

interface TestAttachment {
  attached: boolean;
  objectId: string | null;
  anchorPosition: Vec3 | null;
  anchorRotation: { x: number; y: number; z: number; w: number } | null;
}

const BASE = { x: 2, y: ROBOT_SPEC.base.centerY, z: 2 };

beforeAll(async () => {
  await RAPIER.init();
});

function joints(overrides: Partial<RobotJointState> = {}): RobotJointState {
  return { ...ROBOT_SPEC.defaultJoints, ...overrides };
}

function fixture(attachment: TestAttachment = {
  attached: false,
  objectId: null,
  anchorPosition: null,
  anchorRotation: null
}): {
  world: World;
  objects: Map<string, SimObject>;
  query: ArticulatedCollisionQuery;
} {
  const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
  const objects = new Map<string, SimObject>();
  return {
    world,
    objects,
    query: new ArticulatedCollisionQuery({
      world,
      objects,
      attachment,
      currentBase: () => ({ ...BASE }),
      currentYaw: () => 0
    })
  };
}

function addObject(
  world: World,
  objects: Map<string, SimObject>,
  position: Vec3,
  portable: boolean
): SimObject {
  const body = world.createRigidBody(
    RAPIER.RigidBodyDesc.fixed()
      .setTranslation(position.x, position.y, position.z)
      .setUserData({ kind: "object", id: "payload" })
  );
  const collider = world.createCollider(RAPIER.ColliderDesc.cuboid(0.02, 0.02, 0.02), body);
  const object: SimObject = {
    config: {
      id: "payload",
      kind: "block",
      color: "#ffffff",
      position,
      size: { x: 0.04, y: 0.04, z: 0.04 },
      portable
    },
    body,
    collider,
    locked: false
  };
  objects.set(object.config.id, object);
  return object;
}

describe("ArticulatedCollisionQuery", () => {
  it("filters adjacent links but reports a non-adjacent self-collision", () => {
    const value = fixture();
    try {
      expect(value.query.robot(BASE, 0, joints())).toEqual([]);

      const folded = value.query.arm(joints({
        shoulder: 0,
        elbow: -2.5,
        wrist: -1.7
      }), true);
      expect(folded).toContainEqual(expect.objectContaining({
        collider_kind: "robot"
      }));
    } finally {
      value.world.free();
    }
  });

  it("returns stable external-collision identities, depths, and ordering", () => {
    const value = fixture();
    try {
      const obstacle = value.world.createRigidBody(
        RAPIER.RigidBodyDesc.fixed()
          .setTranslation(BASE.x, BASE.y, BASE.z)
          .setUserData({ kind: "obstacle", id: "barrier" })
      );
      value.world.createCollider(RAPIER.ColliderDesc.cuboid(0.1, 0.1, 0.1), obstacle);
      value.world.step();

      const issues = value.query.robot(BASE, 0, joints());
      expect(issues).toContainEqual(expect.objectContaining({
        segment: "base",
        collider_kind: "obstacle",
        collider_id: "barrier",
        penetration_depth: expect.any(Number)
      }));
      expect(issues).toEqual([...issues].sort((left, right) =>
        collisionKey(left).localeCompare(collisionKey(right))
      ));
    } finally {
      value.world.free();
    }
  });

  it("allows portable finger contact but still rejects fixed objects", () => {
    const value = fixture();
    try {
      const finger = rigTransforms(joints(), BASE, 0).leftFinger;
      const object = addObject(value.world, value.objects, finger.position, true);
      value.world.step();

      expect(value.query.gripper(joints()).some((issue) =>
        issue.collider_id === object.config.id
      )).toBe(false);

      object.config.portable = false;
      expect(value.query.gripper(joints())).toContainEqual(expect.objectContaining({
        segment: "left_finger",
        collider_kind: "object",
        collider_id: object.config.id
      }));
    } finally {
      value.world.free();
    }
  });

  it("preserves the explicit missing-payload state failure", () => {
    const value = fixture({
      attached: true,
      objectId: "missing_payload",
      anchorPosition: null,
      anchorRotation: null
    });
    try {
      expect(value.query.arm(joints(), true)).toContainEqual({
        segment: "attached_payload",
        code: "attached_object_missing",
        collider_kind: "state",
        collider_id: null,
        penetration_depth: null
      });
    } finally {
      value.world.free();
    }
  });
});
