import RAPIER from "@dimforge/rapier3d-compat";
import { beforeAll, describe, expect, it } from "vitest";
import type { Vec3 } from "../domain/schema.js";
import { EntityVisibility } from "./entity-visibility.js";
import type { SimObject } from "./scene-builder.js";

interface VisibilityFixture {
  world: InstanceType<typeof RAPIER.World>;
  object: SimObject;
  visibility: EntityVisibility;
}

beforeAll(async () => {
  await RAPIER.init();
});

function fixture(
  objectPosition: Vec3,
  options: { maximumRange?: number; blockerPosition?: Vec3 } = {}
): VisibilityFixture {
  const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
  const head = world.createRigidBody(
    RAPIER.RigidBodyDesc.fixed()
      .setTranslation(0, 1, 0)
      .setUserData({ kind: "robot", link_id: "sensor_head" })
  );
  const body = world.createRigidBody(
    RAPIER.RigidBodyDesc.fixed()
      .setTranslation(objectPosition.x, objectPosition.y, objectPosition.z)
      .setUserData({ kind: "object", id: "target" })
  );
  const collider = world.createCollider(RAPIER.ColliderDesc.cuboid(0.1, 0.1, 0.1), body);
  if (options.blockerPosition) {
    const blocker = world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed()
        .setTranslation(
          options.blockerPosition.x,
          options.blockerPosition.y,
          options.blockerPosition.z
        )
        .setUserData({ kind: "obstacle", id: "wall" })
    );
    world.createCollider(RAPIER.ColliderDesc.cuboid(0.25, 0.25, 0.25), blocker);
  }
  world.step();
  const object: SimObject = {
    config: {
      id: "target",
      kind: "block",
      color: "#ffffff",
      position: objectPosition,
      size: { x: 0.2, y: 0.2, z: 0.2 },
      portable: false
    },
    body,
    collider,
    locked: false
  };
  return {
    world,
    object,
    visibility: new EntityVisibility({
      world,
      linkBodies: new Map([["sensor_head", head]]),
      maximumRange: options.maximumRange ?? 4
    })
  };
}

describe("EntityVisibility", () => {
  it("accepts an unobstructed object inside the sensor cone", () => {
    const value = fixture({ x: 0, y: 1, z: 2 });
    try {
      expect(value.visibility.isVisible(value.object)).toBe(true);
    } finally {
      value.world.free();
    }
  });

  it("distinguishes range, field-of-view, disabled, and occlusion failures", () => {
    const outOfRange = fixture({ x: 0, y: 1, z: 5 }, { maximumRange: 4 });
    const offAxis = fixture({ x: 2, y: 1, z: 0 });
    const disabled = fixture({ x: 0, y: 1, z: 2 });
    const occluded = fixture(
      { x: 0, y: 1, z: 3 },
      { blockerPosition: { x: 0, y: 1, z: 1.5 } }
    );
    try {
      expect(outOfRange.visibility.failure(outOfRange.object, 0)).toMatchObject({
        reason: "out_of_range",
        visibility_radius: 4
      });

      const aim = offAxis.visibility.failure(offAxis.object, 0);
      expect(aim).toMatchObject({ reason: "outside_field_of_view" });
      expect((aim as { recovery: string }).recovery).toContain("yaw=1.571");

      disabled.object.body.setEnabled(false);
      expect(disabled.visibility.failure(disabled.object, 0)).toMatchObject({
        reason: "not_simulated"
      });

      expect(occluded.visibility.isVisible(occluded.object)).toBe(false);
      expect(occluded.visibility.failure(occluded.object, 0)).toMatchObject({
        reason: "occluded",
        occluded_by: "wall"
      });
    } finally {
      outOfRange.world.free();
      offAxis.world.free();
      disabled.world.free();
      occluded.world.free();
    }
  });
});
