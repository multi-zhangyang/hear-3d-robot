import RAPIER from "@dimforge/rapier3d-compat";
import { beforeAll, describe, expect, it } from "vitest";
import { ROBOT_SPEC } from "./robot-model.js";
import { GraspAttachment } from "./grasp-attachment.js";

describe("GraspAttachment", () => {
  beforeAll(async () => {
    await RAPIER.init();
  });

  it("drops a drifting payload and reports source-backed slip telemetry", () => {
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    try {
      const anchor = world.createRigidBody(
        RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(0, 1, 0)
      );
      const payload = world.createRigidBody(
        RAPIER.RigidBodyDesc.dynamic().setTranslation(0, 1, 0)
      );
      const attachment = new GraspAttachment(world, anchor);
      attachment.install("payload", payload, "constraint_1", "agent_a:grasp");
      expect(attachment.snapshot()).toMatchObject({
        object_id: "payload",
        source_command_id: "agent_a:grasp"
      });

      payload.setTranslation({ x: 0.2, y: 1, z: 0 }, true);
      for (let frame = 1; frame < ROBOT_SPEC.gripper.slipDetectionFrames; frame += 1) {
        expect(attachment.validate()).toBeNull();
      }
      expect(attachment.validate()).toMatchObject({
        objectId: "payload",
        sourceCommandId: "agent_a:grasp",
        detectionFrames: ROBOT_SPEC.gripper.slipDetectionFrames
      });
      expect(attachment.snapshot()).toBeNull();
    } finally {
      world.free();
    }
  });

  it("resets the drift window after the constraint returns within tolerance", () => {
    const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    try {
      const anchor = world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
      const payload = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic());
      const attachment = new GraspAttachment(world, anchor);
      attachment.install("payload", payload, "constraint_2", "agent_b:grasp");
      payload.setTranslation({ x: 0.2, y: 0, z: 0 }, true);
      expect(attachment.validate()).toBeNull();
      payload.setTranslation({ x: 0, y: 0, z: 0 }, true);
      expect(attachment.validate()).toBeNull();
      payload.setTranslation({ x: 0.2, y: 0, z: 0 }, true);
      expect(attachment.validate()).toBeNull();
      expect(attachment.attached).toBe(true);
    } finally {
      world.free();
    }
  });

  it("keeps a constrained payload while both fingers still support it", () => {
    const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
    try {
      const anchor = world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
      const payload = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic());
      const attachment = new GraspAttachment(world, anchor);
      attachment.install("payload", payload, "constraint_3", "agent_c:grasp");
      payload.setRotation({ x: 0, y: Math.sin(0.12), z: 0, w: Math.cos(0.12) }, true);

      for (let frame = 0; frame < ROBOT_SPEC.gripper.slipDetectionFrames + 2; frame += 1) {
        expect(attachment.validate("payload")).toBeNull();
      }
      expect(attachment.attached).toBe(true);

      for (let frame = 1; frame < ROBOT_SPEC.gripper.slipDetectionFrames; frame += 1) {
        expect(attachment.validate(null)).toBeNull();
      }
      expect(attachment.validate(null)).toMatchObject({ objectId: "payload" });
    } finally {
      world.free();
    }
  });
});
