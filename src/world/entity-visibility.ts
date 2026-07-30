import RAPIER from "@dimforge/rapier3d-compat";
import type { JsonValue } from "../domain/schema.js";
import { colliderData, colliderIdentity } from "./collision.js";
import {
  clamp,
  inverseQuaternion,
  quaternion,
  rotateVector,
  scale,
  subtract,
  vector,
  vectorLength,
  yawRotation
} from "./geometry.js";
import { ROBOT_SPEC } from "./robot-model.js";
import type { SimObject } from "./scene-builder.js";

type Collider = InstanceType<typeof RAPIER.Collider>;
type RigidBody = InstanceType<typeof RAPIER.RigidBody>;
type World = InstanceType<typeof RAPIER.World>;

export interface EntityVisibilityOptions {
  world: World;
  linkBodies: ReadonlyMap<string, RigidBody>;
  maximumRange: number;
}

export class EntityVisibility {
  readonly #world: World;
  readonly #linkBodies: ReadonlyMap<string, RigidBody>;
  readonly #maximumRange: number;

  constructor(options: EntityVisibilityOptions) {
    this.#world = options.world;
    this.#linkBodies = options.linkBodies;
    this.#maximumRange = options.maximumRange;
  }

  isVisible(object: SimObject): boolean {
    if (!object.body.isEnabled()) return false;
    const head = this.#linkBodies.get("sensor_head");
    if (!head) return false;
    const origin = vector(head.translation());
    const target = vector(object.body.translation());
    const delta = subtract(target, origin);
    const distance = vectorLength(delta);
    if (distance <= 0.001 || distance > this.#maximumRange) return false;
    const localDirection = rotateVector(
      inverseQuaternion(quaternion(head.rotation())),
      scale(delta, 1 / distance)
    );
    const horizontal = Math.atan2(localDirection.x, localDirection.z);
    const vertical = Math.atan2(localDirection.y, Math.hypot(localDirection.x, localDirection.z));
    if (Math.abs(horizontal) > ROBOT_SPEC.sensorHead.horizontalFieldOfView / 2
      || Math.abs(vertical) > ROBOT_SPEC.sensorHead.verticalFieldOfView / 2) {
      return false;
    }
    const hit = this.#world.castRay(
      new RAPIER.Ray(origin, scale(delta, 1 / distance)),
      distance + 0.02,
      true,
      undefined,
      undefined,
      undefined,
      undefined,
      (collider: Collider) => {
        const data = collider.parent()?.userData as { kind?: string } | undefined;
        return data?.kind !== "robot" && data?.kind !== "ground";
      }
    );
    return hit?.collider === object.collider;
  }

  failure(object: SimObject, bodyYaw: number): JsonValue {
    const entityId = object.config.id;
    if (!object.body.isEnabled()) {
      return {
        entity_id: entityId,
        reason: "not_simulated",
        recovery: `${entityId} is not an active body in this scenario, so no sensing will ever `
          + "reveal it. Use sense_scene to see what does exist and work with those entities."
      };
    }
    const head = this.#linkBodies.get("sensor_head");
    if (!head) return { entity_id: entityId, reason: "sensor_head_missing" };
    const origin = vector(head.translation());
    const target = vector(object.body.translation());
    const delta = subtract(target, origin);
    const distance = vectorLength(delta);
    if (distance > this.#maximumRange) {
      return {
        entity_id: entityId,
        reason: "out_of_range",
        distance: Number(distance.toFixed(3)),
        visibility_radius: this.#maximumRange,
        recovery: `${entityId} is ${distance.toFixed(2)}m away but the sensor only reaches `
          + `${this.#maximumRange}m. Aiming the head will not help. Move the base `
          + "closer first with plan_base_path and execute_base_plan, then inspect again."
      };
    }
    const bodyRelative = rotateVector(inverseQuaternion(yawRotation(bodyYaw)), delta);
    const yaw = clamp(
      Math.atan2(bodyRelative.x, bodyRelative.z),
      ROBOT_SPEC.joints.head_yaw.minimum,
      ROBOT_SPEC.joints.head_yaw.maximum
    );
    const pitch = clamp(
      Math.atan2(bodyRelative.y, Math.hypot(bodyRelative.x, bodyRelative.z)),
      ROBOT_SPEC.joints.head_pitch.minimum,
      ROBOT_SPEC.joints.head_pitch.maximum
    );
    const localDirection = rotateVector(
      inverseQuaternion(quaternion(head.rotation())),
      scale(delta, 1 / Math.max(distance, 1e-6))
    );
    const offAxis = Math.abs(Math.atan2(localDirection.x, localDirection.z))
        > ROBOT_SPEC.sensorHead.horizontalFieldOfView / 2
      || Math.abs(Math.atan2(localDirection.y, Math.hypot(localDirection.x, localDirection.z)))
        > ROBOT_SPEC.sensorHead.verticalFieldOfView / 2;
    if (offAxis) {
      return {
        entity_id: entityId,
        reason: "outside_field_of_view",
        distance: Number(distance.toFixed(3)),
        recovery: `${entityId} is in range but outside the sensor cone. Call set_head_target with `
          + `yaw=${yaw.toFixed(3)} and pitch=${pitch.toFixed(3)} to aim at it, then inspect again.`
      };
    }
    const hit = this.#world.castRay(
      new RAPIER.Ray(origin, scale(delta, 1 / Math.max(distance, 1e-6))),
      distance + 0.02,
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
    const blocker = hit ? colliderIdentity(colliderData(hit.collider)) : null;
    return {
      entity_id: entityId,
      reason: "occluded",
      occluded_by: blocker,
      distance: Number(distance.toFixed(3)),
      recovery: `${entityId} is in range and in view, but ${blocker ?? "another body"} `
        + "blocks the line of sight. Aiming the head will not clear it. Move the base to a "
        + "different side with plan_base_path and execute_base_plan, then inspect again."
    };
  }
}
