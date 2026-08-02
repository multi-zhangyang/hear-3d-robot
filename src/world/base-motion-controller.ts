import type { Vec3 } from "../domain/schema.js";
import { clamp, normalizeAngle, planarDistance } from "./geometry.js";

type BaseLinearSign = -1 | 1;

export interface BaseTargetControlInput {
  current: Vec3;
  target: Vec3;
  yaw: number;
  linearSign: BaseLinearSign;
  maximumLinearVelocity: number;
  maximumAngularVelocity: number;
  timestep: number;
}

export interface BaseTargetControl {
  distance: number;
  bodyYaw: number;
  headingError: number;
  linearVelocity: number;
  angularVelocity: number;
}

/**
 * Closed-loop control for one preflighted path segment.
 *
 * Preflight chooses whether the segment is traversed forward or in reverse,
 * but execution must derive its heading from the live pose. Following only the
 * heading captured at planning time can miss a waypoint by a few centimetres;
 * once past it, a distance-only loop will continue forever in the old direction.
 */
export function controlBaseTowardTarget(input: BaseTargetControlInput): BaseTargetControl {
  const distance = planarDistance(input.current, input.target);
  const travelYaw = Math.atan2(
    input.target.x - input.current.x,
    input.target.z - input.current.z
  );
  const bodyYaw = normalizeAngle(
    travelYaw + (input.linearSign === -1 ? Math.PI : 0)
  );
  const headingError = normalizeAngle(bodyYaw - input.yaw);
  const angularVelocity = clamp(
    headingError * 3,
    -input.maximumAngularVelocity,
    input.maximumAngularVelocity
  );
  const linearVelocity = Math.abs(headingError) > 0.32
    ? 0
    : input.linearSign * Math.min(
        input.maximumLinearVelocity,
        distance / input.timestep
      );

  return {
    distance,
    bodyYaw,
    headingError,
    linearVelocity,
    angularVelocity
  };
}
