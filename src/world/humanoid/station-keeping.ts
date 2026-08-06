import { z } from "zod";
import { yawFromQuaternion } from "../geometry.js";
import {
  releaseReferenceTracking,
  stationaryHumanoidReference,
  targetReference,
  type HumanoidReference
} from "./reference.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

const POSITION_GAIN = 1.2;
const LATERAL_POSITION_GAIN = 0.8;
const YAW_GAIN = 1.8;
const MAXIMUM_FORWARD_SPEED = 0.3;
const MAXIMUM_LATERAL_SPEED = 0.2;
const MAXIMUM_YAW_SPEED = 1;
const MAXIMUM_TRACKED_POSTURE_DRIFT_METERS = 0.1;

export const HumanoidStationKeepingAnchorSchema = z.object({
  x: z.number().finite(),
  z: z.number().finite(),
  yaw: z.number().finite(),
  sourceFrame: z.number().int().nonnegative(),
  sourceWorldRevision: z.number().int().nonnegative()
}).strict();

export type HumanoidStationKeepingAnchor = z.infer<
  typeof HumanoidStationKeepingAnchorSchema
>;

export function captureHumanoidStationKeepingAnchor(
  snapshot: HumanoidSimulationSnapshot,
  sourceFrame: number,
  sourceWorldRevision: number
): HumanoidStationKeepingAnchor {
  return HumanoidStationKeepingAnchorSchema.parse({
    x: snapshot.rootPosition.x,
    z: snapshot.rootPosition.z,
    yaw: yawFromQuaternion(snapshot.rootRotation),
    sourceFrame,
    sourceWorldRevision
  });
}

export function stationKeepingHumanoidReference(
  reference: HumanoidReference,
  snapshot: HumanoidSimulationSnapshot,
  anchor: HumanoidStationKeepingAnchor
): HumanoidReference {
  const validated = HumanoidStationKeepingAnchorSchema.parse(anchor);
  const yaw = yawFromQuaternion(snapshot.rootRotation);
  const deltaX = validated.x - snapshot.rootPosition.x;
  const deltaZ = validated.z - snapshot.rootPosition.z;
  const localForward = deltaX * Math.sin(yaw) + deltaZ * Math.cos(yaw);
  const localLateral = deltaX * Math.cos(yaw) - deltaZ * Math.sin(yaw);
  const yawError = normalizeAngle(validated.yaw - yaw);
  const postureReference = Math.hypot(deltaX, deltaZ)
      > MAXIMUM_TRACKED_POSTURE_DRIFT_METERS
    && reference.jointTrackingWeights.some((weight) => weight > 0)
    ? releaseReferenceTracking(reference)
    : reference;
  return targetReference(stationaryHumanoidReference(postureReference), {
    rootVelocity: [
      clamp(localForward * POSITION_GAIN, -MAXIMUM_FORWARD_SPEED, MAXIMUM_FORWARD_SPEED),
      clamp(
        localLateral * LATERAL_POSITION_GAIN,
        -MAXIMUM_LATERAL_SPEED,
        MAXIMUM_LATERAL_SPEED
      )
    ],
    rootYawVelocity: clamp(
      yawError * YAW_GAIN,
      -MAXIMUM_YAW_SPEED,
      MAXIMUM_YAW_SPEED
    )
  });
}

function normalizeAngle(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
