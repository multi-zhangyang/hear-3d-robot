import { createHash } from "node:crypto";
import type { Vec3 } from "../../domain/schema.js";
import {
  HumanoidMotionPlanSchema,
  type HumanoidMotionPlan
} from "./motion-plan.js";

export const DEFAULT_HUMANOID_PLAN_INTENT_LEASE_SECONDS = 900;

export function humanoidMotionIntentSha256(
  rawPlan: HumanoidMotionPlan
): string {
  const plan = HumanoidMotionPlanSchema.parse(rawPlan);
  return sha256(JSON.stringify(plan));
}

export function humanoidNavigationIntentSha256(target: Vec3): string {
  return sha256(JSON.stringify({
    x: finiteCoordinate(target.x, "x"),
    y: finiteCoordinate(target.y, "y"),
    z: finiteCoordinate(target.z, "z")
  }));
}

export function humanoidPlanExpiryRevision(input: {
  createdRevision: number;
  controlStepSeconds: number;
  leaseSeconds: number;
}): number {
  const createdRevision = nonnegativeSafeInteger(
    input.createdRevision,
    "createdRevision"
  );
  if (!Number.isFinite(input.controlStepSeconds) || input.controlStepSeconds <= 0) {
    throw new Error("Humanoid plan lease requires a positive control step");
  }
  if (!Number.isFinite(input.leaseSeconds) || input.leaseSeconds <= 0) {
    throw new Error("Humanoid plan lease must be positive");
  }
  const leaseRevisions = Math.ceil(input.leaseSeconds / input.controlStepSeconds);
  const expiresRevision = createdRevision + leaseRevisions;
  if (!Number.isSafeInteger(expiresRevision)) {
    throw new Error("Humanoid plan lease revision exceeds the safe integer range");
  }
  return expiresRevision;
}

export function humanoidPlanIntentIsActive(
  worldRevision: number,
  expiresRevision: number
): boolean {
  return nonnegativeSafeInteger(worldRevision, "worldRevision")
    <= nonnegativeSafeInteger(expiresRevision, "expiresRevision");
}

function finiteCoordinate(value: number, axis: string): number {
  if (!Number.isFinite(value)) {
    throw new Error(`Humanoid navigation intent ${axis} must be finite`);
  }
  return Object.is(value, -0) ? 0 : value;
}

function nonnegativeSafeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`Humanoid plan ${name} must be a nonnegative safe integer`);
  }
  return value;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
