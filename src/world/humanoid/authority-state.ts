import { createHash } from "node:crypto";
import type { JsonValue } from "../../domain/schema.js";
import type { HumanoidReference } from "./reference.js";
import type { HumanoidSimulationState } from "./simulation.js";
import type { HumanoidGraspRegistryCheckpoint } from "./grasp-registry.js";
import {
  HumanoidCarriedObjectBindingSetSchema,
  type HumanoidCarriedObjectBindingSet
} from "./carried-object-binding.js";

export interface HumanoidAuthorityIdentity {
  revision: number;
  stateSha256: string;
}

export function humanoidAuthorityStateSha256(input: {
  simulation: HumanoidSimulationState;
  reference: HumanoidReference;
  visibleContactObjectIds: Iterable<string>;
  visibleContactSolidIds: Iterable<string>;
  planRegistryEpoch: number;
  graspRegistry: HumanoidGraspRegistryCheckpoint;
  carriedObjectBindings: HumanoidCarriedObjectBindingSet;
}): string {
  const hash = createHash("sha256");
  updateNumber(hash, input.simulation.time);
  updateNumbers(hash, input.simulation.positions);
  updateNumbers(hash, input.simulation.velocities);
  updateNumbers(hash, input.simulation.controls);
  updateNumbers(hash, input.simulation.activations);
  updateNumbers(hash, input.simulation.accelerationWarmstart);
  if (input.simulation.requestedActuatorTorques) {
    hash.update("requested-actuator-torques\0");
    updateNumbers(hash, input.simulation.requestedActuatorTorques);
  } else {
    hash.update("no-requested-actuator-torques\0");
  }
  if (input.simulation.handCommandTargets) {
    hash.update("hand-command-targets\0");
    updateNumbers(hash, input.simulation.handCommandTargets);
  } else {
    hash.update("no-hand-command-targets\0");
  }
  hash.update("hand-policy-authority\0");
  hash.update(canonicalJson(input.simulation.handPolicyAuthority ?? null));
  hash.update(canonicalJson({
    protocol: input.simulation.controller.protocol,
    version: input.simulation.controller.version,
    implementation: input.simulation.controller.implementation,
    payload: input.simulation.controller.payload
  }));
  updateNumbers(hash, input.reference.jointPositions);
  updateNumbers(hash, input.reference.jointVelocities);
  updateNumbers(hash, input.reference.jointTrackingWeights);
  updateNumbers(hash, input.reference.rootVelocity);
  updateNumber(hash, input.reference.rootYawVelocity);
  updateNumber(hash, input.reference.rootHeight);
  updateNumber(hash, input.reference.rootRoll);
  updateNumber(hash, input.reference.rootPitch);
  const visibleContactObjectIds = [...input.visibleContactObjectIds].sort();
  hash.update(`visible-contact-objects\0${visibleContactObjectIds.length}\0`);
  for (const id of visibleContactObjectIds) {
    hash.update(`${id.length}\0${id}\0`);
  }
  const visibleContactSolidIds = [...input.visibleContactSolidIds].sort();
  hash.update(`visible-contact-solids\0${visibleContactSolidIds.length}\0`);
  for (const id of visibleContactSolidIds) {
    hash.update(`${id.length}\0${id}\0`);
  }
  hash.update("grasp-registry\0");
  hash.update(canonicalJson(input.graspRegistry));
  hash.update("carried-object-bindings\0");
  hash.update(canonicalJson(
    HumanoidCarriedObjectBindingSetSchema.parse(input.carriedObjectBindings)
  ));
  updateNumber(hash, requiredNonnegativeInteger(
    input.planRegistryEpoch,
    "plan registry epoch"
  ));
  return hash.digest("hex");
}

function updateNumbers(
  hash: ReturnType<typeof createHash>,
  values: ArrayLike<number>
): void {
  const length = Buffer.allocUnsafe(4);
  length.writeUInt32BE(values.length);
  hash.update(length);
  for (let index = 0; index < values.length; index += 1) {
    updateNumber(hash, requiredFinite(values[index], `state[${index}]`));
  }
}

function updateNumber(hash: ReturnType<typeof createHash>, value: number): void {
  const finite = requiredFinite(value, "state value");
  const bytes = Buffer.allocUnsafe(8);
  bytes.writeDoubleBE(Object.is(finite, -0) ? 0 : finite);
  hash.update(bytes);
}

function requiredFinite(value: number | undefined, label: string): number {
  if (value === undefined || !Number.isFinite(value)) {
    throw new Error(`Humanoid authority ${label} must be finite`);
  }
  return value;
}

function requiredNonnegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`Humanoid authority ${label} must be a nonnegative safe integer`);
  }
  return value;
}

function canonicalJson(value: JsonValue): string {
  if (value === null) return "null";
  if (typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    return JSON.stringify(requiredFinite(value, "controller state value"));
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  return `{${Object.keys(value).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalJson(value[key]!)}`
  )).join(",")}}`;
}
