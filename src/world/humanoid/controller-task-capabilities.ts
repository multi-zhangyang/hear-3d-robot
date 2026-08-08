import {
  HUMANOID_LEARNED_POLICY_CAPABILITIES,
  type HumanoidLearnedPolicyCapability
} from "../../domain/humanoid-policy.js";
import type { HumanoidReference } from "./reference.js";

const ACTIVE_COMMAND_EPSILON = 1e-9;

export function humanoidControllerTaskCapabilities(
  reference: HumanoidReference,
  required: readonly HumanoidLearnedPolicyCapability[] = []
): HumanoidLearnedPolicyCapability[] {
  const capabilities = new Set<HumanoidLearnedPolicyCapability>([
    "balance",
    ...required
  ]);
  if (Math.hypot(...reference.rootVelocity) > ACTIVE_COMMAND_EPSILON
    || Math.abs(reference.rootYawVelocity) > ACTIVE_COMMAND_EPSILON) {
    capabilities.add("locomotion");
  }
  if (Array.from(reference.jointTrackingWeights).some(
    (weight) => weight > ACTIVE_COMMAND_EPSILON
  )) {
    capabilities.add("joint_reference_tracking");
  }
  return HUMANOID_LEARNED_POLICY_CAPABILITIES.filter((capability) => (
    capabilities.has(capability)
  ));
}
