import {
  HUMANOID_SKILL_CONTRACTS,
  HUMANOID_SKILL_IDS,
  type HumanoidSkillContract,
  type HumanoidSkillId
} from "../../domain/humanoid-skill.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import type {
  HumanoidObjectWorldModel,
  HumanoidObjectWorldModelEntry
} from "./object-world-model.js";
import type { HumanoidSolidToken } from "./solid-observation.js";
import type {
  HumanoidLearnedPolicyCapability
} from "../../domain/humanoid-policy.js";

interface HumanoidSkillCatalogEntry extends HumanoidSkillContract {
  observable_target_ids: string[];
  observable_solid_ids: string[];
  observable_zone_ids: string[];
  remembered_target_ids: string[];
  destination_ids: string[];
  available: boolean;
  unavailable_reasons: string[];
  learned_policy_ready: boolean;
  learned_policy_required_capabilities: HumanoidLearnedPolicyCapability[];
  learned_policy_missing_capabilities: HumanoidLearnedPolicyCapability[];
}

export interface HumanoidSkillCatalog {
  protocol: "humanoid-skill-catalog-v1";
  contract_sha256: string;
  world_frame: number;
  world_revision: number;
  entries: HumanoidSkillCatalogEntry[];
}

export function createHumanoidSkillCatalog(
  world: HumanoidObjectWorldModel,
  solids: readonly HumanoidSolidToken[] = [],
  learnedPolicyCapabilities: readonly HumanoidLearnedPolicyCapability[] = [],
  zoneIds: readonly string[] = []
): HumanoidSkillCatalog {
  const contracts = HUMANOID_SKILL_IDS.map((id) => structuredClone(
    HUMANOID_SKILL_CONTRACTS[id]
  ));
  return {
    protocol: "humanoid-skill-catalog-v1",
    contract_sha256: modelPayloadSha256(contracts),
    world_frame: world.frame,
    world_revision: world.world_revision,
    entries: contracts.map((contract) => catalogEntry(
      contract,
      world.objects,
      solids,
      learnedPolicyCapabilities,
      zoneIds
    ))
  };
}

function catalogEntry(
  contract: HumanoidSkillContract,
  objects: readonly HumanoidObjectWorldModelEntry[],
  solids: readonly HumanoidSolidToken[],
  learnedPolicyCapabilities: readonly HumanoidLearnedPolicyCapability[],
  zoneIds: readonly string[]
): HumanoidSkillCatalogEntry {
  const objectTargetRequired = skillNeedsObject(contract.id);
  const targets = objectTargetRequired ? objects.filter((object) => (
    contract.required_affordances.every(
      (affordance) => object.affordances.includes(affordance)
    ) && skillTargetCompatible(contract.id, object)
  )) : [];
  const observable = targets
    .filter((object) => object.status === "visible")
    .map((object) => object.id)
    .sort();
  const remembered = targets
    .filter((object) => object.status === "remembered")
    .map((object) => object.id)
    .sort();
  const destinations = destinationIds(contract.id, objects);
  const observableSolids = contract.id === "break_block"
    ? solids.filter(({ kind }) => kind === "block").map(({ id }) => id).sort()
    : [];
  const observableZones = contract.id === "navigate_to_zone"
    ? [...new Set(zoneIds)].sort()
    : [];
  const reasons: string[] = [];
  if (objectTargetRequired && observable.length === 0) {
    reasons.push(targets.length === 0
      ? "no object currently satisfies the required affordances"
      : "compatible targets require a current observation");
  }
  if (contract.id === "break_block" && observableSolids.length === 0) {
    reasons.push("no removable block is currently visible");
  }
  if (contract.id === "navigate_to_zone" && observableZones.length === 0) {
    reasons.push("no semantic zone is currently observable");
  }
  if (contract.id === "place" && destinations.length === 0) {
    reasons.push("no observable container, support surface or insertion point");
  }
  const learnedPolicyRequiredCapabilities = [...new Set(
    contract.process.flatMap(({ learned_policy_capabilities: capabilities }) => (
      capabilities
    ))
  )];
  const availableCapabilities = new Set(learnedPolicyCapabilities);
  const learnedPolicyMissingCapabilities = learnedPolicyRequiredCapabilities.filter(
    (capability) => !availableCapabilities.has(capability)
  );
  return {
    ...contract,
    observable_target_ids: observable,
    observable_solid_ids: observableSolids,
    observable_zone_ids: observableZones,
    remembered_target_ids: remembered,
    destination_ids: destinations,
    available: reasons.length === 0,
    unavailable_reasons: reasons,
    learned_policy_ready: learnedPolicyMissingCapabilities.length === 0,
    learned_policy_required_capabilities: learnedPolicyRequiredCapabilities,
    learned_policy_missing_capabilities: learnedPolicyMissingCapabilities
  };
}

function skillTargetCompatible(
  skill: HumanoidSkillId,
  object: HumanoidObjectWorldModelEntry
): boolean {
  if (skill === "open" || skill === "close" || skill === "press" || skill === "turn") {
    return object.articulation !== null;
  }
  if (skill === "approach" || skill === "reach") return true;
  return true;
}

function destinationIds(
  skill: HumanoidSkillId,
  objects: readonly HumanoidObjectWorldModelEntry[]
): string[] {
  if (skill !== "place") return [];
  return objects.filter((object) => object.status === "visible" && (
    object.affordances.includes("container")
      || object.affordances.includes("support_surface")
      || object.interaction_points.some((point) => point.kind === "insert")
  )).map((object) => object.id).sort();
}

function skillNeedsObject(skill: HumanoidSkillId): boolean {
  return [
    "approach", "reach", "grasp", "lift", "carry", "place", "push", "pull",
    "press", "open", "close", "turn", "regrasp", "bimanual_support", "bimanual_carry"
  ].includes(skill);
}
