import { z } from "zod";
import {
  Vec3Schema,
  type Goal,
  type JsonValue,
  type Vec3
} from "../../domain/schema.js";
import {
  BeginHumanoidSkillSchema,
  HUMANOID_SKILL_CONTRACTS,
  HumanoidSkillInvocationSchema,
  humanoidSkillPhaseLearnedPolicyCapabilities,
  type BeginHumanoidSkill,
  type HumanoidSkillInvocation
} from "../../domain/humanoid-skill.js";
import {
  HUMANOID_LEARNED_POLICY_CAPABILITIES,
  type HumanoidLearnedPolicyCapability
} from "../../domain/humanoid-policy.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import type { HumanoidWorldObservation } from "../../world/humanoid/world.js";
import {
  HumanoidEmbodiedSkillIdentitySchema,
  type HumanoidEmbodiedSkillIdentity
} from "../../world/humanoid/embodied-skill-call.js";
import type { HumanoidObjectWorldModelEntry } from "../../world/humanoid/object-world-model.js";
import type { HumanoidSolidToken } from "../../world/humanoid/solid-observation.js";
import { HUMANOID_NAVIGATION_PROFILE } from "../../world/humanoid/environment.js";
import { navigationObstaclePlanarExpansion } from "../../world/navigation.js";
import {
  humanoidArticulationGoal,
  type HumanoidArticulationGoal
} from "./articulation-control.js";
import { alignHumanoidSkillToGoal } from "./goal-skill-alignment.js";

const GRASP_REACH_PRECONDITION_DISTANCE_METERS = 0.12;
const APPROACH_SUPPORT_CLEARANCE_MARGIN_METERS = 0.01;

export type SkillPlanningAction =
  | "plan_humanoid_skill"
  | "plan_whole_body_motion_candidates"
  | "plan_humanoid_navigation";

export interface ActiveHumanoidSkillBinding {
  protocol: "humanoid-active-skill-v1";
  transaction_id: string;
  agent_id: string;
  skill_plan_transaction_id: string | null;
  skill_node_id: string | null;
  invocation: HumanoidSkillInvocation;
  invocation_sha256: string;
  phase: string;
  phase_authority: "navigation" | "whole_body" | "grasp";
  planning_action: SkillPlanningAction;
  observed_frame: number;
  observed_world_revision: number;
  skill_catalog_sha256: string;
  active_goal_sha256?: string;
  recovery_authorized?: boolean;
  target_position: Vec3 | null;
  target_solid: Omit<HumanoidSolidToken, "currentContacts"> | null;
  target_articulation: HumanoidObjectWorldModelEntry["articulation"];
  eligible_interaction_points: HumanoidObjectWorldModelEntry["interaction_points"];
  eligible_interaction_point_ids: string[];
  learned_policy_required_capabilities: HumanoidLearnedPolicyCapability[];
  learned_policy_missing_capabilities: HumanoidLearnedPolicyCapability[];
  control_mode: "learned_policy" | "reference_control_fallback";
}

export function manipulationBasePlacementNavigationBlockerIds(
  observation: HumanoidWorldObservation,
  placement: HumanoidWorldObservation["manipulationBasePlacements"][number]
): string[] {
  const expansion = navigationObstaclePlanarExpansion(
    HUMANOID_NAVIGATION_PROFILE.radius
  ) + APPROACH_SUPPORT_CLEARANCE_MARGIN_METERS;
  return observation.solidTokens.filter((solid) => (
    solid.currentContacts.some((contact) => (
      contact.firstObject === placement.objectId && contact.secondSolid === solid.id
    ) || (
      contact.secondObject === placement.objectId && contact.firstSolid === solid.id
    ))
      && Math.abs(placement.rootWorldTarget.x - solid.center.x)
        <= solid.size.x / 2 + expansion
      && Math.abs(placement.rootWorldTarget.z - solid.center.z)
        <= solid.size.z / 2 + expansion
  )).map(({ id }) => id);
}

export function navigableManipulationBasePlacements(
  observation: HumanoidWorldObservation,
  objectId: string
): HumanoidWorldObservation["manipulationBasePlacements"] {
  return observation.manipulationBasePlacements.filter((placement) => (
    placement.objectId === objectId
      && manipulationBasePlacementNavigationBlockerIds(
        observation,
        placement
      ).length === 0
  ));
}

const PersistedInteractionPointSchema = z.object({
  id: z.string().trim().min(1),
  kind: z.string().trim().min(1),
  compatible_hands: z.enum(["left", "right", "either", "both"]),
  world_position: Vec3Schema,
  approach_direction_world: Vec3Schema.optional(),
  clearance_m: z.number().finite().nonnegative(),
  source: z.enum(["authored", "geometry"])
}).strict();

const PersistedArticulationSchema = z.object({
  joint_id: z.string().trim().min(1),
  parent_object_id: z.string().trim().min(1).nullable(),
  type: z.enum(["hinge", "slide"]),
  semantic: z.string().trim().min(1),
  axis_world: Vec3Schema,
  anchor_world: Vec3Schema,
  position: z.number().finite().nullable(),
  velocity: z.number().finite().nullable(),
  range: z.object({
    minimum: z.number().finite(),
    maximum: z.number().finite()
  }).strict(),
  closed_position: z.number().finite(),
  open_position: z.number().finite(),
  open_fraction: z.number().finite().min(0).max(1).nullable(),
  state: z.enum(["open", "closed", "intermediate", "unobserved"])
}).strict();

const PersistedSolidSchema = z.object({
  id: z.string().trim().min(1),
  sourceId: z.string().trim().min(1),
  kind: z.enum(["block", "fixed_object"]),
  center: Vec3Schema,
  size: Vec3Schema
}).strict();

export const ActiveHumanoidSkillBindingSchema = z.object({
    protocol: z.literal("humanoid-active-skill-v1"),
    transaction_id: z.string().trim().min(1),
    agent_id: z.string().trim().min(1),
    skill_plan_transaction_id: z.string().trim().min(1).nullable(),
    skill_node_id: z.string().trim().min(1).nullable(),
    invocation: HumanoidSkillInvocationSchema,
    invocation_sha256: z.string().regex(/^[a-f0-9]{64}$/),
    phase: z.string().trim().min(1),
    phase_authority: z.enum(["navigation", "whole_body", "grasp"]),
    planning_action: z.enum([
      "plan_humanoid_skill",
      "plan_whole_body_motion_candidates",
      "plan_humanoid_navigation"
    ]),
    observed_frame: z.number().int().nonnegative(),
    observed_world_revision: z.number().int().nonnegative(),
    skill_catalog_sha256: z.string().regex(/^[a-f0-9]{64}$/),
    active_goal_sha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
    recovery_authorized: z.boolean().optional(),
    target_position: Vec3Schema.nullable(),
    target_solid: PersistedSolidSchema.nullable().default(null),
    target_articulation: PersistedArticulationSchema.nullable(),
    eligible_interaction_points: z.array(PersistedInteractionPointSchema),
    eligible_interaction_point_ids: z.array(z.string().trim().min(1)),
    learned_policy_required_capabilities: z.array(
      z.enum(HUMANOID_LEARNED_POLICY_CAPABILITIES)
    ).default([]),
    learned_policy_missing_capabilities: z.array(
      z.enum(HUMANOID_LEARNED_POLICY_CAPABILITIES)
    ).default([]),
    control_mode: z.enum([
      "learned_policy",
      "reference_control_fallback"
    ]).default("reference_control_fallback")
  }).strict().superRefine((binding, context) => {
    if (binding.invocation_sha256 !== modelPayloadSha256(binding.invocation)) {
      context.addIssue({
        code: "custom",
        path: ["invocation_sha256"],
        message: "Active Skill invocation identity is invalid"
      });
    }
    const pointIds = binding.eligible_interaction_points.map(({ id }) => id).sort();
    const indexedIds = [...binding.eligible_interaction_point_ids].sort();
    if (new Set(indexedIds).size !== indexedIds.length
      || JSON.stringify(pointIds) !== JSON.stringify(indexedIds)) {
      context.addIssue({
        code: "custom",
        path: ["eligible_interaction_point_ids"],
        message: "Active Skill interaction-point index is invalid"
      });
    }
  }) as unknown as z.ZodType<ActiveHumanoidSkillBinding>;

export type HumanoidSkillBindingResult =
  | { accepted: true; binding: ActiveHumanoidSkillBinding }
  | {
      accepted: false;
      code: string;
      detail: JsonValue;
    };

export function humanoidEmbodiedSkillIdentity(
  binding: ActiveHumanoidSkillBinding
): HumanoidEmbodiedSkillIdentity {
  return HumanoidEmbodiedSkillIdentitySchema.parse({
    protocol: "humanoid-embodied-skill-identity-v1",
    callId: [
      "skill-call",
      binding.transaction_id,
      binding.phase
    ].join(":"),
    runtimeKind: "semantic_skill",
    agentId: binding.agent_id,
    bindingTransactionId: binding.transaction_id,
    skillPlanTransactionId: binding.skill_plan_transaction_id,
    skillNodeId: binding.skill_node_id,
    skillId: binding.invocation.skill,
    phase: binding.phase,
    invocation: structuredClone(binding.invocation),
    invocationSha256: binding.invocation_sha256,
    skillCatalogSha256: binding.skill_catalog_sha256,
    observedFrame: binding.observed_frame,
    observedWorldRevision: binding.observed_world_revision
  });
}

type InteractionPointValidation =
  | {
      accepted: true;
      eligiblePointIds: string[];
      eligiblePoints: HumanoidObjectWorldModelEntry["interaction_points"];
    }
  | { accepted: false; code: string; detail: JsonValue };

export function bindHumanoidSkill(input: {
  transactionId: string;
  agentId: string;
  request: BeginHumanoidSkill;
  observation: HumanoidWorldObservation;
  articulationGoal?: HumanoidArticulationGoal;
  activeGoal?: Goal;
  recoveryAuthorized?: boolean;
}): HumanoidSkillBindingResult {
  const request = BeginHumanoidSkillSchema.parse(input.request);
  const invocation = request.invocation;
  const contract = HUMANOID_SKILL_CONTRACTS[invocation.skill];
  const process = contract.process.find(({ phase }) => phase === request.phase);
  if (!process) {
    return rejection("skill_phase_unknown", {
      skill: invocation.skill,
      requested_phase: request.phase,
      available_phases: contract.process.map(({ phase, authority }) => ({
        phase,
        authority
      }))
    });
  }
  if (process.authority !== "navigation"
    && process.authority !== "whole_body"
    && process.authority !== "grasp") {
    return rejection("skill_phase_not_actionable", {
      skill: invocation.skill,
      requested_phase: request.phase,
      phase_authority: process.authority,
      reason: process.authority === "sensor"
        ? "The current observation already supplies sensor authority"
        : "Checker authority must be expressed by a physical Motion Option terminal"
    });
  }
  const worldModel = input.observation.interaction.object_world_model;
  if (worldModel.frame !== input.observation.frame
    || worldModel.world_revision !== input.observation.worldRevision) {
    throw new Error("Object world model does not match its observation authority");
  }
  const objects = new Map(worldModel.objects.map((object) => [object.id, object]));
  const targetId = invocationObjectId(invocation);
  const target = targetId ? objects.get(targetId) : undefined;
  const targetSolid = invocation.skill === "break_block"
    ? input.observation.solidTokens.find(({ id }) => id === invocation.solid_id)
    : undefined;
  const explorationFrontier = invocation.skill === "explore"
    ? input.observation.spatialBelief.frontiers.find(
        ({ id }) => id === invocation.frontier_id
      )
    : undefined;
  const targetZone = invocation.skill === "navigate_to_zone"
    || invocation.skill === "carry_to_zone"
    ? input.observation.interaction.zones.find(
        ({ zone_id: id }) => id === invocation.zone_id
      )
    : undefined;
  if ((invocation.skill === "navigate_to_zone"
      || invocation.skill === "carry_to_zone") && !targetZone) {
    return rejection("skill_zone_unavailable", {
      skill: invocation.skill,
      zone_id: invocation.zone_id,
      observable_zone_ids: input.observation.interaction.zones.map(
        ({ zone_id: id }) => id
      )
    });
  }
  if (invocation.skill === "explore" && !explorationFrontier) {
    return rejection("skill_frontier_unavailable", {
      skill: invocation.skill,
      frontier_id: invocation.frontier_id,
      available_frontier_ids: input.observation.spatialBelief.frontiers.map(
        ({ id }) => id
      )
    });
  }
  if (invocation.skill === "explore" && explorationFrontier
    && explorationFrontier.travel_distance_m > invocation.maximum_travel_m) {
    return rejection("skill_frontier_out_of_range", {
      skill: invocation.skill,
      frontier_id: invocation.frontier_id,
      travel_distance_m: explorationFrontier.travel_distance_m,
      maximum_travel_m: invocation.maximum_travel_m
    });
  }
  if (invocation.skill === "break_block" && !targetSolid) {
    return rejection("skill_solid_unobserved", {
      skill: invocation.skill,
      solid_id: invocation.solid_id,
      observable_solid_ids: input.observation.solidTokens.map(({ id }) => id)
    });
  }
  if (invocation.skill === "break_block" && targetSolid?.kind === "fixed_object") {
    return rejection("skill_solid_not_removable", {
      skill: invocation.skill,
      solid_id: invocation.solid_id,
      solid_kind: targetSolid.kind
    });
  }
  if (targetId && (!target || target.status !== "visible")) {
    return rejection("skill_target_unobserved", {
      skill: invocation.skill,
      object_id: targetId,
      status: target?.status ?? "unknown"
    });
  }
  if (target) {
    const missingAffordances = contract.required_affordances.filter(
      (affordance) => !target.affordances.includes(affordance)
    );
    if (missingAffordances.length > 0) {
      return rejection("skill_affordance_missing", {
        skill: invocation.skill,
        object_id: target.id,
        required_affordances: contract.required_affordances,
        observed_affordances: target.affordances,
        missing_affordances: missingAffordances
      });
    }
  }
  const interaction = validateInteractionPoints(invocation, target);
  if (!interaction.accepted) return interaction;
  const semantic = validateSkillSemantics(
    invocation,
    target,
    objects,
    input.observation,
    input.articulationGoal
  );
  if (semantic) return semantic;
  if (input.activeGoal) {
    const alignment = alignHumanoidSkillToGoal({
      goal: input.activeGoal,
      invocation,
      observation: input.observation,
      ...(input.recoveryAuthorized ? { recoveryAuthorized: true } : {})
    });
    if (!alignment.accepted) {
      return rejection("skill_goal_misaligned", {
        skill: invocation.skill,
        active_goal: input.activeGoal,
        reason: alignment.reason,
        recovery: "Choose a Skill whose target advances an active Goal predicate, establishes a matching object prerequisite, or is authorized by current physical recovery evidence"
      });
    }
  }
  const approachPlacement = invocation.skill === "approach"
    && invocation.interaction_point_id !== null
    ? navigableManipulationBasePlacements(
        input.observation,
        invocation.object_id
      ).filter((placement) => (
        placement.interactionPointId === invocation.interaction_point_id
          && placement.handSurface.startsWith(`${invocation.hand}_`)
      )).sort((left, right) => (
        left.ikResidualMeters - right.ikResidualMeters
      ))[0]
    : undefined;
  const learnedPolicyRequiredCapabilities =
    humanoidSkillPhaseLearnedPolicyCapabilities(invocation, request.phase);
  const learnedPolicyCapabilities = new Set(
    input.observation.robot.controller?.learnedPolicy?.capabilities ?? []
  );
  const learnedPolicyMissingCapabilities = learnedPolicyRequiredCapabilities.filter(
    (capability) => !learnedPolicyCapabilities.has(capability)
  );
  const planningAction: SkillPlanningAction = "plan_humanoid_skill";
  return {
    accepted: true,
    binding: {
      protocol: "humanoid-active-skill-v1",
      transaction_id: input.transactionId,
      agent_id: input.agentId,
      skill_plan_transaction_id: request.skill_plan_transaction_id,
      skill_node_id: request.skill_node_id,
      invocation: structuredClone(invocation),
      invocation_sha256: modelPayloadSha256(invocation),
      phase: request.phase,
      phase_authority: process.authority,
      planning_action: planningAction,
      observed_frame: input.observation.frame,
      observed_world_revision: input.observation.worldRevision,
      skill_catalog_sha256:
        input.observation.interaction.skill_catalog.contract_sha256,
      ...(input.activeGoal
        ? { active_goal_sha256: modelPayloadSha256(input.activeGoal) }
        : {}),
      ...(input.recoveryAuthorized ? { recovery_authorized: true } : {}),
      target_position: approachPlacement
        ? { ...approachPlacement.rootWorldTarget }
        : invocation.skill === "carry_to_zone" && target && targetZone
        ? {
            x: input.observation.robot.rootPosition.x
              + targetZone.center.x - target.pose.position.x,
            y: input.observation.robot.rootPosition.y,
            z: input.observation.robot.rootPosition.z
              + targetZone.center.z - target.pose.position.z
          }
        : target
        ? { ...target.pose.position }
        : targetZone ? { ...targetZone.center }
        : explorationFrontier ? { ...explorationFrontier.target }
          : targetSolid ? { ...targetSolid.center } : null,
      target_solid: targetSolid ? {
        id: targetSolid.id,
        sourceId: targetSolid.sourceId,
        kind: targetSolid.kind,
        center: { ...targetSolid.center },
        size: { ...targetSolid.size }
      } : null,
      target_articulation: target?.articulation
        ? structuredClone(target.articulation)
        : null,
      eligible_interaction_points: structuredClone(interaction.eligiblePoints),
      eligible_interaction_point_ids: interaction.eligiblePointIds,
      learned_policy_required_capabilities: learnedPolicyRequiredCapabilities,
      learned_policy_missing_capabilities: learnedPolicyMissingCapabilities,
      control_mode: learnedPolicyMissingCapabilities.length === 0
        ? "learned_policy"
        : "reference_control_fallback"
    }
  };
}

export function validateSkillPlanningReference(input: {
  binding: ActiveHumanoidSkillBinding | undefined;
  action: SkillPlanningAction;
  rawInput: unknown;
  currentWorldRevision: number;
}): { accepted: true } | { accepted: false; code: string; detail: JsonValue } {
  const binding = input.binding;
  if (!binding) {
    return rejection("active_skill_required", {
      action: input.action,
      recovery: "Observe, then bind one model-selected humanoid skill phase before planning"
    });
  }
  const supplied = skillTransactionId(input.rawInput);
  if (supplied !== binding.transaction_id) {
    return rejection("skill_reference_mismatch", {
      action: input.action,
      supplied_skill_transaction_id: supplied,
      active_skill_transaction_id: binding.transaction_id
    });
  }
  if (binding.observed_world_revision !== input.currentWorldRevision) {
    return rejection("skill_world_revision_stale", {
      action: input.action,
      skill: binding.invocation.skill,
      skill_world_revision: binding.observed_world_revision,
      current_world_revision: input.currentWorldRevision,
      recovery: "Observe the changed world and bind the next skill phase again"
    });
  }
  if (binding.planning_action !== input.action) {
    return rejection("skill_phase_authority_mismatch", {
      skill: binding.invocation.skill,
      phase: binding.phase,
      phase_authority: binding.phase_authority,
      requested_action: input.action,
      authorized_action: binding.planning_action
    });
  }
  const outcome = input.action === "plan_humanoid_skill"
    ? null
    : input.action === "plan_humanoid_navigation"
      ? validateNavigationOutcome(binding, input.rawInput)
      : validateMotionOutcome(binding, input.rawInput);
  return outcome ?? { accepted: true };
}

function validateInteractionPoints(
  invocation: HumanoidSkillInvocation,
  target: HumanoidObjectWorldModelEntry | undefined
): InteractionPointValidation {
  if (!target) {
    return { accepted: true, eligiblePointIds: [], eligiblePoints: [] };
  }
  const requested = interactionPointIds(invocation);
  const expectedKinds = interactionPointKinds(invocation.skill);
  const hand = invocationHand(invocation);
  const eligible = target.interaction_points.filter((point) => (
    (expectedKinds.length === 0 || expectedKinds.includes(point.kind))
      && (!hand || point.compatible_hands === "either"
        || point.compatible_hands === "both"
        || point.compatible_hands === hand)
  ));
  const eligibleIds = eligible.map(({ id }) => id).sort();
  for (const pointId of requested) {
    if (!eligibleIds.includes(pointId)) {
      return rejection("skill_interaction_point_missing", {
        skill: invocation.skill,
        object_id: target.id,
        interaction_point_id: pointId,
        expected_kinds: expectedKinds,
        compatible_hand: hand ?? null,
        eligible_interaction_point_ids: eligibleIds
      });
    }
  }
  if (requested.length === 0 && skillNeedsInteractionPoint(invocation.skill)
    && eligibleIds.length === 0) {
    return rejection("skill_interaction_point_missing", {
      skill: invocation.skill,
      object_id: target.id,
      expected_kinds: expectedKinds,
      compatible_hand: hand ?? null,
      eligible_interaction_point_ids: []
    });
  }
  return {
    accepted: true,
    eligiblePointIds: eligibleIds,
    eligiblePoints: structuredClone(eligible)
  };
}

function validateSkillSemantics(
  invocation: HumanoidSkillInvocation,
  target: HumanoidObjectWorldModelEntry | undefined,
  objects: ReadonlyMap<string, HumanoidObjectWorldModelEntry>,
  observation: HumanoidWorldObservation,
  continuationGoal?: HumanoidArticulationGoal
): ReturnType<typeof rejection> | null {
  if (invocation.skill === "approach"
    && invocation.interaction_point_id !== null) {
    const observedObjectPlacements = observation.manipulationBasePlacements.filter(
      (placement) => placement.objectId === invocation.object_id
    );
    const objectPlacements = navigableManipulationBasePlacements(
      observation,
      invocation.object_id
    );
    const selectedPlacements = objectPlacements.filter((placement) => (
      placement.interactionPointId === invocation.interaction_point_id
        && placement.handSurface.startsWith(`${invocation.hand}_`)
    ));
    if (observedObjectPlacements.length > 0 && selectedPlacements.length === 0) {
      return rejection("skill_manipulation_base_unavailable", {
        skill: invocation.skill,
        object_id: invocation.object_id,
        hand: invocation.hand,
        interaction_point_id: invocation.interaction_point_id,
        reason: "the selected hand and interaction point have no navigation-clear live IK-derived base placement",
        reachable_base_placements: manipulationBasePlacementChoices(objectPlacements),
        navigation_blocked_base_placements: observedObjectPlacements
          .filter((placement) => manipulationBasePlacementNavigationBlockerIds(
            observation,
            placement
          ).length > 0)
          .map((placement) => ({
            object_id: placement.objectId,
            interaction_point_id: placement.interactionPointId ?? null,
            hand_surface: placement.handSurface,
            root_world_target: placement.rootWorldTarget,
            root_yaw_radians: placement.rootYawRadians,
            ik_residual_m: placement.ikResidualMeters,
            blocking_solid_ids: manipulationBasePlacementNavigationBlockerIds(
              observation,
              placement
            )
          })),
        recovery: "Choose one exact hand and interaction_point_id pair from reachable_base_placements; do not substitute a merely eligible geometric point"
      });
    }
  }
  if (invocation.skill === "reach") {
    const selectedReachability = observation.manipulationReachability.filter(
      (entry) => entry.objectId === invocation.object_id
        && entry.interactionPointId === invocation.interaction_point_id
        && entry.handSurface.startsWith(`${invocation.hand}_`)
    );
    if (!selectedReachability.some((entry) => entry.ikReferenceReachable)) {
      const objectPlacements = navigableManipulationBasePlacements(
        observation,
        invocation.object_id
      );
      return rejection("skill_reach_pose_unreachable", {
        skill: invocation.skill,
        object_id: invocation.object_id,
        hand: invocation.hand,
        interaction_point_id: invocation.interaction_point_id,
        tolerance_m: invocation.tolerance_m,
        current_reachability: selectedReachability.map((entry) => ({
          hand_surface: entry.handSurface,
          ik_reference_reachable: entry.ikReferenceReachable,
          ik_residual_m: entry.ikResidualMeters
        })),
        reachable_base_placements: manipulationBasePlacementChoices(objectPlacements),
        reason: "the selected wrist target is not reachable from the current physical root pose",
        required_prerequisite_skill: "approach",
        recovery: "Approach one exact live reachable_base_placements sample first, preserving both its root target and root yaw; then observe again before binding reach"
      });
    }
  }
  if (invocation.skill === "grasp") {
    const occupied = observation.interaction.carrying.bindings.find(
      ({ hand }) => hand === invocation.hand
    );
    if (occupied && occupied.object_id !== invocation.object_id) {
      return rejection("skill_precondition_failed", {
        skill: invocation.skill,
        object_id: invocation.object_id,
        hand: invocation.hand,
        occupied_by_object_id: occupied.object_id,
        reason: "selected hand must be free before grasp"
      });
    }
    const verified = observation.interaction.manipulable_objects
      .find(({ object_id: objectId }) => objectId === invocation.object_id)
      ?.grasp.find(({ hand }) => hand === invocation.hand)?.verified === true;
    if (!verified) {
      const point = target?.interaction_points.find(
        ({ id }) => id === invocation.interaction_point_id
      );
      const handSurfaces = observation.handSurfaces.filter(
        ({ hand }) => hand === invocation.hand
      );
      const nearestSurfaceDistance = point && handSurfaces.length > 0
        ? Math.min(...handSurfaces.map(({ worldPosition }) => distance(
            worldPosition,
            point.world_position
          )))
        : null;
      if (nearestSurfaceDistance === null
        || nearestSurfaceDistance > GRASP_REACH_PRECONDITION_DISTANCE_METERS) {
        return rejection("skill_precondition_failed", {
          skill: invocation.skill,
          object_id: invocation.object_id,
          hand: invocation.hand,
          interaction_point_id: invocation.interaction_point_id,
          reason: "selected hand has not reached the live interaction point",
          nearest_hand_surface_distance_m: nearestSurfaceDistance,
          maximum_reach_precondition_distance_m:
            GRASP_REACH_PRECONDITION_DISTANCE_METERS,
          required_prerequisite_skill: "reach"
        });
      }
      if (handSurfaces.length < 2) {
        return rejection("skill_precondition_failed", {
          skill: invocation.skill,
          object_id: invocation.object_id,
          hand: invocation.hand,
          reason: "two observable hand contact surfaces are required before grasp",
          observed_hand_surface_count: handSurfaces.length
        });
      }
    }
  }
  if (invocation.skill === "open" || invocation.skill === "close"
    || invocation.skill === "turn") {
    if (!target?.articulation
      || target.articulation.joint_id !== invocation.joint_id
      || target.articulation.position === null) {
      return rejection("skill_articulation_unobserved", {
        skill: invocation.skill,
        object_id: target?.id ?? invocation.object_id,
        joint_id: invocation.joint_id,
        observed_joint_id: target?.articulation?.joint_id ?? null
      });
    }
    if (continuationGoal && continuationGoal.joint_id !== invocation.joint_id) {
      return rejection("skill_precondition_failed", {
        skill: invocation.skill,
        object_id: target.id,
        joint_id: invocation.joint_id,
        reason: "continued articulation goal does not match the selected joint"
      });
    }
    if (invocation.skill === "turn") {
      if (target.articulation.type !== "hinge") {
        return rejection("skill_precondition_failed", {
          skill: invocation.skill,
          object_id: target.id,
          joint_id: invocation.joint_id,
          reason: "turn requires a rotational articulation"
        });
      }
      let goal: HumanoidArticulationGoal;
      try {
        goal = continuationGoal ?? humanoidArticulationGoal({
          invocation,
          articulation: target.articulation
        });
      } catch (error) {
        return rejection("skill_precondition_failed", {
          skill: invocation.skill,
          object_id: target.id,
          joint_id: invocation.joint_id,
          reason: error instanceof Error ? error.message : String(error),
          current_position: target.articulation.position,
          requested_target_position: continuationGoal?.target_position ?? null,
          observed_range: target.articulation.range
        });
      }
      const remainingDirection = goal.target_position > target.articulation.position
        ? "increasing" : "decreasing";
      if (remainingDirection !== invocation.direction) {
        return rejection("skill_precondition_failed", {
          skill: invocation.skill,
          object_id: target.id,
          joint_id: invocation.joint_id,
          reason: "continued articulation goal has already been reached or crossed",
          current_position: target.articulation.position,
          requested_target_position: goal.target_position
        });
      }
      return null;
    }
    const fraction = target.articulation.open_fraction;
    if (fraction !== null && (invocation.skill === "open"
      ? fraction >= invocation.minimum_open_fraction
      : fraction <= invocation.maximum_open_fraction)) {
      return rejection("skill_precondition_failed", {
        skill: invocation.skill,
        object_id: target.id,
        joint_id: invocation.joint_id,
        reason: invocation.skill === "open"
          ? "articulation is already at the requested open fraction"
          : "articulation is already at the requested closed fraction",
        observed_open_fraction: fraction
      });
    }
  }
  if (invocation.skill === "place") {
    if (invocation.destination.type === "semantic_zone") {
      const destination = invocation.destination;
      const zone = observation.interaction.zones.find(
        ({ zone_id: zoneId }) => zoneId === destination.zone_id
      );
      if (!zone) {
        return rejection("skill_destination_unavailable", {
          skill: invocation.skill,
          destination_id: destination.zone_id,
          required_affordance: "semantic_zone",
          observable_zone_ids: observation.interaction.zones.map(
            ({ zone_id: zoneId }) => zoneId
          )
        });
      }
    } else if (invocation.destination.type !== "world_pose") {
      const destinationId = invocation.destination.object_id;
      const destination = objects.get(destinationId);
      const expected = invocation.destination.type === "container"
        ? "container"
        : invocation.destination.type === "support_surface"
          ? "support_surface"
          : "insertable";
      if (!destination || destination.status !== "visible"
        || !destination.affordances.includes(expected)) {
        return rejection("skill_destination_unavailable", {
          skill: invocation.skill,
          destination_id: destinationId,
          required_affordance: expected,
          observed_status: destination?.status ?? "unknown",
          observed_affordances: destination?.affordances ?? []
        });
      }
      if (invocation.destination.type === "slot") {
        const slot = invocation.destination;
        const point = destination.interaction_points.find(
          ({ id }) => id === slot.interaction_point_id
        );
        if (!point || point.kind !== "insert" || !point.approach_direction_world) {
          return rejection("skill_destination_unavailable", {
            skill: invocation.skill,
            destination_id: destinationId,
            interaction_point_id: slot.interaction_point_id,
            reason: "slot placement requires an observed insert point and insertion direction"
          });
        }
      }
    }
  }
  if (invocation.skill === "regrasp") {
    const grasp = observation.interaction.manipulable_objects
      .find(({ object_id }) => object_id === invocation.object_id)?.grasp
      .find(({ hand }) => hand === invocation.from_hand);
    const carried = observation.interaction.carrying.bindings.some(
      ({ object_id, hand }) => object_id === invocation.object_id
        && hand === invocation.from_hand
    );
    if (!grasp?.verified && !carried) {
      return rejection("skill_precondition_failed", {
        skill: invocation.skill,
        object_id: invocation.object_id,
        from_hand: invocation.from_hand,
        reason: "source hand must have verified support before regrasp"
      });
    }
    const occupied = observation.interaction.carrying.bindings.find(
      ({ hand }) => hand === invocation.to_hand
    );
    if (occupied && occupied.object_id !== invocation.object_id) {
      return rejection("skill_precondition_failed", {
        skill: invocation.skill,
        object_id: invocation.object_id,
        to_hand: invocation.to_hand,
        occupied_by_object_id: occupied.object_id,
        reason: "destination hand is carrying another object"
      });
    }
  }
  const carrying = observation.interaction.carrying.bindings;
  if (invocation.skill === "carry" || invocation.skill === "carry_to_zone"
    || invocation.skill === "place"
    || invocation.skill === "bimanual_carry") {
    const boundHands = carrying
      .filter(({ object_id }) => object_id === invocation.object_id)
      .map(({ hand }) => hand);
    const requiredHands = invocation.skill === "bimanual_carry"
      ? ["left", "right"]
      : invocation.skill === "carry" || invocation.skill === "carry_to_zone"
        || invocation.skill === "place"
        ? invocation.hands === "both" ? ["left", "right"] : [invocation.hands]
        : [];
    if (!requiredHands.every((hand) => boundHands.includes(hand as "left" | "right"))) {
      return rejection("skill_precondition_failed", {
        skill: invocation.skill,
        object_id: invocation.object_id,
        reason: "required carried-object binding is not verified",
        required_hands: requiredHands,
        bound_hands: boundHands
      });
    }
  }
  if (invocation.skill === "lift") {
    const grasp = observation.interaction.manipulable_objects
      .find(({ object_id }) => object_id === invocation.object_id)?.grasp
      .find(({ hand }) => hand === invocation.hand);
    if (!grasp?.verified) {
      return rejection("skill_precondition_failed", {
        skill: invocation.skill,
        object_id: invocation.object_id,
        hand: invocation.hand,
        reason: "verified grasp is required before lift"
      });
    }
  }
  return null;
}

function manipulationBasePlacementChoices(
  placements: HumanoidWorldObservation["manipulationBasePlacements"]
): JsonValue {
  return placements.map((placement) => ({
    object_id: placement.objectId,
    interaction_point_id: placement.interactionPointId ?? null,
    hand_surface: placement.handSurface,
    root_world_target: placement.rootWorldTarget,
    root_yaw_radians: placement.rootYawRadians,
    ik_residual_m: placement.ikResidualMeters
  }));
}

function validateNavigationOutcome(
  binding: ActiveHumanoidSkillBinding,
  rawInput: unknown
): ReturnType<typeof rejection> | null {
  const input = record(rawInput);
  const target = vector(input?.target);
  if (!target) return rejection("skill_navigation_target_invalid", {});
  const invocation = binding.invocation;
  const expected = invocation.skill === "carry"
    || invocation.skill === "bimanual_carry"
    || invocation.skill === "retreat"
    ? invocation.target
    : invocation.skill === "carry_to_zone"
      ? binding.target_position
    : invocation.skill === "explore" || invocation.skill === "navigate_to_zone"
      ? binding.target_position
    : null;
  if (expected && distance(target, expected) > 1e-6) {
    return rejection("skill_navigation_target_mismatch", {
      skill: invocation.skill,
      requested_target: target,
      skill_target: expected
    });
  }
  if (invocation.skill === "approach" && binding.target_position) {
    const selectedPoint = invocation.interaction_point_id === null
      ? undefined
      : binding.eligible_interaction_points.find(
          ({ id }) => id === invocation.interaction_point_id
        );
    const referencePosition = selectedPoint?.world_position ?? binding.target_position;
    const standoff = Math.hypot(
      target.x - referencePosition.x,
      target.z - referencePosition.z
    );
    if (Math.abs(standoff - invocation.standoff_m) > 0.1) {
      return rejection("skill_navigation_target_mismatch", {
        skill: invocation.skill,
        object_id: invocation.object_id,
        interaction_point_id: invocation.interaction_point_id,
        requested_target: target,
        standoff_reference: selectedPoint ? "interaction_point" : "object_origin",
        observed_reference_position: referencePosition,
        requested_standoff_m: invocation.standoff_m,
        resulting_planar_standoff_m: standoff,
        tolerance_m: 0.1,
        recovery: "The navigation target may be physically reachable, but it is not authorized by the active approach Skill. Submit a new local Skill DAG whose approach.standoff_m matches this reachable target, bind that new node, then plan again."
      });
    }
  }
  return null;
}

function validateMotionOutcome(
  binding: ActiveHumanoidSkillBinding,
  rawInput: unknown
): ReturnType<typeof rejection> | null {
  const input = record(rawInput);
  const termination = record(input?.termination);
  const predicates = Array.isArray(termination?.predicates)
    ? termination.predicates.map(record).filter((value) => value !== null)
    : [];
  const invocation = binding.invocation;
  const phase = binding.phase;
  const objectId = invocationObjectId(invocation) ?? undefined;
  const has = (type: string, objectId?: string) => predicates.some((predicate) => (
    predicate?.type === type
      && (objectId === undefined || predicate.object_id === objectId)
  ));
  if ((invocation.skill === "open" || invocation.skill === "close")
    && phase === "actuate_joint") {
    const predicate = predicates.find((candidate) => (
      candidate?.type === "articulation_state"
        && candidate.object_id === invocation.object_id
        && candidate.joint_id === invocation.joint_id
        && candidate.state === invocation.skill
    ));
    const tolerance = typeof predicate?.tolerance === "number"
      ? predicate.tolerance
      : null;
    const strongEnough = tolerance !== null && (invocation.skill === "open"
      ? tolerance <= 1 - invocation.minimum_open_fraction + 1e-9
      : tolerance <= invocation.maximum_open_fraction + 1e-9);
    if (!strongEnough) {
      return rejection("skill_terminal_contract_mismatch", {
        skill: invocation.skill,
        phase,
        required_predicate: {
          type: "articulation_state",
          object_id: invocation.object_id,
          joint_id: invocation.joint_id,
          state: invocation.skill,
          threshold: invocation.skill === "open"
            ? invocation.minimum_open_fraction
            : invocation.maximum_open_fraction
        }
      });
    }
  }
  if (invocation.skill === "turn" && phase === "actuate_joint") {
    const articulation = binding.target_articulation;
    const predicate = predicates.find((candidate) => (
      candidate?.type === "articulation_displaced"
        && candidate.object_id === invocation.object_id
        && candidate.joint_id === invocation.joint_id
    ));
    const originPosition = number(predicate?.origin_position);
    const minimumDelta = number(predicate?.minimum_delta);
    const matched = articulation?.position !== null
      && articulation?.position !== undefined
      && predicate?.direction === invocation.direction
      && originPosition !== null
      && Math.abs(originPosition - articulation.position) <= 1e-6
      && minimumDelta !== null
      && minimumDelta >= invocation.rotation_radians;
    if (!matched) {
      return rejection("skill_terminal_contract_mismatch", {
        skill: invocation.skill,
        phase,
        required_predicate: {
          type: "articulation_displaced",
          object_id: invocation.object_id,
          joint_id: invocation.joint_id,
          origin_position: articulation?.position ?? null,
          direction: invocation.direction,
          minimum_delta: invocation.rotation_radians
        }
      });
    }
  }
  if ((invocation.skill === "push" || invocation.skill === "pull")
    && phase === "apply_force") {
    const mismatch = binding.target_articulation
      ? articulationDisplacementMismatch(binding, {
          directionWorld: invocation.direction_world,
          distanceMeters: invocation.distance_m,
          interactionPointId: invocation.interaction_point_id
        }, predicates)
      : objectDisplacementMismatch(binding, invocation, predicates);
    if (mismatch) return mismatch;
  }
  if (invocation.skill === "press" && phase === "press_stroke") {
    const articulation = binding.target_articulation;
    if (!articulation || articulation.position === null) {
      return rejection("skill_terminal_contract_mismatch", {
        skill: invocation.skill,
        phase,
        reason: "press requires an observable articulation"
      });
    }
    const directionWorld = articulation.open_position >= articulation.closed_position
      ? articulation.axis_world
      : scale(articulation.axis_world, -1);
    const mismatch = articulationDisplacementMismatch(binding, {
      directionWorld,
      distanceMeters: invocation.travel_m,
      interactionPointId: invocation.interaction_point_id
    }, predicates);
    if (mismatch) return mismatch;
  }
  if (invocation.skill === "grasp" && phase === "close_hand_under_contact"
    && !predicates.some((predicate) => predicate?.type === "grasp_verified"
      && predicate.object_id === invocation.object_id
      && predicate.hand === invocation.hand)) {
    return rejection("skill_terminal_contract_mismatch", {
      skill: invocation.skill,
      phase,
      required_predicate: "grasp_verified"
    });
  }
  if (invocation.skill === "regrasp" && phase === "transfer_grasp"
    && (!predicates.some((predicate) => predicate?.type === "grasp_verified"
      && predicate.object_id === invocation.object_id
      && predicate.hand === invocation.to_hand)
      || !predicates.some((predicate) => predicate?.type === "object_released"
        && predicate.object_id === invocation.object_id
        && predicate.hand === invocation.from_hand))) {
    return rejection("skill_terminal_contract_mismatch", {
      skill: invocation.skill,
      phase,
      required_predicates: [
        { type: "grasp_verified", hand: invocation.to_hand },
        { type: "object_released", hand: invocation.from_hand }
      ]
    });
  }
  if (invocation.skill === "bimanual_support"
    && phase === "establish_two_hand_contact"
    && !(["left", "right"] as const).every((hand) => predicates.some(
      (predicate) => predicate?.type === "grasp_verified"
        && predicate.object_id === invocation.object_id
        && predicate.hand === hand
    ))) {
    return rejection("skill_terminal_contract_mismatch", {
      skill: invocation.skill,
      phase,
      required_predicates: [
        { type: "grasp_verified", hand: "left" },
        { type: "grasp_verified", hand: "right" }
      ]
    });
  }
  if (invocation.skill === "lift" && phase === "lift_whole_body"
    && !predicates.some((predicate) => predicate?.type === "grasp_verified"
      && predicate.object_id === invocation.object_id
      && predicate.hand === invocation.hand)) {
    return rejection("skill_terminal_contract_mismatch", {
      skill: invocation.skill,
      phase,
      required_predicate: "grasp_verified"
    });
  }
  if (invocation.skill === "place" && phase === "settle_and_release"
    && ((invocation.release_after_settled
      && (!has("object_released", invocation.object_id)
        || !has("object_settled_on_support", invocation.object_id)))
      || (!invocation.release_after_settled
        && !(invocation.hands === "both"
          ? (["left", "right"] as const).every((hand) => predicates.some(
              (predicate) => predicate?.type === "grasp_verified"
                && predicate.object_id === invocation.object_id
                && predicate.hand === hand
            ))
          : predicates.some((predicate) => predicate?.type === "grasp_verified"
              && predicate.object_id === invocation.object_id
              && predicate.hand === invocation.hands)))
      || !placeRelationPredicatePresent(invocation, predicates))) {
    return rejection("skill_terminal_contract_mismatch", {
      skill: invocation.skill,
      phase,
      required_predicates: [
        ...(invocation.release_after_settled
          ? ["object_released", "object_settled_on_support"]
          : invocation.hands === "both"
            ? [
                { type: "grasp_verified", hand: "left" },
                { type: "grasp_verified", hand: "right" }
              ]
            : [{ type: "grasp_verified", hand: invocation.hands }]),
        placeRelationPredicateName(invocation)
      ]
    });
  }
  if (invocation.skill === "stabilize" && phase === "recover_support"
    && !predicates.some((predicate) => predicate?.type === "balance_stable"
      && typeof predicate.minimum_support_margin_m === "number"
      && predicate.minimum_support_margin_m
        >= invocation.minimum_support_margin_m)) {
    return rejection("skill_terminal_contract_mismatch", {
      skill: invocation.skill,
      phase,
      required_predicate: {
        type: "balance_stable",
        minimum_support_margin_m: invocation.minimum_support_margin_m
      }
    });
  }
  if ((phase === "reach_handle" || phase === "reach_interaction"
    || phase === "solve_whole_body_reach" || phase === "establish_contact")
    && objectId
    && !has("hand_contact_object", objectId)
    && !has("hand_contact_object_any", objectId)
    && !has("hand_contact_object_region", objectId)
    && !has("body_contact_object", objectId)
    && !has("end_effector_near_point")) {
    return rejection("skill_terminal_contract_mismatch", {
      skill: invocation.skill,
      phase,
      required_predicates: [
        "hand_contact_object",
        "hand_contact_object_any",
        "hand_contact_object_region",
        "body_contact_object",
        "end_effector_near_point"
      ]
    });
  }
  return null;
}

function objectDisplacementMismatch(
  binding: ActiveHumanoidSkillBinding,
  invocation: Extract<HumanoidSkillInvocation, { skill: "push" | "pull" }>,
  predicates: Array<Record<string, unknown> | null>
): ReturnType<typeof rejection> | null {
  const origin = binding.target_position;
  const predicate = predicates.find((candidate) => (
    candidate?.type === "object_displaced"
      && candidate.object_id === invocation.object_id
  ));
  const predicateOrigin = vector(predicate?.origin);
  const predicateDirection = vector(predicate?.direction_world);
  const minimumDistance = number(predicate?.minimum_distance_m);
  const maximumLateralError = number(predicate?.maximum_lateral_error_m);
  const matched = origin && predicateOrigin && predicateDirection
    && distance(origin, predicateOrigin) <= 1e-6
    && dot(predicateDirection, invocation.direction_world) >= 1 - 1e-6
    && minimumDistance !== null && minimumDistance >= invocation.distance_m
    && maximumLateralError !== null
    && maximumLateralError <= Math.max(0.05, invocation.distance_m * 0.25);
  return matched ? null : rejection("skill_terminal_contract_mismatch", {
    skill: invocation.skill,
    phase: "apply_force",
    required_predicate: {
      type: "object_displaced",
      object_id: invocation.object_id,
      origin,
      direction_world: invocation.direction_world,
      minimum_distance_m: invocation.distance_m,
      maximum_lateral_error_m: Math.max(0.05, invocation.distance_m * 0.25)
    }
  });
}

function articulationDisplacementMismatch(
  binding: ActiveHumanoidSkillBinding,
  request: {
    directionWorld: Vec3;
    distanceMeters: number;
    interactionPointId: string | null;
  },
  predicates: Array<Record<string, unknown> | null>
): ReturnType<typeof rejection> | null {
  const articulation = binding.target_articulation;
  const point = binding.eligible_interaction_points.find(
    ({ id }) => id === request.interactionPointId
  );
  if (!articulation || articulation.position === null || !point) {
    return rejection("skill_terminal_contract_mismatch", {
      skill: binding.invocation.skill,
      phase: binding.phase,
      reason: "articulated displacement requires a selected observable interaction point"
    });
  }
  const displacement = articulation.type === "slide"
    ? {
        direction: dot(request.directionWorld, articulation.axis_world) >= 0
          ? "increasing" as const : "decreasing" as const,
        minimumDelta: request.distanceMeters
      }
    : hingeDisplacement(
        articulation.axis_world,
        articulation.anchor_world,
        point.world_position,
        request.directionWorld,
        request.distanceMeters
      );
  const objectId = invocationObjectId(binding.invocation);
  if (!objectId) {
    return rejection("skill_terminal_contract_mismatch", {
      skill: binding.invocation.skill,
      phase: binding.phase,
      reason: "articulated skill has no object identity"
    });
  }
  const predicate = predicates.find((candidate) => (
    candidate?.type === "articulation_displaced"
      && candidate.object_id === objectId
      && candidate.joint_id === articulation.joint_id
  ));
  const originPosition = number(predicate?.origin_position);
  const minimumDelta = number(predicate?.minimum_delta);
  const matched = predicate?.direction === displacement.direction
    && originPosition !== null
    && Math.abs(originPosition - articulation.position) <= 1e-6
    && minimumDelta !== null
    && minimumDelta >= displacement.minimumDelta;
  return matched ? null : rejection("skill_terminal_contract_mismatch", {
    skill: binding.invocation.skill,
    phase: binding.phase,
    required_predicate: {
      type: "articulation_displaced",
      object_id: objectId,
      joint_id: articulation.joint_id,
      origin_position: articulation.position,
      direction: displacement.direction,
      minimum_delta: displacement.minimumDelta
    }
  });
}

function hingeDisplacement(
  axis: Vec3,
  anchor: Vec3,
  point: Vec3,
  directionWorld: Vec3,
  distanceMeters: number
): { direction: "increasing" | "decreasing"; minimumDelta: number } {
  const radial = subtract(point, anchor);
  const tangent = cross(axis, radial);
  const radius = length(tangent);
  if (radius <= 1e-6) {
    return { direction: "increasing", minimumDelta: Number.POSITIVE_INFINITY };
  }
  return {
    direction: dot(directionWorld, tangent) >= 0 ? "increasing" : "decreasing",
    minimumDelta: distanceMeters / radius
  };
}

function placeRelationPredicatePresent(
  invocation: Extract<HumanoidSkillInvocation, { skill: "place" }>,
  predicates: Array<Record<string, unknown> | null>
): boolean {
  const relation = predicates.some((predicate) => {
    if (!predicate || predicate.object_id !== invocation.object_id) return false;
    if (invocation.destination.type === "semantic_zone") {
      return predicate.type === "object_in_zone"
        && predicate.zone_id === invocation.destination.zone_id
        && predicate.expected === true
        && typeof predicate.tolerance_m === "number"
        && predicate.tolerance_m <= invocation.destination.tolerance_m;
    }
    if (invocation.destination.type === "container") {
      return predicate.type === "object_inside"
        && predicate.container_id === invocation.destination.object_id
        && predicate.expected === true;
    }
    if (invocation.destination.type === "support_surface") {
      return predicate.type === "object_on"
        && predicate.support_id === invocation.destination.object_id
        && predicate.expected === true;
    }
    if (invocation.destination.type === "world_pose") {
      const target = vector(predicate.target);
      return predicate.type === "object_near_point" && target !== null
        && distance(target, invocation.destination.position) <= 1e-6
        && typeof predicate.tolerance_m === "number"
        && predicate.tolerance_m <= invocation.destination.position_tolerance_m;
    }
    return predicate.type === "object_near_point";
  });
  return relation;
}

function placeRelationPredicateName(
  invocation: Extract<HumanoidSkillInvocation, { skill: "place" }>
): string {
  if (invocation.destination.type === "semantic_zone") return "object_in_zone";
  if (invocation.destination.type === "container") return "object_inside";
  if (invocation.destination.type === "support_surface") return "object_on";
  return "object_near_point";
}

function interactionPointIds(invocation: HumanoidSkillInvocation): string[] {
  if (invocation.skill === "bimanual_support") {
    return [
      invocation.left_interaction_point_id,
      invocation.right_interaction_point_id
    ];
  }
  return "interaction_point_id" in invocation
    && invocation.interaction_point_id !== null
    ? [invocation.interaction_point_id]
    : [];
}

function interactionPointKinds(skill: HumanoidSkillInvocation["skill"]): string[] {
  if (skill === "grasp" || skill === "regrasp" || skill === "bimanual_support") {
    return ["grasp"];
  }
  if (skill === "push") return ["push", "grasp"];
  if (skill === "pull" || skill === "open") return ["pull", "grasp", "turn"];
  if (skill === "close") return ["push", "pull", "grasp", "turn"];
  if (skill === "turn") return ["turn", "grasp"];
  if (skill === "press") return ["press"];
  return [];
}

function skillNeedsInteractionPoint(skill: HumanoidSkillInvocation["skill"]): boolean {
  return [
    "reach", "grasp", "push", "pull", "press", "open", "close", "turn",
    "bimanual_support"
  ].includes(skill);
}

function invocationHand(invocation: HumanoidSkillInvocation): "left" | "right" | null {
  if ("hand" in invocation) return invocation.hand ?? null;
  return null;
}

function invocationObjectId(invocation: HumanoidSkillInvocation): string | null {
  return "object_id" in invocation ? invocation.object_id : null;
}

function skillTransactionId(rawInput: unknown): string | null {
  const value = record(rawInput)?.skill_transaction_id;
  return typeof value === "string" && value ? value : null;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function vector(value: unknown): Vec3 | null {
  const input = record(value);
  return input && [input.x, input.y, input.z].every((part) => (
    typeof part === "number" && Number.isFinite(part)
  ))
    ? { x: input.x as number, y: input.y as number, z: input.z as number }
    : null;
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function subtract(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function scale(value: Vec3, scalar: number): Vec3 {
  return { x: value.x * scalar, y: value.y * scalar, z: value.z * scalar };
}

function dot(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function cross(left: Vec3, right: Vec3): Vec3 {
  return {
    x: left.y * right.z - left.z * right.y,
    y: left.z * right.x - left.x * right.z,
    z: left.x * right.y - left.y * right.x
  };
}

function length(value: Vec3): number {
  return Math.hypot(value.x, value.y, value.z);
}

function rejection(code: string, detail: Record<string, unknown>): {
  accepted: false;
  code: string;
  detail: JsonValue;
} {
  return {
    accepted: false,
    code,
    detail: JSON.parse(JSON.stringify({ automatic_actuation: false, ...detail })) as JsonValue
  };
}
