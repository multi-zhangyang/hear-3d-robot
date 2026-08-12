import { z } from "zod";
import {
  QuaternionSchema,
  Vec3Schema,
  type JsonValue,
  type Quaternion,
  type Vec3
} from "../../domain/schema.js";
import {
  HUMANOID_SKILL_CONTRACTS,
  type HumanoidSkillInvocation
} from "../../domain/humanoid-skill.js";
import {
  HumanoidSkillPlanProposalSchema,
  type HumanoidSkillPlanNode,
  type HumanoidSkillPlanProposal
} from "../../domain/humanoid-skill-plan.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import type { ActiveHumanoidSkillBinding } from "./skill-binding.js";

export interface RegisteredHumanoidSkillPlan {
  protocol: "humanoid-skill-plan-v1";
  transaction_id: string;
  agent_id: string;
  proposal: HumanoidSkillPlanProposal;
  proposal_sha256: string;
  selected_strategy_id: string;
  observed_frame: number;
  world_revision: number;
  physical_anchor?: {
    root_position: Vec3;
    root_rotation: Quaternion;
    carried_object_ids: string[];
    carried_bindings?: Array<{
      object_id: string;
      hand: "left" | "right";
    }> | undefined;
    object_poses: Array<{
      object_id: string;
      position: Vec3;
      rotation: Quaternion;
      articulation?: {
        joint_id: string;
        position: number | null;
      } | null | undefined;
    }>;
  } | undefined;
  in_progress_phase?: {
    node_id: string;
    phase: string;
  } | undefined;
  completed_node_ids: string[];
  completed_phases_by_node: Record<string, string[]>;
}

export const RegisteredHumanoidSkillPlanSchema = z.object({
  protocol: z.literal("humanoid-skill-plan-v1"),
  transaction_id: z.string().trim().min(1),
  agent_id: z.string().trim().min(1),
  proposal: HumanoidSkillPlanProposalSchema,
  proposal_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  selected_strategy_id: z.string().trim().min(1),
  observed_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  physical_anchor: z.object({
    root_position: Vec3Schema,
    root_rotation: QuaternionSchema,
    carried_object_ids: z.array(z.string().trim().min(1)).max(16),
    carried_bindings: z.array(z.object({
      object_id: z.string().trim().min(1),
      hand: z.enum(["left", "right"])
    }).strict()).max(32).optional(),
    object_poses: z.array(z.object({
      object_id: z.string().trim().min(1),
      position: Vec3Schema,
      rotation: QuaternionSchema,
      articulation: z.object({
        joint_id: z.string().trim().min(1),
        position: z.number().finite().nullable()
      }).strict().nullable().optional()
    }).strict()).max(16)
  }).strict().superRefine((anchor, context) => {
    const sorted = [...anchor.carried_object_ids].sort();
    if (new Set(sorted).size !== sorted.length
      || JSON.stringify(anchor.carried_object_ids) !== JSON.stringify(sorted)) {
      context.addIssue({
        code: "custom",
        path: ["carried_object_ids"],
        message: "Skill plan carried-object anchor must be unique and sorted"
      });
    }
    if (anchor.carried_bindings) {
      const bindingKeys = anchor.carried_bindings.map(
        ({ object_id: objectId, hand }) => `${objectId}\0${hand}`
      );
      const sortedBindingKeys = [...bindingKeys].sort();
      if (new Set(sortedBindingKeys).size !== sortedBindingKeys.length
        || JSON.stringify(bindingKeys) !== JSON.stringify(sortedBindingKeys)) {
        context.addIssue({
          code: "custom",
          path: ["carried_bindings"],
          message: "Skill plan carried-object bindings must be unique and sorted"
        });
      }
    }
    const objectIds = anchor.object_poses.map(({ object_id }) => object_id);
    const sortedObjectIds = [...objectIds].sort();
    if (new Set(sortedObjectIds).size !== sortedObjectIds.length
      || JSON.stringify(objectIds) !== JSON.stringify(sortedObjectIds)) {
      context.addIssue({
        code: "custom",
        path: ["object_poses"],
        message: "Skill plan object-pose anchors must be unique and sorted"
      });
    }
  }).optional(),
  in_progress_phase: z.object({
    node_id: z.string().trim().min(1),
    phase: z.string().trim().min(1)
  }).strict().optional(),
  completed_node_ids: z.array(z.string().trim().min(1)).max(16),
  completed_phases_by_node: z.record(
    z.string().trim().min(1),
    z.array(z.string().trim().min(1)).max(16)
  ).default({})
}).strict().superRefine((plan, context) => {
  if (plan.proposal_sha256 !== modelPayloadSha256(plan.proposal)) {
    context.addIssue({
      code: "custom",
      path: ["proposal_sha256"],
      message: "Registered Skill plan proposal identity is invalid"
    });
  }
  if (plan.selected_strategy_id !== plan.proposal.selected_strategy_id) {
    context.addIssue({
      code: "custom",
      path: ["selected_strategy_id"],
      message: "Registered Skill plan selection does not match its proposal"
    });
  }
  const selected = plan.proposal.strategies.find(
    ({ strategy_id }) => strategy_id === plan.selected_strategy_id
  );
  const knownNodeIds = new Set(selected?.nodes.map(({ node_id }) => node_id) ?? []);
  if (new Set(plan.completed_node_ids).size !== plan.completed_node_ids.length
    || plan.completed_node_ids.some((nodeId) => !knownNodeIds.has(nodeId))) {
    context.addIssue({
      code: "custom",
      path: ["completed_node_ids"],
      message: "Registered Skill plan completion state is invalid"
    });
  }
  for (const [nodeId, phases] of Object.entries(
    plan.completed_phases_by_node
  )) {
    const node = selected?.nodes.find((candidate) => candidate.node_id === nodeId);
    const actionable = node ? actionableSkillPhases(node.invocation) : [];
    const isPrefix = phases.every((phase, index) => actionable[index] === phase);
    if (!node || new Set(phases).size !== phases.length || !isPrefix) {
      context.addIssue({
        code: "custom",
        path: ["completed_phases_by_node", nodeId],
        message: "Registered Skill phase progress is invalid"
      });
      continue;
    }
    const nodeCompleted = plan.completed_node_ids.includes(nodeId);
    if (nodeCompleted !== (phases.length === actionable.length)) {
      context.addIssue({
        code: "custom",
        path: ["completed_phases_by_node", nodeId],
        message: "Registered Skill node and phase completion disagree"
      });
    }
  }
  if (plan.in_progress_phase) {
    const node = selected?.nodes.find(
      ({ node_id: nodeId }) => nodeId === plan.in_progress_phase?.node_id
    );
    const completed = node
      ? plan.completed_phases_by_node[node.node_id] ?? []
      : [];
    const expectedPhase = node
      ? actionableSkillPhases(node.invocation)[completed.length]
      : undefined;
    if (!node || plan.completed_node_ids.includes(node.node_id)
      || expectedPhase !== plan.in_progress_phase.phase) {
      context.addIssue({
        code: "custom",
        path: ["in_progress_phase"],
        message: "Registered Skill in-progress phase is not the next incomplete DAG phase"
      });
    }
  }
});

export function registerHumanoidSkillPlan(input: {
  transactionId: string;
  agentId: string;
  proposal: HumanoidSkillPlanProposal;
  observedFrame: number;
  worldRevision: number;
  physicalAnchor: NonNullable<RegisteredHumanoidSkillPlan["physical_anchor"]>;
}): RegisteredHumanoidSkillPlan {
  const proposal = compileHumanoidSkillDependencies(
    HumanoidSkillPlanProposalSchema.parse(input.proposal)
  );
  return RegisteredHumanoidSkillPlanSchema.parse({
    protocol: "humanoid-skill-plan-v1",
    transaction_id: input.transactionId,
    agent_id: input.agentId,
    proposal: structuredClone(proposal),
    proposal_sha256: modelPayloadSha256(proposal),
    selected_strategy_id: proposal.selected_strategy_id,
    observed_frame: input.observedFrame,
    world_revision: input.worldRevision,
    physical_anchor: structuredClone(input.physicalAnchor),
    completed_node_ids: [],
    completed_phases_by_node: {}
  });
}

/**
 * Returns the exact bindings the Motion Agent may choose next from the
 * selected strategy.  This is a projection of the model-authored DAG, not a
 * scheduler decision: parallel dependency-ready nodes are all preserved so
 * the model still owns the semantic choice.
 */
export function readyHumanoidSkillPlanBindings(
  plan: RegisteredHumanoidSkillPlan
): Array<{
  skill_plan_transaction_id: string;
  skill_node_id: string;
  invocation: HumanoidSkillInvocation;
  phase: string;
}> {
  const strategy = plan.proposal.strategies.find(
    ({ strategy_id: strategyId }) => strategyId === plan.selected_strategy_id
  );
  if (!strategy) return [];
  return strategy.nodes.flatMap((node) => {
    if (plan.completed_node_ids.includes(node.node_id)
      || node.depends_on_node_ids.some((dependency) => (
        !plan.completed_node_ids.includes(dependency)
      ))) return [];
    const phases = actionableSkillPhases(node.invocation);
    const completed = completedSkillPhases(plan, node.node_id, node.invocation);
    const phase = phases[completed.length];
    return phase === undefined
      ? []
      : [{
          skill_plan_transaction_id: plan.transaction_id,
          skill_node_id: node.node_id,
          invocation: structuredClone(node.invocation),
          phase
        }];
  });
}

export function compileHumanoidSkillDependencies(
  proposal: HumanoidSkillPlanProposal
): HumanoidSkillPlanProposal {
  const compiled = structuredClone(proposal);
  for (const strategy of compiled.strategies) {
    strategy.nodes.forEach((node, index) => {
      const prerequisiteSkills = HUMANOID_SKILL_CONTRACTS[
        node.invocation.skill
      ].prerequisite_skill_groups;
      for (const alternatives of prerequisiteSkills) {
        const prerequisite = strategy.nodes.slice(0, index).findLast((candidate) => (
          alternatives.includes(candidate.invocation.skill)
            && invocationsShareTarget(candidate.invocation, node.invocation)
            && handsCompatible(candidate.invocation, node.invocation)
        ));
        if (prerequisite
          && !node.depends_on_node_ids.includes(prerequisite.node_id)) {
          node.depends_on_node_ids.push(prerequisite.node_id);
        }
      }
      node.depends_on_node_ids.sort();
    });
  }
  return HumanoidSkillPlanProposalSchema.parse(compiled);
}

function invocationsShareTarget(
  left: HumanoidSkillInvocation,
  right: HumanoidSkillInvocation
): boolean {
  return "object_id" in left && "object_id" in right
    && left.object_id === right.object_id;
}

function handsCompatible(
  left: HumanoidSkillInvocation,
  right: HumanoidSkillInvocation
): boolean {
  const leftHand = "hand" in left ? left.hand : null;
  const rightHand = "hand" in right ? right.hand : null;
  return leftHand === null || rightHand === null || leftHand === rightHand;
}

export function authorizeHumanoidSkillPlanNode(input: {
  plan: RegisteredHumanoidSkillPlan | undefined;
  planTransactionId: string | null;
  nodeId: string | null;
  invocation: HumanoidSkillInvocation;
  phase: string;
  agentId: string;
  currentWorldRevision: number;
}):
  | { accepted: true; node: HumanoidSkillPlanNode }
  | { accepted: false; code: string; detail: JsonValue } {
  const plan = input.plan;
  if (!plan || input.planTransactionId !== plan.transaction_id) {
    return rejection("skill_plan_reference_missing", {
      supplied_skill_plan_transaction_id: input.planTransactionId,
      registered_skill_plan_transaction_id: plan?.transaction_id ?? null
    });
  }
  if (plan.agent_id !== input.agentId) {
    return rejection("skill_plan_agent_mismatch", {
      skill_plan_agent_id: plan.agent_id,
      invoking_agent_id: input.agentId
    });
  }
  if (plan.world_revision !== input.currentWorldRevision) {
    return rejection("skill_plan_world_revision_stale", {
      skill_plan_world_revision: plan.world_revision,
      current_world_revision: input.currentWorldRevision,
      recovery: "Submit a new local Skill DAG from the latest observation"
    });
  }
  const strategy = plan.proposal.strategies.find(
    ({ strategy_id }) => strategy_id === plan.selected_strategy_id
  );
  const node = strategy?.nodes.find(({ node_id }) => node_id === input.nodeId);
  if (!node) {
    return rejection("skill_plan_node_unknown", {
      selected_strategy_id: plan.selected_strategy_id,
      requested_node_id: input.nodeId,
      available_node_ids: strategy?.nodes.map(({ node_id }) => node_id) ?? []
    });
  }
  if (modelPayloadSha256(node.invocation) !== modelPayloadSha256(input.invocation)) {
    return rejection("skill_plan_invocation_mismatch", {
      node_id: node.node_id,
      planned_invocation_sha256: modelPayloadSha256(node.invocation),
      supplied_invocation_sha256: modelPayloadSha256(input.invocation)
    });
  }
  const incomplete = node.depends_on_node_ids.filter(
    (dependency) => !plan.completed_node_ids.includes(dependency)
  );
  if (incomplete.length > 0) {
    return rejection("skill_plan_dependencies_incomplete", {
      node_id: node.node_id,
      incomplete_dependency_node_ids: incomplete,
      completed_node_ids: plan.completed_node_ids
    });
  }
  if (plan.completed_node_ids.includes(node.node_id)) {
    return rejection("skill_plan_node_completed", {
      node_id: node.node_id,
      completed_node_ids: plan.completed_node_ids
    });
  }
  const actionablePhases = actionableSkillPhases(node.invocation);
  const completedPhases = completedSkillPhases(plan, node.node_id, node.invocation);
  const expectedPhase = actionablePhases[completedPhases.length];
  if (input.phase !== expectedPhase) {
    return rejection("skill_plan_phase_out_of_order", {
      node_id: node.node_id,
      requested_phase: input.phase,
      expected_phase: expectedPhase ?? null,
      completed_phases: completedPhases,
      actionable_phases: actionablePhases
    });
  }
  return { accepted: true, node: structuredClone(node) };
}

export function advanceHumanoidSkillPlan(input: {
  plan: RegisteredHumanoidSkillPlan;
  binding: ActiveHumanoidSkillBinding;
  worldRevision: number;
  physicalAnchor: NonNullable<RegisteredHumanoidSkillPlan["physical_anchor"]>;
  executionSucceeded: boolean;
  phasePostconditionSatisfied: boolean;
}): RegisteredHumanoidSkillPlan | null {
  if (!input.executionSucceeded) return null;
  const next = structuredClone(input.plan);
  next.world_revision = input.worldRevision;
  next.physical_anchor = structuredClone(input.physicalAnchor);
  const nodeId = input.binding.skill_node_id;
  const strategy = next.proposal.strategies.find(
    ({ strategy_id: strategyId }) => strategyId === next.selected_strategy_id
  );
  const node = strategy?.nodes.find(({ node_id: candidateId }) => candidateId === nodeId);
  if (!node || nodeId === null) {
    throw new Error("Executed Skill binding does not reference its selected DAG node");
  }
  const actionablePhases = actionableSkillPhases(node.invocation);
  const completedPhases = completedSkillPhases(next, nodeId, node.invocation);
  const expectedPhase = actionablePhases[completedPhases.length];
  if (input.binding.phase !== expectedPhase) {
    throw new Error(
      `Executed Skill phase is out of order: expected ${expectedPhase ?? "none"}, `
      + `received ${input.binding.phase}`
    );
  }
  // A successful controller transaction can be only one bounded physical
  // chunk of a semantic phase (for example, 3 m of navigate_to_zone).  Keep
  // the model-authored DAG and refresh its physical anchor, but do not claim
  // phase or node completion until the phase's world-state postcondition is
  // actually true.
  if (!input.phasePostconditionSatisfied) {
    next.in_progress_phase = {
      node_id: nodeId,
      phase: input.binding.phase
    };
    return RegisteredHumanoidSkillPlanSchema.parse(next);
  }
  delete next.in_progress_phase;
  const advancedPhases = [...completedPhases, input.binding.phase];
  next.completed_phases_by_node = {
    ...next.completed_phases_by_node,
    [nodeId]: advancedPhases
  };
  if (advancedPhases.length === actionablePhases.length
    && !next.completed_node_ids.includes(nodeId)) {
    next.completed_node_ids.push(nodeId);
    next.completed_node_ids.sort();
  }
  return RegisteredHumanoidSkillPlanSchema.parse(next);
}

function actionableSkillPhases(
  invocation: HumanoidSkillInvocation
): string[] {
  return HUMANOID_SKILL_CONTRACTS[invocation.skill].process
    .filter(({ authority }) => authority === "navigation"
      || authority === "whole_body" || authority === "grasp")
    .map(({ phase }) => phase);
}

function completedSkillPhases(
  plan: RegisteredHumanoidSkillPlan,
  nodeId: string,
  invocation: HumanoidSkillInvocation
): string[] {
  const explicit = plan.completed_phases_by_node[nodeId];
  if (explicit) return [...explicit];
  return plan.completed_node_ids.includes(nodeId)
    ? actionableSkillPhases(invocation)
    : [];
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
