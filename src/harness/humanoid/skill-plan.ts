import { z } from "zod";
import type { JsonValue } from "../../domain/schema.js";
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
  completed_node_ids: string[];
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
  completed_node_ids: z.array(z.string().trim().min(1)).max(16)
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
});

export function registerHumanoidSkillPlan(input: {
  transactionId: string;
  agentId: string;
  proposal: HumanoidSkillPlanProposal;
  observedFrame: number;
  worldRevision: number;
}): RegisteredHumanoidSkillPlan {
  const proposal = HumanoidSkillPlanProposalSchema.parse(input.proposal);
  return RegisteredHumanoidSkillPlanSchema.parse({
    protocol: "humanoid-skill-plan-v1",
    transaction_id: input.transactionId,
    agent_id: input.agentId,
    proposal: structuredClone(proposal),
    proposal_sha256: modelPayloadSha256(proposal),
    selected_strategy_id: proposal.selected_strategy_id,
    observed_frame: input.observedFrame,
    world_revision: input.worldRevision,
    completed_node_ids: []
  });
}

export function authorizeHumanoidSkillPlanNode(input: {
  plan: RegisteredHumanoidSkillPlan | undefined;
  planTransactionId: string | null;
  nodeId: string | null;
  invocation: HumanoidSkillInvocation;
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
  return { accepted: true, node: structuredClone(node) };
}

export function advanceHumanoidSkillPlan(input: {
  plan: RegisteredHumanoidSkillPlan;
  binding: ActiveHumanoidSkillBinding;
  worldRevision: number;
  executionSucceeded: boolean;
}): RegisteredHumanoidSkillPlan | null {
  if (!input.executionSucceeded) return null;
  const next = structuredClone(input.plan);
  next.world_revision = input.worldRevision;
  if (humanoidSkillPhaseCompletesNode(input.binding)) {
    const nodeId = input.binding.skill_node_id;
    if (nodeId && !next.completed_node_ids.includes(nodeId)) {
      next.completed_node_ids.push(nodeId);
      next.completed_node_ids.sort();
    }
  }
  return next;
}

function humanoidSkillPhaseCompletesNode(
  binding: ActiveHumanoidSkillBinding
): boolean {
  const actionable = HUMANOID_SKILL_CONTRACTS[binding.invocation.skill].process
    .filter(({ authority }) => authority === "navigation"
      || authority === "whole_body" || authority === "grasp");
  return actionable.at(-1)?.phase === binding.phase;
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
