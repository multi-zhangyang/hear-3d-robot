import { z } from "zod";
import { HumanoidSkillInvocationSchema } from "./humanoid-skill.js";

const HumanoidSkillPlanNodeSchema = z.object({
  node_id: z.string().trim().min(1),
  invocation: HumanoidSkillInvocationSchema,
  depends_on_node_ids: z.array(z.string().trim().min(1)).max(15)
}).strict();

const HumanoidSkillPlanStrategySchema = z.object({
  strategy_id: z.string().trim().min(1),
  rationale: z.string().trim().min(1).max(2_000),
  nodes: z.array(HumanoidSkillPlanNodeSchema).min(1).max(16)
}).strict().superRefine((strategy, context) => {
  const ids = strategy.nodes.map(({ node_id }) => node_id);
  if (new Set(ids).size !== ids.length) {
    context.addIssue({
      code: "custom",
      path: ["nodes"],
      message: "Skill DAG node identities must be unique within a strategy"
    });
    return;
  }
  const known = new Set(ids);
  strategy.nodes.forEach((node, nodeIndex) => {
    if (new Set(node.depends_on_node_ids).size !== node.depends_on_node_ids.length) {
      context.addIssue({
        code: "custom",
        path: ["nodes", nodeIndex, "depends_on_node_ids"],
        message: "Skill DAG dependencies must be unique"
      });
    }
    for (const dependency of node.depends_on_node_ids) {
      if (!known.has(dependency)) {
        context.addIssue({
          code: "custom",
          path: ["nodes", nodeIndex, "depends_on_node_ids"],
          message: `Unknown Skill DAG dependency: ${dependency}`
        });
      }
      if (dependency === node.node_id) {
        context.addIssue({
          code: "custom",
          path: ["nodes", nodeIndex, "depends_on_node_ids"],
          message: "A Skill DAG node cannot depend on itself"
        });
      }
    }
  });
  if (hasCycle(strategy.nodes)) {
    context.addIssue({
      code: "custom",
      path: ["nodes"],
      message: "Skill plan dependencies must form an acyclic graph"
    });
  }
});

export const HumanoidSkillPlanProposalSchema = z.object({
  objective: z.string().trim().min(1).max(1000),
  strategies: z.array(HumanoidSkillPlanStrategySchema).min(1).max(3),
  selected_strategy_id: z.string().trim().min(1)
}).strict().superRefine((proposal, context) => {
  const strategyIds = proposal.strategies.map(({ strategy_id }) => strategy_id);
  if (new Set(strategyIds).size !== strategyIds.length) {
    context.addIssue({
      code: "custom",
      path: ["strategies"],
      message: "Skill plan strategy identities must be unique"
    });
  }
  if (!strategyIds.includes(proposal.selected_strategy_id)) {
    context.addIssue({
      code: "custom",
      path: ["selected_strategy_id"],
      message: "The selected Skill strategy must exist in the proposal"
    });
  }
});

export type HumanoidSkillPlanProposal = z.infer<
  typeof HumanoidSkillPlanProposalSchema
>;
export type HumanoidSkillPlanNode = HumanoidSkillPlanProposal[
  "strategies"
][number]["nodes"][number];

function hasCycle(nodes: readonly HumanoidSkillPlanNode[]): boolean {
  const dependencies = new Map(nodes.map((node) => [
    node.node_id,
    node.depends_on_node_ids
  ]));
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const visit = (nodeId: string): boolean => {
    if (visiting.has(nodeId)) return true;
    if (visited.has(nodeId)) return false;
    visiting.add(nodeId);
    for (const dependency of dependencies.get(nodeId) ?? []) {
      if (visit(dependency)) return true;
    }
    visiting.delete(nodeId);
    visited.add(nodeId);
    return false;
  };
  return nodes.some(({ node_id }) => visit(node_id));
}
