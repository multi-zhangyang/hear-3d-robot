import { describe, expect, it } from "vitest";
import {
  advanceHumanoidSkillPlan,
  authorizeHumanoidSkillPlanNode,
  compileHumanoidSkillDependencies,
  registerHumanoidSkillPlan
} from "./skill-plan.js";
import type { ActiveHumanoidSkillBinding } from "./skill-binding.js";

const proposal = {
  objective: "approach then open",
  strategies: [{
    strategy_id: "selected",
    rationale: "use the currently visible handle",
    nodes: [{
      node_id: "approach",
      invocation: {
        skill: "approach" as const,
        object_id: "door",
        interaction_point_id: "handle",
        standoff_m: 0.6
      },
      depends_on_node_ids: []
    }, {
      node_id: "open",
      invocation: {
        skill: "open" as const,
        object_id: "door",
        interaction_point_id: "handle",
        joint_id: "hinge",
        hand: "right" as const,
        minimum_open_fraction: 0.85
      },
      depends_on_node_ids: ["approach"]
    }]
  }],
  selected_strategy_id: "selected"
};

describe("registered humanoid Skill DAG", () => {
  it("adds dependencies between compatible model-proposed manipulation nodes", () => {
    const compiled = compileHumanoidSkillDependencies({
      objective: "取起物体",
      selected_strategy_id: "direct",
      strategies: [{
        strategy_id: "direct",
        rationale: "使用实时交互点",
        nodes: [{
          node_id: "approach",
          invocation: {
            skill: "approach",
            object_id: "part",
            interaction_point_id: "grasp",
            standoff_m: 0.4
          },
          depends_on_node_ids: []
        }, {
          node_id: "reach",
          invocation: {
            skill: "reach",
            object_id: "part",
            interaction_point_id: "grasp",
            hand: "right",
            tolerance_m: 0.05
          },
          depends_on_node_ids: []
        }, {
          node_id: "grasp",
          invocation: {
            skill: "grasp",
            object_id: "part",
            interaction_point_id: "grasp",
            hand: "right"
          },
          depends_on_node_ids: []
        }, {
          node_id: "lift",
          invocation: {
            skill: "lift",
            object_id: "part",
            hand: "right",
            clearance_m: 0.2
          },
          depends_on_node_ids: []
        }]
      }]
    });
    expect(compiled.strategies[0]!.nodes.map((node) => ({
      id: node.node_id,
      dependencies: node.depends_on_node_ids
    }))).toEqual([
      { id: "approach", dependencies: [] },
      { id: "reach", dependencies: ["approach"] },
      { id: "grasp", dependencies: ["approach", "reach"] },
      { id: "lift", dependencies: ["grasp"] }
    ]);
  });

  it("authorizes only exact nodes with completed dependencies and advances on terminal phases", () => {
    const registered = registerHumanoidSkillPlan({
      transactionId: "skill-plan-1",
      agentId: "motion",
      proposal,
      observedFrame: 4,
      worldRevision: 5
    });
    expect(authorizeHumanoidSkillPlanNode({
      plan: registered,
      planTransactionId: "skill-plan-1",
      nodeId: "open",
      invocation: proposal.strategies[0]!.nodes[1]!.invocation,
      phase: "reach_handle",
      agentId: "motion",
      currentWorldRevision: 5
    })).toMatchObject({
      accepted: false,
      code: "skill_plan_dependencies_incomplete"
    });

    const approachBinding = {
      protocol: "humanoid-active-skill-v1",
      transaction_id: "skill-1",
      agent_id: "motion",
      skill_plan_transaction_id: "skill-plan-1",
      skill_node_id: "approach",
      invocation: proposal.strategies[0]!.nodes[0]!.invocation,
      invocation_sha256: "a".repeat(64),
      phase: "route",
      phase_authority: "navigation",
      planning_action: "plan_humanoid_skill",
      observed_frame: 4,
      observed_world_revision: 5,
      skill_catalog_sha256: "b".repeat(64),
      target_position: { x: 1, y: 1, z: 1 },
      target_solid: null,
      target_articulation: null,
      eligible_interaction_points: [],
      eligible_interaction_point_ids: ["handle"],
      learned_policy_required_capabilities: ["locomotion"],
      learned_policy_missing_capabilities: [],
      control_mode: "learned_policy"
    } as const satisfies ActiveHumanoidSkillBinding;
    const advanced = advanceHumanoidSkillPlan({
      plan: registered,
      binding: approachBinding,
      worldRevision: 6,
      executionSucceeded: true
    });
    expect(advanced).toMatchObject({
      world_revision: 6,
      completed_node_ids: ["approach"]
    });
    expect(authorizeHumanoidSkillPlanNode({
      plan: advanced!,
      planTransactionId: "skill-plan-1",
      nodeId: "open",
      invocation: proposal.strategies[0]!.nodes[1]!.invocation,
      phase: "reach_handle",
      agentId: "motion",
      currentWorldRevision: 6
    })).toMatchObject({ accepted: true, node: { node_id: "open" } });

    const openNode = proposal.strategies[0]!.nodes[1]!;
    const reachHandle = {
      ...approachBinding,
      transaction_id: "skill-2",
      skill_node_id: "open",
      invocation: openNode.invocation,
      phase: "reach_handle",
      phase_authority: "whole_body",
      observed_world_revision: 6
    } as const satisfies ActiveHumanoidSkillBinding;
    const reached = advanceHumanoidSkillPlan({
      plan: advanced!,
      binding: reachHandle,
      worldRevision: 7,
      executionSucceeded: true
    });
    expect(reached).toMatchObject({
      world_revision: 7,
      completed_node_ids: ["approach"],
      completed_phases_by_node: {
        approach: ["route"],
        open: ["reach_handle"]
      }
    });

    expect(authorizeHumanoidSkillPlanNode({
      plan: reached!,
      planTransactionId: "skill-plan-1",
      nodeId: "open",
      invocation: openNode.invocation,
      phase: "actuate_joint",
      agentId: "motion",
      currentWorldRevision: 7
    })).toMatchObject({
      accepted: false,
      code: "skill_plan_phase_out_of_order",
      detail: {
        requested_phase: "actuate_joint",
        expected_phase: "establish_grasp",
        completed_phases: ["reach_handle"]
      }
    });

    const establishGrasp = {
      ...reachHandle,
      transaction_id: "skill-3",
      phase: "establish_grasp",
      phase_authority: "grasp",
      observed_world_revision: 7
    } as const satisfies ActiveHumanoidSkillBinding;
    const grasped = advanceHumanoidSkillPlan({
      plan: reached!,
      binding: establishGrasp,
      worldRevision: 8,
      executionSucceeded: true
    });
    expect(grasped).toMatchObject({
      world_revision: 8,
      completed_node_ids: ["approach"],
      completed_phases_by_node: {
        open: ["reach_handle", "establish_grasp"]
      }
    });
    expect(authorizeHumanoidSkillPlanNode({
      plan: grasped!,
      planTransactionId: "skill-plan-1",
      nodeId: "open",
      invocation: openNode.invocation,
      phase: "actuate_joint",
      agentId: "motion",
      currentWorldRevision: 8
    })).toMatchObject({ accepted: true, node: { node_id: "open" } });

    const actuateJoint = {
      ...reachHandle,
      transaction_id: "skill-4",
      phase: "actuate_joint",
      observed_world_revision: 8
    } as const satisfies ActiveHumanoidSkillBinding;
    expect(advanceHumanoidSkillPlan({
      plan: grasped!,
      binding: actuateJoint,
      worldRevision: 9,
      executionSucceeded: true
    })).toMatchObject({
      world_revision: 9,
      completed_node_ids: ["approach", "open"],
      completed_phases_by_node: {
        open: ["reach_handle", "establish_grasp", "actuate_joint"]
      }
    });
  });
});
