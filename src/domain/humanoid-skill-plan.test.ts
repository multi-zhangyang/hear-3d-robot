import { describe, expect, it } from "vitest";
import { HumanoidSkillPlanProposalSchema } from "./humanoid-skill-plan.js";

const plan = {
  objective: "open, move, and close",
  strategies: [{
    strategy_id: "right-hand",
    rationale: "the right handle point is currently visible",
    nodes: [{
      node_id: "approach-door",
      invocation: {
        skill: "approach",
        object_id: "door",
        interaction_point_id: "handle",
        standoff_m: 0.6
      },
      depends_on_node_ids: []
    }, {
      node_id: "open-door",
      invocation: {
        skill: "open",
        object_id: "door",
        interaction_point_id: "handle",
        joint_id: "hinge",
        hand: "right",
        minimum_open_fraction: 0.85
      },
      depends_on_node_ids: ["approach-door"]
    }]
  }, {
    strategy_id: "left-hand",
    rationale: "an alternative hand can use the same physical handle",
    nodes: [{
      node_id: "open-with-left",
      invocation: {
        skill: "open",
        object_id: "door",
        interaction_point_id: "handle",
        joint_id: "hinge",
        hand: "left",
        minimum_open_fraction: 0.85
      },
      depends_on_node_ids: []
    }]
  }],
  selected_strategy_id: "right-hand"
} as const;

describe("humanoid Skill DAG schema", () => {
  it("accepts multiple model-authored strategies with an explicit selection", () => {
    expect(HumanoidSkillPlanProposalSchema.parse(plan)).toEqual(plan);
  });

  it("rejects missing dependencies and cycles", () => {
    const missing = structuredClone(plan);
    missing.strategies[0]!.nodes[1]!.depends_on_node_ids = ["unknown"];
    expect(HumanoidSkillPlanProposalSchema.safeParse(missing).success).toBe(false);

    const cyclic = structuredClone(plan);
    cyclic.strategies[0]!.nodes[0]!.depends_on_node_ids = ["open-door"];
    expect(HumanoidSkillPlanProposalSchema.safeParse(cyclic).success).toBe(false);
  });
});
