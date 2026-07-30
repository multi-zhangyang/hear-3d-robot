import { describe, expect, it } from "vitest";
import type { AgentSpec, Goal, VoxelMaterial } from "../domain/schema.js";
import { assertGoalCapabilityContract } from "./goal-capability.js";

const goal: Goal = {
  summary: "Place one recovered grass voxel at the target.",
  predicates: [{
    type: "voxel_at",
    coordinate: { column: 70, level: 1, row: 25 },
    material: "grass"
  }]
};

function spec(
  capabilities: string[],
  goalPredicateIndexes = [0],
  mayDelegate = false
): AgentSpec {
  return {
    name: "Voxel branch",
    objective: "Advance the voxel goal from current evidence.",
    success_criteria: ["The delegated outcome is receipt-backed."],
    goal_predicate_indexes: goalPredicateIndexes,
    capabilities,
    may_delegate: mayDelegate,
    references: []
  };
}

function validate(agent: AgentSpec, current: VoxelMaterial | null): void {
  assertGoalCapabilityContract(goal, agent, {
    predicatePassed: (index) => {
      const predicate = goal.predicates[index];
      return predicate?.type === "voxel_at" && predicate.material === current;
    },
    voxelMaterialAt: () => current
  });
}

describe("goal capability contract", () => {
  it("requires place authority when an owned voxel target is currently empty", () => {
    expect(() => validate(spec(["inspect_voxel"]), null)).toThrow(
      "missing capabilities: place_voxel"
    );
    expect(() => validate(spec(["inspect_voxel", "place_voxel"]), null)).not.toThrow();
  });

  it("keeps observation-only leaves as intermediate evidence", () => {
    expect(() => validate(spec(["inspect_voxel"], []), null)).not.toThrow();
  });

  it("requires both mutations when replacing a different material", () => {
    expect(() => validate(spec(["place_voxel"]), "stone")).toThrow(
      "missing capabilities: break_voxel"
    );
    expect(() => validate(spec(["break_voxel", "place_voxel"]), "stone")).not.toThrow();
  });

  it("does not demand an actuator for an already satisfied predicate", () => {
    expect(() => validate(spec(["inspect_voxel"]), "grass")).not.toThrow();
  });

  it("rejects a non-delegating voxel owner that can plan but cannot execute", () => {
    expect(() => validate(spec(["place_voxel", "plan_base_path"]), null)).toThrow(
      "plan_base_path requires execute_base_plan"
    );
    expect(() => validate(spec(["place_voxel", "plan_joint_targets"]), null)).toThrow(
      "plan_joint_targets requires execute_joint_plan"
    );
  });

  it("accepts a non-delegating voxel owner with matched planning and execution", () => {
    expect(() => validate(spec([
      "place_voxel",
      "plan_base_path",
      "execute_base_plan",
      "solve_end_effector_position",
      "execute_joint_plan"
    ]), null)).not.toThrow();
  });

  it("allows a planning-only leaf when it owns no final-state predicate", () => {
    expect(() => validate(spec(["plan_base_path"], []), null)).not.toThrow();
  });

  it("requires a delegating supervisor to retain executors for its descendants", () => {
    expect(() => validate(spec([
      "place_voxel",
      "plan_base_path",
      "solve_end_effector_position"
    ], [0], true), null)).toThrow(
      "solve_end_effector_position requires execute_joint_plan"
    );
    expect(() => validate(spec([
      "place_voxel",
      "plan_base_path",
      "execute_base_plan",
      "solve_end_effector_position",
      "execute_joint_plan"
    ], [0], true), null)).not.toThrow();
  });

  it("does not let an intermediate mutation leaf strand its own arm plan", () => {
    expect(() => validate(spec([
      "inspect_voxel",
      "solve_end_effector_position",
      "break_voxel"
    ], []), "grass")).toThrow(
      "solve_end_effector_position requires execute_joint_plan"
    );
  });

  it("rejects an unmet locomotion owner that can only plan", () => {
    const locomotionGoal: Goal = {
      summary: "Reach the destination.",
      predicates: [{
        type: "robot_at",
        target: { x: 4, y: 0, z: 4 },
        tolerance: 0.35
      }]
    };
    expect(() => assertGoalCapabilityContract(locomotionGoal, spec(["plan_base_path"]), {
      predicatePassed: () => false,
      voxelMaterialAt: () => null
    })).toThrow("has no actuator that can change that state");
  });

  it("accepts locomotion ownership with either base executor", () => {
    const locomotionGoal: Goal = {
      summary: "Reach the destination.",
      predicates: [{
        type: "robot_at",
        target: { x: 4, y: 0, z: 4 },
        tolerance: 0.35
      }]
    };
    for (const capability of ["execute_base_plan", "drive_base"]) {
      expect(() => assertGoalCapabilityContract(
        locomotionGoal,
        spec(["plan_base_path", capability]),
        { predicatePassed: () => false, voxelMaterialAt: () => null }
      )).not.toThrow();
    }
  });

  it("does not demand an actuator for an already-satisfied non-voxel predicate", () => {
    const locomotionGoal: Goal = {
      summary: "Stay at the destination.",
      predicates: [{
        type: "robot_at",
        target: { x: 1, y: 0, z: 1 },
        tolerance: 0.35
      }]
    };
    expect(() => assertGoalCapabilityContract(locomotionGoal, spec(["sense_scene"]), {
      predicatePassed: () => true,
      voxelMaterialAt: () => null
    })).not.toThrow();
  });
});
