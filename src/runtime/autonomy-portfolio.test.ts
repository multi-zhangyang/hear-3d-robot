import { describe, expect, it } from "vitest";
import {
  advancePhysicalTrajectorySha256,
  physicalTrajectoryFrameSha256,
  type PhysicalTrajectoryFrame,
  type PhysicalTrajectorySummary
} from "../domain/physical-trajectory.js";
import {
  assertAutonomyPortfolio,
  evaluateAutonomyPortfolio
} from "./autonomy-portfolio.js";
import { autonomyContentSha256 } from "./autonomy-signature.js";

describe("autonomy portfolio", () => {
  it("requires distinct model decisions, plans and physical traces from one initial state", () => {
    const runs = [
      report("run-1", "a", "d", 0),
      report("run-2", "b", "e", 0.2),
      report("run-3", "c", "f", 0.4)
    ];
    expect(assertAutonomyPortfolio(runs)).toMatchObject({
      run_count: 3,
      initial_state_count: 1,
      same_initial_state: true,
      distinct_goal_signatures: 1,
      distinct_planning_signatures: 3,
      distinct_model_decision_signatures: 3,
      distinct_physical_behavior_signatures: 3,
      materially_distinct_physical_behaviors: 3,
      autonomous_diversity_observed: true
    });
  });

  it("does not confuse different seeds with same-state autonomous diversity", () => {
    const runs = [
      report("run-1", "a", "d", 0, 7),
      report("run-2", "b", "e", 0.2, 8),
      report("run-3", "c", "f", 0.4, 9)
    ];
    expect(evaluateAutonomyPortfolio(runs).same_initial_state).toBe(false);
    expect(() => assertAutonomyPortfolio(runs)).toThrow(
      "same scenario and world seed"
    );
    expect(() => assertAutonomyPortfolio(runs, {
      requireSameInitialState: false
    })).not.toThrow();
  });

  it("rejects textual variation that produces the same plan or physical behavior", () => {
    const runs = [
      report("run-1", "a", "d", 0),
      report("run-2", "b", "d", 0),
      report("run-3", "c", "d", 0)
    ];
    expect(() => assertAutonomyPortfolio(runs)).toThrow(
      "no planning diversity"
    );
  });

  it("rejects different hashes when the actual body trajectories differ only numerically", () => {
    const runs = [
      report("run-1", "a", "d", 0),
      report("run-2", "b", "e", 0.001),
      report("run-3", "c", "f", 0.002)
    ];

    expect(evaluateAutonomyPortfolio(runs)).toMatchObject({
      distinct_physical_behavior_signatures: 3,
      materially_distinct_physical_behaviors: 1,
      autonomous_diversity_observed: false
    });
    expect(() => assertAutonomyPortfolio(runs)).toThrow(
      "no materially different physical behavior"
    );
  });
});

function report(
  runId: string,
  modelMarker: string,
  planningMarker: string,
  behaviorOffset: number,
  seed = 47
) {
  const hash = (marker: string) => marker.repeat(64).slice(0, 64);
  const trajectory = physicalTrajectory(behaviorOffset);
  const worldMutationHashes: string[] = [];
  return {
    version: 3,
    run_id: runId,
    scenario_id: "humanoid_frontier",
    seed,
    status: "succeeded",
    physical_verified: true,
    world_frame: 20,
    world_revision: 20,
    cycle_count: 1,
    model_call_count: 5,
    physical_execution_count: 1,
    physical_frame_count: 20,
    travelled_distance_m: 1,
    selected_goal_hashes: [hash("9")],
    planning_behavior_hashes: [hash(planningMarker)],
    model_response_hashes: [hash(modelMarker)],
    action_sequence: ["observe_humanoid", "plan_humanoid_navigation"],
    goal_signature: hash("9"),
    planning_signature: hash(planningMarker),
    model_decision_signature: hash(modelMarker),
    physical_behavior_signature: autonomyContentSha256({
      trajectories: [trajectory],
      world_mutation_hashes: worldMutationHashes
    }),
    physical_trajectories: [trajectory],
    world_mutation_hashes: worldMutationHashes
  };
}

function physicalTrajectory(offset: number): PhysicalTrajectorySummary {
  const jointNames = Array.from({ length: 43 }, (_, index) => (
    `joint-${String(index).padStart(2, "0")}`
  ));
  let trajectorySha256: string | null = null;
  const samples = Array.from({ length: 21 }, (_, frame) => {
    const ratio = frame / 20;
    const rootX = (1 + offset) * ratio;
    const identity = {
      frame,
      world_revision: frame,
      root_position: { x: rootX, y: 0.8, z: 0 },
      root_rotation: { x: 0, y: 0, z: 0, w: 1 },
      joint_positions: jointNames.map(() => offset * ratio),
      end_effectors: {
        left_wrist: { x: rootX + 0.25, y: 1.1, z: 0 },
        right_wrist: { x: rootX - 0.25, y: 1.1, z: 0 },
        left_ankle: { x: rootX + 0.1, y: 0.1, z: 0 },
        right_ankle: { x: rootX - 0.1, y: 0.1, z: 0 }
      },
      contacts: [],
      objects: [],
      support: "double" as const,
      fallen: false
    };
    const sample: PhysicalTrajectoryFrame = {
      ...identity,
      frame_sha256: physicalTrajectoryFrameSha256(identity)
    };
    trajectorySha256 = advancePhysicalTrajectorySha256(
      trajectorySha256,
      sample.frame_sha256
    );
    return sample;
  });
  return {
    version: 1,
    complete_from_admission: true,
    start_frame: 0,
    end_frame: 20,
    start_world_revision: 0,
    end_world_revision: 20,
    observed_frame_count: 21,
    sample_stride: 1,
    joint_names: jointNames,
    samples,
    root_path_length_m: 1 + offset,
    root_planar_path_length_m: 1 + offset,
    joint_total_variation_rad: Math.abs(offset) * 43,
    end_effector_path_length_m: {
      left_wrist: 1 + offset,
      right_wrist: 1 + offset,
      left_ankle: 1 + offset,
      right_ankle: 1 + offset
    },
    object_path_length_m: {},
    contact_transition_count: 0,
    trajectory_sha256: trajectorySha256!
  };
}
