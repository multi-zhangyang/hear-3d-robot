import type {
  AgentSpec,
  Goal,
  VoxelCoordinate,
  VoxelMaterial
} from "../domain/schema.js";

/**
 * Refuse final-state ownership that the delegated branch cannot physically
 * discharge. This is an authority check, not a planner: it never chooses a
 * coordinate, route, pose, or action order.
 */
export function assertGoalCapabilityContract(
  goal: Goal,
  spec: AgentSpec,
  state: {
    predicatePassed: (index: number) => boolean;
    voxelMaterialAt: (coordinate: VoxelCoordinate) => VoxelMaterial | null;
  }
): void {
  let ownsUnmetVoxelPredicate = false;
  for (const predicateIndex of spec.goal_predicate_indexes) {
    const predicate = goal.predicates[predicateIndex];
    if (!predicate || state.predicatePassed(predicateIndex)) continue;

    if (predicate.type !== "voxel_at") {
      const alternatives = predicateActuatorAlternatives(predicate.type);
      if (alternatives.some((capability) => spec.capabilities.includes(capability))) continue;
      throw new Error(
        `Agent ${spec.name} owns unmet ${predicate.type} goal predicate ${predicateIndex}, but its `
        + `capability grant has no actuator that can change that state; grant one of: `
        + `${alternatives.join(", ")}. A bounded observation or planning leaf must use `
        + "goal_predicate_indexes: []."
      );
    }

    const currentMaterial = state.voxelMaterialAt(predicate.coordinate);
    ownsUnmetVoxelPredicate = true;
    const required = voxelTransitionCapabilities(currentMaterial, predicate.material);
    const missing = required.filter((capability) => !spec.capabilities.includes(capability));
    if (missing.length === 0) continue;

    throw new Error(
      `Agent ${spec.name} owns unmet voxel_at goal predicate ${predicateIndex}, but its `
      + `capability grant cannot change ${JSON.stringify(currentMaterial)} to `
      + `${JSON.stringify(predicate.material)}; missing capabilities: ${missing.join(", ")}. `
      + "A bounded observation or planning leaf must use goal_predicate_indexes: []."
    );
  }

  // A planning-only evidence leaf may intentionally hand a plan to another
  // node, but it must not also carry voxel mutation authority. Once a branch
  // can break or place voxels, every planner in its capability budget needs
  // its executor as well. For supervisors this is capability closure: they do
  // not invoke the tools themselves, but descendants cannot inherit an
  // executor the supervisor was never granted.
  const canMutateVoxels = spec.capabilities.includes("break_voxel")
    || spec.capabilities.includes("place_voxel");
  if (!ownsUnmetVoxelPredicate && !canMutateVoxels) return;
  const missingExecutors = PLANNING_EXECUTION_PAIRS.flatMap(([planner, executor]) =>
    spec.capabilities.includes(planner) && !spec.capabilities.includes(executor)
      ? [`${planner} requires ${executor}`]
      : []
  );
  if (missingExecutors.length === 0) return;
  throw new Error(
    `Agent ${spec.name} has voxel mutation authority or owns an unmet voxel_at predicate, `
    + `but its planning capability budget is not physically closed: `
    + `${missingExecutors.join(", ")}. Add the matching execution capabilities. A pure `
    + "planning leaf may instead use goal_predicate_indexes: [] with no break_voxel or "
    + "place_voxel authority and pass its accepted receipt to a separate executor. A "
    + "supervisor must retain both capabilities so narrower descendants can inherit them."
  );
}

/**
 * Minimal actuator authority for each non-voxel final-state predicate.
 *
 * These are alternatives rather than a scripted recipe. The model still
 * chooses the route, pose and action sequence; the harness only rejects a
 * branch that could never change the state it claims to own. Object predicates
 * intentionally admit several physical strategies (carry, push, arm contact,
 * or gripper interaction), while locomotion predicates require a base actuator.
 */
function predicateActuatorAlternatives(type: Goal["predicates"][number]["type"]): string[] {
  if (type === "robot_at" || type === "robot_in_zone" || type === "terrain_explored") {
    return ["execute_base_plan", "drive_base"];
  }
  if (type === "object_attached") return ["set_gripper_target"];
  return [
    "execute_base_plan",
    "drive_base",
    "execute_joint_plan",
    "set_joint_targets",
    "set_gripper_target"
  ];
}

const PLANNING_EXECUTION_PAIRS = [
  ["plan_base_path", "execute_base_plan"],
  ["plan_joint_targets", "execute_joint_plan"],
  ["solve_end_effector_position", "execute_joint_plan"],
  ["solve_end_effector_pose", "execute_joint_plan"],
  ["plan_arm_retraction", "set_joint_targets"]
] as const;

function voxelTransitionCapabilities(
  current: VoxelMaterial | null,
  expected: VoxelMaterial | null
): string[] {
  if (expected === null) return ["break_voxel"];
  if (current === null) return ["place_voxel"];
  return ["break_voxel", "place_voxel"];
}
