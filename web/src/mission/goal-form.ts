import type { Goal, GoalPredicate, Vec3 } from "../types";

export const predicateOptions = [
  { value: "robot_at", label: "机器人到达坐标" },
  { value: "robot_in_zone", label: "机器人进入区域" },
  { value: "object_in_zone", label: "物体位于区域" },
  { value: "object_at", label: "物体到达坐标" },
  { value: "end_effector_at", label: "末端到达位置" }
] satisfies Array<{
  value: GoalPredicate["type"];
  label: string;
}>;

export function emptyPredicate(type: GoalPredicate["type"]): GoalPredicate {
  if (type === "robot_at") {
    return { type, target: { x: 0, y: 0, z: 0 }, tolerance: 0.25 };
  }
  if (type === "robot_in_zone") return { type, zone_id: "", tolerance: 0.2 };
  if (type === "object_in_zone") {
    return { type, object_id: "", zone_id: "", expected: true, tolerance: 0.05 };
  }
  if (type === "object_at") {
    return { type, object_id: "", target: { x: 0, y: 0, z: 0 }, tolerance: 0.1 };
  }
  return {
    type,
    end_effector: "left_wrist",
    frame: "pelvis",
    target: { x: 0, y: 0, z: 0 },
    tolerance: 0.05,
    stable_frames: 5
  };
}

export function validGoal(goal: Goal): boolean {
  return goal.summary.trim().length > 0
    && goal.predicates.length > 0
    && goal.predicates.every(validPredicate);
}

export function inputNumber(element: HTMLInputElement): number {
  return element.value === "" ? Number.NaN : element.valueAsNumber;
}

function validPredicate(predicate: GoalPredicate): boolean {
  if (predicate.type === "robot_at") {
    return finiteVec3(predicate.target) && predicate.tolerance > 0;
  }
  if (predicate.type === "robot_in_zone") {
    return predicate.zone_id.length > 0 && predicate.tolerance >= 0;
  }
  if (predicate.type === "object_at") {
    return predicate.object_id.length > 0
      && finiteVec3(predicate.target)
      && predicate.tolerance > 0;
  }
  if (predicate.type === "object_in_zone") {
    return predicate.object_id.length > 0
      && predicate.zone_id.length > 0
      && predicate.tolerance >= 0;
  }
  if (predicate.type === "end_effector_at") {
    return finiteVec3(predicate.target)
      && predicate.tolerance > 0
      && predicate.tolerance <= 5
      && Number.isInteger(predicate.stable_frames)
      && predicate.stable_frames >= 1
      && predicate.stable_frames <= 500;
  }
  return predicate satisfies never;
}

function finiteVec3(value: Vec3): boolean {
  return Number.isFinite(value.x) && Number.isFinite(value.y) && Number.isFinite(value.z);
}
