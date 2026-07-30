import type { JsonValue, WorldSnapshot } from "../domain/schema.js";
import type { EndEffectorTarget } from "./kinematics.js";
import type { ArmPose } from "./arm-trajectory.js";

export type StoredArmPlan = EndEffectorArmPlan | JointTargetArmPlan;

interface ArmPlanBase {
  id: string;
  createdRevision: number;
  joints: ArmPose;
  waypoints: ArmPose[];
}

export interface EndEffectorArmPlan extends ArmPlanBase {
  kind: "end_effector";
  target: EndEffectorTarget;
}

export interface JointTargetArmPlan extends ArmPlanBase {
  kind: "joint_targets";
  target: null;
}

type ArmPlanSnapshot = WorldSnapshot["plans"]["arm"][number];

export function snapshotArmPlan(plan: StoredArmPlan): ArmPlanSnapshot {
  return {
    id: plan.id,
    created_revision: plan.createdRevision,
    kind: plan.kind,
    target: plan.target ? structuredClone(plan.target) : null,
    joints: { ...plan.joints },
    waypoints: structuredClone(plan.waypoints)
  };
}

export function restoreArmPlan(
  snapshot: ArmPlanSnapshot,
  currentRevision: number
): StoredArmPlan {
  if (snapshot.created_revision !== currentRevision) {
    throw new Error(`Checkpoint contains a stale arm plan: ${snapshot.id}`);
  }
  const base = {
    id: snapshot.id,
    createdRevision: snapshot.created_revision,
    joints: { ...snapshot.joints },
    waypoints: snapshot.waypoints.length > 0
      ? structuredClone(snapshot.waypoints)
      : [{ ...snapshot.joints }]
  };
  if (snapshot.kind === "joint_targets") {
    if (snapshot.target !== null) {
      throw new Error(`Joint-target plan contains an end-effector target: ${snapshot.id}`);
    }
    return { ...base, kind: "joint_targets", target: null };
  }
  if (snapshot.target === null) {
    throw new Error(`End-effector plan is missing its target: ${snapshot.id}`);
  }
  return {
    ...base,
    kind: "end_effector",
    target: structuredClone(snapshot.target)
  };
}

export function armPlanFocus(plan: StoredArmPlan): JsonValue {
  return plan.kind === "end_effector"
    ? { kind: "end_effector_target", position: plan.target.position }
    : { kind: "joint_targets", joints: { ...plan.joints } };
}

export function endEffectorVerificationTarget(
  plan: StoredArmPlan
): EndEffectorTarget | undefined {
  return plan.kind === "end_effector" ? plan.target : undefined;
}
