import {
  DOF,
  Goal,
  Joint,
  Link,
  SOLVE_STATUS,
  SOLVE_STATUS_NAMES,
  Solver
} from "closed-chain-ik/src/core/index.js";
import type { Vec3 } from "../domain/schema.js";
import {
  ROBOT_SPEC,
  jointLimitIssue,
  type ArmJointName,
  type RobotJointState
} from "./robot-model.js";

export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface EndEffectorTarget {
  position: Vec3;
  orientation?: Quaternion | undefined;
  seed?: {
    shoulder?: number | undefined;
    elbow?: number | undefined;
    wrist?: number | undefined;
  } | undefined;
}

export interface KinematicsSolution {
  joints: Pick<RobotJointState, "shoulder" | "elbow" | "wrist">;
  achievedPosition: Vec3;
  achievedOrientation: Quaternion;
  positionError: number;
  orientationError: number | null;
  status: string;
}

export interface KinematicsFailure {
  code:
    | "ik_not_converged"
    | "ik_residual_too_large"
    | "ik_solution_outside_limits"
    | "ik_solver_error";
  detail: Record<string, unknown>;
}

export const POSITION_TOLERANCE = 0.025;
export const ORIENTATION_TOLERANCE = 0.045;
const SOLVER_LIMIT_EPSILON = 1e-5;

/**
 * How many tolerances of position error still count as "the point is in reach,
 * the orientation is the problem". Errors within a few tolerances are treated
 * as a constraint conflict; larger misses require a different base position.
 */
const NEAR_MISS_FACTOR = 4;

export function solveEndEffectorTarget(input: {
  basePosition: Vec3;
  baseYaw: number;
  currentJoints: RobotJointState;
  target: EndEffectorTarget;
}): KinematicsSolution | KinematicsFailure {
  const targetOrientation = input.target.orientation
    ? normalizeQuaternion(input.target.orientation)
    : undefined;
  const root = new Link();
  root.name = "arm_root";
  const forward = { x: Math.sin(input.baseYaw), y: 0, z: Math.cos(input.baseYaw) };
  root.setPosition(
    input.basePosition.x + forward.x * ROBOT_SPEC.arm.shoulderForwardOffset,
    ROBOT_SPEC.arm.shoulderHeight,
    input.basePosition.z + forward.z * ROBOT_SPEC.arm.shoulderForwardOffset
  );
  const rootRotation = yawRotation(input.baseYaw);
  root.setQuaternion(rootRotation.x, rootRotation.y, rootRotation.z, rootRotation.w);

  const shoulder = revoluteJoint(
    "shoulder",
    -(input.target.seed?.shoulder ?? input.currentJoints.shoulder),
    -ROBOT_SPEC.joints.shoulder.maximum,
    -ROBOT_SPEC.joints.shoulder.minimum
  );
  const upper = new Link();
  upper.name = "upper_arm";
  root.addChild(shoulder);
  shoulder.addChild(upper);

  const elbow = revoluteJoint(
    "elbow",
    -(input.target.seed?.elbow ?? input.currentJoints.elbow),
    -ROBOT_SPEC.joints.elbow.maximum,
    -ROBOT_SPEC.joints.elbow.minimum
  );
  elbow.setPosition(0, 0, ROBOT_SPEC.arm.upperLength);
  const forearm = new Link();
  forearm.name = "forearm";
  upper.addChild(elbow);
  elbow.addChild(forearm);

  const wrist = revoluteJoint(
    "wrist",
    -(input.target.seed?.wrist ?? input.currentJoints.wrist),
    -ROBOT_SPEC.joints.wrist.maximum,
    -ROBOT_SPEC.joints.wrist.minimum
  );
  wrist.setPosition(0, 0, ROBOT_SPEC.arm.forearmLength);
  const tool = new Link();
  tool.name = "tool_center_point";
  tool.setPosition(0, 0, ROBOT_SPEC.arm.wristLength);
  forearm.addChild(wrist);
  wrist.addChild(tool);

  const goal = new Goal();
  goal.name = "tool_goal";
  if (targetOrientation) {
    goal.setGoalDoF(DOF.X, DOF.Y, DOF.Z, DOF.EX, DOF.EY, DOF.EZ);
    goal.setQuaternion(
      targetOrientation.x,
      targetOrientation.y,
      targetOrientation.z,
      targetOrientation.w
    );
  } else {
    goal.setGoalDoF(DOF.X, DOF.Y, DOF.Z);
  }
  goal.setPosition(input.target.position.x, input.target.position.y, input.target.position.z);
  goal.makeClosure(tool);

  const solver = new Solver(root);
  solver.maxIterations = 120;
  solver.translationConvergeThreshold = POSITION_TOLERANCE;
  solver.rotationConvergeThreshold = ORIENTATION_TOLERANCE;
  solver.translationErrorClamp = 0.08;
  solver.rotationErrorClamp = 0.08;
  let statuses: SOLVE_STATUS[];
  try {
    statuses = solver.solve() as SOLVE_STATUS[];
  } catch (error) {
    return {
      code: "ik_solver_error",
      detail: { error: error instanceof Error ? error.message : String(error) }
    };
  }
  const statusNames = statuses.map((status: SOLVE_STATUS) =>
    String(SOLVE_STATUS_NAMES[status] ?? `UNKNOWN_${status}`)
  );
  if (statuses.length === 0) {
    return { code: "ik_not_converged", detail: { statuses: statusNames } };
  }

  tool.updateMatrixWorld();
  const achievedPositionArray = [0, 0, 0];
  const achievedOrientationArray = [0, 0, 0, 1];
  tool.getWorldPosition(achievedPositionArray);
  tool.getWorldQuaternion(achievedOrientationArray);
  const achievedPosition = {
    x: achievedPositionArray[0]!,
    y: achievedPositionArray[1]!,
    z: achievedPositionArray[2]!
  };
  const achievedOrientation = {
    x: achievedOrientationArray[0]!,
    y: achievedOrientationArray[1]!,
    z: achievedOrientationArray[2]!,
    w: achievedOrientationArray[3]!
  };
  const joints = {
    shoulder: normalizeSolverLimit("shoulder", -Number(shoulder.getDoFValue(DOF.EX))),
    elbow: normalizeSolverLimit("elbow", -Number(elbow.getDoFValue(DOF.EX))),
    wrist: normalizeSolverLimit("wrist", -Number(wrist.getDoFValue(DOF.EX)))
  };
  const limitIssue = jointLimitIssue(joints);
  if (limitIssue) {
    return { code: "ik_solution_outside_limits", detail: limitIssue };
  }

  const positionError = distance(achievedPosition, input.target.position);
  const orientationError = targetOrientation
    ? quaternionDistance(achievedOrientation, targetOrientation)
    : null;
  if (!Number.isFinite(positionError)
    || (orientationError !== null && !Number.isFinite(orientationError))
    || positionError > POSITION_TOLERANCE
    || (orientationError !== null && orientationError > ORIENTATION_TOLERANCE)) {
    return {
      code: "ik_residual_too_large",
      detail: {
        statuses: statusNames,
        achieved_position: achievedPosition,
        achieved_orientation: achievedOrientation,
        position_error: positionError,
        orientation_error: orientationError,
        position_tolerance: POSITION_TOLERANCE,
        orientation_tolerance: ORIENTATION_TOLERANCE,
        ...residualRecovery(positionError, orientationError, targetOrientation !== undefined)
      }
    };
  }

  return {
    joints,
    achievedPosition,
    achievedOrientation,
    positionError,
    orientationError,
    status: statusNames.join(",")
  };
}

/**
 * Says which of the two residuals actually failed, and what to change.
 *
 * Position and orientation are evaluated independently. Naming the residual
 * outside tolerance keeps recovery focused on the constraint that can affect
 * the next solve.
 */
function residualRecovery(
  positionError: number,
  orientationError: number | null,
  orientationWasRequested: boolean
): Record<string, unknown> {
  const positionFailed = !Number.isFinite(positionError) || positionError > POSITION_TOLERANCE;
  const orientationFailed = orientationError !== null
    && (!Number.isFinite(orientationError) || orientationError > ORIENTATION_TOLERANCE);
  if (positionFailed && orientationFailed) {
    // Both terms are solved together, so an orientation the arm cannot hold
    // drags the position off its target as well and both residuals fail at
    // once. A small position miss distinguishes an over-constrained pose from
    // a target that is genuinely unreachable from the current base position.
    const nearlyThere = Number.isFinite(positionError)
      && positionError < POSITION_TOLERANCE * NEAR_MISS_FACTOR;
    return {
      failing_residual: "both",
      recovery: orientationWasRequested
        ? "This request constrained orientation and both residuals failed. If the task only needs "
          + "the point, call solve_end_effector_position with the exact same position; that planner "
          + "cannot add a quaternion and lets the wrist rotate freely. If the rotation is genuinely "
          + "required, move the base to a different observed standoff before retrying the pose."
        : nearlyThere
          ? "The point is close but the position solver did not converge. Re-observe the target and "
            + "retry from a different current-world standoff rather than adding an orientation."
        : "Neither the position nor the orientation converged. The arm cannot hold this "
          + "pose from where the base is standing — drive the base closer to the target with "
          + "plan_base_path and execute_base_plan, then solve again."
    };
  }
  if (orientationFailed) {
    return {
      failing_residual: "orientation",
      recovery: "The position converged; only the wrist orientation is out of tolerance. Changing "
        + "the target position will not help. If rotation is not a mission requirement, call "
        + "solve_end_effector_position for the same point. Otherwise request a pose aligned with "
        + "the arm's reaching direction from a different observed standoff."
    };
  }
  return {
    failing_residual: "position",
    recovery: "The requested point is not attainable from the current base pose; the orientation "
      + "is not the problem. Either the point is wrong or the base is too far away. Confirm the "
      + "point first: inspect_entity reports a grasp_pose for a portable object, which is the "
      + "position to pass here. A run guessed one 1.5m from the object it meant to pick up. If "
      + "the point is right, drive the base toward it with plan_base_path and execute_base_plan, "
      + "then solve again — and if this agent was granted neither of those, report_blocked so "
      + "the parent can delegate a node that has them."
  };
}

function normalizeSolverLimit(joint: ArmJointName, value: number): number {
  const limit = ROBOT_SPEC.joints[joint];
  if (value < limit.minimum && value >= limit.minimum - SOLVER_LIMIT_EPSILON) {
    return limit.minimum;
  }
  if (value > limit.maximum && value <= limit.maximum + SOLVER_LIMIT_EPSILON) {
    return limit.maximum;
  }
  return value;
}

function revoluteJoint(name: string, value: number, minimum: number, maximum: number): Joint {
  const joint = new Joint();
  joint.name = name;
  joint.setDoF(DOF.EX);
  joint.setMinLimits(minimum);
  joint.setMaxLimits(maximum);
  joint.setDoFValues(value);
  joint.setRestPoseValues(value);
  joint.restPoseSet = true;
  return joint;
}

function yawRotation(yaw: number): Quaternion {
  return { x: 0, y: Math.sin(yaw / 2), z: 0, w: Math.cos(yaw / 2) };
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

export function quaternionDistance(left: Quaternion, right: Quaternion): number {
  const dot = Math.min(1, Math.abs(
    left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w
  ));
  return 2 * Math.acos(dot);
}

export function normalizeQuaternion(value: Quaternion): Quaternion {
  const length = Math.hypot(value.x, value.y, value.z, value.w);
  return {
    x: value.x / length,
    y: value.y / length,
    z: value.z / length,
    w: value.w / length
  };
}
