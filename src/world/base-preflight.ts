/**
 * Base preflight: deciding whether a navigation path can actually be driven
 * before any of it is driven.
 *
 * The navmesh answers "is there a corridor", which is a question about the
 * footprint disc. It cannot answer "does the whole rig fit while turning and
 * translating along that corridor" — the arm, head, and any carried payload
 * sweep volumes the disc never models. So the path is walked here in small
 * pose steps and each pose is checked against the real rig.
 *
 * The check itself is injected as `collisionsAt`, so this module is pure
 * geometry over poses: it decides *where to look*, and the world decides
 * *what is there*. That is also what makes it testable without a physics world.
 */
import type { JsonValue, Vec3 } from "../domain/schema.js";
import { collisionSetJson, collisionTransitionAllowed, type CollisionIssue } from "./collision.js";
import { normalizeAngle, planarDistance } from "./geometry.js";
import { ROBOT_SPEC } from "./robot-model.js";

/** Whether a segment is driven nose-first (1) or in reverse (-1). */
type BaseLinearSign = -1 | 1;

export interface ResolvedBaseSegment {
  waypointIndex: number;
  start: Vec3;
  target: Vec3;
  bodyYaw: number;
  linearSign: BaseLinearSign;
  distance: number;
}

interface BaseTrajectoryState {
  position: Vec3;
  yaw: number;
  collisions: CollisionIssue[];
}

type WaypointResolution =
  | { ok: true; segment: ResolvedBaseSegment; state: BaseTrajectoryState }
  | { ok: false; failures: JsonValue[] };

type BaseTrajectoryResult =
  | { ok: true; state: BaseTrajectoryState }
  | { ok: false; issue: JsonValue };

export type BasePlanResolution =
  | { ok: true; segments: ResolvedBaseSegment[] }
  | { ok: false; issue: JsonValue };

/** Probes the rig's collisions at a hypothetical base pose without moving it. */
type BaseCollisionProbe = (position: Vec3, yaw: number) => CollisionIssue[];

/**
 * Pose granularity for the sweep. Fine enough that the rig cannot tunnel
 * through a thin obstacle between samples, coarse enough that a long path
 * stays cheap to check.
 */
const LINEAR_STEP = 0.04;
const ANGULAR_STEP = 0.04;
const MAX_INITIAL_EGRESS_DISTANCE = Math.max(1.2, ROBOT_SPEC.base.footprintRadius * 2);

export interface BasePreflightInput {
  waypoints: Vec3[];
  face?: Vec3 | undefined;
  start: Vec3;
  yaw: number;
  collisionsAt: BaseCollisionProbe;
}

/**
 * Walks the navmesh waypoints as drivable segments, checking every intermediate
 * pose. Returns the segments the executor should follow, or the first pose that
 * cannot be reached and why.
 */
export function resolveBasePlan(input: BasePreflightInput): BasePlanResolution {
  const { collisionsAt } = input;
  let state: BaseTrajectoryState = {
    position: input.start,
    yaw: input.yaw,
    collisions: collisionsAt(input.start, input.yaw)
  };
  const segments: ResolvedBaseSegment[] = [];

  // A previously blocked dynamic execution can leave the articulated rig in
  // shallow contact even though its base remains on the navmesh. Before asking
  // it to rotate toward a new route, find a short physically validated escape
  // along its existing heading. This is part of path planning: no pose is
  // applied here, and the model must still execute the accepted plan.
  if (state.collisions.length > 0) {
    const egress = resolveInitialEgress(state, collisionsAt);
    if (egress) {
      segments.push(egress.segment);
      state = egress.state;
    }
  }

  for (let index = 1; index < input.waypoints.length; index += 1) {
    const target = input.waypoints[index]!;
    if (planarDistance(state.position, target) <= 1e-6) continue;
    let selected = resolveWaypoint(state, target, index, collisionsAt);
    if (!selected.ok && index === 1) {
      const clearance = resolveClearanceThenWaypoint(state, target, index, collisionsAt);
      if (clearance) {
        segments.push(clearance.clearance, clearance.waypoint.segment);
        state = clearance.waypoint.state;
        continue;
      }
    }
    if (!selected.ok) {
      return {
        ok: false,
        issue: {
          phase: "resolve_segment",
          waypoint_index: index,
          start: state.position,
          target,
          candidates: selected.failures
        }
      };
    }
    segments.push(selected.segment);
    state = selected.state;
  }

  // Facing is part of the plan's promise, so a face point the rig cannot turn
  // to is a planning failure rather than something the executor discovers.
  if (input.face) {
    const faceYaw = Math.atan2(
      input.face.x - state.position.x,
      input.face.z - state.position.z
    );
    const faced = sweepRotation(
      state,
      faceYaw,
      "face_point",
      input.waypoints.length - 1,
      collisionsAt
    );
    if (!faced.ok) return { ok: false, issue: faced.issue };
  }
  return { ok: true, segments };
}

function resolveWaypoint(
  state: BaseTrajectoryState,
  target: Vec3,
  waypointIndex: number,
  collisionsAt: BaseCollisionProbe
): WaypointResolution {
  const distance = planarDistance(state.position, target);
  const pathYaw = Math.atan2(target.x - state.position.x, target.z - state.position.z);
  // Driving in reverse is legal and sometimes the only way through, but it is
  // the second choice: prefer whichever heading needs the smaller turn, and
  // break exact ties nose-first.
  const candidates = ([
    { bodyYaw: normalizeAngle(pathYaw), linearSign: 1 as const },
    { bodyYaw: normalizeAngle(pathYaw + Math.PI), linearSign: -1 as const }
  ]).sort((left, right) => {
    const turnDifference = Math.abs(normalizeAngle(left.bodyYaw - state.yaw))
      - Math.abs(normalizeAngle(right.bodyYaw - state.yaw));
    return Math.abs(turnDifference) > 1e-9
      ? turnDifference
      : right.linearSign - left.linearSign;
  });
  const failures: JsonValue[] = [];
  for (const candidate of candidates) {
    const segment: ResolvedBaseSegment = {
      waypointIndex,
      start: { ...state.position },
      target: { x: target.x, y: ROBOT_SPEC.base.centerY, z: target.z },
      bodyYaw: candidate.bodyYaw,
      linearSign: candidate.linearSign,
      distance
    };
    const result = sweepSegment(state, segment, collisionsAt);
    if (result.ok) return { ok: true, segment, state: result.state };
    failures.push({
      body_yaw: candidate.bodyYaw,
      linear_direction: candidate.linearSign === 1 ? "forward" : "reverse",
      required_rotation: Math.abs(normalizeAngle(candidate.bodyYaw - state.yaw)),
      issue: result.issue
    });
  }
  return { ok: false, failures };
}

function resolveClearanceThenWaypoint(
  initial: BaseTrajectoryState,
  target: Vec3,
  waypointIndex: number,
  collisionsAt: BaseCollisionProbe
): {
  clearance: ResolvedBaseSegment;
  waypoint: Extract<WaypointResolution, { ok: true }>;
} | undefined {
  const increments = Math.ceil(MAX_INITIAL_EGRESS_DISTANCE / LINEAR_STEP);
  for (let step = 1; step <= increments; step += 1) {
    const distance = step * LINEAR_STEP;
    for (const linearSign of [1, -1] as const) {
      const clearance: ResolvedBaseSegment = {
        waypointIndex: 0,
        start: { ...initial.position },
        target: {
          x: initial.position.x + Math.sin(initial.yaw) * distance * linearSign,
          y: ROBOT_SPEC.base.centerY,
          z: initial.position.z + Math.cos(initial.yaw) * distance * linearSign
        },
        bodyYaw: initial.yaw,
        linearSign,
        distance
      };
      const cleared = sweepSegment(initial, clearance, collisionsAt);
      if (!cleared.ok || cleared.state.collisions.length > 0) continue;
      const waypoint = resolveWaypoint(cleared.state, target, waypointIndex, collisionsAt);
      if (waypoint.ok) return { clearance, waypoint };
    }
  }
  return undefined;
}

function resolveInitialEgress(
  initial: BaseTrajectoryState,
  collisionsAt: BaseCollisionProbe
): { segment: ResolvedBaseSegment; state: BaseTrajectoryState } | undefined {
  const increments = Math.ceil(MAX_INITIAL_EGRESS_DISTANCE / LINEAR_STEP);
  for (let step = 1; step <= increments; step += 1) {
    const distance = step * LINEAR_STEP;
    for (const linearSign of [1, -1] as const) {
      const target = {
        x: initial.position.x + Math.sin(initial.yaw) * distance * linearSign,
        y: ROBOT_SPEC.base.centerY,
        z: initial.position.z + Math.cos(initial.yaw) * distance * linearSign
      };
      const segment: ResolvedBaseSegment = {
        waypointIndex: 0,
        start: { ...initial.position },
        target,
        bodyYaw: initial.yaw,
        linearSign,
        distance
      };
      const result = sweepSegment(initial, segment, collisionsAt);
      if (result.ok && result.state.collisions.length === 0) {
        return { segment, state: result.state };
      }
    }
  }
  return undefined;
}

/** Turn onto the segment heading, then translate along it. */
function sweepSegment(
  initial: BaseTrajectoryState,
  segment: ResolvedBaseSegment,
  collisionsAt: BaseCollisionProbe
): BaseTrajectoryResult {
  const rotated = sweepRotation(
    initial,
    segment.bodyYaw,
    "turn_to_waypoint",
    segment.waypointIndex,
    collisionsAt
  );
  if (!rotated.ok) return rotated;

  let state = rotated.state;
  const steps = Math.max(1, Math.ceil(segment.distance / LINEAR_STEP));
  for (let step = 1; step <= steps; step += 1) {
    const progress = step / steps;
    const next = inspectPose(
      state,
      {
        x: segment.start.x + (segment.target.x - segment.start.x) * progress,
        y: ROBOT_SPEC.base.centerY,
        z: segment.start.z + (segment.target.z - segment.start.z) * progress
      },
      segment.bodyYaw,
      "follow_path",
      segment.waypointIndex,
      progress,
      collisionsAt
    );
    if (!next.ok) return next;
    state = next.state;
  }
  return { ok: true, state };
}

/** Rotate in place to `targetYaw`, checking each intermediate heading. */
function sweepRotation(
  initial: BaseTrajectoryState,
  targetYaw: number,
  phase: string,
  waypointIndex: number,
  collisionsAt: BaseCollisionProbe
): BaseTrajectoryResult {
  let state = initial;
  const delta = normalizeAngle(targetYaw - initial.yaw);
  if (Math.abs(delta) <= 1e-9) return { ok: true, state };
  const steps = Math.max(1, Math.ceil(Math.abs(delta) / ANGULAR_STEP));
  for (let step = 1; step <= steps; step += 1) {
    const progress = step / steps;
    const next = inspectPose(
      state,
      initial.position,
      normalizeAngle(initial.yaw + delta * progress),
      phase,
      waypointIndex,
      progress,
      collisionsAt
    );
    if (!next.ok) return next;
    state = next.state;
  }
  return { ok: true, state };
}

/**
 * Accepts a pose if it does not make contact worse than the pose before it.
 * The comparison matters: the rig may legitimately *start* in contact — resting
 * against a carried object, or nudged by a previous command — and demanding a
 * collision-free pose would refuse to let it move away from what it touches.
 */
function inspectPose(
  current: BaseTrajectoryState,
  position: Vec3,
  yaw: number,
  phase: string,
  waypointIndex: number,
  progress: number,
  collisionsAt: BaseCollisionProbe
): BaseTrajectoryResult {
  const collisions = collisionsAt(position, yaw);
  if (!collisionTransitionAllowed(current.collisions, collisions)) {
    return {
      ok: false,
      issue: {
        phase,
        waypoint_index: waypointIndex,
        progress,
        position,
        yaw,
        collisions: collisionSetJson(collisions)
      }
    };
  }
  return {
    ok: true,
    state: { position: { ...position }, yaw: normalizeAngle(yaw), collisions }
  };
}
