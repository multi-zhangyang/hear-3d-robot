import type { Vec3 } from "../../domain/schema.js";
import type { HumanoidObjectWorldModelEntry } from "../../world/humanoid/object-world-model.js";

type Articulation = NonNullable<HumanoidObjectWorldModelEntry["articulation"]>;

export interface HumanoidArticulationTrajectory {
  final_target_position: number;
  joint_target_position: number;
  joint_delta: number;
  horizon_complete: boolean;
  interaction_waypoints: Vec3[];
  joint_waypoints: number[];
  initial_direction_world: Vec3;
  path_length_m: number;
}

export function solveHumanoidArticulationTrajectory(input: {
  articulation: Articulation;
  interactionPoint: Vec3;
  targetPosition: number;
  maximumPathLengthM?: number;
}): HumanoidArticulationTrajectory {
  const current = input.articulation.position;
  if (current === null) {
    throw new Error("Articulation trajectory requires an observed joint position");
  }
  if (input.targetPosition < input.articulation.range.minimum - 1e-9
    || input.targetPosition > input.articulation.range.maximum + 1e-9) {
    throw new Error("Articulation trajectory target is outside the joint range");
  }
  const requestedDelta = input.targetPosition - current;
  if (Math.abs(requestedDelta) <= 1e-6) {
    throw new Error("Articulation trajectory target must differ from current position");
  }
  const axis = normalize(input.articulation.axis_world);
  const maximumPathLength = input.maximumPathLengthM ?? Number.POSITIVE_INFINITY;
  if (!(maximumPathLength > 0)) {
    throw new Error("Articulation trajectory horizon must be positive");
  }
  const radial = input.articulation.type === "hinge"
    ? subtract(input.interactionPoint, input.articulation.anchor_world)
    : null;
  const radius = radial ? perpendicularLength(radial, axis) : 1;
  if (input.articulation.type === "hinge" && radius <= 1e-6) {
    throw new Error("Articulation interaction point lies on its hinge axis");
  }
  const maximumDelta = input.articulation.type === "hinge"
    ? maximumPathLength / radius
    : maximumPathLength;
  const delta = Math.sign(requestedDelta) * Math.min(
    Math.abs(requestedDelta),
    maximumDelta
  );
  const segmentTarget = current + delta;
  const count = input.articulation.type === "hinge"
    ? sampleCount(Math.abs(delta), Math.PI / 24)
    : sampleCount(Math.abs(delta), 0.025);
  const interactionWaypoints = Array.from({ length: count }, (_, index) => {
    const progress = (index + 1) / count;
    if (input.articulation.type === "slide") {
      return add(input.interactionPoint, scale(axis, delta * progress));
    }
    const rotated = rotateAroundAxis(radial!, axis, delta * progress);
    return add(input.articulation.anchor_world, rotated);
  });
  const jointWaypoints = Array.from({ length: count }, (_, index) => (
    current + delta * (index + 1) / count
  ));
  const segments = [input.interactionPoint, ...interactionWaypoints];
  const pathLength = segments.slice(1).reduce((total, point, index) => (
    total + distance(segments[index]!, point)
  ), 0);
  return {
    final_target_position: input.targetPosition,
    joint_target_position: segmentTarget,
    joint_delta: delta,
    horizon_complete: Math.abs(delta - requestedDelta) <= 1e-9,
    interaction_waypoints: interactionWaypoints,
    joint_waypoints: jointWaypoints,
    initial_direction_world: normalize(subtract(
      interactionWaypoints[0]!,
      input.interactionPoint
    )),
    path_length_m: pathLength
  };
}

function sampleCount(distance: number, maximumStep: number): number {
  return Math.max(2, Math.min(24, Math.ceil(distance / maximumStep)));
}

function perpendicularLength(vector: Vec3, axis: Vec3): number {
  const projection = scale(axis, dot(vector, axis));
  const perpendicular = subtract(vector, projection);
  return Math.hypot(perpendicular.x, perpendicular.y, perpendicular.z);
}

function rotateAroundAxis(vector: Vec3, axis: Vec3, radians: number): Vec3 {
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  return add(
    add(scale(vector, cosine), scale(cross(axis, vector), sine)),
    scale(axis, dot(axis, vector) * (1 - cosine))
  );
}

function add(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x + right.x, y: left.y + right.y, z: left.z + right.z };
}

function subtract(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function scale(value: Vec3, amount: number): Vec3 {
  return { x: value.x * amount, y: value.y * amount, z: value.z * amount };
}

function normalize(value: Vec3): Vec3 {
  const length = Math.hypot(value.x, value.y, value.z);
  if (length <= 1e-9) throw new Error("Articulation trajectory direction is undefined");
  return scale(value, 1 / length);
}

function cross(left: Vec3, right: Vec3): Vec3 {
  return {
    x: left.y * right.z - left.z * right.y,
    y: left.z * right.x - left.x * right.z,
    z: left.x * right.y - left.y * right.x
  };
}

function dot(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}
