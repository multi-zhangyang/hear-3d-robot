import type { Goal, JsonValue, Scenario } from "../domain/schema.js";
import type { HumanoidCheckerResult } from "../domain/humanoid-run.js";
import type { HumanoidWorldSnapshot } from "../world/humanoid/world.js";
import { GoalValidationError } from "./goal-validation.js";

export function assertHumanoidGoalSupported(goal: Goal, scenario: Scenario): void {
  for (const predicate of goal.predicates) {
    if (predicate.type === "robot_in_zone"
      && !scenario.zones.some((zone) => zone.id === predicate.zone_id)) {
      throw new GoalValidationError(`Unknown zone: ${predicate.zone_id}`);
    }
    if (predicate.type === "object_in_zone") {
      const object = scenario.objects.find((candidate) => candidate.id === predicate.object_id);
      if (!object) throw new GoalValidationError(`Unknown object: ${predicate.object_id}`);
      if (!object.portable) throw new GoalValidationError(`Object is not movable: ${predicate.object_id}`);
      if (!scenario.zones.some((zone) => zone.id === predicate.zone_id)) {
        throw new GoalValidationError(`Unknown zone: ${predicate.zone_id}`);
      }
    }
    if (predicate.type === "object_at"
      && !scenario.objects.some((object) => object.id === predicate.object_id)) {
      throw new GoalValidationError(`Unknown object: ${predicate.object_id}`);
    }
  }
}

export function checkHumanoidGoal(
  goal: Goal,
  scenario: Scenario,
  world: HumanoidWorldSnapshot
): HumanoidCheckerResult {
  const checks = goal.predicates.map((predicate, index) => {
    const name = `${index + 1}:${predicate.type}`;
    if (predicate.type === "robot_at") {
      const distance = planarDistance(world.robot.rootPosition, predicate.target);
      return check(name, distance <= predicate.tolerance, {
        distance,
        tolerance: predicate.tolerance,
        position: world.robot.rootPosition,
        target: predicate.target
      });
    }
    if (predicate.type === "robot_in_zone") {
      const zone = scenario.zones.find((candidate) => candidate.id === predicate.zone_id);
      const inside = zone !== undefined
        && Math.abs(world.robot.rootPosition.x - zone.center.x)
          <= zone.size.x / 2 + predicate.tolerance
        && Math.abs(world.robot.rootPosition.z - zone.center.z)
          <= zone.size.z / 2 + predicate.tolerance;
      return check(name, inside, {
        zone_id: predicate.zone_id,
        robot_position: world.robot.rootPosition,
        zone_center: zone?.center ?? null,
        zone_size: zone?.size ?? null,
        inside,
        tolerance: predicate.tolerance
      });
    }
    if (predicate.type === "object_in_zone") {
      const object = world.robot.objects[predicate.object_id];
      const descriptor = scenario.objects.find((candidate) => candidate.id === predicate.object_id);
      const zone = scenario.zones.find((candidate) => candidate.id === predicate.zone_id);
      const inside = object !== undefined && descriptor !== undefined && zone !== undefined
        && objectInsideZone(object.position, descriptor.size, zone, predicate.tolerance);
      return check(
        name,
        object !== undefined && zone !== undefined && inside === predicate.expected,
        {
          object_id: predicate.object_id,
          object_position: object?.position ?? null,
          object_size: descriptor?.size ?? null,
          zone_id: predicate.zone_id,
          zone_center: zone?.center ?? null,
          zone_size: zone?.size ?? null,
          inside,
          expected: predicate.expected,
          tolerance: predicate.tolerance
        }
      );
    }
    if (predicate.type === "object_at") {
      const object = world.robot.objects[predicate.object_id];
      const distance = object ? distance3(object.position, predicate.target) : null;
      return check(name, distance !== null && distance <= predicate.tolerance, {
        object_id: predicate.object_id,
        position: object?.position ?? null,
        target: predicate.target,
        distance,
        tolerance: predicate.tolerance
      });
    }
    return assertNever(predicate);
  });
  return {
    success: checks.every((entry) => entry.passed),
    goal,
    worldFrame: world.frame,
    worldRevision: world.worldRevision,
    checks,
    checkedAt: new Date().toISOString()
  };
}

function assertNever(value: never): never {
  throw new Error(`Unsupported humanoid goal predicate: ${JSON.stringify(value)}`);
}

function check(name: string, passed: boolean, actual: unknown): {
  name: string;
  passed: boolean;
  actual: JsonValue;
} {
  return { name, passed, actual: json(actual) };
}

function objectInsideZone(
  position: { x: number; y: number; z: number },
  size: { x: number; y: number; z: number },
  zone: Scenario["zones"][number],
  tolerance: number
): boolean {
  const bottom = position.y - size.y / 2;
  const surface = zone.center.y + zone.size.y / 2;
  return Math.abs(position.x - zone.center.x) + size.x / 2
      <= zone.size.x / 2 + tolerance
    && Math.abs(position.z - zone.center.z) + size.z / 2
      <= zone.size.z / 2 + tolerance
    && Math.abs(bottom - surface) <= Math.max(tolerance, 0.025);
}

function planarDistance(
  left: { x: number; z: number },
  right: { x: number; z: number }
): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}

function distance3(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number }
): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function json(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}
