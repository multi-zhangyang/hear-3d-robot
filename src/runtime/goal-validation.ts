import {
  GoalPredicateSchema,
  type Goal,
  type GoalPredicate,
  type Scenario
} from "../domain/schema.js";

export class GoalValidationError extends Error {
  readonly statusCode = 400;
}

export function assertScenarioIntegrity(scenarioId: string, scenario: Scenario): void {
  const ids = new Set<string>();
  for (const item of [...scenario.obstacles, ...scenario.objects, ...scenario.zones]) {
    if (ids.has(item.id)) throw new Error(`${scenarioId} contains duplicate entity ID: ${item.id}`);
    ids.add(item.id);
  }
  assertGoalSupported(scenario.default_goal, scenario);
}

export function assertGoalSupported(goal: Goal, scenario: Scenario): void {
  const predicates: GoalPredicate[] = goal.predicates.map((predicate) => (
    GoalPredicateSchema.parse(predicate)
  ));
  for (const predicate of predicates) {
    if (predicate.type === "robot_at") {
      assertPointInBounds(predicate.target, scenario, "Robot target");
      continue;
    }
    if (predicate.type === "end_effector_at") {
      if (predicate.frame === "world") {
        assertEndEffectorWorldTargetInBounds(predicate.target, scenario);
      }
      continue;
    }
    if (predicate.type === "robot_in_zone") {
      if (!scenario.zones.some((candidate) => candidate.id === predicate.zone_id)) {
        throw new GoalValidationError(`Unknown zone: ${predicate.zone_id}`);
      }
      continue;
    }
    const object = scenario.objects.find((candidate) => candidate.id === predicate.object_id);
    if (!object) throw new GoalValidationError(`Unknown object: ${predicate.object_id}`);
    if (predicate.type === "object_in_zone") {
      if (!scenario.zones.some((candidate) => candidate.id === predicate.zone_id)) {
        throw new GoalValidationError(`Unknown zone: ${predicate.zone_id}`);
      }
      continue;
    }
    if (predicate.type === "object_at") {
      assertPointInBounds(predicate.target, scenario, `Target for ${predicate.object_id}`);
      continue;
    }
  }
}

function assertEndEffectorWorldTargetInBounds(
  point: { x: number; y: number; z: number },
  scenario: Scenario
): void {
  if (point.x < 0 || point.z < 0 || point.y < 0
    || point.x > scenario.bounds.width
    || point.z > scenario.bounds.depth) {
    throw new GoalValidationError("End-effector world target is outside the world bounds");
  }
}

function assertPointInBounds(point: { x: number; y: number; z: number }, scenario: Scenario, label: string): void {
  const margin = 0.4;
  if (point.x < margin || point.z < margin || point.y < 0
    || point.x > scenario.bounds.width - margin
    || point.z > scenario.bounds.depth - margin) {
    throw new GoalValidationError(`${label} is outside the world bounds`);
  }
}
