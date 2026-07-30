import type { Goal, Scenario } from "../domain/schema.js";

export class GoalValidationError extends Error {
  readonly statusCode = 400;
}

export function assertScenarioIntegrity(scenarioId: string, scenario: Scenario): void {
  const ids = new Set<string>();
  for (const item of [...scenario.obstacles, ...scenario.objects, ...scenario.zones]) {
    if (ids.has(item.id)) throw new Error(`${scenarioId} contains duplicate entity ID: ${item.id}`);
    ids.add(item.id);
  }
  for (const object of scenario.objects) {
    if (object.container_id) {
      const container = scenario.objects.find((candidate) => candidate.id === object.container_id);
      if (container?.kind !== "container") {
        throw new Error(`${scenarioId}.${object.id} references an invalid container`);
      }
    }
    if (object.key_id) {
      const key = scenario.objects.find((candidate) => candidate.id === object.key_id);
      if (object.kind !== "container" || key?.kind !== "key") {
        throw new Error(`${scenarioId}.${object.id} references an invalid key`);
      }
    }
  }
  const affordanceIds = new Set<string>();
  for (const affordance of scenario.affordances) {
    if (affordanceIds.has(affordance.id)) {
      throw new Error(`${scenarioId} contains duplicate affordance ID: ${affordance.id}`);
    }
    affordanceIds.add(affordance.id);
    const container = scenario.objects.find((candidate) => candidate.id === affordance.container_id);
    const key = scenario.objects.find((candidate) => candidate.id === affordance.key_id);
    if (container?.kind !== "container" || key?.kind !== "key") {
      throw new Error(`${scenarioId}.${affordance.id} references an invalid keyed lock pair`);
    }
    if (container.key_id !== key.id) {
      throw new Error(`${scenarioId}.${affordance.id} does not match ${container.id}.key_id`);
    }
  }
  assertGoalSupported(scenario.default_goal, scenario);
}

export function assertGoalSupported(goal: Goal, scenario: Scenario): void {
  for (const predicate of goal.predicates) {
    if (predicate.type === "robot_at") {
      assertPointInBounds(predicate.target, scenario, "Robot target");
      continue;
    }
    if (predicate.type === "robot_in_zone") {
      if (!scenario.zones.some((candidate) => candidate.id === predicate.zone_id)) {
        throw new GoalValidationError(`Unknown zone: ${predicate.zone_id}`);
      }
      continue;
    }
    if (predicate.type === "terrain_explored") {
      if (!scenario.terrain) {
        throw new GoalValidationError("terrain_explored requires a generated voxel world");
      }
      continue;
    }
    if (predicate.type === "voxel_at") {
      const terrain = scenario.terrain;
      if (!terrain) throw new GoalValidationError("voxel_at requires a generated voxel world");
      const { column, level, row } = predicate.coordinate;
      if (column >= terrain.columns || row >= terrain.rows || level >= terrain.maximum_height) {
        throw new GoalValidationError(
          `Voxel target (${column}, ${level}, ${row}) is outside the editable world`
        );
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
    if (predicate.type === "object_attached" && predicate.expected && !object.portable) {
      throw new GoalValidationError(`Object is not portable: ${predicate.object_id}`);
    }
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
