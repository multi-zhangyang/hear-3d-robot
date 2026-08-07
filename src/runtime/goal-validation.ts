import {
  GoalPredicateSchema,
  type Goal,
  type GoalPredicate,
  type Scenario
} from "../domain/schema.js";
import { assertScenarioChunkIntegrity } from "../domain/scenario-chunk.js";
import { humanoidObjectCapability } from "../world/humanoid/object-capability.js";

export class GoalValidationError extends Error {
  readonly statusCode = 400;
}

export function assertScenarioIntegrity(scenarioId: string, scenario: Scenario): void {
  assertScenarioChunkIntegrity(scenario, scenario.chunk_manifest);
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
    if (predicate.type === "block_removed") {
      if (!scenario.obstacles.some((candidate) => candidate.id === predicate.block_id)) {
        throw new GoalValidationError(`Unknown block: ${predicate.block_id}`);
      }
      continue;
    }
    const object = scenario.objects.find((candidate) => candidate.id === predicate.object_id);
    if (!object) throw new GoalValidationError(`Unknown object: ${predicate.object_id}`);
    if (predicate.type === "object_inside") {
      if (!object.portable) {
        throw new GoalValidationError(`Object is not movable: ${predicate.object_id}`);
      }
      const container = scenario.objects.find((candidate) => (
        candidate.id === predicate.container_id
      ));
      if (!container) throw new GoalValidationError(`Unknown container: ${predicate.container_id}`);
      if (!humanoidObjectCapability(container).affordances.includes("container")) {
        throw new GoalValidationError(`Object is not a container: ${predicate.container_id}`);
      }
      continue;
    }
    if (predicate.type === "object_on") {
      if (!object.portable) {
        throw new GoalValidationError(`Object is not movable: ${predicate.object_id}`);
      }
      const support = scenario.objects.find((candidate) => candidate.id === predicate.support_id);
      if (!support) throw new GoalValidationError(`Unknown support: ${predicate.support_id}`);
      if (!humanoidObjectCapability(support).affordances.includes("support_surface")) {
        throw new GoalValidationError(`Object is not a support surface: ${predicate.support_id}`);
      }
      continue;
    }
    if (predicate.type === "articulation_state") {
      const articulation = humanoidObjectCapability(object).articulation;
      if (!articulation || articulation.joint_id !== predicate.joint_id) {
        throw new GoalValidationError(
          `Unknown articulation ${predicate.joint_id} on ${predicate.object_id}`
        );
      }
      continue;
    }
    if (predicate.type === "object_grasped" || predicate.type === "object_placed") {
      if (!object.portable) {
        throw new GoalValidationError(`Object is not movable: ${predicate.object_id}`);
      }
      if (predicate.type === "object_grasped") continue;
    }
    if (predicate.type === "object_in_zone" || predicate.type === "object_placed") {
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
