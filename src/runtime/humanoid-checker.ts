import type { Goal, JsonValue, Scenario } from "../domain/schema.js";
import { goalSha256 } from "../domain/goal-identity.js";
import {
  HumanoidGoalProgressSchema,
  type HumanoidCheckerResult,
  type HumanoidGoalProgress
} from "../domain/humanoid-run.js";
import {
  humanoidEndEffectorBody,
  humanoidEndEffectorPosition,
  humanoidEndEffectorRotation
} from "../world/humanoid/end-effectors.js";
import {
  normalizeQuaternion,
  quaternionAngularDistance
} from "../world/geometry.js";
import { assessHumanoidObjectSettledOnSupport } from "../world/humanoid/object-settled-support.js";
import type { HumanoidWorldSnapshot } from "../world/humanoid/world.js";
import { GoalValidationError } from "./goal-validation.js";

interface PredicateEvaluation {
  name: string;
  satisfied: boolean;
  actual: Record<string, unknown>;
}

export interface HumanoidGoalAdvance {
  progress: HumanoidGoalProgress;
  checker: HumanoidCheckerResult;
}

export function assertHumanoidGoalSupported(goal: Goal, scenario: Scenario): void {
  for (const predicate of goal.predicates) {
    if (predicate.type === "robot_in_zone"
      && !scenario.zones.some((zone) => zone.id === predicate.zone_id)) {
      throw new GoalValidationError(`Unknown zone: ${predicate.zone_id}`);
    }
    if (predicate.type === "object_in_zone" || predicate.type === "object_placed") {
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
    if (predicate.type === "object_grasped") {
      const object = scenario.objects.find((candidate) => candidate.id === predicate.object_id);
      if (!object) throw new GoalValidationError(`Unknown object: ${predicate.object_id}`);
      if (!object.portable) {
        throw new GoalValidationError(`Object is not movable: ${predicate.object_id}`);
      }
    }
  }
}

export function createHumanoidGoalProgress(
  goal: Goal,
  world: Pick<HumanoidWorldSnapshot, "frame" | "worldRevision">
): HumanoidGoalProgress {
  return HumanoidGoalProgressSchema.parse({
    version: 1,
    goal_sha256: goalSha256(goal),
    predicate_count: goal.predicates.length,
    last_world_frame: world.frame,
    last_world_revision: world.worldRevision,
    predicate_streaks: goal.predicates.map(() => 0)
  });
}

export function assertHumanoidGoalProgressIntegrity(
  goal: Goal,
  world: Pick<HumanoidWorldSnapshot, "frame" | "worldRevision">,
  rawProgress: HumanoidGoalProgress
): void {
  const progress = assertProgressIdentity(goal, rawProgress);
  if (progress.last_world_frame !== world.frame
    || progress.last_world_revision !== world.worldRevision) {
    throw new Error(
      "Humanoid goal progress is not aligned with the authoritative world frame"
    );
  }
}

export function inspectHumanoidGoal(
  goal: Goal,
  scenario: Scenario,
  world: HumanoidWorldSnapshot,
  rawProgress: HumanoidGoalProgress
): HumanoidCheckerResult {
  const progress = assertProgressIdentity(goal, rawProgress);
  assertHumanoidGoalProgressIntegrity(goal, world, progress);
  return checkerResult(
    goal,
    world,
    evaluatePredicates(goal, scenario, world),
    progress
  );
}

export function advanceHumanoidGoal(
  goal: Goal,
  scenario: Scenario,
  world: HumanoidWorldSnapshot,
  rawProgress: HumanoidGoalProgress
): HumanoidGoalAdvance {
  const previous = assertProgressIdentity(goal, rawProgress);
  const sameFrame = world.frame === previous.last_world_frame
    && world.worldRevision === previous.last_world_revision;
  if (sameFrame) {
    return {
      progress: structuredClone(previous),
      checker: inspectHumanoidGoal(goal, scenario, world, previous)
    };
  }
  if (world.frame <= previous.last_world_frame
    || world.worldRevision <= previous.last_world_revision) {
    throw new Error("Humanoid goal progress cannot move backward or advance partially");
  }

  const evaluations = evaluatePredicates(goal, scenario, world);
  const contiguous = world.frame === previous.last_world_frame + 1
    && world.worldRevision === previous.last_world_revision + 1;
  const predicateStreaks = goal.predicates.map((predicate, index) => {
    if (predicate.type !== "end_effector_at") return 0;
    if (!evaluations[index]!.satisfied) return 0;
    const prior = contiguous ? previous.predicate_streaks[index]! : 0;
    return Math.min(prior + 1, predicate.stable_frames);
  });
  const progress = HumanoidGoalProgressSchema.parse({
    version: 1,
    goal_sha256: previous.goal_sha256,
    predicate_count: previous.predicate_count,
    last_world_frame: world.frame,
    last_world_revision: world.worldRevision,
    predicate_streaks: predicateStreaks
  });
  return {
    progress,
    checker: checkerResult(goal, world, evaluations, progress)
  };
}

/**
 * Backward-compatible instantaneous view. New temporal predicates remain
 * fail-closed unless the caller supplies persisted physical-frame progress.
 */
export function checkHumanoidGoal(
  goal: Goal,
  scenario: Scenario,
  world: HumanoidWorldSnapshot,
  progress: HumanoidGoalProgress = createHumanoidGoalProgress(goal, world)
): HumanoidCheckerResult {
  return inspectHumanoidGoal(goal, scenario, world, progress);
}

function assertProgressIdentity(
  goal: Goal,
  rawProgress: HumanoidGoalProgress
): HumanoidGoalProgress {
  const progress = HumanoidGoalProgressSchema.parse(rawProgress);
  if (progress.goal_sha256 !== goalSha256(goal)) {
    throw new Error("Humanoid goal progress belongs to another goal");
  }
  if (progress.predicate_count !== goal.predicates.length) {
    throw new Error("Humanoid goal progress predicate count does not match the goal");
  }
  for (let index = 0; index < goal.predicates.length; index += 1) {
    const predicate = goal.predicates[index]!;
    const streak = progress.predicate_streaks[index]!;
    if (predicate.type === "end_effector_at") {
      if (streak > predicate.stable_frames) {
        throw new Error(`Humanoid goal predicate ${index} has an impossible stability streak`);
      }
    } else if (streak !== 0) {
      throw new Error(`Humanoid goal predicate ${index} cannot carry temporal progress`);
    }
  }
  return progress;
}

function checkerResult(
  goal: Goal,
  world: HumanoidWorldSnapshot,
  evaluations: readonly PredicateEvaluation[],
  progress: HumanoidGoalProgress
): HumanoidCheckerResult {
  const checks = evaluations.map((evaluation, index) => {
    const predicate = goal.predicates[index]!;
    if (predicate.type !== "end_effector_at") {
      return check(evaluation.name, evaluation.satisfied, evaluation.actual);
    }
    const currentStableFrames = progress.predicate_streaks[index]!;
    return check(
      evaluation.name,
      evaluation.satisfied && currentStableFrames >= predicate.stable_frames,
      {
        ...evaluation.actual,
        satisfied: evaluation.satisfied,
        current_stable_frames: currentStableFrames,
        required_stable_frames: predicate.stable_frames
      }
    );
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

function evaluatePredicates(
  goal: Goal,
  scenario: Scenario,
  world: HumanoidWorldSnapshot
): PredicateEvaluation[] {
  return goal.predicates.map((predicate, index) => {
    const name = `${index + 1}:${predicate.type}`;
    if (predicate.type === "robot_at") {
      const distance = planarDistance(world.robot.rootPosition, predicate.target);
      return evaluation(name, distance <= predicate.tolerance, {
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
      return evaluation(name, inside, {
        zone_id: predicate.zone_id,
        robot_position: world.robot.rootPosition,
        zone_center: zone?.center ?? null,
        zone_size: zone?.size ?? null,
        inside,
        tolerance: predicate.tolerance
      });
    }
    if (predicate.type === "block_removed") {
      const block = scenario.obstacles.find((candidate) => (
        candidate.id === predicate.block_id
      ));
      return evaluation(name, block === undefined, {
        block_id: predicate.block_id,
        present: block !== undefined,
        center: block?.center ?? null,
        size: block?.size ?? null
      });
    }
    if (predicate.type === "object_in_zone") {
      const object = world.robot.objects[predicate.object_id];
      const descriptor = scenario.objects.find((candidate) => candidate.id === predicate.object_id);
      const zone = scenario.zones.find((candidate) => candidate.id === predicate.zone_id);
      const inside = object !== undefined && descriptor !== undefined && zone !== undefined
        && objectInsideZone(object.position, descriptor.size, zone, predicate.tolerance);
      return evaluation(
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
    if (predicate.type === "object_placed") {
      const object = world.robot.objects[predicate.object_id];
      const descriptor = scenario.objects.find((candidate) => candidate.id === predicate.object_id);
      const zone = scenario.zones.find((candidate) => candidate.id === predicate.zone_id);
      const inside = object !== undefined && descriptor !== undefined && zone !== undefined
        && objectInsideZone(object.position, descriptor.size, zone, predicate.tolerance);
      const currentAssessments = world.grasp.assessments.filter((assessment) => (
        assessment.frame === world.frame
          && assessment.object_id === predicate.object_id
      ));
      const assessedHands = new Set(currentAssessments.map((assessment) => assessment.hand));
      const evidenceComplete = assessedHands.has("left") && assessedHands.has("right");
      const verifiedHands = currentAssessments
        .filter((assessment) => assessment.grasp_verified)
        .map((assessment) => assessment.hand)
        .sort(compareCodePoints);
      const settledSupport = assessHumanoidObjectSettledOnSupport({
        objectId: predicate.object_id,
        objectObservable: object !== undefined && descriptor !== undefined,
        snapshot: world.robot
      });
      return evaluation(
        name,
        inside
          && evidenceComplete
          && verifiedHands.length === 0
          && settledSupport.status === "satisfied",
        {
          object_id: predicate.object_id,
          object_position: object?.position ?? null,
          object_size: descriptor?.size ?? null,
          zone_id: predicate.zone_id,
          zone_center: zone?.center ?? null,
          zone_size: zone?.size ?? null,
          inside,
          tolerance: predicate.tolerance,
          world_frame: world.frame,
          grasp: {
            contract_sha256: world.grasp.contractSha256,
            evidence_complete: evidenceComplete,
            assessed_hands: [...assessedHands].sort(compareCodePoints),
            verified_hands: verifiedHands,
            assessments: currentAssessments
          },
          settled_support: settledSupport
        }
      );
    }
    if (predicate.type === "object_at") {
      const object = world.robot.objects[predicate.object_id];
      const distance = object ? distance3(object.position, predicate.target) : null;
      return evaluation(name, distance !== null && distance <= predicate.tolerance, {
        object_id: predicate.object_id,
        position: object?.position ?? null,
        target: predicate.target,
        distance,
        tolerance: predicate.tolerance
      });
    }
    if (predicate.type === "object_grasped") {
      const requiredHands = predicate.hand === "either"
        ? ["left", "right"] as const
        : [predicate.hand];
      const currentAssessments = world.grasp.assessments.filter((assessment) => (
        assessment.frame === world.frame
          && assessment.object_id === predicate.object_id
          && requiredHands.includes(assessment.hand)
      ));
      const assessedHands = new Set(currentAssessments.map((assessment) => assessment.hand));
      const evidenceComplete = requiredHands.every((hand) => assessedHands.has(hand));
      const verified = currentAssessments.filter((assessment) => assessment.grasp_verified);
      return evaluation(
        name,
        evidenceComplete && verified.length > 0,
        {
          object_id: predicate.object_id,
          requested_hand: predicate.hand,
          contract_sha256: world.grasp.contractSha256,
          world_frame: world.frame,
          evidence_complete: evidenceComplete,
          assessed_hands: [...assessedHands].sort(compareCodePoints),
          verified_hands: verified.map((assessment) => assessment.hand).sort(compareCodePoints),
          assessments: currentAssessments
        }
      );
    }
    if (predicate.type === "end_effector_at") {
      const position = humanoidEndEffectorPosition(
        world.robot,
        predicate.end_effector,
        predicate.frame
      );
      const distance = position ? distance3(position, predicate.target) : null;
      const targetOrientation = predicate.orientation
        ? normalizeQuaternion(predicate.orientation)
        : undefined;
      const orientation = targetOrientation
        ? observedEndEffectorRotation(world, predicate.end_effector, predicate.frame)
        : undefined;
      const orientationError: number | null | undefined = targetOrientation
        ? orientation
          ? quaternionAngularDistance(targetOrientation, orientation)
          : null
        : undefined;
      const orientationSatisfied = !targetOrientation
        || typeof orientationError === "number"
          && predicate.orientation_tolerance_rad !== undefined
          && orientationError <= predicate.orientation_tolerance_rad;
      return evaluation(
        name,
        distance !== null && distance <= predicate.tolerance && orientationSatisfied,
        {
          end_effector: predicate.end_effector,
          body: humanoidEndEffectorBody(predicate.end_effector),
          frame: predicate.frame,
          position,
          target: predicate.target,
          distance,
          tolerance: predicate.tolerance,
          ...(targetOrientation
            ? {
                orientation: orientation ?? null,
                target_orientation: targetOrientation,
                orientation_error_rad: orientationError,
                orientation_tolerance_rad: predicate.orientation_tolerance_rad
              }
            : {})
        }
      );
    }
    return assertNever(predicate);
  });
}

function observedEndEffectorRotation(
  world: HumanoidWorldSnapshot,
  endEffector: Parameters<typeof humanoidEndEffectorRotation>[1],
  frame: Parameters<typeof humanoidEndEffectorRotation>[2]
): ReturnType<typeof humanoidEndEffectorRotation> {
  try {
    return humanoidEndEffectorRotation(world.robot, endEffector, frame);
  } catch {
    return null;
  }
}

function assertNever(value: never): never {
  throw new Error(`Unsupported humanoid goal predicate: ${JSON.stringify(value)}`);
}

function evaluation(
  name: string,
  satisfied: boolean,
  actual: Record<string, unknown>
): PredicateEvaluation {
  return { name, satisfied, actual };
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

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function json(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}
