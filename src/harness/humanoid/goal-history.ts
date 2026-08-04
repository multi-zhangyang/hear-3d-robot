import type { GoalDAG } from "../../domain/goal-epoch.js";
import type { JsonValue } from "../../domain/schema.js";

export const GOAL_HISTORY_STATUSES = [
  "proposed",
  "active",
  "completed",
  "blocked",
  "abandoned",
  "superseded",
  "expired"
] as const;

export const GOAL_HISTORY_PREDICATE_TYPES = [
  "robot_at",
  "robot_in_zone",
  "block_removed",
  "object_in_zone",
  "object_placed",
  "object_at",
  "object_grasped",
  "end_effector_at"
] as const;

export interface GoalHistoryRecallRequest {
  candidate_ids?: string[];
  before_candidate_sequence?: number;
  statuses?: Array<typeof GOAL_HISTORY_STATUSES[number]>;
  predicate_types?: Array<typeof GOAL_HISTORY_PREDICATE_TYPES[number]>;
  object_ids?: string[];
  solid_ids?: string[];
  zone_ids?: string[];
  limit: number;
}

/**
 * Projects the complete durable Goal DAG into a bounded, read-only history
 * page. Sequence is derived from the append-only candidate insertion order;
 * the returned DAG hash lets a caller detect that a later query saw a newer
 * snapshot.
 */
export function recallGoalHistory(input: {
  goalDAG: GoalDAG;
  currentWorldRevision: number;
  request: GoalHistoryRecallRequest;
}): JsonValue {
  const requestedIds = new Set(input.request.candidate_ids ?? []);
  const epochByCandidate = new Map(input.goalDAG.epochs.map((epoch) => [
    epoch.candidate_id,
    epoch
  ]));
  const candidates = Object.values(input.goalDAG.candidates).map(
    (candidate, index) => ({ candidate, sequence: index + 1 })
  );
  const matches = candidates.filter(({ candidate, sequence }) => {
    if (requestedIds.size > 0 && !requestedIds.has(candidate.candidate_id)) return false;
    if (requestedIds.size === 0
      && input.request.before_candidate_sequence !== undefined
      && sequence >= input.request.before_candidate_sequence) return false;
    if (!matchesAny(input.request.statuses, [candidate.status])) return false;
    const predicates = candidate.goal.predicates;
    if (!matchesAny(
      input.request.predicate_types,
      predicates.map((predicate) => predicate.type)
    )) return false;
    if (!matchesAny(input.request.object_ids, predicates.flatMap(predicateObjectIds))) {
      return false;
    }
    if (!matchesAny(input.request.solid_ids, predicates.flatMap(predicateSolidIds))) {
      return false;
    }
    return matchesAny(input.request.zone_ids, predicates.flatMap(predicateZoneIds));
  });
  const selected = [...matches]
    .sort((left, right) => right.sequence - left.sequence)
    .slice(0, input.request.limit);
  const returnedIds = new Set(selected.map(({ candidate }) => candidate.candidate_id));
  const oldestSequence = selected.at(-1)?.sequence;
  return structuredClone({
    historical_only: true,
    current_world_revision: input.currentWorldRevision,
    goal_dag_state_sha256: input.goalDAG.state_sha256,
    total_candidate_count: candidates.length,
    total_matches: matches.length,
    returned: selected.length,
    candidates: selected.map(({ candidate, sequence }) => {
      const epoch = epochByCandidate.get(candidate.candidate_id);
      return {
        sequence,
        candidate_id: candidate.candidate_id,
        status: candidate.status,
        goal: candidate.goal,
        mission_link: candidate.mission_link,
        dependency_candidate_ids: candidate.dependency_candidate_ids,
        created_world_revision: candidate.created_world_revision,
        resolved_world_revision: candidate.resolved_world_revision,
        epoch: epoch ? {
          epoch_id: epoch.epoch_id,
          epoch_index: epoch.epoch_index,
          status: epoch.status,
          retirement_reason: epoch.retirement_reason,
          created_world_revision: epoch.created_world_revision,
          resolved_world_revision: epoch.resolved_world_revision
        } : null
      };
    }),
    missing_candidate_ids: [...requestedIds].filter((id) => !returnedIds.has(id)),
    next_before_candidate_sequence: requestedIds.size === 0
      && oldestSequence !== undefined
      && oldestSequence > 1
      ? oldestSequence
      : null
  }) as JsonValue;
}

function matchesAny(
  requested: readonly string[] | undefined,
  actual: readonly string[]
): boolean {
  return requested === undefined || requested.length === 0
    || requested.some((value) => actual.includes(value));
}

function predicateObjectIds(
  predicate: GoalDAG["candidates"][string]["goal"]["predicates"][number]
): string[] {
  if (predicate.type === "object_in_zone"
    || predicate.type === "object_placed"
    || predicate.type === "object_at"
    || predicate.type === "object_grasped") {
    return [predicate.object_id];
  }
  return [];
}

function predicateSolidIds(
  predicate: GoalDAG["candidates"][string]["goal"]["predicates"][number]
): string[] {
  return predicate.type === "block_removed" ? [predicate.block_id] : [];
}

function predicateZoneIds(
  predicate: GoalDAG["candidates"][string]["goal"]["predicates"][number]
): string[] {
  if (predicate.type === "robot_in_zone"
    || predicate.type === "object_in_zone"
    || predicate.type === "object_placed") {
    return [predicate.zone_id];
  }
  return [];
}
