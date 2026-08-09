import { GoalHistoryArchiveRecordSchema } from
  "../../domain/goal-history-archive.js";
import {
  goalCandidateSequence,
  type GoalCandidate,
  type GoalDAG,
  type GoalEpoch
} from "../../domain/goal-epoch.js";
import type { JsonValue, Vec3 } from "../../domain/schema.js";
import type { JournalPage } from "../../persistence/run-store.js";

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
  "object_inside",
  "object_on",
  "articulation_state",
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
  world_region?: GoalHistoryWorldRegion;
  limit: number;
}

interface GoalHistoryWorldRegion {
  center: Vec3;
  horizontal_radius_m: number;
  vertical_radius_m?: number;
}

export interface GoalHistoryJournalReader {
  readJournalPage(
    name: "goal_history",
    from: number,
    limit: number
  ): Promise<JournalPage>;
  readJournalTail(name: "goal_history", limit: number): Promise<JournalPage>;
}

interface HistoryEntry {
  candidate: GoalCandidate;
  sequence: number;
  epoch?: GoalEpoch;
}

export async function recallGoalHistory(input: {
  goalDAG: GoalDAG;
  journal: GoalHistoryJournalReader;
  currentWorldRevision: number;
  request: GoalHistoryRecallRequest;
}): Promise<JsonValue> {
  const requestedIds = new Set(input.request.candidate_ids ?? []);
  const currentIds = new Set(Object.keys(input.goalDAG.candidates));
  const currentEpochs = new Map(input.goalDAG.epochs.map((epoch) => [
    epoch.candidate_id,
    epoch
  ]));
  const selected: HistoryEntry[] = [];
  let totalMatches = 0;

  const consider = (entry: HistoryEntry, current: boolean): void => {
    if (!matchesRequest(entry.candidate, entry.sequence, requestedIds, input.request)) {
      return;
    }
    if (!current && currentIds.has(entry.candidate.candidate_id)) {
      const retained = selected.findIndex((existing) => (
        existing.candidate.candidate_id === entry.candidate.candidate_id
      ));
      if (retained >= 0) selected[retained] = entry;
      return;
    }
    totalMatches += 1;
    insertLatest(selected, entry, input.request.limit);
  };

  for (const candidate of Object.values(input.goalDAG.candidates)) {
    const sequence = goalCandidateSequence(input.goalDAG, candidate.candidate_id);
    if (sequence === undefined) {
      throw new Error(`Working Goal candidate has no stable sequence: ${candidate.candidate_id}`);
    }
    consider({
      candidate,
      sequence,
      ...(currentEpochs.get(candidate.candidate_id)
        ? { epoch: currentEpochs.get(candidate.candidate_id)! }
        : {})
    }, true);
  }

  const tail = await input.journal.readJournalTail("goal_history", 1);
  if (tail.total !== input.goalDAG.archive.record_count) {
    throw new Error("Goal history journal is not aligned with the checkpoint archive head");
  }
  let before = tail.total;
  let expectedRecordSha256 = input.goalDAG.archive.last_record_sha256;
  while (before > 0) {
    const from = Math.max(0, before - 128);
    const page = await input.journal.readJournalPage("goal_history", from, before - from);
    if (page.entries.length !== before - from) {
      throw new Error(`Goal history journal stopped before record ${before}`);
    }
    for (let index = page.entries.length - 1; index >= 0; index -= 1) {
      const record = GoalHistoryArchiveRecordSchema.parse(page.entries[index]);
      if (record.sequence !== from + index + 1
        || record.record_sha256 !== expectedRecordSha256) {
        throw new Error(`Goal history journal chain is inconsistent: ${record.sequence}`);
      }
      consider({
        candidate: record.candidate,
        sequence: record.candidate_sequence,
        epoch: record.epoch
      }, false);
      if (record.version === 2) {
        for (const alternate of record.alternate_candidates) {
          consider({
            candidate: alternate.candidate,
            sequence: alternate.candidate_sequence
          }, false);
        }
      }
      expectedRecordSha256 = record.previous_record_sha256;
    }
    before = from;
  }
  if (expectedRecordSha256 !== null) {
    throw new Error("Goal history journal does not terminate at its genesis record");
  }

  selected.sort((left, right) => right.sequence - left.sequence);
  const returnedIds = new Set(selected.map(({ candidate }) => candidate.candidate_id));
  const oldestSequence = selected.at(-1)?.sequence;
  return structuredClone({
    historical_only: true,
    current_world_revision: input.currentWorldRevision,
    goal_dag_state_sha256: input.goalDAG.state_sha256,
    goal_history_archive_sha256: input.goalDAG.archive.last_record_sha256,
    total_candidate_count: input.goalDAG.next_candidate_sequence - 1,
    total_matches: totalMatches,
    returned: selected.length,
    world_region_query: input.request.world_region ?? null,
    candidates: selected.map(projectHistoryEntry),
    missing_candidate_ids: [...requestedIds].filter((id) => !returnedIds.has(id)),
    next_before_candidate_sequence: requestedIds.size === 0
      && oldestSequence !== undefined
      && totalMatches > selected.length
      ? oldestSequence
      : null
  }) as JsonValue;
}

function matchesRequest(
  candidate: GoalCandidate,
  sequence: number,
  requestedIds: ReadonlySet<string>,
  request: GoalHistoryRecallRequest
): boolean {
  if (requestedIds.size > 0 && !requestedIds.has(candidate.candidate_id)) return false;
  if (requestedIds.size === 0
    && request.before_candidate_sequence !== undefined
    && sequence >= request.before_candidate_sequence) return false;
  if (!matchesAny(request.statuses, [candidate.status])) return false;
  const predicates = candidate.goal.predicates;
  if (!matchesAny(request.predicate_types, predicates.map((predicate) => predicate.type))) {
    return false;
  }
  if (!matchesAny(request.object_ids, predicates.flatMap(predicateObjectIds))) return false;
  if (!matchesAny(request.solid_ids, predicates.flatMap(predicateSolidIds))) return false;
  if (!matchesAny(request.zone_ids, predicates.flatMap(predicateZoneIds))) return false;
  return request.world_region === undefined
    || predicates.some((predicate) => predicateMatchesWorldRegion(
      predicate,
      request.world_region!
    ));
}

function insertLatest(
  selected: HistoryEntry[],
  entry: HistoryEntry,
  limit: number
): void {
  selected.push(entry);
  selected.sort((left, right) => right.sequence - left.sequence);
  if (selected.length > limit) selected.pop();
}

function projectHistoryEntry({ candidate, sequence, epoch }: HistoryEntry) {
  return {
    sequence,
    candidate_id: candidate.candidate_id,
    status: candidate.status,
    selection_outcome: epoch ? "selected" : candidate.status === "expired"
      ? "not_selected"
      : null,
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
}

function matchesAny(
  requested: readonly string[] | undefined,
  actual: readonly string[]
): boolean {
  return requested === undefined || requested.length === 0
    || requested.some((value) => actual.includes(value));
}

function predicateObjectIds(
  predicate: GoalCandidate["goal"]["predicates"][number]
): string[] {
  if (predicate.type === "object_inside") {
    return [predicate.object_id, predicate.container_id];
  }
  if (predicate.type === "object_on") {
    return [predicate.object_id, predicate.support_id];
  }
  if (predicate.type === "object_in_zone"
    || predicate.type === "object_placed"
    || predicate.type === "object_at"
    || predicate.type === "object_grasped"
    || predicate.type === "articulation_state") {
    return [predicate.object_id];
  }
  return [];
}

function predicateSolidIds(
  predicate: GoalCandidate["goal"]["predicates"][number]
): string[] {
  return predicate.type === "block_removed" ? [predicate.block_id] : [];
}

function predicateZoneIds(
  predicate: GoalCandidate["goal"]["predicates"][number]
): string[] {
  if (predicate.type === "robot_in_zone"
    || predicate.type === "object_in_zone"
    || predicate.type === "object_placed") {
    return [predicate.zone_id];
  }
  return [];
}

function predicateMatchesWorldRegion(
  predicate: GoalCandidate["goal"]["predicates"][number],
  region: GoalHistoryWorldRegion
): boolean {
  if (predicate.type === "robot_at") {
    return horizontalDistance(predicate.target, region.center)
      <= region.horizontal_radius_m;
  }
  if (predicate.type === "object_at"
    || (predicate.type === "end_effector_at" && predicate.frame === "world")) {
    return horizontalDistance(predicate.target, region.center)
        <= region.horizontal_radius_m
      && (region.vertical_radius_m === undefined
        || Math.abs(predicate.target.y - region.center.y) <= region.vertical_radius_m);
  }
  return false;
}

function horizontalDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}
