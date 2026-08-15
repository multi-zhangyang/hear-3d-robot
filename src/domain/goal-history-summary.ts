import type { GoalPredicate } from "./schema.js";
import { goalConstraintSha256 } from "./goal-identity.js";
import {
  GoalHistoryOutcomeSummarySchema,
  GoalHistorySummarySchema,
  type GoalHistoryDimensionOutcome,
  type GoalHistoryOutcomeSummary,
  type GoalHistoryResolutionStatus,
  type GoalHistorySelectedOutcomes,
  type GoalHistorySummary
} from "./goal-history-summary-schema.js";
import {
  goalCandidateSequence,
  type GoalCandidate,
  type GoalDAG,
  type GoalEpoch
} from "./goal-epoch.js";

export interface GoalHistorySummaryRecordInput {
  sequence: number;
  recordSha256: string;
  candidate: GoalCandidate;
  epoch: GoalEpoch;
  alternateCandidates: readonly GoalCandidate[];
  alternateHistoryComplete: boolean;
}

function createEmptyGoalHistoryOutcomeSummary(): GoalHistoryOutcomeSummary {
  return {
    selected: emptySelectedOutcomes(),
    not_selected: 0,
    goal_outcomes: [],
    predicate_outcomes: [],
    entity_outcomes: []
  };
}

export function createEmptyGoalHistorySummary(): GoalHistorySummary {
  return {
    version: 1,
    archived_epoch_count: 0,
    last_record_sha256: null,
    records_without_alternate_history: 0,
    exact_goal_outcomes_complete: true,
    outcomes: createEmptyGoalHistoryOutcomeSummary()
  };
}

export function appendGoalHistorySummary(
  persisted: GoalHistorySummary,
  input: GoalHistorySummaryRecordInput
): GoalHistorySummary {
  const summary = GoalHistorySummarySchema.parse(persisted);
  if (input.sequence !== summary.archived_epoch_count + 1) {
    throw new Error(`Goal history summary cannot advance to record ${input.sequence}`);
  }
  if (input.epoch.epoch_index + 1 !== input.sequence
    || input.epoch.candidate_id !== input.candidate.candidate_id) {
    throw new Error(`Goal history summary record identity is inconsistent: ${input.sequence}`);
  }
  const status = terminalStatus(input.epoch.status);
  let outcomes = accumulateSelected(
    summary.outcomes,
    input.candidate,
    status,
    input.sequence,
    requiredWorldRevision(input.epoch)
  );
  for (const alternate of input.alternateCandidates) {
    if (alternate.status !== "expired") {
      throw new Error(`Goal history alternate is not unselected: ${alternate.candidate_id}`);
    }
    outcomes = accumulateNotSelected(
      outcomes,
      alternate,
      input.sequence,
      requiredWorldRevision(input.epoch)
    );
  }
  return GoalHistorySummarySchema.parse({
    version: 1,
    archived_epoch_count: input.sequence,
    last_record_sha256: input.recordSha256,
    records_without_alternate_history: summary.records_without_alternate_history
      + (input.alternateHistoryComplete ? 0 : 1),
    exact_goal_outcomes_complete:
      summary.exact_goal_outcomes_complete
        ?? summary.archived_epoch_count === 0,
    outcomes
  });
}

export function goalHistoryLifetimeProjection(goalDAG: GoalDAG) {
  const archived = goalDAG.archive.summary;
  if (goalDAG.archive.record_count > 0 && archived === null) {
    throw new Error("Goal history lifetime summary has not been rebuilt");
  }
  let outcomes = structuredClone(
    archived?.outcomes ?? createEmptyGoalHistoryOutcomeSummary()
  );
  for (const epoch of goalDAG.epochs) {
    const candidate = goalDAG.candidates[epoch.candidate_id];
    if (!candidate) {
      throw new Error(`Working Goal epoch candidate is unavailable: ${epoch.candidate_id}`);
    }
    const sequence = epoch.epoch_index + 1;
    if (epoch.status !== "active") {
      outcomes = accumulateSelected(
        outcomes,
        candidate,
        terminalStatus(epoch.status),
        sequence,
        requiredWorldRevision(epoch)
      );
    }
    for (const alternate of workingSlateAlternates(goalDAG, epoch, candidate)) {
      outcomes = accumulateNotSelected(
        outcomes,
        alternate,
        sequence,
        requiredCandidateWorldRevision(alternate)
      );
    }
  }
  outcomes = GoalHistoryOutcomeSummarySchema.parse(outcomes);
  const goalOutcomes = outcomes.goal_outcomes ?? [];
  return {
    total_selected_epoch_count: goalDAG.next_epoch_index,
    resolved_selected_goal_count: outcomes.selected.total,
    active_selected_goal_count: goalDAG.status === "active" ? 1 : 0,
    archived_selected_goal_count: goalDAG.archive.record_count,
    working_selected_goal_count: goalDAG.epochs.length,
    records_without_alternate_history:
      archived?.records_without_alternate_history ?? 0,
    exact_goal_outcomes_complete: archived
      ? archived.exact_goal_outcomes_complete === true
      : true,
    ...outcomes,
    goal_outcomes: goalOutcomes
  };
}

function accumulateSelected(
  persisted: GoalHistoryOutcomeSummary,
  candidate: GoalCandidate,
  status: GoalHistoryResolutionStatus,
  sequence: number,
  worldRevision: number
): GoalHistoryOutcomeSummary {
  const next = structuredClone(persisted);
  incrementSelected(next.selected, status);
  updateDimensions(next, candidate, (outcome) => {
    incrementSelected(outcome.selected, status);
    outcome.last_selected = {
      epoch_sequence: sequence,
      status,
      world_revision: worldRevision
    };
  });
  return canonicalOutcomeSummary(next);
}

function accumulateNotSelected(
  persisted: GoalHistoryOutcomeSummary,
  candidate: GoalCandidate,
  sequence: number,
  worldRevision: number
): GoalHistoryOutcomeSummary {
  const next = structuredClone(persisted);
  next.not_selected += 1;
  updateDimensions(next, candidate, (outcome) => {
    outcome.not_selected += 1;
    outcome.last_not_selected = {
      epoch_sequence: sequence,
      world_revision: worldRevision
    };
  });
  return canonicalOutcomeSummary(next);
}

function updateDimensions(
  summary: GoalHistoryOutcomeSummary,
  candidate: GoalCandidate,
  update: (outcome: GoalHistoryDimensionOutcome) => void
): void {
  const goalOutcomes = new Map((summary.goal_outcomes ?? []).map((entry) => [
    entry.goal_constraint_sha256,
    entry
  ]));
  const goalIdentity = goalConstraintSha256(candidate.goal);
  const goalOutcome = goalOutcomes.get(goalIdentity) ?? {
    goal_constraint_sha256: goalIdentity,
    ...emptyDimensionOutcome()
  };
  update(goalOutcome);
  goalOutcomes.set(goalIdentity, goalOutcome);
  summary.goal_outcomes = [...goalOutcomes.values()].sort((left, right) => (
    compare(left.goal_constraint_sha256, right.goal_constraint_sha256)
  ));

  const predicates = new Set(candidate.goal.predicates.map((predicate) => predicate.type));
  const predicateOutcomes = new Map(summary.predicate_outcomes.map((entry) => [
    entry.predicate_type,
    entry
  ]));
  for (const predicateType of predicates) {
    const current = predicateOutcomes.get(predicateType) ?? {
      predicate_type: predicateType,
      ...emptyDimensionOutcome()
    };
    update(current);
    predicateOutcomes.set(predicateType, current);
  }
  summary.predicate_outcomes = [...predicateOutcomes.values()].sort((left, right) => (
    compare(left.predicate_type, right.predicate_type)
  ));

  const entityOutcomes = new Map(summary.entity_outcomes.map((entry) => [
    entityKey(entry.entity_kind, entry.entity_id),
    entry
  ]));
  const entities = new Map(candidate.goal.predicates.flatMap(predicateEntities).map((entity) => [
    entityKey(entity.entity_kind, entity.entity_id),
    entity
  ]));
  for (const [key, entity] of entities) {
    const current = entityOutcomes.get(key) ?? {
      ...entity,
      ...emptyDimensionOutcome()
    };
    update(current);
    entityOutcomes.set(key, current);
  }
  summary.entity_outcomes = [...entityOutcomes.values()].sort((left, right) => (
    compare(
      entityKey(left.entity_kind, left.entity_id),
      entityKey(right.entity_kind, right.entity_id)
    )
  ));
}

function workingSlateAlternates(
  goalDAG: GoalDAG,
  epoch: GoalEpoch,
  selected: GoalCandidate
): GoalCandidate[] {
  return Object.values(goalDAG.candidates).filter((candidate) => (
    candidate.status === "expired"
      && candidate.candidate_id !== selected.candidate_id
      && JSON.stringify(candidate.source) === JSON.stringify(selected.source)
      && candidate.resolved_world_revision === epoch.created_world_revision
      && JSON.stringify(candidate.physical_evidence_refs.resolution)
        === JSON.stringify(epoch.physical_evidence_refs.selection)
      && goalCandidateSequence(goalDAG, candidate.candidate_id) !== undefined
  ));
}

function predicateEntities(predicate: GoalPredicate): Array<{
  entity_kind: "object" | "zone" | "solid" | "end_effector";
  entity_id: string;
}> {
  switch (predicate.type) {
    case "robot_at":
      return [];
    case "robot_in_zone":
      return [{ entity_kind: "zone", entity_id: predicate.zone_id }];
    case "block_removed":
      return [{ entity_kind: "solid", entity_id: predicate.block_id }];
    case "object_in_zone":
    case "object_placed":
      return [
        { entity_kind: "object", entity_id: predicate.object_id },
        { entity_kind: "zone", entity_id: predicate.zone_id }
      ];
    case "object_inside":
      return [
        { entity_kind: "object", entity_id: predicate.object_id },
        { entity_kind: "object", entity_id: predicate.container_id }
      ];
    case "object_on":
      return [
        { entity_kind: "object", entity_id: predicate.object_id },
        { entity_kind: "object", entity_id: predicate.support_id }
      ];
    case "object_at":
    case "object_grasped":
    case "articulation_state":
      return [{ entity_kind: "object", entity_id: predicate.object_id }];
    case "end_effector_at":
      return [{ entity_kind: "end_effector", entity_id: predicate.end_effector }];
  }
}

function emptySelectedOutcomes(): GoalHistorySelectedOutcomes {
  return {
    total: 0,
    completed: 0,
    blocked: 0,
    abandoned: 0,
    superseded: 0,
    expired: 0
  };
}

function emptyDimensionOutcome(): GoalHistoryDimensionOutcome {
  return {
    selected: emptySelectedOutcomes(),
    not_selected: 0,
    last_selected: null,
    last_not_selected: null
  };
}

function incrementSelected(
  outcomes: GoalHistorySelectedOutcomes,
  status: GoalHistoryResolutionStatus
): void {
  outcomes.total += 1;
  outcomes[status] += 1;
}

function terminalStatus(status: GoalEpoch["status"]): GoalHistoryResolutionStatus {
  switch (status) {
    case "completed":
    case "blocked":
    case "abandoned":
    case "superseded":
    case "expired":
      return status;
    case "active":
      throw new Error(`Goal history cannot summarize a nonterminal epoch: ${status}`);
  }
}

function requiredWorldRevision(epoch: GoalEpoch): number {
  if (epoch.resolved_world_revision === null) {
    throw new Error(`Goal history epoch is unresolved: ${epoch.epoch_id}`);
  }
  return epoch.resolved_world_revision;
}

function requiredCandidateWorldRevision(candidate: GoalCandidate): number {
  if (candidate.resolved_world_revision === null) {
    throw new Error(`Goal history candidate is unresolved: ${candidate.candidate_id}`);
  }
  return candidate.resolved_world_revision;
}

function canonicalOutcomeSummary(
  summary: GoalHistoryOutcomeSummary
): GoalHistoryOutcomeSummary {
  return GoalHistoryOutcomeSummarySchema.parse(summary);
}

function entityKey(kind: string, id: string): string {
  return `${kind}\0${id}`;
}

function compare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
