import type { GoalDAG } from "../../domain/goal-epoch.js";
import { goalConstraintSha256 } from "../../domain/goal-identity.js";
import type { HumanoidRunMode } from "../../domain/run-mode.js";
import type { Goal, GoalPredicate } from "../../domain/schema.js";
import { goalHistoryLifetimeProjection } from
  "../../domain/goal-history-summary.js";
import type { HumanoidSpatialBeliefObservation } from
  "../../world/humanoid/spatial-belief-map.js";
import type { GoalEvidenceArtifact } from "./goal-evidence.js";

const HISTORY_WINDOW = 128;
const RECENT_EPOCH_LIMIT = 12;

export function createHumanoidAutonomyContext(input: {
  goalDAG: GoalDAG;
  worldEvidence: GoalEvidenceArtifact;
  runMode?: HumanoidRunMode;
  missionGoal?: Goal;
  spatialBelief?: HumanoidSpatialBeliefObservation;
}) {
  if ((input.worldEvidence.version !== 2
    && input.worldEvidence.version !== 3
    && input.worldEvidence.version !== 4)
    || !input.worldEvidence.observation) {
    throw new Error("Autonomy context requires current affordance-bearing world evidence");
  }
  const observation = input.worldEvidence.observation;
  const lifetime = goalHistoryLifetimeProjection(input.goalDAG);
  const epochs = input.goalDAG.epochs.slice(-HISTORY_WINDOW);
  const history = epochs.flatMap((epoch) => {
    const candidate = input.goalDAG.candidates[epoch.candidate_id];
    return candidate ? [{ epoch, candidate }] : [];
  });
  const objectCounts = lifetimeEntityCounts(lifetime.entity_outcomes, "object");
  const solidCounts = lifetimeEntityCounts(lifetime.entity_outcomes, "solid");
  const zoneCounts = lifetimeEntityCounts(lifetime.entity_outcomes, "zone");
  const predicateCounts = new Map(lifetime.predicate_outcomes.map((entry) => [
    entry.predicate_type,
    entry.selected.total
  ]));
  const missionGoalIdentity = input.missionGoal
    ? goalConstraintSha256(input.missionGoal)
    : null;
  const missionGoalOutcomes: OutcomeCounts = {
    total: 0,
    completed: 0,
    unsuccessful: 0,
    active: 0
  };
  const lifetimeMissionGoal = missionGoalIdentity === null
    ? undefined
    : lifetime.goal_outcomes.find((entry) => (
        entry.goal_constraint_sha256 === missionGoalIdentity
      ));
  const bootstrapMissionGoalCompleted = missionGoalOutcomes.completed > 0
    || (lifetimeMissionGoal?.selected.completed ?? 0) > 0
    ? true
    : lifetime.exact_goal_outcomes_complete ? false : null;
  for (const { epoch, candidate } of history) {
    if (missionGoalIdentity !== null
      && goalConstraintSha256(candidate.goal) === missionGoalIdentity) {
      incrementOutcomeValue(missionGoalOutcomes, epoch.status);
    }
    if (epoch.status !== "active") continue;
    for (const predicateType of new Set(candidate.goal.predicates.map(
      (predicate) => predicate.type
    ))) {
      predicateCounts.set(
        predicateType,
        (predicateCounts.get(predicateType) ?? 0) + 1
      );
    }
    incrementActiveEntityCounts(objectCounts, candidate.goal.predicates, predicateObjectId);
    incrementActiveEntityCounts(solidCounts, candidate.goal.predicates, predicateSolidId);
    incrementActiveEntityCounts(zoneCounts, candidate.goal.predicates, predicateZoneId);
  }
  return {
    source_world_frame: input.worldEvidence.evidence.world_frame,
    source_world_revision: input.worldEvidence.evidence.world_revision,
    goal_dag_state_sha256: input.goalDAG.state_sha256,
    selection_authority: "goal_manager_model" as const,
    harness_selection: "none" as const,
    continuous_drive_state: input.runMode === "continuous"
      ? {
          drive_phase: bootstrapMissionGoalCompleted === true
            ? "open_ended"
            : bootstrapMissionGoalCompleted === false
              ? "bootstrap_mission"
              : "history_unknown",
          bootstrap_mission_goal_completed: bootstrapMissionGoalCompleted,
          bootstrap_goal_policy: bootstrapMissionGoalCompleted === true
            ? "model_selected_successor_goals"
            : "submit_and_select_exact_mission_goal",
          exact_mission_goal_history_complete:
            lifetime.exact_goal_outcomes_complete,
          working_exact_mission_goal_outcomes: missionGoalOutcomes,
          untried_visible_object_ids: observation.objects
            .filter((object) => !objectCounts.has(object.id))
            .map((object) => object.id),
          untried_observable_solid_ids: "solids" in observation
            ? observation.solids
                .filter((solid) => !solidCounts.has(solid.id))
                .map((solid) => solid.id)
            : [],
          untried_zone_ids: observation.zones
            .filter((zone) => !zoneCounts.has(zone.id))
            .map((zone) => zone.id),
          spatial_exploration: input.spatialBelief
            ? {
                protocol: input.spatialBelief.protocol,
                frontier_model: input.spatialBelief.frontier_model,
                coverage_ratio: input.spatialBelief.coverage_ratio,
                observed_cell_count: input.spatialBelief.observed_cell_count,
                visited_cell_count: input.spatialBelief.visited_cell_count,
                reachable_free_cell_count:
                  input.spatialBelief.reachable_free_cell_count,
                frontier_candidates: input.spatialBelief.frontiers
              }
            : null
        }
      : null,
    capability_surface: {
      embodiment_predicates: [
        "robot_at",
        "robot_in_zone",
        "end_effector_at"
      ],
      manipulable_object_predicates: [
        "object_grasped",
        "object_at",
        "object_in_zone",
        "object_placed",
        "object_inside",
        "object_on"
      ],
      articulated_object_predicates: ["articulation_state"],
      static_solid_predicates: ["block_removed"]
    },
    object_frontier: observation.objects.map((object) => ({
      object_id: object.id,
      role: object.role,
      portable: object.portable,
      affordances: "affordances" in object ? object.affordances : [],
      articulation: "articulation" in object ? object.articulation : null,
      prior_goal_outcomes: outcome(objectCounts.get(object.id)),
      supported_goal_predicates: [
        ...(object.portable
          ? [
            "object_grasped",
            "object_at",
            "object_in_zone",
            "object_placed",
            "object_inside",
            "object_on"
          ]
          : []),
        ...("articulation" in object && object.articulation
          ? ["articulation_state"]
          : [])
      ]
    })),
    solid_frontier: "solids" in observation
      ? observation.solids.map((solid) => ({
          solid_id: solid.id,
          source_id: solid.source_id,
          kind: solid.kind,
          prior_goal_outcomes: outcome(solidCounts.get(solid.id)),
          supported_goal_predicates: solid.kind === "block"
            ? ["block_removed"]
            : []
        }))
      : [],
    zone_frontier: observation.zones.map((zone) => ({
      zone_id: zone.id,
      prior_goal_outcomes: outcome(zoneCounts.get(zone.id)),
      supported_goal_predicates: ["robot_in_zone", "object_in_zone", "object_placed"]
    })),
    history: {
      total_epoch_count: input.goalDAG.next_epoch_index,
      analyzed_epoch_count: history.length,
      lifetime_outcomes: lifetime,
      predicate_counts: Object.fromEntries(
        [...predicateCounts.entries()].sort(([left], [right]) => compare(left, right))
      ),
      recent_epochs: history.slice(-RECENT_EPOCH_LIMIT).map(({ epoch, candidate }) => ({
        epoch_id: epoch.epoch_id,
        candidate_content_sha256: candidate.content_sha256,
        summary: candidate.goal.summary,
        predicate_types: [...new Set(candidate.goal.predicates.map(
          (predicate) => predicate.type
        ))],
        status: epoch.status,
        created_world_revision: epoch.created_world_revision,
        resolved_world_revision: epoch.resolved_world_revision
      }))
    }
  };
}

interface OutcomeCounts {
  total: number;
  completed: number;
  unsuccessful: number;
  active: number;
}

interface LifetimeEntityOutcome {
  entity_kind: "object" | "zone" | "solid" | "end_effector";
  entity_id: string;
  selected: {
    total: number;
    completed: number;
  };
}

function lifetimeEntityCounts(
  entries: readonly LifetimeEntityOutcome[],
  kind: LifetimeEntityOutcome["entity_kind"]
): Map<string, OutcomeCounts> {
  return new Map(entries.filter((entry) => (
    entry.entity_kind === kind && entry.selected.total > 0
  )).map((entry) => [
    entry.entity_id,
    {
      total: entry.selected.total,
      completed: entry.selected.completed,
      unsuccessful: entry.selected.total - entry.selected.completed,
      active: 0
    }
  ]));
}

function incrementActiveEntityCounts(
  counts: Map<string, OutcomeCounts>,
  predicates: readonly GoalPredicate[],
  identity: (predicate: GoalPredicate) => string | null
): void {
  for (const id of new Set(predicates.map(identity).filter(
    (value): value is string => value !== null
  ))) {
    incrementOutcome(counts, id, "active");
  }
}

function incrementOutcome(
  counts: Map<string, OutcomeCounts>,
  id: string,
  status: "active" | "completed" | "blocked" | "abandoned" | "superseded" | "expired"
): void {
  const current = counts.get(id) ?? {
    total: 0,
    completed: 0,
    unsuccessful: 0,
    active: 0
  };
  incrementOutcomeValue(current, status);
  counts.set(id, current);
}

function incrementOutcomeValue(
  current: OutcomeCounts,
  status: "active" | "completed" | "blocked" | "abandoned" | "superseded" | "expired"
): void {
  current.total += 1;
  if (status === "completed") current.completed += 1;
  else if (status === "active") current.active += 1;
  else current.unsuccessful += 1;
}

function outcome(counts?: OutcomeCounts): OutcomeCounts {
  return counts
    ? { ...counts }
    : { total: 0, completed: 0, unsuccessful: 0, active: 0 };
}

function predicateObjectId(predicate: GoalPredicate): string | null {
  return predicate.type === "object_at"
    || predicate.type === "object_in_zone"
    || predicate.type === "object_placed"
    || predicate.type === "object_grasped"
    || predicate.type === "object_inside"
    || predicate.type === "object_on"
    || predicate.type === "articulation_state"
    ? predicate.object_id
    : null;
}

function predicateZoneId(predicate: GoalPredicate): string | null {
  return predicate.type === "robot_in_zone"
    || predicate.type === "object_in_zone"
    || predicate.type === "object_placed"
    ? predicate.zone_id
    : null;
}

function predicateSolidId(predicate: GoalPredicate): string | null {
  return predicate.type === "block_removed" ? predicate.block_id : null;
}

function compare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
