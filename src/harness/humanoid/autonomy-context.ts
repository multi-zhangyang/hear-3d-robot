import type { GoalDAG } from "../../domain/goal-epoch.js";
import type { GoalPredicate } from "../../domain/schema.js";
import type { GoalEvidenceArtifact } from "./goal-evidence.js";

const HISTORY_WINDOW = 128;
const RECENT_EPOCH_LIMIT = 12;

export function createHumanoidAutonomyContext(input: {
  goalDAG: GoalDAG;
  worldEvidence: GoalEvidenceArtifact;
}) {
  if ((input.worldEvidence.version !== 2
    && input.worldEvidence.version !== 3
    && input.worldEvidence.version !== 4)
    || !input.worldEvidence.observation) {
    throw new Error("Autonomy context requires current affordance-bearing world evidence");
  }
  const observation = input.worldEvidence.observation;
  const epochs = input.goalDAG.epochs.slice(-HISTORY_WINDOW);
  const history = epochs.flatMap((epoch) => {
    const candidate = input.goalDAG.candidates[epoch.candidate_id];
    return candidate ? [{ epoch, candidate }] : [];
  });
  const objectCounts = new Map<string, OutcomeCounts>();
  const solidCounts = new Map<string, OutcomeCounts>();
  const zoneCounts = new Map<string, OutcomeCounts>();
  const predicateCounts = new Map<string, number>();
  for (const { epoch, candidate } of history) {
    for (const predicate of candidate.goal.predicates) {
      predicateCounts.set(
        predicate.type,
        (predicateCounts.get(predicate.type) ?? 0) + 1
      );
      const objectId = predicateObjectId(predicate);
      if (objectId) incrementOutcome(objectCounts, objectId, epoch.status);
      const solidId = predicateSolidId(predicate);
      if (solidId) incrementOutcome(solidCounts, solidId, epoch.status);
      const zoneId = predicateZoneId(predicate);
      if (zoneId) incrementOutcome(zoneCounts, zoneId, epoch.status);
    }
  }
  return {
    source_world_frame: input.worldEvidence.evidence.world_frame,
    source_world_revision: input.worldEvidence.evidence.world_revision,
    goal_dag_state_sha256: input.goalDAG.state_sha256,
    selection_authority: "goal_manager_model" as const,
    harness_selection: "none" as const,
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
  current.total += 1;
  if (status === "completed") current.completed += 1;
  else if (status === "active") current.active += 1;
  else current.unsuccessful += 1;
  counts.set(id, current);
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
