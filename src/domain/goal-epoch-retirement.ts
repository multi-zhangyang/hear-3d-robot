import { z } from "zod";
import {
  GoalDAGValidationError,
  assertGoalModelSource,
  registerGoalEvidence,
  rehashGoalDAG,
  restoreGoalDAG,
  type GoalDAG,
  type GoalHarnessValidation,
  type GoalModelSource
} from "./goal-epoch.js";

export const GOAL_RETIREMENT_STATUSES = [
  "blocked",
  "abandoned",
  "superseded",
  "expired"
] as const;

const GoalRetirementStatusSchema = z.enum(GOAL_RETIREMENT_STATUSES);
const GoalEpochRetirementSchema = z.object({
  status: GoalRetirementStatusSchema,
  retired_by: z.custom<GoalModelSource>(),
  reason: z.string().trim().min(1),
  resolution_evidence_refs: z.array(z.string().trim().min(1)).min(1),
  resolved_world_revision: z.number().int().nonnegative()
}).strict();

export function retireGoalEpoch(
  persisted: GoalDAG,
  input: z.input<typeof GoalEpochRetirementSchema>,
  harness: GoalHarnessValidation
): GoalDAG {
  const dag = restoreGoalDAG(persisted, harness);
  const retirement = GoalEpochRetirementSchema.parse(input);
  assertGoalModelSource(retirement.retired_by, harness, "retire_goal_epoch");
  const epochIndex = dag.epochs.findIndex(
    (epoch) => epoch.epoch_id === dag.current_epoch_id
  );
  const epoch = dag.epochs[epochIndex];
  if (dag.status !== "active" || !epoch || epoch.status !== "active") {
    throw new GoalDAGValidationError(
      "no_active_goal_epoch",
      "There is no active model-selected Goal epoch to retire"
    );
  }
  if (retirement.resolved_world_revision < epoch.created_world_revision) {
    throw new GoalDAGValidationError(
      "world_revision_regression",
      "A Goal epoch cannot retire before it began"
    );
  }
  const evidenceRefs = uniqueSorted(retirement.resolution_evidence_refs);
  if (evidenceRefs.length !== retirement.resolution_evidence_refs.length) {
    throw new GoalDAGValidationError(
      "duplicate_evidence",
      "A Goal retirement cannot repeat physical evidence"
    );
  }
  const evidence = registerGoalEvidence(
    dag.evidence,
    evidenceRefs,
    retirement.resolved_world_revision,
    harness
  );
  if (retirement.status === "blocked" && !evidenceRefs.some(
    (ref) => evidence[ref]?.kind === "action_receipt"
  )) {
    throw new GoalDAGValidationError(
      "blocked_evidence_missing",
      "A blocked Goal requires an action receipt from the retirement revision"
    );
  }
  const candidate = dag.candidates[epoch.candidate_id];
  if (!candidate || candidate.status !== "active") {
    throw new GoalDAGValidationError(
      "active_candidate_missing",
      "The active Goal epoch has no matching active candidate"
    );
  }
  const epochs = [...dag.epochs];
  epochs[epochIndex] = {
    ...epoch,
    status: retirement.status,
    retired_by: retirement.retired_by,
    retirement_reason: retirement.reason,
    physical_evidence_refs: {
      ...epoch.physical_evidence_refs,
      resolution: evidenceRefs
    },
    resolved_world_revision: retirement.resolved_world_revision
  };
  return rehashGoalDAG({
    ...dag,
    status: "awaiting_model_selection",
    candidates: {
      ...dag.candidates,
      [candidate.candidate_id]: {
        ...candidate,
        status: retirement.status,
        physical_evidence_refs: {
          ...candidate.physical_evidence_refs,
          resolution: evidenceRefs
        },
        resolved_world_revision: retirement.resolved_world_revision
      }
    },
    epochs,
    current_epoch_id: null,
    evidence
  });
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) => (
    left < right ? -1 : left > right ? 1 : 0
  ));
}
