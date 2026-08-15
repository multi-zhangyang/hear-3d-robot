import { z } from "zod";
import { GOAL_PREDICATE_TYPES } from "./schema.js";

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);

const GOAL_HISTORY_RESOLUTION_STATUSES = [
  "completed",
  "blocked",
  "abandoned",
  "superseded",
  "expired"
] as const;

const GOAL_HISTORY_ENTITY_KINDS = [
  "object",
  "zone",
  "solid",
  "end_effector"
] as const;

const GoalHistoryResolutionStatusSchema = z.enum(
  GOAL_HISTORY_RESOLUTION_STATUSES
);

const GoalHistorySelectedOutcomesSchema = z.object({
  total: z.number().int().nonnegative(),
  completed: z.number().int().nonnegative(),
  blocked: z.number().int().nonnegative(),
  abandoned: z.number().int().nonnegative(),
  superseded: z.number().int().nonnegative(),
  expired: z.number().int().nonnegative()
}).strict().superRefine((outcomes, context) => {
  if (outcomes.total !== GOAL_HISTORY_RESOLUTION_STATUSES.reduce(
    (total, status) => total + outcomes[status],
    0
  )) {
    context.addIssue({
      code: "custom",
      path: ["total"],
      message: "Selected Goal outcome total is inconsistent"
    });
  }
});

const GoalHistoryLastSelectedSchema = z.object({
  epoch_sequence: z.number().int().positive(),
  status: GoalHistoryResolutionStatusSchema,
  world_revision: z.number().int().nonnegative()
}).strict();

const GoalHistoryLastNotSelectedSchema = z.object({
  epoch_sequence: z.number().int().positive(),
  world_revision: z.number().int().nonnegative()
}).strict();

const GoalHistoryDimensionOutcomeBaseSchema = z.object({
  selected: GoalHistorySelectedOutcomesSchema,
  not_selected: z.number().int().nonnegative(),
  last_selected: GoalHistoryLastSelectedSchema.nullable(),
  last_not_selected: GoalHistoryLastNotSelectedSchema.nullable()
}).strict();

const GoalHistoryDimensionOutcomeSchema = GoalHistoryDimensionOutcomeBaseSchema
  .superRefine(validateDimensionOutcome);

function validateDimensionOutcome(
  outcome: z.infer<typeof GoalHistoryDimensionOutcomeBaseSchema>,
  context: z.RefinementCtx
): void {
  if ((outcome.selected.total === 0) !== (outcome.last_selected === null)) {
    context.addIssue({
      code: "custom",
      path: ["last_selected"],
      message: "Last selected Goal outcome is inconsistent"
    });
  }
  if ((outcome.not_selected === 0) !== (outcome.last_not_selected === null)) {
    context.addIssue({
      code: "custom",
      path: ["last_not_selected"],
      message: "Last unselected Goal outcome is inconsistent"
    });
  }
}

const GoalHistoryPredicateOutcomeSchema = GoalHistoryDimensionOutcomeBaseSchema
  .extend({ predicate_type: z.enum(GOAL_PREDICATE_TYPES) })
  .strict()
  .superRefine(validateDimensionOutcome);

const GoalHistoryGoalOutcomeSchema = GoalHistoryDimensionOutcomeBaseSchema
  .extend({ goal_constraint_sha256: Sha256Schema })
  .strict()
  .superRefine(validateDimensionOutcome);

const GoalHistoryEntityOutcomeSchema = GoalHistoryDimensionOutcomeBaseSchema
  .extend({
    entity_kind: z.enum(GOAL_HISTORY_ENTITY_KINDS),
    entity_id: z.string().trim().min(1)
  })
  .strict()
  .superRefine(validateDimensionOutcome);

export const GoalHistoryOutcomeSummarySchema = z.object({
  selected: GoalHistorySelectedOutcomesSchema,
  not_selected: z.number().int().nonnegative(),
  goal_outcomes: z.array(GoalHistoryGoalOutcomeSchema).optional(),
  predicate_outcomes: z.array(GoalHistoryPredicateOutcomeSchema),
  entity_outcomes: z.array(GoalHistoryEntityOutcomeSchema)
}).strict().superRefine((summary, context) => {
  const predicateKeys = summary.predicate_outcomes.map((entry) => entry.predicate_type);
  const goalKeys = (summary.goal_outcomes ?? []).map(
    (entry) => entry.goal_constraint_sha256
  );
  if (!uniqueSorted(goalKeys)) {
    context.addIssue({
      code: "custom",
      path: ["goal_outcomes"],
      message: "Exact Goal outcome dimensions must be unique and sorted"
    });
  }
  if (!uniqueSorted(predicateKeys)) {
    context.addIssue({
      code: "custom",
      path: ["predicate_outcomes"],
      message: "Goal predicate outcome dimensions must be unique and sorted"
    });
  }
  const entityKeys = summary.entity_outcomes.map(
    (entry) => `${entry.entity_kind}\0${entry.entity_id}`
  );
  if (!uniqueSorted(entityKeys)) {
    context.addIssue({
      code: "custom",
      path: ["entity_outcomes"],
      message: "Goal entity outcome dimensions must be unique and sorted"
    });
  }
});

export const GoalHistorySummarySchema = z.object({
  version: z.literal(1),
  archived_epoch_count: z.number().int().nonnegative(),
  last_record_sha256: Sha256Schema.nullable(),
  records_without_alternate_history: z.number().int().nonnegative(),
  exact_goal_outcomes_complete: z.boolean().optional(),
  outcomes: GoalHistoryOutcomeSummarySchema
}).strict().superRefine((summary, context) => {
  const empty = summary.archived_epoch_count === 0;
  if (empty !== (summary.last_record_sha256 === null)
    || summary.outcomes.selected.total !== summary.archived_epoch_count
    || summary.records_without_alternate_history > summary.archived_epoch_count) {
    context.addIssue({
      code: "custom",
      path: ["archived_epoch_count"],
      message: "Goal history summary archive head is inconsistent"
    });
  }
});

export type GoalHistoryResolutionStatus = z.infer<
  typeof GoalHistoryResolutionStatusSchema
>;
export type GoalHistorySelectedOutcomes = z.infer<
  typeof GoalHistorySelectedOutcomesSchema
>;
export type GoalHistoryDimensionOutcome = z.infer<
  typeof GoalHistoryDimensionOutcomeSchema
>;
export type GoalHistoryOutcomeSummary = z.infer<
  typeof GoalHistoryOutcomeSummarySchema
>;
export type GoalHistorySummary = z.infer<typeof GoalHistorySummarySchema>;

function uniqueSorted(values: readonly string[]): boolean {
  return values.every((value, index) => (
    (index === 0 || values[index - 1]! < value)
      && values.indexOf(value) === index
  ));
}
