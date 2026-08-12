import { z } from "zod";

const DEFAULT_COMPACT_REPLAN_LIMIT = 3;
const DEFAULT_SPECIALIST_CALLS_PER_REPLAN = 8;
const DEFAULT_GOAL_REEVALUATION_CALL_LIMIT = 3;
const DEFAULT_REPLAN_RECOVERY_DEADLINE_MS = 120_000;
const DEFAULT_REPLAN_MODEL_CALL_SLO_MS = 30_000;

const HumanoidReplanModelCallRoleSchema = z.enum([
  "replan_decision",
  "specialist_replan",
  "goal_re_evaluation_decision",
  "goal_re_evaluation"
]);

const HumanoidReplanModelCallStatusSchema = z.enum([
  "started",
  "completed",
  "failed"
]);

export const HumanoidReplanModelCallSchema = z.object({
  model_call_id: z.string().uuid(),
  agent_id: z.string().trim().min(1),
  recovery_tier: z.enum(["compact_replan", "goal_re_evaluation"]),
  role: HumanoidReplanModelCallRoleSchema,
  status: HumanoidReplanModelCallStatusSchema,
  started_at: z.string().datetime(),
  completed_at: z.string().datetime().nullable(),
  latency_ms: z.number().int().nonnegative().nullable(),
  slo_ms: z.number().int().positive(),
  slo_violated: z.boolean().nullable()
}).strict().superRefine((call, context) => {
  const started = Date.parse(call.started_at);
  const completed = call.completed_at === null
    ? null
    : Date.parse(call.completed_at);
  if (call.status === "started") {
    if (completed !== null || call.latency_ms !== null || call.slo_violated !== null) {
      context.addIssue({
        code: "custom",
        message: "A started replan model call cannot carry terminal latency evidence"
      });
    }
    return;
  }
  if (completed === null
    || completed < started
    || call.latency_ms !== completed - started
    || call.slo_violated !== (call.latency_ms > call.slo_ms)) {
    context.addIssue({
      code: "custom",
      message: "A terminal replan model call requires consistent latency and SLO evidence"
    });
  }
});

export const HumanoidReplanBudgetSchema = z.object({
  version: z.literal(1),
  compact_replan_limit: z.number().int().positive(),
  compact_replans_started: z.number().int().nonnegative(),
  specialist_calls_per_replan: z.number().int().positive(),
  goal_reevaluation_call_limit: z.number().int().positive(),
  goal_reevaluation_started: z.boolean(),
  recovery_deadline_ms: z.number().int().positive(),
  model_call_slo_ms: z.number().int().positive(),
  recovery_started_at: z.string().datetime().nullable(),
  recovery_deadline_at: z.string().datetime().nullable(),
  model_calls: z.array(HumanoidReplanModelCallSchema).max(64)
}).strict().superRefine((budget, context) => {
  if (budget.compact_replans_started > budget.compact_replan_limit) {
    context.addIssue({
      code: "custom",
      path: ["compact_replans_started"],
      message: "Compact replan usage exceeds its durable budget"
    });
  }
  if ((budget.recovery_started_at === null)
    !== (budget.recovery_deadline_at === null)) {
    context.addIssue({
      code: "custom",
      path: ["recovery_deadline_at"],
      message: "Recovery start and deadline must be present together"
    });
  }
  if (budget.recovery_started_at && budget.recovery_deadline_at
    && Date.parse(budget.recovery_deadline_at)
      !== Date.parse(budget.recovery_started_at) + budget.recovery_deadline_ms) {
    context.addIssue({
      code: "custom",
      path: ["recovery_deadline_at"],
      message: "Recovery deadline does not match its configured window"
    });
  }
  const ids = budget.model_calls.map((call) => call.model_call_id);
  if (new Set(ids).size !== ids.length) {
    context.addIssue({
      code: "custom",
      path: ["model_calls"],
      message: "Replan model call identities must be unique"
    });
  }
  const compactDecisions = budget.model_calls.filter(
    (call) => call.role === "replan_decision"
  ).length;
  if (compactDecisions !== budget.compact_replans_started) {
    context.addIssue({
      code: "custom",
      path: ["compact_replans_started"],
      message: "Compact replan usage must equal its recorded decisions"
    });
  }
  const specialistCalls = budget.model_calls.filter(
    (call) => call.role === "specialist_replan"
  ).length;
  if (specialistCalls
    > budget.compact_replans_started * budget.specialist_calls_per_replan) {
    context.addIssue({
      code: "custom",
      path: ["model_calls"],
      message: "Specialist replanning exceeded the calls authorized by compact replans"
    });
  }
  const goalCalls = budget.model_calls.filter(
    (call) => call.role === "goal_re_evaluation"
  ).length;
  if (goalCalls > budget.goal_reevaluation_call_limit) {
    context.addIssue({
      code: "custom",
      path: ["model_calls"],
      message: "Goal re-evaluation exceeded its model-call budget"
    });
  }
  const hasGoalReevaluation = budget.model_calls.some(
    (call) => call.recovery_tier === "goal_re_evaluation"
  );
  if (budget.goal_reevaluation_started !== hasGoalReevaluation) {
    context.addIssue({
      code: "custom",
      path: ["goal_reevaluation_started"],
      message: "Goal re-evaluation state must be backed by a recorded model call"
    });
  }
});

export type HumanoidReplanBudget = z.infer<typeof HumanoidReplanBudgetSchema>;
export type HumanoidReplanModelCall = z.infer<typeof HumanoidReplanModelCallSchema>;
export function createHumanoidReplanBudget(): HumanoidReplanBudget {
  return HumanoidReplanBudgetSchema.parse({
    version: 1,
    compact_replan_limit: DEFAULT_COMPACT_REPLAN_LIMIT,
    compact_replans_started: 0,
    specialist_calls_per_replan: DEFAULT_SPECIALIST_CALLS_PER_REPLAN,
    goal_reevaluation_call_limit: DEFAULT_GOAL_REEVALUATION_CALL_LIMIT,
    goal_reevaluation_started: false,
    recovery_deadline_ms: DEFAULT_REPLAN_RECOVERY_DEADLINE_MS,
    model_call_slo_ms: DEFAULT_REPLAN_MODEL_CALL_SLO_MS,
    recovery_started_at: null,
    recovery_deadline_at: null,
    model_calls: []
  });
}

export function beginHumanoidReplanModelCall(
  source: HumanoidReplanBudget,
  input: {
    modelCallId: string;
    agentId: string;
    role: "coordinator" | "motion" | "goal_manager";
    at?: string;
  }
): { budget: HumanoidReplanBudget; call: HumanoidReplanModelCall } {
  const budget = structuredClone(HumanoidReplanBudgetSchema.parse(source));
  const at = input.at ?? new Date().toISOString();
  const startedMs = parseTimestamp(at, "Replan model call start");
  if (budget.model_calls.some((call) => call.model_call_id === input.modelCallId)) {
    throw new Error(`Duplicate replan model call: ${input.modelCallId}`);
  }
  if (budget.recovery_started_at === null) {
    budget.recovery_started_at = at;
    budget.recovery_deadline_at = new Date(
      startedMs + budget.recovery_deadline_ms
    ).toISOString();
  }
  const deadlineExceeded = startedMs > Date.parse(budget.recovery_deadline_at!);
  let recoveryTier: "compact_replan" | "goal_re_evaluation";
  let role: z.infer<typeof HumanoidReplanModelCallRoleSchema>;

  if (input.role === "coordinator") {
    if (budget.goal_reevaluation_started) {
      throw new Error("Replan budget exhausted after Goal re-evaluation escalation");
    }
    if (!deadlineExceeded
      && budget.compact_replans_started < budget.compact_replan_limit) {
      budget.compact_replans_started += 1;
      recoveryTier = "compact_replan";
      role = "replan_decision";
    } else {
      budget.goal_reevaluation_started = true;
      recoveryTier = "goal_re_evaluation";
      role = "goal_re_evaluation_decision";
    }
  } else if (input.role === "motion") {
    const specialistCalls = budget.model_calls.filter(
      (call) => call.role === "specialist_replan"
    ).length;
    const specialistLimit = budget.compact_replans_started
      * budget.specialist_calls_per_replan;
    if (budget.goal_reevaluation_started
      || budget.compact_replans_started === 0
      || specialistCalls >= specialistLimit) {
      throw new Error("Motion specialist has no remaining compact replan authority");
    }
    recoveryTier = "compact_replan";
    role = "specialist_replan";
  } else {
    const goalCalls = budget.model_calls.filter(
      (call) => call.role === "goal_re_evaluation"
    ).length;
    if (goalCalls >= budget.goal_reevaluation_call_limit) {
      throw new Error("Goal re-evaluation model-call budget exhausted");
    }
    budget.goal_reevaluation_started = true;
    recoveryTier = "goal_re_evaluation";
    role = "goal_re_evaluation";
  }

  const call = HumanoidReplanModelCallSchema.parse({
    model_call_id: input.modelCallId,
    agent_id: input.agentId,
    recovery_tier: recoveryTier,
    role,
    status: "started",
    started_at: at,
    completed_at: null,
    latency_ms: null,
    slo_ms: budget.model_call_slo_ms,
    slo_violated: null
  });
  budget.model_calls.push(call);
  return {
    budget: HumanoidReplanBudgetSchema.parse(budget),
    call: structuredClone(call)
  };
}

export function finishHumanoidReplanModelCall(
  source: HumanoidReplanBudget,
  input: {
    modelCallId: string;
    status: "completed" | "failed";
    at?: string;
  }
): { budget: HumanoidReplanBudget; call?: HumanoidReplanModelCall } {
  const budget = structuredClone(HumanoidReplanBudgetSchema.parse(source));
  const index = budget.model_calls.findIndex(
    (call) => call.model_call_id === input.modelCallId
  );
  if (index < 0) return { budget };
  const existing = budget.model_calls[index]!;
  if (existing.status !== "started") {
    throw new Error(`Replan model call is already terminal: ${input.modelCallId}`);
  }
  const at = input.at ?? new Date().toISOString();
  const completedMs = parseTimestamp(at, "Replan model call terminal time");
  const latencyMs = completedMs - Date.parse(existing.started_at);
  if (latencyMs < 0) {
    throw new Error(`Replan model call completed before it started: ${input.modelCallId}`);
  }
  const call = HumanoidReplanModelCallSchema.parse({
    ...existing,
    status: input.status,
    completed_at: at,
    latency_ms: latencyMs,
    slo_violated: latencyMs > existing.slo_ms
  });
  budget.model_calls[index] = call;
  return {
    budget: HumanoidReplanBudgetSchema.parse(budget),
    call: structuredClone(call)
  };
}

export function restoreHumanoidReplanModelCall(
  source: HumanoidReplanBudget,
  rawCall: HumanoidReplanModelCall
): HumanoidReplanBudget {
  const budget = structuredClone(HumanoidReplanBudgetSchema.parse(source));
  const call = HumanoidReplanModelCallSchema.parse(rawCall);
  const existing = budget.model_calls.find(
    (candidate) => candidate.model_call_id === call.model_call_id
  );
  if (existing) {
    const sameStart = existing.agent_id === call.agent_id
      && existing.recovery_tier === call.recovery_tier
      && existing.role === call.role
      && existing.started_at === call.started_at
      && existing.slo_ms === call.slo_ms;
    if (!sameStart
      || (existing.status === "started"
        && JSON.stringify(existing) !== JSON.stringify(call))) {
      throw new Error(`Replan model call restore conflict: ${call.model_call_id}`);
    }
    return budget;
  }
  budget.model_calls.push(call);
  if (call.role === "replan_decision") {
    budget.compact_replans_started += 1;
  }
  if (call.recovery_tier === "goal_re_evaluation") {
    budget.goal_reevaluation_started = true;
  }
  if (budget.recovery_started_at === null) {
    budget.recovery_started_at = call.started_at;
    budget.recovery_deadline_at = new Date(
      Date.parse(call.started_at) + budget.recovery_deadline_ms
    ).toISOString();
  }
  return HumanoidReplanBudgetSchema.parse(budget);
}

export function humanoidReplanBudgetAuthority(
  source: HumanoidReplanBudget,
  at = new Date().toISOString()
): Record<string, unknown> {
  const budget = HumanoidReplanBudgetSchema.parse(source);
  const now = parseTimestamp(at, "Replan budget authority time");
  const remainingCompact = Math.max(
    0,
    budget.compact_replan_limit - budget.compact_replans_started
  );
  const deadlineRemaining = budget.recovery_deadline_at === null
    ? null
    : Math.max(0, Date.parse(budget.recovery_deadline_at) - now);
  const deadlineExceeded = deadlineRemaining === 0
    && budget.recovery_deadline_at !== null
    && now > Date.parse(budget.recovery_deadline_at);
  const compactAvailable = !budget.goal_reevaluation_started
    && !deadlineExceeded
    && remainingCompact > 0;
  const goalAvailable = !budget.goal_reevaluation_started;
  const startedCalls = budget.model_calls.filter(
    (call) => call.status === "started"
  ).length;
  const completedCalls = budget.model_calls.filter(
    (call) => call.status === "completed"
  ).length;
  const failedCalls = budget.model_calls.filter(
    (call) => call.status === "failed"
  ).length;
  const sloViolations = budget.model_calls.filter(
    (call) => call.slo_violated === true
  ).length;
  return {
    version: budget.version,
    status: budget.goal_reevaluation_started
      ? "goal_re_evaluation_in_progress"
      : compactAvailable
        ? "compact_replan_available"
        : "goal_re_evaluation_required",
    recovery_started_at: budget.recovery_started_at,
    recovery_deadline_at: budget.recovery_deadline_at,
    recovery_deadline_remaining_ms: deadlineRemaining,
    recovery_deadline_exceeded: deadlineExceeded,
    compact_replans: {
      limit: budget.compact_replan_limit,
      used: budget.compact_replans_started,
      remaining: remainingCompact,
      available: compactAvailable
    },
    goal_re_evaluation: {
      started: budget.goal_reevaluation_started,
      available: goalAvailable,
      model_call_limit: budget.goal_reevaluation_call_limit,
      model_calls_used: budget.model_calls.filter(
        (call) => call.role === "goal_re_evaluation"
      ).length
    },
    model_calls: {
      total: budget.model_calls.length,
      in_flight: startedCalls,
      completed: completedCalls,
      failed: failedCalls,
      slo_ms: budget.model_call_slo_ms,
      slo_violations: sloViolations
    },
    recovery_layers: [
      {
        tier: "local_controller_recovery",
        budget_consumption: "none",
        authority: "controller"
      },
      {
        tier: "policy_switch",
        budget_consumption: "none",
        authority: "capability_router"
      },
      {
        tier: "compact_replan",
        budget_consumption: "one_coordinator_decision",
        authority: compactAvailable ? "available" : "unavailable"
      },
      {
        tier: "goal_re_evaluation",
        budget_consumption: "bounded_escalation",
        authority: goalAvailable ? "available" : "in_progress_or_consumed"
      }
    ]
  };
}

function parseTimestamp(value: string, label: string): number {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) throw new Error(`${label} is invalid`);
  return parsed;
}
