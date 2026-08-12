import { createHash } from "node:crypto";
import { z } from "zod";
import {
  PersistedHumanoidActionReceiptSchema
} from "../domain/humanoid-run.js";
import {
  humanoidSkillPhaseLearnedPolicyCapabilities,
  HumanoidSkillInvocationSchema
} from "../domain/humanoid-skill.js";
import { PhysicalTrajectorySummarySchema } from
  "../domain/physical-trajectory.js";
import {
  JsonValueSchema,
  Vec3Schema,
  type JsonValue
} from "../domain/schema.js";
import {
  ActiveHumanoidSkillBindingSchema,
  humanoidEmbodiedSkillIdentity,
  type ActiveHumanoidSkillBinding
} from "../harness/humanoid/skill-binding.js";
import {
  HumanoidRecoveryPolicyStateSchema
} from "../harness/humanoid/recovery-policy.js";
import {
  HumanoidEmbodiedSkillContractSchema,
  HumanoidEmbodiedSkillEventSchema,
  HumanoidEmbodiedSkillIdentitySchema,
  HumanoidEmbodiedSkillStatusSchema,
  type HumanoidEmbodiedSkillContract,
  type HumanoidEmbodiedSkillStatus
} from "../world/humanoid/embodied-skill-call.js";
import {
  HUMANOID_LEARNED_POLICY_CAPABILITIES
} from "../domain/humanoid-policy.js";
import { DensePolicyRolloutReferenceSchema } from
  "../domain/humanoid-policy-rollout.js";
import { HumanoidGroundingReceiptSchema } from
  "../domain/humanoid-grounding.js";

const PlanningActionSchema = z.enum([
  "plan_humanoid_skill",
  "plan_whole_body_motion",
  "plan_whole_body_motion_candidates",
  "plan_humanoid_navigation"
]);

const ExecutionActionSchema = z.enum([
  "execute_humanoid_skill",
  "execute_whole_body_motion",
  "execute_humanoid_navigation"
]);

const HarnessRolloutRunSchema = z.object({
  version: z.literal(1),
  run_id: z.string().trim().min(1),
  mission: z.string().trim().min(1),
  scenario_id: z.string().trim().min(1),
  created_at: z.string().datetime()
}).passthrough();

const PlanningAttemptSchema = z.object({
  transaction_id: z.string().trim().min(1),
  journal_index: z.number().int().nonnegative(),
  action: PlanningActionSchema,
  accepted: z.boolean(),
  code: z.string().trim().min(1),
  committed_at: z.string().datetime(),
  world_before_revision: z.number().int().nonnegative(),
  world_after_revision: z.number().int().nonnegative(),
  plan_id: z.string().trim().min(1).nullable(),
  plan_kind: z.enum(["motion", "navigation"]).nullable(),
  intent_sha256: z.string().regex(/^[a-f0-9]{64}$/).nullable(),
  failure_class: z.string().trim().min(1).nullable(),
  failure_reason: z.string().trim().min(1).nullable(),
  policy_contract: HumanoidEmbodiedSkillContractSchema.nullable()
}).strict();

const SkillEventRecordSchema = z.object({
  journal_index: z.number().int().nonnegative(),
  event_id: z.string().trim().min(1),
  at: z.string().datetime(),
  event: HumanoidEmbodiedSkillEventSchema
}).strict();

const TrajectoryIdentitySchema = z.object({
  protocol: z.literal("physical-trajectory-reference-v1"),
  trajectory_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  complete_from_admission: z.boolean(),
  start_frame: z.number().int().nonnegative(),
  end_frame: z.number().int().nonnegative(),
  start_world_revision: z.number().int().nonnegative(),
  end_world_revision: z.number().int().nonnegative(),
  observed_frame_count: z.number().int().positive(),
  sampled_frame_count: z.number().int().positive(),
  sample_stride: z.number().int().positive(),
  controller_usage: JsonValueSchema.nullable()
}).strict();

const ExecutionRecordSchema = z.object({
  transaction_id: z.string().trim().min(1),
  planning_transaction_id: z.string().trim().min(1),
  journal_index: z.number().int().nonnegative(),
  action: ExecutionActionSchema,
  accepted: z.boolean(),
  code: z.string().trim().min(1),
  committed_at: z.string().datetime(),
  world_before_revision: z.number().int().nonnegative(),
  world_after_revision: z.number().int().nonnegative(),
  frame_count: z.number().int().nonnegative(),
  terminal_status: HumanoidEmbodiedSkillStatusSchema.nullable(),
  grounding_receipt: HumanoidGroundingReceiptSchema.nullable(),
  controller_routing: JsonValueSchema.nullable(),
  trajectory: TrajectoryIdentitySchema.nullable()
}).strict();

const OutcomeCategorySchema = z.enum([
  "success",
  "physical_failure",
  "harness_rejection",
  "interrupted",
  "environment_changed",
  "recovery_success",
  "recovery_failure",
  "incomplete"
]);

const BaseOutcomeSchema = z.enum([
  "success",
  "physical_failure",
  "harness_rejection",
  "interrupted",
  "environment_changed",
  "incomplete"
]);

export const HarnessSkillRolloutRecordSchema = z.object({
  protocol: z.literal("hear-harness-skill-rollout-v1"),
  record_id: z.string().regex(/^skill-rollout:[^:]+:[a-f0-9]{24}$/),
  source: z.object({
    run_id: z.string().trim().min(1),
    scenario_id: z.string().trim().min(1),
    mission: z.string().trim().min(1),
    run_created_at: z.string().datetime()
  }).strict(),
  identity: HumanoidEmbodiedSkillIdentitySchema,
  binding: ActiveHumanoidSkillBindingSchema,
  semantic_command: HumanoidSkillInvocationSchema,
  declared_capabilities: z.array(
    z.enum(HUMANOID_LEARNED_POLICY_CAPABILITIES)
  ),
  policy_contract: HumanoidEmbodiedSkillContractSchema.nullable(),
  planning_attempts: z.array(PlanningAttemptSchema).min(1),
  execution: ExecutionRecordSchema.nullable(),
  skill_events: z.array(SkillEventRecordSchema),
  terminal_status: HumanoidEmbodiedSkillStatusSchema.nullable(),
  recovery_policy: HumanoidRecoveryPolicyStateSchema.nullable(),
  outcome: z.object({
    category: OutcomeCategorySchema,
    base: BaseOutcomeSchema,
    is_recovery: z.boolean(),
    code: z.string().trim().min(1),
    recoverability: HumanoidEmbodiedSkillStatusSchema.shape.recoverability.nullable()
  }).strict(),
  dense_policy_rollout: z.discriminatedUnion("available", [
    DensePolicyRolloutReferenceSchema,
    z.object({
      available: z.literal(false),
      dataset_ref: z.null(),
      frame_range: z.null(),
      reason: z.literal(
        "dense observation/action/teacher data requires the dedicated policy-frame sink"
      )
    }).strict()
  ])
}).strict().superRefine((record, context) => {
  if (record.identity.callId !== humanoidEmbodiedSkillIdentity(record.binding).callId) {
    context.addIssue({
      code: "custom",
      path: ["identity", "callId"],
      message: "Rollout identity does not match its Skill binding"
    });
  }
  if (record.execution
    && !record.planning_attempts.some((attempt) => (
      attempt.transaction_id === record.execution!.planning_transaction_id
      && attempt.accepted
    ))) {
    context.addIssue({
      code: "custom",
      path: ["execution", "planning_transaction_id"],
      message: "Execution does not reference an accepted planning attempt"
    });
  }
  if (record.terminal_status
    && record.terminal_status.callId !== record.identity.callId) {
    context.addIssue({
      code: "custom",
      path: ["terminal_status", "callId"],
      message: "Terminal status belongs to another Skill Call"
    });
  }
});

export type HarnessSkillRolloutRecord = z.infer<
  typeof HarnessSkillRolloutRecordSchema
>;

interface IndexedJournalEntry {
  index: number;
  value: unknown;
}

interface SkillCallGroup {
  binding: ActiveHumanoidSkillBinding;
  identity: ReturnType<typeof humanoidEmbodiedSkillIdentity>;
  planning: Array<z.infer<typeof PlanningAttemptSchema>>;
  planningReceipts: Array<{
    index: number;
    receipt: z.infer<typeof PersistedHumanoidActionReceiptSchema>;
  }>;
}

export function createHarnessSkillRolloutRecords(input: {
  run: unknown;
  actions: readonly IndexedJournalEntry[];
  events: readonly IndexedJournalEntry[];
}): HarnessSkillRolloutRecord[] {
  const run = HarnessRolloutRunSchema.parse(input.run);
  const actions = input.actions.map(({ index, value }) => ({
    index,
    receipt: parseJournalAction(value)
  }));
  const groups = collectSkillCallGroups(actions);
  const executions = indexExecutions(actions);
  const recoveryPolicies = indexRecoveryPolicies(actions);
  const events = indexSkillEvents(input.events);
  const records: HarnessSkillRolloutRecord[] = [];

  for (const group of [...groups.values()].sort((left, right) => (
    compareCodePoints(left.identity.callId, right.identity.callId)
  ))) {
    const execution = selectExecution(group, executions);
    const callEvents = events.get(group.identity.callId) ?? [];
    assertEventSequence(group.identity.callId, callEvents);
    const executionRecord = execution
      ? executionRecordFromReceipt(execution.index, execution.receipt)
      : null;
    const terminalStatus = terminalSkillStatus(callEvents, executionRecord);
    const recoveryPolicy = recoveryPolicies.get(group.binding.transaction_id) ?? null;
    const base = baseOutcome(group.planning, executionRecord, callEvents, terminalStatus);
    const category = recoveryPolicy
      ? base === "success"
        ? "recovery_success"
        : base === "incomplete"
          ? "incomplete"
          : "recovery_failure"
      : base;
    const code = terminalStatus?.failure?.code
      ?? executionRecord?.code
      ?? group.planning.at(-1)!.code;
    const policyContract = [...group.planning].reverse().find(
      (attempt) => attempt.accepted && attempt.policy_contract
    )?.policy_contract ?? [...group.planning].reverse().find(
      (attempt) => attempt.policy_contract
    )?.policy_contract ?? null;
    records.push(HarnessSkillRolloutRecordSchema.parse({
      protocol: "hear-harness-skill-rollout-v1",
      record_id: rolloutRecordId(run.run_id, group.identity.callId),
      source: {
        run_id: run.run_id,
        scenario_id: run.scenario_id,
        mission: run.mission,
        run_created_at: run.created_at
      },
      identity: group.identity,
      binding: group.binding,
      semantic_command: group.binding.invocation,
      declared_capabilities: humanoidSkillPhaseLearnedPolicyCapabilities(
        group.binding.invocation,
        group.binding.phase
      ),
      policy_contract: policyContract,
      planning_attempts: group.planning,
      execution: executionRecord,
      skill_events: callEvents,
      terminal_status: terminalStatus,
      recovery_policy: recoveryPolicy,
      outcome: {
        category,
        base,
        is_recovery: recoveryPolicy !== null,
        code,
        recoverability: terminalStatus?.recoverability ?? null
      },
      dense_policy_rollout: {
        available: false,
        dataset_ref: null,
        frame_range: null,
        reason:
          "dense observation/action/teacher data requires the dedicated policy-frame sink"
      }
    }));
  }
  return records;
}

function collectSkillCallGroups(
  actions: ReadonlyArray<{
    index: number;
    receipt: z.infer<typeof PersistedHumanoidActionReceiptSchema>;
  }>
): Map<string, SkillCallGroup> {
  const groups = new Map<string, SkillCallGroup>();
  for (const action of actions) {
    const planning = PlanningActionSchema.safeParse(action.receipt.action);
    if (!planning.success) continue;
    const detail = record(action.receipt.detail);
    const parsedBinding = ActiveHumanoidSkillBindingSchema.safeParse(
      detail?.skill_binding
    );
    if (!parsedBinding.success) continue;
    const binding = parsedBinding.data;
    const identity = humanoidEmbodiedSkillIdentity(binding);
    const existing = groups.get(identity.callId);
    if (existing && JSON.stringify(existing.binding) !== JSON.stringify(binding)) {
      throw new Error(`Skill Call binding changed within ${identity.callId}`);
    }
    const group = existing ?? {
      binding,
      identity,
      planning: [],
      planningReceipts: []
    };
    group.planning.push(planningAttempt(action.index, action.receipt, planning.data));
    group.planningReceipts.push(action);
    groups.set(identity.callId, group);
  }
  for (const group of groups.values()) {
    group.planning.sort((left, right) => left.journal_index - right.journal_index);
    group.planningReceipts.sort((left, right) => left.index - right.index);
  }
  return groups;
}

function planningAttempt(
  index: number,
  receipt: z.infer<typeof PersistedHumanoidActionReceiptSchema>,
  action: z.infer<typeof PlanningActionSchema>
): z.infer<typeof PlanningAttemptSchema> {
  const detail = record(receipt.detail);
  const planKind = detail?.autonomous_plan_kind === "motion"
    || detail?.autonomous_plan_kind === "navigation"
    ? detail.autonomous_plan_kind
    : action === "plan_humanoid_navigation"
      ? "navigation"
      : null;
  return PlanningAttemptSchema.parse({
    transaction_id: receipt.transactionId,
    journal_index: index,
    action,
    accepted: receipt.accepted,
    code: receipt.code,
    committed_at: receipt.committedAt,
    world_before_revision: receipt.worldBeforeRevision,
    world_after_revision: receipt.worldAfterRevision,
    plan_id: nonEmptyString(detail?.plan_id),
    plan_kind: planKind,
    intent_sha256: sha256String(detail?.intent_sha256),
    failure_class: nonEmptyString(detail?.failure_class),
    failure_reason: nonEmptyString(detail?.reason),
    policy_contract: policyContractFromPlanning(detail, planKind)
  });
}

function policyContractFromPlanning(
  detail: Record<string, unknown> | undefined,
  planKind: "motion" | "navigation" | null
): HumanoidEmbodiedSkillContract | null {
  if (!detail) return null;
  const option = record(detail.option);
  if (option?.contract !== undefined) {
    const contract = HumanoidEmbodiedSkillContractSchema.safeParse({
      protocol: "humanoid-embodied-motion-contract-v1",
      option: option.contract
    });
    if (contract.success) return contract.data;
  }
  if (planKind !== "navigation") return null;
  const waypoints = Array.isArray(detail.waypoints)
    ? detail.waypoints.flatMap((value) => {
        const parsed = Vec3Schema.safeParse(value);
        return parsed.success ? [parsed.data] : [];
      })
    : [];
  const target = waypoints.at(-1) ?? Vec3Schema.safeParse(detail.target).data;
  if (!target) return null;
  const attempts = Array.isArray(detail.attempts)
    ? detail.attempts.flatMap((value) => {
        const attempt = record(value);
        return attempt?.accepted === true ? [attempt] : [];
      })
    : [];
  const tolerance = attempts.find((attempt) => (
    typeof attempt.accepted_position_tolerance_m === "number"
  ))?.accepted_position_tolerance_m;
  const heading = navigationHeading(detail.arrival_heading);
  const contract = HumanoidEmbodiedSkillContractSchema.safeParse({
    protocol: "humanoid-embodied-navigation-contract-v1",
    target,
    positionTolerance: typeof tolerance === "number" && tolerance > 0
      ? tolerance
      : 0.06,
    heading
  });
  return contract.success ? contract.data : null;
}

function navigationHeading(value: unknown): JsonValue {
  const heading = record(value);
  if (!heading) return null;
  if (heading.type === "face_point") {
    const target = Vec3Schema.safeParse(heading.target);
    return target.success && typeof heading.tolerance_radians === "number"
      ? {
          type: "face_point",
          target: target.data,
          toleranceRadians: heading.tolerance_radians
        }
      : null;
  }
  if (heading.type === "yaw"
    && typeof heading.yaw_radians === "number"
    && typeof heading.tolerance_radians === "number") {
    return {
      type: "yaw",
      yawRadians: heading.yaw_radians,
      toleranceRadians: heading.tolerance_radians
    };
  }
  return null;
}

function indexExecutions(
  actions: ReadonlyArray<{
    index: number;
    receipt: z.infer<typeof PersistedHumanoidActionReceiptSchema>;
  }>
): Map<string, {
  index: number;
  receipt: z.infer<typeof PersistedHumanoidActionReceiptSchema>;
}> {
  const executions = new Map<string, {
    index: number;
    receipt: z.infer<typeof PersistedHumanoidActionReceiptSchema>;
  }>();
  for (const action of actions) {
    if (!ExecutionActionSchema.safeParse(action.receipt.action).success) continue;
    const planningTransactionId = nonEmptyString(
      record(action.receipt.input)?.planning_transaction_id
    );
    if (!planningTransactionId) continue;
    if (executions.has(planningTransactionId)) {
      throw new Error(
        `Multiple executions reference planning transaction ${planningTransactionId}`
      );
    }
    executions.set(planningTransactionId, action);
  }
  return executions;
}

function selectExecution(
  group: SkillCallGroup,
  executions: ReadonlyMap<string, {
    index: number;
    receipt: z.infer<typeof PersistedHumanoidActionReceiptSchema>;
  }>
) {
  const matching = group.planning.flatMap((attempt) => {
    const execution = executions.get(attempt.transaction_id);
    return execution ? [execution] : [];
  });
  if (matching.length > 1) {
    throw new Error(`Skill Call ${group.identity.callId} has multiple executions`);
  }
  return matching[0];
}

function executionRecordFromReceipt(
  index: number,
  receipt: z.infer<typeof PersistedHumanoidActionReceiptSchema>
): z.infer<typeof ExecutionRecordSchema> {
  const action = ExecutionActionSchema.parse(receipt.action);
  const planningTransactionId = nonEmptyString(
    record(receipt.input)?.planning_transaction_id
  );
  if (!planningTransactionId) {
    throw new Error(`Execution ${receipt.transactionId} has no planning transaction`);
  }
  const detail = record(receipt.detail);
  const result = record(detail?.result);
  const status = HumanoidEmbodiedSkillStatusSchema.safeParse(
    result?.skill_status ?? detail?.skill_status
  );
  const trajectory = PhysicalTrajectorySummarySchema.safeParse(
    detail?.physical_trajectory
  );
  const controllerRouting = result?.controller_routing
    ?? detail?.controller_routing
    ?? null;
  return ExecutionRecordSchema.parse({
    transaction_id: receipt.transactionId,
    planning_transaction_id: planningTransactionId,
    journal_index: index,
    action,
    accepted: receipt.accepted,
    code: receipt.code,
    committed_at: receipt.committedAt,
    world_before_revision: receipt.worldBeforeRevision,
    world_after_revision: receipt.worldAfterRevision,
    frame_count: receipt.frameCount,
    terminal_status: status.success ? status.data : null,
    grounding_receipt: HumanoidGroundingReceiptSchema.safeParse(
      detail?.grounding_receipt
    ).data ?? null,
    controller_routing: JsonValueSchema.parse(controllerRouting),
    trajectory: trajectory.success
      ? {
          protocol: "physical-trajectory-reference-v1",
          trajectory_sha256: trajectory.data.trajectory_sha256,
          complete_from_admission: trajectory.data.complete_from_admission,
          start_frame: trajectory.data.start_frame,
          end_frame: trajectory.data.end_frame,
          start_world_revision: trajectory.data.start_world_revision,
          end_world_revision: trajectory.data.end_world_revision,
          observed_frame_count: trajectory.data.observed_frame_count,
          sampled_frame_count: trajectory.data.samples.length,
          sample_stride: trajectory.data.sample_stride,
          controller_usage: trajectory.data.controller_usage
            ? JsonValueSchema.parse(trajectory.data.controller_usage)
            : null
        }
      : null
  });
}

function indexRecoveryPolicies(
  actions: ReadonlyArray<{
    index: number;
    receipt: z.infer<typeof PersistedHumanoidActionReceiptSchema>;
  }>
): Map<string, z.infer<typeof HumanoidRecoveryPolicyStateSchema>> {
  const policies = new Map<
    string,
    z.infer<typeof HumanoidRecoveryPolicyStateSchema>
  >();
  for (const { receipt } of actions) {
    if (receipt.action !== "begin_humanoid_skill" || !receipt.accepted) continue;
    const detail = record(receipt.detail);
    const binding = ActiveHumanoidSkillBindingSchema.safeParse(detail?.binding);
    const policy = HumanoidRecoveryPolicyStateSchema.safeParse(
      detail?.recovery_policy
    );
    if (!binding.success || !policy.success) continue;
    policies.set(binding.data.transaction_id, policy.data);
  }
  return policies;
}

function indexSkillEvents(
  entries: readonly IndexedJournalEntry[]
): Map<string, Array<z.infer<typeof SkillEventRecordSchema>>> {
  const events = new Map<
    string,
    Array<z.infer<typeof SkillEventRecordSchema>>
  >();
  for (const { index, value } of entries) {
    const envelope = record(value);
    if (envelope?.type !== "humanoid_skill_event") continue;
    const parsed = HumanoidEmbodiedSkillEventSchema.parse(envelope.data);
    const event = SkillEventRecordSchema.parse({
      journal_index: index,
      event_id: envelope.event_id,
      at: envelope.at,
      event: parsed
    });
    const current = events.get(parsed.status.callId) ?? [];
    current.push(event);
    events.set(parsed.status.callId, current);
  }
  for (const current of events.values()) {
    current.sort((left, right) => left.journal_index - right.journal_index);
  }
  return events;
}

function assertEventSequence(
  callId: string,
  events: readonly z.infer<typeof SkillEventRecordSchema>[]
): void {
  for (let index = 0; index < events.length; index += 1) {
    if (events[index]!.event.sequence !== index) {
      throw new Error(`Skill Call ${callId} has a non-contiguous event sequence`);
    }
  }
}

function terminalSkillStatus(
  events: readonly z.infer<typeof SkillEventRecordSchema>[],
  execution: z.infer<typeof ExecutionRecordSchema> | null
): HumanoidEmbodiedSkillStatus | null {
  const terminal = [...events].reverse().find(({ event }) => (
    event.type === "succeeded"
    || event.type === "failed"
    || event.type === "interrupted"
    || event.type === "environment_changed"
  ));
  return terminal?.event.status ?? execution?.terminal_status ?? null;
}

function baseOutcome(
  planning: readonly z.infer<typeof PlanningAttemptSchema>[],
  execution: z.infer<typeof ExecutionRecordSchema> | null,
  events: readonly z.infer<typeof SkillEventRecordSchema>[],
  terminal: HumanoidEmbodiedSkillStatus | null
): z.infer<typeof BaseOutcomeSchema> {
  const lastTerminalEvent = [...events].reverse().find(({ event }) => (
    event.type === "succeeded"
    || event.type === "failed"
    || event.type === "interrupted"
    || event.type === "environment_changed"
  ));
  if (lastTerminalEvent?.event.type === "succeeded") return "success";
  if (lastTerminalEvent?.event.type === "interrupted") return "interrupted";
  if (lastTerminalEvent?.event.type === "environment_changed") {
    return "environment_changed";
  }
  if (lastTerminalEvent?.event.type === "failed") return "physical_failure";
  if (terminal?.state === "succeeded") return "success";
  if (terminal?.state === "interrupted") return "interrupted";
  if (terminal?.state === "failed" || terminal?.state === "uncertain") {
    return "physical_failure";
  }
  if (execution?.code === "execution_grounding_rejected") {
    return "harness_rejection";
  }
  if (execution) return execution.accepted ? "success" : "physical_failure";
  return planning.some((attempt) => attempt.accepted)
    ? "incomplete"
    : "harness_rejection";
}

function parseJournalAction(
  value: unknown
): z.infer<typeof PersistedHumanoidActionReceiptSchema> {
  const action = record(value);
  if (!action) return PersistedHumanoidActionReceiptSchema.parse(value);
  const { runtime_event_id: _runtimeEventId, ...receipt } = action;
  return PersistedHumanoidActionReceiptSchema.parse(receipt);
}

function rolloutRecordId(runId: string, callId: string): string {
  const digest = createHash("sha256")
    .update("hear-harness-skill-rollout-v1\0")
    .update(runId)
    .update("\0")
    .update(callId)
    .digest("hex")
    .slice(0, 24);
  return `skill-rollout:${runId}:${digest}`;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function sha256String(value: unknown): string | null {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value)
    ? value
    : null;
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
