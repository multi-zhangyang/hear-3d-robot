import { z } from "zod";
import {
  actionCommitPayloadSha256,
  actionCommitReceipt,
  actionCommitReceiptSha256,
  type PendingActionCommit
} from "./action-commit-outbox.js";
import { JsonValueSchema, type JsonValue } from "./schema.js";
import {
  ModelDecisionRefSchema,
  type ModelDecisionRef
} from "./model-call-authority.js";
import {
  PhysicalTrajectorySummarySchema,
  type PhysicalTrajectorySummary
} from "./physical-trajectory.js";
import {
  AutonomousCycleRefSchema,
  sameAutonomousCycle,
  type AutonomousCycleRef
} from "./autonomous-cycle.js";
import {
  HumanoidGroundingReceiptSchema,
  type HumanoidGroundingReceipt
} from "./humanoid-grounding.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

const PhysicalHumanoidActionSchema = z.enum([
  "execute_humanoid_skill",
  "execute_whole_body_motion",
  "execute_humanoid_navigation"
]);

export const ExecutionGateToolCallAuthoritySchema = z.object({
  tool_call_id: z.string().trim().min(1),
  tool_name: z.literal("delegate_physics_executor"),
  arguments_sha256: z.string().regex(SHA256_PATTERN),
  normalized_arguments_sha256: z.string().regex(SHA256_PATTERN).optional(),
  deterministic_delegation: z.object({
    contract_id: z.literal("execution_gate_v1"),
    source_input: JsonValueSchema,
    action_input_sha256: z.string().regex(SHA256_PATTERN)
  }).strict()
}).strict();

export type ExecutionGateToolCallAuthority = z.infer<
  typeof ExecutionGateToolCallAuthoritySchema
>;

const ActionExecutionAdmissionSchema = z.object({
  planning_transaction_id: z.string().trim().min(1),
  plan_id: z.string().trim().min(1),
  world_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  authority_state_sha256: z.string().regex(SHA256_PATTERN),
  physical_checkpoint_sha256: z.string().regex(SHA256_PATTERN),
  decision: ModelDecisionRefSchema.optional(),
  tool_call_authority: ExecutionGateToolCallAuthoritySchema.optional(),
  grounding_receipt: HumanoidGroundingReceiptSchema.optional()
}).strict().superRefine((admission, context) => {
  if ((admission.decision === undefined)
    !== (admission.tool_call_authority === undefined)) {
    context.addIssue({
      code: "custom",
      path: ["decision"],
      message: "Physical admission authority envelope is incomplete"
    });
    return;
  }
  const decision = admission.decision;
  const authority = admission.tool_call_authority;
  if (decision && authority && (
    decision.tool_call_id !== authority.tool_call_id
      || decision.tool_arguments_sha256 !== authority.arguments_sha256
      || (decision.normalized_tool_arguments_sha256
        ?? decision.tool_arguments_sha256)
        !== authority.deterministic_delegation.action_input_sha256
  )) {
    context.addIssue({
      code: "custom",
      path: ["tool_call_authority"],
      message: "Physical admission authority is not bound to its model decision"
    });
  }
});

const ActionExecutionPlanTerminalSchema = z.object({
  kind: z.enum(["motion", "navigation"]),
  plan_id: z.string().trim().min(1),
  result_sha256: z.string().regex(SHA256_PATTERN),
  final_frame: z.number().int().nonnegative(),
  final_world_revision: z.number().int().nonnegative()
}).strict();

const ActionExecutionProgressSchema = z.object({
  committed_frame_count: z.number().int().nonnegative(),
  world_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  authority_state_sha256: z.string().regex(SHA256_PATTERN),
  physical_checkpoint_sha256: z.string().regex(SHA256_PATTERN),
  physical_trajectory: PhysicalTrajectorySummarySchema.nullable().default(null),
  completed_plan_terminals: z.array(ActionExecutionPlanTerminalSchema).default([])
}).strict();

const ActionExecutionTerminalIdentitySchema = z.object({
  receipt_sha256: z.string().regex(SHA256_PATTERN),
  action_record_sha256: z.string().regex(SHA256_PATTERN),
  runtime_event_id: z.string().trim().min(1),
  runtime_event_sha256: z.string().regex(SHA256_PATTERN),
  goal_evidence_ref: z.string().trim().min(1),
  goal_evidence_sha256: z.string().regex(SHA256_PATTERN),
  terminal_at: z.string().datetime()
}).strict();

const ActionExecutionLedgerEntrySchema = z.object({
  run_id: z.string().trim().min(1),
  transaction_id: z.string().trim().min(1),
  agent_id: z.string().trim().min(1),
  cycle: AutonomousCycleRefSchema.optional(),
  action: PhysicalHumanoidActionSchema,
  action_fingerprint_sha256: z.string().regex(SHA256_PATTERN),
  intent_sha256: z.string().regex(SHA256_PATTERN),
  admission: ActionExecutionAdmissionSchema,
  progress: ActionExecutionProgressSchema,
  status: z.enum(["admitted", "executing", "terminal"]),
  terminal: ActionExecutionTerminalIdentitySchema.nullable(),
  admitted_at: z.string().datetime(),
  updated_at: z.string().datetime()
}).strict().superRefine((entry, context) => {
  if (entry.intent_sha256 !== actionExecutionIntentSha256(entry)) {
    context.addIssue({
      code: "custom",
      path: ["intent_sha256"],
      message: "Action execution intent integrity hash does not match"
    });
  }
  if (entry.progress.world_frame < entry.admission.world_frame
    || entry.progress.world_revision < entry.admission.world_revision) {
    context.addIssue({
      code: "custom",
      path: ["progress"],
      message: "Action execution progress cannot precede admission"
    });
  }
  const trajectory = entry.progress.physical_trajectory;
  const terminalPlanIds = new Set<string>();
  let previousTerminalRevision = entry.admission.world_revision;
  for (const [index, terminal] of entry.progress.completed_plan_terminals.entries()) {
    if (terminalPlanIds.has(terminal.plan_id)
      || terminal.final_frame > entry.progress.world_frame
      || terminal.final_world_revision > entry.progress.world_revision
      || terminal.final_frame - entry.admission.world_frame
        !== terminal.final_world_revision - entry.admission.world_revision
      || terminal.final_world_revision < previousTerminalRevision) {
      context.addIssue({
        code: "custom",
        path: ["progress", "completed_plan_terminals", index],
        message: "Completed physical plan terminals are not a valid execution prefix"
      });
    }
    terminalPlanIds.add(terminal.plan_id);
    previousTerminalRevision = terminal.final_world_revision;
  }
  if (trajectory && (
    trajectory.end_frame !== entry.progress.world_frame
      || trajectory.end_world_revision !== entry.progress.world_revision
      || trajectory.start_frame < entry.admission.world_frame
      || trajectory.start_world_revision < entry.admission.world_revision
      || trajectory.complete_from_admission && (
        trajectory.start_frame !== entry.admission.world_frame
          || trajectory.start_world_revision !== entry.admission.world_revision
          || trajectory.observed_frame_count
            !== entry.progress.committed_frame_count + 1
      )
  )) {
    context.addIssue({
      code: "custom",
      path: ["progress", "physical_trajectory"],
      message: "Physical trajectory evidence is not aligned with durable execution progress"
    });
  }
  const expectedWorldRevision = entry.admission.world_revision
    + entry.progress.committed_frame_count;
  const expectedWorldFrame = entry.admission.world_frame
    + entry.progress.committed_frame_count;
  if (entry.progress.world_revision !== expectedWorldRevision
    || entry.progress.world_frame !== expectedWorldFrame) {
    context.addIssue({
      code: "custom",
      path: ["progress"],
      message: "Action execution progress is not aligned with its committed frame count"
    });
  }
  if (entry.status === "admitted") {
    if (entry.progress.committed_frame_count !== 0
      || entry.progress.world_frame !== entry.admission.world_frame
      || entry.progress.authority_state_sha256 !== entry.admission.authority_state_sha256
      || entry.progress.physical_checkpoint_sha256
        !== entry.admission.physical_checkpoint_sha256
      || entry.terminal !== null) {
      context.addIssue({
        code: "custom",
        path: ["status"],
        message: "An admitted action cannot contain physical progress or a terminal result"
      });
    }
  } else if (entry.status === "executing") {
    if (entry.progress.committed_frame_count === 0 || entry.terminal !== null) {
      context.addIssue({
        code: "custom",
        path: ["status"],
        message: "An executing action requires progress and cannot contain a terminal result"
      });
    }
  } else if (entry.terminal === null) {
    context.addIssue({
      code: "custom",
      path: ["terminal"],
      message: "A terminal action requires its durable commit identity"
    });
  }
  if (entry.updated_at < entry.admitted_at) {
    context.addIssue({
      code: "custom",
      path: ["updated_at"],
      message: "Action execution update cannot precede admission"
    });
  }
});

export const ActionExecutionLedgerSchema = z.object({
  version: z.literal(1),
  active: z.record(z.string().trim().min(1), ActionExecutionLedgerEntrySchema)
}).strict().superRefine((ledger, context) => {
  const planTransactions = new Map<string, string>();
  for (const [transactionId, entry] of Object.entries(ledger.active)) {
    if (transactionId !== entry.transaction_id) {
      context.addIssue({
        code: "custom",
        path: ["active", transactionId],
        message: "Action execution ledger key does not match its transaction identity"
      });
    }
    const existing = planTransactions.get(entry.admission.plan_id);
    if (existing && existing !== transactionId) {
      context.addIssue({
        code: "custom",
        path: ["active", transactionId, "admission", "plan_id"],
        message: "One physical plan cannot be admitted by multiple action transactions"
      });
    }
    planTransactions.set(entry.admission.plan_id, transactionId);
  }
});

export type ActionExecutionTerminalIdentity = z.infer<
  typeof ActionExecutionTerminalIdentitySchema
>;
export type ActionExecutionLedgerEntry = z.infer<
  typeof ActionExecutionLedgerEntrySchema
>;
export type ActionExecutionLedger = z.infer<typeof ActionExecutionLedgerSchema>;

export const EmptyActionExecutionLedger: ActionExecutionLedger = {
  version: 1,
  active: {}
};

export function stageActionExecutionIntent(
  persisted: ActionExecutionLedger,
  input: {
    runId: string;
    transactionId: string;
    agentId: string;
    action: z.infer<typeof PhysicalHumanoidActionSchema>;
    actionFingerprint: string;
    cycle?: AutonomousCycleRef;
    planningTransactionId: string;
    planId: string;
    worldFrame: number;
    worldRevision: number;
    authorityStateSha256: string;
    physicalCheckpointSha256: string;
    decision: ModelDecisionRef;
    toolCallAuthority: ExecutionGateToolCallAuthority;
    physicalTrajectory?: PhysicalTrajectorySummary;
    groundingReceipt?: HumanoidGroundingReceipt;
    admittedAt?: string;
  }
): ActionExecutionLedger {
  const ledger = restoreActionExecutionLedger(persisted);
  const admittedAt = input.admittedAt ?? new Date().toISOString();
  const identity = {
    run_id: input.runId.trim(),
    transaction_id: input.transactionId.trim(),
    agent_id: input.agentId.trim(),
    ...(input.cycle
      ? { cycle: AutonomousCycleRefSchema.parse(input.cycle) }
      : {}),
    action: input.action,
    action_fingerprint_sha256: actionCommitPayloadSha256(input.actionFingerprint),
    admission: {
      planning_transaction_id: input.planningTransactionId.trim(),
      plan_id: input.planId.trim(),
      world_frame: input.worldFrame,
      world_revision: input.worldRevision,
      authority_state_sha256: input.authorityStateSha256,
      physical_checkpoint_sha256: input.physicalCheckpointSha256,
      decision: ModelDecisionRefSchema.parse(input.decision),
      tool_call_authority: input.toolCallAuthority,
      ...(input.groundingReceipt
        ? {
            grounding_receipt: HumanoidGroundingReceiptSchema.parse(
              input.groundingReceipt
            )
          }
        : {})
    }
  };
  const entry = ActionExecutionLedgerEntrySchema.parse({
    ...identity,
    intent_sha256: actionExecutionIntentSha256(identity),
    progress: {
      committed_frame_count: 0,
      world_frame: input.worldFrame,
      world_revision: input.worldRevision,
      authority_state_sha256: input.authorityStateSha256,
      physical_checkpoint_sha256: input.physicalCheckpointSha256,
      physical_trajectory: input.physicalTrajectory ?? null,
      completed_plan_terminals: []
    },
    status: "admitted",
    terminal: null,
    admitted_at: admittedAt,
    updated_at: admittedAt
  });
  const existing = ledger.active[entry.transaction_id];
  if (existing) {
    if (existing.intent_sha256 !== entry.intent_sha256) {
      throw new Error(`Action execution transaction conflict: ${entry.transaction_id}`);
    }
    return ledger;
  }
  const planOwner = Object.values(ledger.active).find((candidate) => (
    candidate.admission.plan_id === entry.admission.plan_id
  ));
  if (planOwner) {
    throw new Error(
      `Physical plan ${entry.admission.plan_id} is already admitted by ${planOwner.transaction_id}`
    );
  }
  return ActionExecutionLedgerSchema.parse({
    ...ledger,
    active: { ...ledger.active, [entry.transaction_id]: entry }
  });
}

export function recordActionExecutionProgress(
  persisted: ActionExecutionLedger,
  input: {
    transactionId: string;
    committedFrameCount: number;
    worldFrame: number;
    worldRevision: number;
    authorityStateSha256: string;
    physicalCheckpointSha256: string;
    physicalTrajectory?: PhysicalTrajectorySummary;
    completedPlanTerminals?: z.infer<
      typeof ActionExecutionProgressSchema
    >["completed_plan_terminals"];
    updatedAt?: string;
  }
): ActionExecutionLedger {
  const ledger = restoreActionExecutionLedger(persisted);
  const transactionId = input.transactionId.trim();
  const entry = ledger.active[transactionId];
  if (!entry) throw new Error(`Action execution transaction is not admitted: ${transactionId}`);
  if (entry.status === "terminal") {
    throw new Error(`Action execution transaction is already terminal: ${transactionId}`);
  }
  if (input.committedFrameCount < entry.progress.committed_frame_count
    || input.worldFrame < entry.progress.world_frame
    || input.worldRevision < entry.progress.world_revision) {
    throw new Error(`Action execution progress regressed: ${transactionId}`);
  }
  const progress = ActionExecutionProgressSchema.parse({
    committed_frame_count: input.committedFrameCount,
    world_frame: input.worldFrame,
    world_revision: input.worldRevision,
    authority_state_sha256: input.authorityStateSha256,
    physical_checkpoint_sha256: input.physicalCheckpointSha256,
    physical_trajectory: input.physicalTrajectory
      ?? entry.progress.physical_trajectory,
    completed_plan_terminals: input.completedPlanTerminals
      ?? entry.progress.completed_plan_terminals
  });
  if (sameProgress(entry.progress, progress)) return ledger;
  if (progress.committed_frame_count === entry.progress.committed_frame_count) {
    if (progress.world_frame !== entry.progress.world_frame
      || progress.world_revision !== entry.progress.world_revision
      || progress.authority_state_sha256 !== entry.progress.authority_state_sha256
      || progress.committed_frame_count === 0) {
      throw new Error(`Action execution progress identity conflict: ${transactionId}`);
    }
    return replaceEntry(ledger, ActionExecutionLedgerEntrySchema.parse({
      ...entry,
      progress,
      updated_at: input.updatedAt ?? new Date().toISOString()
    }));
  }
  return replaceEntry(ledger, ActionExecutionLedgerEntrySchema.parse({
    ...entry,
    progress,
    status: "executing",
    updated_at: input.updatedAt ?? new Date().toISOString()
  }));
}

export function terminalizeActionExecution(
  persisted: ActionExecutionLedger,
  input: {
    transactionId: string;
    commit: PendingActionCommit;
    terminalAt?: string;
  }
): ActionExecutionLedger {
  const ledger = restoreActionExecutionLedger(persisted);
  const transactionId = input.transactionId.trim();
  const entry = ledger.active[transactionId];
  if (!entry) throw new Error(`Action execution transaction is not admitted: ${transactionId}`);
  if (input.commit.transaction_id !== transactionId) {
    throw new Error(`Action terminal commit transaction conflict: ${transactionId}`);
  }
  const receipt = actionCommitReceipt(input.commit.action_record);
  const receiptObject = jsonObject(receipt);
  const receiptSha256 = actionCommitReceiptSha256(input.commit.action_record);
  if (!receiptObject || !receiptSha256
    || receiptObject.agentId !== entry.agent_id
    || receiptObject.action !== entry.action
    || typeof receiptObject.fingerprint !== "string"
    || actionCommitPayloadSha256(receiptObject.fingerprint)
      !== entry.action_fingerprint_sha256
    || (entry.cycle !== undefined
      && !sameAutonomousCycle(entry.cycle, receiptObject.cycle as AutonomousCycleRef | undefined))
    || receiptObject.worldAfterRevision !== entry.progress.world_revision
    || receiptObject.frameCount !== entry.progress.committed_frame_count
    || input.commit.runtime_event.run_id !== entry.run_id) {
    throw new Error(`Action terminal receipt conflicts with admitted intent: ${transactionId}`);
  }
  const terminal = ActionExecutionTerminalIdentitySchema.parse({
    receipt_sha256: receiptSha256,
    action_record_sha256: input.commit.action_record_sha256,
    runtime_event_id: input.commit.runtime_event_id,
    runtime_event_sha256: input.commit.runtime_event_sha256,
    goal_evidence_ref: input.commit.goal_evidence_ref,
    goal_evidence_sha256: input.commit.goal_evidence_sha256,
    terminal_at: input.terminalAt ?? new Date().toISOString()
  });
  if (entry.status === "terminal") {
    if (!entry.terminal || !sameTerminal(entry.terminal, terminal)) {
      throw new Error(`Action terminal identity conflict: ${transactionId}`);
    }
    return ledger;
  }
  return replaceEntry(ledger, ActionExecutionLedgerEntrySchema.parse({
    ...entry,
    status: "terminal",
    terminal,
    updated_at: terminal.terminal_at
  }));
}

export function acknowledgeTerminalActionExecution(
  persisted: ActionExecutionLedger,
  transactionId: string,
  durableIdentity: ActionExecutionTerminalIdentity
): ActionExecutionLedger {
  const ledger = restoreActionExecutionLedger(persisted);
  const normalized = transactionId.trim();
  const entry = ledger.active[normalized];
  if (!entry) return ledger;
  const identity = ActionExecutionTerminalIdentitySchema.parse(durableIdentity);
  if (entry.status !== "terminal" || !entry.terminal
    || !sameTerminal(entry.terminal, identity)) {
    throw new Error(`Action execution is not durably terminal: ${normalized}`);
  }
  const active = { ...ledger.active };
  delete active[normalized];
  return ActionExecutionLedgerSchema.parse({ ...ledger, active });
}

export function restoreActionExecutionLedger(persisted: unknown): ActionExecutionLedger {
  return ActionExecutionLedgerSchema.parse(persisted);
}

export function activeActionExecutions(
  persisted: ActionExecutionLedger
): ActionExecutionLedgerEntry[] {
  const ledger = restoreActionExecutionLedger(persisted);
  return Object.values(ledger.active)
    .sort((left, right) => left.admitted_at.localeCompare(right.admitted_at))
    .map((entry) => structuredClone(entry));
}

export function actionExecutionFingerprintSha256(fingerprint: string): string {
  return actionCommitPayloadSha256(fingerprint);
}

function actionExecutionIntentSha256(input: {
  run_id: string;
  transaction_id: string;
  agent_id: string;
  cycle?: AutonomousCycleRef | undefined;
  action: z.infer<typeof PhysicalHumanoidActionSchema>;
  action_fingerprint_sha256: string;
  admission: z.infer<typeof ActionExecutionAdmissionSchema>;
}): string {
  return actionCommitPayloadSha256(json({
    version: 1,
    run_id: input.run_id,
    transaction_id: input.transaction_id,
    agent_id: input.agent_id,
    ...(input.cycle ? { cycle: input.cycle } : {}),
    action: input.action,
    action_fingerprint_sha256: input.action_fingerprint_sha256,
    admission: input.admission
  }));
}

function replaceEntry(
  ledger: ActionExecutionLedger,
  entry: ActionExecutionLedgerEntry
): ActionExecutionLedger {
  return ActionExecutionLedgerSchema.parse({
    ...ledger,
    active: { ...ledger.active, [entry.transaction_id]: entry }
  });
}

function sameProgress(
  left: z.infer<typeof ActionExecutionProgressSchema>,
  right: z.infer<typeof ActionExecutionProgressSchema>
): boolean {
  return actionCommitPayloadSha256(json(left)) === actionCommitPayloadSha256(json(right));
}

function sameTerminal(
  left: ActionExecutionTerminalIdentity,
  right: ActionExecutionTerminalIdentity
): boolean {
  return actionCommitPayloadSha256(json(left)) === actionCommitPayloadSha256(json(right));
}

function jsonObject(value: JsonValue | undefined): Record<string, JsonValue> | undefined {
  return value !== undefined && value !== null && typeof value === "object"
    && !Array.isArray(value)
    ? value
    : undefined;
}

function json(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) throw new Error("Action execution state is not serializable");
  return JSON.parse(serialized) as JsonValue;
}
