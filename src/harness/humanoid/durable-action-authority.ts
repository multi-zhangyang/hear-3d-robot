import {
  actionCommitPayloadSha256,
  actionCommitReceiptSha256
} from "../../domain/action-commit-outbox.js";
import {
  actionTransactionFingerprintSha256,
  type ActionTransactionIdentity
} from "../../domain/action-transaction-identity.js";
import {
  PersistedHumanoidActionReceiptSchema,
  humanoidActionReceiptEntriesInCommitOrder
} from "../../domain/humanoid-run.js";
import type { JsonValue } from "../../domain/schema.js";
import type { RunStore } from "../../persistence/run-store.js";
import {
  HumanoidActionRuntimeStateSchema,
  humanoidActionFingerprint,
  type HumanoidActionToolCallAuthority,
  type HumanoidActionReceipt
} from "./runtime.js";

interface DurableActionCommitProof {
  receipt: HumanoidActionReceipt;
  actionRecord: JsonValue;
  runtimeEvent: JsonValue;
  goalEvidence: JsonValue;
}

/**
 * Rebuilds the hot checkpoint action window exclusively from append-only
 * commit proofs. A checkpoint receipt is a cache, never authority by itself.
 */
export async function verifyDurableHumanoidActionWindow(input: {
  store: RunStore;
  runId: string;
  receipts: Readonly<Record<string, HumanoidActionReceipt>>;
  identities: ReadonlyMap<string, ActionTransactionIdentity>;
  assertDecisionAuthority: (
    receipt: HumanoidActionReceipt,
    toolAuthority: HumanoidActionToolCallAuthority | undefined
  ) => void;
}): Promise<Map<string, DurableActionCommitProof>> {
  const entries = humanoidActionReceiptEntriesInCommitOrder(input.receipts);
  if (entries.length === 0) return new Map();

  const requiredTransactions = new Set(entries.map(([transactionId]) => transactionId));
  const actionRecords = await loadUniqueJournalRecords({
    store: input.store,
    journal: "actions",
    requiredIds: requiredTransactions,
    identity: (record) => stringField(record, "transactionId"),
    label: "action transaction"
  });
  const requiredEventIds = new Set<string>();
  for (const [transactionId] of entries) {
    const identity = input.identities.get(transactionId);
    if (!identity) {
      throw new Error(`Committed action has no durable transaction identity: ${transactionId}`);
    }
    requiredEventIds.add(identity.runtime_event_id);
  }
  const runtimeEvents = await loadUniqueJournalRecords({
    store: input.store,
    journal: "events",
    requiredIds: requiredEventIds,
    identity: (record) => stringField(record, "event_id"),
    label: "action runtime event"
  });
  const requiredEvidenceRefs = new Set(
    [...input.identities.values()]
      .filter((identity) => requiredTransactions.has(identity.transaction_id))
      .map((identity) => identity.goal_evidence_ref)
  );
  const goalEvidence = await loadUniqueJournalRecords({
    store: input.store,
    journal: "goal_evidence",
    requiredIds: requiredEvidenceRefs,
    identity: goalEvidenceRef,
    label: "action Goal evidence"
  });

  const verified = new Map<string, DurableActionCommitProof>();
  for (const [transactionId, checkpointReceipt] of entries) {
    const identity = input.identities.get(transactionId)!;
    const actionRecord = actionRecords.get(transactionId);
    if (!actionRecord) {
      throw new Error(`Committed action journal record is missing: ${transactionId}`);
    }
    const durableReceipt = verifyActionRecord(
      actionRecord,
      checkpointReceipt,
      identity
    );
    const runtimeEvent = runtimeEvents.get(identity.runtime_event_id);
    if (!runtimeEvent) {
      throw new Error(`Committed action runtime event is missing: ${transactionId}`);
    }
    verifyRuntimeEvent(runtimeEvent, actionRecord, identity, input.runId);
    const evidence = goalEvidence.get(identity.goal_evidence_ref);
    if (!evidence) {
      throw new Error(`Committed action Goal evidence is missing: ${transactionId}`);
    }
    verifyGoalEvidence(evidence, durableReceipt, identity);
    const eventData = jsonObject(jsonObject(runtimeEvent)?.data ?? null);
    const toolAuthority = eventData?.action_tool_authority === undefined
      ? undefined
      : humanoidActionToolCallAuthority(eventData.action_tool_authority);
    input.assertDecisionAuthority(durableReceipt, toolAuthority);
    verified.set(transactionId, {
      receipt: structuredClone(durableReceipt),
      actionRecord: structuredClone(actionRecord),
      runtimeEvent: structuredClone(runtimeEvent),
      goalEvidence: structuredClone(evidence)
    });
  }
  return verified;
}

function humanoidActionToolCallAuthority(
  value: JsonValue
): HumanoidActionToolCallAuthority {
  const authority = jsonObject(value);
  const delegation = jsonObject(authority?.deterministic_delegation ?? null);
  if (!authority
    || typeof authority.tool_call_id !== "string"
    || typeof authority.tool_name !== "string"
    || typeof authority.arguments_sha256 !== "string"
    || (authority.normalized_arguments_sha256 !== undefined
      && typeof authority.normalized_arguments_sha256 !== "string")
    || (authority.deterministic_delegation !== undefined && (
      !delegation
        || (delegation.contract_id !== "grounding_monitor_v1"
          && delegation.contract_id !== "execution_gate_v1")
        || delegation.source_input === undefined
        || typeof delegation.action_input_sha256 !== "string"
    ))) {
    throw new Error("Durable humanoid action tool authority is malformed");
  }
  return {
    tool_call_id: authority.tool_call_id,
    tool_name: authority.tool_name,
    arguments_sha256: authority.arguments_sha256,
    ...(typeof authority.normalized_arguments_sha256 === "string"
      ? { normalized_arguments_sha256: authority.normalized_arguments_sha256 }
      : {}),
    ...(delegation
      ? {
          deterministic_delegation: {
            contract_id: delegation.contract_id as
              "grounding_monitor_v1" | "execution_gate_v1",
            source_input: structuredClone(delegation.source_input!),
            action_input_sha256: delegation.action_input_sha256 as string
          }
        }
      : {})
  };
}

/**
 * The latest committed action event is the append-only anchor for the mutable
 * Skill DAG/binding/recovery cache stored in the checkpoint.
 */
export function verifyDurableHumanoidActionRuntimeState(input: {
  receipts: Readonly<Record<string, HumanoidActionReceipt>>;
  proofs: ReadonlyMap<string, DurableActionCommitProof>;
  checkpointState: JsonValue | null;
}): void {
  const parsedState = input.checkpointState === null
    ? null
    : HumanoidActionRuntimeStateSchema.parse(input.checkpointState);
  const latestEntry = humanoidActionReceiptEntriesInCommitOrder(input.receipts).at(-1);
  if (!latestEntry) {
    if (parsedState !== null && !actionRuntimeStateIsEmpty(parsedState)) {
      throw new Error("Humanoid action runtime state has no durable action authority");
    }
    return;
  }
  const proof = input.proofs.get(latestEntry[0]);
  if (!proof) {
    throw new Error(`Latest committed action proof is missing: ${latestEntry[0]}`);
  }
  const event = jsonObject(proof.runtimeEvent);
  const data = jsonObject(event?.data ?? null);
  const durableState = data?.action_runtime_state;
  if (durableState === undefined) {
    if (parsedState !== null && !actionRuntimeStateIsEmpty(parsedState)) {
      throw new Error(
        `Humanoid action runtime state has no durable event authority: ${latestEntry[0]}`
      );
    }
    return;
  }
  if (parsedState === null
    || actionCommitPayloadSha256(durableState)
      !== actionCommitPayloadSha256(json(parsedState))) {
    throw new Error(
      `Humanoid action runtime state conflicts with durable event: ${latestEntry[0]}`
    );
  }
}

function verifyGoalEvidence(
  evidence: JsonValue,
  receipt: HumanoidActionReceipt,
  identity: ActionTransactionIdentity
): void {
  const artifact = jsonObject(evidence);
  const evidenceIdentity = jsonObject(artifact?.evidence ?? null);
  const payload = jsonObject(artifact?.payload ?? null);
  if (evidenceIdentity?.ref !== identity.goal_evidence_ref
    || evidenceIdentity.kind !== "action_receipt"
    || payload?.transaction_id !== identity.transaction_id
    || payload.receipt === undefined
    || actionCommitPayloadSha256(evidence) !== identity.goal_evidence_sha256
    || actionCommitPayloadSha256(payload.receipt)
      !== actionCommitPayloadSha256(json(receipt))) {
    throw new Error(
      `Committed action Goal evidence conflicts with durable authority: ${identity.transaction_id}`
    );
  }
}

function verifyActionRecord(
  actionRecord: JsonValue,
  checkpointReceipt: HumanoidActionReceipt,
  identity: ActionTransactionIdentity
): HumanoidActionReceipt {
  const record = jsonObject(actionRecord);
  if (!record || record.runtime_event_id !== identity.runtime_event_id
    || actionCommitPayloadSha256(actionRecord) !== identity.action_record_sha256) {
    throw new Error(
      `Committed action journal identity conflict: ${identity.transaction_id}`
    );
  }
  const { runtime_event_id: _runtimeEventId, ...rawReceipt } = record;
  const durableReceipt = PersistedHumanoidActionReceiptSchema.parse(rawReceipt);
  const canonicalReceiptSha256 = actionCommitReceiptSha256(actionRecord);
  const checkpointSha256 = actionCommitPayloadSha256(json(checkpointReceipt));
  const fingerprint = humanoidActionFingerprint(
    durableReceipt.action,
    durableReceipt.agentId,
    durableReceipt.input
  );
  if (!canonicalReceiptSha256
    || canonicalReceiptSha256 !== identity.receipt_sha256
    || checkpointSha256 !== identity.receipt_sha256
    || durableReceipt.transactionId !== identity.transaction_id
    || durableReceipt.agentId !== identity.agent_id
    || durableReceipt.action !== identity.action
    || durableReceipt.committedAt !== identity.committed_at
    || identity.goal_evidence_ref !== `action:${identity.transaction_id}`
    || actionTransactionFingerprintSha256(fingerprint)
      !== identity.action_fingerprint_sha256
    || durableReceipt.fingerprint !== fingerprint) {
    throw new Error(
      `Committed checkpoint receipt conflicts with durable authority: ${identity.transaction_id}`
    );
  }
  return durableReceipt;
}

function verifyRuntimeEvent(
  runtimeEvent: JsonValue,
  actionRecord: JsonValue,
  identity: ActionTransactionIdentity,
  runId: string
): void {
  const event = jsonObject(runtimeEvent);
  if (!event) {
    throw new Error(`Committed action runtime event is malformed: ${identity.transaction_id}`);
  }
  const { cursor: _cursor, ...withoutCursor } = event;
  const data = jsonObject(event.data ?? null);
  if (event.event_id !== identity.runtime_event_id
    || event.run_id !== runId
    || event.type !== "humanoid_action_committed"
    || actionCommitPayloadSha256(json(withoutCursor)) !== identity.runtime_event_sha256
    || data?.receipt === undefined
    || actionCommitPayloadSha256(data.receipt)
      !== actionCommitPayloadSha256(actionRecord)) {
    throw new Error(
      `Committed action runtime event conflicts with durable authority: ${identity.transaction_id}`
    );
  }
}

async function loadUniqueJournalRecords(input: {
  store: RunStore;
  journal: "actions" | "events" | "goal_evidence";
  requiredIds: ReadonlySet<string>;
  identity: (record: JsonValue) => string | undefined;
  label: string;
}): Promise<Map<string, JsonValue>> {
  const records = new Map<string, JsonValue>();
  await input.store.scanJournal(input.journal, (record) => {
    const id = input.identity(record);
    if (!id || !input.requiredIds.has(id)) return;
    if (records.has(id)) {
      throw new Error(`Duplicate durable ${input.label} identity: ${id}`);
    }
    records.set(id, structuredClone(record));
  });
  return records;
}

function goalEvidenceRef(value: JsonValue): string | undefined {
  const artifact = jsonObject(value);
  const evidence = jsonObject(artifact?.evidence ?? null);
  return typeof evidence?.ref === "string" ? evidence.ref : undefined;
}

function stringField(value: JsonValue, field: string): string | undefined {
  const record = jsonObject(value);
  const candidate = record?.[field];
  return typeof candidate === "string" ? candidate : undefined;
}

function jsonObject(value: JsonValue): Record<string, JsonValue> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}

function actionRuntimeStateIsEmpty(
  state: ReturnType<typeof HumanoidActionRuntimeStateSchema.parse>
): boolean {
  return state.latest_physical_execution_revision === 0
    && state.skill_plans.length === 0
    && Object.keys(state.active_skill_plan_transactions).length === 0
    && state.active_skills.length === 0
    && state.planning_skill_bindings.length === 0
    && state.recovery_policies.length === 0
    && state.navigation_transit_clearance_requirements.length === 0;
}

function json(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) {
    throw new Error("Durable action authority value is not serializable");
  }
  return JSON.parse(serialized) as JsonValue;
}
