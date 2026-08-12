import {
  acknowledgeActionCommit,
  actionCommitPayloadSha256,
  pendingActionCommits,
  type ActionCommitOutbox,
  type PendingActionCommit
} from "../domain/action-commit-outbox.js";
import {
  createActionTransactionIdentity
} from "../domain/action-transaction-identity.js";
import type { JsonValue } from "../domain/schema.js";
import type {
  DurableRuntimeEventRecord,
  JournalName,
} from "./run-store.js";

export interface ActionCommitJournal {
  scanJournal(
    name: JournalName,
    visit: (entry: JsonValue, index: number) => void | Promise<void>
  ): Promise<void>;
  append(name: JournalName, value: JsonValue): Promise<void>;
  appendRuntimeEvents<T extends DurableRuntimeEventRecord>(
    events: readonly T[]
  ): Promise<Array<T & { cursor: string }>>;
}

interface IndexedActionEvent {
  sha256: string;
  cursor: string;
}

interface ActionCommitReconciliationIndex {
  actions: Map<string, string>;
  goalEvidence: Map<string, string>;
  experiences: Map<string, string>;
  transactionIdentities: Map<string, string>;
  events: Map<string, IndexedActionEvent>;
}

const REASSERTABLE_STATE_ANCHOR_EVENTS = new Set([
  "humanoid_physical_state_anchored",
  "humanoid_goal_state_anchored",
  "humanoid_embodied_memory_state_anchored",
  "humanoid_context_memory_state_anchored",
  "humanoid_execution_ledger_state_anchored"
]);

const reconciliationIndexes = new WeakMap<
  object,
  Promise<ActionCommitReconciliationIndex>
>();

export async function reconcileActionCommitOutbox(input: {
  store: ActionCommitJournal;
  outbox: ActionCommitOutbox;
  persist: (outbox: ActionCommitOutbox) => Promise<void>;
  publish?: (event: DurableRuntimeEventRecord & { cursor: string }) => void | Promise<void>;
}): Promise<ActionCommitOutbox> {
  const index = await reconciliationIndex(input.store);
  let outbox = input.outbox;
  for (const entry of pendingActionCommits(outbox)) {
    const event = await ensureActionCommit(input.store, index, entry);
    outbox = acknowledgeActionCommit(outbox, entry.transaction_id);
    await input.persist(outbox);
    await input.publish?.(event);
  }
  return outbox;
}

async function ensureActionCommit(
  store: ActionCommitJournal,
  index: ActionCommitReconciliationIndex,
  entry: PendingActionCommit
): Promise<DurableRuntimeEventRecord & { cursor: string }> {
  const existingAction = index.actions.get(entry.transaction_id);
  if (existingAction) {
    assertMatchingSha256(
      existingAction,
      entry.action_record_sha256,
      `Action journal transaction conflict: ${entry.transaction_id}`
    );
  } else {
    await store.append("actions", entry.action_record);
    index.actions.set(entry.transaction_id, entry.action_record_sha256);
  }

  const existingGoalEvidence = index.goalEvidence.get(entry.goal_evidence_ref);
  if (existingGoalEvidence) {
    assertMatchingSha256(
      existingGoalEvidence,
      entry.goal_evidence_sha256,
      `Goal evidence journal identity conflict: ${entry.goal_evidence_ref}`
    );
  } else {
    await store.append("goal_evidence", entry.goal_evidence_record);
    index.goalEvidence.set(entry.goal_evidence_ref, entry.goal_evidence_sha256);
  }

  if (entry.experience_ref !== undefined
    && entry.experience_record !== undefined
    && entry.experience_sha256 !== undefined) {
    const existingExperience = index.experiences.get(entry.experience_ref);
    if (existingExperience) {
      assertMatchingSha256(
        existingExperience,
        entry.experience_sha256,
        `Embodied experience journal identity conflict: ${entry.experience_ref}`
      );
    } else {
      await store.append("experiences", entry.experience_record);
      index.experiences.set(entry.experience_ref, entry.experience_sha256);
    }
  }

  const transactionIdentity = createActionTransactionIdentity(entry);
  const existingIdentity = index.transactionIdentities.get(entry.transaction_id);
  const transactionIdentitySha256 = actionCommitPayloadSha256(transactionIdentity);
  if (existingIdentity) {
    assertMatchingSha256(
      existingIdentity,
      transactionIdentitySha256,
      `Action transaction identity conflict: ${entry.transaction_id}`
    );
  } else {
    await store.append("action_identities", transactionIdentity);
    index.transactionIdentities.set(entry.transaction_id, transactionIdentitySha256);
  }

  const existingEvent = index.events.get(entry.runtime_event_id);
  if (existingEvent) {
    assertMatchingSha256(
      existingEvent.sha256,
      entry.runtime_event_sha256,
      `Action event journal identity conflict: ${entry.runtime_event_id}`
    );
    return {
      ...structuredClone(entry.runtime_event),
      cursor: existingEvent.cursor
    };
  }
  const [persisted] = await store.appendRuntimeEvents([entry.runtime_event]);
  if (!persisted) throw new Error("Action event journal did not persist its outbox record");
  index.events.set(entry.runtime_event_id, {
    sha256: entry.runtime_event_sha256,
    cursor: persisted.cursor
  });
  return persisted;
}

async function reconciliationIndex(
  store: ActionCommitJournal
): Promise<ActionCommitReconciliationIndex> {
  const key = store as object;
  const existing = reconciliationIndexes.get(key);
  if (existing) return existing;
  const building = buildReconciliationIndex(store);
  reconciliationIndexes.set(key, building);
  try {
    return await building;
  } catch (error) {
    reconciliationIndexes.delete(key);
    throw error;
  }
}

async function buildReconciliationIndex(
  store: ActionCommitJournal
): Promise<ActionCommitReconciliationIndex> {
  const index: ActionCommitReconciliationIndex = {
    actions: new Map(),
    goalEvidence: new Map(),
    experiences: new Map(),
    transactionIdentities: new Map(),
    events: new Map()
  };
  await store.scanJournal("actions", (record) => {
    const transactionId = jsonObject(record)?.transactionId;
    if (typeof transactionId !== "string") return;
    addUnique(index.actions, transactionId, actionCommitPayloadSha256(record), "action");
  });
  await store.scanJournal("goal_evidence", (record) => {
    const ref = goalEvidenceRef(record);
    if (!ref) return;
    addUnique(index.goalEvidence, ref, actionCommitPayloadSha256(record), "Goal evidence");
  });
  await store.scanJournal("experiences", (record) => {
    const ref = jsonObject(record)?.source_ref;
    if (typeof ref !== "string") return;
    addUnique(
      index.experiences,
      ref,
      actionCommitPayloadSha256(record),
      "embodied experience"
    );
  });
  await store.scanJournal("action_identities", (record) => {
    const transactionId = jsonObject(record)?.transaction_id;
    if (typeof transactionId !== "string") return;
    addUnique(
      index.transactionIdentities,
      transactionId,
      actionCommitPayloadSha256(record),
      "action transaction identity"
    );
  });
  await store.scanJournal("events", (record) => {
    const event = jsonObject(record);
    const eventId = event?.event_id;
    const cursor = event?.cursor;
    if (!event || typeof eventId !== "string" || typeof cursor !== "string") return;
    const { cursor: _cursor, ...withoutCursor } = event;
    const sha256 = actionCommitPayloadSha256(withoutCursor);
    const existing = index.events.get(eventId);
    if (existing) {
      if (existing.sha256 === sha256
        && typeof event.type === "string"
        && REASSERTABLE_STATE_ANCHOR_EVENTS.has(event.type)) {
        index.events.set(eventId, { sha256, cursor });
        return;
      }
      throw new Error(`Duplicate durable action event identity: ${eventId}`);
    }
    index.events.set(eventId, {
      sha256,
      cursor
    });
  });
  return index;
}

function addUnique(
  index: Map<string, string>,
  identity: string,
  sha256: string,
  label: string
): void {
  if (index.has(identity)) throw new Error(`Duplicate durable ${label} identity: ${identity}`);
  index.set(identity, sha256);
}

function assertMatchingSha256(actual: string, expected: string, message: string): void {
  if (actual !== expected) throw new Error(message);
}

function jsonObject(value: JsonValue): Record<string, JsonValue> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}

function goalEvidenceRef(value: JsonValue): string | undefined {
  const artifact = jsonObject(value);
  const evidence = jsonObject(artifact?.evidence ?? null);
  return typeof evidence?.ref === "string" ? evidence.ref : undefined;
}
