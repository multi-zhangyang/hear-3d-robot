import {
  ModelCallLifecycleRecordSchema,
  rebuildModelCallAuthorities,
  type ModelCallLifecycleRecord
} from "../../domain/model-call-authority.js";
import type { HumanoidRunCheckpoint } from "../../domain/humanoid-run.js";
import type { RunStore } from "../../persistence/run-store.js";
import {
  GoalEvidenceArtifactSchema,
  type GoalEvidenceArtifact
} from "./goal-evidence.js";

const JOURNAL_SCAN_PAGE = 256;
const RECENT_MODEL_LIFECYCLE_LIMIT = 256;

export function requiredGoalEvidenceRefs(
  checkpoint: HumanoidRunCheckpoint
): Set<string> {
  const refs = new Set(Object.keys(checkpoint.goal_dag.evidence));
  for (const pending of Object.values(checkpoint.action_commit_outbox.pending)) {
    refs.add(pending.goal_evidence_ref);
  }
  for (const execution of Object.values(checkpoint.action_execution_ledger.active)) {
    if (execution.terminal) refs.add(execution.terminal.goal_evidence_ref);
  }
  return refs;
}

export function optionalGoalEvidenceRefs(
  checkpoint: HumanoidRunCheckpoint
): Set<string> {
  return new Set(Object.keys(checkpoint.committed_actions).map(
    (transactionId) => `action:${transactionId}`
  ));
}

export function requiredModelCallIds(
  checkpoint: HumanoidRunCheckpoint
): Set<string> {
  const ids = new Set<string>();
  for (const candidate of Object.values(checkpoint.goal_dag.candidates)) {
    ids.add(candidate.source.model_call_id);
  }
  for (const epoch of checkpoint.goal_dag.epochs) {
    ids.add(epoch.selected_by.model_call_id);
    if (epoch.retired_by) ids.add(epoch.retired_by.model_call_id);
  }
  for (const execution of Object.values(checkpoint.action_execution_ledger.active)) {
    if (execution.admission.decision) {
      ids.add(execution.admission.decision.model_call_id);
    }
  }
  for (const receipt of Object.values(checkpoint.committed_actions)) {
    if (receipt.decision) ids.add(receipt.decision.model_call_id);
  }
  return ids;
}

export function optionalModelCallIds(
  checkpoint: HumanoidRunCheckpoint
): Set<string> {
  return new Set([
    ...(checkpoint.active_cycle?.replan_budget.model_calls.map(
      (call) => call.model_call_id
    ) ?? [])
  ]);
}

export async function loadGoalEvidenceWorkingSet(
  store: RunStore,
  requiredRefs: ReadonlySet<string>,
  optionalRefs: ReadonlySet<string> = new Set()
): Promise<Map<string, GoalEvidenceArtifact>> {
  const retainedRefs = new Set([...requiredRefs, ...optionalRefs]);
  const evidence = new Map<string, GoalEvidenceArtifact>();
  const tail = await store.readJournalTail("goal_evidence", 1);
  for (let from = 0; from < tail.total;) {
    const page = await store.readJournalPage(
      "goal_evidence",
      from,
      Math.min(JOURNAL_SCAN_PAGE, tail.total - from)
    );
    if (page.entries.length === 0) {
      throw new Error(`Goal evidence journal stopped before offset ${tail.total}`);
    }
    for (const rawArtifact of page.entries) {
      const artifact = GoalEvidenceArtifactSchema.parse(rawArtifact);
      const ref = artifact.evidence.ref;
      if (!retainedRefs.has(ref)) continue;
      const existing = evidence.get(ref);
      if (existing && JSON.stringify(existing) !== JSON.stringify(artifact)) {
        throw new Error(`Goal evidence reference was rebound: ${ref}`);
      }
      if (!existing) evidence.set(ref, artifact);
    }
    from += page.entries.length;
  }
  const missing = [...requiredRefs].filter((ref) => !evidence.has(ref));
  if (missing.length > 0) {
    throw new Error(`Physical evidence is unavailable: ${missing[0]}`);
  }
  return evidence;
}

export async function loadModelAuthorityWorkingSet(
  store: RunStore,
  requiredIds: ReadonlySet<string>,
  optionalIds: ReadonlySet<string> = new Set()
): Promise<ModelCallLifecycleRecord[]> {
  const retainedIds = new Set([...requiredIds, ...optionalIds]);
  const pending = new Map<string, IndexedModelRecord>();
  const retained = new Map<string, IndexedModelRecord[]>();
  const recentTerminalIds: string[] = [];
  const tail = await store.readJournalTail("model_calls", 1);
  let journalIndex = 0;
  for (let from = 0; from < tail.total;) {
    const page = await store.readJournalPage(
      "model_calls",
      from,
      Math.min(JOURNAL_SCAN_PAGE, tail.total - from)
    );
    if (page.entries.length === 0) {
      throw new Error(`Model authority journal stopped before offset ${tail.total}`);
    }
    for (const rawRecord of page.entries) {
      const record = ModelCallLifecycleRecordSchema.parse(rawRecord);
      const indexed = { index: journalIndex, record };
      journalIndex += 1;
      if (record.lifecycle === "started") {
        if (pending.has(record.model_call_id)) {
          throw new Error(`Duplicate pending model call start: ${record.model_call_id}`);
        }
        pending.set(record.model_call_id, indexed);
        continue;
      }
      const started = pending.get(record.model_call_id);
      if (!started) {
        throw new Error(`Model call terminal record has no matching start: ${record.model_call_id}`);
      }
      rebuildModelCallAuthorities([started.record, record]);
      pending.delete(record.model_call_id);
      retained.set(record.model_call_id, [started, indexed]);
      recentTerminalIds.push(record.model_call_id);
      if (recentTerminalIds.length > RECENT_MODEL_LIFECYCLE_LIMIT) {
        const expired = recentTerminalIds.shift()!;
        if (!retainedIds.has(expired)) retained.delete(expired);
      }
    }
    from += page.entries.length;
  }
  const retainedPendingIds = new Set([
    ...[...pending.keys()].slice(-RECENT_MODEL_LIFECYCLE_LIMIT),
    ...[...requiredIds].filter((modelCallId) => pending.has(modelCallId)),
    ...[...optionalIds].filter((modelCallId) => pending.has(modelCallId))
  ]);
  for (const modelCallId of retainedPendingIds) {
    const started = pending.get(modelCallId)!;
    retained.set(modelCallId, [started]);
  }
  const missing = [...requiredIds].filter((modelCallId) => !retained.has(modelCallId));
  if (missing.length > 0) {
    throw new Error(`Required model authority is unavailable: ${missing[0]}`);
  }
  return [...retained.values()]
    .flat()
    .sort((left, right) => left.index - right.index)
    .map(({ record }) => record);
}

interface IndexedModelRecord {
  index: number;
  record: ModelCallLifecycleRecord;
}
