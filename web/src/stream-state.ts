import type {
  ContextMemoryState,
  HumanoidActionReceipt,
  HumanoidCheckerResult,
  HumanoidEmbodiedMemoryState,
  HumanoidGoalProgress,
  HumanoidWorldSnapshot,
  ProviderActivity,
  RunListItem,
  RuntimeEvent,
  TaskNode
} from "./types";

export function nextRuntimeEventCursor(
  current: string | undefined,
  event: RuntimeEvent
): string | undefined {
  if (event.cursor) return event.cursor;
  const durable = event.durable ?? event.type !== "world_frames";
  return durable ? event.event_id : current;
}

export function upsertHumanoidAction(
  actions: HumanoidActionReceipt[],
  receipt: HumanoidActionReceipt,
  limit: number
): HumanoidActionReceipt[] {
  const index = actions.findIndex((action) => action.transactionId === receipt.transactionId);
  if (index < 0) return appendRecent(actions, receipt, limit);
  const next = [...actions];
  next[index] = receipt;
  return next;
}

export function appendRecent<T>(current: T[], entry: T, limit: number): T[] {
  if (current.length < limit) return [...current, entry];
  return [...current.slice(current.length - limit + 1), entry];
}

/**
 * Merges a journal tail with its SSE suffix. New provider/framework records
 * persist the durable event id alongside the domain record, so a snapshot cut
 * taken between those two appends cannot make the same activity appear twice.
 * Legacy records have no identity and retain their original append behaviour.
 */
export function upsertRuntimeJournalEntry<T>(current: T[], entry: T, limit: number): T[] {
  const id = runtimeJournalEntryId(entry);
  if (id === null) return appendRecent(current, entry, limit);
  const index = current.findIndex((candidate) => runtimeJournalEntryId(candidate) === id);
  if (index < 0) return appendRecent(current, entry, limit);
  if (current[index] === entry) return current;
  const next = [...current];
  next[index] = entry;
  return next;
}

function runtimeJournalEntryId(value: unknown): string | null {
  const record = asRecord(value);
  return typeof record?.runtime_event_id === "string" && record.runtime_event_id.length > 0
    ? record.runtime_event_id
    : null;
}

export function updateRunListStatus(
  runs: RunListItem[],
  runId: string,
  status: RunListItem["status"],
  error: string | null,
  updatedAt: string
): RunListItem[] {
  return runs.map((run) => run.run_id === runId
    ? { ...run, status, error, updated_at: updatedAt }
    : run);
}

export function failOpenNodes(
  nodes: Record<string, TaskNode>,
  error: string,
  updatedAt: string
): Record<string, TaskNode> {
  return Object.fromEntries(Object.entries(nodes).map(([id, node]) => {
    if (node.status !== "ready" && node.status !== "active" && node.status !== "waiting") {
      return [id, node];
    }
    return [id, {
      ...node,
      status: "failed" as const,
      last_result: { error },
      updated_at: updatedAt
    }];
  }));
}

export function latestProviderActivity(entries: unknown[]): ProviderActivity | null {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const activity = providerActivityFrom(entries[index]);
    if (activity) return activity;
  }
  return null;
}

export function providerActivityFrom(value: unknown): ProviderActivity | null {
  const record = asRecord(value);
  if (!record || typeof record.status !== "string") return null;
  return {
    status: record.status,
    at: typeof record.at === "string" ? record.at : null,
    source: typeof record.source === "string" ? record.source : null
  };
}

export function humanoidWorldSnapshotsFrom(value: unknown): HumanoidWorldSnapshot[] {
  if (isHumanoidWorldSnapshot(value)) return [value];
  const record = asRecord(value);
  if (!record) return [];
  const candidates: HumanoidWorldSnapshot[] = [];
  for (const key of ["world", "snapshot"] as const) {
    if (isHumanoidWorldSnapshot(record[key])) candidates.push(record[key]);
  }
  if (Array.isArray(record.frames)) {
    for (const frame of record.frames) {
      if (isHumanoidWorldSnapshot(frame)) candidates.push(frame);
    }
  }
  return candidates;
}

export function taskNodesFrom(value: unknown): Record<string, TaskNode> | null {
  const record = asRecord(value);
  if (!record) return null;
  for (const node of Object.values(record)) {
    const candidate = asRecord(node);
    if (!candidate || typeof candidate.id !== "string" || typeof candidate.name !== "string"
      || typeof candidate.status !== "string" || !Array.isArray(candidate.child_ids)) return null;
  }
  return record as unknown as Record<string, TaskNode>;
}

export function humanoidActionReceiptFrom(value: unknown): HumanoidActionReceipt | null {
  const record = asRecord(value);
  if (!record || typeof record.transactionId !== "string"
    || typeof record.agentId !== "string" || typeof record.action !== "string"
    || typeof record.fingerprint !== "string" || typeof record.accepted !== "boolean"
    || typeof record.code !== "string" || !Array.isArray(record.channels)) return null;
  return record as unknown as HumanoidActionReceipt;
}

export function humanoidCheckerFrom(value: unknown): HumanoidCheckerResult | null {
  const record = asRecord(value);
  if (!record || typeof record.success !== "boolean"
    || typeof record.worldFrame !== "number"
    || typeof record.worldRevision !== "number"
    || !Array.isArray(record.checks)) return null;
  return record as unknown as HumanoidCheckerResult;
}

export function humanoidGoalProgressFrom(value: unknown): HumanoidGoalProgress | null {
  const record = asRecord(value);
  if (!record || record.version !== 1
    || typeof record.goal_sha256 !== "string"
    || !Number.isInteger(record.predicate_count)
    || !Number.isInteger(record.last_world_frame)
    || !Number.isInteger(record.last_world_revision)
    || !Array.isArray(record.predicate_streaks)
    || record.predicate_streaks.length !== record.predicate_count
    || !record.predicate_streaks.every((streak) => (
      typeof streak === "number" && Number.isInteger(streak) && streak >= 0
    ))) return null;
  return record as unknown as HumanoidGoalProgress;
}

export function contextMemoryFrom(value: unknown): ContextMemoryState | null {
  const record = asRecord(value);
  if (!record || record.version !== 1
    || typeof record.context_window_tokens !== "number"
    || typeof record.compact_trigger_tokens !== "number"
    || typeof record.active_estimated_tokens !== "number"
    || typeof record.total_compactions !== "number"
    || asRecord(record.scopes) === null) return null;
  return record as unknown as ContextMemoryState;
}

export function embodiedMemoryFrom(value: unknown): HumanoidEmbodiedMemoryState | null {
  const record = asRecord(value);
  if (!record || record.version !== 1
    || typeof record.total_episodes !== "number"
    || typeof record.pruned_episodes !== "number"
    || !Array.isArray(record.recent_episodes)) return null;
  for (const episode of record.recent_episodes) {
    const candidate = asRecord(episode);
    if (!candidate || typeof candidate.sequence !== "number"
      || typeof candidate.transaction_id !== "string"
      || typeof candidate.model_summary !== "string"
      || typeof candidate.recorded_at !== "string") return null;
  }
  return record as unknown as HumanoidEmbodiedMemoryState;
}

function isHumanoidWorldSnapshot(value: unknown): value is HumanoidWorldSnapshot {
  const record = asRecord(value);
  const robot = asRecord(record?.robot);
  return record !== null
    && typeof record.frame === "number"
    && typeof record.worldRevision === "number"
    && robot !== null
    && typeof robot.simulatedTime === "number"
    && asRecord(robot.rootPosition) !== null
    && asRecord(robot.links) !== null
    && asRecord(robot.objects) !== null
    && asRecord(record.navigation) !== null;
}

export function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}
