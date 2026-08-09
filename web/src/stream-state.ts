import type {
  ActiveAutonomousCycle,
  ContextMemoryState,
  GoalDAG,
  HumanoidActionReceipt,
  HumanoidCheckerResult,
  HumanoidEmbodiedMemoryState,
  HumanoidGoalProgress,
  HumanoidWorldSnapshot,
  ModelUsageState,
  ProviderActivity,
  RunListItem,
  RuntimeEvent,
  TaskNode
} from "./types";

export function activeAutonomousCycleFrom(value: unknown): ActiveAutonomousCycle | null {
  const record = asRecord(value);
  if (!record
    || typeof record.cycle_id !== "string"
    || !positiveInteger(record.cycle_index)
    || typeof record.goal_epoch_id !== "string"
    || !nonnegativeInteger(record.started_world_frame)
    || !nonnegativeInteger(record.started_world_revision)
    || typeof record.started_at !== "string") return null;
  return record as unknown as ActiveAutonomousCycle;
}

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
    source: typeof record.source === "string" ? record.source : null,
    ...(typeof record.agent_id === "string" ? { agentId: record.agent_id } : {})
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

export function goalDAGFrom(value: unknown): GoalDAG | null {
  const record = asRecord(value);
  const candidates = asRecord(record?.candidates);
  const candidateSequences = asRecord(record?.candidate_sequences);
  const evidence = asRecord(record?.evidence);
  const archive = asRecord(record?.archive);
  if (!record || record.version !== 2
    || (record.status !== "awaiting_model_selection" && record.status !== "active")
    || candidates === null
    || !Object.values(candidates).every((candidate) => asRecord(candidate) !== null)
    || candidateSequences === null
    || !Object.values(candidateSequences).every(positiveInteger)
    || !positiveInteger(record.next_candidate_sequence)
    || !Array.isArray(record.epochs)
    || (record.current_epoch_id !== null && typeof record.current_epoch_id !== "string")
    || !nonnegativeInteger(record.next_epoch_index)
    || evidence === null
    || archive === null
    || !nonnegativeInteger(archive.record_count)
    || (archive.last_record_sha256 !== null
      && typeof archive.last_record_sha256 !== "string")
    || (archive.last_epoch_id !== null && typeof archive.last_epoch_id !== "string")
    || !Array.isArray(archive.retained_candidate_ids)
    || !archive.retained_candidate_ids.every((candidateId) => typeof candidateId === "string")
    || !goalHistorySummary(archive.summary)
    || !goalHistoryArchiveSummaryMatches(archive)
    || typeof record.state_sha256 !== "string") return null;
  return record as unknown as GoalDAG;
}

function goalHistorySummary(value: unknown): boolean {
  if (value === null) return true;
  const summary = asRecord(value);
  const outcomes = asRecord(summary?.outcomes);
  const selected = asRecord(outcomes?.selected);
  if (!summary || summary.version !== 1
    || !nonnegativeInteger(summary.archived_epoch_count)
    || (summary.last_record_sha256 !== null
      && typeof summary.last_record_sha256 !== "string")
    || !nonnegativeInteger(summary.records_without_alternate_history)
    || !outcomes
    || !selected
    || !nonnegativeInteger(selected.total)
    || !nonnegativeInteger(selected.completed)
    || !nonnegativeInteger(selected.blocked)
    || !nonnegativeInteger(selected.abandoned)
    || !nonnegativeInteger(selected.superseded)
    || !nonnegativeInteger(selected.expired)
    || !nonnegativeInteger(outcomes.not_selected)
    || !Array.isArray(outcomes.predicate_outcomes)
    || !Array.isArray(outcomes.entity_outcomes)) return false;
  return selected.total === selected.completed + selected.blocked
      + selected.abandoned + selected.superseded + selected.expired
    && selected.total === summary.archived_epoch_count
    && summary.records_without_alternate_history <= summary.archived_epoch_count;
}

function goalHistoryArchiveSummaryMatches(archive: Record<string, unknown>): boolean {
  if (archive.summary === null) return true;
  const summary = asRecord(archive.summary);
  return summary !== null
    && summary.archived_epoch_count === archive.record_count
    && summary.last_record_sha256 === archive.last_record_sha256;
}

export function contextMemoryFrom(value: unknown): ContextMemoryState | null {
  const record = asRecord(value);
  const scopes = asRecord(record?.scopes);
  if (!record || record.version !== 1
    || typeof record.context_window_tokens !== "number"
    || typeof record.compact_trigger_tokens !== "number"
    || typeof record.active_estimated_tokens !== "number"
    || typeof record.total_compactions !== "number"
    || scopes === null) return null;
  for (const value of Object.values(scopes)) {
    const scope = asRecord(value);
    if (!scope || !validContextScopeBudget(scope)) return null;
  }
  return record as unknown as ContextMemoryState;
}

export function modelUsageFrom(value: unknown): ModelUsageState | null {
  const record = asRecord(value);
  const total = asRecord(record?.total);
  const byAgent = asRecord(record?.by_agent);
  if (!record || record.version !== 1 || !total || !byAgent
    || !validModelUsageTotals(total)
    || !Object.values(byAgent).every((entry) => {
      const totals = asRecord(entry);
      return totals !== null && validModelUsageTotals(totals);
    })) return null;
  return record as unknown as ModelUsageState;
}

function validModelUsageTotals(value: Record<string, unknown>): boolean {
  return [
    value.requests,
    value.reported_requests,
    value.input_tokens,
    value.output_tokens,
    value.total_tokens,
    value.cached_input_tokens,
    value.reasoning_tokens
  ].every(nonnegativeInteger)
    && (value.reported_requests as number) <= (value.requests as number);
}

function validContextScopeBudget(scope: Record<string, unknown>): boolean {
  const values = [
    scope.context_window_tokens,
    scope.compact_trigger_tokens,
    scope.compact_recent_model_turns,
    scope.compact_max_output_tokens
  ];
  const present = values.filter((value) => value !== undefined);
  if (present.length === 0) return true;
  return present.length === values.length
    && positiveInteger(scope.context_window_tokens)
    && positiveInteger(scope.compact_trigger_tokens)
    && nonnegativeInteger(scope.compact_recent_model_turns)
    && positiveInteger(scope.compact_max_output_tokens);
}

function positiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

function nonnegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

export function embodiedMemoryFrom(value: unknown): HumanoidEmbodiedMemoryState | null {
  const record = asRecord(value);
  if (!record || (record.version !== 1 && record.version !== 2)
    || !nonnegativeInteger(record.total_episodes)
    || !nonnegativeInteger(record.pruned_episodes)
    || !Array.isArray(record.recent_episodes)) return null;
  for (const episode of record.recent_episodes) {
    const candidate = asRecord(episode);
    if (!candidate || !positiveInteger(candidate.sequence)
      || typeof candidate.transaction_id !== "string"
      || typeof candidate.model_summary !== "string"
      || typeof candidate.recorded_at !== "string") return null;
  }
  if (record.version === 1) {
    return {
      version: 2,
      total_episodes: record.total_episodes,
      pruned_episodes: record.pruned_episodes,
      recent_episodes: structuredClone(
        record.recent_episodes
      ) as HumanoidEmbodiedMemoryState["recent_episodes"],
      total_experiences: 0,
      pruned_experiences: 0,
      recent_experiences: [],
      outcome_counts: emptyExperienceOutcomeCounts(),
      predicate_outcome_counts: {},
      object_outcome_counts: {},
      solid_outcome_counts: {},
      zone_outcome_counts: {}
    };
  }
  if (!nonnegativeInteger(record.total_experiences)
    || !nonnegativeInteger(record.pruned_experiences)
    || !Array.isArray(record.recent_experiences)
    || !experienceOutcomeCounts(record.outcome_counts)
    || !experienceOutcomeIndex(record.predicate_outcome_counts)
    || !experienceOutcomeIndex(record.object_outcome_counts)
    || (record.solid_outcome_counts !== undefined
      && !experienceOutcomeIndex(record.solid_outcome_counts))
    || !experienceOutcomeIndex(record.zone_outcome_counts)) return null;
  const outcomeCounts = record.outcome_counts;
  if (outcomeCounts.succeeded + outcomeCounts.rejected
    + outcomeCounts.physically_failed !== record.total_experiences) return null;
  for (const experience of record.recent_experiences) {
    const candidate = asRecord(experience);
    if (!candidate
      || !positiveInteger(candidate.sequence)
      || typeof candidate.source_ref !== "string"
      || typeof candidate.transaction_id !== "string"
      || candidate.source_ref !== `action:${candidate.transaction_id}`
      || typeof candidate.accepted !== "boolean"
      || typeof candidate.code !== "string"
      || !["succeeded", "rejected", "physically_failed"].includes(
        String(candidate.outcome)
      )
      || !Array.isArray(candidate.predicate_types)
      || !Array.isArray(candidate.object_ids)
      || !Array.isArray(candidate.solid_ids)
      || !Array.isArray(candidate.zone_ids)
      || typeof candidate.recorded_at !== "string") return null;
  }
  return structuredClone({
    ...record,
    solid_outcome_counts: record.solid_outcome_counts ?? {}
  }) as unknown as HumanoidEmbodiedMemoryState;
}

function emptyExperienceOutcomeCounts() {
  return { succeeded: 0, rejected: 0, physically_failed: 0 };
}

function experienceOutcomeCounts(value: unknown): value is {
  succeeded: number;
  rejected: number;
  physically_failed: number;
} {
  const record = asRecord(value);
  return record !== null
    && nonnegativeInteger(record.succeeded)
    && nonnegativeInteger(record.rejected)
    && nonnegativeInteger(record.physically_failed);
}

function experienceOutcomeIndex(value: unknown): boolean {
  const record = asRecord(value);
  return record !== null
    && Object.values(record).every(experienceOutcomeCounts);
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
