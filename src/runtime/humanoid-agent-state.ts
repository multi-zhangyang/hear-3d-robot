import { createHash } from "node:crypto";
import type { AgentInputItem } from "@openai/agents";
import type { ActionCommitOutbox } from "../domain/action-commit-outbox.js";
import type { ActionExecutionLedger } from "../domain/action-execution-ledger.js";
import {
  humanoidContextMemoryStateSha256,
  type HumanoidGoalProgress
} from "../domain/humanoid-run.js";
import type { ContextMemoryState } from "../domain/schema.js";
import type { FileSession } from "../persistence/file-session.js";
import type {
  AgentSessionStateBaseline
} from "../persistence/run-store.js";

export interface HumanoidAgentStateCheckpoint {
  goal_dag: { state_sha256: string };
  goal_progress: HumanoidGoalProgress | null;
  committed_actions: Readonly<Record<string, unknown>>;
  action_commit_outbox: ActionCommitOutbox;
  action_execution_ledger: ActionExecutionLedger;
  context_memory: ContextMemoryState;
  cycle_index: number;
  active_cycle: { cycle_id: string } | null;
}

export type HumanoidSessionBaseline = ReadonlyMap<
  string,
  readonly AgentInputItem[]
>;

export function humanoidAgentStateFingerprint(
  checkpoint: HumanoidAgentStateCheckpoint
): string {
  const goalProgress = checkpoint.goal_progress === null
    ? null
    : {
        version: checkpoint.goal_progress.version,
        goalSha256: checkpoint.goal_progress.goal_sha256,
        predicateCount: checkpoint.goal_progress.predicate_count,
        predicateStreaks: checkpoint.goal_progress.predicate_streaks
      };
  return createHash("sha256").update(canonicalJson({
    version: 1,
    goalDAGStateSha256: checkpoint.goal_dag.state_sha256,
    goalProgress,
    committedActionIds: Object.keys(checkpoint.committed_actions).sort(compareCodePoints),
    actionCommitOutbox: checkpoint.action_commit_outbox,
    actionExecutionLedger: checkpoint.action_execution_ledger,
    contextMemorySha256: humanoidContextMemoryStateSha256(
      checkpoint.context_memory
    ),
    cycleIndex: checkpoint.cycle_index,
    activeCycleId: checkpoint.active_cycle?.cycle_id ?? null
  })).digest("hex");
}

export async function captureHumanoidSessionBaseline(
  sessions: ReadonlyMap<string, FileSession>
): Promise<HumanoidSessionBaseline> {
  const entries = [...sessions.entries()].sort(([left], [right]) => (
    compareCodePoints(left, right)
  ));
  const snapshots = await Promise.all(entries.map(async ([agentId, session]) => (
    [agentId, await session.getItems()] as const
  )));
  return new Map(snapshots);
}

export async function restoreHumanoidSessionBaseline(
  sessions: ReadonlyMap<string, FileSession>,
  baseline: HumanoidSessionBaseline
): Promise<string[]> {
  if (sessions.size !== baseline.size) {
    throw new Error("Humanoid Agent Session set changed during one model call");
  }
  const entries = [...baseline.entries()].map(([agentId, items]) => {
    const session = sessions.get(agentId);
    if (!session) {
      throw new Error(`Humanoid Agent Session disappeared during recovery: ${agentId}`);
    }
    return { agentId, session, items };
  });
  const candidates = await Promise.all(entries.map(async (entry) => ({
    ...entry,
    current: await entry.session.getItems()
  })));
  const changed = candidates.filter(({ current, items }) => (
    agentInputItemsSha256(current) !== agentInputItemsSha256(items)
  ));
  // Publish the baseline even when this FileSession's cache already matches it:
  // another process may have replaced the durable file after the snapshot.
  // Only the returned ids describe an in-process SDK rollback that the context
  // manager must rebase around.
  await Promise.all(candidates.map(async ({ session, items }) => {
    await session.replaceItems([...items]);
  }));
  return changed.map(({ agentId }) => agentId);
}

export function humanoidSessionBaselineIdentity(
  baseline: HumanoidSessionBaseline
): AgentSessionStateBaseline {
  return Object.fromEntries([...baseline.entries()]
    .sort(([left], [right]) => compareCodePoints(left, right))
    .map(([agentId, items]) => [agentId, {
      item_count: items.length,
      items_sha256: agentInputItemsSha256(items)
    }]));
}

export async function captureHumanoidSessionStateIdentity(
  sessions: ReadonlyMap<string, FileSession>
): Promise<AgentSessionStateBaseline> {
  const entries = [...sessions.entries()].sort(([left], [right]) => (
    compareCodePoints(left, right)
  ));
  const identities = await Promise.all(entries.map(async ([agentId, session]) => {
    const identity = await session.getItemsIdentity();
    return [agentId, {
      item_count: identity.itemCount,
      items_sha256: identity.itemsSha256
    }] as const;
  }));
  return Object.fromEntries(identities);
}

export async function restoreHumanoidSessionStateBaseline(
  sessions: ReadonlyMap<string, FileSession>,
  baseline: AgentSessionStateBaseline
): Promise<boolean> {
  return (await restoreHumanoidSessionStateBaselineDetailed(
    sessions,
    baseline
  )).compatible;
}

export interface HumanoidSessionStateRestore {
  compatible: boolean;
  restored: HumanoidSessionBaseline;
}

export async function restoreHumanoidSessionStateBaselineDetailed(
  sessions: ReadonlyMap<string, FileSession>,
  baseline: AgentSessionStateBaseline
): Promise<HumanoidSessionStateRestore> {
  const agentIds = Object.keys(baseline).sort(compareCodePoints);
  if (sessions.size !== agentIds.length
    || agentIds.some((agentId) => !sessions.has(agentId))) {
    return { compatible: false, restored: new Map() };
  }
  const candidates = await Promise.all(agentIds.map(async (agentId) => {
    const session = sessions.get(agentId)!;
    const current = await session.getItems();
    const identity = baseline[agentId]!;
    if (current.length < identity.item_count) return null;
    const items = current.slice(0, identity.item_count);
    if (agentInputItemsSha256(items) !== identity.items_sha256) return null;
    return { agentId, session, currentCount: current.length, items };
  }));
  if (candidates.some((candidate) => candidate === null)) {
    return { compatible: false, restored: new Map() };
  }
  await Promise.all(candidates.map(async (candidate) => {
    if (!candidate || candidate.currentCount === candidate.items.length) return;
    await candidate.session.replaceItems(candidate.items);
  }));
  return {
    compatible: true,
    restored: new Map(candidates.flatMap((candidate) => (
      candidate && candidate.currentCount !== candidate.items.length
        ? [[candidate.agentId, structuredClone(candidate.items)] as const]
        : []
    )))
  };
}

function agentInputItemsSha256(items: readonly AgentInputItem[]): string {
  return createHash("sha256").update(canonicalJson(items)).digest("hex");
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([left], [right]) => compareCodePoints(left, right))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
    .join(",")}}`;
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
