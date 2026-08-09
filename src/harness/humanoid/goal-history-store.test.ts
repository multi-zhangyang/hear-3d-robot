import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { createCompletedGoalDAG } from "../../domain/goal-history.test-support.js";
import {
  GoalHistoryArchiveRecordSchema,
  type GoalHistoryArchiveRecord
} from "../../domain/goal-history-archive.js";
import type { JsonValue } from "../../domain/schema.js";
import type { JournalPage, RunStore } from "../../persistence/run-store.js";
import { reconcileAndCompactGoalHistory } from "./goal-history-store.js";

describe("Goal history persistence", () => {
  it("reconciles an append-before-checkpoint crash without duplicate records", async () => {
    const fullCheckpoint = createCompletedGoalDAG(15);
    const records: JsonValue[] = [];
    const store = memoryStore(records);
    const compacted = await reconcileAndCompactGoalHistory({
      store,
      goalDAG: fullCheckpoint
    });

    expect(records).toHaveLength(3);
    expect(compacted.epochs).toHaveLength(12);
    expect(compacted.archive.record_count).toBe(3);

    const recovered = await reconcileAndCompactGoalHistory({
      store,
      goalDAG: fullCheckpoint
    });
    expect(recovered).toEqual(compacted);
    expect(records).toHaveLength(3);
  });

  it("rebuilds the bounded lifetime summary for a pre-summary checkpoint", async () => {
    const records: JsonValue[] = [];
    const store = memoryStore(records);
    const compacted = await reconcileAndCompactGoalHistory({
      store,
      goalDAG: createCompletedGoalDAG(15, 3)
    });
    const legacy = structuredClone(compacted) as unknown as Record<string, unknown>;
    const archive = legacy.archive as Record<string, unknown>;
    delete archive.summary;
    delete legacy.state_sha256;
    const stateSha256 = createHash("sha256")
      .update(JSON.stringify(canonicalValue(legacy)))
      .digest("hex");

    const recovered = await reconcileAndCompactGoalHistory({
      store,
      goalDAG: { ...legacy, state_sha256: stateSha256 } as never
    });

    expect(recovered.archive.summary).toMatchObject({
      archived_epoch_count: 3,
      records_without_alternate_history: 0,
      outcomes: {
        selected: { total: 3, completed: 3 },
        not_selected: 6,
        predicate_outcomes: [{
          predicate_type: "robot_at",
          selected: { total: 3, completed: 3 },
          not_selected: 6
        }]
      }
    });
    expect(records).toHaveLength(3);
  });

  it("rebuilds a mixed V1 and V2 journal without inventing legacy alternates", async () => {
    const records: JsonValue[] = [];
    const store = memoryStore(records);
    const compacted = await reconcileAndCompactGoalHistory({
      store,
      goalDAG: createCompletedGoalDAG(15, 3)
    });
    const mixed = mixedArchiveChain(records);
    records.splice(0, records.length, ...mixed);

    const legacy = structuredClone(compacted) as unknown as Record<string, unknown>;
    const archive = legacy.archive as Record<string, unknown>;
    delete archive.summary;
    archive.last_record_sha256 = GoalHistoryArchiveRecordSchema.parse(mixed.at(-1))
      .record_sha256;
    delete legacy.state_sha256;
    const stateSha256 = sha256(legacy);

    const recovered = await reconcileAndCompactGoalHistory({
      store,
      goalDAG: { ...legacy, state_sha256: stateSha256 } as never
    });

    expect(recovered.archive.summary).toMatchObject({
      archived_epoch_count: 3,
      records_without_alternate_history: 1,
      outcomes: {
        selected: { total: 3, completed: 3 },
        not_selected: 4
      }
    });
  });
});

function mixedArchiveChain(records: JsonValue[]): JsonValue[] {
  let previousRecordSha256: string | null = null;
  return records.map((rawRecord, index) => {
    const record = GoalHistoryArchiveRecordSchema.parse(rawRecord);
    const contents = index === 0
      ? v1Contents(record)
      : v2Contents(record, previousRecordSha256);
    const rewritten = {
      ...contents,
      record_sha256: sha256(contents)
    } as unknown as JsonValue;
    previousRecordSha256 = GoalHistoryArchiveRecordSchema.parse(rewritten).record_sha256;
    return rewritten;
  });
}

function v1Contents(record: GoalHistoryArchiveRecord) {
  return {
    version: 1 as const,
    sequence: record.sequence,
    previous_record_sha256: null,
    candidate_sequence: record.candidate_sequence,
    candidate: record.candidate,
    epoch: record.epoch,
    evidence: record.evidence
  };
}

function v2Contents(
  record: GoalHistoryArchiveRecord,
  previousRecordSha256: string | null
) {
  if (record.version !== 2) throw new Error("Expected a V2 Goal archive fixture");
  const { record_sha256: _recordSha256, ...contents } = record;
  return { ...contents, previous_record_sha256: previousRecordSha256 };
}

function sha256(value: unknown): string {
  return createHash("sha256")
    .update(JSON.stringify(canonicalValue(value)))
    .digest("hex");
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([key, entry]) => [key, canonicalValue(entry)]));
  }
  return value;
}

function memoryStore(records: JsonValue[]): RunStore {
  return {
    append: async (_name: "goal_history", record: JsonValue) => {
      records.push(structuredClone(record));
    },
    readJournalTail: async (_name: "goal_history", limit: number): Promise<JournalPage> => ({
      entries: records.slice(-limit),
      next: null,
      total: records.length
    }),
    readJournalPage: async (
      _name: "goal_history",
      from: number,
      limit: number
    ): Promise<JournalPage> => ({
      entries: records.slice(from, from + limit),
      next: from + limit < records.length ? from + limit : null,
      total: records.length
    })
  } as unknown as RunStore;
}
