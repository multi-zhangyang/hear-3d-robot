import { describe, expect, it } from "vitest";
import { createCompletedGoalDAG } from "../../domain/goal-history.test-support.js";
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
});

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
