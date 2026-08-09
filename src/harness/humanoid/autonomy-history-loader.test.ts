import { describe, expect, it } from "vitest";
import type { JsonValue } from "../../domain/schema.js";
import type { JournalName, JournalPage, RunStore } from "../../persistence/run-store.js";
import { createActionGoalEvidence } from "./goal-evidence.js";
import {
  loadGoalEvidenceWorkingSet,
  loadModelAuthorityWorkingSet
} from "./autonomy-history-loader.js";

describe("autonomy history loading", () => {
  it("scans long Goal evidence journals into an exact bounded working set", async () => {
    const artifacts = Array.from({ length: 1_000 }, (_, index) => (
      createActionGoalEvidence({
        transactionId: `transaction-${index}`,
        worldFrame: index,
        worldRevision: index,
        receipt: {
          transactionId: `transaction-${index}`,
          worldAfterRevision: index,
          accepted: true
        }
      })
    ));
    const store = journalStore({ goal_evidence: artifacts as JsonValue[] });
    const loaded = await loadGoalEvidenceWorkingSet(store, new Set([
      "action:transaction-2",
      "action:transaction-997"
    ]));

    expect([...loaded.keys()]).toEqual([
      "action:transaction-2",
      "action:transaction-997"
    ]);
  });

  it("retains required and recent model authorities without loading the full journal", async () => {
    const records: JsonValue[] = [];
    for (let index = 0; index < 400; index += 1) {
      const modelCallId = uuid(index);
      records.push({
        version: 1,
        lifecycle: "started",
        model_call_id: modelCallId,
        agent_id: "humanoid-coordinator",
        at: "2026-08-08T00:00:00.000Z"
      }, {
        version: 1,
        lifecycle: "completed",
        model_call_id: modelCallId,
        agent_id: "humanoid-coordinator",
        response_id: `response-${index}`,
        response_output_sha256: index.toString(16).padStart(64, "0"),
        tool_calls: [],
        at: "2026-08-08T00:00:01.000Z"
      });
    }
    const pendingId = uuid(500);
    records.push({
      version: 1,
      lifecycle: "started",
      model_call_id: pendingId,
      agent_id: "humanoid-coordinator",
      at: "2026-08-08T00:00:02.000Z"
    });
    const loaded = await loadModelAuthorityWorkingSet(
      journalStore({ model_calls: records }),
      new Set([uuid(0), uuid(399)])
    );

    expect(loaded).toHaveLength(515);
    expect(loaded.some((record) => record.model_call_id === uuid(0))).toBe(true);
    expect(loaded.some((record) => record.model_call_id === uuid(399))).toBe(true);
    expect(loaded.at(-1)).toMatchObject({
      lifecycle: "started",
      model_call_id: pendingId
    });
  });
});

function journalStore(journals: Partial<Record<JournalName, JsonValue[]>>): RunStore {
  const entries = (name: JournalName) => journals[name] ?? [];
  return {
    readJournalPage: async (
      name: JournalName,
      from: number,
      limit: number
    ): Promise<JournalPage> => ({
      entries: entries(name).slice(from, from + limit),
      next: from + limit < entries(name).length ? from + limit : null,
      total: entries(name).length
    }),
    readJournalTail: async (
      name: JournalName,
      limit: number
    ): Promise<JournalPage> => ({
      entries: entries(name).slice(-limit),
      next: null,
      total: entries(name).length
    })
  } as RunStore;
}

function uuid(index: number): string {
  return `00000000-0000-4000-8000-${index.toString(16).padStart(12, "0")}`;
}
