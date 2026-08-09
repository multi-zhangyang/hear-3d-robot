import {
  applyGoalHistoryArchiveRecord,
  createGoalHistoryArchiveRecord,
  GoalHistoryArchiveRecordSchema
} from "../../domain/goal-history-archive.js";
import {
  GoalDAGSchema,
  rehashGoalDAG,
  type GoalDAG
} from "../../domain/goal-epoch.js";
import {
  appendGoalHistorySummary,
  createEmptyGoalHistorySummary
} from "../../domain/goal-history-summary.js";
import type { GoalHistorySummary } from
  "../../domain/goal-history-summary-schema.js";
import type { RunStore } from "../../persistence/run-store.js";
import { json } from "./run-runtime-persistence.js";

const WORKING_GOAL_EPOCH_LIMIT = 12;

export async function reconcileAndCompactGoalHistory(input: {
  store: RunStore;
  goalDAG: GoalDAG;
  epochLimit?: number;
}): Promise<GoalDAG> {
  const epochLimit = input.epochLimit ?? WORKING_GOAL_EPOCH_LIMIT;
  if (!Number.isSafeInteger(epochLimit) || epochLimit < 1) {
    throw new Error("Working Goal epoch limit must be positive");
  }
  let goalDAG = GoalDAGSchema.parse(input.goalDAG);
  const tail = await input.store.readJournalTail("goal_history", 1);
  if (tail.total < goalDAG.archive.record_count) {
    throw new Error("Goal history archive journal is shorter than its checkpoint head");
  }
  if (goalDAG.archive.summary === null) {
    const summary = await rebuildGoalHistorySummary(
      input.store,
      goalDAG.archive.record_count
    );
    if (summary.last_record_sha256 !== goalDAG.archive.last_record_sha256) {
      throw new Error("Rebuilt Goal history summary does not match the checkpoint head");
    }
    goalDAG = rehashGoalDAG({
      ...goalDAG,
      archive: { ...goalDAG.archive, summary }
    });
  }
  if (goalDAG.archive.record_count > 0) {
    const reflected = await input.store.readJournalPage(
      "goal_history",
      goalDAG.archive.record_count - 1,
      1
    );
    const head = GoalHistoryArchiveRecordSchema.parse(reflected.entries[0]);
    if (head.sequence !== goalDAG.archive.record_count
      || head.record_sha256 !== goalDAG.archive.last_record_sha256
      || head.epoch.epoch_id !== goalDAG.archive.last_epoch_id) {
      throw new Error("Goal history archive checkpoint head does not match its journal");
    }
  }

  let offset = goalDAG.archive.record_count;
  while (offset < tail.total) {
    const page = await input.store.readJournalPage(
      "goal_history",
      offset,
      Math.min(64, tail.total - offset)
    );
    if (page.entries.length === 0) {
      throw new Error(`Goal history archive stopped before record ${tail.total}`);
    }
    for (const rawRecord of page.entries) {
      goalDAG = applyGoalHistoryArchiveRecord(goalDAG, rawRecord);
      offset += 1;
    }
  }

  while (goalDAG.epochs.length > epochLimit) {
    const record = createGoalHistoryArchiveRecord(goalDAG);
    await input.store.append("goal_history", json(record));
    goalDAG = applyGoalHistoryArchiveRecord(goalDAG, record);
  }
  return goalDAG;
}

async function rebuildGoalHistorySummary(
  store: RunStore,
  recordCount: number
): Promise<GoalHistorySummary> {
  let summary = createEmptyGoalHistorySummary();
  let offset = 0;
  let expectedPreviousSha256: string | null = null;
  while (offset < recordCount) {
    const page = await store.readJournalPage(
      "goal_history",
      offset,
      Math.min(64, recordCount - offset)
    );
    if (page.entries.length === 0) {
      throw new Error(`Goal history summary stopped before record ${recordCount}`);
    }
    for (const rawRecord of page.entries) {
      const record = GoalHistoryArchiveRecordSchema.parse(rawRecord);
      if (record.sequence !== offset + 1
        || record.previous_record_sha256 !== expectedPreviousSha256) {
        throw new Error(`Goal history summary chain is inconsistent: ${record.sequence}`);
      }
      summary = appendGoalHistorySummary(summary, {
        sequence: record.sequence,
        recordSha256: record.record_sha256,
        candidate: record.candidate,
        epoch: record.epoch,
        alternateCandidates: record.version === 2
          ? record.alternate_candidates.map((entry) => entry.candidate)
          : [],
        alternateHistoryComplete: record.version === 2
      });
      expectedPreviousSha256 = record.record_sha256;
      offset += 1;
    }
  }
  return summary;
}
