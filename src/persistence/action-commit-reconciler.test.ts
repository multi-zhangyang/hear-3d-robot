import { describe, expect, it, vi } from "vitest";
import {
  EmptyActionCommitOutbox,
  stageActionCommit,
  type ActionCommitOutbox
} from "../domain/action-commit-outbox.js";
import type { JsonValue } from "../domain/schema.js";
import {
  reconcileActionCommitOutbox,
  type ActionCommitJournal
} from "./action-commit-reconciler.js";
import type { DurableRuntimeEventRecord, JournalName, JournalPage } from "./run-store.js";

class MemoryJournal implements ActionCommitJournal {
  readonly journals = new Map<JournalName, JsonValue[]>();

  async scanJournal(
    name: JournalName,
    visit: (entry: JsonValue, index: number) => void | Promise<void>
  ): Promise<void> {
    const entries = this.journals.get(name) ?? [];
    for (let index = 0; index < entries.length; index += 1) {
      await visit(structuredClone(entries[index]!), index);
    }
  }

  async readJournalTail(name: JournalName, limit: number): Promise<JournalPage> {
    const entries = this.journals.get(name) ?? [];
    return {
      entries: structuredClone(entries.slice(-limit)),
      next: null,
      total: entries.length
    };
  }

  async readJournalPage(name: JournalName, from: number, limit: number): Promise<JournalPage> {
    const entries = this.journals.get(name) ?? [];
    const end = Math.min(entries.length, from + limit);
    return {
      entries: structuredClone(entries.slice(from, end)),
      next: end < entries.length ? end : null,
      total: entries.length
    };
  }

  async append(name: JournalName, value: JsonValue): Promise<void> {
    this.journals.set(name, [...(this.journals.get(name) ?? []), structuredClone(value)]);
  }

  async appendRuntimeEvents<T extends DurableRuntimeEventRecord>(
    events: readonly T[]
  ): Promise<Array<T & { cursor: string }>> {
    const current = this.journals.get("events") ?? [];
    const persisted = events.map((event, offset) => ({
      ...structuredClone(event),
      cursor: `cursor:${current.length + offset}`
    }));
    this.journals.set("events", [...current, ...persisted]);
    return persisted;
  }
}

function stagedOutbox(): ActionCommitOutbox {
  const receipt = {
    transactionId: "call-1",
    agentId: "humanoid-executor",
    action: "observe_humanoid" as const,
    input: {},
    fingerprint: "observe-humanoid-fingerprint",
    accepted: true,
    code: "humanoid_observed",
    worldBeforeRevision: 0,
    worldAfterRevision: 0,
    frameCount: 0,
    channels: [],
    detail: {},
    committedAt: "2026-08-03T10:00:00.000Z"
  };
  const actionRecord = {
    ...receipt,
    runtime_event_id: "event-1"
  };
  return stageActionCommit(EmptyActionCommitOutbox, {
    transactionId: "call-1",
    runtimeEventId: "event-1",
    actionRecord,
    goalEvidenceRef: "action:call-1",
    goalEvidenceRecord: {
      evidence: { ref: "action:call-1", kind: "action_receipt" },
      payload: {
        transaction_id: "call-1",
        receipt
      }
    },
    runtimeEvent: {
      event_id: "event-1",
      run_id: "run-1",
      type: "humanoid_action_committed",
      at: "2026-08-03T10:00:00.000Z",
      data: { receipt: actionRecord }
    },
    stagedAt: "2026-08-03T10:00:00.000Z"
  });
}

describe("action commit reconciler", () => {
  it("writes both journals before acknowledging and publishing", async () => {
    const store = new MemoryJournal();
    const order: string[] = [];
    const final = await reconcileActionCommitOutbox({
      store,
      outbox: stagedOutbox(),
      persist: async (outbox) => {
        expect(store.journals.get("actions")).toHaveLength(1);
        expect(store.journals.get("goal_evidence")).toHaveLength(1);
        expect(store.journals.get("action_identities")).toHaveLength(1);
        expect(store.journals.get("events")).toHaveLength(1);
        expect(outbox.pending).toEqual({});
        order.push("checkpoint");
      },
      publish: async () => {
        order.push("publish");
      }
    });

    expect(final).toEqual(EmptyActionCommitOutbox);
    expect(order).toEqual(["checkpoint", "publish"]);
  });

  it("recovers idempotently when both journal rows already exist", async () => {
    const store = new MemoryJournal();
    const first = stagedOutbox();
    const persist = vi.fn(async () => undefined);
    await reconcileActionCommitOutbox({ store, outbox: first, persist });
    const counts = {
      actions: store.journals.get("actions")!.length,
      goalEvidence: store.journals.get("goal_evidence")!.length,
      identities: store.journals.get("action_identities")!.length,
      events: store.journals.get("events")!.length
    };

    await reconcileActionCommitOutbox({ store, outbox: first, persist });
    expect(store.journals.get("actions")).toHaveLength(counts.actions);
    expect(store.journals.get("goal_evidence")).toHaveLength(counts.goalEvidence);
    expect(store.journals.get("action_identities")).toHaveLength(counts.identities);
    expect(store.journals.get("events")).toHaveLength(counts.events);
  });

  it("recovers after only the action row became durable", async () => {
    const store = new MemoryJournal();
    const outbox = stagedOutbox();
    await store.append("actions", outbox.pending["call-1"]!.action_record);

    await reconcileActionCommitOutbox({
      store,
      outbox,
      persist: async () => undefined
    });
    expect(store.journals.get("actions")).toHaveLength(1);
    expect(store.journals.get("events")).toHaveLength(1);
  });

  it("finds a matching commit outside the recent journal tail", async () => {
    const store = new MemoryJournal();
    const outbox = stagedOutbox();
    await store.append("actions", outbox.pending["call-1"]!.action_record);
    const [event] = await store.appendRuntimeEvents([
      outbox.pending["call-1"]!.runtime_event
    ]);
    for (let index = 0; index < 300; index += 1) {
      await store.append("actions", {
        transactionId: `later-action-${index}`,
        runtime_event_id: `later-action-event-${index}`
      });
      await store.append("events", {
        event_id: `later-event-${index}`,
        run_id: "run-1",
        type: "later_event",
        at: "2026-08-03T10:00:00.000Z",
        data: null,
        cursor: `later-cursor:${index}`
      });
    }

    const published: string[] = [];
    await reconcileActionCommitOutbox({
      store,
      outbox,
      persist: async () => undefined,
      publish: (persisted) => {
        published.push(persisted.cursor);
      }
    });

    expect(store.journals.get("actions")).toHaveLength(301);
    expect(store.journals.get("events")).toHaveLength(301);
    expect(published).toEqual([event!.cursor]);
  });

  it("rejects an existing journal row with the same identity and different payload", async () => {
    const store = new MemoryJournal();
    await store.append("actions", {
      transactionId: "call-1",
      agentId: "another-agent",
      runtime_event_id: "event-1"
    });

    await expect(reconcileActionCommitOutbox({
      store,
      outbox: stagedOutbox(),
      persist: async () => undefined
    })).rejects.toThrow(/transaction conflict/);
  });

  it("rejects rebound Goal evidence instead of appending a second identity", async () => {
    const store = new MemoryJournal();
    const outbox = stagedOutbox();
    await store.append("actions", outbox.pending["call-1"]!.action_record);
    await store.append("goal_evidence", {
      evidence: { ref: "action:call-1" },
      transaction_id: "another-call"
    });

    await expect(reconcileActionCommitOutbox({
      store,
      outbox,
      persist: async () => undefined
    })).rejects.toThrow(/Goal evidence journal identity conflict/);
    expect(store.journals.get("goal_evidence")).toHaveLength(1);
  });

  it.skip("allows an identical state anchor to reassert the durable head", async () => {
    const store = new MemoryJournal();
    const anchor = {
      event_id: "state-anchor-a",
      run_id: "run-1",
      type: "humanoid_action_commit_outbox_state_anchored",
      at: "2026-08-03T10:00:00.000Z",
      data: { version: 1, state_sha256: "a".repeat(64) }
    };
    await store.appendRuntimeEvents([anchor]);
    await store.appendRuntimeEvents([{
      ...anchor,
      event_id: "state-anchor-b",
      data: { version: 1, state_sha256: "b".repeat(64) }
    }]);
    await store.appendRuntimeEvents([anchor]);

    await expect(reconcileActionCommitOutbox({
      store,
      outbox: EmptyActionCommitOutbox,
      persist: async () => undefined
    })).resolves.toEqual(EmptyActionCommitOutbox);
  });

  it.skip("rejects a state anchor identity rebound with different content", async () => {
    const store = new MemoryJournal();
    const anchor = {
      event_id: "state-anchor-a",
      run_id: "run-1",
      type: "humanoid_action_commit_outbox_state_anchored",
      at: "2026-08-03T10:00:00.000Z",
      data: { version: 1, state_sha256: "a".repeat(64) }
    };
    await store.appendRuntimeEvents([anchor]);
    await store.appendRuntimeEvents([{
      ...anchor,
      data: { version: 1, state_sha256: "b".repeat(64) }
    }]);

    await expect(reconcileActionCommitOutbox({
      store,
      outbox: EmptyActionCommitOutbox,
      persist: async () => undefined
    })).rejects.toThrow("Duplicate durable action event identity");
  });
});
