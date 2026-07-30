import { randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import type {
  JsonValue,
  RunCheckpoint,
  RunLifecycleEvent,
  RunLifecycleEventType
} from "../domain/schema.js";
import type { RunStore } from "./run-store.js";

export type LifecycleEventSink = (
  event: RunLifecycleEvent
) => void | Promise<void>;

export function createLifecycleEvent(input: {
  runId: string;
  type: RunLifecycleEventType;
  at: string;
  data: JsonValue;
}): RunLifecycleEvent {
  return {
    event_id: randomUUID(),
    run_id: input.runId,
    type: input.type,
    at: input.at,
    data: structuredClone(input.data)
  };
}

export async function reconcileLifecycleOutbox(input: {
  store: RunStore;
  checkpoint: RunCheckpoint;
  persistCheckpoint: () => Promise<void>;
  eventSink?: LifecycleEventSink;
}): Promise<void> {
  const pending = structuredClone(input.checkpoint.pending_lifecycle_events);
  if (pending.length === 0) return;

  const expectedById = new Map<string, RunLifecycleEvent>();
  for (const event of pending) {
    if (event.run_id !== input.store.definition.run_id) {
      throw new Error(`Lifecycle event ${event.event_id} belongs to another run`);
    }
    const existing = expectedById.get(event.event_id);
    if (existing && !isDeepStrictEqual(existing, event)) {
      throw new Error(`Lifecycle outbox contains conflicting event ${event.event_id}`);
    }
    expectedById.set(event.event_id, event);
  }

  const published = new Set<string>();
  await input.store.scanJournal("events", (entry) => {
    const eventId = eventIdentifier(entry);
    if (!eventId) return;
    const expected = expectedById.get(eventId);
    if (!expected) return;
    if (!matchesLifecycleEvent(entry, expected)) {
      throw new Error(`Lifecycle event ${eventId} conflicts with the event journal`);
    }
    published.add(eventId);
  });

  for (const event of pending) {
    if (published.has(event.event_id)) continue;
    const [persisted] = await input.store.appendRuntimeEvents([event]);
    published.add(event.event_id);
    await input.eventSink?.(structuredClone(persisted!));
  }

  const previous = input.checkpoint.pending_lifecycle_events;
  input.checkpoint.pending_lifecycle_events = previous.filter(
    (event) => !published.has(event.event_id)
  );
  try {
    await input.persistCheckpoint();
  } catch (error) {
    input.checkpoint.pending_lifecycle_events = previous;
    throw error;
  }
}

function eventIdentifier(value: JsonValue): string | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
  return typeof value.event_id === "string" ? value.event_id : undefined;
}

function matchesLifecycleEvent(value: JsonValue, expected: RunLifecycleEvent): boolean {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const { cursor: _cursor, ...event } = value;
  return isDeepStrictEqual(event, expected);
}
