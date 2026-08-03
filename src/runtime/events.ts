import type { JsonValue } from "../domain/schema.js";

export interface RuntimeEvent {
  event_id: string;
  run_id: string;
  type: string;
  at: string;
  data: JsonValue;
  durable?: boolean;
  cursor?: string;
}

export type RuntimeEventSink = (event: RuntimeEvent) => void | Promise<void>;
