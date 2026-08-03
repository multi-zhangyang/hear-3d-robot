import type {
  ContextCompactionSummary,
  ContextMemoryState,
  JsonValue,
  TaskNode
} from "../domain/schema.js";
import type { RunStore } from "../persistence/run-store.js";

interface ContextReceiptView {
  accepted: boolean;
  worldRevision: number;
}

export interface LongRunContextRuntime {
  readonly rootAgentId: string;
  readonly signal: AbortSignal | undefined;
  readonly store: RunStore;
  activeNode(agentId?: string): TaskNode;
  contextAnchor(agentId: string): JsonValue;
  contextMemory(): ContextMemoryState;
  contextWorldIdentity(): { worldRevision: number };
  contextReceipts(): Record<string, ContextReceiptView>;
  assertContextSummaryEvidence(summary: ContextCompactionSummary): void;
  updateContextMemory(state: ContextMemoryState, journalRecord?: JsonValue): Promise<void>;
  recordCompactionModelCall(agentId: string): Promise<void>;
  reconcileCompactionModelCalls(agentId: string, additionalCalls: number): Promise<void>;
  recordProvider(event: JsonValue, agentId?: string): Promise<void>;
}

export interface ModelTelemetryRuntime {
  readonly rootAgentId: string;
  activeNode(agentId?: string): TaskNode;
  recordModelCallStarted(agentId: string): Promise<void>;
}
