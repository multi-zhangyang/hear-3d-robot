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
  recordModelCallStarted(agentId: string): Promise<string | undefined>;
  recordModelCallCompleted?(input: {
    modelCallId: string;
    agentId: string;
    responseId: string;
    responseOutputSha256: string;
    toolCalls: Array<{
      toolCallId: string;
      toolName: string;
      argumentsSha256: string;
    }>;
  }): Promise<void>;
  recordModelCallFailed?(modelCallId: string, agentId: string): Promise<void>;
  modelProgressSnapshot?(agentId: string): ModelProgressSnapshot;
  modelProgressRecoveryEpoch?(): number;
}

export interface ModelProgressReceipt {
  transactionId: string;
  agentId: string;
  action: string;
  accepted: boolean;
  code: string;
  worldBeforeRevision: number;
  worldAfterRevision: number;
  frameCount: number;
}

export interface ModelProgressSnapshot {
  worldRevision: number;
  cycleIndex: number;
  checkerSuccess: boolean;
  goalStateSha256: string;
  receipts: ModelProgressReceipt[];
}
