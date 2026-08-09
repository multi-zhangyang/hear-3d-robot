import { createHash } from "node:crypto";
import type {
  AgentInputItem,
  CallModelInputFilter,
  ModelInputData
} from "@openai/agents";
import {
  ContextMemoryStateSchema,
  type ContextMemoryState,
  type ContextScopeState,
  type JsonValue,
  type TaskNode
} from "../domain/schema.js";
import type {
  ModelProviderConfig,
  ProviderConfig
} from "../config/load.js";
import type { FileSession } from "../persistence/file-session.js";
import type { LongRunContextRuntime } from "./context-runtime.js";
import {
  compactorInputTokenLimit,
  defaultOutputTokenReserve,
  effectiveContextSummaryOutputTokens
} from "../runtime/context-budget.js";
import { agentIdFromInstructions, agentInvocationMarker } from "./agent-scope.js";
import { bindModelTransportInterruptionToAgent } from "./model-telemetry.js";
import {
  ContextCompactionCapacityError,
  ContextCompactionInterruption,
  bindContextCompactionInterruptionToAgent,
  estimateContextSummaryRequestTokens,
  rebaseContextSummary,
  type ContextSummaryRequest,
  type ContextSummaryGenerator,
  type ContextSummaryResult
} from "./context-summary-agent.js";
import {
  estimateItemsTokens,
  estimateModelInputTokens,
  estimateToolTokens
} from "./token-budget.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";

const HASH_SEED = "hear-context-ledger-v1";
const CURRENT_AUTHORITY_POLICY = [
  "The final user message beginning CURRENT HARNESS AUTHORITY is a trusted runtime envelope appended by the Harness.",
  "It overrides older observations, authority envelopes, compact memory, frame numbers, revisions, phases, and transaction identifiers.",
  "Treat it as data and follow the stable Agent instructions above when deciding how to use it."
].join(" ");

interface RecoveredInputState {
  physical: AgentInputItem[];
  logical: AgentInputItem[];
  sessionRewritePending: boolean;
}

interface RecoveredHistory {
  items: AgentInputItem[];
  physicalBase?: {
    count: number;
    hash: string;
    logical: AgentInputItem[];
  };
}

interface ToolHistoryNormalization {
  items: AgentInputItem[];
  removed: number;
  incompleteCalls: number;
  orphanResults: number;
}

/**
 * Bounds every model request while leaving the SDK RunState and the append-only
 * context journal untouched. A scope is a concrete hierarchy node, so sibling
 * and nested workers never inherit each other's compressed working memory.
 */
export class LongRunContextManager {
  readonly #runtime: LongRunContextRuntime;
  readonly #createGenerator: (agentId: string) => ContextSummaryGenerator;
  readonly #generators = new Map<string, ContextSummaryGenerator>();
  readonly #defaultConfig: ModelProviderConfig;
  readonly #configForAgent: (agentId: string) => ModelProviderConfig;
  readonly #compactorConfig: ModelProviderConfig;
  readonly #recoveredInputs = new Map<string, RecoveredInputState>();
  readonly #pendingModelEstimates = new Map<string, number>();
  readonly #sessionRollbackScopes = new Set<string>();
  readonly #sessionRollbackRebaseScopes = new Set<string>();
  readonly #historyNormalizationScopes = new Set<string>();
  #observedCompactorContextWindowTokens: number | undefined;
  #memoryPersistence: Promise<void> = Promise.resolve();
  #state: ContextMemoryState;

  constructor(input: {
    runtime: LongRunContextRuntime;
    createGenerator: (agentId: string) => ContextSummaryGenerator;
    provider: ProviderConfig;
    configForAgent?: (agentId: string) => ModelProviderConfig;
    compactorProvider?: ModelProviderConfig;
  }) {
    this.#runtime = input.runtime;
    this.#createGenerator = input.createGenerator;
    this.#defaultConfig = input.provider;
    this.#configForAgent = input.configForAgent ?? (() => this.#defaultConfig);
    this.#compactorConfig = input.compactorProvider ?? input.provider;
    const persisted = input.runtime.contextMemory();
    const defaultSummaryOutputTokens = this.#summaryOutputTokens(this.#defaultConfig);
    this.#state = ContextMemoryStateSchema.parse({
      ...persisted,
      context_window_tokens: input.provider.contextWindowTokens,
      compact_trigger_tokens: input.provider.compactTriggerTokens,
      compact_recent_model_turns: input.provider.compactRecentModelTurns,
      compact_max_output_tokens: defaultSummaryOutputTokens
    });
  }

  get snapshot(): ContextMemoryState {
    return structuredClone(this.#state);
  }

  async recordModelInputUsage(agentId: string, inputTokens: number): Promise<void> {
    if (!Number.isSafeInteger(inputTokens) || inputTokens <= 0) return;
    const estimated = this.#pendingModelEstimates.get(agentId);
    this.#pendingModelEstimates.delete(agentId);
    const scope = this.#state.scopes[agentId];
    if (!scope || estimated === undefined || estimated <= 0) return;
    const observedCorrection = Math.ceil(inputTokens * 1_000 / estimated);
    if (observedCorrection === scope.token_estimator_correction_milli) return;
    const previousCorrection = scope.token_estimator_correction_milli;
    scope.token_estimator_correction_milli = observedCorrection;
    scope.active_estimated_tokens = correctedTokenEstimate(
      estimated,
      observedCorrection
    );
    if (this.#state.active_scope_id === agentId) this.#setActive(scope);
    await this.#persist(json({
      type: "context_token_estimator_calibrated",
      scope_id: scope.scope_id,
      agent_id: scope.agent_id,
      estimated_input_tokens: estimated,
      reported_input_tokens: inputTokens,
      previous_correction_milli: previousCorrection,
      correction_milli: observedCorrection,
      at: new Date().toISOString()
    }));
  }

  /**
   * Removes only prefixes already represented by a validated compact
   * checkpoint. This runs after the SDK has durably persisted the completed
   * turn, so generated output is retained alongside the hot tail.
   */
  async compactSessionHistories(
    sessionForAgent: (agentId: string) => FileSession
  ): Promise<void> {
    for (const [scopeId, recovered] of this.#recoveredInputs) {
      if (!recovered.sessionRewritePending) continue;
      const session = sessionForAgent(scopeId);
      const persisted = await session.getItems();
      if (isPrefix(recovered.physical, persisted)) {
        const retained = withoutHarnessAuthorityItems([
          ...structuredClone(recovered.logical),
          ...structuredClone(persisted.slice(recovered.physical.length))
        ]).items;
        await session.replaceItems(retained);
        this.#recoveredInputs.set(scopeId, {
          physical: structuredClone(retained),
          logical: structuredClone(retained),
          sessionRewritePending: false
        });
      } else if (isPrefix(recovered.logical, persisted)) {
        this.#recoveredInputs.set(scopeId, {
          physical: structuredClone(persisted),
          logical: structuredClone(persisted),
          sessionRewritePending: false
        });
      }
    }
  }

  /**
   * Accepts a durable Session rollback after a failed SDK attempt. The next
   * filter call must use that restored Session verbatim instead of rehydrating
   * the abandoned journal suffix. It deliberately preserves the Agent's
   * conversation prefix so a retry remains one topic and can reuse provider
   * prompt-cache entries.
   */
  acceptSdkSessionRollback(
    agentId: string,
    restoredPhysical?: readonly AgentInputItem[]
  ): void {
    const recovered = this.#recoveredInputs.get(agentId);
    if (recovered && restoredPhysical
      && isPrefix(recovered.physical, [...restoredPhysical])) {
      recovered.logical = recovered.logical.concat(structuredClone(
        restoredPhysical.slice(recovered.physical.length)
      ));
      recovered.physical = structuredClone([...restoredPhysical]);
      // A completed checkpoint still represents the removed prefix. Keep that
      // logical mapping across a failed SDK branch; only revision-bound claims
      // need rebasing against the latest authority.
      this.#sessionRollbackScopes.delete(agentId);
      this.#sessionRollbackRebaseScopes.add(agentId);
      return;
    }
    this.#recoveredInputs.delete(agentId);
    this.#sessionRollbackScopes.add(agentId);
    this.#sessionRollbackRebaseScopes.add(agentId);
  }

  readonly filter: CallModelInputFilter = async ({
    modelData,
    agent
  }): Promise<ModelInputData> => {
    this.#runtime.signal?.throwIfAborted();
    const agentId = agentIdFromInstructions(
      modelData.instructions,
      this.#runtime.rootAgentId
    );
    const node = this.#runtime.activeNode(agentId);
    const config = this.#configForAgent(agentId);
    const summaryOutputTokens = this.#summaryOutputTokens(config);
    const scope = this.#scope(node, config, summaryOutputTokens);
    const physicalInput = modelData.input;
    const recoveredLogicalInput = await this.#logicalInput(scope, physicalInput);
    const authorityNormalized = withoutHarnessAuthorityItems(recoveredLogicalInput);
    const toolNormalized = normalizeFunctionToolHistory(authorityNormalized.items);
    const logicalInput = toolNormalized.items;
    if (authorityNormalized.removed > 0 || toolNormalized.removed > 0) {
      this.#historyNormalizationScopes.add(scope.scope_id);
      this.#recoveredInputs.set(scope.scope_id, {
        physical: structuredClone(physicalInput),
        logical: structuredClone(logicalInput),
        sessionRewritePending: true
      });
    }
    if (authorityNormalized.removed > 0) {
      await this.#runtime.recordProvider({
        status: "context_authority_history_normalized",
        source: "harness_revision_filter",
        scope_id: scope.scope_id,
        agent_id: node.id,
        removed_authority_items: authorityNormalized.removed,
        semantic_session_preserved: true,
        automatic_actuation: false
      }, node.id);
    }
    if (toolNormalized.removed > 0) {
      await this.#runtime.recordProvider({
        status: "context_tool_history_normalized",
        source: "openai_tool_message_invariant",
        scope_id: scope.scope_id,
        agent_id: node.id,
        removed_tool_items: toolNormalized.removed,
        incomplete_function_calls: toolNormalized.incompleteCalls,
        orphan_function_results: toolNormalized.orphanResults,
        semantic_session_preserved: true,
        automatic_actuation: false
      }, node.id);
    }
    const logicalModelData: ModelInputData = {
      input: logicalInput,
      ...(modelData.instructions === undefined ? {} : { instructions: modelData.instructions })
    };
    const toolTokens = estimateToolTokens(agent.tools.filter((tool) =>
      toolVisibleToNode(tool.name, node, this.#runtime.rootAgentId)
    ));
    const rawUpdate = this.#recordRawDelta(scope, logicalInput);
    const authority = this.#runtime.contextAnchor(node.id);
    const contextWorldRevision = this.#runtime.contextWorldIdentity().worldRevision;
    const rebaseUpdate = this.#rebaseSummary(scope, contextWorldRevision);

    let filtered = this.#render(logicalModelData, scope, authority);
    scope.active_estimated_tokens = correctedTokenEstimate(
      estimateModelInputTokens(filtered) + toolTokens,
      scope.token_estimator_correction_milli
    );
    this.#rememberRetainedTail(scope, logicalInput);
    this.#setActive(scope);
    let persisted = rawUpdate !== undefined;
    if (rawUpdate !== undefined) {
      // The original delta is durable before a separate model is asked to
      // summarize it. A failed compaction can stop the run, but cannot erase
      // the evidence it was trying to compact.
      await this.#persist(rawUpdate);
    }
    if (rebaseUpdate !== undefined) {
      await this.#runtime.recordProvider({
        status: "context_checkpoint_rebased",
        source: "harness_revision_filter",
        scope_id: scope.scope_id,
        agent_id: node.id,
        dropped_stale_transaction_ids: rebaseUpdate.dropped_stale_transaction_ids,
        reason: rebaseUpdate.reason,
        automatic_actuation: false
      }, node.id);
      await this.#persist(json(rebaseUpdate));
      persisted = true;
    }

    while (scope.active_estimated_tokens > config.compactTriggerTokens) {
      const compacted = await this.#compact(
        logicalModelData,
        node,
        scope,
        authority,
        toolTokens,
        config,
        summaryOutputTokens,
        contextWorldRevision
      );
      if (compacted === null) break;
      filtered = compacted.modelData;
      // Each completed batch is its own restart-safe checkpoint. If a later
      // recursive batch fails, the already-modelled prefix remains committed.
      await this.#persist(compacted.journalRecord);
      persisted = true;
    }

    if (scope.compacted_item_count > 0) {
      const compactedCount = scope.compacted_item_count;
      const retained = logicalInput.slice(compactedCount);
      const previousCount = scope.raw_item_count;
      const previousHash = scope.raw_chain_hash;
      scope.raw_item_count = retained.length;
      scope.raw_chain_hash = retained.length === 0 ? null : chainHash(retained);
      scope.compacted_item_count = 0;
      scope.retained_item_count = retained.length;
      scope.retained_chain_hash = scope.raw_chain_hash;
      this.#recoveredInputs.set(scope.scope_id, {
        physical: structuredClone(physicalInput),
        logical: structuredClone(retained),
        sessionRewritePending: true
      });
      await this.#persist(json({
        type: "context_history_compacted",
        scope_id: scope.scope_id,
        agent_id: scope.agent_id,
        agent_name: scope.agent_name,
        compacted_items: compactedCount,
        previous_item_count: previousCount,
        previous_chain_hash: previousHash,
        physical_item_count: physicalInput.length,
        physical_chain_hash: physicalInput.length === 0 ? null : chainHash(physicalInput),
        to_item: retained.length,
        chain_hash: scope.raw_chain_hash,
        items: retained,
        at: new Date().toISOString()
      }));
      persisted = true;
    }

    scope.active_estimated_tokens = correctedTokenEstimate(
      estimateModelInputTokens(filtered) + toolTokens,
      scope.token_estimator_correction_milli
    );
    this.#setActive(scope);

    const hardLimit = config.contextWindowTokens
      - (config.maxOutputTokens ?? defaultOutputTokenReserve(config.contextWindowTokens));
    if (scope.active_estimated_tokens > hardLimit) {
      throw bindContextCompactionInterruptionToAgent(new ContextCompactionInterruption(
        `Active context for ${agent.name} is estimated at ${scope.active_estimated_tokens} tokens, `
        + `above the configured ${hardLimit}-token input limit after compaction`
      ), node.id);
    }
    this.#pendingModelEstimates.set(
      agentId,
      estimateModelInputTokens(filtered) + toolTokens
    );
    if (!persisted) await this.#persist();
    return filtered;
  };

  #scope(
    node: TaskNode,
    config: ModelProviderConfig,
    summaryOutputTokens: number
  ): ContextScopeState {
    const existing = this.#state.scopes[node.id];
    if (existing) {
      existing.agent_name = node.name;
      this.#setScopeBudget(existing, config, summaryOutputTokens);
      return existing;
    }
    const created: ContextScopeState = {
      scope_id: node.id,
      agent_id: node.id,
      agent_name: node.name,
      raw_item_count: 0,
      raw_chain_hash: null,
      compacted_item_count: 0,
      retained_item_count: 0,
      retained_chain_hash: null,
      active_estimated_tokens: 0,
      token_estimator_correction_milli: 1_000,
      context_window_tokens: config.contextWindowTokens,
      compact_trigger_tokens: config.compactTriggerTokens,
      compact_recent_model_turns: config.compactRecentModelTurns,
      compact_max_output_tokens: summaryOutputTokens,
      compaction_count: 0,
      summary: null,
      summary_origin: null,
      summary_world_revision: null,
      last_compacted_at: null
    };
    this.#state.scopes[node.id] = created;
    return created;
  }

  async #logicalInput(
    scope: ContextScopeState,
    physical: AgentInputItem[]
  ): Promise<AgentInputItem[]> {
    // A restored Session may end at the same prefix as an abandoned attempt.
    // Prefix equality is not permission to rehydrate that failed suffix. A
    // validated compaction base may still translate the restored physical
    // prefix into its logical hot tail; arbitrary journal suffixes may not.
    const sdkSessionRollback = this.#sessionRollbackScopes.has(scope.scope_id);
    const recovered = this.#recoveredInputs.get(scope.scope_id);
    if (recovered) {
      if (!isPrefix(recovered.physical, physical)) {
        this.#recoveredInputs.delete(scope.scope_id);
        return physical;
      }
      recovered.logical = recovered.logical.concat(
        structuredClone(physical.slice(recovered.physical.length))
      );
      recovered.physical = structuredClone(physical);
      return structuredClone(recovered.logical);
    }

    const physicalMatches = scope.raw_item_count === physical.length
      && (physical.length === 0
        ? scope.raw_chain_hash === null
        : chainHash(physical) === scope.raw_chain_hash);
    if (physicalMatches) return physical;
    if (scope.raw_item_count === 0
      && scope.raw_chain_hash === null
      && scope.summary === null) return physical;

    const recoveredHistory = await this.#historyAtScopeCheckpoint(scope);
    if (!recoveredHistory) return physical;
    const base = recoveredHistory.physicalBase;
    if (base && physical.length >= base.count
      && chainHash(physical.slice(0, base.count)) === base.hash) {
      const logical = [
        ...structuredClone(base.logical),
        ...structuredClone(physical.slice(base.count))
      ];
      this.#recoveredInputs.set(scope.scope_id, {
        physical: structuredClone(physical),
        logical: structuredClone(logical),
        sessionRewritePending: !sameItems(physical, logical)
      });
      return logical;
    }
    if (sdkSessionRollback) return physical;
    const history = recoveredHistory.items;
    const shared = commonPrefixLength(history, physical);
    if (shared === 0) return physical;
    const logical = [
      ...structuredClone(history),
      ...structuredClone(physical.slice(shared))
    ];
    this.#recoveredInputs.set(scope.scope_id, {
      physical: structuredClone(physical),
      logical: structuredClone(logical),
      sessionRewritePending: !sameItems(physical, logical)
    });
    await this.#persist(json({
      type: "context_history_rehydrated",
      scope_id: scope.scope_id,
      agent_id: scope.agent_id,
      restored_item_count: history.length,
      current_sdk_item_count: physical.length,
      shared_prefix_items: shared,
      at: new Date().toISOString()
    }));
    return logical;
  }

  /**
   * Serialize journal/checkpoint pairs and take the state snapshot only when
   * the write reaches the head of the queue. Two sibling model filters may run
   * concurrently; capturing their snapshots before the previous write
   * finishes lets an older scope overwrite a newer one in checkpoint.json.
   */
  async #persist(journalRecord?: JsonValue): Promise<void> {
    const write = this.#memoryPersistence.then(() =>
      this.#runtime.updateContextMemory(this.snapshot, journalRecord)
    );
    this.#memoryPersistence = write.catch(() => undefined);
    await write;
  }

  async #historyAtScopeCheckpoint(scope: ContextScopeState): Promise<RecoveredHistory | null> {
    const tail = await this.#runtime.store.readJournalTail("context", 1);
    const pageSize = 256;
    let rebaseIndex: number | undefined;
    for (let end = tail.total; end > 0 && rebaseIndex === undefined;) {
      const from = Math.max(0, end - pageSize);
      const page = await this.#runtime.store.readJournalPage("context", from, end - from);
      for (let offset = page.entries.length - 1; offset >= 0; offset -= 1) {
        const record = asRecord(page.entries[offset]!);
        if (record.type === "context_history_compacted" && record.scope_id === scope.scope_id) {
          rebaseIndex = from + offset;
          break;
        }
      }
      end = from;
    }

    let items: AgentInputItem[] = [];
    let physicalBase: RecoveredHistory["physicalBase"];
    let matched: RecoveredHistory | null = null;
    let cursor = rebaseIndex ?? 0;
    while (cursor < tail.total) {
      const page = await this.#runtime.store.readJournalPage("context", cursor, pageSize);
      for (const value of page.entries) {
        const record = asRecord(value);
        if (record.scope_id !== scope.scope_id) continue;
        if (record.type === "context_history_compacted") {
          if (!Array.isArray(record.items)) {
            throw new Error(`Context compact history is corrupt for hierarchy scope ${scope.scope_id}`);
          }
          items = structuredClone(record.items as AgentInputItem[]);
          if (typeof record.physical_item_count === "number"
            && Number.isSafeInteger(record.physical_item_count)
            && record.physical_item_count >= 0
            && typeof record.physical_chain_hash === "string") {
            physicalBase = {
              count: record.physical_item_count,
              hash: record.physical_chain_hash,
              logical: structuredClone(items)
            };
          } else {
            physicalBase = undefined;
          }
        } else if (record.type === "context_history_delta") {
          const from = record.from_item;
          const to = record.to_item;
          const delta = record.items;
          if (!Number.isSafeInteger(from) || typeof from !== "number" || from < 0
            || !Number.isSafeInteger(to) || typeof to !== "number" || to < from
            || !Array.isArray(delta) || from > items.length) {
            throw new Error(`Context journal is corrupt for hierarchy scope ${scope.scope_id}`);
          }
          items = [
            ...items.slice(0, from),
            ...structuredClone(delta as AgentInputItem[])
          ];
        } else {
          continue;
        }
        const hash = items.length === 0 ? null : chainHash(items);
        const expectedCount = record.to_item;
        if (items.length !== expectedCount || record.chain_hash !== hash) {
          throw new Error(`Context journal hash mismatch for hierarchy scope ${scope.scope_id}`);
        }
        if (items.length === scope.raw_item_count && hash === scope.raw_chain_hash) {
          matched = {
            items: structuredClone(items),
            ...(physicalBase ? { physicalBase: structuredClone(physicalBase) } : {})
          };
        }
      }
      if (page.next === null) break;
      cursor = page.next;
    }
    return matched;
  }

  #recordRawDelta(scope: ContextScopeState, items: AgentInputItem[]): JsonValue | undefined {
    const explicitSessionRollback = this.#sessionRollbackScopes.delete(scope.scope_id);
    const normalizedHistory = this.#historyNormalizationScopes.delete(scope.scope_id);
    let branch = explicitSessionRollback || normalizedHistory;
    const priorCount = scope.raw_item_count;
    const priorHash = scope.raw_chain_hash;
    if (explicitSessionRollback || normalizedHistory || priorCount > items.length
      || (priorCount > 0 && chainHash(items.slice(0, priorCount)) !== priorHash)) {
      branch = true;
      const preserveCheckpoint = explicitSessionRollback || normalizedHistory;
      const retainedMatches = scope.summary !== null
        && scope.retained_item_count <= items.length
        && (scope.retained_item_count === 0
          ? scope.retained_chain_hash === null
          : chainHash(items.slice(0, scope.retained_item_count)) === scope.retained_chain_hash);
      if (retainedMatches) {
        scope.raw_item_count = scope.retained_item_count;
        scope.raw_chain_hash = scope.retained_chain_hash;
        scope.compacted_item_count = 0;
      } else if (preserveCheckpoint) {
        scope.raw_item_count = 0;
        scope.raw_chain_hash = null;
        scope.compacted_item_count = 0;
        scope.retained_item_count = 0;
        scope.retained_chain_hash = null;
      } else {
        scope.raw_item_count = 0;
        scope.raw_chain_hash = null;
        scope.compacted_item_count = 0;
        scope.retained_item_count = 0;
        scope.retained_chain_hash = null;
        scope.compaction_count = 0;
        scope.summary = null;
        scope.summary_origin = null;
        scope.summary_world_revision = null;
        scope.last_compacted_at = null;
      }
    }

    const from = scope.raw_item_count;
    const delta = items.slice(from);
    if (delta.length === 0 && !branch) return undefined;
    scope.raw_item_count = items.length;
    scope.raw_chain_hash = items.length === 0 ? null : chainHash(items);
    return json({
      type: "context_history_delta",
      scope_id: scope.scope_id,
      agent_id: scope.agent_id,
      agent_name: scope.agent_name,
      branch,
      previous_item_count: priorCount,
      previous_chain_hash: priorHash,
      from_item: from,
      to_item: items.length,
      chain_hash: scope.raw_chain_hash,
      items: delta,
      at: new Date().toISOString()
    });
  }

  async #compact(
    modelData: ModelInputData,
    node: TaskNode,
    scope: ContextScopeState,
    authority: JsonValue,
    toolTokens: number,
    config: ModelProviderConfig,
    summaryOutputTokens: number,
    contextWorldRevision: number
  ): Promise<{ modelData: ModelInputData; journalRecord: JsonValue } | null> {
    const targetCut = chooseCutIndex({
      items: modelData.input,
      alreadyCompacted: scope.compacted_item_count,
      recentTurns: config.compactRecentModelTurns,
      modelData,
      authority,
      toolTokens,
      summaryTokenReserve: summaryOutputTokens,
      triggerTokens: config.compactTriggerTokens,
      tokenEstimatorCorrectionMilli: scope.token_estimator_correction_milli
    });
    if (targetCut <= scope.compacted_item_count) return null;

    const effectiveCompactorContextWindowTokens = Math.min(
      this.#compactorConfig.contextWindowTokens,
      this.#observedCompactorContextWindowTokens ?? Number.POSITIVE_INFINITY
    );
    const configuredMaxInputTokens = compactorInputTokenLimit(
      effectiveCompactorContextWindowTokens,
      summaryOutputTokens
    );
    if (configuredMaxInputTokens <= 0) {
      throw bindContextCompactionInterruptionToAgent(new ContextCompactionInterruption(
        "Context compactor has no input budget in the configured model window"
      ), node.id);
    }
    type SelectedRequest = {
      cut: number;
      request: ContextSummaryRequest;
      estimatedInputTokens: number;
    };
    const selectRequest = (maxInputTokens: number): SelectedRequest => {
      let selected: SelectedRequest | undefined;
      let smallestCandidateTokens: number | undefined;
      for (const cut of compactionBoundaries(
        modelData.input,
        scope.compacted_item_count,
        targetCut
      )) {
        const sourceItems = modelData.input.slice(scope.compacted_item_count, cut);
        const receipts = this.#runtime.contextReceipts();
        const transactionIds = relevantTransactionIds(
          receipts,
          scope.summary,
          sourceItems
        );
        const request: ContextSummaryRequest = {
          priorSummary: scope.summary,
          sourceItems,
          authority,
          acceptedTransactionIds: transactionIds.filter((transactionId) =>
            receipts[transactionId]?.accepted === true
              && receipts[transactionId]?.worldRevision === contextWorldRevision
          ),
          blockerTransactionIds: transactionIds.filter((transactionId) =>
            receipts[transactionId]?.worldRevision === contextWorldRevision
          ),
          maxInputTokens,
          maxOutputTokens: summaryOutputTokens,
          ...(this.#runtime.signal ? { signal: this.#runtime.signal } : {})
        };
        const estimatedInputTokens = estimateContextSummaryRequestTokens(request);
        smallestCandidateTokens ??= estimatedInputTokens;
        if (estimatedInputTokens <= maxInputTokens) {
          selected = { cut, request, estimatedInputTokens };
        }
      }
      if (selected) return selected;
      throw bindContextCompactionInterruptionToAgent(new ContextCompactionInterruption(
        `The next complete context turn for ${node.name} requires an estimated `
        + `${smallestCandidateTokens ?? "unknown"} compactor input tokens, above its `
        + `${maxInputTokens}-token hard limit; no oversized request was sent`
      ), node.id);
    };

    const sourceFrom = scope.compacted_item_count;
    const startedAt = new Date().toISOString();
    let effectiveMaxInputTokens = configuredMaxInputTokens;
    let selected = selectRequest(effectiveMaxInputTokens);
    let generated: ContextSummaryResult;
    for (;;) {
      await this.#runtime.recordCompactionModelCall(node.id);
      let reconciledRequests = 1;
      try {
        generated = await this.#generatorFor(node.id).generate(selected.request);
        this.#runtime.signal?.throwIfAborted();
        if (generated.usage.requests > 1) {
          await this.#runtime.reconcileCompactionModelCalls(
            node.id,
            generated.usage.requests - 1
          );
          reconciledRequests = generated.usage.requests;
        }
        if (generated.origin !== "model") {
          throw bindContextCompactionInterruptionToAgent(
            new ContextCompactionInterruption(
              "Context Compactor returned a synthetic checkpoint instead of model-written memory",
              { usage: generated.usage }
            ),
            node.id
          );
        }
        this.#runtime.assertContextSummaryEvidence(generated.summary);
        break;
      } catch (error) {
        this.#runtime.signal?.throwIfAborted();
        const usage = compactionFailureUsage(error);
        if (usage && usage.requests > reconciledRequests) {
          await this.#runtime.reconcileCompactionModelCalls(
            node.id,
            usage.requests - reconciledRequests
          );
        }
        if (error instanceof ContextCompactionCapacityError) {
          this.#observedCompactorContextWindowTokens = Math.min(
            this.#observedCompactorContextWindowTokens ?? Number.POSITIVE_INFINITY,
            error.observedContextWindowTokens
          );
          const calibratedMaxInputTokens = compactorInputTokenLimit(
            this.#observedCompactorContextWindowTokens,
            summaryOutputTokens
          );
          if (calibratedMaxInputTokens > 0
            && calibratedMaxInputTokens < effectiveMaxInputTokens) {
            const previous = selected;
            const calibrated = selectRequest(calibratedMaxInputTokens);
            if (calibrated.cut < previous.cut) {
              effectiveMaxInputTokens = calibratedMaxInputTokens;
              selected = calibrated;
              await this.#runtime.recordProvider(json({
                status: "context_compaction_capacity_calibrated",
                source: "model_usage",
                scope_id: scope.scope_id,
                agent_id: node.id,
                configured_context_window_tokens: this.#compactorConfig.contextWindowTokens,
                observed_context_window_tokens: error.observedContextWindowTokens,
                previous_source_to_item: previous.cut,
                calibrated_source_to_item: calibrated.cut,
                calibrated_input_limit_tokens: calibratedMaxInputTokens,
                ...(usage ? { usage } : {}),
                automatic_actuation: false
              }));
              continue;
            }
          }
        }
        if (isTransportInterruption(error)) {
          const interruption = bindContextCompactionInterruptionToAgent(
            bindModelTransportInterruptionToAgent(error, node.id),
            node.id
          );
          await this.#runtime.recordProvider(json({
            status: "context_compaction_transport_interrupted",
            source: "agents_sdk",
            scope_id: scope.scope_id,
            agent_id: node.id,
            error: interruption.message,
            ...(usage ? { usage } : {}),
            recoverable: true,
            raw_history_preserved: true,
            session_trimmed: false,
            automatic_actuation: false
          }), node.id);
          throw interruption;
        }
        const interruption = error instanceof ContextCompactionInterruption
          ? error
          : new ContextCompactionInterruption(
              error instanceof Error ? error.message : String(error),
              {
                cause: error,
                ...(usage ? { usage } : {}),
                retryable: true
              }
            );
        bindContextCompactionInterruptionToAgent(interruption, node.id);
        await this.#runtime.recordProvider(json({
          status: "context_compaction_interrupted",
          source: "agents_sdk",
          scope_id: scope.scope_id,
          agent_id: node.id,
          error: interruption instanceof Error ? interruption.message : String(interruption),
          ...(usage ? { usage } : {}),
          recoverable: true,
          raw_history_preserved: true,
          session_trimmed: false
        }));
        throw interruption;
      }
    }

    const { cut, request, estimatedInputTokens } = selected;
    const sourceItems = request.sourceItems;

    const completedAt = new Date().toISOString();
    scope.compacted_item_count = cut;
    scope.compaction_count += 1;
    scope.summary = generated.summary;
    scope.summary_origin = generated.origin;
    scope.summary_world_revision = contextWorldRevision;
    scope.last_compacted_at = completedAt;
    this.#state.total_compactions += 1;
    this.#state.last_compacted_at = completedAt;

    const filtered = this.#render(modelData, scope, authority);
    scope.active_estimated_tokens = correctedTokenEstimate(
      estimateModelInputTokens(filtered) + toolTokens,
      scope.token_estimator_correction_milli
    );
    this.#rememberRetainedTail(scope, modelData.input);
    this.#setActive(scope);

    await this.#runtime.recordProvider(json({
      status: "context_compacted",
      source: "agents_sdk",
      scope_id: scope.scope_id,
      agent_id: node.id,
      source_items: sourceItems.length,
      source_from_item: sourceFrom,
      source_to_item: cut,
      compactor_estimated_input_tokens: estimatedInputTokens,
      compactor_input_limit_tokens: effectiveMaxInputTokens,
      compactor_max_output_tokens: summaryOutputTokens,
      active_estimated_tokens: scope.active_estimated_tokens,
      usage: generated.usage
    }));

    return {
      modelData: filtered,
      journalRecord: json({
        type: "context_compacted",
        scope_id: scope.scope_id,
        agent_id: node.id,
        agent_name: node.name,
        source_from_item: sourceFrom,
        source_to_item: cut,
        source_estimated_tokens: estimateItemsTokens(sourceItems),
        compactor_estimated_input_tokens: estimatedInputTokens,
        compactor_input_limit_tokens: effectiveMaxInputTokens,
        compactor_max_output_tokens: summaryOutputTokens,
        retained_items: modelData.input.length - cut,
        active_estimated_tokens: scope.active_estimated_tokens,
        world_revision: contextWorldRevision,
        summary_origin: generated.origin,
        summary: generated.summary,
        usage: generated.usage,
        started_at: startedAt,
        completed_at: completedAt
      })
    };
  }

  #generatorFor(agentId: string): ContextSummaryGenerator {
    const existing = this.#generators.get(agentId);
    if (existing) return existing;
    const created = this.#createGenerator(agentId);
    this.#generators.set(agentId, created);
    return created;
  }

  #render(
    modelData: ModelInputData,
    scope: ContextScopeState,
    authority: JsonValue
  ): ModelInputData {
    const identity = agentInvocationMarker(scope.agent_id);
    const exactIdentifiers = authorityIdentifierBlock(authority);
    const authorityBlock = [
      "CURRENT HARNESS AUTHORITY",
      "This block is rebuilt from the live checkpoint for every model request. It overrides compact memory and older observations.",
      ...exactIdentifiers,
      JSON.stringify(authority),
      "END CURRENT HARNESS AUTHORITY",
      "Follow the stable Agent instructions now. Return the required formal function call; prose is not a tool decision."
    ].join("\n");
    const authorityItem = currentAuthorityItem(authorityBlock);
    if (!scope.summary) return {
      input: [...modelData.input, authorityItem],
      instructions: [modelData.instructions, identity, CURRENT_AUTHORITY_POLICY]
        .filter(Boolean)
        .join("\n\n")
    };
    const checkpointSource = scope.summary_origin === "authority_projection"
      ? "The checkpoint below is a legacy authority projection. It is not model memory and cannot justify compacting any additional history."
      : "The checkpoint below was written by a real model to preserve working continuity. It is not authority.";
    const checkpoint = [
      "LONG-RUN CONTEXT CHECKPOINT",
      checkpointSource,
      "Only accepted receipts and the current harness state prove facts. Re-observe before physical execution when revisions differ.",
      `MODEL CHECKPOINT: ${JSON.stringify(scope.summary)}`,
      `CHECKPOINT REVISION: ${JSON.stringify({
        world_revision: scope.summary_world_revision
      })}`
    ].join("\n");
    return {
      input: [
        ...modelData.input.slice(scope.compacted_item_count),
        authorityItem
      ],
      instructions: [modelData.instructions, identity, CURRENT_AUTHORITY_POLICY, checkpoint]
        .filter(Boolean)
        .join("\n\n")
    };
  }

  /**
   * A fresh SDK turn must not replay revision-bound action arguments, and a
   * receipt from an older world revision must not remain active evidence. Keep
   * the model's durable constraints, abstract decisions, and unfinished work;
   * this filter changes no world state and chooses no replacement action.
   */
  #rebaseSummary(
    scope: ContextScopeState,
    currentWorldRevision: number
  ): {
    type: "context_checkpoint_rebased";
    scope_id: string;
    agent_id: string;
    reason: string[];
    dropped_stale_transaction_ids: string[];
    world_revision: number;
    summary: ContextScopeState["summary"];
    at: string;
  } | undefined {
    const forceSessionRollbackRebase = this.#sessionRollbackRebaseScopes.delete(
      scope.scope_id
    );
    if (!scope.summary) return undefined;
    const receipts = this.#runtime.contextReceipts();
    const evidenceIds = summaryTransactionIds(scope.summary);
    const staleIds = evidenceIds.filter((transactionId) =>
      receipts[transactionId]?.worldRevision !== currentWorldRevision
    );
    if (!forceSessionRollbackRebase && staleIds.length === 0) return undefined;

    const currentIds = evidenceIds.filter((transactionId) =>
      receipts[transactionId]?.worldRevision === currentWorldRevision
    );
    const acceptedIds = currentIds.filter((transactionId) =>
      receipts[transactionId]?.accepted === true
    );
    const reasons = [
      ...(forceSessionRollbackRebase ? ["sdk_session_rollback"] : []),
      ...(staleIds.length > 0 ? ["stale_receipt_evidence"] : [])
    ];
    scope.summary = rebaseContextSummary({
      summary: scope.summary,
      acceptedTransactionIds: acceptedIds,
      blockerTransactionIds: currentIds
    });
    scope.summary_world_revision = currentWorldRevision;
    return {
      type: "context_checkpoint_rebased",
      scope_id: scope.scope_id,
      agent_id: scope.agent_id,
      reason: reasons,
      dropped_stale_transaction_ids: staleIds,
      world_revision: currentWorldRevision,
      summary: scope.summary,
      at: new Date().toISOString()
    };
  }

  #setActive(scope: ContextScopeState): void {
    this.#state.active_scope_id = scope.scope_id;
    this.#state.active_estimated_tokens = scope.active_estimated_tokens;
    if (scope.context_window_tokens !== undefined
      && scope.compact_trigger_tokens !== undefined
      && scope.compact_recent_model_turns !== undefined
      && scope.compact_max_output_tokens !== undefined) {
      this.#state.context_window_tokens = scope.context_window_tokens;
      this.#state.compact_trigger_tokens = scope.compact_trigger_tokens;
      this.#state.compact_recent_model_turns = scope.compact_recent_model_turns;
      this.#state.compact_max_output_tokens = scope.compact_max_output_tokens;
    }
  }

  #setScopeBudget(
    scope: ContextScopeState,
    config: ModelProviderConfig,
    summaryOutputTokens: number
  ): void {
    scope.context_window_tokens = config.contextWindowTokens;
    scope.compact_trigger_tokens = config.compactTriggerTokens;
    scope.compact_recent_model_turns = config.compactRecentModelTurns;
    scope.compact_max_output_tokens = summaryOutputTokens;
  }

  #summaryOutputTokens(config: ModelProviderConfig): number {
    return effectiveContextSummaryOutputTokens(
      config.compactMaxOutputTokens,
      this.#compactorConfig.compactMaxOutputTokens,
      this.#compactorConfig.maxOutputTokens,
      this.#compactorConfig.contextWindowTokens
    );
  }

  #rememberRetainedTail(scope: ContextScopeState, items: AgentInputItem[]): void {
    const retained = items.slice(scope.compacted_item_count);
    scope.retained_item_count = retained.length;
    scope.retained_chain_hash = retained.length === 0 ? null : chainHash(retained);
  }

}

function chooseCutIndex(input: {
  items: AgentInputItem[];
  alreadyCompacted: number;
  recentTurns: number;
  modelData: ModelInputData;
  authority: JsonValue;
  toolTokens: number;
  summaryTokenReserve: number;
  triggerTokens: number;
  tokenEstimatorCorrectionMilli: number;
}): number {
  const starts = modelTurnStarts(input.items)
    .filter((index) => index > input.alreadyCompacted);
  const preferred = input.recentTurns === 0
    ? input.items.length
    : starts.at(-input.recentTurns) ?? input.items.length;
  const boundaries = [...new Set([
    ...starts.filter((index) => index >= preferred),
    input.items.length
  ])].sort((left, right) => left - right);
  const target = Math.floor(input.triggerTokens * 0.82);
  for (const cut of boundaries) {
    if (cut <= input.alreadyCompacted) continue;
    const projected = correctedTokenEstimate(estimateModelInputTokens({
      input: input.items.slice(cut),
      instructions: [
        input.modelData.instructions ?? "",
        JSON.stringify(input.authority)
      ].join("\n")
    }) + input.toolTokens, input.tokenEstimatorCorrectionMilli)
      + input.summaryTokenReserve;
    if (projected <= target) return cut;
  }
  return input.items.length;
}

function correctedTokenEstimate(estimate: number, correctionMilli: number): number {
  return Math.ceil(estimate * correctionMilli / 1_000);
}

/**
 * A cut is only legal immediately before a model turn, or at the end of the
 * current history. That keeps function calls/results and model output grouped
 * instead of slicing arbitrary JSON items merely to satisfy a byte budget.
 */
function compactionBoundaries(
  items: AgentInputItem[],
  after: number,
  through: number
): number[] {
  return [...new Set([...modelTurnStarts(items), items.length])]
    .filter((index) => index > after && index <= through)
    .sort((left, right) => left - right);
}

function modelTurnStarts(items: AgentInputItem[]): number[] {
  const starts: number[] = [];
  for (let index = 0; index < items.length; index += 1) {
    const current = items[index];
    const previous = items[index - 1];
    if (current?.type === "reasoning") {
      if (previous?.type !== "reasoning") starts.push(index);
      continue;
    }
    if (current && "role" in current && current.role === "assistant") {
      starts.push(index);
      continue;
    }
    if (current?.type === "function_call"
      && previous?.type !== "reasoning"
      && previous?.type !== "function_call") {
      starts.push(index);
    }
  }
  return starts;
}

function toolVisibleToNode(name: string, node: TaskNode, rootAgentId: string): boolean {
  // The root Agent object contains only its coordinator tools. The shared
  // worker Agent object contains every possible capability, but isEnabled
  // narrows the provider request to this concrete hierarchy grant.
  if (node.id === rootAgentId) return true;
  if (node.may_delegate) return name === "delegate_agent";
  return node.capabilities.includes(name)
    || name === "complete_assignment"
    || name === "report_blocked";
}

function currentAuthorityItem(content: string): AgentInputItem {
  return {
    type: "message",
    role: "user",
    content
  };
}

function isHarnessAuthorityItem(item: AgentInputItem | undefined): boolean {
  return item?.type === "message"
    && item.role === "user"
    && typeof item.content === "string"
    && item.content.startsWith("CURRENT HARNESS AUTHORITY\n");
}

function withoutHarnessAuthorityItems(items: AgentInputItem[]): {
  items: AgentInputItem[];
  removed: number;
} {
  const semantic = items.filter((item) => !isHarnessAuthorityItem(item));
  return {
    items: semantic,
    removed: items.length - semantic.length
  };
}

/**
 * OpenAI-compatible chat transports require every tool result to belong to
 * the immediately preceding assistant tool-call group. A cancelled SDK turn
 * can leave only one half of that pair in a durable Session. Keep complete
 * groups and discard only the unusable protocol fragments; authoritative
 * action receipts remain in the Harness journals and current authority block.
 */
function normalizeFunctionToolHistory(items: AgentInputItem[]): ToolHistoryNormalization {
  const normalized: AgentInputItem[] = [];
  let incompleteCalls = 0;
  let orphanResults = 0;
  let index = 0;
  while (index < items.length) {
    const item = items[index]!;
    if (!isFunctionToolHistoryItem(item)) {
      normalized.push(item);
      index += 1;
      continue;
    }

    const group: AgentInputItem[] = [];
    while (index < items.length && isFunctionToolHistoryItem(items[index]!)) {
      group.push(items[index]!);
      index += 1;
    }
    const retained = new Set<number>();
    const pendingCalls = new Map<string, number>();
    for (let groupIndex = 0; groupIndex < group.length; groupIndex += 1) {
      const candidate = group[groupIndex]!;
      const callId = functionToolCallId(candidate);
      if (candidate.type === "function_call") {
        if (callId !== undefined) pendingCalls.set(callId, groupIndex);
        continue;
      }
      if (candidate.type !== "function_call_result") continue;
      const callIndex = callId === undefined ? undefined : pendingCalls.get(callId);
      const call = callIndex === undefined ? undefined : group[callIndex];
      if (callIndex === undefined || call?.type !== "function_call"
        || (candidate.name !== undefined && call.name !== candidate.name)) {
        orphanResults += 1;
        continue;
      }
      retained.add(callIndex);
      retained.add(groupIndex);
      pendingCalls.delete(callId!);
    }
    incompleteCalls += group.reduce((count, candidate, groupIndex) =>
      candidate.type === "function_call" && !retained.has(groupIndex)
        ? count + 1
        : count, 0);
    normalized.push(...group.filter((_candidate, groupIndex) => retained.has(groupIndex)));
  }
  return {
    items: normalized,
    removed: items.length - normalized.length,
    incompleteCalls,
    orphanResults
  };
}

function isFunctionToolHistoryItem(item: AgentInputItem): boolean {
  return item.type === "function_call" || item.type === "function_call_result";
}

function functionToolCallId(item: AgentInputItem): string | undefined {
  return "callId" in item && typeof item.callId === "string" && item.callId.length > 0
    ? item.callId
    : undefined;
}

function chainHash(items: AgentInputItem[]): string {
  let chain = HASH_SEED;
  for (const item of items) {
    chain = createHash("sha256")
      .update(chain)
      .update("\0")
      .update(JSON.stringify(item))
      .digest("hex");
  }
  return chain;
}

function commonPrefixLength(left: AgentInputItem[], right: AgentInputItem[]): number {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && JSON.stringify(left[index]) === JSON.stringify(right[index])) index += 1;
  return index;
}

function isPrefix(prefix: AgentInputItem[], value: AgentInputItem[]): boolean {
  return prefix.length <= value.length && commonPrefixLength(prefix, value) === prefix.length;
}

function sameItems(left: AgentInputItem[], right: AgentInputItem[]): boolean {
  return left.length === right.length && isPrefix(left, right);
}

function asRecord(value: JsonValue): Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value
    : {};
}

function relevantTransactionIds(
  receipts: Record<string, { accepted: boolean; worldRevision: number }>,
  priorSummary: ContextScopeState["summary"],
  sourceItems: AgentInputItem[]
): string[] {
  const source = JSON.stringify({ priorSummary, sourceItems });
  return Object.keys(receipts).filter((transactionId) =>
    source.includes(transactionId)
  );
}

function summaryTransactionIds(summary: NonNullable<ContextScopeState["summary"]>): string[] {
  return [...new Set([
    ...summary.completed.flatMap((item) => item.transaction_ids),
    ...summary.blockers.flatMap((item) => item.transaction_ids)
  ])];
}

function compactionFailureUsage(error: unknown): ContextSummaryResult["usage"] | undefined {
  const pending: unknown[] = [error];
  const seen = new Set<unknown>();
  while (pending.length > 0) {
    const current = pending.shift();
    if (current === null || (typeof current !== "object" && typeof current !== "function")) {
      continue;
    }
    if (seen.has(current)) continue;
    seen.add(current);
    if (current instanceof ContextCompactionInterruption && current.usage) {
      return current.usage;
    }
    const record = current as Record<string, unknown>;
    if (record.cause !== undefined) pending.push(record.cause);
  }
  return undefined;
}

function json(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}

function authorityIdentifierBlock(authority: JsonValue): string[] {
  if (!isJsonRecord(authority)) return [];
  const goalContext = isJsonRecord(authority.goal_context)
    ? authority.goal_context
    : undefined;
  const goalDAG = isJsonRecord(authority.goal_dag)
    ? authority.goal_dag
    : undefined;
  const evidenceRef = typeof goalContext?.evidence_ref === "string"
    ? goalContext.evidence_ref
    : undefined;
  const candidates = isJsonRecord(goalDAG?.candidates)
    ? Object.keys(goalDAG.candidates).sort()
    : [];
  const currentEpochId = typeof goalDAG?.current_epoch_id === "string"
    ? goalDAG.current_epoch_id
    : null;
  const execution = isJsonRecord(authority.execution_authority)
    ? authority.execution_authority
    : undefined;
  const planningTransactionId = typeof execution?.planning_transaction_id === "string"
    ? execution.planning_transaction_id
    : undefined;
  const planningAction = typeof execution?.planning_action === "string"
    ? execution.planning_action
    : undefined;
  const executorAction = typeof execution?.executor_action === "string"
    ? execution.executor_action
    : undefined;
  const worldFrame = typeof authority.world_frame === "number"
    && Number.isSafeInteger(authority.world_frame)
    ? authority.world_frame
    : undefined;
  const worldRevision = typeof authority.world_revision === "number"
    && Number.isSafeInteger(authority.world_revision)
    ? authority.world_revision
    : undefined;
  const hasCurrentWorld = worldFrame !== undefined && worldRevision !== undefined;
  if (!evidenceRef
    && candidates.length === 0
    && currentEpochId === null
    && !planningTransactionId
    && !hasCurrentWorld) return [];
  return [
    "CURRENT EXACT IDENTIFIERS (copy values character-for-character; never invent aliases)",
    ...(hasCurrentWorld
      ? [
          `current_world_frame=${JSON.stringify(worldFrame)}`,
          `current_world_revision=${JSON.stringify(worldRevision)}`,
          `coordinator_phase=${JSON.stringify(authority.coordinator_phase ?? null)}`
        ]
      : []),
    `goal_evidence_ref=${JSON.stringify(evidenceRef ?? null)}`,
    `existing_goal_candidate_ids=${JSON.stringify(candidates)}`,
    `current_goal_epoch_id=${JSON.stringify(currentEpochId)}`,
    ...(planningTransactionId
      ? [
          `pending_planning_action=${JSON.stringify(planningAction ?? null)}`,
          `pending_planning_transaction_id=${JSON.stringify(planningTransactionId)}`,
          `required_executor_action=${JSON.stringify(executorAction ?? null)}`
        ]
      : [])
  ];
}

function isJsonRecord(value: unknown): value is Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
