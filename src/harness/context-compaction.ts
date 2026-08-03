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
import type { ProviderConfig } from "../config/load.js";
import type { FileSession } from "../persistence/file-session.js";
import type { LongRunContextRuntime } from "./context-runtime.js";
import { compactorInputTokenLimit } from "../runtime/context-budget.js";
import { agentIdFromModelPayload, agentInvocationMarker } from "./agent-scope.js";
import {
  ContextCompactionInterruption,
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

const HASH_SEED = "hear-context-ledger-v1";

interface RecoveredInputState {
  physical: AgentInputItem[];
  logical: AgentInputItem[];
}

interface RecoveredHistory {
  items: AgentInputItem[];
  physicalBase?: {
    count: number;
    hash: string;
    logical: AgentInputItem[];
  };
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
  readonly #config: ProviderConfig;
  readonly #sessionForAgent: ((agentId: string) => FileSession) | undefined;
  readonly #recoveredInputs = new Map<string, RecoveredInputState>();
  readonly #freshTurnScopes = new Set<string>();
  readonly #freshTurnRebaseScopes = new Set<string>();
  #memoryPersistence: Promise<void> = Promise.resolve();
  #state: ContextMemoryState;

  constructor(input: {
    runtime: LongRunContextRuntime;
    createGenerator: (agentId: string) => ContextSummaryGenerator;
    provider: ProviderConfig;
    sessionForAgent?: (agentId: string) => FileSession;
  }) {
    this.#runtime = input.runtime;
    this.#createGenerator = input.createGenerator;
    this.#config = input.provider;
    this.#sessionForAgent = input.sessionForAgent;
    const persisted = input.runtime.contextMemory();
    this.#state = ContextMemoryStateSchema.parse({
      ...persisted,
      context_window_tokens: input.provider.contextWindowTokens,
      compact_trigger_tokens: input.provider.compactTriggerTokens,
      compact_recent_model_turns: input.provider.compactRecentModelTurns,
      compact_max_output_tokens: input.provider.compactMaxOutputTokens
    });
  }

  get snapshot(): ContextMemoryState {
    return structuredClone(this.#state);
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
      const compactedPrefix = recovered.physical.length - recovered.logical.length;
      if (compactedPrefix <= 0) continue;
      const session = sessionForAgent(scopeId);
      const persisted = await session.getItems();
      if (isPrefix(recovered.physical, persisted)) {
        const retained = persisted.slice(compactedPrefix);
        await session.replaceItems(retained);
        this.#recoveredInputs.set(scopeId, {
          physical: structuredClone(retained),
          logical: structuredClone(retained)
        });
      } else if (isPrefix(recovered.logical, persisted)) {
        this.#recoveredInputs.set(scopeId, {
          physical: structuredClone(persisted),
          logical: structuredClone(persisted)
        });
      }
    }
  }

  /**
   * Starts a clean SDK conversation after repeated responses without a formal
   * decision. Durable raw history stays in context.jsonl and the last validated
   * model checkpoint remains available; only the poisoned short-term branch is
   * replaced on the next filter call.
   */
  startFreshSdkTurn(agentId: string): void {
    this.#recoveredInputs.delete(agentId);
    this.#freshTurnScopes.add(agentId);
    this.#freshTurnRebaseScopes.add(agentId);
  }

  readonly filter: CallModelInputFilter = async ({
    modelData,
    agent
  }): Promise<ModelInputData> => {
    this.#runtime.signal?.throwIfAborted();
    const agentId = agentIdFromModelPayload(modelData.input, this.#runtime.rootAgentId);
    const node = this.#runtime.activeNode(agentId);
    const scope = this.#scope(node);
    const physicalInput = modelData.input;
    const logicalInput = await this.#logicalInput(scope, physicalInput);
    const logicalModelData: ModelInputData = {
      input: logicalInput,
      ...(modelData.instructions === undefined ? {} : { instructions: modelData.instructions })
    };
    const toolTokens = estimateToolTokens(agent.tools.filter((tool) =>
      toolVisibleToNode(tool.name, node, this.#runtime.rootAgentId)
    ));
    const rawUpdate = this.#recordRawDelta(scope, logicalInput);
    const authority = this.#runtime.contextAnchor(node.id);
    const rebaseUpdate = this.#rebaseSummary(scope);

    let filtered = this.#render(logicalModelData, scope, authority);
    scope.active_estimated_tokens = estimateModelInputTokens(filtered) + toolTokens;
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

    while (scope.active_estimated_tokens > this.#config.compactTriggerTokens) {
      const compacted = await this.#compact(logicalModelData, node, scope, authority, toolTokens);
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
        logical: structuredClone(retained)
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
      await this.#compactPersistedSession(scope.scope_id);
      persisted = true;
    }

    const hardLimit = this.#config.contextWindowTokens - this.#config.maxOutputTokens;
    if (scope.active_estimated_tokens > hardLimit) {
      throw new ContextCompactionInterruption(
        `Active context for ${agent.name} is estimated at ${scope.active_estimated_tokens} tokens, `
        + `above the configured ${hardLimit}-token input limit after compaction`
      );
    }
    if (!persisted) await this.#persist();
    return filtered;
  };

  #scope(node: TaskNode): ContextScopeState {
    const existing = this.#state.scopes[node.id];
    if (existing) {
      existing.agent_name = node.name;
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
    // An explicit fresh turn may deliberately start with the same mission
    // prompt as the abandoned branch. Prefix equality is not permission to
    // rehydrate that branch; #recordRawDelta will preserve only its validated
    // compact checkpoint and begin a new raw chain.
    if (this.#freshTurnScopes.has(scope.scope_id)) return physical;
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
        logical: structuredClone(logical)
      });
      await this.#compactPersistedSession(scope.scope_id);
      return logical;
    }
    const history = recoveredHistory.items;
    const shared = commonPrefixLength(history, physical);
    if (shared === 0) return physical;
    const logical = [
      ...structuredClone(history),
      ...structuredClone(physical.slice(shared))
    ];
    this.#recoveredInputs.set(scope.scope_id, {
      physical: structuredClone(physical),
      logical: structuredClone(logical)
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

  async #compactPersistedSession(scopeId: string): Promise<void> {
    const sessionForAgent = this.#sessionForAgent;
    const recovered = this.#recoveredInputs.get(scopeId);
    if (!sessionForAgent || !recovered) return;
    const compactedPrefix = recovered.physical.length - recovered.logical.length;
    if (compactedPrefix <= 0) return;
    const session = sessionForAgent(scopeId);
    const persisted = await session.getItems();
    const physicalStart = sequencePrefixIndex(recovered.physical, persisted);
    if (physicalStart < 0) return;
    const remove = Math.min(
      persisted.length,
      Math.max(0, compactedPrefix - physicalStart)
    );
    if (remove > 0) await session.replaceItems(persisted.slice(remove));
  }

  #recordRawDelta(scope: ContextScopeState, items: AgentInputItem[]): JsonValue | undefined {
    const explicitFreshTurn = this.#freshTurnScopes.delete(scope.scope_id);
    let branch = explicitFreshTurn;
    const priorCount = scope.raw_item_count;
    const priorHash = scope.raw_chain_hash;
    if (explicitFreshTurn || priorCount > items.length
      || (priorCount > 0 && chainHash(items.slice(0, priorCount)) !== priorHash)) {
      branch = true;
      const preserveCheckpoint = explicitFreshTurn;
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
    toolTokens: number
  ): Promise<{ modelData: ModelInputData; journalRecord: JsonValue } | null> {
    const targetCut = chooseCutIndex({
      items: modelData.input,
      alreadyCompacted: scope.compacted_item_count,
      recentTurns: this.#config.compactRecentModelTurns,
      modelData,
      authority,
      toolTokens,
      summaryTokenReserve: this.#config.compactMaxOutputTokens,
      triggerTokens: this.#config.compactTriggerTokens
    });
    if (targetCut <= scope.compacted_item_count) return null;

    const maxInputTokens = compactorInputTokenLimit(
      this.#config.contextWindowTokens,
      this.#config.compactMaxOutputTokens
    );
    if (maxInputTokens <= 0) {
      throw new ContextCompactionInterruption(
        "Context compactor has no input budget in the configured model window"
      );
    }
    let selected: {
      cut: number;
      request: ContextSummaryRequest;
      estimatedInputTokens: number;
    } | undefined;
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
      const { worldRevision: currentWorldRevision } = this.#runtime.contextWorldIdentity();
      const request: ContextSummaryRequest = {
        priorSummary: scope.summary,
        sourceItems,
        authority,
        acceptedTransactionIds: transactionIds.filter((transactionId) =>
          receipts[transactionId]?.accepted === true
            && receipts[transactionId]?.worldRevision === currentWorldRevision
        ),
        blockerTransactionIds: transactionIds.filter((transactionId) =>
          receipts[transactionId]?.worldRevision === currentWorldRevision
        ),
        maxInputTokens,
        ...(this.#runtime.signal ? { signal: this.#runtime.signal } : {})
      };
      const estimatedInputTokens = estimateContextSummaryRequestTokens(request);
      smallestCandidateTokens ??= estimatedInputTokens;
      if (estimatedInputTokens <= maxInputTokens) {
        selected = { cut, request, estimatedInputTokens };
      }
    }
    if (!selected) {
      throw new ContextCompactionInterruption(
        `The next complete context turn for ${node.name} requires an estimated `
        + `${smallestCandidateTokens ?? "unknown"} compactor input tokens, above its `
        + `${maxInputTokens}-token hard limit; no oversized request was sent`
      );
    }

    const { cut, request, estimatedInputTokens } = selected;
    const sourceItems = request.sourceItems;
    const sourceFrom = scope.compacted_item_count;
    const startedAt = new Date().toISOString();
    await this.#runtime.recordCompactionModelCall(node.id);

    let generated: ContextSummaryResult;
    let reconciledRequests = 1;
    try {
      generated = await this.#generatorFor(node.id).generate(request);
      this.#runtime.signal?.throwIfAborted();
      if (generated.usage.requests > 1) {
        await this.#runtime.reconcileCompactionModelCalls(
          node.id,
          generated.usage.requests - 1
        );
        reconciledRequests = generated.usage.requests;
      }
      if (generated.origin !== "model") {
        throw new ContextCompactionInterruption(
          "Context Compactor returned a synthetic checkpoint instead of model-written memory",
          { usage: generated.usage }
        );
      }
      this.#runtime.assertContextSummaryEvidence(generated.summary);
    } catch (error) {
      this.#runtime.signal?.throwIfAborted();
      const usage = compactionFailureUsage(error);
      if (usage && usage.requests > reconciledRequests) {
        await this.#runtime.reconcileCompactionModelCalls(
          node.id,
          usage.requests - reconciledRequests
        );
      }
      const interruption = error instanceof ContextCompactionInterruption
        ? error
        : new ContextCompactionInterruption(
            error instanceof Error ? error.message : String(error),
            { cause: error, ...(usage ? { usage } : {}) }
          );
      await this.#runtime.recordProvider(json({
        status: "context_compaction_interrupted",
        source: "agents_sdk",
        scope_id: scope.scope_id,
        agent_id: node.id,
        error: interruption instanceof Error ? interruption.message : String(interruption),
        recoverable: true,
        raw_history_preserved: true,
        session_trimmed: false
      }));
      throw interruption;
    }

    const completedAt = new Date().toISOString();
    const world = this.#runtime.contextWorldIdentity();
    scope.compacted_item_count = cut;
    scope.compaction_count += 1;
    scope.summary = generated.summary;
    scope.summary_origin = generated.origin;
    scope.summary_world_revision = world.worldRevision;
    scope.last_compacted_at = completedAt;
    this.#state.total_compactions += 1;
    this.#state.last_compacted_at = completedAt;

    const filtered = this.#render(modelData, scope, authority);
    scope.active_estimated_tokens = estimateModelInputTokens(filtered) + toolTokens;
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
      compactor_input_limit_tokens: maxInputTokens,
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
        compactor_input_limit_tokens: maxInputTokens,
        retained_items: modelData.input.length - cut,
        active_estimated_tokens: scope.active_estimated_tokens,
        world_revision: world.worldRevision,
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
    const authorityBlock = [
      "CURRENT HARNESS AUTHORITY",
      "This block is rebuilt from the live checkpoint for every model request. It overrides compact memory and older observations.",
      JSON.stringify(authority)
    ].join("\n");
    if (!scope.summary) return {
      input: modelData.input,
      instructions: [modelData.instructions, identity, authorityBlock]
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
      input: modelData.input.slice(scope.compacted_item_count),
      instructions: [modelData.instructions, identity, authorityBlock, checkpoint]
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
    scope: ContextScopeState
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
    const forceFreshRebase = this.#freshTurnRebaseScopes.delete(scope.scope_id);
    if (!scope.summary) return undefined;
    const receipts = this.#runtime.contextReceipts();
    const { worldRevision: currentWorldRevision } = this.#runtime.contextWorldIdentity();
    const evidenceIds = summaryTransactionIds(scope.summary);
    const staleIds = evidenceIds.filter((transactionId) =>
      receipts[transactionId]?.worldRevision !== currentWorldRevision
    );
    const revisionChanged = scope.summary_world_revision !== currentWorldRevision;
    if (!forceFreshRebase && !revisionChanged && staleIds.length === 0) return undefined;

    const currentIds = evidenceIds.filter((transactionId) =>
      receipts[transactionId]?.worldRevision === currentWorldRevision
    );
    const acceptedIds = currentIds.filter((transactionId) =>
      receipts[transactionId]?.accepted === true
    );
    const reasons = [
      ...(forceFreshRebase ? ["fresh_sdk_turn"] : []),
      ...(revisionChanged ? ["world_identity_changed"] : []),
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
    const projected = estimateModelInputTokens({
      input: input.items.slice(cut),
      instructions: [
        input.modelData.instructions ?? "",
        JSON.stringify(input.authority)
      ].join("\n")
    }) + input.toolTokens + input.summaryTokenReserve;
    if (projected <= target) return cut;
  }
  return input.items.length;
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

function sequencePrefixIndex(value: AgentInputItem[], prefix: AgentInputItem[]): number {
  if (prefix.length === 0) return value.length;
  const maximum = value.length - prefix.length;
  // Session rows can repeat byte-for-byte across model turns. The persisted
  // Session is the newest matching branch, so choosing an older occurrence can
  // trim live rows that merely resemble an archived prefix. Search backwards;
  // ambiguity then keeps extra context instead of deleting current context.
  for (let start = maximum; start >= 0; start -= 1) {
    let matches = true;
    for (let offset = 0; offset < prefix.length; offset += 1) {
      if (JSON.stringify(value[start + offset]) !== JSON.stringify(prefix[offset])) {
        matches = false;
        break;
      }
    }
    if (matches) return start;
  }
  return -1;
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
