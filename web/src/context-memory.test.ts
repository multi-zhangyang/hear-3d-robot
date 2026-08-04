import { describe, expect, it } from "vitest";
import type { ContextMemoryState, ContextScopeState } from "./types";
import { activeContextUsage } from "./context-memory";

describe("active context budget", () => {
  it("uses the restored active scope instead of the coordinator envelope", () => {
    const memory = contextMemory({
      ...contextScope(),
      active_estimated_tokens: 4_000,
      context_window_tokens: 32_768,
      compact_trigger_tokens: 8_000,
      compact_recent_model_turns: 2,
      compact_max_output_tokens: 768
    });

    expect(activeContextUsage(memory)).toEqual({
      scopeId: "worker",
      activeEstimatedTokens: 4_000,
      contextWindowTokens: 32_768,
      compactTriggerTokens: 8_000,
      compactRecentModelTurns: 2,
      compactMaxOutputTokens: 768,
      loadFraction: 0.5
    });
  });

  it("reads an old v1 scope through its recorded global budget without mutating it", () => {
    const scope = contextScope();
    const memory = contextMemory(scope);
    const before = structuredClone(memory);

    expect(activeContextUsage(memory)).toMatchObject({
      scopeId: "worker",
      activeEstimatedTokens: 3_000,
      contextWindowTokens: 65_536,
      compactTriggerTokens: 12_000,
      compactRecentModelTurns: 4,
      compactMaxOutputTokens: 2_048,
      loadFraction: 0.25
    });
    expect(memory).toEqual(before);
    expect(memory.scopes.worker).not.toHaveProperty("compact_trigger_tokens");
  });
});

function contextMemory(scope: ContextScopeState): ContextMemoryState {
  return {
    version: 1,
    context_window_tokens: 65_536,
    compact_trigger_tokens: 12_000,
    compact_recent_model_turns: 4,
    compact_max_output_tokens: 2_048,
    active_scope_id: scope.scope_id,
    active_estimated_tokens: 111,
    total_compactions: 0,
    last_compacted_at: null,
    scopes: { [scope.scope_id]: scope }
  };
}

function contextScope(): ContextScopeState {
  return {
    scope_id: "worker",
    agent_id: "worker",
    agent_name: "工作智能体",
    raw_item_count: 0,
    raw_chain_hash: null,
    compacted_item_count: 0,
    retained_item_count: 0,
    retained_chain_hash: null,
    active_estimated_tokens: 3_000,
    compaction_count: 0,
    summary: null,
    summary_origin: null,
    summary_world_revision: null,
    last_compacted_at: null
  };
}
