import type { ContextMemoryState, ContextScopeState } from "./types";

export interface ActiveContextUsage {
  scopeId: string | null;
  activeEstimatedTokens: number;
  contextWindowTokens: number;
  compactTriggerTokens: number;
  compactRecentModelTurns: number;
  compactMaxOutputTokens: number;
  loadFraction: number;
}

export function activeContextUsage(memory: ContextMemoryState): ActiveContextUsage {
  const scope = memory.active_scope_id
    ? memory.scopes[memory.active_scope_id]
    : undefined;
  const budget = hasCompleteBudget(scope) ? {
    contextWindowTokens: scope.context_window_tokens,
    compactTriggerTokens: scope.compact_trigger_tokens,
    compactRecentModelTurns: scope.compact_recent_model_turns,
    compactMaxOutputTokens: scope.compact_max_output_tokens
  } : {
    contextWindowTokens: memory.context_window_tokens,
    compactTriggerTokens: memory.compact_trigger_tokens,
    compactRecentModelTurns: memory.compact_recent_model_turns,
    compactMaxOutputTokens: memory.compact_max_output_tokens
  };
  const activeEstimatedTokens = scope?.active_estimated_tokens
    ?? memory.active_estimated_tokens;
  return {
    scopeId: scope?.scope_id ?? memory.active_scope_id,
    activeEstimatedTokens,
    ...budget,
    loadFraction: Math.min(
      1,
      Math.max(0, activeEstimatedTokens / budget.compactTriggerTokens)
    )
  };
}

function hasCompleteBudget(scope: ContextScopeState | undefined): scope is ContextScopeState & {
  context_window_tokens: number;
  compact_trigger_tokens: number;
  compact_recent_model_turns: number;
  compact_max_output_tokens: number;
} {
  return scope?.context_window_tokens !== undefined
    && scope.compact_trigger_tokens !== undefined
    && scope.compact_recent_model_turns !== undefined
    && scope.compact_max_output_tokens !== undefined;
}
