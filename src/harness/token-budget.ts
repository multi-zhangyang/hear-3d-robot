import type { AgentInputItem, ModelInputData } from "@openai/agents";

interface BudgetedTool {
  name: string;
  description?: string;
  parameters?: unknown;
  strict?: boolean;
}

export function estimateModelInputTokens(data: ModelInputData): number {
  return estimateTextTokens(data.instructions ?? "") + estimateItemsTokens(data.input);
}

export function estimateItemsTokens(items: AgentInputItem[]): number {
  return items.reduce((total, item) => total + 8 + estimateJsonTokens(item), 0);
}

/**
 * Tool definitions are sent on every provider request even though an Agents
 * SDK input filter only exposes instructions and items. Include the actual
 * JSON schema when available; the fixed framing reserve covers provider field
 * names and message/tool wrappers that are not represented by the SDK object.
 */
export function estimateToolTokens(tools: readonly BudgetedTool[]): number {
  return tools.reduce((total, tool) => total + 128 + estimateJsonTokens({
    type: "function",
    name: tool.name,
    description: tool.description ?? "",
    ...(tool.parameters === undefined ? {} : { parameters: tool.parameters }),
    ...(tool.strict === undefined ? {} : { strict: tool.strict })
  }), 0);
}

export function estimateTextTokens(value: string): number {
  if (value.length === 0) return 0;
  // Provider-neutral and deliberately conservative: English/JSON normally
  // average nearer four bytes per token, while CJK stays close to this bound.
  return Math.max(1, Math.ceil(Buffer.byteLength(value, "utf8") / 3));
}

function estimateJsonTokens(value: unknown): number {
  return estimateTextTokens(JSON.stringify(value));
}
