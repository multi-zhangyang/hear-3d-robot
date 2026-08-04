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

function estimateTextTokens(value: string): number {
  if (value.length === 0) return 0;
  // Structured tool traffic contains hashes, decimals and JSON punctuation,
  // which is materially denser than prose for modern tokenizers. This
  // provider-neutral baseline is calibrated further from reported usage.
  return Math.max(1, Math.ceil(Buffer.byteLength(value, "utf8") * 4 / 9));
}

function estimateJsonTokens(value: unknown): number {
  return estimateTextTokens(JSON.stringify(value));
}
