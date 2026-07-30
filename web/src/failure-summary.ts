/**
 * Runs record failures with `errorMessage()`, which deliberately keeps the whole
 * structured error — nested causes, status codes, socket counters. That is the
 * right thing to persist, because it is the evidence you need when a run dies
 * halfway. It is the wrong thing to paste into a banner: a wall of JSON tells a
 * reader nothing about what broke.
 *
 * So the banner reads the same string and reduces it to a headline plus the
 * fields worth naming, keeping the raw text available underneath. Nothing here
 * knows about any particular provider; it walks the shapes that JavaScript
 * errors and HTTP responses already have.
 */

export interface FailureSummary {
  /** One line, safe to render on its own. */
  headline: string;
  /** Named facts worth surfacing next to the headline — status, code, cause. */
  facts: { label: string; value: string }[];
  /** The original string, shown only when the reader asks for it. */
  raw: string;
  /** False when the stored text was already a plain sentence. */
  structured: boolean;
}

const MAX_HEADLINE = 200;

/**
 * A `cause` chain runs from general to specific: a fetch that lost its socket
 * surfaces as `TypeError: terminated` with the actual reason buried one or two
 * levels down. The innermost link that carries a message is the one a reader
 * wants, so the walk keeps descending and remembers the last useful message.
 */
export function summarizeFailure(error: string): FailureSummary {
  const trimmed = error.trim();
  const parsed = parseObject(trimmed);
  if (!parsed) {
    return { headline: truncate(trimmed), facts: [], raw: trimmed, structured: false };
  }

  const facts: { label: string; value: string }[] = [];
  const status = firstOf(parsed, ["status", "statusCode", "status_code"]);
  if (status !== undefined) facts.push({ label: "状态", value: String(status) });

  let node: Record<string, unknown> | undefined = parsed;
  let headline = "";
  let code: string | undefined;
  let depth = 0;
  while (node && depth < 8) {
    const message = readString(node, "message") ?? readString(node, "reason");
    if (message) headline = message;
    const nodeCode = readString(node, "code");
    if (nodeCode) code = nodeCode;
    const nested: unknown = node.error ?? node.cause;
    node = typeof nested === "object" && nested !== null ? (nested as Record<string, unknown>) : undefined;
    depth += 1;
  }

  if (!headline) headline = readString(parsed, "name") ?? truncate(trimmed);
  if (code) facts.push({ label: "代码", value: code });
  const type = readString(parsed, "type");
  if (type) facts.push({ label: "类型", value: type });

  return { headline: truncate(headline), facts, raw: trimmed, structured: true };
}

function parseObject(text: string): Record<string, unknown> | undefined {
  if (!text.startsWith("{")) return undefined;
  try {
    const value: unknown = JSON.parse(text);
    return typeof value === "object" && value !== null && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : undefined;
  } catch {
    // A run can also fail with a plain sentence that happens to start with a
    // brace; treating it as prose is the safe reading.
    return undefined;
  }
}

function readString(node: Record<string, unknown>, key: string): string | undefined {
  const value = node[key];
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function firstOf(node: Record<string, unknown>, keys: string[]): string | number | undefined {
  for (const key of keys) {
    const value = node[key];
    if (typeof value === "number" || typeof value === "string") return value;
  }
  return undefined;
}

function truncate(text: string): string {
  return text.length > MAX_HEADLINE ? `${text.slice(0, MAX_HEADLINE - 1)}…` : text;
}
