const SENSITIVE_FIELD = /^(?:authorization|proxy-authorization|x-api-key|api[-_]?key|access[-_]?token|refresh[-_]?token|secret|password|cookie|set-cookie)$/i;

export function errorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  const normalized = normalize(error, new WeakSet<object>());
  if (typeof normalized === "string") return normalized;
  try {
    const serialized = JSON.stringify(normalized);
    if (serialized && serialized !== "{}") return serialized;
  } catch {
    // The recursive normalizer removes cycles before serialization.
  }
  return Object.prototype.toString.call(error);
}

function normalize(value: unknown, seen: WeakSet<object>): unknown {
  if (value === null || value === undefined) return value ?? null;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "symbol" || typeof value === "function") return String(value);
  if (typeof value !== "object") return String(value);
  if (seen.has(value)) return "[circular]";
  seen.add(value);

  if (value instanceof Date) return value.toISOString();
  if (Array.isArray(value)) return value.map((item) => normalize(item, seen));
  const result: Record<string, unknown> = {};
  if (value instanceof Error) {
    result.name = value.name;
    result.message = value.message;
    if (value.cause !== undefined) result.cause = normalize(value.cause, seen);
    // SDK errors can expose an enumerable `state` containing the entire agent,
    // prompt, tool schemas, and generated history. Persisting that object made
    // one compaction failure several megabytes large. Keep only small transport
    // metadata that is useful for diagnosis; the full SDK state already lives
    // in the dedicated run artifacts.
    for (const key of ["status", "statusCode", "code", "type", "request_id", "requestId"] as const) {
      const item = (value as unknown as Record<string, unknown>)[key];
      if (item !== undefined) result[key] = normalize(item, seen);
    }
    return result;
  }
  for (const [key, item] of Object.entries(value)) {
    result[key] = SENSITIVE_FIELD.test(key) ? "[redacted]" : normalize(item, seen);
  }
  return result;
}
