import type { ZodType } from "zod";

const INVALID_TOOL_INPUT_ERROR = "InvalidToolInputError";

interface ValidationIssue {
  path: string;
  code: string;
  message: string;
}

interface InvalidInputAttempt {
  fingerprint: string;
  count: number;
}

export interface ToolInputRecovery {
  preflight(
    input: string,
    schema: ZodType,
    toolName: string
  ): string | undefined;
}

export function recoverInvalidToolInputOutput(
  output: string,
  input: string,
  schema: ZodType,
  toolName: string,
  recovery: ToolInputRecovery
): string {
  const parsed = jsonObject(output);
  if (parsed?.accepted !== false
    || parsed.tool !== toolName
    || (parsed.code !== "invalid_tool_input"
      && parsed.code !== "repeated_invalid_tool_input")) {
    return output;
  }
  const recovered = recovery.preflight(input, schema, toolName);
  if (recovered === undefined) return output;
  return JSON.stringify({ ...parsed, ...jsonObject(recovered) });
}

export function createToolInputRecovery(): ToolInputRecovery {
  let previousAttempt: InvalidInputAttempt | undefined;
  return {
    preflight(input, schema, toolName) {
      const result = validateToolInput(input, schema);
      if (result.valid) {
        previousAttempt = undefined;
        return undefined;
      }
      const fingerprint = invalidInputFingerprint(
        toolName,
        result.decoded,
        result.normalizedInput,
        result.issues
      );
      const count = previousAttempt?.fingerprint === fingerprint
        ? previousAttempt.count + 1
        : 1;
      previousAttempt = { fingerprint, count };
      return invalidToolInputResultForIssues(
        toolName,
        result.issues,
        count > 1
          ? {
              count,
              fields: repeatedInvalidFields(result.decoded, result.issues)
            }
          : undefined
      );
    }
  };
}

/**
 * Keeps the Agents SDK's native model-correction loop for malformed tool
 * arguments while allowing every execution, transport, and abort failure to
 * retain its normal failure semantics.
 */
export function invalidToolInputResult(error: unknown, toolName: string): string {
  if (!matchesInvalidToolInput(error, toolName)) throw error;
  return invalidToolInputResultForIssues(
    toolName,
    objectRecord(objectRecord(error)?.originalError)?.issues
  );
}

/**
 * Agent.asTool captures its default error handler when the SDK tool is built,
 * so changing the public errorFunction later cannot recover schema failures.
 * Validate the same Zod contract at the public invoke boundary and reject the
 * model call without entering the delegated agent.
 */
export function preflightAgentToolInput(
  input: string,
  schema: ZodType,
  toolName: string
): string | undefined {
  const result = validateToolInput(input, schema);
  return result.valid
    ? undefined
    : invalidToolInputResultForIssues(toolName, result.issues);
}

function invalidToolInputResultForIssues(
  toolName: string,
  issues: unknown,
  repeated?: {
    count: number;
    fields: Array<{ path: string; value: string | number | boolean | null }>;
  }
): string {
  const normalizedIssues = validationIssues(issues);
  return JSON.stringify({
    accepted: false,
    code: repeated ? "repeated_invalid_tool_input" : "invalid_tool_input",
    tool: toolName,
    error: "The tool arguments were not valid for the declared schema.",
    validation_issues: normalizedIssues,
    ...(repeated
      ? {
          repeated_attempt: {
            count: repeated.count,
            invalid_fields: repeated.fields
          }
        }
      : {}),
    automatic_actuation: false,
    next_response_contract: {
      mode: "corrected_tool_call_only",
      tool: toolName,
      preserve_valid_fields: true,
      narration_allowed: false
    },
    recovery: repeated
      ? "Your next response must call this same tool once with materially corrected complete arguments. Preserve valid fields unless a listed correction requires a dependent change. Do not narrate, restate the schema, recalculate unrelated fields, or submit these same arguments again. The harness will not repair or replace rejected values."
      : "Your next response must call this same tool once with complete arguments that resolve every validation issue. Preserve valid fields unless a listed correction requires a dependent change. Do not narrate, restate the schema, or recalculate unrelated fields. The harness will not repair the rejected arguments."
  });
}

function validateToolInput(
  input: string,
  schema: ZodType
): {
  valid: true;
} | {
  valid: false;
  decoded: unknown;
  normalizedInput: string;
  issues: ValidationIssue[];
} {
  let decoded: unknown;
  try {
    decoded = JSON.parse(input);
  } catch {
    return {
      valid: false,
      decoded: undefined,
      normalizedInput: input,
      issues: [{
        path: "",
        code: "invalid_json",
        message: "Tool arguments must be valid JSON."
      }]
    };
  }
  const result = schema.safeParse(decoded);
  if (result.success) return { valid: true };
  return {
    valid: false,
    decoded,
    normalizedInput: stableJson(decoded),
    issues: validationIssues(result.error.issues)
  };
}

function invalidInputFingerprint(
  toolName: string,
  decoded: unknown,
  normalizedInput: string,
  issues: readonly ValidationIssue[]
): string {
  const issueIdentity = issues
    .map((issue) => {
      const invalidValue = issue.path
        ? valueAtPath(decoded, issue.path)
        : undefined;
      const valueIdentity = issue.path
        ? invalidValue === undefined ? "<missing>" : stableJson(invalidValue)
        : normalizedInput;
      return `${issue.path}\u0000${issue.code}\u0000${issue.message}`
        + `\u0000${valueIdentity}`;
    })
    .sort()
    .join("\u0001");
  return `${toolName}\u0002${issueIdentity}`;
}

function repeatedInvalidFields(
  decoded: unknown,
  issues: readonly ValidationIssue[]
): Array<{ path: string; value: string | number | boolean | null }> {
  if (decoded === undefined) return [];
  const fields: Array<{
    path: string;
    value: string | number | boolean | null;
  }> = [];
  for (const issue of issues) {
    if (!issue.path) continue;
    const value = valueAtPath(decoded, issue.path);
    if (value !== null
      && typeof value !== "string"
      && typeof value !== "number"
      && typeof value !== "boolean") continue;
    fields.push({ path: issue.path, value });
  }
  return fields;
}

function valueAtPath(value: unknown, path: string): unknown {
  let current = value;
  for (const part of path.split(".")) {
    if (current === null || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(",")}]`;
  }
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => (
    `${JSON.stringify(key)}:${stableJson(record[key])}`
  )).join(",")}}`;
}

function matchesInvalidToolInput(error: unknown, toolName: string): boolean {
  if (!error || typeof error !== "object") return false;
  const record = error as Record<string, unknown>;
  if (record.name !== INVALID_TOOL_INPUT_ERROR) return false;
  const invocation = objectRecord(record.toolInvocation);
  const details = objectRecord(invocation?.details);
  const call = objectRecord(details?.toolCall);
  const calledTool = call?.name;
  return calledTool === undefined || calledTool === toolName;
}

function objectRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object"
    ? value as Record<string, unknown>
    : undefined;
}

function jsonObject(value: string): Record<string, unknown> | undefined {
  try {
    return objectRecord(JSON.parse(value));
  } catch {
    return undefined;
  }
}

function validationIssues(values: unknown): ValidationIssue[] {
  if (!Array.isArray(values)) return [];
  const seen = new Set<string>();
  return values.flatMap((value) => {
    const issue = objectRecord(value);
    if (!issue || typeof issue.message !== "string") return [];
    const path = typeof issue.path === "string"
      ? issue.path
      : Array.isArray(issue.path)
        ? issue.path.filter((part): part is string | number =>
            typeof part === "string" || typeof part === "number"
          ).join(".")
        : "";
    const identity = `${path}\u0000${issue.message}`;
    if (seen.has(identity)) return [];
    seen.add(identity);
    return [{
      path,
      code: typeof issue.code === "string" ? issue.code : "validation_error",
      message: issue.message
    }];
  }).slice(0, 16);
}
