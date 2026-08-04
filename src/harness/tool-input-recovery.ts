import type { ZodType } from "zod";

const INVALID_TOOL_INPUT_ERROR = "InvalidToolInputError";

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
  let decoded: unknown;
  try {
    decoded = JSON.parse(input);
  } catch {
    return invalidToolInputResultForIssues(toolName, [{
      path: [],
      code: "invalid_json",
      message: "Tool arguments must be valid JSON."
    }]);
  }
  const result = schema.safeParse(decoded);
  if (result.success) return undefined;
  return invalidToolInputResultForIssues(toolName, result.error.issues);
}

function invalidToolInputResultForIssues(
  toolName: string,
  issues: unknown
): string {
  return JSON.stringify({
    accepted: false,
    code: "invalid_tool_input",
    tool: toolName,
    error: "The tool arguments were not valid for the declared schema.",
    validation_issues: validationIssues(issues),
    automatic_actuation: false,
    recovery: "Generate a new complete tool call from current evidence and resolve every validation issue. The harness will not repair the rejected arguments."
  });
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

function validationIssues(values: unknown): Array<{
  path: string;
  code: string;
  message: string;
}> {
  if (!Array.isArray(values)) return [];
  return values.slice(0, 16).flatMap((value) => {
    const issue = objectRecord(value);
    if (!issue || typeof issue.message !== "string") return [];
    const path = Array.isArray(issue.path)
      ? issue.path.filter((part): part is string | number =>
          typeof part === "string" || typeof part === "number"
        ).join(".")
      : "";
    return [{
      path,
      code: typeof issue.code === "string" ? issue.code : "validation_error",
      message: issue.message
    }];
  });
}
