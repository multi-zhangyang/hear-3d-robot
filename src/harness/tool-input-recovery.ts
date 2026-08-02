const INVALID_TOOL_INPUT_ERROR = "InvalidToolInputError";

/**
 * Keeps the Agents SDK's native model-correction loop for malformed tool
 * arguments while allowing every execution, transport, and abort failure to
 * retain its normal failure semantics.
 */
export function invalidToolInputResult(error: unknown, toolName: string): string {
  if (!matchesInvalidToolInput(error, toolName)) throw error;
  return JSON.stringify({
    accepted: false,
    code: "invalid_tool_input",
    tool: toolName,
    error: "The tool arguments were not valid for the declared schema.",
    validation_issues: validationIssues(error),
    automatic_actuation: false,
    recovery: "Generate a new complete tool call from current evidence and resolve every validation issue. The harness will not repair the rejected arguments."
  });
}

function matchesInvalidToolInput(error: unknown, toolName: string): boolean {
  if (!error || typeof error !== "object") return false;
  const record = error as Record<string, unknown>;
  if (record.name !== INVALID_TOOL_INPUT_ERROR) return false;
  const invocation = objectRecord(record.toolInvocation);
  if (!invocation || typeof invocation.input !== "string") return false;
  const details = objectRecord(invocation.details);
  const call = objectRecord(details?.toolCall);
  const calledTool = call?.name;
  return calledTool === undefined || calledTool === toolName;
}

function objectRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object"
    ? value as Record<string, unknown>
    : undefined;
}

function validationIssues(error: unknown): Array<{
  path: string;
  code: string;
  message: string;
}> {
  const original = objectRecord(objectRecord(error)?.originalError);
  if (!Array.isArray(original?.issues)) return [];
  return original.issues.slice(0, 16).flatMap((value) => {
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
