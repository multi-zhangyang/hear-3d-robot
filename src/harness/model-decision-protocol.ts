import type {
  Model,
  ModelRequest,
  ModelResponse,
  StreamEvent
} from "@openai/agents";
import { recordAgentInvocationDecisionInterruption } from "./agent-scope.js";
import { ModelDecisionStallError } from "./model-telemetry.js";

const MAX_IN_PLACE_DECISION_RESPONSES = 2;

export interface ToolChoiceProtocolRejection {
  agentId: string;
  mode: "named" | "required";
  toolName?: string;
  reason?: "constraint_ignored";
}

export interface ModelDecisionProtocolRetry {
  agentId: string;
  completedResponseCount: number;
  availableToolNames: string[];
  constraint: "native" | "prompted_auto";
}

/**
 * Uses the standard named-tool protocol when the Harness exposes one legal
 * transition, then strengthens multi-tool recovery after a model completes a
 * branch without a decision. Capability negotiation stays attached to the
 * same model facade; decision recovery expires when authority changes.
 */
export class ModelDecisionProtocolRecovery {
  readonly #authorityFingerprint: () => string;
  readonly #availableToolNames: (
    agentId: string,
    exposedToolNames: readonly string[]
  ) => readonly string[];
  readonly #requiredAtAuthority = new Map<string, string>();
  readonly #namedUnsupported = new Set<string>();
  readonly #requiredUnsupported = new Set<string>();

  constructor(
    authorityFingerprint: () => string,
    availableToolNames: (
      agentId: string,
      exposedToolNames: readonly string[]
    ) => readonly string[] = (_agentId, exposedToolNames) => exposedToolNames
  ) {
    this.#authorityFingerprint = authorityFingerprint;
    this.#availableToolNames = availableToolNames;
  }

  requireToolDecision(agentId: string): boolean {
    const authorityFingerprint = this.#authorityFingerprint();
    const activated = this.#requiredAtAuthority.get(agentId) !== authorityFingerprint;
    this.#requiredAtAuthority.set(agentId, authorityFingerprint);
    return activated;
  }

  requiresToolDecision(agentId: string): boolean {
    const requiredAt = this.#requiredAtAuthority.get(agentId);
    if (requiredAt === undefined) return false;
    if (requiredAt === this.#authorityFingerprint()) return true;
    this.#requiredAtAuthority.delete(agentId);
    return false;
  }

  rejectRequiredToolChoice(agentId: string): void {
    this.#requiredUnsupported.add(agentId);
  }

  requiredToolChoiceSupported(agentId: string): boolean {
    return !this.#requiredUnsupported.has(agentId);
  }

  namedToolChoiceSupported(agentId: string): boolean {
    return !this.#namedUnsupported.has(agentId);
  }

  rejectNamedToolChoice(agentId: string): void {
    this.#namedUnsupported.add(agentId);
  }

  availableToolNames(agentId: string, exposedToolNames: readonly string[]): string[] {
    const exposed = new Set(exposedToolNames);
    return this.#availableToolNames(agentId, exposedToolNames)
      .filter((name, index, names) => exposed.has(name) && names.indexOf(name) === index);
  }

  clear(): void {
    this.#requiredAtAuthority.clear();
  }
}

export function withModelDecisionProtocolRecovery(
  model: Model,
  agentId: string,
  recovery: ModelDecisionProtocolRecovery,
  onToolChoiceUnsupported?: (
    event: ToolChoiceProtocolRejection
  ) => void | Promise<void>,
  onDecisionRetry?: (
    event: ModelDecisionProtocolRetry
  ) => void | Promise<void>
): Model {
  const rejectChoice = async (
    prepared: PreparedToolChoiceRequest,
    reason?: ToolChoiceProtocolRejection["reason"]
  ): Promise<void> => {
    if (prepared.mode === "named") {
      recovery.rejectNamedToolChoice(agentId);
    } else if (prepared.mode === "required") {
      recovery.rejectRequiredToolChoice(agentId);
    } else {
      return;
    }
    await onToolChoiceUnsupported?.({
      agentId,
      mode: prepared.mode,
      ...(prepared.mode === "named" ? { toolName: String(prepared.choice) } : {}),
      ...(reason ? { reason } : {})
    });
  };
  const recordNoDecision = async (
    request: ModelRequest,
    completedResponseCount: number,
    prepared: PreparedToolChoiceRequest
  ): Promise<void> => {
    recovery.requireToolDecision(agentId);
    if (prepared.mode !== "configured") {
      await rejectChoice(prepared, "constraint_ignored");
    }
    await onDecisionRetry?.({
      agentId,
      completedResponseCount,
      availableToolNames: recovery.availableToolNames(
        agentId,
        requestToolNames(request)
      ),
      constraint: recovery.requiredToolChoiceSupported(agentId)
        ? "native"
        : "prompted_auto"
    });
  };
  return {
    getResponse: async (request) => {
      if (!requestNeedsNativeToolDecision(request)) {
        return model.getResponse(request);
      }
      let completedResponseCount = 0;
      for (let decisionAttempt = 0;
        decisionAttempt < MAX_IN_PLACE_DECISION_RESPONSES;
        decisionAttempt += 1) {
        const decisionRequest = decisionAttempt === 0
          ? request
          : promptedDecisionRequest(request, agentId, recovery);
        const candidates = toolChoiceCandidates(decisionRequest, agentId, recovery);
        for (let index = 0; index < candidates.length; index += 1) {
          const prepared = candidates[index]!;
          try {
            const response = await model.getResponse(prepared.request);
            if (hasNativeToolDecision(response.output)) return response;
            completedResponseCount += 1;
            await recordNoDecision(request, completedResponseCount, prepared);
            break;
          } catch (error) {
            if (index === candidates.length - 1
              || prepared.mode === "configured"
              || !isUnsupportedToolChoiceError(error)) throw error;
            await rejectChoice(prepared);
          }
        }
      }
      throw noDecisionError(agentId, completedResponseCount);
    },
    getStreamedResponse: (request) => ({
      async *[Symbol.asyncIterator]() {
        if (!requestNeedsNativeToolDecision(request)) {
          yield* model.getStreamedResponse(request);
          return;
        }
        let completedResponseCount = 0;
        for (let decisionAttempt = 0;
          decisionAttempt < MAX_IN_PLACE_DECISION_RESPONSES;
          decisionAttempt += 1) {
          const decisionRequest = decisionAttempt === 0
            ? request
            : promptedDecisionRequest(request, agentId, recovery);
          const candidates = toolChoiceCandidates(decisionRequest, agentId, recovery);
          for (let index = 0; index < candidates.length; index += 1) {
            const prepared = candidates[index]!;
            const buffered: StreamEvent[] = [];
            try {
              for await (const event of model.getStreamedResponse(prepared.request)) {
                buffered.push(event);
              }
              const output = completedStreamOutput(buffered);
              if (output && hasNativeToolDecision(output)) {
                yield* buffered;
                return;
              }
              completedResponseCount += 1;
              await recordNoDecision(request, completedResponseCount, prepared);
              break;
            } catch (error) {
              if (index === candidates.length - 1
                || prepared.mode === "configured"
                || !isUnsupportedToolChoiceError(error)) throw error;
              await rejectChoice(prepared);
            }
          }
        }
        throw noDecisionError(agentId, completedResponseCount);
      }
    }),
    ...(model.getRetryAdvice
      ? {
          getRetryAdvice: (input) => model.getRetryAdvice!({
            ...input,
            request: toolChoiceCandidates(input.request, agentId, recovery)[0]!.request
          })
        }
      : {})
  };
}

type ToolChoice = NonNullable<ModelRequest["modelSettings"]["toolChoice"]>;

interface PreparedToolChoiceRequest {
  request: ModelRequest;
  choice: ToolChoice | undefined;
  mode: "named" | "required" | "configured";
}

function toolChoiceCandidates(
  request: ModelRequest,
  agentId: string,
  recovery: ModelDecisionProtocolRecovery
): PreparedToolChoiceRequest[] {
  const candidates: PreparedToolChoiceRequest[] = [];
  const seen = new Set<string>();
  const configured = request.modelSettings.toolChoice;
  const recoveryRequired = recovery.requiresToolDecision(agentId);
  const toolNames = requestToolNames(request);
  const add = (
    choice: ToolChoice | undefined,
    mode: PreparedToolChoiceRequest["mode"]
  ): void => {
    const key = JSON.stringify(choice);
    if (seen.has(key)) return;
    seen.add(key);
    const { toolChoice: _toolChoice, ...settingsWithoutToolChoice } =
      request.modelSettings;
    candidates.push({
      request: choice === configured
        ? request
        : {
            ...request,
            modelSettings: choice === undefined
              ? settingsWithoutToolChoice
              : { ...settingsWithoutToolChoice, toolChoice: choice }
          },
      choice,
      mode
    });
  };

  // When the Harness exposes exactly one transition, a standard named tool
  // choice removes the ambiguous text-final branch without choosing arguments
  // or inventing an action. Endpoints that reject named choice are negotiated
  // once and continue with their configured protocol.
  if (toolNames.length === 1
    && (configured !== "none" || recoveryRequired)
    && recovery.namedToolChoiceSupported(agentId)) {
    add(toolNames[0]!, "named");
  }
  if (recoveryRequired && recovery.requiredToolChoiceSupported(agentId)) {
    add("required", "required");
  }
  add(configured, "configured");
  return candidates;
}

function requestToolNames(request: ModelRequest): string[] {
  return [
    ...request.tools.flatMap((tool) => (
      tool.type === "function" && tool.name.trim().length > 0 ? [tool.name] : []
    )),
    ...(request.handoffs ?? []).map((handoff) => handoff.toolName)
  ].filter((name, index, names) => names.indexOf(name) === index);
}

function requestNeedsNativeToolDecision(request: ModelRequest): boolean {
  return request.outputType === "text" && requestToolNames(request).length > 0;
}

function hasNativeToolDecision(output: ModelResponse["output"]): boolean {
  return output.some((item) => item.type === "function_call");
}

function completedStreamOutput(
  events: readonly StreamEvent[]
): ModelResponse["output"] | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!;
    if (event.type === "response_done") return event.response.output;
  }
  return undefined;
}

function promptedDecisionRequest(
  request: ModelRequest,
  agentId: string,
  recovery: ModelDecisionProtocolRecovery
): ModelRequest {
  const toolNames = recovery.availableToolNames(agentId, requestToolNames(request));
  const protocol = [
    "HARNESS NATIVE FUNCTION DECISION RECOVERY",
    `Bound agent: ${agentId}`,
    "This retry has exactly one purpose: make the model select and emit one native function_call from the functions currently exposed by the Harness.",
    `Available functions: ${toolNames.join(", ")}`,
    "Choose the function and all arguments yourself from the live authority and prior real receipts.",
    "Do not answer with prose, a JSON imitation, a tool name, or an explanation. The Harness will not parse text or choose an action for you."
  ].join("\n");
  return {
    ...request,
    tools: request.tools.filter((tool) => (
      tool.type !== "function" || toolNames.includes(tool.name)
    )),
    handoffs: request.handoffs.filter((handoff) => (
      toolNames.includes(handoff.toolName)
    )),
    input: appendDecisionProtocol(request.input, protocol)
  };
}

function appendDecisionProtocol(
  input: ModelRequest["input"],
  protocol: string
): ModelRequest["input"] {
  if (typeof input === "string") return `${input}\n\n${protocol}`;
  const items = [...input];
  const last = items.at(-1);
  if (last?.type === "message"
    && last.role === "user"
    && typeof last.content === "string") {
    items[items.length - 1] = {
      ...last,
      content: `${last.content}\n\n${protocol}`
    };
    return items;
  }
  items.push({ type: "message", role: "user", content: protocol });
  return items;
}

function noDecisionError(
  agentId: string,
  completedResponseCount: number
): ModelDecisionStallError {
  const error = new ModelDecisionStallError(
    agentId,
    `${agentId} returned ${completedResponseCount} completed model responses `
      + "without a native function_call"
  );
  recordAgentInvocationDecisionInterruption(error);
  return error;
}

function isUnsupportedToolChoiceError(error: unknown): boolean {
  const pending: unknown[] = [error];
  const seen = new Set<object>();
  const messages: string[] = [];
  while (pending.length > 0 && seen.size < 16) {
    const current = pending.shift();
    if (typeof current === "string") {
      messages.push(current);
      continue;
    }
    if (current === null || typeof current !== "object" || seen.has(current)) continue;
    seen.add(current);
    const record = current as Record<string, unknown>;
    if (typeof record.message === "string") messages.push(record.message);
    for (const key of ["cause", "error", "responseBody", "data"]) {
      if (record[key] !== undefined) pending.push(record[key]);
    }
    if (Array.isArray(record.errors)) pending.push(...record.errors);
  }
  const detail = messages.join(" ").toLowerCase();
  const namesToolChoice = detail.includes("tool_choice")
    || detail.includes("tool choice");
  const rejectsSetting = detail.includes("not support")
    || detail.includes("unsupported")
    || detail.includes("invalid_request")
    || detail.includes("invalid request")
    || detail.includes("not allowed");
  return namesToolChoice && rejectsSetting;
}
