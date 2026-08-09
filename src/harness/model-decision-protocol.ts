import type { Model, ModelRequest } from "@openai/agents";

export interface ToolChoiceProtocolRejection {
  agentId: string;
  mode: "named" | "required";
  toolName?: string;
}

/**
 * Uses the standard named-tool protocol when the Harness exposes one legal
 * transition, then strengthens multi-tool recovery after a model completes a
 * branch without a decision. Capability negotiation stays attached to the
 * same model facade; decision recovery expires when authority changes.
 */
export class ModelDecisionProtocolRecovery {
  readonly #authorityFingerprint: () => string;
  readonly #requiredAtAuthority = new Map<string, string>();
  readonly #namedUnsupported = new Set<string>();
  readonly #requiredUnsupported = new Set<string>();

  constructor(authorityFingerprint: () => string) {
    this.#authorityFingerprint = authorityFingerprint;
  }

  requireToolDecision(agentId: string): boolean {
    if (this.#requiredUnsupported.has(agentId)) return false;
    const authorityFingerprint = this.#authorityFingerprint();
    const activated = this.#requiredAtAuthority.get(agentId) !== authorityFingerprint;
    this.#requiredAtAuthority.set(agentId, authorityFingerprint);
    return activated;
  }

  requiresToolDecision(agentId: string): boolean {
    if (this.#requiredUnsupported.has(agentId)) return false;
    const requiredAt = this.#requiredAtAuthority.get(agentId);
    if (requiredAt === undefined) return false;
    if (requiredAt === this.#authorityFingerprint()) return true;
    this.#requiredAtAuthority.delete(agentId);
    return false;
  }

  rejectRequiredToolChoice(agentId: string): void {
    this.#requiredUnsupported.add(agentId);
    this.#requiredAtAuthority.delete(agentId);
  }

  namedToolChoiceSupported(agentId: string): boolean {
    return !this.#namedUnsupported.has(agentId);
  }

  rejectNamedToolChoice(agentId: string): void {
    this.#namedUnsupported.add(agentId);
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
  ) => void | Promise<void>
): Model {
  const rejectChoice = async (
    prepared: PreparedToolChoiceRequest
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
      ...(prepared.mode === "named" ? { toolName: String(prepared.choice) } : {})
    });
  };
  return {
    getResponse: async (request) => {
      const candidates = toolChoiceCandidates(request, agentId, recovery);
      for (let index = 0; index < candidates.length; index += 1) {
        const prepared = candidates[index]!;
        try {
          return await model.getResponse(prepared.request);
        } catch (error) {
          if (index === candidates.length - 1
            || prepared.mode === "configured"
            || !isUnsupportedToolChoiceError(error)) throw error;
          await rejectChoice(prepared);
        }
      }
      throw new Error("Tool-choice negotiation produced no model request");
    },
    getStreamedResponse: (request) => ({
      async *[Symbol.asyncIterator]() {
        const candidates = toolChoiceCandidates(request, agentId, recovery);
        for (let index = 0; index < candidates.length; index += 1) {
          const prepared = candidates[index]!;
          let yielded = false;
          try {
            for await (const event of model.getStreamedResponse(prepared.request)) {
              yielded = true;
              yield event;
            }
            return;
          } catch (error) {
            if (index === candidates.length - 1
              || prepared.mode === "configured"
              || yielded
              || !isUnsupportedToolChoiceError(error)) throw error;
            await rejectChoice(prepared);
          }
        }
        throw new Error("Tool-choice negotiation produced no streamed model request");
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
  const toolNames = [
    ...request.tools.flatMap((tool) => (
      tool.type === "function" && tool.name.trim().length > 0 ? [tool.name] : []
    )),
    ...(request.handoffs ?? []).map((handoff) => handoff.toolName)
  ].filter((name, index, names) => names.indexOf(name) === index);
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
  if (recoveryRequired) add("required", "required");
  add(configured, "configured");
  return candidates;
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
