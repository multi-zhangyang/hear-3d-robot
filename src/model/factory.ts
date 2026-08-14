import { createHash } from "node:crypto";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { Model, ModelRequest } from "@openai/agents";
import { aisdk } from "@openai/agents-extensions/ai-sdk";
import {
  DEFAULT_MODEL_REQUEST_TIMEOUT_MS,
  type ModelProviderConfig,
  type ProviderConfig
} from "../config/load.js";

export interface ProviderIdentity {
  protocol: ModelProviderConfig["protocol"];
  model: string;
}

export interface ConfiguredModelOptions {
  promptCacheKey?: string;
  onPromptCacheStatus?: (event: PromptCacheStatus) => void | Promise<void>;
  onPromptCacheRequest?: (event: PromptCacheRequestTrace) => void | Promise<void>;
}

export interface PromptCacheStatus {
  status: "unsupported";
  compatibilityRetry: boolean;
}

export interface PromptCacheRequestTrace {
  requestSha256: string;
  messageCount: number;
  previousMessageCount: number | null;
  commonMessagePrefixCount: number;
  commonMessagePrefixBytes: number;
  appendOnlyMessagePrefix: boolean;
  toolCount: number;
  toolsStable: boolean;
  settingsStable: boolean;
  cacheAffinityPresent: boolean;
}

export function createConfiguredModel(
  config: ModelProviderConfig,
  options: ConfiguredModelOptions = {}
): Model {
  const timedFetch = requestTimedFetch(
    config.requestTimeoutMs ?? DEFAULT_MODEL_REQUEST_TIMEOUT_MS,
    fetch,
    options.onPromptCacheRequest
  );
  let model: Model;
  if (config.protocol === "openai_compatible") {
    const provider = createOpenAICompatible({
      name: "configured-openai-compatible",
      baseURL: config.baseUrl,
      apiKey: config.apiKey,
      fetch: openAICompatibleRequestFetch(timedFetch)
    });
    model = agentsModelFromAiSdk(
      provider.chatModel(config.model),
      true
    );
  } else if (config.protocol === "openai_responses") {
    const provider = createOpenAI({
      name: "configured-openai-responses",
      baseURL: config.baseUrl,
      apiKey: config.apiKey,
      fetch: timedFetch
    });
    model = agentsModelFromAiSdk(provider.responses(config.model));
  } else {
    const provider = createAnthropic({
      name: "configured-anthropic-messages",
      baseURL: config.baseUrl,
      apiKey: config.apiKey,
      fetch: timedFetch
    });
    model = agentsModelFromAiSdk(provider.messages(config.model));
  }
  return options.promptCacheKey
    ? withPromptCacheAffinity(
        model,
        config.protocol,
        options.promptCacheKey,
        options.onPromptCacheStatus
      )
    : model;
}

/**
 * The 0.15 AI SDK adapter can materialize `rawUsage: undefined`, while the
 * Agents `Model` contract requires an absent optional property. Keep that
 * package-level declaration mismatch at this integration boundary and retain
 * the adapter's native retry advice for the Agents runner.
 */
function agentsModelFromAiSdk(
  model: Parameters<typeof aisdk>[0],
  normalizeCompatibleToolArguments = false
): Model {
  const adapted = aisdk(model);
  const getResponse: Model["getResponse"] = async (request) => {
    const rawResponse = await adapted.getResponse(request);
    const response = normalizeCompatibleToolArguments
      ? normalizedCompatibleModelResponse(rawResponse)
      : rawResponse;
    if (response.rawUsage === undefined) {
      const { rawUsage: _rawUsage, ...normalized } = response;
      return normalized;
    }
    return { ...response, rawUsage: response.rawUsage };
  };
  const getStreamedResponse: Model["getStreamedResponse"] = async function* (
    request
  ) {
    for await (const event of adapted.getStreamedResponse(request)) {
      if (normalizeCompatibleToolArguments && event.type === "response_done") {
        yield {
          ...event,
          response: normalizedCompatibleModelResponse(event.response)
        };
      } else {
        yield event;
      }
    }
  };
  return {
    getResponse,
    getStreamedResponse,
    getRetryAdvice: (input) => adapted.getRetryAdvice(input)
  };
}

function normalizedCompatibleModelResponse<T>(response: T): T {
  if (!isRecord(response) || !Array.isArray(response.output)) return response;
  let changed = false;
  const output = response.output.map((item) => {
    if (!isRecord(item) || item.type !== "function_call"
      || typeof item.arguments !== "string") return item;
    const normalized = normalizeDuplicatedTrailingObjectDelimiters(
      item.arguments
    );
    if (normalized === item.arguments) return item;
    changed = true;
    return { ...item, arguments: normalized };
  });
  return changed ? { ...response, output } as T : response;
}

export function normalizeDuplicatedTrailingObjectDelimiters(input: string): string {
  if (jsonObject(input) !== undefined) return input;
  let candidate = input.trimEnd();
  for (let removed = 0; removed < 4 && candidate.endsWith("}"); removed += 1) {
    candidate = candidate.slice(0, -1).trimEnd();
    if (jsonObject(candidate) !== undefined) return candidate;
  }
  return input;
}

function jsonObject(value: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

export function openAICompatibleRequestFetch(
  implementation: typeof fetch = fetch
): typeof fetch {
  return async (input, init) => {
    const body = normalizedOpenAICompatibleBody(init?.body);
    return implementation(input, body === undefined ? init : { ...init, body });
  };
}

function normalizedOpenAICompatibleBody(
  body: BodyInit | null | undefined
): BodyInit | null | undefined {
  if (typeof body !== "string") return body;
  let value: unknown;
  try {
    value = JSON.parse(body);
  } catch {
    return body;
  }
  if (!isRecord(value) || !Array.isArray(value.messages)) return body;
  let messages = value.messages.map((message) => (
    isRecord(message) ? { ...message } : message
  ));
  let changed = false;
  for (let index = messages.length - 2; index >= 0; index -= 1) {
    const message = messages[index];
    const next = messages[index + 1];
    if (!isEmptyAssistantMessage(message) || !isRecord(next)
      || next.role !== "assistant") continue;
    const reasoning = message.reasoning_content;
    if (typeof reasoning === "string" && reasoning.length > 0
      && !(typeof next.reasoning_content === "string"
        && next.reasoning_content.length > 0)) {
      next.reasoning_content = reasoning;
    }
    messages.splice(index, 1);
    changed = true;
  }
  const toolHistory = normalizedOpenAICompatibleToolHistory(messages);
  messages = toolHistory.messages;
  changed = changed || toolHistory.changed;
  if (!changed) return body;
  return JSON.stringify({ ...value, messages });
}

function normalizedOpenAICompatibleToolHistory(
  messages: unknown[]
): { messages: unknown[]; changed: boolean } {
  const normalized: unknown[] = [];
  let changed = false;
  for (let index = 0; index < messages.length;) {
    const message = messages[index];
    if (isRecord(message) && message.role === "assistant"
      && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      let resultEnd = index + 1;
      while (resultEnd < messages.length
        && messageRole(messages[resultEnd]) === "tool") {
        resultEnd += 1;
      }
      const callIds = message.tool_calls.flatMap((toolCall) => (
        isRecord(toolCall) && typeof toolCall.id === "string" && toolCall.id.length > 0
          ? [toolCall.id]
          : []
      ));
      const resultIds = messages.slice(index + 1, resultEnd).flatMap((result) => (
        isRecord(result)
          && typeof result.tool_call_id === "string"
          && result.tool_call_id.length > 0
          ? [result.tool_call_id]
          : []
      ));
      const complete = callIds.length === message.tool_calls.length
        && new Set(callIds).size === callIds.length
        && resultIds.length === callIds.length
        && new Set(resultIds).size === resultIds.length
        && resultIds.every((id) => callIds.includes(id));
      if (complete) {
        normalized.push(message, ...messages.slice(index + 1, resultEnd));
      } else {
        const semantic = { ...message };
        delete semantic.tool_calls;
        if (!isEmptyAssistantMessage(semantic)) normalized.push(semantic);
        changed = true;
      }
      index = resultEnd;
      continue;
    }
    if (isRecord(message) && message.role === "tool") {
      changed = true;
      index += 1;
      continue;
    }
    normalized.push(message);
    index += 1;
  }
  return { messages: normalized, changed };
}

function messageRole(value: unknown): unknown {
  return isRecord(value) ? value.role : undefined;
}

function isEmptyAssistantMessage(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value) || value.role !== "assistant") return false;
  const hasContent = typeof value.content === "string"
    ? value.content.length > 0
    : Array.isArray(value.content) && value.content.length > 0;
  const hasToolCall = Array.isArray(value.tool_calls) && value.tool_calls.length > 0;
  const hasFunctionCall = isRecord(value.function_call);
  const hasRefusal = typeof value.refusal === "string" && value.refusal.length > 0;
  return !hasContent && !hasToolCall && !hasFunctionCall && !hasRefusal;
}

/**
 * Keeps one long-running Agent on one provider cache route without exposing
 * the endpoint, model name, run identifier, or Agent identifier to telemetry.
 */
export function promptCacheAffinityKey(input: {
  namespace: string;
  agentId: string;
  provider: ModelProviderConfig;
}): string {
  return createHash("sha256")
    .update("hear-prompt-cache-affinity-v1\0")
    .update(input.namespace)
    .update("\0")
    .update(input.agentId)
    .update("\0")
    .update(input.provider.protocol)
    .update("\0")
    .update(new URL(input.provider.baseUrl).href)
    .update("\0")
    .update(input.provider.model)
    .update("\0")
    .update(input.provider.apiKey)
    .digest("hex");
}

export function promptCacheProviderData(
  protocol: ModelProviderConfig["protocol"],
  promptCacheKey: string,
  existing: Record<string, unknown> = {}
): Record<string, unknown> {
  if (protocol === "openai_compatible") {
    return mergeProviderOptions(existing, "configured-openai-compatible", {
      prompt_cache_key: promptCacheKey
    });
  }
  if (protocol === "openai_responses") {
    return mergeProviderOptions(existing, "openai", {
      promptCacheKey
    });
  }
  return mergeProviderOptions(existing, "anthropic", {
    cacheControl: { type: "ephemeral" }
  });
}

export function withPromptCacheAffinity(
  model: Model,
  protocol: ModelProviderConfig["protocol"],
  promptCacheKey: string,
  onStatus?: (event: PromptCacheStatus) => void | Promise<void>
): Model {
  let capability: "unknown" | "supported" | "unsupported" = "unknown";
  const prepare = (request: ModelRequest): ModelRequest => ({
    ...request,
    modelSettings: {
      ...request.modelSettings,
      providerData: promptCacheProviderData(
        protocol,
        promptCacheKey,
        request.modelSettings.providerData
      )
    }
  });
  const reportUnsupported = async (): Promise<void> => {
    if (capability === "unsupported") return;
    capability = "unsupported";
    await onStatus?.({ status: "unsupported", compatibilityRetry: true });
  };
  const getResponse: Model["getResponse"] = async (request) => {
    if (capability === "unsupported") return model.getResponse(request);
    try {
      const response = await model.getResponse(prepare(request));
      capability = "supported";
      return response;
    } catch (error) {
      if (!isUnsupportedPromptCacheError(error)) throw error;
      await reportUnsupported();
      return model.getResponse(request);
    }
  };
  const getStreamedResponse: Model["getStreamedResponse"] = (request) => ({
    async *[Symbol.asyncIterator]() {
      if (capability === "unsupported") {
        yield* model.getStreamedResponse(request);
        return;
      }
      let yielded = false;
      try {
        for await (const event of model.getStreamedResponse(prepare(request))) {
          yielded = true;
          capability = "supported";
          yield event;
        }
      } catch (error) {
        if (yielded || !isUnsupportedPromptCacheError(error)) throw error;
        await reportUnsupported();
        yield* model.getStreamedResponse(request);
      }
    }
  });
  return {
    getResponse,
    getStreamedResponse,
    ...(model.getRetryAdvice
      ? {
          getRetryAdvice: (input) => model.getRetryAdvice!({
            ...input,
            request: prepare(input.request)
          })
        }
      : {})
  };
}

function isUnsupportedPromptCacheError(error: unknown): boolean {
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
  const namesCacheSetting = detail.includes("prompt_cache")
    || detail.includes("prompt cache")
    || detail.includes("cache_control")
    || detail.includes("cache control");
  const rejectsSetting = detail.includes("unsupported")
    || detail.includes("unknown parameter")
    || detail.includes("unrecognized")
    || detail.includes("not allowed")
    || detail.includes("extra inputs are not permitted");
  return namesCacheSetting && rejectsSetting;
}

function mergeProviderOptions(
  existing: Record<string, unknown>,
  providerKey: string,
  additions: Record<string, unknown>
): Record<string, unknown> {
  const providerOptions = isRecord(existing.providerOptions)
    ? existing.providerOptions
    : {};
  const selected = isRecord(providerOptions[providerKey])
    ? providerOptions[providerKey]
    : {};
  return {
    ...existing,
    providerOptions: {
      ...providerOptions,
      [providerKey]: {
        ...selected,
        ...additions
      }
    }
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

interface CacheRequestSurface {
  requestSha256: string;
  messages: Array<{ sha256: string; bytes: number }>;
  toolsSha256: string;
  toolCount: number;
  settingsSha256: string;
  cacheAffinityPresent: boolean;
}

function cacheRequestSurface(body: BodyInit | null | undefined): CacheRequestSurface | undefined {
  if (typeof body !== "string") return undefined;
  let value: unknown;
  try {
    value = JSON.parse(body);
  } catch {
    return undefined;
  }
  if (!isRecord(value)) return undefined;
  const promptItems: unknown[] = [];
  if (value.system !== undefined) promptItems.push({ system: value.system });
  if (Array.isArray(value.messages)) promptItems.push(...value.messages);
  else if (Array.isArray(value.input)) promptItems.push(...value.input);
  else if (value.input !== undefined) promptItems.push({ input: value.input });
  const messages = promptItems.map((item) => {
    const serialized = JSON.stringify(item);
    return {
      sha256: sha256(serialized),
      bytes: Buffer.byteLength(serialized)
    };
  });
  const tools = Array.isArray(value.tools) ? value.tools : [];
  const settings = Object.fromEntries(Object.entries(value).filter(([key]) => (
    key !== "system" && key !== "messages" && key !== "input" && key !== "tools"
  )));
  return {
    requestSha256: sha256(body),
    messages,
    toolsSha256: sha256(JSON.stringify(tools)),
    toolCount: tools.length,
    settingsSha256: sha256(JSON.stringify(settings)),
    cacheAffinityPresent: typeof value.prompt_cache_key === "string"
      || containsCacheControl(value)
  };
}

function promptCacheRequestTrace(
  current: CacheRequestSurface,
  previous: CacheRequestSurface | undefined
): PromptCacheRequestTrace {
  let commonMessagePrefixCount = 0;
  let commonMessagePrefixBytes = 0;
  if (previous) {
    const maximum = Math.min(previous.messages.length, current.messages.length);
    while (commonMessagePrefixCount < maximum) {
      const prior = previous.messages[commonMessagePrefixCount]!;
      const next = current.messages[commonMessagePrefixCount]!;
      if (prior.sha256 !== next.sha256) break;
      commonMessagePrefixBytes += next.bytes;
      commonMessagePrefixCount += 1;
    }
  }
  return {
    requestSha256: current.requestSha256,
    messageCount: current.messages.length,
    previousMessageCount: previous?.messages.length ?? null,
    commonMessagePrefixCount,
    commonMessagePrefixBytes,
    appendOnlyMessagePrefix: previous !== undefined
      && commonMessagePrefixCount === previous.messages.length
      && current.messages.length >= previous.messages.length,
    toolCount: current.toolCount,
    toolsStable: previous === undefined || current.toolsSha256 === previous.toolsSha256,
    settingsStable: previous === undefined
      || current.settingsSha256 === previous.settingsSha256,
    cacheAffinityPresent: current.cacheAffinityPresent
  };
}

function containsCacheControl(value: unknown, depth = 0): boolean {
  if (depth > 4 || value === null || typeof value !== "object") return false;
  if (Array.isArray(value)) {
    return value.some((entry) => containsCacheControl(entry, depth + 1));
  }
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (key === "cache_control" || key === "cacheControl") return true;
    if (containsCacheControl(entry, depth + 1)) return true;
  }
  return false;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

export function requestTimedFetch(
  timeoutMs: number,
  implementation: typeof fetch = fetch,
  onPromptCacheRequest?: (event: PromptCacheRequestTrace) => void | Promise<void>
): typeof fetch {
  let previousRequest: CacheRequestSurface | undefined;
  return async (input, init) => {
    const currentRequest = onPromptCacheRequest
      ? cacheRequestSurface(init?.body)
      : undefined;
    if (currentRequest) {
      await onPromptCacheRequest?.(promptCacheRequestTrace(
        currentRequest,
        previousRequest
      ));
      previousRequest = currentRequest;
    }
    const timeout = new AbortController();
    const upstream = init?.signal;
    const signal = upstream
      ? AbortSignal.any([upstream, timeout.signal])
      : timeout.signal;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let timedOut = false;
    const clearTimer = (): void => {
      if (timer === undefined) return;
      clearTimeout(timer);
      timer = undefined;
    };
    const armTimer = (): void => {
      clearTimer();
      timer = setTimeout(() => {
        timedOut = true;
        timeout.abort();
      }, timeoutMs);
    };
    const normalizeError = (error: unknown): unknown => (
      timedOut && !upstream?.aborted
        ? modelRequestTimeoutError(timeoutMs)
        : error
    );
    armTimer();
    try {
      const response = await implementation(input, { ...init, signal });
      armTimer();
      if (!response.body) {
        clearTimer();
        return response;
      }
      const reader = response.body.getReader();
      const body = new ReadableStream<Uint8Array>({
        async pull(controller) {
          try {
            const chunk = await reader.read();
            if (chunk.done) {
              clearTimer();
              controller.close();
              return;
            }
            armTimer();
            controller.enqueue(chunk.value);
          } catch (error) {
            clearTimer();
            controller.error(normalizeError(error));
          }
        },
        async cancel(reason) {
          clearTimer();
          await reader.cancel(reason);
        }
      });
      return new Response(body, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers
      });
    } catch (error) {
      clearTimer();
      throw normalizeError(error);
    }
  };
}

function modelRequestTimeoutError(timeoutMs: number): TypeError {
  const cause = Object.assign(new Error(
    `Model request produced no data for ${timeoutMs}ms`
  ), {
    code: "ETIMEDOUT"
  });
  return new TypeError("fetch failed", { cause });
}

export function providerIdentity(config: ProviderConfig | ModelProviderConfig): ProviderIdentity {
  return {
    protocol: config.protocol,
    model: config.model
  };
}
