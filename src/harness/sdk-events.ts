import type { ModelResponse, RunStreamEvent } from "@openai/agents";
import type { JsonValue } from "../domain/schema.js";

export function sdkEventJson(event: RunStreamEvent): JsonValue | undefined {
  if (event.type === "agent_updated_stream_event") {
    return { type: event.type, agent: event.agent.name };
  }
  if (event.type === "run_item_stream_event") {
    return {
      type: event.type,
      name: event.name,
      item: json(event.item.toJSON())
    };
  }
  if (event.data.type === "response_started") {
    return {
      type: event.type,
      source: event.source ?? null,
      data: { type: event.data.type }
    };
  }
  if (event.data.type === "response_done") {
    return {
      type: event.type,
      source: event.source ?? null,
      data: {
        type: event.data.type,
        response_id: event.data.response.id,
        output_types: event.data.response.output.map((item) => item.type ?? "unknown"),
        usage: json(event.data.response.usage)
      }
    };
  }
  // Token deltas are transport noise. Durable tool and message items arrive as
  // run_item_stream_event records, while response_done retains final usage.
  return undefined;
}

export function providerEventJson(event: RunStreamEvent): JsonValue | undefined {
  if (event.type !== "raw_model_stream_event") return undefined;
  if (event.data.type === "response_started") {
    return { status: "contacted", source: event.source ?? null };
  }
  if (event.data.type === "response_done") {
    const { hasText, hasDecision } = modelResponseDisposition(event.data.response.output);
    return {
      status: hasText || hasDecision ? "usable_stream" : "no_text",
      source: event.source ?? null,
      response_id: event.data.response.id,
      has_text: hasText,
      has_decision: hasDecision,
      usage: json(event.data.response.usage)
    };
  }
  return undefined;
}

export function modelResponseDisposition(output: ModelResponse["output"]): {
  hasText: boolean;
  hasDecision: boolean;
} {
  return {
    hasText: output.some((item) => {
      if (item.type !== "message" || !Array.isArray(item.content)) return false;
      return item.content.some((content) =>
        typeof content === "object"
          && content !== null
          && "type" in content
          && content.type === "output_text"
          && "text" in content
          && typeof content.text === "string"
          && content.text.trim() !== ""
      );
    }),
    hasDecision: output.some((item) => item.type === "function_call")
  };
}

function json(value: unknown): JsonValue {
  const text = JSON.stringify(value);
  if (text === undefined) return null;
  return JSON.parse(text) as JsonValue;
}
