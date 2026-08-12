import { createHash } from "node:crypto";
import { z } from "zod";
import {
  AutonomousCycleRefSchema,
  sameAutonomousCycle
} from "./autonomous-cycle.js";
import { HumanoidReplanModelCallSchema } from "./humanoid-replan-budget.js";

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);

export const ModelDecisionRefSchema = z.object({
  agent_id: z.string().trim().min(1),
  agent_manifest_sha256: Sha256Schema,
  agent_manifest_epoch_id: z.string().uuid(),
  model_call_id: z.string().uuid(),
  response_id: z.string().trim().min(1),
  response_output_sha256: Sha256Schema,
  tool_call_id: z.string().trim().min(1),
  tool_arguments_sha256: Sha256Schema,
  normalized_tool_arguments_sha256: Sha256Schema.optional()
}).strict();

export type ModelDecisionRef = z.infer<typeof ModelDecisionRefSchema>;

const ModelToolCallSchema = z.object({
  tool_call_id: z.string().trim().min(1),
  tool_name: z.string().trim().min(1),
  arguments_sha256: Sha256Schema
}).strict();

const ModelCallStartedSchema = z.object({
  version: z.literal(1),
  lifecycle: z.literal("started"),
  model_call_id: z.string().uuid(),
  agent_id: z.string().trim().min(1),
  cycle: AutonomousCycleRefSchema.optional(),
  replan_budget_call: HumanoidReplanModelCallSchema.optional(),
  at: z.string().datetime()
}).strict().superRefine((record, context) => {
  const call = record.replan_budget_call;
  if (call && (call.model_call_id !== record.model_call_id
    || call.agent_id !== record.agent_id
    || call.started_at !== record.at
    || call.status !== "started")) {
    context.addIssue({
      code: "custom",
      path: ["replan_budget_call"],
      message: "Model call replan budget evidence does not match its lifecycle start"
    });
  }
});

const ModelCallCompletedSchema = z.object({
  version: z.literal(1),
  lifecycle: z.literal("completed"),
  model_call_id: z.string().uuid(),
  agent_id: z.string().trim().min(1),
  response_id: z.string().trim().min(1),
  response_output_sha256: Sha256Schema,
  tool_calls: z.array(ModelToolCallSchema),
  cycle: AutonomousCycleRefSchema.optional(),
  at: z.string().datetime()
}).strict().superRefine((record, context) => {
  const callIds = record.tool_calls.map((entry) => entry.tool_call_id);
  if (new Set(callIds).size !== callIds.length) {
    context.addIssue({
      code: "custom",
      path: ["tool_calls"],
      message: "A model response cannot repeat an SDK tool call identity"
    });
  }
});

const ModelCallFailedSchema = z.object({
  version: z.literal(1),
  lifecycle: z.literal("failed"),
  model_call_id: z.string().uuid(),
  agent_id: z.string().trim().min(1),
  cycle: AutonomousCycleRefSchema.optional(),
  at: z.string().datetime()
}).strict();

export const ModelCallLifecycleRecordSchema = z.discriminatedUnion("lifecycle", [
  ModelCallStartedSchema,
  ModelCallCompletedSchema,
  ModelCallFailedSchema
]);

export type ModelCallLifecycleRecord = z.infer<typeof ModelCallLifecycleRecordSchema>;
export type ModelCallAuthority = z.infer<typeof ModelCallCompletedSchema> & {
  started_at: string;
};

export function rebuildModelCallAuthorities(
  records: readonly unknown[]
): Map<string, ModelCallAuthority> {
  const started = new Map<string, z.infer<typeof ModelCallStartedSchema>>();
  const terminal = new Set<string>();
  const authorities = new Map<string, ModelCallAuthority>();
  for (const raw of records) {
    const record = ModelCallLifecycleRecordSchema.parse(raw);
    if (record.lifecycle === "started") {
      if (started.has(record.model_call_id) || terminal.has(record.model_call_id)) {
        throw new Error(`Duplicate model call start: ${record.model_call_id}`);
      }
      started.set(record.model_call_id, record);
      continue;
    }
    const origin = started.get(record.model_call_id);
    if (!origin || origin.agent_id !== record.agent_id || terminal.has(record.model_call_id)) {
      throw new Error(`Model call terminal record has no matching start: ${record.model_call_id}`);
    }
    if ((origin.cycle !== undefined || record.cycle !== undefined)
      && !sameAutonomousCycle(origin.cycle, record.cycle)) {
      throw new Error(`Model call cycle identity changed: ${record.model_call_id}`);
    }
    if (record.at < origin.at) {
      throw new Error(`Model call terminal record precedes its start: ${record.model_call_id}`);
    }
    terminal.add(record.model_call_id);
    if (record.lifecycle === "completed") {
      authorities.set(record.model_call_id, {
        ...record,
        started_at: origin.at
      });
    }
  }
  return authorities;
}

export function authorityForToolCall(
  authorities: ReadonlyMap<string, ModelCallAuthority>,
  agentId: string,
  toolCallId: string,
  toolName: string
): ModelCallAuthority | undefined {
  const matches = [...authorities.values()].filter((authority) => (
    authority.agent_id === agentId
      && authority.tool_calls.some((toolCall) => (
        toolCall.tool_call_id === toolCallId && toolCall.tool_name === toolName
      ))
  ));
  return matches.length === 1 ? structuredClone(matches[0]) : undefined;
}

export function modelPayloadSha256(value: unknown): string {
  return createHash("sha256")
    .update(JSON.stringify(canonicalValue(value)))
    .digest("hex");
}

export function modelToolArgumentsSha256(rawArguments: string): string {
  let payload: unknown;
  try {
    payload = JSON.parse(rawArguments);
  } catch {
    payload = rawArguments;
  }
  return modelPayloadSha256(payload);
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([key, entry]) => [key, canonicalValue(entry)]));
  }
  return value;
}
