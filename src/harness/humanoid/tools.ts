import {
  tool,
  type FunctionTool,
  type Tool
} from "@openai/agents";
import { z } from "zod";
import {
  HumanoidActionDescriptions,
  HumanoidActionInputs,
  type HumanoidActionName
} from "./actions.js";
import type { JsonValue } from "../../domain/schema.js";
import {
  modelPayloadSha256,
  modelToolArgumentsSha256
} from "../../domain/model-call-authority.js";
import {
  createToolInputRecovery,
  invalidToolInputResult,
} from "../tool-input-recovery.js";
import type {
  HumanoidActionInvoker,
  HumanoidActionReceipt,
} from "./runtime.js";
import {
  HUMANOID_EXPERIENCE_OUTCOMES,
  HUMANOID_GOAL_PREDICATE_TYPES
} from "./embodied-recall.js";
import { modelToolReceiptDetail } from "./receipt-context.js";

export type { HumanoidActionInvoker } from "./runtime.js";

const HumanoidEmbodiedRecallInputSchema = z.object({
  source_refs: z.array(
    z.string().max(320).regex(/^(?:episode:[1-9]\d*|action:\S+)$/)
  ).max(64).nullable().optional()
    .describe("要精确召回的 episode:N 或 action:transactionId 来源标识"),
  before_sequence: z.number().int().positive().nullable().optional()
    .describe("分页时只返回指定 episode sequence 之前的历史"),
  before_experience_sequence: z.number().int().positive().nullable().optional()
    .describe("语义经验分页时只返回指定 experience sequence 之前的历史"),
  outcomes: z.array(z.enum(HUMANOID_EXPERIENCE_OUTCOMES)).max(3)
    .nullable().optional()
    .describe("按真实执行结果筛选经验；同一字段内为任一匹配"),
  predicate_types: z.array(z.enum(HUMANOID_GOAL_PREDICATE_TYPES)).max(7)
    .nullable().optional()
    .describe("按当时模型 Goal 的谓词类型筛选经验"),
  object_ids: z.array(z.string().trim().min(1).max(160)).max(32)
    .nullable().optional()
    .describe("按当时 Goal 引用的对象筛选经验"),
  solid_ids: z.array(z.string().trim().min(1).max(160)).max(32)
    .nullable().optional()
    .describe("按当时 Goal 或世界修改引用的静态方块筛选经验"),
  zone_ids: z.array(z.string().trim().min(1).max(160)).max(32)
    .nullable().optional()
    .describe("按当时 Goal 引用的区域筛选经验"),
  limit: z.number().int().min(1).max(64)
    .describe("本次最多返回的历史事件数")
}).strict().superRefine((input, context) => {
  if (input.source_refs
    && new Set(input.source_refs).size !== input.source_refs.length) {
    context.addIssue({
      code: "custom",
      path: ["source_refs"],
      message: "Embodied history source references must be unique"
    });
  }
  if (input.source_refs && input.source_refs.length > input.limit) {
    context.addIssue({
      code: "custom",
      path: ["limit"],
      message: "Recall limit must cover every requested source reference"
    });
  }
  for (const field of [
    "outcomes",
    "predicate_types",
    "object_ids",
    "solid_ids",
    "zone_ids"
  ] as const) {
    const values = input[field];
    if (values && new Set(values).size !== values.length) {
      context.addIssue({
        code: "custom",
        path: [field],
        message: "Embodied semantic recall filters must be unique"
      });
    }
  }
  const semantic = input.before_experience_sequence != null
    || (input.outcomes?.length ?? 0) > 0
    || (input.predicate_types?.length ?? 0) > 0
    || (input.object_ids?.length ?? 0) > 0
    || (input.solid_ids?.length ?? 0) > 0
    || (input.zone_ids?.length ?? 0) > 0;
  if (semantic && (input.source_refs?.length ?? 0) > 0) {
    context.addIssue({
      code: "custom",
      path: ["source_refs"],
      message: "Exact source recall and semantic experience filters are separate queries"
    });
  }
});

export interface HumanoidEmbodiedRecallInvoker {
  recallEmbodiedHistory(request: {
    source_refs?: string[];
    before_sequence?: number;
    before_experience_sequence?: number;
    outcomes?: Array<typeof HUMANOID_EXPERIENCE_OUTCOMES[number]>;
    predicate_types?: Array<typeof HUMANOID_GOAL_PREDICATE_TYPES[number]>;
    object_ids?: string[];
    solid_ids?: string[];
    zone_ids?: string[];
    limit: number;
  }): Promise<JsonValue>;
}

export function createHumanoidActionTools(
  runtime: HumanoidActionInvoker,
  agentId: string,
  allowedActions: readonly HumanoidActionName[] = humanoidActionNames()
): Tool[] {
  const unique = new Set(allowedActions);
  if (unique.size !== allowedActions.length) {
    throw new Error(`Duplicate humanoid action grant for ${agentId}`);
  }
  return allowedActions.map((name) => humanoidActionTool(runtime, agentId, name));
}

export function createHumanoidEmbodiedRecallTool(
  runtime: HumanoidEmbodiedRecallInvoker
): FunctionTool<unknown, typeof HumanoidEmbodiedRecallInputSchema, string> {
  const name = "recall_embodied_history";
  return tool<typeof HumanoidEmbodiedRecallInputSchema, unknown, string>({
    name,
    description: "只读召回带来源标识的具身历史。既可按 episode:N 或 action:transactionId 精确读取，也可按真实结果、Goal 谓词、对象、静态方块和区域检索持久经验。所有结果均为 historical_only，不代表当前传感或当前物理状态。",
    parameters: HumanoidEmbodiedRecallInputSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: (_context, error) => historicalValidationError(error, name),
    execute: async (input) => historicalRecallOutput(
      await runtime.recallEmbodiedHistory({
        ...(input.source_refs ? { source_refs: input.source_refs } : {}),
        ...(input.before_sequence != null
          ? { before_sequence: input.before_sequence }
          : {}),
        ...(input.before_experience_sequence != null
          ? { before_experience_sequence: input.before_experience_sequence }
          : {}),
        ...(input.outcomes?.length ? { outcomes: input.outcomes } : {}),
        ...(input.predicate_types?.length
          ? { predicate_types: input.predicate_types }
          : {}),
        ...(input.object_ids?.length ? { object_ids: input.object_ids } : {}),
        ...(input.solid_ids?.length ? { solid_ids: input.solid_ids } : {}),
        ...(input.zone_ids?.length ? { zone_ids: input.zone_ids } : {}),
        limit: input.limit
      })
    )
  });
}

function humanoidActionTool(
  runtime: HumanoidActionInvoker,
  agentId: string,
  name: HumanoidActionName
): FunctionTool<unknown, z.ZodObject, string> {
  const parameters: z.ZodObject = HumanoidActionInputs[name];
  const inputRecovery = createToolInputRecovery();
  const actionTool = tool<z.ZodObject, unknown, string>({
    name,
    description: HumanoidActionDescriptions[name],
    parameters,
    strict: true,
    isEnabled: () => runtime.isActionAvailable?.(name, agentId) ?? true,
    timeoutBehavior: "raise_exception",
    errorFunction: (_context, error) => invalidToolInputResult(error, name),
    execute: async (input, _context, details) => {
      const toolCall = details?.toolCall;
      const transactionId = toolCall?.callId;
      if (!transactionId) throw new Error(`SDK did not provide a call ID for ${name}`);
      if (toolCall.name !== name) {
        throw new Error(`SDK tool identity mismatch for ${name}`);
      }
      const argumentsSha256 = modelToolArgumentsSha256(toolCall.arguments);
      const normalizedArgumentsSha256 = modelPayloadSha256(input);
      const receipt: HumanoidActionReceipt = await runtime.invoke(
        name,
        input,
        transactionId,
        agentId,
        {
          tool_call_id: transactionId,
          tool_name: name,
          arguments_sha256: argumentsSha256,
          ...(normalizedArgumentsSha256 === argumentsSha256
            ? {}
            : { normalized_arguments_sha256: normalizedArgumentsSha256 })
        }
      );
      return humanoidActionReceiptModelOutput(receipt);
    }
  });
  if (name === "plan_whole_body_motion_candidates") {
    actionTool.parameters = referencedJsonSchema(parameters);
  }
  const invoke = actionTool.invoke;
  actionTool.invoke = async (context, input, details) => {
    const rejection = inputRecovery.preflight(input, parameters, name);
    if (rejection !== undefined) return rejection;
    return invoke(context, input, details);
  };
  return actionTool;
}

function humanoidActionReceiptModelOutput(receipt: HumanoidActionReceipt): string {
  return JSON.stringify({
    transactionId: receipt.transactionId,
    agentId: receipt.agentId,
    action: receipt.action,
    accepted: receipt.accepted,
    code: receipt.code,
    worldBeforeRevision: receipt.worldBeforeRevision,
    worldAfterRevision: receipt.worldAfterRevision,
    frameCount: receipt.frameCount,
    channels: receipt.channels,
    detail: modelToolReceiptDetail(receipt),
    committedAt: receipt.committedAt,
    durable_evidence: {
      source_ref: `action:${receipt.transactionId}`,
      receipt_sha256: modelPayloadSha256(receipt),
      full_receipt_persisted: true
    }
  });
}

function referencedJsonSchema(
  parameters: z.ZodObject
): FunctionTool<unknown, z.ZodObject, string>["parameters"] {
  const schema = z.toJSONSchema(parameters, {
    cycles: "ref",
    reused: "ref",
    unrepresentable: "any"
  });
  if (schema.type !== "object" || schema.properties === undefined) {
    throw new Error("Humanoid action parameters must produce an object JSON schema");
  }
  return schema as FunctionTool<unknown, z.ZodObject, string>["parameters"];
}

function humanoidActionNames(): HumanoidActionName[] {
  return Object.keys(HumanoidActionInputs) as HumanoidActionName[];
}

function historicalRecallOutput(value: JsonValue): string {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Embodied history recall returned a non-object payload");
  }
  return JSON.stringify({
    ...value,
    historical_only: true
  });
}

function historicalValidationError(error: unknown, toolName: string): string {
  const recovered = JSON.parse(
    invalidToolInputResult(error, toolName)
  ) as Record<string, unknown>;
  return JSON.stringify({
    ...recovered,
    historical_only: true
  });
}
