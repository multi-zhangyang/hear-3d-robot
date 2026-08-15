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
  recoverInvalidToolInputOutput
} from "../tool-input-recovery.js";
import type {
  HumanoidActionInvoker,
  HumanoidActionReceipt,
  HumanoidActionToolCallAuthority,
} from "./runtime.js";
import type { NeuralRolloutExecutionAdmission } from
  "../../domain/action-execution-ledger.js";
import {
  HUMANOID_EXPERIENCE_OUTCOMES,
  HUMANOID_GOAL_PREDICATE_TYPES
} from "./embodied-recall.js";
import { modelToolReceiptDetail } from "./receipt-context.js";
import { HUMANOID_NEURAL_AGENT_IDS } from "./neural-hierarchy-contract.js";

export type { HumanoidActionInvoker } from "./runtime.js";

const HumanoidEmbodiedRecallInputSchema = z.object({
  query_mode: z.enum(["chronological_or_exact", "semantic"])
    .describe("chronological_or_exact 按时间或 source_refs 精确读取；semantic 按结果和 Goal 实体筛选经验。两种模式互斥"),
  source_refs: z.array(
    z.string().max(320).regex(/^(?:episode:[1-9]\d*|action:\S+)$/)
  ).max(64).nullable().optional()
    .describe("仅 chronological_or_exact 模式使用；要精确召回的 episode:N 或 action:transactionId 来源标识"),
  before_sequence: z.number().int().positive().nullable().optional()
    .describe("仅 chronological_or_exact 模式使用；分页时只返回指定 episode sequence 之前的历史"),
  before_experience_sequence: z.number().int().positive().nullable().optional()
    .describe("仅 semantic 模式使用；语义经验分页时只返回指定 experience sequence 之前的历史"),
  outcomes: z.array(z.enum(HUMANOID_EXPERIENCE_OUTCOMES)).max(3)
    .nullable().optional()
    .describe("按真实执行结果筛选经验；同一字段内为任一匹配"),
  predicate_types: z.array(z.enum(HUMANOID_GOAL_PREDICATE_TYPES))
    .max(HUMANOID_GOAL_PREDICATE_TYPES.length)
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
  if (input.query_mode === "semantic" && !semantic) {
    context.addIssue({
      code: "custom",
      path: ["query_mode"],
      message: "Semantic recall requires at least one semantic filter or before_experience_sequence"
    });
  }
  if (input.query_mode === "semantic" && input.before_sequence != null) {
    context.addIssue({
      code: "custom",
      path: ["before_sequence"],
      message: "Semantic recall must set before_sequence to null"
    });
  }
  if (input.query_mode === "semantic" && (input.source_refs?.length ?? 0) > 0) {
    context.addIssue({
      code: "custom",
      path: ["source_refs"],
      message: "Semantic recall must set source_refs to null; the current failure is already present in the Recovery input"
    });
  }
  if (input.query_mode === "chronological_or_exact" && semantic) {
    context.addIssue({
      code: "custom",
      path: ["query_mode"],
      message: "Semantic filters require query_mode=semantic; otherwise set every semantic filter to null or []"
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
  allowedActions: readonly HumanoidActionName[] = humanoidActionNames(),
  options: { availability?: "dynamic" | "stable" } = {}
): Tool[] {
  const unique = new Set(allowedActions);
  if (unique.size !== allowedActions.length) {
    throw new Error(`Duplicate humanoid action grant for ${agentId}`);
  }
  if (agentId === HUMANOID_NEURAL_AGENT_IDS.motorIntent
    && unique.has("plan_whole_body_motion")) {
    throw new Error(
      "Production Motion Agent cannot receive raw dense-motion authoring authority"
    );
  }
  return allowedActions.map((name) => humanoidActionTool(
    runtime,
    agentId,
    name,
    options.availability ?? "dynamic"
  ));
}

/**
 * Execute a physical/runtime action from the owning hierarchical tool call.
 *
 * The owning parent episode remains the model decision authority. The
 * deterministic service is only the actor: it cannot invent a second model
 * response, rewrite the accepted plan, or manufacture a tool-call identity.
 */
export async function invokeDeterministicHumanoidAction(input: {
  runtime: HumanoidActionInvoker;
  actorAgentId: string;
  sourceToolName: string;
  sourceInput: unknown;
  action: HumanoidActionName;
  actionInput: unknown;
  contractId: NonNullable<
    HumanoidActionToolCallAuthority["deterministic_delegation"]
  >["contract_id"];
  neuralRolloutCertificate?: NeuralRolloutExecutionAdmission;
  details?: {
    toolCall?: {
      callId?: string;
      name?: string;
      arguments: string;
    };
    signal?: AbortSignal;
  };
  /**
   * Runtime-only observer for deterministic hierarchy services that need the
   * authoritative, unprojected receipt. The model still receives only the
   * bounded projection returned by humanoidActionReceiptModelOutput().
   */
  onReceipt?: (receipt: HumanoidActionReceipt) => void | Promise<void>;
}): Promise<string> {
  const toolCall = input.details?.toolCall;
  const transactionId = toolCall?.callId;
  if (!transactionId) {
    throw new Error(`SDK did not provide a call ID for ${input.sourceToolName}`);
  }
  if (toolCall.name !== input.sourceToolName) {
    throw new Error(`SDK tool identity mismatch for ${input.sourceToolName}`);
  }
  const argumentsSha256 = modelToolArgumentsSha256(toolCall.arguments);
  const normalizedArgumentsSha256 = modelPayloadSha256(input.sourceInput);
  const authority: HumanoidActionToolCallAuthority = {
    tool_call_id: transactionId,
    tool_name: input.sourceToolName,
    arguments_sha256: argumentsSha256,
    ...(normalizedArgumentsSha256 === argumentsSha256
      ? {}
      : { normalized_arguments_sha256: normalizedArgumentsSha256 }),
    deterministic_delegation: {
      contract_id: input.contractId,
      source_input: input.sourceInput as JsonValue,
      action_input_sha256: modelPayloadSha256(input.actionInput)
    }
  };
  const invocationOptions = {
    ...(input.details?.signal ? { signal: input.details.signal } : {}),
    ...(input.neuralRolloutCertificate
      ? { neuralRolloutCertificate: input.neuralRolloutCertificate }
      : {})
  };
  const receipt = input.details?.signal || input.neuralRolloutCertificate
    ? await input.runtime.invoke(
        input.action,
        input.actionInput,
        transactionId,
        input.actorAgentId,
        authority,
        invocationOptions
      )
    : await input.runtime.invoke(
        input.action,
        input.actionInput,
        transactionId,
        input.actorAgentId,
        authority
      );
  await input.onReceipt?.(structuredClone(receipt));
  return humanoidActionReceiptModelOutput(receipt);
}

export function createHumanoidEmbodiedRecallTool(
  runtime: HumanoidEmbodiedRecallInvoker
): FunctionTool<unknown, typeof HumanoidEmbodiedRecallInputSchema, string> {
  const name = "recall_embodied_history";
  const inputRecovery = createToolInputRecovery();
  const recallTool = tool<typeof HumanoidEmbodiedRecallInputSchema, unknown, string>({
    name,
    description: "只读召回带来源标识的具身历史。必须明确选择一种互斥 query_mode：chronological_or_exact 按 episode:N、action:transactionId 或时间读取；semantic 按真实结果、Goal 谓词、对象、静态方块和区域检索持久经验。所有结果均为 historical_only，不代表当前传感或当前物理状态。",
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
  const invoke = recallTool.invoke;
  recallTool.invoke = async (context, input, details) => {
    const output = await invoke(context, input, details);
    return recoverInvalidToolInputOutput(
      output,
      input,
      HumanoidEmbodiedRecallInputSchema,
      name,
      inputRecovery
    );
  };
  return recallTool;
}

function humanoidActionTool(
  runtime: HumanoidActionInvoker,
  agentId: string,
  name: HumanoidActionName,
  availability: "dynamic" | "stable"
): FunctionTool<unknown, z.ZodObject, string> {
  const parameters: z.ZodObject = HumanoidActionInputs[name];
  const inputRecovery = createToolInputRecovery();
  const actionTool = tool<z.ZodObject, unknown, string>({
    name,
    description: HumanoidActionDescriptions[name],
    parameters,
    strict: true,
    isEnabled: () => availability === "stable"
      || (runtime.isActionAvailable?.(name, agentId) ?? true),
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
      const authority = {
        tool_call_id: transactionId,
        tool_name: name,
        arguments_sha256: argumentsSha256,
        ...(normalizedArgumentsSha256 === argumentsSha256
          ? {}
          : { normalized_arguments_sha256: normalizedArgumentsSha256 })
      };
      const receipt: HumanoidActionReceipt = details?.signal
        ? await runtime.invoke(
            name,
            input,
            transactionId,
            agentId,
            authority,
            { signal: details.signal }
          )
        : await runtime.invoke(
            name,
            input,
            transactionId,
            agentId,
            authority
          );
      return humanoidActionReceiptModelOutput(receipt);
    }
  });
  if (name === "plan_whole_body_motion_candidates") {
    actionTool.parameters = referencedJsonSchema(parameters);
  }
  const invoke = actionTool.invoke;
  actionTool.invoke = async (context, input, details) => {
    const output = await invoke(context, input, details);
    return recoverInvalidToolInputOutput(
      output,
      input,
      parameters,
      name,
      inputRecovery
    );
  };
  return actionTool;
}

export function humanoidActionReceiptModelOutput(
  receipt: HumanoidActionReceipt
): string {
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
  removeJsonSchemaDefaults(schema);
  return schema as FunctionTool<unknown, z.ZodObject, string>["parameters"];
}

function removeJsonSchemaDefaults(value: unknown): void {
  if (Array.isArray(value)) {
    value.forEach(removeJsonSchemaDefaults);
    return;
  }
  if (value === null || typeof value !== "object") return;
  const record = value as Record<string, unknown>;
  delete record.default;
  Object.values(record).forEach(removeJsonSchemaDefaults);
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
