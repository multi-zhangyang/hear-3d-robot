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
import { invalidToolInputResult } from "../tool-input-recovery.js";
import type {
  HumanoidActionReceipt,
  HumanoidActionRuntime
} from "./runtime.js";

export type HumanoidActionInvoker = Pick<HumanoidActionRuntime, "invoke">;

const HumanoidEmbodiedRecallInputSchema = z.object({
  source_refs: z.array(
    z.string().max(320).regex(/^(?:episode:[1-9]\d*|action:\S+)$/)
  ).max(64).nullable().optional()
    .describe("要精确召回的 episode:N 或 action:transactionId 来源标识"),
  before_sequence: z.number().int().positive().nullable().optional()
    .describe("分页时只返回指定 episode sequence 之前的历史"),
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
});

export interface HumanoidEmbodiedRecallInvoker {
  recallEmbodiedHistory(request: {
    source_refs?: string[];
    before_sequence?: number;
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
    description: "只读召回带来源标识的具身历史，包括完成的 episode 与 actions.jsonl 中真实的 execute_* 成功、停滞或失败回执。返回内容始终标记 historical_only=true，不代表当前传感、当前可见性或当前物理状态。",
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
  return tool<z.ZodObject, unknown, string>({
    name,
    description: HumanoidActionDescriptions[name],
    parameters,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: (_context, error) => invalidToolInputResult(error, name),
    execute: async (input, _context, details) => {
      const transactionId = details?.toolCall?.callId;
      if (!transactionId) throw new Error(`SDK did not provide a call ID for ${name}`);
      const receipt: HumanoidActionReceipt = await runtime.invoke(
        name,
        input,
        transactionId,
        agentId
      );
      return JSON.stringify(receipt);
    }
  });
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
