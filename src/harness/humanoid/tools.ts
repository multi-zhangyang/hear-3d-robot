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
import { invalidToolInputResult } from "../tool-input-recovery.js";
import type {
  HumanoidActionReceipt,
  HumanoidActionRuntime
} from "./runtime.js";

export type HumanoidActionInvoker = Pick<HumanoidActionRuntime, "invoke">;

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
