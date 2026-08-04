import { tool, type FunctionTool } from "@openai/agents";
import { z } from "zod";
import { goalSha256 } from "../../domain/goal-identity.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import { GoalSchema, type JsonValue } from "../../domain/schema.js";
import { GOAL_RETIREMENT_STATUSES } from "../../domain/goal-epoch-retirement.js";
import { invalidToolInputResult } from "../tool-input-recovery.js";

const CandidateIdSchema = z.string().regex(/^goal-candidate:[a-f0-9]{64}$/)
  .describe(
    "只能逐字复制 CURRENT HARNESS AUTHORITY 的 existing_goal_candidate_ids；"
    + "不能填写本批 proposal_id，列表为空时依赖必须为空"
  );

const SubmitGoalCandidatesSchema = z.object({
  candidates: z.array(z.object({
    proposal_id: z.string().trim().min(1),
    mission_link: z.string().trim().min(1),
    goal: GoalSchema,
    dependency_candidate_ids: z.array(CandidateIdSchema).max(32)
      .describe("仅引用提交前已经存在的 Goal candidate；初始候选批次填写 []")
  }).strict()).min(2).max(3)
}).strict().superRefine((input, context) => {
  const proposalIds = input.candidates.map((candidate) => candidate.proposal_id);
  if (new Set(proposalIds).size !== proposalIds.length) {
    context.addIssue({
      code: "custom",
      path: ["candidates"],
      message: "Goal candidate proposal identities must be unique"
    });
  }
  const contentHashes = input.candidates.map((candidate) => goalSha256(candidate.goal));
  if (new Set(contentHashes).size !== contentHashes.length) {
    context.addIssue({
      code: "custom",
      path: ["candidates"],
      message: "A Goal candidate batch must contain distinct Goal contents"
    });
  }
});

const SelectGoalCandidateSchema = z.object({
  candidate_id: CandidateIdSchema.describe(
    "逐字复制 submit_goal_candidates 成功回执中的 candidate_ids 之一"
  )
}).strict();

const RetireGoalEpochSchema = z.object({
  status: z.enum(GOAL_RETIREMENT_STATUSES),
  reason: z.string().trim().min(1),
  evidence_refs: z.array(z.string().trim().min(1)).min(1).max(32)
}).strict().superRefine((input, context) => {
  if (new Set(input.evidence_refs).size !== input.evidence_refs.length) {
    context.addIssue({
      code: "custom",
      path: ["evidence_refs"],
      message: "Goal retirement evidence references must be unique"
    });
  }
});

export interface GoalManagerRuntime {
  submitGoalCandidates(
    input: z.infer<typeof SubmitGoalCandidatesSchema>,
    authority: GoalToolCallAuthority
  ): Promise<JsonValue>;
  selectGoalCandidate(
    input: z.infer<typeof SelectGoalCandidateSchema>,
    authority: GoalToolCallAuthority
  ): Promise<JsonValue>;
  retireGoalEpoch(
    input: z.infer<typeof RetireGoalEpochSchema>,
    authority: GoalToolCallAuthority
  ): Promise<JsonValue>;
}

export interface GoalToolCallAuthority {
  tool_call_id: string;
  tool_name: string;
  arguments_sha256: string;
}

export function createGoalManagerTools(
  runtime: GoalManagerRuntime
): Array<FunctionTool<unknown, z.ZodObject, string>> {
  return [
    goalTool(
      "submit_goal_candidates",
      SubmitGoalCandidatesSchema,
      (input, authority) => runtime.submitGoalCandidates(
        SubmitGoalCandidatesSchema.parse(input),
        authority
      )
    ),
    goalTool(
      "select_goal_candidate",
      SelectGoalCandidateSchema,
      (input, authority) => runtime.selectGoalCandidate(
        SelectGoalCandidateSchema.parse(input),
        authority
      )
    ),
    goalTool(
      "retire_goal_epoch",
      RetireGoalEpochSchema,
      (input, authority) => runtime.retireGoalEpoch(
        RetireGoalEpochSchema.parse(input),
        authority
      )
    )
  ];
}

function goalTool(
  name: string,
  parameters: z.ZodObject,
  invoke: (
    input: Record<string, unknown>,
    authority: GoalToolCallAuthority
  ) => Promise<JsonValue>
): FunctionTool<unknown, z.ZodObject, string> {
  return tool<z.ZodObject, unknown, string>({
    name,
    description: goalToolDescription(name),
    parameters,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: (_context, error) => invalidToolInputResult(error, name),
    execute: async (input, _context, details) => {
      const toolCallId = details?.toolCall?.callId;
      if (!toolCallId) throw new Error(`SDK did not provide a call ID for ${name}`);
      try {
        return JSON.stringify(await invoke(input, {
          tool_call_id: toolCallId,
          tool_name: name,
          arguments_sha256: modelPayloadSha256(input)
        }));
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") throw error;
        return JSON.stringify({
          accepted: false,
          code: "goal_transition_rejected",
          tool: name,
          error: error instanceof Error ? error.message : String(error),
          automatic_selection: false,
          automatic_actuation: false,
          recovery: "根据本回执与最新 CURRENT HARNESS AUTHORITY 重新调用正确的 Goal 工具；Harness 不会修补候选、选择替代目标或执行动作。"
        });
      }
    }
  });
}

function goalToolDescription(name: string): string {
  if (name === "submit_goal_candidates") {
    return "一次提交 2–3 个由当前模型提出的长期任务候选。每个候选必须绑定当前物理证据、可观察谓词、依赖和任务推进关系；Harness 不生成或补充候选。";
  }
  if (name === "select_goal_candidate") {
    return "从已提交且依赖已完成的候选中显式选择下一 Goal epoch。Harness 不打分、不随机选择，也不提供替代目标。";
  }
  return "用当前物理证据将 active Goal 退役为 blocked、abandoned、superseded 或 expired。退役后必须由后续模型调用重新提交并选择，Harness 不自动替换。";
}
