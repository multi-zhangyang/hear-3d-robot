import { tool, type FunctionTool } from "@openai/agents";
import { z } from "zod";
import { goalSha256 } from "../../domain/goal-identity.js";
import {
  modelPayloadSha256,
  modelToolArgumentsSha256
} from "../../domain/model-call-authority.js";
import {
  GoalSchema,
  Vec3Schema,
  type JsonValue
} from "../../domain/schema.js";
import { GOAL_RETIREMENT_STATUSES } from "../../domain/goal-epoch-retirement.js";
import {
  createToolInputRecovery,
  invalidToolInputResult,
  recoverInvalidToolInputOutput
} from "../tool-input-recovery.js";
import {
  GOAL_HISTORY_PREDICATE_TYPES,
  GOAL_HISTORY_STATUSES,
  type GoalHistoryRecallRequest
} from "./goal-history.js";

const CandidateIdSchema = z.string().regex(/^goal-candidate:[a-f0-9]{64}$/)
  .describe(
    "只能逐字复制 CURRENT GOAL MANAGER INVOCATION 的 existing_goal_candidate_ids；"
    + "不能填写本批 proposal_id，列表为空时依赖必须为空"
  );

const CandidateSequenceSchema = z.number().int().positive().describe(
  "逐字复制候选提交回执或 CURRENT GOAL MANAGER INVOCATION 中的 candidate_sequence"
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
  input.candidates.forEach((candidate, candidateIndex) => {
    const predicateHashes = candidate.goal.predicates.map(modelPayloadSha256);
    if (new Set(predicateHashes).size === predicateHashes.length) return;
    context.addIssue({
      code: "custom",
      path: ["candidates", candidateIndex, "goal", "predicates"],
      message: "A Goal candidate cannot repeat an identical predicate"
    });
  });
});

const SelectGoalCandidateSchema = z.object({
  candidate_sequence: CandidateSequenceSchema
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

const ContinueGoalEpochSchema = z.object({
  reason: z.string().trim().min(1)
}).strict();

const RecallGoalHistorySchema = z.object({
  candidate_ids: z.array(CandidateIdSchema).max(32).nullable().optional()
    .describe("精确召回的 Goal candidate 标识"),
  before_candidate_sequence: z.number().int().positive().nullable().optional()
    .describe("分页时只返回该 candidate sequence 之前的历史"),
  statuses: z.array(z.enum(GOAL_HISTORY_STATUSES)).max(GOAL_HISTORY_STATUSES.length)
    .nullable().optional(),
  predicate_types: z.array(z.enum(GOAL_HISTORY_PREDICATE_TYPES))
    .max(GOAL_HISTORY_PREDICATE_TYPES.length).nullable().optional(),
  object_ids: z.array(z.string().trim().min(1).max(160)).max(32)
    .nullable().optional(),
  solid_ids: z.array(z.string().trim().min(1).max(160)).max(32)
    .nullable().optional(),
  zone_ids: z.array(z.string().trim().min(1).max(160)).max(32)
    .nullable().optional(),
  world_region: z.object({
    center: Vec3Schema,
    horizontal_radius_m: z.number().finite().positive().max(1_000_000),
    vertical_radius_m: z.number().finite().positive().max(1_000_000)
      .nullable().optional()
  }).strict().nullable().optional()
    .describe("按世界空间范围召回 robot_at、object_at 或 world-frame end_effector_at 历史"),
  limit: z.number().int().min(1).max(32)
}).strict().superRefine((input, context) => {
  for (const field of [
    "candidate_ids",
    "statuses",
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
        message: "Goal history recall values must be unique"
      });
    }
  }
  if (input.candidate_ids && input.candidate_ids.length > input.limit) {
    context.addIssue({
      code: "custom",
      path: ["limit"],
      message: "Goal history recall limit must cover every exact candidate identity"
    });
  }
  if (input.candidate_ids?.length && input.before_candidate_sequence != null) {
    context.addIssue({
      code: "custom",
      path: ["before_candidate_sequence"],
      message: "Exact Goal candidate recall cannot use chronological pagination"
    });
  }
});

export interface GoalManagerRuntime {
  recallGoalHistory(request: GoalHistoryRecallRequest): Promise<JsonValue>;
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
  continueGoalEpoch(
    input: z.infer<typeof ContinueGoalEpochSchema>,
    authority: GoalToolCallAuthority
  ): Promise<JsonValue>;
}

export interface GoalToolCallAuthority {
  tool_call_id: string;
  tool_name: string;
  arguments_sha256: string;
  normalized_arguments_sha256?: string;
}

export function createGoalManagerTools(
  runtime: GoalManagerRuntime
): Array<FunctionTool<unknown, z.ZodObject, string>> {
  return [
    goalHistoryTool(runtime),
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
    ),
    goalTool(
      "continue_goal_epoch",
      ContinueGoalEpochSchema,
      (input, authority) => runtime.continueGoalEpoch(
        ContinueGoalEpochSchema.parse(input),
        authority
      )
    )
  ];
}

function goalHistoryTool(
  runtime: GoalManagerRuntime
): FunctionTool<unknown, typeof RecallGoalHistorySchema, string> {
  const name = "recall_goal_history";
  const inputRecovery = createToolInputRecovery();
  const historyTool = tool<typeof RecallGoalHistorySchema, unknown, string>({
    name,
    description: "只读召回完整 Goal DAG 中未装入当前工作集的候选与结果，并区分 selected 与 not_selected。可按 candidate、状态、谓词、对象、方块、语义区域或世界空间范围检索；历史结果不能代替当前物理观察。",
    parameters: RecallGoalHistorySchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: (_context, error) => historicalGoalError(error, name),
    execute: async (input) => JSON.stringify(await runtime.recallGoalHistory({
      ...(input.candidate_ids?.length ? { candidate_ids: input.candidate_ids } : {}),
      ...(input.before_candidate_sequence != null
        ? { before_candidate_sequence: input.before_candidate_sequence }
        : {}),
      ...(input.statuses?.length ? { statuses: input.statuses } : {}),
      ...(input.predicate_types?.length
        ? { predicate_types: input.predicate_types }
        : {}),
      ...(input.object_ids?.length ? { object_ids: input.object_ids } : {}),
      ...(input.solid_ids?.length ? { solid_ids: input.solid_ids } : {}),
      ...(input.zone_ids?.length ? { zone_ids: input.zone_ids } : {}),
      ...(input.world_region
        ? {
            world_region: {
              center: input.world_region.center,
              horizontal_radius_m: input.world_region.horizontal_radius_m,
              ...(input.world_region.vertical_radius_m == null
                ? {}
                : { vertical_radius_m: input.world_region.vertical_radius_m })
            }
          }
        : {}),
      limit: input.limit
    }))
  });
  const invoke = historyTool.invoke;
  historyTool.invoke = async (context, input, details) => {
    const output = await invoke(context, input, details);
    return recoverInvalidToolInputOutput(
      output,
      input,
      RecallGoalHistorySchema,
      name,
      inputRecovery
    );
  };
  return historyTool;
}

function goalTool(
  name: string,
  parameters: z.ZodObject,
  invoke: (
    input: Record<string, unknown>,
    authority: GoalToolCallAuthority
  ) => Promise<JsonValue>
): FunctionTool<unknown, z.ZodObject, string> {
  const inputRecovery = createToolInputRecovery();
  const transitionTool = tool<z.ZodObject, unknown, string>({
    name,
    description: goalToolDescription(name),
    parameters,
    strict: true,
    timeoutBehavior: "raise_exception",
    errorFunction: (_context, error) => invalidToolInputResult(error, name),
    execute: async (input, _context, details) => {
      const toolCall = details?.toolCall;
      const toolCallId = toolCall?.callId;
      if (!toolCallId) throw new Error(`SDK did not provide a call ID for ${name}`);
      if (toolCall.name !== name) {
        throw new Error(`SDK tool identity mismatch for ${name}`);
      }
      const argumentsSha256 = modelToolArgumentsSha256(toolCall.arguments);
      const normalizedArgumentsSha256 = modelPayloadSha256(input);
      try {
        return JSON.stringify(await invoke(input, {
          tool_call_id: toolCallId,
          tool_name: name,
          arguments_sha256: argumentsSha256,
          ...(normalizedArgumentsSha256 === argumentsSha256
            ? {}
            : { normalized_arguments_sha256: normalizedArgumentsSha256 })
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
          recovery: "根据本回执与本次 CURRENT GOAL MANAGER INVOCATION 重新调用正确的 Goal 工具；Harness 不会修补候选、选择替代目标或执行动作。"
        });
      }
    }
  });
  const sdkInvoke = transitionTool.invoke;
  transitionTool.invoke = async (context, input, details) => {
    const output = await sdkInvoke(context, input, details);
    return recoverInvalidToolInputOutput(
      output,
      input,
      parameters,
      name,
      inputRecovery
    );
  };
  return transitionTool;
}

function goalToolDescription(name: string): string {
  if (name === "submit_goal_candidates") {
    return "一次提交 2–3 个由当前模型提出的长期任务候选。每个候选必须绑定当前物理证据、可观察谓词、依赖和任务推进关系；Harness 不生成或补充候选。";
  }
  if (name === "select_goal_candidate") {
    return "用候选回执中的短序号显式选择一个已提交且依赖已完成的 Goal。短序号与持久候选身份一对一对应；Harness 不打分、不随机选择，也不提供替代目标。";
  }
  if (name === "continue_goal_epoch") {
    return "目标重评后保留当前 active Goal，结束已经失败的自主周期并以新的恢复预算开始下一周期。只有 Goal 仍物理可达、但当前周期的 compact replan 已耗尽时使用。只说明 Goal 为何仍应继续；不得指定 Skill、手、交互点、坐标、路线或动作参数，下一周期必须从新鲜物理观察重新选择。";
  }
  return "用当前物理证据将 active Goal 退役为 blocked、abandoned、superseded 或 expired。退役后必须由后续模型调用重新提交并选择，Harness 不自动替换。";
}

function historicalGoalError(error: unknown, toolName: string): string {
  const recovered = JSON.parse(
    invalidToolInputResult(error, toolName)
  ) as Record<string, unknown>;
  return JSON.stringify({ ...recovered, historical_only: true });
}
