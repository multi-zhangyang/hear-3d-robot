/**
 * Legacy V2 five-role Agent graph.
 *
 * This module is intentionally excluded from the production mission Runner.
 * It remains only to decode and verify pre-V3 manifests/checkpoints while the
 * V3 neural hierarchy starts a fresh, isolated Agent epoch. New runtime code
 * must import neural-agents.ts and neural-hierarchy-contract.ts instead.
 */
import {
  Agent,
  Runner,
  tool,
  type CallModelInputFilter,
  type FunctionTool,
  type Model,
  type ModelSettings,
  type RunStreamEvent,
  type Session,
  type ToolInputParameters,
  type ToolUseBehavior
} from "@openai/agents";
import { z } from "zod";
import type { JsonValue } from "../../domain/schema.js";
import {
  providerConfigForRole,
  type AgentModelRole,
  type ModelProviderConfig,
  type ProviderConfig
} from "../../config/load.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import type { HumanoidActionInvoker } from "./runtime.js";
import {
  createHumanoidActionTools,
  createHumanoidEmbodiedRecallTool,
  invokeDeterministicHumanoidAction,
  type HumanoidEmbodiedRecallInvoker
} from "./tools.js";
import {
  createGoalManagerTools,
  type GoalManagerRuntime
} from "./goal-manager-tools.js";
import {
  agentInvocationMarker,
  scopeAgentToolInvocation,
  withAgentInvocation
} from "../agent-scope.js";
import { createToolInputRecovery } from "../tool-input-recovery.js";
import { ModelDecisionStallError } from "../model-telemetry.js";
import type { HumanoidCycleCompletionReadiness } from "./cycle-causal-evidence.js";
import type { HumanoidAutonomyReadiness } from "./run-runtime.js";
import { GOAL_HISTORY_PREDICATE_TYPES } from "./goal-history.js";

const SpecialistDelegationSchema = z.object({}).strict();
const ExecutePlanTaskSchema = z.object({
  kind: z.literal("execute_plan"),
  planning_action: z.enum([
    "plan_humanoid_skill",
    "plan_whole_body_motion",
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
  ]).describe("逐字复制已接受规划回执 action"),
  planning_transaction_id: z.string().trim().min(1)
    .describe("逐字复制已接受规划回执 transactionId")
}).strict();
const RemoveWorldBlockTaskSchema = z.object({
  kind: z.literal("remove_world_block"),
  solid_id: z.string().trim().min(1).describe("目标 solid_id"),
  execution_transaction_id: z.string().trim().min(1)
    .describe("先前成功物理执行的 transactionId")
}).strict();
const ExecutionTaskSchema = z.object({
  objective: z.string().trim().min(1),
  execution: z.discriminatedUnion("kind", [
    ExecutePlanTaskSchema,
    RemoveWorldBlockTaskSchema
  ])
}).strict();
const CycleCompletionSchema = z.object({
  summary: z.string().trim().min(1),
  evidence_transaction_ids: z.array(z.string().trim().min(1)).min(1),
  next_intent: z.string().trim().min(1).optional()
}).strict().superRefine((value, context) => {
  if (new Set(value.evidence_transaction_ids).size !== value.evidence_transaction_ids.length) {
    context.addIssue({
      code: "custom",
      path: ["evidence_transaction_ids"],
      message: "Cycle evidence transaction identifiers must be unique"
    });
  }
});
const SatisfiedGoalCompletionSchema = z.object({
  summary: z.string().trim().min(1)
}).strict();
export const HUMANOID_AGENT_IDS = {
  goalManager: "humanoid-goal-manager",
  coordinator: "humanoid-coordinator",
  sentry: "humanoid-sentry",
  motionPlanner: "humanoid-motion-planner",
  motion: "humanoid-motion-reference",
  executor: "humanoid-executor"
} as const;

const MOTION_PLAN_ARTIFACT_MAX_CHARACTERS = 32_000;
const MOTION_COLLABORATION_ACTIONS = new Set([
  "submit_humanoid_skill_plan",
  "begin_humanoid_skill",
  "plan_humanoid_skill",
  "plan_whole_body_motion_candidates",
  "plan_humanoid_navigation"
]);

export function goalManagerInvocationInput(
  authority: JsonValue
): string {
  const root = jsonRecord(authority);
  const goalDAG = jsonRecord(root.goal_dag);
  const goalContext = jsonRecord(root.goal_context);
  const autonomy = jsonRecord(goalContext.autonomy);
  const autonomyHistory = jsonRecord(autonomy.history);
  const continuousDriveState = jsonRecord(autonomy.continuous_drive_state);
  const observation = jsonRecord(goalContext.observation);
  const candidates = jsonRecord(goalDAG.candidates);
  const projection = jsonRecord(goalDAG.context_projection);
  const candidateReferences = Object.entries(candidates).map(([candidateId, value]) => {
    const candidate = jsonRecord(value);
    return {
      candidate_sequence: candidate.candidate_sequence ?? null,
      proposal_id: candidate.proposal_id ?? null,
      candidate_id: candidateId,
      status: candidate.status ?? null,
      goal: candidate.goal ?? null,
      mission_link: candidate.mission_link ?? null,
      dependency_candidate_ids: Array.isArray(candidate.dependency_candidate_ids)
        ? candidate.dependency_candidate_ids
        : [],
      dependency_candidates: Array.isArray(candidate.dependency_candidates)
        ? candidate.dependency_candidates
        : []
    };
  });
  const solids = Array.isArray(observation.solids)
    ? observation.solids.map(jsonRecord)
    : [];
  const objects = Array.isArray(observation.objects)
    ? observation.objects.map(jsonRecord)
    : [];
  const interaction = jsonRecord(root.interaction);
  const spatialBelief = jsonRecord(root.spatial_belief);
  const carrying = jsonRecord(interaction.carrying);
  const carryBindings = Array.isArray(carrying.bindings)
    ? carrying.bindings.map(jsonRecord)
    : [];
  const recentReceipts = Array.isArray(root.recent_receipts)
    ? root.recent_receipts.map(jsonRecord)
    : [];
  const exact = {
    run_mode: root.run_mode ?? null,
    mission_goal: root.mission_goal ?? null,
    goal_dag_status: goalDAG.status ?? null,
    autonomy_readiness: root.autonomy_readiness ?? null,
    recovery_authority: root.recovery_authority ?? null,
    previous_cycle_transition: root.previous_cycle_transition ?? null,
    goal_state: root.goal_state ?? null,
    current_goal_evidence_ref: goalContext.evidence_ref ?? null,
    existing_goal_candidate_ids: Object.keys(candidates).sort(),
    existing_goal_candidates: candidateReferences,
    current_goal_epoch_id: goalDAG.current_epoch_id ?? null,
    recent_action_evidence: recentReceipts.flatMap((receipt) => (
      typeof receipt.transaction_id === "string"
        ? [{
            evidence_ref: `action:${receipt.transaction_id}`,
            transaction_id: receipt.transaction_id,
            action: receipt.action ?? null,
            accepted: receipt.accepted ?? null,
            code: receipt.code ?? null,
            world_after_revision: receipt.world_after_revision ?? null,
            frame_count: receipt.frame_count ?? null,
            detail: receipt.detail ?? {}
          }]
        : []
    )),
    candidate_history: {
      total: projection.total_candidate_count ?? Object.keys(candidates).length,
      visible: Object.keys(candidates).length,
      truncated: projection.history_truncated === true,
      lifetime_outcomes: autonomyHistory.lifetime_outcomes ?? null
    },
    continuous_drive_state: {
      bootstrap_mission_goal_completed:
        continuousDriveState.bootstrap_mission_goal_completed ?? null,
      exact_mission_goal_history_complete:
        continuousDriveState.exact_mission_goal_history_complete ?? null,
      working_exact_mission_goal_outcomes:
        continuousDriveState.working_exact_mission_goal_outcomes ?? null,
      untried_visible_object_ids:
        stringArray(continuousDriveState.untried_visible_object_ids),
      untried_observable_solid_ids:
        stringArray(continuousDriveState.untried_observable_solid_ids),
      untried_zone_ids: stringArray(continuousDriveState.untried_zone_ids)
    },
    observable_goal_surface: {
      predicate_types: [...GOAL_HISTORY_PREDICATE_TYPES],
      zone_ids: stringArray(observation.zone_ids),
      visible_object_ids: stringArray(observation.visible_object_ids),
      portable_object_ids: objects.flatMap((object) => (
        object.portable === true && typeof object.id === "string" ? [object.id] : []
      )),
      object_affordances: objects.flatMap((object) => (
        typeof object.id === "string"
          ? [{
              object_id: object.id,
              affordances: stringArray(object.affordances),
              articulation: object.articulation ?? null
            }]
          : []
      )),
      removable_block_ids: solids.flatMap((solid) => (
        solid.kind === "block" && typeof solid.id === "string" ? [solid.id] : []
      )),
      carrying: {
        phase: carrying.phase ?? null,
        object_ids: carryBindings.flatMap((binding) => (
          typeof binding.object_id === "string" ? [binding.object_id] : []
        )),
        continuation_verified: carrying.continuation_verified ?? null
      },
      exploration: {
        visibility_model: spatialBelief.visibility_model ?? null,
        frontier_model: spatialBelief.frontier_model ?? null,
        coverage_ratio: spatialBelief.coverage_ratio ?? null,
        observed_cell_count: spatialBelief.observed_cell_count ?? null,
        free_cell_count: spatialBelief.free_cell_count ?? null,
        occupied_cell_count: spatialBelief.occupied_cell_count ?? null,
        reachable_free_cell_count: spatialBelief.reachable_free_cell_count ?? null,
        frontier_candidates: Array.isArray(spatialBelief.frontiers)
          ? spatialBelief.frontiers.slice(0, 12)
          : []
      },
      object_frontier: Array.isArray(autonomy.object_frontier)
        ? autonomy.object_frontier
        : [],
      solid_frontier: Array.isArray(autonomy.solid_frontier)
        ? autonomy.solid_frontier
        : [],
      zone_frontier: Array.isArray(autonomy.zone_frontier)
        ? autonomy.zone_frontier
        : []
    }
  };
  return [
    "请基于当前权威状态独立管理下一 Goal。",
    "CURRENT GOAL MANAGER INVOCATION",
    "以下状态与标识来自本次调用的权威物理状态。候选提交和选择会由 Harness 绑定本次证据；工具中仍需填写的标识必须逐字复制。",
    JSON.stringify(exact)
  ].join("\n\n");
}

export function motionInvocationInput(authority: JsonValue): string {
  const root = jsonRecord(authority);
  const goalDAG = jsonRecord(root.goal_dag);
  const collaborationResults = Array.isArray(root.recent_receipts)
    ? root.recent_receipts
        .map(jsonRecord)
        .filter((receipt) => (
          receipt.agent_id === HUMANOID_AGENT_IDS.motion
          && typeof receipt.action === "string"
          && MOTION_COLLABORATION_ACTIONS.has(receipt.action)
        ))
        .slice(-6)
    : [];
  return [
    "请基于本次 Sentry Grounding Snapshot，独立决定并提交推进当前 active Goal 的下一个可真实执行事务；不得沿用上级坐标或动作参数。",
    "CURRENT MOTION DELEGATION",
    JSON.stringify({
      run_mode: root.run_mode ?? null,
      current_goal_epoch_id: goalDAG.current_epoch_id ?? null,
      active_goal: root.active_goal ?? null,
      autonomy_readiness: root.autonomy_readiness ?? null,
      active_cycle: root.active_cycle ?? null,
      planning_tool_state: root.planning_tool_state ?? null,
      grounding_snapshot: root.grounding_snapshot ?? null,
      collaboration_results: collaborationResults
    })
  ].join("\n\n");
}

export function motionActorInvocationInput(
  plannerArtifact: string,
  authority: JsonValue
): string {
  const root = jsonRecord(authority);
  const goalDAG = jsonRecord(root.goal_dag);
  return [
    "将独立 Motion Planner 的本轮语义计划落实为一个正式 Harness 工具调用。",
    "MOTION PLANNER ARTIFACT",
    plannerArtifact,
    "END MOTION PLANNER ARTIFACT",
    "CURRENT MOTION ACTOR AUTHORITY",
    "计划 artifact 只提供语义选择，不提供权限。所有事务标识、binding、world revision 和当前可用动作必须以以下权威状态为准并逐字复制。",
    JSON.stringify({
      run_mode: root.run_mode ?? null,
      current_goal_epoch_id: goalDAG.current_epoch_id ?? null,
      active_goal: root.active_goal ?? null,
      autonomy_readiness: root.autonomy_readiness ?? null,
      active_cycle: root.active_cycle ?? null,
      planning_tool_state: root.planning_tool_state ?? null,
      grounding_snapshot: root.grounding_snapshot ?? null
    }),
    "END CURRENT MOTION ACTOR AUTHORITY"
  ].join("\n\n");
}

export function coordinatorInvocationInput(authority: JsonValue): string {
  return [
    "CURRENT COORDINATOR STEP",
    "This is the only authoritative state at the start of this run. After a tool call, use the latest coordinator_step_result.coordinator_state instead; older state is history.",
    JSON.stringify(coordinatorAuthorityProjection(authority)),
    "END CURRENT COORDINATOR STEP"
  ].join("\n");
}

export function coordinatorAuthorityProjection(authority: JsonValue): JsonValue {
  const root = jsonRecord(authority);
  const goalDAG = jsonRecord(root.goal_dag);
  const robot = jsonRecord(root.robot);
  return {
    run_mode: root.run_mode ?? null,
    mission_goal: root.mission_goal ?? null,
    goal_dag: {
      status: goalDAG.status ?? null,
      current_epoch_id: goalDAG.current_epoch_id ?? null
    },
    active_goal: root.active_goal ?? null,
    active_cycle: root.active_cycle ?? null,
    autonomy_readiness: root.autonomy_readiness ?? null,
    cycle_completion: root.cycle_completion ?? null,
    execution_authority: root.execution_authority ?? null,
    recovery_authority: root.recovery_authority ?? null,
    world_frame: root.world_frame ?? null,
    world_revision: root.world_revision ?? null,
    robot: {
      fallen: robot.fallen ?? null,
      balance: robot.balance ?? null,
      navigation: robot.navigation ?? null
    },
    recent_receipts: root.recent_receipts ?? []
  };
}

export const HUMANOID_AGENT_TOOL_CONTRACTS = {
  goalManager: {
    dispatchKind: "model_agent",
    toolName: "delegate_goal_manager",
    targetRole: "goal_manager",
    targetAgentId: HUMANOID_AGENT_IDS.goalManager,
    inputBuilderContract: "goal_manager_authority_envelope_v2",
    inputBuilder: () => goalManagerInvocationInput({}),
    runOptions: {
      sessionAgentId: HUMANOID_AGENT_IDS.goalManager,
      contextSource: "parent_run_context",
      maxTurns: "unbounded"
    },
    resumeContextStrategy: "merge",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "nested_agent_final_output_text"
  },
  sentry: {
    dispatchKind: "deterministic_service",
    toolName: "delegate_humanoid_sentry",
    targetRole: "sentry",
    targetAgentId: HUMANOID_AGENT_IDS.sentry,
    targetName: "异步物理 Grounding Monitor",
    inputBuilderContract: "grounding_monitor_direct_v1",
    implementationContract: "observe_humanoid_from_coordinator_tool_call_v1",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "formal_action_receipt"
  },
  motion: {
    dispatchKind: "model_pipeline",
    toolName: "delegate_motion_reference",
    targetRole: "motion",
    targetAgentId: HUMANOID_AGENT_IDS.motion,
    inputBuilderContract: "motion_planner_actor_pipeline_v1",
    inputBuilder: () => motionInvocationInput({}),
    pipeline: {
      plannerAgentId: HUMANOID_AGENT_IDS.motionPlanner,
      plannerSessionAgentId: HUMANOID_AGENT_IDS.motionPlanner,
      actorAgentId: HUMANOID_AGENT_IDS.motion,
      actorSessionAgentId: HUMANOID_AGENT_IDS.motion,
      artifactContract: "bounded_motion_plan_artifact_v1",
      authorityContract: "fresh_motion_authority_envelope_v1"
    },
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "formal_action_receipt"
  },
  executor: {
    dispatchKind: "deterministic_service",
    toolName: "delegate_physics_executor",
    targetRole: "executor",
    targetAgentId: HUMANOID_AGENT_IDS.executor,
    targetName: "确定性物理 Execution Gate",
    inputBuilderContract: "validated_execution_gate_v1",
    implementationContract: "accepted_plan_to_runtime_action_v1",
    includeInputSchema: false,
    needsApproval: false,
    outputContract: "formal_action_receipt"
  }
} as const;

type ModelAgentToolContract =
  typeof HUMANOID_AGENT_TOOL_CONTRACTS.goalManager;

export const HUMANOID_CAPABILITIES = [
  "recall_goal_history",
  "submit_goal_candidates",
  "select_goal_candidate",
  "retire_goal_epoch",
  "continue_goal_epoch",
  "observe_humanoid",
  "submit_humanoid_skill_plan",
  "begin_humanoid_skill",
  "recall_embodied_history",
  "plan_humanoid_skill",
  "execute_humanoid_skill",
  "plan_whole_body_motion",
  "plan_whole_body_motion_candidates",
  "execute_whole_body_motion",
  "plan_humanoid_navigation",
  "execute_humanoid_navigation",
  "remove_world_block"
] as const;

type HumanoidHierarchyRuntime = HumanoidActionInvoker
& HumanoidEmbodiedRecallInvoker & GoalManagerRuntime & {
  contextAnchor(agentId: string): JsonValue;
  validateCycleEvidence(evidenceTransactionIds: readonly string[]): HumanoidActionReceipt;
  cycleCompletionReadiness(): HumanoidCycleCompletionReadiness;
  autonomyReadiness(): HumanoidAutonomyReadiness;
  executorDelegationAvailable(): boolean;
  goalRetirementDelegationAvailable(): boolean;
  sentryDelegationAvailable?(): boolean;
  motionDelegationAvailable?(): boolean;
  validateSatisfiedGoal(): JsonValue;
};

export interface HumanoidAgentHierarchy {
  goalManager: Agent;
  coordinator: Agent;
  motionPlanner: Agent;
  motion: Agent;
  sentry: HumanoidDeterministicService;
  executor: HumanoidDeterministicService;
  goalManagerSession: Session;
  coordinatorSession: Session;
  session(agentId: string): Session | undefined;
}

interface HumanoidDeterministicService {
  id: string;
  name: string;
  kind: "deterministic_service";
  implementationContract: string;
}

/**
 * @deprecated Historical V2 five-role hierarchy. Production mission runs use
 * createHumanoidNeuralAgentHierarchy exclusively. Keep this factory only for
 * pre-V3 manifest/checkpoint migration fixtures; do not attach it to a Runner.
 */
export function createHumanoidAgentHierarchy(input: {
  createModel: (agentId: string, provider: ModelProviderConfig) => Model;
  createSession: (agentId: string) => Session;
  callModelInputFilter: CallModelInputFilter;
  provider: ProviderConfig;
  runtime: HumanoidHierarchyRuntime;
  onAgentStream?: (agentId: string, event: RunStreamEvent) => void | Promise<void>;
}): HumanoidAgentHierarchy {
  const models = new Set<Model>();
  const sessions = new Map<string, Session>();
  const sessionOwners = new Set<Session>();
  const ownModel = (agentId: string): Model => {
    const model = input.createModel(
      agentId,
      providerConfigForRole(input.provider, humanoidAgentRole(agentId))
    );
    if (models.has(model)) {
      throw new Error(`Humanoid hierarchy cannot share one Model facade: ${agentId}`);
    }
    models.add(model);
    return model;
  };
  const ownSession = (agentId: string): Session => {
    const session = input.createSession(agentId);
    if (sessionOwners.has(session)) {
      throw new Error(`Humanoid hierarchy cannot share one Session: ${agentId}`);
    }
    sessionOwners.add(session);
    sessions.set(agentId, session);
    return session;
  };
  const modelSettings = (
    agentId: string,
    overrides: { thinking?: "enabled" | "disabled"; toolChoice?: "auto" | "required" }
      = {}
  ): ModelSettings => {
    const provider = providerConfigForRole(input.provider, humanoidAgentRole(agentId));
    return {
      temperature: provider.temperature,
      ...(provider.reasoningEffort === undefined
        ? {}
        : { reasoning: { effort: provider.reasoningEffort } }),
      ...(provider.maxOutputTokens === undefined
        ? {}
        : { maxTokens: provider.maxOutputTokens }),
      ...(provider.protocol === "openai_compatible"
        && provider.model.toLowerCase().includes("deepseek")
        ? {
            // DeepSeek returns reasoning_content even when its gateway exposes no
            // explicit reasoning-effort switch. The top-level marker tells the
            // official Agents AI SDK adapter whether reasoning_content belongs in
            // replayed tool-call history. The provider option is the independent
            // transport control that the OpenAI-compatible adapter serializes into
            // the Chat Completions request. Keeping both is intentional: the Actor
            // must actually disable thinking before requiring a tool call.
            providerData: {
              thinking: { type: overrides.thinking ?? "enabled" },
              providerOptions: {
                "configured-openai-compatible": {
                  thinking: { type: overrides.thinking ?? "enabled" }
                }
              }
            }
          }
        : {}),
      parallelToolCalls: false,
      // The Harness targets the OpenAI-compatible Chat Completions capability
      // baseline. Some thinking models reject `required`, so provider capability
      // selection stays at the transport boundary instead of leaking a
      // Responses-only assumption into Agent orchestration.
      toolChoice: overrides.toolChoice ?? provider.toolChoice ?? "auto"
    };
  };

  const goalManager = new Agent({
    name: "自主目标管理智能体",
    instructions: scopedInstructions(
      HUMANOID_AGENT_IDS.goalManager,
      goalManagerInstructions()
    ),
    model: ownModel(HUMANOID_AGENT_IDS.goalManager),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.goalManager),
    tools: createGoalManagerTools(input.runtime),
    resetToolChoice: false,
    toolUseBehavior: verifiedStatusToolUseBehavior({
      select_goal_candidate: "goal_candidate_selected",
      retire_goal_epoch: "goal_epoch_retired",
      continue_goal_epoch: "goal_epoch_continued"
    })
  });
  const motionPlanner = new Agent({
    name: "全身运动规划智能体",
    instructions: scopedInstructions(
      HUMANOID_AGENT_IDS.motionPlanner,
      motionPlannerInstructions()
    ),
    model: ownModel(HUMANOID_AGENT_IDS.motionPlanner),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.motionPlanner),
    tools: [createHumanoidEmbodiedRecallTool(input.runtime)],
    resetToolChoice: false,
    toolUseBehavior: "run_llm_again"
  });
  const motion = new Agent({
    name: "全身运动动作智能体",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.motion, motionActorInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.motion),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.motion, {
      thinking: "disabled",
      toolChoice: "required"
    }),
    tools: [
      ...createHumanoidActionTools(
        input.runtime,
        HUMANOID_AGENT_IDS.motion,
        [
          "submit_humanoid_skill_plan",
          "begin_humanoid_skill",
          "plan_humanoid_skill",
          "plan_whole_body_motion_candidates",
          "plan_humanoid_navigation"
        ],
        { availability: "stable" }
      )
    ],
    resetToolChoice: false,
    toolUseBehavior: receiptToolUseBehavior([
      "submit_humanoid_skill_plan",
      "begin_humanoid_skill",
      "plan_humanoid_skill",
      "plan_whole_body_motion_candidates",
      "plan_humanoid_navigation"
    ])
  });
  const sentry: HumanoidDeterministicService = {
    id: HUMANOID_AGENT_IDS.sentry,
    name: HUMANOID_AGENT_TOOL_CONTRACTS.sentry.targetName,
    kind: "deterministic_service",
    implementationContract:
      HUMANOID_AGENT_TOOL_CONTRACTS.sentry.implementationContract
  };
  const executor: HumanoidDeterministicService = {
    id: HUMANOID_AGENT_IDS.executor,
    name: HUMANOID_AGENT_TOOL_CONTRACTS.executor.targetName,
    kind: "deterministic_service",
    implementationContract:
      HUMANOID_AGENT_TOOL_CONTRACTS.executor.implementationContract
  };
  const goalManagerSession = ownSession(HUMANOID_AGENT_IDS.goalManager);
  const motionPlannerSession = ownSession(HUMANOID_AGENT_IDS.motionPlanner);
  ownSession(HUMANOID_AGENT_IDS.motion);
  const coordinatorSession = ownSession(HUMANOID_AGENT_IDS.coordinator);
  const agentToolSession = (contract: ModelAgentToolContract): Session => {
    const session = sessions.get(contract.runOptions.sessionAgentId);
    if (!session) {
      throw new Error(
        `Agent-as-tool ${contract.toolName} requires Session owner `
        + contract.runOptions.sessionAgentId
      );
    }
    return session;
  };
  const goalManagerContract = HUMANOID_AGENT_TOOL_CONTRACTS.goalManager;
  const coordinator = new Agent({
    name: "人形自主协调智能体",
    instructions: scopedInstructions(HUMANOID_AGENT_IDS.coordinator, coordinatorInstructions()),
    model: ownModel(HUMANOID_AGENT_IDS.coordinator),
    modelSettings: modelSettings(HUMANOID_AGENT_IDS.coordinator, {
      thinking: "disabled",
      toolChoice: "required"
    }),
    tools: [
      createHumanoidEmbodiedRecallTool(input.runtime),
      recoverAgentToolInput(scopeAgentToolInvocation(
        HUMANOID_AGENT_IDS.goalManager,
        requireDelegatedDecision(
          HUMANOID_AGENT_IDS.goalManager,
          goalManager.asTool({
        toolName: goalManagerContract.toolName,
        toolDescription: "让独立目标管理智能体基于当前物理证据提交 2–3 个候选并显式选择下一 Goal，或证据化退役当前不可达 Goal。",
        customOutputExtractor: ({ finalOutput }) => serializeAgentToolOutput(finalOutput),
        parameters: SpecialistDelegationSchema,
        inputBuilder: () => goalManagerInvocationInput(
          input.runtime.contextAnchor(HUMANOID_AGENT_IDS.goalManager)
        ),
        includeInputSchema: goalManagerContract.includeInputSchema,
        needsApproval: goalManagerContract.needsApproval,
        runConfig: { callModelInputFilter: input.callModelInputFilter },
        runOptions: {
          session: agentToolSession(goalManagerContract),
          maxTurns: null,
          // OpenAI-compatible chat adapters commonly emit response-local
          // reasoning ids such as `reasoning-0`. They are not durable global
          // identities; persisting them lets the SDK deduplicator replace an
          // older reasoning item and breaks the append-only cache prefix.
          reasoningItemIdPolicy: "omit"
        },
        resumeState: { contextStrategy: goalManagerContract.resumeContextStrategy },
        ...(input.onAgentStream
          ? {
              onStream: ({ event }) => input.onAgentStream!(
                HUMANOID_AGENT_IDS.goalManager,
                event
              )
            }
          : {})
          }),
          isFormalGoalManagerResult
        )
      ), SpecialistDelegationSchema),
      groundingMonitorTool(input.runtime),
      recoverAgentToolInput(createMotionPipelineTool({
        runtime: input.runtime,
        planner: motionPlanner,
        actor: motion,
        plannerSession: motionPlannerSession,
        actorSession: sessions.get(HUMANOID_AGENT_IDS.motion)!,
        callModelInputFilter: input.callModelInputFilter,
        ...(input.onAgentStream ? { onAgentStream: input.onAgentStream } : {})
      }), SpecialistDelegationSchema),
      executionGateTool(input.runtime),
      cycleCompletionTool(input.runtime),
      satisfiedGoalCompletionTool(input.runtime)
    ],
    resetToolChoice: false,
    toolUseBehavior: coordinatorStepToolUseBehavior()
  });
  guardCoordinatorToolExecution(coordinator, input.runtime);

  return {
    goalManager,
    coordinator,
    sentry,
    motionPlanner,
    motion,
    executor,
    goalManagerSession,
    coordinatorSession,
    session: (agentId) => sessions.get(agentId)
  };
}

function groundingMonitorTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof SpecialistDelegationSchema, string> {
  const contract = HUMANOID_AGENT_TOOL_CONTRACTS.sentry;
  const groundingTool = tool<typeof SpecialistDelegationSchema, unknown, string>({
    name: contract.toolName,
    description: "从当前 MuJoCo、本体感觉、接触与头部传感器直接生成受限权威观察；该服务不调用模型，也不产生目标或动作。",
    parameters: SpecialistDelegationSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    execute: (params, _context, details) => invokeDeterministicHumanoidAction({
      runtime,
      actorAgentId: HUMANOID_AGENT_IDS.sentry,
      sourceToolName: contract.toolName,
      sourceInput: params,
      action: "observe_humanoid",
      actionInput: {},
      contractId: "grounding_monitor_v1",
      ...(details ? { details } : {})
    })
  });
  return recoverAgentToolInput(
    scopeAgentToolInvocation(HUMANOID_AGENT_IDS.sentry, groundingTool),
    SpecialistDelegationSchema
  );
}

function createMotionPipelineTool(input: {
  runtime: HumanoidHierarchyRuntime;
  planner: Agent;
  actor: Agent;
  plannerSession: Session;
  actorSession: Session;
  callModelInputFilter: CallModelInputFilter;
  onAgentStream?: (agentId: string, event: RunStreamEvent) => void | Promise<void>;
}): FunctionTool<unknown, typeof SpecialistDelegationSchema, string> {
  const contract = HUMANOID_AGENT_TOOL_CONTRACTS.motion;
  const runner = new Runner({
    tracingDisabled: true,
    traceIncludeSensitiveData: false,
    callModelInputFilter: input.callModelInputFilter,
    toolExecution: { maxFunctionToolConcurrency: 1 },
    toolNotFoundBehavior: "return_error_to_model",
    reasoningItemIdPolicy: "omit",
    modelSettings: { parallelToolCalls: false }
  });
  const drain = async <TStream extends {
    completed: Promise<void>;
    [Symbol.asyncIterator](): AsyncIterator<RunStreamEvent>;
  }>(
    agentId: string,
    stream: TStream
  ): Promise<void> => {
    for await (const event of stream) await input.onAgentStream?.(agentId, event);
    await stream.completed;
  };
  const runPlanner = async (
    authority: JsonValue,
    signal?: AbortSignal
  ): Promise<string> => withAgentInvocation(
    HUMANOID_AGENT_IDS.motionPlanner,
    async () => {
      if (!input.onAgentStream) {
        const result = await runner.run(
          input.planner,
          motionInvocationInput(authority),
          {
            session: input.plannerSession,
            maxTurns: null,
            reasoningItemIdPolicy: "omit",
            toolExecution: { maxFunctionToolConcurrency: 1 },
            ...(signal ? { signal } : {})
          }
        );
        return boundedMotionPlannerArtifact(result.finalOutput);
      }
      const stream = await runner.run(
        input.planner,
        motionInvocationInput(authority),
        {
          stream: true,
          session: input.plannerSession,
          maxTurns: null,
          reasoningItemIdPolicy: "omit",
          toolExecution: { maxFunctionToolConcurrency: 1 },
          ...(signal ? { signal } : {})
        }
      );
      await drain(HUMANOID_AGENT_IDS.motionPlanner, stream);
      return boundedMotionPlannerArtifact(stream.finalOutput);
    }
  );
  const runActor = async (
    plannerArtifact: string,
    authority: JsonValue,
    signal?: AbortSignal
  ): Promise<string> => withAgentInvocation(
    HUMANOID_AGENT_IDS.motion,
    async () => {
      if (!input.onAgentStream) {
        const result = await runner.run(
          input.actor,
          motionActorInvocationInput(plannerArtifact, authority),
          {
            session: input.actorSession,
            maxTurns: null,
            reasoningItemIdPolicy: "omit",
            toolExecution: { maxFunctionToolConcurrency: 1 },
            ...(signal ? { signal } : {})
          }
        );
        return serializeAgentToolOutput(result.finalOutput);
      }
      const stream = await runner.run(
        input.actor,
        motionActorInvocationInput(plannerArtifact, authority),
        {
          stream: true,
          session: input.actorSession,
          maxTurns: null,
          reasoningItemIdPolicy: "omit",
          toolExecution: { maxFunctionToolConcurrency: 1 },
          ...(signal ? { signal } : {})
        }
      );
      await drain(HUMANOID_AGENT_IDS.motion, stream);
      return serializeAgentToolOutput(stream.finalOutput);
    }
  );
  return scopeAgentToolInvocation(HUMANOID_AGENT_IDS.motion, tool({
    name: contract.toolName,
    description: "运行独立 Motion Planner → Motion Actor 管线：Planner 保留语义推理，Actor 只把本轮有界计划和最新权威状态落实为一个正式 Harness 动作工具。",
    parameters: SpecialistDelegationSchema,
    strict: true,
    needsApproval: contract.needsApproval,
    timeoutBehavior: "raise_exception",
    errorFunction: null,
    execute: async (_params, _context, details) => {
      details?.signal?.throwIfAborted();
      const plannerAuthority = input.runtime.contextAnchor(
        HUMANOID_AGENT_IDS.motionPlanner
      );
      const plannerArtifact = await runPlanner(plannerAuthority, details?.signal);
      const actorAuthority = input.runtime.contextAnchor(HUMANOID_AGENT_IDS.motion);
      const output = await runActor(plannerArtifact, actorAuthority, details?.signal);
      if (isFormalActionReceipt(output, HUMANOID_AGENT_IDS.motion, [
        "submit_humanoid_skill_plan",
        "begin_humanoid_skill",
        "plan_humanoid_skill",
        "plan_whole_body_motion_candidates",
        "plan_humanoid_navigation"
      ])) return output;
      throw new ModelDecisionStallError(
        HUMANOID_AGENT_IDS.motion,
        `${HUMANOID_AGENT_IDS.motion} did not return its required terminal tool result`
      );
    }
  }));
}

function boundedMotionPlannerArtifact(output: unknown): string {
  const text = serializeAgentToolOutput(output).trim();
  if (text.length === 0) return "Planner returned no prose plan; derive the one legal action from current authority without inventing identifiers.";
  if (text.length <= MOTION_PLAN_ARTIFACT_MAX_CHARACTERS) return text;
  return text.slice(0, MOTION_PLAN_ARTIFACT_MAX_CHARACTERS);
}

function executionGateTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof ExecutionTaskSchema, string> {
  const contract = HUMANOID_AGENT_TOOL_CONTRACTS.executor;
  const executionTool = tool<typeof ExecutionTaskSchema, unknown, string>({
    name: contract.toolName,
    description: "确定性消费当前 Harness 已接受的规划授权并映射到唯一正式物理动作，或提交已有稳定接触证明的方块拆除事务；不会调用模型、重规划或改写参数。",
    parameters: ExecutionTaskSchema,
    strict: true,
    timeoutBehavior: "raise_exception",
    execute: (params, _context, details) => {
      const selected = executionGateAction(params.execution);
      return invokeDeterministicHumanoidAction({
        runtime,
        actorAgentId: HUMANOID_AGENT_IDS.executor,
        sourceToolName: contract.toolName,
        sourceInput: params,
        action: selected.action,
        actionInput: selected.input,
        contractId: "execution_gate_v1",
        ...(details ? { details } : {})
      });
    }
  });
  return recoverAgentToolInput(
    scopeAgentToolInvocation(HUMANOID_AGENT_IDS.executor, executionTool),
    ExecutionTaskSchema
  );
}

function executionGateAction(
  execution: z.infer<typeof ExecutionTaskSchema>["execution"]
): {
  action: "execute_humanoid_skill"
    | "execute_whole_body_motion"
    | "execute_humanoid_navigation"
    | "remove_world_block";
  input: Record<string, string>;
} {
  if (execution.kind === "remove_world_block") {
    return {
      action: "remove_world_block",
      input: {
        solid_id: execution.solid_id,
        execution_transaction_id: execution.execution_transaction_id
      }
    };
  }
  const action = {
    plan_humanoid_skill: "execute_humanoid_skill",
    plan_whole_body_motion: "execute_whole_body_motion",
    plan_whole_body_motion_candidates: "execute_whole_body_motion",
    plan_humanoid_navigation: "execute_humanoid_navigation"
  } as const;
  return {
    action: action[execution.planning_action],
    input: { planning_transaction_id: execution.planning_transaction_id }
  };
}

function guardCoordinatorToolExecution(
  coordinator: Agent,
  runtime: HumanoidHierarchyRuntime
): void {
  for (const coordinatorTool of coordinator.tools) {
    if (coordinatorTool.type !== "function") continue;
    const name = coordinatorTool.name;
    const invoke = coordinatorTool.invoke;
    coordinatorTool.invoke = async (context, toolInput, details) => {
      if (coordinatorToolAvailable(name, runtime)) {
        const output = await invoke(context, toolInput, details);
        if (name === "complete_autonomous_cycle"
          || name === "complete_satisfied_goal") {
          return output;
        }
        return coordinatorStepResult(name, output, runtime);
      }
      return JSON.stringify({
        status: "coordinator_step_result",
        tool: name,
        result: {
          accepted: false,
          code: "autonomy_readiness_rejected",
          automatic_actuation: false
        },
        coordinator_state: coordinatorAuthorityProjection(
          runtime.contextAnchor(HUMANOID_AGENT_IDS.coordinator)
        )
      });
    };
  }
}

function coordinatorStepResult(
  toolName: string,
  output: unknown,
  runtime: HumanoidHierarchyRuntime
): string {
  let result = output;
  if (typeof output === "string") {
    try {
      result = JSON.parse(output) as JsonValue;
    } catch {
      // Preserve a non-JSON recall result as text; the authority projection is
      // still the only source for the next physical phase.
    }
  }
  return JSON.stringify({
    status: "coordinator_step_result",
    tool: toolName,
    result: coordinatorDelegationResult(toolName, result),
    coordinator_state: coordinatorAuthorityProjection(
      runtime.contextAnchor(HUMANOID_AGENT_IDS.coordinator)
    )
  });
}

/**
 * Manager-facing result of a delegation. Specialist observations, geometry,
 * and planner diagnostics remain owned by the specialist Session or the
 * deterministic Runtime; the Manager receives only the control-plane result
 * and obtains current authority from coordinator_state.
 */
function coordinatorDelegationResult(
  toolName: string,
  output: unknown
): JsonValue {
  const value = output !== null && typeof output === "object" && !Array.isArray(output)
    ? output as Record<string, unknown>
    : undefined;
  if (!value) return {
    kind: "delegation_result",
    tool: toolName,
    accepted: false,
    code: "invalid_delegation_result"
  };
  if (toolName === "recall_embodied_history") {
    return JSON.parse(JSON.stringify({
      kind: "recall_result",
      tool: toolName,
      ...value
    })) as JsonValue;
  }
  if (typeof value.transactionId === "string"
    && typeof value.action === "string"
    && typeof value.accepted === "boolean") {
    return JSON.parse(JSON.stringify({
      kind: "delegation_result",
      tool: toolName,
      owner_agent_id: value.agentId ?? null,
      transaction_id: value.transactionId,
      action: value.action,
      accepted: value.accepted,
      code: value.code ?? null,
      world_before_revision: value.worldBeforeRevision ?? null,
      world_after_revision: value.worldAfterRevision ?? null,
      frame_count: value.frameCount ?? null,
      channels: value.channels ?? []
    })) as JsonValue;
  }
  if (typeof value.accepted === "boolean" || typeof value.code === "string") {
    return JSON.parse(JSON.stringify({
      kind: "delegation_result",
      tool: toolName,
      accepted: value.accepted ?? false,
      code: value.code ?? null,
      status: value.status ?? null,
      ...(value.validation_issues === undefined
        ? {}
        : { validation_issues: value.validation_issues }),
      automatic_actuation: value.automatic_actuation ?? false
    })) as JsonValue;
  }
  return JSON.parse(JSON.stringify({
    kind: "delegation_result",
    tool: toolName,
    status: value.status ?? null,
    epoch_id: value.epoch_id ?? null,
    cycle_id: value.cycle_id ?? null,
    candidate_id: value.candidate_id ?? null,
    retirement_status: value.retirement_status ?? null,
    world_revision: value.world_revision ?? value.resolved_world_revision ?? null
  })) as JsonValue;
}

function coordinatorToolAvailable(
  name: string,
  runtime: HumanoidHierarchyRuntime
): boolean {
  const completion = runtime.cycleCompletionReadiness();
  const phase = runtime.autonomyReadiness();
  if (name === "complete_autonomous_cycle") {
    return completion.status === "ready"
      && completion.observed_after_execution
      && phase === "complete_cycle";
  }
  if (name === "complete_satisfied_goal") return phase === "complete_satisfied_goal";
  if (name === "delegate_goal_manager") {
    return phase === "goal_selection"
      || runtime.goalRetirementDelegationAvailable();
  }
  if (name === "delegate_humanoid_sentry") {
    const phaseAllowsObservation = phase === "observe_or_plan"
      || phase === "plan"
      || phase === "post_failure_observation"
      || phase === "replan_or_retire"
      || phase === "post_execution";
    return phaseAllowsObservation
      && (runtime.sentryDelegationAvailable?.() ?? true);
  }
  if (name === "delegate_motion_reference") {
    return (phase === "observe_or_plan"
      || phase === "plan"
      || phase === "replan_or_retire")
      && (runtime.motionDelegationAvailable?.() ?? true);
  }
  if (name === "delegate_physics_executor") {
    return runtime.executorDelegationAvailable();
  }
  return phase === "observe_or_plan" || phase === "plan";
}

export function humanoidAgentRole(agentId: string): Exclude<AgentModelRole, "compactor"> {
  switch (agentId) {
    case HUMANOID_AGENT_IDS.goalManager:
      return "goal_manager";
    case HUMANOID_AGENT_IDS.coordinator:
      return "coordinator";
    case HUMANOID_AGENT_IDS.sentry:
      return "sentry";
    case HUMANOID_AGENT_IDS.motionPlanner:
      return "motion_planner";
    case HUMANOID_AGENT_IDS.motion:
      return "motion";
    case HUMANOID_AGENT_IDS.executor:
      return "executor";
    default:
      throw new Error(`Unknown humanoid agent identity: ${agentId}`);
  }
}

function scopedInstructions(agentId: string, instructions: string): string {
  return `${agentInvocationMarker(agentId)}\n${instructions}`;
}

function jsonRecord(value: unknown): Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, JsonValue>
    : {};
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function cycleCompletionTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof CycleCompletionSchema, string> {
  return tool<typeof CycleCompletionSchema, unknown, string>({
    name: "complete_autonomous_cycle",
    description: "完成一次真实自主循环。证据必须包含最近一次未被后续物理执行取代的成功回执；若该执行产生世界修改，还必须同时引用对应 mutation 回执。",
    parameters: CycleCompletionSchema,
    strict: true,
    execute: (input) => {
      const execution = runtime.validateCycleEvidence(input.evidence_transaction_ids);
      return JSON.stringify({
        status: "cycle_completed",
        summary: input.summary,
        evidence_transaction_ids: input.evidence_transaction_ids,
        world_revision: execution.worldAfterRevision,
        executed_action: execution.action,
        ...(input.next_intent ? { next_intent: input.next_intent } : {})
      });
    }
  });
}

function satisfiedGoalCompletionTool(
  runtime: HumanoidHierarchyRuntime
): FunctionTool<unknown, typeof SatisfiedGoalCompletionSchema, string> {
  return tool<typeof SatisfiedGoalCompletionSchema, unknown, string>({
    name: "complete_satisfied_goal",
    description: "当前 active Goal 已由实时物理状态直接满足且本周期无需重复执行时，提交 Harness 复验并完成该 Goal。",
    parameters: SatisfiedGoalCompletionSchema,
    strict: true,
    execute: (input) => JSON.stringify({
      status: "satisfied_goal_completed",
      summary: input.summary,
      verification: runtime.validateSatisfiedGoal()
    })
  });
}

function receiptToolUseBehavior(
  terminalActions: readonly string[]
): ToolUseBehavior {
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const output = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output);
      try {
        const receipt = JSON.parse(output) as Partial<HumanoidActionReceipt>;
        if (receipt.transactionId
          && receipt.action === result.tool.name
          && typeof receipt.accepted === "boolean"
          && (!receipt.accepted
            || terminalActions.includes(result.tool.name))) {
          return {
            isFinalOutput: true,
            isInterrupted: undefined,
            finalOutput: output
          };
        }
      } catch {
        // The SDK feeds malformed model arguments back through errorFunction.
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function verifiedStatusToolUseBehavior(
  statusByTool: Readonly<Record<string, string>>
): ToolUseBehavior {
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const expectedStatus = statusByTool[result.tool.name];
      if (!expectedStatus) continue;
      const output = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output);
      try {
        const parsed = JSON.parse(output) as { status?: unknown };
        if (parsed.status === expectedStatus) {
          return {
            isFinalOutput: true,
            isInterrupted: undefined,
            finalOutput: output
          };
        }
      } catch {
        // Rejected or malformed terminal output is fed back to the deciding model.
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function coordinatorStepToolUseBehavior(): ToolUseBehavior {
  const terminalStatusByTool: Readonly<Record<string, string>> = {
    complete_autonomous_cycle: "cycle_completed",
    complete_satisfied_goal: "satisfied_goal_completed"
  };
  return (_context, results) => {
    for (const result of results) {
      if (result.type !== "function_output") continue;
      const output = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output);
      const parsed = delegatedOutputObject(output);
      if (!parsed) continue;
      const terminalStatus = terminalStatusByTool[result.tool.name];
      const terminal = terminalStatus !== undefined
        && parsed.status === terminalStatus;
      if (terminal) {
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: output
        };
      }
    }
    return { isFinalOutput: false, isInterrupted: undefined };
  };
}

function requireDelegatedDecision<
  TContext,
  TParameters extends ToolInputParameters,
  TResult,
  TTool extends FunctionTool<TContext, TParameters, TResult>
>(
  agentId: string,
  agentTool: TTool,
  isFormalDecision: (output: unknown) => boolean
): TTool {
  const invoke = agentTool.invoke;
  agentTool.invoke = async (context, input, details) => {
    details?.signal?.throwIfAborted();
    const output = await invoke(context, input, details);
    if (isFormalDecision(output)) return output;
    throw new ModelDecisionStallError(
      agentId,
      `${agentId} did not return its required terminal tool result`
    );
  };
  return agentTool;
}

function isFormalGoalManagerResult(output: unknown): boolean {
  const result = delegatedOutputObject(output);
  return result?.status === "goal_candidate_selected"
    || result?.status === "goal_epoch_retired"
    || result?.status === "goal_epoch_continued";
}

function isFormalActionReceipt(
  output: unknown,
  agentId: string,
  allowedActions: readonly string[]
): boolean {
  const receipt = delegatedOutputObject(output);
  return typeof receipt?.transactionId === "string"
    && receipt.transactionId.length > 0
    && receipt.agentId === agentId
    && typeof receipt.action === "string"
    && allowedActions.includes(receipt.action)
    && typeof receipt.accepted === "boolean";
}

function delegatedOutputObject(output: unknown): Record<string, unknown> | undefined {
  let parsed = output;
  if (typeof output === "string") {
    try {
      parsed = JSON.parse(output);
    } catch {
      return undefined;
    }
  }
  return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
    ? parsed as Record<string, unknown>
    : undefined;
}

function serializeAgentToolOutput(output: unknown): string {
  if (typeof output === "string") return output;
  return output === undefined ? "" : JSON.stringify(output);
}

function recoverAgentToolInput<
  TContext,
  TParameters extends ToolInputParameters,
  TResult,
  TTool extends FunctionTool<TContext, TParameters, TResult>
>(agentTool: TTool, schema: z.ZodType): TTool {
  const inputRecovery = createToolInputRecovery();
  const invoke = agentTool.invoke;
  agentTool.invoke = async (context, input, details) => {
    const rejection = inputRecovery.preflight(input, schema, agentTool.name);
    if (rejection !== undefined) return rejection;
    return invoke(context, input, details);
  };
  return agentTool;
}

function coordinatorInstructions(): string {
  return [
    "你是人形机器人的 Manager Agent。你始终拥有任务；Goal Manager 和 Motion 仅作为有界 agent-as-tool 专家，Grounding Monitor 与 Execution Gate 是确定性服务。",
    "每次模型响应必须且只能调用一个正式工具。不要在工具调用之前或之后输出说明、摘要或普通聊天；你不能直接改变物理世界。",
    "运行开始时读取 CURRENT COORDINATOR STEP。每次工具返回后，只读取最新 coordinator_step_result.coordinator_state 作为当前权威状态；更早的状态、回执和召回内容均为历史。",
    "严格按 autonomy_readiness 选择下一层级动作：goal_selection 或 goal_transition 调用 delegate_goal_manager；observe_or_plan 或 post_failure_observation 调用 delegate_humanoid_sentry；plan 调用 delegate_motion_reference；execute_plan 调用 delegate_physics_executor；post_execution 调用 delegate_humanoid_sentry；complete_cycle 调用 complete_autonomous_cycle；complete_satisfied_goal 调用 complete_satisfied_goal。",
    "replan_or_retire 时依据最新失败观察和 recovery_authority 自主选择 delegate_motion_reference 重新规划，或在物理证据证明 Goal 本身不可继续时调用 delegate_goal_manager。规划失败本身只否定策略，不自动证明 Goal blocked。",
    "delegate_goal_manager、delegate_humanoid_sentry 和 delegate_motion_reference 的参数必须是 {}。不得替专家预选 Goal、坐标、手、Link、接触、Skill、路线或动作参数。",
    "execute_plan 时只使用最新 execution_authority，逐字复制 planning_action 与 planning_transaction_id；不得使用历史 transactionId、内部 plan_id 或被拒绝回执。",
    "物理执行后必须重新感知。只有最新 autonomy state 同时给出 autonomy_readiness=complete_cycle、cycle_completion.status=ready 和 observed_after_execution=true 时，才逐字提交其中的 evidence_transaction_ids。",
    "召回结果只用于避免重复失败，不能代替当前传感或授权。工具被阶段拒绝时，读取同一回执中的最新 coordinator_state 并改用该阶段合法工具。",
    "不得使用固定动作表、预设路径、程序随机列表或假执行；运动意图必须由 Motion 基于实时几何生成，低层控制器负责逐帧运动。"
  ].join("\n");
}

function goalManagerInstructions(): string {
  return [
    "你是人形层级智能体的自主目标管理节点，拥有独立模型、独立 Session 和长期 Goal DAG。",
    "mission 与 mission_goal 是长期约束，不是当前 active Goal。你必须自行决定下一阶段如何推进它，并在每个候选的 mission_link 中说明关联；Harness 不会生成、补全、排序或替换候选。",
    "run_mode=mission 时，任务只有在一个经过物理验收且 predicates 与 mission_goal 完全一致的 Goal 完成后才会结束；summary 只是说明文本，不改变物理语义。一个 active Goal 可以跨越多次观察、规划、抓取、导航和执行周期，不需要把每个动作阶段都改写成新 Goal。只有当前物理证据证明最终 Goal 尚不可观察或存在必须先独立验收的因果前置条件时，才选择阶段 Goal；条件成熟后必须提交并选择完整 mission predicates。run_mode=continuous 时，完成当前 Goal 后继续基于新观察选择下一 Goal。",
    "阶段 Goal 必须有当前证据支持且确实有助于 mission_goal；不要把普通障碍物想象成必经阻塞。导航能够绕行时优先选择直接可验收的移动或 mission_goal，只有真实规划/接触证据表明具名静态方块必须被处理时，才选择 block_removed。",
    "Goal 中每个 predicate 都是必须真实完成的合取义务，不是说明或确认字段。不得为了显得完整而加入 mission_goal 未要求的物体、接触、抓取、方块或区域谓词；若候选包含 mission_goal 的任一谓词，该候选必须逐字段保持完整 mission_goal，不得改 tolerance、删减谓词或拼接额外条件。",
    "run_mode=mission 且 mission_goal 的谓词已有对应能力、当前没有规划拒绝证明其受阻时，候选中必须包含完整 mission_goal，并优先选择它；阶段 Goal 只用于有当前物理证据的必要前置条件，不能无端扩大任务范围。",
    "区域是具有水平范围与支撑面的语义对象：要求物体进入或稳放到区域时使用 object_in_zone 或 object_placed 并逐字复制 zone_id，不能把 zone.center 当作物体中心的 object_at target。zone.center.y 描述区域平面，不是物体落稳后的中心高度。一个 Goal 内不得重复相同 predicate。",
    "robot_at 阶段 Goal 的 target 必须是机器人根可实际占据的自由站位，不能直接复制动态物体、承托台或障碍物中心。已有导航拒绝回执给出 projected target、partial endpoint 或 chunk target 时，应把这些物理可行性证据与阶段目的结合，而不是再次提交同一个不可占据点。",
    "导航的 target.y 不参与平面可达性，投影拒绝中的 distance 衡量 XZ 平面偏差；把 y 改成地面高度不能修复同一 XZ 站位。相邻 robot_at 站位已被物理规划连续拒绝、但当前观察显示目标物体仍可见且双腕距离已进入全身操作可达范围时，不得继续提交物体中心或同一障碍边界附近的 robot_at 变体；应选择 object_grasped 或完整 mission_goal，让 Motion 用实时末端几何继续操作。",
    "goal_dag.status=awaiting_model_selection 时，先检查 CURRENT GOAL MANAGER INVOCATION 的 existing_goal_candidates。若仍有 status=proposed、dependency_candidates 均为 completed、且符合当前证据与长期任务的候选，可直接调用 select_goal_candidate；若没有合适候选，才调用 submit_goal_candidates 一次提交 2–3 个内容不同的新候选。不得重复提交已有 Goal 内容来代替选择。Harness 会把提交与随后选择分别绑定到对应模型请求所见的当前物理证据，不要求你转录证据哈希。候选间不能引用本批 proposal_id 或尚未生成的 candidate_id；dependency_candidate_ids 只能逐字引用 existing_goal_candidate_ids，列表为空时每个候选都必须填写 []。每个谓词必须能由当前证据和后续 checker 真实观察。goal_context.observation 提供当前可见物体、视觉或接触可观察静态方块的真实位姿、关系、接触和区域；block_removed 只能逐字引用当前 solids 中 kind=block 的 id。candidate_history.lifetime_outcomes 汇总整个运行期按谓词、对象、区域、方块和末端划分的 selected、not_selected 与真实终态，只提供历史，不包含 Harness 评分或候选；records_without_alternate_history 大于 0 表示相应旧记录没有保存未采用候选，不得把未知解释为零次。",
    "候选必须根据当前可供性和长期约束产生，并主动比较近期 Goal 的对象、区域、谓词组合和结果。context_projection.total_epoch_count 大于 visible_epoch_count 时，选择或提交任何包含 robot_at、object_at 或 world-frame end_effector_at 的候选之前，必须先调用 recall_goal_history.world_region 查询你准备选择的世界坐标；水平半径不得小于对应谓词 tolerance。其他 history_truncated=true 或需要核对更早结果的情况也应调用 recall_goal_history 检索完整持久 Goal DAG。召回只筛选历史，不能代替当前观察，也不能替你选择候选；除非恢复、依赖或当前物理状态确有必要，不要重复已完成的相同空间目标。自主差异来自你的模型选择，不得用随机坐标、随机电机动作或固定目标表冒充新颖性。",
    "直接选择既有候选时，逐字复制 CURRENT GOAL MANAGER INVOCATION 中的 candidate_sequence；提交成功后必须在新的模型响应中从提交回执里选择 candidate_sequence 并调用 select_goal_candidate。同一次 submit_goal_candidates 调用产生一个互斥决策批次：模型选中一个候选后，该批其余候选会以 expired 结果随所选 epoch 持久归档；以后若新物理证据使其中一种方向重新合适，应在新的模型调用中重新提出，而不能复用旧批次。该短序号与持久哈希身份一对一对应，不能猜测。必须显式选择一个依赖已完成的候选；不能让程序随机选择，也不能让 Harness 替你插入固定候选。",
    "涉及便携物体时必须按当前物理依赖判断 Goal：未持握物体就不能把远离该物体、仅进入目标区域当成放置任务的充分前置；observable_goal_surface.carrying 只有在列出该物体且 continuation_verified=true 时才证明它可随导航继续携带。可见且可操作的目标物体、其双腕距离、当前抓取评估与目标区域共同决定是直接选择完整 mission_goal，还是先选择 object_grasped 等可验收前置 Goal；不得用固定抓取顺序替代这一判断。",
    "goal_dag.status=active 时不得提交或选择新目标。只有当前观察与物理 action receipt 证明 Goal 谓词本身不可继续时，才能调用 retire_goal_epoch；若当前物理状态已满足完整 mission_goal、但 active Goal 是阶段目标，应引用 current_goal_evidence_ref 将阶段目标退役为 superseded，使下一轮能选择完整 mission_goal 验收。blocked 必须引用 CURRENT GOAL MANAGER INVOCATION.recent_action_evidence 中真实存在的 action evidence_ref。零物理帧的规划拒绝、plan_revalidation_failed、repeated_planning_failure 只否定一次策略，不能证明 Goal blocked；Goal 重评时若 Goal 仍可达，必须调用 continue_goal_epoch 保留它。Goal Manager 只能说明 Goal 为何仍应继续，不得为下一周期指定 Skill、手、交互点、坐标、路线或动作参数；下一周期必须从新鲜物理观察重新选择。Harness 会结束当前失败周期、刷新 compact replan 预算，但不会替 Motion 选择 Skill 或参数。manipulation_base_placement_required 且 detail.reachable_base_placements 非空只证明需要先移动基座。退役不会自动选择替代目标。",
    "active Goal 可以跨越导航、重新观察、接触和操作多个周期。候选若在 mission_link 中声明依赖另一个阶段，该依赖必须已经是 completed candidate 并写入 dependency_candidate_ids；不能把尚未完成的前置阶段和后继阶段放进同一批后直接选择后继。",
    "retire_goal_epoch 的 evidence_refs 必须逐字复制 current_goal_evidence_ref 或 recent_action_evidence.evidence_ref，绝不能填写裸 transaction_id；一次退役引用的全部证据必须属于同一 world revision。select_goal_candidate 的 candidate_sequence 与所有 dependency_candidate_ids 也必须逐字使用上下文或工具回执中真实存在的标识，不得猜测。",
    "最终结果必须来自 select_goal_candidate、retire_goal_epoch 或 continue_goal_epoch 工具。"
  ].join("\n");
}

function motionPlannerInstructions(): string {
  return [
    "你是全身 Motion Planner Agent，拥有独立模型、独立 Session，并只决定本轮语义目标与策略。你不拥有物理动作工具，也不能直接改变 Harness 或世界。",
    "CURRENT MOTION DELEGATION.grounding_snapshot 是 Sentry 在本次 coordinator phase 捕获并由 Harness 绑定给你的唯一当前物理事实。你没有感知工具；快照缺失或 world_revision 不匹配时不得规划。",
    "根据 active Goal、实时空间信念、对象世界模型、Affordance、关节状态、掌指几何、平衡和近期真实失败，自主选择当前局部阶段。不得使用固定巡逻点、预设动作序列、随机电机噪声或猜测坐标。",
    "每个 Skill 必须对 active Goal 具有可验证因果关系：空间目标应推进对应位置或区域谓词，操作 Skill 应匹配 Goal 中的对象、方块或关节，准备阶段应建立同一实体的真实前置条件。与 Goal 无关或令目标距离增加的另一 frontier 会被 Harness 拒绝；真实物理失败授权的安全恢复除外。",
    "观察中的 control_authority 区分 MuJoCo 物理后端、已加载学习策略的真实能力和任务空间生成器。active_control 是当前物理帧实际执行的控制来源：learned_policy 为学习控制，reference_control 为参考控制，hybrid_control 为学习式下肢运动与参考式上肢跟踪的同帧组合；transition 表示尚未结束的连续交接。只有 learned_policy.capabilities 明确列出的能力才能称为已经由策略学习；未列出的接触操作能力不能靠叙述冒充已训练，是否可执行仍以当前控制后端的完整 MuJoCo 预演为准。",
    "观察回执的 interaction.available_skills 中 learned_policy_ready 与 learned_policy_missing_capabilities 表示训练策略是否完整覆盖该 Skill；缺少训练能力的 Skill 不可执行，不能用参考控制冒充已训练策略。",
    "观察后若当前没有仍与 world_revision 一致的 Skill 计划，明确提出 submit_humanoid_skill_plan 所需的短程 Skill DAG；已有有效计划时继续其中依赖已满足的节点。你必须自己选择策略、Skill、目标对象或 frontier、交互点、手、操作方向及依赖。",
    "若 planning_tool_state.ready_skill_bindings 非空，从中自主选择一项，并在计划中明确要求 Motion Actor 将 skill_plan_transaction_id、skill_node_id、invocation、phase 原样复制到 begin_humanoid_skill。若有多个 ready binding，选择权属于你而非 Harness。",
    "长程语义 Skill 可能由多个真实物理 chunk 完成。物理 chunk 成功不等于 Skill phase 完成；若下一次委派的 planning_tool_state 保留 skill_plan 且给出 ready_skill_bindings，从中选择并原样调用 begin_humanoid_skill，继续同一模型提交的 DAG，禁止重复提交整个 DAG。",
    "planning_tool_state.active_skill 存在时，只计划其 planning_action，并要求 Actor 逐字复制该绑定要求的事务标识。",
    "plan_humanoid_skill 会从实时几何生成站位、任务空间轨迹和终止契约，再通过 Recast、IK 与 MuJoCo 完整预演；不能提交关节角或低层路线绕过它。planning_tool_state.transit_clearance.status=required 时，可调用 plan_humanoid_navigation 或 plan_whole_body_motion_candidates，并逐字传递其中 skill_transaction_id；候选仍必须由你选择且通过新的 MuJoCo 预演。",
    "explore 必须选择当前 spatial_belief.frontiers 中真实存在的 frontier；自主差异来自模型对信息增益、路程、覆盖率、近期经历和长期 Goal 的判断，不得由程序随机替代。",
    "操作物体时保持模型选定的对象、手、交互点和策略。向语义区域搬运已持握物体时使用 carry_to_zone 并逐字引用 zone_id；确定性导航层会根据当前物体相对根节点的真实偏移计算机器人终点，禁止把 zone.center 直接当作机器人根目标。对 object_in_zone 或 object_placed Goal，place.destination 使用 semantic_zone 并逐字引用 zone_id，让确定性求解器从区域支撑面和物体尺寸计算落点；不得把 zone.center 猜成 world_pose。approach、reach、grasp、lift、carry、carry_to_zone、place、push、pull、press、open、close、turn、regrasp 与双手 Skill 的低层几何由求解器从当前状态计算，物理拒绝不会被伪装为成功。",
    "break_block 只能选择当前 solid_tokens 中 kind=block 的实体，并由你选择手、strike 或 press 策略；必须先完成可达接近，再以真实稳定掌指接触取得拆除权限。固定物体不能拆除。",
    "planning_tool_state.recovery_policy 来自真实失败分类；从其中允许的恢复 Skill 中作出新的模型选择。不得重复完全相同的失败方案，也不得让 Harness 改换语义目标、对象、手或策略。",
    "CURRENT MOTION DELEGATION.collaboration_results 是 Motion Actor 最近正式工具结果的有界投影，不是 Actor 会话历史。若最新结果被拒绝，必须读取 code、failure_class、attempts、skill_binding 和可用性状态，改变造成失败的语义选择（如交互面、手、standoff、Skill 或策略），不得重新提交同一 invocation；若结果与 planning_tool_state 冲突，以当前 planning_tool_state 为准。",
    "recall_embodied_history 只用于比较过去策略结果。召回内容始终是 historical_only；任何新动作都必须重新观察并绑定当前 world_revision。",
    "最终输出一个自包含、有界的 Motion Plan Artifact：说明应调用的唯一正式工具、语义选择、需逐字复制的权威标识以及理由。不要声称工具已经执行，不要输出 Harness 回执。"
  ].join("\n");
}

function motionActorInstructions(): string {
  return [
    "你是 Motion Actor Agent，拥有独立模型和独立 Session。你不接收 Planner 的会话历史，只接收本轮有界 Motion Plan Artifact 与最新 CURRENT MOTION ACTOR AUTHORITY。",
    "每次响应必须且只能调用一个正式工具，不得输出普通聊天。你不重新规划、不改变 Planner 的语义选择、不调用历史召回。",
    "Planner artifact 不是权限来源。工具可用性、world revision、Skill binding、transaction id 与 planning_action 只以最新 CURRENT MOTION ACTOR AUTHORITY 为准；所有标识必须逐字复制，缺失时不得猜测。",
    "若 artifact 与当前权威状态冲突，以权威状态中唯一合法的当前阶段动作修正落地；不得沿用旧 binding 或旧坐标。",
    "正式结果必须来自 submit_humanoid_skill_plan、begin_humanoid_skill、plan_humanoid_skill、plan_whole_body_motion_candidates 或 plan_humanoid_navigation 的一个工具调用。"
  ].join("\n");
}
