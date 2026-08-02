import {
  Agent,
  tool,
  type AgentInputItem,
  type CallModelInputFilter,
  type FunctionTool,
  type Model,
  type ModelSettings,
  type OutputGuardrail,
  type RunContext,
  type RunToolCallOutputItem,
  type Session,
  type Tool
} from "@openai/agents";
import { isDeepStrictEqual } from "node:util";
import { z } from "zod";
import type { ProviderConfig } from "../config/load.js";
import {
  AgentSpecSchema,
  type AgentSpec
} from "../domain/schema.js";
import {
  AgentSkillInputs,
  SkillDescriptions,
  ToolDescriptions,
  ToolInputs,
  type SkillName,
  type ToolName
} from "../runtime/actions.js";
import { errorMessage } from "../runtime/error-message.js";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import { coordinatorInstructions, workerInstructions } from "./agent-prompts.js";
import {
  DelegationDrainRegistry,
  type DelegationDrainHandle
} from "./delegation-drain.js";
import {
  assertEvidenceRequirementsJointlySatisfiable,
  evidenceContractGuide,
  receiptEvidenceRequirement
} from "./evidence-contract.js";
import {
  agentInvocationMarker,
  currentAgentInvocationId,
  currentAgentInvocationIsRecovery,
  currentAgentInvocationTransportInterruption,
  withAgentInvocation
} from "./agent-scope.js";
import { HarnessRuntimeContext } from "./runtime-context.js";
import {
  providerEventJson,
  sdkEventJson
} from "./sdk-events.js";
import { invalidToolInputResult } from "./tool-input-recovery.js";
import { withModelTelemetry } from "./model-telemetry.js";

export {
  ModelDecisionGuard,
  ModelDecisionStallError
} from "./model-telemetry.js";

const CheckerInput = z.object({}).strict();
const MissionCompletionInput = z.object({
  summary: z.string().trim().min(1)
}).strict();
const MissionOutcomeSchema = MissionCompletionInput.extend({
  status: z.literal("completed")
}).strict();
const DELEGATABLE_CAPABILITIES = [
  ...Object.keys(ToolInputs),
  ...Object.keys(AgentSkillInputs)
] as [string, ...string[]];
const DelegationReferenceInputSchema = z.object({
  transaction_id: z.string().trim().min(1)
}).strict();
export const DelegationSpecSchema = AgentSpecSchema.omit({
  capabilities: true,
  references: true
}).extend({
  capabilities: z.array(z.enum(DELEGATABLE_CAPABILITIES)),
  references: z.array(DelegationReferenceInputSchema).default([])
}).strict().superRefine((spec, context) => {
  if (spec.evidence_requirements.length !== spec.success_criteria.length) {
    context.addIssue({
      code: "custom",
      path: ["evidence_requirements"],
      message: "one typed evidence requirement is required per success criterion"
    });
  }
  try {
    assertEvidenceRequirementsJointlySatisfiable(spec.evidence_requirements);
  } catch (error) {
    context.addIssue({
      code: "custom",
      path: ["evidence_requirements"],
      message: errorMessage(error)
    });
  }
});
const AgentInvocationSchema = z.object({
  node_id: z.string().min(1),
  spec: AgentSpecSchema
}).strict();
const WorkerEvidenceSchema = z.object({
  criterion_index: z.number().int().nonnegative(),
  transaction_ids: z.array(z.string().trim().min(1)).min(1)
}).strict();
const CompletedWorkerOutcomeSchema = z.object({
  status: z.literal("completed"),
  summary: z.string().trim().min(1),
  evidence: z.array(WorkerEvidenceSchema).min(1),
  unmet_criteria: z.array(z.number().int().nonnegative()).length(0)
}).strict();
const BlockedWorkerOutcomeSchema = z.object({
  status: z.literal("blocked"),
  summary: z.string().trim().min(1),
  evidence: z.array(WorkerEvidenceSchema).min(1),
  unmet_criteria: z.array(z.number().int().nonnegative()).min(1)
}).strict();
export const WorkerOutcomeSchema = z.discriminatedUnion("status", [
  CompletedWorkerOutcomeSchema,
  BlockedWorkerOutcomeSchema
]);
const DELEGATE_TOOL_NAME = "delegate_agent";
const CHECKER_TOOL_NAME = "check_mission";
const COMPLETE_MISSION_TOOL_NAME = "complete_mission";
const COMPLETE_ASSIGNMENT_TOOL_NAME = "complete_assignment";
const REPORT_BLOCKED_TOOL_NAME = "report_blocked";

export interface AgentHierarchy {
  root: Agent<HarnessAgentContext>;
  /** Creates one SDK Agent object for one concrete hierarchy node. */
  createWorker(nodeId: string, spec: AgentSpec): Agent<HarnessAgentContext>;
}

export interface HarnessAgentContext {
  runId: string;
}

export interface AgentToolTopology {
  root: string[];
  worker: string[];
}

export function capabilityCatalog(): string[] {
  const capabilities = [...DELEGATABLE_CAPABILITIES];
  if (new Set(capabilities).size !== capabilities.length) {
    throw new Error("Tool and skill capability names must be unique");
  }
  return capabilities;
}

export function agentToolTopology(): AgentToolTopology {
  return {
    root: [DELEGATE_TOOL_NAME, CHECKER_TOOL_NAME, COMPLETE_MISSION_TOOL_NAME],
    worker: [
      ...capabilityCatalog(),
      DELEGATE_TOOL_NAME,
      COMPLETE_ASSIGNMENT_TOOL_NAME,
      REPORT_BLOCKED_TOOL_NAME
    ]
  };
}

export function createAgentHierarchy(input: {
  /** A concrete hierarchy node owns the Model adapter returned by this call. */
  createModel: () => Model;
  /** Nested Agent-as-tool runs use one durable SDK Session per concrete node. */
  createSession?: (agentId: string) => Session;
  provider: ProviderConfig;
  runtime: HarnessRuntimeContext;
  callModelInputFilter?: CallModelInputFilter;
  onModelResponseCompleted?: (agentId: string) => void | Promise<void>;
}): AgentHierarchy {
  const modelSettings: ModelSettings = {
    temperature: input.provider.temperature,
    maxTokens: input.provider.maxOutputTokens,
    parallelToolCalls: true,
    toolChoice: "required"
  };
  let delegate: FunctionTool<HarnessAgentContext, typeof DelegationSpecSchema>;
  const workers = new Map<string, {
    agent: Agent<HarnessAgentContext>;
    spec: AgentSpec;
  }>();
  const createWorker = (
    nodeId: string,
    spec: AgentSpec
  ): Agent<HarnessAgentContext> => {
    const existing = workers.get(nodeId);
    if (existing) {
      if (!isDeepStrictEqual(existing.spec, spec)) {
        throw new Error(`Hierarchy node ${nodeId} cannot change its agent specification`);
      }
      return existing.agent;
    }
    // Role templates are reusable, but a concrete hierarchy node gets its own
    // Agent and Model facade. Provider credentials/configuration may be equal;
    // neither the SDK Agent state nor decision guard is shared by siblings.
    const worker = new Agent<HarnessAgentContext>({
      name: spec.name,
      instructions: workerInstructions(spec, input.runtime),
      model: withModelTelemetry(
        input.createModel(),
        input.runtime,
        nodeId,
        input.onModelResponseCompleted
      ),
      modelSettings,
      tools: actionTools(input.runtime, true),
      resetToolChoice: false,
      toolUseBehavior: (_context, toolResults) => {
        const completion = toolResults.find((result) =>
          result.type === "function_output"
            && isWorkerTerminalTool(result.tool.name)
            && validWorkerTerminalOutcome(result.tool.name, result.output)
        );
        if (!completion || completion.type !== "function_output") {
          return { isFinalOutput: false, isInterrupted: undefined };
        }
        return {
          isFinalOutput: true,
          isInterrupted: undefined,
          finalOutput: workerTerminalOutcomeText(completion.tool.name, completion.output)
        };
      }
    });
    worker.tools.push(delegate, completionTool(input.runtime), blockedTool(input.runtime));
    assertToolTopology("worker", worker.tools.map((entry) => entry.name));
    workers.set(nodeId, { agent: worker, spec: structuredClone(spec) });
    return worker;
  };
  delegate = delegationTool(
    createWorker,
    input.runtime,
    input.callModelInputFilter,
    input.createSession
  );

  const root = new Agent<HarnessAgentContext>({
    name: "Mission Coordinator",
    instructions: coordinatorInstructions(),
    model: withModelTelemetry(
      input.createModel(),
      input.runtime,
      input.runtime.rootAgentId,
      input.onModelResponseCompleted
    ),
    modelSettings,
    tools: [delegate, checkerTool(input.runtime), missionCompletionTool(input.runtime)],
    resetToolChoice: false,
    toolUseBehavior: { stopAtToolNames: [COMPLETE_MISSION_TOOL_NAME] },
    outputGuardrails: [missionCompletionSubmitted(), checkerPassed(input.runtime)]
  });
  const hierarchy = { root, createWorker };
  assertToolTopology("root", root.tools.map((entry) => entry.name));
  return hierarchy;
}

function assertToolTopology(agentName: keyof AgentToolTopology, actual: string[]): void {
  const expected = agentToolTopology();
  const required = expected[agentName];
  if (actual.length !== required.length
    || actual.some((name, index) => name !== required[index])) {
    throw new Error(`${agentName} tool topology does not match its harness contract`);
  }
}

function actionTools(
  runtime: HarnessRuntimeContext,
  includeSkills: boolean
): Tool<HarnessAgentContext>[] {
  const tools: Tool<HarnessAgentContext>[] = [];
  for (const name of Object.keys(ToolInputs) as ToolName[]) {
    tools.push(actionTool("tool", name, ToolInputs[name], ToolDescriptions[name], runtime));
  }
  if (includeSkills) {
    for (const name of Object.keys(AgentSkillInputs) as SkillName[]) {
      tools.push(actionTool("skill", name, AgentSkillInputs[name], SkillDescriptions[name], runtime));
    }
  }
  return tools;
}

function actionTool(
  kind: "tool" | "skill",
  name: ToolName | SkillName,
  parameters: z.ZodObject,
  description: string,
  runtime: HarnessRuntimeContext
): FunctionTool<HarnessAgentContext, z.ZodObject, string> {
  return tool({
    name,
    description,
    parameters,
    strict: true,
    isEnabled: ({ runContext }) => {
      requireRuntime(runContext, runtime);
      return runtime.isCapabilityEnabled(name, agentIdFromContext(runContext, runtime));
    },
    execute: async (toolInput, context, details) => {
      const activeRuntime = requireRuntime(context, runtime);
      const transactionId = details?.toolCall?.callId;
      if (!transactionId) throw new Error(`SDK did not provide a call ID for ${name}`);
      const agentId = agentIdFromContext(context, activeRuntime);
      return kind === "tool"
        ? activeRuntime.invokeTool(name, toolInput, transactionId, agentId)
        : activeRuntime.invokeSkill(name, toolInput, transactionId, agentId);
    }
  });
}

function checkerTool(
  runtime: HarnessRuntimeContext
): FunctionTool<HarnessAgentContext, typeof CheckerInput, string> {
  return tool<typeof CheckerInput, HarnessAgentContext, string>({
    name: CHECKER_TOOL_NAME,
    description: "Evaluate every requested final-state predicate against the current physics state. The verdict only changes when a body command changes the world, so calling it again without delegating work in between returns the identical answer and is refused as a repeated action.",
    parameters: CheckerInput,
    strict: true,
    // Once the current revision is proved, complete_mission is the only valid
    // coordinator transition. Dynamic SDK tool enablement keeps the model from
    // spending another turn re-checking an immutable verdict.
    isEnabled: ({ runContext }) => {
      requireRuntime(runContext, runtime);
      return !runtime.checkerSatisfiedCurrentWorld();
    },
    execute: async (toolInput, context, details) => {
      const activeRuntime = requireRuntime(context, runtime);
      const transactionId = details?.toolCall?.callId;
      if (!transactionId) throw new Error("SDK did not provide a call ID for check_mission");
      return activeRuntime.invokeChecker(
        toolInput,
        transactionId,
        agentIdFromContext(context, activeRuntime)
      );
    }
  });
}

function missionCompletionTool(
  runtime: HarnessRuntimeContext
): FunctionTool<HarnessAgentContext, typeof MissionCompletionInput, string> {
  return tool<typeof MissionCompletionInput, HarnessAgentContext, string>({
    name: COMPLETE_MISSION_TOOL_NAME,
    description: "Submit the completed mission after check_mission confirms the current world state.",
    parameters: MissionCompletionInput,
    strict: true,
    isEnabled: ({ runContext }) => {
      requireRuntime(runContext, runtime);
      return runtime.checkerSatisfiedCurrentWorld();
    },
    execute: async ({ summary }, context) => {
      const activeRuntime = requireRuntime(context, runtime);
      if (!activeRuntime.checkerSatisfiedCurrentWorld()) {
        throw new Error("complete_mission requires Checker success for the current world revision");
      }
      return JSON.stringify(MissionOutcomeSchema.parse({ status: "completed", summary }));
    }
  });
}

function completionTool(
  runtime: HarnessRuntimeContext
): FunctionTool<HarnessAgentContext, typeof CompletedWorkerOutcomeSchema, string> {
  return tool<typeof CompletedWorkerOutcomeSchema, HarnessAgentContext, string>({
    name: COMPLETE_ASSIGNMENT_TOOL_NAME,
    description: "Submit the verified completed result of this assignment and end the worker run.",
    parameters: CompletedWorkerOutcomeSchema,
    strict: true,
    execute: async (outcome, context) => {
      const activeRuntime = requireRuntime(context, runtime);
      const active = activeRuntime.activeNode(agentIdFromContext(context, activeRuntime));
      activeRuntime.assertChildEvidence(
        active.id,
        outcome.status,
        outcome.evidence,
        outcome.unmet_criteria
      );
      return JSON.stringify(outcome);
    }
  });
}

function blockedTool(
  runtime: HarnessRuntimeContext
): FunctionTool<HarnessAgentContext, typeof BlockedWorkerOutcomeSchema, string> {
  return tool<typeof BlockedWorkerOutcomeSchema, HarnessAgentContext, string>({
    name: REPORT_BLOCKED_TOOL_NAME,
    description: "Report a leaf assignment blocked by source-backed evidence. Supervisory agents must delegate recovery instead.",
    parameters: BlockedWorkerOutcomeSchema,
    strict: true,
    isEnabled: ({ runContext }) => {
      const activeRuntime = requireRuntime(runContext, runtime);
      return !activeRuntime.activeNode(agentIdFromContext(runContext, activeRuntime)).may_delegate;
    },
    execute: async (outcome, context) => {
      const activeRuntime = requireRuntime(context, runtime);
      const active = activeRuntime.activeNode(agentIdFromContext(context, activeRuntime));
      activeRuntime.assertChildEvidence(
        active.id,
        outcome.status,
        outcome.evidence,
        outcome.unmet_criteria
      );
      return JSON.stringify(outcome);
    }
  });
}

function delegationTool(
  createWorker: (nodeId: string, spec: AgentSpec) => Agent<HarnessAgentContext>,
  runtime: HarnessRuntimeContext,
  callModelInputFilter: CallModelInputFilter | undefined,
  createSession: ((agentId: string) => Session) | undefined
): FunctionTool<HarnessAgentContext, typeof DelegationSpecSchema> {
  const drainRegistry = new DelegationDrainRegistry();
  const frontierEvidence = receiptEvidenceRequirement(
    0,
    "navigate_frontier",
    { kind: "body", channel: "base" }
  );
  const description = [
    "Create a capability-scoped child agent and wait for its real model run to return.",
    "A supervisory model may emit several delegate_agent calls in one response for independent children; the Agents SDK runs them concurrently up to the configured tool limit, while the harness arbitrates disjoint body-channel leases.",
    "Do not parallelize children that need one another's receipts, contend for a body channel, or combine a fixed world-space plan with a sibling base mutation. An arm plan_joint_targets receipt is relative to the robot and may join disjoint base/head execution; solve_end_effector_position/pose may not.",
    "The child may create further children only when may_delegate is true.",
    "Provide one evidence_requirements entry for each zero-based success criterion. Use goal_predicate for owned final-state predicates; use receipt for an exact action/effect/target/freshness contract. references may be omitted when there is no prior receipt to grant.",
    "Capabilities are operations available during the child's process, not separate completion criteria. A planning receipt consumed by execute_base_plan or execute_joint_plan is verified from the terminal execution receipt's planning_transaction_id, so never declare the plan and its matching execution as separate terminal requirements.",
    `For one frontier movement, normally grant the same may_delegate=false leaf both survey_terrain and navigate_frontier, with exactly one success criterion and this terminal evidence contract: ${JSON.stringify(frontierEvidence)}. That leaf surveys first, then its model chooses choice_id and calls the physical skill. Do not split observation and navigation across children unless the survey child has completed and its exact survey_terrain transaction_id is included in the navigation child's references. The harness verifies provenance and never chooses or substitutes a frontier.`,
    "The child receives only the listed formal capabilities and accepted action receipts explicitly granted through references; its capability list must be a strict subset of the active parent's list.",
    "goal_predicate_indexes are zero-based indexes into Mission goal.predicates. A supervisor must own at least one and complete_assignment is rejected until every owned predicate passes in the live world. A bounded observation or planning leaf that cannot actuate its final state must use [].",
    "An unmet voxel_at owner must retain the break_voxel and/or place_voxel authority required by the current-to-target material transition. An observation-only voxel leaf must use goal_predicate_indexes: [].",
    "Any branch with break_voxel/place_voxel authority that receives plan_base_path, plan_joint_targets, solve_end_effector_position/pose, or plan_arm_retraction must also retain its matching execute_base_plan, execute_joint_plan, or set_joint_targets capability. This includes supervisors, which need the complete capability budget so narrower descendants can inherit executors. A pure planning leaf with no mutation authority must use goal_predicate_indexes: [], then pass its accepted receipt to an executor.",
    "Set may_delegate=false for an agent that should invoke observation, planning, or body tools itself. A may_delegate=true supervisor cannot invoke those capabilities and must create still-narrower children.",
    "To hand off a prior result, put only its exact receipt transaction_id in references. The harness resolves and validates the action name; never copy an internal plan_id into the objective.",
    `Capabilities available for grants:\n${capabilityGuide()}`,
    `Receipt evidence contracts by action:\n${evidenceContractGuide()}`
  ].join(" ");
  return tool<typeof DelegationSpecSchema, HarnessAgentContext, string>({
    name: DELEGATE_TOOL_NAME,
    description,
    parameters: DelegationSpecSchema,
    strict: true,
    // Invalid model arguments return to the SDK's native correction loop
    // without invoking the hierarchy. Every real execution or transport error
    // is rethrown so the mission recovery boundary keeps owning it.
    errorFunction: (_context, error) => invalidToolInputResult(error, DELEGATE_TOOL_NAME),
    isEnabled: ({ runContext }) => {
      requireRuntime(runContext, runtime);
      if (runtime.checkerSatisfiedCurrentWorld()) return false;
      const parent = invocationFromContext(runContext, runtime);
      return runtime.canDelegate(parent.spec, parent.nodeId);
    },
    execute: async (childInput, context, details) => {
      if (!context) throw new Error("Agent runtime context is required for delegation");
      const activeRuntime = requireRuntime(context, runtime);
      let childSpec: AgentSpec | undefined;
      let entry: Awaited<ReturnType<HarnessRuntimeContext["beginDelegation"]>> | undefined;
      let drain: DelegationDrainHandle | undefined;
      try {
        const parent = invocationFromContext(context, activeRuntime);
        const callId = details?.toolCall?.callId;
        if (!callId) throw new Error("SDK did not provide a call ID for delegate_agent");
        drain = drainRegistry.register(
          parent.nodeId,
          callId,
          currentAgentInvocationIsRecovery()
        );
        childSpec = {
          ...childInput,
          references: activeRuntime.acceptedActionReferences(
            childInput.references.map((reference) => reference.transaction_id)
          )
        };
        entry = await activeRuntime.beginDelegation(
          parent.spec,
          childSpec,
          callId,
          parent.nodeId,
          drain.sourceCallIds,
          drain.recoveryState
        );
        if (entry.cached_output !== undefined) return entry.cached_output;

        const childNodeId = entry.node.id;
        const session = createSession?.(childNodeId);
        const invocationInput = agentInvocationInput(childNodeId, childSpec, activeRuntime);
        const sessionBaseline = session
          ? await prepareSessionForAgentInvocation(session, invocationInput)
          : undefined;
        const nativeAgentTool = delegatedAgentTool(
          createWorker(childNodeId, childSpec),
          runtime,
          description,
          callModelInputFilter,
          session,
          childSpec.may_delegate ? 4 : 1
        );
        const result = await withAgentInvocation(childNodeId, async () => {
          let value: unknown;
          try {
            value = await nativeAgentTool.invoke(
              context,
              JSON.stringify({ node_id: childNodeId, spec: childSpec }),
              details
            );
          } catch (error) {
            const interruption = currentAgentInvocationTransportInterruption();
            if (interruption) {
              await restoreInterruptedSession(session, sessionBaseline, interruption);
              throw interruption;
            }
            if (activeRuntime.signal?.aborted) {
              await restoreInterruptedSession(session, sessionBaseline, error);
            }
            throw error;
          }
          const interruption = currentAgentInvocationTransportInterruption();
          if (!interruption) return value;
          await restoreInterruptedSession(session, sessionBaseline, interruption);
          throw interruption;
        }, !entry.created);
        const output = typeof result === "string" ? result : JSON.stringify(result);
        const parsedOutcome = WorkerOutcomeSchema.safeParse(
          typeof output === "string" ? safeJson(output) : undefined
        );
        if (!parsedOutcome.success) {
          throw new Error(
            typeof output === "string" && output.trim() !== ""
              ? output
              : `${childSpec.name} returned no valid terminal-tool output`
          );
        }
        const outcome = parsedOutcome.data;
        const verifiedEvidence = activeRuntime.assertChildEvidence(
          entry.node.id,
          outcome.status,
          outcome.evidence,
          outcome.unmet_criteria
        );
        if (outcome.status === "blocked") {
          await activeRuntime.blockChild(entry.node.id, outcome.summary);
          return JSON.stringify({
            accepted: false,
            code: "child_agent_blocked",
            agent_id: entry.node.id,
            agent_name: childSpec.name,
            ...activeRuntime.worldIdentity(),
            outcome,
            verified_evidence: verifiedEvidence,
            automatic_actuation: false,
            recovery: "Use the cited receipts to delegate a fresh capability-appropriate model node. The harness does not perform the blocked movement."
          });
        }
        const references = activeRuntime.acceptedActionReferences(
          outcome.evidence.flatMap((item) => item.transaction_ids)
        ).map((reference) => ({ transaction_id: reference.transaction_id }));
        const completedOutput = JSON.stringify({
          ...outcome,
          verified_evidence: verifiedEvidence,
          references,
          ...activeRuntime.worldIdentity()
        });
        await activeRuntime.completeChild(entry.node.id, completedOutput);
        return completedOutput;
      } catch (error) {
        try {
          rethrowDelegationInterruption(error, activeRuntime.signal);
        } catch (interruption) {
          await drain?.settleAndDrain();
          throw interruption;
        }
        if (!entry || !childSpec) {
          return JSON.stringify({
            accepted: false,
            code: "delegation_rejected",
            ...activeRuntime.worldIdentity(),
            error: errorMessage(error),
            automatic_actuation: false
          });
        }
        await activeRuntime.failChild(entry.node.id, errorMessage(error));
        return JSON.stringify({
          accepted: false,
          code: "child_agent_failed",
          agent_id: entry.node.id,
          agent_name: childSpec.name,
          ...activeRuntime.worldIdentity(),
          error: errorMessage(error),
          automatic_actuation: false,
          recovery: "Delegate a fresh model-run node from current world evidence. For movement, re-survey and select a different reachable frontier; no programmatic action was substituted."
        });
      } finally {
        drain?.settle();
      }
    }
  });
}

export function rethrowDelegationInterruption(
  error: unknown,
  signal?: AbortSignal
): void {
  signal?.throwIfAborted();
  if (isTransportInterruption(error)) throw error;
}

function delegatedAgentTool(
  worker: Agent<HarnessAgentContext>,
  runtime: HarnessRuntimeContext,
  description: string,
  callModelInputFilter: CallModelInputFilter | undefined,
  session: Session | undefined,
  maxFunctionToolConcurrency: number
) {
  return worker.asTool({
    toolName: DELEGATE_TOOL_NAME,
    toolDescription: description,
    parameters: AgentInvocationSchema,
    inputBuilder: ({ params }) => agentInvocationInput(params.node_id, params.spec, runtime),
    includeInputSchema: true,
    runConfig: {
      tracingDisabled: true,
      traceIncludeSensitiveData: false,
      toolExecution: { maxFunctionToolConcurrency },
      toolNotFoundBehavior: "return_error_to_model",
      ...(callModelInputFilter ? { callModelInputFilter } : {})
    },
    runOptions: {
      maxTurns: null,
      toolExecution: { maxFunctionToolConcurrency },
      toolNotFoundBehavior: "return_error_to_model",
      ...(callModelInputFilter ? { callModelInputFilter } : {}),
      ...(session ? { session } : {}),
      ...(runtime.signal ? { signal: runtime.signal } : {})
    },
    onStream: async ({ event }) => {
      const agentId = currentAgentInvocationId();
      if (!agentId) throw new Error("Nested agent stream lost its hierarchy identity");
      const frameworkEvent = sdkEventJson(event);
      if (frameworkEvent) {
        await runtime.recordFramework(runtime.frameworkScope(agentId), frameworkEvent, agentId);
      }
      const providerEvent = providerEventJson(event);
      if (providerEvent) await runtime.recordProvider(providerEvent, agentId);
    },
    customOutputExtractor: async (result) => {
      const completions = result.newItems.filter((item): item is RunToolCallOutputItem =>
        item.type === "tool_call_output_item"
          && item.rawItem.type === "function_call_result"
          && isWorkerTerminalTool(item.rawItem.name)
          && item.rawItem.status === "completed"
          && validWorkerTerminalOutcome(item.rawItem.name, item.output)
      );
      if (completions.length !== 1) {
        throw new Error(`${worker.name} must call exactly one valid terminal tool`);
      }
      const completion = completions[0]!;
      const rawItem = completion.rawItem;
      if (rawItem.type !== "function_call_result" || !isWorkerTerminalTool(rawItem.name)) {
        throw new Error(`${worker.name} terminal result has an invalid SDK item type`);
      }
      const outcome = parseWorkerTerminalOutcome(rawItem.name, completion.output);
      if (result.finalOutput !== completion.output) {
        throw new Error(`${worker.name} completion output does not match its final output`);
      }
      return JSON.stringify(outcome);
    }
  });
}

export function agentInvocationInput(
  nodeId: string,
  spec: AgentSpec,
  runtime: Pick<HarnessRuntimeContext, "goal">
): string {
  const normalizedSpec = AgentSpecSchema.parse(spec);
  const params = { node_id: nodeId, spec: normalizedSpec };
  return [
    agentInvocationMarker(nodeId),
    `Agent invocation: ${JSON.stringify(params)}`,
    `Agent specification: ${JSON.stringify(normalizedSpec)}`,
    `Mission goal: ${JSON.stringify(runtime.goal())}`
  ].join("\n");
}

async function prepareSessionForAgentInvocation(
  session: Session,
  invocationInput: string
): Promise<AgentInputItem[]> {
  const items = await session.getItems();
  const danglingOpeningInput: AgentInputItem = {
    type: "message",
    role: "user",
    content: invocationInput
  };
  if (!isDeepStrictEqual(items.at(-1), danglingOpeningInput)) return items;

  const baseline = items.slice(0, -1);
  const replaceItems = atomicSessionReplace(session);
  if (!replaceItems) {
    throw new Error(
      "Agent Session ends with an interrupted invocation input and cannot be repaired atomically"
    );
  }
  await replaceItems(baseline);
  const repaired = await session.getItems();
  if (!isDeepStrictEqual(repaired, baseline)) {
    throw new Error("Agent Session did not preserve the repaired invocation baseline");
  }
  return repaired;
}

async function restoreInterruptedSession(
  session: Session | undefined,
  baseline: AgentInputItem[] | undefined,
  interruption: unknown
): Promise<void> {
  if (!session || !baseline) return;
  try {
    const current = await session.getItems();
    if (isDeepStrictEqual(current, baseline)) return;
    const replaceItems = atomicSessionReplace(session);
    if (!replaceItems) {
      throw new Error("Agent Session cannot be restored atomically after an interrupted model request");
    }
    await replaceItems(baseline);
    const restored = await session.getItems();
    if (!isDeepStrictEqual(restored, baseline)) {
      throw new Error("Agent Session did not preserve the restored invocation baseline");
    }
  } catch (restoreError) {
    throw new AggregateError(
      [interruption, restoreError],
      "Agent Session could not be restored after an interrupted model request"
    );
  }
}

function atomicSessionReplace(
  session: Session
): ((items: AgentInputItem[]) => Promise<void>) | undefined {
  const replaceItems = (session as Session & {
    replaceItems?: (items: AgentInputItem[]) => Promise<void>;
  }).replaceItems;
  return replaceItems ? (items) => replaceItems.call(session, items) : undefined;
}

function missionCompletionSubmitted(): OutputGuardrail<"text", HarnessAgentContext> {
  return {
    name: "complete_mission_required",
    execute: async ({ agentOutput }) => {
      const parsed = typeof agentOutput === "string"
        ? safeJson(agentOutput)
        : undefined;
      const accepted = MissionOutcomeSchema.safeParse(parsed).success;
      return {
        tripwireTriggered: !accepted,
        outputInfo: { complete_mission_submitted: accepted }
      };
    }
  };
}

function checkerPassed(runtime: HarnessRuntimeContext): OutputGuardrail<"text", HarnessAgentContext> {
  return {
    name: "checker_success_required",
    execute: async () => ({
      tripwireTriggered: !runtime.checkerSatisfiedCurrentWorld(),
      outputInfo: runtime.checkpoint.checker ?? { success: false, reason: "checker_not_called" }
    })
  };
}

function requireRuntime(
  context: RunContext<HarnessAgentContext> | undefined,
  runtime: HarnessRuntimeContext
): HarnessRuntimeContext {
  if (context?.context.runId !== runtime.runId) {
    throw new Error("Agent runtime context mismatch");
  }
  return runtime;
}

function invocationFromContext(
  context: RunContext<HarnessAgentContext> | undefined,
  runtime: HarnessRuntimeContext
): { nodeId: string; spec: AgentSpec | null } {
  if (context?.toolInput === undefined) return { nodeId: runtime.rootAgentId, spec: null };
  const invocation = AgentInvocationSchema.parse(context.toolInput);
  return { nodeId: invocation.node_id, spec: invocation.spec };
}

function agentIdFromContext(
  context: RunContext<HarnessAgentContext> | undefined,
  runtime: HarnessRuntimeContext
): string {
  return invocationFromContext(context, runtime).nodeId;
}

function parseWorkerTerminalOutcome(
  toolName: string,
  value: unknown
): z.infer<typeof WorkerOutcomeSchema> {
  if (typeof value !== "string") {
    throw new Error(`${toolName} returned a non-text result`);
  }
  const parsed = JSON.parse(value);
  return workerTerminalSchema(toolName).parse(parsed);
}

function validWorkerTerminalOutcome(toolName: string, value: unknown): boolean {
  const parsed = typeof value === "string" ? safeJson(value) : value;
  return workerTerminalSchema(toolName).safeParse(parsed).success;
}

function workerTerminalOutcomeText(toolName: string, value: unknown): string {
  if (typeof value === "string") return value;
  return JSON.stringify(workerTerminalSchema(toolName).parse(value));
}

function workerTerminalSchema(toolName: string) {
  if (toolName === COMPLETE_ASSIGNMENT_TOOL_NAME) return CompletedWorkerOutcomeSchema;
  if (toolName === REPORT_BLOCKED_TOOL_NAME) return BlockedWorkerOutcomeSchema;
  throw new Error(`Unknown worker terminal tool: ${toolName}`);
}

function isWorkerTerminalTool(name: string): boolean {
  return name === COMPLETE_ASSIGNMENT_TOOL_NAME || name === REPORT_BLOCKED_TOOL_NAME;
}

function safeJson(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
}

function capabilityGuide(): string {
  return [
    `Observation and planning tools: ${Object.keys(ToolDescriptions).join(", ")}`,
    `Physical skill tools: ${Object.keys(SkillDescriptions).join(", ")}`,
    "Planning receipts pair plan_base_path with execute_base_plan and "
      + "plan_joint_targets/solve_end_effector_position/solve_end_effector_pose with execute_joint_plan. "
      + "Frontier exploration pairs survey_terrain with model-selected navigate_frontier."
  ].join("\n");
}
