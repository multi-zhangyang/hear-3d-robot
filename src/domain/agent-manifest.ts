import { z } from "zod";

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const ReasoningEffortSchema = z.enum([
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max"
]);

const AgentToolUseBehaviorIdentitySchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("sdk_flag"),
    value: z.enum(["run_llm_again", "stop_on_first_tool"])
  }).strict(),
  z.object({
    kind: z.literal("stop_at_tool_names"),
    tool_names: z.array(z.string().trim().min(1)).min(1)
  }).strict(),
  z.object({
    kind: z.literal("harness_callback"),
    contract_id: z.string().trim().min(1),
    terminal_tool_names: z.array(z.string().trim().min(1)).min(1)
  }).strict()
]);

const AgentToolContractSchema = z.object({
  dispatch_kind: z.enum([
    "model_agent",
    "model_pipeline",
    "deterministic_service"
  ]).optional(),
  tool_name: z.string().trim().min(1),
  target_role: z.enum(["goal_manager", "sentry", "motion", "executor"]),
  target_agent_id: z.string().trim().min(1),
  target_agent_name: z.string().trim().min(1),
  tool_schema_sha256: Sha256Schema,
  input_builder_contract: z.enum([
    "objective_text_v1",
    "goal_manager_authority_envelope_v1",
    "goal_manager_authority_envelope_v2",
    "live_authority_delegation_v1",
    "motion_authority_envelope_v1",
    "motion_planner_actor_pipeline_v1",
    "validated_execution_task_json_v1",
    "grounding_monitor_direct_v1",
    "validated_execution_gate_v1"
  ]),
  input_builder_sha256: Sha256Schema,
  implementation_contract: z.string().trim().min(1).optional(),
  run_options: z.object({
    session_agent_id: z.string().trim().min(1),
    context_source: z.literal("parent_run_context"),
    max_turns: z.literal("unbounded")
  }).strict().optional(),
  resume_context_strategy: z.enum(["merge", "replace", "preferSerialized"]).optional(),
  pipeline: z.object({
    planner_agent_id: z.string().trim().min(1),
    planner_session_agent_id: z.string().trim().min(1),
    actor_agent_id: z.string().trim().min(1),
    actor_session_agent_id: z.string().trim().min(1),
    artifact_contract: z.literal("bounded_motion_plan_artifact_v1"),
    authority_contract: z.literal("fresh_motion_authority_envelope_v1")
  }).strict().optional(),
  include_input_schema: z.literal(false),
  needs_approval: z.literal(false),
  output_contract: z.enum([
    "nested_agent_final_output_text",
    "formal_action_receipt"
  ])
}).strict().superRefine((contract, context) => {
  const dispatch = contract.dispatch_kind ?? "model_agent";
  if (dispatch === "model_agent") {
    if (!contract.run_options) {
      context.addIssue({
        code: "custom",
        path: ["run_options"],
        message: "A model Agent delegation requires owned Session run options"
      });
    }
    if (!contract.resume_context_strategy) {
      context.addIssue({
        code: "custom",
        path: ["resume_context_strategy"],
        message: "A model Agent delegation requires a resume context strategy"
      });
    }
    if (contract.output_contract !== "nested_agent_final_output_text") {
      context.addIssue({
        code: "custom",
        path: ["output_contract"],
        message: "A model Agent delegation must return nested final output"
      });
    }
  } else if (dispatch === "model_pipeline") {
    if (!contract.pipeline) {
      context.addIssue({
        code: "custom",
        path: ["pipeline"],
        message: "A model pipeline requires planner and actor ownership"
      });
    }
    if (contract.run_options || contract.resume_context_strategy
      || contract.implementation_contract) {
      context.addIssue({
        code: "custom",
        path: ["dispatch_kind"],
        message: "A model pipeline owns separate planner and actor Sessions"
      });
    }
    if (contract.output_contract !== "formal_action_receipt") {
      context.addIssue({
        code: "custom",
        path: ["output_contract"],
        message: "A model pipeline must return the Actor's formal action receipt"
      });
    }
  } else {
    if (!contract.implementation_contract) {
      context.addIssue({
        code: "custom",
        path: ["implementation_contract"],
        message: "A deterministic service requires an implementation contract"
      });
    }
    if (contract.run_options || contract.resume_context_strategy) {
      context.addIssue({
        code: "custom",
        path: ["dispatch_kind"],
        message: "A deterministic service must not own model Session run options"
      });
    }
    if (contract.output_contract !== "formal_action_receipt") {
      context.addIssue({
        code: "custom",
        path: ["output_contract"],
        message: "A deterministic service must return a formal action receipt"
      });
    }
  }
});

const AgentModelSettingsIdentitySchema = z.object({
  request_timeout_ms: z.number().int().positive(),
  temperature: z.number().min(0).max(2),
  reasoning_effort: ReasoningEffortSchema.optional(),
  max_output_tokens: z.number().int().positive().optional(),
  context_window_tokens: z.number().int().positive(),
  compact_trigger_tokens: z.number().int().positive(),
  compact_recent_model_turns: z.number().int().nonnegative(),
  compact_max_output_tokens: z.number().int().positive().optional()
}).strict();

const AgentModelIdentitySchema = z.object({
  execution_kind: z.literal("model").optional(),
  agent_id: z.string().trim().min(1),
  agent_name: z.string().trim().min(1),
  role: z.enum([
    "goal_manager",
    "coordinator",
    "sentry",
    "motion_planner",
    "motion",
    "executor",
    "compactor"
  ]),
  protocol: z.enum([
    "openai_compatible",
    "openai_responses",
    "anthropic_messages"
  ]),
  model: z.string().trim().min(1),
  endpoint_sha256: Sha256Schema,
  instructions_sha256: Sha256Schema,
  tool_schema_sha256: Sha256Schema,
  sdk_model_settings: z.record(z.string(), z.json()),
  reset_tool_choice: z.boolean(),
  tool_use_behavior: AgentToolUseBehaviorIdentitySchema,
  settings: AgentModelSettingsIdentitySchema
}).strict();

const AgentDeterministicServiceIdentitySchema = z.object({
  execution_kind: z.literal("deterministic_service"),
  agent_id: z.string().trim().min(1),
  agent_name: z.string().trim().min(1),
  role: z.enum(["sentry", "executor"]),
  implementation_contract: z.string().trim().min(1),
  decision_authority_role: z.literal("coordinator")
}).strict();

const AgentServiceOrLegacyModelIdentitySchema = z.union([
  AgentDeterministicServiceIdentitySchema,
  AgentModelIdentitySchema
]);

export const LegacyAgentManifestSchema = z.object({
  version: z.literal(1),
  runtime: z.literal("humanoid_g1"),
  harness_contract_version: z.number().int().positive(),
  epoch_id: z.string().uuid(),
  created_at: z.string().datetime(),
  runtime_sdk_identity: z.record(z.string().min(1), z.string().min(1)),
  agents: z.object({
    goal_manager: AgentModelIdentitySchema,
    coordinator: AgentModelIdentitySchema,
    sentry: AgentServiceOrLegacyModelIdentitySchema,
    motion_planner: AgentModelIdentitySchema,
    motion: AgentModelIdentitySchema,
    executor: AgentServiceOrLegacyModelIdentitySchema,
    compactor: AgentModelIdentitySchema
  }).strict(),
  agent_tool_contracts: z.array(AgentToolContractSchema),
  identity_sha256: Sha256Schema
}).strict().superRefine((manifest, context) => {
  for (const role of [
    "goal_manager",
    "coordinator",
    "sentry",
    "motion_planner",
    "motion",
    "executor",
    "compactor"
  ] as const) {
    if (manifest.agents[role].role !== role) {
      context.addIssue({
        code: "custom",
        path: ["agents", role, "role"],
        message: "Agent manifest role does not match its profile key"
      });
    }
  }
  const toolNames = new Set<string>();
  for (const [index, contract] of manifest.agent_tool_contracts.entries()) {
    if (toolNames.has(contract.tool_name)) {
      context.addIssue({
        code: "custom",
        path: ["agent_tool_contracts", index, "tool_name"],
        message: "Agent-as-tool contract names must be unique"
      });
    }
    toolNames.add(contract.tool_name);
    const target = manifest.agents[contract.target_role];
    if (contract.target_agent_id !== target.agent_id) {
      context.addIssue({
        code: "custom",
        path: ["agent_tool_contracts", index, "target_agent_id"],
        message: "Agent-as-tool target id does not match its target role"
      });
    }
    if (contract.target_agent_name !== target.agent_name) {
      context.addIssue({
        code: "custom",
        path: ["agent_tool_contracts", index, "target_agent_name"],
        message: "Agent-as-tool target name does not match its target role"
      });
    }
    if (contract.run_options
      && contract.run_options.session_agent_id !== target.agent_id) {
      context.addIssue({
        code: "custom",
        path: ["agent_tool_contracts", index, "run_options", "session_agent_id"],
        message: "Agent-as-tool Session owner does not match its target role"
      });
    }
    if (contract.dispatch_kind === "model_pipeline") {
      const pipeline = contract.pipeline;
      const planner = manifest.agents.motion_planner;
      const actor = manifest.agents.motion;
      if (!pipeline
        || pipeline.planner_agent_id !== planner.agent_id
        || pipeline.planner_session_agent_id !== planner.agent_id
        || pipeline.actor_agent_id !== actor.agent_id
        || pipeline.actor_session_agent_id !== actor.agent_id
        || contract.target_agent_id !== actor.agent_id) {
        context.addIssue({
          code: "custom",
          path: ["agent_tool_contracts", index, "pipeline"],
          message: "Motion pipeline identities must match independent Planner and Actor nodes"
        });
      }
    }
    if (contract.dispatch_kind === "deterministic_service") {
      if (target.execution_kind !== "deterministic_service"
        || target.implementation_contract !== contract.implementation_contract) {
        context.addIssue({
          code: "custom",
          path: ["agent_tool_contracts", index, "implementation_contract"],
          message: "Deterministic service contract does not match its target identity"
        });
      }
    }
  }
});

const NeuralAgentIdentityBaseSchema = z.object({
  agent_id: z.string().trim().min(1),
  agent_name: z.string().trim().min(1),
  parent_agent_id: z.string().trim().min(1).nullable(),
  layer: z.enum([
    "executive",
    "action_selection",
    "perceptual_association",
    "sensorimotor",
    "premotor",
    "motor_planning",
    "predictive_rollout",
    "controller",
    "reflex",
    "body"
  ]),
  pathway: z.enum([
    "executive_control",
    "goal_valuation",
    "perceptual_association",
    "sensorimotor_selection",
    "cerebellar_prediction",
    "interoceptive_risk",
    "premotor_composition",
    "motor_intent",
    "physical_execution",
    "ascending_feedback"
  ]),
  orchestration_kind: z.enum([
    "root_runner",
    "agent_tool",
    "exclusive_lease_episode",
    "runtime_service",
    "controller_loop",
    "physical_plant"
  ]),
  session_mode: z.enum(["independent_file_session", "none"]),
  cadence: z.enum([
    "mission_event",
    "goal_event",
    "world_event",
    "skill_event",
    "rollout_event",
    "execution_transaction",
    "recovery_event",
    "controller_tick",
    "physics_tick"
  ]),
  maximum_correction_scope: z.enum([
    "none",
    "local",
    "pathway",
    "supervisory"
  ]),
  parallel_group: z.enum([
    "perception_interpretation",
    "sensorimotor_assessment"
  ]).optional(),
  parallel_safe: z.boolean(),
  physical_write_authority: z.boolean(),
  capabilities: z.array(z.string().trim().min(1))
});

const NeuralModelAgentIdentitySchema = NeuralAgentIdentityBaseSchema.extend({
  execution_kind: z.literal("model_agent"),
  provider_profile: z.enum([
    "executive",
    "associative",
    "sensorimotor",
    "motor_intent"
  ]),
  protocol: z.enum([
    "openai_compatible",
    "openai_responses",
    "anthropic_messages"
  ]),
  model: z.string().trim().min(1),
  endpoint_sha256: Sha256Schema,
  instructions_sha256: Sha256Schema,
  tool_schema_sha256: Sha256Schema,
  output_schema_sha256: Sha256Schema,
  sdk_model_settings: z.record(z.string(), z.json()),
  reset_tool_choice: z.boolean(),
  settings: AgentModelSettingsIdentitySchema
}).strict();

const NeuralRuntimeNodeIdentitySchema = NeuralAgentIdentityBaseSchema.extend({
  execution_kind: z.enum([
    "deterministic_service",
    "learned_controller"
  ]),
  implementation_contract: z.string().trim().min(1)
}).strict();

export const NeuralAgentIdentitySchema = z.discriminatedUnion(
  "execution_kind",
  [NeuralModelAgentIdentitySchema, NeuralRuntimeNodeIdentitySchema]
);

export const NeuralControlEdgeSchema = z.object({
  parent_agent_id: z.string().trim().min(1),
  child_agent_id: z.string().trim().min(1),
  orchestration_kind: z.enum([
    "agent_tool",
    "exclusive_lease_episode",
    "runtime_service",
    "controller_loop",
    "physical_plant"
  ]),
  contract_id: z.string().trim().min(1),
  tool_name: z.string().trim().min(1).optional(),
  session_agent_id: z.string().trim().min(1).optional(),
  parallel_group: z.enum([
    "perception_interpretation",
    "sensorimotor_assessment"
  ]).optional()
}).strict();

export const NeuralSignalContractSchema = z.object({
  source_agent_id: z.string().trim().min(1),
  target_agent_id: z.string().trim().min(1),
  direction: z.enum(["descending", "ascending", "reentrant"]),
  signal_kinds: z.array(z.enum([
    "goal_context",
    "goal_selected",
    "sensory_evidence",
    "scene_interpretation",
    "memory_retrieval",
    "perceptual_belief",
    "affordance_hypothesis",
    "risk_assessment",
    "forward_prediction",
    "prediction_error",
    "skill_proposal",
    "skill_commitment",
    "motor_intent",
    "rollout_result",
    "execution_receipt",
    "skill_completed",
    "skill_failed",
    "escalation"
  ])).min(1)
}).strict();

export const NeuralAgentManifestSchema = z.object({
  version: z.literal(3),
  runtime: z.literal("humanoid_g1"),
  harness_contract_version: z.number().int().positive(),
  neural_contract_version: z.number().int().positive(),
  epoch_id: z.string().uuid(),
  created_at: z.string().datetime(),
  runtime_sdk_identity: z.record(z.string().min(1), z.string().min(1)),
  root_agent_id: z.string().trim().min(1),
  agents: z.record(z.string().trim().min(1), NeuralAgentIdentitySchema),
  control_edges: z.array(NeuralControlEdgeSchema),
  signal_contracts: z.array(NeuralSignalContractSchema),
  identity_sha256: Sha256Schema
}).strict().superRefine((manifest, context) => {
  const identities = Object.values(manifest.agents);
  if (identities.length === 0) {
    context.addIssue({
      code: "custom",
      path: ["agents"],
      message: "A neural Agent manifest cannot be empty"
    });
    return;
  }
  const roots = identities.filter((agent) => agent.parent_agent_id === null);
  if (roots.length !== 1 || roots[0]?.agent_id !== manifest.root_agent_id
    || roots[0]?.orchestration_kind !== "root_runner") {
    context.addIssue({
      code: "custom",
      path: ["root_agent_id"],
      message: "A neural Agent manifest requires exactly one root runner"
    });
  }
  let physicalWriters = 0;
  for (const [agentId, agent] of Object.entries(manifest.agents)) {
    if (agentId !== agent.agent_id) {
      context.addIssue({
        code: "custom",
        path: ["agents", agentId, "agent_id"],
        message: "Neural manifest keys must equal structural Agent ids"
      });
    }
    if (agent.physical_write_authority) physicalWriters += 1;
    if (agent.execution_kind === "model_agent"
      && agent.session_mode !== "independent_file_session") {
      context.addIssue({
        code: "custom",
        path: ["agents", agentId, "session_mode"],
        message: "Every model Agent owns an independent Session"
      });
    }
    if (agent.execution_kind !== "model_agent" && agent.session_mode !== "none") {
      context.addIssue({
        code: "custom",
        path: ["agents", agentId, "session_mode"],
        message: "Runtime nodes cannot own model Sessions"
      });
    }
    if (agent.execution_kind === "model_agent"
      && ["execution_transaction", "controller_tick", "physics_tick"]
        .includes(agent.cadence)) {
      context.addIssue({
        code: "custom",
        path: ["agents", agentId, "cadence"],
        message: "Model Agents cannot run at execution, controller, or physics cadence"
      });
    }
  }
  if (physicalWriters !== 1) {
    context.addIssue({
      code: "custom",
      path: ["agents"],
      message: "The neural hierarchy requires exactly one physical writer"
    });
  }
  const childOwners = new Map<string, string>();
  for (const [index, edge] of manifest.control_edges.entries()) {
    const parent = manifest.agents[edge.parent_agent_id];
    const child = manifest.agents[edge.child_agent_id];
    if (!parent || !child || child.parent_agent_id !== edge.parent_agent_id) {
      context.addIssue({
        code: "custom",
        path: ["control_edges", index],
        message: "Control edges must match the declared parent-child tree"
      });
      continue;
    }
    if (childOwners.has(edge.child_agent_id)) {
      context.addIssue({
        code: "custom",
        path: ["control_edges", index, "child_agent_id"],
        message: "A neural node cannot have multiple control parents"
      });
    }
    childOwners.set(edge.child_agent_id, edge.parent_agent_id);
    if (edge.orchestration_kind !== child.orchestration_kind) {
      context.addIssue({
        code: "custom",
        path: ["control_edges", index, "orchestration_kind"],
        message: "Control edge orchestration must match its child node"
      });
    }
    if (child.execution_kind === "model_agent"
      && child.orchestration_kind === "agent_tool"
      && (edge.tool_name === undefined
        || edge.session_agent_id !== child.agent_id)) {
      context.addIssue({
        code: "custom",
        path: ["control_edges", index],
        message: "An Agent-as-tool edge must bind the child's own Session"
      });
    }
  }
  for (const agent of identities) {
    if (agent.parent_agent_id !== null
      && childOwners.get(agent.agent_id) !== agent.parent_agent_id) {
      context.addIssue({
        code: "custom",
        path: ["agents", agent.agent_id, "parent_agent_id"],
        message: "Every non-root neural node requires one matching control edge"
      });
    }
  }
  for (const [index, signal] of manifest.signal_contracts.entries()) {
    if (!manifest.agents[signal.source_agent_id]
      || !manifest.agents[signal.target_agent_id]) {
      context.addIssue({
        code: "custom",
        path: ["signal_contracts", index],
        message: "Neural signal contracts must reference structural nodes"
      });
    }
  }
});

export const AgentManifestSchema = z.union([
  LegacyAgentManifestSchema,
  NeuralAgentManifestSchema
]);

/**
 * Minimal, immutable identity retained for validating decisions created by an
 * archived Agent epoch.  It intentionally omits prompts, tools, Sessions, SDK
 * state, and orchestration contracts: historical authority may prove an old
 * decision, but it can never be resumed as a live hierarchy.
 */
export const HistoricalAgentAuthorityManifestSchema = z.object({
  epoch_id: z.string().uuid(),
  identity_sha256: Sha256Schema,
  agent_ids: z.array(z.string().trim().min(1)).min(1),
  goal_manager_agent_id: z.string().trim().min(1),
  grounding_manager_agent_id: z.string().trim().min(1),
  execution_manager_agent_id: z.string().trim().min(1)
}).strict().superRefine((manifest, context) => {
  const ids = new Set(manifest.agent_ids);
  if (ids.size !== manifest.agent_ids.length) {
    context.addIssue({
      code: "custom",
      path: ["agent_ids"],
      message: "Historical Agent authority identities must be unique"
    });
  }
  for (const [field, agentId] of [
    ["goal_manager_agent_id", manifest.goal_manager_agent_id],
    ["grounding_manager_agent_id", manifest.grounding_manager_agent_id],
    ["execution_manager_agent_id", manifest.execution_manager_agent_id]
  ] as const) {
    if (!ids.has(agentId)) {
      context.addIssue({
        code: "custom",
        path: [field],
        message: "Historical Manager authority must name an archived Agent identity"
      });
    }
  }
});

export type LegacyAgentManifest = z.infer<typeof LegacyAgentManifestSchema>;
export type NeuralAgentManifest = z.infer<typeof NeuralAgentManifestSchema>;
export type AgentManifest = z.infer<typeof AgentManifestSchema>;
export type HistoricalAgentAuthorityManifest = z.infer<
  typeof HistoricalAgentAuthorityManifestSchema
>;
export type AgentModelIdentity = z.infer<typeof AgentModelIdentitySchema>;
export type AgentDeterministicServiceIdentity = z.infer<
  typeof AgentDeterministicServiceIdentitySchema
>;
export type NeuralAgentIdentity = z.infer<typeof NeuralAgentIdentitySchema>;
