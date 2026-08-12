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
  dispatch_kind: z.enum(["model_agent", "deterministic_service"]).optional(),
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

export const AgentManifestSchema = z.object({
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

export type AgentManifest = z.infer<typeof AgentManifestSchema>;
export type AgentModelIdentity = z.infer<typeof AgentModelIdentitySchema>;
export type AgentDeterministicServiceIdentity = z.infer<
  typeof AgentDeterministicServiceIdentitySchema
>;
