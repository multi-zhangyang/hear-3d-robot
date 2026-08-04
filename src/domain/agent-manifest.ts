import { z } from "zod";

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);

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
  tool_name: z.string().trim().min(1),
  target_role: z.enum(["goal_manager", "sentry", "motion", "executor"]),
  target_agent_id: z.string().trim().min(1),
  target_agent_name: z.string().trim().min(1),
  tool_schema_sha256: Sha256Schema,
  input_builder_contract: z.enum([
    "objective_text_v1",
    "goal_manager_authority_envelope_v1",
    "validated_execution_task_json_v1"
  ]),
  input_builder_sha256: Sha256Schema,
  run_options: z.object({
    session_agent_id: z.string().trim().min(1),
    context_source: z.literal("parent_run_context"),
    max_turns: z.literal("sdk_default")
  }).strict(),
  resume_context_strategy: z.enum(["merge", "replace", "preferSerialized"]),
  include_input_schema: z.literal(false),
  needs_approval: z.literal(false),
  output_contract: z.literal("nested_agent_final_output_text")
}).strict();

const AgentModelSettingsIdentitySchema = z.object({
  request_timeout_ms: z.number().int().positive(),
  temperature: z.number().min(0).max(2),
  max_output_tokens: z.number().int().positive().optional(),
  context_window_tokens: z.number().int().positive(),
  compact_trigger_tokens: z.number().int().positive(),
  compact_recent_model_turns: z.number().int().nonnegative(),
  compact_max_output_tokens: z.number().int().positive().optional()
}).strict();

const AgentModelIdentitySchema = z.object({
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
    sentry: AgentModelIdentitySchema,
    motion: AgentModelIdentitySchema,
    executor: AgentModelIdentitySchema,
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
    if (contract.run_options.session_agent_id !== target.agent_id) {
      context.addIssue({
        code: "custom",
        path: ["agent_tool_contracts", index, "run_options", "session_agent_id"],
        message: "Agent-as-tool Session owner does not match its target role"
      });
    }
  }
});

export type AgentManifest = z.infer<typeof AgentManifestSchema>;
export type AgentModelIdentity = z.infer<typeof AgentModelIdentitySchema>;
