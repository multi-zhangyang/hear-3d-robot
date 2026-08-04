import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadDotenv } from "dotenv";
import { parse as parseYaml } from "yaml";
import { z } from "zod";
import {
  ScenarioCatalogSchema,
  type Scenario,
  type ScenarioTemplate
} from "../domain/schema.js";
import {
  compactorInputTokenLimit,
  defaultCompactTriggerTokens,
  defaultOutputTokenReserve
} from "../runtime/context-budget.js";
import { assertScenarioIntegrity } from "../runtime/goal-validation.js";
import { assertHumanoidGoalSupported } from "../runtime/humanoid-checker.js";
import { materializeScenario } from "../world/world-generator.js";

const bundledConfigDirectory = fileURLToPath(new URL("../../config/", import.meta.url));

export const AGENT_MODEL_ROLES = [
  "goal_manager",
  "coordinator",
  "sentry",
  "motion",
  "executor",
  "compactor"
] as const;

export type AgentModelRole = typeof AGENT_MODEL_ROLES[number];

const ProviderConfigShape = {
  protocol: z.enum([
    "openai_compatible",
    "openai_responses",
    "anthropic_messages"
  ]),
  baseUrl: z.url(),
  model: z.string().min(1),
  apiKey: z.string().min(1),
  requestTimeoutMs: z.number().int().min(5_000).max(10 * 60_000).optional(),
  temperature: z.number().min(0).max(2),
  maxOutputTokens: z.number().int().positive().optional(),
  contextWindowTokens: z.number().int().positive(),
  compactTriggerTokens: z.number().int().positive(),
  compactRecentModelTurns: z.number().int().nonnegative(),
  compactMaxOutputTokens: z.number().int().positive().optional()
} as const;

const ModelProviderConfigSchema = z.object(ProviderConfigShape).strict().superRefine(
  validateProviderBudget
);

const ProviderConfigSchema = z.object({
  ...ProviderConfigShape,
  agentModels: z.object({
    goal_manager: ModelProviderConfigSchema,
    coordinator: ModelProviderConfigSchema,
    sentry: ModelProviderConfigSchema,
    motion: ModelProviderConfigSchema,
    executor: ModelProviderConfigSchema,
    compactor: ModelProviderConfigSchema
  }).strict().optional()
}).strict().superRefine(validateProviderBudget);

function validateProviderBudget(
  config: z.infer<typeof ModelProviderConfigSchema>,
  context: z.RefinementCtx
): void {
  const fallbackReserve = defaultOutputTokenReserve(config.contextWindowTokens);
  const exceedsWindow = config.compactTriggerTokens >= config.contextWindowTokens;
  const exceedsConfiguredOutputHeadroom = config.maxOutputTokens !== undefined
    && config.compactTriggerTokens + config.maxOutputTokens >= config.contextWindowTokens;
  if (exceedsWindow || exceedsConfiguredOutputHeadroom) {
    context.addIssue({
      code: "custom",
      path: ["compactTriggerTokens"],
      message: "AI_COMPACT_TRIGGER_TOKENS must leave room for any explicitly configured AI_MAX_OUTPUT_TOKENS"
    });
  }
  if (compactorInputTokenLimit(
    config.contextWindowTokens,
    config.compactMaxOutputTokens ?? fallbackReserve
  ) <= 0) {
    context.addIssue({
      code: "custom",
      path: ["compactMaxOutputTokens"],
      message: "AI_COMPACT_MAX_OUTPUT_TOKENS leaves no room for bounded compactor repair turns"
    });
  }
}

export type ModelProviderConfig = z.infer<typeof ModelProviderConfigSchema>;
export type ProviderConfig = z.infer<typeof ProviderConfigSchema>;

/**
 * The scenario catalog holds templates, not worlds.
 *
 * A generated template has no positions in it at all — the terrain, the robot's
 * pose and stance, and every entity placement come from the run's seed. So the
 * catalog can only hand out a world once a seed exists, and a run stores the
 * seed it was given rather than the world it produced.
 */
export interface RuntimeCatalog {
  templates: Record<string, ScenarioTemplate>;
  materialize(scenarioId: string, seed: number): Scenario;
}

export interface ServerConfig {
  host: string;
  port: number;
  password: string;
  runsDir: string;
}

export function loadEnvironment(path = resolve(process.cwd(), ".env")): void {
  loadDotenv({ path, quiet: true });
}

export function loadProviderConfig(env: NodeJS.ProcessEnv = process.env): ProviderConfig {
  const contextWindowTokens = numberFromEnv(env.AI_CONTEXT_WINDOW_TOKENS, 262_144);
  const maxOutputTokens = optionalNumberFromEnv(env.AI_MAX_OUTPUT_TOKENS);
  const compactMaxOutputTokens = optionalNumberFromEnv(env.AI_COMPACT_MAX_OUTPUT_TOKENS);
  const inherited = ModelProviderConfigSchema.parse({
    protocol: env.AI_PROVIDER,
    baseUrl: env.AI_BASE_URL,
    model: env.AI_MODEL,
    apiKey: env.AI_API_KEY,
    requestTimeoutMs: numberFromEnv(env.AI_REQUEST_TIMEOUT_MS, 90_000),
    temperature: numberFromEnv(env.AI_TEMPERATURE, 0.2),
    maxOutputTokens,
    contextWindowTokens,
    compactTriggerTokens: numberFromEnv(
      env.AI_COMPACT_TRIGGER_TOKENS,
      defaultCompactTriggerTokens(contextWindowTokens)
    ),
    compactRecentModelTurns: numberFromEnv(env.AI_COMPACT_RECENT_MODEL_TURNS, 4),
    compactMaxOutputTokens
  });
  const agentModels = Object.fromEntries(AGENT_MODEL_ROLES.map((role) => [
    role,
    loadAgentModelConfig(env, role, inherited)
  ])) as Record<AgentModelRole, ModelProviderConfig>;
  return ProviderConfigSchema.parse({ ...inherited, agentModels });
}

export function providerConfigForRole(
  config: ProviderConfig,
  role: AgentModelRole
): ModelProviderConfig {
  const selected = config.agentModels?.[role];
  if (selected) return ModelProviderConfigSchema.parse(selected);
  const { agentModels: _agentModels, ...inherited } = config;
  return ModelProviderConfigSchema.parse(inherited);
}

export function loadServerConfig(env: NodeJS.ProcessEnv = process.env): ServerConfig {
  const port = numberFromEnv(env.HEAR_PORT, 8765);
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error("HEAR_PORT must be an integer between 1 and 65535");
  }
  const host = env.HEAR_HOST?.trim() || "127.0.0.1";
  const password = env.HEAR_OPERATOR_PASSWORD?.trim() || "";
  if (!password && !isLoopbackHost(host)) {
    throw new Error(
      "HEAR_OPERATOR_PASSWORD is required when HEAR_HOST is not a loopback address"
    );
  }
  return {
    host,
    port,
    password,
    runsDir: resolve(env.HEAR_RUNS_DIR?.trim() || "runs")
  };
}

export async function loadRuntimeCatalog(
  configDir = bundledConfigDirectory
): Promise<RuntimeCatalog> {
  const scenariosText = await readFile(resolve(configDir, "scenarios.yaml"), "utf8");
  const templates = ScenarioCatalogSchema.parse(parseYaml(scenariosText)).scenarios;
  // Validating a template means validating a world built from it: the integrity
  // rules are about positions and references, and a generated template only has
  // the latter until a seed fills in the former. Seed 0 is as good as any and
  // costs one generation per scenario at startup, which catches an impossible
  // template — one whose entities cannot fit the terrain it asks for — here
  // rather than on the operator's first run.
  for (const [scenarioId, template] of Object.entries(templates)) {
    const scenario = materializeScenario(template, 0);
    assertScenarioIntegrity(scenarioId, scenario);
    assertHumanoidGoalSupported(scenario.default_goal, scenario);
  }
  return {
    templates,
    materialize(scenarioId, seed) {
      const template = templates[scenarioId];
      if (!template) throw new Error(`Unknown scenario: ${scenarioId}`);
      const scenario = materializeScenario(template, seed);
      assertScenarioIntegrity(scenarioId, scenario);
      assertHumanoidGoalSupported(scenario.default_goal, scenario);
      return scenario;
    }
  };
}

function numberFromEnv(value: string | undefined, defaultValue: number): number {
  if (value === undefined || value.trim() === "") return defaultValue;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`Invalid numeric environment value: ${value}`);
  return parsed;
}

function optionalNumberFromEnv(
  value: string | undefined,
  inherited?: number
): number | undefined {
  if (value === undefined || value.trim() === "") return inherited;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`Invalid numeric environment value: ${value}`);
  return parsed;
}

function loadAgentModelConfig(
  env: NodeJS.ProcessEnv,
  role: AgentModelRole,
  inherited: ModelProviderConfig
): ModelProviderConfig {
  const prefix = `AI_${role.toUpperCase()}_`;
  const contextWindowTokens = numberFromEnv(
    env[`${prefix}CONTEXT_WINDOW_TOKENS`],
    inherited.contextWindowTokens
  );
  const maxOutputTokens = optionalNumberFromEnv(
    env[`${prefix}MAX_OUTPUT_TOKENS`],
    inherited.maxOutputTokens
  );
  const inheritedTriggerWasExplicit = hasText(env.AI_COMPACT_TRIGGER_TOKENS);
  return ModelProviderConfigSchema.parse({
    protocol: stringFromEnv(env[`${prefix}PROVIDER`], inherited.protocol),
    baseUrl: stringFromEnv(env[`${prefix}BASE_URL`], inherited.baseUrl),
    model: stringFromEnv(env[`${prefix}MODEL`], inherited.model),
    apiKey: stringFromEnv(env[`${prefix}API_KEY`], inherited.apiKey),
    requestTimeoutMs: numberFromEnv(
      env[`${prefix}REQUEST_TIMEOUT_MS`],
      inherited.requestTimeoutMs ?? 90_000
    ),
    temperature: numberFromEnv(env[`${prefix}TEMPERATURE`], inherited.temperature),
    maxOutputTokens,
    contextWindowTokens,
    compactTriggerTokens: numberFromEnv(
      env[`${prefix}COMPACT_TRIGGER_TOKENS`],
      inheritedTriggerWasExplicit
        ? inherited.compactTriggerTokens
        : defaultCompactTriggerTokens(contextWindowTokens)
    ),
    compactRecentModelTurns: numberFromEnv(
      env[`${prefix}COMPACT_RECENT_MODEL_TURNS`],
      inherited.compactRecentModelTurns
    ),
    compactMaxOutputTokens: optionalNumberFromEnv(
      env[`${prefix}COMPACT_MAX_OUTPUT_TOKENS`],
      inherited.compactMaxOutputTokens
    )
  });
}

function stringFromEnv(value: string | undefined, inherited: string): string {
  return value?.trim() ? value : inherited;
}

function hasText(value: string | undefined): boolean {
  return value !== undefined && value.trim() !== "";
}

function isLoopbackHost(host: string): boolean {
  const normalized = host.trim().toLowerCase();
  if (normalized === "localhost" || normalized === "::1") return true;
  const match = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(normalized);
  if (!match) return false;
  const octets = match.slice(1).map(Number);
  return octets.every((octet) => Number.isInteger(octet) && octet >= 0 && octet <= 255)
    && octets[0] === 127;
}
