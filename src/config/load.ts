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
import { compactorInputTokenLimit } from "../runtime/context-budget.js";
import { assertScenarioIntegrity } from "../runtime/goal-validation.js";
import { materializeScenario } from "../world/world-generator.js";

const bundledConfigDirectory = fileURLToPath(new URL("../../config/", import.meta.url));

const ProviderConfigSchema = z.object({
  protocol: z.enum([
    "openai_compatible",
    "openai_responses",
    "anthropic_messages"
  ]),
  baseUrl: z.url(),
  model: z.string().min(1),
  apiKey: z.string().min(1),
  temperature: z.number().min(0).max(2),
  maxOutputTokens: z.number().int().positive(),
  contextWindowTokens: z.number().int().positive(),
  compactTriggerTokens: z.number().int().positive(),
  compactRecentModelTurns: z.number().int().nonnegative(),
  compactMaxOutputTokens: z.number().int().positive()
}).superRefine((config, context) => {
  const reserved = config.maxOutputTokens + config.compactMaxOutputTokens;
  if (config.compactTriggerTokens + reserved >= config.contextWindowTokens) {
    context.addIssue({
      code: "custom",
      path: ["compactTriggerTokens"],
      message: "AI_COMPACT_TRIGGER_TOKENS plus output reserves must be below AI_CONTEXT_WINDOW_TOKENS"
    });
  }
  if (compactorInputTokenLimit(
    config.contextWindowTokens,
    config.compactMaxOutputTokens
  ) <= 0) {
    context.addIssue({
      code: "custom",
      path: ["compactMaxOutputTokens"],
      message: "AI_COMPACT_MAX_OUTPUT_TOKENS leaves no room for bounded compactor repair turns"
    });
  }
});

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
  materialize(scenarioId: string, seed: number, motionSeed?: number): Scenario;
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
  const contextWindowTokens = numberFromEnv(env.AI_CONTEXT_WINDOW_TOKENS, 65_536);
  return ProviderConfigSchema.parse({
    protocol: env.AI_PROVIDER,
    baseUrl: env.AI_BASE_URL,
    model: env.AI_MODEL,
    apiKey: env.AI_API_KEY,
    temperature: numberFromEnv(env.AI_TEMPERATURE, 0.2),
    // Bound unproductive generations while leaving enough room for model
    // reasoning and the resulting tool call, both of which consume this budget.
    maxOutputTokens: numberFromEnv(env.AI_MAX_OUTPUT_TOKENS, 8192),
    contextWindowTokens,
    compactTriggerTokens: numberFromEnv(
      env.AI_COMPACT_TRIGGER_TOKENS,
      // The advertised context window is a hard ceiling, not a useful working
      // set. Long tool histories can make planning quality collapse well
      // before that ceiling, so compact around 27.5% for the default 65k
      // window while still scaling down for smaller models.
      Math.min(18_000, Math.floor(contextWindowTokens * 0.4))
    ),
    compactRecentModelTurns: numberFromEnv(env.AI_COMPACT_RECENT_MODEL_TURNS, 4),
    // Compaction is itself a reasoning/tool-call run. A 2k ceiling can be
    // exhausted before a reasoning model emits the required typed checkpoint,
    // so keep enough output room for the decision while still validating the
    // complete repair-turn envelope against the configured context window.
    compactMaxOutputTokens: numberFromEnv(env.AI_COMPACT_MAX_OUTPUT_TOKENS, 4096)
  });
}

export function loadServerConfig(env: NodeJS.ProcessEnv = process.env): ServerConfig {
  const port = numberFromEnv(env.HEAR_PORT, 8765);
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error("HEAR_PORT must be an integer between 1 and 65535");
  }
  return {
    host: env.HEAR_HOST?.trim() || "127.0.0.1",
    port,
    password: env.HEAR_OPERATOR_PASSWORD?.trim() || "",
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
    assertScenarioIntegrity(scenarioId, materializeScenario(template, 0, 0));
  }
  return {
    templates,
    materialize(scenarioId, seed, motionSeed) {
      const template = templates[scenarioId];
      if (!template) throw new Error(`Unknown scenario: ${scenarioId}`);
      const scenario = materializeScenario(template, seed, motionSeed);
      assertScenarioIntegrity(scenarioId, scenario);
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
