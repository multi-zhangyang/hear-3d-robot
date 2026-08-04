#!/usr/bin/env node
import { realpath } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  loadEnvironment,
  loadProviderConfig,
  loadRuntimeCatalog,
  loadServerConfig,
  type ProviderConfig
} from "./config/load.js";
import { GoalSchema } from "./domain/schema.js";
import { HumanoidRunModeSchema } from "./domain/run-mode.js";
import type { MutationFence } from "./persistence/mutation-fence.js";
import { resolveRunDirectory } from "./persistence/run-store.js";
import {
  resumeHumanoidMission,
  startHumanoidMission
} from "./runtime/humanoid-mission-runner.js";
import { errorMessage } from "./runtime/error-message.js";
import { RunPauseRequestedError } from "./runtime/run-pause.js";
import {
  acquireOperatorLease,
  type OperatorLeaseOptions
} from "./server/operator-lease.js";
import { createOperatorServer } from "./server/operator-server.js";

loadEnvironment();

if (await isMainModule(process.argv[1], import.meta.url)) {
  await main(process.argv.slice(2));
}

export async function isMainModule(
  entryPath: string | undefined,
  moduleUrl: string
): Promise<boolean> {
  if (!entryPath) return false;
  try {
    const [entryRealPath, moduleRealPath] = await Promise.all([
      realpath(entryPath),
      realpath(fileURLToPath(moduleUrl))
    ]);
    return entryRealPath === moduleRealPath;
  } catch {
    return moduleUrl === pathToFileURL(entryPath).href;
  }
}

async function main(argv: string[]): Promise<void> {
  const command = argv[0] ?? "help";
  const options = parseOptions(argv.slice(1));
  try {
    if (command === "scenarios") {
      const catalog = await loadRuntimeCatalog();
      for (const [id, template] of Object.entries(catalog.templates)) {
        const shape = template.kind === "procedural"
          ? `procedural ${template.generate.bounds.width}x${template.generate.bounds.depth}`
          : "authored";
        process.stdout.write(`${id}\t${template.title}\t${shape}\n`);
      }
      return;
    }
    if (command === "run") {
      requireConfirmation(options);
      const mission = required(options, "mission");
      const scenarioId = required(options, "scenario");
      const goal = GoalSchema.parse(JSON.parse(required(options, "goal")));
      const seed = options.seed === undefined ? undefined : parseSeed(options.seed);
      const runMode = HumanoidRunModeSchema.parse(options.mode ?? "mission");
      const [catalog, provider, server] = await Promise.all([
        loadRuntimeCatalog(),
        Promise.resolve(loadProviderConfig()),
        Promise.resolve(loadServerConfig())
      ]);
      const result = await withMissionSignals(server.runsDir, (signal, mutationFence) => startHumanoidMission({
        runsDir: server.runsDir,
        mission,
        scenarioId,
        goal,
        catalog,
        provider,
        runMode,
        ...(seed === undefined ? {} : { seed }),
        signal,
        mutationFence
      }));
      process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
      return;
    }
    if (command === "resume") {
      requireConfirmation(options);
      const runId = required(options, "run");
      const [catalog, provider, server] = await Promise.all([
        loadRuntimeCatalog(),
        Promise.resolve(loadProviderConfig()),
        Promise.resolve(loadServerConfig())
      ]);
      const result = await withMissionSignals(server.runsDir, (signal, mutationFence) => resumeHumanoidMission({
        runDir: resolveRunDirectory(server.runsDir, runId),
        catalog,
        provider,
        signal,
        mutationFence
      }));
      process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
      return;
    }
    if (command === "operator") {
      const [catalog, server] = await Promise.all([
        loadRuntimeCatalog(),
        Promise.resolve(loadServerConfig())
      ]);
      const host = options.host ?? server.host;
      const port = options.port ? Number(options.port) : server.port;
      if (!Number.isInteger(port) || port < 1 || port > 65_535) {
        throw new Error("--port must be an integer between 1 and 65535");
      }
      let provider: ProviderConfig | undefined;
      let providerError: string | undefined;
      try {
        provider = loadProviderConfig();
      } catch (error) {
        providerError = errorMessage(error);
      }
      const app = await createOperatorServer({
        server: { ...server, host, port },
        catalog,
        ...(provider ? { provider } : {}),
        ...(providerError ? { providerError } : {}),
        dev: options.dev === "true"
      });
      installOperatorSignals(app);
      await app.listen({ host, port });
      return;
    }
    printHelp();
    if (command !== "help" && command !== "--help" && command !== "-h") process.exitCode = 1;
  } catch (error) {
    process.stderr.write(`${errorMessage(error)}\n`);
    process.exitCode = 1;
  }
}

function installOperatorSignals(app: Awaited<ReturnType<typeof createOperatorServer>>): void {
  let closing = false;
  const close = (): void => {
    if (closing) return;
    closing = true;
    void app.close().catch((error) => {
      process.stderr.write(`${errorMessage(error)}\n`);
      process.exitCode = 1;
    });
  };
  process.on("SIGINT", close);
  process.on("SIGTERM", close);
  app.addHook("onClose", async () => {
    process.off("SIGINT", close);
    process.off("SIGTERM", close);
  });
}

function parseOptions(args: string[]): Record<string, string> {
  const options: Record<string, string> = {};
  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];
    if (!token?.startsWith("--")) throw new Error(`Unexpected argument: ${token ?? ""}`);
    const name = token.slice(2);
    const next = args[index + 1];
    if (!next || next.startsWith("--")) {
      options[name] = "true";
    } else {
      options[name] = next;
      index += 1;
    }
  }
  return options;
}

function required(options: Record<string, string>, name: string): string {
  const value = options[name]?.trim();
  if (!value) throw new Error(`--${name} is required`);
  return value;
}

/** A world seed, so a generated scenario can be rebuilt deliberately. */
function parseSeed(value: string): number {
  const seed = Number(value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffff_ffff) {
    throw new Error("--seed must be an integer between 0 and 4294967295");
  }
  return seed;
}

function requireConfirmation(options: Record<string, string>): void {
  if (options.confirm !== "true") throw new Error("--confirm is required");
}

export async function withMissionSignals<T>(
  runsDir: string,
  operation: (signal: AbortSignal, mutationFence: MutationFence) => Promise<T>,
  leaseOptions?: OperatorLeaseOptions
): Promise<T> {
  const lease = await acquireOperatorLease(runsDir, leaseOptions);
  const controller = new AbortController();
  const signal = AbortSignal.any([controller.signal, lease.signal]);
  const interrupt = (): void => controller.abort(
    new RunPauseRequestedError("Mission paused by process signal")
  );
  process.on("SIGINT", interrupt);
  process.on("SIGTERM", interrupt);
  try {
    await lease.assertOwned();
    return await operation(signal, lease);
  } finally {
    process.off("SIGINT", interrupt);
    process.off("SIGTERM", interrupt);
    await lease.release();
  }
}

function printHelp(): void {
  process.stdout.write([
    "HEAR",
    "",
    "  hear scenarios",
    "  hear run --scenario ID --mission TEXT --goal JSON [--mode mission|continuous] [--seed N] --confirm",
    "  hear resume --run RUN_ID --confirm",
    "  hear operator [--host HOST] [--port PORT] [--dev]",
    ""
  ].join("\n"));
}
