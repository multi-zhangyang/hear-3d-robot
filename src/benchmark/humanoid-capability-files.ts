import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import {
  createHumanoidCapabilityBenchmarkReport,
  measureHumanoidCapabilityRun,
  type HumanoidCapabilityBenchmarkReport
} from "./humanoid-capability.js";
import {
  addModelUsage,
  EmptyModelUsageState,
  modelUsageDeltaFromProviderEvent,
  ModelUsageStateSchema,
  type ModelUsageState
} from "../domain/model-usage.js";

export async function loadHumanoidCapabilityBenchmark(
  runsPath: string,
  generatedAt = new Date().toISOString()
): Promise<HumanoidCapabilityBenchmarkReport> {
  const runDirectories = await discoverRunDirectories(resolve(runsPath));
  const runs = await Promise.all(runDirectories.map(async (runDir) => {
    const [definition, checkpoint, actions, events, providerEvents] = await Promise.all([
      readJson(resolve(runDir, "run.json")),
      readJson(resolve(runDir, "checkpoint.json")),
      readJsonLines(resolve(runDir, "actions.jsonl")),
      readJsonLines(resolve(runDir, "events.jsonl")),
      readJsonLines(resolve(runDir, "provider.jsonl"))
    ]);
    const modelUsage = recoverModelUsage(providerEvents);
    return measureHumanoidCapabilityRun({
      definition,
      checkpoint,
      actions,
      events,
      ...(modelUsage ? { modelUsage } : {})
    });
  }));
  return createHumanoidCapabilityBenchmarkReport(runs, generatedAt);
}

export async function writeHumanoidCapabilityBenchmark(
  outputPath: string,
  report: HumanoidCapabilityBenchmarkReport
): Promise<void> {
  const destination = resolve(outputPath);
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

async function discoverRunDirectories(root: string): Promise<string[]> {
  if (await exists(resolve(root, "run.json"))) return [root];
  const entries = await readdir(root, { withFileTypes: true });
  const directories = entries.filter((entry) => (
    entry.isDirectory() && !entry.name.startsWith(".")
  )).map((entry) => resolve(root, entry.name)).sort(compareCodePoints);
  if (directories.length === 0) {
    throw new Error(`Capability benchmark found no run directories in ${root}`);
  }
  return directories;
}

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await readFile(path, "utf8")) as unknown;
}

async function readJsonLines(path: string): Promise<unknown[]> {
  try {
    return (await readFile(path, "utf8")).split(/\r?\n/).filter(Boolean).map(
      (line) => JSON.parse(line) as unknown
    );
  } catch (error) {
    if (isMissing(error)) return [];
    throw error;
  }
}

function recoverModelUsage(events: readonly unknown[]): ModelUsageState | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = record(events[index]);
    if (event && "model_usage" in event) {
      return ModelUsageStateSchema.parse(event.model_usage);
    }
  }
  let recovered = EmptyModelUsageState;
  let available = false;
  for (const event of events) {
    const delta = modelUsageDeltaFromProviderEvent(event);
    if (!delta) continue;
    available = true;
    recovered = addModelUsage(recovered, delta, eventTime(event));
  }
  return available ? recovered : undefined;
}

function eventTime(value: unknown): string {
  const at = record(value)?.at;
  return typeof at === "string" && Number.isFinite(Date.parse(at))
    ? at
    : new Date(0).toISOString();
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

async function exists(path: string): Promise<boolean> {
  try {
    await readFile(path);
    return true;
  } catch (error) {
    if (isMissing(error)) return false;
    throw error;
  }
}

function isMissing(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
