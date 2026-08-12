import { randomUUID } from "node:crypto";
import {
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  writeFile
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import {
  createHarnessSkillRolloutRecords,
  HarnessSkillRolloutRecordSchema,
  type HarnessSkillRolloutRecord
} from "./harness-rollout-dataset.js";
import { loadDensePolicyRolloutReference } from
  "./dense-policy-rollout-files.js";

export interface HarnessRolloutDataset {
  runs_path: string;
  run_count: number;
  records: HarnessSkillRolloutRecord[];
}

export interface HarnessRolloutDatasetSummary {
  protocol: "hear-harness-rollout-export-v1";
  runs_path: string;
  output_path: string;
  run_count: number;
  record_count: number;
  outcome_counts: Record<string, number>;
  dense_policy_rollout_count: number;
}

export async function loadHarnessRolloutDataset(
  runsPath: string,
  options: { denseRolloutsPath?: string } = {}
): Promise<HarnessRolloutDataset> {
  const root = resolve(runsPath);
  const runDirectories = await discoverRunDirectories(root);
  const records = (await Promise.all(runDirectories.map(async (runDir) => {
    const [run, actions, events] = await Promise.all([
      readJson(resolve(runDir, "run.json")),
      readJsonLines(resolve(runDir, "actions.jsonl")),
      readJsonLines(resolve(runDir, "events.jsonl"))
    ]);
    const records = createHarnessSkillRolloutRecords({ run, actions, events });
    if (!options.denseRolloutsPath) return records;
    return Promise.all(records.map(async (record) => {
      const dense = await loadDensePolicyRolloutReference({
        rootDir: options.denseRolloutsPath!,
        runId: record.source.run_id,
        callId: record.identity.callId,
        execution: record.execution
          ? {
              frameCount: record.execution.frame_count,
              worldBeforeRevision: record.execution.world_before_revision,
              worldAfterRevision: record.execution.world_after_revision
            }
          : null
      });
      return dense
        ? HarnessSkillRolloutRecordSchema.parse({
            ...record,
            dense_policy_rollout: dense
          })
        : record;
    }));
  }))).flat().sort((left, right) => compareCodePoints(
    left.record_id,
    right.record_id
  ));
  return {
    runs_path: root,
    run_count: runDirectories.length,
    records
  };
}

export async function writeHarnessRolloutDataset(
  outputPath: string,
  dataset: HarnessRolloutDataset
): Promise<HarnessRolloutDatasetSummary> {
  const destination = resolve(outputPath);
  const records = dataset.records.map((record) => (
    HarnessSkillRolloutRecordSchema.parse(record)
  ));
  const temporary = `${destination}.${randomUUID()}.tmp`;
  await mkdir(dirname(destination), { recursive: true });
  try {
    await writeFile(
      temporary,
      records.map((record) => JSON.stringify(record)).join("\n")
        + (records.length > 0 ? "\n" : ""),
      "utf8"
    );
    await rename(temporary, destination);
  } finally {
    await rm(temporary, { force: true });
  }
  return {
    protocol: "hear-harness-rollout-export-v1",
    runs_path: dataset.runs_path,
    output_path: destination,
    run_count: dataset.run_count,
    record_count: records.length,
    outcome_counts: countOutcomes(records),
    dense_policy_rollout_count: records.filter(
      ({ dense_policy_rollout }) => dense_policy_rollout.available
    ).length
  };
}

async function discoverRunDirectories(root: string): Promise<string[]> {
  if (await exists(resolve(root, "run.json"))) return [root];
  const entries = await readdir(root, { withFileTypes: true });
  const directories = entries.filter((entry) => (
    entry.isDirectory() && !entry.name.startsWith(".")
  )).map((entry) => resolve(root, entry.name)).sort(compareCodePoints);
  if (directories.length === 0) {
    throw new Error(`Harness rollout export found no run directories in ${root}`);
  }
  return directories;
}

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await readFile(path, "utf8")) as unknown;
}

async function readJsonLines(
  path: string
): Promise<Array<{ index: number; value: unknown }>> {
  let content: string;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if (isMissing(error)) return [];
    throw error;
  }
  return content.split(/\r?\n/).filter(Boolean).map((line, index) => ({
    index,
    value: JSON.parse(line) as unknown
  }));
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

function countOutcomes(
  records: readonly HarnessSkillRolloutRecord[]
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const record of records) {
    counts[record.outcome.category] = (counts[record.outcome.category] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => (
    compareCodePoints(left, right)
  )));
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
