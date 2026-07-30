import { cp, mkdir, rm } from "node:fs/promises";
import { resolve } from "node:path";

// Kept out of `test-results/`, which Playwright wipes as the run starts.
export const E2E_RUNS_DIR = resolve(process.cwd(), ".e2e-runs");

const FIXTURE_RUNS_DIR = resolve(
  process.env.HEAR_E2E_RUNS_SOURCE ?? resolve(process.cwd(), "tests", "fixtures", "runs")
);

export default async function globalSetup(): Promise<void> {
  await rm(E2E_RUNS_DIR, { recursive: true, force: true });
  await mkdir(E2E_RUNS_DIR, { recursive: true });
  await cp(FIXTURE_RUNS_DIR, E2E_RUNS_DIR, { recursive: true });
}
