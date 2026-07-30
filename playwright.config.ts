import { defineConfig } from "@playwright/test";
import { E2E_RUNS_DIR } from "./tests/browser/e2e-runs.js";

const port = Number(process.env.HEAR_E2E_PORT ?? 8877);
const operatorPassword = process.env.HEAR_E2E_PASSWORD ?? "hear-e2e-local";

export default defineConfig({
  testDir: "./tests/browser",
  // SwiftShader startup and the first 80×80 voxel upload can be noticeably
  // slower on a cold shared CI runner than on a warm development machine.
  // Keep every rendering assertion, but give the full production WebGL path a
  // realistic ceiling instead of turning runner contention into a false fail.
  timeout: 120_000,
  expect: { timeout: 12_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
    video: "retain-on-failure",
    launchOptions: {
      args: ["--use-gl=angle", "--use-angle=swiftshader"]
    }
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 960 } } },
    {
      name: "mobile",
      use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true }
    }
  ],
  webServer: {
    command: "pnpm exec tsx tests/browser/start-operator.ts",
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
    // The spec asserts against a recorded run, so it must not read — or write to —
    // whatever happens to be in the developer's own runs directory.
    env: {
      HEAR_RUNS_DIR: E2E_RUNS_DIR,
      HEAR_HOST: "127.0.0.1",
      HEAR_PORT: String(port),
      HEAR_OPERATOR_PASSWORD: operatorPassword,
      AI_PROVIDER: "",
      AI_BASE_URL: "",
      AI_MODEL: "",
      AI_API_KEY: ""
    }
  }
});
