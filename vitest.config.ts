import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts", "web/src/**/*.test.ts"],
    environment: "node",
    // Several suites intentionally exercise real MuJoCo, fsync, subprocess
    // heartbeats, and lease takeover deadlines. Running those files beside
    // CPU-heavy simulation suites changes the timing behavior being tested.
    fileParallelism: false,
    coverage: {
      reporter: ["text", "json-summary"]
    }
  }
});
