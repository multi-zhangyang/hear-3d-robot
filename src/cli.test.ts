import { randomUUID } from "node:crypto";
import { link, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { hostname, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import { isMainModule, withMissionSignals } from "./cli.js";

const FAST_LEASE = {
  heartbeatIntervalMs: 10,
  leaseDurationMs: 60,
  reclaimConfirmationMs: 20
};

describe("CLI mission lease fencing", () => {
  it("recognizes a packaged CLI launched through a bin symlink", async () => {
    const modulePath = fileURLToPath(new URL("./cli.ts", import.meta.url));
    const directory = await mkdtemp(join(
      process.platform === "win32" ? dirname(modulePath) : tmpdir(),
      "hear-cli-symlink-"
    ));
    const binPath = join(directory, "hear");
    try {
      try {
        await symlink(modulePath, binPath, "file");
      } catch (error) {
        if (!isWindowsSymlinkPrivilegeError(error)) throw error;
        await link(modulePath, binPath);
      }
      await expect(isMainModule(binPath, pathToFileURL(modulePath).href)).resolves.toBe(true);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("aborts a running CLI mission when its lease token is replaced", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-cli-lease-"));
    const lockPath = join(runsDir, ".operator.lock");
    const replacement = {
      version: 2,
      token: randomUUID(),
      pid: process.pid,
      hostname: hostname(),
      started_at: new Date().toISOString(),
      heartbeat_interval_ms: FAST_LEASE.heartbeatIntervalMs,
      lease_duration_ms: FAST_LEASE.leaseDurationMs
    };
    let missionSignal: AbortSignal | undefined;
    try {
      const running = withMissionSignals(runsDir, async (signal) => {
        missionSignal = signal;
        await writeFile(lockPath, `${JSON.stringify(replacement)}\n`, "utf8");
        await aborted(signal);
        signal.throwIfAborted();
      }, FAST_LEASE);

      await expect(running).rejects.toThrow(/Operator lease/);
      expect(missionSignal?.aborted).toBe(true);
      expect(JSON.parse(await readFile(lockPath, "utf8"))).toMatchObject({
        token: replacement.token
      });
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });
});

function isWindowsSymlinkPrivilegeError(error: unknown): boolean {
  return process.platform === "win32"
    && error instanceof Error
    && "code" in error
    && error.code === "EPERM";
}

function aborted(signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.resolve();
  return new Promise((resolveAbort, rejectAbort) => {
    const timer = setTimeout(() => rejectAbort(new Error("Lease signal did not abort")), 2_000);
    timer.unref();
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      resolveAbort();
    }, { once: true });
  });
}
