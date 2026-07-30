import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { mkdtemp, rm, utimes, writeFile } from "node:fs/promises";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { acquireOperatorLease, OperatorLeaseLostError } from "./operator-lease.js";

const FAST_LEASE = {
  heartbeatIntervalMs: 10,
  leaseDurationMs: 60,
  reclaimConfirmationMs: 20
};

describe("Operator runs-directory lease", () => {
  it("admits one owner and releases only its own lease", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-operator-lease-"));
    try {
      const first = await acquireOperatorLease(runsDir, FAST_LEASE);
      await expect(acquireOperatorLease(runsDir, FAST_LEASE)).rejects.toThrow("already owned");
      await first.release();
      const next = await acquireOperatorLease(runsDir, FAST_LEASE);
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 30));
      expect(first.signal.aborted).toBe(false);
      await expect(next.assertOwned()).resolves.toBeUndefined();
      await next.release();
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("reclaims a same-host lease after its process has ended", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-stale-operator-lease-"));
    const child = spawn(process.execPath, ["-e", ""], { stdio: "ignore" });
    const deadPid = child.pid;
    await once(child, "exit");
    if (!deadPid) throw new Error("Child process had no pid");
    await writeFile(join(runsDir, ".operator.lock"), `${JSON.stringify({
      token: randomUUID(),
      pid: deadPid,
      hostname: hostname(),
      started_at: "2026-07-30T00:00:00.000Z"
    })}\n`);
    try {
      const lease = await acquireOperatorLease(runsDir);
      await lease.release();
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("reclaims a renewable lease whose pid has been reused but heartbeat stopped", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-reused-pid-lease-"));
    const path = join(runsDir, ".operator.lock");
    await writeFile(path, `${JSON.stringify({
      version: 2,
      token: randomUUID(),
      pid: process.pid,
      hostname: hostname(),
      started_at: "2026-07-30T00:00:00.000Z",
      heartbeat_interval_ms: FAST_LEASE.heartbeatIntervalMs,
      lease_duration_ms: FAST_LEASE.leaseDurationMs
    })}\n`);
    const old = new Date("2000-01-01T00:00:00.000Z");
    await utimes(path, old, old);
    try {
      const lease = await acquireOperatorLease(runsDir, FAST_LEASE);
      expect(lease.signal.aborted).toBe(false);
      await lease.release();
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("does not reclaim an owner that renews while stale takeover is being confirmed", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-heartbeating-lease-"));
    const path = join(runsDir, ".operator.lock");
    const owner = await acquireOperatorLease(runsDir, FAST_LEASE);
    const old = new Date("2000-01-01T00:00:00.000Z");
    await utimes(path, old, old);
    try {
      await expect(acquireOperatorLease(runsDir, FAST_LEASE)).rejects.toThrow("already owned");
      await expect(owner.assertOwned()).resolves.toBeUndefined();
      expect(owner.signal.aborted).toBe(false);
    } finally {
      await owner.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("allows exactly one contender to reclaim the same stale lease", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-racing-operator-lease-"));
    const child = spawn(process.execPath, ["-e", ""], { stdio: "ignore" });
    const deadPid = child.pid;
    await once(child, "exit");
    if (!deadPid) throw new Error("Child process had no pid");
    await writeFile(join(runsDir, ".operator.lock"), `${JSON.stringify({
      token: randomUUID(),
      pid: deadPid,
      hostname: hostname(),
      started_at: "2026-07-30T00:00:00.000Z"
    })}\n`);
    try {
      const attempts = await Promise.allSettled(
        Array.from({ length: 12 }, () => acquireOperatorLease(runsDir))
      );
      const owners = attempts.filter(
        (attempt): attempt is PromiseFulfilledResult<Awaited<ReturnType<typeof acquireOperatorLease>>> =>
          attempt.status === "fulfilled"
      );
      expect(owners).toHaveLength(1);
      await owners[0]!.value.release();

      const next = await acquireOperatorLease(runsDir);
      await next.release();
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("never removes a replacement lease during release", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-replaced-operator-lease-"));
    const first = await acquireOperatorLease(runsDir);
    const replacement = {
      token: randomUUID(),
      pid: process.pid,
      hostname: hostname(),
      started_at: new Date().toISOString()
    };
    await writeFile(
      join(runsDir, ".operator.lock"),
      `${JSON.stringify(replacement)}\n`,
      "utf8"
    );
    try {
      await first.release();
      await expect(acquireOperatorLease(runsDir)).rejects.toThrow("already owned");
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("fences an owner whose token was replaced and preserves the replacement", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-lost-operator-lease-"));
    const first = await acquireOperatorLease(runsDir, FAST_LEASE);
    const replacement = {
      version: 2,
      token: randomUUID(),
      pid: process.pid,
      hostname: hostname(),
      started_at: new Date().toISOString(),
      heartbeat_interval_ms: FAST_LEASE.heartbeatIntervalMs,
      lease_duration_ms: FAST_LEASE.leaseDurationMs
    };
    await writeFile(
      join(runsDir, ".operator.lock"),
      `${JSON.stringify(replacement)}\n`,
      "utf8"
    );
    try {
      let ownershipFailure: unknown;
      try {
        await first.assertOwned();
      } catch (error) {
        ownershipFailure = error;
      }
      expect(ownershipFailure).toBeInstanceOf(OperatorLeaseLostError);
      expect((ownershipFailure as Error).message).toMatch(
        /^(?:Operator lease token was replaced|Operator lease token changed during renewal)$/
      );
      expect(first.signal.aborted).toBe(true);
      expect(first.signal.reason).toBe(ownershipFailure);
      await first.release();
      await expect(acquireOperatorLease(runsDir, FAST_LEASE)).rejects.toThrow("already owned");
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("reclaims only aged malformed lease artifacts", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-malformed-operator-lease-"));
    const path = join(runsDir, ".operator.lock");
    await writeFile(path, "{\"interrupted\":", "utf8");
    try {
      await expect(acquireOperatorLease(runsDir)).rejects.toThrow("initializing or unreadable");
      const old = new Date("2000-01-01T00:00:00.000Z");
      await utimes(path, old, old);
      const recovered = await acquireOperatorLease(runsDir);
      await recovered.release();
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });
});
