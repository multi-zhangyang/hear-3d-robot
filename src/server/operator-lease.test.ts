import { fork, spawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { cp, mkdir, mkdtemp, readFile, rm, stat, utimes, writeFile } from "node:fs/promises";
import { hostname, tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { RunStore } from "../persistence/run-store.js";
import { acquireOperatorLease, OperatorLeaseLostError } from "./operator-lease.js";

const FAST_LEASE = {
  heartbeatIntervalMs: 10,
  leaseDurationMs: 60,
  reclaimConfirmationMs: 20
};
const CROSS_PROCESS_LEASE = {
  heartbeatIntervalMs: 25,
  leaseDurationMs: 150,
  reclaimConfirmationMs: 50
};
const FENCE_CHILD = fileURLToPath(
  new URL("../../tests/helpers/operator-fence-child.mjs", import.meta.url)
);
const RUN_FIXTURE = resolve(
  process.cwd(),
  "tests/fixtures/runs/20000101T000000Z_fetch_red_block_00000000"
);

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

  it("does not reclaim an aged guard from a live in-progress mutation", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-live-mutation-guard-"));
    const owner = await acquireOperatorLease(runsDir);
    let enter!: () => void;
    let finish!: () => void;
    const entered = new Promise<void>((resolveEntered) => {
      enter = resolveEntered;
    });
    const gate = new Promise<void>((resolveFinish) => {
      finish = resolveFinish;
    });
    try {
      const mutation = owner.runMutation(async () => {
        enter();
        await gate;
      });
      await entered;
      const guardPath = join(runsDir, ".operator.lock.guard");
      const guard = JSON.parse(await readFile(guardPath, "utf8")) as { token: string };
      const old = new Date("2000-01-01T00:00:00.000Z");
      await utimes(guardPath, old, old);

      const contender = acquireOperatorLease(runsDir, FAST_LEASE);
      await delay(100);
      expect(JSON.parse(await readFile(guardPath, "utf8"))).toMatchObject({
        token: guard.token,
        pid: process.pid
      });

      finish();
      await mutation;
      await expect(contender).rejects.toThrow("already owned");
      await expect(owner.assertOwned()).resolves.toBeUndefined();
    } finally {
      finish();
      await owner.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("never rewrites or removes an aged valid guard held by another host", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-foreign-guard-"));
    const guardPath = join(runsDir, ".operator.lock.guard");
    const foreignGuard = {
      token: randomUUID(),
      pid: process.pid,
      hostname: `${hostname()}-foreign`,
      started_at: "2026-07-30T00:00:00.000Z"
    };
    const serialized = `${JSON.stringify(foreignGuard)}\n`;
    await writeFile(guardPath, serialized, "utf8");
    const old = new Date("2000-01-01T00:00:00.000Z");
    await utimes(guardPath, old, old);
    const before = await stat(guardPath);
    try {
      let rejection: unknown;
      try {
        await acquireOperatorLease(runsDir, FAST_LEASE);
      } catch (error) {
        rejection = error;
      }

      expect(rejection).toBeInstanceOf(Error);
      const message = (rejection as Error).message;
      expect(message).toBe("Runs directory lease guard is held by another host");
      expect(message).not.toContain(foreignGuard.hostname);
      expect(message).not.toContain(foreignGuard.token);
      expect(message).not.toContain(String(foreignGuard.pid));
      expect(await readFile(guardPath, "utf8")).toBe(serialized);
      const after = await stat(guardPath);
      expect({
        dev: after.dev,
        ino: after.ino,
        size: after.size,
        mtimeMs: after.mtimeMs
      }).toEqual({
        dev: before.dev,
        ino: before.ino,
        size: before.size,
        mtimeMs: before.mtimeMs
      });
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("keeps an expired foreign lease fenced until its foreign guard is released", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-foreign-guard-fence-"));
    const leasePath = join(runsDir, ".operator.lock");
    const guardPath = `${leasePath}.guard`;
    const foreignHostname = `${hostname()}-foreign`;
    const foreignLease = {
      version: 2,
      token: randomUUID(),
      pid: process.pid,
      hostname: foreignHostname,
      started_at: "2026-07-30T00:00:00.000Z",
      heartbeat_interval_ms: FAST_LEASE.heartbeatIntervalMs,
      lease_duration_ms: FAST_LEASE.leaseDurationMs
    };
    const foreignGuard = {
      token: randomUUID(),
      pid: process.pid,
      hostname: foreignHostname,
      started_at: "2026-07-30T00:00:00.000Z"
    };
    await writeFile(leasePath, `${JSON.stringify(foreignLease)}\n`, "utf8");
    await writeFile(guardPath, `${JSON.stringify(foreignGuard)}\n`, "utf8");
    const old = new Date("2000-01-01T00:00:00.000Z");
    await utimes(leasePath, old, old);
    await utimes(guardPath, old, old);
    let contender: Awaited<ReturnType<typeof acquireOperatorLease>> | undefined;
    try {
      await expect(acquireOperatorLease(runsDir, FAST_LEASE))
        .rejects.toThrow("lease guard is held by another host");
      expect(JSON.parse(await readFile(leasePath, "utf8"))).toMatchObject({
        token: foreignLease.token,
        hostname: foreignHostname
      });
      expect(JSON.parse(await readFile(guardPath, "utf8"))).toEqual(foreignGuard);

      await rm(guardPath);
      contender = await acquireOperatorLease(runsDir, FAST_LEASE);
      expect(JSON.parse(await readFile(leasePath, "utf8"))).not.toMatchObject({
        token: foreignLease.token
      });
      await expect(contender.assertOwned()).resolves.toBeUndefined();
    } finally {
      await contender?.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("reclaims a valid guard only after its same-host process has ended", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-dead-local-guard-"));
    const guardPath = join(runsDir, ".operator.lock.guard");
    const child = spawn(process.execPath, ["-e", ""], { stdio: "ignore" });
    const deadPid = child.pid;
    await once(child, "exit");
    if (!deadPid) throw new Error("Child process had no pid");
    await writeFile(guardPath, `${JSON.stringify({
      token: randomUUID(),
      pid: deadPid,
      hostname: hostname(),
      started_at: "2026-07-30T00:00:00.000Z"
    })}\n`, "utf8");
    let lease: Awaited<ReturnType<typeof acquireOperatorLease>> | undefined;
    try {
      lease = await acquireOperatorLease(runsDir, FAST_LEASE);
      await expect(lease.assertOwned()).resolves.toBeUndefined();
      await expect(stat(guardPath)).rejects.toMatchObject({ code: "ENOENT" });
    } finally {
      await lease?.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("reclaims an aged malformed guard", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-malformed-guard-"));
    const guardPath = join(runsDir, ".operator.lock.guard");
    await writeFile(guardPath, "{\"interrupted\":", "utf8");
    const old = new Date("2000-01-01T00:00:00.000Z");
    await utimes(guardPath, old, old);
    let lease: Awaited<ReturnType<typeof acquireOperatorLease>> | undefined;
    try {
      lease = await acquireOperatorLease(runsDir, FAST_LEASE);
      await expect(lease.assertOwned()).resolves.toBeUndefined();
      await expect(stat(guardPath)).rejects.toMatchObject({ code: "ENOENT" });
    } finally {
      await lease?.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("does not retry a guarded mutation whose operation throws EEXIST", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-guard-operation-error-"));
    const owner = await acquireOperatorLease(runsDir, FAST_LEASE);
    let attempts = 0;
    const collision = Object.assign(new Error("mutation target already exists"), {
      code: "EEXIST"
    });
    try {
      await expect(owner.runMutation(async () => {
        attempts += 1;
        throw collision;
      })).rejects.toBe(collision);
      expect(attempts).toBe(1);
      await expect(owner.assertOwned()).resolves.toBeUndefined();
    } finally {
      await owner.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("retries guard cleanup without replacing the operation error", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-guard-cleanup-retry-"));
    const owner = await acquireOperatorLease(runsDir, FAST_LEASE);
    const guardPath = join(runsDir, ".operator.lock.guard");
    const operationFailure = new Error("injected persistence failure");
    let restoreGuard: Promise<void> | undefined;
    try {
      await expect(owner.runMutation(async () => {
        const guard = await readFile(guardPath, "utf8");
        await rm(guardPath, { force: true });
        await mkdir(guardPath);
        restoreGuard = new Promise<void>((resolveRestore, rejectRestore) => {
          setTimeout(() => {
            void (async () => {
              await rm(guardPath, { recursive: true, force: true });
              await writeFile(guardPath, guard, "utf8");
            })().then(resolveRestore, rejectRestore);
          }, 15);
        });
        throw operationFailure;
      })).rejects.toBe(operationFailure);
      await restoreGuard;
      expect(owner.signal.aborted).toBe(false);
      await expect(owner.runMutation(async () => "next mutation"))
        .resolves.toBe("next mutation");
    } finally {
      await restoreGuard?.catch(() => undefined);
      await rm(guardPath, { recursive: true, force: true });
      await owner.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("fences the owner when failed-operation guard cleanup cannot complete", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-guard-cleanup-loss-"));
    const owner = await acquireOperatorLease(runsDir, FAST_LEASE);
    const guardPath = join(runsDir, ".operator.lock.guard");
    const operationFailure = new Error("original mutation failure");
    try {
      await expect(owner.runMutation(async () => {
        await rm(guardPath, { force: true });
        await mkdir(guardPath);
        throw operationFailure;
      })).rejects.toBe(operationFailure);

      expect(owner.signal.aborted).toBe(true);
      expect(owner.signal.reason).toBeInstanceOf(OperatorLeaseLostError);
      expect((owner.signal.reason as Error).message)
        .toContain("lease guard cleanup failed");

      let subsequentOperationRan = false;
      const startedAt = Date.now();
      await expect(owner.runMutation(async () => {
        subsequentOperationRan = true;
      })).rejects.toBe(owner.signal.reason);
      expect(subsequentOperationRan).toBe(false);
      expect(Date.now() - startedAt).toBeLessThan(500);
    } finally {
      await rm(guardPath, { recursive: true, force: true });
      await owner.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("post-renews a long successful mutation before allowing a contender through", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-long-mutation-renewal-"));
    const restoreHeartbeats = suppressLeaseHeartbeats();
    const owner = await acquireOperatorLease(runsDir, FAST_LEASE);
    let enter!: () => void;
    let finish!: () => void;
    const entered = new Promise<void>((resolveEntered) => {
      enter = resolveEntered;
    });
    const gate = new Promise<void>((resolveFinish) => {
      finish = resolveFinish;
    });
    let mutation: Promise<void> | undefined;
    let contender: ReturnType<typeof acquireOperatorLease> | undefined;
    try {
      mutation = owner.runMutation(async () => {
        enter();
        await gate;
      });
      await entered;
      await delay(FAST_LEASE.leaseDurationMs + 20);
      let contenderSettled = false;
      contender = acquireOperatorLease(runsDir, FAST_LEASE);
      void contender.then(
        () => { contenderSettled = true; },
        () => { contenderSettled = true; }
      );
      await delay(30);
      expect(contenderSettled).toBe(false);

      finish();
      await mutation;
      await expect(contender).rejects.toThrow("already owned");
      expect(owner.signal.aborted).toBe(false);
      await expect(owner.assertOwned()).resolves.toBeUndefined();
    } finally {
      finish();
      await mutation?.catch(() => undefined);
      const acquired = await contender?.catch(() => undefined);
      await acquired?.release();
      await owner.release();
      restoreHeartbeats();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("post-renews a long failed mutation without replacing its original error", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-long-failed-renewal-"));
    const restoreHeartbeats = suppressLeaseHeartbeats();
    const owner = await acquireOperatorLease(runsDir, FAST_LEASE);
    const operationFailure = new Error("injected persistence failure");
    let enter!: () => void;
    let finish!: () => void;
    const entered = new Promise<void>((resolveEntered) => {
      enter = resolveEntered;
    });
    const gate = new Promise<void>((resolveFinish) => {
      finish = resolveFinish;
    });
    let mutation: Promise<void> | undefined;
    let contender: ReturnType<typeof acquireOperatorLease> | undefined;
    try {
      mutation = owner.runMutation(async () => {
        enter();
        await gate;
        throw operationFailure;
      });
      await entered;
      await delay(FAST_LEASE.leaseDurationMs + 20);
      contender = acquireOperatorLease(runsDir, FAST_LEASE);
      await delay(30);

      finish();
      await expect(mutation).rejects.toBe(operationFailure);
      await expect(contender).rejects.toThrow("already owned");
      expect(owner.signal.aborted).toBe(false);
      await expect(owner.assertOwned()).resolves.toBeUndefined();
    } finally {
      finish();
      await mutation?.catch(() => undefined);
      const acquired = await contender?.catch(() => undefined);
      await acquired?.release();
      await owner.release();
      restoreHeartbeats();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("preserves a mutation error when its error-path renewal also loses ownership", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-failed-renewal-loss-"));
    const owner = await acquireOperatorLease(runsDir, FAST_LEASE);
    const operationFailure = new Error("original mutation error");
    const replacement = {
      version: 2,
      token: randomUUID(),
      pid: process.pid,
      hostname: hostname(),
      started_at: new Date().toISOString(),
      heartbeat_interval_ms: FAST_LEASE.heartbeatIntervalMs,
      lease_duration_ms: FAST_LEASE.leaseDurationMs
    };
    try {
      await expect(owner.runMutation(async () => {
        await writeFile(
          join(runsDir, ".operator.lock"),
          `${JSON.stringify(replacement)}\n`,
          "utf8"
        );
        throw operationFailure;
      })).rejects.toBe(operationFailure);
      expect(owner.signal.aborted).toBe(true);
      expect(JSON.parse(await readFile(join(runsDir, ".operator.lock"), "utf8")))
        .toMatchObject({ token: replacement.token });
    } finally {
      await owner.release();
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

  it("rejects a paused stale process before it can overwrite the new owner's checkpoint", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-checkpoint-fence-"));
    const store = await copyRunStore(runsDir);
    const child = forkFenceChild(runsDir, store.runDir, "checkpoint");
    const exited = once(child, "exit");
    let nextOwner: Awaited<ReturnType<typeof acquireOperatorLease>> | undefined;
    try {
      await childMessage(child, "owned");
      await delay(250);
      nextOwner = await acquireOperatorLease(runsDir, CROSS_PROCESS_LEASE);
      const nextStore = await RunStore.open(store.runDir, { mutationFence: nextOwner });
      const checkpoint = await nextStore.readCheckpoint();
      await nextStore.writeCheckpoint({
        ...checkpoint,
        error: "new-owner",
        updated_at: new Date().toISOString()
      });

      const result = childMessage(child, "write_result");
      child.send("write");
      await expect(result).resolves.toMatchObject({
        accepted: false,
        error_name: "OperatorLeaseLostError",
        error_message: expect.stringMatching(
          /lease token (?:was replaced|changed during renewal)/
        )
      });
      await exited;

      expect((await nextStore.readCheckpoint()).error).toBe("new-owner");
      await expect(nextOwner.assertOwned()).resolves.toBeUndefined();
    } finally {
      if (child.exitCode === null && child.signalCode === null) child.kill();
      await nextOwner?.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  }, 20_000);

  it("rejects a paused stale journal append without damaging the new owner's offsets", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-journal-fence-"));
    const store = await createRunStore(runsDir, "Fence an indexed journal append");
    const child = forkFenceChild(runsDir, store.runDir, "journal");
    const exited = once(child, "exit");
    let nextOwner: Awaited<ReturnType<typeof acquireOperatorLease>> | undefined;
    try {
      await childMessage(child, "owned");
      await delay(250);
      nextOwner = await acquireOperatorLease(runsDir, CROSS_PROCESS_LEASE);
      const nextStore = await RunStore.open(store.runDir, { mutationFence: nextOwner });
      const first = { owner: "new-owner", sequence: 1 };
      const second = { owner: "new-owner", sequence: 2 };
      await nextStore.append("events", first);

      const result = childMessage(child, "write_result");
      child.send("write");
      await expect(result).resolves.toMatchObject({
        accepted: false,
        error_name: "OperatorLeaseLostError"
      });
      await exited;
      await nextStore.append("events", second);

      expect(await nextStore.readJournal("events")).toEqual([first, second]);
      const offsets = await readFile(join(store.runDir, "events.offsets"));
      expect((await stat(join(store.runDir, "events.offsets"))).size).toBe(16);
      expect([
        Number(offsets.readBigUInt64LE(0)),
        Number(offsets.readBigUInt64LE(8))
      ]).toEqual([0, Buffer.byteLength(`${JSON.stringify(first)}\n`)]);
      await expect(nextOwner.assertOwned()).resolves.toBeUndefined();
    } finally {
      if (child.exitCode === null && child.signalCode === null) child.kill();
      await nextOwner?.release();
      await rm(runsDir, { recursive: true, force: true });
    }
  }, 20_000);
});

async function createRunStore(runsDir: string, mission: string): Promise<RunStore> {
  const catalog = await loadRuntimeCatalog();
  const scenario = catalog.materialize("open_navigation", 0);
  return RunStore.create(runsDir, {
    mission,
    scenarioId: "open_navigation",
    scenario,
    goal: scenario.default_goal
  });
}

async function copyRunStore(runsDir: string): Promise<RunStore> {
  const destination = join(runsDir, basename(RUN_FIXTURE));
  await cp(RUN_FIXTURE, destination, { recursive: true });
  return RunStore.open(destination);
}

function forkFenceChild(
  runsDir: string,
  runDir: string,
  mode: "checkpoint" | "journal"
): ChildProcess {
  return fork(FENCE_CHILD, [runsDir, runDir, mode, "5000"], {
    execArgv: ["--import", "tsx"],
    stdio: ["ignore", "ignore", "pipe", "ipc"]
  });
}

function childMessage(
  child: ChildProcess,
  type: string
): Promise<Record<string, unknown>> {
  return new Promise((resolveMessage, rejectMessage) => {
    const timer = setTimeout(() => {
      cleanup();
      rejectMessage(new Error(`Timed out waiting for child message ${type}`));
    }, 10_000);
    timer.unref();
    const onMessage = (value: unknown): void => {
      if (typeof value !== "object" || value === null || !("type" in value)
        || value.type !== type) return;
      cleanup();
      resolveMessage(value as Record<string, unknown>);
    };
    const onExit = (code: number | null, signal: NodeJS.Signals | null): void => {
      cleanup();
      rejectMessage(new Error(`Fence child exited before ${type}: ${code ?? signal ?? "unknown"}`));
    };
    const onError = (error: Error): void => {
      cleanup();
      rejectMessage(error);
    };
    const cleanup = (): void => {
      clearTimeout(timer);
      child.off("message", onMessage);
      child.off("exit", onExit);
      child.off("error", onError);
    };
    child.on("message", onMessage);
    child.once("exit", onExit);
    child.once("error", onError);
  });
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

function suppressLeaseHeartbeats(): () => void {
  const dormant = setTimeout(() => undefined, 60_000);
  dormant.unref();
  const interval = vi.spyOn(globalThis, "setInterval").mockReturnValue(dormant);
  return () => {
    interval.mockRestore();
    clearTimeout(dormant);
  };
}
