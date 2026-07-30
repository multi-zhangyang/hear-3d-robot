import { randomUUID } from "node:crypto";
import {
  mkdir,
  open,
  readFile,
  rename,
  rm,
  stat,
  unlink,
  utimes
} from "node:fs/promises";
import { hostname } from "node:os";
import { resolve } from "node:path";
import { z } from "zod";

const LegacyLeaseRecordSchema = z.object({
  token: z.string().uuid(),
  pid: z.number().int().positive(),
  hostname: z.string().min(1),
  started_at: z.string().datetime()
}).strict();

const RenewableLeaseRecordSchema = LegacyLeaseRecordSchema.extend({
  version: z.literal(2),
  heartbeat_interval_ms: z.number().int().positive(),
  lease_duration_ms: z.number().int().positive()
}).strict();

const LeaseRecordSchema = z.union([
  RenewableLeaseRecordSchema,
  LegacyLeaseRecordSchema
]);
const GuardRecordSchema = z.object({
  token: z.string().uuid(),
  created_at: z.string().datetime()
}).strict();

type LeaseRecord = z.infer<typeof LeaseRecordSchema>;
type RenewableLeaseRecord = z.infer<typeof RenewableLeaseRecordSchema>;

interface LeaseSnapshot {
  record: LeaseRecord | undefined;
  identity: string;
  modifiedAt: number;
}

interface LeaseTiming {
  heartbeatIntervalMs: number;
  leaseDurationMs: number;
  reclaimConfirmationMs: number;
}

const MAX_ACQUIRE_ATTEMPTS = 16;
const INVALID_LEASE_GRACE_MS = 30_000;
const GUARD_RETRY_MS = 10;
const MAX_GUARD_ATTEMPTS = 1_000;
const RENAME_RETRY_DELAYS_MS = [10, 25, 50, 100] as const;

export interface OperatorLeaseOptions {
  heartbeatIntervalMs?: number;
  leaseDurationMs?: number;
  reclaimConfirmationMs?: number;
}

export interface OperatorLease {
  readonly signal: AbortSignal;
  assertOwned(): Promise<void>;
  release(): Promise<void>;
}

export async function acquireOperatorLease(
  runsDir: string,
  options: OperatorLeaseOptions = {}
): Promise<OperatorLease> {
  const timing = leaseTiming(options);
  await mkdir(runsDir, { recursive: true });
  const path = resolve(runsDir, ".operator.lock");
  const record: RenewableLeaseRecord = {
    version: 2,
    token: randomUUID(),
    pid: process.pid,
    hostname: hostname(),
    started_at: new Date().toISOString(),
    heartbeat_interval_ms: timing.heartbeatIntervalMs,
    lease_duration_ms: timing.leaseDurationMs
  };

  return withLeaseGuard(path, async () => {
    for (let attempt = 0; attempt < MAX_ACQUIRE_ATTEMPTS; attempt += 1) {
      try {
        await createExclusiveJsonFile(path, record);
        return new RenewableOperatorLease(path, record);
      } catch (error) {
        if (!isExists(error)) throw error;
      }

      const current = await readLeaseSnapshot(path);
      if (!current) continue;
      if (isRenewable(current.record)) {
        if (!leaseExpired(current, current.record.lease_duration_ms)) {
          throw ownedLeaseError(current.record);
        }
        const confirmationMs = Math.max(
          timing.reclaimConfirmationMs,
          current.record.heartbeat_interval_ms * 2
        );
        await delay(confirmationMs);
        const confirmed = await readLeaseSnapshot(path);
        if (
          !confirmed
          || !sameSnapshot(confirmed, current)
          || !leaseExpired(confirmed, current.record.lease_duration_ms)
        ) {
          throw ownedLeaseError(confirmed?.record ?? current.record);
        }
        await removeLeaseSnapshot(path, confirmed, record.token);
        continue;
      }

      const staleLegacyProcess = current.record
        && sameHostname(current.record.hostname, record.hostname)
        && !processExists(current.record.pid);
      const abandonedInvalidLease = !current.record
        && Date.now() - current.modifiedAt >= INVALID_LEASE_GRACE_MS;
      if (staleLegacyProcess || abandonedInvalidLease) {
        await removeLeaseSnapshot(path, current, record.token);
        continue;
      }
      throw current.record
        ? ownedLeaseError(current.record)
        : new OperatorLeaseError(
            "Runs directory is already owned by an initializing or unreadable existing lease"
          );
    }
    throw new OperatorLeaseError("Runs directory lease could not be acquired");
  });
}

export class OperatorLeaseError extends Error {}

export class OperatorLeaseLostError extends Error {
  readonly statusCode = 503;
}

class RenewableOperatorLease implements OperatorLease {
  readonly signal: AbortSignal;
  readonly #path: string;
  readonly #record: RenewableLeaseRecord;
  readonly #controller = new AbortController();
  readonly #heartbeat: ReturnType<typeof setInterval>;
  #tail: Promise<void> = Promise.resolve();
  #closing = false;
  #heartbeatPending = false;
  #releaseOperation: Promise<void> | undefined;

  constructor(path: string, record: RenewableLeaseRecord) {
    this.#path = path;
    this.#record = record;
    this.signal = this.#controller.signal;
    this.#heartbeat = setInterval(() => {
      if (this.#heartbeatPending) return;
      this.#heartbeatPending = true;
      void this.#enqueue(async () => {
        if (this.#closing || this.signal.aborted) return;
        try {
          await this.#renewAndVerify();
        } catch (error) {
          this.#lose(error);
        }
      }).finally(() => {
        this.#heartbeatPending = false;
      });
    }, record.heartbeat_interval_ms);
    this.#heartbeat.unref();
  }

  async assertOwned(): Promise<void> {
    if (this.#closing) throw new OperatorLeaseLostError("Operator lease is closing");
    this.signal.throwIfAborted();
    await this.#enqueue(async () => {
      if (this.#closing) throw new OperatorLeaseLostError("Operator lease is closing");
      this.signal.throwIfAborted();
      try {
        await this.#renewAndVerify();
      } catch (error) {
        const lost = leaseLostError(error);
        this.#lose(lost);
        throw lost;
      }
    });
  }

  release(): Promise<void> {
    if (this.#releaseOperation) return this.#releaseOperation;
    this.#closing = true;
    clearInterval(this.#heartbeat);
    this.#releaseOperation = this.#enqueue(async () => {
      await withLeaseGuard(this.#path, () => releaseLease(this.#path, this.#record.token));
    });
    return this.#releaseOperation;
  }

  async #renewAndVerify(): Promise<void> {
    const before = await readLeaseSnapshot(this.#path);
    if (before?.record?.token !== this.#record.token) {
      throw new OperatorLeaseLostError("Operator lease token was replaced");
    }
    const now = new Date();
    await utimes(this.#path, now, now);
    const after = await readLeaseSnapshot(this.#path);
    if (after?.record?.token !== this.#record.token) {
      throw new OperatorLeaseLostError("Operator lease token changed during renewal");
    }
  }

  #enqueue(operation: () => Promise<void>): Promise<void> {
    const result = this.#tail.then(operation, operation);
    this.#tail = result.then(() => undefined, () => undefined);
    return result;
  }

  #lose(reason: unknown): void {
    if (this.signal.aborted || this.#closing) return;
    clearInterval(this.#heartbeat);
    this.#controller.abort(leaseLostError(reason));
  }
}

async function withLeaseGuard<T>(path: string, operation: () => Promise<T>): Promise<T> {
  const guardPath = `${path}.guard`;
  const guard = { token: randomUUID(), created_at: new Date().toISOString() };
  for (let attempt = 0; attempt < MAX_GUARD_ATTEMPTS; attempt += 1) {
    try {
      await createExclusiveJsonFile(guardPath, guard);
      try {
        return await operation();
      } finally {
        await releaseTokenFile(guardPath, guard.token);
      }
    } catch (error) {
      if (!isExists(error)) throw error;
      const existing = await readLeaseSnapshot(guardPath);
      if (!existing) continue;
      if (Date.now() - existing.modifiedAt >= INVALID_LEASE_GRACE_MS) {
        await removeLeaseSnapshot(guardPath, existing, guard.token);
        continue;
      }
      await delay(GUARD_RETRY_MS);
    }
  }
  throw new OperatorLeaseError("Runs directory lease guard could not be acquired");
}

async function createExclusiveJsonFile(path: string, value: unknown): Promise<void> {
  const handle = await open(path, "wx", 0o600);
  let complete = false;
  try {
    await handle.writeFile(`${JSON.stringify(value)}\n`, "utf8");
    await handle.sync();
    complete = true;
  } finally {
    await handle.close();
    if (!complete) {
      await unlink(path).catch((error) => {
        if (!isMissing(error)) throw error;
      });
    }
  }
}

async function readLeaseSnapshot(path: string): Promise<LeaseSnapshot | undefined> {
  let info;
  try {
    info = await stat(path);
  } catch (error) {
    if (isMissing(error)) return undefined;
    throw error;
  }
  let record: LeaseRecord | undefined;
  if (info.isFile()) {
    try {
      record = LeaseRecordSchema.parse(JSON.parse(await readFile(path, "utf8")));
    } catch (error) {
      if (isMissing(error)) return undefined;
      if (!(error instanceof SyntaxError) && !(error instanceof z.ZodError)) throw error;
    }
  }
  return {
    record,
    identity: `${info.dev}:${info.ino}:${info.size}:${info.mtimeMs}`,
    modifiedAt: info.mtimeMs
  };
}

/** Moves the exact stale lease aside while the acquisition guard excludes contenders. */
async function removeLeaseSnapshot(
  path: string,
  expected: LeaseSnapshot,
  contenderToken: string
): Promise<void> {
  const quarantine = `${path}.stale.${contenderToken}`;
  try {
    await renameWithRetry(path, quarantine);
  } catch (error) {
    if (isMissing(error)) return;
    throw error;
  }
  const moved = await readLeaseSnapshot(quarantine);
  if (!moved || !sameSnapshot(moved, expected)) {
    try {
      await renameWithRetry(quarantine, path);
    } catch (error) {
      if (!isExists(error)) throw error;
      throw new OperatorLeaseError("Runs directory lease changed during stale-owner recovery");
    }
    return;
  }
  await rm(quarantine, { recursive: true, force: true });
}

async function releaseLease(path: string, token: string): Promise<void> {
  const current = await readLeaseSnapshot(path);
  if (current?.record?.token !== token) return;
  const quarantine = `${path}.release.${token}`;
  try {
    await renameWithRetry(path, quarantine);
  } catch (error) {
    if (isMissing(error)) return;
    throw error;
  }
  const moved = await readLeaseSnapshot(quarantine);
  if (moved?.record?.token !== token) {
    try {
      await renameWithRetry(quarantine, path);
    } catch (error) {
      if (!isExists(error)) throw error;
      throw new OperatorLeaseError("Runs directory lease changed during release");
    }
    return;
  }
  await rm(quarantine, { recursive: true, force: true });
}

async function releaseTokenFile(path: string, token: string): Promise<void> {
  let current: z.infer<typeof GuardRecordSchema> | undefined;
  try {
    current = GuardRecordSchema.parse(JSON.parse(await readFile(path, "utf8")));
  } catch (error) {
    if (isMissing(error)) return;
    if (!(error instanceof SyntaxError) && !(error instanceof z.ZodError)) throw error;
  }
  if (current?.token !== token) return;
  const quarantine = `${path}.release.${token}`;
  try {
    await renameWithRetry(path, quarantine);
  } catch (error) {
    if (isMissing(error)) return;
    throw error;
  }
  let movedToken: string | undefined;
  try {
    movedToken = GuardRecordSchema.parse(
      JSON.parse(await readFile(quarantine, "utf8"))
    ).token;
  } catch (error) {
    if (!isMissing(error) && !(error instanceof SyntaxError) && !(error instanceof z.ZodError)) {
      throw error;
    }
  }
  if (movedToken !== token) {
    try {
      await renameWithRetry(quarantine, path);
    } catch (error) {
      if (!isExists(error)) throw error;
      throw new OperatorLeaseError("Runs directory lease guard changed during release");
    }
    return;
  }
  await rm(quarantine, { recursive: true, force: true });
}

function leaseTiming(options: OperatorLeaseOptions): LeaseTiming {
  const heartbeatIntervalMs = options.heartbeatIntervalMs ?? 1_000;
  const leaseDurationMs = options.leaseDurationMs ?? 10_000;
  const reclaimConfirmationMs = options.reclaimConfirmationMs ?? 2_000;
  if (
    !Number.isSafeInteger(heartbeatIntervalMs) || heartbeatIntervalMs < 1
    || !Number.isSafeInteger(leaseDurationMs) || leaseDurationMs < heartbeatIntervalMs * 3
    || !Number.isSafeInteger(reclaimConfirmationMs)
    || reclaimConfirmationMs < heartbeatIntervalMs
  ) {
    throw new Error(
      "Operator lease timing requires positive integers, a duration of at least three heartbeats, and confirmation of at least one heartbeat"
    );
  }
  return { heartbeatIntervalMs, leaseDurationMs, reclaimConfirmationMs };
}

function isRenewable(record: LeaseRecord | undefined): record is RenewableLeaseRecord {
  return record !== undefined && "version" in record && record.version === 2;
}

function leaseExpired(snapshot: LeaseSnapshot, durationMs: number): boolean {
  return Date.now() - snapshot.modifiedAt >= durationMs;
}

function sameSnapshot(left: LeaseSnapshot, right: LeaseSnapshot): boolean {
  return left.identity === right.identity
    && left.record?.token === right.record?.token;
}

function ownedLeaseError(record: LeaseRecord): OperatorLeaseError {
  return new OperatorLeaseError(
    `Runs directory is already owned by ${record.hostname} pid ${record.pid} since ${record.started_at}`
  );
}

function leaseLostError(reason: unknown): OperatorLeaseLostError {
  if (reason instanceof OperatorLeaseLostError) return reason;
  const detail = reason instanceof Error ? reason.message : String(reason);
  return new OperatorLeaseLostError(`Operator lease was lost: ${detail}`);
}

function sameHostname(left: string, right: string): boolean {
  return left.toLocaleLowerCase("en-US") === right.toLocaleLowerCase("en-US");
}

function processExists(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !(error instanceof Error && "code" in error && error.code === "ESRCH");
  }
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

async function renameWithRetry(source: string, destination: string): Promise<void> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      await rename(source, destination);
      return;
    } catch (error) {
      const waitMs = RENAME_RETRY_DELAYS_MS[attempt];
      if (waitMs === undefined || !isTransientRenameFailure(error)) throw error;
      await delay(waitMs);
    }
  }
}

function isTransientRenameFailure(error: unknown): boolean {
  if (!(error instanceof Error) || !("code" in error)) return false;
  return error.code === "EACCES" || error.code === "EBUSY" || error.code === "EPERM";
}

function isExists(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "EEXIST";
}

function isMissing(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}
