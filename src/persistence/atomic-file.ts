import { setTimeout as delay } from "node:timers/promises";
import { mkdir, open, rename, stat, unlink } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { dirname } from "node:path";

const RENAME_RETRY_DELAYS_MS = [10, 25, 50, 100, 250, 500, 1_000] as const;

/**
 * Replaces a UTF-8 file from a same-directory temporary file. The temporary is
 * flushed before it becomes visible, transient rename sharing failures are
 * retried, and every failed path removes its private artifact.
 */
export async function writeTextAtomically(destination: string, value: string): Promise<void> {
  const directory = dirname(destination);
  await ensureDirectoryDurably(directory);
  const temporary = `${destination}.${randomUUID()}.tmp`;
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    handle = await open(temporary, "wx", 0o600);
    await handle.writeFile(value, "utf8");
    await handle.sync();
    await handle.close();
    handle = undefined;
    await replaceFileAtomically(temporary, destination);
  } finally {
    await handle?.close().catch(() => undefined);
    await unlink(temporary).catch((error) => {
      if (!isMissing(error)) throw error;
    });
  }
}

/**
 * Creates only missing path components and flushes the parent entry for every
 * component this call observed as missing. Directory fsync is not implemented
 * consistently on every supported platform, so syncDirectory retains the same
 * narrow unsupported-operation fallback used after an atomic rename.
 */
async function ensureDirectoryDurably(directory: string): Promise<void> {
  if (await pathExists(directory)) return;
  const parent = dirname(directory);
  if (parent === directory) throw new Error(`Cannot create filesystem root: ${directory}`);
  await ensureDirectoryDurably(parent);
  try {
    await mkdir(directory);
  } catch (error) {
    // A concurrent creator may win after pathExists. Its new parent entry still
    // needs the same flush before this writer starts publishing files beneath it.
    if (!isAlreadyExists(error)) throw error;
  }
  await syncDirectory(parent);
}

/** Publishes an already flushed same-directory temporary file. */
export async function replaceFileAtomically(source: string, destination: string): Promise<void> {
  await publishPathAtomically(source, destination);
}

/**
 * Publishes a same-parent staged file or directory and durably records the new
 * directory entry. Windows scanners and indexers can briefly hold a sharing
 * lease on either path, so directory epochs use the same bounded transient
 * rename retry as atomic files instead of failing an otherwise valid resume.
 */
export async function publishPathAtomically(
  source: string,
  destination: string
): Promise<void> {
  await renameWithRetry(source, destination);
  await syncDirectory(dirname(destination));
}

async function renameWithRetry(source: string, destination: string): Promise<void> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      await rename(source, destination);
      return;
    } catch (error) {
      const wait = RENAME_RETRY_DELAYS_MS[attempt];
      if (wait === undefined || !isTransientRenameFailure(error)) throw error;
      await delay(wait);
    }
  }
}

async function syncDirectory(directory: string): Promise<void> {
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    handle = await open(directory, "r");
    await handle.sync();
  } catch (error) {
    if (!isUnsupportedDirectorySync(error)) throw error;
  } finally {
    await handle?.close().catch(() => undefined);
  }
}

function isTransientRenameFailure(error: unknown): boolean {
  const code = errorCode(error);
  return code === "EACCES" || code === "EBUSY" || code === "EPERM";
}

function isUnsupportedDirectorySync(error: unknown): boolean {
  const code = errorCode(error);
  return code === "EACCES"
    || code === "EBADF"
    || code === "EINVAL"
    || code === "EISDIR"
    || code === "ENOTSUP"
    || code === "EPERM";
}

function isMissing(error: unknown): boolean {
  return errorCode(error) === "ENOENT";
}

function isAlreadyExists(error: unknown): boolean {
  return errorCode(error) === "EEXIST";
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (isMissing(error)) return false;
    throw error;
  }
}

function errorCode(error: unknown): string | undefined {
  return error instanceof Error && "code" in error && typeof error.code === "string"
    ? error.code
    : undefined;
}
