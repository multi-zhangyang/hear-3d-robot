import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  open,
  stat,
  truncate,
  unlink,
  writeFile,
  type FileHandle
} from "node:fs/promises";
import { resolve } from "node:path";
import { replaceFileAtomically } from "./atomic-file.js";

const OFFSET_BYTES = 8;
const REBUILD_BATCH_ENTRIES = 8_192;
const VERIFIED_JOURNAL_CACHE_LIMIT = 2_048;
const CONTENT_PROOF_WINDOW_BYTES = 4_096;
const CONTENT_PROOF_MAX_ATTEMPTS = 3;
const journalOperations = new Map<string, Promise<void>>();
const verifiedJournals = new Map<string, VerifiedJournal>();

interface FileFingerprint {
  size: bigint;
  mtimeNs: bigint;
  ctimeNs: bigint;
  dev: bigint;
  ino: bigint;
  contentProof: string;
}

type FileMetadata = Omit<FileFingerprint, "contentProof">;

interface FileSnapshot {
  size: number;
  fingerprint: FileFingerprint;
}

interface VerifiedJournal {
  data: FileFingerprint | undefined;
  index: FileFingerprint | undefined;
  state: JournalIndexState;
}

export interface JournalIndexState {
  count: number;
  completeByteLength: number;
  dataByteLength: number;
}

export interface JournalWindow {
  lines: string[];
  next: number | null;
  total: number;
}

/**
 * Appends complete JSONL records and their byte offsets as one serialized
 * journal operation. A crash can leave data ahead of the index; the next open
 * validates or rebuilds the index and discards only an unterminated tail before
 * appending.
 */
export async function appendIndexedRecords(
  dataPath: string,
  indexPath: string,
  records: readonly string[]
): Promise<JournalIndexState> {
  return appendIndexedRecordsBuilt(dataPath, indexPath, () => records);
}

/**
 * Builds records only after the journal's current count is locked. This lets a
 * caller persist an ordinal-derived cursor in the same operation that assigns
 * the ordinal; calculating it with a later tail read would race another writer.
 */
export async function appendIndexedRecordsBuilt(
  dataPath: string,
  indexPath: string,
  build: (startIndex: number) => readonly string[]
): Promise<JournalIndexState> {
  return withJournalOperation(dataPath, indexPath, async () => {
    const state = await ensureJournalIndexUnlocked(dataPath, indexPath);
    const records = build(state.count);
    if (records.length === 0) return state;
    if (state.dataByteLength !== state.completeByteLength) {
      await truncate(dataPath, state.completeByteLength);
    }

    const encoded = records.map((record) => Buffer.from(`${record}\n`, "utf8"));
    const offsets = Buffer.allocUnsafe(encoded.length * OFFSET_BYTES);
    let byteOffset = state.completeByteLength;
    for (let index = 0; index < encoded.length; index += 1) {
      offsets.writeBigUInt64LE(BigInt(byteOffset), index * OFFSET_BYTES);
      byteOffset += encoded[index]!.byteLength;
    }
    // Publish data before its offset index and flush both in that order. The
    // checkpoint can safely claim a journal record only after this returns;
    // without the syncs a power loss could persist the later checkpoint while
    // losing the context/action record it references.
    await appendDurably(dataPath, Buffer.concat(encoded));
    await appendDurably(indexPath, offsets);
    const next = {
      count: state.count + encoded.length,
      completeByteLength: byteOffset,
      dataByteLength: byteOffset
    };
    await rememberJournalState(dataPath, indexPath, next);
    return next;
  });
}

async function appendDurably(path: string, value: Uint8Array): Promise<void> {
  const file = await open(path, "a", 0o600);
  try {
    await file.writeFile(value);
    await file.sync();
  } finally {
    await file.close();
  }
}

export async function readIndexedWindow(
  dataPath: string,
  indexPath: string,
  from: number,
  limit: number,
  maximumBytes?: number
): Promise<JournalWindow> {
  return withJournalOperation(dataPath, indexPath, async () => {
    if (maximumBytes !== undefined
      && (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1)) {
      throw new Error("Journal byte limit must be a positive safe integer");
    }
    const state = await ensureJournalIndexUnlocked(dataPath, indexPath);
    if (from >= state.count) return { lines: [], next: null, total: state.count };
    const requestedEnd = Math.min(state.count, from + limit);
    const offsets = await readOffsets(
      indexPath,
      from,
      requestedEnd - from + (requestedEnd < state.count ? 1 : 0)
    );
    const startByte = offsets[0]!;
    let endIndex = requestedEnd;
    if (maximumBytes !== undefined) {
      endIndex = from + 1;
      for (let candidate = from + 1; candidate <= requestedEnd; candidate += 1) {
        const candidateEndByte = candidate < state.count
          ? offsets[candidate - from]!
          : state.completeByteLength;
        // One oversized record must still make progress. The caller already
        // needs that record in order to deliver or inspect it; the byte budget
        // prevents several large records from being materialized together.
        if (candidate > from + 1 && candidateEndByte - startByte > maximumBytes) break;
        endIndex = candidate;
        if (candidateEndByte - startByte > maximumBytes) break;
      }
    }
    const endByte = endIndex < state.count
      ? offsets[endIndex - from]!
      : state.completeByteLength;
    const text = await readRange(dataPath, startByte, endByte);
    return {
      lines: completeLines(text),
      next: endIndex < state.count ? endIndex : null,
      total: state.count
    };
  });
}

export async function readIndexedTail(
  dataPath: string,
  indexPath: string,
  limit: number
): Promise<JournalWindow> {
  return withJournalOperation(dataPath, indexPath, async () => {
    const state = await ensureJournalIndexUnlocked(dataPath, indexPath);
    const from = Math.max(0, state.count - limit);
    if (state.count === 0) return { lines: [], next: null, total: 0 };
    const offsets = await readOffsets(indexPath, from, state.count - from);
    const text = await readRange(dataPath, offsets[0]!, state.completeByteLength);
    return { lines: completeLines(text), next: null, total: state.count };
  });
}

/** Ensures the fixed-width offset file exactly describes complete JSONL rows. */
export async function ensureJournalIndex(
  dataPath: string,
  indexPath: string
): Promise<JournalIndexState> {
  return withJournalOperation(
    dataPath,
    indexPath,
    () => ensureJournalIndexUnlocked(dataPath, indexPath)
  );
}

async function ensureJournalIndexUnlocked(
  dataPath: string,
  indexPath: string
): Promise<JournalIndexState> {
  const [dataInfo, indexInfo] = await Promise.all([
    optionalFileSnapshot(dataPath),
    optionalFileSnapshot(indexPath)
  ]);
  const cached = cachedJournalState(dataPath, indexPath, dataInfo, indexInfo);
  if (cached) return cached;

  if (!dataInfo || dataInfo.size === 0) {
    await writeFile(indexPath, new Uint8Array());
    const empty = { count: 0, completeByteLength: 0, dataByteLength: 0 };
    await rememberJournalState(dataPath, indexPath, empty);
    return empty;
  }

  if (indexInfo && indexInfo.size % OFFSET_BYTES === 0) {
    const count = indexInfo.size / OFFSET_BYTES;
    const valid = await validateJournalIndex(dataPath, indexPath, count, dataInfo.size);
    if (valid) {
      await rememberJournalState(dataPath, indexPath, valid);
      return valid;
    }
  }
  const rebuilt = await rebuildJournalIndex(dataPath, indexPath, dataInfo.size);
  await rememberJournalState(dataPath, indexPath, rebuilt);
  return rebuilt;
}

async function withJournalOperation<T>(
  dataPath: string,
  indexPath: string,
  operation: () => Promise<T>
): Promise<T> {
  const key = journalKey(dataPath, indexPath);
  const previous = journalOperations.get(key) ?? Promise.resolve();
  let release!: () => void;
  const gate = new Promise<void>((resolveGate) => {
    release = resolveGate;
  });
  const tail = previous.then(() => gate);
  journalOperations.set(key, tail);
  await previous;
  try {
    return await operation();
  } finally {
    release();
    if (journalOperations.get(key) === tail) journalOperations.delete(key);
  }
}

function journalKey(dataPath: string, indexPath: string): string {
  return `${resolve(dataPath)}\0${resolve(indexPath)}`;
}

function cachedJournalState(
  dataPath: string,
  indexPath: string,
  data: FileSnapshot | undefined,
  index: FileSnapshot | undefined
): JournalIndexState | undefined {
  const key = journalKey(dataPath, indexPath);
  const cached = verifiedJournals.get(key);
  if (!cached) return undefined;
  if (!sameFingerprint(cached.data, data?.fingerprint)
    || !sameFingerprint(cached.index, index?.fingerprint)) {
    verifiedJournals.delete(key);
    return undefined;
  }
  verifiedJournals.delete(key);
  verifiedJournals.set(key, cached);
  return { ...cached.state };
}

async function rememberJournalState(
  dataPath: string,
  indexPath: string,
  state: JournalIndexState
): Promise<void> {
  const [data, index] = await Promise.all([
    optionalFileSnapshot(dataPath),
    optionalFileSnapshot(indexPath)
  ]);
  const key = journalKey(dataPath, indexPath);
  if ((data?.size ?? 0) !== state.dataByteLength
    || (index?.size ?? 0) !== state.count * OFFSET_BYTES) {
    verifiedJournals.delete(key);
    return;
  }
  verifiedJournals.delete(key);
  verifiedJournals.set(key, {
    data: data?.fingerprint,
    index: index?.fingerprint,
    state: { ...state }
  });
  while (verifiedJournals.size > VERIFIED_JOURNAL_CACHE_LIMIT) {
    const oldest = verifiedJournals.keys().next().value as string | undefined;
    if (oldest === undefined) break;
    verifiedJournals.delete(oldest);
  }
}

function sameFingerprint(
  left: FileFingerprint | undefined,
  right: FileFingerprint | undefined
): boolean {
  if (!left || !right) return left === right;
  return left.size === right.size
    && left.mtimeNs === right.mtimeNs
    && left.ctimeNs === right.ctimeNs
    && left.dev === right.dev
    && left.ino === right.ino
    && left.contentProof === right.contentProof;
}

async function rebuildJournalIndex(
  dataPath: string,
  indexPath: string,
  dataByteLength: number
): Promise<JournalIndexState> {
  const temporary = `${indexPath}.${randomUUID()}.tmp`;
  const output = await open(temporary, "wx", 0o600);
  let outputClosed = false;
  try {
    const scan = await scanJournalOffsets(dataPath, dataByteLength, async (batch) => {
      await writeFully(output, encodeOffsets(batch));
      return true;
    });
    if (!scan) throw new Error("Journal data changed during index rebuild");
    await output.sync();
    await output.close();
    outputClosed = true;
    await replaceFileAtomically(temporary, indexPath);
    return {
      count: scan.count,
      completeByteLength: scan.completeByteLength,
      dataByteLength
    };
  } finally {
    if (!outputClosed) await output.close().catch(() => undefined);
    await unlink(temporary).catch(() => undefined);
  }
}

async function validateJournalIndex(
  dataPath: string,
  indexPath: string,
  indexCount: number,
  dataByteLength: number
): Promise<JournalIndexState | undefined> {
  const index = await open(indexPath, "r");
  let compared = 0;
  try {
    const scan = await scanJournalOffsets(dataPath, dataByteLength, async (expected) => {
      if (compared + expected.length > indexCount) return false;
      const matches = await offsetBatchMatches(index, compared, expected);
      if (!matches) return false;
      compared += expected.length;
      return true;
    });
    if (!scan || compared !== indexCount) return undefined;
    return {
      count: scan.count,
      completeByteLength: scan.completeByteLength,
      dataByteLength
    };
  } finally {
    await index.close();
  }
}

interface JournalOffsetScan {
  count: number;
  completeByteLength: number;
}

/**
 * Scans at most the snapshotted data length and publishes offsets in bounded
 * batches. Returning false from the visitor stops validation at the first
 * mismatch without allocating a range derived from a damaged offset.
 */
async function scanJournalOffsets(
  dataPath: string,
  dataByteLength: number,
  visit: (offsets: readonly number[]) => Promise<boolean>
): Promise<JournalOffsetScan | undefined> {
  let absolute = 0;
  let lineStart = 0;
  let count = 0;
  let completeByteLength = 0;
  let batch: number[] = [];
  const stream = createReadStream(dataPath, { end: dataByteLength - 1 });
  try {
    for await (const raw of stream) {
      const chunk = Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
      for (let index = 0; index < chunk.length; index += 1) {
        if (chunk[index] !== 0x0a) continue;
        const lineEnd = absolute + index;
        if (lineEnd > lineStart) {
          batch.push(lineStart);
          count += 1;
        }
        completeByteLength = lineEnd + 1;
        lineStart = completeByteLength;
        if (batch.length >= REBUILD_BATCH_ENTRIES) {
          if (!await visit(batch)) return undefined;
          batch = [];
        }
      }
      absolute += chunk.length;
    }
    if (absolute !== dataByteLength) return undefined;
    if (batch.length > 0 && !await visit(batch)) return undefined;
    return { count, completeByteLength };
  } finally {
    stream.destroy();
  }
}

async function offsetBatchMatches(
  file: FileHandle,
  from: number,
  expected: readonly number[]
): Promise<boolean> {
  const encoded = Buffer.allocUnsafe(expected.length * OFFSET_BYTES);
  const position = from * OFFSET_BYTES;
  if (!await readFully(file, encoded, position)) return false;
  return expected.every((offset, index) =>
    encoded.readBigUInt64LE(index * OFFSET_BYTES) === BigInt(offset)
  );
}

async function readFully(file: FileHandle, buffer: Buffer, position: number): Promise<boolean> {
  let offset = 0;
  while (offset < buffer.length) {
    const { bytesRead } = await file.read(
      buffer,
      offset,
      buffer.length - offset,
      position + offset
    );
    if (bytesRead === 0) return false;
    offset += bytesRead;
  }
  return true;
}

async function writeFully(file: FileHandle, value: Uint8Array): Promise<void> {
  let offset = 0;
  while (offset < value.byteLength) {
    const { bytesWritten } = await file.write(
      value,
      offset,
      value.byteLength - offset,
      null
    );
    if (bytesWritten === 0) throw new Error("Journal offset index write made no progress");
    offset += bytesWritten;
  }
}

function encodeOffsets(offsets: readonly number[]): Buffer {
  const buffer = Buffer.allocUnsafe(offsets.length * OFFSET_BYTES);
  offsets.forEach((offset, index) => {
    buffer.writeBigUInt64LE(BigInt(offset), index * OFFSET_BYTES);
  });
  return buffer;
}

async function readOffsets(indexPath: string, from: number, count: number): Promise<number[]> {
  if (count <= 0) return [];
  const file = await open(indexPath, "r");
  const buffer = Buffer.allocUnsafe(count * OFFSET_BYTES);
  try {
    if (!await readFully(file, buffer, from * OFFSET_BYTES)) {
      throw new Error("Journal offset index ended unexpectedly");
    }
  } finally {
    await file.close();
  }
  return Array.from({ length: count }, (_, index) =>
    Number(buffer.readBigUInt64LE(index * OFFSET_BYTES))
  );
}

async function readRange(path: string, start: number, end: number): Promise<string> {
  const length = end - start;
  if (length <= 0) return "";
  const file = await open(path, "r");
  const buffer = Buffer.allocUnsafe(length);
  try {
    if (!await readFully(file, buffer, start)) throw new Error("Journal data ended unexpectedly");
  } finally {
    await file.close();
  }
  return buffer.toString("utf8");
}

function completeLines(text: string): string[] {
  const lines = text.split("\n");
  if (lines.at(-1) === "") lines.pop();
  return lines.filter((line) => line.length > 0);
}

async function optionalFileSnapshot(path: string): Promise<FileSnapshot | undefined> {
  for (let attempt = 0; attempt < CONTENT_PROOF_MAX_ATTEMPTS; attempt += 1) {
    try {
      const before = fileMetadata(await stat(path, { bigint: true }));
      if (before.size > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new Error("Journal file is too large to index safely");
      }
      const size = Number(before.size);
      const contentProof = await sampledContentProof(path, size);
      const after = fileMetadata(await stat(path, { bigint: true }));
      if (!sameMetadata(before, after)) continue;
      return {
        size,
        fingerprint: { ...after, contentProof }
      };
    } catch (error) {
      if (isMissing(error)) return undefined;
      throw error;
    }
  }
  throw new Error(`Journal file changed while fingerprinting: ${path}`);
}

function fileMetadata(info: {
  size: bigint;
  mtimeNs: bigint;
  ctimeNs: bigint;
  dev: bigint;
  ino: bigint;
}): FileMetadata {
  return {
    size: info.size,
    mtimeNs: info.mtimeNs,
    ctimeNs: info.ctimeNs,
    dev: info.dev,
    ino: info.ino
  };
}

function sameMetadata(left: FileMetadata, right: FileMetadata): boolean {
  return left.size === right.size
    && left.mtimeNs === right.mtimeNs
    && left.ctimeNs === right.ctimeNs
    && left.dev === right.dev
    && left.ino === right.ino;
}

async function sampledContentProof(path: string, size: number): Promise<string> {
  const digest = createHash("sha256");
  digest.update(`${size}\0`);
  if (size === 0) return digest.digest("base64url");

  const file = await open(path, "r");
  try {
    for (const window of contentProofWindows(size)) {
      const buffer = Buffer.allocUnsafe(window.length);
      if (!await readFully(file, buffer, window.position)) {
        throw new Error(`Journal file ended while fingerprinting: ${path}`);
      }
      digest.update(`${window.position}:${window.length}\0`);
      digest.update(buffer);
    }
  } finally {
    await file.close();
  }
  return digest.digest("base64url");
}

function contentProofWindows(size: number): Array<{ position: number; length: number }> {
  const maximumSampleBytes = CONTENT_PROOF_WINDOW_BYTES * 3;
  if (size <= maximumSampleBytes) return [{ position: 0, length: size }];
  const lastPosition = size - CONTENT_PROOF_WINDOW_BYTES;
  return [
    { position: 0, length: CONTENT_PROOF_WINDOW_BYTES },
    {
      position: Math.floor(lastPosition / 2),
      length: CONTENT_PROOF_WINDOW_BYTES
    },
    { position: lastPosition, length: CONTENT_PROOF_WINDOW_BYTES }
  ];
}

function isMissing(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}
