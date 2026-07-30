import { createReadStream } from "node:fs";
import {
  mkdtemp,
  open,
  readFile,
  readdir,
  rm,
  stat,
  utimes,
  writeFile
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  appendIndexedRecords,
  ensureJournalIndex,
  readIndexedWindow
} from "./journal-index.js";

vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return { ...actual, createReadStream: vi.fn(actual.createReadStream) };
});

vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs/promises")>();
  return { ...actual, stat: vi.fn(actual.stat) };
});

const RECORDS = [
  JSON.stringify({ index: 0, value: "zero" }),
  JSON.stringify({ index: 1, value: "a longer first payload" }),
  JSON.stringify({ index: 2, value: "middle" }),
  JSON.stringify({ index: 3, value: "another differently sized payload" }),
  JSON.stringify({ index: 4, value: "tail" })
] as const;

describe("journal offset index repair", () => {
  it.each([
    { location: "first", entry: 0 },
    { location: "middle", entry: 2 },
    { location: "last", entry: RECORDS.length - 1 }
  ])("rebuilds a damaged $location offset", async ({ entry }) => {
    const fixture = await createFixture();
    try {
      const expected = await readFile(fixture.indexPath);
      const damaged = Buffer.from(expected);
      const original = damaged.readBigUInt64LE(entry * 8);
      damaged.writeBigUInt64LE(original + 1n, entry * 8);
      await writeFile(fixture.indexPath, damaged);

      const state = await ensureJournalIndex(fixture.dataPath, fixture.indexPath);

      expect(state).toEqual({
        count: RECORDS.length,
        completeByteLength: Buffer.byteLength(`${RECORDS.join("\n")}\n`),
        dataByteLength: Buffer.byteLength(`${RECORDS.join("\n")}\n`)
      });
      expect(await readFile(fixture.indexPath)).toEqual(expected);
      expect(await readIndexedWindow(fixture.dataPath, fixture.indexPath, 0, RECORDS.length))
        .toEqual({ lines: [...RECORDS], next: null, total: RECORDS.length });
      expect((await readdir(fixture.directory)).filter((name) => name.endsWith(".tmp")))
        .toEqual([]);
    } finally {
      await rm(fixture.directory, { recursive: true, force: true });
    }
  });

  it("finishes an index rebuild when individual writes are short", async () => {
    const fixture = await createFixture();
    let restoreWrite: (() => void) | undefined;
    try {
      const expected = await readFile(fixture.indexPath);
      await writeFile(fixture.indexPath, Buffer.from([0xff]));

      const probe = await open(join(fixture.directory, "write-probe"), "w");
      const prototype = Object.getPrototypeOf(probe) as FileHandlePrototype;
      await probe.close();
      const originalWrite = prototype.write;
      const spy = vi.spyOn(prototype, "write").mockImplementation(async function (
        this: unknown,
        ...args: unknown[]
      ): Promise<unknown> {
        const [value, rawOffset, rawLength, rawPosition] = args;
        if (!(value instanceof Uint8Array)) return Reflect.apply(originalWrite, this, args);
        const offset = typeof rawOffset === "number" ? rawOffset : 0;
        const length = typeof rawLength === "number" ? rawLength : value.byteLength - offset;
        return Reflect.apply(originalWrite, this, [
          value,
          offset,
          Math.min(length, 7),
          rawPosition ?? null
        ]);
      });
      restoreWrite = () => spy.mockRestore();

      expect(await ensureJournalIndex(fixture.dataPath, fixture.indexPath)).toMatchObject({
        count: RECORDS.length
      });
      expect(spy.mock.calls.length).toBeGreaterThan(1);
      expect(await readFile(fixture.indexPath)).toEqual(expected);
    } finally {
      restoreWrite?.();
      await rm(fixture.directory, { recursive: true, force: true });
    }
  });

  it("serializes concurrent readers around one damaged index rebuild", async () => {
    const fixture = await createFixture();
    try {
      const expected = await readFile(fixture.indexPath);
      const damaged = Buffer.from(expected);
      damaged.writeBigUInt64LE(damaged.readBigUInt64LE(16) + 1n, 16);
      await writeFile(fixture.indexPath, damaged);

      const windows = await Promise.all(Array.from({ length: 12 }, (_, from) =>
        readIndexedWindow(fixture.dataPath, fixture.indexPath, from % RECORDS.length, 1)
      ));

      expect(windows.every((window) => window.lines.length === 1 && window.total === RECORDS.length))
        .toBe(true);
      expect(await readFile(fixture.indexPath)).toEqual(expected);
      expect((await readdir(fixture.directory)).filter((name) => name.endsWith(".tmp")))
        .toEqual([]);
    } finally {
      await rm(fixture.directory, { recursive: true, force: true });
    }
  });

  it("reuses verified fingerprints until an external middle-offset edit", async () => {
    const fixture = await createFixture();
    const streamSpy = vi.mocked(createReadStream);
    try {
      const oldTimestamp = new Date("2000-01-01T00:00:00.000Z");
      await utimes(fixture.indexPath, oldTimestamp, oldTimestamp);
      streamSpy.mockClear();

      await readIndexedWindow(fixture.dataPath, fixture.indexPath, 0, 1);
      expect(streamSpy).toHaveBeenCalledTimes(1);

      for (let from = 0; from < RECORDS.length; from += 1) {
        await readIndexedWindow(fixture.dataPath, fixture.indexPath, from, 1);
      }
      await appendIndexedRecords(fixture.dataPath, fixture.indexPath, [
        JSON.stringify({ index: RECORDS.length, value: "appended" })
      ]);
      await readIndexedWindow(fixture.dataPath, fixture.indexPath, RECORDS.length, 1);
      expect(streamSpy).toHaveBeenCalledTimes(1);

      const expected = await readFile(fixture.indexPath);
      const damaged = Buffer.from(expected);
      damaged.writeBigUInt64LE(damaged.readBigUInt64LE(16) + 1n, 16);
      await writeFile(fixture.indexPath, damaged);

      expect(await readIndexedWindow(fixture.dataPath, fixture.indexPath, 2, 1))
        .toMatchObject({ lines: [RECORDS[2]], total: RECORDS.length + 1 });
      expect(streamSpy).toHaveBeenCalledTimes(3);
      expect(await readFile(fixture.indexPath)).toEqual(expected);
    } finally {
      streamSpy.mockClear();
      await rm(fixture.directory, { recursive: true, force: true });
    }
  });

  it("repairs repeated same-metadata offset rewrites without rescanning stable reads", async () => {
    const fixture = await createFixture();
    const streamSpy = vi.mocked(createReadStream);
    const statSpy = vi.mocked(stat);
    const actualFs = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
    const frozenDataInfo = await actualFs.stat(fixture.dataPath, { bigint: true });
    const frozenIndexInfo = await actualFs.stat(fixture.indexPath, { bigint: true });
    const expectedData = await readFile(fixture.dataPath);
    const expected = await readFile(fixture.indexPath);
    try {
      statSpy.mockImplementation(((path, options) => {
        if (String(path) === fixture.dataPath) {
          return Promise.resolve(frozenDataInfo);
        }
        if (String(path) === fixture.indexPath) {
          return Promise.resolve(frozenIndexInfo);
        }
        return actualFs.stat(path, options);
      }) as typeof stat);
      streamSpy.mockClear();

      for (let iteration = 0; iteration < 64; iteration += 1) {
        const entry = iteration % RECORDS.length;
        const damaged = Buffer.from(expected);
        damaged.writeBigUInt64LE(
          damaged.readBigUInt64LE(entry * 8) + BigInt(iteration + 1),
          entry * 8
        );
        await writeFile(fixture.indexPath, damaged);

        expect(await ensureJournalIndex(fixture.dataPath, fixture.indexPath)).toMatchObject({
          count: RECORDS.length
        });
        expect(await readFile(fixture.indexPath)).toEqual(expected);
      }

      const changedData = Buffer.from(expectedData);
      const changedAt = changedData.indexOf("zero");
      expect(changedAt).toBeGreaterThanOrEqual(0);
      changedData.write("ZERO", changedAt, "utf8");
      const indexRepairScans = streamSpy.mock.calls.length;
      await writeFile(fixture.dataPath, changedData);
      await ensureJournalIndex(fixture.dataPath, fixture.indexPath);
      expect(streamSpy).toHaveBeenCalledTimes(indexRepairScans + 1);

      await writeFile(fixture.dataPath, expectedData);
      await ensureJournalIndex(fixture.dataPath, fixture.indexPath);
      const rebuildScans = streamSpy.mock.calls.length;
      for (let iteration = 0; iteration < 64; iteration += 1) {
        await readIndexedWindow(
          fixture.dataPath,
          fixture.indexPath,
          iteration % RECORDS.length,
          1
        );
      }
      expect(streamSpy).toHaveBeenCalledTimes(rebuildScans);
    } finally {
      statSpy.mockImplementation(actualFs.stat);
      streamSpy.mockClear();
      await rm(fixture.directory, { recursive: true, force: true });
    }
  });
});

interface FileHandlePrototype {
  write: (...args: unknown[]) => Promise<unknown>;
}

async function createFixture(): Promise<{
  directory: string;
  dataPath: string;
  indexPath: string;
}> {
  const directory = await mkdtemp(join(tmpdir(), "hear-journal-index-"));
  const dataPath = join(directory, "events.jsonl");
  const indexPath = join(directory, "events.offsets");
  await appendIndexedRecords(dataPath, indexPath, RECORDS);
  return { directory, dataPath, indexPath };
}
