import { mkdir, mkdtemp, open, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { writeTextAtomically } from "./atomic-file.js";

describe("durable atomic text files", () => {
  it("publishes one complete replacement and leaves no private temporary files", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-atomic-file-"));
    const nested = join(directory, "状态 files");
    const destination = join(nested, "checkpoint.json");
    const values = Array.from({ length: 16 }, (_, index) => JSON.stringify({ index }));
    try {
      await Promise.all(values.map((value) => writeTextAtomically(destination, `${value}\n`)));
      expect(values).toContain((await readFile(destination, "utf8")).trim());
      expect((await readdir(nested)).filter((entry) => entry.endsWith(".tmp"))).toEqual([]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("cleans its temporary file when the destination cannot be replaced", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-atomic-failure-"));
    const destination = join(directory, "occupied");
    await writeTextAtomically(join(directory, "existing.txt"), "preserved\n");
    await mkdir(destination);
    try {
      await expect(writeTextAtomically(destination, "never visible\n")).rejects.toBeDefined();
      expect((await readdir(directory)).filter((entry) => entry.endsWith(".tmp"))).toEqual([]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("syncs only parents of directories created by this write", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-atomic-directory-sync-"));
    const destination = join(directory, "new parent", "new child", "checkpoint.json");
    const prototype = await syncPrototype(directory);
    const sync = vi.spyOn(prototype, "sync");
    try {
      await writeTextAtomically(destination, "first\n");
      // Two newly created directory entries, one temporary file flush, and the
      // existing destination-directory flush after rename.
      expect(sync).toHaveBeenCalledTimes(4);

      sync.mockClear();
      await writeTextAtomically(destination, "second\n");
      expect(sync).toHaveBeenCalledTimes(2);
      expect(await readFile(destination, "utf8")).toBe("second\n");
    } finally {
      sync.mockRestore();
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("keeps the existing fallback when a platform rejects directory sync", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-atomic-directory-unsupported-"));
    const destination = join(directory, "new directory", "checkpoint.json");
    const prototype = await syncPrototype(directory);
    const original = prototype.sync;
    let calls = 0;
    const sync = vi.spyOn(prototype, "sync").mockImplementation(function (this: unknown) {
      calls += 1;
      if (calls === 1) return Promise.reject(systemError("EPERM"));
      return Reflect.apply(original, this, []);
    });
    try {
      await writeTextAtomically(destination, "durable where supported\n");
      expect(await readFile(destination, "utf8")).toBe("durable where supported\n");
    } finally {
      sync.mockRestore();
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("propagates an unexpected parent-directory sync failure before writing", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-atomic-directory-failure-"));
    const destination = join(directory, "new directory", "checkpoint.json");
    const prototype = await syncPrototype(directory);
    const sync = vi.spyOn(prototype, "sync").mockRejectedValueOnce(systemError("EIO"));
    try {
      await expect(writeTextAtomically(destination, "not published\n"))
        .rejects.toMatchObject({ code: "EIO" });
      await expect(readFile(destination, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
      expect((await readdir(join(directory, "new directory")))
        .filter((entry) => entry.endsWith(".tmp"))).toEqual([]);
    } finally {
      sync.mockRestore();
      await rm(directory, { recursive: true, force: true });
    }
  });
});

interface FileHandleSyncPrototype {
  sync: () => Promise<void>;
}

async function syncPrototype(directory: string): Promise<FileHandleSyncPrototype> {
  const probePath = join(directory, "sync-probe");
  const probe = await open(probePath, "w");
  const prototype = Object.getPrototypeOf(probe) as FileHandleSyncPrototype;
  await probe.close();
  await rm(probePath, { force: true });
  return prototype;
}

function systemError(code: string): Error & { code: string } {
  return Object.assign(new Error(`Injected ${code}`), { code });
}
