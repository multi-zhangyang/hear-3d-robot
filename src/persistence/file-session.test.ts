import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import type { AgentInputItem } from "@openai/agents";
import { FileSession } from "./file-session.js";

const atomicWriteControl = vi.hoisted(() => ({ failAfterPublish: false }));

vi.mock("./atomic-file.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./atomic-file.js")>();
  return {
    ...actual,
    writeTextAtomically: async (path: string, value: string) => {
      await actual.writeTextAtomically(path, value);
      if (atomicWriteControl.failAfterPublish) {
        atomicWriteControl.failAfterPublish = false;
        throw new Error("injected directory durability failure");
      }
    }
  };
});

describe("FileSession", () => {
  it("persists, limits, pops, and clears SDK input items", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-session-"));
    const path = join(directory, "scoped", "agent_1.json");
    const session = new FileSession(path, "run_1");
    const items: AgentInputItem[] = [
      { role: "user", content: "first" },
      { role: "user", content: "second" }
    ];
    await session.addItems(items);
    expect(await session.getItems(1)).toEqual([items[1]]);
    expect(await new FileSession(path, "run_1").getItems()).toEqual(items);
    expect(await session.popItem()).toEqual(items[1]);
    await session.replaceItems([{ role: "user", content: "compacted tail" }]);
    expect(await new FileSession(path, "run_1").getItems()).toEqual([
      { role: "user", content: "compacted tail" }
    ]);
    await session.clearSession();
    expect(await session.getItems()).toEqual([]);
  });

  it("rotates a 10k-turn SDK branch to a bounded hot tail and recovers it", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-long-session-"));
    const path = join(directory, "sessions", "agent_long.json");
    try {
      const session = new FileSession(path, "run_long");
      const history: AgentInputItem[] = Array.from({ length: 10_240 }, (_, index) => ({
        role: "user",
        content: `turn-${index}`
      }));
      await session.addItems(history);
      const hotTail = history.slice(-16);
      await session.replaceItems(hotTail);

      expect(await session.getItems()).toEqual(hotTail);
      expect(await new FileSession(path, "run_long").getItems()).toEqual(hotTail);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("invalidates its cache when a durability error follows a published rename", async () => {
    const directory = await mkdtemp(join(tmpdir(), "hear-session-cache-failure-"));
    const path = join(directory, "session.json");
    const session = new FileSession(path, "run_cache_failure");
    try {
      await session.addItems([{ role: "user", content: "old item" }]);
      atomicWriteControl.failAfterPublish = true;
      await expect(session.replaceItems([{ role: "user", content: "published item" }]))
        .rejects.toThrow("injected directory durability failure");
      await session.addItems([{ role: "user", content: "later item" }]);
      expect(await session.getItems()).toEqual([
        { role: "user", content: "published item" },
        { role: "user", content: "later item" }
      ]);
    } finally {
      atomicWriteControl.failAfterPublish = false;
      await rm(directory, { recursive: true, force: true });
    }
  });
});
