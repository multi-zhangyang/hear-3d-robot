import { describe, expect, it } from "vitest";
import { ExplorationMap } from "./exploration.js";

describe("terrain exploration bitmap", () => {
  it("marks each cell once and restores exactly from a checkpoint", () => {
    const map = new ExplorationMap(37);
    for (const index of [0, 1, 7, 8, 16, 31, 36]) {
      expect(map.mark(index)).toBe(true);
      expect(map.mark(index)).toBe(false);
    }
    expect(map.mark(-1)).toBe(false);
    expect(map.mark(37)).toBe(false);

    const persisted = map.state();
    const restored = new ExplorationMap(37);
    restored.restore(persisted);

    expect(restored.state()).toEqual(persisted);
    expect(restored.seen).toBe(7);
    for (let index = 0; index < 37; index += 1) {
      expect(restored.has(index)).toBe([0, 1, 7, 8, 16, 31, 36].includes(index));
    }
  });

  it("does not trust a stale persisted seen counter", () => {
    const source = new ExplorationMap(10);
    source.mark(2);
    source.mark(9);
    const state = source.state();
    const restored = new ExplorationMap(10);
    restored.restore({ ...state, seen: 999 });

    expect(restored.seen).toBe(2);
    expect(restored.state().seen).toBe(2);
  });
});
