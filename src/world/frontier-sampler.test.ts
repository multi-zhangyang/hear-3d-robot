import { describe, expect, it } from "vitest";
import type { Terrain } from "../domain/schema.js";
import { buildFrontierSample } from "./frontier-sampler.js";

const terrain: Terrain = {
  cell: 1,
  block: 1,
  columns: 7,
  rows: 7,
  heights: new Array<number>(49).fill(0)
};

const explored = new Set<number>();
for (let row = 2; row <= 4; row += 1) {
  for (let column = 2; column <= 4; column += 1) explored.add(row * 7 + column);
}

function sample(motionSeed: number) {
  return buildFrontierSample({
    terrain,
    here: { column: 3, row: 3 },
    isExplored: (index) => explored.has(index),
    motionSeed,
    worldRevision: 2,
    exploredCount: explored.size,
    robotYaw: 0
  });
}

describe("frontier motion entropy", () => {
  it("is stable inside one persisted motion stream", () => {
    expect(sample(1234)).toEqual(sample(1234));
  });

  it("changes equally useful route ordering without selecting a route", () => {
    const first = sample(1234);
    const second = sample(5678);
    expect(first.sampleId).not.toBe(second.sampleId);
    expect(first.candidates.map((candidate) => candidate.target))
      .not.toEqual(second.candidates.map((candidate) => candidate.target));
    expect(first.candidates.every((candidate) => candidate.choiceId.startsWith("frontier_")))
      .toBe(true);
  });

  it("keeps higher information gain ahead of lower gain", () => {
    const candidates = sample(42).candidates;
    for (let index = 1; index < candidates.length; index += 1) {
      expect(candidates[index - 1]!.unseen).toBeGreaterThanOrEqual(candidates[index]!.unseen);
    }
  });
});
