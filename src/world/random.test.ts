import { describe, expect, it } from "vitest";
import { createRandom, deriveSeed, randomBetween, randomInteger, shuffle } from "./random.js";

describe("seeded world randomness", () => {
  it("repeats exactly for the same seed and diverges for another seed", () => {
    const sequence = (seed: number): number[] => {
      const random = createRandom(seed);
      return Array.from({ length: 12 }, () => random());
    };

    expect(sequence(1_337)).toEqual(sequence(1_337));
    expect(sequence(1_337)).not.toEqual(sequence(1_338));
    expect(sequence(0)).toEqual(sequence(0));
    expect(sequence(1_337).every((value) => value >= 0 && value < 1)).toBe(true);
  });

  it("gives named streams independent deterministic seeds", () => {
    expect(deriveSeed(42, "terrain")).toBe(deriveSeed(42, "terrain"));
    expect(deriveSeed(42, "terrain")).not.toBe(deriveSeed(42, "placement"));
    expect(deriveSeed(42, "terrain")).not.toBe(deriveSeed(43, "terrain"));
  });

  it("keeps integer, floating-point, and shuffle results inside their contracts", () => {
    const random = createRandom(9_001);
    const integers = Array.from({ length: 200 }, () => randomInteger(random, -3, 4));
    const floats = Array.from({ length: 200 }, () => randomBetween(random, 2.5, 7.25));
    const shuffled = shuffle([1, 2, 3, 4, 5, 6], random);

    expect(integers.every((value) => Number.isInteger(value) && value >= -3 && value <= 4)).toBe(true);
    expect(floats.every((value) => value >= 2.5 && value < 7.25)).toBe(true);
    expect([...shuffled].sort((left, right) => left - right)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});
