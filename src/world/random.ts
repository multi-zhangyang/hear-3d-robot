/**
 * Seeded pseudo-randomness.
 *
 * Every source of variation in this project has to be replayable. A run stores
 * the world it was given, and resuming from a checkpoint has to rebuild exactly
 * that world — so nothing may call the platform's `Math.random`. Variation
 * comes from one integer chosen once per run and carried in the scenario; every
 * generator below is a pure function of that integer.
 */

/** A fast, well-distributed 32-bit generator. Returns values in [0, 1). */
export function createRandom(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

/**
 * Folds a label and an integer into a new seed.
 *
 * Different aspects of a run — terrain, the start pose, the order candidate
 * approach poses are offered in — must vary independently rather than moving in
 * lockstep, or two of them drawn from the same stream would correlate. Naming
 * each stream gives every one its own sequence from the same run seed.
 */
export function deriveSeed(seed: number, label: string): number {
  let hash = Math.imul(seed >>> 0, 0x85ebca6b) >>> 0;
  for (let index = 0; index < label.length; index += 1) {
    hash = Math.imul(hash ^ label.charCodeAt(index), 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** An integer in [minimum, maximum]. */
export function randomInteger(random: () => number, minimum: number, maximum: number): number {
  return minimum + Math.floor(random() * (maximum - minimum + 1));
}

/** A value in [minimum, maximum). */
export function randomBetween(random: () => number, minimum: number, maximum: number): number {
  return minimum + random() * (maximum - minimum);
}

/** Fisher-Yates, in place, so ordering is shuffled without bias. */
export function shuffle<T>(items: T[], random: () => number): T[] {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    const held = items[index]!;
    items[index] = items[swap]!;
    items[swap] = held;
  }
  return items;
}
