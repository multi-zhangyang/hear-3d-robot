/**
 * What the robot has seen.
 *
 * A world larger than the sensor is a world the agent has to explore, and
 * exploring only means something if the world remembers where the robot has
 * already been. Every column whose centre falls inside the sensor cone, within
 * range, and with clear line of sight is marked once and stays marked.
 *
 * The record is a bitmap over terrain columns because it belongs in every world
 * frame: a checkpoint is a frame, so resuming has to recover the whole frontier
 * rather than a delta, and a list of indices would outgrow the rest of the
 * frame within a few hundred cells.
 */
import type { Terrain, WorldSnapshot } from "../domain/schema.js";

export type ExploredState = WorldSnapshot["explored"];

export class ExplorationMap {
  readonly #bits: Uint8Array;
  readonly #total: number;
  #seen = 0;

  constructor(total: number) {
    this.#total = total;
    this.#bits = new Uint8Array(Math.ceil(total / 8));
  }

  static forTerrain(terrain: Terrain | undefined): ExplorationMap {
    return new ExplorationMap(terrain ? terrain.columns * terrain.rows : 0);
  }

  get total(): number {
    return this.#total;
  }

  get seen(): number {
    return this.#seen;
  }

  has(index: number): boolean {
    if (index < 0 || index >= this.#total) return false;
    return (this.#bits[index >> 3]! & (1 << (index & 7))) !== 0;
  }

  /** Marks a column seen. Returns whether this was the first time. */
  mark(index: number): boolean {
    if (index < 0 || index >= this.#total || this.has(index)) return false;
    this.#bits[index >> 3]! |= 1 << (index & 7);
    this.#seen += 1;
    return true;
  }

  state(): ExploredState {
    return {
      cells: encodeBits(this.#bits),
      seen: this.#seen,
      total: this.#total
    };
  }

  /** Rebuilds the frontier recorded in a checkpoint. */
  restore(state: ExploredState): void {
    const decoded = decodeBits(state.cells, this.#bits.length);
    this.#bits.set(decoded);
    this.#seen = 0;
    for (let index = 0; index < this.#total; index += 1) {
      if (this.has(index)) this.#seen += 1;
    }
  }
}

function encodeBits(bits: Uint8Array): string {
  return Buffer.from(bits).toString("base64");
}

function decodeBits(encoded: string, length: number): Uint8Array {
  const decoded = Buffer.from(encoded, "base64");
  const bits = new Uint8Array(length);
  bits.set(decoded.subarray(0, length));
  return bits;
}
