import type { HumanoidWorldSnapshot } from "../types";

const DEFAULT_CAPACITY = 120;
const MAXIMUM_VISUAL_LAG_SECONDS = 0.22;
const CATCH_UP_DELAY_SECONDS = 0.04;
const EPSILON = 1e-6;

type Listener = () => void;

export class HumanoidFrameBuffer {
  readonly #capacity: number;
  readonly #listeners = new Set<Listener>();
  #frames: HumanoidWorldSnapshot[] = [];
  #playhead: number | null = null;
  #lastWallTime: number | null = null;
  #version = 0;

  constructor(capacity = DEFAULT_CAPACITY) {
    if (!Number.isSafeInteger(capacity) || capacity < 2) {
      throw new Error("Humanoid frame capacity must be at least two");
    }
    this.#capacity = capacity;
  }

  get latest(): HumanoidWorldSnapshot | null {
    return this.#frames.at(-1) ?? null;
  }

  get version(): number {
    return this.#version;
  }

  get pending(): boolean {
    const latest = this.latest;
    return latest !== null && this.#playhead !== null
      && latest.robot.simulatedTime - this.#playhead > EPSILON;
  }

  reset(frame: HumanoidWorldSnapshot): void {
    assertFrame(frame);
    this.#frames = [frame];
    this.#playhead = frame.robot.simulatedTime;
    this.#lastWallTime = null;
    this.#changed();
  }

  clear(): void {
    if (this.#frames.length === 0) return;
    this.#frames = [];
    this.#playhead = null;
    this.#lastWallTime = null;
    this.#changed();
  }

  push(incoming: readonly HumanoidWorldSnapshot[]): number {
    const ordered = incoming
      .filter(validFrame)
      .sort((left, right) => left.frame - right.frame
        || left.robot.simulatedTime - right.robot.simulatedTime);
    let accepted = 0;
    for (const frame of ordered) {
      const latest = this.latest;
      if (!latest) {
        this.#frames.push(frame);
        this.#playhead = frame.robot.simulatedTime;
        accepted += 1;
        continue;
      }
      if (frame.frame < latest.frame
        || frame.robot.simulatedTime + EPSILON < latest.robot.simulatedTime) continue;
      if (frame.frame === latest.frame) this.#frames[this.#frames.length - 1] = frame;
      else this.#frames.push(frame);
      accepted += 1;
    }
    if (accepted === 0) return 0;
    if (this.#frames.length > this.#capacity) {
      this.#frames.splice(0, this.#frames.length - this.#capacity);
    }
    this.#lastWallTime = null;
    this.#changed();
    return accepted;
  }

  sample(wallTimeMs: number, live: boolean): HumanoidWorldSnapshot | null {
    const latest = this.latest;
    if (!latest) return null;
    if (!Number.isFinite(wallTimeMs)) throw new Error("Frame sample time must be finite");
    if (!live) {
      this.#frames = [latest];
      this.#playhead = latest.robot.simulatedTime;
      this.#lastWallTime = wallTimeMs;
      return latest;
    }
    this.#playhead ??= this.#frames[0]!.robot.simulatedTime;
    if (this.#lastWallTime !== null) {
      this.#playhead = Math.min(
        latest.robot.simulatedTime,
        this.#playhead + Math.max(0, wallTimeMs - this.#lastWallTime) / 1000
      );
    }
    this.#lastWallTime = wallTimeMs;
    if (latest.robot.simulatedTime - this.#playhead > MAXIMUM_VISUAL_LAG_SECONDS) {
      this.#playhead = Math.max(
        this.#frames[0]!.robot.simulatedTime,
        latest.robot.simulatedTime - CATCH_UP_DELAY_SECONDS
      );
    }
    let selected = this.#frames[0]!;
    let selectedIndex = 0;
    for (let index = 1; index < this.#frames.length; index += 1) {
      const candidate = this.#frames[index]!;
      if (candidate.robot.simulatedTime > this.#playhead + EPSILON) {
        if (candidate.robot.simulatedTime - this.#playhead <= CATCH_UP_DELAY_SECONDS + EPSILON) {
          selected = candidate;
          selectedIndex = index;
          this.#playhead = candidate.robot.simulatedTime;
        }
        break;
      }
      selected = candidate;
      selectedIndex = index;
    }
    if (selectedIndex > 0) this.#frames.splice(0, selectedIndex);
    return selected;
  }

  subscribe(listener: Listener): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  #changed(): void {
    this.#version += 1;
    for (const listener of [...this.#listeners]) listener();
  }
}

function validFrame(value: HumanoidWorldSnapshot): boolean {
  return Number.isFinite(value.frame)
    && value.frame >= 0
    && Number.isFinite(value.worldRevision)
    && value.worldRevision >= 0
    && Number.isFinite(value.robot.simulatedTime)
    && value.robot.simulatedTime >= 0;
}

function assertFrame(value: HumanoidWorldSnapshot): void {
  if (!validFrame(value)) throw new Error("Invalid authoritative humanoid frame");
}
