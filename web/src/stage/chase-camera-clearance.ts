import type { TerrainDefinition, VoxelWorldState } from "../types";

export interface CameraPoint {
  x: number;
  y: number;
  z: number;
}

export interface SurfaceHeightField {
  readonly cellSize: number;
  readonly maximumHeight?: number;
  heightAt(x: number, z: number): number | null;
}

export interface ChaseCameraClearance {
  preferred: CameraPoint;
  target: CameraPoint;
  terrain: SurfaceHeightField;
  aspect: number;
  targetRadius?: number;
}

/**
 * A compact height projection of the same baseline and mutations rendered by
 * VoxelTerrain. Updating it is revision-gated, so following the robot does not
 * rebuild the field on every interpolated frame.
 */
export class VoxelSurfaceHeightField implements SurfaceHeightField {
  readonly cellSize: number;
  readonly #terrain: TerrainDefinition;
  readonly #baseline: Float32Array;
  #surface: Float32Array;
  #revision: number | null | undefined;
  #maximumHeight: number;

  constructor(terrain: TerrainDefinition) {
    this.#terrain = terrain;
    this.cellSize = terrain.cell;
    this.#baseline = Float32Array.from(
      terrain.heights,
      (height) => Math.max(0, height) * terrain.block
    );
    this.#surface = this.#baseline.slice();
    this.#maximumHeight = maximumOf(this.#surface);
  }

  get maximumHeight(): number {
    return this.#maximumHeight;
  }

  update(state: VoxelWorldState | null | undefined): void {
    const revision = state?.revision ?? null;
    if (revision === this.#revision) return;
    this.#revision = revision;
    this.#surface = this.#baseline.slice();
    if (!state || state.mutations.length === 0) {
      this.#maximumHeight = maximumOf(this.#surface);
      return;
    }

    const mutationsByCell = new Map<number, Map<number, boolean>>();
    for (const mutation of state.mutations) {
      const { column, level, row } = mutation.coordinate;
      if (
        column < 0 || column >= this.#terrain.columns
        || row < 0 || row >= this.#terrain.rows
        || level < 0
      ) continue;
      const cell = row * this.#terrain.columns + column;
      const levels = mutationsByCell.get(cell) ?? new Map<number, boolean>();
      levels.set(level, mutation.after !== null);
      mutationsByCell.set(cell, levels);
    }

    for (const [cell, levels] of mutationsByCell) {
      const baselineHeight = Math.max(0, this.#terrain.heights[cell] ?? 0);
      const highestCandidate = Math.max(baselineHeight - 1, ...levels.keys());
      let highestOccupied = -1;
      for (let level = highestCandidate; level >= 0; level -= 1) {
        const occupied = levels.has(level) ? levels.get(level) === true : level < baselineHeight;
        if (!occupied) continue;
        highestOccupied = level;
        break;
      }
      this.#surface[cell] = (highestOccupied + 1) * this.#terrain.block;
    }
    this.#maximumHeight = maximumOf(this.#surface);
  }

  heightAt(x: number, z: number): number | null {
    const column = Math.floor(x / this.#terrain.cell);
    const row = Math.floor(z / this.#terrain.cell);
    if (
      column < 0 || column >= this.#terrain.columns
      || row < 0 || row >= this.#terrain.rows
    ) return null;
    return this.#surface[row * this.#terrain.columns + column] ?? null;
  }
}

/**
 * Raises a chase camera only as far as the real voxel surface requires. The
 * solve always starts from the preferred pose, making repeated calls
 * idempotent rather than accumulating height.
 */
export function clearChaseCameraSightline({
  preferred,
  target,
  terrain,
  aspect,
  targetRadius
}: ChaseCameraClearance): CameraPoint {
  const horizontalDistance = Math.hypot(preferred.x - target.x, preferred.z - target.z);
  if (horizontalDistance < 1e-6) return { ...preferred };

  const cellSize = Math.max(0.05, terrain.cellSize);
  const protectedRadius = Math.max(targetRadius ?? 0, cellSize * 0.72);
  const start = Math.min(0.82, protectedRadius / horizontalDistance);
  const samples = Math.max(2, Math.ceil(horizontalDistance / (cellSize * 0.2)));
  const margin = cellSize * (aspect < 0.72 ? 0.78 : 0.28);
  let requiredCameraY = preferred.y;

  for (let index = 1; index <= samples; index += 1) {
    const progress = index / samples;
    if (progress < start) continue;
    const x = target.x + (preferred.x - target.x) * progress;
    const z = target.z + (preferred.z - target.z) * progress;
    const surface = terrain.heightAt(x, z);
    if (surface === null) continue;
    const requiredAtSample = surface + margin;
    const cameraY = target.y + (requiredAtSample - target.y) / progress;
    requiredCameraY = Math.max(requiredCameraY, cameraY);
  }

  const framingLiftLimit = Math.max(cellSize * 2.2, horizontalDistance * 1.15);
  const worldLiftLimit = terrain.maximumHeight === undefined
    ? framingLiftLimit
    : Math.max(cellSize * 2.2, terrain.maximumHeight + margin - target.y);
  const maximumY = preferred.y + Math.min(framingLiftLimit, worldLiftLimit);
  return { x: preferred.x, y: Math.min(requiredCameraY, maximumY), z: preferred.z };
}

export function easeChaseCameraHeight(current: number, resolved: number): number {
  const rate = resolved > current ? 0.38 : 0.2;
  return current + (resolved - current) * rate;
}

function maximumOf(values: Float32Array): number {
  let maximum = 0;
  for (const value of values) maximum = Math.max(maximum, value);
  return maximum;
}
