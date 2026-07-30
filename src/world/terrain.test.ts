import { describe, expect, it } from "vitest";
import type { Terrain } from "../domain/schema.js";
import { generateTerrain, terrainBoxes, walkableCells } from "./terrain.js";

describe("voxel terrain", () => {
  it("is deterministic, bounded, and leaves one connected walkable region", () => {
    for (const seed of [0, 1, 73, 91_337]) {
      const terrain = generateTerrain({
        size: 48,
        cell: 1,
        block: 0.9,
        relief: 4,
        density: 0.3
      }, seed);
      expect(terrain).toEqual(generateTerrain({
        size: 48,
        cell: 1,
        block: 0.9,
        relief: 4,
        density: 0.3
      }, seed));
      expect(terrain.heights).toHaveLength(48 * 48);
      expect(terrain.heights.every((height) => height >= 0 && height <= 4)).toBe(true);

      const walkable = walkableCells(terrain);
      expect(walkable.length).toBeGreaterThan(48 * 48 * 0.4);
      expect(connectedFloorCount(terrain, walkable[0]!)).toBe(walkable.length);

      for (let index = 0; index < 48; index += 1) {
        expect(terrain.heights[index]).toBe(4);
        expect(terrain.heights[(47 * 48) + index]).toBe(4);
        expect(terrain.heights[index * 48]).toBe(4);
        expect(terrain.heights[(index * 48) + 47]).toBe(4);
      }
    }
  });

  it("tracks the requested interior density without degenerating into empty or solid noise", () => {
    const fractions = [7, 8, 9, 10].map((seed) => {
      const terrain = generateTerrain({
        size: 64,
        cell: 1,
        block: 1,
        relief: 3,
        density: 0.28
      }, seed);
      let raised = 0;
      let total = 0;
      for (let row = 1; row < terrain.rows - 1; row += 1) {
        for (let column = 1; column < terrain.columns - 1; column += 1) {
          total += 1;
          if (terrain.heights[row * terrain.columns + column]! > 0) raised += 1;
        }
      }
      return raised / total;
    });

    expect(fractions.every((fraction) => fraction >= 0.24 && fraction <= 0.42)).toBe(true);
    expect(new Set(fractions.map((fraction) => fraction.toFixed(4))).size).toBeGreaterThan(1);
  });

  it("merges boxes without adding, dropping, or changing a voxel", () => {
    const terrain: Terrain = {
      cell: 0.5,
      block: 0.75,
      columns: 5,
      rows: 4,
      heights: [
        1, 1, 0, 2, 2,
        1, 1, 0, 2, 2,
        0, 0, 3, 3, 2,
        4, 0, 3, 3, 0
      ]
    };
    const boxes = terrainBoxes(terrain);
    const reconstructed = new Array<number>(terrain.heights.length).fill(0);

    for (const box of boxes) {
      const startColumn = Math.round((box.center.x - box.size.x / 2) / terrain.cell);
      const startRow = Math.round((box.center.z - box.size.z / 2) / terrain.cell);
      const width = Math.round(box.size.x / terrain.cell);
      const depth = Math.round(box.size.z / terrain.cell);
      const height = Math.round(box.size.y / terrain.block);
      expect(box.center.y).toBeCloseTo(box.size.y / 2, 10);
      for (let row = startRow; row < startRow + depth; row += 1) {
        for (let column = startColumn; column < startColumn + width; column += 1) {
          const index = row * terrain.columns + column;
          expect(reconstructed[index]).toBe(0);
          reconstructed[index] = height;
        }
      }
    }

    expect(reconstructed).toEqual(terrain.heights);
    const voxelVolume = terrain.heights.reduce((sum, height) => sum + height, 0)
      * terrain.cell * terrain.cell * terrain.block;
    const boxVolume = boxes.reduce((sum, box) => sum + box.size.x * box.size.y * box.size.z, 0);
    expect(boxVolume).toBeCloseTo(voxelVolume, 10);
    expect(boxes.length).toBeLessThan(terrain.heights.filter((height) => height > 0).length);
  });
});

function connectedFloorCount(terrain: Terrain, start: { column: number; row: number }): number {
  const visited = new Set<number>();
  const queue = [start.row * terrain.columns + start.column];
  while (queue.length > 0) {
    const current = queue.pop()!;
    if (visited.has(current)) continue;
    visited.add(current);
    const column = current % terrain.columns;
    const row = (current - column) / terrain.columns;
    for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const nextColumn = column + dx;
      const nextRow = row + dz;
      if (nextColumn < 0 || nextRow < 0
        || nextColumn >= terrain.columns || nextRow >= terrain.rows) continue;
      const next = nextRow * terrain.columns + nextColumn;
      if (terrain.heights[next] === 0 && !visited.has(next)) queue.push(next);
    }
  }
  return visited.size;
}
