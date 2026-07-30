import { describe, expect, it } from "vitest";
import type { TerrainDefinition, VoxelWorldState } from "../types";
import {
  clearChaseCameraSightline,
  easeChaseCameraHeight,
  VoxelSurfaceHeightField,
  type CameraPoint,
  type SurfaceHeightField
} from "./chase-camera-clearance";

function stripField(height: (x: number, z: number) => number | null): SurfaceHeightField {
  return { cellSize: 1, heightAt: height };
}

const target: CameraPoint = { x: 4, y: 0.2, z: 0.5 };
const preferred: CameraPoint = { x: 0, y: 2.4, z: 0.5 };

describe("chase camera clearance", () => {
  it("leaves a clear preferred pose unchanged", () => {
    const result = clearChaseCameraSightline({
      preferred,
      target,
      terrain: stripField(() => 0),
      aspect: 16 / 9
    });

    expect(result).toEqual(preferred);
  });

  it("raises the camera when a high voxel crosses the sightline", () => {
    const result = clearChaseCameraSightline({
      preferred,
      target,
      terrain: stripField((x) => x < 1 ? 2.7 : 0),
      aspect: 16 / 9
    });

    expect(result.y).toBeGreaterThan(preferred.y);
  });

  it("ignores high terrain outside the sightline", () => {
    const result = clearChaseCameraSightline({
      preferred,
      target,
      terrain: stripField((_x, z) => z > 1 ? 8 : 0),
      aspect: 16 / 9
    });

    expect(result).toEqual(preferred);
  });

  it("keeps extra foreground clearance in a portrait viewport", () => {
    const terrain = stripField((x) => x < 1 ? 2.7 : 0);
    const landscape = clearChaseCameraSightline({
      preferred,
      target,
      terrain,
      aspect: 16 / 9
    });
    const portrait = clearChaseCameraSightline({
      preferred,
      target,
      terrain,
      aspect: 390 / 844
    });

    expect(portrait.y).toBeGreaterThan(landscape.y);
  });

  it("does not mutate inputs or keep lifting an already-cleared pose", () => {
    const originalPreferred = { ...preferred };
    const originalTarget = { ...target };
    const terrain = stripField((x) => x < 1 ? 2.7 : 0);
    const once = clearChaseCameraSightline({ preferred, target, terrain, aspect: 1 });
    const twice = clearChaseCameraSightline({ preferred: once, target, terrain, aspect: 1 });

    expect(preferred).toEqual(originalPreferred);
    expect(target).toEqual(originalTarget);
    expect(twice).toEqual(once);
  });

  it("caps an impossible near-wall solve so framing cannot run away", () => {
    const result = clearChaseCameraSightline({
      preferred,
      target,
      terrain: { cellSize: 1, maximumHeight: 100, heightAt: () => 100 },
      aspect: 16 / 9
    });

    expect(result.y).toBeLessThanOrEqual(preferred.y + 4.6);
  });

  it("converges across consecutive follow frames instead of resetting each frame", () => {
    const resolved = 4.6;
    let current = preferred.y;
    const heights: number[] = [];
    for (let frame = 0; frame < 14; frame += 1) {
      current = easeChaseCameraHeight(current, resolved);
      heights.push(current);
    }

    expect(heights.every((height, index) => index === 0 || height > heights[index - 1]!)).toBe(true);
    expect(heights.at(-1)).toBeCloseTo(resolved, 2);
  });
});

describe("voxel surface height field", () => {
  const terrain: TerrainDefinition = {
    cell: 1,
    columns: 2,
    rows: 1,
    block: 0.9,
    chunk_size: 2,
    maximum_height: 8,
    heights: [2, 0]
  };

  it("projects authoritative removals and placements without changing their data", () => {
    const state: VoxelWorldState = {
      version: 1,
      revision: 3,
      chunk_size: 2,
      load_radius_chunks: 1,
      loaded_chunks: [{ column: 0, row: 0 }],
      mutations: [
        {
          coordinate: { column: 0, level: 1, row: 0 },
          before: "grass",
          after: null,
          revision: 1,
          source_command_id: "remove",
          source_agent_id: "agent"
        },
        {
          coordinate: { column: 1, level: 4, row: 0 },
          before: null,
          after: "placed",
          revision: 3,
          source_command_id: "place",
          source_agent_id: "agent"
        }
      ],
      inventory: { grass: 0, dirt: 0, stone: 0, sand: 0, placed: 0 }
    };
    const original = structuredClone(state);
    const field = new VoxelSurfaceHeightField(terrain);

    field.update(state);

    expect(field.heightAt(0.5, 0.5)).toBeCloseTo(0.9);
    expect(field.heightAt(1.5, 0.5)).toBeCloseTo(4.5);
    expect(field.heightAt(-0.1, 0.5)).toBeNull();
    expect(state).toEqual(original);
  });
});
