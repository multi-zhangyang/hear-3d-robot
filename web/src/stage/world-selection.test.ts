import { describe, expect, it } from "vitest";
import type { WorldSnapshot } from "../types";
import {
  describeWorldSelection,
  sameWorldSelection,
  worldSelectionKey
} from "./world-selection";

describe("world selection", () => {
  it("keeps a voxel identity stable while detecting an authoritative material change", () => {
    const grass = {
      kind: "voxel" as const,
      coordinate: { column: 4, level: 2, row: 7 },
      material: "grass" as const
    };
    const stone = { ...grass, material: "stone" as const };

    expect(worldSelectionKey(grass)).toBe("voxel:4:2:7");
    expect(sameWorldSelection(grass, { ...grass })).toBe(true);
    expect(sameWorldSelection(grass, stone)).toBe(false);
  });

  it("describes a selected entity from the latest world snapshot", () => {
    const frame = {
      robot: { position: { x: 1, y: 0.4, z: 2 } },
      objects: [{ id: "red_block", kind: "block", position: { x: 8.25, y: 0.5, z: 9.75 } }],
      obstacles: [],
      zones: []
    } as unknown as WorldSnapshot;

    expect(describeWorldSelection({
      kind: "entity",
      entityType: "object",
      id: "red_block"
    }, frame)).toEqual({
      badge: "方块",
      title: "红色方块",
      detail: "8.3, 0.5, 9.8"
    });

    frame.objects[0]!.position.x = 10;
    expect(describeWorldSelection({
      kind: "entity",
      entityType: "object",
      id: "red_block"
    }, frame).detail).toBe("10.0, 0.5, 9.8");
  });
});
