import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { humanoidEnvironment } from "./environment.js";
import {
  humanoidPhysicalRegion,
  humanoidPhysicalRegionIncludesBox
} from "./physical-region.js";

const SCENARIO = ScenarioSchema.parse({
  title: "Large physical region",
  seed: 29,
  bounds: { width: 60, depth: 60 },
  visibility_radius: 8,
  robot: { x: 5, z: 5, yaw: 0 },
  obstacles: [
    {
      id: "local-block",
      center: { x: 6, y: 1, z: 6 },
      size: { x: 2, y: 2, z: 2 }
    },
    {
      id: "long-wall",
      center: { x: 30, y: 1, z: 18 },
      size: { x: 56, y: 2, z: 1 }
    },
    {
      id: "remote-block",
      center: { x: 52, y: 1, z: 52 },
      size: { x: 2, y: 2, z: 2 }
    }
  ],
  objects: [
    {
      id: "near-fixed",
      kind: "crate",
      color: "#777777",
      position: { x: 8, y: 0.5, z: 8 },
      size: { x: 1, y: 1, z: 1 },
      portable: false
    },
    {
      id: "far-fixed",
      kind: "crate",
      color: "#777777",
      position: { x: 52, y: 0.5, z: 48 },
      size: { x: 1, y: 1, z: 1 },
      portable: false
    },
    {
      id: "portable",
      kind: "crate",
      color: "#aa7744",
      position: { x: 50, y: 0.25, z: 50 },
      size: { x: 0.5, y: 0.5, z: 0.5 },
      portable: true
    }
  ],
  zones: [],
  default_goal: {
    summary: "Explore",
    predicates: [{
      type: "robot_at",
      target: { x: 50, y: 0, z: 50 },
      tolerance: 0.4
    }]
  }
});

describe("humanoid physical region", () => {
  it("keeps a one-chunk collision margin and changes only at chunk boundaries", () => {
    const initial = humanoidPhysicalRegion(SCENARIO, [{ x: 5, z: 5 }]);
    const sameChunk = humanoidPhysicalRegion(SCENARIO, [{ x: 11.9, z: 5 }]);
    const nextChunk = humanoidPhysicalRegion(SCENARIO, [{ x: 12.1, z: 5 }]);

    expect(initial.chunkIds).toEqual([
      "chunk_0_0",
      "chunk_1_0",
      "chunk_0_1",
      "chunk_1_1"
    ]);
    expect(sameChunk.key).toBe(initial.key);
    expect(nextChunk.key).not.toBe(initial.key);
    expect(nextChunk.chunkIds).toContain("chunk_2_0");
  });

  it("supports disjoint robot and moving-object regions without filling between them", () => {
    const region = humanoidPhysicalRegion(SCENARIO, [
      { x: 5, z: 5 },
      { x: 52, z: 52 }
    ]);

    expect(region.anchorChunkIds).toEqual(["chunk_0_0", "chunk_4_4"]);
    expect(region.chunkIds).toHaveLength(8);
    expect(region.chunkIds).not.toContain("chunk_2_2");
  });

  it("loads intersecting static geometry while retaining portable physics state", () => {
    const region = humanoidPhysicalRegion(SCENARIO, [{ x: 5, z: 5 }]);
    const environment = humanoidEnvironment(SCENARIO, region);

    expect(environment.solids?.map(({ id }) => id)).toEqual([
      "local-block",
      "long-wall",
      "object-near-fixed"
    ]);
    expect(environment.objects?.map(({ id }) => id)).toEqual(["portable"]);
    expect(humanoidPhysicalRegionIncludesBox(region, SCENARIO.obstacles[2]!)).toBe(false);
  });
});
