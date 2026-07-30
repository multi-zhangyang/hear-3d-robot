import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { NavigationMesh } from "./navigation.js";
import type { TerrainBox } from "./terrain.js";

describe("NavigationMesh build bounds", () => {
  it("builds a scoped mesh when the world contains more solids than the call stack accepts", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 0);
    const outsideScope: TerrainBox = {
      center: { x: 1_000, y: 2, z: 1_000 },
      size: { x: 1, y: 4, z: 1 }
    };
    const terrainSolids = Array<TerrainBox>(150_000).fill(outsideScope);

    const navigation = await NavigationMesh.create(scenario, {
      region: {
        minimum: { x: 0, z: 0 },
        maximum: { x: 1, z: 1 }
      },
      terrainSolids
    });

    navigation.dispose();
    expect(terrainSolids).toHaveLength(150_000);
  }, 30_000);
});
