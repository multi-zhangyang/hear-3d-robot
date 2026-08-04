import { describe, expect, it } from "vitest";
import { DEFAULT_SCENARIO_CHUNK_SIZE } from "./scenario-chunk.js";
import { ScenarioSchema, type Scenario } from "./schema.js";

const legacyScenario = {
  title: "Chunk integrity fixture",
  seed: 7,
  bounds: { width: 25, depth: 19 },
  visibility_radius: 8,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [
    {
      id: "west_wall",
      center: { x: 4, y: 1, z: 4 },
      size: { x: 1, y: 2, z: 1 }
    },
    {
      id: "east_wall",
      center: { x: 20, y: 1, z: 4 },
      size: { x: 1, y: 2, z: 1 }
    }
  ],
  objects: [{
    id: "supply_crate",
    kind: "crate",
    color: "#a46d3c",
    position: { x: 13, y: 0.25, z: 13 },
    size: { x: 0.5, y: 0.5, z: 0.5 },
    portable: true
  }],
  zones: [{
    id: "arrival",
    color: "#55aa88",
    center: { x: 24, y: 0.01, z: 18 },
    size: { x: 1, y: 0.02, z: 1 }
  }],
  default_goal: {
    summary: "Reach the arrival zone.",
    predicates: [{
      type: "robot_in_zone" as const,
      zone_id: "arrival",
      tolerance: 0.2
    }]
  }
};

describe("Scenario chunk manifest schema", () => {
  it("upgrades a legacy scenario with the canonical versioned grid", () => {
    const parsed = ScenarioSchema.parse(legacyScenario);

    expect(parsed.chunk_manifest).toMatchObject({
      version: 1,
      chunk_size: DEFAULT_SCENARIO_CHUNK_SIZE,
      grid: { columns: 3, rows: 2 }
    });
    expect(parsed.chunk_manifest.chunks.map(({ id, coordinate, bounds }) => ({
      id,
      coordinate,
      bounds
    }))).toEqual([
      chunk("chunk_0_0", 0, 0, 0, 0, 12, 12),
      chunk("chunk_1_0", 1, 0, 12, 0, 24, 12),
      chunk("chunk_2_0", 2, 0, 24, 0, 25, 12),
      chunk("chunk_0_1", 0, 1, 0, 12, 12, 19),
      chunk("chunk_1_1", 1, 1, 12, 12, 24, 19),
      chunk("chunk_2_1", 2, 1, 24, 12, 25, 19)
    ]);
    expect(entityOwners(parsed)).toEqual({
      east_wall: "chunk_1_0",
      west_wall: "chunk_0_0",
      supply_crate: "chunk_1_1",
      arrival: "chunk_2_1"
    });
    expect(ScenarioSchema.parse(parsed)).toEqual(parsed);
  });

  it("keeps chunk identity and bounds stable when entities move or the seed changes", () => {
    const first = ScenarioSchema.parse(legacyScenario);
    const moved = ScenarioSchema.parse({
      ...legacyScenario,
      seed: 8,
      objects: legacyScenario.objects.map((object) => ({
        ...object,
        position: { ...object.position, x: 2, z: 16 }
      }))
    });

    expect(moved.chunk_manifest.chunks.map(({ id, coordinate, bounds }) => ({
      id,
      coordinate,
      bounds
    }))).toEqual(first.chunk_manifest.chunks.map(({ id, coordinate, bounds }) => ({
      id,
      coordinate,
      bounds
    })));
    expect(entityOwners(first).supply_crate).toBe("chunk_1_1");
    expect(entityOwners(moved).supply_crate).toBe("chunk_0_1");
  });

  it.each([
    ["missing entity", (scenario: Scenario) => {
      owner(scenario, "supply_crate").entity_ids.objects = [];
    }, "is missing from the chunk manifest"],
    ["duplicate entity", (scenario: Scenario) => {
      owner(scenario, "supply_crate").entity_ids.objects.push("supply_crate");
    }, "must be unique and sorted"],
    ["wrong owner", (scenario: Scenario) => {
      owner(scenario, "supply_crate").entity_ids.objects = [];
      scenario.chunk_manifest.chunks[0]!.entity_ids.objects.push("supply_crate");
    }, "must belong to chunk_1_1"],
    ["unknown entity", (scenario: Scenario) => {
      scenario.chunk_manifest.chunks[0]!.entity_ids.zones.push("unknown_zone");
    }, "references unknown zones entity"],
    ["wrong bounds", (scenario: Scenario) => {
      scenario.chunk_manifest.chunks[0]!.bounds.maximum.x = 11;
    }, "bounds do not match"],
    ["unstable ID", (scenario: Scenario) => {
      scenario.chunk_manifest.chunks[0]!.id = "seed_specific_chunk";
    }, "must use stable ID chunk_0_0"],
    ["noncanonical order", (scenario: Scenario) => {
      scenario.chunk_manifest.chunks.reverse();
    }, "canonical row-major order"]
  ])("rejects a manifest with %s", (_label, mutate, expected) => {
    const scenario = structuredClone(ScenarioSchema.parse(legacyScenario));
    mutate(scenario);
    const result = ScenarioSchema.safeParse(scenario);

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues.map(({ message }) => message).join("\n")).toContain(expected);
    }
  });

  it("rejects an entity anchor outside the world instead of silently clamping ownership", () => {
    const result = ScenarioSchema.safeParse({
      ...legacyScenario,
      objects: legacyScenario.objects.map((object) => ({
        ...object,
        position: { ...object.position, x: 26 }
      }))
    });

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues.map(({ message }) => message).join("\n"))
        .toContain("anchor is outside the world bounds");
    }
  });
});

function chunk(
  id: string,
  column: number,
  row: number,
  minimumX: number,
  minimumZ: number,
  maximumX: number,
  maximumZ: number
) {
  return {
    id,
    coordinate: { column, row },
    bounds: {
      minimum: { x: minimumX, z: minimumZ },
      maximum: { x: maximumX, z: maximumZ }
    }
  };
}

function entityOwners(scenario: Scenario): Record<string, string> {
  const result: Record<string, string> = {};
  for (const chunk of scenario.chunk_manifest.chunks) {
    for (const category of ["obstacles", "objects", "zones"] as const) {
      for (const id of chunk.entity_ids[category]) result[id] = chunk.id;
    }
  }
  return result;
}

function owner(scenario: Scenario, entityId: string) {
  const result = scenario.chunk_manifest.chunks.find((chunk) => (
    Object.values(chunk.entity_ids).some((ids) => ids.includes(entityId))
  ));
  if (!result) throw new Error(`Missing fixture owner for ${entityId}`);
  return result;
}
