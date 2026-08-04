import { mkdtemp, readFile, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { RunStore } from "./run-store.js";

describe("RunStore scenario chunk persistence", () => {
  it("writes and completely reparses the materialized manifest", async () => {
    const { store, scenario } = await createStore("persist");
    const raw = JSON.parse(await readFile(join(store.runDir, "run.json"), "utf8"));

    expect(raw.scenario.chunk_manifest).toEqual(scenario.chunk_manifest);
    const reopened = await RunStore.open(store.runDir);
    expect(reopened.definition.scenario.chunk_manifest).toEqual(scenario.chunk_manifest);
    expect(reopened.definition.scenario).toEqual(scenario);
  });

  it("deterministically upgrades a legacy run definition without a manifest", async () => {
    const { store, scenario } = await createStore("legacy");
    const path = join(store.runDir, "run.json");
    const raw = JSON.parse(await readFile(path, "utf8"));
    delete raw.scenario.chunk_manifest;
    await writeFile(path, `${JSON.stringify(raw, null, 2)}\n`, "utf8");

    const reopened = await RunStore.open(store.runDir);
    expect(reopened.definition.scenario.chunk_manifest).toEqual(scenario.chunk_manifest);
  });

  it("refuses a persisted manifest whose entity ownership was corrupted", async () => {
    const { store } = await createStore("corrupt");
    const path = join(store.runDir, "run.json");
    const raw = JSON.parse(await readFile(path, "utf8"));
    const owner = raw.scenario.chunk_manifest.chunks.find((chunk: {
      entity_ids: { objects: string[] };
    }) => chunk.entity_ids.objects.includes("courtyard_crate"));
    owner.entity_ids.objects = [];
    await writeFile(path, `${JSON.stringify(raw, null, 2)}\n`, "utf8");

    await expect(RunStore.open(store.runDir)).rejects.toThrow(
      "is missing from the chunk manifest"
    );
  });

  it("initializes a sparse V1 delta file and upgrades an old run when first changed", async () => {
    const { store } = await createStore("delta-legacy");
    const path = join(store.runDir, "chunk-deltas.json");
    const initial = await store.readScenarioChunkDeltaState();
    expect(initial).toMatchObject({ version: 1, revision: 0, chunks: [] });
    expect(JSON.parse(await readFile(path, "utf8"))).toEqual(initial);

    await unlink(path);
    const definitionPath = join(store.runDir, "run.json");
    const legacyDefinition = JSON.parse(await readFile(definitionPath, "utf8"));
    delete legacyDefinition.scenario.chunk_manifest;
    await writeFile(definitionPath, `${JSON.stringify(legacyDefinition, null, 2)}\n`, "utf8");
    const reopened = await RunStore.open(store.runDir);
    expect(await reopened.readScenarioChunkDeltaState()).toMatchObject({
      version: 1,
      revision: 0,
      chunks: []
    });
    const changed = await reopened.applyScenarioChunkDeltaMutation({
      type: "put_dynamic_entity",
      entity: courtyardCrate(15, 13)
    });

    expect(changed).toMatchObject({
      revision: 1,
      changed_chunk_ids: ["chunk_0_0", "chunk_1_1"]
    });
    expect(JSON.parse(await readFile(path, "utf8"))).toEqual(changed);
    expect(changed.chunks.flatMap(({ dynamic_entities }) => dynamic_entities))
      .toHaveLength(1);
    expect(changed.chunks.find(({ dynamic_entities }) => dynamic_entities.length > 0)?.chunk_id)
      .toBe("chunk_1_1");
  });

  it("serializes concurrent delta mutations without losing either entity", async () => {
    const { store } = await createStore("delta-concurrent");
    await Promise.all([
      store.applyScenarioChunkDeltaMutation({
        type: "create_block",
        block: placedBlock("placed:block-west", 2, 8)
      }),
      store.applyScenarioChunkDeltaMutation({
        type: "create_block",
        block: placedBlock("placed:block-east", 14, 8)
      })
    ]);

    const state = await store.readScenarioChunkDeltaState();
    expect(state.revision).toBe(2);
    expect(state.chunks.flatMap(({ blocks }) => blocks.map(({ id }) => id)).sort()).toEqual([
      "placed:block-east",
      "placed:block-west"
    ]);
  });

  it("commits a physical world batch with one durable chunk revision", async () => {
    const { store } = await createStore("delta-batch");
    const state = await store.applyScenarioChunkDeltaMutations([
      {
        type: "create_block",
        block: placedBlock("placed:block-west", 2, 8)
      },
      {
        type: "create_block",
        block: placedBlock("placed:block-east", 14, 8)
      },
      {
        type: "put_dynamic_entity",
        entity: courtyardCrate(15, 13)
      }
    ]);

    expect(state.revision).toBe(1);
    expect(state.changed_chunk_ids).toEqual([
      "chunk_0_0",
      "chunk_1_0",
      "chunk_1_1"
    ]);
    expect(state.chunks.every(({ revision }) => revision === 1)).toBe(true);
    expect(JSON.parse(await readFile(join(store.runDir, "chunk-deltas.json"), "utf8")))
      .toEqual(state);
  });

  it("keeps the last durable file byte-for-byte when a mutation is invalid", async () => {
    const { store } = await createStore("delta-atomic-reject");
    const path = join(store.runDir, "chunk-deltas.json");
    await store.applyScenarioChunkDeltaMutation({
      type: "create_block",
      block: placedBlock("placed:block-safe", 2, 8)
    });
    const before = await readFile(path, "utf8");

    await expect(store.applyScenarioChunkDeltaMutation({
      type: "remove_dynamic_entity",
      entity_id: "unknown_entity"
    })).rejects.toThrow("Unknown scenario entity");

    expect(await readFile(path, "utf8")).toBe(before);
    expect((await store.readScenarioChunkDeltaState()).revision).toBe(1);
  });

  it("hard-rejects duplicated entities in a corrupted delta file", async () => {
    const { store } = await createStore("delta-corrupt");
    const path = join(store.runDir, "chunk-deltas.json");
    await store.applyScenarioChunkDeltaMutation({
      type: "create_block",
      block: placedBlock("placed:block-duplicate", 2, 8)
    });
    const raw = JSON.parse(await readFile(path, "utf8"));
    raw.chunks[0].blocks.push(structuredClone(raw.chunks[0].blocks[0]));
    await writeFile(path, `${JSON.stringify(raw, null, 2)}\n`, "utf8");

    await expect(store.readScenarioChunkDeltaState()).rejects.toThrow(
      "entity records must be unique and sorted"
    );
  });
});

async function createStore(label: string) {
  const runsDir = await mkdtemp(join(tmpdir(), `hear-chunks-${label}-`));
  const catalog = await loadRuntimeCatalog();
  const scenario = catalog.materialize("humanoid_courtyard", 123);
  const store = await RunStore.create(runsDir, {
    mission: "Exercise durable scenario chunks",
    scenarioId: "humanoid_courtyard",
    scenario,
    goal: scenario.default_goal
  });
  return { store, scenario };
}

function placedBlock(id: string, x: number, z: number) {
  return {
    id,
    center: { x, y: 0.5, z },
    size: { x: 1, y: 1, z: 1 },
    material: "stone",
    properties: {}
  };
}

function courtyardCrate(x: number, z: number) {
  return {
    id: "courtyard_crate",
    kind: "crate",
    color: "#bd7844",
    position: { x, y: 0.25, z },
    rotation: { x: 0, y: 0, z: 0, w: 1 },
    linear_velocity: { x: 0, y: 0, z: 0 },
    angular_velocity: { x: 0, y: 0, z: 0 },
    size: { x: 0.5, y: 0.5, z: 0.5 },
    portable: true,
    properties: {}
  };
}
