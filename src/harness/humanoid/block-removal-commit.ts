import {
  assertScenarioBlockRemovalApplied,
  projectScenarioBlockRemoval,
  scenarioBlockRemovalMutations,
  type ScenarioBlockRemovalTransaction
} from "../../domain/scenario-block-removal.js";
import type { ScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
import type { RunStore } from "../../persistence/run-store.js";

export async function reconcileHumanoidBlockRemoval(
  store: RunStore,
  transaction: ScenarioBlockRemovalTransaction
): Promise<ScenarioChunkDeltaState> {
  const current = await store.readScenarioChunkDeltaState();
  const mutations = scenarioBlockRemovalMutations(
    store.definition.scenario,
    current,
    transaction
  );
  const committed = mutations.length > 0
    ? await store.applyScenarioChunkDeltaMutations(mutations)
    : current;
  assertScenarioBlockRemovalApplied(
    store.definition.scenario,
    committed,
    transaction
  );
  return committed;
}

export function projectHumanoidBlockRemoval(
  store: Pick<RunStore, "definition">,
  chunks: ScenarioChunkDeltaState,
  transaction: ScenarioBlockRemovalTransaction
): ScenarioChunkDeltaState {
  return projectScenarioBlockRemoval(
    store.definition.scenario,
    chunks,
    transaction
  );
}
