import type { Scenario } from "../../domain/schema.js";
import {
  createScenarioPhysicalWorldDelta,
  assertScenarioPhysicalWorldDeltaApplied,
  scenarioPhysicalWorldDeltaMutations,
  type ScenarioPhysicalDynamicEntity,
  type ScenarioPhysicalWorldDelta
} from "../../domain/scenario-physical-delta.js";
import {
  applyScenarioChunkDeltaMutations,
  materializeScenarioChunkDeltaState,
  resolveScenarioChunkDeltaEntity
} from "../../domain/scenario-chunk-delta.js";
import {
  restoreScenarioChunkDeltaState,
  type ScenarioChunkDeltaState
} from "../../domain/scenario-chunk-delta-schema.js";
import type { RunStore } from "../../persistence/run-store.js";
import type { HumanoidSimulationSnapshot } from "../../world/humanoid/simulation.js";

interface HumanoidPhysicalWorldFrame {
  frame: number;
  worldRevision: number;
  robot: Pick<HumanoidSimulationSnapshot, "objects">;
}

export function captureHumanoidPhysicalWorldDelta(input: {
  scenario: Scenario;
  chunks: ScenarioChunkDeltaState;
  world: HumanoidPhysicalWorldFrame;
  transactionId: string;
}): ScenarioPhysicalWorldDelta | undefined {
  const entities: ScenarioPhysicalDynamicEntity[] = [];
  for (const [id, physical] of Object.entries(input.world.robot.objects)) {
    const current = resolveScenarioChunkDeltaEntity(input.scenario, input.chunks, id);
    if (current.category !== "dynamic_entity"
      || !current.state.present
      || !current.state.portable) {
      throw new Error(`MuJoCo object is not an active portable scenario entity: ${id}`);
    }
    const candidate: ScenarioPhysicalDynamicEntity = {
      id,
      position: structuredClone(physical.position),
      rotation: structuredClone(physical.rotation),
      linear_velocity: structuredClone(physical.linearVelocity),
      angular_velocity: structuredClone(physical.angularVelocity),
      physical_authority: {
        source: "humanoid_mujoco",
        transaction_id: input.transactionId,
        world_frame: input.world.frame,
        world_revision: input.world.worldRevision
      }
    };
    if (!samePhysicalState(current.state, candidate)) entities.push(candidate);
  }
  if (entities.length === 0) return undefined;
  return createScenarioPhysicalWorldDelta({
    version: 1,
    transaction_id: input.transactionId,
    source_world_frame: input.world.frame,
    source_world_revision: input.world.worldRevision,
    base_chunk_revision: input.chunks.revision,
    entities
  });
}

export async function reconcileHumanoidPhysicalWorldDelta(
  store: RunStore,
  record: ScenarioPhysicalWorldDelta
): Promise<ScenarioChunkDeltaState> {
  const current = await store.readScenarioChunkDeltaState();
  const mutations = scenarioPhysicalWorldDeltaMutations(
    store.definition.scenario,
    current,
    record
  );
  const committed = mutations.length > 0
    ? await store.applyScenarioChunkDeltaMutations(mutations)
    : current;
  assertScenarioPhysicalWorldDeltaApplied(
    store.definition.scenario,
    committed,
    record
  );
  return committed;
}

export function projectHumanoidPhysicalWorldDelta(
  scenario: Scenario,
  chunks: ScenarioChunkDeltaState,
  record: ScenarioPhysicalWorldDelta
): ScenarioChunkDeltaState {
  return applyScenarioChunkDeltaMutations(
    scenario,
    chunks,
    scenarioPhysicalWorldDeltaMutations(scenario, chunks, record)
  );
}

export function assertHumanoidPhysicalWorldDeltaRecovery(input: {
  scenario: Scenario;
  chunks: ScenarioChunkDeltaState;
  world: HumanoidPhysicalWorldFrame;
}): void {
  const chunks = restoreScenarioChunkDeltaState(input.scenario, input.chunks);
  const activeScenario = materializeScenarioChunkDeltaState(input.scenario, chunks);
  const expectedPortableIds = activeScenario.objects
    .filter((object) => object.portable)
    .map((object) => object.id)
    .sort();
  const checkpointPortableIds = Object.keys(input.world.robot.objects).sort();
  if (JSON.stringify(expectedPortableIds) !== JSON.stringify(checkpointPortableIds)) {
    throw new Error(
      "Portable entity topology changed without a matching physical checkpoint migration"
    );
  }
  for (const chunk of chunks.chunks) {
    for (const entity of chunk.dynamic_entities) {
      const authority = entity.physical_authority;
      if (!authority) continue;
      if (authority.world_revision > input.world.worldRevision
        || authority.world_frame > input.world.frame) {
        throw new Error(
          `Durable scenario entity is ahead of the physical checkpoint: ${entity.id}`
        );
      }
      if (authority.world_revision !== input.world.worldRevision) continue;
      const physical = input.world.robot.objects[entity.id];
      if (!physical || !samePhysicalState(entity, {
        id: entity.id,
        position: physical.position,
        rotation: physical.rotation,
        linear_velocity: physical.linearVelocity,
        angular_velocity: physical.angularVelocity,
        physical_authority: authority
      })) {
        throw new Error(
          `Durable scenario entity conflicts with the physical checkpoint: ${entity.id}`
        );
      }
    }
  }
}

function samePhysicalState(
  current: {
    position: ScenarioPhysicalDynamicEntity["position"];
    rotation: ScenarioPhysicalDynamicEntity["rotation"];
    linear_velocity: ScenarioPhysicalDynamicEntity["linear_velocity"];
    angular_velocity: ScenarioPhysicalDynamicEntity["angular_velocity"];
  },
  expected: ScenarioPhysicalDynamicEntity
): boolean {
  return sameVector(current.position, expected.position)
    && sameQuaternion(current.rotation, expected.rotation)
    && sameVector(current.linear_velocity, expected.linear_velocity)
    && sameVector(current.angular_velocity, expected.angular_velocity);
}

function sameVector(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number }
): boolean {
  return left.x === right.x && left.y === right.y && left.z === right.z;
}

function sameQuaternion(
  left: { x: number; y: number; z: number; w: number },
  right: { x: number; y: number; z: number; w: number }
): boolean {
  return left.x === right.x && left.y === right.y
    && left.z === right.z && left.w === right.w;
}
