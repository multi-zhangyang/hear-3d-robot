import type { Scenario } from "../../domain/schema.js";
import {
  materializeScenarioChunkDeltaState
} from "../../domain/scenario-chunk-delta.js";
import type {
  ScenarioChunkDeltaState
} from "../../domain/scenario-chunk-delta-schema.js";

export interface HumanoidScenarioSynchronization {
  scenario: Scenario;
  chunkRevision: number;
  changed: boolean;
  requiresResourceRebuild: boolean;
  changedDomains: Array<"geometry" | "objects" | "zones">;
}

export function analyzeHumanoidScenarioSynchronization(input: {
  current: Scenario;
  baseline: Scenario;
  chunks: ScenarioChunkDeltaState;
}): HumanoidScenarioSynchronization {
  const scenario = materializeScenarioChunkDeltaState(input.baseline, input.chunks);
  assertPortableTopologyCompatible(input.current, scenario);

  const geometryChanged = !sameJson(
    physicalGeometryProjection(input.current),
    physicalGeometryProjection(scenario)
  );
  const objectsChanged = !sameJson(
    objectDescriptorProjection(input.current),
    objectDescriptorProjection(scenario)
  );
  const zonesChanged = !sameJson(input.current.zones, scenario.zones);
  const changedDomains: HumanoidScenarioSynchronization["changedDomains"] = [];
  if (geometryChanged) changedDomains.push("geometry");
  if (objectsChanged) changedDomains.push("objects");
  if (zonesChanged) changedDomains.push("zones");
  return {
    scenario,
    chunkRevision: input.chunks.revision,
    changed: changedDomains.length > 0,
    requiresResourceRebuild: geometryChanged,
    changedDomains
  };
}

function assertPortableTopologyCompatible(current: Scenario, next: Scenario): void {
  const currentPortable = current.objects
    .filter(({ portable }) => portable)
    .map(({ id }) => id);
  const nextPortable = next.objects
    .filter(({ portable }) => portable)
    .map(({ id }) => id);
  if (!sameJson(currentPortable, nextPortable)) {
    throw new Error(
      "Portable entity topology requires an explicit physical checkpoint migration"
    );
  }
}

function physicalGeometryProjection(scenario: Scenario): unknown {
  return {
    obstacles: scenario.obstacles,
    objects: scenario.objects.map((object) => ({
      id: object.id,
      size: object.size,
      portable: object.portable,
      ...(object.portable ? {} : { position: object.position })
    }))
  };
}

function objectDescriptorProjection(scenario: Scenario): unknown {
  return scenario.objects.map((object) => ({
    id: object.id,
    kind: object.kind,
    color: object.color,
    size: object.size,
    portable: object.portable
  }));
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}
