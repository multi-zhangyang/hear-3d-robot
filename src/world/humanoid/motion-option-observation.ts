import type { Scenario } from "../../domain/schema.js";
import { HumanoidGraspRegistry } from "./grasp-registry.js";
import {
  detectHumanoidMotionOption,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionDetection,
  type HumanoidMotionOptionDetectorInput,
  type HumanoidMotionOptionObservableObject
} from "./motion-option.js";
import type {
  HumanoidSimulation,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import { observableHumanoidSolidIds } from "./solid-observation.js";
import { humanoidObjectCapability } from "./object-capability.js";

interface HumanoidMotionOptionObservation {
  contract: HumanoidMotionOptionContract;
  scenario: Scenario;
}

export interface HumanoidSimulationOptionObservationInput {
  simulation: HumanoidSimulation;
  snapshot: HumanoidSimulationSnapshot;
  option: HumanoidMotionOptionObservation;
  boundObjectIds?: ReadonlySet<string> | undefined;
  graspRegistry?: HumanoidGraspRegistry | undefined;
  worldFrame?: number | undefined;
  trackedObjectIds?: ReadonlySet<string> | undefined;
}

export function detectHumanoidMotionOptionFromSimulation(
  input: HumanoidSimulationOptionObservationInput
): HumanoidMotionOptionDetection {
  return detectHumanoidMotionOption(
    input.option.contract,
    humanoidMotionOptionDetectorInputFromSimulation(input)
  );
}

export function humanoidMotionOptionDetectorInputFromSimulation(
  input: HumanoidSimulationOptionObservationInput
): HumanoidMotionOptionDetectorInput {
  const visible = input.simulation.senseObjects(
    input.option.scenario.visibility_radius
  ).objects;
  const sensedSolids = input.simulation.senseSolids(
    input.option.scenario.visibility_radius
  );
  const trackedObjectIds = input.trackedObjectIds ?? new Set<string>();
  const observableObjects: HumanoidMotionOptionObservableObject[] = [];
  for (const descriptor of input.option.scenario.objects) {
    const tracked = trackedObjectIds.has(descriptor.id);
    const object = visible[descriptor.id]
      ?? (tracked ? input.snapshot.objects[descriptor.id] : undefined);
    if (!object
      || input.boundObjectIds
        && !input.boundObjectIds.has(descriptor.id)
        && !tracked) continue;
    observableObjects.push({
      id: descriptor.id,
      position: { ...object.position },
      rotation: { ...object.rotation },
      size: { ...descriptor.size },
      ...observableArticulation(descriptor, object)
    });
  }
  return {
    snapshot: input.snapshot,
    observableObjects,
    observableSolidIds: observableHumanoidSolidIds(
      sensedSolids,
      input.snapshot.contacts
    ),
    zones: input.option.scenario.zones,
    ...(input.graspRegistry && input.worldFrame !== undefined
      ? {
          graspAssessments: input.graspRegistry.bindingsForOption(
            input.option.contract,
            input.worldFrame
          )
        }
      : {})
  };
}

function observableArticulation(
  descriptor: Scenario["objects"][number],
  observed: HumanoidSimulationSnapshot["objects"][string]
): Pick<
  HumanoidMotionOptionObservableObject,
  "articulation" | "container" | "supportSurface"
> {
  const capability = humanoidObjectCapability(descriptor);
  return {
    ...(capability.articulation && observed.articulation
      ? {
          articulation: {
            jointId: capability.articulation.joint_id,
            position: observed.articulation.position,
            velocity: observed.articulation.velocity,
            closedPosition: capability.articulation.closed_position,
            openPosition: capability.articulation.open_position
          }
        }
      : {}),
    ...(capability.container
      ? { container: structuredClone(capability.container) }
      : {}),
    ...(capability.supportSurface
      ? { supportSurface: structuredClone(capability.supportSurface) }
      : {})
  };
}
