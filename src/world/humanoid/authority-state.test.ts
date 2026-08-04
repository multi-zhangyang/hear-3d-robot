import { describe, expect, it } from "vitest";
import { neutralHumanoidReference } from "./reference.js";
import { humanoidAuthorityStateSha256 } from "./authority-state.js";
import type { HumanoidSimulationState } from "./simulation.js";
import { HumanoidGraspRegistry } from "./grasp-registry.js";
import { admitHumanoidCarriedObjectBindings } from "./carried-object-binding.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

describe("humanoidAuthorityStateSha256", () => {
  it("is deterministic across independent state copies and object key order", () => {
    const first = state({ nested: { beta: 2, alpha: 1 }, count: 4 });
    const second = state({ count: 4, nested: { alpha: 1, beta: 2 } });
    const reference = neutralHumanoidReference();

    expect(digest(first, reference, ["crate", "beacon"], 4)).toBe(
      digest(second, reference, ["beacon", "crate"], 4)
    );
  });

  it("binds simulation, controller, requested torque and reference state", () => {
    const simulation = state({ count: 4 });
    const reference = neutralHumanoidReference();
    const baseline = digest(simulation, reference, ["crate"], 2);
    const changedSimulation = {
      ...simulation,
      positions: simulation.positions.slice()
    };
    changedSimulation.positions[0] += 0.001;
    const changedReference = {
      ...reference,
      jointPositions: reference.jointPositions.slice()
    };
    changedReference.jointPositions[0] += 0.001;

    expect(digest(changedSimulation, reference, ["crate"], 2)).not.toBe(baseline);
    expect(digest(simulation, changedReference, ["crate"], 2)).not.toBe(baseline);
    expect(digest({
        ...simulation,
        requestedActuatorTorques: new Float64Array([0.25, 0.5])
      }, reference, ["crate"], 2)).not.toBe(baseline);
  });

  it("binds visibility memory and plan registry epochs independently of physics", () => {
    const simulation = state({ count: 4 });
    const reference = neutralHumanoidReference();
    const baseline = digest(simulation, reference, ["crate"], 2);

    expect(digest(simulation, reference, ["crate", "beacon"], 2)).not.toBe(baseline);
    expect(digest(simulation, reference, ["crate"], 2, ["block-a"])).not.toBe(
      baseline
    );
    expect(digest(simulation, reference, ["crate"], 3)).not.toBe(baseline);
  });

  it("binds the complete grasp registry authority state", () => {
    const simulation = state({ count: 4 });
    const reference = neutralHumanoidReference();
    const registry = graspRegistry(0);
    const carriedObjectBindings = emptyCarriedBindings(registry, 0);
    const baseline = humanoidAuthorityStateSha256({
      simulation,
      reference,
      visibleContactObjectIds: [],
      visibleContactSolidIds: [],
      planRegistryEpoch: 0,
      graspRegistry: registry.checkpoint(),
      carriedObjectBindings
    });
    registry.observe(1, { objects: {} } as HumanoidSimulationSnapshot);

    expect(humanoidAuthorityStateSha256({
      simulation,
      reference,
      visibleContactObjectIds: [],
      visibleContactSolidIds: [],
      planRegistryEpoch: 0,
      graspRegistry: registry.checkpoint(),
      carriedObjectBindings: emptyCarriedBindings(registry, 1)
    })).not.toBe(baseline);
  });

  it("binds carried-object authority independently of the grasp checkpoint", () => {
    const simulation = state({ count: 4 });
    const reference = neutralHumanoidReference();
    const registry = graspRegistry(0);
    const atRevisionZero = emptyCarriedBindings(registry, 0);
    const atRevisionOne = emptyCarriedBindings(registry, 1);

    expect(humanoidAuthorityStateSha256({
      simulation,
      reference,
      visibleContactObjectIds: [],
      visibleContactSolidIds: [],
      planRegistryEpoch: 0,
      graspRegistry: registry.checkpoint(),
      carriedObjectBindings: atRevisionZero
    })).not.toBe(humanoidAuthorityStateSha256({
      simulation,
      reference,
      visibleContactObjectIds: [],
      visibleContactSolidIds: [],
      planRegistryEpoch: 0,
      graspRegistry: registry.checkpoint(),
      carriedObjectBindings: atRevisionOne
    }));
  });
});

function digest(
  simulation: HumanoidSimulationState,
  reference: ReturnType<typeof neutralHumanoidReference>,
  visibleContactObjectIds: Iterable<string>,
  planRegistryEpoch: number,
  visibleContactSolidIds: Iterable<string> = []
): string {
  const registry = graspRegistry(0);
  return humanoidAuthorityStateSha256({
    simulation,
    reference,
    visibleContactObjectIds,
    visibleContactSolidIds,
    planRegistryEpoch,
    graspRegistry: registry.checkpoint(),
    carriedObjectBindings: emptyCarriedBindings(registry, 0)
  });
}

function emptyCarriedBindings(
  registry: HumanoidGraspRegistry,
  currentWorldRevision: number
) {
  return admitHumanoidCarriedObjectBindings({
    registry,
    currentFrame: registry.lastFrame ?? 0,
    currentWorldRevision,
    requests: []
  });
}

function graspRegistry(frame: number): HumanoidGraspRegistry {
  const registry = new HumanoidGraspRegistry({ portableObjectIds: [] });
  registry.observe(frame, { objects: {} } as HumanoidSimulationSnapshot);
  return registry;
}

function state(payload: HumanoidSimulationState["controller"]["payload"]): HumanoidSimulationState {
  return {
    time: 1.25,
    positions: new Float64Array([1, 2]),
    velocities: new Float64Array([3, 4]),
    controls: new Float64Array([5, 6]),
    activations: new Float64Array([7]),
    accelerationWarmstart: new Float64Array([8, 9]),
    requestedActuatorTorques: new Float64Array([0.1, 0.2]),
    controller: {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: "authority-state-test",
      payload
    }
  };
}
