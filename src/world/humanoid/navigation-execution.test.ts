import { describe, expect, it } from "vitest";
import type { NavigationPlan } from "../navigation.js";
import {
  HumanoidNavigationExecution,
  HumanoidNavigationExecutionProgressSchema
} from "./navigation-execution.js";
import { neutralHumanoidReference, type HumanoidReference } from "./reference.js";
import type {
  HumanoidContactSnapshot,
  HumanoidSimulation,
  HumanoidSimulationSnapshot
} from "./simulation.js";

describe("humanoid navigation execution progress", () => {
  it("continues the same physical route from a durable frame cursor", async () => {
    const uninterruptedSimulation = new NavigationSimulation();
    const uninterrupted = new HumanoidNavigationExecution({
      plan: route,
      reference: neutralHumanoidReference(),
      simulation: uninterruptedSimulation.asHumanoidSimulation()
    });
    expect(uninterruptedSimulation.controllerResetCount).toBe(1);
    await complete(uninterrupted, uninterruptedSimulation);
    const uninterruptedResult = uninterrupted.result();

    const firstSimulation = new NavigationSimulation();
    const interrupted = new HumanoidNavigationExecution({
      plan: route,
      reference: neutralHumanoidReference(),
      simulation: firstSimulation.asHumanoidSimulation()
    });
    expect(firstSimulation.controllerResetCount).toBe(1);
    for (let frame = 0; frame < 3; frame += 1) {
      expect((await interrupted.step(firstSimulation.asHumanoidSimulation())).done).toBe(false);
    }
    const progress = interrupted.checkpoint();
    expect(progress).toMatchObject({
      version: 1,
      waypoint_index: 1,
      committed_frame_count: 3,
      stopping_frame_count: 0
    });

    const resumedSimulation = new NavigationSimulation(firstSimulation.position);
    const resumed = new HumanoidNavigationExecution({
      plan: route,
      reference: interrupted.reference,
      simulation: resumedSimulation.asHumanoidSimulation(),
      progress
    });
    expect(resumedSimulation.controllerResetCount).toBe(0);
    await complete(resumed, resumedSimulation);
    const resumedResult = resumed.result();

    expect(resumedResult.completed).toBe(true);
    expect(resumedResult.frames).toBe(uninterruptedResult.frames);
    expect(resumedResult.travelledDistance).toBeCloseTo(
      uninterruptedResult.travelledDistance,
      12
    );
    expect(resumedResult.final.rootPosition).toEqual(
      uninterruptedResult.final.rootPosition
    );
  });

  it("rejects progress that cannot belong to the route", () => {
    expect(() => HumanoidNavigationExecutionProgressSchema.parse({
      version: 1,
      start_root_position: { x: 0, y: 0.8, z: 0 },
      waypoint_index: 1,
      committed_frame_count: 2,
      stopping_frame_count: 3
    })).toThrow(/stopping progress/i);

    const simulation = new NavigationSimulation();
    expect(() => new HumanoidNavigationExecution({
      plan: route,
      reference: neutralHumanoidReference(),
      simulation: simulation.asHumanoidSimulation(),
      progress: {
        version: 1,
        start_root_position: { x: 0, y: 0.8, z: 0 },
        waypoint_index: route.waypoints.length + 1,
        committed_frame_count: 1,
        stopping_frame_count: 0
      }
    })).toThrow(/waypoint progress/i);
  });

  it("allows only the exact hand-surface contact authorized for a carried object", async () => {
    const allowedContact = contact({
      firstHandLink: "left_hand_thumb_2_link",
      secondObject: "crate"
    });
    const allowedSimulation = new NavigationSimulation(undefined, [allowedContact]);
    const allowed = new HumanoidNavigationExecution({
      plan: route,
      reference: neutralHumanoidReference(),
      simulation: allowedSimulation.asHumanoidSimulation(),
      contactConstraints: [{
        hand_surface: "left_hand_thumb_2_link",
        object_id: "crate",
        required: false
      }]
    });

    expect((await allowed.step(allowedSimulation.asHumanoidSimulation())).done).toBe(false);

    const blockedSimulation = new NavigationSimulation(undefined, [contact({
      firstHandLink: "left_hand_index_1_link",
      secondObject: "crate"
    })]);
    const blocked = new HumanoidNavigationExecution({
      plan: route,
      reference: neutralHumanoidReference(),
      simulation: blockedSimulation.asHumanoidSimulation(),
      contactConstraints: [{
        hand_surface: "left_hand_thumb_2_link",
        object_id: "crate",
        required: false
      }]
    });

    expect((await blocked.step(blockedSimulation.asHumanoidSimulation())).done).toBe(true);
    expect(blocked.result()).toMatchObject({
      completed: false,
      reason: "environment_contact:left_hand_index_1_link:crate"
    });
  });

  it("commits external grasp evidence in the same physical frame", async () => {
    const simulation = new NavigationSimulation();
    const execution = new HumanoidNavigationExecution({
      plan: route,
      reference: neutralHumanoidReference(),
      simulation: simulation.asHumanoidSimulation()
    });

    const prepared = await execution.prepareFrame(simulation.asHumanoidSimulation());
    expect(prepared).not.toBeNull();
    expect(() => execution.checkpoint()).toThrow(/uncommitted/i);
    expect(() => execution.result()).toThrow(/uncommitted/i);

    const committed = execution.commitPreparedFrame("carried_grasp_lost:crate:left");
    expect(committed.done).toBe(true);
    expect(execution.checkpoint().committed_frame_count).toBe(1);
    expect(execution.result()).toMatchObject({
      completed: false,
      frames: 1,
      reason: "carried_grasp_lost:crate:left"
    });
  });

  it("preserves learned controller history when navigation continues a carried grasp", () => {
    const simulation = new NavigationSimulation();
    new HumanoidNavigationExecution({
      plan: route,
      reference: neutralHumanoidReference(),
      simulation: simulation.asHumanoidSimulation(),
      graspTargets: [{} as never]
    });

    expect(simulation.controllerResetCount).toBe(0);
  });

  it("lets a stable slow gait finish a physical route without a false timeout", async () => {
    const slowRoute: NavigationPlan = {
      waypoints: [
        { x: 0, y: 0.8, z: 0 },
        { x: 0, y: 0.8, z: 3 }
      ],
      distance: 3,
      resolvedTarget: { x: 0, y: 0.8, z: 3 },
      projectionDistance: 0
    };
    const simulation = new NavigationSimulation(undefined, [], 3, 0.006);
    const execution = new HumanoidNavigationExecution({
      plan: slowRoute,
      reference: neutralHumanoidReference(),
      simulation: simulation.asHumanoidSimulation()
    });

    for (let guard = 0; !execution.done && guard < 1_000; guard += 1) {
      await execution.step(simulation.asHumanoidSimulation());
    }

    expect(execution.done).toBe(true);
    expect(execution.result()).toMatchObject({ completed: true });
    expect(execution.result().frames).toBeGreaterThan(330);
  });
});

const route: NavigationPlan = {
  waypoints: [
    { x: 0, y: 0.8, z: 0 },
    { x: 0, y: 0.8, z: 1 }
  ],
  distance: 1,
  resolvedTarget: { x: 0, y: 0.8, z: 1 },
  projectionDistance: 0
};

async function complete(
  execution: HumanoidNavigationExecution,
  simulation: NavigationSimulation
): Promise<void> {
  for (let guard = 0; !execution.done && guard < 100; guard += 1) {
    await execution.step(simulation.asHumanoidSimulation());
  }
  expect(execution.done).toBe(true);
}

class NavigationSimulation {
  readonly position: { x: number; y: number; z: number };
  readonly contacts: HumanoidContactSnapshot[];
  readonly destinationZ: number;
  readonly progressPerStep: number;
  controllerResetCount = 0;

  constructor(
    position = { x: 0, y: 0.8, z: 0 },
    contacts: readonly HumanoidContactSnapshot[] = [],
    destinationZ = 1,
    progressPerStep = 0.2
  ) {
    this.position = { ...position };
    this.contacts = structuredClone(contacts);
    this.destinationZ = destinationZ;
    this.progressPerStep = progressPerStep;
  }

  asHumanoidSimulation(): HumanoidSimulation {
    return this as unknown as HumanoidSimulation;
  }

  controllerDescriptor(): {
    controlStepSeconds: number;
  } {
    return { controlStepSeconds: 0.1 };
  }

  resetController(_reference: HumanoidReference): void {
    this.controllerResetCount += 1;
  }

  snapshot(): HumanoidSimulationSnapshot {
    return {
      rootPosition: { ...this.position },
      rootRotation: { x: 0, y: 0, z: 0, w: 1 },
      contacts: structuredClone(this.contacts),
      fallen: false
    } as unknown as HumanoidSimulationSnapshot;
  }

  async step(reference: HumanoidReference): Promise<HumanoidSimulationSnapshot> {
    if (reference.rootVelocity[0] > 0) {
      this.position.z = Math.min(
        this.destinationZ,
        this.position.z + this.progressPerStep
      );
    }
    return this.snapshot();
  }
}

function contact(
  overrides: Partial<HumanoidContactSnapshot>
): HumanoidContactSnapshot {
  return {
    position: { x: 0, y: 0.8, z: 0 },
    normal: { x: 1, y: 0, z: 0 },
    normalForce: 10,
    firstBody: null,
    secondBody: null,
    firstObject: null,
    secondObject: null,
    firstHandLink: null,
    secondHandLink: null,
    ...overrides
  };
}
