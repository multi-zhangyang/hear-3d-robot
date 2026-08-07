import { describe, expect, it } from "vitest";
import type { Scenario } from "../../domain/schema.js";
import { HumanoidMotionOptionContractSchema } from "./motion-option.js";
import { humanoidMotionOptionDetectorInputFromSimulation } from "./motion-option-observation.js";
import type {
  HumanoidSimulation,
  HumanoidSimulationSnapshot
} from "./simulation.js";

const identity = { x: 0, y: 0, z: 0, w: 1 };
const visibleState = {
  id: "visible",
  position: { x: 1, y: 0.1, z: 1 },
  rotation: identity,
  linearVelocity: { x: 0, y: 0, z: 0 },
  angularVelocity: { x: 0, y: 0, z: 0 }
};
const trackedState = {
  ...visibleState,
  id: "tracked-release",
  position: { x: 2, y: 0.1, z: 2 },
  rotation: { x: 0, y: 0, z: Math.SQRT1_2, w: Math.SQRT1_2 }
};
const scenario = {
  visibility_radius: 2,
  objects: [{
    id: "visible",
    size: { x: 0.2, y: 0.2, z: 0.2 }
  }, {
    id: "tracked-release",
    size: { x: 0.03, y: 0.27, z: 0.03 }
  }],
  zones: []
} as unknown as Scenario;
const option = {
  contract: HumanoidMotionOptionContractSchema.parse({
    option_id: "transaction-observation",
    predicates: [{
      type: "object_released",
      object_id: "tracked-release",
      hand: "left"
    }],
    stable_steps: 1
  }),
  scenario
};
const snapshot = {
  contacts: [],
  objects: {
    visible: visibleState,
    "tracked-release": trackedState
  }
} as unknown as HumanoidSimulationSnapshot;
const simulation = {
  senseObjects: () => ({ objects: { visible: visibleState } }),
  senseSolids: () => ({ solids: {} })
} as unknown as HumanoidSimulation;

describe("motion option transaction observation", () => {
  it("binds an observable articulation descriptor to live MuJoCo joint state", () => {
    const doorState = {
      ...visibleState,
      id: "door",
      articulation: {
        type: "hinge" as const,
        position: 0.8,
        velocity: 0.03,
        minimum: 0,
        maximum: 1,
        normalized: 0.8
      }
    };
    const doorScenario = {
      visibility_radius: 2,
      objects: [{
        id: "door",
        size: { x: 0.6, y: 0.9, z: 0.05 },
        capability: {
          articulation: {
            joint_id: "door-hinge",
            type: "hinge",
            semantic: "door",
            axis: { x: 0, y: 1, z: 0 },
            anchor_world: { x: 0, y: 0, z: 0 },
            range: { minimum: 0, maximum: 1 },
            initial_position: 0,
            closed_position: 0,
            open_position: 1,
            damping: 0.5,
            friction_loss: 0.05
          }
        }
      }],
      zones: []
    } as unknown as Scenario;
    const doorSimulation = {
      senseObjects: () => ({ objects: { door: doorState } }),
      senseSolids: () => ({ solids: {} })
    } as unknown as HumanoidSimulation;
    const result = humanoidMotionOptionDetectorInputFromSimulation({
      simulation: doorSimulation,
      snapshot: {
        contacts: [],
        objects: { door: doorState }
      } as unknown as HumanoidSimulationSnapshot,
      option: {
        contract: HumanoidMotionOptionContractSchema.parse({
          option_id: "observe-door",
          predicates: [{
            type: "articulation_state",
            object_id: "door",
            joint_id: "door-hinge",
            state: "open",
            tolerance: 0.2
          }],
          stable_steps: 1
        }),
        scenario: doorScenario
      }
    });
    expect(result.observableObjects[0]?.articulation).toEqual({
      jointId: "door-hinge",
      position: 0.8,
      velocity: 0.03,
      closedPosition: 0,
      openPosition: 1
    });
  });

  it("keeps ordinary hidden objects private but follows the explicitly tracked release object", () => {
    const ordinary = humanoidMotionOptionDetectorInputFromSimulation({
      simulation,
      snapshot,
      option
    });
    expect(ordinary.observableObjects.map((object) => object.id)).toEqual(["visible"]);

    const release = humanoidMotionOptionDetectorInputFromSimulation({
      simulation,
      snapshot,
      option,
      boundObjectIds: new Set(),
      trackedObjectIds: new Set(["tracked-release"])
    });
    expect(release.observableObjects).toEqual([{
      id: "tracked-release",
      position: trackedState.position,
      rotation: trackedState.rotation,
      size: { x: 0.03, y: 0.27, z: 0.03 }
    }]);
  });
});
