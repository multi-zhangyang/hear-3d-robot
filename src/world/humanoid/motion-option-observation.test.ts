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
