import { describe, expect, it } from "vitest";
import type { HumanoidManipulationReachabilityObservation } from "./world-contract.js";
import {
  basePlacementProbeAlignments,
  manipulationBasePreservesBodyClearance
} from "./manipulation-reachability.js";

describe("humanoid manipulation base-placement probes", () => {
  it("preserves the best recoverable surface for each hand", () => {
    const selected = basePlacementProbeAlignments([
      alignment("left_hand_index_1_link", 0.01),
      alignment("left_hand_middle_1_link", 0.02),
      alignment("right_hand_index_1_link", 0.04),
      alignment("right_hand_middle_1_link", 0.03),
      alignment("right_hand_thumb_2_link", null)
    ]);

    expect(selected.map(({ handSurface }) => handSurface)).toEqual([
      "left_hand_index_1_link",
      "right_hand_middle_1_link"
    ]);
  });

  it("rejects a base that crosses the interaction side body clearance", () => {
    const target = {
      worldPosition: { x: 0, y: 1, z: 1 },
      approachDirection: { x: 0, y: 0, z: 1 },
      clearanceMeters: 0.08
    };

    expect(manipulationBasePreservesBodyClearance(
      { x: 0, y: 0.75, z: 0.7 },
      target
    )).toBe(true);
    expect(manipulationBasePreservesBodyClearance(
      { x: 0, y: 0.75, z: 0.75 },
      target
    )).toBe(false);
    expect(manipulationBasePreservesBodyClearance(
      { x: 0, y: 0.75, z: 0.35 },
      target
    )).toBe(true);
  });

  it("uses radial body clearance for a vertical approach", () => {
    const target = {
      worldPosition: { x: 0, y: 0.8, z: 1 },
      approachDirection: { x: 0, y: -1, z: 0 },
      clearanceMeters: 0.04
    };

    expect(manipulationBasePreservesBodyClearance(
      { x: 0, y: 0.75, z: 0.7 },
      target
    )).toBe(true);
    expect(manipulationBasePreservesBodyClearance(
      { x: 0, y: 0.75, z: 0.81 },
      target
    )).toBe(false);
  });
});

function alignment(
  handSurface: HumanoidManipulationReachabilityObservation["handSurface"],
  ikResidualMeters: number | null
): HumanoidManipulationReachabilityObservation {
  return {
    objectId: "assembly_rod",
    handSurface,
    wristWorldTarget: { x: 0, y: 0, z: 0 },
    ikReferenceReachable: false,
    ikResidualMeters
  };
}
