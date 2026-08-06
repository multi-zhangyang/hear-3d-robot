import { describe, expect, it } from "vitest";
import type { HumanoidManipulationReachabilityObservation } from "./world-contract.js";
import { basePlacementProbeAlignments } from "./manipulation-reachability.js";

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
