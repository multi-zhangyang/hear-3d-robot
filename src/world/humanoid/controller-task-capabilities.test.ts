import { describe, expect, it } from "vitest";
import { humanoidControllerTaskCapabilities } from
  "./controller-task-capabilities.js";
import { neutralHumanoidReference } from "./reference.js";

describe("humanoid controller task capabilities", () => {
  it("derives the low-level capabilities from the actual reference", () => {
    const standing = neutralHumanoidReference();
    expect(humanoidControllerTaskCapabilities(standing)).toEqual(["balance"]);

    const walking = neutralHumanoidReference();
    walking.rootVelocity[0] = 0.2;
    expect(humanoidControllerTaskCapabilities(walking)).toEqual([
      "balance",
      "locomotion"
    ]);

    const articulated = neutralHumanoidReference();
    articulated.jointTrackingWeights[3] = 1;
    expect(humanoidControllerTaskCapabilities(articulated)).toEqual([
      "balance",
      "joint_reference_tracking"
    ]);
  });

  it("merges semantic requirements in the canonical capability order", () => {
    const reference = neutralHumanoidReference();
    reference.rootYawVelocity = 0.1;
    reference.jointTrackingWeights[8] = 0.5;
    expect(humanoidControllerTaskCapabilities(reference, [
      "bimanual_manipulation",
      "contact_rich_manipulation",
      "locomotion"
    ])).toEqual([
      "balance",
      "locomotion",
      "joint_reference_tracking",
      "contact_rich_manipulation",
      "bimanual_manipulation"
    ]);
  });
});
