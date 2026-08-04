import { describe, expect, it } from "vitest";
import { G1_RIG_PARTS } from "./g1-rig";

describe("G1 Web rig morphology", () => {
  it("renders articulated palms and all fourteen finger links from upstream meshes", () => {
    const handParts = G1_RIG_PARTS.filter((part) => part.mesh.includes("_hand_"));
    const articulated = handParts.filter((part) => part.body.includes("_hand_"));
    expect(handParts).toHaveLength(16);
    expect(articulated).toHaveLength(14);
    expect(handParts.map((part) => part.mesh)).toContain("left_hand_palm_link");
    expect(handParts.map((part) => part.mesh)).toContain("right_hand_palm_link");
    expect(G1_RIG_PARTS.some((part) => part.mesh.includes("rubber_hand"))).toBe(false);
    expect(G1_RIG_PARTS).toEqual(expect.arrayContaining([
      expect.objectContaining({ mesh: "torso_link_rev_1_0" }),
      expect.objectContaining({ mesh: "waist_roll_link_rev_1_0" }),
      expect.objectContaining({ mesh: "waist_yaw_link_rev_1_0" })
    ]));
  });
});
