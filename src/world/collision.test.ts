import { describe, expect, it } from "vitest";
import { collisionTransitionAllowed, type CollisionIssue } from "./collision.js";

const touching: CollisionIssue = {
  segment: "base",
  collider_kind: "voxel",
  collider_id: "voxel_1",
  penetration_depth: 0
};

describe("collision transition", () => {
  it("allows an existing zero-depth contact while still rejecting penetration growth", () => {
    expect(collisionTransitionAllowed([touching], [{
      ...touching,
      penetration_depth: 0.000005
    }])).toBe(true);
    expect(collisionTransitionAllowed([touching], [{
      ...touching,
      penetration_depth: 0.0002
    }])).toBe(false);
  });

  it("requires a real penetration to shrink or disappear", () => {
    const penetrating = { ...touching, penetration_depth: 0.02 };
    expect(collisionTransitionAllowed([penetrating], [{
      ...penetrating,
      penetration_depth: 0.01
    }])).toBe(true);
    expect(collisionTransitionAllowed([penetrating], [{
      ...penetrating,
      penetration_depth: 0.02001
    }])).toBe(false);
    expect(collisionTransitionAllowed([penetrating], [])).toBe(true);
  });
});
