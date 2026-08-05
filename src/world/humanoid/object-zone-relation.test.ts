import { describe, expect, it } from "vitest";
import { humanoidObjectZoneRelation } from "./object-zone-relation.js";

const ZONE = {
  center: { x: 0, y: 0, z: 0 },
  size: { x: 1, y: 0.04, z: 1 }
};

describe("humanoid object-zone relation", () => {
  it("uses the oriented world extents of a placed object", () => {
    const unrotated = humanoidObjectZoneRelation({
      object: {
        position: { x: 0.32, y: 0.12, z: 0 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        size: { x: 0.3, y: 0.2, z: 0.8 }
      },
      zone: ZONE,
      tolerance: 0
    });
    const quarterTurn = humanoidObjectZoneRelation({
      object: {
        position: { x: 0.32, y: 0.12, z: 0 },
        rotation: {
          x: 0,
          y: Math.sin(Math.PI / 4),
          z: 0,
          w: Math.cos(Math.PI / 4)
        },
        size: { x: 0.3, y: 0.2, z: 0.8 }
      },
      zone: ZONE,
      tolerance: 0
    });

    expect(unrotated.inside).toBe(true);
    expect(quarterTurn.inside).toBe(false);
    expect(quarterTurn.horizontalClearance.x).toBeLessThan(0);
  });

  it("reports horizontal fit and support-height evidence separately", () => {
    const relation = humanoidObjectZoneRelation({
      object: {
        position: { x: 0, y: 0.2, z: 0 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        size: { x: 0.2, y: 0.2, z: 0.2 }
      },
      zone: ZONE,
      tolerance: 0.01
    });

    expect(relation.horizontalClearance.x).toBeCloseTo(0.41);
    expect(relation.horizontalClearance.z).toBeCloseTo(0.41);
    expect(relation.horizontalClearance.minimum).toBeCloseTo(0.41);
    expect(relation.supportHeightError).toBeCloseTo(0.08);
    expect(relation.inside).toBe(false);
  });
});
