import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { humanoidEnvironment } from "./environment.js";

describe("humanoidEnvironment", () => {
  it("derives portable-object mass from metric volume with bounded physical density", () => {
    const environment = humanoidEnvironment(ScenarioSchema.parse({
      title: "Portable-object mass field",
      seed: 1,
      bounds: { width: 4, depth: 4 },
      visibility_radius: 4,
      robot: { x: 0, z: 0, yaw: 0 },
      obstacles: [],
      objects: [
        portableObject("small", 0.04),
        portableObject("medium", 0.1),
        portableObject("large", 0.3)
      ],
      zones: [],
      default_goal: {
        summary: "保持站立",
        predicates: [{
          type: "robot_at",
          target: { x: 0, y: 0, z: 0 },
          tolerance: 1
        }]
      }
    }));

    expect(environment.objects?.map(({ id }) => id)).toEqual([
      "small",
      "medium",
      "large"
    ]);
    expect(environment.objects?.[0]?.mass).toBe(0.25);
    expect(environment.objects?.[1]?.mass).toBeCloseTo(0.6, 12);
    expect(environment.objects?.[2]?.mass).toBe(2);
  });
});

function portableObject(id: string, edge: number) {
  return {
    id,
    kind: "block",
    color: "#8b6b45",
    position: { x: 1, y: edge / 2, z: 1 },
    size: { x: edge, y: edge, z: edge },
    portable: true
  };
}
