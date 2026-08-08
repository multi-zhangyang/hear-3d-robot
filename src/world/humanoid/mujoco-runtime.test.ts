import { describe, expect, it } from "vitest";
import {
  createHumanoidScenePath,
  loadHumanoidMujoco,
  removeHumanoidScene
} from "./mujoco-runtime.js";

describe("humanoid interaction geometry", () => {
  it("builds pull affordances as graspable bars and press affordances as points", async () => {
    const runtime = await loadHumanoidMujoco();
    const path = createHumanoidScenePath(runtime, [], [{
      id: "fixture",
      center: { x: 0, y: 1, z: 2 },
      size: { x: 0.8, y: 1, z: 0.08 },
      mass: 2,
      mobility: { type: "fixed" },
      interactionPoints: [{
        id: "handle",
        kind: "pull",
        localPosition: { x: 0.2, y: 0, z: -0.08 },
        clearanceMeters: 0.08
      }, {
        id: "button",
        kind: "press",
        localPosition: { x: -0.2, y: 0, z: -0.05 },
        clearanceMeters: 0.04
      }]
    }]);
    const model = runtime.MjModel.from_xml_path(path);
    try {
      const handle = runtime.mj_name2id(
        model,
        runtime.mjtObj.mjOBJ_GEOM.value,
        "world-object-interaction-0-0-handle"
      );
      const button = runtime.mj_name2id(
        model,
        runtime.mjtObj.mjOBJ_GEOM.value,
        "world-object-interaction-0-1-button"
      );
      expect(model.geom_type[handle]).toBe(runtime.mjtGeom.mjGEOM_CAPSULE.value);
      expect(model.geom_size[handle * 3 + 1]).toBeGreaterThan(
        model.geom_size[handle * 3]!
      );
      expect(model.geom_type[button]).toBe(runtime.mjtGeom.mjGEOM_SPHERE.value);
    } finally {
      model.delete();
      removeHumanoidScene(runtime, path);
    }
  }, 30_000);
});
