import { ScenarioSchema } from "../../domain/schema.js";
import { rebuildScenarioChunkManifest } from "../../domain/scenario-chunk.js";

export const humanoidWorldTestScenario = ScenarioSchema.parse({
  title: "Humanoid field",
  seed: 7,
  bounds: { width: 10, depth: 10 },
  visibility_radius: 6,
  robot: { x: 1.5, z: 1.5, yaw: 0 },
  obstacles: [{
    id: "column",
    center: { x: 6, y: 1, z: 6 },
    size: { x: 1, y: 2, z: 1 }
  }],
  objects: [],
  zones: [],
  default_goal: {
    summary: "到达开放区域",
    predicates: [{
      type: "robot_at",
      target: { x: 1.5, y: 0, z: 2.2 },
      tolerance: 0.25
    }]
  }
});

export const humanoidWorldPerceptionTestScenario = ScenarioSchema.parse(
  rebuildScenarioChunkManifest({
    ...humanoidWorldTestScenario,
    title: "Humanoid perception field",
    obstacles: [],
    objects: [{
      id: "crate",
      kind: "crate",
      color: "#8b6b45",
      position: { x: 1.5, y: 0.15, z: 3.2 },
      size: { x: 0.3, y: 0.3, z: 0.3 },
      portable: true
    }]
  })
);
