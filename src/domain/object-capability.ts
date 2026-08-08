import { z } from "zod";

const Vector3Schema = z.object({
  x: z.number().finite(),
  y: z.number().finite(),
  z: z.number().finite()
}).strict();

const UnitVector3Schema = Vector3Schema.refine(
  (value) => Math.abs(Math.hypot(value.x, value.y, value.z) - 1) <= 1e-3,
  "direction must be normalized"
);

const OBJECT_AFFORDANCES = [
  "graspable",
  "pushable",
  "pullable",
  "openable",
  "closeable",
  "pressable",
  "rotatable",
  "container",
  "support_surface",
  "insertable",
  "movable"
] as const;

const ObjectAffordanceSchema = z.enum(OBJECT_AFFORDANCES);
export type ObjectAffordance = z.infer<typeof ObjectAffordanceSchema>;

const OBJECT_INTERACTION_POINT_KINDS = [
  "grasp",
  "push",
  "pull",
  "press",
  "turn",
  "insert",
  "support"
] as const;

const ObjectInteractionPointSchema = z.object({
  id: z.string().trim().min(1),
  kind: z.enum(OBJECT_INTERACTION_POINT_KINDS),
  local_position: Vector3Schema,
  approach_direction: UnitVector3Schema.optional(),
  compatible_hands: z.enum(["left", "right", "either", "both"]).default("either"),
  clearance_m: z.number().finite().nonnegative().default(0.04)
}).strict();

const ObjectArticulationSchema = z.object({
  joint_id: z.string().trim().min(1),
  parent_object_id: z.string().trim().min(1).optional(),
  type: z.enum(["hinge", "slide"]),
  semantic: z.string().trim().min(1).max(64),
  axis: UnitVector3Schema,
  anchor_world: Vector3Schema,
  range: z.object({
    minimum: z.number().finite(),
    maximum: z.number().finite()
  }).strict(),
  initial_position: z.number().finite().default(0),
  closed_position: z.number().finite(),
  open_position: z.number().finite(),
  damping: z.number().finite().nonnegative().default(0.5),
  friction_loss: z.number().finite().nonnegative().default(0.05)
}).strict().superRefine((joint, context) => {
  if (joint.range.minimum >= joint.range.maximum) {
    context.addIssue({
      code: "custom",
      path: ["range"],
      message: "articulation range minimum must be less than maximum"
    });
  }
  for (const [field, value] of [
    ["initial_position", joint.initial_position],
    ["closed_position", joint.closed_position],
    ["open_position", joint.open_position]
  ] as const) {
    if (value < joint.range.minimum || value > joint.range.maximum) {
      context.addIssue({
        code: "custom",
        path: [field],
        message: `${field} must be inside the articulation range`
      });
    }
  }
  if (Math.abs(joint.open_position - joint.closed_position) <= 1e-6) {
    context.addIssue({
      code: "custom",
      path: ["open_position"],
      message: "open and closed articulation positions must differ"
    });
  }
});

const ObjectContainerSchema = z.object({
  interior_center: Vector3Schema,
  interior_size: Vector3Schema.refine(
    ({ x, y, z: depth }) => x > 0 && y > 0 && depth > 0,
    "container interior size components must be positive"
  ),
  opening_direction: UnitVector3Schema,
  wall_thickness_m: z.number().finite().positive().default(0.035)
}).strict();

const ObjectSupportSurfaceSchema = z.object({
  local_center: Vector3Schema,
  size: Vector3Schema.refine(
    ({ x, y, z: depth }) => x > 0 && y >= 0 && depth > 0,
    "support surface size must have positive horizontal extent"
  ),
  normal: UnitVector3Schema
}).strict();

export const ScenarioObjectCapabilitySchema = z.object({
  shape: z.enum(["box", "sphere", "cylinder", "capsule"]).default("box"),
  mass_kg: z.number().finite().positive().optional(),
  density_kg_m3: z.number().finite().positive().optional(),
  friction: z.object({
    sliding: z.number().finite().nonnegative(),
    torsional: z.number().finite().nonnegative(),
    rolling: z.number().finite().nonnegative()
  }).strict().optional(),
  affordances: z.array(ObjectAffordanceSchema).default([]),
  interaction_points: z.array(ObjectInteractionPointSchema).default([]),
  articulation: ObjectArticulationSchema.optional(),
  container: ObjectContainerSchema.optional(),
  support_surface: ObjectSupportSurfaceSchema.optional()
}).strict().superRefine((capability, context) => {
  const ids = capability.interaction_points.map((point) => point.id);
  if (new Set(ids).size !== ids.length) {
    context.addIssue({
      code: "custom",
      path: ["interaction_points"],
      message: "interaction point identities must be unique"
    });
  }
  if (new Set(capability.affordances).size !== capability.affordances.length) {
    context.addIssue({
      code: "custom",
      path: ["affordances"],
      message: "object affordances must be unique"
    });
  }
  if (capability.mass_kg !== undefined && capability.density_kg_m3 !== undefined) {
    context.addIssue({
      code: "custom",
      path: ["density_kg_m3"],
      message: "specify either mass or density, not both"
    });
  }
});

export type ScenarioObjectCapability = z.infer<typeof ScenarioObjectCapabilitySchema>;
