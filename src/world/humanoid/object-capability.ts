import type { Scenario, Vec3 } from "../../domain/schema.js";
import type {
  ObjectAffordance,
  ScenarioObjectCapability
} from "../../domain/object-capability.js";

type ScenarioObject = Scenario["objects"][number];

interface HumanoidObjectInteractionPoint {
  id: string;
  kind: "grasp" | "push" | "pull" | "press" | "turn" | "insert" | "support";
  localPosition: Vec3;
  approachDirection?: Vec3;
  compatibleHands: "left" | "right" | "either" | "both";
  clearanceMeters: number;
  source: "authored" | "geometry";
}

export interface HumanoidObjectCapabilityDescriptor {
  shape: "box" | "sphere" | "cylinder" | "capsule";
  massKg: number;
  friction: {
    sliding: number;
    torsional: number;
    rolling: number;
  };
  mobility: "fixed" | "free" | "articulated";
  affordances: ObjectAffordance[];
  interactionPoints: HumanoidObjectInteractionPoint[];
  articulation?: NonNullable<ScenarioObjectCapability["articulation"]>;
  container?: NonNullable<ScenarioObjectCapability["container"]>;
  supportSurface?: NonNullable<ScenarioObjectCapability["support_surface"]>;
}

const LEGACY_DENSITY_KG_M3 = 600;
const LEGACY_MINIMUM_MASS_KG = 0.25;
const LEGACY_MAXIMUM_MASS_KG = 2;

export function humanoidObjectCapability(
  object: ScenarioObject
): HumanoidObjectCapabilityDescriptor {
  const configured = object.capability;
  const shape = configured?.shape ?? "box";
  const volume = objectVolume(shape, object.size);
  const massKg = configured?.mass_kg
    ?? (configured?.density_kg_m3 === undefined
      ? clamp(
          volume * LEGACY_DENSITY_KG_M3,
          LEGACY_MINIMUM_MASS_KG,
          LEGACY_MAXIMUM_MASS_KG
        )
      : volume * configured.density_kg_m3);
  const mobility = configured?.articulation
    ? "articulated" as const
    : object.portable ? "free" as const : "fixed" as const;
  const affordances = new Set<ObjectAffordance>(configured?.affordances ?? []);
  const authored = (configured?.interaction_points ?? []).map((point) => ({
    id: point.id,
    kind: point.kind,
    localPosition: { ...point.local_position },
    ...(point.approach_direction
      ? { approachDirection: { ...point.approach_direction } }
      : {}),
    compatibleHands: point.compatible_hands,
    clearanceMeters: point.clearance_m,
    source: "authored" as const
  }));
  deriveAffordances({
    affordances,
    mobility,
    massKg,
    maximumExtent: Math.max(object.size.x, object.size.y, object.size.z),
    articulated: configured?.articulation !== undefined,
    interactionPointKinds: new Set(authored.map(({ kind }) => kind)),
    container: configured?.container !== undefined,
    supportSurface: configured?.support_surface !== undefined
  });
  const generated = authored.some((point) => point.kind === "grasp")
    || !affordances.has("graspable")
    ? []
    : geometryGraspPoints(shape, object.size);
  return {
    shape,
    massKg,
    friction: configured?.friction
      ? { ...configured.friction }
      : { sliding: 0.8, torsional: 0.01, rolling: 0.001 },
    mobility,
    affordances: [...affordances].sort(),
    interactionPoints: [...authored, ...generated],
    ...(configured?.articulation
      ? { articulation: structuredClone(configured.articulation) }
      : {}),
    ...(configured?.container
      ? { container: structuredClone(configured.container) }
      : {}),
    ...(configured?.support_surface
      ? { supportSurface: structuredClone(configured.support_surface) }
      : {})
  };
}

function deriveAffordances(input: {
  affordances: Set<ObjectAffordance>;
  mobility: HumanoidObjectCapabilityDescriptor["mobility"];
  massKg: number;
  maximumExtent: number;
  articulated: boolean;
  interactionPointKinds: ReadonlySet<HumanoidObjectInteractionPoint["kind"]>;
  container: boolean;
  supportSurface: boolean;
}): void {
  if (input.mobility === "free") {
    input.affordances.add("movable");
    input.affordances.add("pushable");
    input.affordances.add("pullable");
    if (input.massKg <= 8 && input.maximumExtent <= 1.2) {
      input.affordances.add("graspable");
    }
  }
  if (input.interactionPointKinds.has("grasp")) {
    input.affordances.add("graspable");
  }
  if (input.interactionPointKinds.has("push")) {
    input.affordances.add("pushable");
  }
  if (input.interactionPointKinds.has("pull")) {
    input.affordances.add("pullable");
  }
  if (input.interactionPointKinds.has("press")) {
    input.affordances.add("pressable");
  }
  if (input.interactionPointKinds.has("turn")) {
    input.affordances.add("rotatable");
  }
  if (input.interactionPointKinds.has("insert")) {
    input.affordances.add("insertable");
  }
  if (input.articulated && (input.interactionPointKinds.has("grasp")
    || input.interactionPointKinds.has("push")
    || input.interactionPointKinds.has("pull"))) {
    input.affordances.add("openable");
    input.affordances.add("closeable");
  }
  if (input.container) input.affordances.add("container");
  if (input.supportSurface) input.affordances.add("support_surface");
}

function geometryGraspPoints(
  shape: HumanoidObjectCapabilityDescriptor["shape"],
  size: Vec3
): HumanoidObjectInteractionPoint[] {
  const clearance = Math.max(0.025, Math.min(0.08, Math.min(size.x, size.y, size.z) / 3));
  const radial = shape === "sphere"
    ? Math.min(size.x, size.y, size.z) / 2
    : shape === "cylinder" || shape === "capsule"
      ? Math.min(size.x, size.z) / 2
      : null;
  const x = radial ?? size.x / 2;
  const z = radial ?? size.z / 2;
  const points: Array<[string, Vec3, Vec3]> = [
    ["geometry-x-positive", { x, y: 0, z: 0 }, { x: -1, y: 0, z: 0 }],
    ["geometry-x-negative", { x: -x, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }],
    ["geometry-z-positive", { x: 0, y: 0, z }, { x: 0, y: 0, z: -1 }],
    ["geometry-z-negative", { x: 0, y: 0, z: -z }, { x: 0, y: 0, z: 1 }]
  ];
  if (shape !== "capsule") {
    points.push([
      "geometry-top",
      { x: 0, y: size.y / 2, z: 0 },
      { x: 0, y: -1, z: 0 }
    ]);
  }
  return points.map(([id, localPosition, approachDirection]) => ({
    id,
    kind: "grasp",
    localPosition,
    approachDirection,
    compatibleHands: "either",
    clearanceMeters: clearance,
    source: "geometry"
  }));
}

function objectVolume(
  shape: HumanoidObjectCapabilityDescriptor["shape"],
  size: Vec3
): number {
  if (shape === "box") return size.x * size.y * size.z;
  if (shape === "sphere") {
    const radius = Math.min(size.x, size.y, size.z) / 2;
    return 4 * Math.PI * radius ** 3 / 3;
  }
  const radius = Math.min(size.x, size.z) / 2;
  const cylindricalHeight = shape === "capsule"
    ? Math.max(0, size.y - radius * 2)
    : size.y;
  return Math.PI * radius ** 2 * cylindricalHeight
    + (shape === "capsule" ? 4 * Math.PI * radius ** 3 / 3 : 0);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
