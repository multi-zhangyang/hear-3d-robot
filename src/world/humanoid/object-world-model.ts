import type { Scenario, Vec3 } from "../../domain/schema.js";
import {
  add,
  inverseQuaternion,
  rotateVector,
  subtract
} from "../geometry.js";
import type { HumanoidObjectToken } from "./object-memory.js";
import { humanoidObjectCapability } from "./object-capability.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

export interface HumanoidObjectWorldModelEntry {
  id: string;
  kind: string;
  role: "manipulable" | "fixture";
  status: "visible" | "remembered";
  authority: "mujoco_exact" | "sensor_history";
  pose: HumanoidObjectToken["pose"];
  size: Vec3;
  shape: "box" | "sphere" | "cylinder" | "capsule";
  physical: {
    mass_kg: number;
    friction: {
      sliding: number;
      torsional: number;
      rolling: number;
    };
    mobility: "fixed" | "free" | "articulated";
  };
  affordances: string[];
  interaction_points: Array<{
    id: string;
    kind: string;
    compatible_hands: "left" | "right" | "either" | "both";
    world_position: Vec3;
    approach_direction_world?: Vec3;
    clearance_m: number;
    source: "authored" | "geometry";
  }>;
  articulation: null | {
    joint_id: string;
    parent_object_id: string | null;
    type: "hinge" | "slide";
    semantic: string;
    axis_world: Vec3;
    anchor_world: Vec3;
    position: number | null;
    velocity: number | null;
    range: { minimum: number; maximum: number };
    closed_position: number;
    open_position: number;
    open_fraction: number | null;
    state: "open" | "closed" | "intermediate" | "unobserved";
  };
  relations: {
    contained_by: string[];
    contains: string[];
    supported_by: string[];
    supports: string[];
    connected_to: string[];
  };
  current_contact_count: number;
}

export interface HumanoidObjectWorldModel {
  frame: number;
  world_revision: number;
  objects: HumanoidObjectWorldModelEntry[];
}

export function createHumanoidObjectWorldModel(input: {
  frame: number;
  worldRevision: number;
  scenario: Scenario;
  robot: HumanoidSimulationSnapshot;
  objectTokens: readonly HumanoidObjectToken[];
}): HumanoidObjectWorldModel {
  const descriptors = new Map(input.scenario.objects.map((object) => [object.id, object]));
  const tokens = new Map(input.objectTokens.map((token) => [token.id, token]));
  const relationSets = new Map<string, {
    containedBy: Set<string>;
    contains: Set<string>;
    supportedBy: Set<string>;
    supports: Set<string>;
    connectedTo: Set<string>;
  }>();
  for (const id of tokens.keys()) {
    relationSets.set(id, {
      containedBy: new Set(),
      contains: new Set(),
      supportedBy: new Set(),
      supports: new Set(),
      connectedTo: new Set()
    });
  }
  for (const [id, token] of tokens) {
    const descriptor = descriptors.get(id);
    if (!descriptor) throw new Error(`Object world model is missing descriptor: ${id}`);
    const capability = humanoidObjectCapability(descriptor);
    const parentId = capability.articulation?.parent_object_id;
    if (parentId && relationSets.has(parentId)) {
      relationSets.get(id)!.connectedTo.add(parentId);
      relationSets.get(parentId)!.connectedTo.add(id);
    }
    for (const [containerId, containerToken] of tokens) {
      if (containerId === id) continue;
      const containerDescriptor = descriptors.get(containerId);
      if (!containerDescriptor) continue;
      const container = humanoidObjectCapability(containerDescriptor).container;
      if (container && objectInsideContainer(token, descriptor.size, containerToken, container)) {
        relationSets.get(id)!.containedBy.add(containerId);
        relationSets.get(containerId)!.contains.add(id);
      }
      const support = humanoidObjectCapability(containerDescriptor).supportSurface;
      if (support && objectOnSupport(token, descriptor.size, containerToken, support)) {
        relationSets.get(id)!.supportedBy.add(containerId);
        relationSets.get(containerId)!.supports.add(id);
      }
    }
  }
  const objects = [...tokens.values()]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((token): HumanoidObjectWorldModelEntry => {
      const descriptor = descriptors.get(token.id);
      if (!descriptor) throw new Error(`Object world model is missing descriptor: ${token.id}`);
      const capability = humanoidObjectCapability(descriptor);
      const snapshot = token.observable ? input.robot.objects[token.id] : undefined;
      const relations = relationSets.get(token.id)!;
      const pose = objectTokenPose(token, snapshot ?? {
        position: descriptor.position,
        rotation: { x: 0, y: 0, z: 0, w: 1 }
      });
      return {
        id: token.id,
        kind: descriptor.kind,
        role: token.role,
        status: token.status,
        authority: token.authority,
        pose: structuredClone(pose),
        size: { ...descriptor.size },
        shape: capability.shape,
        physical: {
          mass_kg: capability.massKg,
          friction: { ...capability.friction },
          mobility: capability.mobility
        },
        affordances: [...capability.affordances],
        interaction_points: capability.interactionPoints.map((point) => ({
          id: point.id,
          kind: point.kind,
          compatible_hands: point.compatibleHands,
          world_position: add(
            pose.position,
            rotateVector(pose.rotation, point.localPosition)
          ),
          ...(point.approachDirection
            ? {
                approach_direction_world: rotateVector(
                  pose.rotation,
                  point.approachDirection
                )
              }
            : {}),
          clearance_m: point.clearanceMeters,
          source: point.source
        })),
        articulation: capability.articulation
          ? articulationState(capability.articulation, snapshot?.articulation)
          : null,
        relations: {
          contained_by: [...relations.containedBy].sort(),
          contains: [...relations.contains].sort(),
          supported_by: [...relations.supportedBy].sort(),
          supports: [...relations.supports].sort(),
          connected_to: [...relations.connectedTo].sort()
        },
        current_contact_count: token.currentContacts?.length ?? 0
      };
    });
  return {
    frame: input.frame,
    world_revision: input.worldRevision,
    objects
  };
}

function articulationState(
  descriptor: NonNullable<ReturnType<typeof humanoidObjectCapability>["articulation"]>,
  observed: HumanoidSimulationSnapshot["objects"][string]["articulation"] | undefined
): NonNullable<HumanoidObjectWorldModelEntry["articulation"]> {
  const openFraction = observed
    ? clamp01(
        (observed.position - descriptor.closed_position)
          / (descriptor.open_position - descriptor.closed_position)
      )
    : null;
  return {
    joint_id: descriptor.joint_id,
    parent_object_id: descriptor.parent_object_id ?? null,
    type: descriptor.type,
    semantic: descriptor.semantic,
    axis_world: { ...descriptor.axis },
    anchor_world: { ...descriptor.anchor_world },
    position: observed?.position ?? null,
    velocity: observed?.velocity ?? null,
    range: { ...descriptor.range },
    closed_position: descriptor.closed_position,
    open_position: descriptor.open_position,
    open_fraction: openFraction,
    state: openFraction === null
      ? "unobserved"
      : openFraction >= 0.9
        ? "open"
        : openFraction <= 0.1 ? "closed" : "intermediate"
  };
}

function objectInsideContainer(
  object: HumanoidObjectToken,
  objectSize: Vec3,
  container: HumanoidObjectToken,
  descriptor: NonNullable<ReturnType<typeof humanoidObjectCapability>["container"]>
): boolean {
  if (!object.observable || !container.observable) return false;
  return humanoidObjectInsideContainerGeometry({
    object: { ...objectTokenPose(object), size: objectSize },
    container: objectTokenPose(container),
    descriptor,
    tolerance: 0
  });
}

function objectOnSupport(
  object: HumanoidObjectToken,
  objectSize: Vec3,
  support: HumanoidObjectToken,
  descriptor: NonNullable<ReturnType<typeof humanoidObjectCapability>["supportSurface"]>
): boolean {
  if (!object.observable || !support.observable) return false;
  return humanoidObjectOnSupportGeometry({
    object: { ...objectTokenPose(object), size: objectSize },
    support: objectTokenPose(support),
    descriptor,
    tolerance: 0.06
  });
}

export function humanoidObjectInsideContainerGeometry(input: {
  object: { position: Vec3; rotation: HumanoidObjectToken["rotation"]; size: Vec3 };
  container: { position: Vec3; rotation: HumanoidObjectToken["rotation"] };
  descriptor: NonNullable<ReturnType<typeof humanoidObjectCapability>["container"]>;
  tolerance: number;
}): boolean {
  const local = rotateVector(
    inverseQuaternion(input.container.rotation),
    subtract(input.object.position, input.container.position)
  );
  const offset = subtract(local, input.descriptor.interior_center);
  return Math.abs(offset.x) + input.object.size.x / 2
      <= input.descriptor.interior_size.x / 2 + input.tolerance
    && Math.abs(offset.y) + input.object.size.y / 2
      <= input.descriptor.interior_size.y / 2 + input.tolerance
    && Math.abs(offset.z) + input.object.size.z / 2
      <= input.descriptor.interior_size.z / 2 + input.tolerance;
}

export function humanoidObjectOnSupportGeometry(input: {
  object: { position: Vec3; rotation: HumanoidObjectToken["rotation"]; size: Vec3 };
  support: { position: Vec3; rotation: HumanoidObjectToken["rotation"] };
  descriptor: NonNullable<ReturnType<typeof humanoidObjectCapability>["supportSurface"]>;
  tolerance: number;
}): boolean {
  const local = rotateVector(
    inverseQuaternion(input.support.rotation),
    subtract(input.object.position, input.support.position)
  );
  const offset = subtract(local, input.descriptor.local_center);
  const normalDistance = dot(offset, input.descriptor.normal) - objectExtentAlong(
    input.object.size,
    input.descriptor.normal
  );
  return Math.abs(normalDistance) <= input.tolerance
    && Math.abs(offset.x) <= input.descriptor.size.x / 2
      + input.object.size.x / 2 + input.tolerance
    && Math.abs(offset.z) <= input.descriptor.size.z / 2
      + input.object.size.z / 2 + input.tolerance;
}

function objectExtentAlong(size: Vec3, direction: Vec3): number {
  return (Math.abs(direction.x) * size.x
    + Math.abs(direction.y) * size.y
    + Math.abs(direction.z) * size.z) / 2;
}

function dot(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function objectTokenPose(
  token: HumanoidObjectToken,
  fallback?: Pick<HumanoidObjectToken["pose"], "position" | "rotation">
): HumanoidObjectToken["pose"] {
  return token.pose ?? {
    position: { ...(token.position ?? fallback?.position ?? { x: 0, y: 0, z: 0 }) },
    rotation: {
      ...(token.rotation ?? fallback?.rotation ?? { x: 0, y: 0, z: 0, w: 1 })
    }
  };
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
