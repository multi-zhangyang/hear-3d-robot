import { z } from "zod";
import type { Quaternion, Vec3 } from "../../domain/schema.js";
import { QuaternionSchema, Vec3Schema } from "../../domain/schema.js";
import type { HumanoidObjectSnapshot } from "./simulation.js";

const HumanoidObjectRoleSchema = z.enum(["manipulable", "fixture"]);
export type HumanoidObjectRole = z.infer<typeof HumanoidObjectRoleSchema>;

export interface HumanoidObjectStateDescriptor {
  id: string;
  kind: string;
  size: Vec3;
  portable: boolean;
}

const HumanoidRoleObjectStateSchema = z.object({
  id: z.string().trim().min(1),
  role: HumanoidObjectRoleSchema,
  kind: z.string().trim().min(1),
  size: Vec3Schema,
  portable: z.boolean(),
  state: z.enum(["active", "historical"]),
  authority: z.enum(["mujoco_exact", "sensor_history"]),
  exact: z.boolean(),
  observable: z.boolean(),
  pose: z.object({
    position: Vec3Schema,
    rotation: QuaternionSchema
  }).strict(),
  linearVelocity: Vec3Schema,
  angularVelocity: Vec3Schema,
  frame: z.number().int().nonnegative(),
  worldRevision: z.number().int().nonnegative(),
  observedFrame: z.number().int().nonnegative().nullable(),
  observedWorldRevision: z.number().int().nonnegative().nullable()
}).strict().superRefine((value, context) => {
  const activeExact = value.state === "active"
    && value.authority === "mujoco_exact"
    && value.exact;
  const historical = value.state === "historical"
    && value.authority === "sensor_history"
    && !value.exact
    && !value.observable;
  if (!activeExact && !historical) {
    context.addIssue({
      code: "custom",
      path: ["state"],
      message: "Humanoid object state authority does not match its lifecycle"
    });
  }
  if (value.observable && (value.observedFrame !== value.frame
    || value.observedWorldRevision !== value.worldRevision)) {
    context.addIssue({
      code: "custom",
      path: ["observable"],
      message: "Observable humanoid object state must belong to the current frame and revision"
    });
  }
  if (value.state === "active" && !value.observable
    && (value.observedFrame !== null || value.observedWorldRevision !== null)) {
    context.addIssue({
      code: "custom",
      path: ["observedFrame"],
      message: "Unobservable active humanoid state cannot carry a sensor observation identity"
    });
  }
  if (value.state === "historical"
    && (value.observedFrame !== value.frame
      || value.observedWorldRevision !== value.worldRevision)) {
    context.addIssue({
      code: "custom",
      path: ["observedFrame"],
      message: "Historical humanoid object state must retain its observation identity"
    });
  }
});

export type HumanoidRoleObjectState = z.infer<typeof HumanoidRoleObjectStateSchema>;

export interface HumanoidHistoricalObjectObservation {
  id: string;
  position: Vec3;
  rotation: Quaternion;
  linearVelocity: Vec3;
  angularVelocity: Vec3;
  lastSeenFrame: number;
  lastSeenRevision: number;
}

export class HumanoidAuthoritativeObjectFrame {
  readonly #descriptors: ReadonlyMap<string, HumanoidObjectStateDescriptor>;
  readonly #states = new Map<string, HumanoidRoleObjectState>();
  #frame: number | null = null;
  #worldRevision: number | null = null;

  constructor(descriptors: Iterable<HumanoidObjectStateDescriptor>) {
    const entries = [...descriptors].map((descriptor) => [
      descriptor.id,
      cloneDescriptor(descriptor)
    ] as const);
    this.#descriptors = new Map(entries);
    if (this.#descriptors.size !== entries.length) {
      throw new Error("Duplicate humanoid object state descriptor");
    }
  }

  refresh(
    frame: number,
    worldRevision: number,
    authoritativeObjects: Readonly<Record<string, HumanoidObjectSnapshot>>,
    observableObjectIds: ReadonlySet<string>
  ): void {
    if (!Number.isSafeInteger(frame) || frame < 0
      || !Number.isSafeInteger(worldRevision) || worldRevision < 0) {
      throw new Error("Humanoid object state identity must be nonnegative safe integers");
    }
    for (const id of observableObjectIds) {
      if (!authoritativeObjects[id]) {
        throw new Error(`Observable humanoid object has no authoritative state: ${id}`);
      }
    }
    const next = new Map<string, HumanoidRoleObjectState>();
    for (const [id, object] of Object.entries(authoritativeObjects)) {
      const descriptor = this.#descriptors.get(id);
      if (!descriptor) {
        throw new Error(`Authoritative humanoid state references an unknown object: ${id}`);
      }
      if (object.id !== id) {
        throw new Error(`Authoritative humanoid object identity mismatch: ${id}`);
      }
      const observable = observableObjectIds.has(id);
      next.set(id, HumanoidRoleObjectStateSchema.parse({
        id,
        role: humanoidObjectRole(descriptor),
        kind: descriptor.kind,
        size: { ...descriptor.size },
        portable: descriptor.portable,
        state: "active",
        authority: "mujoco_exact",
        exact: true,
        observable,
        pose: {
          position: { ...object.position },
          rotation: { ...object.rotation }
        },
        linearVelocity: { ...object.linearVelocity },
        angularVelocity: { ...object.angularVelocity },
        frame,
        worldRevision,
        observedFrame: observable ? frame : null,
        observedWorldRevision: observable ? worldRevision : null
      }));
    }
    this.#states.clear();
    for (const [id, state] of next) this.#states.set(id, state);
    this.#frame = frame;
    this.#worldRevision = worldRevision;
  }

  activeStates(): HumanoidRoleObjectState[] {
    return [...this.#states.values()]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map(cloneState);
  }

  observableStates(frame: number, worldRevision: number): HumanoidRoleObjectState[] {
    if (frame !== this.#frame || worldRevision !== this.#worldRevision) return [];
    return this.activeStates().filter((state) => state.observable);
  }
}

export function historicalHumanoidObjectState(
  descriptor: HumanoidObjectStateDescriptor,
  observation: HumanoidHistoricalObjectObservation
): HumanoidRoleObjectState {
  if (descriptor.id !== observation.id) {
    throw new Error(`Historical humanoid object identity mismatch: ${observation.id}`);
  }
  return HumanoidRoleObjectStateSchema.parse({
    id: observation.id,
    role: humanoidObjectRole(descriptor),
    kind: descriptor.kind,
    size: { ...descriptor.size },
    portable: descriptor.portable,
    state: "historical",
    authority: "sensor_history",
    exact: false,
    observable: false,
    pose: {
      position: { ...observation.position },
      rotation: { ...observation.rotation }
    },
    linearVelocity: { ...observation.linearVelocity },
    angularVelocity: { ...observation.angularVelocity },
    frame: observation.lastSeenFrame,
    worldRevision: observation.lastSeenRevision,
    observedFrame: observation.lastSeenFrame,
    observedWorldRevision: observation.lastSeenRevision
  });
}

function humanoidObjectRole(
  descriptor: Pick<HumanoidObjectStateDescriptor, "portable">
): HumanoidObjectRole {
  return descriptor.portable ? "manipulable" : "fixture";
}

function cloneDescriptor(
  descriptor: HumanoidObjectStateDescriptor
): HumanoidObjectStateDescriptor {
  return {
    id: descriptor.id,
    kind: descriptor.kind,
    size: { ...descriptor.size },
    portable: descriptor.portable
  };
}

function cloneState(state: HumanoidRoleObjectState): HumanoidRoleObjectState {
  return structuredClone(state);
}
