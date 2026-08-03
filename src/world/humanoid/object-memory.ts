import { z } from "zod";
import type { Scenario, Vec3 } from "../../domain/schema.js";
import { QuaternionSchema, Vec3Schema } from "../../domain/schema.js";
import {
  inverseQuaternion,
  rotateVector,
  subtract,
  vectorLength
} from "../geometry.js";
import { humanoidObjectContacts } from "./motion-plan.js";
import {
  historicalHumanoidObjectState,
  HumanoidAuthoritativeObjectFrame,
  type HumanoidObjectRole,
  type HumanoidObjectStateDescriptor,
  type HumanoidRoleObjectState
} from "./object-state.js";
import type {
  HumanoidObjectSnapshot,
  HumanoidSimulationSnapshot
} from "./simulation.js";

const ObjectStateSchema = z.object({
  id: z.string().trim().min(1),
  position: Vec3Schema,
  rotation: QuaternionSchema,
  linearVelocity: Vec3Schema,
  angularVelocity: Vec3Schema,
  firstSeenRevision: z.number().int().nonnegative(),
  lastSeenRevision: z.number().int().nonnegative(),
  lastSeenFrame: z.number().int().nonnegative(),
  observationCount: z.number().int().positive()
}).strict();

export const HumanoidObjectMemoryCheckpointSchema = z.object({
  version: z.literal(1),
  records: z.array(ObjectStateSchema)
}).strict();

export type HumanoidObjectMemoryCheckpoint = z.infer<
  typeof HumanoidObjectMemoryCheckpointSchema
>;

export interface HumanoidObjectToken {
  id: string;
  role: HumanoidObjectRole;
  kind: string;
  color: string;
  size: Vec3;
  portable: boolean;
  status: "visible" | "remembered";
  state: HumanoidRoleObjectState["state"];
  authority: HumanoidRoleObjectState["authority"];
  exact: boolean;
  observable: boolean;
  pose: HumanoidRoleObjectState["pose"];
  observedFrame: number;
  observedWorldRevision: number;
  position: Vec3;
  rotation: HumanoidObjectSnapshot["rotation"];
  linearVelocity: Vec3;
  angularVelocity: Vec3;
  firstSeenRevision: number;
  lastSeenRevision: number;
  lastSeenFrame: number;
  observationCount: number;
  ageRevisions: number;
  relation: {
    distanceToRobot: number;
    bearingRadians: number;
    verticalOffset: number;
    distanceToLeftWrist: number;
    distanceToRightWrist: number;
  };
  currentContacts: Array<{
    body: string;
    normalForce: number;
  }>;
}

interface ObjectDescriptor extends HumanoidObjectStateDescriptor {
  color: string;
}

type ObjectRecord = z.infer<typeof ObjectStateSchema>;

export class HumanoidObjectMemory {
  readonly #descriptors: ReadonlyMap<string, ObjectDescriptor>;
  readonly #records = new Map<string, ObjectRecord>();
  readonly #authoritativeFrame: HumanoidAuthoritativeObjectFrame;

  constructor(
    scenario: Scenario,
    checkpoint?: HumanoidObjectMemoryCheckpoint
  ) {
    this.#descriptors = new Map(scenario.objects.map((object) => [object.id, {
      id: object.id,
      kind: object.kind,
      color: object.color,
      size: { ...object.size },
      portable: object.portable
    }]));
    this.#authoritativeFrame = new HumanoidAuthoritativeObjectFrame(
      this.#descriptors.values()
    );
    if (!checkpoint) return;
    const parsed = HumanoidObjectMemoryCheckpointSchema.parse(checkpoint);
    for (const record of parsed.records) {
      if (!this.#descriptors.has(record.id)) {
        throw new Error(`Humanoid object memory references an unknown object: ${record.id}`);
      }
      this.#records.set(record.id, structuredClone(record));
    }
  }

  observe(
    frame: number,
    worldRevision: number,
    visibleObjects: Readonly<Record<string, HumanoidObjectSnapshot>>
  ): void {
    this.refresh(
      frame,
      worldRevision,
      visibleObjects,
      new Set(Object.keys(visibleObjects))
    );
  }

  refresh(
    frame: number,
    worldRevision: number,
    authoritativeObjects: Readonly<Record<string, HumanoidObjectSnapshot>>,
    observableObjectIds: ReadonlySet<string>
  ): void {
    this.#authoritativeFrame.refresh(
      frame,
      worldRevision,
      authoritativeObjects,
      observableObjectIds
    );
    for (const state of this.#authoritativeFrame.observableStates(frame, worldRevision)) {
      const id = state.id;
      const previous = this.#records.get(id);
      this.#records.set(id, {
        id,
        position: { ...state.pose.position },
        rotation: { ...state.pose.rotation },
        linearVelocity: { ...state.linearVelocity },
        angularVelocity: { ...state.angularVelocity },
        firstSeenRevision: previous?.firstSeenRevision ?? worldRevision,
        lastSeenRevision: worldRevision,
        lastSeenFrame: frame,
        observationCount: previous
          && (previous.lastSeenRevision !== worldRevision || previous.lastSeenFrame !== frame)
          ? previous.observationCount + 1
          : previous?.observationCount ?? 1
      });
    }
  }

  tokens(
    snapshot: HumanoidSimulationSnapshot,
    frame: number,
    worldRevision: number
  ): HumanoidObjectToken[] {
    const contacts = humanoidObjectContacts(snapshot);
    const observable = new Map(this.#authoritativeFrame
      .observableStates(frame, worldRevision)
      .map((state) => [state.id, state]));
    return [...this.#records.values()]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((record) => {
        const descriptor = this.#descriptors.get(record.id);
        if (!descriptor) throw new Error(`Missing humanoid object descriptor: ${record.id}`);
        const active = observable.get(record.id);
        const state = active ?? historicalHumanoidObjectState(descriptor, record);
        const position = state.pose.position;
        const delta = subtract(position, snapshot.rootPosition);
        const local = rotateVector(inverseQuaternion(snapshot.rootRotation), delta);
        return {
          id: record.id,
          role: state.role,
          kind: descriptor.kind,
          color: descriptor.color,
          size: { ...descriptor.size },
          portable: descriptor.portable,
          status: active ? "visible" : "remembered",
          state: state.state,
          authority: state.authority,
          exact: state.exact,
          observable: state.observable,
          pose: structuredClone(state.pose),
          observedFrame: record.lastSeenFrame,
          observedWorldRevision: record.lastSeenRevision,
          position: { ...position },
          rotation: { ...state.pose.rotation },
          linearVelocity: { ...state.linearVelocity },
          angularVelocity: { ...state.angularVelocity },
          firstSeenRevision: record.firstSeenRevision,
          lastSeenRevision: record.lastSeenRevision,
          lastSeenFrame: record.lastSeenFrame,
          observationCount: record.observationCount,
          ageRevisions: Math.max(0, worldRevision - record.lastSeenRevision),
          relation: {
            distanceToRobot: vectorLength(delta),
            bearingRadians: Math.atan2(local.x, local.z),
            verticalOffset: delta.y,
            distanceToLeftWrist: distance(
              position,
              snapshot.links.left_wrist_yaw_link.position
            ),
            distanceToRightWrist: distance(
              position,
              snapshot.links.right_wrist_yaw_link.position
            )
          },
          currentContacts: (active ? contacts : [])
            .filter((contact) => contact.objectId === record.id)
            .map((contact) => ({
              body: contact.body,
              normalForce: contact.normalForce
            }))
        };
      });
  }

  checkpoint(): HumanoidObjectMemoryCheckpoint {
    return HumanoidObjectMemoryCheckpointSchema.parse({
      version: 1,
      records: [...this.#records.values()].map((record) => structuredClone(record))
    });
  }

  observedObjectIds(frame: number, worldRevision: number): ReadonlySet<string> {
    return new Set(this.#authoritativeFrame
      .observableStates(frame, worldRevision)
      .map((state) => state.id));
  }

  observableObjectStates(
    frame: number,
    worldRevision: number
  ): HumanoidRoleObjectState[] {
    return this.#authoritativeFrame.observableStates(frame, worldRevision);
  }

  activeObjectStates(): HumanoidRoleObjectState[] {
    return this.#authoritativeFrame.activeStates();
  }
}

function distance(left: Vec3, right: Vec3): number {
  return vectorLength(subtract(left, right));
}
