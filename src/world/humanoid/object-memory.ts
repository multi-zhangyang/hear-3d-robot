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
  kind: string;
  color: string;
  size: Vec3;
  portable: boolean;
  status: "visible" | "remembered";
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

interface ObjectDescriptor {
  id: string;
  kind: string;
  color: string;
  size: Vec3;
  portable: boolean;
}

type ObjectRecord = z.infer<typeof ObjectStateSchema>;

export class HumanoidObjectMemory {
  readonly #descriptors: ReadonlyMap<string, ObjectDescriptor>;
  readonly #records = new Map<string, ObjectRecord>();

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
    for (const [id, object] of Object.entries(visibleObjects)) {
      if (!this.#descriptors.has(id)) {
        throw new Error(`Humanoid sensor returned an unknown object: ${id}`);
      }
      const previous = this.#records.get(id);
      this.#records.set(id, {
        id,
        position: { ...object.position },
        rotation: { ...object.rotation },
        linearVelocity: { ...object.linearVelocity },
        angularVelocity: { ...object.angularVelocity },
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
    worldRevision: number,
    visibleObjectIds: ReadonlySet<string>
  ): HumanoidObjectToken[] {
    const contacts = humanoidObjectContacts(snapshot);
    return [...this.#records.values()]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((record) => {
        const descriptor = this.#descriptors.get(record.id);
        if (!descriptor) throw new Error(`Missing humanoid object descriptor: ${record.id}`);
        const delta = subtract(record.position, snapshot.rootPosition);
        const local = rotateVector(inverseQuaternion(snapshot.rootRotation), delta);
        return {
          id: record.id,
          kind: descriptor.kind,
          color: descriptor.color,
          size: { ...descriptor.size },
          portable: descriptor.portable,
          status: visibleObjectIds.has(record.id) ? "visible" : "remembered",
          position: { ...record.position },
          rotation: { ...record.rotation },
          linearVelocity: { ...record.linearVelocity },
          angularVelocity: { ...record.angularVelocity },
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
              record.position,
              snapshot.links.left_wrist_yaw_link.position
            ),
            distanceToRightWrist: distance(
              record.position,
              snapshot.links.right_wrist_yaw_link.position
            )
          },
          currentContacts: contacts
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

  observedObjectIds(worldRevision: number): ReadonlySet<string> {
    return new Set([...this.#records.values()]
      .filter((record) => record.lastSeenRevision === worldRevision)
      .map((record) => record.id));
  }
}

function distance(left: Vec3, right: Vec3): number {
  return vectorLength(subtract(left, right));
}
