import type {
  FrameTransforms,
  JointStates,
  SceneUpdate,
  Time
} from "@foxglove/schemas";
import {
  FrameTransforms as FrameTransformsSchema,
  JointStates as JointStatesSchema,
  SceneUpdate as SceneUpdateSchema
} from "@foxglove/schemas/jsonschema.js";
import { McapWriter, type IWritable } from "@mcap/core";
import type { JsonValue } from "../domain/schema.js";
import type { AnyRunCheckpoint } from "../domain/run-checkpoint.js";
import type { RunDefinition } from "../persistence/run-store.js";
import type { RuntimeEvent } from "../runtime/events.js";
import { HumanoidWorldSnapshotSchema } from
  "../world/humanoid/snapshot-schema.js";
import type { HumanoidWorldSnapshot } from
  "../world/humanoid/world-contract.js";

const encoder = new TextEncoder();
const WORLD_FRAME = "hear/world";

const RuntimeEventJsonSchema = {
  title: "hear.RuntimeEvent",
  description: "One durable HEAR Harness or physical runtime event.",
  type: "object",
  properties: {
    event_id: { type: "string" },
    run_id: { type: "string" },
    type: { type: "string" },
    at: { type: "string", format: "date-time" },
    data: {}
  },
  required: ["event_id", "run_id", "type", "at", "data"],
  additionalProperties: true
} as const;

const WorldSnapshotJsonSchema = {
  title: "hear.HumanoidWorldSnapshot",
  description: "Authoritative HEAR MuJoCo projection. The source coordinate system is Y-up.",
  type: "object",
  properties: {
    frame: { type: "integer", minimum: 0 },
    worldRevision: { type: "integer", minimum: 0 },
    robot: {
      type: "object",
      properties: {
        simulatedTime: { type: "number", minimum: 0 },
        contactCount: { type: "integer", minimum: 0 },
        fallen: { type: "boolean" },
        balance: {
          type: "object",
          properties: {
            upright: { type: "number" },
            supportMargin: { type: ["number", "null"] },
            support: { type: "string" }
          },
          additionalProperties: true
        },
        feet: { type: "object", additionalProperties: true },
        joints: { type: "object", additionalProperties: true }
      },
      required: ["simulatedTime", "contactCount", "fallen", "balance"],
      additionalProperties: true
    },
    navigation: { type: "object", additionalProperties: true }
  },
  required: ["frame", "worldRevision", "robot", "navigation"],
  additionalProperties: true
} as const;

export interface FoxgloveMcapArtifact {
  data: Buffer;
  messageCount: number;
  snapshotCount: number;
}

interface PendingMessage {
  channelId: number;
  logTime: bigint;
  order: number;
  data: Uint8Array;
}

/**
 * Builds a self-contained, read-only MCAP projection for Foxglove. The export
 * deliberately contains no command channel and cannot be used to bypass the
 * Serial Executor's sole-writer boundary.
 */
export async function buildFoxgloveMcap(input: {
  definition: RunDefinition;
  checkpoint: AnyRunCheckpoint;
  events: readonly JsonValue[];
}): Promise<FoxgloveMcapArtifact> {
  const output = new MemoryWritable();
  const writer = new McapWriter({ writable: output, chunkSize: 4 * 1024 * 1024 });
  await writer.start({ profile: "foxglove", library: "HEAR" });

  const runtimeEventChannel = await registerJsonChannel(
    writer,
    "/hear/runtime/events",
    RuntimeEventJsonSchema
  );
  const snapshotChannel = await registerJsonChannel(
    writer,
    "/hear/world/snapshot",
    WorldSnapshotJsonSchema
  );
  const transformChannel = await registerJsonChannel(
    writer,
    "/hear/world/transforms",
    FrameTransformsSchema
  );
  const jointChannel = await registerJsonChannel(
    writer,
    "/hear/world/joints",
    JointStatesSchema
  );
  const sceneChannel = await registerJsonChannel(
    writer,
    "/hear/world/scene",
    SceneUpdateSchema
  );

  await writer.addMetadata({
    name: "hear.run",
    metadata: new Map([
      ["run_id", input.definition.run_id],
      ["runtime", input.definition.runtime],
      ["scenario_id", input.definition.scenario_id],
      ["authority", "read-only projection; Serial Executor is the sole physical writer"],
      ["source_coordinates", "HEAR Y-up"],
      ["foxglove_coordinates", "Z-up via +90deg X basis rotation"]
    ])
  });

  const epoch = isoNanoseconds(input.definition.created_at, 0n);
  const messages: PendingMessage[] = [];
  const snapshotIdentities = new Set<string>();
  let order = 0;
  let snapshotCount = 0;

  const appendSnapshot = (snapshot: HumanoidWorldSnapshot): void => {
    const identity = `${snapshot.frame}:${snapshot.worldRevision}`;
    if (snapshotIdentities.has(identity)) return;
    snapshotIdentities.add(identity);
    snapshotCount += 1;
    const logTime = epoch + secondsNanoseconds(snapshot.robot.simulatedTime);
    messages.push(
      message(snapshotChannel, logTime, order++, snapshot),
      message(transformChannel, logTime, order++, frameTransforms(snapshot, logTime)),
      message(jointChannel, logTime, order++, jointStates(snapshot, logTime)),
      message(sceneChannel, logTime, order++, diagnosticScene(snapshot, logTime))
    );
  };

  for (const raw of input.events) {
    const event = runtimeEvent(raw, input.definition.run_id);
    if (!event) continue;
    messages.push(message(
      runtimeEventChannel,
      isoNanoseconds(event.at, epoch),
      order++,
      event
    ));
    for (const snapshot of worldSnapshots(event.data)) appendSnapshot(snapshot);
  }

  if (input.checkpoint.runtime === "humanoid_g1") {
    appendSnapshot(input.checkpoint.world);
  }

  messages.sort((left, right) => (
    left.logTime < right.logTime ? -1
      : left.logTime > right.logTime ? 1
        : left.order - right.order
  ));
  const sequences = new Map<number, number>();
  for (const pending of messages) {
    const sequence = sequences.get(pending.channelId) ?? 0;
    sequences.set(pending.channelId, sequence + 1);
    await writer.addMessage({
      channelId: pending.channelId,
      sequence,
      logTime: pending.logTime,
      publishTime: pending.logTime,
      data: pending.data
    });
  }
  await writer.end();
  return {
    data: output.buffer(),
    messageCount: messages.length,
    snapshotCount
  };
}

async function registerJsonChannel(
  writer: McapWriter,
  topic: string,
  schema: { title: string }
): Promise<number> {
  const schemaId = await writer.registerSchema({
    name: schema.title,
    encoding: "jsonschema",
    data: jsonBytes(schema)
  });
  return writer.registerChannel({
    schemaId,
    topic,
    messageEncoding: "json",
    metadata: new Map([
      ["hear.authority", "read_only"],
      ["hear.physical_writer", "false"]
    ])
  });
}

function message(
  channelId: number,
  logTime: bigint,
  order: number,
  value: unknown
): PendingMessage {
  return { channelId, logTime, order, data: jsonBytes(value) };
}

function jsonBytes(value: unknown): Uint8Array {
  return encoder.encode(JSON.stringify(value));
}

function runtimeEvent(value: JsonValue, runId: string): RuntimeEvent | null {
  if (!isRecord(value)
    || typeof value.event_id !== "string"
    || typeof value.run_id !== "string"
    || value.run_id !== runId
    || typeof value.type !== "string"
    || typeof value.at !== "string"
    || !("data" in value)) return null;
  return value as unknown as RuntimeEvent;
}

function worldSnapshots(value: JsonValue): HumanoidWorldSnapshot[] {
  const record = isRecord(value) ? value : null;
  if (!record) return [];
  const candidates: unknown[] = [record.world, record.snapshot];
  if (Array.isArray(record.frames)) candidates.push(...record.frames);
  const snapshots: HumanoidWorldSnapshot[] = [];
  for (const candidate of candidates) {
    const parsed = HumanoidWorldSnapshotSchema.safeParse(candidate);
    if (parsed.success) snapshots.push(parsed.data);
  }
  return snapshots;
}

function frameTransforms(
  snapshot: HumanoidWorldSnapshot,
  logTime: bigint
): FrameTransforms {
  const timestamp = foxgloveTime(logTime);
  const links = {
    ...snapshot.robot.links,
    ...snapshot.robot.hands.links
  };
  return {
    transforms: [
      ...Object.entries(links).map(([name, link]) => ({
        timestamp,
        parent_frame_id: WORLD_FRAME,
        child_frame_id: `g1/${name}`,
        translation: zUpVector(link.position),
        rotation: zUpQuaternion(link.rotation)
      })),
      ...Object.values(snapshot.robot.objects).map((object) => ({
        timestamp,
        parent_frame_id: WORLD_FRAME,
        child_frame_id: `objects/${safeFrameId(object.id)}`,
        translation: zUpVector(object.position),
        rotation: zUpQuaternion(object.rotation)
      }))
    ]
  };
}

function jointStates(snapshot: HumanoidWorldSnapshot, logTime: bigint): JointStates {
  return {
    timestamp: foxgloveTime(logTime),
    joints: [
      ...Object.entries(snapshot.robot.joints).map(([name, joint]) => ({
        name,
        position: joint.position,
        velocity: joint.velocity,
        ...(joint.effort ? { effort: joint.effort.appliedNewtonMeters } : {})
      })),
      ...Object.entries(snapshot.robot.hands.joints).map(([name, joint]) => ({
        name,
        position: joint.position,
        velocity: joint.velocity,
        effort: joint.appliedNewtonMeters
      }))
    ]
  };
}

function diagnosticScene(
  snapshot: HumanoidWorldSnapshot,
  logTime: bigint
): SceneUpdate {
  const identity = {
    position: { x: 0, y: 0, z: 0 },
    orientation: { x: 0, y: 0, z: 0, w: 1 }
  };
  const linkSpheres = Object.entries({
    ...snapshot.robot.links,
    ...snapshot.robot.hands.links
  }).map(([name, link]) => ({
    pose: {
      position: zUpVector(link.position),
      orientation: zUpQuaternion(link.rotation)
    },
    size: { x: 0.045, y: 0.045, z: 0.045 },
    color: name.includes("hand")
      ? { r: 0.29, g: 0.88, b: 0.72, a: 0.9 }
      : { r: 0.72, g: 0.8, b: 0.84, a: 0.82 }
  }));
  const contactSpheres = snapshot.robot.contacts.map((contact) => {
    const diameter = Math.min(0.16, 0.025 + Math.sqrt(contact.normalForce) * 0.006);
    const hand = contact.firstHandLink !== null || contact.secondHandLink !== null;
    return {
      pose: { position: zUpVector(contact.position), orientation: identity.orientation },
      size: { x: diameter, y: diameter, z: diameter },
      color: hand
        ? { r: 0.96, g: 0.71, b: 0.28, a: 0.94 }
        : { r: 0.36, g: 0.88, b: 0.76, a: 0.86 }
    };
  });
  const navigationPoints = snapshot.navigation.waypoints.map(zUpVector);
  return {
    deletions: [],
    entities: [{
      timestamp: foxgloveTime(logTime),
      frame_id: WORLD_FRAME,
      id: "hear-humanoid-diagnostics",
      lifetime: { sec: 0, nsec: 0 },
      frame_locked: false,
      metadata: [
        { key: "frame", value: String(snapshot.frame) },
        { key: "world_revision", value: String(snapshot.worldRevision) },
        { key: "support", value: snapshot.robot.balance.support }
      ],
      arrows: [],
      cubes: [],
      spheres: [
        ...linkSpheres,
        ...contactSpheres,
        {
          pose: {
            position: zUpVector(snapshot.robot.balance.centerOfMass),
            orientation: identity.orientation
          },
          size: { x: 0.075, y: 0.075, z: 0.075 },
          color: { r: 0.98, g: 0.73, b: 0.3, a: 1 }
        }
      ],
      cylinders: [],
      lines: navigationPoints.length >= 2 ? [{
        type: 0,
        pose: identity,
        thickness: 0.018,
        scale_invariant: false,
        points: navigationPoints,
        color: { r: 0.3, g: 0.92, b: 0.78, a: 0.92 },
        colors: [],
        indices: []
      }] : [],
      triangles: [],
      texts: [],
      models: []
    }]
  };
}

function foxgloveTime(nanoseconds: bigint): Time {
  const safe = nanoseconds < 0n ? 0n : nanoseconds;
  return {
    sec: Number(safe / 1_000_000_000n),
    nsec: Number(safe % 1_000_000_000n)
  };
}

function isoNanoseconds(value: string, fallback: bigint): bigint {
  const milliseconds = Date.parse(value);
  return Number.isFinite(milliseconds)
    ? BigInt(Math.round(milliseconds)) * 1_000_000n
    : fallback;
}

function secondsNanoseconds(value: number): bigint {
  return BigInt(Math.round(Math.max(0, value) * 1_000_000_000));
}

function zUpVector(vector: { x: number; y: number; z: number }): {
  x: number;
  y: number;
  z: number;
} {
  return { x: vector.x, y: -vector.z, z: vector.y };
}

/** Applies the +90 degree X basis rotation: q' = basis * q * basis^-1. */
function zUpQuaternion(quaternion: {
  x: number;
  y: number;
  z: number;
  w: number;
}): { x: number; y: number; z: number; w: number } {
  const half = Math.SQRT1_2;
  const basis = { x: half, y: 0, z: 0, w: half };
  return multiplyQuaternion(
    multiplyQuaternion(basis, quaternion),
    { x: -half, y: 0, z: 0, w: half }
  );
}

function multiplyQuaternion(
  left: { x: number; y: number; z: number; w: number },
  right: { x: number; y: number; z: number; w: number }
): { x: number; y: number; z: number; w: number } {
  return {
    x: left.w * right.x + left.x * right.w
      + left.y * right.z - left.z * right.y,
    y: left.w * right.y - left.x * right.z
      + left.y * right.w + left.z * right.x,
    z: left.w * right.z + left.x * right.y
      - left.y * right.x + left.z * right.w,
    w: left.w * right.w - left.x * right.x
      - left.y * right.y - left.z * right.z
  };
}

function safeFrameId(value: string): string {
  return value.replace(/[^a-zA-Z0-9_.-]+/g, "_");
}

function isRecord(value: unknown): value is Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

class MemoryWritable implements IWritable {
  readonly #chunks: Buffer[] = [];
  #position = 0n;

  position(): bigint {
    return this.#position;
  }

  async write(value: Uint8Array): Promise<void> {
    const copy = Buffer.from(value);
    this.#chunks.push(copy);
    this.#position += BigInt(copy.byteLength);
  }

  buffer(): Buffer {
    return Buffer.concat(this.#chunks);
  }
}
