import {
  HUMANOID_END_EFFECTORS,
  type HumanoidEndEffector,
  type Vec3
} from "../../domain/schema.js";
import {
  PhysicalTrajectoryFrameSchema,
  PhysicalTrajectorySummarySchema,
  advancePhysicalTrajectorySha256,
  physicalTrajectoryFrameSha256,
  type PhysicalTrajectoryFrame,
  type PhysicalTrajectorySummary
} from "../../domain/physical-trajectory.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import { HUMANOID_JOINT_NAMES } from "../../world/humanoid/model.js";
import { G1_HAND_JOINT_NAMES } from "../../world/humanoid/morphology.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";

const MAX_TRAJECTORY_SAMPLES = 64;
const JOINT_NAMES = [...HUMANOID_JOINT_NAMES, ...G1_HAND_JOINT_NAMES]
  .sort(compareCodePoints);

export function createPhysicalTrajectory(
  world: HumanoidWorldSnapshot,
  completeFromAdmission = true
): PhysicalTrajectorySummary {
  const frame = captureFrame(world);
  return PhysicalTrajectorySummarySchema.parse({
    version: 1,
    complete_from_admission: completeFromAdmission,
    start_frame: frame.frame,
    end_frame: frame.frame,
    start_world_revision: frame.world_revision,
    end_world_revision: frame.world_revision,
    observed_frame_count: 1,
    sample_stride: 1,
    joint_names: JOINT_NAMES,
    samples: [frame],
    root_path_length_m: 0,
    root_planar_path_length_m: 0,
    joint_total_variation_rad: 0,
    end_effector_path_length_m: endEffectorNumbers(0),
    object_path_length_m: Object.fromEntries(frame.objects.map(({ id }) => [id, 0])),
    contact_transition_count: 0,
    trajectory_sha256: advancePhysicalTrajectorySha256(null, frame.frame_sha256)
  });
}

export function advancePhysicalTrajectory(
  persisted: PhysicalTrajectorySummary,
  world: HumanoidWorldSnapshot
): PhysicalTrajectorySummary {
  const summary = PhysicalTrajectorySummarySchema.parse(persisted);
  const frame = captureFrame(world);
  const previous = summary.samples.at(-1)!;
  if (frame.frame === summary.end_frame
    && frame.world_revision === summary.end_world_revision) {
    return summary;
  }
  if (frame.frame !== summary.end_frame + 1
    || frame.world_revision !== summary.end_world_revision + 1) {
    throw new Error(
      `Physical trajectory is not contiguous: ${summary.end_frame}/${summary.end_world_revision}`
      + ` -> ${frame.frame}/${frame.world_revision}`
    );
  }

  const objectDistance = { ...summary.object_path_length_m };
  const previousObjects = new Map(previous.objects.map((object) => [object.id, object.position]));
  for (const object of frame.objects) {
    objectDistance[object.id] = rounded(
      (objectDistance[object.id] ?? 0)
        + distance(previousObjects.get(object.id), object.position)
    );
  }
  let stride = summary.sample_stride;
  let samples = [...summary.samples];
  const terminal = samples.at(-1)!;
  if (terminal.frame !== summary.start_frame
    && (terminal.frame - summary.start_frame) % stride !== 0) {
    samples.pop();
  }
  samples.push(frame);
  while (samples.length > MAX_TRAJECTORY_SAMPLES) {
    stride *= 2;
    samples = samples.filter((sample, index) => (
      index === 0
        || index === samples.length - 1
        || (sample.frame - summary.start_frame) % stride === 0
    ));
  }

  return PhysicalTrajectorySummarySchema.parse({
    ...summary,
    end_frame: frame.frame,
    end_world_revision: frame.world_revision,
    observed_frame_count: summary.observed_frame_count + 1,
    sample_stride: stride,
    samples,
    root_path_length_m: rounded(
      summary.root_path_length_m + distance(previous.root_position, frame.root_position)
    ),
    root_planar_path_length_m: rounded(
      summary.root_planar_path_length_m
        + Math.hypot(
          frame.root_position.x - previous.root_position.x,
          frame.root_position.z - previous.root_position.z
        )
    ),
    joint_total_variation_rad: rounded(
      summary.joint_total_variation_rad
        + frame.joint_positions.reduce((total, position, index) => (
          total + Math.abs(position - previous.joint_positions[index]!)
        ), 0)
    ),
    end_effector_path_length_m: Object.fromEntries(HUMANOID_END_EFFECTORS.map((name) => [
      name,
      rounded(
        summary.end_effector_path_length_m[name]
          + distance(previous.end_effectors[name], frame.end_effectors[name])
      )
    ])),
    object_path_length_m: objectDistance,
    contact_transition_count: summary.contact_transition_count
      + (sameContactState(previous, frame) ? 0 : 1),
    trajectory_sha256: advancePhysicalTrajectorySha256(
      summary.trajectory_sha256,
      frame.frame_sha256
    )
  });
}

function captureFrame(world: HumanoidWorldSnapshot): PhysicalTrajectoryFrame {
  const robot = world.robot;
  const identity = {
    frame: world.frame,
    world_revision: world.worldRevision,
    root_position: vector(robot.rootPosition),
    root_rotation: {
      x: rounded(robot.rootRotation.x),
      y: rounded(robot.rootRotation.y),
      z: rounded(robot.rootRotation.z),
      w: rounded(robot.rootRotation.w)
    },
    joint_positions: JOINT_NAMES.map((name) => rounded(
      name in robot.joints
        ? robot.joints[name as keyof typeof robot.joints]!.position
        : robot.hands.joints[name as keyof typeof robot.hands.joints]!.position
    )),
    end_effectors: Object.fromEntries(HUMANOID_END_EFFECTORS.map((name) => {
      const position = humanoidEndEffectorPosition(robot, name, "world");
      if (!position) throw new Error(`Missing humanoid end-effector state: ${name}`);
      return [name, vector(position)];
    })) as Record<HumanoidEndEffector, Vec3>,
    contacts: aggregateContacts(robot.contacts),
    objects: Object.values(robot.objects).map((object) => ({
      id: object.id,
      position: vector(object.position)
    })).sort((left, right) => compareCodePoints(left.id, right.id)),
    support: robot.balance.support,
    fallen: robot.fallen
  };
  return PhysicalTrajectoryFrameSchema.parse({
    ...identity,
    frame_sha256: physicalTrajectoryFrameSha256(identity)
  });
}

function aggregateContacts(
  contacts: HumanoidWorldSnapshot["robot"]["contacts"]
): Array<{ key: string; normal_force_n: number }> {
  const forceByKey = new Map<string, number>();
  for (const contact of contacts) {
    const key = contactKey(contact);
    forceByKey.set(key, rounded((forceByKey.get(key) ?? 0) + contact.normalForce));
  }
  return [...forceByKey].map(([key, normalForce]) => ({
    key,
    normal_force_n: normalForce
  })).sort((left, right) => compareCodePoints(left.key, right.key));
}

function contactKey(contact: HumanoidWorldSnapshot["robot"]["contacts"][number]): string {
  const first = contactParty(contact, "first");
  const second = contactParty(contact, "second");
  return [first, second].sort(compareCodePoints).join("<->");
}

function contactParty(
  contact: HumanoidWorldSnapshot["robot"]["contacts"][number],
  side: "first" | "second"
): string {
  const title = side === "first" ? "first" : "second";
  const values = [
    contact[`${title}Body`],
    contact[`${title}Object`],
    contact[`${title}Solid`],
    contact[`${title}HandLink`]
  ].filter((value): value is string => typeof value === "string");
  return values.length > 0 ? values.sort(compareCodePoints).join("+") : "environment";
}

function sameContactState(
  left: PhysicalTrajectoryFrame,
  right: PhysicalTrajectoryFrame
): boolean {
  return JSON.stringify(left.contacts.map(({ key }) => key))
    === JSON.stringify(right.contacts.map(({ key }) => key));
}

function endEffectorNumbers(value: number): Record<HumanoidEndEffector, number> {
  return Object.fromEntries(
    HUMANOID_END_EFFECTORS.map((name) => [name, value])
  ) as Record<HumanoidEndEffector, number>;
}

function vector(value: Vec3): Vec3 {
  return { x: rounded(value.x), y: rounded(value.y), z: rounded(value.z) };
}

function distance(left: Vec3 | undefined, right: Vec3): number {
  return left
    ? Math.hypot(right.x - left.x, right.y - left.y, right.z - left.z)
    : 0;
}

function rounded(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
