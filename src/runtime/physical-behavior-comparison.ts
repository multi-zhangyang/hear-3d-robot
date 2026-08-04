import {
  PhysicalTrajectorySummarySchema,
  type PhysicalTrajectoryFrame,
  type PhysicalTrajectorySummary
} from "../domain/physical-trajectory.js";
import { HUMANOID_END_EFFECTORS } from "../domain/schema.js";

const RESAMPLED_POINT_COUNT = 32;

export interface PhysicalBehaviorDifference {
  materially_different: boolean;
  reasons: string[];
  root_path_rms_m: number;
  final_root_distance_m: number;
  root_path_length_delta_m: number;
  joint_trajectory_rms_rad: number;
  end_effector_trajectory_rms_m: number;
  maximum_object_final_distance_m: number;
  contact_sequence_different: boolean;
  physical_frame_count_delta: number;
}

export function comparePhysicalTrajectories(
  leftRaw: readonly unknown[],
  rightRaw: readonly unknown[]
): PhysicalBehaviorDifference {
  const left = leftRaw.map((entry) => PhysicalTrajectorySummarySchema.parse(entry));
  const right = rightRaw.map((entry) => PhysicalTrajectorySummarySchema.parse(entry));
  if (left.length === 0 || right.length === 0) {
    throw new Error("Physical behavior comparison requires trajectories from both runs");
  }
  const leftFrames = orderedFrames(left);
  const rightFrames = orderedFrames(right);
  const rootPathRms = seriesRms(
    leftFrames.map((frame) => vector(frame.root_position)),
    rightFrames.map((frame) => vector(frame.root_position))
  );
  const jointRms = seriesRms(
    leftFrames.map((frame) => frame.joint_positions),
    rightFrames.map((frame) => frame.joint_positions)
  );
  const endEffectorRms = seriesRms(
    leftFrames.map(endEffectorVector),
    rightFrames.map(endEffectorVector)
  );
  const leftFinal = leftFrames.at(-1)!;
  const rightFinal = rightFrames.at(-1)!;
  const finalRootDistance = euclidean(
    vector(leftFinal.root_position),
    vector(rightFinal.root_position)
  );
  const rootPathLengthDelta = Math.abs(
    sum(left, "root_path_length_m") - sum(right, "root_path_length_m")
  );
  const objectDistance = maximumObjectDistance(leftFinal, rightFinal);
  const contactSequenceDifferent = contactSequence(leftFrames)
    !== contactSequence(rightFrames);
  const physicalFrameCountDelta = Math.abs(frameCount(left) - frameCount(right));
  const reasons = [
    ...(rootPathRms >= 0.08 ? ["root_path"] : []),
    ...(finalRootDistance >= 0.1 ? ["final_root"] : []),
    ...(rootPathLengthDelta >= 0.15 ? ["root_path_length"] : []),
    ...(jointRms >= 0.06 ? ["joint_trajectory"] : []),
    ...(endEffectorRms >= 0.06 ? ["end_effector_trajectory"] : []),
    ...(objectDistance >= 0.05 ? ["object_motion"] : []),
    ...(contactSequenceDifferent
      && (jointRms >= 0.025 || endEffectorRms >= 0.025 || objectDistance >= 0.02)
      ? ["contact_sequence"]
      : [])
  ];
  return {
    materially_different: reasons.length > 0,
    reasons,
    root_path_rms_m: rounded(rootPathRms),
    final_root_distance_m: rounded(finalRootDistance),
    root_path_length_delta_m: rounded(rootPathLengthDelta),
    joint_trajectory_rms_rad: rounded(jointRms),
    end_effector_trajectory_rms_m: rounded(endEffectorRms),
    maximum_object_final_distance_m: rounded(objectDistance),
    contact_sequence_different: contactSequenceDifferent,
    physical_frame_count_delta: physicalFrameCountDelta
  };
}

function orderedFrames(
  trajectories: readonly PhysicalTrajectorySummary[]
): PhysicalTrajectoryFrame[] {
  const byFrame = new Map<number, PhysicalTrajectoryFrame>();
  for (const trajectory of trajectories) {
    for (const frame of trajectory.samples) {
      const existing = byFrame.get(frame.frame);
      if (existing && existing.world_revision !== frame.world_revision) {
        throw new Error(`Physical trajectory frame ${frame.frame} has conflicting revisions`);
      }
      byFrame.set(frame.frame, frame);
    }
  }
  return [...byFrame.values()].sort((left, right) => left.frame - right.frame);
}

function seriesRms(left: readonly number[][], right: readonly number[][]): number {
  const leftSamples = resample(left, RESAMPLED_POINT_COUNT);
  const rightSamples = resample(right, RESAMPLED_POINT_COUNT);
  if (leftSamples[0]!.length !== rightSamples[0]!.length) {
    throw new Error("Physical trajectory dimensions do not match");
  }
  let squared = 0;
  let count = 0;
  for (let sample = 0; sample < RESAMPLED_POINT_COUNT; sample += 1) {
    for (let dimension = 0; dimension < leftSamples[sample]!.length; dimension += 1) {
      const difference = leftSamples[sample]![dimension]!
        - rightSamples[sample]![dimension]!;
      squared += difference * difference;
      count += 1;
    }
  }
  return Math.sqrt(squared / count);
}

function resample(series: readonly number[][], count: number): number[][] {
  if (series.length === 0) throw new Error("Cannot resample an empty physical trajectory");
  const dimensions = series[0]!.length;
  if (dimensions === 0 || series.some((sample) => sample.length !== dimensions)) {
    throw new Error("Physical trajectory samples have inconsistent dimensions");
  }
  if (series.length === 1) {
    return Array.from({ length: count }, () => [...series[0]!]);
  }
  return Array.from({ length: count }, (_, index) => {
    const position = index * (series.length - 1) / (count - 1);
    const lower = Math.floor(position);
    const upper = Math.min(series.length - 1, Math.ceil(position));
    const blend = position - lower;
    return series[lower]!.map((value, dimension) => (
      value + (series[upper]![dimension]! - value) * blend
    ));
  });
}

function endEffectorVector(frame: PhysicalTrajectoryFrame): number[] {
  return HUMANOID_END_EFFECTORS.flatMap((name) => vector(frame.end_effectors[name]));
}

function vector(value: { x: number; y: number; z: number }): number[] {
  return [value.x, value.y, value.z];
}

function maximumObjectDistance(
  left: PhysicalTrajectoryFrame,
  right: PhysicalTrajectoryFrame
): number {
  const rightObjects = new Map(right.objects.map((object) => [object.id, object.position]));
  let maximum = 0;
  for (const object of left.objects) {
    const other = rightObjects.get(object.id);
    if (other) maximum = Math.max(maximum, euclidean(vector(object.position), vector(other)));
  }
  return maximum;
}

function contactSequence(frames: readonly PhysicalTrajectoryFrame[]): string {
  return JSON.stringify(frames.map((frame) => frame.contacts.map(({ key }) => key)));
}

function frameCount(trajectories: readonly PhysicalTrajectorySummary[]): number {
  return trajectories.reduce((total, trajectory) => (
    total + trajectory.observed_frame_count - 1
  ), 0);
}

function sum(
  trajectories: readonly PhysicalTrajectorySummary[],
  key: "root_path_length_m"
): number {
  return trajectories.reduce((total, trajectory) => total + trajectory[key], 0);
}

function euclidean(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length) throw new Error("Vector dimensions do not match");
  return Math.hypot(...left.map((value, index) => value - right[index]!));
}

function rounded(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
