import type { JsonValue, Vec3 } from "../domain/schema.js";
import {
  armReachMetrics,
  type ArmReachMetrics
} from "./arm-reach-diagnosis.js";

export type ArmWorkspaceFit =
  | "preferred"
  | "folded"
  | "out_of_span"
  | "off_plane"
  | "unknown";

export const VOXEL_AFFORDANCE_CONTRACT_VERSION = 2;

export interface VoxelInteractionTarget {
  normal: Vec3;
  interaction_point: Vec3;
}

export interface VoxelStandoffCandidate {
  target: Vec3;
  radius: number;
  distanceToEntity: number;
  distanceToRobot: number;
  axisAlignmentError: number;
}

export interface RankedVoxelStandoff {
  candidate: VoxelStandoffCandidate;
  interaction: VoxelInteractionTarget | null;
  metrics: ArmReachMetrics | null;
  fit: ArmWorkspaceFit;
}

export function armWorkspaceFit(metrics: ArmReachMetrics | null): ArmWorkspaceFit {
  if (!metrics) return "unknown";
  if (!metrics.targetWithinReach) return "out_of_span";
  if (metrics.armMotionPlaneLateralError > metrics.armMotionPlaneTolerance) {
    return "off_plane";
  }
  return metrics.targetDistanceFromShoulder < metrics.preferredMinimumArmReach
    ? "folded"
    : "preferred";
}

export function voxelAffordanceContractStale(
  action: string,
  detail: JsonValue | null
): boolean {
  if (action !== "inspect_voxel" && action !== "scan_voxels") return false;
  if (typeof detail !== "object" || detail === null || Array.isArray(detail)) return true;
  return detail.affordance_contract_version !== VOXEL_AFFORDANCE_CONTRACT_VERSION;
}

/** Preserve material/provenance evidence from an older observation while
 * removing geometry ranked under a superseded affordance contract. */
export function withoutVoxelDynamicAffordances(detail: JsonValue): JsonValue {
  if (typeof detail !== "object" || detail === null || Array.isArray(detail)) return detail;
  const {
    reachable_standoff_poses: _standoffs,
    exposed_faces: _faces,
    placement_interaction_points: _placementPoints,
    ...stable
  } = detail;
  return {
    ...stable,
    affordance_contract_stale: true,
    omitted_dynamic_affordances: [
      "reachable_standoff_poses",
      "exposed_faces",
      "placement_interaction_points"
    ],
    recovery: "Call inspect_voxel again under the current affordance contract before planning or arm actuation."
  };
}

export function rankVoxelStandoffs(input: {
  candidates: readonly VoxelStandoffCandidate[];
  interactions: readonly VoxelInteractionTarget[];
}): RankedVoxelStandoff[] {
  return input.candidates.map((candidate): RankedVoxelStandoff => {
    const rankedTargets = input.interactions.map((interaction) => {
      const yaw = Math.atan2(
        interaction.interaction_point.x - candidate.target.x,
        interaction.interaction_point.z - candidate.target.z
      );
      const metrics = armReachMetrics({
        base: candidate.target,
        yaw,
        target: interaction.interaction_point
      });
      return { interaction, metrics, fit: armWorkspaceFit(metrics) };
    }).sort(compareInteractionFits);
    const best = rankedTargets[0];
    return best
      ? { candidate, ...best }
      : { candidate, interaction: null, metrics: null, fit: "unknown" };
  }).sort((left, right) =>
    fitRank(left.fit) - fitRank(right.fit)
    || comfortError(left.metrics) - comfortError(right.metrics)
    || left.candidate.distanceToRobot - right.candidate.distanceToRobot
    || left.candidate.axisAlignmentError - right.candidate.axisAlignmentError
    || left.candidate.radius - right.candidate.radius
  );
}

function compareInteractionFits(
  left: { metrics: ArmReachMetrics; fit: ArmWorkspaceFit },
  right: { metrics: ArmReachMetrics; fit: ArmWorkspaceFit }
): number {
  return fitRank(left.fit) - fitRank(right.fit)
    || comfortError(left.metrics) - comfortError(right.metrics);
}

function fitRank(fit: ArmWorkspaceFit): number {
  if (fit === "preferred") return 0;
  if (fit === "folded") return 1;
  if (fit === "off_plane") return 2;
  if (fit === "out_of_span") return 3;
  return 4;
}

function comfortError(metrics: ArmReachMetrics | null): number {
  if (!metrics) return Number.POSITIVE_INFINITY;
  const middle = (metrics.preferredMinimumArmReach + metrics.maximumArmReach) / 2;
  return Math.abs(metrics.targetDistanceFromShoulder - middle);
}
