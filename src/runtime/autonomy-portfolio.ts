import { z } from "zod";
import { PhysicalTrajectorySummarySchema } from "../domain/physical-trajectory.js";
import { autonomyContentSha256 } from "./autonomy-signature.js";
import {
  comparePhysicalTrajectories,
  type PhysicalBehaviorDifference
} from "./physical-behavior-comparison.js";

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);

const AutonomyRunEvidenceSchema = z.object({
  version: z.literal(3),
  run_id: z.string().trim().min(1),
  scenario_id: z.string().trim().min(1),
  seed: z.number().int().min(0).max(0xffff_ffff),
  status: z.enum(["succeeded", "paused"]),
  physical_verified: z.literal(true),
  world_frame: z.number().int().positive(),
  world_revision: z.number().int().positive(),
  cycle_count: z.number().int().positive(),
  model_call_count: z.number().int().positive(),
  physical_execution_count: z.number().int().positive(),
  physical_frame_count: z.number().int().positive(),
  travelled_distance_m: z.number().finite().nonnegative(),
  selected_goal_hashes: z.array(Sha256Schema).min(1),
  planning_behavior_hashes: z.array(Sha256Schema).min(1),
  model_response_hashes: z.array(Sha256Schema).min(1),
  action_sequence: z.array(z.string().trim().min(1)).min(1),
  goal_signature: Sha256Schema,
  planning_signature: Sha256Schema,
  model_decision_signature: Sha256Schema,
  physical_behavior_signature: Sha256Schema,
  physical_trajectories: z.array(PhysicalTrajectorySummarySchema).min(1),
  world_mutation_hashes: z.array(Sha256Schema)
}).passthrough().superRefine((run, context) => {
  if (run.physical_trajectories.some((trajectory) => (
    !trajectory.complete_from_admission
  ))) {
    context.addIssue({
      code: "custom",
      path: ["physical_trajectories"],
      message: "Autonomy evidence cannot use a partial physical trajectory"
    });
  }
  const physicalFrames = run.physical_trajectories.reduce((total, trajectory) => (
    total + trajectory.observed_frame_count - 1
  ), 0);
  if (physicalFrames !== run.physical_frame_count
    || run.physical_trajectories.length !== run.physical_execution_count) {
    context.addIssue({
      code: "custom",
      path: ["physical_frame_count"],
      message: "Autonomy physical totals do not match their authoritative trajectories"
    });
  }
  const expectedSignature = autonomyContentSha256({
    trajectories: run.physical_trajectories,
    world_mutation_hashes: run.world_mutation_hashes
  });
  if (run.physical_behavior_signature !== expectedSignature) {
    context.addIssue({
      code: "custom",
      path: ["physical_behavior_signature"],
      message: "Physical behavior signature contains nonphysical or missing evidence"
    });
  }
});

type AutonomyRunEvidence = z.infer<typeof AutonomyRunEvidenceSchema>;

export interface AutonomyPortfolioEvaluation {
  version: 1;
  run_count: number;
  initial_state_count: number;
  same_initial_state: boolean;
  distinct_goal_signatures: number;
  distinct_planning_signatures: number;
  distinct_model_decision_signatures: number;
  distinct_physical_behavior_signatures: number;
  materially_distinct_physical_behaviors: number;
  physical_comparisons: Array<PhysicalBehaviorDifference & {
    left_run_id: string;
    right_run_id: string;
  }>;
  autonomous_diversity_observed: boolean;
  runs: AutonomyRunEvidence[];
}

export function evaluateAutonomyPortfolio(
  rawRuns: readonly unknown[]
): AutonomyPortfolioEvaluation {
  const runs = rawRuns.map((run) => AutonomyRunEvidenceSchema.parse(run));
  const initialStates = new Set(runs.map((run) => (
    `${run.scenario_id}\0${String(run.seed)}`
  )));
  const distinctGoalSignatures = uniqueCount(runs, "goal_signature");
  const distinctPlanningSignatures = uniqueCount(runs, "planning_signature");
  const distinctModelDecisionSignatures = uniqueCount(
    runs,
    "model_decision_signature"
  );
  const distinctPhysicalBehaviorSignatures = uniqueCount(
    runs,
    "physical_behavior_signature"
  );
  const physicalComparisons = pairwisePhysicalComparisons(runs);
  const materiallyDistinctPhysicalBehaviors = physicalBehaviorClusterCount(runs);
  return {
    version: 1,
    run_count: runs.length,
    initial_state_count: initialStates.size,
    same_initial_state: initialStates.size === 1,
    distinct_goal_signatures: distinctGoalSignatures,
    distinct_planning_signatures: distinctPlanningSignatures,
    distinct_model_decision_signatures: distinctModelDecisionSignatures,
    distinct_physical_behavior_signatures: distinctPhysicalBehaviorSignatures,
    materially_distinct_physical_behaviors: materiallyDistinctPhysicalBehaviors,
    physical_comparisons: physicalComparisons,
    autonomous_diversity_observed: runs.length >= 2
      && distinctPlanningSignatures >= 2
      && distinctModelDecisionSignatures >= 2
      && distinctPhysicalBehaviorSignatures >= 2
      && materiallyDistinctPhysicalBehaviors >= 2,
    runs: runs.map((run) => structuredClone(run))
  };
}

export function assertAutonomyPortfolio(
  rawRuns: readonly unknown[],
  options: {
    minimumRuns?: number;
    requireSameInitialState?: boolean;
  } = {}
): AutonomyPortfolioEvaluation {
  const minimumRuns = options.minimumRuns ?? 3;
  if (!Number.isSafeInteger(minimumRuns) || minimumRuns < 2) {
    throw new Error("Autonomy portfolio minimumRuns must be an integer of at least two");
  }
  const evaluation = evaluateAutonomyPortfolio(rawRuns);
  if (evaluation.run_count < minimumRuns) {
    throw new Error(
      `Autonomy portfolio requires at least ${String(minimumRuns)} verified runs`
    );
  }
  if ((options.requireSameInitialState ?? true) && !evaluation.same_initial_state) {
    throw new Error(
      "Autonomy diversity must be measured from the same scenario and world seed"
    );
  }
  if (evaluation.distinct_model_decision_signatures < 2) {
    throw new Error("Real model responses showed no decision diversity");
  }
  if (evaluation.distinct_planning_signatures < 2) {
    throw new Error("Real model decisions produced no planning diversity");
  }
  if (evaluation.distinct_physical_behavior_signatures < 2) {
    throw new Error("Real model planning produced no physical behavior diversity");
  }
  if (evaluation.materially_distinct_physical_behaviors < 2) {
    throw new Error(
      "Real model decisions produced no materially different physical behavior"
    );
  }
  return evaluation;
}

function pairwisePhysicalComparisons(
  runs: readonly AutonomyRunEvidence[]
): AutonomyPortfolioEvaluation["physical_comparisons"] {
  const comparisons: AutonomyPortfolioEvaluation["physical_comparisons"] = [];
  for (let left = 0; left < runs.length; left += 1) {
    for (let right = left + 1; right < runs.length; right += 1) {
      comparisons.push({
        left_run_id: runs[left]!.run_id,
        right_run_id: runs[right]!.run_id,
        ...compareRunPhysicalBehavior(runs[left]!, runs[right]!)
      });
    }
  }
  return comparisons;
}

function physicalBehaviorClusterCount(runs: readonly AutonomyRunEvidence[]): number {
  const representatives: AutonomyRunEvidence[] = [];
  for (const run of runs) {
    if (representatives.some((representative) => (
      !compareRunPhysicalBehavior(representative, run).materially_different
    ))) continue;
    representatives.push(run);
  }
  return representatives.length;
}

function compareRunPhysicalBehavior(
  left: AutonomyRunEvidence,
  right: AutonomyRunEvidence
): PhysicalBehaviorDifference {
  const physical = comparePhysicalTrajectories(
    left.physical_trajectories,
    right.physical_trajectories
  );
  const worldMutationDifferent = JSON.stringify(left.world_mutation_hashes)
    !== JSON.stringify(right.world_mutation_hashes);
  if (!worldMutationDifferent) return physical;
  return {
    ...physical,
    materially_different: true,
    reasons: [...physical.reasons, "world_mutation"]
  };
}

function uniqueCount(
  runs: readonly AutonomyRunEvidence[],
  key: "goal_signature"
    | "planning_signature"
    | "model_decision_signature"
    | "physical_behavior_signature"
): number {
  return new Set(runs.map((run) => run[key])).size;
}
