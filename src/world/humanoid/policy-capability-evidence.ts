import { z } from "zod";
import type { JsonValue } from "../../domain/schema.js";
import type { HumanoidEmbodiedSkillCall } from "./embodied-skill-call.js";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import type { HumanoidPolicyState } from "./whole-body-controller.js";

const EVIDENCE_PROTOCOL = "humanoid-policy-capability-evidence-v1";
const MINIMUM_OUTCOMES_FOR_HARD_ADMISSION = 8;
const MINIMUM_SUCCESSES_FOR_DISTRIBUTION = 5;
const MINIMUM_SUCCESS_LOWER_BOUND = 0.45;
const MAXIMUM_OOD_SCORE = 4;
const MAXIMUM_RECENT_OUTCOMES = 32;
const WILSON_Z_90 = 1.644_853_626_951_472_2;

const STATE_VARIANCE_FLOORS = [
  0.08, 0.08, 0.08,
  0.12, 0.12, 0.12,
  0.04,
  0.08,
  0.20,
  0.35,
  1
] as const;

const COMMAND_VARIANCE_FLOORS = [
  0.06, 0.06, 0.08,
  0.025,
  0.5, 0.04, 0.04, 0.04,
  0.5, 0.04, 0.04, 0.04,
  1, 1
] as const;

export type HumanoidPolicySkillFamily =
  | `semantic:${string}`
  | "navigation"
  | "station_keeping"
  | "legacy_motion";

export type HumanoidPolicyOutcome = "succeeded" | "failed" | "interrupted";

export interface HumanoidPolicyCapabilityObservation {
  state: number[];
  command: number[];
  jointPositions: number[];
}

export interface HumanoidPolicyCapabilityPosterior {
  outcomes: number;
  successes: number;
  failures: number;
  posteriorMean: number;
  lowerBound: number;
  upperBound: number;
  recentSuccessRate: number | null;
  transitionAttempts: number;
  transitionSuccesses: number;
}

type HumanoidPolicyAdmissionReason =
  | "cold_start"
  | "capability_supported"
  | "insufficient_success_posterior"
  | "entry_state_ood"
  | "command_ood"
  | "memory_bridge_completed"
  | "memory_bridge_timeout";

export interface HumanoidPolicyAdmissionAssessment {
  protocol: "humanoid-policy-admission-v1";
  implementation: string;
  skillFamily: HumanoidPolicySkillFamily;
  admitted: boolean;
  reason: HumanoidPolicyAdmissionReason;
  coldStart: boolean;
  entryStateOodScore: number | null;
  commandOodScore: number | null;
  posterior: HumanoidPolicyCapabilityPosterior;
  successfulEntryPrototype: number[] | null;
}

export const HumanoidPolicyCapabilityObservationSchema = z.object({
  state: z.array(z.number().finite()).length(STATE_VARIANCE_FLOORS.length),
  command: z.array(z.number().finite()).length(COMMAND_VARIANCE_FLOORS.length),
  jointPositions: z.array(z.number().finite()).length(HUMANOID_JOINT_NAMES.length)
}).strict();

const HumanoidPolicyCapabilityPosteriorSchema = z.object({
  outcomes: z.number().int().nonnegative(),
  successes: z.number().int().nonnegative(),
  failures: z.number().int().nonnegative(),
  posteriorMean: z.number().finite().min(0).max(1),
  lowerBound: z.number().finite().min(0).max(1),
  upperBound: z.number().finite().min(0).max(1),
  recentSuccessRate: z.number().finite().min(0).max(1).nullable(),
  transitionAttempts: z.number().int().nonnegative(),
  transitionSuccesses: z.number().int().nonnegative()
}).strict();

export const HumanoidPolicyAdmissionAssessmentSchema = z.object({
  protocol: z.literal("humanoid-policy-admission-v1"),
  implementation: z.string().trim().min(1),
  skillFamily: z.custom<HumanoidPolicySkillFamily>((value) => (
    typeof value === "string" && (
      value === "navigation"
      || value === "station_keeping"
      || value === "legacy_motion"
      || value.startsWith("semantic:") && value.length > "semantic:".length
    )
  )),
  admitted: z.boolean(),
  reason: z.enum([
    "cold_start",
    "capability_supported",
    "insufficient_success_posterior",
    "entry_state_ood",
    "command_ood",
    "memory_bridge_completed",
    "memory_bridge_timeout"
  ]),
  coldStart: z.boolean(),
  entryStateOodScore: z.number().finite().nonnegative().nullable(),
  commandOodScore: z.number().finite().nonnegative().nullable(),
  posterior: HumanoidPolicyCapabilityPosteriorSchema,
  successfulEntryPrototype: z.array(z.number().finite())
    .length(HUMANOID_JOINT_NAMES.length).nullable()
}).strict();

interface RunningVectorStatistics {
  count: number;
  mean: number[];
  m2: number[];
}

interface CapabilityEvidenceEntry {
  implementation: string;
  skillFamily: HumanoidPolicySkillFamily;
  outcomes: number;
  successes: number;
  recent: HumanoidPolicyOutcome[];
  successfulEntryStates: RunningVectorStatistics;
  successfulCommands: RunningVectorStatistics;
  successfulJointEntries: RunningVectorStatistics;
  transitionAttempts: number;
  transitionSuccesses: number;
}

const RunningVectorStatisticsSchema = z.object({
  count: z.number().int().nonnegative(),
  mean: z.array(z.number().finite()),
  m2: z.array(z.number().finite().nonnegative())
}).strict().superRefine((statistics, context) => {
  if (statistics.mean.length !== statistics.m2.length) {
    context.addIssue({
      code: "custom",
      message: "Running-vector mean and second moment must have equal dimensions"
    });
  }
  if (statistics.count === 0
    && (statistics.mean.length > 0 || statistics.m2.length > 0)) {
    context.addIssue({
      code: "custom",
      message: "Empty running-vector statistics cannot contain dimensions"
    });
  }
});

const CapabilityEvidenceEntrySchema = z.object({
  implementation: z.string().trim().min(1),
  skill_family: z.string().trim().min(1),
  outcomes: z.number().int().nonnegative(),
  successes: z.number().int().nonnegative(),
  recent: z.array(z.enum(["succeeded", "failed", "interrupted"]))
    .max(MAXIMUM_RECENT_OUTCOMES),
  successful_entry_states: RunningVectorStatisticsSchema,
  successful_commands: RunningVectorStatisticsSchema,
  successful_joint_entries: RunningVectorStatisticsSchema,
  transition_attempts: z.number().int().nonnegative(),
  transition_successes: z.number().int().nonnegative()
}).strict().superRefine((entry, context) => {
  if (entry.successes > entry.outcomes) {
    context.addIssue({ code: "custom", message: "Successes cannot exceed outcomes" });
  }
  if (entry.transition_successes > entry.transition_attempts) {
    context.addIssue({
      code: "custom",
      message: "Transition successes cannot exceed transition attempts"
    });
  }
  for (const statistics of [
    entry.successful_entry_states,
    entry.successful_commands,
    entry.successful_joint_entries
  ]) {
    if (statistics.count !== entry.successes) {
      context.addIssue({
        code: "custom",
        message: "Successful feature statistics must match the success count"
      });
    }
  }
  for (const [statistics, dimensions, label] of [
    [entry.successful_entry_states, STATE_VARIANCE_FLOORS.length, "entry state"],
    [entry.successful_commands, COMMAND_VARIANCE_FLOORS.length, "command"],
    [entry.successful_joint_entries, HUMANOID_JOINT_NAMES.length, "joint entry"]
  ] as const) {
    if (statistics.count > 0 && statistics.mean.length !== dimensions) {
      context.addIssue({
        code: "custom",
        message: `Successful ${label} statistics have invalid dimensions`
      });
    }
  }
});

const CapabilityEvidenceStateSchema = z.object({
  protocol: z.literal(EVIDENCE_PROTOCOL),
  entries: z.array(CapabilityEvidenceEntrySchema)
}).strict();

export class HumanoidPolicyCapabilityEvidenceRegistry {
  readonly #entries = new Map<string, CapabilityEvidenceEntry>();

  assess(input: {
    implementation: string;
    state: HumanoidPolicyState;
    taskCommand: HumanoidEmbodiedSkillCall;
  }): HumanoidPolicyAdmissionAssessment {
    const skillFamily = humanoidPolicySkillFamily(input.taskCommand);
    const observation = humanoidPolicyCapabilityObservation(
      input.state,
      input.taskCommand
    );
    const entry = this.#entry(input.implementation, skillFamily, false);
    const posterior = capabilityPosterior(entry);
    const coldStart = posterior.outcomes < MINIMUM_OUTCOMES_FOR_HARD_ADMISSION;
    const entryStateOodScore = entry
      && entry.successfulEntryStates.count >= MINIMUM_SUCCESSES_FOR_DISTRIBUTION
      ? diagonalOodScore(
          observation.state,
          entry.successfulEntryStates,
          STATE_VARIANCE_FLOORS
        )
      : null;
    const commandOodScore = entry
      && entry.successfulCommands.count >= MINIMUM_SUCCESSES_FOR_DISTRIBUTION
      ? diagonalOodScore(
          observation.command,
          entry.successfulCommands,
          COMMAND_VARIANCE_FLOORS
        )
      : null;

    let admitted = true;
    let reason: HumanoidPolicyAdmissionReason = coldStart
      ? "cold_start"
      : "capability_supported";
    if (!coldStart && posterior.lowerBound < MINIMUM_SUCCESS_LOWER_BOUND) {
      admitted = false;
      reason = "insufficient_success_posterior";
    } else if (!coldStart && entryStateOodScore !== null
      && entryStateOodScore > MAXIMUM_OOD_SCORE) {
      admitted = false;
      reason = "entry_state_ood";
    } else if (!coldStart && commandOodScore !== null
      && commandOodScore > MAXIMUM_OOD_SCORE) {
      admitted = false;
      reason = "command_ood";
    }

    return {
      protocol: "humanoid-policy-admission-v1",
      implementation: input.implementation,
      skillFamily,
      admitted,
      reason,
      coldStart,
      entryStateOodScore,
      commandOodScore,
      posterior,
      successfulEntryPrototype: entry?.successfulJointEntries.mean.length
        ? [...entry.successfulJointEntries.mean]
        : null
    };
  }

  record(input: {
    implementation: string;
    skillFamily: HumanoidPolicySkillFamily;
    observation: HumanoidPolicyCapabilityObservation;
    outcome: HumanoidPolicyOutcome;
    transitionAttempted: boolean;
  }): HumanoidPolicyCapabilityPosterior {
    assertObservation(input.observation);
    const entry = this.#entry(input.implementation, input.skillFamily, true)!;
    entry.outcomes += 1;
    entry.recent.push(input.outcome);
    if (entry.recent.length > MAXIMUM_RECENT_OUTCOMES) entry.recent.shift();
    if (input.transitionAttempted) {
      entry.transitionAttempts += 1;
      if (input.outcome === "succeeded") entry.transitionSuccesses += 1;
    }
    if (input.outcome === "succeeded") {
      entry.successes += 1;
      updateRunningStatistics(entry.successfulEntryStates, input.observation.state);
      updateRunningStatistics(entry.successfulCommands, input.observation.command);
      updateRunningStatistics(
        entry.successfulJointEntries,
        input.observation.jointPositions
      );
    }
    return capabilityPosterior(entry);
  }

  summaries(): ReadonlyArray<{
    implementation: string;
    skillFamily: HumanoidPolicySkillFamily;
    posterior: HumanoidPolicyCapabilityPosterior;
  }> {
    return [...this.#entries.values()]
      .sort((left, right) => evidenceKey(
        left.implementation,
        left.skillFamily
      ).localeCompare(evidenceKey(right.implementation, right.skillFamily)))
      .map((entry) => ({
        implementation: entry.implementation,
        skillFamily: entry.skillFamily,
        posterior: capabilityPosterior(entry)
      }));
  }

  captureState(): JsonValue {
    return {
      protocol: EVIDENCE_PROTOCOL,
      entries: [...this.#entries.values()]
        .sort((left, right) => evidenceKey(
          left.implementation,
          left.skillFamily
        ).localeCompare(evidenceKey(right.implementation, right.skillFamily)))
        .map((entry) => ({
          implementation: entry.implementation,
          skill_family: entry.skillFamily,
          outcomes: entry.outcomes,
          successes: entry.successes,
          recent: [...entry.recent],
          successful_entry_states: statisticsJson(entry.successfulEntryStates),
          successful_commands: statisticsJson(entry.successfulCommands),
          successful_joint_entries: statisticsJson(entry.successfulJointEntries),
          transition_attempts: entry.transitionAttempts,
          transition_successes: entry.transitionSuccesses
        }))
    };
  }

  restoreState(value: JsonValue): void {
    const restored = CapabilityEvidenceStateSchema.parse(value);
    const entries = new Map<string, CapabilityEvidenceEntry>();
    for (const item of restored.entries) {
      const skillFamily = parseSkillFamily(item.skill_family);
      const key = evidenceKey(item.implementation, skillFamily);
      if (entries.has(key)) {
        throw new Error("Duplicate humanoid policy capability evidence entry");
      }
      entries.set(key, {
        implementation: item.implementation,
        skillFamily,
        outcomes: item.outcomes,
        successes: item.successes,
        recent: [...item.recent],
        successfulEntryStates: cloneStatistics(item.successful_entry_states),
        successfulCommands: cloneStatistics(item.successful_commands),
        successfulJointEntries: cloneStatistics(item.successful_joint_entries),
        transitionAttempts: item.transition_attempts,
        transitionSuccesses: item.transition_successes
      });
    }
    this.#entries.clear();
    for (const [key, entry] of entries) this.#entries.set(key, entry);
  }

  #entry(
    implementation: string,
    skillFamily: HumanoidPolicySkillFamily,
    create: boolean
  ): CapabilityEvidenceEntry | undefined {
    const key = evidenceKey(implementation, skillFamily);
    let entry = this.#entries.get(key);
    if (!entry && create) {
      entry = {
        implementation,
        skillFamily,
        outcomes: 0,
        successes: 0,
        recent: [],
        successfulEntryStates: emptyStatistics(),
        successfulCommands: emptyStatistics(),
        successfulJointEntries: emptyStatistics(),
        transitionAttempts: 0,
        transitionSuccesses: 0
      };
      this.#entries.set(key, entry);
    }
    return entry;
  }
}

function humanoidPolicySkillFamily(
  taskCommand: HumanoidEmbodiedSkillCall
): HumanoidPolicySkillFamily {
  const identity = taskCommand.identity;
  if (identity.runtimeKind === "semantic_skill") {
    if (!identity.skillId) {
      throw new Error("Semantic humanoid Skill call is missing its Skill identity");
    }
    return `semantic:${identity.skillId}`;
  }
  return identity.runtimeKind;
}

export function humanoidPolicyCapabilityObservation(
  state: HumanoidPolicyState,
  taskCommand: HumanoidEmbodiedSkillCall
): HumanoidPolicyCapabilityObservation {
  const jointVelocities = finiteArray(state.jointVelocities, "joint velocities");
  const jointPositions = finiteArray(state.jointPositions, "joint positions");
  if (jointPositions.length !== HUMANOID_JOINT_NAMES.length
    || jointVelocities.length !== HUMANOID_JOINT_NAMES.length) {
    throw new Error("Humanoid policy evidence requires all body joints");
  }
  const linear = state.environment?.rootLinearVelocity ?? [0, 0, 0];
  const angular = state.rootAngularVelocity;
  const upright = clamp(
    1 - 2 * (state.rootQuaternion[1] ** 2 + state.rootQuaternion[2] ** 2),
    -1,
    1
  );
  const jointVelocityRms = Math.sqrt(
    jointVelocities.reduce((sum, value) => sum + value * value, 0)
      / Math.max(1, jointVelocities.length)
  );
  const command = taskCommand.command;
  return {
    state: [
      ...linear,
      ...angular,
      Math.acos(upright),
      jointVelocityRms,
      Math.max(0, ...jointVelocities.map(Math.abs)),
      Math.log1p(state.environment?.contacts.length ?? 0),
      state.environment ? 1 : 0
    ],
    command: [
      command.baseTwist.forwardMetersPerSecond,
      command.baseTwist.lateralMetersPerSecond,
      command.baseTwist.yawRadiansPerSecond,
      command.rootHeightMeters,
      ...(command.leftWristPositionPelvis
        ? [1, ...vec3(command.leftWristPositionPelvis)]
        : [0, 0, 0, 0]),
      ...(command.rightWristPositionPelvis
        ? [1, ...vec3(command.rightWristPositionPelvis)]
        : [0, 0, 0, 0]),
      command.endEffectors.length,
      command.grasps.length
    ],
    jointPositions
  };
}

function capabilityPosterior(
  entry: CapabilityEvidenceEntry | undefined
): HumanoidPolicyCapabilityPosterior {
  const outcomes = entry?.outcomes ?? 0;
  const successes = entry?.successes ?? 0;
  const failures = outcomes - successes;
  const interval = wilsonInterval(successes + 1, outcomes + 2);
  const recent = entry?.recent ?? [];
  return {
    outcomes,
    successes,
    failures,
    posteriorMean: (successes + 1) / (outcomes + 2),
    lowerBound: interval.lower,
    upperBound: interval.upper,
    recentSuccessRate: recent.length > 0
      ? recent.filter((outcome) => outcome === "succeeded").length / recent.length
      : null,
    transitionAttempts: entry?.transitionAttempts ?? 0,
    transitionSuccesses: entry?.transitionSuccesses ?? 0
  };
}

function wilsonInterval(successes: number, outcomes: number): {
  lower: number;
  upper: number;
} {
  const probability = successes / outcomes;
  const zSquared = WILSON_Z_90 ** 2;
  const denominator = 1 + zSquared / outcomes;
  const center = probability + zSquared / (2 * outcomes);
  const spread = WILSON_Z_90 * Math.sqrt(
    probability * (1 - probability) / outcomes
      + zSquared / (4 * outcomes ** 2)
  );
  return {
    lower: clamp((center - spread) / denominator, 0, 1),
    upper: clamp((center + spread) / denominator, 0, 1)
  };
}

function diagonalOodScore(
  values: readonly number[],
  statistics: RunningVectorStatistics,
  varianceFloors: readonly number[]
): number {
  if (values.length !== statistics.mean.length
    || values.length !== varianceFloors.length) {
    throw new Error("Humanoid policy OOD evidence dimensions do not match");
  }
  const squaredScore = values.reduce((sum, value, index) => {
    const sampleVariance = statistics.count > 1
      ? statistics.m2[index]! / (statistics.count - 1)
      : 0;
    const standardDeviation = Math.max(
      Math.sqrt(sampleVariance),
      varianceFloors[index]!
    );
    const z = (value - statistics.mean[index]!) / standardDeviation;
    return sum + z * z;
  }, 0);
  return Math.sqrt(squaredScore / Math.max(1, values.length));
}

function updateRunningStatistics(
  statistics: RunningVectorStatistics,
  values: readonly number[]
): void {
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error("Humanoid policy evidence features must be finite");
  }
  if (statistics.count === 0) {
    statistics.count = 1;
    statistics.mean = [...values];
    statistics.m2 = values.map(() => 0);
    return;
  }
  if (statistics.mean.length !== values.length) {
    throw new Error("Humanoid policy evidence feature dimensions changed");
  }
  statistics.count += 1;
  for (let index = 0; index < values.length; index += 1) {
    const delta = values[index]! - statistics.mean[index]!;
    const mean = statistics.mean[index]! + delta / statistics.count;
    statistics.mean[index] = mean;
    const nextDelta = values[index]! - mean;
    statistics.m2[index] = statistics.m2[index]! + delta * nextDelta;
  }
}

function assertObservation(observation: HumanoidPolicyCapabilityObservation): void {
  const dimensions = [
    [observation.state, STATE_VARIANCE_FLOORS.length, "state"],
    [observation.command, COMMAND_VARIANCE_FLOORS.length, "command"],
    [observation.jointPositions, HUMANOID_JOINT_NAMES.length, "joint entry"]
  ] as const;
  for (const [values, expected, label] of dimensions) {
    if (values.length !== expected || values.some((value) => !Number.isFinite(value))) {
      throw new Error(`Invalid humanoid policy ${label} evidence`);
    }
  }
}

function evidenceKey(
  implementation: string,
  skillFamily: HumanoidPolicySkillFamily
): string {
  return `${implementation}\u0000${skillFamily}`;
}

function parseSkillFamily(value: string): HumanoidPolicySkillFamily {
  if (value === "navigation" || value === "station_keeping"
    || value === "legacy_motion"
    || value.startsWith("semantic:") && value.length > "semantic:".length) {
    return value as HumanoidPolicySkillFamily;
  }
  throw new Error(`Invalid humanoid policy Skill family: ${value}`);
}

function emptyStatistics(): RunningVectorStatistics {
  return { count: 0, mean: [], m2: [] };
}

function statisticsJson(statistics: RunningVectorStatistics): JsonValue {
  return {
    count: statistics.count,
    mean: [...statistics.mean],
    m2: [...statistics.m2]
  };
}

function cloneStatistics(statistics: RunningVectorStatistics): RunningVectorStatistics {
  return {
    count: statistics.count,
    mean: [...statistics.mean],
    m2: [...statistics.m2]
  };
}

function finiteArray(values: ArrayLike<number>, label: string): number[] {
  const result = Array.from(values);
  if (result.some((value) => !Number.isFinite(value))) {
    throw new Error(`Humanoid policy ${label} must be finite`);
  }
  return result;
}

function vec3(value: { x: number; y: number; z: number }): [number, number, number] {
  return [value.x, value.y, value.z];
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
