import {
  blockedHumanoidContacts,
  humanoidContactConstraintKey,
  humanoidEnvironmentContacts,
  humanoidEnvironmentContactKey,
  missingRequiredHumanoidContacts,
  type HumanoidContactConstraint,
  type HumanoidMotionValidation
} from "./motion-plan.js";
import { applyHumanoidMotionArtifactFrame } from "./motion-frame-application.js";
import { humanoidMotionArtifactSha256 } from "./motion-artifact.js";
import { humanoidMotionContactEvidenceSha256 } from "./motion-contact-evidence.js";
import { detectHumanoidMotionDrift } from "./motion-rollout.js";
import {
  advanceHumanoidMotionOptionMonitor,
  detectHumanoidMotionOption,
  type HumanoidMotionOptionDetection,
  type HumanoidMotionOptionDetectorInput
} from "./motion-option.js";
import {
  accumulateHumanoidPhysicalSafetyFrame,
  completeHumanoidPhysicalSafetyEvidence,
  createHumanoidPhysicalSafetyAccumulator,
  type HumanoidPhysicalSafetyEvidence
} from "./physical-safety.js";
import type { HumanoidMotionOptionExecutionState } from "./checkpoint.js";
import type { HumanoidReference } from "./reference.js";
import type {
  HumanoidSimulation,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import type { StoredHumanoidMotionPlan } from "./world-plan-state.js";
import { humanoidMotionPlanHasPlanarRootMotion } from "./motion-plan-schema.js";
import type { G1ContactAwareGraspTarget } from "./contact-aware-grasp-servo.js";
import { humanoidCarriedObjectContactConstraints } from "./carried-object-binding.js";
import {
  HumanoidCarryTaskSpaceTargetsSchema,
  type HumanoidCarryTaskSpaceTarget
} from "./carry-task-space-servo.js";
import {
  captureHumanoidStationKeepingAnchor,
  type HumanoidStationKeepingAnchor
} from "./station-keeping.js";

export interface HumanoidMotionExecutionStep {
  snapshot?: HumanoidSimulationSnapshot;
  done: boolean;
}

export interface HumanoidMotionExecutionResult {
  frames: number;
  failures: HumanoidMotionValidation["failures"];
  reference: HumanoidReference;
  physicalSafety?: HumanoidPhysicalSafetyEvidence;
}

export class HumanoidMotionExecution {
  readonly #stored: StoredHumanoidMotionPlan;
  readonly #detectorInput: (
    snapshot: HumanoidSimulationSnapshot
  ) => HumanoidMotionOptionDetectorInput;
  readonly #commitPhysicalFrame: (
    snapshot: HumanoidSimulationSnapshot
  ) => string | undefined;
  readonly #failures: HumanoidMotionValidation["failures"];
  readonly #constraints: NonNullable<
    StoredHumanoidMotionPlan["plan"]["contact_constraints"]
  >;
  readonly #modelConstraints: NonNullable<
    StoredHumanoidMotionPlan["plan"]["contact_constraints"]
  >;
  readonly #satisfiedContacts: Set<string>;
  readonly #frameLimit: number;
  readonly #graspTargets: readonly G1ContactAwareGraspTarget[];
  readonly #carryTaskSpaceTargets: readonly HumanoidCarryTaskSpaceTarget[];
  readonly #stationKeepingRequired: boolean;
  #stationKeepingAnchor: HumanoidStationKeepingAnchor | undefined;
  #stationKeepingCommand: [number, number] = [0, 0];
  #frames = 0;
  #reference: HumanoidReference;
  #lastOptionDetection: HumanoidMotionOptionDetection | null = null;
  #result: HumanoidMotionExecutionResult | undefined;

  constructor(input: {
    stored: StoredHumanoidMotionPlan;
    reference: HumanoidReference;
    detectorInput: (
      snapshot: HumanoidSimulationSnapshot
    ) => HumanoidMotionOptionDetectorInput;
    graspTargets?: readonly G1ContactAwareGraspTarget[];
    carryTaskSpaceTargets?: readonly HumanoidCarryTaskSpaceTarget[];
    stationKeepingAnchor?: HumanoidStationKeepingAnchor;
    commitPhysicalFrame?: (
      snapshot: HumanoidSimulationSnapshot
    ) => string | undefined;
  }) {
    this.#stored = input.stored;
    this.#reference = input.reference;
    this.#detectorInput = input.detectorInput;
    this.#commitPhysicalFrame = input.commitPhysicalFrame ?? (() => undefined);
    this.#graspTargets = input.graspTargets?.map((target) => ({ ...target })) ?? [];
    this.#carryTaskSpaceTargets = HumanoidCarryTaskSpaceTargetsSchema.parse(
      input.carryTaskSpaceTargets ?? input.stored.carriedObjectTaskSpaceTargets ?? []
    );
    this.#stationKeepingRequired = !humanoidMotionPlanHasPlanarRootMotion(
      input.stored.plan
    );
    this.#stationKeepingAnchor = this.#stationKeepingRequired
      && input.stationKeepingAnchor
      ? structuredClone(input.stationKeepingAnchor)
      : undefined;
    if (this.#stationKeepingRequired
      && input.stored.progress.nextFrameIndex > 0) {
      this.#stationKeepingCommand = [...input.reference.rootVelocity];
    }
    this.#failures = input.stored.progress.failure
      ? [validationFailure(input.stored.progress.failure)]
      : [];
    this.#modelConstraints = input.stored.plan.contact_constraints ?? [];
    this.#constraints = mergeContactConstraints(
      this.#modelConstraints,
      input.stored.carriedObjectBindings
        ? humanoidCarriedObjectContactConstraints(
            input.stored.carriedObjectBindings
          )
        : []
    );
    this.#satisfiedContacts = new Set(
      input.stored.progress.satisfiedContactKeys
    );
    input.stored.progress.satisfiedContactKeys = [...this.#satisfiedContacts].sort();
    const restoredContactEvidence = this.#contactEvidenceSha256();
    if (input.stored.progress.satisfiedContactEvidenceSha256 === undefined
      && this.#satisfiedContacts.size > 0) {
      throw new Error(
        "Executed humanoid contact evidence is missing its prefix identity"
      );
    }
    if (input.stored.progress.satisfiedContactEvidenceSha256 !== undefined
      && input.stored.progress.satisfiedContactEvidenceSha256
        !== restoredContactEvidence) {
      throw new Error(
        "Humanoid contact evidence does not match its executed prefix"
      );
    }
    input.stored.progress.satisfiedContactEvidenceSha256 = restoredContactEvidence;
    this.#frameLimit = input.stored.option
      ? input.stored.option.certificate.validated_frame_limit
      : input.stored.artifact.frames.length;
    if (input.stored.option && !terminalOption(input.stored.option.status)) {
      input.stored.option.status = "executing";
    }
  }

  get done(): boolean {
    return this.#result !== undefined;
  }

  get reference(): HumanoidReference {
    return this.#reference;
  }

  async step(simulation: HumanoidSimulation): Promise<HumanoidMotionExecutionStep> {
    if (this.#result) throw new Error("Humanoid motion execution is already complete");
    if (this.#shouldFinishBeforeStep()) {
      this.#finish();
      return { done: true };
    }
    const index = this.#stored.progress.nextFrameIndex;
    const frame = this.#stored.artifact.frames[index]!;
    if (this.#stationKeepingRequired
      && !this.#stationKeepingAnchor
      && typeof simulation.snapshot === "function") {
      this.#stationKeepingAnchor = captureHumanoidStationKeepingAnchor(
        simulation.snapshot(),
        0,
        0
      );
    }
    if (this.#stored.option?.monitor.phase === "awaiting_precondition"
      && !this.#acceptPrecondition(index, simulation.snapshot())) {
      this.#finish();
      return { done: true };
    }

    const applied = await applyHumanoidMotionArtifactFrame(simulation, frame, {
      graspTargets: this.#graspTargets,
      carryTaskSpaceTargets: this.#carryTaskSpaceTargets,
      taskId: this.#stored.option?.contract.option_id
        ?? `motion-plan:${this.#stored.plan.id}`,
      ...(this.#stationKeepingAnchor
        ? {
            stationKeepingAnchor: this.#stationKeepingAnchor,
            stationKeepingCommand: this.#stationKeepingCommand
          }
        : {})
    });
    const { reference, snapshot } = applied;
    this.#frames += 1;
    this.#reference = reference;
    if (this.#stationKeepingAnchor) {
      this.#stationKeepingCommand = [...reference.rootVelocity];
    }
    this.#stored.progress.physicalSafety = accumulateHumanoidPhysicalSafetyFrame(
      this.#stored.progress.physicalSafety ?? createHumanoidPhysicalSafetyAccumulator(),
      index + 1,
      snapshot
    );
    this.#updateRolloutDrift(index, frame.atSeconds, snapshot);
    this.#updateContacts(snapshot);
    this.#detectPhysicalFailure(frame.atSeconds, snapshot);
    this.#stored.progress.nextFrameIndex = index + 1;
    this.#stored.progress.satisfiedContactKeys = [...this.#satisfiedContacts].sort();
    this.#stored.progress.satisfiedContactEvidenceSha256 =
      this.#contactEvidenceSha256();
    const committedFailure = this.#commitPhysicalFrame(snapshot);
    if (committedFailure && this.#failures.length === 0) {
      this.#failures.push({
        code: "motion_constraint_violated",
        atSeconds: frame.atSeconds,
        message: committedFailure
      });
      this.#stored.progress.failure = {
        code: "motion_constraint_violated",
        atSeconds: frame.atSeconds
      };
    }
    if (this.#stored.option) this.#updateOption(index, frame.atSeconds, snapshot);

    if (this.#shouldFinishBeforeStep()) this.#finish();
    return { snapshot, done: this.#result !== undefined };
  }

  result(): HumanoidMotionExecutionResult {
    if (!this.#result) throw new Error("Humanoid motion execution is not complete");
    return structuredClone(this.#result);
  }

  #shouldFinishBeforeStep(): boolean {
    return this.#failures.length > 0
      || this.#stored.progress.nextFrameIndex >= this.#frameLimit
      || this.#stored.option !== null && terminalOption(this.#stored.option.status);
  }

  #contactEvidenceSha256(): string {
    return humanoidMotionContactEvidenceSha256({
      planId: this.#stored.plan.id,
      intentSha256: this.#stored.intentSha256,
      artifactSha256: humanoidMotionArtifactSha256(this.#stored.artifact),
      nextFrameIndex: this.#stored.progress.nextFrameIndex,
      satisfiedContactKeys: this.#stored.progress.satisfiedContactKeys
    });
  }

  #acceptPrecondition(
    index: number,
    snapshot: HumanoidSimulationSnapshot
  ): boolean {
    const option = this.#stored.option!;
    const update = advanceHumanoidMotionOptionMonitor(
      option.contract,
      option.monitor,
      this.#detectorInput(snapshot)
    );
    option.monitor = update.state;
    option.successStreak = update.state.terminalStableSteps;
    this.#lastOptionDetection = update.detection;
    option.lastEvidence = jsonOptionEvidence(update.detection, update.state);
    if (update.observationStatus === "satisfied") return true;
    const atSeconds = index === 0
      ? 0
      : this.#stored.artifact.frames[index - 1]!.atSeconds;
    const uncertain = update.observationStatus === "uncertain";
    this.#failures.push({
      code: uncertain ? "motion_goal_uncertain" : "motion_constraint_violated",
      atSeconds,
      message: uncertain
        ? "Motion option precondition is not observable before execution"
        : "Motion option precondition is not satisfied before execution"
    });
    if (!uncertain) {
      this.#stored.progress.failure = {
        code: "motion_constraint_violated",
        atSeconds
      };
    }
    option.status = uncertain ? "goal_unmet" : "failed";
    option.actualTerminationFrame = index;
    option.terminationReason = uncertain
      ? "motion_goal_uncertain"
      : "motion_constraint_violated";
    return false;
  }

  #updateRolloutDrift(
    index: number,
    atSeconds: number,
    snapshot: HumanoidSimulationSnapshot
  ): void {
    if (!this.#stored.option || !this.#stored.rollout) return;
    const predicted = this.#stored.rollout.frames[index];
    if (!predicted || Math.abs(predicted.atSeconds - atSeconds) > 1e-9) {
      throw new Error("Humanoid motion rollout frame does not match its artifact");
    }
    const drift = detectHumanoidMotionDrift(
      snapshot,
      predicted,
      this.#stored.rollout.limits
    );
    this.#stored.progress.lastDrift = drift;
    this.#stored.progress.driftStreak = drift.drifted
      ? this.#stored.progress.driftStreak + 1
      : 0;
  }

  #updateContacts(snapshot: HumanoidSimulationSnapshot): void {
    const allowed = new Set(this.#modelConstraints.map(humanoidContactConstraintKey));
    for (const contact of humanoidEnvironmentContacts(snapshot)) {
      if (contact.objectId === null && contact.solidId === null) continue;
      const key = humanoidEnvironmentContactKey(contact);
      if (allowed.has(key)) this.#satisfiedContacts.add(key);
    }
  }

  #detectPhysicalFailure(
    atSeconds: number,
    snapshot: HumanoidSimulationSnapshot
  ): void {
    const blockedContacts = blockedHumanoidContacts(snapshot, this.#constraints);
    if (blockedContacts.length > 0) {
      const bodies = [...new Set(blockedContacts.flatMap((contact) => (
        "body" in contact ? [contact.body] : []
      )))].sort();
      const handSurfaces = [...new Set(blockedContacts.flatMap((contact) => (
        "handSurface" in contact ? [contact.handSurface] : []
      )))].sort();
      const contacts = blockedContacts.map((contact) => ({ ...contact }));
      this.#failures.push({
        code: "environment_contact",
        atSeconds,
        ...(bodies.length > 0 ? { bodies } : {}),
        ...(handSurfaces.length > 0 ? { handSurfaces } : {}),
        contacts
      });
      this.#stored.progress.failure = {
        code: "environment_contact",
        atSeconds,
        ...(bodies.length > 0 ? { bodies: [...bodies] } : {}),
        ...(handSurfaces.length > 0
          ? { handSurfaces: [...handSurfaces] }
          : {}),
        contacts: contacts.map((contact) => ({ ...contact }))
      };
      return;
    }
    if (snapshot.fallen) {
      this.#failures.push({ code: "fallen", atSeconds });
      this.#stored.progress.failure = { code: "fallen", atSeconds };
      return;
    }
    if (this.#stored.option
      && this.#stored.rollout
      && this.#stored.progress.lastDrift?.drifted
      && this.#stored.progress.driftStreak >= this.#stored.rollout.limits.consecutive_steps) {
      const failure = {
        code: "execution_drift" as const,
        atSeconds,
        drift: { ...this.#stored.progress.lastDrift },
        message: "Physical execution persistently diverged from its validated rollout"
      };
      this.#failures.push(failure);
      this.#stored.progress.failure = {
        code: failure.code,
        atSeconds: failure.atSeconds,
        drift: { ...failure.drift }
      };
    }
  }

  #updateOption(
    index: number,
    atSeconds: number,
    snapshot: HumanoidSimulationSnapshot
  ): void {
    const option = this.#stored.option!;
    const detectorInput = this.#detectorInput(snapshot);
    if (this.#failures.length > 0) {
      this.#lastOptionDetection = detectHumanoidMotionOption(
        option.contract,
        detectorInput
      );
      option.lastEvidence = jsonOptionEvidence(
        this.#lastOptionDetection,
        option.monitor
      );
      option.status = "failed";
      option.actualTerminationFrame = index + 1;
      option.terminationReason = this.#stored.progress.failure?.code
        ?? "environment_contact";
      return;
    }
    const update = advanceHumanoidMotionOptionMonitor(
      option.contract,
      option.monitor,
      detectorInput
    );
    option.monitor = update.state;
    option.successStreak = update.state.terminalStableSteps;
    this.#lastOptionDetection = update.detection;
    option.lastEvidence = jsonOptionEvidence(update.detection, update.state);
    if (update.state.phase === "indeterminate") {
      this.#failures.push({
        code: "motion_goal_uncertain",
        atSeconds,
        message: "Motion option lost observable evidence for its during constraint"
      });
      option.status = "goal_unmet";
      option.actualTerminationFrame = index + 1;
      option.terminationReason = "motion_goal_uncertain";
      return;
    }
    if (update.state.phase === "violated") {
      this.#failures.push({
        code: "motion_constraint_violated",
        atSeconds,
        message: "Motion option violated its during constraint"
      });
      this.#stored.progress.failure = {
        code: "motion_constraint_violated",
        atSeconds
      };
      option.status = "failed";
      option.actualTerminationFrame = index + 1;
      option.terminationReason = "motion_constraint_violated";
      return;
    }
    if (update.state.phase === "succeeded"
      && missingRequiredHumanoidContacts(
        this.#modelConstraints,
        this.#satisfiedContacts
      ).length === 0
      && snapshot.balance.support !== "none") {
      option.status = "succeeded";
      option.actualTerminationFrame = index + 1;
      option.terminationReason = "physical_success";
    }
  }

  #finish(): void {
    const option = this.#stored.option;
    if (option) {
      if (option.status === "succeeded"
        && missingRequiredHumanoidContacts(
          this.#modelConstraints,
          this.#satisfiedContacts
        ).length > 0) {
        option.status = "goal_unmet";
        option.terminationReason = "motion_goal_unmet";
      }
      if (!terminalOption(option.status)) {
        const uncertain = this.#lastOptionDetection?.hasUncertain ?? true;
        option.status = "goal_unmet";
        option.actualTerminationFrame = this.#stored.progress.nextFrameIndex;
        option.terminationReason = uncertain
          ? "motion_goal_uncertain"
          : "motion_goal_unmet";
        this.#failures.push({
          code: uncertain ? "motion_goal_uncertain" : "motion_goal_unmet",
          atSeconds: this.#stored.artifact.frames[
            Math.max(0, this.#stored.progress.nextFrameIndex - 1)
          ]!.atSeconds,
          message: uncertain
            ? "Motion option ended without observable success evidence"
            : "Motion option exhausted its verified horizon before physical success"
        });
      }
    } else if (this.#failures.length === 0) {
      const missing = missingRequiredHumanoidContacts(
        this.#modelConstraints,
        this.#satisfiedContacts
      );
      if (missing.length > 0) {
        this.#failures.push({
          code: "required_contact_missing",
          atSeconds: this.#stored.plan.duration_seconds,
          constraints: missing
        });
      }
    }
    const physicalSafety = this.#stored.progress.physicalSafety
      ? completeHumanoidPhysicalSafetyEvidence(this.#stored.progress.physicalSafety)
      : undefined;
    this.#result = {
      frames: this.#frames,
      failures: this.#failures,
      reference: this.#reference,
      ...(physicalSafety ? { physicalSafety } : {})
    };
  }
}

function terminalOption(status: HumanoidMotionOptionExecutionState["status"]): boolean {
  return status === "succeeded" || status === "failed" || status === "goal_unmet";
}

function mergeContactConstraints(
  primary: readonly HumanoidContactConstraint[],
  additional: readonly HumanoidContactConstraint[]
): HumanoidContactConstraint[] {
  const merged = new Map<string, HumanoidContactConstraint>();
  for (const constraint of [...additional, ...primary]) {
    merged.set(humanoidContactConstraintKey(constraint), { ...constraint });
  }
  return [...merged.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, constraint]) => constraint);
}

function validationFailure(
  failure: NonNullable<StoredHumanoidMotionPlan["progress"]["failure"]>
): HumanoidMotionValidation["failures"][number] {
  if (failure.code === "fallen") return { code: "fallen", atSeconds: failure.atSeconds };
  if (failure.code === "execution_drift") {
    return {
      code: "execution_drift",
      atSeconds: failure.atSeconds,
      ...(failure.drift ? { drift: { ...failure.drift } } : {}),
      message: "Physical execution persistently diverged from its validated rollout"
    };
  }
  if (failure.code === "motion_constraint_violated") {
    return {
      code: "motion_constraint_violated",
      atSeconds: failure.atSeconds,
      message: "Motion option violated its during constraint"
    };
  }
  return {
    code: "environment_contact",
    atSeconds: failure.atSeconds,
    ...(failure.bodies ? { bodies: [...failure.bodies] } : {}),
    ...(failure.handSurfaces
      ? { handSurfaces: [...failure.handSurfaces] }
      : {}),
    ...(failure.contacts
      ? { contacts: failure.contacts.map((contact) => ({ ...contact })) }
      : {})
  };
}

function jsonOptionEvidence(
  detection: HumanoidMotionOptionDetection,
  monitor: NonNullable<StoredHumanoidMotionPlan["option"]>["monitor"]
): NonNullable<StoredHumanoidMotionPlan["option"]>["lastEvidence"] {
  return JSON.parse(JSON.stringify({
    predicates: detection.evidence,
    phases: detection.phases,
    monitor
  })) as NonNullable<StoredHumanoidMotionPlan["option"]>["lastEvidence"];
}
