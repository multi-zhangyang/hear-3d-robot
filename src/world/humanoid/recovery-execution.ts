import {
  accumulateHumanoidPhysicalSafetyFrame,
  completeHumanoidPhysicalSafetyEvidence,
  createHumanoidPhysicalSafetyAccumulator,
  type HumanoidPhysicalSafetyEvidence
} from "./physical-safety.js";
import {
  controllerTaskCommand
} from "./motion-frame-application.js";
import {
  humanoidContactConstraintKey,
  humanoidEnvironmentContactKey,
  humanoidEnvironmentContacts
} from "./motion-plan.js";
import type { HumanoidEnvironmentContact } from "./motion-contact-policy.js";
import type { HumanoidReference } from "./reference.js";
import { humanoidMotionArtifactSha256 } from "./motion-artifact.js";
import { humanoidMotionContactEvidenceSha256 } from "./motion-contact-evidence.js";
import type {
  HumanoidPolicyFrameSink,
  HumanoidSimulation,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import type { StoredHumanoidMotionPlan } from "./world-plan-state.js";
import type { HumanoidSkillProgressEvidence } from "./skill-event-stream.js";

export type HumanoidRecoveryFailureCode =
  | "recovery_timeout"
  | "unauthorized_scene_contact"
  | "contact_force_limit"
  | "contact_impact_limit"
  | "joint_velocity_limit"
  | "joint_limit_violation"
  | "numerical_instability"
  | "recovery_controller_unavailable";

export interface HumanoidRecoveryFailure {
  code: HumanoidRecoveryFailureCode;
  atSeconds: number;
  detail: string;
  contacts?: HumanoidEnvironmentContact[];
}

export interface HumanoidRecoveryExecutionResult {
  frames: number;
  completed: boolean;
  reference: HumanoidReference;
  failure: HumanoidRecoveryFailure | null;
  physicalSafety?: HumanoidPhysicalSafetyEvidence;
  recoveryImplementation: string | null;
  handoffSteps: number;
}

export class HumanoidRecoveryExecution {
  readonly #stored: StoredHumanoidMotionPlan;
  readonly #policyFrameSink: HumanoidPolicyFrameSink | undefined;
  readonly #skillWindowMaximumSteps: number;
  readonly #skillWindowStepOffset: number;
  readonly #initialControllerImplementation: string | null;
  #reference: HumanoidReference;
  #frames = 0;
  #stableSteps = 0;
  #handoffSteps = 0;
  #handoffStarted = false;
  #handoffCompleted = false;
  #recoveryImplementation: string | null = null;
  #previousTotalContactForceN: number | null = null;
  #failure: HumanoidRecoveryFailure | null = null;
  #result: HumanoidRecoveryExecutionResult | null = null;

  constructor(input: {
    stored: StoredHumanoidMotionPlan;
    reference: HumanoidReference;
    initialSnapshot: HumanoidSimulationSnapshot;
    policyFrameSink?: HumanoidPolicyFrameSink;
    skillWindow?: { maximumSteps: number; stepOffset: number };
  }) {
    if (!input.stored.recoveryContract) {
      throw new Error("Humanoid recovery execution requires a recovery contract");
    }
    this.#stored = input.stored;
    this.#reference = input.reference;
    this.#policyFrameSink = input.policyFrameSink;
    this.#skillWindowStepOffset = input.skillWindow?.stepOffset ?? 0;
    this.#skillWindowMaximumSteps = input.skillWindow?.maximumSteps
      ?? input.stored.recoveryContract.maximumSteps;
    if (this.#skillWindowMaximumSteps
      < this.#skillWindowStepOffset + input.stored.recoveryContract.maximumSteps) {
      throw new Error("Humanoid recovery Skill window cannot contain this contract");
    }
    this.#initialControllerImplementation =
      input.initialSnapshot.controllerExecution?.activeImplementation ?? null;
    const persisted = input.stored.progress.recovery;
    if (persisted) {
      this.#stableSteps = persisted.stableSteps;
      this.#handoffSteps = persisted.handoffSteps;
      this.#recoveryImplementation = persisted.recoveryImplementation;
      this.#handoffStarted = persisted.handoffStarted;
      this.#handoffCompleted = persisted.handoffCompleted;
      this.#previousTotalContactForceN =
        persisted.previousTotalContactForceN;
    }
    const transition = input.initialSnapshot.controllerExecution?.transition;
    if (transition) {
      this.#recoveryImplementation ??= transition.fromImplementation;
    } else if (!input.initialSnapshot.fallen
      && input.initialSnapshot.controllerExecution) {
      this.#recoveryImplementation ??=
        input.initialSnapshot.controllerExecution.activeImplementation;
    }
    this.#syncProgress();
  }

  get done(): boolean {
    return this.#result !== null;
  }

  get reference(): HumanoidReference {
    return this.#reference;
  }

  skillProgressEvidence(): HumanoidSkillProgressEvidence {
    const contract = this.#stored.recoveryContract!;
    return {
      elapsedRatio: boundedRatio(
        this.#stored.progress.nextFrameIndex,
        contract.maximumSteps
      ),
      physicalCompletionRatio: boundedRatio(
        Math.min(this.#stableSteps, contract.stableSteps)
          + Math.min(this.#handoffSteps, contract.handoffSteps),
        contract.stableSteps + contract.handoffSteps
      ),
      satisfiedPredicateRatio: this.#completedHandoff() ? 1 : 0,
      stableSteps: this.#stableSteps,
      requiredStableSteps: contract.stableSteps,
      confidence: this.#recoveryImplementation === null ? 0 : 1
    };
  }

  async step(
    simulation: HumanoidSimulation,
    authority: { worldFrame: number; worldRevision: number },
    commitPhysicalFrame: (snapshot: HumanoidSimulationSnapshot) => void
  ): Promise<{ snapshot?: HumanoidSimulationSnapshot; done: boolean }> {
    if (this.#result) throw new Error("Humanoid recovery execution is already complete");
    const contract = this.#stored.recoveryContract!;
    const index = this.#stored.progress.nextFrameIndex;
    if (index >= contract.maximumSteps) {
      this.#failure = {
        code: "recovery_timeout",
        atSeconds: index * simulation.controllerDescriptor().controlStepSeconds,
        detail: "Recovery expert exhausted its bounded control horizon"
      };
      this.#finish();
      return { done: true };
    }
    const taskCommand = controllerTaskCommand({
      taskId: `recovery:${this.#stored.plan.id}`,
      taskGoal: contract,
      ...(this.#stored.skillCallIdentity
        ? { skillIdentity: this.#stored.skillCallIdentity }
        : {}),
      authority,
      controlWindow: {
        maximumSteps: this.#skillWindowMaximumSteps,
        stepIndex: this.#skillWindowStepOffset + index
      },
      authorizedContacts: contract.authorizedContacts,
      recoverySafety: true,
      controlStepSeconds: simulation.controllerDescriptor().controlStepSeconds,
      reference: this.#reference,
      taskSpaceTargets: [],
      carryTaskSpaceTargets: [],
      graspTargets: []
    });
    const snapshot = await simulation.step(this.#reference, {
      trackedJointPolicyCommand: "measured",
      taskCommand,
      ...(this.#policyFrameSink
        ? { policyFrameSink: this.#policyFrameSink }
        : {})
    });
    this.#frames += 1;
    this.#stored.progress.nextFrameIndex = index + 1;
    this.#stored.progress.satisfiedContactEvidenceSha256 =
      humanoidMotionContactEvidenceSha256({
        planId: this.#stored.plan.id,
        intentSha256: this.#stored.intentSha256,
        artifactSha256: humanoidMotionArtifactSha256(this.#stored.artifact),
        nextFrameIndex: this.#stored.progress.nextFrameIndex,
        satisfiedContactKeys: this.#stored.progress.satisfiedContactKeys
      });
    this.#observeController(snapshot);
    this.#captureSafety(index + 1, snapshot);
    const totalContactForceN = snapshot.contacts.reduce(
      (sum, { normalForce }) => sum + normalForce,
      0
    );
    this.#failure ??= humanoidRecoverySafetyFailure(
      snapshot,
      contract,
      index + 1,
      this.#previousTotalContactForceN,
      simulation.controllerDescriptor().controlStepSeconds
    );
    this.#previousTotalContactForceN = totalContactForceN;
    this.#updateStableStanding(snapshot);
    this.#syncProgress();
    commitPhysicalFrame(snapshot);
    if (this.#failure
      || this.#completedHandoff()
      || this.#stored.progress.nextFrameIndex >= contract.maximumSteps) {
      if (!this.#failure && !this.#completedHandoff()) {
        this.#failure = {
          code: "recovery_timeout",
          atSeconds: snapshot.simulatedTime,
          detail: "Recovery expert did not complete stable handoff before timeout"
        };
      }
      this.#finish();
    }
    return { snapshot, done: this.#result !== null };
  }

  result(): HumanoidRecoveryExecutionResult {
    if (!this.#result) throw new Error("Humanoid recovery execution is not complete");
    return structuredClone(this.#result);
  }

  #observeController(snapshot: HumanoidSimulationSnapshot): void {
    const execution = snapshot.controllerExecution;
    if (!execution) return;
    if (execution.transition) {
      this.#recoveryImplementation ??= execution.transition.fromImplementation;
    } else if (execution.activeImplementation !== this.#initialControllerImplementation) {
      this.#recoveryImplementation ??= execution.activeImplementation;
    }
    const recoveryImplementation = this.#recoveryImplementation;
    if (!recoveryImplementation) return;
    if (execution.transition?.fromImplementation === recoveryImplementation) {
      this.#handoffStarted = true;
      this.#handoffCompleted = false;
      this.#handoffSteps += 1;
      return;
    }
    if (execution.activeImplementation === recoveryImplementation) {
      if (this.#handoffStarted) {
        this.#handoffStarted = false;
        this.#handoffCompleted = false;
        this.#handoffSteps = 0;
      }
      return;
    }
    if (this.#handoffStarted && execution.transition === null) {
      this.#handoffSteps += 1;
      this.#handoffCompleted = this.#handoffSteps
        >= this.#stored.recoveryContract!.handoffSteps;
    }
  }

  #captureSafety(frame: number, snapshot: HumanoidSimulationSnapshot): void {
    try {
      this.#stored.progress.physicalSafety =
        accumulateHumanoidPhysicalSafetyFrame(
          this.#stored.progress.physicalSafety
            ?? createHumanoidPhysicalSafetyAccumulator(),
          frame,
          snapshot
        );
    } catch (error) {
      this.#stored.progress.physicalSafety = undefined;
      this.#failure = {
        code: "numerical_instability",
        atSeconds: snapshot.simulatedTime,
        detail: error instanceof Error ? error.message : String(error)
      };
    }
  }

  #updateStableStanding(snapshot: HumanoidSimulationSnapshot): void {
    this.#stableSteps = humanoidRecoveryStandingSatisfied(
      snapshot,
      this.#stored.recoveryContract!
    ) ? this.#stableSteps + 1 : 0;
  }

  #completedHandoff(): boolean {
    const contract = this.#stored.recoveryContract!;
    return this.#recoveryImplementation !== null
      && this.#stableSteps >= contract.stableSteps
      && this.#handoffCompleted
      && this.#handoffSteps >= contract.handoffSteps;
  }

  #finish(): void {
    this.#syncProgress();
    if (this.#failure) {
      this.#stored.progress.failure = {
        code: "motion_constraint_violated",
        atSeconds: this.#failure.atSeconds
      };
    }
    const physicalSafety = this.#stored.progress.physicalSafety
      ? completeHumanoidPhysicalSafetyEvidence(
          this.#stored.progress.physicalSafety
        )
      : undefined;
    this.#result = {
      frames: this.#frames,
      completed: this.#failure === null,
      reference: this.#reference,
      failure: this.#failure ? structuredClone(this.#failure) : null,
      ...(physicalSafety ? { physicalSafety } : {}),
      recoveryImplementation: this.#recoveryImplementation,
      handoffSteps: this.#handoffSteps
    };
  }

  #syncProgress(): void {
    this.#stored.progress.recovery = {
      stableSteps: this.#stableSteps,
      handoffSteps: this.#handoffSteps,
      recoveryImplementation: this.#recoveryImplementation,
      handoffStarted: this.#handoffStarted,
      handoffCompleted: this.#handoffCompleted,
      previousTotalContactForceN: this.#previousTotalContactForceN
    };
  }
}

export function humanoidRecoveryStandingSatisfied(
  snapshot: HumanoidSimulationSnapshot,
  contract: NonNullable<StoredHumanoidMotionPlan["recoveryContract"]>
): boolean {
  const standing = contract.standing;
  const pelvis = snapshot.links.pelvis;
  const maximumJointSpeed = Math.max(
    ...Object.values(snapshot.joints).map(({ velocity }) => Math.abs(velocity))
  );
  return snapshot.rootPosition.y >= standing.minimumRootHeightMeters
    && snapshot.balance.upright >= standing.minimumUpright
    && snapshot.balance.supportMargin !== null
    && snapshot.balance.supportMargin >= contract.minimumSupportMarginMeters
    && (!standing.requireBothFeetContact
      || snapshot.feet.left.touching && snapshot.feet.right.touching)
    && magnitude(pelvis.linearVelocity)
      <= standing.maximumRootLinearSpeedMetersPerSecond
    && magnitude(pelvis.angularVelocity)
      <= standing.maximumRootAngularSpeedRadiansPerSecond
    && maximumJointSpeed <= standing.maximumJointSpeedRadiansPerSecond;
}

export function humanoidRecoverySafetyFailure(
  snapshot: HumanoidSimulationSnapshot,
  contract: NonNullable<StoredHumanoidMotionPlan["recoveryContract"]>,
  frame: number,
  previousTotalContactForceN: number | null,
  controlStepSeconds: number
): HumanoidRecoveryFailure | null {
  const atSeconds = snapshot.simulatedTime;
  const allowed = new Set(
    contract.authorizedContacts.map(humanoidContactConstraintKey)
  );
  const unauthorized = humanoidEnvironmentContacts(snapshot).filter((contact) => (
    (contact.objectId !== null || contact.solidId !== null)
      && !allowed.has(humanoidEnvironmentContactKey(contact))
  ));
  if (unauthorized.length > 0) {
    return {
      code: "unauthorized_scene_contact",
      atSeconds,
      detail: "Recovery contacted a scene object or solid outside its explicit authority",
      contacts: unauthorized.map((contact) => ({ ...contact }))
    };
  }
  const limits = contract.safetyLimits;
  const peakContact = Math.max(0, ...snapshot.contacts.map(({ normalForce }) => normalForce));
  const totalContact = snapshot.contacts.reduce(
    (sum, { normalForce }) => sum + normalForce,
    0
  );
  if (peakContact > limits.maximumPeakContactNormalForceN
    || totalContact > limits.maximumTotalContactNormalForceN) {
    return {
      code: "contact_force_limit",
      atSeconds,
      detail: `Recovery contact force exceeded its catastrophic limit at frame ${frame}`
    };
  }
  if (previousTotalContactForceN !== null
    && (totalContact - previousTotalContactForceN) / controlStepSeconds
      > limits.maximumTotalContactForceRiseRateNPerSecond) {
    return {
      code: "contact_impact_limit",
      atSeconds,
      detail: `Recovery contact-force rise exceeded its catastrophic limit at frame ${frame}`
    };
  }
  const safety = snapshot.joints;
  const maximumJointSpeed = Math.max(
    ...Object.values(safety).map(({ velocity }) => Math.abs(velocity))
  );
  if (maximumJointSpeed > limits.maximumJointSpeedRadiansPerSecond) {
    return {
      code: "joint_velocity_limit",
      atSeconds,
      detail: `Recovery joint speed exceeded ${limits.maximumJointSpeedRadiansPerSecond}rad/s`
    };
  }
  const minimumJointMargin = Math.min(...Object.values(safety).map((joint) => (
    Math.min(joint.position - joint.minimum, joint.maximum - joint.position)
  )));
  if (minimumJointMargin < limits.minimumJointLimitMarginRadians) {
    return {
      code: "joint_limit_violation",
      atSeconds,
      detail: "Recovery exceeded a physical joint limit"
    };
  }
  return null;
}

function magnitude(vector: { x: number; y: number; z: number }): number {
  return Math.hypot(vector.x, vector.y, vector.z);
}

function boundedRatio(value: number, total: number): number {
  return total <= 0 ? 0 : Math.min(1, Math.max(0, value / total));
}
