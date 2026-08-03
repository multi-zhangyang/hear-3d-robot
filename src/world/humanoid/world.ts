import type { Scenario, Vec3 } from "../../domain/schema.js";
import {
  NavigationMesh,
  type NavigationObstacle,
  type NavigationPlan
} from "../navigation.js";
import {
  HUMANOID_NAVIGATION_PROFILE,
  humanoidEnvironment
} from "./environment.js";
import {
  HumanoidWorldCheckpointSchema,
  type HumanoidMotionExecutionProgress,
  type HumanoidMotionOptionExecutionState,
  type HumanoidWorldCheckpoint
} from "./checkpoint.js";
import {
  blockedHumanoidContacts,
  humanoidContactKey,
  humanoidObjectContacts,
  HumanoidMotionCandidateBatchSchema,
  HumanoidMotionPlanSchema,
  missingRequiredHumanoidContacts,
  occupiedHumanoidChannels,
  prepareHumanoidMotion,
  TaskSpaceHumanoidMotionGenerator,
  type HumanoidMotionPlan,
  type HumanoidMotionCandidateBatch,
  type HumanoidBodyChannel,
  type HumanoidMotionGenerator,
  type HumanoidMotionOptionCertificate,
  type HumanoidMotionValidation,
  type PreparedHumanoidMotion
} from "./motion-plan.js";
import type { HumanoidMotionGeneratorDescriptor } from "./motion-generator-contract.js";
import {
  humanoidMotionArtifactSummary,
  humanoidMotionArtifactSha256,
  hydrateHumanoidReference,
  serializeHumanoidReference,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import {
  detectHumanoidMotionDrift,
  humanoidMotionRolloutSha256,
  type HumanoidMotionDriftEvidence,
  type HumanoidMotionRollout
} from "./motion-rollout.js";
import {
  advanceHumanoidMotionOptionMonitor,
  createHumanoidMotionOptionMonitorState,
  detectHumanoidMotionOption,
  humanoidMotionOptionContractSha256,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionDetection,
  type HumanoidMotionOptionDetectorInput,
  type HumanoidMotionOptionObservableObject
} from "./motion-option.js";
import {
  HumanoidObjectMemory,
  type HumanoidObjectMemoryCheckpoint,
  type HumanoidObjectToken
} from "./object-memory.js";
import {
  neutralHumanoidReference,
  releaseReferenceTracking,
  targetReference,
  type HumanoidReference
} from "./reference.js";
import {
  HumanoidSimulation,
  type HumanoidSimulationSnapshot
} from "./simulation.js";

export interface HumanoidWorldSnapshot {
  frame: number;
  worldRevision: number;
  motionGenerator: HumanoidMotionGeneratorDescriptor;
  robot: HumanoidSimulationSnapshot;
  navigation: {
    planId: string | null;
    status: "idle" | "planned" | "executing" | "completed" | "blocked";
    target: Vec3 | null;
    waypoints: Vec3[];
    waypointIndex: number | null;
  };
}

export interface HumanoidWorldObservation {
  frame: number;
  worldRevision: number;
  motionGenerator: HumanoidMotionGeneratorDescriptor;
  sensor: ReturnType<HumanoidSimulation["senseObjects"]>["sensor"];
  robot: Omit<HumanoidSimulationSnapshot, "objects">;
  objectTokens: HumanoidObjectToken[];
  navigation: HumanoidWorldSnapshot["navigation"];
}

export interface WholeBodyPlanReceipt {
  accepted: boolean;
  planId: string;
  createdRevision: number;
  channels: HumanoidBodyChannel[];
  motion: ReturnType<typeof humanoidMotionArtifactSummary> | null;
  validation: HumanoidMotionValidation;
}

interface WholeBodyCandidateEvaluation {
  rank: number;
  planId: string;
  intent: string;
  channels: HumanoidBodyChannel[];
  motion: ReturnType<typeof humanoidMotionArtifactSummary> | null;
  optionCertificate: HumanoidMotionOptionCertificate | null;
  validation: HumanoidMotionValidation;
}

export interface WholeBodyCandidatePlanReceipt {
  accepted: boolean;
  planId: string;
  selectedCandidateId: string | null;
  selectedRank: number | null;
  createdRevision: number;
  channels: HumanoidBodyChannel[];
  motion: ReturnType<typeof humanoidMotionArtifactSummary> | null;
  option: {
    contract: HumanoidMotionOptionContract;
    certificate: HumanoidMotionOptionCertificate;
  } | null;
  selection: "model_rank_then_physics";
  candidates: WholeBodyCandidateEvaluation[];
}

export interface HumanoidExecutionReceipt {
  accepted: boolean;
  code: "motion_completed" | "navigation_completed" | "plan_stale"
    | "motion_failed" | "navigation_blocked" | "motion_option_succeeded"
    | "motion_goal_unmet" | "motion_goal_uncertain"
    | "motion_execution_drifted" | "motion_constraint_violated";
  frames: number;
  finalSnapshot: HumanoidWorldSnapshot;
  detail: {
    failures?: HumanoidMotionValidation["failures"];
    reason?: string;
    travelledDistance?: number;
    motion?: ReturnType<typeof humanoidMotionArtifactSummary>;
    option?: {
      option_id: string;
      status: HumanoidMotionOptionExecutionState["status"];
      termination_reason: HumanoidMotionOptionExecutionState["terminationReason"];
      full_frame_count: number;
      executed_prefix_frame_count: number;
      predicted_termination_frame: number;
      actual_termination_frame: number | null;
      artifact_sha256: string;
      rollout_sha256: string;
      drift_streak: number;
      drift_evidence: HumanoidMotionDriftEvidence | null;
      evidence: HumanoidMotionOptionExecutionState["lastEvidence"];
    };
  };
}

export interface NavigationPlanReceipt {
  accepted: boolean;
  planId: string;
  createdRevision: number;
  target: Vec3;
  chunkTarget: Vec3;
  waypoints: Vec3[];
  distance: number;
  remainingDistance: number;
  reason?: string;
}

export type HumanoidFrameSink = (snapshot: HumanoidWorldSnapshot) => void | Promise<void>;

interface StoredMotionPlan {
  plan: HumanoidMotionPlan;
  artifact: HumanoidMotionArtifact;
  rollout: HumanoidMotionRollout | null;
  createdRevision: number;
  option: HumanoidMotionOptionExecutionState | null;
  progress: HumanoidMotionExecutionProgress;
}

interface StoredNavigationPlan {
  id: string;
  plan: NavigationPlan;
  requestedTarget: Vec3;
  createdRevision: number;
}

export interface HumanoidWorldOptions {
  motionGeneratorFactory?: () => Promise<HumanoidMotionGenerator>;
}

interface RouteRun {
  completed: boolean;
  reason?: string;
  frames: number;
  reference: HumanoidReference;
  final: HumanoidSimulationSnapshot;
  travelledDistance: number;
}

const NAVIGATION_CHUNK_DISTANCE = 3;

export class HumanoidWorld {
  readonly #scenario: Scenario;
  readonly #simulation: HumanoidSimulation;
  readonly #navigation: NavigationMesh;
  readonly #objectMemory: HumanoidObjectMemory;
  readonly #motionGenerator: HumanoidMotionGenerator;
  readonly #motions = new Map<string, StoredMotionPlan>();
  readonly #routes = new Map<string, StoredNavigationPlan>();
  #reference = neutralHumanoidReference();
  #frame = 0;
  #worldRevision = 0;
  #routeSequence = 0;
  #navigationState: HumanoidWorldSnapshot["navigation"] = {
    planId: null,
    status: "idle",
    target: null,
    waypoints: [],
    waypointIndex: null
  };

  static async create(
    scenario: Scenario,
    checkpoint?: HumanoidWorldCheckpoint,
    options: HumanoidWorldOptions = {}
  ): Promise<HumanoidWorld> {
    const [simulation, navigation, motionGenerator] = await Promise.all([
      HumanoidSimulation.create(humanoidEnvironment(scenario)),
      NavigationMesh.create(scenario, undefined, HUMANOID_NAVIGATION_PROFILE),
      options.motionGeneratorFactory
        ? options.motionGeneratorFactory()
        : Promise.resolve(new TaskSpaceHumanoidMotionGenerator())
    ]);
    try {
      const world = new HumanoidWorld(
        scenario,
        simulation,
        navigation,
        motionGenerator,
        checkpoint?.objectMemory
      );
      if (checkpoint) world.#restore(checkpoint);
      else await world.#settle(80);
      return world;
    } catch (error) {
      navigation.dispose();
      await simulation.dispose();
      await motionGenerator.dispose();
      throw error;
    }
  }

  private constructor(
    scenario: Scenario,
    simulation: HumanoidSimulation,
    navigation: NavigationMesh,
    motionGenerator: HumanoidMotionGenerator,
    objectMemoryCheckpoint?: HumanoidObjectMemoryCheckpoint
  ) {
    this.#scenario = scenario;
    this.#simulation = simulation;
    this.#navigation = navigation;
    this.#motionGenerator = motionGenerator;
    this.#objectMemory = new HumanoidObjectMemory(scenario, objectMemoryCheckpoint);
  }

  snapshot(): HumanoidWorldSnapshot {
    return {
      frame: this.#frame,
      worldRevision: this.#worldRevision,
      motionGenerator: structuredClone(this.#motionGenerator.descriptor),
      robot: this.#simulation.snapshot(),
      navigation: structuredClone(this.#navigationState)
    };
  }

  observe(): HumanoidWorldObservation {
    const snapshot = this.snapshot();
    const sensed = this.#simulation.senseObjects(this.#scenario.visibility_radius);
    this.#objectMemory.refresh(
      snapshot.frame,
      snapshot.worldRevision,
      snapshot.robot.objects,
      new Set(Object.keys(sensed.objects))
    );
    const { objects: _objects, ...robot } = snapshot.robot;
    return {
      frame: snapshot.frame,
      worldRevision: snapshot.worldRevision,
      motionGenerator: structuredClone(snapshot.motionGenerator),
      sensor: sensed.sensor,
      robot,
      objectTokens: this.#objectMemory.tokens(
        snapshot.robot,
        snapshot.frame,
        snapshot.worldRevision
      ),
      navigation: structuredClone(snapshot.navigation)
    };
  }

  checkpoint(): HumanoidWorldCheckpoint {
    const simulation = this.#simulation.captureState();
    return HumanoidWorldCheckpointSchema.parse({
      version: 1,
      motionGenerator: structuredClone(this.#motionGenerator.descriptor),
      frame: this.#frame,
      worldRevision: this.#worldRevision,
      routeSequence: this.#routeSequence,
      simulation: {
        time: simulation.time,
        positions: [...simulation.positions],
        velocities: [...simulation.velocities],
        controls: [...simulation.controls],
        activations: [...simulation.activations],
        accelerationWarmstart: [...simulation.accelerationWarmstart],
        controller: structuredClone(simulation.controller)
      },
      reference: serializeHumanoidReference(this.#reference),
      motions: [...this.#motions.values()].map((entry) => structuredClone(entry)),
      routes: [...this.#routes.values()].map((entry) => structuredClone(entry)),
      navigation: structuredClone(this.#navigationState),
      objectMemory: this.#objectMemory.checkpoint()
    });
  }

  consumablePlanIds(): string[] {
    return [...this.#motions.keys(), ...this.#routes.keys()];
  }

  async planWholeBodyMotion(rawPlan: HumanoidMotionPlan): Promise<WholeBodyPlanReceipt> {
    const plan = HumanoidMotionPlanSchema.parse(rawPlan);
    if (this.#motions.has(plan.id)) throw new Error(`Duplicate humanoid motion plan: ${plan.id}`);
    const prepared = await prepareHumanoidMotion(
      this.#simulation,
      plan,
      this.#reference,
      {
        contactObjectIds: this.#objectMemory.observedObjectIds(
          this.#frame,
          this.#worldRevision
        )
      },
      this.#motionGenerator
    );
    if (prepared.validation.feasible && prepared.artifact) {
      this.#motions.set(plan.id, {
        plan: structuredClone(plan),
        artifact: structuredClone(prepared.artifact),
        rollout: null,
        createdRevision: this.#worldRevision,
        option: null,
        progress: {
          nextFrameIndex: 0,
          satisfiedContactKeys: [],
          driftStreak: 0,
          lastDrift: null,
          failure: null
        }
      });
    }
    return {
      accepted: prepared.validation.feasible,
      planId: plan.id,
      createdRevision: this.#worldRevision,
      channels: occupiedHumanoidChannels(plan),
      motion: prepared.artifact
        ? humanoidMotionArtifactSummary(prepared.artifact)
        : null,
      validation: prepared.validation
    };
  }

  async planWholeBodyMotionCandidates(
    rawBatch: HumanoidMotionCandidateBatch
  ): Promise<WholeBodyCandidatePlanReceipt> {
    const batch = HumanoidMotionCandidateBatchSchema.parse(rawBatch);
    for (const candidate of batch.candidates) {
      if (this.#motions.has(candidate.id)) {
        throw new Error(`Duplicate humanoid motion plan: ${candidate.id}`);
      }
    }
    const visibleContactObjects = this.#objectMemory.observedObjectIds(
      this.#frame,
      this.#worldRevision
    );
    const prepared: Array<{
      rank: number;
      plan: HumanoidMotionPlan;
      channels: HumanoidBodyChannel[];
      result: PreparedHumanoidMotion;
    }> = [];
    for (let index = 0; index < batch.candidates.length; index += 1) {
      const plan = batch.candidates[index]!;
      const result = await prepareHumanoidMotion(
        this.#simulation,
        plan,
        this.#reference,
        {
          contactObjectIds: visibleContactObjects,
          motionOption: {
            contract: batch.termination,
            scenario: this.#scenario
          }
        },
        this.#motionGenerator
      );
      prepared.push({
        rank: index + 1,
        plan,
        channels: occupiedHumanoidChannels(plan),
        result
      });
    }
    const selected = prepared.find((candidate) => (
      candidate.result.validation.feasible
      && candidate.result.artifact !== null
      && candidate.result.rollout !== null
      && candidate.result.optionCertificate !== null
    ));
    if (selected?.result.artifact
      && selected.result.rollout
      && selected.result.optionCertificate) {
      this.#motions.set(selected.plan.id, {
        plan: structuredClone(selected.plan),
        artifact: structuredClone(selected.result.artifact),
        rollout: structuredClone(selected.result.rollout),
        createdRevision: this.#worldRevision,
        option: {
          contract: structuredClone(batch.termination),
          certificate: structuredClone(selected.result.optionCertificate),
          monitor: createHumanoidMotionOptionMonitorState(batch.termination),
          status: "planned",
          successStreak: 0,
          actualTerminationFrame: null,
          terminationReason: null,
          lastEvidence: null
        },
        progress: {
          nextFrameIndex: 0,
          satisfiedContactKeys: [],
          driftStreak: 0,
          lastDrift: null,
          failure: null
        }
      });
    }
    return {
      accepted: selected !== undefined,
      planId: selected?.plan.id ?? "",
      selectedCandidateId: selected?.plan.id ?? null,
      selectedRank: selected?.rank ?? null,
      createdRevision: this.#worldRevision,
      channels: selected?.channels ?? [],
      motion: selected?.result.artifact
        ? humanoidMotionArtifactSummary(selected.result.artifact)
        : null,
      option: selected?.result.optionCertificate
        ? {
            contract: structuredClone(batch.termination),
            certificate: structuredClone(selected.result.optionCertificate)
          }
        : null,
      selection: "model_rank_then_physics",
      candidates: prepared.map((candidate) => ({
        rank: candidate.rank,
        planId: candidate.plan.id,
        intent: candidate.plan.intent,
        channels: candidate.channels,
        motion: candidate.result.artifact
          ? humanoidMotionArtifactSummary(candidate.result.artifact)
          : null,
        optionCertificate: candidate.result.optionCertificate
          ? structuredClone(candidate.result.optionCertificate)
          : null,
        validation: candidate.result.validation
      }))
    };
  }

  async executeWholeBodyMotion(
    planId: string,
    frameSink?: HumanoidFrameSink
  ): Promise<HumanoidExecutionReceipt> {
    const stored = this.#motions.get(planId);
    if (!stored) throw new Error(`Unknown humanoid motion plan: ${planId}`);
    if (stored.option) assertMotionOptionIntegrity(stored);
    const expectedRevision = stored.createdRevision + stored.progress.nextFrameIndex;
    if (expectedRevision !== this.#worldRevision) {
      this.#motions.delete(planId);
      return this.#receipt(false, "plan_stale", 0, {
        reason: `expected_revision=${expectedRevision}, world_revision=${this.#worldRevision}`
      });
    }
    if (stored.option && isTerminalMotionOption(stored.option)) {
      if (stored.option.status === "succeeded"
        && missingRequiredHumanoidContacts(
          stored.plan.contact_constraints ?? [],
          new Set(stored.progress.satisfiedContactKeys)
        ).length > 0) {
        stored.option.status = "goal_unmet";
        stored.option.terminationReason = "motion_goal_unmet";
      }
      const receipt = this.#motionOptionReceipt(stored, 0);
      this.#reference = releaseReferenceTracking(this.#reference);
      this.#motions.delete(planId);
      return receipt;
    }
    let frames = 0;
    const failures: HumanoidMotionValidation["failures"] = stored.progress.failure
      ? [validationFailure(stored.progress.failure)]
      : [];
    const constraints = stored.plan.contact_constraints ?? [];
    const satisfiedContacts = new Set(stored.progress.satisfiedContactKeys);
    let lastReference = this.#reference;
    let lastOptionDetection: HumanoidMotionOptionDetection | null = null;
    if (stored.option) stored.option.status = "executing";
    const frameLimit = stored.option
      ? stored.option.certificate.validated_frame_limit
      : stored.artifact.frames.length;
    for (
      let index = stored.progress.nextFrameIndex;
      index < frameLimit && failures.length === 0;
      index += 1
    ) {
      const frame = stored.artifact.frames[index]!;
      if (stored.option?.monitor.phase === "awaiting_precondition") {
        const update = advanceHumanoidMotionOptionMonitor(
          stored.option.contract,
          stored.option.monitor,
          this.#motionOptionDetectorInput(this.#simulation.snapshot())
        );
        stored.option.monitor = update.state;
        stored.option.successStreak = update.state.terminalStableSteps;
        lastOptionDetection = update.detection;
        stored.option.lastEvidence = jsonOptionEvidence(
          update.detection,
          update.state
        );
        if (update.observationStatus !== "satisfied") {
          const atSeconds = index === 0
            ? 0
            : stored.artifact.frames[index - 1]!.atSeconds;
          const uncertain = update.observationStatus === "uncertain";
          failures.push({
            code: uncertain ? "motion_goal_uncertain" : "motion_constraint_violated",
            atSeconds,
            message: uncertain
              ? "Motion option precondition is not observable before execution"
              : "Motion option precondition is not satisfied before execution"
          });
          if (!uncertain) {
            stored.progress.failure = {
              code: "motion_constraint_violated",
              atSeconds
            };
          }
          stored.option.status = uncertain ? "goal_unmet" : "failed";
          stored.option.actualTerminationFrame = index;
          stored.option.terminationReason = uncertain
            ? "motion_goal_uncertain"
            : "motion_constraint_violated";
          break;
        }
      }
      const reference = hydrateHumanoidReference(frame.reference);
      const snapshot = await this.#simulation.step(reference);
      frames += 1;
      lastReference = reference;
      this.#reference = reference;
      if (stored.option && stored.rollout) {
        const predicted = stored.rollout.frames[index];
        if (!predicted || Math.abs(predicted.atSeconds - frame.atSeconds) > 1e-9) {
          throw new Error("Humanoid motion rollout frame does not match its artifact");
        }
        const drift = detectHumanoidMotionDrift(
          snapshot,
          predicted,
          stored.rollout.limits
        );
        stored.progress.lastDrift = drift;
        stored.progress.driftStreak = drift.drifted
          ? stored.progress.driftStreak + 1
          : 0;
      }
      for (const contact of humanoidObjectContacts(snapshot)) {
        if (contact.objectId === null) continue;
        if (constraints.some((constraint) => (
          constraint.body === contact.body && constraint.object_id === contact.objectId
        ))) {
          satisfiedContacts.add(humanoidContactKey(contact.body, contact.objectId));
        }
      }
      const blockedContacts = blockedHumanoidContacts(snapshot, constraints);
      if (blockedContacts.length > 0) {
        const bodies = [...new Set(blockedContacts.map((contact) => contact.body))];
        const contacts = blockedContacts.map((contact) => ({ ...contact }));
        const failure: HumanoidMotionValidation["failures"][number] = {
          code: "environment_contact",
          atSeconds: frame.atSeconds,
          bodies,
          contacts
        };
        failures.push(failure);
        stored.progress.failure = {
          code: "environment_contact",
          atSeconds: frame.atSeconds,
          bodies: [...bodies],
          contacts: contacts.map((contact) => ({ ...contact }))
        };
      } else if (snapshot.fallen) {
        const failure = { code: "fallen", atSeconds: frame.atSeconds } as const;
        failures.push(failure);
        stored.progress.failure = failure;
      } else if (stored.option
        && stored.rollout
        && stored.progress.lastDrift?.drifted
        && stored.progress.driftStreak >= stored.rollout.limits.consecutive_steps) {
        const failure = {
          code: "execution_drift",
          atSeconds: frame.atSeconds,
          drift: { ...stored.progress.lastDrift },
          message: "Physical execution persistently diverged from its validated rollout"
        } as const;
        failures.push(failure);
        stored.progress.failure = {
          code: failure.code,
          atSeconds: failure.atSeconds,
          drift: { ...failure.drift }
        };
      }
      stored.progress.nextFrameIndex = index + 1;
      stored.progress.satisfiedContactKeys = [...satisfiedContacts];
      if (stored.option) {
        const detectorInput = this.#motionOptionDetectorInput(snapshot);
        if (failures.length > 0) {
          lastOptionDetection = detectHumanoidMotionOption(
            stored.option.contract,
            detectorInput
          );
          stored.option.lastEvidence = jsonOptionEvidence(
            lastOptionDetection,
            stored.option.monitor
          );
          stored.option.status = "failed";
          stored.option.actualTerminationFrame = index + 1;
          stored.option.terminationReason = stored.progress.failure?.code
            ?? "environment_contact";
        } else {
          const update = advanceHumanoidMotionOptionMonitor(
            stored.option.contract,
            stored.option.monitor,
            detectorInput
          );
          stored.option.monitor = update.state;
          stored.option.successStreak = update.state.terminalStableSteps;
          lastOptionDetection = update.detection;
          stored.option.lastEvidence = jsonOptionEvidence(
            update.detection,
            update.state
          );
          if (update.state.phase === "indeterminate") {
            failures.push({
              code: "motion_goal_uncertain",
              atSeconds: frame.atSeconds,
              message: "Motion option lost observable evidence for its during constraint"
            });
            stored.option.status = "goal_unmet";
            stored.option.actualTerminationFrame = index + 1;
            stored.option.terminationReason = "motion_goal_uncertain";
          } else if (update.state.phase === "violated") {
            const failure = {
              code: "motion_constraint_violated",
              atSeconds: frame.atSeconds,
              message: "Motion option violated its during constraint"
            } as const;
            failures.push(failure);
            stored.progress.failure = {
              code: failure.code,
              atSeconds: failure.atSeconds
            };
            stored.option.status = "failed";
            stored.option.actualTerminationFrame = index + 1;
            stored.option.terminationReason = failure.code;
          }
          const missingRequired = missingRequiredHumanoidContacts(
            constraints,
            satisfiedContacts
          );
          if (update.state.phase === "succeeded"
            && missingRequired.length === 0
            && snapshot.balance.support !== "none"
            && failures.length === 0) {
            stored.option.status = "succeeded";
            stored.option.actualTerminationFrame = index + 1;
            stored.option.terminationReason = "physical_success";
          }
        }
      }
      await this.#commitFrame(frameSink, undefined, { motionPlanId: planId });
      if (stored.option && isTerminalMotionOption(stored.option)) break;
    }
    if (stored.option) {
      if (!isTerminalMotionOption(stored.option)) {
        const uncertain = lastOptionDetection?.hasUncertain ?? true;
        stored.option.status = "goal_unmet";
        stored.option.actualTerminationFrame = stored.progress.nextFrameIndex;
        stored.option.terminationReason = uncertain
          ? "motion_goal_uncertain"
          : "motion_goal_unmet";
        failures.push({
          code: uncertain ? "motion_goal_uncertain" : "motion_goal_unmet",
          atSeconds: stored.artifact.frames[
            Math.max(0, stored.progress.nextFrameIndex - 1)
          ]!.atSeconds,
          message: uncertain
            ? "Motion option ended without observable success evidence"
            : "Motion option exhausted its verified horizon before physical success"
        });
      }
      this.#reference = releaseReferenceTracking(lastReference);
      const receipt = this.#motionOptionReceipt(stored, frames, failures);
      this.#motions.delete(planId);
      return receipt;
    }
    const missingContacts = missingRequiredHumanoidContacts(
      constraints,
      satisfiedContacts
    );
    if (failures.length === 0 && missingContacts.length > 0) {
      failures.push({
        code: "required_contact_missing",
        atSeconds: stored.plan.duration_seconds,
        constraints: missingContacts
      });
    }
    this.#reference = releaseReferenceTracking(lastReference);
    this.#motions.delete(planId);
    return this.#receipt(
      failures.length === 0,
      failures.length === 0 ? "motion_completed" : "motion_failed",
      frames,
      {
        motion: humanoidMotionArtifactSummary(stored.artifact),
        ...(failures.length === 0 ? {} : { failures })
      }
    );
  }

  async planNavigation(target: Vec3): Promise<NavigationPlanReceipt> {
    const start = this.#simulation.snapshot().rootPosition;
    let completePlan: NavigationPlan;
    try {
      completePlan = this.#navigation.plan(start, target, this.#dynamicNavigationObstacles());
    } catch (error) {
      return {
        accepted: false,
        planId: "",
        createdRevision: this.#worldRevision,
        target: { ...target },
        chunkTarget: { ...start },
        waypoints: [],
        distance: 0,
        remainingDistance: 0,
        reason: error instanceof Error ? error.message : String(error)
      };
    }
    const plan = navigationChunk(completePlan, NAVIGATION_CHUNK_DISTANCE);
    const remainingDistance = Math.max(0, completePlan.distance - plan.distance);
    const state = this.#simulation.captureState();
    let preview: RouteRun;
    try {
      preview = await this.#followRoute(plan, this.#reference);
    } finally {
      this.#simulation.restoreState(state);
    }
    if (!preview.completed) {
      return {
        accepted: false,
        planId: "",
        createdRevision: this.#worldRevision,
        target: { ...target },
        chunkTarget: { ...plan.resolvedTarget },
        waypoints: plan.waypoints.map((point) => ({ ...point })),
        distance: plan.distance,
        remainingDistance,
        reason: preview.reason ?? "physical_preview_failed"
      };
    }
    const planId = `humanoid-route-${this.#routeSequence++}`;
    this.#routes.set(planId, {
      id: planId,
      plan,
      requestedTarget: { ...target },
      createdRevision: this.#worldRevision
    });
    this.#navigationState = {
      planId,
      status: "planned",
      target: { ...target },
      waypoints: plan.waypoints.map((point) => ({ ...point })),
      waypointIndex: 1
    };
    return {
      accepted: true,
      planId,
      createdRevision: this.#worldRevision,
      target: { ...target },
      chunkTarget: { ...plan.resolvedTarget },
      waypoints: plan.waypoints.map((point) => ({ ...point })),
      distance: plan.distance,
      remainingDistance
    };
  }

  async executeNavigation(
    planId: string,
    frameSink?: HumanoidFrameSink
  ): Promise<HumanoidExecutionReceipt> {
    const stored = this.#routes.get(planId);
    if (!stored) throw new Error(`Unknown humanoid navigation plan: ${planId}`);
    if (stored.createdRevision !== this.#worldRevision) {
      this.#routes.delete(planId);
      this.#navigationState.status = "blocked";
      return this.#receipt(false, "plan_stale", 0, {
        reason: `plan_revision=${stored.createdRevision}, world_revision=${this.#worldRevision}`
      });
    }
    this.#navigationState.status = "executing";
    const run = await this.#followRoute(stored.plan, this.#reference, async (snapshot, index) => {
      this.#navigationState.waypointIndex = index;
      await this.#commitFrame(frameSink, snapshot, { routePlanId: planId });
    });
    this.#reference = run.reference;
    this.#routes.delete(planId);
    this.#navigationState.status = run.completed ? "completed" : "blocked";
    return this.#receipt(
      run.completed,
      run.completed ? "navigation_completed" : "navigation_blocked",
      run.frames,
      {
        ...(run.reason ? { reason: run.reason } : {}),
        travelledDistance: run.travelledDistance
      }
    );
  }

  async dispose(): Promise<void> {
    this.#navigation.dispose();
    await Promise.all([
      this.#simulation.dispose(),
      this.#motionGenerator.dispose()
    ]);
  }

  async #settle(steps: number): Promise<void> {
    for (let index = 0; index < steps; index += 1) {
      await this.#simulation.step(this.#reference);
      this.#frame += 1;
    }
    if (this.#simulation.snapshot().fallen) {
      throw new Error("Humanoid could not reach a stable initial stance");
    }
  }

  #restore(rawCheckpoint: HumanoidWorldCheckpoint): void {
    const checkpoint = HumanoidWorldCheckpointSchema.parse(rawCheckpoint);
    const expectedGenerator = this.#motionGenerator.descriptor;
    if (checkpoint.motionGenerator.protocol !== expectedGenerator.protocol
      || checkpoint.motionGenerator.implementation !== expectedGenerator.implementation
      || checkpoint.motionGenerator.motionClass !== expectedGenerator.motionClass
      || checkpoint.motionGenerator.sampling !== expectedGenerator.sampling) {
      throw new Error(
        `Humanoid motion generator mismatch: checkpoint=${checkpoint.motionGenerator.implementation}, `
        + `runtime=${expectedGenerator.implementation}`
      );
    }
    this.#simulation.restoreState({
      time: checkpoint.simulation.time,
      positions: Float64Array.from(checkpoint.simulation.positions),
      velocities: Float64Array.from(checkpoint.simulation.velocities),
      controls: Float64Array.from(checkpoint.simulation.controls),
      activations: Float64Array.from(checkpoint.simulation.activations),
      accelerationWarmstart: Float64Array.from(
        checkpoint.simulation.accelerationWarmstart
      ),
      controller: structuredClone(checkpoint.simulation.controller)
    });
    this.#reference = hydrateHumanoidReference(checkpoint.reference);
    this.#frame = checkpoint.frame;
    this.#worldRevision = checkpoint.worldRevision;
    this.#routeSequence = checkpoint.routeSequence;
    this.#motions.clear();
    for (const entry of checkpoint.motions) {
      const expectedRevision = entry.createdRevision + entry.progress.nextFrameIndex;
      if (expectedRevision === checkpoint.worldRevision) {
        const restored = structuredClone(entry);
        restored.progress.satisfiedContactKeys = [];
        this.#motions.set(entry.plan.id, restored);
      }
    }
    this.#routes.clear();
    for (const entry of checkpoint.routes) {
      if (entry.createdRevision === checkpoint.worldRevision) {
        this.#routes.set(entry.id, structuredClone(entry));
      }
    }
    this.#navigationState = structuredClone(checkpoint.navigation);
    const navigationPlanId = this.#navigationState.planId;
    if (this.#navigationState.status === "executing"
      || (this.#navigationState.status === "planned"
        && (navigationPlanId === null || !this.#routes.has(navigationPlanId)))) {
      if (navigationPlanId) this.#routes.delete(navigationPlanId);
      this.#navigationState.planId = null;
      this.#navigationState.status = "blocked";
      this.#navigationState.waypointIndex = null;
    }
  }

  async #followRoute(
    plan: NavigationPlan,
    initialReference: HumanoidReference,
    onFrame?: (snapshot: HumanoidSimulationSnapshot, waypointIndex: number) => Promise<void>
  ): Promise<RouteRun> {
    const start = this.#simulation.snapshot();
    let final = start;
    let reference = initialReference;
    let waypointIndex = Math.min(1, plan.waypoints.length - 1);
    let frames = 0;
    const maximumFrames = Math.ceil(Math.min(60, plan.distance / 0.12 + 8) / 0.02);
    while (frames < maximumFrames && waypointIndex < plan.waypoints.length) {
      const waypoint = plan.waypoints[waypointIndex]!;
      const dx = waypoint.x - final.rootPosition.x;
      const dz = waypoint.z - final.rootPosition.z;
      const distance = Math.hypot(dx, dz);
      if (distance <= 0.18) {
        waypointIndex += 1;
        continue;
      }
      const yaw = yawFromQuaternion(final.rootRotation);
      const desiredYaw = Math.atan2(dx, dz);
      const yawError = normalizeAngle(desiredYaw - yaw);
      const localForward = dx * Math.sin(yaw) + dz * Math.cos(yaw);
      const localLateral = dx * Math.cos(yaw) - dz * Math.sin(yaw);
      reference = targetReference(reference, {
        rootVelocity: [
          clamp(localForward * 0.9, -0.3, 0.48),
          clamp(localLateral * 0.8, -0.22, 0.22)
        ],
        rootYawVelocity: clamp(yawError * 1.8, -1, 1)
      });
      final = await this.#simulation.step(reference);
      frames += 1;
      await onFrame?.(final, waypointIndex);
      if (final.fallen) {
        return routeResult(false, "fallen", frames, reference, start, final);
      }
      const blockedContacts = blockedHumanoidContacts(final, []);
      if (blockedContacts.length > 0) {
        return routeResult(
          false,
          `environment_contact:${blockedContacts.map((contact) => (
            `${contact.body}:${contact.objectId ?? "environment"}`
          )).join(",")}`,
          frames,
          reference,
          start,
          final
        );
      }
    }
    if (waypointIndex < plan.waypoints.length) {
      const waypoint = plan.waypoints[waypointIndex]!;
      return routeResult(
        false,
        `navigation_timeout:position=${point(final.rootPosition)},target=${point(waypoint)}`,
        frames,
        reference,
        start,
        final
      );
    }
    reference = targetReference(reference, {
      rootVelocity: [0, 0],
      rootYawVelocity: 0
    });
    for (let index = 0; index < 30; index += 1) {
      final = await this.#simulation.step(reference);
      frames += 1;
      await onFrame?.(final, plan.waypoints.length - 1);
      if (final.fallen) return routeResult(false, "fallen_while_stopping", frames, reference, start, final);
      if (blockedHumanoidContacts(final, []).length > 0) {
        return routeResult(false, "contact_while_stopping", frames, reference, start, final);
      }
    }
    return routeResult(true, undefined, frames, reference, start, final);
  }

  async #commitFrame(
    sink?: HumanoidFrameSink,
    _snapshot?: HumanoidSimulationSnapshot,
    activePlan: { motionPlanId?: string; routePlanId?: string } = {}
  ): Promise<void> {
    this.#frame += 1;
    this.#worldRevision += 1;
    this.#pruneUnconsumablePlans(activePlan);
    await sink?.(this.snapshot());
  }

  #pruneUnconsumablePlans(activePlan: {
    motionPlanId?: string;
    routePlanId?: string;
  }): void {
    for (const [planId, stored] of this.#motions) {
      if (planId === activePlan.motionPlanId) continue;
      const expectedRevision = stored.createdRevision + stored.progress.nextFrameIndex;
      if (expectedRevision !== this.#worldRevision) this.#motions.delete(planId);
    }
    for (const [planId, stored] of this.#routes) {
      if (planId === activePlan.routePlanId) continue;
      if (stored.createdRevision !== this.#worldRevision) this.#routes.delete(planId);
    }
    const navigationPlanId = this.#navigationState.planId;
    if (navigationPlanId
      && navigationPlanId !== activePlan.routePlanId
      && !this.#routes.has(navigationPlanId)) {
      this.#navigationState.planId = null;
      this.#navigationState.status = "blocked";
      this.#navigationState.waypointIndex = null;
    }
  }

  #dynamicNavigationObstacles(): NavigationObstacle[] {
    const objects = this.#simulation.snapshot().objects;
    return this.#scenario.objects
      .filter((object) => object.portable)
      .map((object) => ({
        id: `object-${object.id}`,
        center: { ...(objects[object.id]?.position ?? object.position) },
        halfExtents: {
          x: object.size.x / 2,
          y: object.size.y / 2,
          z: object.size.z / 2
        },
        yaw: 0
      }));
  }

  #motionOptionDetectorInput(
    snapshot: HumanoidSimulationSnapshot
  ): HumanoidMotionOptionDetectorInput {
    const sensed = this.#simulation.senseObjects(
      this.#scenario.visibility_radius
    );
    const observedFrame = this.#frame + 1;
    const observedWorldRevision = this.#worldRevision + 1;
    this.#objectMemory.refresh(
      observedFrame,
      observedWorldRevision,
      snapshot.objects,
      new Set(Object.keys(sensed.objects))
    );
    const observableObjects: HumanoidMotionOptionObservableObject[] = this.#objectMemory
      .observableObjectStates(observedFrame, observedWorldRevision)
      .map((object) => ({
        id: object.id,
        position: { ...object.pose.position },
        size: { ...object.size }
      }));
    return {
      snapshot,
      observableObjects,
      zones: this.#scenario.zones
    };
  }

  #motionOptionReceipt(
    stored: StoredMotionPlan,
    frames: number,
    failures: HumanoidMotionValidation["failures"] = []
  ): HumanoidExecutionReceipt {
    const option = stored.option;
    if (!option || !isTerminalMotionOption(option)) {
      throw new Error("Humanoid motion option has no terminal physical result");
    }
    const accepted = option.status === "succeeded";
    const code: HumanoidExecutionReceipt["code"] = accepted
      ? "motion_option_succeeded"
      : option.status === "failed"
        ? option.terminationReason === "execution_drift"
          ? "motion_execution_drifted"
          : option.terminationReason === "motion_constraint_violated"
            ? "motion_constraint_violated"
            : "motion_failed"
        : option.terminationReason === "motion_goal_uncertain"
          ? "motion_goal_uncertain"
          : "motion_goal_unmet";
    return this.#receipt(accepted, code, frames, {
      motion: humanoidMotionArtifactSummary(stored.artifact),
      ...(failures.length === 0 ? {} : { failures }),
      option: {
        option_id: option.contract.option_id,
        status: option.status,
        termination_reason: option.terminationReason,
        full_frame_count: stored.artifact.frames.length,
        executed_prefix_frame_count: stored.progress.nextFrameIndex,
        predicted_termination_frame: option.certificate.predicted_termination_frame,
        actual_termination_frame: option.actualTerminationFrame,
        artifact_sha256: option.certificate.artifact_sha256,
        rollout_sha256: option.certificate.rollout_sha256,
        drift_streak: stored.progress.driftStreak,
        drift_evidence: stored.progress.lastDrift
          ? { ...stored.progress.lastDrift }
          : null,
        evidence: option.lastEvidence
      }
    });
  }

  #receipt(
    accepted: boolean,
    code: HumanoidExecutionReceipt["code"],
    frames: number,
    detail: HumanoidExecutionReceipt["detail"]
  ): HumanoidExecutionReceipt {
    return {
      accepted,
      code,
      frames,
      finalSnapshot: this.snapshot(),
      detail
    };
  }
}

function navigationChunk(plan: NavigationPlan, maximumDistance: number): NavigationPlan {
  if (plan.distance <= maximumDistance || plan.waypoints.length < 2) {
    return structuredClone(plan);
  }
  const waypoints: Vec3[] = [{ ...plan.waypoints[0]! }];
  let distance = 0;
  for (let index = 1; index < plan.waypoints.length; index += 1) {
    const from = plan.waypoints[index - 1]!;
    const to = plan.waypoints[index]!;
    const segment = Math.hypot(to.x - from.x, to.z - from.z);
    if (distance + segment <= maximumDistance) {
      waypoints.push({ ...to });
      distance += segment;
      continue;
    }
    const remaining = maximumDistance - distance;
    const ratio = segment <= 1e-9 ? 0 : remaining / segment;
    const endpoint = {
      x: from.x + (to.x - from.x) * ratio,
      y: from.y + (to.y - from.y) * ratio,
      z: from.z + (to.z - from.z) * ratio
    };
    waypoints.push(endpoint);
    return {
      waypoints,
      distance: maximumDistance,
      resolvedTarget: { ...endpoint },
      projectionDistance: 0
    };
  }
  return structuredClone(plan);
}

function routeResult(
  completed: boolean,
  reason: string | undefined,
  frames: number,
  reference: HumanoidReference,
  start: HumanoidSimulationSnapshot,
  final: HumanoidSimulationSnapshot
): RouteRun {
  return {
    completed,
    ...(reason ? { reason } : {}),
    frames,
    reference,
    final,
    travelledDistance: Math.hypot(
      final.rootPosition.x - start.rootPosition.x,
      final.rootPosition.z - start.rootPosition.z
    )
  };
}

function yawFromQuaternion(rotation: HumanoidSimulationSnapshot["rootRotation"]): number {
  return Math.atan2(
    2 * (rotation.w * rotation.y + rotation.x * rotation.z),
    1 - 2 * (rotation.y * rotation.y + rotation.z * rotation.z)
  );
}

function normalizeAngle(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function point(value: Vec3): string {
  return `${value.x.toFixed(3)},${value.y.toFixed(3)},${value.z.toFixed(3)}`;
}

function validationFailure(
  failure: NonNullable<HumanoidMotionExecutionProgress["failure"]>
): HumanoidMotionValidation["failures"][number] {
  if (failure.code === "fallen") {
    return { code: "fallen", atSeconds: failure.atSeconds };
  }
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
    ...(failure.contacts
      ? { contacts: failure.contacts.map((contact) => ({ ...contact })) }
      : {})
  };
}

function isTerminalMotionOption(
  option: HumanoidMotionOptionExecutionState
): boolean {
  return option.status === "succeeded"
    || option.status === "failed"
    || option.status === "goal_unmet";
}

function assertMotionOptionIntegrity(stored: StoredMotionPlan): void {
  const option = stored.option;
  if (!option) return;
  if (!stored.rollout) {
    throw new Error("Humanoid motion option is missing its validated rollout");
  }
  const certificate = option.certificate;
  const predictedFrame = stored.artifact.frames[
    certificate.predicted_termination_frame - 1
  ];
  const valid = certificate.artifact_sha256
      === humanoidMotionArtifactSha256(stored.artifact)
    && certificate.contract_sha256
      === humanoidMotionOptionContractSha256(option.contract)
    && option.monitor.contractSha256 === certificate.contract_sha256
    && option.successStreak === option.monitor.terminalStableSteps
    && certificate.rollout_sha256
      === humanoidMotionRolloutSha256(stored.rollout)
    && certificate.rollout_frame_count === stored.rollout.frames.length
    && certificate.rollout_frame_count === stored.artifact.frames.length
    && certificate.drift_consecutive_steps
      === stored.rollout.limits.consecutive_steps
    && certificate.validated_frame_limit === stored.artifact.frames.length
    && certificate.predicted_termination_frame >= certificate.stable_steps
    && certificate.predicted_termination_frame <= certificate.validated_frame_limit
    && certificate.stable_steps === option.contract.stable_steps
    && predictedFrame !== undefined
    && Math.abs(predictedFrame.atSeconds - certificate.predicted_at_seconds) <= 1e-9;
  if (!valid) {
    throw new Error("Humanoid motion option certificate integrity check failed");
  }
}

function jsonOptionEvidence(
  detection: HumanoidMotionOptionDetection,
  monitor: HumanoidMotionOptionExecutionState["monitor"]
): HumanoidMotionOptionExecutionState["lastEvidence"] {
  return JSON.parse(JSON.stringify({
    predicates: detection.evidence,
    phases: detection.phases,
    monitor
  })) as HumanoidMotionOptionExecutionState[
    "lastEvidence"
  ];
}
