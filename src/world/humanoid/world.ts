import type { Scenario, Vec3 } from "../../domain/schema.js";
import type { ScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
import {
  NavigationPlanningError,
  type NavigationObstacle,
  type NavigationPlan
} from "../navigation.js";
import {
  HumanoidWorldCheckpointSchema,
  type HumanoidWorldCheckpoint
} from "./checkpoint.js";
import {
  HumanoidMotionCandidateBatchSchema,
  HumanoidMotionPlanSchema,
  humanoidGraspContactAuthorizationFailures,
  occupiedHumanoidChannels,
  prepareHumanoidMotion,
  TaskSpaceHumanoidMotionGenerator,
  type HumanoidMotionPlan,
  type HumanoidMotionCandidateBatch,
  type HumanoidBodyChannel,
  type HumanoidMotionGenerator,
  type HumanoidMotionValidationOptions,
  type HumanoidMotionValidation,
  type PreparedHumanoidMotion
} from "./motion-plan.js";
import {
  humanoidMotionArtifactSummary,
  humanoidMotionArtifactSha256,
  hydrateHumanoidReference,
  serializeHumanoidReference
} from "./motion-artifact.js";
import {
  createHumanoidMotionOptionMonitorState,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionDetectorInput,
  type HumanoidMotionOptionObservableObject
} from "./motion-option.js";
import {
  HumanoidObjectMemory,
  type HumanoidObjectMemoryCheckpoint
} from "./object-memory.js";
import { humanoidObjectCapability } from "./object-capability.js";
import {
  neutralHumanoidReference,
  releaseReferenceTracking,
  stationaryHumanoidReference,
  type HumanoidReference
} from "./reference.js";
import {
  HumanoidSimulation,
  type HumanoidSimulationState,
  type HumanoidSimulationSnapshot
} from "./simulation.js";
import { HumanoidRolloutSimulationPool } from "./rollout-simulation-pool.js";
import { HumanoidNavigationPlanner } from "./navigation-planner.js";
import { createHumanoidControlStepPacer } from "./control-step-pacer.js";
import {
  HumanoidAuthorityAdmissionError,
  HumanoidAuthorityLoop,
  type HumanoidAuthorityCommandHandle
} from "./authority-loop.js";
import {
  humanoidAuthorityStateSha256,
  type HumanoidAuthorityIdentity
} from "./authority-state.js";
import { HumanoidMotionExecution } from "./motion-execution.js";
import {
  HumanoidNavigationExecution,
  carryNavigationFailure,
  previewHumanoidNavigation
} from "./navigation-execution.js";
import { boundedNavigationChunk } from "./navigation-plan.js";
import {
  captureHumanoidCarryTaskSpaceTargets,
  type HumanoidCarryTaskSpaceTarget
} from "./carry-task-space-servo.js";
import { humanoidMotionContactEvidenceSha256 } from "./motion-contact-evidence.js";
import {
  HumanoidGraspRegistry,
  type HumanoidGraspRegistryCheckpoint
} from "./grasp-registry.js";
import {
  humanoidCarriedObjectBindingSetSha256,
  humanoidCarriedObjectContactConstraints,
  humanoidCarriedObjectContinuationEvidence,
  humanoidCarriedObjectUnauthorizedContacts,
  type HumanoidCarriedObjectBindingSet
} from "./carried-object-binding.js";
import type {
  StoredHumanoidMotionPlan,
  StoredHumanoidNavigationPlan
} from "./world-plan-state.js";
import {
  assertMotionOptionIntegrity,
  isTerminalMotionOption,
  storedMotionPhysicalSafety
} from "./world-plan-state.js";
import { restoreHumanoidWorldPlans } from "./world-plan-restore.js";
import {
  createHumanoidPlanTerminal,
  humanoidPlanTerminalReceipt
} from "./execution-terminal.js";
import {
  DEFAULT_HUMANOID_PLAN_INTENT_LEASE_SECONDS,
  humanoidMotionIntentSha256,
  humanoidNavigationIntentSha256,
  humanoidPlanExpiryRevision,
  humanoidPlanIntentIsActive
} from "./plan-lifecycle.js";
import type {
  HumanoidExecutionReceipt,
  HumanoidExecutionOptions,
  HumanoidFrameSink,
  HumanoidWorldPersistenceState,
  HumanoidWorldObservation,
  HumanoidWorldOptions,
  HumanoidWorldScenarioSynchronizationReceipt,
  HumanoidWorldSnapshot,
  NavigationPlanReceipt,
  WholeBodyCandidatePlanReceipt,
  WholeBodyPlanReceipt
} from "./world-contract.js";
import {
  probeHumanoidManipulationReachability,
  type HumanoidManipulationReachabilityMap
} from "./manipulation-reachability.js";
import {
  contactAwareG1GraspTargetsForBindings,
  contactAwareG1GraspTargetsForOption,
  mergeG1ContactAwareGraspTargets
} from "./contact-aware-grasp-servo.js";
import { HumanoidCarriedObjectLifecycle } from "./carried-object-lifecycle.js";
import {
  authorizeHumanoidCarriedObjectRelease,
} from "./carried-object-release.js";
import { carryMotionFailure } from "./carried-object-execution.js";
import {
  analyzeHumanoidScenarioSynchronization
} from "./scenario-synchronization.js";
import {
  createHumanoidPhysicsResources,
  createHumanoidWorldResources,
  disposeHumanoidPhysicsResources,
  disposeHumanoidWorldResources
} from "./world-resources.js";
import {
  humanoidPhysicalRegion,
  type HumanoidPhysicalRegion
} from "./physical-region.js";
import {
  observableHumanoidSolidIds,
  visibleHumanoidSolidTokens
} from "./solid-observation.js";
import { humanoidDynamicNavigationObstacles } from "./navigation-obstacles.js";
import { createHumanoidInteractionObservation } from "./interaction-observation.js";
import { g1HandCoordinationFromJointTargets } from "./hand-coordination.js";
import {
  captureHumanoidStationKeepingAnchor,
  stationKeepingHumanoidReference,
  type HumanoidStationKeepingAnchor
} from "./station-keeping.js";
import {
  HumanoidSpatialBeliefMap,
  type HumanoidSpatialBeliefMapCheckpoint
} from "./spatial-belief-map.js";
import { onlineNavigationReplanDecision } from "./online-navigation-replanner.js";
import type {
  HumanoidWholeBodyControllerFactory
} from "./whole-body-controller.js";

export interface WholeBodyMotionPlanningOptions {
  retainTerminalJointTracking?: boolean;
}
import type { HumanoidNavigationArrivalHeading } from "./navigation-arrival.js";

export type {
  HumanoidExecutionReceipt,
  HumanoidExecutionOptions,
  HumanoidFrameSink,
  HumanoidPersistenceSink,
  HumanoidWorldPersistenceState,
  HumanoidWorldObservation,
  HumanoidWorldOptions,
  HumanoidWorldScenarioSynchronizationReceipt,
  HumanoidWorldSnapshot,
  NavigationPlanReceipt,
  WholeBodyCandidatePlanReceipt,
  WholeBodyPlanReceipt
} from "./world-contract.js";

type PlanRevalidationEvidence = NonNullable<
  HumanoidExecutionReceipt["detail"]["revalidation"]
>;

interface MotionPlanRevalidation {
  accepted: boolean;
  evidence: PlanRevalidationEvidence;
  validation?: HumanoidMotionValidation;
  reason?: string;
}

interface MotionPlanningContext {
  frame: number;
  worldRevision: number;
  stateSha256: string;
  simulation: HumanoidSimulationState;
  baseline: HumanoidReference;
  visibleContactObjectIds: ReadonlySet<string>;
  visibleContactSolidIds: ReadonlySet<string>;
  graspRegistry: HumanoidGraspRegistryCheckpoint;
  carriedObjectBindings: HumanoidCarriedObjectBindingSet;
  carriedObjectTaskSpaceTargets: HumanoidCarryTaskSpaceTarget[];
}

interface NavigationIntentValidation {
  accepted: boolean;
  start: Vec3;
  plan: NavigationPlan | null;
  projectedTarget?: Vec3;
  projectionDistance?: number;
  remainingDistance: number;
  arrivalHeading: HumanoidNavigationArrivalHeading | null;
  releaseJointTracking: boolean;
  partialEndpoint?: Vec3;
  previewFrames?: number;
  previewTravelledDistance?: number;
  reason?: string;
}

interface NavigationPlanningContext {
  frame: number;
  worldRevision: number;
  stateSha256: string;
  start: Vec3;
  simulation: HumanoidSimulationState;
  baseline: HumanoidReference;
  obstacles: NavigationObstacle[];
  graspRegistry: HumanoidGraspRegistryCheckpoint;
  carriedObjectBindings: HumanoidCarriedObjectBindingSet;
  carriedObjectTaskSpaceTargets: HumanoidCarryTaskSpaceTarget[];
}

const NAVIGATION_CHUNK_DISTANCE = 3;
const ACTIVE_OBJECT_LINEAR_SPEED_METERS_PER_SECOND = 0.03;
const ACTIVE_OBJECT_ANGULAR_SPEED_RADIANS_PER_SECOND = 0.1;

export class HumanoidWorld {
  #scenario: Scenario;
  #simulation: HumanoidSimulation;
  #rolloutPool: HumanoidRolloutSimulationPool;
  #navigation: HumanoidNavigationPlanner;
  #physicalRegion: HumanoidPhysicalRegion;
  #objectMemory: HumanoidObjectMemory;
  #spatialBelief: HumanoidSpatialBeliefMap;
  readonly #graspRegistry: HumanoidGraspRegistry;
  readonly #motionGenerator: HumanoidMotionGenerator;
  readonly #controllerFactory: HumanoidWholeBodyControllerFactory | undefined;
  readonly #planIntentLeaseSeconds: number;
  readonly #motions = new Map<string, StoredHumanoidMotionPlan>();
  readonly #routes = new Map<string, StoredHumanoidNavigationPlan>();
  readonly #authority: HumanoidAuthorityLoop<HumanoidWorldSnapshot>;
  #carriedObjectLifecycle: HumanoidCarriedObjectLifecycle | null = null;
  #reference = neutralHumanoidReference();
  #stationKeepingAnchor: HumanoidStationKeepingAnchor | null = null;
  #frame = 0;
  #worldRevision = 0;
  #routeSequence = 0;
  #planRegistryEpoch = 0;
  #physicalSafety: HumanoidWorldSnapshot["physicalSafety"];
  #manipulationReachabilityCache: {
    worldRevision: number;
    evidence: HumanoidManipulationReachabilityMap;
  } | null = null;
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
    const activeScenario = options.scenarioChunks
      ? analyzeHumanoidScenarioSynchronization({
          current: scenario,
          baseline: scenario,
          chunks: options.scenarioChunks
        }).scenario
      : scenario;
    let resources: Awaited<ReturnType<typeof createHumanoidWorldResources>> | undefined;
    let motionGenerator: HumanoidMotionGenerator | undefined;
    try {
      resources = await createHumanoidWorldResources(
        activeScenario,
        checkpoint
          ? [checkpointRootAnchor(checkpoint)]
          : undefined,
        options.controllerFactory
      );
      motionGenerator = options.motionGeneratorFactory
        ? await options.motionGeneratorFactory()
        : new TaskSpaceHumanoidMotionGenerator();
      const world = new HumanoidWorld(
        activeScenario,
        resources.simulation,
        resources.rolloutPool,
        resources.navigation,
        resources.physicalRegion,
        motionGenerator,
        options.controllerFactory,
        options.planIntentLeaseSeconds
          ?? DEFAULT_HUMANOID_PLAN_INTENT_LEASE_SECONDS,
        checkpoint?.objectMemory,
        checkpoint?.graspRegistry,
        checkpoint?.spatialBelief
      );
      if (checkpoint) world.#restore(checkpoint);
      else await world.#settle(80);
      await world.#ensurePhysicalRegion();
      return world;
    } catch (error) {
      await Promise.allSettled([
        ...(resources ? [disposeHumanoidWorldResources(resources)] : []),
        ...(motionGenerator ? [motionGenerator.dispose()] : [])
      ]);
      throw error;
    }
  }

  private constructor(
    scenario: Scenario,
    simulation: HumanoidSimulation,
    rolloutPool: HumanoidRolloutSimulationPool,
    navigation: HumanoidNavigationPlanner,
    physicalRegion: HumanoidPhysicalRegion,
    motionGenerator: HumanoidMotionGenerator,
    controllerFactory: HumanoidWholeBodyControllerFactory | undefined,
    planIntentLeaseSeconds: number,
    objectMemoryCheckpoint?: HumanoidObjectMemoryCheckpoint,
    graspRegistryCheckpoint?: HumanoidGraspRegistryCheckpoint,
    spatialBeliefCheckpoint?: HumanoidSpatialBeliefMapCheckpoint
  ) {
    this.#scenario = scenario;
    this.#simulation = simulation;
    this.#rolloutPool = rolloutPool;
    this.#navigation = navigation;
    this.#physicalRegion = physicalRegion;
    this.#motionGenerator = motionGenerator;
    this.#controllerFactory = controllerFactory;
    this.#planIntentLeaseSeconds = planIntentLeaseSeconds;
    this.#objectMemory = new HumanoidObjectMemory(scenario, objectMemoryCheckpoint);
    this.#spatialBelief = new HumanoidSpatialBeliefMap(
      scenario,
      spatialBeliefCheckpoint
    );
    this.#graspRegistry = new HumanoidGraspRegistry({
      portableObjectIds: scenario.objects
        .filter((object) => object.portable)
        .map((object) => object.id),
      ...(graspRegistryCheckpoint
        ? { checkpoint: graspRegistryCheckpoint }
        : {})
    });
    this.#planExpiryRevision(0);
    this.#authority = new HumanoidAuthorityLoop({
      identity: () => this.#authorityIdentity(),
      stationaryStep: () => this.#stepStationaryAuthority()
    });
  }

  snapshot(): HumanoidWorldSnapshot {
    return {
      frame: this.#frame,
      worldRevision: this.#worldRevision,
      motionGenerator: structuredClone(this.#motionGenerator.descriptor),
      ...(this.#physicalSafety
        ? { physicalSafety: structuredClone(this.#physicalSafety) }
        : {}),
      robot: this.#simulation.snapshot(),
      grasp: this.#currentGraspState(),
      navigation: structuredClone(this.#navigationState)
    };
  }

  observe(): HumanoidWorldObservation {
    const snapshot = this.snapshot();
    const sensed = this.#refreshCurrentObjectMemory();
    const sensedSolids = this.#simulation.senseSolids(this.#scenario.visibility_radius);
    const objectTokens = this.#objectMemory.tokens(
      snapshot.robot,
      snapshot.frame,
      snapshot.worldRevision
    );
    const handSurfaces = this.#simulation.handSurfaceObservations(snapshot.robot);
    const solidTokens = visibleHumanoidSolidTokens({
      scenario: this.#scenario,
      sensed: sensedSolids,
      contacts: snapshot.robot.contacts
    });
    this.#spatialBelief.observe({
      frame: snapshot.frame,
      rootPosition: snapshot.robot.rootPosition,
      sensor: sensed.sensor,
      visibleSolids: solidTokens
    });
    const { objects: _objects, ...robot } = snapshot.robot;
    return {
      frame: snapshot.frame,
      worldRevision: snapshot.worldRevision,
      motionGenerator: structuredClone(snapshot.motionGenerator),
      sensor: sensed.sensor,
      robot,
      handCoordination: g1HandCoordinationFromJointTargets(
        Object.fromEntries(Object.entries(snapshot.robot.hands.joints).map(([name, joint]) => (
          [name, joint.target]
        ))) as Record<keyof typeof snapshot.robot.hands.joints, number>
      ),
      handSurfaces,
      manipulationReachability: [],
      manipulationBasePlacements: [],
      objectTokens,
      solidTokens,
      spatialBelief: this.#spatialBelief.observation(snapshot.robot.rootPosition),
      grasp: structuredClone(snapshot.grasp),
      interaction: createHumanoidInteractionObservation({
        frame: snapshot.frame,
        worldRevision: snapshot.worldRevision,
        scenario: this.#scenario,
        robot: snapshot.robot,
        objectTokens,
        solidTokens,
        grasp: snapshot.grasp,
        graspContract: this.#graspRegistry.contract,
        carried: this.#requiredCarriedObjectLifecycle().checkpoint()
      }),
      navigation: structuredClone(snapshot.navigation)
    };
  }

  async observeManipulationReachability(): Promise<HumanoidWorldObservation> {
    const observation = this.observe();
    const cached = this.#manipulationReachabilityCache;
    if (cached?.worldRevision === observation.worldRevision) {
      observation.manipulationReachability = structuredClone(
        cached.evidence.alignments
      );
      observation.manipulationBasePlacements = structuredClone(
        cached.evidence.basePlacements
      );
      return observation;
    }
    const authoritativeState = this.#simulation.captureState();
    const evidence = await this.#rolloutPool.lease(
      authoritativeState,
      (simulation) => probeHumanoidManipulationReachability({
        simulation,
        authoritativeState,
        reference: this.#reference,
        robot: this.#simulation.snapshot(),
        objectTokens: observation.objectTokens,
        handSurfaces: observation.handSurfaces,
        interactionTargets: manipulationInteractionTargets(observation)
      })
    );
    if (this.#worldRevision !== observation.worldRevision) {
      throw new Error(
        "Humanoid world changed while manipulation reachability was being observed"
      );
    }
    this.#manipulationReachabilityCache = {
      worldRevision: observation.worldRevision,
      evidence: structuredClone(evidence)
    };
    observation.manipulationReachability = evidence.alignments;
    observation.manipulationBasePlacements = evidence.basePlacements;
    return observation;
  }

  checkpoint(): HumanoidWorldCheckpoint {
    const simulation = this.#simulation.captureState();
    return HumanoidWorldCheckpointSchema.parse({
      version: 1,
      motionGenerator: structuredClone(this.#motionGenerator.descriptor),
      frame: this.#frame,
      worldRevision: this.#worldRevision,
      routeSequence: this.#routeSequence,
      planRegistryEpoch: this.#planRegistryEpoch,
      simulation: {
        time: simulation.time,
        positions: [...simulation.positions],
        velocities: [...simulation.velocities],
        controls: [...simulation.controls],
        activations: [...simulation.activations],
        accelerationWarmstart: [...simulation.accelerationWarmstart],
        ...(simulation.requestedActuatorTorques
          ? { requestedActuatorTorques: [...simulation.requestedActuatorTorques] }
          : {}),
        ...(simulation.handCommandTargets
          ? { handCommandTargets: [...simulation.handCommandTargets] }
          : {}),
        controller: structuredClone(simulation.controller)
      },
      reference: serializeHumanoidReference(this.#reference),
      stationKeeping: this.#stationKeepingAnchor
        ? structuredClone(this.#stationKeepingAnchor)
        : null,
      motions: [...this.#motions.values()].map((entry) => structuredClone(entry)),
      routes: [...this.#routes.values()].map((entry) => structuredClone(entry)),
      navigation: structuredClone(this.#navigationState),
      graspRegistry: this.#currentGraspRegistryCheckpoint(),
      carriedObjectLifecycle: this.#requiredCarriedObjectLifecycle().checkpoint(),
      ...(this.#physicalSafety
        ? { physicalSafety: structuredClone(this.#physicalSafety) }
        : {}),
      objectMemory: this.#objectMemory.checkpoint(),
      spatialBelief: this.#spatialBelief.checkpoint()
    });
  }

  capturePersistenceState(): Promise<HumanoidWorldPersistenceState> {
    return this.#authority.capture(() => this.#capturePersistenceState());
  }

  flushFramePublications(): Promise<void> {
    return this.#authority.flushPublications();
  }

  async synchronizeScenarioChunks(
    baseline: Scenario,
    chunks: ScenarioChunkDeltaState
  ): Promise<HumanoidWorldScenarioSynchronizationReceipt> {
    const analysis = analyzeHumanoidScenarioSynchronization({
      current: this.#scenario,
      baseline,
      chunks
    });
    if (!analysis.changed) {
      return {
        changed: false,
        chunkRevision: analysis.chunkRevision,
        resourceRebuilt: false,
        changedDomains: [],
        invalidatedPlanIds: []
      };
    }

    const captured = await this.#authority.capture(() => {
      if (this.#authority.busy) {
        throw new Error("Scenario synchronization requires an idle physical authority loop");
      }
      return {
        identity: this.#authorityIdentity(),
        simulation: this.#simulation.captureState(),
        physicalAnchors: this.#physicalAnchors(),
        objectMemory: this.#objectMemory.checkpoint(),
        spatialBelief: this.#spatialBelief.checkpoint()
      };
    });
    const nextObjectMemory = new HumanoidObjectMemory(
      analysis.scenario,
      captured.objectMemory
    );
    const nextSpatialBelief = new HumanoidSpatialBeliefMap(
      analysis.scenario,
      captured.spatialBelief
    );
    const nextResources = analysis.requiresResourceRebuild
      ? await createHumanoidWorldResources(
          analysis.scenario,
          captured.physicalAnchors,
          this.#controllerFactory
        )
      : undefined;
    let resourcesInstalled = false;
    try {
      nextResources?.simulation.restoreState(captured.simulation);
      const previousResources = await this.#authority.capture(() => {
        const actual = this.#authorityIdentity();
        if (actual.revision !== captured.identity.revision
          || actual.stateSha256 !== captured.identity.stateSha256) {
          throw new Error("Physical state changed while scenario resources were rebuilding");
        }
        const previous = nextResources
          ? {
              simulation: this.#simulation,
              rolloutPool: this.#rolloutPool,
              navigation: this.#navigation,
              physicalRegion: this.#physicalRegion
            }
          : undefined;
        const invalidatedPlanIds = [
          ...this.#motions.keys(),
          ...this.#routes.keys()
        ].sort();
        this.#scenario = analysis.scenario;
        this.#objectMemory = nextObjectMemory;
        this.#spatialBelief = nextSpatialBelief;
        if (nextResources) {
          this.#simulation = nextResources.simulation;
          this.#rolloutPool = nextResources.rolloutPool;
          this.#navigation = nextResources.navigation;
          this.#physicalRegion = nextResources.physicalRegion;
        }
        this.#motions.clear();
        this.#routes.clear();
        this.#planRegistryEpoch += 1;
        this.#frame += 1;
        this.#worldRevision += 1;
        this.#physicalSafety = undefined;
        this.#navigationState = {
          planId: null,
          status: "idle",
          target: null,
          waypoints: [],
          waypointIndex: null
        };
        const synchronizedSnapshot = this.#simulation.snapshot();
        this.#observeGraspFrame(this.#frame, synchronizedSnapshot);
        this.#observeCarriedObjectFrame(synchronizedSnapshot);
        this.#refreshCurrentObjectMemory();
        return { previous, invalidatedPlanIds };
      });
      resourcesInstalled = nextResources !== undefined;
      if (previousResources.previous) {
        await disposeHumanoidWorldResources(previousResources.previous);
      }
      return {
        changed: true,
        chunkRevision: analysis.chunkRevision,
        resourceRebuilt: analysis.requiresResourceRebuild,
        changedDomains: [...analysis.changedDomains],
        invalidatedPlanIds: previousResources.invalidatedPlanIds
      };
    } catch (error) {
      if (nextResources && !resourcesInstalled) {
        await disposeHumanoidWorldResources(nextResources);
      }
      throw error;
    }
  }

  #capturePersistenceState(): HumanoidWorldPersistenceState {
    return {
      world: this.snapshot(),
      worldCheckpoint: this.checkpoint(),
      authority: this.#authorityIdentity()
    };
  }

  consumablePlanIds(): string[] {
    return [
      ...[...this.#motions.entries()]
        .filter(([, stored]) => stored.terminal !== null
          || humanoidPlanIntentIsActive(
            this.#worldRevision,
            stored.expiresRevision
          ))
        .map(([planId]) => planId),
      ...[...this.#routes.entries()]
        .filter(([, stored]) => stored.terminal !== null
          || stored.progress.committed_frame_count > 0
          || humanoidPlanIntentIsActive(
            this.#worldRevision,
            stored.expiresRevision
          ))
        .map(([planId]) => planId)
    ];
  }

  async planWholeBodyMotion(
    rawPlan: HumanoidMotionPlan,
    options: WholeBodyMotionPlanningOptions = {}
  ): Promise<WholeBodyPlanReceipt> {
    const plan = HumanoidMotionPlanSchema.parse(rawPlan);
    const context = await this.#authority.capture(() => {
      if (this.#motions.has(plan.id)) {
        throw new Error(`Duplicate humanoid motion plan: ${plan.id}`);
      }
      return this.#captureMotionPlanningContext();
    });
    const createdRevision = context.worldRevision;
    const expiresRevision = this.#planExpiryRevision(createdRevision);
    const intentSha256 = humanoidMotionIntentSha256(plan);
    const prepared = await this.#prepareMotionIntent(
      plan,
      {
        contactObjectIds: context.visibleContactObjectIds,
        contactSolidIds: context.visibleContactSolidIds
      },
      context
    );
    await this.#authority.capture(() => {
      if (!humanoidPlanIntentIsActive(this.#worldRevision, expiresRevision)) {
        throw new Error("Humanoid motion planning exceeded its intent lease");
      }
      if (this.#motions.has(plan.id)) {
        throw new Error(`Duplicate humanoid motion plan: ${plan.id}`);
      }
      if (prepared.validation.feasible && prepared.artifact) {
        const stored: StoredHumanoidMotionPlan = {
          plan: structuredClone(plan),
          artifact: structuredClone(prepared.artifact),
          rollout: null,
          retainTerminalJointTracking:
            options.retainTerminalJointTracking === true,
          createdRevision,
          validatedRevision: createdRevision,
          validatedStateSha256: context.stateSha256,
          expiresRevision,
          intentSha256,
          revalidationCount: 0,
          terminal: null,
          option: null,
          carriedObjectBindings: structuredClone(context.carriedObjectBindings),
          carriedObjectTaskSpaceTargets: structuredClone(
            context.carriedObjectTaskSpaceTargets
          ),
          carriedObjectContinuation: null,
          carriedObjectUnauthorizedContacts: [],
          progress: {
            nextFrameIndex: 0,
            satisfiedContactKeys: [],
            satisfiedContactEvidenceSha256: humanoidMotionContactEvidenceSha256({
              planId: plan.id,
              intentSha256,
              artifactSha256: humanoidMotionArtifactSha256(prepared.artifact),
              nextFrameIndex: 0,
              satisfiedContactKeys: []
            }),
            driftStreak: 0,
            lastDrift: null,
            failure: null
          }
        };
        this.#motions.set(plan.id, stored);
        this.#planRegistryEpoch += 1;
        if (this.#worldRevision === createdRevision) {
          stored.validatedStateSha256 = this.#authorityStateSha256();
        }
      }
    });
    return {
      accepted: prepared.validation.feasible,
      planId: plan.id,
      createdRevision,
      validatedStateSha256: context.stateSha256,
      expiresRevision,
      intentSha256,
      channels: occupiedHumanoidChannels(plan),
      motion: prepared.artifact
        ? humanoidMotionArtifactSummary(prepared.artifact)
        : null,
      validation: prepared.validation
    };
  }

  async planWholeBodyMotionCandidates(
    rawBatch: HumanoidMotionCandidateBatch,
    options: WholeBodyMotionPlanningOptions = {}
  ): Promise<WholeBodyCandidatePlanReceipt> {
    const batch = HumanoidMotionCandidateBatchSchema.parse(rawBatch);
    this.#graspRegistry.bindingsForOption(batch.termination, this.#frame);
    const graspAuthorizationFailures = humanoidGraspContactAuthorizationFailures(
      batch,
      (contractSha256) => {
        if (contractSha256 !== this.#graspRegistry.contractSha256) {
          throw new Error(
            "Humanoid grasp predicate contract hash does not match authority"
          );
        }
        return this.#graspRegistry.contract.minimum_distinct_contact_links;
      }
    );
    if (graspAuthorizationFailures.length > 0) {
      const failure = graspAuthorizationFailures[0]!;
      throw new Error(
        `Humanoid grasp candidate ${failure.candidateIndex + 1} authorizes `
        + `${failure.authorizedContactSurfaces.length} contact surfaces for `
        + `${failure.objectId}/${failure.hand}; authority requires `
        + failure.minimumDistinctContactSurfaces
      );
    }
    const context = await this.#authority.capture(() => {
      for (const candidate of batch.candidates) {
        if (this.#motions.has(candidate.id)) {
          throw new Error(`Duplicate humanoid motion plan: ${candidate.id}`);
        }
      }
      return this.#captureMotionPlanningContext();
    });
    const createdRevision = context.worldRevision;
    const expiresRevision = this.#planExpiryRevision(createdRevision);
    const visibleContactObjects = context.visibleContactObjectIds;
    const visibleContactSolids = context.visibleContactSolidIds;
    const prepared: Array<{
      rank: number;
      plan: HumanoidMotionPlan;
      channels: HumanoidBodyChannel[];
      result: PreparedHumanoidMotion;
    }> = [];
    for (let index = 0; index < batch.candidates.length; index += 1) {
      const plan = batch.candidates[index]!;
      const result = await this.#prepareMotionIntent(
        plan,
        {
          contactObjectIds: visibleContactObjects,
          contactSolidIds: visibleContactSolids,
          motionOption: {
            contract: batch.termination,
            scenario: this.#scenario
          }
        },
        context
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
    await this.#authority.capture(() => {
      if (!humanoidPlanIntentIsActive(this.#worldRevision, expiresRevision)) {
        throw new Error("Humanoid candidate planning exceeded its intent lease");
      }
      for (const candidate of batch.candidates) {
        if (this.#motions.has(candidate.id)) {
          throw new Error(`Duplicate humanoid motion plan: ${candidate.id}`);
        }
      }
      if (selected?.result.artifact
        && selected.result.rollout
        && selected.result.optionCertificate) {
        const stored: StoredHumanoidMotionPlan = {
        plan: structuredClone(selected.plan),
        artifact: structuredClone(selected.result.artifact),
        rollout: structuredClone(selected.result.rollout),
        retainTerminalJointTracking:
          options.retainTerminalJointTracking === true,
        createdRevision,
        validatedRevision: createdRevision,
        validatedStateSha256: context.stateSha256,
        expiresRevision,
        intentSha256: humanoidMotionIntentSha256(selected.plan),
          revalidationCount: 0,
          terminal: null,
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
        carriedObjectBindings: structuredClone(context.carriedObjectBindings),
        carriedObjectTaskSpaceTargets: structuredClone(
          context.carriedObjectTaskSpaceTargets
        ),
        carriedObjectContinuation: null,
        carriedObjectUnauthorizedContacts: [],
        progress: {
          nextFrameIndex: 0,
          satisfiedContactKeys: [],
          satisfiedContactEvidenceSha256: humanoidMotionContactEvidenceSha256({
            planId: selected.plan.id,
            intentSha256: humanoidMotionIntentSha256(selected.plan),
            artifactSha256: humanoidMotionArtifactSha256(
              selected.result.artifact
            ),
            nextFrameIndex: 0,
            satisfiedContactKeys: []
          }),
          driftStreak: 0,
          lastDrift: null,
          failure: null
        }
        };
        this.#motions.set(selected.plan.id, stored);
        this.#planRegistryEpoch += 1;
        if (this.#worldRevision === createdRevision) {
          stored.validatedStateSha256 = this.#authorityStateSha256();
        }
      }
    });
    return {
      accepted: selected !== undefined,
      planId: selected?.plan.id ?? "",
      selectedCandidateId: selected?.plan.id ?? null,
      selectedRank: selected?.rank ?? null,
      createdRevision,
      validatedStateSha256: context.stateSha256,
      expiresRevision,
      intentSha256: selected ? humanoidMotionIntentSha256(selected.plan) : null,
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
        intentSha256: humanoidMotionIntentSha256(candidate.plan),
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
    frameSink?: HumanoidFrameSink,
    options: HumanoidExecutionOptions = {}
  ): Promise<HumanoidExecutionReceipt> {
    const initial = await this.#authority.capture(() => {
      const stored = this.#motions.get(planId);
      return {
        stored,
        terminalReceipt: stored?.terminal
          ? humanoidPlanTerminalReceipt(stored.terminal, this.snapshot())
          : null
      };
    });
    const stored = initial.stored;
    if (!stored) throw new Error(`Unknown humanoid motion plan: ${planId}`);
    if (initial.terminalReceipt) return initial.terminalReceipt;
    const revalidation = await this.#revalidateMotionIntent(stored);
    if (!revalidation.accepted) {
      const preconditionFailure = stored.option
        ? revalidation.validation?.failures.find((failure) => (
            failure.code === "motion_constraint_violated"
            && failure.atSeconds === 0
          ))
        : undefined;
      if (stored.option && preconditionFailure) {
        const terminalized = await this.#authority.capture(() => {
          if (this.#motions.get(planId) !== stored || !stored.option) return null;
          stored.option.status = "failed";
          stored.option.actualTerminationFrame = 0;
          stored.option.terminationReason = "motion_constraint_violated";
          const receipt = this.#motionOptionReceipt(
            stored,
            0,
            revalidation.validation?.failures ?? [],
            revalidation.evidence
          );
          return this.#finalizeMotionPlan(stored, receipt, options.retainTerminal ?? false);
        });
        if (terminalized) {
          return terminalized;
        }
      }
      return this.#authority.capture(() => {
        const receipt = this.#receipt(false, "plan_revalidation_failed", 0, {
          reason: revalidation.reason
            ?? "The unchanged model motion intent is no longer physically admissible",
          ...(revalidation.validation
            ? { failures: revalidation.validation.failures }
            : {}),
          revalidation: revalidation.evidence
        });
        return this.#finalizeMotionPlan(stored, receipt, options.retainTerminal ?? false);
      });
    }
    if (stored.option) assertMotionOptionIntegrity(stored);
    const admission: HumanoidAuthorityIdentity = {
      revision: stored.validatedRevision + stored.progress.nextFrameIndex,
      stateSha256: revalidation.evidence.admission_state_sha256
    };
    const evidence: PlanRevalidationEvidence = {
      ...revalidation.evidence,
      admission_state_sha256: admission.stateSha256
    };
    const releaseAuthority = stored.option
      ? authorizeHumanoidCarriedObjectRelease({
          contract: stored.option.contract,
          bindingSet: stored.carriedObjectBindings
        })
      : null;
    const releaseTrackedObjectIds = new Set(
      releaseAuthority?.bindings.map((binding) => binding.objectId) ?? []
    );
    let execution: HumanoidMotionExecution | undefined;
    let handle: HumanoidAuthorityCommandHandle<HumanoidExecutionReceipt>;
    try {
      handle = await this.#authority.submit({
        id: `motion:${planId}`,
        source: "motion",
        admission,
        ...(frameSink ? { frameSink } : {}),
        admit: () => {
          if (this.#motions.get(planId) !== stored) {
            throw new Error(`Humanoid motion plan became unavailable: ${planId}`);
          }
          this.#stationKeepingAnchor = null;
          const expectedRevision = stored.validatedRevision
            + stored.progress.nextFrameIndex;
          if (expectedRevision !== this.#worldRevision) {
            throw new Error(
              `Humanoid motion plan is stale: expected_revision=${expectedRevision}, `
              + `world_revision=${this.#worldRevision}`
            );
          }
          if (releaseAuthority) {
            const lifecycle = this.#requiredCarriedObjectLifecycle();
            if (lifecycle.phase !== "release_pending"
              && lifecycle.phase !== "released") {
              lifecycle.beginRelease({
                currentFrame: this.#frame,
                currentWorldRevision: this.#worldRevision
              });
            }
          }
          execution = new HumanoidMotionExecution({
            stored,
            reference: this.#reference,
            stationKeepingAnchor: captureHumanoidStationKeepingAnchor(
              this.#simulation.snapshot(),
              this.#frame,
              this.#worldRevision
            ),
            graspTargets: mergeG1ContactAwareGraspTargets(
              contactAwareG1GraspTargetsForBindings({
                bindings: stored.carriedObjectBindings.bindings,
                graspRegistry: this.#graspRegistry
              }),
              stored.option
                ? contactAwareG1GraspTargetsForOption({
                    option: stored.option.contract,
                    graspContract: this.#graspRegistry.contract
                  })
                : []
            ),
            carryTaskSpaceTargets: stored.carriedObjectTaskSpaceTargets,
            detectorInput: (snapshot) => this.#motionOptionDetectorInput(
              snapshot,
              stored.option!.contract,
              releaseTrackedObjectIds
            ),
            commitPhysicalFrame: (snapshot) => {
              this.#commitFrameState({ motionPlanId: planId });
              this.#observeGraspFrame(this.#frame, snapshot);
              const continuation = humanoidCarriedObjectContinuationEvidence({
                state: stored.carriedObjectBindings,
                registry: this.#graspRegistry,
                currentFrame: this.#frame,
                currentWorldRevision: this.#worldRevision
              });
              const unauthorized = humanoidCarriedObjectUnauthorizedContacts(
                stored.carriedObjectBindings,
                snapshot.contacts
              );
              stored.carriedObjectContinuation = continuation;
              stored.carriedObjectUnauthorizedContacts = unauthorized;
              this.#observeCarriedObjectFrame(snapshot);
              return carryMotionFailure({
                continuation,
                unauthorized,
                snapshot,
                releaseAuthority,
                lifecyclePhase: this.#requiredCarriedObjectLifecycle().phase
              });
            }
          });
        },
        step: async () => {
          options.signal?.throwIfAborted();
          if (!execution) throw new Error("Humanoid motion was not admitted");
          await this.#ensurePhysicalRegion();
          const step = await execution.step(this.#simulation);
          this.#reference = execution.reference;
          if (step.snapshot) {
            const physicalSafety = storedMotionPhysicalSafety(stored);
            if (physicalSafety) this.#physicalSafety = { planId, evidence: physicalSafety };
          }
          if (!step.done) {
            const snapshot = this.snapshot();
            await options.persistenceSink?.(this.#capturePersistenceState());
            return {
              ...(frameSink && !options.persistenceSink ? { snapshot } : {}),
              done: false
            };
          }
          const result = execution.result();
          if (releaseAuthority && stored.option?.status !== "succeeded") {
            this.#requiredCarriedObjectLifecycle().cancelRelease({
              currentFrame: this.#frame,
              currentWorldRevision: this.#worldRevision
            });
          }
          this.#acquireSuccessfulOptionGrasp(stored);
          const carrying = this.#requiredCarriedObjectLifecycle().active !== null;
          this.#reference = carrying
            ? stationaryHumanoidReference(result.reference)
            : result.failures.length === 0 && stored.retainTerminalJointTracking
              ? stationaryHumanoidReference(result.reference)
              : releaseReferenceTracking(result.reference);
          const physicalSafety = result.physicalSafety ?? storedMotionPhysicalSafety(stored);
          const receipt = stored.option
            ? this.#motionOptionReceipt(stored, result.frames, result.failures, evidence)
            : this.#receipt(
                result.failures.length === 0,
                result.failures.length === 0 ? "motion_completed" : "motion_failed",
                result.frames,
                {
                  motion: humanoidMotionArtifactSummary(stored.artifact),
                  ...(physicalSafety ? { physical_safety: physicalSafety } : {}),
                  ...(result.failures.length > 0 ? { failures: result.failures } : {}),
                  carry: {
                    binding_set: structuredClone(stored.carriedObjectBindings),
                    continuation: stored.carriedObjectContinuation
                      ? structuredClone(stored.carriedObjectContinuation)
                      : null,
                    unauthorized_contacts: structuredClone(
                      stored.carriedObjectUnauthorizedContacts
                    )
                  },
                  revalidation: evidence
                }
              );
          const finalized = this.#finalizeMotionPlan(
            stored,
            receipt,
            options.retainTerminal ?? false
          );
          if (step.snapshot) {
            await options.persistenceSink?.(this.#capturePersistenceState());
          }
          return {
            ...(step.snapshot && frameSink && !options.persistenceSink
              ? { snapshot: finalized.finalSnapshot }
              : {}),
            done: true,
            result: finalized
          };
        }
      });
    } catch (error) {
      if (!(error instanceof HumanoidAuthorityAdmissionError)) throw error;
      return this.#authority.capture(() => {
        const receipt = this.#receipt(false, "plan_stale", 0, {
          reason: error.message,
          revalidation: {
            ...evidence,
            accepted: false,
            admission_state_sha256: error.actual.stateSha256
          }
        });
        return this.#finalizeMotionPlan(stored, receipt, options.retainTerminal ?? false);
      });
    }
    return this.#driveAuthorityCommand(handle, options);
  }

  async acknowledgeWholeBodyMotion(
    planId: string,
    resultSha256: string
  ): Promise<boolean> {
    return this.#authority.capture(() => {
      const stored = this.#motions.get(planId);
      if (!stored) return false;
      if (!stored.terminal) {
        throw new Error(`Humanoid motion plan is not terminal: ${planId}`);
      }
      if (stored.terminal.result_sha256 !== resultSha256) {
        throw new Error(`Humanoid motion terminal acknowledgement mismatch: ${planId}`);
      }
      return this.#deleteMotionPlan(planId);
    });
  }

  async #prepareMotionIntent(
    plan: HumanoidMotionPlan,
    options: HumanoidMotionValidationOptions,
    context?: MotionPlanningContext
  ): Promise<PreparedHumanoidMotion> {
    const source = context ?? this.#captureMotionPlanningContext();
    const previewGraspRegistry = new HumanoidGraspRegistry({
      portableObjectIds: source.graspRegistry.portable_object_ids,
      contract: source.graspRegistry.contract,
      checkpoint: source.graspRegistry
    });
    return this.#rolloutPool.lease(source.simulation, (simulation) => (
      prepareHumanoidMotion(
        simulation,
        plan,
        source.baseline,
        {
          ...options,
          graspRegistry: previewGraspRegistry,
          worldFrame: source.frame,
          worldRevision: source.worldRevision,
          carriedObjectBindings: source.carriedObjectBindings,
          carriedObjectTaskSpaceTargets: source.carriedObjectTaskSpaceTargets
        },
        this.#motionGenerator
      )
    ));
  }

  #captureMotionPlanningContext(): MotionPlanningContext {
    const snapshot = this.#simulation.snapshot();
    const baseline = hydrateHumanoidReference(
      serializeHumanoidReference(this.#reference)
    );
    const observedContactObjectIds = new Set(this.#objectMemory
      .activeObjectStates()
      .filter((object) => object.observable)
      .map((object) => object.id));
    const currentlyVisibleObjectIds = new Set(Object.keys(
      this.#simulation.senseObjects(this.#scenario.visibility_radius).objects
    ));
    const visibleContactObjectIds = new Set(
      [...observedContactObjectIds].filter((id) => currentlyVisibleObjectIds.has(id))
    );
    const authorityContactObjectIds = this.#objectMemory.observedObjectIds(
      this.#frame,
      this.#worldRevision
    );
    const visibleContactSolidIds = new Set(observableHumanoidSolidIds(
      this.#simulation.senseSolids(this.#scenario.visibility_radius),
      snapshot.contacts
    ));
    const graspRegistry = this.#currentGraspRegistryCheckpoint();
    const carriedObjectBindings = this.#currentCarriedObjectBindings();
    const carriedObjectTaskSpaceTargets = captureHumanoidCarryTaskSpaceTargets({
      snapshot,
      bindings: carriedObjectBindings
    });
    const simulation = this.#simulation.captureState();
    return {
      frame: this.#frame,
      worldRevision: this.#worldRevision,
      stateSha256: humanoidAuthorityStateSha256({
        simulation,
        reference: baseline,
        visibleContactObjectIds: authorityContactObjectIds,
        visibleContactSolidIds,
        planRegistryEpoch: this.#planRegistryEpoch,
        graspRegistry,
        carriedObjectBindings
      }),
      simulation,
      baseline,
      visibleContactObjectIds,
      visibleContactSolidIds,
      graspRegistry,
      carriedObjectBindings,
      carriedObjectTaskSpaceTargets
    };
  }

  async #revalidateMotionIntent(
    stored: StoredHumanoidMotionPlan
  ): Promise<MotionPlanRevalidation> {
    if (stored.intentSha256 !== humanoidMotionIntentSha256(stored.plan)) {
      throw new Error("Humanoid motion intent changed after model planning");
    }
    const context = await this.#authority.capture(() => {
      return this.#captureMotionPlanningContext();
    });
    const previousValidationRevision = stored.validatedRevision;
    let validationRevision = stored.validatedRevision;
    let validationStateSha256 = stored.validatedStateSha256;
    let admissionStateSha256 = context.stateSha256;
    const evidence = (
      performed: boolean,
      accepted: boolean,
      revalidationCount: number
    ): PlanRevalidationEvidence => ({
      performed,
      accepted,
      intent_sha256: stored.intentSha256,
      planning_revision: stored.createdRevision,
      previous_validation_revision: previousValidationRevision,
      validation_revision: validationRevision,
      validation_state_sha256: validationStateSha256,
      admission_state_sha256: admissionStateSha256,
      expires_revision: stored.expiresRevision,
      revalidation_count: revalidationCount
    });
    if (this.#motions.get(stored.plan.id) !== stored) {
      return {
        accepted: false,
        reason: "humanoid_motion_plan_no_longer_registered",
        evidence: evidence(false, false, stored.revalidationCount)
      };
    }
    if (!humanoidPlanIntentIsActive(context.worldRevision, stored.expiresRevision)) {
      return {
        accepted: false,
        reason: `intent_expired_at_revision=${stored.expiresRevision}, world_revision=${context.worldRevision}`,
        evidence: evidence(false, false, stored.revalidationCount)
      };
    }
    const expectedRevision = stored.validatedRevision + stored.progress.nextFrameIndex;
    if (expectedRevision === context.worldRevision
      && (stored.progress.nextFrameIndex > 0
        || stored.validatedStateSha256 === context.stateSha256)) {
      return {
        accepted: true,
        evidence: evidence(false, true, stored.revalidationCount)
      };
    }
    if (stored.progress.nextFrameIndex > 0) {
      return {
        accepted: false,
        reason: `partially_executed_intent_expected_revision=${expectedRevision}, world_revision=${context.worldRevision}`,
        evidence: evidence(false, false, stored.revalidationCount)
      };
    }
    const nextRevalidationCount = stored.revalidationCount + 1;
    validationRevision = context.worldRevision;
    validationStateSha256 = context.stateSha256;
    const prepared = await this.#prepareMotionIntent(
      stored.plan,
      {
        contactObjectIds: context.visibleContactObjectIds,
        contactSolidIds: context.visibleContactSolidIds,
        ...(stored.option
          ? {
              motionOption: {
                contract: stored.option.contract,
                scenario: this.#scenario
              }
            }
          : {})
      },
      context
    );
    const optionReady = !stored.option
      || prepared.rollout !== null && prepared.optionCertificate !== null;
    if (!prepared.validation.feasible || !prepared.artifact || !optionReady) {
      return {
        accepted: false,
        reason: "unchanged_intent_failed_latest_state_validation",
        validation: prepared.validation,
        evidence: evidence(true, false, nextRevalidationCount)
      };
    }
    const applied = await this.#authority.capture(() => {
      const identity = this.#authorityIdentity();
      admissionStateSha256 = identity.stateSha256;
      if (this.#motions.get(stored.plan.id) !== stored
        || identity.revision !== context.worldRevision
        || identity.stateSha256 !== context.stateSha256) {
        return false;
      }
      stored.artifact = structuredClone(prepared.artifact!);
      stored.rollout = prepared.rollout ? structuredClone(prepared.rollout) : null;
      stored.validatedRevision = context.worldRevision;
      stored.validatedStateSha256 = context.stateSha256;
      stored.revalidationCount = nextRevalidationCount;
      stored.carriedObjectBindings = structuredClone(
        context.carriedObjectBindings
      );
      stored.carriedObjectTaskSpaceTargets = structuredClone(
        context.carriedObjectTaskSpaceTargets
      );
      stored.carriedObjectContinuation = null;
      stored.carriedObjectUnauthorizedContacts = [];
      stored.progress = {
        nextFrameIndex: 0,
        satisfiedContactKeys: [],
        satisfiedContactEvidenceSha256: humanoidMotionContactEvidenceSha256({
          planId: stored.plan.id,
          intentSha256: stored.intentSha256,
          artifactSha256: humanoidMotionArtifactSha256(prepared.artifact!),
          nextFrameIndex: 0,
          satisfiedContactKeys: []
        }),
        driftStreak: 0,
        lastDrift: null,
        failure: null
      };
      if (stored.option) {
        if (!prepared.optionCertificate) {
          throw new Error("Validated humanoid option is missing its certificate");
        }
        stored.option = {
          contract: structuredClone(stored.option.contract),
          certificate: structuredClone(prepared.optionCertificate),
          monitor: createHumanoidMotionOptionMonitorState(stored.option.contract),
          status: "planned",
          successStreak: 0,
          actualTerminationFrame: null,
          terminationReason: null,
          lastEvidence: null
        };
      }
      return true;
    });
    if (!applied) {
      return {
        accepted: false,
        reason: "world_state_changed_during_motion_revalidation",
        validation: prepared.validation,
        evidence: evidence(true, false, nextRevalidationCount)
      };
    }
    return {
      accepted: true,
      validation: prepared.validation,
      evidence: evidence(true, true, nextRevalidationCount)
    };
  }

  async #validateNavigationIntent(
    target: Vec3,
    requestedArrivalHeading: HumanoidNavigationArrivalHeading | null,
    context?: NavigationPlanningContext,
    acceptedPositionToleranceMeters: number | null = null
  ): Promise<NavigationIntentValidation> {
    const source = context ?? this.#captureNavigationPlanningContext();
    const start = { ...source.start };
    let completePlan: NavigationPlan;
    try {
      completePlan = await this.#navigation.plan(
        start,
        target,
        source.obstacles
      );
    } catch (error) {
      return {
        accepted: false,
        start,
        plan: null,
        ...(error instanceof NavigationPlanningError && error.projectedTarget
          ? { projectedTarget: error.projectedTarget }
          : {}),
        ...(error instanceof NavigationPlanningError
          && error.projectionDistance !== undefined
          ? { projectionDistance: error.projectionDistance }
          : {}),
        remainingDistance: 0,
        arrivalHeading: null,
        releaseJointTracking: false,
        reason: error instanceof Error ? error.message : String(error)
      };
    }
    const plan = boundedNavigationChunk(completePlan, NAVIGATION_CHUNK_DISTANCE);
    const remainingDistance = Math.max(0, completePlan.distance - plan.distance);
    const arrivalHeading = remainingDistance <= 1e-9
      ? requestedArrivalHeading
      : null;
    const previewWith = (baseline: HumanoidReference) => {
      const previewGraspRegistry = new HumanoidGraspRegistry({
        portableObjectIds: source.graspRegistry.portable_object_ids,
        contract: source.graspRegistry.contract,
        checkpoint: source.graspRegistry
      });
      return this.#rolloutPool.lease(
        source.simulation,
        (simulation) => previewHumanoidNavigation(
          plan,
          baseline,
          simulation,
          previewGraspRegistry,
          source.frame,
          source.worldRevision,
          source.carriedObjectBindings,
          source.carriedObjectTaskSpaceTargets,
          arrivalHeading,
          acceptedPositionToleranceMeters
        )
      );
    };
    let releaseJointTracking = false;
    let preview = await previewWith(source.baseline);
    if (!preview.completed
      && source.carriedObjectBindings.bindings.length === 0
      && source.baseline.jointTrackingWeights.some((weight) => weight > 0)) {
      const releasedPreview = await previewWith(
        releaseReferenceTracking(source.baseline)
      );
      if (releasedPreview.completed) {
        preview = releasedPreview;
        releaseJointTracking = true;
      }
    }
    return {
      accepted: preview.completed,
      start,
      plan,
      remainingDistance,
      arrivalHeading,
      releaseJointTracking,
      ...(!preview.completed
        ? {
            partialEndpoint: { ...preview.final.rootPosition },
            previewFrames: preview.frames,
            previewTravelledDistance: preview.travelledDistance
          }
        : {}),
      ...(preview.completed
        ? {}
        : {
            reason: `${preview.reason ?? "physical_preview_failed"}`
              + `; preview_frames=${preview.frames}`
              + `; preview_travelled_m=${preview.travelledDistance.toFixed(6)}`
          })
    };
  }

  #captureNavigationPlanningContext(): NavigationPlanningContext {
    const snapshot = this.#simulation.snapshot();
    const baseline = hydrateHumanoidReference(
      serializeHumanoidReference(this.#reference)
    );
    const graspRegistry = this.#currentGraspRegistryCheckpoint();
    const carriedObjectBindings = this.#currentCarriedObjectBindings();
    const carriedObjectIds = new Set(
      carriedObjectBindings.bindings.map((binding) => binding.object_id)
    );
    const visibleContactObjectIds = this.#objectMemory.observedObjectIds(
      this.#frame,
      this.#worldRevision
    );
    const visibleContactSolidIds = observableHumanoidSolidIds(
      this.#simulation.senseSolids(this.#scenario.visibility_radius),
      snapshot.contacts
    );
    const carriedObjectTaskSpaceTargets = captureHumanoidCarryTaskSpaceTargets({
      snapshot,
      bindings: carriedObjectBindings
    });
    const simulation = this.#simulation.captureState();
    return {
      frame: this.#frame,
      worldRevision: this.#worldRevision,
      stateSha256: humanoidAuthorityStateSha256({
        simulation,
        reference: baseline,
        visibleContactObjectIds,
        visibleContactSolidIds,
        planRegistryEpoch: this.#planRegistryEpoch,
        graspRegistry,
        carriedObjectBindings
      }),
      start: { ...snapshot.rootPosition },
      simulation,
      baseline,
      obstacles: humanoidDynamicNavigationObstacles({
        scenario: this.#scenario,
        objectSnapshots: snapshot.objects,
        excludedPortableObjectIds: carriedObjectIds
      }),
      graspRegistry,
      carriedObjectBindings,
      carriedObjectTaskSpaceTargets
    };
  }

  async planNavigation(
    target: Vec3,
    requestedArrivalHeading: HumanoidNavigationArrivalHeading | null = null,
    acceptedPositionToleranceMeters: number | null = null
  ): Promise<NavigationPlanReceipt> {
    const context = await this.#authority.capture(() => {
      return this.#captureNavigationPlanningContext();
    });
    const createdRevision = context.worldRevision;
    const expiresRevision = this.#planExpiryRevision(createdRevision);
    const intentSha256 = humanoidNavigationIntentSha256(
      target,
      requestedArrivalHeading,
      acceptedPositionToleranceMeters
    );
    const validation = await this.#validateNavigationIntent(
      target,
      requestedArrivalHeading,
      context,
      acceptedPositionToleranceMeters
    );
    if (!validation.accepted || !validation.plan) {
      return {
        accepted: false,
        planId: "",
        createdRevision,
        validatedStateSha256: context.stateSha256,
        expiresRevision,
        intentSha256,
        target: { ...target },
        requestedArrivalHeading: requestedArrivalHeading
          ? structuredClone(requestedArrivalHeading)
          : null,
        arrivalHeading: null,
        acceptedPositionToleranceMeters,
        chunkTarget: {
          ...(validation.plan?.resolvedTarget
            ?? validation.projectedTarget
            ?? validation.start)
        },
        waypoints: validation.plan?.waypoints.map((point) => ({ ...point })) ?? [],
        distance: validation.plan?.distance ?? 0,
        remainingDistance: validation.remainingDistance,
        ...(validation.partialEndpoint
          ? { partialEndpoint: { ...validation.partialEndpoint } }
          : {}),
        ...(validation.previewFrames === undefined
          ? {}
          : { previewFrames: validation.previewFrames }),
        ...(validation.previewTravelledDistance === undefined
          ? {}
          : { previewTravelledDistance: validation.previewTravelledDistance }),
        carry: navigationCarryReceipt(context.carriedObjectBindings),
        reason: validation.reason ?? "physical_preview_failed"
      };
    }
    const plan = validation.plan;
    let planId = "";
    await this.#authority.capture(() => {
      if (!humanoidPlanIntentIsActive(this.#worldRevision, expiresRevision)) {
        throw new Error("Humanoid navigation planning exceeded its intent lease");
      }
      planId = `humanoid-route-${this.#routeSequence++}`;
      const stored: StoredHumanoidNavigationPlan = {
        id: planId,
        plan,
        requestedTarget: { ...target },
        requestedArrivalHeading: requestedArrivalHeading
          ? structuredClone(requestedArrivalHeading)
          : null,
        arrivalHeading: validation.arrivalHeading
          ? structuredClone(validation.arrivalHeading)
          : null,
        acceptedPositionToleranceMeters,
        releaseJointTracking: validation.releaseJointTracking,
        createdRevision,
        validatedRevision: createdRevision,
        validatedStateSha256: context.stateSha256,
        expiresRevision,
        intentSha256,
        revalidationCount: 0,
        carriedObjectBindings: structuredClone(context.carriedObjectBindings),
        carriedObjectTaskSpaceTargets: structuredClone(
          context.carriedObjectTaskSpaceTargets
        ),
        carriedObjectContinuation: null,
        carriedObjectUnauthorizedContacts: [],
        progress: {
          version: 1,
          start_root_position: { ...context.start },
          waypoint_index: Math.min(1, plan.waypoints.length - 1),
          committed_frame_count: 0,
          stopping_frame_count: 0
        },
        terminal: null
      };
      this.#routes.set(planId, stored);
      this.#planRegistryEpoch += 1;
      if (this.#worldRevision === createdRevision) {
        stored.validatedStateSha256 = this.#authorityStateSha256();
      }
      this.#navigationState = {
        planId,
        status: "planned",
        target: { ...target },
        waypoints: plan.waypoints.map((point) => ({ ...point })),
        waypointIndex: 1
      };
    });
    return {
      accepted: true,
      planId,
      createdRevision,
      validatedStateSha256: context.stateSha256,
      expiresRevision,
      intentSha256,
      target: { ...target },
      chunkTarget: { ...plan.resolvedTarget },
      requestedArrivalHeading: requestedArrivalHeading
        ? structuredClone(requestedArrivalHeading)
        : null,
      arrivalHeading: validation.arrivalHeading
        ? structuredClone(validation.arrivalHeading)
        : null,
      acceptedPositionToleranceMeters,
      waypoints: plan.waypoints.map((point) => ({ ...point })),
      distance: plan.distance,
      remainingDistance: validation.remainingDistance,
      carry: navigationCarryReceipt(context.carriedObjectBindings)
    };
  }

  async executeNavigation(
    planId: string,
    frameSink?: HumanoidFrameSink,
    options: HumanoidExecutionOptions = {}
  ): Promise<HumanoidExecutionReceipt> {
    const captured = await this.#authority.capture(() => {
      const stored = this.#routes.get(planId);
      return {
        stored,
        context: this.#captureNavigationPlanningContext(),
        terminalReceipt: stored?.terminal
          ? humanoidPlanTerminalReceipt(stored.terminal, this.snapshot())
          : null
      };
    });
    const stored = captured.stored;
    if (!stored) throw new Error(`Unknown humanoid navigation plan: ${planId}`);
    if (captured.terminalReceipt) return captured.terminalReceipt;
    if (stored.intentSha256 !== humanoidNavigationIntentSha256(
      stored.requestedTarget,
      stored.requestedArrivalHeading,
      stored.acceptedPositionToleranceMeters
    )) {
      throw new Error("Humanoid navigation intent changed after model planning");
    }
    const previousValidationRevision = stored.validatedRevision;
    let validationRevision = stored.validatedRevision;
    let validationStateSha256 = stored.validatedStateSha256;
    const admissionStateSha256 = captured.context.stateSha256;
    const revalidationEvidence = (
      performed: boolean,
      accepted: boolean,
      revalidationCount: number
    ): PlanRevalidationEvidence => ({
      performed,
      accepted,
      intent_sha256: stored.intentSha256,
      planning_revision: stored.createdRevision,
      previous_validation_revision: previousValidationRevision,
      validation_revision: validationRevision,
      validation_state_sha256: validationStateSha256,
      admission_state_sha256: admissionStateSha256,
      expires_revision: stored.expiresRevision,
      revalidation_count: revalidationCount
    });
    const expectedRevision = stored.validatedRevision
      + stored.progress.committed_frame_count;
    if (stored.progress.committed_frame_count > 0
      && expectedRevision !== captured.context.worldRevision) {
      return this.#authority.capture(() => {
        const receipt = this.#receipt(false, "plan_stale", 0, {
          reason: `partially_executed_route_expected_revision=${expectedRevision}, world_revision=${captured.context.worldRevision}`,
          revalidation: revalidationEvidence(false, false, stored.revalidationCount)
        });
        return this.#finalizeNavigationPlan(
          stored,
          receipt,
          options.retainTerminal ?? false
        );
      });
    }
    if (stored.progress.committed_frame_count === 0
      && !humanoidPlanIntentIsActive(
        captured.context.worldRevision,
        stored.expiresRevision
      )) {
      return this.#authority.capture(() => {
        const receipt = this.#receipt(false, "plan_stale", 0, {
          reason: `intent_expired_at_revision=${stored.expiresRevision}, world_revision=${captured.context.worldRevision}`,
          revalidation: revalidationEvidence(false, false, stored.revalidationCount)
        });
        return this.#finalizeNavigationPlan(
          stored,
          receipt,
          options.retainTerminal ?? false
        );
      });
    }
    let revalidation = revalidationEvidence(false, true, stored.revalidationCount);
    if (stored.progress.committed_frame_count === 0
      && (stored.validatedRevision !== captured.context.worldRevision
      || stored.validatedStateSha256 !== captured.context.stateSha256)) {
      const validation = await this.#validateNavigationIntent(
        stored.requestedTarget,
        stored.requestedArrivalHeading,
        captured.context,
        stored.acceptedPositionToleranceMeters
      );
      const nextRevalidationCount = stored.revalidationCount + 1;
      validationRevision = captured.context.worldRevision;
      validationStateSha256 = captured.context.stateSha256;
      if (!validation.accepted || !validation.plan) {
        revalidation = revalidationEvidence(true, false, nextRevalidationCount);
        return this.#authority.capture(() => {
          const receipt = this.#receipt(false, "plan_revalidation_failed", 0, {
            reason: validation.reason
              ?? "The unchanged navigation target is no longer physically admissible",
            revalidation
          });
          return this.#finalizeNavigationPlan(
            stored,
            receipt,
            options.retainTerminal ?? false
          );
        });
      }
      const applied = await this.#authority.capture(() => {
        const identity = this.#authorityIdentity();
        if (this.#routes.get(planId) !== stored
          || identity.revision !== captured.context.worldRevision
          || identity.stateSha256 !== captured.context.stateSha256) {
          return false;
        }
        stored.plan = structuredClone(validation.plan!);
        stored.arrivalHeading = validation.arrivalHeading
          ? structuredClone(validation.arrivalHeading)
          : null;
        stored.releaseJointTracking = validation.releaseJointTracking;
        stored.validatedRevision = identity.revision;
        stored.validatedStateSha256 = identity.stateSha256;
        stored.revalidationCount = nextRevalidationCount;
        stored.carriedObjectBindings = structuredClone(
          captured.context.carriedObjectBindings
        );
        stored.carriedObjectTaskSpaceTargets = structuredClone(
          captured.context.carriedObjectTaskSpaceTargets
        );
        stored.carriedObjectContinuation = null;
        stored.carriedObjectUnauthorizedContacts = [];
        stored.progress = {
          version: 1,
          start_root_position: { ...validation.start },
          waypoint_index: Math.min(1, validation.plan!.waypoints.length - 1),
          committed_frame_count: 0,
          stopping_frame_count: 0
        };
        this.#navigationState = {
          planId,
          status: "planned",
          target: { ...stored.requestedTarget },
          waypoints: stored.plan.waypoints.map((point) => ({ ...point })),
          waypointIndex: 1
        };
        return true;
      });
      if (!applied) {
        return this.#authority.capture(() => {
          const receipt = this.#receipt(false, "plan_stale", 0, {
            reason: "world_state_changed_during_navigation_revalidation",
            revalidation: {
              ...revalidationEvidence(true, false, nextRevalidationCount),
              admission_state_sha256: this.#authorityStateSha256()
            }
          });
          return this.#finalizeNavigationPlan(
            stored,
            receipt,
            options.retainTerminal ?? false
          );
        });
      }
      revalidation = revalidationEvidence(true, true, nextRevalidationCount);
    }
    const admission: HumanoidAuthorityIdentity = {
      revision: stored.validatedRevision + stored.progress.committed_frame_count,
      stateSha256: revalidation.admission_state_sha256
    };
    revalidation = {
      ...revalidation,
      admission_state_sha256: admission.stateSha256
    };
    let execution: HumanoidNavigationExecution | undefined;
    const onlineReplans: Array<{
      attempt: number;
      trigger: string;
      world_revision: number;
      accepted: boolean;
      reason: string | null;
      waypoint_count: number;
    }> = [];
    let handle: HumanoidAuthorityCommandHandle<HumanoidExecutionReceipt>;
    try {
      handle = await this.#authority.submit({
        id: `navigation:${planId}`,
        source: "navigation",
        admission,
        ...(frameSink ? { frameSink } : {}),
        admit: () => {
          if (this.#routes.get(planId) !== stored) {
            throw new Error(`Humanoid navigation plan became unavailable: ${planId}`);
          }
          this.#stationKeepingAnchor = null;
          if (stored.releaseJointTracking) {
            this.#reference = releaseReferenceTracking(this.#reference);
          }
          execution = new HumanoidNavigationExecution({
            plan: stored.plan,
            reference: this.#reference,
            simulation: this.#simulation,
            progress: stored.progress,
            contactConstraints: humanoidCarriedObjectContactConstraints(
              stored.carriedObjectBindings
            ),
            graspTargets: contactAwareG1GraspTargetsForBindings({
              bindings: stored.carriedObjectBindings.bindings,
              graspRegistry: this.#graspRegistry
            }),
            carryTaskSpaceTargets: stored.carriedObjectTaskSpaceTargets,
            arrivalHeading: stored.arrivalHeading,
            acceptedPositionToleranceMeters: stored.acceptedPositionToleranceMeters
          });
          this.#navigationState.status = "executing";
        },
        step: async () => {
          options.signal?.throwIfAborted();
          if (!execution) throw new Error("Humanoid navigation was not admitted");
          await this.#ensurePhysicalRegion();
          const prepared = await execution.prepareFrame(this.#simulation);
          this.#reference = execution.reference;
          const step = prepared
            ? (() => {
                this.#commitFrameState({ routePlanId: planId });
                this.#observeGraspFrame(this.#frame, prepared.snapshot);
                const continuation = humanoidCarriedObjectContinuationEvidence({
                  state: stored.carriedObjectBindings,
                  registry: this.#graspRegistry,
                  currentFrame: this.#frame,
                  currentWorldRevision: this.#worldRevision
                });
                const unauthorized = humanoidCarriedObjectUnauthorizedContacts(
                  stored.carriedObjectBindings,
                  prepared.snapshot.contacts
                );
                stored.carriedObjectContinuation = continuation;
                stored.carriedObjectUnauthorizedContacts = unauthorized;
                this.#observeCarriedObjectFrame(prepared.snapshot);
                return execution!.commitPreparedFrame(
                  carryNavigationFailure(continuation, unauthorized, prepared.snapshot)
                );
              })()
            : {
                waypointIndex: stored.progress.waypoint_index,
                done: true
              };
          if (prepared) {
            stored.progress = execution.checkpoint();
            this.#navigationState.waypointIndex = step.waypointIndex;
          }
          if (!step.done) {
            const snapshot = this.snapshot();
            await options.persistenceSink?.(this.#capturePersistenceState());
            return {
              ...(frameSink && !options.persistenceSink ? { snapshot } : {}),
              done: false
            };
          }
          const result = execution.result();
          const replanDecision = onlineNavigationReplanDecision({
            reason: result.reason,
            fallen: result.final.fallen,
            attempts: stored.progress.online_replan_count ?? 0
          });
          if (!result.completed && replanDecision.replan) {
            const context = this.#captureNavigationPlanningContext();
            const replanned = await this.#validateNavigationIntent(
              stored.requestedTarget,
              stored.requestedArrivalHeading,
              context,
              stored.acceptedPositionToleranceMeters
            );
            const attempt = (stored.progress.online_replan_count ?? 0) + 1;
            onlineReplans.push({
              attempt,
              trigger: result.reason ?? "navigation_blocked",
              world_revision: context.worldRevision,
              accepted: replanned.accepted && replanned.plan !== null,
              reason: replanned.reason ?? null,
              waypoint_count: replanned.plan?.waypoints.length ?? 0
            });
            if (replanned.accepted && replanned.plan) {
              validationRevision = context.worldRevision;
              validationStateSha256 = context.stateSha256;
              stored.plan = structuredClone(replanned.plan);
              stored.arrivalHeading = replanned.arrivalHeading
                ? structuredClone(replanned.arrivalHeading)
                : null;
              stored.releaseJointTracking = replanned.releaseJointTracking;
              stored.revalidationCount += 1;
              revalidation = {
                ...revalidationEvidence(true, true, stored.revalidationCount),
                admission_state_sha256: context.stateSha256
              };
              stored.carriedObjectBindings = structuredClone(
                context.carriedObjectBindings
              );
              stored.carriedObjectTaskSpaceTargets = structuredClone(
                context.carriedObjectTaskSpaceTargets
              );
              stored.carriedObjectContinuation = null;
              stored.carriedObjectUnauthorizedContacts = [];
              stored.progress = {
                version: 1,
                start_root_position: { ...stored.progress.start_root_position },
                segment_start_root_position: { ...context.start },
                waypoint_index: Math.min(1, replanned.plan.waypoints.length - 1),
                committed_frame_count: stored.progress.committed_frame_count,
                online_replan_count: attempt,
                stopping_frame_count: 0,
                stopping_settled_frame_count: 0,
                arrival_position_latched: false
              };
              execution = new HumanoidNavigationExecution({
                plan: stored.plan,
                reference: this.#reference,
                simulation: this.#simulation,
                progress: stored.progress,
                contactConstraints: humanoidCarriedObjectContactConstraints(
                  stored.carriedObjectBindings
                ),
                graspTargets: contactAwareG1GraspTargetsForBindings({
                  bindings: stored.carriedObjectBindings.bindings,
                  graspRegistry: this.#graspRegistry
                }),
                carryTaskSpaceTargets: stored.carriedObjectTaskSpaceTargets,
                arrivalHeading: stored.arrivalHeading,
                acceptedPositionToleranceMeters: stored.acceptedPositionToleranceMeters
              });
              this.#navigationState = {
                planId,
                status: "executing",
                target: { ...stored.requestedTarget },
                waypoints: stored.plan.waypoints.map((point) => ({ ...point })),
                waypointIndex: stored.progress.waypoint_index
              };
              await options.persistenceSink?.(this.#capturePersistenceState());
              return {
                ...(prepared && frameSink && !options.persistenceSink
                  ? { snapshot: this.snapshot() }
                  : {}),
                done: false
              };
            }
          }
          this.#navigationState.status = result.completed ? "completed" : "blocked";
          const receipt = this.#receipt(
            result.completed,
            result.completed ? "navigation_completed" : "navigation_blocked",
            result.frames,
            {
              ...(result.reason ? { reason: result.reason } : {}),
              travelledDistance: result.travelledDistance,
              online_replans: onlineReplans,
              carry: {
                binding_set: structuredClone(stored.carriedObjectBindings),
                continuation: stored.carriedObjectContinuation
                  ? structuredClone(stored.carriedObjectContinuation)
                  : null,
                unauthorized_contacts: structuredClone(
                  stored.carriedObjectUnauthorizedContacts
                )
              },
              revalidation
            }
          );
          const finalized = this.#finalizeNavigationPlan(
            stored,
            receipt,
            options.retainTerminal ?? false
          );
          await options.persistenceSink?.(this.#capturePersistenceState());
          return {
            ...(step.snapshot && frameSink && !options.persistenceSink
              ? { snapshot: finalized.finalSnapshot }
              : {}),
            done: true,
            result: finalized
          };
        }
      });
    } catch (error) {
      if (!(error instanceof HumanoidAuthorityAdmissionError)) throw error;
      return this.#authority.capture(() => {
        const receipt = this.#receipt(false, "plan_stale", 0, {
          reason: error.message,
          revalidation: {
            ...revalidation,
            accepted: false,
            admission_state_sha256: error.actual.stateSha256
          }
        });
        return this.#finalizeNavigationPlan(
          stored,
          receipt,
          options.retainTerminal ?? false
        );
      });
    }
    return this.#driveAuthorityCommand(handle, options);
  }

  async acknowledgeNavigation(
    planId: string,
    resultSha256: string
  ): Promise<boolean> {
    return this.#authority.capture(() => {
      const stored = this.#routes.get(planId);
      if (!stored) return false;
      if (!stored.terminal) {
        throw new Error(`Humanoid navigation plan is not terminal: ${planId}`);
      }
      if (stored.terminal.result_sha256 !== resultSha256) {
        throw new Error(`Humanoid navigation terminal acknowledgement mismatch: ${planId}`);
      }
      const deleted = this.#deleteRoutePlan(planId);
      if (this.#navigationState.planId === planId) {
        this.#navigationState.planId = null;
        this.#navigationState.waypointIndex = null;
      }
      return deleted;
    });
  }

  async advanceStationary(frameSink?: HumanoidFrameSink): Promise<HumanoidWorldSnapshot | null> {
    const tick = await this.#authority.tick(frameSink);
    return tick.snapshot;
  }

  async #driveAuthorityCommand<Result>(
    handle: HumanoidAuthorityCommandHandle<Result>,
    options: HumanoidExecutionOptions
  ): Promise<Result> {
    const pacer = createHumanoidControlStepPacer({
      controlStepSeconds: this.#simulation.controllerDescriptor().controlStepSeconds,
      realtime: options.realtime ?? false,
      ...(options.signal ? { signal: options.signal } : {})
    });
    try {
      while (!handle.settled) {
        await pacer.waitForNextStep();
        if (handle.settled) break;
        await this.#authority.tick();
      }
      const result = await handle.result;
      await handle.publication;
      return result;
    } catch (error) {
      await this.#authority.cancel(handle.id, error);
      throw error;
    }
  }

  #deleteMotionPlan(planId: string): boolean {
    const deleted = this.#motions.delete(planId);
    if (deleted) this.#planRegistryEpoch += 1;
    return deleted;
  }

  #finalizeMotionPlan(
    stored: StoredHumanoidMotionPlan,
    receipt: HumanoidExecutionReceipt,
    retainTerminal: boolean
  ): HumanoidExecutionReceipt {
    const planId = stored.plan.id;
    if (!retainTerminal) {
      this.#deleteMotionPlan(planId);
      return receipt;
    }
    if (stored.terminal) {
      return humanoidPlanTerminalReceipt(stored.terminal, this.snapshot());
    }
    const terminal = createHumanoidPlanTerminal({
      planId,
      totalFrames: stored.progress.nextFrameIndex,
      receipt
    });
    stored.terminal = terminal;
    if (!this.#motions.has(planId)) {
      this.#motions.set(planId, stored);
      this.#planRegistryEpoch += 1;
    }
    return {
      ...receipt,
      terminalResultSha256: terminal.result_sha256
    };
  }

  #deleteRoutePlan(planId: string): boolean {
    const deleted = this.#routes.delete(planId);
    if (deleted) this.#planRegistryEpoch += 1;
    return deleted;
  }

  #finalizeNavigationPlan(
    stored: StoredHumanoidNavigationPlan,
    receipt: HumanoidExecutionReceipt,
    retainTerminal: boolean
  ): HumanoidExecutionReceipt {
    const planId = stored.id;
    if (!retainTerminal) {
      this.#deleteRoutePlan(planId);
      if (this.#navigationState.planId === planId) {
        this.#navigationState.planId = null;
        this.#navigationState.waypointIndex = null;
      }
      return receipt;
    }
    if (stored.terminal) {
      return humanoidPlanTerminalReceipt(stored.terminal, this.snapshot());
    }
    const terminal = createHumanoidPlanTerminal({
      planId,
      totalFrames: stored.progress.committed_frame_count,
      receipt
    });
    stored.terminal = terminal;
    if (!this.#routes.has(planId)) {
      this.#routes.set(planId, stored);
      this.#planRegistryEpoch += 1;
    }
    return {
      ...receipt,
      terminalResultSha256: terminal.result_sha256
    };
  }

  #authorityIdentity(): HumanoidAuthorityIdentity {
    return {
      revision: this.#worldRevision,
      stateSha256: this.#authorityStateSha256()
    };
  }

  #authorityStateSha256(): string {
    const carriedObjectBindings = this.#currentCarriedObjectBindings();
    const snapshot = this.#simulation.snapshot();
    const visibleContactObjectIds = this.#objectMemory.observedObjectIds(
      this.#frame,
      this.#worldRevision
    );
    const visibleContactSolidIds = observableHumanoidSolidIds(
      this.#simulation.senseSolids(this.#scenario.visibility_radius),
      snapshot.contacts
    );
    const simulation = this.#simulation.captureState();
    return humanoidAuthorityStateSha256({
      simulation,
      reference: this.#reference,
      visibleContactObjectIds,
      visibleContactSolidIds,
      planRegistryEpoch: this.#planRegistryEpoch,
      graspRegistry: this.#currentGraspRegistryCheckpoint(),
      carriedObjectBindings
    });
  }

  #currentGraspRegistryCheckpoint(): HumanoidGraspRegistryCheckpoint {
    const checkpoint = this.#graspRegistry.checkpoint();
    if (checkpoint.last_frame !== this.#frame) {
      throw new Error(
        `Humanoid grasp registry frame is not authoritative: `
        + `${String(checkpoint.last_frame)} != ${this.#frame}`
      );
    }
    return checkpoint;
  }

  #currentGraspState(): HumanoidWorldSnapshot["grasp"] {
    return {
      contractSha256: this.#graspRegistry.contractSha256,
      assessments: this.#graspRegistry.assessmentsForFrame(this.#frame)
    };
  }

  #currentCarriedObjectBindings(): HumanoidCarriedObjectBindingSet {
    return this.#requiredCarriedObjectLifecycle().bindingSet(
      this.#frame,
      this.#worldRevision
    );
  }

  #acquireSuccessfulOptionGrasp(stored: StoredHumanoidMotionPlan): void {
    if (stored.option?.status !== "succeeded"
      || this.#requiredCarriedObjectLifecycle().active !== null) return;
    const verified = new Set(this.#graspRegistry.assessmentsForFrame(this.#frame)
      .filter((assessment) => assessment.grasp_verified)
      .map((assessment) => `${assessment.object_id}\0${assessment.hand}`));
    const requests = stored.option.contract.predicates.flatMap((predicate) => (
      predicate.type === "grasp_verified"
        && verified.has(`${predicate.object_id}\0${predicate.hand}`)
        ? [{ object_id: predicate.object_id, hand: predicate.hand }]
        : []
    ));
    if (requests.length === 0) return;
    this.#requiredCarriedObjectLifecycle().acquire({
      currentFrame: this.#frame,
      currentWorldRevision: this.#worldRevision,
      requests
    });
  }

  #requiredCarriedObjectLifecycle(): HumanoidCarriedObjectLifecycle {
    if (!this.#carriedObjectLifecycle) {
      throw new Error("Humanoid carried-object lifecycle is not initialized");
    }
    return this.#carriedObjectLifecycle;
  }

  #observeGraspFrame(
    frame: number,
    snapshot: HumanoidSimulationSnapshot
  ): void {
    this.#graspRegistry.observe(frame, snapshot);
  }

  #observeCarriedObjectFrame(snapshot: HumanoidSimulationSnapshot): void {
    this.#requiredCarriedObjectLifecycle().observe({
      currentFrame: this.#frame,
      currentWorldRevision: this.#worldRevision,
      contacts: snapshot.contacts
    });
  }

  #refreshCurrentObjectMemory(): ReturnType<HumanoidSimulation["senseObjects"]> {
    const snapshot = this.#simulation.snapshot();
    const sensed = this.#simulation.senseObjects(this.#scenario.visibility_radius);
    this.#objectMemory.refresh(
      this.#frame,
      this.#worldRevision,
      snapshot.objects,
      new Set(Object.keys(sensed.objects))
    );
    return sensed;
  }

  #physicalAnchors(
    snapshot: HumanoidSimulationSnapshot = this.#simulation.snapshot()
  ): Array<Pick<Vec3, "x" | "z">> {
    const carriedObjectIds = new Set(
      this.#carriedObjectLifecycle?.active?.bindings.map(({ object_id }) => object_id)
        ?? []
    );
    return [
      { x: snapshot.rootPosition.x, z: snapshot.rootPosition.z },
      ...Object.values(snapshot.objects)
        .filter((object) => carriedObjectIds.has(object.id)
          || Math.hypot(
            object.linearVelocity.x,
            object.linearVelocity.y,
            object.linearVelocity.z
          ) >= ACTIVE_OBJECT_LINEAR_SPEED_METERS_PER_SECOND
          || Math.hypot(
            object.angularVelocity.x,
            object.angularVelocity.y,
            object.angularVelocity.z
          ) >= ACTIVE_OBJECT_ANGULAR_SPEED_RADIANS_PER_SECOND)
        .map(({ position }) => ({ x: position.x, z: position.z }))
    ];
  }

  async #ensurePhysicalRegion(): Promise<void> {
    const anchors = this.#physicalAnchors();
    const requested = humanoidPhysicalRegion(this.#scenario, anchors);
    if (requested.key === this.#physicalRegion.key) return;
    const captured = this.#simulation.captureState();
    const next = await createHumanoidPhysicsResources(this.#scenario, anchors);
    let installed = false;
    try {
      next.simulation.restoreState(captured);
      const previous = {
        simulation: this.#simulation,
        rolloutPool: this.#rolloutPool,
        physicalRegion: this.#physicalRegion
      };
      this.#simulation = next.simulation;
      this.#rolloutPool = next.rolloutPool;
      this.#physicalRegion = next.physicalRegion;
      installed = true;
      await disposeHumanoidPhysicsResources(previous);
    } catch (error) {
      if (!installed) await disposeHumanoidPhysicsResources(next);
      throw error;
    }
  }

  async #stepStationaryAuthority(): Promise<HumanoidWorldSnapshot> {
    await this.#ensurePhysicalRegion();
    const current = this.#simulation.snapshot();
    this.#stationKeepingAnchor ??= captureHumanoidStationKeepingAnchor(
      current,
      this.#frame,
      this.#worldRevision
    );
    this.#reference = stationKeepingHumanoidReference(
      this.#reference,
      current,
      this.#stationKeepingAnchor
    );
    const snapshot = await this.#simulation.step(this.#reference);
    this.#commitFrameState();
    this.#observeGraspFrame(this.#frame, snapshot);
    this.#observeCarriedObjectFrame(snapshot);
    return this.snapshot();
  }

  #planExpiryRevision(createdRevision: number): number {
    return humanoidPlanExpiryRevision({
      createdRevision,
      controlStepSeconds: this.#simulation.controllerDescriptor().controlStepSeconds,
      leaseSeconds: this.#planIntentLeaseSeconds
    });
  }

  async dispose(): Promise<void> {
    await this.#authority.dispose();
    await Promise.all([
      disposeHumanoidWorldResources({
        simulation: this.#simulation,
        rolloutPool: this.#rolloutPool,
        navigation: this.#navigation,
        physicalRegion: this.#physicalRegion
      }),
      this.#motionGenerator.dispose()
    ]);
  }

  async #settle(steps: number): Promise<void> {
    for (let index = 0; index < steps; index += 1) {
      await this.#simulation.step(this.#reference);
    }
    const snapshot = this.#simulation.snapshot();
    if (snapshot.fallen) {
      throw new Error("Humanoid could not reach a stable initial stance");
    }
    this.#stationKeepingAnchor = captureHumanoidStationKeepingAnchor(snapshot, 0, 0);
    this.#observeGraspFrame(0, snapshot);
    this.#carriedObjectLifecycle = new HumanoidCarriedObjectLifecycle({
      registry: this.#graspRegistry,
      currentFrame: 0,
      currentWorldRevision: 0
    });
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
      ...(checkpoint.simulation.requestedActuatorTorques
        ? {
            requestedActuatorTorques: Float64Array.from(
              checkpoint.simulation.requestedActuatorTorques
            )
          }
        : {}),
      ...(checkpoint.simulation.handCommandTargets
        ? {
            handCommandTargets: Float64Array.from(
              checkpoint.simulation.handCommandTargets
            )
          }
        : {}),
      controller: structuredClone(checkpoint.simulation.controller)
    });
    this.#reference = hydrateHumanoidReference(checkpoint.reference);
    this.#stationKeepingAnchor = checkpoint.stationKeeping
      ? structuredClone(checkpoint.stationKeeping)
      : null;
    this.#frame = checkpoint.frame;
    this.#worldRevision = checkpoint.worldRevision;
    this.#routeSequence = checkpoint.routeSequence;
    this.#planRegistryEpoch = checkpoint.planRegistryEpoch;
    this.#physicalSafety = checkpoint.physicalSafety
      ? structuredClone(checkpoint.physicalSafety)
      : undefined;
    if (JSON.stringify(this.#graspRegistry.checkpoint())
      !== JSON.stringify(checkpoint.graspRegistry)) {
      throw new Error("Humanoid grasp registry restore did not preserve its checkpoint");
    }
    this.#currentGraspRegistryCheckpoint();
    this.#carriedObjectLifecycle = new HumanoidCarriedObjectLifecycle({
      registry: this.#graspRegistry,
      currentFrame: this.#frame,
      currentWorldRevision: this.#worldRevision,
      checkpoint: checkpoint.carriedObjectLifecycle
    });
    if (!checkpoint.carriedObjectLifecycle) {
      const legacyRequests = this.#graspRegistry.assessmentsForFrame(this.#frame)
        .filter((assessment) => assessment.grasp_verified)
        .map((assessment) => ({
          object_id: assessment.object_id,
          hand: assessment.hand
        }));
      if (legacyRequests.length > 0) {
        this.#carriedObjectLifecycle.acquire({
          currentFrame: this.#frame,
          currentWorldRevision: this.#worldRevision,
          requests: legacyRequests
        });
      }
    }
    const restoredStateSha256 = this.#authorityStateSha256();
    const restoredPlans = restoreHumanoidWorldPlans({
      checkpoint,
      snapshot: this.#simulation.snapshot(),
      graspRegistry: this.#graspRegistry,
      currentBindings: this.#currentCarriedObjectBindings(),
      restoredStateSha256,
      planExpiryRevision: (createdRevision) => this.#planExpiryRevision(createdRevision)
    });
    this.#motions.clear();
    for (const [planId, stored] of restoredPlans.motions) {
      this.#motions.set(planId, stored);
    }
    this.#routes.clear();
    for (const [planId, stored] of restoredPlans.routes) {
      this.#routes.set(planId, stored);
    }
    this.#planRegistryEpoch = restoredPlans.planRegistryEpoch;
    this.#physicalSafety = restoredPlans.physicalSafety;
    this.#navigationState = restoredPlans.navigation;
  }

  #commitFrameState(
    activePlan: { motionPlanId?: string; routePlanId?: string } = {}
  ): void {
    this.#frame += 1;
    this.#worldRevision += 1;
    this.#pruneUnconsumablePlans(activePlan);
  }

  #pruneUnconsumablePlans(activePlan: {
    motionPlanId?: string;
    routePlanId?: string;
  }): void {
    const physicalExecutionCommitted = activePlan.motionPlanId !== undefined
      || activePlan.routePlanId !== undefined;
    let registryChanged = false;
    for (const [planId, stored] of this.#motions) {
      if (planId === activePlan.motionPlanId) continue;
      if (stored.terminal) continue;
      if (physicalExecutionCommitted) {
        this.#motions.delete(planId);
        registryChanged = true;
        continue;
      }
      if (stored.progress.nextFrameIndex > 0) {
        const expectedRevision = stored.validatedRevision
          + stored.progress.nextFrameIndex;
        if (expectedRevision !== this.#worldRevision) {
          this.#motions.delete(planId);
          registryChanged = true;
        }
        continue;
      }
      if (!humanoidPlanIntentIsActive(this.#worldRevision, stored.expiresRevision)) {
        this.#motions.delete(planId);
        registryChanged = true;
      }
    }
    for (const [planId, stored] of this.#routes) {
      if (planId === activePlan.routePlanId) continue;
      if (stored.terminal) continue;
      if (physicalExecutionCommitted) {
        this.#routes.delete(planId);
        registryChanged = true;
        continue;
      }
      if (stored.progress.committed_frame_count > 0) {
        const expectedRevision = stored.validatedRevision
          + stored.progress.committed_frame_count;
        if (expectedRevision !== this.#worldRevision) {
          this.#routes.delete(planId);
          registryChanged = true;
        }
        continue;
      }
      if (!humanoidPlanIntentIsActive(this.#worldRevision, stored.expiresRevision)) {
        this.#routes.delete(planId);
        registryChanged = true;
      }
    }
    if (registryChanged) this.#planRegistryEpoch += 1;
    const navigationPlanId = this.#navigationState.planId;
    if (navigationPlanId
      && navigationPlanId !== activePlan.routePlanId
      && !this.#routes.has(navigationPlanId)) {
      this.#navigationState.planId = null;
      this.#navigationState.status = "blocked";
      this.#navigationState.waypointIndex = null;
    }
  }

  #motionOptionDetectorInput(
    snapshot: HumanoidSimulationSnapshot,
    option: HumanoidMotionOptionContract,
    trackedObjectIds: ReadonlySet<string> = new Set()
  ): HumanoidMotionOptionDetectorInput {
    const sensed = this.#simulation.senseObjects(
      this.#scenario.visibility_radius
    );
    const sensedSolids = this.#simulation.senseSolids(
      this.#scenario.visibility_radius
    );
    const observedFrame = this.#frame;
    const observedWorldRevision = this.#worldRevision;
    this.#objectMemory.refresh(
      observedFrame,
      observedWorldRevision,
      snapshot.objects,
      new Set(Object.keys(sensed.objects))
    );
    const observableObjects = new Map(this.#objectMemory
      .observableObjectStates(observedFrame, observedWorldRevision)
      .map((object): [string, HumanoidMotionOptionObservableObject] => [object.id, {
        id: object.id,
        position: { ...object.pose.position },
        rotation: { ...object.pose.rotation },
        size: { ...object.size },
        ...this.#observableArticulation(object.id, snapshot)
      }]));
    for (const objectId of trackedObjectIds) {
      if (observableObjects.has(objectId)) continue;
      const state = snapshot.objects[objectId];
      const descriptor = this.#scenario.objects.find((object) => object.id === objectId);
      if (!state || !descriptor) continue;
      observableObjects.set(objectId, {
        id: objectId,
        position: { ...state.position },
        rotation: { ...state.rotation },
        size: { ...descriptor.size },
        ...this.#observableArticulation(objectId, snapshot)
      });
    }
    return {
      snapshot,
      observableObjects: [...observableObjects.values()],
      observableSolidIds: observableHumanoidSolidIds(
        sensedSolids,
        snapshot.contacts
      ),
      zones: this.#scenario.zones,
      graspAssessments: this.#graspRegistry.bindingsForOption(
        option,
        observedFrame
      )
    };
  }

  #observableArticulation(
    objectId: string,
    snapshot: HumanoidSimulationSnapshot
  ): Pick<
    HumanoidMotionOptionObservableObject,
    "articulation" | "container" | "supportSurface"
  > {
    const descriptor = this.#scenario.objects.find((object) => object.id === objectId);
    const capability = descriptor ? humanoidObjectCapability(descriptor) : undefined;
    const observed = snapshot.objects[objectId]?.articulation;
    if (!capability) return {};
    return {
      ...(capability.articulation && observed
        ? {
            articulation: {
              jointId: capability.articulation.joint_id,
              position: observed.position,
              velocity: observed.velocity,
              closedPosition: capability.articulation.closed_position,
              openPosition: capability.articulation.open_position
            }
          }
        : {}),
      ...(capability.container
        ? { container: structuredClone(capability.container) }
        : {}),
      ...(capability.supportSurface
        ? { supportSurface: structuredClone(capability.supportSurface) }
        : {})
    };
  }

  #motionOptionReceipt(
    stored: StoredHumanoidMotionPlan,
    frames: number,
    failures: HumanoidMotionValidation["failures"] = [],
    revalidation?: PlanRevalidationEvidence
  ): HumanoidExecutionReceipt {
    const option = stored.option;
    if (!option || !isTerminalMotionOption(option)) {
      throw new Error("Humanoid motion option has no terminal physical result");
    }
    const physicalSafety = storedMotionPhysicalSafety(stored);
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
      ...(physicalSafety ? { physical_safety: physicalSafety } : {}),
      ...(failures.length === 0 ? {} : { failures }),
      ...(revalidation ? { revalidation } : {}),
      carry: {
        binding_set: structuredClone(stored.carriedObjectBindings),
        continuation: stored.carriedObjectContinuation
          ? structuredClone(stored.carriedObjectContinuation)
          : null,
        unauthorized_contacts: structuredClone(
          stored.carriedObjectUnauthorizedContacts
        )
      },
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

function manipulationInteractionTargets(
  observation: HumanoidWorldObservation
): Array<{
  objectId: string;
  interactionPointId: string;
  worldPosition: Vec3;
  approachDirection?: Vec3;
  preferredGraspAxis?: Vec3;
  clearanceMeters?: number;
}> {
  const wrists = [
    observation.robot.links.left_wrist_yaw_link.position,
    observation.robot.links.right_wrist_yaw_link.position
  ];
  return observation.interaction.object_world_model.objects.flatMap((object) => {
    if (object.status !== "visible"
      || object.role !== "manipulable" && object.articulation === null) {
      return [];
    }
    return object.interaction_points
      .filter(({ kind }) => [
        "grasp", "push", "pull", "press", "turn"
      ].includes(kind))
      .sort((left, right) => minimumPointDistance(left.world_position, wrists)
        - minimumPointDistance(right.world_position, wrists))
      .slice(0, 2)
      .map((point) => ({
        objectId: object.id,
        interactionPointId: point.id,
        worldPosition: { ...point.world_position },
        ...(point.approach_direction_world
          ? { approachDirection: { ...point.approach_direction_world } }
          : {}),
        ...(object.articulation?.axis_world
          ? { preferredGraspAxis: { ...object.articulation.axis_world } }
          : {}),
        clearanceMeters: point.clearance_m
      }));
  });
}

function minimumPointDistance(point: Vec3, references: readonly Vec3[]): number {
  return Math.min(...references.map((reference) => Math.hypot(
    point.x - reference.x,
    point.y - reference.y,
    point.z - reference.z
  )));
}

function checkpointRootAnchor(
  checkpoint: HumanoidWorldCheckpoint
): Pick<Vec3, "x" | "z"> {
  const z = checkpoint.simulation.positions[0];
  const x = checkpoint.simulation.positions[1];
  if (!Number.isFinite(x) || !Number.isFinite(z)) {
    throw new Error("Humanoid checkpoint is missing a finite root position");
  }
  return { x: x!, z: z! };
}

function navigationCarryReceipt(
  state: HumanoidCarriedObjectBindingSet
): NavigationPlanReceipt["carry"] {
  return {
    binding_set_sha256: humanoidCarriedObjectBindingSetSha256(state),
    bindings: state.bindings.map((binding) => ({
      object_id: binding.object_id,
      hand: binding.hand
    }))
  };
}
