import { z } from "zod";
import { Vec3Schema, type Vec3 } from "../../domain/schema.js";
import type { NavigationPlan } from "../navigation.js";
import {
  blockedHumanoidContacts,
  HumanoidContactConstraintSchema,
  type HumanoidContactConstraint
} from "./motion-plan.js";
import {
  stationaryHumanoidReference,
  targetReference,
  type HumanoidReference
} from "./reference.js";
import type {
  HumanoidPolicyFrameSink,
  HumanoidSimulation,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import {
  contactAwareG1GraspTargetsForBindings,
  contactAwareG1GraspJointTargets,
  g1CarriedGraspRequiresNoslip,
  type G1ContactAwareGraspTarget
} from "./contact-aware-grasp-servo.js";
import {
  applyHumanoidCarryTaskSpaceServo,
  HumanoidCarryTaskSpaceTargetsSchema,
  type HumanoidCarryTaskSpaceTarget
} from "./carry-task-space-servo.js";
import { HumanoidGraspRegistry } from "./grasp-registry.js";
import {
  humanoidCarriedObjectContactConstraints,
  humanoidCarriedObjectContinuationEvidence,
  humanoidCarriedObjectUnauthorizedContacts,
  type HumanoidCarriedObjectBindingSet
} from "./carried-object-binding.js";
import { g1HandObjectContacts } from "./hand-contact-evidence.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  rotateVector,
  subtract,
  yawFromQuaternion
} from "../geometry.js";
import {
  HumanoidNavigationArrivalHeadingSchema,
  humanoidNavigationArrivalHeadingError,
  humanoidNavigationArrivalHeadingSatisfied,
  humanoidNavigationShouldBeginBraking,
  humanoidNavigationStoppingDistance,
  type HumanoidNavigationArrivalHeading
} from "./navigation-arrival.js";
import { humanoidControllerTaskCapabilities } from
  "./controller-task-capabilities.js";
import {
  humanoidNavigationCollisionEvidence,
  type HumanoidNavigationCollisionEvidence
} from "./navigation-collision-evidence.js";
import {
  HumanoidEmbodiedSkillCallSchema,
  legacyHumanoidEmbodiedSkillIdentity,
  type HumanoidEmbodiedSkillIdentity
} from "./embodied-skill-call.js";
import type { HumanoidSkillProgressEvidence } from "./skill-event-stream.js";

const NAVIGATION_PROGRESS_BUDGET_SPEED_METERS_PER_SECOND = 0.05;
const NAVIGATION_PRECISION_PROGRESS_BUDGET_SPEED_METERS_PER_SECOND = 0.01;
const NAVIGATION_WAYPOINT_TOLERANCE_METERS = 0.18;
const NAVIGATION_FINAL_WAYPOINT_TOLERANCE_METERS = 0.06;
const CARRY_NAVIGATION_FINAL_WAYPOINT_TOLERANCE_METERS = 0.12;
const NAVIGATION_PHYSICAL_DEADBAND_TOLERANCE_METERS = 0.08;
const NAVIGATION_POSITION_DISCRETIZATION_METERS = 0.001;
const NAVIGATION_EFFECTIVE_COMMAND_MARGIN_METERS_PER_SECOND = 0.01;
const NAVIGATION_ALREADY_AT_TARGET_TOLERANCE_METERS = 0.02;
const NAVIGATION_MINIMUM_SHORT_ROUTE_PROGRESS_METERS = 0.02;
const NAVIGATION_MINIMUM_SHORT_ROUTE_PROGRESS_RATIO = 0.5;
const NAVIGATION_STARTUP_SECONDS = 10;
const NAVIGATION_FINAL_CONVERGENCE_BUDGET_SECONDS = 6;
// Learned locomotion can arrive positionally before its yaw response has
// converged, especially after a long station-keeping period.  Manipulation
// approaches require the declared wrist-facing base yaw, so timing out the
// preview while the robot is upright, inside the position envelope, and still
// rotating rejects a physically feasible approach.  Keep this as bounded
// physical rollout time rather than weakening the heading contract.
const NAVIGATION_ARRIVAL_HEADING_BUDGET_SECONDS = 20;
const NAVIGATION_ARRIVAL_HEADING_RECOVERY_HYSTERESIS_RADIANS = 0.03;
const NAVIGATION_ARRIVAL_HEADING_SETTLE_MARGIN_RADIANS = 0.03;
const NAVIGATION_ARRIVAL_POSITION_LATCH_HYSTERESIS_METERS = 0.01;
const NAVIGATION_STOP_SETTLE_BUDGET_SECONDS = 6;
const NAVIGATION_STOP_SETTLED_STEPS = 4;
const NAVIGATION_STOP_PLANAR_SPEED_TOLERANCE_METERS_PER_SECOND = 0.08;
const NAVIGATION_STOP_YAW_SPEED_TOLERANCE_RADIANS_PER_SECOND = 0.08;
const NAVIGATION_STOP_COMMAND_TOLERANCE_METERS_PER_SECOND = 0.08;
const NAVIGATION_STOP_POSITION_HOLD_COMMAND_METERS_PER_SECOND = 0.06;
const NAVIGATION_PHYSICAL_DEADBAND_COMMAND_TOLERANCE_METERS_PER_SECOND = 0.11;
const NAVIGATION_MINIMUM_EFFECTIVE_ARRIVAL_YAW_RADIANS_PER_SECOND = 0.45;
const CARRY_MINIMUM_EFFECTIVE_ARRIVAL_YAW_RADIANS_PER_SECOND = 0.3;
const MAXIMUM_NAVIGATION_SECONDS = 90;
const MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED = 1;
const MAXIMUM_YAW_ACCELERATION_RADIANS_PER_SECOND_SQUARED = 2;
const CARRY_MAXIMUM_FORWARD_SPEED_METERS_PER_SECOND = 0.3;
const CARRY_MAXIMUM_REVERSE_SPEED_METERS_PER_SECOND = 0.3;
const CARRY_MAXIMUM_LATERAL_SPEED_METERS_PER_SECOND = 0.12;
const CARRY_MAXIMUM_YAW_SPEED_RADIANS_PER_SECOND = 0.55;
const CARRY_MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED = 0.3;
const CARRY_MAXIMUM_YAW_ACCELERATION_RADIANS_PER_SECOND_SQUARED = 0.7;
const CARRY_MAXIMUM_WRIST_REFERENCE_CORRECTION_RADIANS = 0.06;
const CARRY_NOSLIP_SOLVER_ITERATIONS = 2;

export interface HumanoidNavigationExecutionResult {
  completed: boolean;
  reason?: string;
  blockingContacts?: HumanoidNavigationCollisionEvidence[];
  frames: number;
  reference: HumanoidReference;
  final: HumanoidSimulationSnapshot;
  travelledDistance: number;
}

export interface HumanoidNavigationExecutionStep {
  snapshot?: HumanoidSimulationSnapshot;
  waypointIndex: number;
  done: boolean;
}

export interface HumanoidNavigationPreparedFrame {
  snapshot: HumanoidSimulationSnapshot;
  waypointIndex: number;
  stopping: boolean;
}

export const HumanoidNavigationExecutionProgressSchema = z.object({
  version: z.literal(1),
  start_root_position: Vec3Schema,
  segment_start_root_position: Vec3Schema.optional(),
  waypoint_index: z.number().int().nonnegative(),
  committed_frame_count: z.number().int().nonnegative(),
  online_replan_count: z.number().int().nonnegative().optional(),
  stopping_frame_count: z.number().int().nonnegative(),
  stopping_settled_frame_count: z.number().int().nonnegative().optional(),
  arrival_position_latched: z.boolean().optional()
}).strict().superRefine((progress, context) => {
  if (progress.stopping_frame_count > progress.committed_frame_count) {
    context.addIssue({
      code: "custom",
      path: ["stopping_frame_count"],
      message: "Navigation stopping progress cannot exceed committed physical frames"
    });
  }
  if ((progress.stopping_settled_frame_count ?? 0)
    > progress.stopping_frame_count) {
    context.addIssue({
      code: "custom",
      path: ["stopping_settled_frame_count"],
      message: "Navigation settled progress cannot exceed stopping progress"
    });
  }
});

export type HumanoidNavigationExecutionProgress = z.infer<
  typeof HumanoidNavigationExecutionProgressSchema
>;

export class HumanoidNavigationExecution {
  readonly #plan: NavigationPlan;
  readonly #startRootPosition: Vec3;
  readonly #segmentStartRootPosition: Vec3;
  readonly #maximumTravelFrames: number;
  readonly #stoppingFrames: number;
  readonly #maximumStoppingFrames: number;
  readonly #controlStepSeconds: number;
  readonly #commandResponseHorizonSeconds: number;
  readonly #brakingResponseHorizonSeconds: number;
  readonly #linearAcceleration: number;
  readonly #yawAcceleration: number;
  readonly #carrying: boolean;
  readonly #minimumEffectivePlanarSpeed: number;
  readonly #minimumEffectiveYawSpeed: number;
  readonly #precisionArrival: boolean;
  readonly #contactConstraints: readonly HumanoidContactConstraint[];
  readonly #graspTargets: readonly G1ContactAwareGraspTarget[];
  readonly #carryTaskSpaceTargets: readonly HumanoidCarryTaskSpaceTarget[];
  readonly #arrivalHeading: HumanoidNavigationArrivalHeading | null;
  readonly #requestedPositionToleranceMeters: number | null;
  readonly #onlineReplanCount: number;
  readonly #skillIdentity: HumanoidEmbodiedSkillIdentity;
  readonly #policyFrameSink: HumanoidPolicyFrameSink | undefined;
  readonly #skillWindowMaximumSteps: number | undefined;
  readonly #skillWindowStepOffset: number;
  readonly #initialTargetDistance: number;
  #reference: HumanoidReference;
  #final: HumanoidSimulationSnapshot;
  #waypointIndex: number;
  #frames = 0;
  #stopFrames = 0;
  #stopSettledFrames = 0;
  #arrivalPositionLatched = false;
  #result: HumanoidNavigationExecutionResult | undefined;
  #pendingFrame: HumanoidNavigationPreparedFrame | undefined;

  #skillWindow(): {
    maximumSteps: number;
    stepIndex: number;
    remainingSteps: number;
  } {
    const maximumSteps = this.#skillWindowMaximumSteps ?? Math.max(
      1,
      this.#maximumTravelFrames + this.#maximumStoppingFrames
    );
    const stepIndex = this.#skillWindowStepOffset + this.#frames;
    return {
      maximumSteps,
      stepIndex,
      remainingSteps: Math.max(0, maximumSteps - stepIndex)
    };
  }

  constructor(input: {
    plan: NavigationPlan;
    reference: HumanoidReference;
    simulation: HumanoidSimulation;
    progress?: HumanoidNavigationExecutionProgress;
    contactConstraints?: readonly HumanoidContactConstraint[];
    graspTargets?: readonly G1ContactAwareGraspTarget[];
    carryTaskSpaceTargets?: readonly HumanoidCarryTaskSpaceTarget[];
    arrivalHeading?: HumanoidNavigationArrivalHeading | null;
    acceptedPositionToleranceMeters?: number | null;
    skillIdentity?: HumanoidEmbodiedSkillIdentity;
    policyFrameSink?: HumanoidPolicyFrameSink;
    skillWindow?: {
      maximumSteps: number;
      stepOffset: number;
    };
  }) {
    this.#plan = structuredClone(input.plan);
    this.#policyFrameSink = input.policyFrameSink;
    this.#reference = input.reference;
    const progress = input.progress
      ? HumanoidNavigationExecutionProgressSchema.parse(input.progress)
      : undefined;
    const carrying = (input.graspTargets?.length ?? 0) > 0
      || (input.carryTaskSpaceTargets?.length ?? 0) > 0;
    this.#carrying = carrying;
    if (!carrying && (!progress || progress.committed_frame_count === 0)
      && (input.skillWindow?.stepOffset ?? 0) === 0) {
      input.simulation.resetController(input.reference);
    }
    this.#final = input.simulation.snapshot();
    this.#skillIdentity = input.skillIdentity
      ? structuredClone(input.skillIdentity)
      : legacyHumanoidEmbodiedSkillIdentity({
          callId: "navigation-execution",
          runtimeKind: "navigation",
          phase: "navigate",
          observedFrame: progress?.committed_frame_count ?? 0,
          observedWorldRevision: progress?.committed_frame_count ?? 0
        });
    this.#skillWindowStepOffset = input.skillWindow?.stepOffset ?? 0;
    this.#skillWindowMaximumSteps = input.skillWindow?.maximumSteps;
    if (!Number.isSafeInteger(this.#skillWindowStepOffset)
      || this.#skillWindowStepOffset < 0
      || (this.#skillWindowMaximumSteps !== undefined
        && (!Number.isSafeInteger(this.#skillWindowMaximumSteps)
          || this.#skillWindowMaximumSteps <= this.#skillWindowStepOffset))) {
      throw new Error("Navigation Skill window is invalid");
    }
    this.#arrivalHeading = input.arrivalHeading === null
      || input.arrivalHeading === undefined
      ? null
      : HumanoidNavigationArrivalHeadingSchema.parse(input.arrivalHeading);
    this.#requestedPositionToleranceMeters = input.acceptedPositionToleranceMeters === null
      || input.acceptedPositionToleranceMeters === undefined
      ? null
      : navigationPositionTolerance(input.acceptedPositionToleranceMeters);
    this.#onlineReplanCount = progress?.online_replan_count ?? 0;
    const controller = input.simulation.controllerDescriptor();
    const controlStepSeconds = controller.controlStepSeconds;
    this.#controlStepSeconds = controlStepSeconds;
    this.#commandResponseHorizonSeconds = controller.commandResponseHorizonSeconds
      ?? controlStepSeconds;
    this.#brakingResponseHorizonSeconds = this.#commandResponseHorizonSeconds;
    this.#minimumEffectivePlanarSpeed = controller
      .minimumEffectivePlanarSpeedMetersPerSecond ?? 0;
    this.#minimumEffectiveYawSpeed = controller
      .minimumEffectiveYawSpeedRadiansPerSecond
        ?? NAVIGATION_MINIMUM_EFFECTIVE_ARRIVAL_YAW_RADIANS_PER_SECOND;
    this.#linearAcceleration = carrying
      ? CARRY_MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED
      : MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED;
    this.#yawAcceleration = carrying
      ? CARRY_MAXIMUM_YAW_ACCELERATION_RADIANS_PER_SECOND_SQUARED
      : MAXIMUM_YAW_ACCELERATION_RADIANS_PER_SECOND_SQUARED;
    const maximumPlanarSpeed = carrying
      ? Math.hypot(
          CARRY_MAXIMUM_FORWARD_SPEED_METERS_PER_SECOND,
          CARRY_MAXIMUM_LATERAL_SPEED_METERS_PER_SECOND
        )
      : Math.hypot(0.48, 0.22);
    const maximumYawSpeed = carrying
      ? CARRY_MAXIMUM_YAW_SPEED_RADIANS_PER_SECOND
      : 1;
    this.#stoppingFrames = Math.ceil(
      Math.max(
        maximumPlanarSpeed / this.#linearAcceleration,
        maximumYawSpeed / this.#yawAcceleration
      ) / controlStepSeconds
        + this.#commandResponseHorizonSeconds / controlStepSeconds
    );
    this.#maximumStoppingFrames = this.#stoppingFrames + Math.ceil(
      NAVIGATION_STOP_SETTLE_BUDGET_SECONDS / controlStepSeconds
    );
    this.#contactConstraints = (input.contactConstraints ?? []).map((constraint) => (
      HumanoidContactConstraintSchema.parse(constraint)
    ));
    this.#graspTargets = input.graspTargets?.map((target) => ({ ...target })) ?? [];
    this.#carryTaskSpaceTargets = HumanoidCarryTaskSpaceTargetsSchema.parse(
      input.carryTaskSpaceTargets ?? []
    );
    this.#startRootPosition = progress
      ? { ...progress.start_root_position }
      : { ...this.#final.rootPosition };
    this.#segmentStartRootPosition = progress?.segment_start_root_position
      ? { ...progress.segment_start_root_position }
      : { ...this.#final.rootPosition };
    const finalTarget = this.#plan.waypoints.at(-1)!;
    this.#initialTargetDistance = Math.hypot(
      finalTarget.x - this.#startRootPosition.x,
      finalTarget.z - this.#startRootPosition.z
    );
    const initialFinalDistance = Math.hypot(
      finalTarget.x - this.#segmentStartRootPosition.x,
      finalTarget.z - this.#segmentStartRootPosition.z
    );
    this.#precisionArrival = initialFinalDistance <= (
      NAVIGATION_FINAL_WAYPOINT_TOLERANCE_METERS
        + NAVIGATION_POSITION_DISCRETIZATION_METERS
        + humanoidNavigationStoppingDistance({
          planarSpeedMetersPerSecond: maximumPlanarSpeed,
          maximumDecelerationMetersPerSecondSquared: this.#linearAcceleration,
          commandResponseHorizonSeconds: this.#brakingResponseHorizonSeconds
        })
    );
    const progressBudgetSpeed = this.#precisionArrival
      ? NAVIGATION_PRECISION_PROGRESS_BUDGET_SPEED_METERS_PER_SECOND
      : NAVIGATION_PROGRESS_BUDGET_SPEED_METERS_PER_SECOND;
    this.#maximumTravelFrames = (progress?.online_replan_count ?? 0) > 0
      ? (progress?.committed_frame_count ?? 0) + Math.ceil(
          Math.min(
            MAXIMUM_NAVIGATION_SECONDS,
            this.#plan.distance / progressBudgetSpeed
              + NAVIGATION_STARTUP_SECONDS
              + NAVIGATION_FINAL_CONVERGENCE_BUDGET_SECONDS
              + (this.#arrivalHeading === null
                ? 0
                : NAVIGATION_ARRIVAL_HEADING_BUDGET_SECONDS)
          ) / controlStepSeconds
        )
      : Math.ceil(
      Math.min(
        MAXIMUM_NAVIGATION_SECONDS,
        this.#plan.distance / progressBudgetSpeed
          + NAVIGATION_STARTUP_SECONDS
          + NAVIGATION_FINAL_CONVERGENCE_BUDGET_SECONDS
          + (this.#arrivalHeading === null
            ? 0
            : NAVIGATION_ARRIVAL_HEADING_BUDGET_SECONDS)
      ) / controlStepSeconds
        );
    this.#waypointIndex = progress?.waypoint_index
      ?? Math.min(1, this.#plan.waypoints.length - 1);
    this.#frames = progress?.committed_frame_count ?? 0;
    this.#stopFrames = progress?.stopping_frame_count ?? 0;
    this.#stopSettledFrames = progress?.stopping_settled_frame_count ?? 0;
    this.#arrivalPositionLatched = progress?.arrival_position_latched ?? false;
    this.#assertProgress();
  }

  get done(): boolean {
    return this.#result !== undefined;
  }

  get reference(): HumanoidReference {
    return this.#reference;
  }

  skillProgressEvidence(): HumanoidSkillProgressEvidence {
    const physicalCompletionRatio = this.#initialTargetDistance <= 1e-9
      ? this.#finalWaypointSatisfied() ? 1 : 0
      : Math.min(1, Math.max(
          0,
          1 - this.#finalWaypointDistance() / this.#initialTargetDistance
        ));
    const maximumFrames = Math.max(
      1,
      this.#maximumTravelFrames + this.#maximumStoppingFrames
    );
    return {
      elapsedRatio: Math.min(1, this.#frames / maximumFrames),
      physicalCompletionRatio,
      satisfiedPredicateRatio: this.#result?.completed ? 1 : 0,
      stableSteps: this.#stopSettledFrames,
      requiredStableSteps: NAVIGATION_STOP_SETTLED_STEPS,
      confidence: 1
    };
  }

  checkpoint(): HumanoidNavigationExecutionProgress {
    if (this.#pendingFrame) {
      throw new Error("Cannot checkpoint an uncommitted humanoid navigation frame");
    }
    return HumanoidNavigationExecutionProgressSchema.parse({
      version: 1,
      start_root_position: this.#startRootPosition,
      segment_start_root_position: this.#segmentStartRootPosition,
      waypoint_index: this.#waypointIndex,
      committed_frame_count: this.#frames,
      online_replan_count: this.#onlineReplanCount,
      stopping_frame_count: this.#stopFrames,
      stopping_settled_frame_count: this.#stopSettledFrames,
      arrival_position_latched: this.#arrivalPositionLatched
    });
  }

  async step(
    simulation: HumanoidSimulation,
    authority?: { worldFrame: number; worldRevision: number }
  ): Promise<HumanoidNavigationExecutionStep> {
    const prepared = await this.prepareFrame(simulation, authority);
    return prepared
      ? this.commitPreparedFrame()
      : this.#terminalStep();
  }

  async prepareFrame(
    simulation: HumanoidSimulation,
    authority?: { worldFrame: number; worldRevision: number }
  ): Promise<HumanoidNavigationPreparedFrame | null> {
    if (this.#result) throw new Error("Humanoid navigation execution is already complete");
    if (this.#pendingFrame) {
      throw new Error("Humanoid navigation already has an uncommitted physical frame");
    }
    if (this.#waypointIndex === this.#plan.waypoints.length) {
      if (this.#arrivalPositionLatched
        && this.#finalWaypointDistance() > this.#finalAcceptedPositionTolerance()
          + NAVIGATION_ARRIVAL_POSITION_LATCH_HYSTERESIS_METERS) {
        this.#arrivalPositionLatched = false;
        this.#waypointIndex = this.#plan.waypoints.length - 1;
        this.#stopFrames = 0;
        this.#stopSettledFrames = 0;
      } else {
        if (this.#resolveSettledStop()) return null;
        if (this.#stopFrames >= this.#maximumStoppingFrames) {
          if (this.#physicallySettledWithinDeadband()) {
            this.#finish(true);
            return null;
          }
          this.#finish(false, this.#failedToSettleReason());
          return null;
        }
      }
    }
    while (this.#waypointIndex < this.#plan.waypoints.length) {
      const waypoint = this.#plan.waypoints[this.#waypointIndex]!;
      const dx = waypoint.x - this.#final.rootPosition.x;
      const dz = waypoint.z - this.#final.rootPosition.z;
      const distance = Math.hypot(dx, dz);
      if (this.#waypointIndex < this.#plan.waypoints.length - 1
        ? distance <= NAVIGATION_WAYPOINT_TOLERANCE_METERS
        : this.#finalWaypointSatisfied()) {
        this.#waypointIndex += 1;
        continue;
      }
      // A committed control step can enter the terminal envelope on the exact
      // budget boundary.  Evaluate that physical state before declaring a
      // timeout; otherwise a valid arrival is discarded one frame later.
      if (this.#frames >= this.#maximumTravelFrames) {
        this.#finish(
          false,
          `navigation_timeout:position=${point(this.#final.rootPosition)},target=${point(waypoint)}`
            + `;distance=${this.#finalWaypointDistance().toFixed(6)}`
            + `;accepted_distance=${this.#finalAcceptedPositionTolerance().toFixed(6)}`
            + `;heading_error=${this.#arrivalHeading === null
              ? "none"
              : Math.abs(humanoidNavigationArrivalHeadingError(
                  this.#arrivalHeading,
                  this.#final.rootPosition,
                  yawFromQuaternion(this.#final.rootRotation)
                )).toFixed(6)}`
            + `;command=${this.#reference.rootVelocity.map((value) => (
              value.toFixed(3)
            )).join(",")},yaw=${this.#reference.rootYawVelocity.toFixed(3)}`
            + `;support=${this.#final.balance.support}`
            + `;upright=${this.#final.balance.upright.toFixed(3)}`
        );
        return null;
      }
      const yaw = yawFromQuaternion(this.#final.rootRotation);
      const finalWaypoint = this.#waypointIndex === this.#plan.waypoints.length - 1;
      const acceptedPositionTolerance = finalWaypoint
        ? this.#finalAcceptedPositionTolerance()
        : NAVIGATION_FINAL_WAYPOINT_TOLERANCE_METERS;
      if (finalWaypoint
        && this.#arrivalHeading !== null
        && this.#arrivalPositionLatched
        && distance > acceptedPositionTolerance
          + NAVIGATION_ARRIVAL_POSITION_LATCH_HYSTERESIS_METERS) {
        this.#arrivalPositionLatched = false;
      }
      if (finalWaypoint
        && this.#arrivalHeading !== null
        && distance <= acceptedPositionTolerance) {
        this.#arrivalPositionLatched = true;
      }
      const aligningArrivalHeading = finalWaypoint
        && this.#arrivalPositionLatched
        && this.#arrivalHeading !== null;
      const arrivalHeadingReadyForPositionRecovery = aligningArrivalHeading
        && Math.abs(humanoidNavigationArrivalHeadingError(
          this.#arrivalHeading!,
          this.#final.rootPosition,
          yaw
        )) <= this.#arrivalHeading!.tolerance_radians
          + NAVIGATION_ARRIVAL_HEADING_RECOVERY_HYSTERESIS_RADIANS;
      if (finalWaypoint
        && this.#arrivalHeading === null
        && humanoidNavigationShouldBeginBraking({
          distanceMeters: distance,
          acceptedDistanceMeters: acceptedPositionTolerance,
          planarSpeedMetersPerSecond: this.#planarApproachSpeed(dx, dz, distance),
          ...((this.#carrying || this.#arrivalPositionLatched)
            ? {
                commandedPlanarSpeedMetersPerSecond: Math.hypot(
                  ...this.#reference.rootVelocity
                )
              }
            : {}),
          maximumDecelerationMetersPerSecondSquared: this.#linearAcceleration,
          commandResponseHorizonSeconds: this.#brakingResponseHorizonSeconds
        })) {
        this.#waypointIndex = this.#plan.waypoints.length;
        break;
      }
      const travelYaw = Math.atan2(dx, dz);
      const yawError = aligningArrivalHeading
        ? humanoidNavigationArrivalHeadingError(
            this.#arrivalHeading!,
            this.#final.rootPosition,
            yaw
          )
        : shortestLocomotionYawError(travelYaw, yaw);
      const localForward = dx * Math.sin(yaw) + dz * Math.cos(yaw);
      const localLateral = dx * Math.cos(yaw) - dz * Math.sin(yaw);
      const carrying = this.#carrying;
      let desiredForward = clamp(
        localForward * 1.4,
        carrying ? -CARRY_MAXIMUM_REVERSE_SPEED_METERS_PER_SECOND : -0.3,
        carrying ? CARRY_MAXIMUM_FORWARD_SPEED_METERS_PER_SECOND : 0.48
      );
      let desiredLateral = clamp(
        localLateral * 0.8,
        carrying ? -CARRY_MAXIMUM_LATERAL_SPEED_METERS_PER_SECOND : -0.22,
        carrying ? CARRY_MAXIMUM_LATERAL_SPEED_METERS_PER_SECOND : 0.22
      );
      const minimumPlanarSpeed = this.#minimumPlanarCommandSpeed();
      const desiredPlanarSpeed = Math.hypot(desiredForward, desiredLateral);
      if ((!aligningArrivalHeading || arrivalHeadingReadyForPositionRecovery)
        && distance > acceptedPositionTolerance
        && desiredPlanarSpeed > 1e-9
        && desiredPlanarSpeed < minimumPlanarSpeed) {
        const scale = minimumPlanarSpeed / desiredPlanarSpeed;
        desiredForward *= scale;
        desiredLateral *= scale;
      }
      desiredForward = clamp(
        desiredForward,
        carrying ? -CARRY_MAXIMUM_REVERSE_SPEED_METERS_PER_SECOND : -0.3,
        carrying ? CARRY_MAXIMUM_FORWARD_SPEED_METERS_PER_SECOND : 0.48
      );
      desiredLateral = clamp(
        desiredLateral,
        carrying ? -CARRY_MAXIMUM_LATERAL_SPEED_METERS_PER_SECOND : -0.22,
        carrying ? CARRY_MAXIMUM_LATERAL_SPEED_METERS_PER_SECOND : 0.22
      );
      if (carrying && finalWaypoint) {
        const safeApproachSpeed = Math.max(
          minimumPlanarSpeed,
          this.#maximumSafeApproachSpeed(
            distance,
            acceptedPositionTolerance
          )
        );
        const plannedSpeed = Math.hypot(desiredForward, desiredLateral);
        if (plannedSpeed > safeApproachSpeed && plannedSpeed > 1e-9) {
          const scale = safeApproachSpeed / plannedSpeed;
          desiredForward *= scale;
          desiredLateral *= scale;
        }
      }
      const linearStep = this.#linearAcceleration * this.#controlStepSeconds;
      const yawStep = this.#yawAcceleration * this.#controlStepSeconds;
      const desiredYawVelocity = aligningArrivalHeading
        ? this.#arrivalYawVelocityTarget(yawError)
        : clamp(
            yawError * 1.8,
            carrying ? -CARRY_MAXIMUM_YAW_SPEED_RADIANS_PER_SECOND : -1,
            carrying ? CARRY_MAXIMUM_YAW_SPEED_RADIANS_PER_SECOND : 1
          );
      this.#reference = targetReference(this.#reference, {
        rootVelocity: [
          approach(this.#reference.rootVelocity[0], desiredForward, linearStep),
          approach(this.#reference.rootVelocity[1], desiredLateral, linearStep)
        ],
        rootYawVelocity: approach(
          this.#reference.rootYawVelocity,
          desiredYawVelocity,
          yawStep
        )
      });
      return this.#preparePhysicalFrame(simulation, false, authority);
    }

    const stoppingVelocity = this.#stoppingStationKeepingVelocityTarget();
    const stoppingLinearStep = this.#linearAcceleration * this.#controlStepSeconds;
    this.#reference = targetReference(this.#reference, {
      rootVelocity: [
        approach(
          this.#reference.rootVelocity[0],
          stoppingVelocity[0],
          stoppingLinearStep
        ),
        approach(
          this.#reference.rootVelocity[1],
          stoppingVelocity[1],
          stoppingLinearStep
        )
      ],
      rootYawVelocity: approach(
        this.#reference.rootYawVelocity,
        this.#stoppingYawVelocityTarget(),
        this.#yawAcceleration * this.#controlStepSeconds
      )
    });
    return this.#preparePhysicalFrame(simulation, true, authority);
  }

  #stoppingStationKeepingVelocityTarget(): readonly [number, number] {
    const target = this.#plan.waypoints.at(-1)!;
    const dx = target.x - this.#final.rootPosition.x;
    const dz = target.z - this.#final.rootPosition.z;
    const distance = Math.hypot(dx, dz);
    if (this.#requestedPositionToleranceMeters !== null
      && distance <= this.#finalAcceptedPositionTolerance()) {
      return [0, 0];
    }
    const yaw = yawFromQuaternion(this.#final.rootRotation);
    let forward = (dx * Math.sin(yaw) + dz * Math.cos(yaw)) * 1.4;
    let lateral = (dx * Math.cos(yaw) - dz * Math.sin(yaw)) * 0.8;
    if (!this.#carrying && distance > this.#finalAcceptedPositionTolerance()) {
      const speed = Math.hypot(forward, lateral);
      const minimumSpeed = this.#minimumPlanarCommandSpeed();
      if (speed > 1e-9 && speed < minimumSpeed) {
        const scale = minimumSpeed / speed;
        forward *= scale;
        lateral *= scale;
      }
      return [
        clamp(forward, -0.3, 0.48),
        clamp(lateral, -0.22, 0.22)
      ];
    }
    return [
      clamp(
        forward,
        -NAVIGATION_STOP_POSITION_HOLD_COMMAND_METERS_PER_SECOND,
        NAVIGATION_STOP_POSITION_HOLD_COMMAND_METERS_PER_SECOND
      ),
      clamp(
        lateral,
        -NAVIGATION_STOP_POSITION_HOLD_COMMAND_METERS_PER_SECOND,
        NAVIGATION_STOP_POSITION_HOLD_COMMAND_METERS_PER_SECOND
      )
    ];
  }

  #stoppingYawVelocityTarget(): number {
    if (!this.#arrivalPositionLatched || this.#arrivalHeading === null) return 0;
    const error = humanoidNavigationArrivalHeadingError(
      this.#arrivalHeading,
      this.#final.rootPosition,
      yawFromQuaternion(this.#final.rootRotation)
    );
    // Once the physical heading satisfies the declared arrival contract, stop
    // commanding rotation. Continuing to servo a sub-tolerance error makes the
    // stop gate require a near-zero yaw command while the controller is still
    // deliberately producing one, so an otherwise valid arrival can never
    // accumulate settled frames.
    if (this.#arrivalHeadingCaptured(error)) return 0;
    return this.#arrivalYawVelocityTarget(error);
  }

  #arrivalHeadingCaptured(error: number): boolean {
    if (this.#arrivalHeading === null) return true;
    // Capture the learned controller inside the declared tolerance rather
    // than exactly on its boundary.  Otherwise the yaw command decays after
    // crossing the limit and the physical response drifts a few milliradians
    // back outside it, creating an endless rotate/stop oscillation.
    const captureTolerance = Math.max(
      0.03,
      this.#arrivalHeading.tolerance_radians
        - NAVIGATION_ARRIVAL_HEADING_SETTLE_MARGIN_RADIANS
    );
    return Math.abs(error) <= captureTolerance;
  }

  #arrivalYawVelocityTarget(error: number): number {
    const maximum = this.#carrying
      ? CARRY_MAXIMUM_YAW_SPEED_RADIANS_PER_SECOND
      : 1;
    const minimum = this.#carrying
      ? CARRY_MINIMUM_EFFECTIVE_ARRIVAL_YAW_RADIANS_PER_SECOND
      : this.#minimumEffectiveYawSpeed;
    const proportional = clamp(error * 1.8, -maximum, maximum);
    if (Math.abs(proportional) >= minimum) return proportional;
    // The locomotion policy has a small-yaw command deadband.  Staying below
    // it leaves a stable nonzero heading error forever even though the robot
    // remains upright and positionally settled.  Hold the minimum effective
    // command until the declared heading tolerance is physically crossed;
    // the normal stop gate then ramps the command back to zero.
    return Math.sign(error) * minimum;
  }

  #maximumSafeApproachSpeed(
    distance: number,
    acceptedDistance: number
  ): number {
    const brakingDistance = Math.max(0, distance - acceptedDistance);
    const response = this.#brakingResponseHorizonSeconds;
    const acceleration = this.#linearAcceleration;
    return acceleration * (
      Math.sqrt(
        response * response + 2 * brakingDistance / acceleration
      ) - response
    );
  }

  commitPreparedFrame(externalFailure?: string): HumanoidNavigationExecutionStep {
    const pending = this.#pendingFrame;
    if (!pending) throw new Error("Humanoid navigation has no prepared physical frame");
    this.#pendingFrame = undefined;
    this.#final = pending.snapshot;
    this.#frames += 1;
    if (pending.stopping) {
      this.#stopFrames += 1;
      this.#stopSettledFrames = this.#stoppingFrameIsSettled()
        ? this.#stopSettledFrames + 1
        : 0;
    }
    const blockedContacts = blockedHumanoidContacts(
      this.#final,
      this.#contactConstraints
    );
    const failure = this.#final.fallen
      ? pending.stopping ? "fallen_while_stopping" : "fallen"
      : blockedContacts.length > 0
        ? pending.stopping
          ? `contact_while_stopping:${environmentContact(
              this.#final,
              this.#contactConstraints
            )}`
          : environmentContact(this.#final, this.#contactConstraints)
        : externalFailure;
    if (failure) {
      this.#finish(
        false,
        failure,
        blockedContacts.length > 0
          ? humanoidNavigationCollisionEvidence(
              this.#final,
              this.#contactConstraints
            )
          : undefined
      );
    }
    if (!this.#result && pending.stopping) {
      if (!this.#resolveSettledStop()
        && this.#stopFrames >= this.#maximumStoppingFrames) {
        if (this.#physicallySettledWithinDeadband()) {
          this.#finish(true);
        } else {
          this.#finish(false, this.#failedToSettleReason());
        }
      }
    }
    return {
      snapshot: this.#final,
      waypointIndex: pending.waypointIndex,
      done: this.#result !== undefined
    };
  }

  result(): HumanoidNavigationExecutionResult {
    if (this.#pendingFrame) {
      throw new Error("Humanoid navigation result cannot include an uncommitted frame");
    }
    if (!this.#result) throw new Error("Humanoid navigation execution is not complete");
    return structuredClone(this.#result);
  }

  async #preparePhysicalFrame(
    simulation: HumanoidSimulation,
    stopping: boolean,
    authority?: { worldFrame: number; worldRevision: number }
  ): Promise<HumanoidNavigationPreparedFrame> {
    this.#reference = applyHumanoidCarryTaskSpaceServo({
      simulation,
      reference: this.#reference,
      targets: this.#carryTaskSpaceTargets,
      maximumReferenceCorrectionRadians:
        CARRY_MAXIMUM_WRIST_REFERENCE_CORRECTION_RADIANS
    });
    if (this.#graspTargets.length > 0) {
      const snapshot = simulation.snapshot();
      const requestedJointTargets = simulation.handJointCommandTargets();
      const controlled = contactAwareG1GraspJointTargets({
        requestedJointTargets,
        snapshot,
        targets: this.#graspTargets
      });
      simulation.applyHandServoJointTargets(controlled.jointTargets);
    }
    const snapshot = await simulation.step(this.#reference, {
      trackedJointPolicyCommand: this.#carrying ? "neutral" : "measured",
      noslipIterations: g1CarriedGraspRequiresNoslip({
        snapshot: simulation.snapshot(),
        targets: this.#graspTargets
      }) ? CARRY_NOSLIP_SOLVER_ITERATIONS : 0,
      taskCommand: HumanoidEmbodiedSkillCallSchema.parse({
        protocol: "humanoid-embodied-skill-call-v2",
        identity: this.#skillIdentity,
        authority: {
          source: this.#skillIdentity.runtimeKind === "semantic_skill"
            ? "agent_harness"
            : "deterministic_runtime",
          ...(authority ?? {
            worldFrame: this.#skillIdentity.observedFrame + this.#frames,
            worldRevision:
              this.#skillIdentity.observedWorldRevision + this.#frames
          })
        },
        window: {
          mode: "autonomous_closed_loop",
          replanPolicy: "event_driven",
          controlStepSeconds: this.#controlStepSeconds,
          ...this.#skillWindow()
        },
        requestedCapabilities: humanoidControllerTaskCapabilities(
          this.#reference,
          [
            "locomotion",
            ...(this.#carrying ? ["contact_rich_manipulation" as const] : []),
            ...(new Set(this.#graspTargets.map(({ hand }) => hand)).size > 1
              ? ["bimanual_manipulation" as const]
              : [])
          ]
        ),
        command: {
          baseTwist: {
            forwardMetersPerSecond: this.#reference.rootVelocity[0],
            lateralMetersPerSecond: this.#reference.rootVelocity[1],
            yawRadiansPerSecond: this.#reference.rootYawVelocity
          },
          rootHeightMeters: this.#reference.rootHeight,
          leftWristPositionPelvis: navigationWristCommandInPelvis(
            this.#carryTaskSpaceTargets,
            "left_wrist_yaw_link",
            this.#final
          ),
          rightWristPositionPelvis: navigationWristCommandInPelvis(
            this.#carryTaskSpaceTargets,
            "right_wrist_yaw_link",
            this.#final
          ),
          endEffectors: this.#carryTaskSpaceTargets.map((target) => ({
            body: target.body,
            frame: target.frame,
            position: { ...target.position },
            tolerance: target.tolerance,
            orientation: { ...target.orientation },
            orientationTolerance: target.orientationTolerance
          })),
          grasps: this.#graspTargets.map((target) => ({
            objectId: target.objectId,
            hand: target.hand,
            minimumNormalForceN: target.minimumNormalForceN,
            minimumDistinctContactSurfaces:
              target.minimumDistinctContactSurfaces ?? 1
          }))
        },
        contract: {
          protocol: "humanoid-embodied-navigation-contract-v1",
          target: { ...this.#plan.waypoints.at(-1)! },
          positionTolerance: this.#finalAcceptedPositionTolerance(),
          heading: this.#arrivalHeading?.type === "face_point"
            ? {
                type: "face_point" as const,
                target: { ...this.#arrivalHeading.target },
                toleranceRadians: this.#arrivalHeading.tolerance_radians
              }
            : this.#arrivalHeading?.type === "yaw"
              ? {
                  type: "yaw" as const,
                  yawRadians: this.#arrivalHeading.yaw_radians,
                  toleranceRadians: this.#arrivalHeading.tolerance_radians
                }
              : null
        },
        safety: {
          authorizedContacts: this.#contactConstraints.map((contact) => ({
            ...contact
          })),
          stopOnFall: true,
          stopOnUnauthorizedContact: true,
          stopOnContractViolation: true
        },
        feedback: {
          mode: "event_driven",
          progressDelta: 0.1,
          events: [
            "accepted",
            "progress",
            "succeeded",
            "failed",
            "interrupted",
            "environment_changed"
          ]
        }
      }),
      ...(this.#policyFrameSink
        ? { policyFrameSink: this.#policyFrameSink }
        : {})
    });
    const prepared = {
      snapshot,
      waypointIndex: Math.min(
        this.#waypointIndex,
        this.#plan.waypoints.length - 1
      ),
      stopping
    };
    this.#pendingFrame = prepared;
    return prepared;
  }

  #terminalStep(): HumanoidNavigationExecutionStep {
    if (!this.#result) {
      throw new Error("Humanoid navigation produced no physical frame or terminal result");
    }
    return {
      waypointIndex: Math.min(
        this.#waypointIndex,
        this.#plan.waypoints.length - 1
      ),
      done: true
    };
  }

  #finish(
    completed: boolean,
    reason?: string,
    blockingContacts?: HumanoidNavigationCollisionEvidence[]
  ): void {
    if (completed) this.#reference = stationaryHumanoidReference(this.#reference);
    this.#result = {
      completed,
      ...(reason ? { reason } : {}),
      ...(blockingContacts && blockingContacts.length > 0
        ? { blockingContacts: structuredClone(blockingContacts) }
        : {}),
      frames: this.#frames,
      reference: this.#reference,
      final: this.#final,
      travelledDistance: Math.hypot(
        this.#final.rootPosition.x - this.#startRootPosition.x,
        this.#final.rootPosition.z - this.#startRootPosition.z
      )
    };
  }

  #finalWaypointDistance(): number {
    const target = this.#plan.waypoints.at(-1)!;
    return Math.hypot(
      target.x - this.#final.rootPosition.x,
      target.z - this.#final.rootPosition.z
    );
  }

  #resolveSettledStop(): boolean {
    if (this.#stopFrames < this.#stoppingFrames
      || this.#stopSettledFrames < NAVIGATION_STOP_SETTLED_STEPS) {
      return false;
    }
    if (this.#finalWaypointSatisfied()) {
      this.#finish(true);
      return true;
    }
    this.#waypointIndex = this.#plan.waypoints.length - 1;
    this.#stopFrames = 0;
    this.#stopSettledFrames = 0;
    return false;
  }

  #stoppingFrameIsSettled(): boolean {
    const pelvis = this.#final.links?.pelvis;
    const planarSpeed = pelvis
      ? Math.hypot(pelvis.linearVelocity.x, pelvis.linearVelocity.z)
      : 0;
    const yawSpeed = Math.abs(pelvis?.angularVelocity.y ?? 0);
    return planarSpeed
        <= NAVIGATION_STOP_PLANAR_SPEED_TOLERANCE_METERS_PER_SECOND
      && yawSpeed <= NAVIGATION_STOP_YAW_SPEED_TOLERANCE_RADIANS_PER_SECOND
      && Math.hypot(...this.#reference.rootVelocity)
        <= NAVIGATION_STOP_COMMAND_TOLERANCE_METERS_PER_SECOND;
  }

  #failedToSettleReason(): string {
    const pelvis = this.#final.links?.pelvis;
    const deadbandAcceptedDistance = this.#boundedFinalPositionTolerance(Math.max(
      this.#baseFinalPositionTolerance(),
      NAVIGATION_PHYSICAL_DEADBAND_TOLERANCE_METERS
    )) + NAVIGATION_POSITION_DISCRETIZATION_METERS;
    return "navigation_failed_to_settle"
      + `:position=${point(this.#final.rootPosition)}`
      + `;distance=${this.#finalWaypointDistance().toFixed(6)}`
      + `;pelvis_speed=${pelvis
        ? Math.hypot(pelvis.linearVelocity.x, pelvis.linearVelocity.z).toFixed(6)
        : "unavailable"}`
      + `;pelvis_yaw_speed=${pelvis
        ? Math.abs(pelvis.angularVelocity.y).toFixed(6)
        : "unavailable"}`
      + `;command=${this.#reference.rootVelocity.map((value) => (
        value.toFixed(3)
      )).join(",")}`
      + `;deadband_accepted_distance=${deadbandAcceptedDistance.toFixed(6)}`
      + `;heading_error=${this.#arrivalHeading === null
        ? "none"
        : Math.abs(humanoidNavigationArrivalHeadingError(
            this.#arrivalHeading,
            this.#final.rootPosition,
            yawFromQuaternion(this.#final.rootRotation)
          )).toFixed(6)}`
      + `;yaw_command=${this.#reference.rootYawVelocity.toFixed(6)}`;
  }

  #planarApproachSpeed(dx: number, dz: number, distance: number): number {
    const commandedSpeed = Math.hypot(...this.#reference.rootVelocity);
    const pelvisVelocity = this.#final.links?.pelvis?.linearVelocity;
    if (!pelvisVelocity || distance <= 1e-9) {
      return commandedSpeed + 1e-9 < this.#minimumPlanarCommandSpeed()
        ? 0
        : commandedSpeed;
    }
    return Math.max(
      0,
      (pelvisVelocity.x * dx + pelvisVelocity.z * dz) / distance
    );
  }

  #minimumPlanarCommandSpeed(): number {
    if (!this.#carrying && !this.#precisionArrival) return 0;
    return Math.max(
      this.#carrying ? 0.07 : 0.1,
      this.#minimumEffectivePlanarSpeed > 0
        ? this.#minimumEffectivePlanarSpeed
          + NAVIGATION_EFFECTIVE_COMMAND_MARGIN_METERS_PER_SECOND
        : 0
    );
  }

  #finalWaypointSatisfied(): boolean {
    const currentDistance = this.#finalWaypointDistance();
    if (currentDistance > this.#finalAcceptedPositionTolerance()) return false;
    if (this.#arrivalHeading !== null
      && !this.#arrivalHeadingCaptured(humanoidNavigationArrivalHeadingError(
        this.#arrivalHeading,
        this.#final.rootPosition,
        yawFromQuaternion(this.#final.rootRotation)
      ))) return false;
    return true;
  }

  #finalPositionTolerance(): number {
    return this.#boundedFinalPositionTolerance(
      this.#baseFinalPositionTolerance()
    );
  }

  #boundedFinalPositionTolerance(baseTolerance: number): number {
    if (this.#requestedPositionToleranceMeters !== null) {
      return baseTolerance;
    }
    const target = this.#plan.waypoints.at(-1)!;
    const initialDistance = Math.hypot(
      target.x - this.#startRootPosition.x,
      target.z - this.#startRootPosition.z
    );
    if (initialDistance <= NAVIGATION_ALREADY_AT_TARGET_TOLERANCE_METERS) {
      return this.#baseFinalPositionTolerance();
    }
    const requiredProgress = Math.min(
      initialDistance,
      Math.max(
        NAVIGATION_MINIMUM_SHORT_ROUTE_PROGRESS_METERS,
        initialDistance * NAVIGATION_MINIMUM_SHORT_ROUTE_PROGRESS_RATIO
      )
    );
    return Math.min(
      baseTolerance,
      initialDistance - requiredProgress
    );
  }

  #physicallySettledWithinDeadband(): boolean {
    const pelvis = this.#final.links?.pelvis;
    const planarSpeed = pelvis
      ? Math.hypot(pelvis.linearVelocity.x, pelvis.linearVelocity.z)
      : 0;
    const yawSpeed = Math.abs(pelvis?.angularVelocity.y ?? 0);
    const acceptedDistance = this.#boundedFinalPositionTolerance(Math.max(
      this.#baseFinalPositionTolerance(),
      NAVIGATION_PHYSICAL_DEADBAND_TOLERANCE_METERS
    )) + NAVIGATION_POSITION_DISCRETIZATION_METERS;
    return this.#finalWaypointDistance() <= acceptedDistance
      && humanoidNavigationArrivalHeadingSatisfied(
        this.#arrivalHeading,
        this.#final.rootPosition,
        yawFromQuaternion(this.#final.rootRotation)
      )
      && planarSpeed
        <= NAVIGATION_STOP_PLANAR_SPEED_TOLERANCE_METERS_PER_SECOND
      && yawSpeed <= NAVIGATION_STOP_YAW_SPEED_TOLERANCE_RADIANS_PER_SECOND
      && Math.hypot(...this.#reference.rootVelocity)
        <= Math.max(
          NAVIGATION_STOP_COMMAND_TOLERANCE_METERS_PER_SECOND,
          NAVIGATION_PHYSICAL_DEADBAND_COMMAND_TOLERANCE_METERS_PER_SECOND,
          this.#minimumPlanarCommandSpeed()
            + NAVIGATION_EFFECTIVE_COMMAND_MARGIN_METERS_PER_SECOND
        );
  }

  #baseFinalPositionTolerance(): number {
    const physicalMinimum = this.#carrying
      ? CARRY_NAVIGATION_FINAL_WAYPOINT_TOLERANCE_METERS
      : NAVIGATION_FINAL_WAYPOINT_TOLERANCE_METERS;
    return Math.max(
      physicalMinimum,
      this.#requestedPositionToleranceMeters ?? physicalMinimum
    );
  }

  #finalAcceptedPositionTolerance(): number {
    return this.#finalPositionTolerance() + NAVIGATION_POSITION_DISCRETIZATION_METERS;
  }

  #assertProgress(): void {
    if (this.#plan.waypoints.length === 0) {
      throw new Error("Humanoid navigation execution requires at least one waypoint");
    }
    if (this.#waypointIndex > this.#plan.waypoints.length) {
      throw new Error("Humanoid navigation waypoint progress exceeds its plan");
    }
    const stopping = this.#waypointIndex === this.#plan.waypoints.length;
    if ((!stopping && (this.#stopFrames !== 0 || this.#stopSettledFrames !== 0))
      || this.#stopFrames > this.#maximumStoppingFrames
      || this.#stopSettledFrames > this.#stopFrames) {
      throw new Error("Humanoid navigation stopping progress is inconsistent with its route");
    }
    if (this.#arrivalPositionLatched && this.#arrivalHeading === null) {
      throw new Error("Humanoid navigation position latch requires an arrival heading");
    }
    if (this.#frames > this.#maximumTravelFrames + this.#maximumStoppingFrames) {
      throw new Error("Humanoid navigation progress exceeds its physical frame limit");
    }
  }
}

export async function previewHumanoidNavigation(
  plan: NavigationPlan,
  reference: HumanoidReference,
  simulation: HumanoidSimulation,
  graspRegistry: HumanoidGraspRegistry,
  startFrame: number,
  startWorldRevision: number,
  carriedObjectBindings: HumanoidCarriedObjectBindingSet,
  carriedObjectTaskSpaceTargets: readonly HumanoidCarryTaskSpaceTarget[],
  arrivalHeading: HumanoidNavigationArrivalHeading | null = null,
  acceptedPositionToleranceMeters: number | null = null
): Promise<ReturnType<HumanoidNavigationExecution["result"]>> {
  if (graspRegistry.lastFrame !== startFrame) {
    throw new Error("Humanoid navigation preview grasp registry is not frame-aligned");
  }
  const execution = new HumanoidNavigationExecution({
    plan,
    reference,
    simulation,
    contactConstraints: humanoidCarriedObjectContactConstraints(
      carriedObjectBindings
    ),
    graspTargets: contactAwareG1GraspTargetsForBindings({
      bindings: carriedObjectBindings.bindings,
      graspRegistry
    }),
    carryTaskSpaceTargets: carriedObjectTaskSpaceTargets,
    arrivalHeading,
    acceptedPositionToleranceMeters
  });
  let frame = startFrame;
  let worldRevision = startWorldRevision;
  while (!execution.done) {
    const prepared = await execution.prepareFrame(simulation);
    if (!prepared) continue;
    frame += 1;
    worldRevision += 1;
    graspRegistry.observe(frame, prepared.snapshot);
    const continuation = humanoidCarriedObjectContinuationEvidence({
      state: carriedObjectBindings,
      registry: graspRegistry,
      currentFrame: frame,
      currentWorldRevision: worldRevision
    });
    const unauthorized = humanoidCarriedObjectUnauthorizedContacts(
      carriedObjectBindings,
      prepared.snapshot.contacts
    );
    execution.commitPreparedFrame(
      carryNavigationFailure(continuation, unauthorized, prepared.snapshot)
    );
  }
  return execution.result();
}

export function carryNavigationFailure(
  continuation: ReturnType<typeof humanoidCarriedObjectContinuationEvidence>,
  unauthorized: ReturnType<typeof humanoidCarriedObjectUnauthorizedContacts>,
  snapshot: HumanoidSimulationSnapshot
): string | undefined {
  const failed = continuation.bindings.find((binding) => !binding.continued);
  if (failed) {
    const contacts = g1HandObjectContacts(snapshot.contacts, failed.object_id)
      .filter((contact) => contact.hand === failed.hand)
      .map((contact) => `${contact.handLink}@${contact.normalForce.toFixed(3)}N`)
      .join(",");
    const wrist = snapshot.links[
      failed.hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
    ];
    const object = snapshot.objects[failed.object_id];
    const physicalState = object
      ? {
          relative_translation: rotateVector(
            inverseQuaternion(wrist.rotation),
            subtract(object.position, wrist.position)
          ),
          relative_rotation: normalizeQuaternion(multiplyQuaternion(
            inverseQuaternion(wrist.rotation),
            object.rotation
          )),
          relative_angular_velocity: subtract(
            object.angularVelocity,
            wrist.angularVelocity
          ),
          wrist_rotation: wrist.rotation,
          object_position: object.position,
          contacts: g1HandObjectContacts(snapshot.contacts, failed.object_id)
            .filter((contact) => contact.hand === failed.hand)
            .map((contact) => ({
              digit: contact.handLink,
              position: contact.position,
              normal_from_hand: contact.normalFromHand,
              normal_force_n: contact.normalForce
            }))
        }
      : null;
    return `carried_grasp_lost:${failed.object_id}:${failed.hand}:${
      failed.failure ?? "unknown"
    }:${failed.detail ?? "no_detail"}; contacts=${contacts || "none"}; state=${
      JSON.stringify(physicalState)
    }`;
  }
  const collision = unauthorized[0];
  return collision
    ? `carried_object_collision:${collision.object_id}:${collision.counterpart_kind}`
    : undefined;
}

function navigationWristCommandInPelvis(
  targets: readonly HumanoidCarryTaskSpaceTarget[],
  body: "left_wrist_yaw_link" | "right_wrist_yaw_link",
  snapshot: HumanoidSimulationSnapshot
): Vec3 | null {
  const target = targets.find((candidate) => candidate.body === body);
  if (!target || target.frame === "torso") return null;
  if (target.frame === "pelvis") return { ...target.position };
  return rotateVector(
    inverseQuaternion(snapshot.rootRotation),
    subtract(target.position, snapshot.rootPosition)
  );
}

function environmentContact(
  snapshot: HumanoidSimulationSnapshot,
  constraints: readonly HumanoidContactConstraint[]
): string {
  const blocked = blockedHumanoidContacts(snapshot, constraints);
  const base = `environment_contact:${blocked.map((contact) => (
    `${"body" in contact ? contact.body : contact.handSurface}`
      + `:${contact.objectId ?? contact.solidId ?? "environment"}`
  )).join(",")}`;
  const environmentBodies = new Set(blocked.flatMap((contact) => (
    "body" in contact && contact.objectId === null ? [contact.body] : []
  )));
  if (environmentBodies.size === 0) return base;
  const evidence = snapshot.contacts.filter((contact) => (
    contact.firstObject === null
      && contact.secondObject === null
      && (contact.firstBody !== null && environmentBodies.has(contact.firstBody)
        || contact.secondBody !== null && environmentBodies.has(contact.secondBody))
  )).map((contact) => (
    `${contact.firstBody ?? contact.secondBody}`
      + `@${point(contact.position)}`
      + `/${contact.normalForce.toFixed(3)}N`
  ));
  return `${base};time=${snapshot.simulatedTime.toFixed(3)};contacts=${
    evidence.join(",") || "none"
  }`;
}

function normalizeAngle(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function shortestLocomotionYawError(travelYaw: number, currentYaw: number): number {
  const forward = normalizeAngle(travelYaw - currentYaw);
  const reverse = normalizeAngle(travelYaw + Math.PI - currentYaw);
  return Math.abs(reverse) < Math.abs(forward) ? reverse : forward;
}

function approach(current: number, target: number, maximumDelta: number): number {
  return current + clamp(target - current, -maximumDelta, maximumDelta);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function navigationPositionTolerance(value: number): number {
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new Error("Navigation position tolerance must be within (0, 1] meters");
  }
  return value;
}

function point(value: Vec3): string {
  return `${value.x.toFixed(3)},${value.y.toFixed(3)},${value.z.toFixed(3)}`;
}
