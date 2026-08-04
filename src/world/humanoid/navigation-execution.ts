import { z } from "zod";
import { Vec3Schema, type Vec3 } from "../../domain/schema.js";
import type { NavigationPlan } from "../navigation.js";
import {
  blockedHumanoidContacts,
  HumanoidContactConstraintSchema,
  type HumanoidContactConstraint
} from "./motion-plan.js";
import { targetReference, type HumanoidReference } from "./reference.js";
import type {
  HumanoidSimulation,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import {
  contactAwareG1GraspTargetsForBindings,
  contactAwareG1GraspJointTargets,
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
  subtract
} from "../geometry.js";

const NAVIGATION_PROGRESS_BUDGET_SPEED_METERS_PER_SECOND = 0.05;
const NAVIGATION_STARTUP_SECONDS = 10;
const MAXIMUM_NAVIGATION_SECONDS = 90;
const MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED = 1;
const MAXIMUM_YAW_ACCELERATION_RADIANS_PER_SECOND_SQUARED = 2;
const CARRY_MAXIMUM_FORWARD_SPEED_METERS_PER_SECOND = 0.22;
const CARRY_MAXIMUM_REVERSE_SPEED_METERS_PER_SECOND = 0.16;
const CARRY_MAXIMUM_LATERAL_SPEED_METERS_PER_SECOND = 0.12;
const CARRY_MAXIMUM_YAW_SPEED_RADIANS_PER_SECOND = 0.55;
const CARRY_MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED = 0.3;
const CARRY_MAXIMUM_YAW_ACCELERATION_RADIANS_PER_SECOND_SQUARED = 0.7;
const CARRY_MAXIMUM_WRIST_REFERENCE_CORRECTION_RADIANS = 0.06;

export interface HumanoidNavigationExecutionResult {
  completed: boolean;
  reason?: string;
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
  waypoint_index: z.number().int().nonnegative(),
  committed_frame_count: z.number().int().nonnegative(),
  stopping_frame_count: z.number().int().nonnegative()
}).strict().superRefine((progress, context) => {
  if (progress.stopping_frame_count > progress.committed_frame_count) {
    context.addIssue({
      code: "custom",
      path: ["stopping_frame_count"],
      message: "Navigation stopping progress cannot exceed committed physical frames"
    });
  }
});

export type HumanoidNavigationExecutionProgress = z.infer<
  typeof HumanoidNavigationExecutionProgressSchema
>;

export class HumanoidNavigationExecution {
  readonly #plan: NavigationPlan;
  readonly #startRootPosition: Vec3;
  readonly #maximumTravelFrames: number;
  readonly #stoppingFrames: number;
  readonly #controlStepSeconds: number;
  readonly #contactConstraints: readonly HumanoidContactConstraint[];
  readonly #graspTargets: readonly G1ContactAwareGraspTarget[];
  readonly #carryTaskSpaceTargets: readonly HumanoidCarryTaskSpaceTarget[];
  #reference: HumanoidReference;
  #final: HumanoidSimulationSnapshot;
  #waypointIndex: number;
  #frames = 0;
  #stopFrames = 0;
  #result: HumanoidNavigationExecutionResult | undefined;
  #pendingFrame: HumanoidNavigationPreparedFrame | undefined;

  constructor(input: {
    plan: NavigationPlan;
    reference: HumanoidReference;
    simulation: HumanoidSimulation;
    progress?: HumanoidNavigationExecutionProgress;
    contactConstraints?: readonly HumanoidContactConstraint[];
    graspTargets?: readonly G1ContactAwareGraspTarget[];
    carryTaskSpaceTargets?: readonly HumanoidCarryTaskSpaceTarget[];
  }) {
    this.#plan = structuredClone(input.plan);
    this.#reference = input.reference;
    const progress = input.progress
      ? HumanoidNavigationExecutionProgressSchema.parse(input.progress)
      : undefined;
    const carrying = (input.graspTargets?.length ?? 0) > 0
      || (input.carryTaskSpaceTargets?.length ?? 0) > 0;
    if (!carrying && (!progress || progress.committed_frame_count === 0)) {
      input.simulation.resetController(input.reference);
    }
    this.#final = input.simulation.snapshot();
    const controlStepSeconds = input.simulation.controllerDescriptor().controlStepSeconds;
    this.#controlStepSeconds = controlStepSeconds;
    this.#maximumTravelFrames = Math.ceil(
      Math.min(
        MAXIMUM_NAVIGATION_SECONDS,
        this.#plan.distance / NAVIGATION_PROGRESS_BUDGET_SPEED_METERS_PER_SECOND
          + NAVIGATION_STARTUP_SECONDS
      ) / controlStepSeconds
    );
    this.#stoppingFrames = Math.ceil(0.6 / controlStepSeconds);
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
    this.#waypointIndex = progress?.waypoint_index
      ?? Math.min(1, this.#plan.waypoints.length - 1);
    this.#frames = progress?.committed_frame_count ?? 0;
    this.#stopFrames = progress?.stopping_frame_count ?? 0;
    this.#assertProgress();
  }

  get done(): boolean {
    return this.#result !== undefined;
  }

  get reference(): HumanoidReference {
    return this.#reference;
  }

  checkpoint(): HumanoidNavigationExecutionProgress {
    if (this.#pendingFrame) {
      throw new Error("Cannot checkpoint an uncommitted humanoid navigation frame");
    }
    return HumanoidNavigationExecutionProgressSchema.parse({
      version: 1,
      start_root_position: this.#startRootPosition,
      waypoint_index: this.#waypointIndex,
      committed_frame_count: this.#frames,
      stopping_frame_count: this.#stopFrames
    });
  }

  async step(simulation: HumanoidSimulation): Promise<HumanoidNavigationExecutionStep> {
    const prepared = await this.prepareFrame(simulation);
    return prepared
      ? this.commitPreparedFrame()
      : this.#terminalStep();
  }

  async prepareFrame(
    simulation: HumanoidSimulation
  ): Promise<HumanoidNavigationPreparedFrame | null> {
    if (this.#result) throw new Error("Humanoid navigation execution is already complete");
    if (this.#pendingFrame) {
      throw new Error("Humanoid navigation already has an uncommitted physical frame");
    }
    if (this.#waypointIndex === this.#plan.waypoints.length
      && this.#stopFrames >= this.#stoppingFrames) {
      this.#finish(true);
      return null;
    }
    while (this.#waypointIndex < this.#plan.waypoints.length) {
      if (this.#frames >= this.#maximumTravelFrames) {
        const waypoint = this.#plan.waypoints[this.#waypointIndex]!;
        this.#finish(
          false,
          `navigation_timeout:position=${point(this.#final.rootPosition)},target=${point(waypoint)}`
            + `;command=${this.#reference.rootVelocity.map((value) => (
              value.toFixed(3)
            )).join(",")},yaw=${this.#reference.rootYawVelocity.toFixed(3)}`
            + `;support=${this.#final.balance.support}`
            + `;upright=${this.#final.balance.upright.toFixed(3)}`
        );
        return null;
      }
      const waypoint = this.#plan.waypoints[this.#waypointIndex]!;
      const dx = waypoint.x - this.#final.rootPosition.x;
      const dz = waypoint.z - this.#final.rootPosition.z;
      const distance = Math.hypot(dx, dz);
      if (distance <= 0.18) {
        this.#waypointIndex += 1;
        continue;
      }
      const yaw = yawFromQuaternion(this.#final.rootRotation);
      const travelYaw = Math.atan2(dx, dz);
      const yawError = shortestLocomotionYawError(travelYaw, yaw);
      const localForward = dx * Math.sin(yaw) + dz * Math.cos(yaw);
      const localLateral = dx * Math.cos(yaw) - dz * Math.sin(yaw);
      const carrying = this.#graspTargets.length > 0;
      const desiredForward = clamp(
        localForward * 1.4,
        carrying ? -CARRY_MAXIMUM_REVERSE_SPEED_METERS_PER_SECOND : -0.3,
        carrying ? CARRY_MAXIMUM_FORWARD_SPEED_METERS_PER_SECOND : 0.48
      );
      const desiredLateral = clamp(
        localLateral * 0.8,
        carrying ? -CARRY_MAXIMUM_LATERAL_SPEED_METERS_PER_SECOND : -0.22,
        carrying ? CARRY_MAXIMUM_LATERAL_SPEED_METERS_PER_SECOND : 0.22
      );
      const linearStep = (carrying
        ? CARRY_MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED
        : MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED)
        * this.#controlStepSeconds;
      const yawStep = (carrying
        ? CARRY_MAXIMUM_YAW_ACCELERATION_RADIANS_PER_SECOND_SQUARED
        : MAXIMUM_YAW_ACCELERATION_RADIANS_PER_SECOND_SQUARED)
        * this.#controlStepSeconds;
      this.#reference = targetReference(this.#reference, {
        rootVelocity: [
          approach(this.#reference.rootVelocity[0], desiredForward, linearStep),
          approach(this.#reference.rootVelocity[1], desiredLateral, linearStep)
        ],
        rootYawVelocity: approach(
          this.#reference.rootYawVelocity,
          clamp(
            yawError * 1.8,
            carrying ? -CARRY_MAXIMUM_YAW_SPEED_RADIANS_PER_SECOND : -1,
            carrying ? CARRY_MAXIMUM_YAW_SPEED_RADIANS_PER_SECOND : 1
          ),
          yawStep
        )
      });
      return this.#preparePhysicalFrame(simulation, false);
    }

    this.#reference = targetReference(this.#reference, {
      rootVelocity: [
        approach(
          this.#reference.rootVelocity[0],
          0,
          MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED
            * this.#controlStepSeconds
        ),
        approach(
          this.#reference.rootVelocity[1],
          0,
          MAXIMUM_LINEAR_ACCELERATION_METERS_PER_SECOND_SQUARED
            * this.#controlStepSeconds
        )
      ],
      rootYawVelocity: approach(
        this.#reference.rootYawVelocity,
        0,
        MAXIMUM_YAW_ACCELERATION_RADIANS_PER_SECOND_SQUARED
          * this.#controlStepSeconds
      )
    });
    return this.#preparePhysicalFrame(simulation, true);
  }

  commitPreparedFrame(externalFailure?: string): HumanoidNavigationExecutionStep {
    const pending = this.#pendingFrame;
    if (!pending) throw new Error("Humanoid navigation has no prepared physical frame");
    this.#pendingFrame = undefined;
    this.#final = pending.snapshot;
    this.#frames += 1;
    if (pending.stopping) this.#stopFrames += 1;
    const failure = this.#final.fallen
      ? pending.stopping ? "fallen_while_stopping" : "fallen"
      : blockedHumanoidContacts(this.#final, this.#contactConstraints).length > 0
        ? pending.stopping
          ? "contact_while_stopping"
          : environmentContact(this.#final, this.#contactConstraints)
        : externalFailure;
    if (failure) this.#finish(false, failure);
    if (!this.#result
      && pending.stopping
      && this.#stopFrames >= this.#stoppingFrames) {
      this.#finish(true);
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
    stopping: boolean
  ): Promise<HumanoidNavigationPreparedFrame> {
    this.#reference = applyHumanoidCarryTaskSpaceServo({
      simulation,
      reference: this.#reference,
      targets: this.#carryTaskSpaceTargets,
      maximumReferenceCorrectionRadians:
        CARRY_MAXIMUM_WRIST_REFERENCE_CORRECTION_RADIANS
    });
    if (this.#graspTargets.length > 0) {
      const controlled = contactAwareG1GraspJointTargets({
        requestedJointTargets: simulation.handJointCommandTargets(),
        snapshot: simulation.snapshot(),
        targets: this.#graspTargets
      });
      simulation.applyHandServoJointTargets(controlled.jointTargets);
    }
    const snapshot = await simulation.step(this.#reference);
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

  #finish(completed: boolean, reason?: string): void {
    this.#result = {
      completed,
      ...(reason ? { reason } : {}),
      frames: this.#frames,
      reference: this.#reference,
      final: this.#final,
      travelledDistance: Math.hypot(
        this.#final.rootPosition.x - this.#startRootPosition.x,
        this.#final.rootPosition.z - this.#startRootPosition.z
      )
    };
  }

  #assertProgress(): void {
    if (this.#plan.waypoints.length === 0) {
      throw new Error("Humanoid navigation execution requires at least one waypoint");
    }
    if (this.#waypointIndex > this.#plan.waypoints.length) {
      throw new Error("Humanoid navigation waypoint progress exceeds its plan");
    }
    const stopping = this.#waypointIndex === this.#plan.waypoints.length;
    if ((!stopping && this.#stopFrames !== 0)
      || this.#stopFrames > this.#stoppingFrames) {
      throw new Error("Humanoid navigation stopping progress is inconsistent with its route");
    }
    if (this.#frames > this.#maximumTravelFrames + this.#stoppingFrames) {
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
  carriedObjectTaskSpaceTargets: readonly HumanoidCarryTaskSpaceTarget[]
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
    carryTaskSpaceTargets: carriedObjectTaskSpaceTargets
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

function environmentContact(
  snapshot: HumanoidSimulationSnapshot,
  constraints: readonly HumanoidContactConstraint[]
): string {
  const blocked = blockedHumanoidContacts(snapshot, constraints);
  const base = `environment_contact:${blocked.map((contact) => (
    `${"body" in contact ? contact.body : contact.handSurface}`
      + `:${contact.objectId ?? "environment"}`
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

function yawFromQuaternion(rotation: HumanoidSimulationSnapshot["rootRotation"]): number {
  return Math.atan2(
    2 * (rotation.w * rotation.y + rotation.x * rotation.z),
    1 - 2 * (rotation.y * rotation.y + rotation.z * rotation.z)
  );
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

function point(value: Vec3): string {
  return `${value.x.toFixed(3)},${value.y.toFixed(3)},${value.z.toFixed(3)}`;
}
