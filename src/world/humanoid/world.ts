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
  type HumanoidWorldCheckpoint
} from "./checkpoint.js";
import {
  blockedHumanoidContacts,
  humanoidContactKey,
  humanoidObjectContacts,
  HumanoidMotionPlanSchema,
  missingRequiredHumanoidContacts,
  occupiedHumanoidChannels,
  prepareHumanoidMotion,
  TaskSpaceHumanoidMotionGenerator,
  type HumanoidMotionPlan,
  type HumanoidBodyChannel,
  type HumanoidMotionGenerator,
  type HumanoidMotionValidation
} from "./motion-plan.js";
import type { HumanoidMotionGeneratorDescriptor } from "./motion-generator-contract.js";
import {
  humanoidMotionArtifactSummary,
  hydrateHumanoidReference,
  serializeHumanoidReference,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import {
  HumanoidObjectMemory,
  type HumanoidObjectMemoryCheckpoint,
  type HumanoidObjectToken
} from "./object-memory.js";
import {
  neutralHumanoidReference,
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

export interface HumanoidExecutionReceipt {
  accepted: boolean;
  code: "motion_completed" | "navigation_completed" | "plan_stale"
    | "motion_failed" | "navigation_blocked";
  frames: number;
  finalSnapshot: HumanoidWorldSnapshot;
  detail: {
    failures?: HumanoidMotionValidation["failures"];
    reason?: string;
    travelledDistance?: number;
    motion?: ReturnType<typeof humanoidMotionArtifactSummary>;
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
  createdRevision: number;
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
    this.#objectMemory.observe(
      snapshot.frame,
      snapshot.worldRevision,
      sensed.objects
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
        snapshot.worldRevision,
        new Set(Object.keys(sensed.objects))
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

  async planWholeBodyMotion(rawPlan: HumanoidMotionPlan): Promise<WholeBodyPlanReceipt> {
    const plan = HumanoidMotionPlanSchema.parse(rawPlan);
    if (this.#motions.has(plan.id)) throw new Error(`Duplicate humanoid motion plan: ${plan.id}`);
    const prepared = await prepareHumanoidMotion(
      this.#simulation,
      plan,
      this.#reference,
      { contactObjectIds: this.#objectMemory.observedObjectIds(this.#worldRevision) },
      this.#motionGenerator
    );
    if (prepared.validation.feasible && prepared.artifact) {
      this.#motions.set(plan.id, {
        plan: structuredClone(plan),
        artifact: structuredClone(prepared.artifact),
        createdRevision: this.#worldRevision,
        progress: {
          nextFrameIndex: 0,
          satisfiedContactKeys: [],
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

  async executeWholeBodyMotion(
    planId: string,
    frameSink?: HumanoidFrameSink
  ): Promise<HumanoidExecutionReceipt> {
    const stored = this.#motions.get(planId);
    if (!stored) throw new Error(`Unknown humanoid motion plan: ${planId}`);
    const expectedRevision = stored.createdRevision + stored.progress.nextFrameIndex;
    if (expectedRevision !== this.#worldRevision) {
      this.#motions.delete(planId);
      return this.#receipt(false, "plan_stale", 0, {
        reason: `expected_revision=${expectedRevision}, world_revision=${this.#worldRevision}`
      });
    }
    let frames = 0;
    const failures: HumanoidMotionValidation["failures"] = stored.progress.failure
      ? [validationFailure(stored.progress.failure)]
      : [];
    const constraints = stored.plan.contact_constraints ?? [];
    const satisfiedContacts = new Set(stored.progress.satisfiedContactKeys);
    let lastReference = this.#reference;
    for (
      let index = stored.progress.nextFrameIndex;
      index < stored.artifact.frames.length && failures.length === 0;
      index += 1
    ) {
      const frame = stored.artifact.frames[index]!;
      const reference = hydrateHumanoidReference(frame.reference);
      const snapshot = await this.#simulation.step(reference);
      frames += 1;
      lastReference = reference;
      this.#reference = reference;
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
      }
      stored.progress.nextFrameIndex = index + 1;
      stored.progress.satisfiedContactKeys = [...satisfiedContacts];
      await this.#commitFrame(frameSink);
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
    this.#reference = lastReference;
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
      await this.#commitFrame(frameSink, snapshot);
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
        this.#motions.set(entry.plan.id, structuredClone(entry));
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
    _snapshot?: HumanoidSimulationSnapshot
  ): Promise<void> {
    this.#frame += 1;
    this.#worldRevision += 1;
    await sink?.(this.snapshot());
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
  return {
    code: "environment_contact",
    atSeconds: failure.atSeconds,
    ...(failure.bodies ? { bodies: [...failure.bodies] } : {}),
    ...(failure.contacts
      ? { contacts: failure.contacts.map((contact) => ({ ...contact })) }
      : {})
  };
}
