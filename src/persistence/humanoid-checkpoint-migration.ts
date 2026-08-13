import type { Scenario } from "../domain/schema.js";
import { EmptyContextMemoryState } from "../domain/schema.js";
import {
  HumanoidRunCheckpointSchema,
  LegacyHumanoidRunCheckpointSchema,
  PreGraspHumanoidRunCheckpointV6Schema,
  completeHumanoidRunCheckpointPhysicalMigration,
  type HumanoidRunCheckpoint
} from "../domain/humanoid-run.js";
import { HumanoidWorldCheckpointSchema } from "../world/humanoid/checkpoint.js";
import { humanoidEnvironment } from "../world/humanoid/environment.js";
import { HumanoidSimulation } from "../world/humanoid/simulation.js";
import { HumanoidWorldSnapshotSchema } from "../world/humanoid/snapshot-schema.js";
import {
  HumanoidGraspRegistry,
  HumanoidGraspRegistryCheckpointSchema
} from "../world/humanoid/grasp-registry.js";
import { humanoidMotionArtifactSha256 } from "../world/humanoid/motion-artifact.js";
import { humanoidMotionIntentSha256 } from "../world/humanoid/plan-lifecycle.js";
import { humanoidMotionContactEvidenceSha256 } from "../world/humanoid/motion-contact-evidence.js";
import {
  NEURAL_HIERARCHY_CONTRACT_VERSION,
  createNeuralHierarchyState
} from "../domain/neural-hierarchy.js";

export interface NormalizedHumanoidRunCheckpoint {
  checkpoint: HumanoidRunCheckpoint;
  migrated: boolean;
}

export async function normalizeHumanoidRunCheckpoint(
  raw: unknown,
  scenario: Scenario,
  options: { freshNeuralHierarchyEpoch?: boolean } = {}
): Promise<NormalizedHumanoidRunCheckpoint> {
  const source = options.freshNeuralHierarchyEpoch
    ? checkpointWithFreshNeuralHierarchy(raw)
    : raw;
  assertCompatibleNeuralHierarchyCheckpoint(source);
  if (!humanoidRunCheckpointNeedsPhysicalMigration(source)) {
    return {
      checkpoint: HumanoidRunCheckpointSchema.parse(source),
      migrated: options.freshNeuralHierarchyEpoch === true
    };
  }

  if (preGraspV6NeedsPhysicalMigration(source)) {
    return migratePreGraspV6(source, scenario);
  }

  const legacy = LegacyHumanoidRunCheckpointSchema.parse(source);
  const simulation = await HumanoidSimulation.create(humanoidEnvironment(scenario));
  try {
    const source = legacy.world_checkpoint.simulation;
    simulation.restoreState({
      time: source.time,
      positions: Float64Array.from(source.positions),
      velocities: Float64Array.from(source.velocities),
      controls: Float64Array.from(source.controls),
      activations: Float64Array.from(source.activations),
      accelerationWarmstart: Float64Array.from(source.accelerationWarmstart),
      ...(source.requestedActuatorTorques === undefined
        ? {}
        : {
            requestedActuatorTorques: Float64Array.from(
              source.requestedActuatorTorques
            )
          }),
      controller: structuredClone(source.controller)
    });

    const state = simulation.captureState();
    const graspRegistry = new HumanoidGraspRegistry({
      portableObjectIds: scenario.objects
        .filter((object) => object.portable)
        .map((object) => object.id)
    });
    const graspAssessments = graspRegistry.observe(
      legacy.world_checkpoint.frame,
      simulation.snapshot()
    );
    const retiredNavigation = {
      planId: null,
      status: "idle" as const,
      target: null,
      waypoints: [],
      waypointIndex: null
    };
    const {
      physicalSafety: _legacyCheckpointSafety,
      ...checkpointWithoutSafety
    } = legacy.world_checkpoint;
    const worldCheckpoint = HumanoidWorldCheckpointSchema.parse({
      ...checkpointWithoutSafety,
      routeSequence: legacy.world_checkpoint.routeSequence + 1,
      planRegistryEpoch: legacy.world_checkpoint.planRegistryEpoch + 1,
      motions: [],
      routes: [],
      navigation: retiredNavigation,
      graspRegistry: graspRegistry.checkpoint(),
      simulation: {
        time: state.time,
        positions: [...state.positions],
        velocities: [...state.velocities],
        controls: [...state.controls],
        activations: [...state.activations],
        accelerationWarmstart: [...state.accelerationWarmstart],
        ...(state.requestedActuatorTorques === undefined
          ? {}
          : { requestedActuatorTorques: [...state.requestedActuatorTorques] }),
        ...(state.handCommandTargets === undefined
          ? {}
          : { handCommandTargets: [...state.handCommandTargets] }),
        handPolicyAuthority: state.handPolicyAuthority
          ? structuredClone(state.handPolicyAuthority)
          : null,
        controller: structuredClone(state.controller)
      }
    });
    const {
      physicalSafety: _legacySnapshotSafety,
      ...worldWithoutSafety
    } = legacy.world;
    const world = HumanoidWorldSnapshotSchema.parse({
      ...worldWithoutSafety,
      robot: simulation.snapshot(),
      grasp: {
        contractSha256: graspRegistry.contractSha256,
        assessments: graspAssessments
      },
      navigation: retiredNavigation
    });
    return {
      checkpoint: completeHumanoidRunCheckpointPhysicalMigration({
        checkpoint: legacy,
        world,
        worldCheckpoint
      }),
      migrated: true
    };
  } finally {
    await simulation.dispose();
  }
}

function checkpointWithFreshNeuralHierarchy(raw: unknown): unknown {
  assertFreshNeuralHierarchyEpochSafe(raw);
  const source = structuredClone(raw as Record<string, unknown>);
  const hierarchy = createNeuralHierarchyState();
  source.neural_hierarchy_state = hierarchy;
  source.action_runtime_state = emptyActionRuntimeStateForFreshEpoch(
    source,
    hierarchy.epoch_id
  );
  const previousContext = recordProperty(source, "context_memory");
  source.context_memory = {
    ...structuredClone(EmptyContextMemoryState),
    ...(typeof previousContext.context_window_tokens === "number"
      ? { context_window_tokens: previousContext.context_window_tokens }
      : {}),
    ...(typeof previousContext.compact_trigger_tokens === "number"
      ? { compact_trigger_tokens: previousContext.compact_trigger_tokens }
      : {}),
    ...(typeof previousContext.compact_recent_model_turns === "number"
      ? { compact_recent_model_turns: previousContext.compact_recent_model_turns }
      : {}),
    ...(typeof previousContext.compact_max_output_tokens === "number"
      ? { compact_max_output_tokens: previousContext.compact_max_output_tokens }
      : {})
  };
  source.context_memory_state_anchor = null;
  source.active_agent_id = source.root_id;
  source.active_agent_ids = [source.root_id];
  return source;
}

function emptyActionRuntimeStateForFreshEpoch(
  source: Record<string, unknown>,
  neuralHierarchyEpochId: string
): Record<string, unknown> {
  const previousState = recordProperty(source, "action_runtime_state");
  const persistedRevision = nonnegativeInteger(
    previousState.latest_physical_execution_revision
  ) ?? 0;
  const receipts = recordProperty(source, "committed_actions");
  const durablePhysicalRevision = Object.values(receipts).reduce<number>((latest, value) => {
    if (!isRecord(value)
      || value.accepted !== true
      || !isPhysicalExecutionAction(value.action)) return latest;
    return Math.max(latest, nonnegativeInteger(value.worldAfterRevision) ?? 0);
  }, 0);
  return {
    version: 1,
    neural_hierarchy_epoch_id: neuralHierarchyEpochId,
    latest_physical_execution_revision: Math.max(
      persistedRevision,
      durablePhysicalRevision
    ),
    skill_plans: [],
    active_skill_plan_transactions: {},
    active_skills: [],
    planning_skill_bindings: [],
    recovery_policies: [],
    navigation_transit_clearance_requirements: [],
    latest_grounding_observation: null
  };
}

function isPhysicalExecutionAction(value: unknown): boolean {
  return value === "execute_humanoid_skill"
    || value === "execute_whole_body_motion"
    || value === "execute_humanoid_navigation";
}

function nonnegativeInteger(value: unknown): number | undefined {
  return typeof value === "number"
    && Number.isInteger(value)
    && value >= 0
    ? value
    : undefined;
}

export function assertFreshNeuralHierarchyEpochSafe(raw: unknown): void {
  if (!isRecord(raw)) throw new Error("Humanoid checkpoint must be an object");
  if (raw.status === "succeeded") {
    throw new Error("A succeeded run cannot start a fresh Agent/hierarchy epoch");
  }
  const ledger = recordProperty(raw, "action_execution_ledger");
  const activeExecutions = recordProperty(ledger, "active");
  const outbox = recordProperty(raw, "action_commit_outbox");
  const pendingCommits = recordProperty(outbox, "pending");
  if (Object.keys(activeExecutions).length > 0
    || Object.keys(pendingCommits).length > 0) {
    throw new Error(
      "A fresh Agent/hierarchy epoch cannot start while a physical execution or "
        + "action commit is unfinished; recover that transaction first"
    );
  }
}

function assertCompatibleNeuralHierarchyCheckpoint(raw: unknown): void {
  if (!isRecord(raw)) return;
  const hierarchy = raw.neural_hierarchy_state;
  if (!isRecord(hierarchy) || typeof hierarchy.version !== "number") return;
  if (hierarchy.version === NEURAL_HIERARCHY_CONTRACT_VERSION) return;
  throw new Error(
    `The persisted neural hierarchy uses contract v${hierarchy.version}, but this `
      + `runtime requires v${NEURAL_HIERARCHY_CONTRACT_VERSION} with invocation-scoped `
      + "parent fork/join identity. It cannot reuse the old Agent epoch because doing so "
      + "could attach rollout feedback to the wrong hierarchy episode. Physical state, "
      + "Goal state, and embodied memory remain unchanged; resume through an explicit fresh "
      + "Agent/hierarchy epoch."
  );
}

export function humanoidRunCheckpointNeedsPhysicalMigration(
  value: unknown
): value is Record<string, unknown> & { version: 4 | 5 | 6 } {
  return typeof value === "object"
    && value !== null
    && !Array.isArray(value)
    && "version" in value
    && (value.version === 4
      || value.version === 5
      || preGraspV6NeedsPhysicalMigration(value));
}

async function migratePreGraspV6(
  raw: Record<string, unknown> & { version: 6 },
  scenario: Scenario
): Promise<NormalizedHumanoidRunCheckpoint> {
  const worldRecord = recordProperty(raw, "world");
  const checkpointRecord = recordProperty(raw, "world_checkpoint");
  const worldHasGrasp = Object.hasOwn(worldRecord, "grasp");
  const checkpointHasRegistry = Object.hasOwn(
    checkpointRecord,
    "graspRegistry"
  );
  if (worldHasGrasp && !checkpointHasRegistry) {
    throw new Error(
      "Pre-grasp V6 checkpoint has display evidence without its authoritative registry"
    );
  }

  const parseable = structuredClone(raw);
  delete recordProperty(parseable, "world").grasp;
  delete recordProperty(parseable, "world_checkpoint").graspRegistry;
  const checkpoint = PreGraspHumanoidRunCheckpointV6Schema.parse(parseable);
  if (checkpoint.world.frame !== checkpoint.world_checkpoint.frame
    || checkpoint.world.worldRevision
      !== checkpoint.world_checkpoint.worldRevision) {
    throw new Error(
      "Pre-grasp V6 world snapshot and physical checkpoint are not aligned"
    );
  }

  const simulation = await HumanoidSimulation.create(humanoidEnvironment(scenario));
  try {
    const source = checkpoint.world_checkpoint.simulation;
    simulation.restoreState({
      time: source.time,
      positions: Float64Array.from(source.positions),
      velocities: Float64Array.from(source.velocities),
      controls: Float64Array.from(source.controls),
      activations: Float64Array.from(source.activations),
      accelerationWarmstart: Float64Array.from(source.accelerationWarmstart),
      ...(source.requestedActuatorTorques === undefined
        ? {}
        : {
            requestedActuatorTorques: Float64Array.from(
              source.requestedActuatorTorques
            )
          }),
      controller: structuredClone(source.controller)
    });
    const physicalSnapshot = simulation.snapshot();
    const existingRegistry = checkpointHasRegistry
      ? HumanoidGraspRegistryCheckpointSchema.parse(
          checkpointRecord.graspRegistry
        )
      : undefined;
    const graspRegistry = new HumanoidGraspRegistry({
      portableObjectIds: scenario.objects
        .filter((object) => object.portable)
        .map((object) => object.id),
      ...(existingRegistry ? { checkpoint: existingRegistry } : {})
    });
    const assessments = existingRegistry
      ? (() => {
          if (graspRegistry.lastFrame !== checkpoint.world.frame) {
            throw new Error(
              "Pre-grasp V6 registry is not aligned with its world frame"
            );
          }
          return graspRegistry.assessmentsForFrame(checkpoint.world.frame);
        })()
      : graspRegistry.observe(checkpoint.world.frame, physicalSnapshot);
    const state = simulation.captureState();
    const worldCheckpoint = HumanoidWorldCheckpointSchema.parse({
      ...checkpoint.world_checkpoint,
      graspRegistry: graspRegistry.checkpoint(),
      motions: checkpoint.world_checkpoint.motions.map((motion) => ({
        ...motion,
        progress: {
          ...motion.progress,
          satisfiedContactEvidenceSha256: humanoidMotionContactEvidenceSha256({
            planId: motion.plan.id,
            intentSha256: motion.intentSha256
              ?? humanoidMotionIntentSha256(motion.plan),
            artifactSha256: humanoidMotionArtifactSha256(motion.artifact),
            nextFrameIndex: motion.progress.nextFrameIndex,
            satisfiedContactKeys: motion.progress.satisfiedContactKeys
          })
        }
      })),
      simulation: {
        time: state.time,
        positions: [...state.positions],
        velocities: [...state.velocities],
        controls: [...state.controls],
        activations: [...state.activations],
        accelerationWarmstart: [...state.accelerationWarmstart],
        ...(state.requestedActuatorTorques === undefined
          ? {}
          : { requestedActuatorTorques: [...state.requestedActuatorTorques] }),
        ...(state.handCommandTargets === undefined
          ? {}
          : { handCommandTargets: [...state.handCommandTargets] }),
        handPolicyAuthority: state.handPolicyAuthority
          ? structuredClone(state.handPolicyAuthority)
          : null,
        controller: structuredClone(state.controller)
      }
    });
    const world = HumanoidWorldSnapshotSchema.parse({
      ...checkpoint.world,
      robot: physicalSnapshot,
      grasp: {
        contractSha256: graspRegistry.contractSha256,
        assessments
      }
    });
    return {
      checkpoint: HumanoidRunCheckpointSchema.parse({
        ...checkpoint,
        world,
        world_checkpoint: worldCheckpoint
      }),
      migrated: true
    };
  } finally {
    await simulation.dispose();
  }
}

function preGraspV6NeedsPhysicalMigration(
  value: unknown
): value is Record<string, unknown> & { version: 6 } {
  if (!isRecord(value) || value.version !== 6) return false;
  const world = value.world;
  const checkpoint = value.world_checkpoint;
  return !isRecord(world)
    || !isRecord(checkpoint)
    || !Object.hasOwn(world, "grasp")
    || !Object.hasOwn(checkpoint, "graspRegistry");
}

function recordProperty(
  value: Record<string, unknown>,
  property: string
): Record<string, unknown> {
  const nested = value[property];
  if (!isRecord(nested)) {
    throw new Error(`Humanoid checkpoint ${property} must be an object`);
  }
  return nested;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
