import type { Scenario } from "../domain/schema.js";
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

export interface NormalizedHumanoidRunCheckpoint {
  checkpoint: HumanoidRunCheckpoint;
  migrated: boolean;
}

export async function normalizeHumanoidRunCheckpoint(
  raw: unknown,
  scenario: Scenario
): Promise<NormalizedHumanoidRunCheckpoint> {
  if (!humanoidRunCheckpointNeedsPhysicalMigration(raw)) {
    return {
      checkpoint: HumanoidRunCheckpointSchema.parse(raw),
      migrated: false
    };
  }

  if (preGraspV6NeedsPhysicalMigration(raw)) {
    return migratePreGraspV6(raw, scenario);
  }

  const legacy = LegacyHumanoidRunCheckpointSchema.parse(raw);
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
