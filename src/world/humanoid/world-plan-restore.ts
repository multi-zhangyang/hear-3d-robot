import {
  humanoidMotionArtifactSha256
} from "./motion-artifact.js";
import type { HumanoidWorldCheckpoint } from "./checkpoint.js";
import {
  admitHumanoidCarriedObjectBindings,
  humanoidCarriedObjectContinuationEvidence,
  humanoidCarriedObjectUnauthorizedContacts,
  verifyHumanoidCarriedObjectBindingSet,
  type HumanoidCarriedObjectBindingSet
} from "./carried-object-binding.js";
import { captureHumanoidCarryTaskSpaceTargets } from "./carry-task-space-servo.js";
import type { HumanoidGraspRegistry } from "./grasp-registry.js";
import { humanoidMotionContactEvidenceSha256 } from "./motion-contact-evidence.js";
import {
  humanoidMotionIntentSha256,
  humanoidNavigationIntentSha256,
  humanoidPlanIntentIsActive
} from "./plan-lifecycle.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";
import {
  storedMotionPhysicalSafety,
  type StoredHumanoidMotionPlan,
  type StoredHumanoidNavigationPlan
} from "./world-plan-state.js";
import type { HumanoidWorldSnapshot } from "./world-contract.js";
import { legacyHumanoidEmbodiedSkillIdentity } from "./embodied-skill-call.js";

export interface RestoredHumanoidWorldPlans {
  motions: Map<string, StoredHumanoidMotionPlan>;
  routes: Map<string, StoredHumanoidNavigationPlan>;
  planRegistryEpoch: number;
  physicalSafety: HumanoidWorldSnapshot["physicalSafety"];
  navigation: HumanoidWorldSnapshot["navigation"];
}

export function restoreHumanoidWorldPlans(input: {
  checkpoint: HumanoidWorldCheckpoint;
  snapshot: HumanoidSimulationSnapshot;
  graspRegistry: HumanoidGraspRegistry;
  currentBindings: HumanoidCarriedObjectBindingSet;
  restoredStateSha256: string;
  planExpiryRevision: (createdRevision: number) => number;
}): RestoredHumanoidWorldPlans {
  const motions = restoreMotions(input);
  const routes = restoreRoutes(input);
  const pruned = motions.pruned || routes.pruned;
  const physicalSafety = input.checkpoint.physicalSafety
    ? structuredClone(input.checkpoint.physicalSafety)
    : firstMotionPhysicalSafety(motions.plans);
  return {
    motions: motions.plans,
    routes: routes.plans,
    planRegistryEpoch: input.checkpoint.planRegistryEpoch + (pruned ? 1 : 0),
    physicalSafety,
    navigation: restoreNavigationState(
      input.checkpoint.navigation,
      routes.plans
    )
  };
}

function restoreMotions(input: {
  checkpoint: HumanoidWorldCheckpoint;
  snapshot: HumanoidSimulationSnapshot;
  graspRegistry: HumanoidGraspRegistry;
  currentBindings: HumanoidCarriedObjectBindingSet;
  restoredStateSha256: string;
  planExpiryRevision: (createdRevision: number) => number;
}): {
  plans: Map<string, StoredHumanoidMotionPlan>;
  pruned: boolean;
} {
  const plans = new Map<string, StoredHumanoidMotionPlan>();
  let pruned = false;
  for (const entry of input.checkpoint.motions) {
    const validatedRevision = entry.validatedRevision ?? entry.createdRevision;
    const expiresRevision = entry.expiresRevision
      ?? input.planExpiryRevision(entry.createdRevision);
    const intentSha256 = entry.intentSha256
      ?? humanoidMotionIntentSha256(entry.plan);
    const carriedObjectBindings = entry.carriedObjectBindings
      ?? input.currentBindings;
    const carriedObjectTaskSpaceTargets = entry.carriedObjectTaskSpaceTargets.length > 0
      ? entry.carriedObjectTaskSpaceTargets
      : captureHumanoidCarryTaskSpaceTargets({
          snapshot: input.snapshot,
          bindings: carriedObjectBindings
        });
    const restoredContinuation = entry.progress.nextFrameIndex > 0
      && entry.terminal === null
      ? humanoidCarriedObjectContinuationEvidence({
          state: carriedObjectBindings,
          registry: input.graspRegistry,
          currentFrame: input.checkpoint.frame,
          currentWorldRevision: input.checkpoint.worldRevision
        })
      : entry.carriedObjectContinuation;
    const restoredUnauthorizedContacts = entry.progress.nextFrameIndex > 0
      && entry.terminal === null
      ? humanoidCarriedObjectUnauthorizedContacts(
          carriedObjectBindings,
          input.snapshot.contacts
        )
      : entry.carriedObjectUnauthorizedContacts;
    const expectedRevision = validatedRevision + entry.progress.nextFrameIndex;
    const carryRecoverable = entry.terminal !== null
      || entry.progress.nextFrameIndex === 0
      || restoredContinuation?.continued === true
        && restoredUnauthorizedContacts.length === 0;
    const recoverable = carryRecoverable && (entry.terminal !== null
      || entry.progress.nextFrameIndex > 0
        ? expectedRevision === input.checkpoint.worldRevision
        : humanoidPlanIntentIsActive(
            input.checkpoint.worldRevision,
            expiresRevision
          ));
    if (!recoverable) {
      pruned = true;
      continue;
    }
    if (entry.progress.satisfiedContactKeys.length > 0
      && entry.progress.satisfiedContactEvidenceSha256 === undefined) {
      throw new Error("Executed humanoid contact evidence is missing its prefix identity");
    }
    const restored: StoredHumanoidMotionPlan = {
      ...structuredClone(entry),
      skillCallIdentity: entry.skillCallIdentity
        ?? legacyHumanoidEmbodiedSkillIdentity({
          callId: `restored-motion:${entry.plan.id}`,
          runtimeKind: "legacy_motion",
          phase: "execute_reference",
          observedFrame: Math.max(
            0,
            input.checkpoint.frame - entry.progress.nextFrameIndex
          ),
          observedWorldRevision: entry.createdRevision
        }),
      retainTerminalJointTracking: entry.retainTerminalJointTracking ?? false,
      validatedRevision,
      validatedStateSha256: entry.validatedStateSha256
        ?? input.restoredStateSha256,
      expiresRevision,
      intentSha256,
      revalidationCount: entry.revalidationCount ?? 0,
      carriedObjectBindings,
      carriedObjectTaskSpaceTargets,
      carriedObjectContinuation: restoredContinuation,
      carriedObjectUnauthorizedContacts: restoredUnauthorizedContacts
    };
    restored.progress.satisfiedContactEvidenceSha256 ??=
      humanoidMotionContactEvidenceSha256({
        planId: restored.plan.id,
        intentSha256: restored.intentSha256,
        artifactSha256: humanoidMotionArtifactSha256(restored.artifact),
        nextFrameIndex: restored.progress.nextFrameIndex,
        satisfiedContactKeys: restored.progress.satisfiedContactKeys
      });
    plans.set(entry.plan.id, restored);
  }
  return { plans, pruned };
}

function restoreRoutes(input: {
  checkpoint: HumanoidWorldCheckpoint;
  snapshot: HumanoidSimulationSnapshot;
  graspRegistry: HumanoidGraspRegistry;
  restoredStateSha256: string;
  planExpiryRevision: (createdRevision: number) => number;
}): {
  plans: Map<string, StoredHumanoidNavigationPlan>;
  pruned: boolean;
} {
  const plans = new Map<string, StoredHumanoidNavigationPlan>();
  let pruned = false;
  for (const entry of input.checkpoint.routes) {
    const validatedRevision = entry.validatedRevision ?? entry.createdRevision;
    const expiresRevision = entry.expiresRevision
      ?? input.planExpiryRevision(entry.createdRevision);
    const progress = entry.progress ?? {
      version: 1 as const,
      start_root_position: { ...input.snapshot.rootPosition },
      waypoint_index: Math.min(1, entry.plan.waypoints.length - 1),
      committed_frame_count: 0,
      stopping_frame_count: 0
    };
    const carriedObjectBindings = entry.carriedObjectBindings
      ?? admitHumanoidCarriedObjectBindings({
        registry: input.graspRegistry,
        currentFrame: input.checkpoint.frame,
        currentWorldRevision: input.checkpoint.worldRevision,
        requests: []
      });
    const carriedObjectTaskSpaceTargets = entry.carriedObjectTaskSpaceTargets.length > 0
      ? entry.carriedObjectTaskSpaceTargets
      : captureHumanoidCarryTaskSpaceTargets({
          snapshot: input.snapshot,
          bindings: carriedObjectBindings
        });
    if (progress.committed_frame_count === 0
      && carriedObjectBindings.source_frame === input.checkpoint.frame
      && carriedObjectBindings.source_world_revision === input.checkpoint.worldRevision) {
      verifyHumanoidCarriedObjectBindingSet({
        registry: input.graspRegistry,
        currentFrame: input.checkpoint.frame,
        currentWorldRevision: input.checkpoint.worldRevision,
        state: carriedObjectBindings
      });
    }
    const restoredContinuation = progress.committed_frame_count > 0
      && entry.terminal === null
      ? humanoidCarriedObjectContinuationEvidence({
          state: carriedObjectBindings,
          registry: input.graspRegistry,
          currentFrame: input.checkpoint.frame,
          currentWorldRevision: input.checkpoint.worldRevision
        })
      : entry.carriedObjectContinuation;
    const restoredUnauthorizedContacts = progress.committed_frame_count > 0
      && entry.terminal === null
      ? humanoidCarriedObjectUnauthorizedContacts(
          carriedObjectBindings,
          input.snapshot.contacts
        )
      : entry.carriedObjectUnauthorizedContacts;
    const expectedRevision = validatedRevision + progress.committed_frame_count;
    const carryRecoverable = entry.terminal !== null
      || progress.committed_frame_count === 0
      || restoredContinuation?.continued === true
        && restoredUnauthorizedContacts.length === 0;
    const recoverable = carryRecoverable && (entry.terminal !== null
      || progress.committed_frame_count > 0
        ? expectedRevision === input.checkpoint.worldRevision
        : humanoidPlanIntentIsActive(
            input.checkpoint.worldRevision,
            expiresRevision
          ));
    if (!recoverable) {
      pruned = true;
      continue;
    }
    plans.set(entry.id, {
      ...structuredClone(entry),
      skillCallIdentity: entry.skillCallIdentity
        ?? legacyHumanoidEmbodiedSkillIdentity({
          callId: `restored-navigation:${entry.id}`,
          runtimeKind: "navigation",
          phase: "navigate",
          observedFrame: Math.max(
            0,
            input.checkpoint.frame - progress.committed_frame_count
          ),
          observedWorldRevision: entry.createdRevision
        }),
      validatedRevision,
      validatedStateSha256: entry.validatedStateSha256
        ?? input.restoredStateSha256,
      expiresRevision,
      intentSha256: entry.intentSha256
        ?? humanoidNavigationIntentSha256(
          entry.requestedTarget,
          entry.requestedArrivalHeading,
          entry.acceptedPositionToleranceMeters
        ),
      revalidationCount: entry.revalidationCount ?? 0,
      carriedObjectBindings,
      carriedObjectTaskSpaceTargets,
      carriedObjectContinuation: restoredContinuation,
      carriedObjectUnauthorizedContacts: restoredUnauthorizedContacts,
      progress
    });
  }
  return { plans, pruned };
}

function firstMotionPhysicalSafety(
  motions: ReadonlyMap<string, StoredHumanoidMotionPlan>
): HumanoidWorldSnapshot["physicalSafety"] {
  for (const [planId, stored] of motions) {
    const evidence = storedMotionPhysicalSafety(stored);
    if (evidence) return { planId, evidence };
  }
  return undefined;
}

function restoreNavigationState(
  source: HumanoidWorldSnapshot["navigation"],
  routes: ReadonlyMap<string, StoredHumanoidNavigationPlan>
): HumanoidWorldSnapshot["navigation"] {
  const navigation = structuredClone(source);
  const planId = navigation.planId;
  if (navigation.status === "executing"
    && planId !== null
    && routes.has(planId)) {
    navigation.status = "planned";
    navigation.waypointIndex = routes.get(planId)!.progress.waypoint_index;
  } else if ((navigation.status === "executing" || navigation.status === "planned")
    && (planId === null || !routes.has(planId))) {
    navigation.planId = null;
    navigation.status = "blocked";
    navigation.waypointIndex = null;
  }
  return navigation;
}
