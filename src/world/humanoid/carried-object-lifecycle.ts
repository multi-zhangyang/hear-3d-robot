import { z } from "zod";
import {
  HumanoidCarriedObjectBindingSetSchema,
  HumanoidCarriedObjectContinuationEvidenceSchema,
  HumanoidCarriedObjectUnauthorizedContactSchema,
  admitHumanoidCarriedObjectBindings,
  humanoidCarriedObjectBindingSetSha256,
  humanoidCarriedObjectContinuationEvidence,
  humanoidCarriedObjectUnauthorizedContacts,
  verifyHumanoidCarriedObjectBindingSet,
  type HumanoidCarriedObjectBindingRequest,
  type HumanoidCarriedObjectBindingSet,
  type HumanoidCarriedObjectContinuationEvidence,
  type HumanoidCarriedObjectUnauthorizedContact
} from "./carried-object-binding.js";
import type { HumanoidGraspRegistry } from "./grasp-registry.js";
import { humanoidObjectHandContacts } from "./object-release.js";
import type { HumanoidContactSnapshot } from "./simulation.js";

const FrameSchema = z.number().int().nonnegative();
const WorldRevisionSchema = z.number().int().nonnegative();

export const HumanoidCarriedObjectLifecycleCheckpointSchema = z.object({
  protocol: z.literal("humanoid-carried-object-lifecycle-v1"),
  phase: z.enum([
    "idle",
    "acquired",
    "carrying",
    "release_pending",
    "released",
    "lost"
  ]),
  active_binding_set: HumanoidCarriedObjectBindingSetSchema.nullable(),
  last_continuation: HumanoidCarriedObjectContinuationEvidenceSchema.nullable(),
  last_unauthorized_contacts: z.array(
    HumanoidCarriedObjectUnauthorizedContactSchema
  ),
  transition_frame: FrameSchema,
  transition_world_revision: WorldRevisionSchema,
  transition_reason: z.enum([
    "initialized",
    "grasp_acquired",
    "grasp_continued",
    "release_started",
    "release_cancelled",
    "release_completed",
    "grasp_lost"
  ])
}).strict().superRefine((state, context) => {
  const active = state.active_binding_set !== null
    && state.active_binding_set.bindings.length > 0;
  const activePhase = state.phase === "acquired"
    || state.phase === "carrying"
    || state.phase === "release_pending";
  if (active !== activePhase) {
    context.addIssue({
      code: "custom",
      path: ["active_binding_set"],
      message: "Carried-object lifecycle phase does not match its active binding"
    });
  }
  if (state.active_binding_set?.bindings.length === 0) {
    context.addIssue({
      code: "custom",
      path: ["active_binding_set"],
      message: "An active carried-object binding set cannot be empty"
    });
  }
  if (state.last_continuation && state.active_binding_set
    && state.last_continuation.binding_set_sha256
      !== humanoidCarriedObjectBindingSetSha256(state.active_binding_set)) {
    context.addIssue({
      code: "custom",
      path: ["last_continuation"],
      message: "Carried-object continuation does not match the active binding"
    });
  }
  if (state.phase === "carrying"
    && state.last_continuation?.continued !== true) {
    context.addIssue({
      code: "custom",
      path: ["last_continuation"],
      message: "A carrying lifecycle requires successful continuation evidence"
    });
  }
});

export type HumanoidCarriedObjectLifecycleCheckpoint = z.infer<
  typeof HumanoidCarriedObjectLifecycleCheckpointSchema
>;

export interface HumanoidCarriedObjectLifecycleObservation {
  bindingSet: HumanoidCarriedObjectBindingSet;
  continuation: HumanoidCarriedObjectContinuationEvidence;
  unauthorizedContacts: HumanoidCarriedObjectUnauthorizedContact[];
  phase: HumanoidCarriedObjectLifecycleCheckpoint["phase"];
}

export class HumanoidCarriedObjectLifecycle {
  readonly #registry: HumanoidGraspRegistry;
  #state: HumanoidCarriedObjectLifecycleCheckpoint;

  constructor(input: {
    registry: HumanoidGraspRegistry;
    currentFrame: number;
    currentWorldRevision: number;
    checkpoint?: HumanoidCarriedObjectLifecycleCheckpoint | null;
  }) {
    this.#registry = input.registry;
    const currentFrame = FrameSchema.parse(input.currentFrame);
    const currentWorldRevision = WorldRevisionSchema.parse(
      input.currentWorldRevision
    );
    this.#state = input.checkpoint
      ? HumanoidCarriedObjectLifecycleCheckpointSchema.parse(input.checkpoint)
      : {
          protocol: "humanoid-carried-object-lifecycle-v1",
          phase: "idle",
          active_binding_set: null,
          last_continuation: null,
          last_unauthorized_contacts: [],
          transition_frame: currentFrame,
          transition_world_revision: currentWorldRevision,
          transition_reason: "initialized"
        };
    this.#verifyCurrentAuthority(currentFrame, currentWorldRevision);
  }

  get phase(): HumanoidCarriedObjectLifecycleCheckpoint["phase"] {
    return this.#state.phase;
  }

  get active(): HumanoidCarriedObjectBindingSet | null {
    return this.#state.active_binding_set
      ? structuredClone(this.#state.active_binding_set)
      : null;
  }

  checkpoint(): HumanoidCarriedObjectLifecycleCheckpoint {
    return HumanoidCarriedObjectLifecycleCheckpointSchema.parse(
      structuredClone(this.#state)
    );
  }

  bindingSet(
    currentFrame: number,
    currentWorldRevision: number
  ): HumanoidCarriedObjectBindingSet {
    if (this.#state.active_binding_set) {
      return structuredClone(this.#state.active_binding_set);
    }
    return admitHumanoidCarriedObjectBindings({
      registry: this.#registry,
      currentFrame,
      currentWorldRevision,
      requests: []
    });
  }

  acquire(input: {
    currentFrame: number;
    currentWorldRevision: number;
    requests: readonly HumanoidCarriedObjectBindingRequest[];
  }): HumanoidCarriedObjectBindingSet {
    if (input.requests.length === 0) {
      throw new Error("Carried-object acquisition requires at least one binding request");
    }
    if (this.#state.active_binding_set) {
      throw new Error("A carried-object binding is already active");
    }
    const bindingSet = admitHumanoidCarriedObjectBindings({
      registry: this.#registry,
      currentFrame: input.currentFrame,
      currentWorldRevision: input.currentWorldRevision,
      requests: input.requests
    });
    this.#state = HumanoidCarriedObjectLifecycleCheckpointSchema.parse({
      protocol: "humanoid-carried-object-lifecycle-v1",
      phase: "acquired",
      active_binding_set: bindingSet,
      last_continuation: null,
      last_unauthorized_contacts: [],
      transition_frame: input.currentFrame,
      transition_world_revision: input.currentWorldRevision,
      transition_reason: "grasp_acquired"
    });
    return structuredClone(bindingSet);
  }

  beginRelease(input: {
    currentFrame: number;
    currentWorldRevision: number;
  }): void {
    if (!this.#state.active_binding_set) {
      throw new Error("Cannot release an object without an active carried binding");
    }
    this.#transition(
      "release_pending",
      input.currentFrame,
      input.currentWorldRevision,
      "release_started"
    );
  }

  cancelRelease(input: {
    currentFrame: number;
    currentWorldRevision: number;
  }): void {
    if (this.#state.phase !== "release_pending"
      || !this.#state.active_binding_set
      || this.#state.last_continuation?.continued !== true) return;
    this.#transition(
      "carrying",
      input.currentFrame,
      input.currentWorldRevision,
      "release_cancelled"
    );
  }

  observe(input: {
    currentFrame: number;
    currentWorldRevision: number;
    contacts: readonly HumanoidContactSnapshot[];
  }): HumanoidCarriedObjectLifecycleObservation | null {
    const bindingSet = this.#state.active_binding_set;
    if (!bindingSet) return null;
    const continuation = humanoidCarriedObjectContinuationEvidence({
      state: bindingSet,
      registry: this.#registry,
      currentFrame: input.currentFrame,
      currentWorldRevision: input.currentWorldRevision
    });
    const unauthorizedContacts = humanoidCarriedObjectUnauthorizedContacts(
      bindingSet,
      input.contacts
    );
    this.#state.last_continuation = continuation;
    this.#state.last_unauthorized_contacts = unauthorizedContacts;
    if (!continuation.continued) {
      const releaseSeparated = this.#state.phase === "release_pending"
        && bindingSet.bindings.every((binding) => (
          humanoidObjectHandContacts(
            input.contacts,
            binding.object_id,
            binding.hand
          ).length === 0
        ));
      if (this.#state.phase !== "release_pending" || releaseSeparated) {
        this.#state.active_binding_set = null;
        this.#transition(
          releaseSeparated ? "released" : "lost",
          input.currentFrame,
          input.currentWorldRevision,
          releaseSeparated ? "release_completed" : "grasp_lost"
        );
      }
    } else if (this.#state.phase === "acquired") {
      this.#transition(
        "carrying",
        input.currentFrame,
        input.currentWorldRevision,
        "grasp_continued"
      );
    }
    this.#state = HumanoidCarriedObjectLifecycleCheckpointSchema.parse(this.#state);
    return {
      bindingSet: structuredClone(bindingSet),
      continuation: structuredClone(continuation),
      unauthorizedContacts: structuredClone(unauthorizedContacts),
      phase: this.#state.phase
    };
  }

  #verifyCurrentAuthority(
    currentFrame: number,
    currentWorldRevision: number
  ): void {
    const active = this.#state.active_binding_set;
    if (!active) return;
    if (active.source_frame > currentFrame
      || active.source_world_revision > currentWorldRevision) {
      throw new Error("Carried-object lifecycle binding is newer than the world");
    }
    if (active.source_frame === currentFrame
      && active.source_world_revision === currentWorldRevision) {
      verifyHumanoidCarriedObjectBindingSet({
        registry: this.#registry,
        currentFrame,
        currentWorldRevision,
        state: active
      });
      return;
    }
    const continuation = humanoidCarriedObjectContinuationEvidence({
      state: active,
      registry: this.#registry,
      currentFrame,
      currentWorldRevision
    });
    if (!continuation.continued) {
      throw new Error("Restored carried-object binding lacks current continuation evidence");
    }
  }

  #transition(
    phase: HumanoidCarriedObjectLifecycleCheckpoint["phase"],
    frame: number,
    worldRevision: number,
    reason: HumanoidCarriedObjectLifecycleCheckpoint["transition_reason"]
  ): void {
    this.#state.phase = phase;
    this.#state.transition_frame = FrameSchema.parse(frame);
    this.#state.transition_world_revision = WorldRevisionSchema.parse(worldRevision);
    this.#state.transition_reason = reason;
  }
}
