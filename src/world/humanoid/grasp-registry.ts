import { z } from "zod";
import {
  HumanoidGraspAssessmentSchema,
  HumanoidGraspContractSchema,
  HumanoidGraspTracker,
  HumanoidGraspTrackerCheckpointSchema,
  humanoidGraspContractSha256,
  humanoidGraspObservation,
  type HumanoidGraspAssessment,
  type HumanoidGraspContract,
  type HumanoidGraspHand
} from "./grasp-tracker.js";
import {
  HumanoidMotionOptionContractSchema,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionGraspAssessmentBinding
} from "./motion-option.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const HANDS = ["left", "right"] as const satisfies readonly HumanoidGraspHand[];

const parsedDefaultContract = HumanoidGraspContractSchema.parse({
  protocol: "humanoid-grasp-contract-v1",
  world_up: { x: 0, y: 1, z: 0 },
  minimum_distinct_contact_links: 2,
  minimum_contact_normal_force_n: 5,
  maximum_opposing_normal_dot: -0.5,
  maximum_opposing_position_dot: -0.5,
  minimum_opposing_contact_separation_m: 0.02,
  minimum_contact_radial_distance_m: 0.005,
  maximum_relative_translation_drift_m: 0.015,
  maximum_relative_rotation_drift_rad: 0.1,
  minimum_relative_pose_stable_frames: 8,
  minimum_lift_m: 0.04,
  minimum_lifted_hold_frames: 8,
  minimum_support_normal_force_n: 2,
  minimum_support_up_dot: 0.7
});
Object.freeze(parsedDefaultContract.world_up);
Object.freeze(parsedDefaultContract);

export const DEFAULT_HUMANOID_GRASP_CONTRACT: HumanoidGraspContract =
  parsedDefaultContract;

export const HumanoidGraspRegistryCheckpointSchema = z.object({
  protocol: z.literal("humanoid-grasp-registry-checkpoint-v1"),
  contract: HumanoidGraspContractSchema,
  contract_sha256: z.string().regex(SHA256_PATTERN),
  portable_object_ids: z.array(z.string().trim().min(1)),
  tracker: HumanoidGraspTrackerCheckpointSchema,
  last_frame: z.number().int().nonnegative().nullable(),
  last_assessments: z.array(HumanoidGraspAssessmentSchema)
}).strict().superRefine((checkpoint, context) => {
  const expectedHash = humanoidGraspContractSha256(checkpoint.contract);
  if (checkpoint.contract_sha256 !== expectedHash) {
    context.addIssue({
      code: "custom",
      path: ["contract_sha256"],
      message: "Grasp registry contract hash does not match its contract"
    });
  }
  if (humanoidGraspContractSha256(checkpoint.tracker.contract) !== expectedHash) {
    context.addIssue({
      code: "custom",
      path: ["tracker", "contract"],
      message: "Grasp registry tracker contract does not match"
    });
  }

  validateSortedUniqueStrings(
    checkpoint.portable_object_ids,
    "portable object id",
    ["portable_object_ids"],
    context
  );
  const allowedObjects = new Set(checkpoint.portable_object_ids);
  const trackKeys = new Set<string>();
  let previousTrackKey: string | null = null;
  checkpoint.tracker.tracks.forEach((track, index) => {
    const key = objectHandKey(track.object_id, track.hand);
    if (!allowedObjects.has(track.object_id)) {
      context.addIssue({
        code: "custom",
        path: ["tracker", "tracks", index, "object_id"],
        message: `Grasp registry track references unknown object: ${track.object_id}`
      });
    }
    if (previousTrackKey !== null && compareStrings(previousTrackKey, key) >= 0) {
      context.addIssue({
        code: "custom",
        path: ["tracker", "tracks", index],
        message: "Grasp registry tracks must be unique and deterministically sorted"
      });
    }
    if (checkpoint.last_frame === null || track.last_frame > checkpoint.last_frame) {
      context.addIssue({
        code: "custom",
        path: ["tracker", "tracks", index, "last_frame"],
        message: "Grasp registry track frame exceeds the registry frame"
      });
    }
    previousTrackKey = key;
    trackKeys.add(key);
  });

  const assessmentKeys = new Set<string>();
  let previousAssessmentKey: string | null = null;
  checkpoint.last_assessments.forEach((assessment, index) => {
    const key = objectHandKey(assessment.object_id, assessment.hand);
    if (!allowedObjects.has(assessment.object_id)) {
      context.addIssue({
        code: "custom",
        path: ["last_assessments", index, "object_id"],
        message: `Grasp registry assessment references unknown object: ${assessment.object_id}`
      });
    }
    if (previousAssessmentKey !== null
      && compareStrings(previousAssessmentKey, key) >= 0) {
      context.addIssue({
        code: "custom",
        path: ["last_assessments", index],
        message: "Grasp registry assessments must be unique and deterministically sorted"
      });
    }
    if (checkpoint.last_frame === null
      || assessment.frame !== checkpoint.last_frame) {
      context.addIssue({
        code: "custom",
        path: ["last_assessments", index, "frame"],
        message: "Grasp registry assessment is not from the registry frame"
      });
    }
    const track = checkpoint.tracker.tracks.find((candidate) => (
      objectHandKey(candidate.object_id, candidate.hand) === key
    ));
    if (!track || track.last_frame !== assessment.frame) {
      context.addIssue({
        code: "custom",
        path: ["last_assessments", index],
        message: "Grasp registry assessment does not match its tracker track"
      });
    }
    previousAssessmentKey = key;
    assessmentKeys.add(key);
  });

  for (const track of checkpoint.tracker.tracks) {
    if (track.last_frame === checkpoint.last_frame
      && !assessmentKeys.has(objectHandKey(track.object_id, track.hand))) {
      context.addIssue({
        code: "custom",
        path: ["last_assessments"],
        message: "Grasp registry is missing the current assessment for a tracker track"
      });
    }
  }
  for (const assessment of checkpoint.last_assessments) {
    if (!trackKeys.has(objectHandKey(assessment.object_id, assessment.hand))) {
      context.addIssue({
        code: "custom",
        path: ["last_assessments"],
        message: "Grasp registry assessment has no tracker track"
      });
    }
  }

  const assessmentHands = new Map<string, Set<HumanoidGraspHand>>();
  for (const assessment of checkpoint.last_assessments) {
    const hands = assessmentHands.get(assessment.object_id) ?? new Set();
    hands.add(assessment.hand);
    assessmentHands.set(assessment.object_id, hands);
  }
  for (const [objectId, hands] of assessmentHands) {
    if (hands.size !== HANDS.length) {
      context.addIssue({
        code: "custom",
        path: ["last_assessments"],
        message: `Grasp registry current object is missing a hand assessment: ${objectId}`
      });
    }
  }
});

export type HumanoidGraspRegistryCheckpoint = z.infer<
  typeof HumanoidGraspRegistryCheckpointSchema
>;

export interface HumanoidGraspRegistryOptions {
  portableObjectIds: readonly string[];
  contract?: HumanoidGraspContract;
  checkpoint?: HumanoidGraspRegistryCheckpoint;
}

export class HumanoidGraspRegistry {
  readonly #contract: HumanoidGraspContract;
  readonly #contractSha256: string;
  readonly #portableObjectIds: string[];
  readonly #portableObjectIdSet: ReadonlySet<string>;
  readonly #tracker: HumanoidGraspTracker;
  #lastFrame: number | null = null;
  #lastAssessments: HumanoidGraspAssessment[] = [];

  constructor(options: HumanoidGraspRegistryOptions) {
    this.#contract = HumanoidGraspContractSchema.parse(
      options.contract ?? DEFAULT_HUMANOID_GRASP_CONTRACT
    );
    this.#contractSha256 = humanoidGraspContractSha256(this.#contract);
    this.#portableObjectIds = normalizedObjectIds(options.portableObjectIds);
    this.#portableObjectIdSet = new Set(this.#portableObjectIds);

    if (!options.checkpoint) {
      this.#tracker = new HumanoidGraspTracker(this.#contract);
      return;
    }
    const checkpoint = HumanoidGraspRegistryCheckpointSchema.parse(
      options.checkpoint
    );
    if (checkpoint.contract_sha256 !== this.#contractSha256
      || JSON.stringify(checkpoint.contract) !== JSON.stringify(this.#contract)) {
      throw new Error("Humanoid grasp registry checkpoint contract does not match");
    }
    if (JSON.stringify(checkpoint.portable_object_ids)
      !== JSON.stringify(this.#portableObjectIds)) {
      throw new Error("Humanoid grasp registry checkpoint object set does not match");
    }
    this.#tracker = new HumanoidGraspTracker(this.#contract, checkpoint.tracker);
    this.#lastFrame = checkpoint.last_frame;
    this.#lastAssessments = structuredClone(checkpoint.last_assessments);
  }

  get contract(): HumanoidGraspContract {
    return structuredClone(this.#contract);
  }

  get contractSha256(): string {
    return this.#contractSha256;
  }

  get portableObjectIds(): readonly string[] {
    return [...this.#portableObjectIds];
  }

  get lastFrame(): number | null {
    return this.#lastFrame;
  }

  observe(
    rawFrame: number,
    snapshot: HumanoidSimulationSnapshot
  ): HumanoidGraspAssessment[] {
    const frame = z.number().int().nonnegative().parse(rawFrame);
    if (this.#lastFrame !== null) {
      if (frame < this.#lastFrame) {
        throw new Error(
          `Humanoid grasp registry frame moved backwards: ${frame} < ${this.#lastFrame}`
        );
      }
      if (frame === this.#lastFrame) return structuredClone(this.#lastAssessments);
    }

    const observations = this.#portableObjectIds
      .filter((objectId) => snapshot.objects[objectId] !== undefined)
      .flatMap((objectId) => {
        const object = snapshot.objects[objectId]!;
        if (object.id !== objectId) {
          throw new Error(
            `Humanoid grasp snapshot object key does not match its id: ${objectId}/${object.id}`
          );
        }
        return HANDS.map((hand) => humanoidGraspObservation({
          frame,
          objectId,
          hand,
          snapshot
        }));
      });
    const assessments = observations.map((observation) => (
      this.#tracker.observe(observation)
    )).sort(compareAssessments);
    this.#lastFrame = frame;
    this.#lastAssessments = structuredClone(assessments);
    return structuredClone(assessments);
  }

  assessmentsForFrame(rawFrame: number): HumanoidGraspAssessment[] {
    const frame = z.number().int().nonnegative().parse(rawFrame);
    return frame === this.#lastFrame
      ? structuredClone(this.#lastAssessments)
      : [];
  }

  bindingsForOption(
    rawOption: HumanoidMotionOptionContract,
    rawCurrentFrame: number
  ): HumanoidMotionOptionGraspAssessmentBinding[] {
    const option = HumanoidMotionOptionContractSchema.parse(rawOption);
    const currentFrame = z.number().int().nonnegative().parse(rawCurrentFrame);
    const current = new Map(this.assessmentsForFrame(currentFrame).map((assessment) => (
      [objectHandKey(assessment.object_id, assessment.hand), assessment]
    )));
    const referenced = new Set<string>();
    const bindings: HumanoidMotionOptionGraspAssessmentBinding[] = [];
    option.predicates.forEach((predicate, predicateIndex) => {
      if (predicate.type !== "grasp_verified") return;
      if (predicate.grasp_contract_sha256 !== this.#contractSha256) {
        throw new Error(
          `Humanoid grasp predicate ${predicateIndex} contract hash does not match authority`
        );
      }
      if (!this.#portableObjectIdSet.has(predicate.object_id)) {
        throw new Error(
          `Humanoid grasp predicate ${predicateIndex} references unknown portable object: ${predicate.object_id}`
        );
      }
      const key = objectHandKey(predicate.object_id, predicate.hand);
      if (referenced.has(key)) {
        throw new Error(
          `Duplicate humanoid grasp predicate object and hand: ${predicate.object_id}/${predicate.hand}`
        );
      }
      referenced.add(key);
      const assessment = current.get(key);
      if (!assessment) return;
      bindings.push({
        predicate_index: predicateIndex,
        contract_sha256: this.#contractSha256,
        assessment: structuredClone(assessment)
      });
    });
    return bindings;
  }

  checkpoint(): HumanoidGraspRegistryCheckpoint {
    return HumanoidGraspRegistryCheckpointSchema.parse({
      protocol: "humanoid-grasp-registry-checkpoint-v1",
      contract: this.#contract,
      contract_sha256: this.#contractSha256,
      portable_object_ids: this.#portableObjectIds,
      tracker: this.#tracker.checkpoint(),
      last_frame: this.#lastFrame,
      last_assessments: this.#lastAssessments
    });
  }

  fork(): HumanoidGraspRegistry {
    return new HumanoidGraspRegistry({
      contract: this.#contract,
      portableObjectIds: this.#portableObjectIds,
      checkpoint: this.checkpoint()
    });
  }
}

function normalizedObjectIds(rawObjectIds: readonly string[]): string[] {
  const objectIds = rawObjectIds.map((objectId) => (
    z.string().trim().min(1).parse(objectId)
  )).sort(compareStrings);
  for (let index = 1; index < objectIds.length; index += 1) {
    if (objectIds[index] === objectIds[index - 1]) {
      throw new Error(`Duplicate portable object id: ${objectIds[index]}`);
    }
  }
  return objectIds;
}

function validateSortedUniqueStrings(
  values: readonly string[],
  label: string,
  path: (string | number)[],
  context: z.RefinementCtx
): void {
  for (let index = 1; index < values.length; index += 1) {
    if (compareStrings(values[index - 1]!, values[index]!) >= 0) {
      context.addIssue({
        code: "custom",
        path: [...path, index],
        message: `Grasp registry ${label}s must be unique and deterministically sorted`
      });
    }
  }
}

function compareAssessments(
  left: HumanoidGraspAssessment,
  right: HumanoidGraspAssessment
): number {
  return compareStrings(
    objectHandKey(left.object_id, left.hand),
    objectHandKey(right.object_id, right.hand)
  );
}

function objectHandKey(objectId: string, hand: HumanoidGraspHand): string {
  return `${objectId}\0${hand}`;
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
