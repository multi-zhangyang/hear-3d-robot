import { createHash } from "node:crypto";
import { z } from "zod";
import {
  GoalSchema,
  type GoalPredicate
} from "./schema.js";
import { goalSha256 } from "./goal-identity.js";
import { ModelDecisionRefSchema } from "./model-call-authority.js";
import {
  GoalHistorySummarySchema
} from "./goal-history-summary-schema.js";

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const CandidateIdSchema = z.string().regex(/^goal-candidate:[a-f0-9]{64}$/);
const EpochIdSchema = z.string().regex(/^goal-epoch:[a-f0-9]{64}$/);

const GoalModelSourceSchema = ModelDecisionRefSchema;

const GoalPhysicalEvidenceBaseSchema = z.object({
  ref: z.string().trim().min(1),
  content_sha256: Sha256Schema,
  world_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative()
});

export const GoalPhysicalEvidenceSchema = z.discriminatedUnion("kind", [
  GoalPhysicalEvidenceBaseSchema.extend({
    kind: z.literal("world_observation")
  }).strict(),
  GoalPhysicalEvidenceBaseSchema.extend({
    kind: z.literal("world_checkpoint")
  }).strict(),
  GoalPhysicalEvidenceBaseSchema.extend({
    kind: z.literal("action_receipt")
  }).strict(),
  GoalPhysicalEvidenceBaseSchema.extend({
    kind: z.literal("goal_evaluation"),
    goal_content_sha256: Sha256Schema
  }).strict()
]);

const PhysicalEvidenceRefsSchema = z.object({
  proposal: z.array(z.string().trim().min(1)).min(1),
  resolution: z.array(z.string().trim().min(1))
}).strict();

const EpochPhysicalEvidenceRefsSchema = z.object({
  selection: z.array(z.string().trim().min(1)).min(1),
  resolution: z.array(z.string().trim().min(1))
}).strict();

const GoalCandidateBaseSchema = z.object({
  candidate_id: CandidateIdSchema,
  proposal_id: z.string().trim().min(1),
  source: GoalModelSourceSchema,
  goal: GoalSchema,
  mission_link: z.string().trim().min(1),
  identity_sha256: Sha256Schema,
  content_sha256: Sha256Schema,
  integrity_sha256: Sha256Schema,
  dependency_candidate_ids: z.array(CandidateIdSchema),
  status: z.enum([
    "proposed",
    "active",
    "completed",
    "blocked",
    "abandoned",
    "superseded",
    "expired"
  ]),
  physical_evidence_refs: PhysicalEvidenceRefsSchema,
  created_world_revision: z.number().int().nonnegative(),
  resolved_world_revision: z.number().int().nonnegative().nullable()
}).strict();

export const GoalCandidateSchema = GoalCandidateBaseSchema.superRefine(
  (candidate, context) => {
    if (candidate.identity_sha256 !== candidateIdentitySha256(candidate)) {
      context.addIssue({
        code: "custom",
        path: ["identity_sha256"],
        message: "Goal candidate model identity hash does not match its source"
      });
    }
    if (candidate.content_sha256 !== goalSha256(candidate.goal)) {
      context.addIssue({
        code: "custom",
        path: ["content_sha256"],
        message: "Goal candidate content hash does not match its goal"
      });
    }
    const integrity = candidateIntegritySha256(candidate);
    if (candidate.integrity_sha256 !== integrity
      || candidate.candidate_id !== `goal-candidate:${integrity}`) {
      context.addIssue({
        code: "custom",
        path: ["integrity_sha256"],
        message: "Goal candidate immutable identity is inconsistent"
      });
    }
    if (!isUniqueSorted(candidate.dependency_candidate_ids)) {
      context.addIssue({
        code: "custom",
        path: ["dependency_candidate_ids"],
        message: "Goal candidate dependencies must be unique and sorted"
      });
    }
    if (!isUniqueSorted(candidate.physical_evidence_refs.proposal)
      || !isUniqueSorted(candidate.physical_evidence_refs.resolution)) {
      context.addIssue({
        code: "custom",
        path: ["physical_evidence_refs"],
        message: "Goal candidate evidence references must be unique and sorted"
      });
    }
    const terminal = candidate.status !== "proposed" && candidate.status !== "active";
    if (terminal !== (candidate.resolved_world_revision !== null)
      || terminal !== (candidate.physical_evidence_refs.resolution.length > 0)) {
      context.addIssue({
        code: "custom",
        path: ["resolved_world_revision"],
        message: "Goal candidate resolution state requires revision and evidence together"
      });
    }
    if (candidate.resolved_world_revision !== null
      && candidate.resolved_world_revision < candidate.created_world_revision) {
      context.addIssue({
        code: "custom",
        path: ["resolved_world_revision"],
        message: "Goal candidate resolution cannot precede its creation"
      });
    }
  }
);

const GoalEpochBaseSchema = z.object({
  epoch_id: EpochIdSchema,
  epoch_index: z.number().int().nonnegative(),
  previous_epoch_id: EpochIdSchema.nullable(),
  candidate_id: CandidateIdSchema,
  candidate_source: GoalModelSourceSchema,
  selected_by: GoalModelSourceSchema,
  candidate_identity_sha256: Sha256Schema,
  candidate_content_sha256: Sha256Schema,
  dependency_candidate_ids: z.array(CandidateIdSchema),
  identity_sha256: Sha256Schema,
  status: z.enum([
    "active",
    "completed",
    "blocked",
    "abandoned",
    "superseded",
    "expired"
  ]),
  retired_by: GoalModelSourceSchema.nullable(),
  retirement_reason: z.string().trim().min(1).nullable(),
  physical_evidence_refs: EpochPhysicalEvidenceRefsSchema,
  created_world_revision: z.number().int().nonnegative(),
  resolved_world_revision: z.number().int().nonnegative().nullable()
}).strict();

export const GoalEpochSchema = GoalEpochBaseSchema.superRefine((epoch, context) => {
  const identity = epochIdentitySha256(epoch);
  if (epoch.identity_sha256 !== identity
    || epoch.epoch_id !== `goal-epoch:${identity}`) {
    context.addIssue({
      code: "custom",
      path: ["identity_sha256"],
      message: "Goal epoch immutable identity is inconsistent"
    });
  }
  if (!isUniqueSorted(epoch.dependency_candidate_ids)
    || !isUniqueSorted(epoch.physical_evidence_refs.selection)
    || !isUniqueSorted(epoch.physical_evidence_refs.resolution)) {
    context.addIssue({
      code: "custom",
      path: ["physical_evidence_refs"],
      message: "Goal epoch dependencies and evidence references must be unique and sorted"
    });
  }
  const terminal = epoch.status !== "active";
  if (terminal !== (epoch.resolved_world_revision !== null)
    || terminal !== (epoch.physical_evidence_refs.resolution.length > 0)) {
    context.addIssue({
      code: "custom",
      path: ["resolved_world_revision"],
      message: "Goal epoch resolution state requires revision and evidence together"
    });
  }
  if (epoch.resolved_world_revision !== null
    && epoch.resolved_world_revision < epoch.created_world_revision) {
    context.addIssue({
      code: "custom",
      path: ["resolved_world_revision"],
      message: "Goal epoch resolution cannot precede its creation"
    });
  }
  const retired = epoch.status !== "active" && epoch.status !== "completed";
  if (retired !== (epoch.retired_by !== null)
    || retired !== (epoch.retirement_reason !== null)) {
    context.addIssue({
      code: "custom",
      path: ["retired_by"],
      message: "A retired Goal epoch requires model provenance and a reason"
    });
  }
});

const GoalDAGArchiveStateSchema = z.object({
  record_count: z.number().int().nonnegative(),
  last_record_sha256: Sha256Schema.nullable(),
  last_epoch_id: EpochIdSchema.nullable(),
  retained_candidate_ids: z.array(CandidateIdSchema),
  summary: GoalHistorySummarySchema.nullable()
}).strict().superRefine((archive, context) => {
  const empty = archive.record_count === 0;
  if (empty !== (archive.last_record_sha256 === null)
    || empty !== (archive.last_epoch_id === null)) {
    context.addIssue({
      code: "custom",
      path: ["record_count"],
      message: "Goal history archive head is inconsistent"
    });
  }
  if (!isUniqueSorted(archive.retained_candidate_ids)) {
    context.addIssue({
      code: "custom",
      path: ["retained_candidate_ids"],
      message: "Retained archived Goal candidates must be unique and sorted"
    });
  }
  if (archive.summary
    && (archive.summary.archived_epoch_count !== archive.record_count
      || archive.summary.last_record_sha256 !== archive.last_record_sha256)) {
    context.addIssue({
      code: "custom",
      path: ["summary"],
      message: "Goal history summary does not match the archive head"
    });
  }
});

type GoalDAGArchiveState = z.infer<typeof GoalDAGArchiveStateSchema>;

const EmptyGoalDAGArchiveState: GoalDAGArchiveState = {
  record_count: 0,
  last_record_sha256: null,
  last_epoch_id: null,
  retained_candidate_ids: [],
  summary: {
    version: 1,
    archived_epoch_count: 0,
    last_record_sha256: null,
    records_without_alternate_history: 0,
    outcomes: {
      selected: {
        total: 0,
        completed: 0,
        blocked: 0,
        abandoned: 0,
        superseded: 0,
        expired: 0
      },
      not_selected: 0,
      predicate_outcomes: [],
      entity_outcomes: []
    }
  }
};

const LegacyEmptyGoalDAGArchiveState: GoalDAGArchiveState = {
  ...EmptyGoalDAGArchiveState,
  summary: null
};

const GoalDAGBaseSchema = z.object({
  version: z.literal(2),
  status: z.enum(["awaiting_model_selection", "active"]),
  candidates: z.record(CandidateIdSchema, GoalCandidateSchema),
  candidate_sequences: z.record(CandidateIdSchema, z.number().int().positive()),
  next_candidate_sequence: z.number().int().positive(),
  epochs: z.array(GoalEpochSchema),
  current_epoch_id: EpochIdSchema.nullable(),
  next_epoch_index: z.number().int().nonnegative(),
  evidence: z.record(z.string().trim().min(1), GoalPhysicalEvidenceSchema),
  archive: GoalDAGArchiveStateSchema,
  state_sha256: Sha256Schema
}).strict();

const ValidatedGoalDAGSchema = GoalDAGBaseSchema.superRefine((dag, context) => {
  if (dag.state_sha256 !== goalDAGStateSha256(dag)
    && (dag.archive.summary !== null
      || dag.state_sha256 !== goalDAGWithoutArchiveSummaryStateSha256(dag))
    && (!isLegacyCompatibleGoalDAG(dag)
      || dag.state_sha256 !== legacyGoalDAGStateSha256(dag))) {
    context.addIssue({
      code: "custom",
      path: ["state_sha256"],
      message: "Goal DAG state hash does not match its persisted contents"
    });
  }

  const candidateIds = Object.keys(dag.candidates);
  const sequenceIds = Object.keys(dag.candidate_sequences);
  if (canonicalJson(candidateIds.sort(compareCodePoints))
    !== canonicalJson(sequenceIds.sort(compareCodePoints))) {
    context.addIssue({
      code: "custom",
      path: ["candidate_sequences"],
      message: "Goal candidate sequences must cover the working candidate set"
    });
  }
  const sequences = Object.values(dag.candidate_sequences);
  if (new Set(sequences).size !== sequences.length
    || sequences.some((sequence) => sequence >= dag.next_candidate_sequence)) {
    context.addIssue({
      code: "custom",
      path: ["candidate_sequences"],
      message: "Goal candidate sequences must be unique and precede the next sequence"
    });
  }

  for (const [key, candidate] of Object.entries(dag.candidates)) {
    if (key !== candidate.candidate_id) {
      context.addIssue({
        code: "custom",
        path: ["candidates", key, "candidate_id"],
        message: "Goal candidate record key does not match its identity"
      });
    }
    for (const dependencyId of candidate.dependency_candidate_ids) {
      if (!dag.candidates[dependencyId]) {
        context.addIssue({
          code: "custom",
          path: ["candidates", key, "dependency_candidate_ids"],
          message: `Goal candidate dependency is missing: ${dependencyId}`
        });
      }
    }
    checkEvidenceRefs(
      dag,
      candidate.physical_evidence_refs.proposal,
      candidate.created_world_revision,
      ["candidates", key, "physical_evidence_refs", "proposal"],
      context
    );
    if (candidate.resolved_world_revision !== null) {
      checkEvidenceRefs(
        dag,
        candidate.physical_evidence_refs.resolution,
        candidate.resolved_world_revision,
        ["candidates", key, "physical_evidence_refs", "resolution"],
        context,
        candidate.status === "completed" ? candidate.content_sha256 : undefined
      );
    }
  }
  checkDependencyCycles(dag, context);

  for (const [ref, evidence] of Object.entries(dag.evidence)) {
    if (ref !== evidence.ref) {
      context.addIssue({
        code: "custom",
        path: ["evidence", ref, "ref"],
        message: "Physical evidence record key does not match its reference"
      });
    }
  }

  if (dag.next_epoch_index !== dag.archive.record_count + dag.epochs.length) {
    context.addIssue({
      code: "custom",
      path: ["next_epoch_index"],
      message: "Goal DAG epoch index is not contiguous"
    });
  }

  const seenCandidates = new Set<string>();
  dag.epochs.forEach((epoch, index) => {
    const candidate = dag.candidates[epoch.candidate_id];
    const expectedIndex = dag.archive.record_count + index;
    const expectedPreviousEpochId = dag.epochs[index - 1]?.epoch_id
      ?? dag.archive.last_epoch_id;
    if (epoch.epoch_index !== expectedIndex
      || epoch.previous_epoch_id !== expectedPreviousEpochId) {
      context.addIssue({
        code: "custom",
        path: ["epochs", index],
        message: "Goal epoch chain is not contiguous"
      });
    }
    if (!candidate) return;
    if (seenCandidates.has(candidate.candidate_id)) {
      context.addIssue({
        code: "custom",
        path: ["epochs", index, "candidate_id"],
        message: "A goal candidate cannot be activated more than once"
      });
    }
    seenCandidates.add(candidate.candidate_id);
    if (canonicalJson(epoch.candidate_source) !== canonicalJson(candidate.source)
      || epoch.candidate_identity_sha256 !== candidate.identity_sha256
      || epoch.candidate_content_sha256 !== candidate.content_sha256
      || canonicalJson(epoch.dependency_candidate_ids)
        !== canonicalJson(candidate.dependency_candidate_ids)) {
      context.addIssue({
        code: "custom",
        path: ["epochs", index, "candidate_id"],
        message: "Goal epoch does not pin the selected candidate identity"
      });
    }
    if (epoch.created_world_revision < candidate.created_world_revision
      || epoch.status !== candidate.status
      || epoch.resolved_world_revision !== candidate.resolved_world_revision
      || canonicalJson(epoch.physical_evidence_refs.resolution)
        !== canonicalJson(candidate.physical_evidence_refs.resolution)) {
      context.addIssue({
        code: "custom",
        path: ["epochs", index, "status"],
        message: "Goal epoch and candidate lifecycle are inconsistent"
      });
    }
    if (candidate.status !== "proposed") {
      for (const dependencyId of candidate.dependency_candidate_ids) {
        const dependency = dag.candidates[dependencyId];
        if (dependency?.status !== "completed"
          || dependency.resolved_world_revision === null
          || dependency.resolved_world_revision > epoch.created_world_revision) {
          context.addIssue({
            code: "custom",
            path: ["epochs", index, "dependency_candidate_ids"],
            message: "An activated goal epoch requires completed dependencies"
          });
        }
      }
    }
    checkEvidenceRefs(
      dag,
      epoch.physical_evidence_refs.selection,
      epoch.created_world_revision,
      ["epochs", index, "physical_evidence_refs", "selection"],
      context
    );
    if (epoch.resolved_world_revision !== null) {
      checkEvidenceRefs(
        dag,
        epoch.physical_evidence_refs.resolution,
        epoch.resolved_world_revision,
        ["epochs", index, "physical_evidence_refs", "resolution"],
        context,
        epoch.status === "completed" ? epoch.candidate_content_sha256 : undefined
      );
    }
  });
  const retainedArchived = new Set(dag.archive.retained_candidate_ids);
  const workingAlternates = new Set(Object.values(dag.candidates).flatMap((candidate) => (
    dag.epochs.some((epoch) => isExpiredSlateAlternate(dag, candidate, epoch))
      ? [candidate.candidate_id]
      : []
  )));
  for (const candidate of Object.values(dag.candidates)) {
    const activated = seenCandidates.has(candidate.candidate_id);
    const retained = retainedArchived.has(candidate.candidate_id);
    const alternate = workingAlternates.has(candidate.candidate_id);
    if ((candidate.status === "proposed" && activated)
      || (candidate.status !== "proposed" && !activated && !retained && !alternate)
      || (retained && (activated
        || candidate.status === "proposed"
        || candidate.status === "active"))) {
      context.addIssue({
        code: "custom",
        path: ["candidates", candidate.candidate_id, "status"],
        message: "Goal candidate lifecycle does not match the epoch chain"
      });
    }
  }
  const expectedRetained = Object.values(dag.candidates)
    .filter((candidate) => candidate.status !== "proposed"
      && candidate.status !== "active"
      && !seenCandidates.has(candidate.candidate_id)
      && !workingAlternates.has(candidate.candidate_id))
    .map((candidate) => candidate.candidate_id)
    .sort(compareCodePoints);
  if (canonicalJson(expectedRetained)
    !== canonicalJson(dag.archive.retained_candidate_ids)) {
    context.addIssue({
      code: "custom",
      path: ["archive", "retained_candidate_ids"],
      message: "Goal history archive retention does not match working dependencies"
    });
  }

  const activeEpochs = dag.epochs.filter((epoch) => epoch.status === "active");
  const activeCandidates = Object.values(dag.candidates)
    .filter((candidate) => candidate.status === "active");
  if (dag.status === "active") {
    if (activeEpochs.length !== 1 || activeCandidates.length !== 1
      || dag.current_epoch_id !== activeEpochs[0]?.epoch_id
      || activeEpochs[0]?.candidate_id !== activeCandidates[0]?.candidate_id
      || dag.epochs.at(-1)?.epoch_id !== dag.current_epoch_id) {
      context.addIssue({
        code: "custom",
        path: ["current_epoch_id"],
        message: "Active Goal DAG state must identify exactly one latest active epoch"
      });
    }
  } else if (dag.current_epoch_id !== null
    || activeEpochs.length > 0
    || activeCandidates.length > 0) {
    context.addIssue({
      code: "custom",
      path: ["current_epoch_id"],
      message: "A Goal DAG awaiting model selection cannot carry an active goal"
    });
  }
});

export const GoalDAGSchema = z.preprocess(
  normalizePersistedGoalDAG,
  ValidatedGoalDAGSchema
);

export type GoalModelSource = z.infer<typeof GoalModelSourceSchema>;
export type GoalPhysicalEvidence = z.infer<typeof GoalPhysicalEvidenceSchema>;
export type GoalCandidate = z.infer<typeof GoalCandidateSchema>;
export type GoalEpoch = z.infer<typeof GoalEpochSchema>;
export type GoalDAG = z.infer<typeof GoalDAGSchema>;

export interface GoalHarnessValidation {
  authorized_model_sources: ReadonlyArray<{
    agent_id: string;
    agent_manifest_sha256: string;
    agent_manifest_epoch_id: string;
  }>;
  is_model_call_authoritative(
    source: GoalModelSource,
    expectedToolName:
      | "submit_goal_candidates"
      | "select_goal_candidate"
      | "retire_goal_epoch"
      | "continue_goal_epoch"
  ): boolean;
  evidence_by_ref(ref: string): unknown;
  is_predicate_observable(input: {
    predicate: GoalPredicate;
    predicate_index: number;
    world_revision: number;
    evidence_refs: readonly string[];
  }): boolean;
}

const GoalCandidateProposalSchema = z.object({
  proposal_id: z.string().trim().min(1),
  source: GoalModelSourceSchema,
  goal: GoalSchema,
  mission_link: z.string().trim().min(1),
  dependency_candidate_ids: z.array(CandidateIdSchema),
  proposal_evidence_refs: z.array(z.string().trim().min(1)).min(1),
  created_world_revision: z.number().int().nonnegative()
}).strict();

const GoalCandidateSelectionSchema = z.object({
  candidate_id: CandidateIdSchema,
  selected_by: GoalModelSourceSchema,
  selection_evidence_refs: z.array(z.string().trim().min(1)).min(1),
  created_world_revision: z.number().int().nonnegative()
}).strict();

const GoalEpochCompletionSchema = z.object({
  resolution_evidence_refs: z.array(z.string().trim().min(1)).min(1),
  resolved_world_revision: z.number().int().nonnegative()
}).strict();

export class GoalDAGValidationError extends Error {
  constructor(
    readonly code: string,
    message: string
  ) {
    super(message);
    this.name = "GoalDAGValidationError";
  }
}

export function createGoalDAG(): GoalDAG {
  return rehashGoalDAG({
    version: 2,
    status: "awaiting_model_selection",
    candidates: {},
    candidate_sequences: {},
    next_candidate_sequence: 1,
    epochs: [],
    current_epoch_id: null,
    next_epoch_index: 0,
    evidence: {},
    archive: EmptyGoalDAGArchiveState
  });
}

export function proposeGoalCandidate(
  persisted: GoalDAG,
  input: z.input<typeof GoalCandidateProposalSchema>,
  harness: GoalHarnessValidation
): GoalDAG {
  const dag = restoreGoalDAG(persisted, harness);
  const proposal = GoalCandidateProposalSchema.parse(input);
  assertGoalModelSource(proposal.source, harness, "submit_goal_candidates");
  const dependencies = uniqueSorted(proposal.dependency_candidate_ids);
  if (dependencies.length !== proposal.dependency_candidate_ids.length) {
    throw new GoalDAGValidationError(
      "duplicate_dependency",
      "A model goal candidate cannot repeat a dependency"
    );
  }
  for (const dependencyId of dependencies) {
    const dependency = dag.candidates[dependencyId];
    if (!dependency) {
      throw new GoalDAGValidationError(
        "missing_dependency",
        `Model goal candidate references a missing dependency: ${dependencyId}`
      );
    }
    if (dependency.status !== "completed"
      || dependency.resolved_world_revision === null
      || dependency.resolved_world_revision > proposal.created_world_revision) {
      throw new GoalDAGValidationError(
        "dependency_not_completed",
        `Model goal candidate dependency is not complete: ${dependencyId}`
      );
    }
  }
  const proposalEvidenceRefs = uniqueSorted(proposal.proposal_evidence_refs);
  if (proposalEvidenceRefs.length !== proposal.proposal_evidence_refs.length) {
    throw new GoalDAGValidationError(
      "duplicate_evidence",
      "A model goal candidate cannot repeat physical evidence"
    );
  }
  const evidence = registerGoalEvidence(
    dag.evidence,
    proposalEvidenceRefs,
    proposal.created_world_revision,
    harness
  );
  proposal.goal.predicates.forEach((predicate, predicateIndex) => {
    if (!harness.is_predicate_observable({
      predicate,
      predicate_index: predicateIndex,
      world_revision: proposal.created_world_revision,
      evidence_refs: proposalEvidenceRefs
    })) {
      throw new GoalDAGValidationError(
        "predicate_not_observable",
        `Model goal predicate ${predicateIndex} is not observable at the proposal revision`
      );
    }
  });

  const immutable = {
    proposal_id: proposal.proposal_id,
    source: proposal.source,
    goal: proposal.goal,
    mission_link: proposal.mission_link,
    dependency_candidate_ids: dependencies,
    proposal_evidence_refs: proposalEvidenceRefs,
    created_world_revision: proposal.created_world_revision
  };
  const identitySha256 = candidateIdentitySha256(immutable);
  const contentSha256 = goalSha256(proposal.goal);
  const integritySha256 = candidateIntegritySha256({
    ...immutable,
    identity_sha256: identitySha256,
    content_sha256: contentSha256,
    physical_evidence_refs: {
      proposal: proposalEvidenceRefs
    }
  });
  const candidateId = `goal-candidate:${integritySha256}`;
  if (dag.candidates[candidateId]) {
    throw new GoalDAGValidationError(
      "duplicate_candidate",
      `Model goal candidate already exists: ${candidateId}`
    );
  }
  const candidate = GoalCandidateSchema.parse({
    candidate_id: candidateId,
    proposal_id: proposal.proposal_id,
    source: proposal.source,
    goal: proposal.goal,
    mission_link: proposal.mission_link,
    identity_sha256: identitySha256,
    content_sha256: contentSha256,
    integrity_sha256: integritySha256,
    dependency_candidate_ids: dependencies,
    status: "proposed",
    physical_evidence_refs: {
      proposal: proposalEvidenceRefs,
      resolution: []
    },
    created_world_revision: proposal.created_world_revision,
    resolved_world_revision: null
  });
  return rehashGoalDAG({
    ...dag,
    candidates: { ...dag.candidates, [candidateId]: candidate },
    candidate_sequences: {
      ...dag.candidate_sequences,
      [candidateId]: dag.next_candidate_sequence
    },
    next_candidate_sequence: dag.next_candidate_sequence + 1,
    evidence
  });
}

export function selectGoalCandidate(
  persisted: GoalDAG,
  input: z.input<typeof GoalCandidateSelectionSchema>,
  harness: GoalHarnessValidation
): GoalDAG {
  const dag = restoreGoalDAG(persisted, harness);
  const selection = GoalCandidateSelectionSchema.parse(input);
  if (dag.status !== "awaiting_model_selection") {
    throw new GoalDAGValidationError(
      "goal_epoch_active",
      "A model cannot select another goal while an epoch is active"
    );
  }
  assertGoalModelSource(selection.selected_by, harness, "select_goal_candidate");
  const candidate = dag.candidates[selection.candidate_id];
  if (!candidate || candidate.status !== "proposed") {
    throw new GoalDAGValidationError(
      "candidate_not_proposed",
      `The model-selected goal candidate is unavailable: ${selection.candidate_id}`
    );
  }
  if (selection.created_world_revision < candidate.created_world_revision) {
    throw new GoalDAGValidationError(
      "world_revision_regression",
      "A goal epoch cannot begin before its candidate was created"
    );
  }
  for (const dependencyId of candidate.dependency_candidate_ids) {
    const dependency = dag.candidates[dependencyId];
    if (dependency?.status !== "completed"
      || dependency.resolved_world_revision === null
      || dependency.resolved_world_revision > selection.created_world_revision) {
      throw new GoalDAGValidationError(
        "dependency_not_completed",
        `The model-selected goal dependency is not complete: ${dependencyId}`
      );
    }
  }
  const selectionEvidenceRefs = uniqueSorted(selection.selection_evidence_refs);
  if (selectionEvidenceRefs.length !== selection.selection_evidence_refs.length) {
    throw new GoalDAGValidationError(
      "duplicate_evidence",
      "A goal epoch selection cannot repeat physical evidence"
    );
  }
  const evidence = registerGoalEvidence(
    dag.evidence,
    selectionEvidenceRefs,
    selection.created_world_revision,
    harness
  );
  const epochImmutable = {
    epoch_index: dag.next_epoch_index,
    previous_epoch_id: dag.epochs.at(-1)?.epoch_id ?? dag.archive.last_epoch_id,
    candidate_id: candidate.candidate_id,
    candidate_source: candidate.source,
    selected_by: selection.selected_by,
    candidate_identity_sha256: candidate.identity_sha256,
    candidate_content_sha256: candidate.content_sha256,
    dependency_candidate_ids: candidate.dependency_candidate_ids,
    physical_evidence_refs: { selection: selectionEvidenceRefs },
    created_world_revision: selection.created_world_revision
  };
  const identitySha256 = epochIdentitySha256(epochImmutable);
  const epoch = GoalEpochSchema.parse({
    epoch_id: `goal-epoch:${identitySha256}`,
    ...epochImmutable,
    identity_sha256: identitySha256,
    status: "active",
    retired_by: null,
    retirement_reason: null,
    physical_evidence_refs: {
      selection: selectionEvidenceRefs,
      resolution: []
    },
    resolved_world_revision: null
  });
  const candidates = Object.fromEntries(Object.entries(dag.candidates).map(([
    candidateId,
    entry
  ]) => {
    if (candidateId === candidate.candidate_id) {
      return [candidateId, { ...entry, status: "active" as const }];
    }
    if (entry.status !== "proposed" || !sameCandidateSlate(entry, candidate)) {
      return [candidateId, entry];
    }
    return [candidateId, {
      ...entry,
      status: "expired" as const,
      physical_evidence_refs: {
        ...entry.physical_evidence_refs,
        resolution: selectionEvidenceRefs
      },
      resolved_world_revision: selection.created_world_revision
    }];
  }));
  return rehashGoalDAG({
    ...dag,
    status: "active",
    candidates,
    epochs: [...dag.epochs, epoch],
    current_epoch_id: epoch.epoch_id,
    next_epoch_index: dag.next_epoch_index + 1,
    evidence
  });
}

export function completeGoalEpoch(
  persisted: GoalDAG,
  input: z.input<typeof GoalEpochCompletionSchema>,
  harness: GoalHarnessValidation
): GoalDAG {
  const dag = restoreGoalDAG(persisted, harness);
  const completion = GoalEpochCompletionSchema.parse(input);
  const epochIndex = dag.epochs.findIndex(
    (epoch) => epoch.epoch_id === dag.current_epoch_id
  );
  const epoch = dag.epochs[epochIndex];
  if (dag.status !== "active" || !epoch || epoch.status !== "active") {
    throw new GoalDAGValidationError(
      "no_active_goal_epoch",
      "There is no active model-selected goal epoch to complete"
    );
  }
  if (completion.resolved_world_revision < epoch.created_world_revision) {
    throw new GoalDAGValidationError(
      "world_revision_regression",
      "A goal epoch cannot complete before it began"
    );
  }
  const completionEvidenceRefs = uniqueSorted(completion.resolution_evidence_refs);
  if (completionEvidenceRefs.length !== completion.resolution_evidence_refs.length) {
    throw new GoalDAGValidationError(
      "duplicate_evidence",
      "A goal epoch completion cannot repeat physical evidence"
    );
  }
  const evidence = registerGoalEvidence(
    dag.evidence,
    completionEvidenceRefs,
    completion.resolved_world_revision,
    harness
  );
  const candidate = dag.candidates[epoch.candidate_id];
  if (!candidate || candidate.status !== "active") {
    throw new GoalDAGValidationError(
      "active_candidate_missing",
      "The active goal epoch has no matching active candidate"
    );
  }
  const hasBoundGoalEvaluation = completionEvidenceRefs.some((ref) => {
    const entry = evidence[ref];
    return entry?.kind === "goal_evaluation"
      && entry.goal_content_sha256 === candidate.content_sha256;
  });
  if (!hasBoundGoalEvaluation) {
    throw new GoalDAGValidationError(
      "goal_evaluation_missing",
      "Goal completion requires physical evaluation evidence bound to the goal content"
    );
  }
  const completedEpoch: GoalEpoch = {
    ...epoch,
    status: "completed",
    physical_evidence_refs: {
      ...epoch.physical_evidence_refs,
      resolution: completionEvidenceRefs
    },
    resolved_world_revision: completion.resolved_world_revision
  };
  const epochs = [...dag.epochs];
  epochs[epochIndex] = completedEpoch;
  return rehashGoalDAG({
    ...dag,
    status: "awaiting_model_selection",
    candidates: {
      ...dag.candidates,
      [candidate.candidate_id]: {
        ...candidate,
        status: "completed",
        physical_evidence_refs: {
          ...candidate.physical_evidence_refs,
          resolution: completionEvidenceRefs
        },
        resolved_world_revision: completion.resolved_world_revision
      }
    },
    epochs,
    current_epoch_id: null,
    evidence
  });
}

export function restoreGoalDAG(
  persisted: unknown,
  harness: GoalHarnessValidation
): GoalDAG {
  const dag = GoalDAGSchema.parse(persisted);
  for (const candidate of Object.values(dag.candidates)) {
    assertGoalModelSource(candidate.source, harness, "submit_goal_candidates");
    candidate.goal.predicates.forEach((predicate, predicateIndex) => {
      if (!harness.is_predicate_observable({
        predicate,
        predicate_index: predicateIndex,
        world_revision: candidate.created_world_revision,
        evidence_refs: candidate.physical_evidence_refs.proposal
      })) {
        throw new GoalDAGValidationError(
          "predicate_not_observable",
          `Persisted model goal predicate ${predicateIndex} is no longer verifiable`
        );
      }
    });
  }
  for (const epoch of dag.epochs) {
    assertGoalModelSource(epoch.selected_by, harness, "select_goal_candidate");
    if (epoch.retired_by) {
      assertGoalModelSource(epoch.retired_by, harness, "retire_goal_epoch");
    }
  }
  for (const evidence of Object.values(dag.evidence)) {
    const authoritative = GoalPhysicalEvidenceSchema.safeParse(
      harness.evidence_by_ref(evidence.ref)
    );
    if (!authoritative.success) {
      throw new GoalDAGValidationError(
        "evidence_unavailable",
        `Physical evidence is unavailable during Goal DAG recovery: ${evidence.ref}`
      );
    }
    if (canonicalJson(authoritative.data) !== canonicalJson(evidence)) {
      throw new GoalDAGValidationError(
        "evidence_mismatch",
        `Physical evidence changed during Goal DAG recovery: ${evidence.ref}`
      );
    }
  }
  return normalizeSelectedSlateAlternates(dag);
}

function normalizeSelectedSlateAlternates(dag: GoalDAG): GoalDAG {
  const latestEpochBySlate = new Map<string, GoalEpoch>();
  for (const epoch of dag.epochs) {
    const selected = dag.candidates[epoch.candidate_id];
    if (selected) latestEpochBySlate.set(candidateSlateKey(selected), epoch);
  }
  let changed = false;
  const candidates = Object.fromEntries(Object.entries(dag.candidates).map(([
    candidateId,
    candidate
  ]) => {
    if (candidate.status !== "proposed") return [candidateId, candidate];
    const epoch = latestEpochBySlate.get(candidateSlateKey(candidate));
    if (!epoch || epoch.candidate_id === candidateId) return [candidateId, candidate];
    changed = true;
    return [candidateId, {
      ...candidate,
      status: "expired" as const,
      physical_evidence_refs: {
        ...candidate.physical_evidence_refs,
        resolution: epoch.physical_evidence_refs.selection
      },
      resolved_world_revision: epoch.created_world_revision
    }];
  }));
  return changed ? rehashGoalDAG({ ...dag, candidates }) : dag;
}

function isExpiredSlateAlternate(
  dag: GoalDAG,
  candidate: GoalCandidate,
  epoch: GoalEpoch
): boolean {
  const selected = dag.candidates[epoch.candidate_id];
  return candidate.status === "expired"
    && candidate.candidate_id !== epoch.candidate_id
    && selected !== undefined
    && sameCandidateSlate(candidate, selected)
    && candidate.resolved_world_revision === epoch.created_world_revision
    && canonicalJson(candidate.physical_evidence_refs.resolution)
      === canonicalJson(epoch.physical_evidence_refs.selection);
}

function sameCandidateSlate(left: GoalCandidate, right: GoalCandidate): boolean {
  return candidateSlateKey(left) === candidateSlateKey(right);
}

function candidateSlateKey(candidate: GoalCandidate): string {
  return canonicalJson(candidate.source);
}

function candidateIdentitySha256(candidate: {
  proposal_id: string;
  source: GoalModelSource;
}): string {
  return sha256(canonicalJson({
    proposal_id: candidate.proposal_id,
    source: candidate.source
  }));
}

function candidateIntegritySha256(candidate: {
  identity_sha256: string;
  content_sha256: string;
  mission_link: string;
  dependency_candidate_ids: readonly string[];
  physical_evidence_refs: { proposal: readonly string[] };
  created_world_revision: number;
}): string {
  return sha256(canonicalJson({
    identity_sha256: candidate.identity_sha256,
    content_sha256: candidate.content_sha256,
    mission_link: candidate.mission_link,
    dependency_candidate_ids: candidate.dependency_candidate_ids,
    proposal_evidence_refs: candidate.physical_evidence_refs.proposal,
    created_world_revision: candidate.created_world_revision
  }));
}

function epochIdentitySha256(epoch: {
  epoch_index: number;
  previous_epoch_id: string | null;
  candidate_id: string;
  candidate_source: GoalModelSource;
  selected_by: GoalModelSource;
  candidate_identity_sha256: string;
  candidate_content_sha256: string;
  dependency_candidate_ids: readonly string[];
  physical_evidence_refs: { selection: readonly string[] };
  created_world_revision: number;
}): string {
  return sha256(canonicalJson({
    epoch_index: epoch.epoch_index,
    previous_epoch_id: epoch.previous_epoch_id,
    candidate_id: epoch.candidate_id,
    candidate_source: epoch.candidate_source,
    selected_by: epoch.selected_by,
    candidate_identity_sha256: epoch.candidate_identity_sha256,
    candidate_content_sha256: epoch.candidate_content_sha256,
    dependency_candidate_ids: epoch.dependency_candidate_ids,
    selection_evidence_refs: epoch.physical_evidence_refs.selection,
    created_world_revision: epoch.created_world_revision
  }));
}

function goalDAGStateSha256(dag: { state_sha256?: string } & Record<string, unknown>): string {
  const { state_sha256: _stateSha256, ...contents } = dag;
  return sha256(canonicalJson(contents));
}

function legacyGoalDAGStateSha256(dag: z.infer<typeof GoalDAGBaseSchema>): string {
  const {
    state_sha256: _stateSha256,
    candidate_sequences: _candidateSequences,
    next_candidate_sequence: _nextCandidateSequence,
    archive: _archive,
    ...contents
  } = dag;
  return sha256(canonicalJson({ ...contents, version: 1 }));
}

function goalDAGWithoutArchiveSummaryStateSha256(
  dag: z.infer<typeof GoalDAGBaseSchema>
): string {
  const {
    state_sha256: _stateSha256,
    archive,
    ...contents
  } = dag;
  const { summary: _summary, ...legacyArchive } = archive;
  return sha256(canonicalJson({ ...contents, archive: legacyArchive }));
}

function isLegacyCompatibleGoalDAG(dag: z.infer<typeof GoalDAGBaseSchema>): boolean {
  const expectedSequences = Object.fromEntries(
    Object.keys(dag.candidates).map((candidateId, index) => [candidateId, index + 1])
  );
  return dag.archive.record_count === 0
    && canonicalJson(dag.archive) === canonicalJson(LegacyEmptyGoalDAGArchiveState)
    && dag.next_candidate_sequence === Object.keys(dag.candidates).length + 1
    && canonicalJson(dag.candidate_sequences) === canonicalJson(expectedSequences);
}

function normalizePersistedGoalDAG(value: unknown): unknown {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return value;
  const dag = value as Record<string, unknown>;
  if (dag.version === 2) {
    const archive = dag.archive;
    if (archive === null || typeof archive !== "object" || Array.isArray(archive)
      || Object.hasOwn(archive, "summary")) return value;
    return {
      ...dag,
      archive: { ...archive, summary: null }
    };
  }
  if (dag.version !== 1) return value;
  const candidates = dag.candidates !== null
    && typeof dag.candidates === "object"
    && !Array.isArray(dag.candidates)
    ? dag.candidates as Record<string, unknown>
    : {};
  const candidateSequences = Object.fromEntries(
    Object.keys(candidates).map((candidateId, index) => [candidateId, index + 1])
  );
  return {
    ...dag,
    version: 2,
    candidate_sequences: dag.candidate_sequences ?? candidateSequences,
    next_candidate_sequence: dag.next_candidate_sequence
      ?? Object.keys(candidates).length + 1,
    archive: dag.archive && typeof dag.archive === "object" && !Array.isArray(dag.archive)
      ? { ...dag.archive, summary: null }
      : LegacyEmptyGoalDAGArchiveState
  };
}

export function goalCandidateSequence(
  goalDAG: GoalDAG,
  candidateId: string
): number | undefined {
  return goalDAG.candidate_sequences?.[candidateId]
    ?? (Object.keys(goalDAG.candidates).indexOf(candidateId) + 1 || undefined);
}

export function goalCandidateBySequence(
  goalDAG: GoalDAG,
  sequence: number
): GoalCandidate | undefined {
  const candidateId = Object.entries(goalDAG.candidate_sequences).find(
    ([, candidateSequence]) => candidateSequence === sequence
  )?.[0];
  return candidateId ? goalDAG.candidates[candidateId] : undefined;
}

export function rehashGoalDAG(
  draft: Omit<GoalDAG, "state_sha256"> & { state_sha256?: string }
): GoalDAG {
  const contents = { ...draft };
  delete contents.state_sha256;
  return GoalDAGSchema.parse({
    ...contents,
    state_sha256: goalDAGStateSha256(contents)
  });
}

export function assertGoalModelSource(
  source: GoalModelSource,
  harness: GoalHarnessValidation,
  expectedToolName:
    | "submit_goal_candidates"
    | "select_goal_candidate"
    | "retire_goal_epoch"
    | "continue_goal_epoch"
): void {
  const authorized = harness.authorized_model_sources.some((entry) => (
    entry.agent_id === source.agent_id
      && entry.agent_manifest_sha256 === source.agent_manifest_sha256
      && entry.agent_manifest_epoch_id === source.agent_manifest_epoch_id
  ));
  if (!authorized
    || !harness.is_model_call_authoritative(source, expectedToolName)) {
    throw new GoalDAGValidationError(
      "unauthorized_model_source",
      `Goal model source is not authorized for this recovery domain: ${source.agent_id}`
    );
  }
}

export function registerGoalEvidence(
  existing: Readonly<Record<string, GoalPhysicalEvidence>>,
  refs: readonly string[],
  expectedWorldRevision: number,
  harness: GoalHarnessValidation
): Record<string, GoalPhysicalEvidence> {
  const next = { ...existing };
  for (const ref of refs) {
    const parsed = GoalPhysicalEvidenceSchema.safeParse(harness.evidence_by_ref(ref));
    if (!parsed.success || parsed.data.ref !== ref) {
      throw new GoalDAGValidationError(
        "evidence_unavailable",
        `Physical evidence is unavailable: ${ref}`
      );
    }
    if (parsed.data.world_revision !== expectedWorldRevision) {
      throw new GoalDAGValidationError(
        "evidence_revision_mismatch",
        `Physical evidence ${ref} does not belong to world revision ${expectedWorldRevision}`
      );
    }
    const persisted = existing[ref];
    if (persisted && canonicalJson(persisted) !== canonicalJson(parsed.data)) {
      throw new GoalDAGValidationError(
        "evidence_mismatch",
        `Physical evidence reference was rebound: ${ref}`
      );
    }
    next[ref] = parsed.data;
  }
  return next;
}

function checkEvidenceRefs(
  dag: z.infer<typeof GoalDAGBaseSchema>,
  refs: readonly string[],
  expectedWorldRevision: number,
  path: PropertyKey[],
  context: z.RefinementCtx,
  expectedGoalContentSha256?: string
): void {
  let exactRevision = false;
  let exactGoalEvaluation = false;
  for (const ref of refs) {
    const evidence = dag.evidence[ref];
    if (!evidence) {
      context.addIssue({
        code: "custom",
        path,
        message: `Goal lifecycle references missing physical evidence: ${ref}`
      });
      continue;
    }
    if (evidence.world_revision === expectedWorldRevision) {
      exactRevision = true;
      if (evidence.kind === "goal_evaluation"
        && evidence.goal_content_sha256 === expectedGoalContentSha256) {
        exactGoalEvaluation = true;
      }
    }
  }
  if (!exactRevision) {
    context.addIssue({
      code: "custom",
      path,
      message: "Goal lifecycle requires physical evidence from its exact world revision"
    });
  }
  if (expectedGoalContentSha256 !== undefined && !exactGoalEvaluation) {
    context.addIssue({
      code: "custom",
      path,
      message: "Goal completion requires an evaluation bound to its exact content hash"
    });
  }
}

function checkDependencyCycles(
  dag: z.infer<typeof GoalDAGBaseSchema>,
  context: z.RefinementCtx
): void {
  const complete = new Set<string>();
  const visiting = new Set<string>();
  const visit = (candidateId: string): boolean => {
    if (complete.has(candidateId)) return false;
    if (visiting.has(candidateId)) return true;
    visiting.add(candidateId);
    const candidate = dag.candidates[candidateId];
    for (const dependencyId of candidate?.dependency_candidate_ids ?? []) {
      if (visit(dependencyId)) return true;
    }
    visiting.delete(candidateId);
    complete.add(candidateId);
    return false;
  };
  for (const candidateId of Object.keys(dag.candidates)) {
    if (!visit(candidateId)) continue;
    context.addIssue({
      code: "custom",
      path: ["candidates", candidateId, "dependency_candidate_ids"],
      message: "Goal candidate dependencies must form an acyclic graph"
    });
    return;
  }
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort(compareCodePoints);
}

function isUniqueSorted(values: readonly string[]): boolean {
  return canonicalJson(values) === canonicalJson(uniqueSorted(values));
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([key, entry]) => [key, canonicalValue(entry)]));
  }
  return value;
}
