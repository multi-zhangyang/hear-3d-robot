import { createHash } from "node:crypto";
import { z } from "zod";
import {
  GoalCandidateSchema,
  GoalEpochSchema,
  GoalPhysicalEvidenceSchema,
  goalCandidateSequence,
  rehashGoalDAG,
  type GoalCandidate,
  type GoalDAG,
  type GoalEpoch
} from "./goal-epoch.js";
import {
  appendGoalHistorySummary,
  createEmptyGoalHistorySummary
} from "./goal-history-summary.js";

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);

const GoalHistoryArchiveRecordV1BaseSchema = z.object({
  version: z.literal(1),
  sequence: z.number().int().positive(),
  previous_record_sha256: Sha256Schema.nullable(),
  candidate_sequence: z.number().int().positive(),
  candidate: GoalCandidateSchema,
  epoch: GoalEpochSchema,
  evidence: z.record(z.string().trim().min(1), GoalPhysicalEvidenceSchema),
  record_sha256: Sha256Schema
}).strict();

const GoalHistoryAlternateCandidateSchema = z.object({
  candidate_sequence: z.number().int().positive(),
  candidate: GoalCandidateSchema
}).strict();

const GoalHistoryArchiveRecordV2BaseSchema = z.object({
  version: z.literal(2),
  kind: z.literal("epoch"),
  sequence: z.number().int().positive(),
  previous_record_sha256: Sha256Schema.nullable(),
  candidate_sequence: z.number().int().positive(),
  candidate: GoalCandidateSchema,
  alternate_candidates: z.array(GoalHistoryAlternateCandidateSchema),
  epoch: GoalEpochSchema,
  evidence: z.record(z.string().trim().min(1), GoalPhysicalEvidenceSchema),
  record_sha256: Sha256Schema
}).strict();

const GoalHistoryArchiveRecordV1Schema = GoalHistoryArchiveRecordV1BaseSchema
  .superRefine((record, context) => {
    validateArchiveRecord(record, [], context);
  });

const GoalHistoryArchiveRecordV2Schema = GoalHistoryArchiveRecordV2BaseSchema
  .superRefine((record, context) => {
    validateArchiveRecord(record, record.alternate_candidates, context);
  });

export const GoalHistoryArchiveRecordSchema = z.union([
  GoalHistoryArchiveRecordV1Schema,
  GoalHistoryArchiveRecordV2Schema
]);

function validateArchiveRecord(
  record: z.infer<typeof GoalHistoryArchiveRecordV1BaseSchema>
    | z.infer<typeof GoalHistoryArchiveRecordV2BaseSchema>,
  alternates: Array<z.infer<typeof GoalHistoryAlternateCandidateSchema>>,
  context: z.RefinementCtx
): void {
    if (record.sequence !== record.epoch.epoch_index + 1
      || record.epoch.candidate_id !== record.candidate.candidate_id
      || record.epoch.status !== record.candidate.status
      || record.epoch.status === "active"
      || record.candidate.status === "active") {
      context.addIssue({
        code: "custom",
        path: ["epoch"],
        message: "Goal history record does not contain one terminal epoch"
      });
    }
    if ((record.sequence === 1) !== (record.previous_record_sha256 === null)) {
      context.addIssue({
        code: "custom",
        path: ["previous_record_sha256"],
        message: "Goal history record chain head is inconsistent"
      });
    }
    const alternateSequences = alternates.map((entry) => entry.candidate_sequence);
    const alternateIds = alternates.map((entry) => entry.candidate.candidate_id);
    if (new Set(alternateSequences).size !== alternateSequences.length
      || new Set(alternateIds).size !== alternateIds.length
      || alternateSequences.some((sequence) => sequence === record.candidate_sequence)
      || alternateIds.some((candidateId) => candidateId === record.candidate.candidate_id)
      || canonicalJson([...alternates].sort(compareAlternateCandidates))
        !== canonicalJson(alternates)) {
      context.addIssue({
        code: "custom",
        path: ["alternate_candidates"],
        message: "Goal history alternate candidates must be unique and sequence ordered"
      });
    }
    for (const { candidate } of alternates) {
      if (candidate.status !== "expired"
        || !sameCandidateSlate(candidate, record.candidate)
        || candidate.resolved_world_revision !== record.epoch.created_world_revision
        || canonicalJson(candidate.physical_evidence_refs.resolution)
          !== canonicalJson(record.epoch.physical_evidence_refs.selection)) {
        context.addIssue({
          code: "custom",
          path: ["alternate_candidates", candidate.candidate_id],
          message: "Goal history alternate candidate is not bound to the selected slate"
        });
      }
    }
    const referencedEvidence = evidenceRefs(
      record.candidate,
      record.epoch,
      alternates.map((entry) => entry.candidate)
    );
    if (canonicalJson(Object.keys(record.evidence).sort(compareCodePoints))
      !== canonicalJson(referencedEvidence)) {
      context.addIssue({
        code: "custom",
        path: ["evidence"],
        message: "Goal history record must contain its exact physical evidence"
      });
    }
    for (const [ref, evidence] of Object.entries(record.evidence)) {
      if (evidence.ref !== ref) {
        context.addIssue({
          code: "custom",
          path: ["evidence", ref],
          message: "Goal history evidence key does not match its identity"
        });
      }
    }
    if (record.record_sha256 !== archiveRecordSha256(record)) {
      context.addIssue({
        code: "custom",
        path: ["record_sha256"],
        message: "Goal history archive record hash does not match its contents"
      });
    }
}

export type GoalHistoryArchiveRecord = z.infer<
  typeof GoalHistoryArchiveRecordSchema
>;

export function createGoalHistoryArchiveRecord(
  goalDAG: GoalDAG
): GoalHistoryArchiveRecord {
  const epoch = goalDAG.epochs[0];
  if (!epoch || epoch.status === "active") {
    throw new Error("Goal history archive requires the oldest terminal epoch");
  }
  if (epoch.epoch_index !== goalDAG.archive.record_count) {
    throw new Error("Goal history archive can only advance its contiguous epoch prefix");
  }
  const candidate = goalDAG.candidates[epoch.candidate_id];
  const candidateSequence = candidate
    ? goalCandidateSequence(goalDAG, candidate.candidate_id)
    : undefined;
  if (!candidate || candidateSequence === undefined) {
    throw new Error(`Goal history archive candidate is unavailable: ${epoch.candidate_id}`);
  }
  const alternateCandidates = Object.values(goalDAG.candidates)
    .filter((entry) => isExpiredSlateAlternate(goalDAG, entry, epoch))
    .map((entry) => {
      const candidateSequence = goalCandidateSequence(goalDAG, entry.candidate_id);
      if (candidateSequence === undefined) {
        throw new Error(`Goal history alternate has no sequence: ${entry.candidate_id}`);
      }
      return { candidate_sequence: candidateSequence, candidate: entry };
    })
    .sort(compareAlternateCandidates);
  const refs = evidenceRefs(
    candidate,
    epoch,
    alternateCandidates.map((entry) => entry.candidate)
  );
  const evidence = Object.fromEntries(refs.map((ref) => {
    const artifact = goalDAG.evidence[ref];
    if (!artifact) throw new Error(`Goal history archive evidence is unavailable: ${ref}`);
    return [ref, artifact];
  }));
  const contents = {
    version: 2 as const,
    kind: "epoch" as const,
    sequence: epoch.epoch_index + 1,
    previous_record_sha256: goalDAG.archive.last_record_sha256,
    candidate_sequence: candidateSequence,
    candidate,
    alternate_candidates: alternateCandidates,
    epoch,
    evidence
  };
  return GoalHistoryArchiveRecordSchema.parse({
    ...contents,
    record_sha256: archiveRecordSha256(contents)
  });
}

export function applyGoalHistoryArchiveRecord(
  persisted: GoalDAG,
  rawRecord: unknown
): GoalDAG {
  const record = GoalHistoryArchiveRecordSchema.parse(rawRecord);
  const epoch = persisted.epochs[0];
  const candidate = persisted.candidates[record.candidate.candidate_id];
  if (record.sequence !== persisted.archive.record_count + 1
    || record.previous_record_sha256 !== persisted.archive.last_record_sha256
    || !epoch
    || canonicalJson(epoch) !== canonicalJson(record.epoch)
    || !candidate
    || canonicalJson(candidate) !== canonicalJson(record.candidate)
    || goalCandidateSequence(persisted, candidate.candidate_id)
      !== record.candidate_sequence) {
    throw new Error(`Goal history archive record cannot advance checkpoint: ${record.sequence}`);
  }
  const expectedAlternates = Object.values(persisted.candidates)
    .filter((entry) => isExpiredSlateAlternate(persisted, entry, epoch))
    .map((entry) => ({
      candidate_sequence: goalCandidateSequence(persisted, entry.candidate_id),
      candidate: entry
    }))
    .sort(compareAlternateCandidates);
  const recordAlternates = record.version === 2 ? record.alternate_candidates : [];
  if (record.version === 2
    && (expectedAlternates.some((entry) => entry.candidate_sequence === undefined)
      || canonicalJson(expectedAlternates) !== canonicalJson(recordAlternates))) {
    throw new Error(`Goal history archive alternates changed: ${record.sequence}`);
  }
  for (const alternate of recordAlternates) {
    const persistedCandidate = persisted.candidates[alternate.candidate.candidate_id];
    if (!persistedCandidate
      || canonicalJson(persistedCandidate) !== canonicalJson(alternate.candidate)
      || goalCandidateSequence(persisted, persistedCandidate.candidate_id)
        !== alternate.candidate_sequence) {
      throw new Error(
        `Goal history archive alternate is unavailable: ${alternate.candidate.candidate_id}`
      );
    }
  }
  for (const [ref, evidence] of Object.entries(record.evidence)) {
    if (canonicalJson(persisted.evidence[ref]) !== canonicalJson(evidence)) {
      throw new Error(`Goal history archive evidence changed: ${ref}`);
    }
  }

  const epochs = persisted.epochs.slice(1);
  const archivedAlternateIds = new Set(recordAlternates.map(
    (entry) => entry.candidate.candidate_id
  ));
  const roots = new Set(epochs.map((entry) => entry.candidate_id));
  for (const workingCandidate of Object.values(persisted.candidates)) {
    if (workingCandidate.status === "proposed" || workingCandidate.status === "active") {
      roots.add(workingCandidate.candidate_id);
    } else if (!archivedAlternateIds.has(workingCandidate.candidate_id)
      && (epochs.some((workingEpoch) => (
        isExpiredSlateAlternate(persisted, workingCandidate, workingEpoch)
      )) || (record.version === 1
        && isExpiredSlateAlternate(persisted, workingCandidate, epoch)))) {
      roots.add(workingCandidate.candidate_id);
    }
  }
  const retainedIds = dependencyClosure(persisted.candidates, roots);
  const candidates = Object.fromEntries(Object.entries(persisted.candidates)
    .filter(([candidateId]) => retainedIds.has(candidateId)));
  const candidateSequences = Object.fromEntries(Object.entries(
    persisted.candidate_sequences
  ).filter(([candidateId]) => retainedIds.has(candidateId)));
  const retainedEpochCandidates = new Set(epochs.map((entry) => entry.candidate_id));
  const retainedArchivedCandidateIds = Object.values(candidates)
    .filter((entry) => entry.status !== "proposed"
      && entry.status !== "active"
      && !retainedEpochCandidates.has(entry.candidate_id)
      && !epochs.some((workingEpoch) => (
        isExpiredSlateAlternate(persisted, entry, workingEpoch)
      )))
    .map((entry) => entry.candidate_id)
    .sort(compareCodePoints);
  const retainedEvidence = new Set<string>();
  for (const workingCandidate of Object.values(candidates)) {
    workingCandidate.physical_evidence_refs.proposal.forEach(
      (ref) => retainedEvidence.add(ref)
    );
    workingCandidate.physical_evidence_refs.resolution.forEach(
      (ref) => retainedEvidence.add(ref)
    );
  }
  for (const workingEpoch of epochs) {
    workingEpoch.physical_evidence_refs.selection.forEach(
      (ref) => retainedEvidence.add(ref)
    );
    workingEpoch.physical_evidence_refs.resolution.forEach(
      (ref) => retainedEvidence.add(ref)
    );
  }
  const evidence = Object.fromEntries(Object.entries(persisted.evidence)
    .filter(([ref]) => retainedEvidence.has(ref)));

  const currentSummary = persisted.archive.summary
    ?? (persisted.archive.record_count === 0
      ? createEmptyGoalHistorySummary()
      : undefined);
  if (!currentSummary) {
    throw new Error("Goal history summary must be rebuilt before advancing the archive");
  }
  const summary = appendGoalHistorySummary(currentSummary, {
    sequence: record.sequence,
    recordSha256: record.record_sha256,
    candidate: record.candidate,
    epoch: record.epoch,
    alternateCandidates: recordAlternates.map((entry) => entry.candidate),
    alternateHistoryComplete: record.version === 2
  });
  return rehashGoalDAG({
    ...persisted,
    candidates,
    candidate_sequences: candidateSequences,
    epochs,
    evidence,
    archive: {
      record_count: record.sequence,
      last_record_sha256: record.record_sha256,
      last_epoch_id: record.epoch.epoch_id,
      retained_candidate_ids: retainedArchivedCandidateIds,
      summary
    }
  });
}

function dependencyClosure(
  candidates: Readonly<Record<string, GoalCandidate>>,
  roots: ReadonlySet<string>
): Set<string> {
  const retained = new Set<string>();
  const pending = [...roots];
  while (pending.length > 0) {
    const candidateId = pending.pop()!;
    if (retained.has(candidateId)) continue;
    const candidate = candidates[candidateId];
    if (!candidate) throw new Error(`Working Goal dependency is unavailable: ${candidateId}`);
    retained.add(candidateId);
    pending.push(...candidate.dependency_candidate_ids);
  }
  return retained;
}

function evidenceRefs(
  candidate: GoalCandidate,
  epoch: GoalEpoch,
  alternates: readonly GoalCandidate[] = []
): string[] {
  return [...new Set([
    ...candidate.physical_evidence_refs.proposal,
    ...candidate.physical_evidence_refs.resolution,
    ...alternates.flatMap((entry) => entry.physical_evidence_refs.proposal),
    ...alternates.flatMap((entry) => entry.physical_evidence_refs.resolution),
    ...epoch.physical_evidence_refs.selection,
    ...epoch.physical_evidence_refs.resolution
  ])].sort(compareCodePoints);
}

function isExpiredSlateAlternate(
  goalDAG: GoalDAG,
  candidate: GoalCandidate,
  epoch: GoalEpoch
): boolean {
  const selected = goalDAG.candidates[epoch.candidate_id];
  return candidate.status === "expired"
    && candidate.candidate_id !== epoch.candidate_id
    && selected !== undefined
    && sameCandidateSlate(candidate, selected)
    && candidate.resolved_world_revision === epoch.created_world_revision
    && canonicalJson(candidate.physical_evidence_refs.resolution)
      === canonicalJson(epoch.physical_evidence_refs.selection);
}

function sameCandidateSlate(left: GoalCandidate, right: GoalCandidate): boolean {
  return canonicalJson(left.source) === canonicalJson(right.source);
}

function compareAlternateCandidates(
  left: { candidate_sequence: number | undefined },
  right: { candidate_sequence: number | undefined }
): number {
  return (left.candidate_sequence ?? Number.MAX_SAFE_INTEGER)
    - (right.candidate_sequence ?? Number.MAX_SAFE_INTEGER);
}

function archiveRecordSha256(record: {
  record_sha256?: string;
  [key: string]: unknown;
}): string {
  const { record_sha256: _recordSha256, ...contents } = record;
  return createHash("sha256").update(canonicalJson(contents)).digest("hex");
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
      .sort(([left], [right]) => compareCodePoints(left, right))
      .map(([key, entry]) => [key, canonicalValue(entry)]));
  }
  return value;
}
