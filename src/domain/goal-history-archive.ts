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

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);

const GoalHistoryArchiveRecordBaseSchema = z.object({
  version: z.literal(1),
  sequence: z.number().int().positive(),
  previous_record_sha256: Sha256Schema.nullable(),
  candidate_sequence: z.number().int().positive(),
  candidate: GoalCandidateSchema,
  epoch: GoalEpochSchema,
  evidence: z.record(z.string().trim().min(1), GoalPhysicalEvidenceSchema),
  record_sha256: Sha256Schema
}).strict();

export const GoalHistoryArchiveRecordSchema = GoalHistoryArchiveRecordBaseSchema
  .superRefine((record, context) => {
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
    const referencedEvidence = evidenceRefs(record.candidate, record.epoch);
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
  });

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
  const refs = evidenceRefs(candidate, epoch);
  const evidence = Object.fromEntries(refs.map((ref) => {
    const artifact = goalDAG.evidence[ref];
    if (!artifact) throw new Error(`Goal history archive evidence is unavailable: ${ref}`);
    return [ref, artifact];
  }));
  const contents = {
    version: 1 as const,
    sequence: epoch.epoch_index + 1,
    previous_record_sha256: goalDAG.archive.last_record_sha256,
    candidate_sequence: candidateSequence,
    candidate,
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
  for (const [ref, evidence] of Object.entries(record.evidence)) {
    if (canonicalJson(persisted.evidence[ref]) !== canonicalJson(evidence)) {
      throw new Error(`Goal history archive evidence changed: ${ref}`);
    }
  }

  const epochs = persisted.epochs.slice(1);
  const roots = new Set(epochs.map((entry) => entry.candidate_id));
  for (const workingCandidate of Object.values(persisted.candidates)) {
    if (workingCandidate.status === "proposed" || workingCandidate.status === "active") {
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
      && !retainedEpochCandidates.has(entry.candidate_id))
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
      retained_candidate_ids: retainedArchivedCandidateIds
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

function evidenceRefs(candidate: GoalCandidate, epoch: GoalEpoch): string[] {
  return [...new Set([
    ...candidate.physical_evidence_refs.proposal,
    ...candidate.physical_evidence_refs.resolution,
    ...epoch.physical_evidence_refs.selection,
    ...epoch.physical_evidence_refs.resolution
  ])].sort(compareCodePoints);
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
