import type { GoalDAG } from "../../domain/goal-epoch.js";

export const CONTEXT_GOAL_EPOCH_LIMIT = 12;
export const CONTEXT_PROPOSED_GOAL_LIMIT = 12;

/**
 * Keeps the durable Goal DAG authoritative while presenting a bounded working
 * set to models. Older outcomes remain available through bounded Goal-history
 * recall and append-only storage; they are not replayed into every request.
 */
export function goalDAGContextView(goalDAG: GoalDAG) {
  const candidateSequences = new Map(
    Object.keys(goalDAG.candidates).map((candidateId, index) => [candidateId, index + 1])
  );
  const epochs = goalDAG.epochs.slice(-CONTEXT_GOAL_EPOCH_LIMIT);
  const candidateIds = new Set(epochs.map((epoch) => epoch.candidate_id));
  const proposed = Object.values(goalDAG.candidates)
    .filter((candidate) => candidate.status === "proposed")
    .slice(-CONTEXT_PROPOSED_GOAL_LIMIT);
  for (const candidate of proposed) candidateIds.add(candidate.candidate_id);
  const activeEpoch = goalDAG.current_epoch_id === null
    ? undefined
    : goalDAG.epochs.find((epoch) => epoch.epoch_id === goalDAG.current_epoch_id);
  if (activeEpoch) candidateIds.add(activeEpoch.candidate_id);

  const candidates = Object.fromEntries([...candidateIds].flatMap((candidateId) => {
    const candidate = goalDAG.candidates[candidateId];
    const candidateSequence = candidateSequences.get(candidateId);
    return candidate && candidateSequence
      ? [[candidateId, { ...candidate, candidate_sequence: candidateSequence }] as const]
      : [];
  }));
  const evidenceRefs = new Set<string>();
  for (const candidate of Object.values(candidates)) {
    candidate.physical_evidence_refs.proposal.forEach((ref) => evidenceRefs.add(ref));
    candidate.physical_evidence_refs.resolution.forEach((ref) => evidenceRefs.add(ref));
  }
  for (const epoch of epochs) {
    epoch.physical_evidence_refs.selection.forEach((ref) => evidenceRefs.add(ref));
    epoch.physical_evidence_refs.resolution.forEach((ref) => evidenceRefs.add(ref));
  }
  const evidence = Object.fromEntries([...evidenceRefs].flatMap((ref) => {
    const artifact = goalDAG.evidence[ref];
    return artifact ? [[ref, artifact] as const] : [];
  }));

  const totalCandidateCount = Object.keys(goalDAG.candidates).length;
  const totalEvidenceCount = Object.keys(goalDAG.evidence).length;
  return {
    version: goalDAG.version,
    status: goalDAG.status,
    candidates,
    epochs,
    current_epoch_id: goalDAG.current_epoch_id,
    next_epoch_index: goalDAG.next_epoch_index,
    evidence,
    state_sha256: goalDAG.state_sha256,
    context_projection: {
      total_candidate_count: totalCandidateCount,
      visible_candidate_count: Object.keys(candidates).length,
      total_epoch_count: goalDAG.epochs.length,
      visible_epoch_count: epochs.length,
      total_evidence_count: totalEvidenceCount,
      visible_evidence_count: Object.keys(evidence).length,
      history_truncated: goalDAG.epochs.length > epochs.length
        || totalCandidateCount > Object.keys(candidates).length
        || totalEvidenceCount > Object.keys(evidence).length
    }
  };
}
