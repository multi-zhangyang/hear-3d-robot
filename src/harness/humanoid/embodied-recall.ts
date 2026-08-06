import type {
  HumanoidEmbodiedEpisode,
  HumanoidEmbodiedExperience,
  HumanoidEmbodiedMemoryState
} from "../../domain/humanoid-run.js";
import {
  HumanoidEmbodiedEpisodeSchema,
  HumanoidEmbodiedExperienceSchema,
  HumanoidEmbodiedMemoryStateSchema
} from "../../domain/humanoid-run.js";
import type { JsonValue } from "../../domain/schema.js";
import type { RunStore } from "../../persistence/run-store.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import { embodiedActionJournalReceipt, json } from "./run-runtime-persistence.js";
import { modelReceiptDetail } from "./receipt-context.js";

const RECALL_PAGE_SIZE = 64;

export const HUMANOID_EXPERIENCE_OUTCOMES = [
  "succeeded",
  "rejected",
  "physically_failed"
] as const;

export const HUMANOID_GOAL_PREDICATE_TYPES = [
  "robot_at",
  "robot_in_zone",
  "object_in_zone",
  "object_placed",
  "object_at",
  "object_grasped",
  "block_removed",
  "end_effector_at"
] as const;

export interface HumanoidEmbodiedRecallRequest {
  source_refs?: string[];
  before_sequence?: number;
  before_experience_sequence?: number;
  outcomes?: Array<typeof HUMANOID_EXPERIENCE_OUTCOMES[number]>;
  predicate_types?: Array<typeof HUMANOID_GOAL_PREDICATE_TYPES[number]>;
  object_ids?: string[];
  solid_ids?: string[];
  zone_ids?: string[];
  limit: number;
}

type HistoricalHumanoidAction = Pick<HumanoidActionReceipt,
  | "transactionId"
  | "agentId"
  | "decision"
  | "cycle"
  | "action"
  | "accepted"
  | "code"
  | "worldBeforeRevision"
  | "worldAfterRevision"
  | "frameCount"
  | "channels"
  | "committedAt"
> & {
  detail: JsonValue;
  source_ref: string;
  historical_only: true;
  full_receipt_persisted: true;
};

export async function recallHumanoidEmbodiedHistory(input: {
  store: RunStore;
  memory: HumanoidEmbodiedMemoryState;
  currentWorldRevision: number;
  request: HumanoidEmbodiedRecallRequest;
}): Promise<JsonValue> {
  const memory = HumanoidEmbodiedMemoryStateSchema.parse(input.memory);
  return hasSemanticFilters(input.request)
    ? recallSemanticExperiences(input.store, memory, input.currentWorldRevision, input.request)
    : recallChronologicalHistory(input.store, memory, input.currentWorldRevision, input.request);
}

async function recallSemanticExperiences(
  store: RunStore,
  memory: HumanoidEmbodiedMemoryState,
  currentWorldRevision: number,
  request: HumanoidEmbodiedRecallRequest
): Promise<JsonValue> {
  const experiences = new Map<string, HumanoidEmbodiedExperience>();
  const consider = (raw: HumanoidEmbodiedExperience): void => {
    const experience = HumanoidEmbodiedExperienceSchema.parse(raw);
    if (request.before_experience_sequence !== undefined
      && experience.sequence >= request.before_experience_sequence) return;
    if (!matchesSemanticFilters(experience, request)) return;
    experiences.set(experience.source_ref, experience);
  };
  for (const experience of [...memory.recent_experiences].reverse()) {
    consider(experience);
    if (experiences.size >= request.limit) break;
  }
  if (experiences.size < request.limit) {
    const tail = await store.readJournalTail("experiences", 1);
    for (let end = tail.total; end > 0 && experiences.size < request.limit;) {
      const from = Math.max(0, end - RECALL_PAGE_SIZE);
      const page = await store.readJournalPage("experiences", from, end - from);
      for (let index = page.entries.length - 1; index >= 0; index -= 1) {
        consider(HumanoidEmbodiedExperienceSchema.parse(page.entries[index]));
        if (experiences.size >= request.limit) break;
      }
      end = from;
    }
  }
  const selectedExperiences = [...experiences.values()]
    .sort((left, right) => right.sequence - left.sequence)
    .slice(0, request.limit);
  const actions = await actionsBySourceRefs(
    store,
    new Set(selectedExperiences.map((experience) => experience.source_ref))
  );
  const orderedSourceRefs = selectedExperiences.map((experience) => experience.source_ref);
  return json({
    historical_only: true,
    current_world_revision: currentWorldRevision,
    semantic_query: semanticQuery(request),
    ordered_source_refs: orderedSourceRefs,
    episodes: [],
    experiences: selectedExperiences.map(historicalExperience),
    actions: orderedSourceRefs.flatMap((sourceRef) => {
      const action = actions.get(sourceRef);
      return action ? [action] : [];
    }),
    missing_source_refs: orderedSourceRefs.filter((sourceRef) => !actions.has(sourceRef)),
    next_before_sequence: null,
    next_before_experience_sequence: selectedExperiences.length > 0
      && Math.min(...selectedExperiences.map((experience) => experience.sequence)) > 1
      ? Math.min(...selectedExperiences.map((experience) => experience.sequence))
      : null
  });
}

async function recallChronologicalHistory(
  store: RunStore,
  memory: HumanoidEmbodiedMemoryState,
  currentWorldRevision: number,
  request: HumanoidEmbodiedRecallRequest
): Promise<JsonValue> {
  const requestedRefs = new Set(request.source_refs ?? []);
  const requestedEpisodeRefs = new Set(
    [...requestedRefs].filter((sourceRef) => sourceRef.startsWith("episode:"))
  );
  const requestedActionRefs = new Set(
    [...requestedRefs].filter((sourceRef) => sourceRef.startsWith("action:"))
  );
  const episodes = new Map<string, HumanoidEmbodiedEpisode>();
  let actionBeforeTime: string | undefined;
  const considerEpisode = (raw: HumanoidEmbodiedEpisode): void => {
    const episode = HumanoidEmbodiedEpisodeSchema.parse(raw);
    const sourceRef = episode.source_ref ?? `episode:${episode.sequence}`;
    if (requestedRefs.size === 0 && request.before_sequence === episode.sequence) {
      actionBeforeTime = episode.recorded_at;
    }
    if (request.before_sequence !== undefined
      && episode.sequence >= request.before_sequence) return;
    if (requestedRefs.size > 0 && !requestedEpisodeRefs.has(sourceRef)) return;
    episodes.set(sourceRef, { ...episode, source_ref: sourceRef });
  };
  for (const episode of [...memory.recent_episodes].reverse()) considerEpisode(episode);

  const enoughEpisodes = (): boolean => requestedRefs.size > 0
    ? [...requestedEpisodeRefs].every((sourceRef) => episodes.has(sourceRef))
    : episodes.size >= request.limit;
  if (!enoughEpisodes()) {
    const tail = await store.readJournalTail("episodes", 1);
    for (let end = tail.total; end > 0 && !enoughEpisodes();) {
      const from = Math.max(0, end - RECALL_PAGE_SIZE);
      const page = await store.readJournalPage("episodes", from, end - from);
      for (let index = page.entries.length - 1; index >= 0; index -= 1) {
        considerEpisode(HumanoidEmbodiedEpisodeSchema.parse(page.entries[index]));
        if (enoughEpisodes()) break;
      }
      end = from;
    }
  }

  const actions = new Map<string, HistoricalHumanoidAction>();
  const enoughActions = (): boolean => requestedRefs.size > 0
    ? [...requestedActionRefs].every((sourceRef) => actions.has(sourceRef))
    : actions.size >= request.limit;
  if (!enoughActions()) {
    const tail = await store.readJournalTail("actions", 1);
    for (let end = tail.total; end > 0 && !enoughActions();) {
      const from = Math.max(0, end - RECALL_PAGE_SIZE);
      const page = await store.readJournalPage("actions", from, end - from);
      for (let index = page.entries.length - 1; index >= 0; index -= 1) {
        const receipt = embodiedActionJournalReceipt(page.entries[index]!);
        if (!receipt) continue;
        if (requestedRefs.size === 0
          && actionBeforeTime !== undefined
          && receipt.committedAt >= actionBeforeTime) continue;
        const sourceRef = `action:${receipt.transactionId}`;
        if (requestedRefs.size > 0 && !requestedActionRefs.has(sourceRef)) continue;
        actions.set(sourceRef, historicalAction(receipt));
        if (enoughActions()) break;
      }
      end = from;
    }
  }

  const selectedRecords = [
    ...[...episodes.values()].map((episode) => ({
      kind: "episode" as const,
      sourceRef: episode.source_ref!,
      recordedAt: episode.recorded_at,
      value: episode
    })),
    ...[...actions.values()].map((action) => ({
      kind: "action" as const,
      sourceRef: action.source_ref,
      recordedAt: action.committedAt,
      value: action
    }))
  ].sort((left, right) => (
    compare(right.recordedAt, left.recordedAt)
      || compare(right.kind, left.kind)
      || compare(right.sourceRef, left.sourceRef)
  )).slice(0, request.limit);
  const selectedEpisodes = selectedRecords.flatMap((record) => (
    record.kind === "episode" ? [record.value as HumanoidEmbodiedEpisode] : []
  ));
  const selectedActions = selectedRecords.flatMap((record) => (
    record.kind === "action" ? [record.value as HistoricalHumanoidAction] : []
  ));
  const selectedActionRefs = new Set(selectedActions.map((action) => action.source_ref));
  const experiences = await experiencesBySourceRefs(store, memory, selectedActionRefs);
  const returnedRefs = new Set(selectedRecords.map((record) => record.sourceRef));
  return json({
    historical_only: true,
    current_world_revision: currentWorldRevision,
    semantic_query: null,
    ordered_source_refs: selectedRecords.map((record) => record.sourceRef),
    episodes: selectedEpisodes,
    experiences: selectedActions.flatMap((action) => {
      const experience = experiences.get(action.source_ref);
      return experience ? [historicalExperience(experience)] : [];
    }),
    actions: selectedActions,
    missing_source_refs: [...requestedRefs].filter((sourceRef) => !returnedRefs.has(sourceRef)),
    next_before_sequence: requestedRefs.size === 0
      && selectedEpisodes.length > 0
      && Math.min(...selectedEpisodes.map((episode) => episode.sequence)) > 1
      ? Math.min(...selectedEpisodes.map((episode) => episode.sequence))
      : null,
    next_before_experience_sequence: null
  });
}

async function actionsBySourceRefs(
  store: RunStore,
  refs: Set<string>
): Promise<Map<string, HistoricalHumanoidAction>> {
  const actions = new Map<string, HistoricalHumanoidAction>();
  if (refs.size === 0) return actions;
  const tail = await store.readJournalTail("actions", 1);
  for (let end = tail.total; end > 0 && actions.size < refs.size;) {
    const from = Math.max(0, end - RECALL_PAGE_SIZE);
    const page = await store.readJournalPage("actions", from, end - from);
    for (let index = page.entries.length - 1; index >= 0; index -= 1) {
      const receipt = embodiedActionJournalReceipt(page.entries[index]!);
      if (!receipt) continue;
      const sourceRef = `action:${receipt.transactionId}`;
      if (refs.has(sourceRef)) actions.set(sourceRef, historicalAction(receipt));
    }
    end = from;
  }
  return actions;
}

async function experiencesBySourceRefs(
  store: RunStore,
  memory: HumanoidEmbodiedMemoryState,
  refs: Set<string>
): Promise<Map<string, HumanoidEmbodiedExperience>> {
  const experiences = new Map<string, HumanoidEmbodiedExperience>();
  if (refs.size === 0) return experiences;
  for (const experience of memory.recent_experiences) {
    if (refs.has(experience.source_ref)) {
      experiences.set(experience.source_ref, experience);
    }
  }
  if (experiences.size >= refs.size) return experiences;
  const tail = await store.readJournalTail("experiences", 1);
  for (let end = tail.total; end > 0 && experiences.size < refs.size;) {
    const from = Math.max(0, end - RECALL_PAGE_SIZE);
    const page = await store.readJournalPage("experiences", from, end - from);
    for (let index = page.entries.length - 1; index >= 0; index -= 1) {
      const experience = HumanoidEmbodiedExperienceSchema.parse(page.entries[index]);
      if (refs.has(experience.source_ref)) {
        experiences.set(experience.source_ref, experience);
      }
    }
    end = from;
  }
  return experiences;
}

function matchesSemanticFilters(
  experience: HumanoidEmbodiedExperience,
  request: HumanoidEmbodiedRecallRequest
): boolean {
  return matches(request.outcomes, [experience.outcome])
    && matches(request.predicate_types, experience.predicate_types)
    && matches(request.object_ids, experience.object_ids)
    && matches(request.solid_ids, experience.solid_ids)
    && matches(request.zone_ids, experience.zone_ids);
}

function matches(
  requested: readonly string[] | undefined,
  actual: readonly string[]
): boolean {
  return !requested || requested.length === 0
    || requested.some((value) => actual.includes(value));
}

function hasSemanticFilters(request: HumanoidEmbodiedRecallRequest): boolean {
  return request.before_experience_sequence !== undefined
    || (request.outcomes?.length ?? 0) > 0
    || (request.predicate_types?.length ?? 0) > 0
    || (request.object_ids?.length ?? 0) > 0
    || (request.solid_ids?.length ?? 0) > 0
    || (request.zone_ids?.length ?? 0) > 0;
}

function semanticQuery(request: HumanoidEmbodiedRecallRequest) {
  return {
    outcomes: request.outcomes ?? [],
    predicate_types: request.predicate_types ?? [],
    object_ids: request.object_ids ?? [],
    solid_ids: request.solid_ids ?? [],
    zone_ids: request.zone_ids ?? []
  };
}

function historicalAction(receipt: HumanoidActionReceipt): HistoricalHumanoidAction {
  return {
    transactionId: receipt.transactionId,
    agentId: receipt.agentId,
    ...(receipt.decision ? { decision: structuredClone(receipt.decision) } : {}),
    ...(receipt.cycle ? { cycle: structuredClone(receipt.cycle) } : {}),
    action: receipt.action,
    accepted: receipt.accepted,
    code: receipt.code,
    worldBeforeRevision: receipt.worldBeforeRevision,
    worldAfterRevision: receipt.worldAfterRevision,
    frameCount: receipt.frameCount,
    channels: [...receipt.channels],
    detail: json(modelReceiptDetail(receipt.detail)),
    committedAt: receipt.committedAt,
    source_ref: `action:${receipt.transactionId}`,
    historical_only: true,
    full_receipt_persisted: true
  };
}

function historicalExperience(experience: HumanoidEmbodiedExperience) {
  return { ...experience, historical_only: true as const };
}

function compare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
