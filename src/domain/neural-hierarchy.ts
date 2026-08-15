import { randomUUID } from "node:crypto";
import { z } from "zod";
import { modelPayloadSha256 } from "./model-call-authority.js";
import { JsonValueSchema, type JsonValue } from "./schema.js";

/**
 * Durable contract between cognitive Agents and the continuous robot control
 * stack.  This is deliberately a signal protocol, not a shared conversation:
 * every consumer receives a bounded, world-versioned message and keeps its own
 * SDK Session.
 */
export const NEURAL_HIERARCHY_CONTRACT_VERSION = 3 as const;

export const NeuralLayerSchema = z.enum([
  "executive",
  "action_selection",
  "perceptual_association",
  "sensorimotor",
  "premotor",
  "motor_planning",
  "predictive_rollout",
  "controller",
  "reflex",
  "body"
]);

export type NeuralLayer = z.infer<typeof NeuralLayerSchema>;

export const NeuralPathwaySchema = z.enum([
  "executive_control",
  "goal_valuation",
  "perceptual_association",
  "sensorimotor_selection",
  "cerebellar_prediction",
  "interoceptive_risk",
  "premotor_composition",
  "motor_intent",
  "physical_execution",
  "ascending_feedback"
]);

export type NeuralPathway = z.infer<typeof NeuralPathwaySchema>;

export const NeuralSignalKindSchema = z.enum([
  "goal_context",
  "goal_selected",
  "sensory_evidence",
  "scene_interpretation",
  "memory_retrieval",
  "perceptual_belief",
  "affordance_hypothesis",
  "risk_assessment",
  "forward_prediction",
  "prediction_error",
  "skill_proposal",
  "skill_commitment",
  "motor_intent",
  "rollout_result",
  "execution_receipt",
  "skill_completed",
  "skill_failed",
  "escalation"
]);

export type NeuralSignalKind = z.infer<typeof NeuralSignalKindSchema>;

export const NeuralSignalDirectionSchema = z.enum([
  "descending",
  "ascending",
  "reentrant"
]);

export type NeuralSignalDirection = z.infer<
  typeof NeuralSignalDirectionSchema
>;

export const NeuralSignalStatusSchema = z.enum([
  "pending",
  "consumed",
  "expired",
  "superseded"
]);

export const NeuralSignalSchema = z.object({
  signal_id: z.string().uuid(),
  sequence: z.number().int().positive(),
  kind: NeuralSignalKindSchema,
  pathway: NeuralPathwaySchema,
  direction: NeuralSignalDirectionSchema,
  source_node_id: z.string().trim().min(1),
  source_layer: NeuralLayerSchema,
  target_node_id: z.string().trim().min(1),
  target_layer: NeuralLayerSchema,
  world_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  ttl_revisions: z.number().int().nonnegative(),
  priority: z.number().int().min(0).max(100),
  invocation_id: z.string().uuid(),
  parent_invocation_id: z.string().uuid().nullable(),
  parent_episode_id: z.string().uuid(),
  causal_parent_ids: z.array(z.string().uuid()).max(64),
  authority_lease_id: z.string().uuid().nullable().default(null),
  source_authority_lease_id: z.string().uuid().nullable().default(null),
  payload: JsonValueSchema,
  status: NeuralSignalStatusSchema,
  created_at: z.string().datetime(),
  consumed_at: z.string().datetime().optional()
}).strict().superRefine((signal, context) => {
  if (new Set(signal.causal_parent_ids).size !== signal.causal_parent_ids.length) {
    context.addIssue({
      code: "custom",
      path: ["causal_parent_ids"],
      message: "Neural signal causal parents must be unique"
    });
  }
  if (signal.causal_parent_ids.includes(signal.signal_id)) {
    context.addIssue({
      code: "custom",
      path: ["causal_parent_ids"],
      message: "A neural signal cannot be its own causal parent"
    });
  }
  if (signal.parent_invocation_id === signal.invocation_id) {
    context.addIssue({
      code: "custom",
      path: ["parent_invocation_id"],
      message: "A neural invocation cannot be its own parent"
    });
  }
  if (signal.status === "consumed" && signal.consumed_at === undefined) {
    context.addIssue({
      code: "custom",
      path: ["consumed_at"],
      message: "A consumed neural signal requires a consumption timestamp"
    });
  }
  const descending = signal.direction === "descending";
  if (descending !== (signal.authority_lease_id !== null)) {
    context.addIssue({
      code: "custom",
      path: ["authority_lease_id"],
      message: "Only a descending neural signal carries current child authority"
    });
  }
  if (descending === (signal.source_authority_lease_id !== null)) {
    context.addIssue({
      code: "custom",
      path: ["source_authority_lease_id"],
      message: "Ascending and reentrant signals must retain source-lease provenance"
    });
  }
});

export type NeuralSignal = z.infer<typeof NeuralSignalSchema>;

export const NeuralSkillCommitmentSchema = z.object({
  commitment_id: z.string().uuid(),
  goal_epoch_id: z.string().trim().min(1),
  owner_node_id: z.string().trim().min(1),
  skill: z.string().trim().min(1),
  state: z.enum([
    "proposed",
    "committed",
    "executing",
    "completed",
    "failed",
    "released"
  ]),
  established_world_revision: z.number().int().nonnegative(),
  last_validated_world_revision: z.number().int().nonnegative(),
  termination_contract: JsonValueSchema,
  source_signal_ids: z.array(z.string().uuid()).min(1).max(64),
  transition_signal_ids: z.array(z.string().uuid()).max(64).default([]),
  transition_interrupt_ids: z.array(z.string().uuid()).max(64).default([]),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
}).strict().superRefine((commitment, context) => {
  if (commitment.last_validated_world_revision
    < commitment.established_world_revision) {
    context.addIssue({
      code: "custom",
      path: ["last_validated_world_revision"],
      message: "A skill commitment cannot be validated before it was established"
    });
  }
});

export type NeuralSkillCommitment = z.infer<
  typeof NeuralSkillCommitmentSchema
>;

/**
 * A plant-originated safety event is not an Agent message and does not borrow
 * an SDK invocation lease. It is a durable afferent fact captured by Body,
 * relayed by Reflex, and acknowledged by the nearest supervisory owner.
 */
export const NeuralSafetyInterruptSchema = z.object({
  interrupt_id: z.string().uuid(),
  kind: z.enum(["stationary_fall"]),
  source_node_id: z.string().trim().min(1),
  relay_node_id: z.string().trim().min(1),
  target_node_id: z.string().trim().min(1),
  world_frame: z.number().int().nonnegative(),
  world_revision: z.number().int().nonnegative(),
  commitment_id: z.string().uuid().nullable(),
  status: z.enum(["pending", "acknowledged", "resolved"]),
  detail: JsonValueSchema,
  created_at: z.string().datetime(),
  acknowledged_at: z.string().datetime().nullable(),
  acknowledged_by_node_id: z.string().trim().min(1).nullable(),
  resolved_at: z.string().datetime().nullable(),
  resolved_world_revision: z.number().int().nonnegative().nullable()
}).strict().superRefine((interrupt, context) => {
  const acknowledged = interrupt.status !== "pending";
  if (acknowledged !== (interrupt.acknowledged_at !== null)
    || acknowledged !== (interrupt.acknowledged_by_node_id !== null)) {
    context.addIssue({
      code: "custom",
      path: ["acknowledged_at"],
      message: "An acknowledged safety interrupt requires its supervisory owner and time"
    });
  }
  const resolved = interrupt.status === "resolved";
  if (resolved !== (interrupt.resolved_at !== null)
    || resolved !== (interrupt.resolved_world_revision !== null)) {
    context.addIssue({
      code: "custom",
      path: ["resolved_at"],
      message: "A resolved safety interrupt requires its resolution revision and time"
    });
  }
  if (interrupt.resolved_world_revision !== null
    && interrupt.resolved_world_revision < interrupt.world_revision) {
    context.addIssue({
      code: "custom",
      path: ["resolved_world_revision"],
      message: "A safety interrupt cannot resolve before its originating world revision"
    });
  }
});

export type NeuralSafetyInterrupt = z.infer<
  typeof NeuralSafetyInterruptSchema
>;

export const NeuralPredictionErrorSchema = z.object({
  error_id: z.string().uuid(),
  source_signal_id: z.string().uuid(),
  observer_node_id: z.string().trim().min(1),
  world_revision: z.number().int().nonnegative(),
  magnitude: z.number().finite().nonnegative(),
  tolerance: z.number().finite().nonnegative(),
  correction_scope: z.enum(["local", "pathway", "supervisory"]),
  corrected: z.boolean(),
  detail: JsonValueSchema,
  observed_at: z.string().datetime()
}).strict();

export type NeuralPredictionError = z.infer<
  typeof NeuralPredictionErrorSchema
>;

export const NeuralPathwayCadenceSchema = z.object({
  pathway: NeuralPathwaySchema,
  minimum_interval_ms: z.number().int().nonnegative(),
  maximum_staleness_ms: z.number().int().positive(),
  last_dispatched_at: z.string().datetime().nullable(),
  last_completed_at: z.string().datetime().nullable(),
  dispatch_count: z.number().int().nonnegative()
}).strict().superRefine((cadence, context) => {
  if (cadence.maximum_staleness_ms < cadence.minimum_interval_ms) {
    context.addIssue({
      code: "custom",
      path: ["maximum_staleness_ms"],
      message: "Pathway staleness horizon must cover its minimum cadence"
    });
  }
});

export type NeuralPathwayCadence = z.infer<
  typeof NeuralPathwayCadenceSchema
>;

export const NeuralHarnessPhaseSchema = z.enum([
  "bootstrapping",
  "goal_valuation",
  "perception",
  "skill_proposal",
  "commitment_authorization",
  "motor_assessment",
  "motor_planning",
  "rollout_review",
  "execution",
  "feedback",
  "safety_interrupt",
  "recovery",
  "cycle_completion",
  "terminal"
]);

export type NeuralHarnessPhase = z.infer<typeof NeuralHarnessPhaseSchema>;

export const NeuralHarnessPhaseStateSchema = z.object({
  phase: NeuralHarnessPhaseSchema,
  sequence: z.number().int().nonnegative(),
  goal_epoch_id: z.string().trim().min(1).nullable(),
  commitment_id: z.string().uuid().nullable(),
  world_revision: z.number().int().nonnegative(),
  entered_by_node_id: z.string().trim().min(1),
  reason: z.string().trim().min(1),
  entered_at: z.string().datetime()
}).strict();

export type NeuralHarnessPhaseState = z.infer<
  typeof NeuralHarnessPhaseStateSchema
>;

export const NeuralAuthorityLeaseStatusSchema = z.enum([
  "active",
  "suspended",
  "closed",
  "revoked",
  "expired"
]);

export const NeuralAuthorityLeaseSchema = z.object({
  lease_id: z.string().uuid(),
  issuing_parent_node_id: z.string().trim().min(1),
  target_child_node_id: z.string().trim().min(1),
  goal_epoch_id: z.string().trim().min(1).nullable(),
  commitment_id: z.string().uuid().nullable(),
  issued_world_revision: z.number().int().nonnegative(),
  expires_world_revision: z.number().int().nonnegative(),
  allowed_signal_kinds: z.array(NeuralSignalKindSchema).min(1).max(64),
  correction_scope: z.enum(["ordinary", "local", "pathway", "supervisory"]),
  invocation_id: z.string().uuid(),
  parent_invocation_id: z.string().uuid().nullable(),
  parent_episode_id: z.string().uuid(),
  exclusive: z.boolean(),
  suspended_lease_ids: z.array(z.string().uuid()).max(64),
  status: NeuralAuthorityLeaseStatusSchema,
  issued_at: z.string().datetime(),
  expires_at: z.string().datetime(),
  closed_at: z.string().datetime().nullable(),
  close_reason: z.string().trim().min(1).nullable()
}).strict().superRefine((lease, context) => {
  if (lease.expires_world_revision < lease.issued_world_revision) {
    context.addIssue({
      code: "custom",
      path: ["expires_world_revision"],
      message: "A neural authority lease cannot expire before it is issued"
    });
  }
  if (new Set(lease.allowed_signal_kinds).size !== lease.allowed_signal_kinds.length
    || new Set(lease.suspended_lease_ids).size !== lease.suspended_lease_ids.length) {
    context.addIssue({
      code: "custom",
      path: ["allowed_signal_kinds"],
      message: "Neural authority lease lists must be unique"
    });
  }
  const terminal = ["closed", "revoked", "expired"].includes(lease.status);
  if (terminal !== (lease.closed_at !== null)
    || terminal !== (lease.close_reason !== null)) {
    context.addIssue({
      code: "custom",
      path: ["closed_at"],
      message: "A terminal neural authority lease requires its close time and reason"
    });
  }
});

export type NeuralAuthorityLease = z.infer<typeof NeuralAuthorityLeaseSchema>;

export const NeuralPlanningActionSchema = z.enum([
  "plan_humanoid_skill",
  "plan_whole_body_motion",
  "plan_whole_body_motion_candidates",
  "plan_humanoid_navigation"
]);

export type NeuralPlanningAction = z.infer<typeof NeuralPlanningActionSchema>;

export const NeuralRolloutCertificateStatusSchema = z.enum([
  "active",
  "consumed",
  "revoked",
  "expired"
]);

export const NeuralRolloutCertificateSchema = z.object({
  certificate_id: z.string().uuid(),
  commitment_id: z.string().uuid(),
  goal_epoch_id: z.string().trim().min(1),
  issued_by_node_id: z.string().trim().min(1),
  planning_transaction_id: z.string().trim().min(1),
  planning_action: NeuralPlanningActionSchema,
  rollout_signal_id: z.string().uuid(),
  predictive_signal_id: z.string().uuid(),
  rollout_payload_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  rollout_invocation_id: z.string().uuid(),
  predictive_invocation_id: z.string().uuid(),
  rollout_world_revision: z.number().int().nonnegative(),
  issued_world_revision: z.number().int().nonnegative(),
  expires_world_revision: z.number().int().nonnegative(),
  status: NeuralRolloutCertificateStatusSchema,
  execution_transaction_id: z.string().trim().min(1).nullable(),
  issued_at: z.string().datetime(),
  closed_at: z.string().datetime().nullable(),
  close_reason: z.string().trim().min(1).nullable()
}).strict().superRefine((certificate, context) => {
  if (certificate.expires_world_revision < certificate.issued_world_revision) {
    context.addIssue({
      code: "custom",
      path: ["expires_world_revision"],
      message: "A rollout certificate cannot expire before it is issued"
    });
  }
  const terminal = certificate.status !== "active";
  if (terminal !== (certificate.closed_at !== null)
    || terminal !== (certificate.close_reason !== null)) {
    context.addIssue({
      code: "custom",
      path: ["closed_at"],
      message: "A terminal rollout certificate requires close metadata"
    });
  }
  if ((certificate.status === "consumed")
    !== (certificate.execution_transaction_id !== null)) {
    context.addIssue({
      code: "custom",
      path: ["execution_transaction_id"],
      message: "Only a consumed rollout certificate names its physical transaction"
    });
  }
});

export type NeuralRolloutCertificate = z.infer<
  typeof NeuralRolloutCertificateSchema
>;

export const NeuralReflexArcStatusSchema = z.enum([
  "idle",
  "active",
  "succeeded",
  "failed",
  "interrupted"
]);

export type NeuralReflexArcStatus = z.infer<
  typeof NeuralReflexArcStatusSchema
>;

const NeuralReflexArcModeCountsSchema = z.object({
  learned_policy: z.number().int().nonnegative(),
  reference_control: z.number().int().nonnegative(),
  hybrid_control: z.number().int().nonnegative()
}).strict();

const NeuralReflexArcRouteCountsSchema = z.object({
  direct: z.number().int().nonnegative(),
  primary: z.number().int().nonnegative(),
  fallback: z.number().int().nonnegative(),
  upper_body_overlay: z.number().int().nonnegative()
}).strict();

export const NeuralReflexArcStateSchema = z.object({
  protocol: z.literal("neural-reflex-arc-state-v1"),
  status: NeuralReflexArcStatusSchema,
  execution_transaction_id: z.string().trim().min(1).nullable(),
  commitment_id: z.string().uuid().nullable(),
  controller_tick_count: z.number().int().nonnegative(),
  physics_substep_count: z.number().int().nonnegative(),
  mode_tick_counts: NeuralReflexArcModeCountsSchema,
  route_tick_counts: NeuralReflexArcRouteCountsSchema,
  local_correction_tick_count: z.number().int().nonnegative(),
  peak_weighted_joint_tracking_error: z.number().finite().nonnegative().nullable(),
  last_weighted_joint_tracking_error: z.number().finite().nonnegative().nullable(),
  peak_controller_reference_delta: z.number().finite().nonnegative().nullable(),
  last_controller_reference_delta: z.number().finite().nonnegative().nullable(),
  maximum_joint_velocity: z.number().finite().nonnegative().nullable(),
  non_foot_contact_tick_count: z.number().int().nonnegative(),
  support_state: z.enum(["none", "left", "right", "double"]),
  fallen: z.boolean(),
  controller_implementation: z.string().trim().min(1).nullable(),
  controller_mode: z.enum([
    "learned_policy",
    "reference_control",
    "hybrid_control"
  ]).nullable(),
  controller_route: z.enum([
    "direct",
    "primary",
    "fallback",
    "upper_body_overlay"
  ]).nullable(),
  started_simulated_time_seconds: z.number().finite().nonnegative().nullable(),
  last_simulated_time_seconds: z.number().finite().nonnegative().nullable(),
  terminal_code: z.string().trim().min(1).nullable(),
  started_at: z.string().datetime().nullable(),
  updated_at: z.string().datetime(),
  completed_at: z.string().datetime().nullable()
}).strict().superRefine((arc, context) => {
  const modeTicks = Object.values(arc.mode_tick_counts).reduce(
    (total, count) => total + count,
    0
  );
  const routeTicks = Object.values(arc.route_tick_counts).reduce(
    (total, count) => total + count,
    0
  );
  if (modeTicks !== arc.controller_tick_count
    || routeTicks !== arc.controller_tick_count) {
    context.addIssue({
      code: "custom",
      path: ["controller_tick_count"],
      message: "Reflex mode and route counts must cover every controller tick"
    });
  }
  if (arc.local_correction_tick_count > arc.controller_tick_count
    || arc.non_foot_contact_tick_count > arc.controller_tick_count) {
    context.addIssue({
      code: "custom",
      path: ["local_correction_tick_count"],
      message: "Reflex event counts cannot exceed controller ticks"
    });
  }
  const idle = arc.status === "idle";
  const terminal = arc.status === "succeeded"
    || arc.status === "failed"
    || arc.status === "interrupted";
  if (idle !== (arc.execution_transaction_id === null)
    || idle !== (arc.commitment_id === null)
    || idle !== (arc.started_at === null)
    || idle !== (arc.started_simulated_time_seconds === null)
    || idle !== (arc.last_simulated_time_seconds === null)) {
    context.addIssue({
      code: "custom",
      path: ["execution_transaction_id"],
      message: "Only a bound reflex arc carries execution identity and time bounds"
    });
  }
  if (terminal !== (arc.completed_at !== null)
    || terminal !== (arc.terminal_code !== null)) {
    context.addIssue({
      code: "custom",
      path: ["completed_at"],
      message: "A terminal reflex arc requires terminal metadata"
    });
  }
  const observed = arc.controller_tick_count > 0;
  const observedMetrics = [
    arc.peak_weighted_joint_tracking_error,
    arc.last_weighted_joint_tracking_error,
    arc.peak_controller_reference_delta,
    arc.last_controller_reference_delta,
    arc.maximum_joint_velocity,
    arc.controller_implementation,
    arc.controller_mode,
    arc.controller_route
  ];
  if (observedMetrics.some((value) => (value !== null) !== observed)) {
    context.addIssue({
      code: "custom",
      path: ["controller_tick_count"],
      message: "Reflex controller metrics must exactly match observed tick coverage"
    });
  }
  if ((arc.peak_weighted_joint_tracking_error ?? 0)
      < (arc.last_weighted_joint_tracking_error ?? 0)
    || (arc.peak_controller_reference_delta ?? 0)
      < (arc.last_controller_reference_delta ?? 0)) {
    context.addIssue({
      code: "custom",
      path: ["peak_weighted_joint_tracking_error"],
      message: "Reflex peak errors cannot be smaller than their latest sample"
    });
  }
  if (arc.started_simulated_time_seconds !== null
    && arc.last_simulated_time_seconds !== null
    && arc.last_simulated_time_seconds < arc.started_simulated_time_seconds) {
    context.addIssue({
      code: "custom",
      path: ["last_simulated_time_seconds"],
      message: "Reflex simulated time cannot regress"
    });
  }
  if (idle && (arc.controller_tick_count !== 0
    || arc.physics_substep_count !== 0
    || arc.local_correction_tick_count !== 0
    || arc.non_foot_contact_tick_count !== 0
    || arc.fallen)) {
    context.addIssue({
      code: "custom",
      path: ["status"],
      message: "An idle reflex arc cannot contain physical observations"
    });
  }
});

export type NeuralReflexArcState = z.infer<typeof NeuralReflexArcStateSchema>;

function initialNeuralReflexArcState(
  at = new Date().toISOString()
): NeuralReflexArcState {
  return {
    protocol: "neural-reflex-arc-state-v1",
    status: "idle",
    execution_transaction_id: null,
    commitment_id: null,
    controller_tick_count: 0,
    physics_substep_count: 0,
    mode_tick_counts: {
      learned_policy: 0,
      reference_control: 0,
      hybrid_control: 0
    },
    route_tick_counts: {
      direct: 0,
      primary: 0,
      fallback: 0,
      upper_body_overlay: 0
    },
    local_correction_tick_count: 0,
    peak_weighted_joint_tracking_error: null,
    last_weighted_joint_tracking_error: null,
    peak_controller_reference_delta: null,
    last_controller_reference_delta: null,
    maximum_joint_velocity: null,
    non_foot_contact_tick_count: 0,
    support_state: "none",
    fallen: false,
    controller_implementation: null,
    controller_mode: null,
    controller_route: null,
    started_simulated_time_seconds: null,
    last_simulated_time_seconds: null,
    terminal_code: null,
    started_at: null,
    updated_at: at,
    completed_at: null
  };
}

const DEFAULT_HARNESS_OWNER = "humanoid-executive";

function initialNeuralHarnessPhase(
  at = new Date().toISOString()
): NeuralHarnessPhaseState {
  return {
    phase: "bootstrapping",
    sequence: 0,
    goal_epoch_id: null,
    commitment_id: null,
    world_revision: 0,
    entered_by_node_id: DEFAULT_HARNESS_OWNER,
    reason: "neural_hierarchy_created",
    entered_at: at
  };
}

export const NeuralHierarchyStateSchema = z.object({
  version: z.literal(NEURAL_HIERARCHY_CONTRACT_VERSION),
  epoch_id: z.string().uuid(),
  next_sequence: z.number().int().positive(),
  signals: z.record(z.string().uuid(), NeuralSignalSchema),
  active_skill_commitment: NeuralSkillCommitmentSchema.nullable(),
  safety_interrupts: z.record(
    z.string().uuid(),
    NeuralSafetyInterruptSchema
  ).default({}),
  prediction_errors: z.array(NeuralPredictionErrorSchema).max(256),
  pathway_cadences: z.record(NeuralPathwaySchema, NeuralPathwayCadenceSchema),
  harness_phase: NeuralHarnessPhaseStateSchema.default(
    () => initialNeuralHarnessPhase()
  ),
  authority_leases: z.record(
    z.string().uuid(),
    NeuralAuthorityLeaseSchema
  ).default({}),
  rollout_certificates: z.record(
    z.string().uuid(),
    NeuralRolloutCertificateSchema
  ).default({}),
  reflex_arc: NeuralReflexArcStateSchema.default(
    () => initialNeuralReflexArcState()
  ),
  updated_at: z.string().datetime()
}).strict().superRefine((state, context) => {
  const sequences = new Set<number>();
  let maximumSequence = 0;
  for (const [signalId, signal] of Object.entries(state.signals)) {
    if (signalId !== signal.signal_id) {
      context.addIssue({
        code: "custom",
        path: ["signals", signalId, "signal_id"],
        message: "Neural signal record key must match its signal identity"
      });
    }
    if (sequences.has(signal.sequence)) {
      context.addIssue({
        code: "custom",
        path: ["signals", signalId, "sequence"],
        message: "Neural signal sequences must be unique"
      });
    }
    sequences.add(signal.sequence);
    maximumSequence = Math.max(maximumSequence, signal.sequence);
    for (const parentId of signal.causal_parent_ids) {
      const parent = state.signals[parentId];
      if (!parent || parent.sequence >= signal.sequence) {
        context.addIssue({
          code: "custom",
          path: ["signals", signalId, "causal_parent_ids"],
          message: "Neural causal parents must exist and precede their child"
        });
      }
    }
  const provenanceLeaseId = signal.direction === "descending"
      ? signal.authority_lease_id
      : signal.source_authority_lease_id;
    const provenanceLease = provenanceLeaseId
      ? state.authority_leases[provenanceLeaseId]
      : undefined;
    if (!provenanceLease
      || provenanceLease.invocation_id !== signal.invocation_id
      || provenanceLease.parent_invocation_id !== signal.parent_invocation_id
      || provenanceLease.parent_episode_id !== signal.parent_episode_id) {
      context.addIssue({
        code: "custom",
        path: ["signals", signalId, "invocation_id"],
        message: "Neural signal invocation must exactly match its authority lease"
      });
    }
  }
  for (const [interruptId, interrupt] of Object.entries(state.safety_interrupts)) {
    if (interruptId !== interrupt.interrupt_id) {
      context.addIssue({
        code: "custom",
        path: ["safety_interrupts", interruptId, "interrupt_id"],
        message: "Safety interrupt record key must match its identity"
      });
    }
  }
  for (const interruptId of state.active_skill_commitment
    ?.transition_interrupt_ids ?? []) {
    if (!state.safety_interrupts[interruptId]) {
      context.addIssue({
        code: "custom",
        path: ["active_skill_commitment", "transition_interrupt_ids"],
        message: "Skill commitment transition references an unknown safety interrupt"
      });
    }
  }
  if (state.next_sequence <= maximumSequence) {
    context.addIssue({
      code: "custom",
      path: ["next_sequence"],
      message: "Neural signal sequence head must follow every durable signal"
    });
  }
  for (const pathway of NeuralPathwaySchema.options) {
    if (state.pathway_cadences[pathway]?.pathway !== pathway) {
      context.addIssue({
        code: "custom",
        path: ["pathway_cadences", pathway],
        message: "Neural hierarchy requires one cadence state per pathway"
      });
    }
  }
  const activeLeaseTargets = new Set<string>();
  for (const [leaseId, lease] of Object.entries(state.authority_leases)) {
    if (leaseId !== lease.lease_id) {
      context.addIssue({
        code: "custom",
        path: ["authority_leases", leaseId, "lease_id"],
        message: "Neural authority lease record key must match its identity"
      });
    }
    for (const suspendedId of lease.suspended_lease_ids) {
      if (!state.authority_leases[suspendedId]) {
        context.addIssue({
          code: "custom",
          path: ["authority_leases", leaseId, "suspended_lease_ids"],
          message: "A recovery lease cannot reference an unknown suspended lease"
        });
      }
    }
    if (lease.status !== "active") continue;
    if (activeLeaseTargets.has(lease.target_child_node_id)) {
      context.addIssue({
        code: "custom",
        path: ["authority_leases", leaseId, "target_child_node_id"],
        message: "A neural child cannot have two active control leases"
      });
    }
    activeLeaseTargets.add(lease.target_child_node_id);
  }
  const activeCertificateCommitments = new Set<string>();
  for (const [certificateId, certificate] of Object.entries(
    state.rollout_certificates
  )) {
    if (certificateId !== certificate.certificate_id) {
      context.addIssue({
        code: "custom",
        path: ["rollout_certificates", certificateId, "certificate_id"],
        message: "Rollout certificate record key must match its identity"
      });
    }
    const rollout = state.signals[certificate.rollout_signal_id];
    const predictive = state.signals[certificate.predictive_signal_id];
    if (!rollout || rollout.kind !== "rollout_result"
      || rollout.direction !== "reentrant") {
      context.addIssue({
        code: "custom",
        path: ["rollout_certificates", certificateId, "rollout_signal_id"],
        message: "Rollout certificate must reference real reentrant rollout feedback"
      });
    } else if (modelPayloadSha256(rollout.payload)
      !== certificate.rollout_payload_sha256) {
      context.addIssue({
        code: "custom",
        path: ["rollout_certificates", certificateId, "rollout_payload_sha256"],
        message: "Rollout certificate payload hash does not match its rollout signal"
      });
    } else if (rollout.invocation_id !== certificate.rollout_invocation_id) {
      context.addIssue({
        code: "custom",
        path: ["rollout_certificates", certificateId, "rollout_invocation_id"],
        message: "Rollout certificate must retain its exact invocation episode"
      });
    }
    if (!predictive || predictive.kind !== "forward_prediction") {
      context.addIssue({
        code: "custom",
        path: ["rollout_certificates", certificateId, "predictive_signal_id"],
        message: "Rollout certificate must reference Predictive acceptance"
      });
    } else if (predictive.invocation_id !== certificate.predictive_invocation_id) {
      context.addIssue({
        code: "custom",
        path: ["rollout_certificates", certificateId, "predictive_signal_id"],
        message: "Predictive acceptance must belong to the certified invocation episode"
      });
    }
    if (certificate.status !== "active") continue;
    if (activeCertificateCommitments.has(certificate.commitment_id)) {
      context.addIssue({
        code: "custom",
        path: ["rollout_certificates", certificateId, "commitment_id"],
        message: "A commitment cannot have two active rollout certificates"
      });
    }
    activeCertificateCommitments.add(certificate.commitment_id);
    if (!state.active_skill_commitment
      || state.active_skill_commitment.commitment_id !== certificate.commitment_id
      || state.active_skill_commitment.goal_epoch_id !== certificate.goal_epoch_id) {
      context.addIssue({
        code: "custom",
        path: ["rollout_certificates", certificateId, "commitment_id"],
        message: "Active rollout certificate must belong to the active commitment"
      });
    }
  }
});

export type NeuralHierarchyState = z.infer<typeof NeuralHierarchyStateSchema>;

const DEFAULT_PATHWAY_CADENCES_MS: Readonly<Record<
  NeuralPathway,
  readonly [minimumIntervalMs: number, maximumStalenessMs: number]
>> = {
  executive_control: [2_000, 30_000],
  goal_valuation: [5_000, 60_000],
  perceptual_association: [100, 2_000],
  sensorimotor_selection: [250, 5_000],
  cerebellar_prediction: [50, 1_000],
  interoceptive_risk: [25, 500],
  premotor_composition: [100, 2_000],
  motor_intent: [50, 1_000],
  physical_execution: [0, 500],
  ascending_feedback: [25, 500]
};

export function createNeuralHierarchyState(
  at = new Date().toISOString(),
  epochId = randomUUID()
): NeuralHierarchyState {
  return NeuralHierarchyStateSchema.parse({
    version: NEURAL_HIERARCHY_CONTRACT_VERSION,
    epoch_id: epochId,
    next_sequence: 1,
    signals: {},
    active_skill_commitment: null,
    prediction_errors: [],
    harness_phase: initialNeuralHarnessPhase(at),
    authority_leases: {},
    rollout_certificates: {},
    reflex_arc: initialNeuralReflexArcState(at),
    pathway_cadences: Object.fromEntries(
      NeuralPathwaySchema.options.map((pathway) => {
        const [minimumIntervalMs, maximumStalenessMs]
          = DEFAULT_PATHWAY_CADENCES_MS[pathway];
        return [pathway, {
          pathway,
          minimum_interval_ms: minimumIntervalMs,
          maximum_staleness_ms: maximumStalenessMs,
          last_dispatched_at: null,
          last_completed_at: null,
          dispatch_count: 0
        }];
      })
    ),
    updated_at: at
  });
}

export interface BeginNeuralReflexArcInput {
  executionTransactionId: string;
  commitmentId: string;
  simulatedTimeSeconds: number;
  at?: string;
}

export function beginNeuralReflexArc(
  stateInput: NeuralHierarchyState,
  input: BeginNeuralReflexArcInput
): NeuralHierarchyState {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const executionTransactionId = input.executionTransactionId.trim();
  if (!executionTransactionId) {
    throw new Error("A reflex arc requires a physical execution transaction");
  }
  if (!Number.isFinite(input.simulatedTimeSeconds)
    || input.simulatedTimeSeconds < 0) {
    throw new Error("A reflex arc requires nonnegative simulated time");
  }
  const existing = state.reflex_arc;
  if (existing.status === "active") {
    if (existing.execution_transaction_id !== executionTransactionId
      || existing.commitment_id !== input.commitmentId) {
      throw new Error(
        `Reflex arc ${existing.execution_transaction_id} must finish before ${executionTransactionId}`
      );
    }
    return state;
  }
  const at = input.at ?? new Date().toISOString();
  state.reflex_arc = NeuralReflexArcStateSchema.parse({
    ...initialNeuralReflexArcState(at),
    status: "active",
    execution_transaction_id: executionTransactionId,
    commitment_id: input.commitmentId,
    started_simulated_time_seconds: input.simulatedTimeSeconds,
    last_simulated_time_seconds: input.simulatedTimeSeconds,
    started_at: at
  });
  state.updated_at = at;
  return NeuralHierarchyStateSchema.parse(state);
}

export interface RecordNeuralReflexArcFrameInput
  extends BeginNeuralReflexArcInput {
  controllerMode: "learned_policy" | "reference_control" | "hybrid_control";
  controllerRoute: "direct" | "primary" | "fallback" | "upper_body_overlay";
  controllerImplementation: string;
  physicsSubsteps: number;
  weightedJointTrackingError: number;
  controllerReferenceDelta: number;
  maximumJointVelocity: number;
  localCorrection: boolean;
  nonFootContact: boolean;
  supportState: "none" | "left" | "right" | "double";
  fallen: boolean;
}

export function recordNeuralReflexArcFrame(
  stateInput: NeuralHierarchyState,
  input: RecordNeuralReflexArcFrameInput
): NeuralHierarchyState {
  let state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  if (state.reflex_arc.status !== "active") {
    state = beginNeuralReflexArc(state, input);
  }
  const arc = state.reflex_arc;
  if (arc.execution_transaction_id !== input.executionTransactionId.trim()
    || arc.commitment_id !== input.commitmentId) {
    throw new Error("A controller frame cannot cross reflex execution ownership");
  }
  if (!Number.isSafeInteger(input.physicsSubsteps) || input.physicsSubsteps <= 0
    || ![
      input.simulatedTimeSeconds,
      input.weightedJointTrackingError,
      input.controllerReferenceDelta,
      input.maximumJointVelocity
    ].every((value) => Number.isFinite(value) && value >= 0)) {
    throw new Error("A reflex controller frame contains invalid physical metrics");
  }
  if (arc.controller_tick_count > 0
    && input.simulatedTimeSeconds <= arc.last_simulated_time_seconds!) {
    throw new Error("A reflex controller frame cannot replay or regress simulated time");
  }
  const controllerImplementation = input.controllerImplementation.trim();
  if (!controllerImplementation) {
    throw new Error("A reflex controller frame requires an implementation identity");
  }
  const at = input.at ?? new Date().toISOString();
  state.reflex_arc = NeuralReflexArcStateSchema.parse({
    ...arc,
    controller_tick_count: arc.controller_tick_count + 1,
    physics_substep_count: arc.physics_substep_count + input.physicsSubsteps,
    mode_tick_counts: {
      ...arc.mode_tick_counts,
      [input.controllerMode]: arc.mode_tick_counts[input.controllerMode] + 1
    },
    route_tick_counts: {
      ...arc.route_tick_counts,
      [input.controllerRoute]: arc.route_tick_counts[input.controllerRoute] + 1
    },
    local_correction_tick_count: arc.local_correction_tick_count
      + (input.localCorrection ? 1 : 0),
    peak_weighted_joint_tracking_error: Math.max(
      arc.peak_weighted_joint_tracking_error ?? 0,
      input.weightedJointTrackingError
    ),
    last_weighted_joint_tracking_error: input.weightedJointTrackingError,
    peak_controller_reference_delta: Math.max(
      arc.peak_controller_reference_delta ?? 0,
      input.controllerReferenceDelta
    ),
    last_controller_reference_delta: input.controllerReferenceDelta,
    maximum_joint_velocity: Math.max(
      arc.maximum_joint_velocity ?? 0,
      input.maximumJointVelocity
    ),
    non_foot_contact_tick_count: arc.non_foot_contact_tick_count
      + (input.nonFootContact ? 1 : 0),
    support_state: input.supportState,
    fallen: arc.fallen || input.fallen,
    controller_implementation: controllerImplementation,
    controller_mode: input.controllerMode,
    controller_route: input.controllerRoute,
    last_simulated_time_seconds: input.simulatedTimeSeconds,
    updated_at: at
  });
  state.updated_at = at;
  return NeuralHierarchyStateSchema.parse(state);
}

export function completeNeuralReflexArc(
  stateInput: NeuralHierarchyState,
  input: {
    executionTransactionId: string;
    status: "succeeded" | "failed" | "interrupted";
    terminalCode: string;
    simulatedTimeSeconds?: number;
    at?: string;
  }
): NeuralHierarchyState {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const executionTransactionId = input.executionTransactionId.trim();
  const terminalCode = input.terminalCode.trim();
  const arc = state.reflex_arc;
  if (!executionTransactionId || !terminalCode) {
    throw new Error("A terminal reflex arc requires transaction and result identities");
  }
  if (arc.status !== "active") {
    if (arc.execution_transaction_id === executionTransactionId
      && arc.status === input.status
      && arc.terminal_code === terminalCode) return state;
    throw new Error(`No active reflex arc belongs to ${executionTransactionId}`);
  }
  if (arc.execution_transaction_id !== executionTransactionId) {
    throw new Error(
      `Reflex arc ${arc.execution_transaction_id} cannot finish as ${executionTransactionId}`
    );
  }
  const simulatedTimeSeconds = input.simulatedTimeSeconds
    ?? arc.last_simulated_time_seconds!;
  if (!Number.isFinite(simulatedTimeSeconds)
    || simulatedTimeSeconds < arc.last_simulated_time_seconds!) {
    throw new Error("A terminal reflex arc cannot regress simulated time");
  }
  const at = input.at ?? new Date().toISOString();
  state.reflex_arc = NeuralReflexArcStateSchema.parse({
    ...arc,
    status: input.status,
    last_simulated_time_seconds: simulatedTimeSeconds,
    terminal_code: terminalCode,
    updated_at: at,
    completed_at: at
  });
  state.updated_at = at;
  return NeuralHierarchyStateSchema.parse(state);
}

export function neuralReflexArcSummary(
  arcInput: NeuralReflexArcState
): JsonValue {
  const arc = NeuralReflexArcStateSchema.parse(structuredClone(arcInput));
  return {
    protocol: "neural-reflex-arc-summary-v1",
    status: arc.status,
    execution_transaction_id: arc.execution_transaction_id,
    commitment_id: arc.commitment_id,
    controller_ticks: arc.controller_tick_count,
    physics_substeps: arc.physics_substep_count,
    mode_ticks: arc.mode_tick_counts,
    route_ticks: arc.route_tick_counts,
    local_correction_ticks: arc.local_correction_tick_count,
    peak_weighted_joint_tracking_error:
      arc.peak_weighted_joint_tracking_error,
    last_weighted_joint_tracking_error:
      arc.last_weighted_joint_tracking_error,
    peak_controller_reference_delta: arc.peak_controller_reference_delta,
    last_controller_reference_delta: arc.last_controller_reference_delta,
    maximum_joint_velocity: arc.maximum_joint_velocity,
    non_foot_contact_ticks: arc.non_foot_contact_tick_count,
    support_state: arc.support_state,
    fallen: arc.fallen,
    controller: arc.controller_implementation === null
      ? null
      : {
          implementation: arc.controller_implementation,
          mode: arc.controller_mode,
          route: arc.controller_route
        },
    simulated_duration_seconds:
      arc.started_simulated_time_seconds === null
        || arc.last_simulated_time_seconds === null
        ? null
        : Math.max(
            0,
            arc.last_simulated_time_seconds
              - arc.started_simulated_time_seconds
          ),
    terminal_code: arc.terminal_code
  };
}

export interface PublishNeuralSignalInput {
  kind: NeuralSignalKind;
  pathway: NeuralPathway;
  direction: NeuralSignalDirection;
  sourceNodeId: string;
  sourceLayer: NeuralLayer;
  targetNodeId: string;
  targetLayer: NeuralLayer;
  worldFrame: number;
  worldRevision: number;
  ttlRevisions: number;
  priority: number;
  invocationId: string;
  parentInvocationId: string | null;
  parentEpisodeId: string;
  causalParentIds?: readonly string[];
  authorityLeaseId?: string | null;
  sourceAuthorityLeaseId?: string | null;
  payload: JsonValue;
  signalId?: string;
  liveInvocationIds?: readonly string[];
  at?: string;
}

export function publishNeuralSignal(
  stateInput: NeuralHierarchyState,
  input: PublishNeuralSignalInput
): { state: NeuralHierarchyState; signal: NeuralSignal } {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const at = input.at ?? new Date().toISOString();
  const signal = NeuralSignalSchema.parse({
    signal_id: input.signalId ?? randomUUID(),
    sequence: state.next_sequence,
    kind: input.kind,
    pathway: input.pathway,
    direction: input.direction,
    source_node_id: input.sourceNodeId,
    source_layer: input.sourceLayer,
    target_node_id: input.targetNodeId,
    target_layer: input.targetLayer,
    world_frame: input.worldFrame,
    world_revision: input.worldRevision,
    ttl_revisions: input.ttlRevisions,
    priority: input.priority,
    invocation_id: input.invocationId,
    parent_invocation_id: input.parentInvocationId,
    parent_episode_id: input.parentEpisodeId,
    causal_parent_ids: [...(input.causalParentIds ?? [])],
    authority_lease_id: input.authorityLeaseId ?? null,
    source_authority_lease_id: input.sourceAuthorityLeaseId ?? null,
    payload: structuredClone(input.payload),
    status: "pending",
    created_at: at
  });
  if (state.signals[signal.signal_id]) {
    throw new Error(`Duplicate neural signal identity: ${signal.signal_id}`);
  }
  for (const parentId of signal.causal_parent_ids) {
    if (!state.signals[parentId]) {
      throw new Error(`Neural signal references an unknown causal parent: ${parentId}`);
    }
  }
  if (signal.authority_lease_id !== null
    && !state.authority_leases[signal.authority_lease_id]) {
    throw new Error(
      `Neural signal references an unknown authority lease: ${signal.authority_lease_id}`
    );
  }
  if (signal.source_authority_lease_id !== null
    && !state.authority_leases[signal.source_authority_lease_id]) {
    throw new Error(
      `Neural signal references an unknown source authority lease: ${signal.source_authority_lease_id}`
    );
  }
  const leaseId = signal.direction === "descending"
    ? signal.authority_lease_id
    : signal.source_authority_lease_id;
  const lease = leaseId === null ? undefined : state.authority_leases[leaseId];
  if (!lease || lease.invocation_id !== signal.invocation_id
    || lease.parent_invocation_id !== signal.parent_invocation_id
    || lease.parent_episode_id !== signal.parent_episode_id) {
    throw new Error("Neural signal invocation identity does not match its authority lease");
  }
  state.signals[signal.signal_id] = signal;
  state.next_sequence += 1;
  state.updated_at = at;
  return {
    state: compactNeuralHierarchyState(
      state,
      input.worldRevision,
      128,
      input.liveInvocationIds
    ),
    signal: structuredClone(signal)
  };
}

export function pendingNeuralSignals(input: {
  state: NeuralHierarchyState;
  targetNodeId?: string;
  kinds?: readonly NeuralSignalKind[];
  worldRevision: number;
  liveInvocationIds?: readonly string[];
}): NeuralSignal[] {
  const kinds = input.kinds ? new Set(input.kinds) : undefined;
  const liveInvocationIds = new Set(input.liveInvocationIds ?? []);
  return Object.values(input.state.signals)
    .filter((signal) => signal.status === "pending"
      && (liveInvocationIds.has(signal.invocation_id)
        || liveInvocationIds.has(signal.parent_episode_id)
        || input.worldRevision <= signal.world_revision + signal.ttl_revisions)
      && (input.targetNodeId === undefined
        || signal.target_node_id === input.targetNodeId)
      && (kinds === undefined || kinds.has(signal.kind)))
    .sort((left, right) => (
      right.priority - left.priority || left.sequence - right.sequence
    ))
    .map((signal) => structuredClone(signal));
}

export function consumeNeuralSignals(
  stateInput: NeuralHierarchyState,
  signalIds: readonly string[],
  at = new Date().toISOString()
): NeuralHierarchyState {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  for (const signalId of new Set(signalIds)) {
    const signal = state.signals[signalId];
    if (!signal) throw new Error(`Cannot consume unknown neural signal: ${signalId}`);
    if (signal.status !== "pending") continue;
    signal.status = "consumed";
    signal.consumed_at = at;
  }
  state.updated_at = at;
  return NeuralHierarchyStateSchema.parse(state);
}

export function markNeuralPathwayDispatch(
  stateInput: NeuralHierarchyState,
  pathway: NeuralPathway,
  phase: "started" | "completed",
  at = new Date().toISOString()
): NeuralHierarchyState {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const cadence = state.pathway_cadences[pathway];
  if (phase === "started") {
    cadence.last_dispatched_at = at;
    cadence.dispatch_count += 1;
  } else {
    cadence.last_completed_at = at;
  }
  state.updated_at = at;
  return NeuralHierarchyStateSchema.parse(state);
}

const NEURAL_HARNESS_PHASE_TRANSITIONS: Readonly<Record<
  NeuralHarnessPhase,
  ReadonlySet<NeuralHarnessPhase>
>> = {
  bootstrapping: new Set([
    "goal_valuation", "perception", "safety_interrupt", "terminal"
  ]),
  goal_valuation: new Set([
    "perception", "cycle_completion", "safety_interrupt", "terminal"
  ]),
  perception: new Set([
    "skill_proposal",
    "motor_assessment",
    "recovery",
    "goal_valuation",
    "cycle_completion",
    "safety_interrupt",
    "terminal"
  ]),
  skill_proposal: new Set([
    "commitment_authorization",
    "perception",
    "goal_valuation",
    "recovery",
    "safety_interrupt",
    "terminal"
  ]),
  commitment_authorization: new Set([
    "motor_assessment",
    "skill_proposal",
    "goal_valuation",
    "safety_interrupt",
    "terminal"
  ]),
  motor_assessment: new Set([
    "motor_planning",
    "skill_proposal",
    "perception",
    "recovery",
    "safety_interrupt",
    "terminal"
  ]),
  motor_planning: new Set([
    "rollout_review",
    "motor_assessment",
    "recovery",
    "safety_interrupt",
    "terminal"
  ]),
  rollout_review: new Set([
    "execution",
    "motor_assessment",
    "skill_proposal",
    "recovery",
    "safety_interrupt",
    "terminal"
  ]),
  execution: new Set(["feedback", "recovery", "safety_interrupt", "terminal"]),
  feedback: new Set([
    "perception",
    "motor_assessment",
    "skill_proposal",
    "cycle_completion",
    "recovery",
    "safety_interrupt",
    "terminal"
  ]),
  safety_interrupt: new Set(["perception", "recovery", "terminal"]),
  recovery: new Set([
    "motor_assessment",
    "skill_proposal",
    "perception",
    "goal_valuation",
    "safety_interrupt",
    "terminal"
  ]),
  cycle_completion: new Set([
    "goal_valuation", "perception", "safety_interrupt", "terminal"
  ]),
  terminal: new Set()
};

export function transitionNeuralHarnessPhase(
  stateInput: NeuralHierarchyState,
  input: {
    phase: NeuralHarnessPhase;
    goalEpochId: string | null;
    commitmentId: string | null;
    worldRevision: number;
    enteredByNodeId: string;
    reason: string;
    liveInvocationIds?: readonly string[];
    resumeFromTerminal?: boolean;
    at?: string;
  }
): NeuralHierarchyState {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const next = NeuralHarnessPhaseSchema.parse(input.phase);
  const previous = state.harness_phase;
  const explicitTerminalResume = previous.phase === "terminal"
    && input.resumeFromTerminal === true;
  if (next !== previous.phase
    && !explicitTerminalResume
    && !NEURAL_HARNESS_PHASE_TRANSITIONS[previous.phase].has(next)) {
    throw new Error(
      `Invalid neural Harness phase transition: ${previous.phase} -> ${next}`
    );
  }
  if (input.worldRevision < previous.world_revision) {
    throw new Error("Neural Harness phase cannot move to an older world revision");
  }
  const active = state.active_skill_commitment;
  if (input.commitmentId !== null
    && (!active || active.commitment_id !== input.commitmentId)) {
    throw new Error(`Harness phase references unknown commitment: ${input.commitmentId}`);
  }
  if (active && input.commitmentId === active.commitment_id
    && input.goalEpochId !== active.goal_epoch_id) {
    throw new Error("Harness phase Goal epoch does not own its skill commitment");
  }
  const at = input.at ?? new Date().toISOString();
  const liveInvocationIds = new Set(input.liveInvocationIds ?? []);
  state.harness_phase = NeuralHarnessPhaseStateSchema.parse({
    phase: next,
    sequence: previous.sequence + 1,
    goal_epoch_id: input.goalEpochId,
    commitment_id: input.commitmentId,
    world_revision: input.worldRevision,
    entered_by_node_id: input.enteredByNodeId,
    reason: input.reason,
    entered_at: at
  });
  for (const lease of Object.values(state.authority_leases)) {
    if (lease.status !== "active" && lease.status !== "suspended") continue;
    const invocationIsLive = liveInvocationIds.has(lease.invocation_id);
    // A nested child may legitimately advance the global phase before its
    // typed result has unwound through every structural parent. Preserve only
    // leases on that exact live invocation stack long enough to publish those
    // already-authorized return edges; non-live leases are revoked immediately.
    // Terminal entry always closes the graph.
    const preserveLiveReturnPath = next !== "terminal" && invocationIsLive;
    const incompatibleGoal = !preserveLiveReturnPath
      && lease.goal_epoch_id !== null
      && lease.goal_epoch_id !== input.goalEpochId;
    const incompatibleCommitment = !preserveLiveReturnPath
      && lease.commitment_id !== null
      && lease.commitment_id !== input.commitmentId;
    const expiredRevision = !invocationIsLive
      && input.worldRevision > lease.expires_world_revision;
    const expiredTime = !invocationIsLive
      && Date.parse(at) > Date.parse(lease.expires_at);
    if (!incompatibleGoal && !incompatibleCommitment
      && !expiredRevision && !expiredTime) continue;
    lease.status = expiredRevision || expiredTime ? "expired" : "revoked";
    lease.closed_at = at;
    lease.close_reason = expiredRevision || expiredTime
      ? "phase_transition_expired"
      : "phase_authority_changed";
  }
  state.updated_at = at;
  return NeuralHierarchyStateSchema.parse(state);
}

export function issueNeuralAuthorityLease(
  stateInput: NeuralHierarchyState,
  input: {
    issuingParentNodeId: string;
    targetChildNodeId: string;
    goalEpochId: string | null;
    commitmentId: string | null;
    worldRevision: number;
    expiresWorldRevision: number;
    expiresAt: string;
    allowedSignalKinds: readonly NeuralSignalKind[];
    correctionScope?: NeuralAuthorityLease["correction_scope"];
    invocationId?: string;
    parentInvocationId?: string | null;
    parentEpisodeId: string;
    liveInvocationIds?: readonly string[];
    exclusive?: boolean;
    suspendLeaseIds?: readonly string[];
    leaseId?: string;
    at?: string;
  }
): { state: NeuralHierarchyState; lease: NeuralAuthorityLease } {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const at = input.at ?? new Date().toISOString();
  if (input.worldRevision < state.harness_phase.world_revision) {
    throw new Error("Authority lease cannot precede the current Harness phase revision");
  }
  if (input.goalEpochId !== state.harness_phase.goal_epoch_id) {
    throw new Error("Authority lease must bind the current Harness Goal epoch");
  }
  if (input.commitmentId !== state.harness_phase.commitment_id) {
    throw new Error("Authority lease must bind the current Harness commitment");
  }
  if (Date.parse(input.expiresAt) <= Date.parse(at)) {
    throw new Error("Authority lease expiration must be in the future");
  }
  const liveInvocationIds = new Set(input.liveInvocationIds ?? []);
  for (const existing of Object.values(state.authority_leases)) {
    if (existing.status !== "active" && existing.status !== "suspended") continue;
    if (liveInvocationIds.has(existing.invocation_id)) continue;
    if (input.worldRevision <= existing.expires_world_revision
      && Date.parse(at) <= Date.parse(existing.expires_at)) continue;
    existing.status = "expired";
    existing.closed_at = at;
    existing.close_reason = "authority_horizon_elapsed_before_new_episode";
  }
  const duplicate = Object.values(state.authority_leases).find((lease) => (
    lease.target_child_node_id === input.targetChildNodeId
      && lease.status === "active"
  ));
  if (duplicate) {
    const sameInvocation = input.invocationId !== undefined
      && duplicate.invocation_id === input.invocationId
      && duplicate.issuing_parent_node_id === input.issuingParentNodeId
      && duplicate.parent_invocation_id === (input.parentInvocationId ?? null)
      && duplicate.parent_episode_id === input.parentEpisodeId
      && duplicate.goal_epoch_id === input.goalEpochId
      && duplicate.commitment_id === input.commitmentId;
    if (sameInvocation) {
      return {
        state: NeuralHierarchyStateSchema.parse(state),
        lease: structuredClone(duplicate)
      };
    }
    throw new Error(
      `Neural child already has active authority lease ${duplicate.lease_id}`
    );
  }
  const activeExclusiveSibling = Object.values(state.authority_leases).find(
    (lease) => (
      lease.status === "active"
        && lease.exclusive
        && lease.issuing_parent_node_id === input.issuingParentNodeId
        && lease.target_child_node_id !== input.targetChildNodeId
    )
  );
  if (activeExclusiveSibling) {
    throw new Error(
      `Exclusive neural lease ${activeExclusiveSibling.lease_id} freezes its sibling branches`
    );
  }
  const suspendLeaseIds = [...new Set(input.suspendLeaseIds ?? [])];
  if (!input.exclusive && suspendLeaseIds.length > 0) {
    throw new Error("Only an exclusive neural lease may suspend another branch");
  }
  for (const leaseId of suspendLeaseIds) {
    const suspended = state.authority_leases[leaseId];
    if (!suspended || suspended.status !== "active") {
      throw new Error(`Cannot suspend inactive neural authority lease: ${leaseId}`);
    }
    if (suspended.issuing_parent_node_id !== input.issuingParentNodeId) {
      throw new Error("An exclusive lease may suspend only its parent's own branches");
    }
    suspended.status = "suspended";
  }
  const lease = NeuralAuthorityLeaseSchema.parse({
    lease_id: input.leaseId ?? randomUUID(),
    issuing_parent_node_id: input.issuingParentNodeId,
    target_child_node_id: input.targetChildNodeId,
    goal_epoch_id: input.goalEpochId,
    commitment_id: input.commitmentId,
    issued_world_revision: input.worldRevision,
    expires_world_revision: input.expiresWorldRevision,
    allowed_signal_kinds: [...new Set(input.allowedSignalKinds)],
    correction_scope: input.correctionScope ?? "ordinary",
    invocation_id: input.invocationId ?? randomUUID(),
    parent_invocation_id: input.parentInvocationId ?? null,
    parent_episode_id: input.parentEpisodeId,
    exclusive: input.exclusive ?? false,
    suspended_lease_ids: suspendLeaseIds,
    status: "active",
    issued_at: at,
    expires_at: input.expiresAt,
    closed_at: null,
    close_reason: null
  });
  state.authority_leases[lease.lease_id] = lease;
  state.updated_at = at;
  return {
    state: NeuralHierarchyStateSchema.parse(state),
    lease: structuredClone(lease)
  };
}

export function closeNeuralAuthorityLease(
  stateInput: NeuralHierarchyState,
  input: {
    leaseId: string;
    closedByNodeId: string;
    reason: string;
    status?: "closed" | "revoked" | "expired";
    resumeSuspended?: boolean;
    worldRevision: number;
    liveInvocationIds?: readonly string[];
    at?: string;
  }
): NeuralHierarchyState {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const lease = state.authority_leases[input.leaseId];
  if (!lease) throw new Error(`Unknown neural authority lease: ${input.leaseId}`);
  if (lease.status === "closed" || lease.status === "revoked"
    || lease.status === "expired") return state;
  if (input.closedByNodeId !== lease.issuing_parent_node_id
    && input.closedByNodeId !== lease.target_child_node_id) {
    throw new Error("Only the issuing parent or target child may close a neural lease");
  }
  const at = input.at ?? new Date().toISOString();
  lease.status = input.status ?? "closed";
  lease.closed_at = at;
  lease.close_reason = input.reason;
  if (input.resumeSuspended) {
    if (!lease.exclusive) {
      throw new Error("Only an exclusive lease can resume suspended branches");
    }
    const liveInvocationIds = new Set(input.liveInvocationIds ?? []);
    for (const suspendedId of lease.suspended_lease_ids) {
      const suspended = state.authority_leases[suspendedId];
      if (!suspended || suspended.status !== "suspended") continue;
      const invocationIsLive = liveInvocationIds.has(suspended.invocation_id);
      const validRevision = invocationIsLive
        || input.worldRevision <= suspended.expires_world_revision;
      const validTime = invocationIsLive
        || Date.parse(at) <= Date.parse(suspended.expires_at);
      const validPhase = suspended.goal_epoch_id === state.harness_phase.goal_epoch_id
        && suspended.commitment_id === state.harness_phase.commitment_id;
      if (validRevision && validTime && validPhase) {
        suspended.status = "active";
      } else {
        suspended.status = "expired";
        suspended.closed_at = at;
        suspended.close_reason = "exclusive_lease_closed_after_authority_expired";
      }
    }
  }
  state.updated_at = at;
  return NeuralHierarchyStateSchema.parse(state);
}

export function activeNeuralAuthorityLease(input: {
  state: NeuralHierarchyState;
  targetChildNodeId: string;
  worldRevision: number;
  signalKind?: NeuralSignalKind;
  goalEpochId?: string | null;
  commitmentId?: string | null;
  liveInvocationIds?: readonly string[];
  now?: number;
}): NeuralAuthorityLease | undefined {
  const now = input.now ?? Date.now();
  const liveInvocationIds = new Set(input.liveInvocationIds ?? []);
  const lease = Object.values(input.state.authority_leases).find((candidate) => (
    candidate.target_child_node_id === input.targetChildNodeId
      && candidate.status === "active"
      && input.worldRevision >= candidate.issued_world_revision
      && (liveInvocationIds.has(candidate.invocation_id)
        || (input.worldRevision <= candidate.expires_world_revision
          && now <= Date.parse(candidate.expires_at)))
      && (input.signalKind === undefined
        || candidate.allowed_signal_kinds.includes(input.signalKind))
      && (input.goalEpochId === undefined
        || candidate.goal_epoch_id === input.goalEpochId)
      && (input.commitmentId === undefined
        || candidate.commitment_id === input.commitmentId)
  ));
  return lease ? structuredClone(lease) : undefined;
}

export function issueNeuralRolloutCertificate(
  stateInput: NeuralHierarchyState,
  input: {
    commitmentId: string;
    goalEpochId: string;
    issuedByNodeId: string;
    planningTransactionId: string;
    planningAction: NeuralPlanningAction;
    rolloutSignalId: string;
    predictiveSignalId: string;
  rolloutPayloadSha256: string;
    rolloutInvocationId: string;
    predictiveInvocationId: string;
    worldRevision: number;
    expiresWorldRevision: number;
    certificateId?: string;
    at?: string;
  }
): { state: NeuralHierarchyState; certificate: NeuralRolloutCertificate } {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const commitment = state.active_skill_commitment;
  if (!commitment || commitment.commitment_id !== input.commitmentId
    || commitment.goal_epoch_id !== input.goalEpochId
    || commitment.state !== "committed") {
    throw new Error("Rollout certificate requires the current committed Skill");
  }
  if (state.harness_phase.phase !== "rollout_review"
    || state.harness_phase.commitment_id !== input.commitmentId) {
    throw new Error("Rollout certificate can be issued only during rollout review");
  }
  const rollout = state.signals[input.rolloutSignalId];
  const predictive = state.signals[input.predictiveSignalId];
  if (!rollout || rollout.kind !== "rollout_result"
    || rollout.direction !== "reentrant") {
    throw new Error("Rollout certificate requires real reentrant rollout feedback");
  }
  if (rollout.invocation_id !== input.rolloutInvocationId) {
    throw new Error("Rollout certificate invocation does not match rollout feedback");
  }
  if (modelPayloadSha256(rollout.payload) !== input.rolloutPayloadSha256) {
    throw new Error("Rollout certificate payload hash does not match its rollout signal");
  }
  if (!predictive || predictive.kind !== "forward_prediction"
    || predictive.source_node_id !== input.issuedByNodeId
    || !predictive.causal_parent_ids.includes(rollout.signal_id)
    || predictive.invocation_id !== input.predictiveInvocationId) {
    throw new Error("Predictive acceptance must causally descend from the rollout");
  }
  if (input.worldRevision < rollout.world_revision
    || input.expiresWorldRevision < input.worldRevision) {
    throw new Error("Rollout certificate world revision is invalid");
  }
  for (const certificate of Object.values(state.rollout_certificates)) {
    if (certificate.status !== "active") continue;
    if (certificate.commitment_id === input.commitmentId) {
      throw new Error(
        `Commitment already has active rollout certificate ${certificate.certificate_id}`
      );
    }
  }
  const at = input.at ?? new Date().toISOString();
  const certificate = NeuralRolloutCertificateSchema.parse({
    certificate_id: input.certificateId ?? randomUUID(),
    commitment_id: input.commitmentId,
    goal_epoch_id: input.goalEpochId,
    issued_by_node_id: input.issuedByNodeId,
    planning_transaction_id: input.planningTransactionId,
    planning_action: input.planningAction,
    rollout_signal_id: input.rolloutSignalId,
    predictive_signal_id: input.predictiveSignalId,
    rollout_payload_sha256: input.rolloutPayloadSha256,
    rollout_invocation_id: input.rolloutInvocationId,
    predictive_invocation_id: input.predictiveInvocationId,
    rollout_world_revision: rollout.world_revision,
    issued_world_revision: input.worldRevision,
    expires_world_revision: input.expiresWorldRevision,
    status: "active",
    execution_transaction_id: null,
    issued_at: at,
    closed_at: null,
    close_reason: null
  });
  state.rollout_certificates[certificate.certificate_id] = certificate;
  state.updated_at = at;
  return {
    state: NeuralHierarchyStateSchema.parse(state),
    certificate: structuredClone(certificate)
  };
}

export function consumeNeuralRolloutCertificate(
  stateInput: NeuralHierarchyState,
  input: {
    certificateId: string;
    commitmentId: string;
    planningTransactionId: string;
    planningAction: NeuralPlanningAction;
    executionTransactionId: string;
    worldRevision: number;
    at?: string;
  }
): { state: NeuralHierarchyState; certificate: NeuralRolloutCertificate } {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const certificate = state.rollout_certificates[input.certificateId];
  if (certificate?.status === "consumed"
    && certificate.execution_transaction_id === input.executionTransactionId
    && certificate.commitment_id === input.commitmentId
    && certificate.planning_transaction_id === input.planningTransactionId
    && certificate.planning_action === input.planningAction) {
    return { state, certificate: structuredClone(certificate) };
  }
  if (!certificate || certificate.status !== "active") {
    throw new Error(`Unknown active rollout certificate: ${input.certificateId}`);
  }
  if (certificate.commitment_id !== input.commitmentId
    || certificate.planning_transaction_id !== input.planningTransactionId
    || certificate.planning_action !== input.planningAction) {
    throw new Error("Rollout certificate does not bind the requested physical plan");
  }
  const rollout = state.signals[certificate.rollout_signal_id];
  if (!rollout || modelPayloadSha256(rollout.payload)
    !== certificate.rollout_payload_sha256) {
    throw new Error("Rollout certificate payload integrity failed at physical admission");
  }
  if (input.worldRevision < certificate.issued_world_revision
    || input.worldRevision > certificate.expires_world_revision) {
    throw new Error("Rollout certificate is stale for physical execution");
  }
  const commitment = state.active_skill_commitment;
  if (!commitment || commitment.commitment_id !== input.commitmentId
    || commitment.state !== "executing") {
    throw new Error("Rollout certificate requires an executing commitment");
  }
  const at = input.at ?? new Date().toISOString();
  certificate.status = "consumed";
  certificate.execution_transaction_id = input.executionTransactionId;
  certificate.closed_at = at;
  certificate.close_reason = "serial_executor_admitted_physical_transaction";
  state.updated_at = at;
  return {
    state: NeuralHierarchyStateSchema.parse(state),
    certificate: structuredClone(certificate)
  };
}

export function establishNeuralSkillCommitment(
  stateInput: NeuralHierarchyState,
  input: {
    goalEpochId: string;
    ownerNodeId: string;
    skill: string;
    worldRevision: number;
    terminationContract: JsonValue;
    sourceSignalIds: readonly string[];
    commitmentId?: string;
    at?: string;
  }
): { state: NeuralHierarchyState; commitment: NeuralSkillCommitment } {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const at = input.at ?? new Date().toISOString();
  const active = state.active_skill_commitment;
  if (active && !["completed", "failed", "released"].includes(active.state)) {
    throw new Error(
      `Cannot replace active neural skill commitment ${active.commitment_id}`
    );
  }
  for (const signalId of new Set(input.sourceSignalIds)) {
    if (!state.signals[signalId]) {
      throw new Error(`Skill commitment references an unknown signal: ${signalId}`);
    }
  }
  const commitment = NeuralSkillCommitmentSchema.parse({
    commitment_id: input.commitmentId ?? randomUUID(),
    goal_epoch_id: input.goalEpochId,
    owner_node_id: input.ownerNodeId,
    skill: input.skill,
    state: "committed",
    established_world_revision: input.worldRevision,
    last_validated_world_revision: input.worldRevision,
    termination_contract: structuredClone(input.terminationContract),
    source_signal_ids: [...new Set(input.sourceSignalIds)],
    transition_signal_ids: [],
    transition_interrupt_ids: [],
    created_at: at,
    updated_at: at
  });
  state.active_skill_commitment = commitment;
  state.updated_at = at;
  return {
    state: NeuralHierarchyStateSchema.parse(state),
    commitment: structuredClone(commitment)
  };
}

const SKILL_COMMITMENT_TRANSITIONS: Readonly<Record<
  NeuralSkillCommitment["state"],
  ReadonlySet<NeuralSkillCommitment["state"]>
>> = {
  proposed: new Set(["committed", "released"]),
  committed: new Set(["executing", "failed", "released"]),
  executing: new Set(["completed", "failed", "released"]),
  completed: new Set(),
  failed: new Set(),
  released: new Set()
};

export function transitionNeuralSkillCommitment(
  stateInput: NeuralHierarchyState,
  input: {
    commitmentId: string;
    ownerNodeId: string;
    state: NeuralSkillCommitment["state"];
    worldRevision: number;
    sourceSignalIds?: readonly string[];
    sourceInterruptIds?: readonly string[];
    at?: string;
  }
): { state: NeuralHierarchyState; commitment: NeuralSkillCommitment } {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const commitment = state.active_skill_commitment;
  if (!commitment || commitment.commitment_id !== input.commitmentId) {
    throw new Error(`Unknown active neural skill commitment: ${input.commitmentId}`);
  }
  if (commitment.owner_node_id !== input.ownerNodeId) {
    throw new Error(
      `Only ${commitment.owner_node_id} may transition commitment ${input.commitmentId}`
    );
  }
  if (input.worldRevision < commitment.last_validated_world_revision) {
    throw new Error("A neural skill commitment cannot move to an older world revision");
  }
  if (input.state !== commitment.state
    && !SKILL_COMMITMENT_TRANSITIONS[commitment.state].has(input.state)) {
    throw new Error(
      `Invalid neural skill commitment transition: ${commitment.state} -> ${input.state}`
    );
  }
  for (const signalId of new Set(input.sourceSignalIds ?? [])) {
    if (!state.signals[signalId]) {
      throw new Error(`Commitment transition references unknown signal: ${signalId}`);
    }
  }
  for (const interruptId of new Set(input.sourceInterruptIds ?? [])) {
    const interrupt = state.safety_interrupts[interruptId];
    if (!interrupt) {
      throw new Error(
        `Commitment transition references unknown safety interrupt: ${interruptId}`
      );
    }
    if (interrupt.commitment_id !== null
      && interrupt.commitment_id !== commitment.commitment_id) {
      throw new Error(
        `Safety interrupt ${interruptId} belongs to another commitment`
      );
    }
  }
  const at = input.at ?? new Date().toISOString();
  commitment.state = input.state;
  commitment.transition_signal_ids = [...new Set([
    ...commitment.transition_signal_ids,
    ...(input.sourceSignalIds ?? [])
  ])].slice(-64);
  commitment.transition_interrupt_ids = [...new Set([
    ...commitment.transition_interrupt_ids,
    ...(input.sourceInterruptIds ?? [])
  ])].slice(-64);
  commitment.last_validated_world_revision = input.worldRevision;
  commitment.updated_at = at;
  if (["completed", "failed", "released"].includes(input.state)) {
    for (const certificate of Object.values(state.rollout_certificates)) {
      if (certificate.status !== "active"
        || certificate.commitment_id !== commitment.commitment_id) continue;
      certificate.status = "revoked";
      certificate.closed_at = at;
      certificate.close_reason = `skill_commitment_${input.state}`;
    }
  }
  state.updated_at = at;
  return {
    state: NeuralHierarchyStateSchema.parse(state),
    commitment: structuredClone(commitment)
  };
}

export function recordNeuralSafetyInterrupt(
  stateInput: NeuralHierarchyState,
  input: {
    kind: NeuralSafetyInterrupt["kind"];
    sourceNodeId: string;
    relayNodeId: string;
    targetNodeId: string;
    worldFrame: number;
    worldRevision: number;
    commitmentId: string | null;
    detail: JsonValue;
    interruptId?: string;
    at?: string;
  }
): { state: NeuralHierarchyState; interrupt: NeuralSafetyInterrupt } {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const existing = Object.values(state.safety_interrupts).find((interrupt) => (
    interrupt.status !== "resolved"
      && interrupt.kind === input.kind
      && interrupt.world_frame === input.worldFrame
      && interrupt.world_revision === input.worldRevision
  ));
  if (existing) return { state, interrupt: structuredClone(existing) };
  const at = input.at ?? new Date().toISOString();
  const interrupt = NeuralSafetyInterruptSchema.parse({
    interrupt_id: input.interruptId ?? randomUUID(),
    kind: input.kind,
    source_node_id: input.sourceNodeId,
    relay_node_id: input.relayNodeId,
    target_node_id: input.targetNodeId,
    world_frame: input.worldFrame,
    world_revision: input.worldRevision,
    commitment_id: input.commitmentId,
    status: "pending",
    detail: structuredClone(input.detail),
    created_at: at,
    acknowledged_at: null,
    acknowledged_by_node_id: null,
    resolved_at: null,
    resolved_world_revision: null
  });
  state.safety_interrupts[interrupt.interrupt_id] = interrupt;
  state.updated_at = at;
  return {
    state: NeuralHierarchyStateSchema.parse(state),
    interrupt: structuredClone(interrupt)
  };
}

export function acknowledgeNeuralSafetyInterrupt(
  stateInput: NeuralHierarchyState,
  input: {
    interruptId: string;
    acknowledgedByNodeId: string;
    worldRevision: number;
    at?: string;
  }
): {
  state: NeuralHierarchyState;
  interrupt: NeuralSafetyInterrupt;
  commitment: NeuralSkillCommitment | null;
} {
  let state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const interrupt = state.safety_interrupts[input.interruptId];
  if (!interrupt) throw new Error(`Unknown neural safety interrupt: ${input.interruptId}`);
  if (input.worldRevision < interrupt.world_revision) {
    throw new Error("A safety interrupt cannot be acknowledged at an older world revision");
  }
  if (interrupt.status === "resolved") {
    throw new Error(`Safety interrupt is already resolved: ${input.interruptId}`);
  }
  let commitment: NeuralSkillCommitment | null = null;
  const active = state.active_skill_commitment;
  if (active && !["completed", "failed", "released"].includes(active.state)) {
    if (interrupt.commitment_id !== active.commitment_id) {
      throw new Error("Pending safety interrupt does not own the active commitment");
    }
    const transitioned = transitionNeuralSkillCommitment(state, {
      commitmentId: active.commitment_id,
      ownerNodeId: active.owner_node_id,
      state: "failed",
      worldRevision: input.worldRevision,
      sourceInterruptIds: [interrupt.interrupt_id],
      ...(input.at === undefined ? {} : { at: input.at })
    });
    state = transitioned.state;
    commitment = transitioned.commitment;
  }
  const stored = state.safety_interrupts[input.interruptId]!;
  const at = input.at ?? new Date().toISOString();
  if (stored.status === "pending") {
    stored.status = "acknowledged";
    stored.acknowledged_at = at;
    stored.acknowledged_by_node_id = input.acknowledgedByNodeId;
    state.updated_at = at;
  }
  return {
    state: NeuralHierarchyStateSchema.parse(state),
    interrupt: structuredClone(stored),
    commitment
  };
}

export function resolveNeuralSafetyInterrupts(
  stateInput: NeuralHierarchyState,
  input: {
    worldRevision: number;
    interruptIds?: readonly string[];
    at?: string;
  }
): { state: NeuralHierarchyState; resolved: NeuralSafetyInterrupt[] } {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  const at = input.at ?? new Date().toISOString();
  const selected = input.interruptIds
    ? new Set(input.interruptIds)
    : null;
  const resolved: NeuralSafetyInterrupt[] = [];
  for (const interrupt of Object.values(state.safety_interrupts)) {
    if (interrupt.status !== "acknowledged") continue;
    if (selected && !selected.has(interrupt.interrupt_id)) continue;
    if (input.worldRevision < interrupt.world_revision) {
      throw new Error("A safety interrupt cannot resolve at an older world revision");
    }
    interrupt.status = "resolved";
    interrupt.resolved_at = at;
    interrupt.resolved_world_revision = input.worldRevision;
    resolved.push(structuredClone(interrupt));
  }
  if (resolved.length > 0) state.updated_at = at;
  return { state: NeuralHierarchyStateSchema.parse(state), resolved };
}

export function pendingNeuralSafetyInterrupts(
  stateInput: NeuralHierarchyState
): NeuralSafetyInterrupt[] {
  const state = NeuralHierarchyStateSchema.parse(stateInput);
  return Object.values(state.safety_interrupts)
    .filter((interrupt) => interrupt.status !== "resolved")
    .sort((left, right) => (
      right.world_revision - left.world_revision
        || right.created_at.localeCompare(left.created_at)
    ))
    .map((interrupt) => structuredClone(interrupt));
}

export function appendNeuralPredictionError(
  stateInput: NeuralHierarchyState,
  input: {
    sourceSignalId: string;
    observerNodeId: string;
    worldRevision: number;
    magnitude: number;
    tolerance: number;
    correctionScope: NeuralPredictionError["correction_scope"];
    detail: JsonValue;
    errorId?: string;
    at?: string;
  }
): { state: NeuralHierarchyState; error: NeuralPredictionError } {
  const state = NeuralHierarchyStateSchema.parse(structuredClone(stateInput));
  if (!state.signals[input.sourceSignalId]) {
    throw new Error(
      `Prediction error references an unknown neural signal: ${input.sourceSignalId}`
    );
  }
  const error = NeuralPredictionErrorSchema.parse({
    error_id: input.errorId ?? randomUUID(),
    source_signal_id: input.sourceSignalId,
    observer_node_id: input.observerNodeId,
    world_revision: input.worldRevision,
    magnitude: input.magnitude,
    tolerance: input.tolerance,
    correction_scope: input.correctionScope,
    corrected: input.magnitude <= input.tolerance,
    detail: structuredClone(input.detail),
    observed_at: input.at ?? new Date().toISOString()
  });
  state.prediction_errors.push(error);
  state.prediction_errors = state.prediction_errors.slice(-256);
  state.updated_at = error.observed_at;
  return {
    state: NeuralHierarchyStateSchema.parse(state),
    error: structuredClone(error)
  };
}

export function neuralPathwayDue(
  state: NeuralHierarchyState,
  pathway: NeuralPathway,
  now = Date.now()
): boolean {
  const cadence = state.pathway_cadences[pathway];
  if (cadence.last_dispatched_at === null) return true;
  return now - Date.parse(cadence.last_dispatched_at)
    >= cadence.minimum_interval_ms;
}

export function compactNeuralHierarchyState(
  stateInput: NeuralHierarchyState,
  worldRevision: number,
  retainedTerminalSignals = 128,
  liveInvocationIdsInput: readonly string[] = []
): NeuralHierarchyState {
  const state = structuredClone(stateInput);
  const liveInvocationIds = new Set(liveInvocationIdsInput);
  for (const signal of Object.values(state.signals)) {
    if (signal.status === "pending"
      && !liveInvocationIds.has(signal.invocation_id)
      && !liveInvocationIds.has(signal.parent_episode_id)
      && worldRevision > signal.world_revision + signal.ttl_revisions) {
      signal.status = "expired";
    }
  }
  const terminal = Object.values(state.signals)
    .filter((signal) => signal.status !== "pending")
    .sort((left, right) => right.sequence - left.sequence);
  const retain = new Set(terminal.slice(0, retainedTerminalSignals).map(
    (signal) => signal.signal_id
  ));
  const protectedSignalIds = new Set<string>([
    ...Object.values(state.signals)
      .filter((signal) => signal.status === "pending")
      .map((signal) => signal.signal_id),
    ...retain,
    ...(state.active_skill_commitment?.source_signal_ids ?? []),
    ...(state.active_skill_commitment?.transition_signal_ids ?? []),
    ...Object.values(state.rollout_certificates).flatMap((certificate) => [
      certificate.rollout_signal_id,
      certificate.predictive_signal_id
    ])
  ]);
  const causalFrontier = [...protectedSignalIds];
  while (causalFrontier.length > 0) {
    const signalId = causalFrontier.pop()!;
    if (protectedSignalIds.has(signalId) && !state.signals[signalId]) continue;
    const signal = state.signals[signalId];
    if (!signal) continue;
    for (const parentId of signal.causal_parent_ids) {
      if (protectedSignalIds.has(parentId)) continue;
      protectedSignalIds.add(parentId);
      causalFrontier.push(parentId);
    }
  }
  for (const signal of terminal.slice(retainedTerminalSignals)) {
    if (!protectedSignalIds.has(signal.signal_id)) {
      delete state.signals[signal.signal_id];
    }
  }
  const retainedLeaseIds = new Set<string>();
  for (const signal of Object.values(state.signals)) {
    if (signal.authority_lease_id) retainedLeaseIds.add(signal.authority_lease_id);
    if (signal.source_authority_lease_id) {
      retainedLeaseIds.add(signal.source_authority_lease_id);
    }
  }
  for (const lease of Object.values(state.authority_leases)) {
    if (lease.status === "active" || lease.status === "suspended") {
      retainedLeaseIds.add(lease.lease_id);
      for (const suspendedId of lease.suspended_lease_ids) retainedLeaseIds.add(suspendedId);
    }
  }
  const terminalLeases = Object.values(state.authority_leases)
    .filter((lease) => lease.status !== "active" && lease.status !== "suspended")
    .sort((left, right) => Date.parse(right.issued_at) - Date.parse(left.issued_at));
  for (const lease of terminalLeases.slice(128)) {
    if (!retainedLeaseIds.has(lease.lease_id)) delete state.authority_leases[lease.lease_id];
  }
  state.prediction_errors = state.prediction_errors.slice(-256);
  return NeuralHierarchyStateSchema.parse(state);
}
