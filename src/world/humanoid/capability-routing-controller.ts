import type { JsonValue } from "../../domain/schema.js";
import type { HumanoidLearnedPolicyCapability } from
  "../../domain/humanoid-policy.js";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import {
  G1HandCoordinationSchema
} from "./hand-coordination.js";
import {
  HumanoidHandPolicyAuthorityStateSchema
} from "./hand-policy-authority.js";
import type { HumanoidReference } from "./reference.js";
import {
  HumanoidPolicyAdmissionAssessmentSchema,
  HumanoidPolicyCapabilityEvidenceRegistry,
  HumanoidPolicyCapabilityObservationSchema,
  humanoidPolicyCapabilityObservation,
  type HumanoidPolicyAdmissionAssessment,
  type HumanoidPolicyCapabilityObservation
} from "./policy-capability-evidence.js";
import type {
  HumanoidControllerCapabilityEvidenceSummary,
  HumanoidControllerDescriptor,
  HumanoidControllerExecutionState,
  HumanoidControllerInferenceOptions,
  HumanoidControllerInferenceTrace,
  HumanoidControllerTensorTrace,
  HumanoidControllerSkillOutcome,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidHandSynergyCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

const ROUTING_STATE_PROTOCOL_V1 = "humanoid-controller-capability-routing-state-v1";
const ROUTING_STATE_PROTOCOL_V2 = "humanoid-controller-capability-routing-state-v2";
const ROUTING_STATE_PROTOCOL_V3 = "humanoid-controller-capability-routing-state-v3";
const ROUTING_STATE_PROTOCOL_V4 = "humanoid-controller-capability-routing-state-v4";
const MEMORY_BRIDGE_MAXIMUM_SECONDS = 4;
const MEMORY_BRIDGE_REQUIRED_STABLE_STEPS = 5;
const MEMORY_BRIDGE_ENTRY_RELEASE_SCORE = 2.5;
const MEMORY_BRIDGE_MAXIMUM_JOINT_RMS_ERROR = 0.22;
const MEMORY_BRIDGE_MAXIMUM_JOINT_VELOCITY = 0.8;
const REFERENCE_CONTROL_CAPABILITIES = [
  "balance",
  "locomotion",
  "joint_reference_tracking"
] as const satisfies readonly HumanoidLearnedPolicyCapability[];
type ControllerRoute = "primary" | "fallback" | "upper_body_overlay";

const FIRST_UPPER_BODY_JOINT_INDEX = HUMANOID_JOINT_NAMES.findIndex((joint) => (
  joint.includes("shoulder")
));

interface ControllerHandoff {
  fromImplementation: string;
  toImplementation: string;
  source: HumanoidJointPositionCommand;
  completedSteps: number;
  totalSteps: number;
}

interface ControllerAdmission {
  callId: string;
  assessment: HumanoidPolicyAdmissionAssessment;
  observation: HumanoidPolicyCapabilityObservation;
  primarySteps: number;
  fallbackSteps: number;
  upperBodyOverlaySteps: number;
  memoryBridgeSteps: number;
  transitionAttempted: boolean;
  recorded: boolean;
  memoryBridge: ControllerMemoryBridge | null;
}

interface ControllerMemoryBridge {
  protocol: "humanoid-policy-memory-bridge-v1";
  phase: "guiding" | "completed" | "timed_out" | "aborted";
  trigger: "entry_state_ood";
  targetJointPositions: number[];
  completedSteps: number;
  maximumSteps: number;
  stableSteps: number;
  requiredStableSteps: number;
  entryStateOodScore: number;
  jointPrototypeRmsError: number;
  maximumJointVelocity: number;
}

export function humanoidControllerNeedsReferenceFallback(
  descriptor: HumanoidControllerDescriptor
): boolean {
  const learnedPolicy = descriptor.learnedPolicy;
  return learnedPolicy !== undefined
    && descriptor.capabilityRouting === undefined
    && REFERENCE_CONTROL_CAPABILITIES.some((capability) => (
      !learnedPolicy.capabilities.includes(capability)
    ));
}

export class CapabilityRoutingHumanoidController
implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor;
  readonly #primary: HumanoidWholeBodyController;
  readonly #fallback: HumanoidWholeBodyController;
  readonly #handoffSteps: number;
  #active: ControllerRoute | null = null;
  #lastCommand: HumanoidJointPositionCommand | null = null;
  #handoff: ControllerHandoff | null = null;
  readonly #evidence = new HumanoidPolicyCapabilityEvidenceRegistry();
  readonly #admissions = new Map<string, ControllerAdmission>();
  #lastAdmissionCallId: string | null = null;
  #lastInferenceTrace: HumanoidControllerInferenceTrace | null = null;

  constructor(
    primary: HumanoidWholeBodyController,
    fallback: HumanoidWholeBodyController
  ) {
    assertRoutingPair(primary.descriptor, fallback.descriptor);
    this.#primary = primary;
    this.#fallback = fallback;
    this.#handoffSteps = controllerHandoffSteps(
      primary.descriptor,
      fallback.descriptor
    );
    this.descriptor = Object.freeze({
      ...primary.descriptor,
      capabilityRouting: Object.freeze({
        protocol: "humanoid-controller-capability-routing-v1" as const,
        strategy: "capability_evidence" as const,
        fallback: Object.freeze({
          mode: "reference_control" as const,
          implementation: fallback.descriptor.implementation
        })
      })
    });
  }

  executionState(): HumanoidControllerExecutionState {
    const active = this.#active ?? "primary";
    const admission = this.#lastAdmissionCallId
      ? this.#admissions.get(this.#lastAdmissionCallId)
      : undefined;
    return {
      protocol: "humanoid-controller-execution-v1",
      mode: active === "fallback"
        ? "reference_control"
        : active === "upper_body_overlay"
          ? "hybrid_control"
          : "learned_policy",
      activeImplementation: this.#routeImplementation(active),
      transition: this.#handoff
        ? {
            fromImplementation: this.#handoff.fromImplementation,
            toImplementation: this.#handoff.toImplementation,
            progress: this.#handoff.completedSteps / this.#handoff.totalSteps,
            durationSeconds:
              this.#handoff.totalSteps * this.descriptor.controlStepSeconds
          }
        : null,
      ...(admission
        ? {
            routing: {
              callId: admission.callId,
              route: active,
              assessment: structuredClone(admission.assessment),
              attribution: {
                primarySteps: admission.primarySteps,
                fallbackSteps: admission.fallbackSteps,
                upperBodyOverlaySteps: admission.upperBodyOverlaySteps,
                memoryBridgeSteps: admission.memoryBridgeSteps
              },
              memoryBridge: admission.memoryBridge
                ? memoryBridgeExecution(admission.memoryBridge)
                : null
            }
          }
        : {})
    };
  }

  capabilityEvidence(): readonly HumanoidControllerCapabilityEvidenceSummary[] {
    return this.#evidence.summaries();
  }

  inferenceTrace(): HumanoidControllerInferenceTrace | null {
    return this.#lastInferenceTrace
      ? structuredClone(this.#lastInferenceTrace)
      : null;
  }

  recordSkillOutcome(outcome: HumanoidControllerSkillOutcome): void {
    const admission = this.#admissions.get(outcome.identity.callId);
    if (!admission || admission.recorded) return;
    admission.recorded = true;
    if (admission.primarySteps === 0) return;
    const actualFamily = outcome.identity.runtimeKind === "semantic_skill"
      ? outcome.identity.skillId ? `semantic:${outcome.identity.skillId}` : null
      : outcome.identity.runtimeKind;
    if (actualFamily !== admission.assessment.skillFamily) {
      throw new Error("Humanoid policy outcome does not match its admitted Skill family");
    }
    const posterior = this.#evidence.record({
      implementation: this.#primary.descriptor.implementation,
      skillFamily: admission.assessment.skillFamily,
      observation: admission.observation,
      outcome: outcome.outcome === "succeeded" && admission.fallbackSteps > 0
        ? "interrupted"
        : outcome.outcome,
      transitionAttempted: admission.transitionAttempted
    });
    admission.assessment = {
      ...admission.assessment,
      posterior
    };
  }

  reset(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): void {
    this.#primary.reset(state, reference, options);
    this.#fallback.reset(state, reference, options);
    this.#active = this.#selectedRoute(state, reference, options, false);
    this.#lastCommand = null;
    this.#handoff = null;
    this.#lastInferenceTrace = null;
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): Promise<HumanoidJointPositionCommand> {
    const selected = this.#selectedRoute(state, reference, options, true);
    const previous = this.#active;
    const routeChanged = selected !== previous;
    const routeReference = this.#routeReference(selected, reference, options);
    if (routeChanged) {
      this.#resetNewRouteParticipants(
        previous,
        selected,
        state,
        routeReference,
        options
      );
    }
    const admission = options.taskCommand
      ? this.#admissions.get(options.taskCommand.identity.callId)
      : undefined;
    const target = await this.#inferRoute(
      selected,
      state,
      routeReference,
      options
    );
    this.#lastInferenceTrace = routingInferenceTrace(
      this.descriptor.implementation,
      selected,
      selected === "upper_body_overlay"
        ? [
            controllerTensorTrace(this.#primary, "primary"),
            controllerTensorTrace(this.#fallback, "fallback")
          ]
        : [controllerTensorTrace(
            this.#controller(selected),
            selected === "primary" ? "primary" : "fallback"
          )]
    );
    if (admission) {
      if (selected === "fallback") {
        if (admission.memoryBridge?.phase === "guiding") {
          admission.memoryBridgeSteps += 1;
        } else {
          admission.fallbackSteps += 1;
        }
      } else {
        admission.primarySteps += 1;
        if (selected === "upper_body_overlay") {
          admission.upperBodyOverlaySteps += 1;
        }
        if (previous === "fallback") admission.transitionAttempted = true;
      }
    }
    if (routeChanged) {
      this.#active = selected;
      this.#handoff = previous !== null
        && this.#lastCommand !== null
        && this.#handoffSteps > 1
        ? {
            fromImplementation:
              this.#routeImplementation(previous),
            toImplementation: this.#routeImplementation(selected),
            source: cloneCommand(this.#lastCommand),
            completedSteps: 0,
            totalSteps: this.#handoffSteps
          }
        : null;
    }
    const command = this.#handoff
      ? advanceHandoff(this.#handoff, target)
      : cloneCommand(target);
    if (this.#handoff
      && this.#handoff.completedSteps === this.#handoff.totalSteps) {
      this.#handoff = null;
    }
    this.#lastCommand = cloneCommand(command);
    return command;
  }

  advanceHistory(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): void {
    const selected = this.#selectedRoute(state, reference, options, false);
    if (selected !== this.#active) {
      throw new Error("Humanoid controller capability route changed within one control step");
    }
    if (selected === "upper_body_overlay") {
      this.#primary.advanceHistory(state, reference, options);
      this.#fallback.advanceHistory(state, reference, options);
      return;
    }
    this.#controller(selected).advanceHistory(
      state,
      this.#routeReference(selected, reference, options),
      options
    );
  }

  captureState(): HumanoidControllerState {
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: this.descriptor.implementation,
      payload: {
        protocol: ROUTING_STATE_PROTOCOL_V4,
        active: this.#active,
        primary: controllerStateJson(this.#primary.captureState()),
        fallback: controllerStateJson(this.#fallback.captureState()),
        last_command: this.#lastCommand
          ? controllerCommandJson(this.#lastCommand)
          : null,
        handoff: this.#handoff
          ? {
              from_implementation: this.#handoff.fromImplementation,
              to_implementation: this.#handoff.toImplementation,
              source: controllerCommandJson(this.#handoff.source),
              completed_steps: this.#handoff.completedSteps,
              total_steps: this.#handoff.totalSteps
            }
          : null,
        capability_evidence: this.#evidence.captureState(),
        admissions: [...this.#admissions.values()].map(controllerAdmissionJson),
        last_admission_call_id: this.#lastAdmissionCallId
      }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    if (state.protocol !== "humanoid-controller-state-v1"
      || state.version !== 1
      || state.implementation !== this.descriptor.implementation) {
      throw new Error("Invalid humanoid capability-routing controller state");
    }
    const routed = routingStatePayload(state.payload);
    if (!routed) {
      this.#primary.restoreState(state);
      this.#active = "primary";
      this.#lastCommand = null;
      this.#handoff = null;
      this.#evidence.restoreState({
        protocol: "humanoid-policy-capability-evidence-v1",
        entries: []
      });
      this.#admissions.clear();
      this.#lastAdmissionCallId = null;
      return;
    }
    if (routed.primary.implementation !== this.#primary.descriptor.implementation
      || routed.fallback.implementation !== this.#fallback.descriptor.implementation) {
      throw new Error("Humanoid capability-routing state belongs to another controller pair");
    }
    assertRestoredHandoff(
      routed,
      this.#primary.descriptor,
      this.#fallback.descriptor
    );
    this.#primary.restoreState(routed.primary);
    this.#fallback.restoreState(routed.fallback);
    this.#active = routed.active;
    this.#lastCommand = routed.lastCommand
      ? cloneCommand(routed.lastCommand)
      : null;
    this.#handoff = routed.handoff
      ? cloneHandoff(routed.handoff)
      : null;
    this.#evidence.restoreState(routed.capabilityEvidence ?? {
      protocol: "humanoid-policy-capability-evidence-v1",
      entries: []
    });
    this.#admissions.clear();
    for (const admission of routed.admissions) {
      const bridge = admission.memoryBridge;
      if (bridge && bridge.maximumSteps !== memoryBridgeMaximumSteps(
        this.descriptor.controlStepSeconds
      )) {
        throw new Error("Humanoid policy Memory Bridge timing changed across restore");
      }
      this.#admissions.set(admission.callId, cloneAdmission(admission));
    }
    this.#lastAdmissionCallId = routed.lastAdmissionCallId;
    this.#lastInferenceTrace = null;
  }

  async dispose(): Promise<void> {
    const results = await Promise.allSettled([
      this.#primary.dispose(),
      this.#fallback.dispose()
    ]);
    const failures = results.flatMap((result) => (
      result.status === "rejected" ? [result.reason] : []
    ));
    if (failures.length > 0) {
      throw new AggregateError(failures, "Unable to dispose humanoid controller pair");
    }
  }

  #selectedRoute(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions,
    advanceMemoryBridge: boolean
  ): ControllerRoute {
    const declared = declaredRoute(this.#primary.descriptor, reference, options);
    const taskCommand = options.taskCommand;
    if (!taskCommand) {
      this.#lastAdmissionCallId = null;
      return declared;
    }
    const callId = taskCommand.identity.callId;
    const existing = this.#admissions.get(callId);
    if (existing) {
      this.#lastAdmissionCallId = callId;
      if (advanceMemoryBridge && existing.memoryBridge?.phase === "guiding") {
        this.#advanceMemoryBridge(existing, state, taskCommand);
      }
      if (existing.memoryBridge?.phase === "guiding") return "fallback";
      return declared === "fallback" || !existing.assessment.admitted
        ? "fallback"
        : declared;
    }
    if (declared === "fallback") {
      this.#lastAdmissionCallId = null;
      return declared;
    }
    const observation = humanoidPolicyCapabilityObservation(state, taskCommand);
    const assessment = this.#evidence.assess({
      implementation: this.#primary.descriptor.implementation,
      state,
      taskCommand
    });
    this.#admissions.set(callId, {
      callId,
      assessment,
      observation,
      primarySteps: 0,
      fallbackSteps: 0,
      upperBodyOverlaySteps: 0,
      memoryBridgeSteps: 0,
      transitionAttempted: false,
      recorded: false,
      memoryBridge: assessment.reason === "entry_state_ood"
        && assessment.successfulEntryPrototype
        && memoryBridgeEligible(taskCommand)
        ? createMemoryBridge(
            assessment,
            state,
            this.descriptor.controlStepSeconds
          )
        : null
    });
    this.#lastAdmissionCallId = callId;
    while (this.#admissions.size > 32) {
      const oldest = this.#admissions.keys().next().value as string | undefined;
      if (!oldest || oldest === callId) break;
      this.#admissions.delete(oldest);
    }
    return assessment.admitted ? declared : "fallback";
  }

  #advanceMemoryBridge(
    admission: ControllerAdmission,
    state: HumanoidPolicyState,
    taskCommand: NonNullable<HumanoidControllerInferenceOptions["taskCommand"]>
  ): void {
    const bridge = admission.memoryBridge;
    if (!bridge || bridge.phase !== "guiding") return;
    bridge.completedSteps += 1;
    const observation = humanoidPolicyCapabilityObservation(state, taskCommand);
    const refreshed = this.#evidence.assess({
      implementation: this.#primary.descriptor.implementation,
      state,
      taskCommand
    });
    bridge.entryStateOodScore = refreshed.entryStateOodScore
      ?? bridge.entryStateOodScore;
    bridge.jointPrototypeRmsError = jointRmsError(
      observation.jointPositions,
      bridge.targetJointPositions
    );
    bridge.maximumJointVelocity = Math.max(
      0,
      ...Array.from(state.jointVelocities, Math.abs)
    );
    if (refreshed.reason === "command_ood"
      || refreshed.reason === "insufficient_success_posterior") {
      bridge.phase = "aborted";
      admission.assessment = refreshed;
      return;
    }
    const insideEntryRegion = bridge.entryStateOodScore
        <= MEMORY_BRIDGE_ENTRY_RELEASE_SCORE
      && bridge.jointPrototypeRmsError <= MEMORY_BRIDGE_MAXIMUM_JOINT_RMS_ERROR
      && bridge.maximumJointVelocity <= MEMORY_BRIDGE_MAXIMUM_JOINT_VELOCITY;
    bridge.stableSteps = insideEntryRegion ? bridge.stableSteps + 1 : 0;
    if (bridge.stableSteps >= bridge.requiredStableSteps) {
      bridge.phase = "completed";
      admission.observation = observation;
      admission.assessment = {
        ...refreshed,
        admitted: true,
        reason: "memory_bridge_completed"
      };
      return;
    }
    if (bridge.completedSteps >= bridge.maximumSteps) {
      bridge.phase = "timed_out";
      admission.assessment = {
        ...refreshed,
        admitted: false,
        reason: "memory_bridge_timeout"
      };
    }
  }

  #routeReference(
    route: ControllerRoute,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions
  ): HumanoidReference {
    if (route !== "fallback" || !options.taskCommand) return reference;
    const bridge = this.#admissions.get(options.taskCommand.identity.callId)
      ?.memoryBridge;
    return bridge?.phase === "guiding"
      ? memoryBridgeReference(reference, bridge.targetJointPositions)
      : reference;
  }

  #controller(route: Exclude<ControllerRoute, "upper_body_overlay">):
  HumanoidWholeBodyController {
    return route === "primary" ? this.#primary : this.#fallback;
  }

  #routeImplementation(route: ControllerRoute): string {
    if (route === "upper_body_overlay") {
      return `${this.#primary.descriptor.implementation}+${this.#fallback.descriptor.implementation}`;
    }
    return this.#controller(route).descriptor.implementation;
  }

  #resetNewRouteParticipants(
    previous: ControllerRoute | null,
    selected: ControllerRoute,
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions
  ): void {
    if (selected !== "fallback" && previous !== "primary"
      && previous !== "upper_body_overlay") {
      this.#primary.reset(state, reference, options);
    }
    if (selected !== "primary" && previous !== "fallback"
      && previous !== "upper_body_overlay") {
      this.#fallback.reset(state, reference, options);
    }
  }

  async #inferRoute(
    route: ControllerRoute,
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions
  ): Promise<HumanoidJointPositionCommand> {
    if (route !== "upper_body_overlay") {
      return this.#controller(route).infer(state, reference, options);
    }
    const primary = await this.#primary.infer(state, reference, options);
    const fallback = await this.#fallback.infer(state, reference, options);
    return upperBodyOverlayCommand(primary, fallback, reference);
  }
}

function declaredRoute(
  descriptor: HumanoidControllerDescriptor,
  reference: HumanoidReference,
  options: HumanoidControllerInferenceOptions
): ControllerRoute {
  const available = new Set(descriptor.learnedPolicy?.capabilities ?? []);
  const requested = options.taskCommand?.requestedCapabilities ?? [];
  const missingRequestedCapabilities = requested.filter((capability) => (
    !available.has(capability)
  ));
  const activeTrackingIndexes = Array.from(
    reference.jointTrackingWeights,
    (weight, index) => weight > 0 ? index : -1
  ).filter((index) => index >= 0);
  const unsupportedJointReference = !available.has("joint_reference_tracking")
    && activeTrackingIndexes.length > 0;
  const missingBeyondJointTracking = missingRequestedCapabilities.some(
    (capability) => capability !== "joint_reference_tracking"
  );
  if (missingBeyondJointTracking) return "fallback";
  if (!unsupportedJointReference) return "primary";
  return activeTrackingIndexes.every((index) => (
    index >= FIRST_UPPER_BODY_JOINT_INDEX
  ))
    ? "upper_body_overlay"
    : "fallback";
}

function controllerTensorTrace(
  controller: HumanoidWholeBodyController,
  role: HumanoidControllerTensorTrace["role"]
): HumanoidControllerTensorTrace | null {
  const trace = controller.inferenceTrace?.();
  const component = trace?.components[0];
  return component
    ? {
        ...structuredClone(component),
        role,
        implementation: controller.descriptor.implementation
      }
    : null;
}

function routingInferenceTrace(
  implementation: string,
  route: Exclude<HumanoidControllerInferenceTrace["route"], "direct">,
  components: Array<HumanoidControllerTensorTrace | null>
): HumanoidControllerInferenceTrace {
  return {
    protocol: "humanoid-controller-inference-trace-v1",
    implementation,
    route,
    components: components.filter(
      (component): component is HumanoidControllerTensorTrace => component !== null
    )
  };
}

function createMemoryBridge(
  assessment: HumanoidPolicyAdmissionAssessment,
  state: HumanoidPolicyState,
  controlStepSeconds: number
): ControllerMemoryBridge {
  const targetJointPositions = assessment.successfulEntryPrototype;
  if (!targetJointPositions || assessment.entryStateOodScore === null) {
    throw new Error("Memory Bridge requires a learned entry region and prototype");
  }
  const jointPositions = Array.from(state.jointPositions);
  return {
    protocol: "humanoid-policy-memory-bridge-v1",
    phase: "guiding",
    trigger: "entry_state_ood",
    targetJointPositions: [...targetJointPositions],
    completedSteps: 0,
    maximumSteps: memoryBridgeMaximumSteps(controlStepSeconds),
    stableSteps: 0,
    requiredStableSteps: MEMORY_BRIDGE_REQUIRED_STABLE_STEPS,
    entryStateOodScore: assessment.entryStateOodScore,
    jointPrototypeRmsError: jointRmsError(
      jointPositions,
      targetJointPositions
    ),
    maximumJointVelocity: Math.max(
      0,
      ...Array.from(state.jointVelocities, Math.abs)
    )
  };
}

function memoryBridgeMaximumSteps(controlStepSeconds: number): number {
  return Math.max(
    MEMORY_BRIDGE_REQUIRED_STABLE_STEPS,
    Math.ceil(MEMORY_BRIDGE_MAXIMUM_SECONDS / controlStepSeconds)
  );
}

function memoryBridgeEligible(
  taskCommand: NonNullable<HumanoidControllerInferenceOptions["taskCommand"]>
): boolean {
  const capabilities = new Set(taskCommand.requestedCapabilities);
  return taskCommand.command.grasps.length === 0
    && !capabilities.has("contact_rich_manipulation")
    && !capabilities.has("bimanual_manipulation");
}

function memoryBridgeReference(
  reference: HumanoidReference,
  targetJointPositions: readonly number[]
): HumanoidReference {
  if (targetJointPositions.length !== HUMANOID_JOINT_NAMES.length
    || targetJointPositions.some((position) => !Number.isFinite(position))) {
    throw new Error("Memory Bridge joint prototype is invalid");
  }
  return {
    jointPositions: Float64Array.from(targetJointPositions),
    jointVelocities: new Float64Array(HUMANOID_JOINT_NAMES.length),
    jointTrackingWeights: new Float64Array(HUMANOID_JOINT_NAMES.length).fill(1),
    rootVelocity: [0, 0],
    rootYawVelocity: 0,
    rootHeight: reference.rootHeight,
    rootRoll: 0,
    rootPitch: 0
  };
}

function memoryBridgeExecution(
  bridge: ControllerMemoryBridge
): NonNullable<
  NonNullable<HumanoidControllerExecutionState["routing"]>["memoryBridge"]
> {
  return {
    protocol: bridge.protocol,
    phase: bridge.phase,
    trigger: bridge.trigger,
    completedSteps: bridge.completedSteps,
    maximumSteps: bridge.maximumSteps,
    stableSteps: bridge.stableSteps,
    requiredStableSteps: bridge.requiredStableSteps,
    progress: bridge.phase === "completed" || bridge.phase === "timed_out"
      ? 1
      : bridge.completedSteps / bridge.maximumSteps,
    entryStateOodScore: bridge.entryStateOodScore,
    jointPrototypeRmsError: bridge.jointPrototypeRmsError,
    maximumJointVelocity: bridge.maximumJointVelocity
  };
}

function jointRmsError(
  actual: readonly number[],
  target: readonly number[]
): number {
  if (actual.length !== target.length || actual.length === 0) {
    throw new Error("Memory Bridge joint vectors do not match");
  }
  return Math.sqrt(actual.reduce((sum, position, index) => {
    const error = position - target[index]!;
    return sum + error * error;
  }, 0) / actual.length);
}

function assertRoutingPair(
  primary: HumanoidControllerDescriptor,
  fallback: HumanoidControllerDescriptor
): void {
  if (!humanoidControllerNeedsReferenceFallback(primary)) {
    throw new Error("Primary controller does not require reference-control routing");
  }
  if (primary.implementation === fallback.implementation) {
    throw new Error("Humanoid capability routing requires distinct controller implementations");
  }
  if (primary.actuation !== fallback.actuation
    || primary.controlStepSeconds !== fallback.controlStepSeconds
    || primary.physicsStepSeconds !== fallback.physicsStepSeconds) {
    throw new Error("Humanoid capability-routing controllers must use identical timing and actuation");
  }
  const fallbackCapabilities = new Set(fallback.learnedPolicy?.capabilities ?? []);
  if (REFERENCE_CONTROL_CAPABILITIES.some((capability) => (
    !fallbackCapabilities.has(capability)
  ))) {
    throw new Error("Reference fallback must support balance, locomotion, and joint tracking");
  }
}

function controllerStateJson(state: HumanoidControllerState): JsonValue {
  return {
    protocol: state.protocol,
    version: state.version,
    implementation: state.implementation,
    payload: structuredClone(state.payload)
  };
}

interface RoutingStatePayload {
  active: ControllerRoute | null;
  primary: HumanoidControllerState;
  fallback: HumanoidControllerState;
  lastCommand: HumanoidJointPositionCommand | null;
  handoff: ControllerHandoff | null;
  capabilityEvidence: JsonValue | null;
  admissions: ControllerAdmission[];
  lastAdmissionCallId: string | null;
}

function routingStatePayload(value: JsonValue): RoutingStatePayload | null {
  if (!isRecord(value)
    || (value.protocol !== ROUTING_STATE_PROTOCOL_V1
      && value.protocol !== ROUTING_STATE_PROTOCOL_V2
      && value.protocol !== ROUTING_STATE_PROTOCOL_V3
      && value.protocol !== ROUTING_STATE_PROTOCOL_V4)) return null;
  if (value.active !== null
    && value.active !== "primary"
    && value.active !== "fallback"
    && (value.protocol !== ROUTING_STATE_PROTOCOL_V3
      && value.protocol !== ROUTING_STATE_PROTOCOL_V4
      || value.active !== "upper_body_overlay")) {
    throw new Error("Invalid humanoid capability-routing active branch");
  }
  const primary = controllerStateFrom(value.primary);
  const fallback = controllerStateFrom(value.fallback);
  if (!primary || !fallback) {
    throw new Error("Invalid humanoid capability-routing nested controller state");
  }
  if (value.protocol === ROUTING_STATE_PROTOCOL_V1) {
    return {
      active: value.active,
      primary,
      fallback,
      lastCommand: null,
      handoff: null,
      capabilityEvidence: null,
      admissions: [],
      lastAdmissionCallId: null
    };
  }
  const lastCommand = value.last_command === null
    ? null
    : controllerCommandFrom(value.last_command);
  const handoff = value.handoff === null
    ? null
    : controllerHandoffFrom(value.handoff);
  if ((value.last_command !== null && !lastCommand)
    || (value.handoff !== null && !handoff)) {
    throw new Error("Invalid humanoid capability-routing handoff state");
  }
  if (value.protocol !== ROUTING_STATE_PROTOCOL_V4) return {
    active: value.active,
    primary,
    fallback,
    lastCommand,
    handoff,
    capabilityEvidence: null,
    admissions: [],
    lastAdmissionCallId: null
  };
  if (value.capability_evidence === undefined
    || !Array.isArray(value.admissions)
    || (value.last_admission_call_id !== null
      && typeof value.last_admission_call_id !== "string")) {
    throw new Error("Invalid humanoid capability-routing evidence state");
  }
  const admissions = value.admissions.map(controllerAdmissionFrom);
  const callIds = new Set(admissions.map(({ callId }) => callId));
  if (callIds.size !== admissions.length
    || (value.last_admission_call_id !== null
      && !callIds.has(value.last_admission_call_id))) {
    throw new Error("Invalid humanoid capability-routing admission identity");
  }
  return {
    active: value.active,
    primary,
    fallback,
    lastCommand,
    handoff,
    capabilityEvidence: structuredClone(value.capability_evidence),
    admissions,
    lastAdmissionCallId: value.last_admission_call_id
  };
}

function controllerCommandJson(command: HumanoidJointPositionCommand): JsonValue {
  return {
    kind: command.kind,
    positions: [...command.positions],
    stiffness: [...command.stiffness],
    damping: [...command.damping],
    ...(command.handSynergy
      ? {
          hand_synergy: {
            protocol: command.handSynergy.protocol,
            authority: structuredClone(command.handSynergy.authority),
            action: [...command.handSynergy.action],
            coordination: structuredClone(command.handSynergy.coordination),
            maximum_closing_joint_lead_radians:
              command.handSynergy.maximumClosingJointLeadRadians
          }
        }
      : {})
  };
}

function controllerAdmissionJson(admission: ControllerAdmission): JsonValue {
  const assessment = admission.assessment;
  return {
    call_id: admission.callId,
    assessment: {
      protocol: assessment.protocol,
      implementation: assessment.implementation,
      skillFamily: assessment.skillFamily,
      admitted: assessment.admitted,
      reason: assessment.reason,
      coldStart: assessment.coldStart,
      entryStateOodScore: assessment.entryStateOodScore,
      commandOodScore: assessment.commandOodScore,
      posterior: {
        outcomes: assessment.posterior.outcomes,
        successes: assessment.posterior.successes,
        failures: assessment.posterior.failures,
        posteriorMean: assessment.posterior.posteriorMean,
        lowerBound: assessment.posterior.lowerBound,
        upperBound: assessment.posterior.upperBound,
        recentSuccessRate: assessment.posterior.recentSuccessRate,
        transitionAttempts: assessment.posterior.transitionAttempts,
        transitionSuccesses: assessment.posterior.transitionSuccesses
      },
      successfulEntryPrototype: assessment.successfulEntryPrototype
        ? [...assessment.successfulEntryPrototype]
        : null
    },
    observation: {
      state: [...admission.observation.state],
      command: [...admission.observation.command],
      jointPositions: [...admission.observation.jointPositions]
    },
    primary_steps: admission.primarySteps,
    fallback_steps: admission.fallbackSteps,
    upper_body_overlay_steps: admission.upperBodyOverlaySteps,
    memory_bridge_steps: admission.memoryBridgeSteps,
    transition_attempted: admission.transitionAttempted,
    recorded: admission.recorded,
    memory_bridge: admission.memoryBridge
      ? controllerMemoryBridgeJson(admission.memoryBridge)
      : null
  };
}

function controllerAdmissionFrom(value: JsonValue): ControllerAdmission {
  if (!isRecord(value)
    || typeof value.call_id !== "string"
    || value.call_id.length === 0
    || !Number.isSafeInteger(value.primary_steps)
    || typeof value.primary_steps !== "number"
    || value.primary_steps < 0
    || !Number.isSafeInteger(value.fallback_steps)
    || typeof value.fallback_steps !== "number"
    || value.fallback_steps < 0
    || !Number.isSafeInteger(value.upper_body_overlay_steps)
    || typeof value.upper_body_overlay_steps !== "number"
    || value.upper_body_overlay_steps < 0
    || value.upper_body_overlay_steps > value.primary_steps
    || (value.memory_bridge_steps !== undefined
      && (!Number.isSafeInteger(value.memory_bridge_steps)
        || typeof value.memory_bridge_steps !== "number"
        || value.memory_bridge_steps < 0))
    || typeof value.transition_attempted !== "boolean"
    || typeof value.recorded !== "boolean") {
    throw new Error("Invalid humanoid capability-routing admission state");
  }
  const memoryBridge = value.memory_bridge === undefined
    || value.memory_bridge === null
    ? null
    : controllerMemoryBridgeFrom(value.memory_bridge);
  const admission: ControllerAdmission = {
    callId: value.call_id,
    assessment: HumanoidPolicyAdmissionAssessmentSchema.parse(value.assessment),
    observation: HumanoidPolicyCapabilityObservationSchema.parse(value.observation),
    primarySteps: value.primary_steps,
    fallbackSteps: value.fallback_steps,
    upperBodyOverlaySteps: value.upper_body_overlay_steps,
    memoryBridgeSteps: typeof value.memory_bridge_steps === "number"
      ? value.memory_bridge_steps
      : 0,
    transitionAttempted: value.transition_attempted,
    recorded: value.recorded,
    memoryBridge
  };
  assertAdmissionMemoryBridge(admission);
  return admission;
}

function cloneAdmission(admission: ControllerAdmission): ControllerAdmission {
  return {
    callId: admission.callId,
    assessment: structuredClone(admission.assessment),
    observation: structuredClone(admission.observation),
    primarySteps: admission.primarySteps,
    fallbackSteps: admission.fallbackSteps,
    upperBodyOverlaySteps: admission.upperBodyOverlaySteps,
    memoryBridgeSteps: admission.memoryBridgeSteps,
    transitionAttempted: admission.transitionAttempted,
    recorded: admission.recorded,
    memoryBridge: admission.memoryBridge
      ? cloneMemoryBridge(admission.memoryBridge)
      : null
  };
}

function controllerMemoryBridgeJson(bridge: ControllerMemoryBridge): JsonValue {
  return {
    protocol: bridge.protocol,
    phase: bridge.phase,
    trigger: bridge.trigger,
    target_joint_positions: [...bridge.targetJointPositions],
    completed_steps: bridge.completedSteps,
    maximum_steps: bridge.maximumSteps,
    stable_steps: bridge.stableSteps,
    required_stable_steps: bridge.requiredStableSteps,
    entry_state_ood_score: bridge.entryStateOodScore,
    joint_prototype_rms_error: bridge.jointPrototypeRmsError,
    maximum_joint_velocity: bridge.maximumJointVelocity
  };
}

function controllerMemoryBridgeFrom(value: JsonValue): ControllerMemoryBridge {
  if (!isRecord(value)
    || value.protocol !== "humanoid-policy-memory-bridge-v1"
    || (value.phase !== "guiding" && value.phase !== "completed"
      && value.phase !== "timed_out" && value.phase !== "aborted")
    || value.trigger !== "entry_state_ood"
    || !Number.isSafeInteger(value.completed_steps)
    || typeof value.completed_steps !== "number"
    || value.completed_steps < 0
    || !Number.isSafeInteger(value.maximum_steps)
    || typeof value.maximum_steps !== "number"
    || value.maximum_steps < MEMORY_BRIDGE_REQUIRED_STABLE_STEPS
    || value.completed_steps > value.maximum_steps
    || !Number.isSafeInteger(value.stable_steps)
    || typeof value.stable_steps !== "number"
    || value.stable_steps < 0
    || !Number.isSafeInteger(value.required_stable_steps)
    || value.required_stable_steps !== MEMORY_BRIDGE_REQUIRED_STABLE_STEPS
    || value.stable_steps > value.required_stable_steps
    || typeof value.entry_state_ood_score !== "number"
    || !Number.isFinite(value.entry_state_ood_score)
    || value.entry_state_ood_score < 0
    || typeof value.joint_prototype_rms_error !== "number"
    || !Number.isFinite(value.joint_prototype_rms_error)
    || value.joint_prototype_rms_error < 0
    || typeof value.maximum_joint_velocity !== "number"
    || !Number.isFinite(value.maximum_joint_velocity)
    || value.maximum_joint_velocity < 0) {
    throw new Error("Invalid humanoid policy Memory Bridge state");
  }
  const targetJointPositions = finiteVector(value.target_joint_positions, false);
  if (!targetJointPositions) {
    throw new Error("Invalid humanoid policy Memory Bridge joint prototype");
  }
  return {
    protocol: value.protocol,
    phase: value.phase,
    trigger: value.trigger,
    targetJointPositions,
    completedSteps: value.completed_steps,
    maximumSteps: value.maximum_steps,
    stableSteps: value.stable_steps,
    requiredStableSteps: value.required_stable_steps,
    entryStateOodScore: value.entry_state_ood_score,
    jointPrototypeRmsError: value.joint_prototype_rms_error,
    maximumJointVelocity: value.maximum_joint_velocity
  };
}

function cloneMemoryBridge(bridge: ControllerMemoryBridge): ControllerMemoryBridge {
  return {
    ...bridge,
    targetJointPositions: [...bridge.targetJointPositions]
  };
}

function assertAdmissionMemoryBridge(admission: ControllerAdmission): void {
  const bridge = admission.memoryBridge;
  if (!bridge) {
    if (admission.memoryBridgeSteps !== 0) {
      throw new Error("Humanoid policy admission has Memory Bridge steps without a bridge");
    }
    return;
  }
  const prototype = admission.assessment.successfulEntryPrototype;
  if (!prototype || prototype.some((value, index) => (
    value !== bridge.targetJointPositions[index]
  ))) {
    throw new Error("Humanoid policy Memory Bridge does not match its entry prototype");
  }
  const coherent = bridge.phase === "guiding"
    ? !admission.assessment.admitted
      && admission.assessment.reason === "entry_state_ood"
    : bridge.phase === "completed"
      ? admission.assessment.admitted
        && admission.assessment.reason === "memory_bridge_completed"
      : bridge.phase === "timed_out"
        ? !admission.assessment.admitted
          && admission.assessment.reason === "memory_bridge_timeout"
        : !admission.assessment.admitted
          && (admission.assessment.reason === "command_ood"
            || admission.assessment.reason === "insufficient_success_posterior");
  if (!coherent) {
    throw new Error("Humanoid policy Memory Bridge phase conflicts with admission");
  }
}

function controllerCommandFrom(
  value: JsonValue | undefined
): HumanoidJointPositionCommand | null {
  if (!isRecord(value) || value.kind !== "joint_position_pd") return null;
  const positions = finiteVector(value.positions, false);
  const stiffness = finiteVector(value.stiffness, true);
  const damping = finiteVector(value.damping, true);
  if (!positions || !stiffness || !damping) return null;
  const handSynergy = handSynergyCommandFrom(value.hand_synergy);
  if (value.hand_synergy !== undefined && !handSynergy) return null;
  return {
    kind: "joint_position_pd",
    positions: Float64Array.from(positions),
    stiffness: Float64Array.from(stiffness),
    damping: Float64Array.from(damping),
    ...(handSynergy ? { handSynergy } : {})
  };
}

function handSynergyCommandFrom(
  value: JsonValue | undefined
): HumanoidHandSynergyCommand | null {
  if (!isRecord(value)
    || value.protocol !== "humanoid-authorized-hand-synergy-command-v1") {
    return null;
  }
  const authority = HumanoidHandPolicyAuthorityStateSchema.safeParse(
    value.authority
  );
  const coordination = G1HandCoordinationSchema.safeParse(value.coordination);
  if (!authority.success || !coordination.success
    || value.maximum_closing_joint_lead_radians !== 0.25
    || !Array.isArray(value.action)
    || value.action.length !== 8
    || value.action.some((entry) => typeof entry !== "number"
      || !Number.isFinite(entry) || entry < -1 || entry > 1)) {
    return null;
  }
  return {
    protocol: "humanoid-authorized-hand-synergy-command-v1",
    authority: authority.data,
    action: Float64Array.from(value.action as number[]),
    coordination: coordination.data,
    maximumClosingJointLeadRadians: 0.25
  };
}

function controllerHandoffFrom(value: JsonValue | undefined): ControllerHandoff | null {
  if (!isRecord(value)
    || typeof value.from_implementation !== "string"
    || value.from_implementation.length === 0
    || typeof value.to_implementation !== "string"
    || value.to_implementation.length === 0
    || !Number.isSafeInteger(value.completed_steps)
    || typeof value.completed_steps !== "number"
    || value.completed_steps <= 0
    || !Number.isSafeInteger(value.total_steps)
    || typeof value.total_steps !== "number"
    || value.total_steps <= 1
    || value.completed_steps >= value.total_steps) return null;
  const source = controllerCommandFrom(value.source);
  return source
    ? {
        fromImplementation: value.from_implementation,
        toImplementation: value.to_implementation,
        source,
        completedSteps: value.completed_steps,
        totalSteps: value.total_steps
      }
    : null;
}

function finiteVector(
  value: JsonValue | undefined,
  nonnegative: boolean
): number[] | null {
  if (!Array.isArray(value)
    || value.length !== HUMANOID_JOINT_NAMES.length
    || value.some((entry) => typeof entry !== "number"
      || !Number.isFinite(entry)
      || (nonnegative && entry < 0))) return null;
  return value as number[];
}

function assertRestoredHandoff(
  state: RoutingStatePayload,
  primary: HumanoidControllerDescriptor,
  fallback: HumanoidControllerDescriptor
): void {
  if (state.lastCommand !== null && state.active === null) {
    throw new Error("Humanoid capability-routing command has no active controller");
  }
  if (!state.handoff) return;
  const implementations = {
    primary: primary.implementation,
    fallback: fallback.implementation,
    upper_body_overlay: `${primary.implementation}+${fallback.implementation}`
  } as const;
  const activeImplementation = state.active === null
    ? null
    : implementations[state.active];
  const validSources = Object.entries(implementations)
    .filter(([route]) => route !== state.active)
    .map(([, implementation]) => implementation);
  if (!activeImplementation
    || !state.lastCommand
    || state.handoff.toImplementation !== activeImplementation
    || !validSources.includes(state.handoff.fromImplementation)
    || state.handoff.totalSteps !== controllerHandoffSteps(primary, fallback)) {
    throw new Error("Humanoid capability-routing handoff does not match the controller pair");
  }
}

function upperBodyOverlayCommand(
  primary: HumanoidJointPositionCommand,
  fallback: HumanoidJointPositionCommand,
  reference: HumanoidReference
): HumanoidJointPositionCommand {
  assertCommandDimensions(primary);
  assertCommandDimensions(fallback);
  const blend = (
    primaryValues: Float64Array,
    fallbackValues: Float64Array
  ) => Float64Array.from(primaryValues, (value, index) => {
    const weight = index >= FIRST_UPPER_BODY_JOINT_INDEX
      ? reference.jointTrackingWeights[index]!
      : 0;
    return value + (fallbackValues[index]! - value) * weight;
  });
  return {
    kind: "joint_position_pd",
    positions: blend(primary.positions, fallback.positions),
    stiffness: blend(primary.stiffness, fallback.stiffness),
    damping: blend(primary.damping, fallback.damping),
    ...(primary.handSynergy
      ? { handSynergy: cloneHandSynergy(primary.handSynergy) }
      : {})
  };
}

function controllerHandoffSteps(
  primary: HumanoidControllerDescriptor,
  fallback: HumanoidControllerDescriptor
): number {
  const responseHorizonSeconds = Math.max(
    primary.commandResponseHorizonSeconds ?? 0,
    fallback.commandResponseHorizonSeconds ?? 0
  );
  return Math.max(
    1,
    Math.ceil(responseHorizonSeconds / primary.controlStepSeconds - 1e-12)
  );
}

function advanceHandoff(
  handoff: ControllerHandoff,
  target: HumanoidJointPositionCommand
): HumanoidJointPositionCommand {
  assertCommandDimensions(target);
  const nextStep = handoff.completedSteps + 1;
  const linearProgress = Math.min(1, nextStep / handoff.totalSteps);
  const amount = linearProgress * linearProgress * (3 - 2 * linearProgress);
  handoff.completedSteps = nextStep;
  return {
    kind: "joint_position_pd",
    positions: blendVector(handoff.source.positions, target.positions, amount),
    stiffness: blendVector(handoff.source.stiffness, target.stiffness, amount),
    damping: blendVector(handoff.source.damping, target.damping, amount),
    ...(target.handSynergy
      ? { handSynergy: cloneHandSynergy(target.handSynergy) }
      : {})
  };
}

function blendVector(
  source: Float64Array,
  target: Float64Array,
  amount: number
): Float64Array {
  return Float64Array.from(source, (value, index) => (
    value + (target[index]! - value) * amount
  ));
}

function assertCommandDimensions(command: HumanoidJointPositionCommand): void {
  if (command.positions.length !== HUMANOID_JOINT_NAMES.length
    || command.stiffness.length !== HUMANOID_JOINT_NAMES.length
    || command.damping.length !== HUMANOID_JOINT_NAMES.length) {
    throw new Error("Humanoid capability-routing command has an invalid joint count");
  }
  if (command.handSynergy) {
    const hand = command.handSynergy;
    if (hand.protocol !== "humanoid-authorized-hand-synergy-command-v1"
      || !HumanoidHandPolicyAuthorityStateSchema.safeParse(hand.authority).success
      || !G1HandCoordinationSchema.safeParse(hand.coordination).success
      || hand.maximumClosingJointLeadRadians !== 0.25
      || hand.action.length !== 8
      || Array.from(hand.action).some((value) => (
        !Number.isFinite(value) || value < -1 || value > 1
      ))) {
      throw new Error("Humanoid capability-routing command has invalid hand synergy");
    }
  }
}

function cloneCommand(
  command: HumanoidJointPositionCommand
): HumanoidJointPositionCommand {
  assertCommandDimensions(command);
  return {
    kind: "joint_position_pd",
    positions: command.positions.slice(),
    stiffness: command.stiffness.slice(),
    damping: command.damping.slice(),
    ...(command.handSynergy
      ? { handSynergy: cloneHandSynergy(command.handSynergy) }
      : {})
  };
}

function cloneHandSynergy(
  command: HumanoidHandSynergyCommand
): HumanoidHandSynergyCommand {
  return {
    protocol: command.protocol,
    authority: structuredClone(command.authority),
    action: command.action.slice(),
    coordination: structuredClone(command.coordination),
    maximumClosingJointLeadRadians: command.maximumClosingJointLeadRadians
  };
}

function cloneHandoff(handoff: ControllerHandoff): ControllerHandoff {
  return {
    ...handoff,
    source: cloneCommand(handoff.source)
  };
}

function controllerStateFrom(value: JsonValue | undefined): HumanoidControllerState | null {
  if (!isRecord(value)
    || value.protocol !== "humanoid-controller-state-v1"
    || value.version !== 1
    || typeof value.implementation !== "string"
    || value.implementation.length === 0
    || !("payload" in value)) {
    return null;
  }
  return {
    protocol: "humanoid-controller-state-v1",
    version: 1,
    implementation: value.implementation,
    payload: structuredClone(value.payload as JsonValue)
  };
}

function isRecord(value: JsonValue | undefined): value is Record<string, JsonValue> {
  return value !== null && value !== undefined
    && typeof value === "object" && !Array.isArray(value);
}
