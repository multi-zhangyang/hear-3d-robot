import type { JsonValue } from "../../domain/schema.js";
import type { HumanoidLearnedPolicyCapability } from
  "../../domain/humanoid-policy.js";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import type { HumanoidReference } from "./reference.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerExecutionState,
  HumanoidControllerInferenceOptions,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

const ROUTING_STATE_PROTOCOL_V1 = "humanoid-controller-capability-routing-state-v1";
const ROUTING_STATE_PROTOCOL_V2 = "humanoid-controller-capability-routing-state-v2";
const REFERENCE_CONTROL_CAPABILITIES = [
  "balance",
  "locomotion",
  "joint_reference_tracking"
] as const satisfies readonly HumanoidLearnedPolicyCapability[];
type ControllerBranch = "primary" | "fallback";

interface ControllerHandoff {
  fromImplementation: string;
  toImplementation: string;
  source: HumanoidJointPositionCommand;
  completedSteps: number;
  totalSteps: number;
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
  #active: ControllerBranch | null = null;
  #lastCommand: HumanoidJointPositionCommand | null = null;
  #handoff: ControllerHandoff | null = null;

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
        strategy: "declared_capabilities" as const,
        fallback: Object.freeze({
          mode: "reference_control" as const,
          implementation: fallback.descriptor.implementation
        })
      })
    });
  }

  executionState(): HumanoidControllerExecutionState {
    const active = this.#active ?? "primary";
    const controller = this.#controller(active);
    return {
      protocol: "humanoid-controller-execution-v1",
      mode: active === "fallback" ? "reference_control" : "learned_policy",
      activeImplementation: controller.descriptor.implementation,
      transition: this.#handoff
        ? {
            fromImplementation: this.#handoff.fromImplementation,
            toImplementation: this.#handoff.toImplementation,
            progress: this.#handoff.completedSteps / this.#handoff.totalSteps,
            durationSeconds:
              this.#handoff.totalSteps * this.descriptor.controlStepSeconds
          }
        : null
    };
  }

  reset(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): void {
    this.#primary.reset(state, reference, options);
    this.#fallback.reset(state, reference, options);
    this.#active = selectedBranch(this.#primary.descriptor, reference, options);
    this.#lastCommand = null;
    this.#handoff = null;
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): Promise<HumanoidJointPositionCommand> {
    const selected = selectedBranch(this.#primary.descriptor, reference, options);
    const controller = this.#controller(selected);
    const previous = this.#active;
    const routeChanged = selected !== previous;
    if (routeChanged) {
      controller.reset(state, reference, options);
    }
    const target = await controller.infer(state, reference, options);
    if (routeChanged) {
      this.#active = selected;
      this.#handoff = previous !== null
        && this.#lastCommand !== null
        && this.#handoffSteps > 1
        ? {
            fromImplementation:
              this.#controller(previous).descriptor.implementation,
            toImplementation: controller.descriptor.implementation,
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
    const selected = selectedBranch(this.#primary.descriptor, reference, options);
    if (selected !== this.#active) {
      throw new Error("Humanoid controller capability route changed within one control step");
    }
    this.#controller(selected).advanceHistory(state, reference, options);
  }

  captureState(): HumanoidControllerState {
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: this.descriptor.implementation,
      payload: {
        protocol: ROUTING_STATE_PROTOCOL_V2,
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
          : null
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

  #controller(branch: ControllerBranch): HumanoidWholeBodyController {
    return branch === "primary" ? this.#primary : this.#fallback;
  }
}

function selectedBranch(
  descriptor: HumanoidControllerDescriptor,
  reference: HumanoidReference,
  options: HumanoidControllerInferenceOptions
): ControllerBranch {
  const available = new Set(descriptor.learnedPolicy?.capabilities ?? []);
  const requested = options.taskCommand?.requestedCapabilities ?? [];
  const missingRequestedCapability = requested.some((capability) => (
    !available.has(capability)
  ));
  const unsupportedJointReference = !available.has("joint_reference_tracking")
    && Array.from(reference.jointTrackingWeights).some((weight) => weight > 0);
  return missingRequestedCapability || unsupportedJointReference
    ? "fallback"
    : "primary";
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
  active: ControllerBranch | null;
  primary: HumanoidControllerState;
  fallback: HumanoidControllerState;
  lastCommand: HumanoidJointPositionCommand | null;
  handoff: ControllerHandoff | null;
}

function routingStatePayload(value: JsonValue): RoutingStatePayload | null {
  if (!isRecord(value)
    || (value.protocol !== ROUTING_STATE_PROTOCOL_V1
      && value.protocol !== ROUTING_STATE_PROTOCOL_V2)) return null;
  if (value.active !== null && value.active !== "primary" && value.active !== "fallback") {
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
      handoff: null
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
  return {
    active: value.active,
    primary,
    fallback,
    lastCommand,
    handoff
  };
}

function controllerCommandJson(command: HumanoidJointPositionCommand): JsonValue {
  return {
    kind: command.kind,
    positions: [...command.positions],
    stiffness: [...command.stiffness],
    damping: [...command.damping]
  };
}

function controllerCommandFrom(
  value: JsonValue | undefined
): HumanoidJointPositionCommand | null {
  if (!isRecord(value) || value.kind !== "joint_position_pd") return null;
  const positions = finiteVector(value.positions, false);
  const stiffness = finiteVector(value.stiffness, true);
  const damping = finiteVector(value.damping, true);
  if (!positions || !stiffness || !damping) return null;
  return {
    kind: "joint_position_pd",
    positions: Float64Array.from(positions),
    stiffness: Float64Array.from(stiffness),
    damping: Float64Array.from(damping)
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
  const active = state.active === "primary" ? primary
    : state.active === "fallback" ? fallback
      : null;
  const inactive = state.active === "primary" ? fallback
    : state.active === "fallback" ? primary
      : null;
  if (!active
    || !inactive
    || !state.lastCommand
    || state.handoff.toImplementation !== active.implementation
    || state.handoff.fromImplementation !== inactive.implementation
    || state.handoff.totalSteps !== controllerHandoffSteps(primary, fallback)) {
    throw new Error("Humanoid capability-routing handoff does not match the controller pair");
  }
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
    damping: blendVector(handoff.source.damping, target.damping, amount)
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
}

function cloneCommand(
  command: HumanoidJointPositionCommand
): HumanoidJointPositionCommand {
  assertCommandDimensions(command);
  return {
    kind: "joint_position_pd",
    positions: command.positions.slice(),
    stiffness: command.stiffness.slice(),
    damping: command.damping.slice()
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
