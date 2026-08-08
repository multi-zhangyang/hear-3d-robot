import type { JsonValue } from "../../domain/schema.js";
import type { HumanoidReference } from "./reference.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerInferenceOptions,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

const ROUTING_STATE_PROTOCOL = "humanoid-controller-capability-routing-state-v1";
type ControllerBranch = "primary" | "fallback";

export function humanoidControllerNeedsReferenceFallback(
  descriptor: HumanoidControllerDescriptor
): boolean {
  const learnedPolicy = descriptor.learnedPolicy;
  return learnedPolicy !== undefined
    && descriptor.capabilityRouting === undefined
    && !learnedPolicy.capabilities.includes("joint_reference_tracking");
}

export class CapabilityRoutingHumanoidController
implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor;
  readonly #primary: HumanoidWholeBodyController;
  readonly #fallback: HumanoidWholeBodyController;
  #active: ControllerBranch | null = null;

  constructor(
    primary: HumanoidWholeBodyController,
    fallback: HumanoidWholeBodyController
  ) {
    assertRoutingPair(primary.descriptor, fallback.descriptor);
    this.#primary = primary;
    this.#fallback = fallback;
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

  reset(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): void {
    this.#primary.reset(state, reference, options);
    this.#fallback.reset(state, reference, options);
    this.#active = selectedBranch(this.#primary.descriptor, reference, options);
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): Promise<HumanoidJointPositionCommand> {
    const selected = selectedBranch(this.#primary.descriptor, reference, options);
    const controller = this.#controller(selected);
    if (selected !== this.#active) {
      controller.reset(state, reference, options);
      this.#active = selected;
    }
    return controller.infer(state, reference, options);
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
        protocol: ROUTING_STATE_PROTOCOL,
        active: this.#active,
        primary: controllerStateJson(this.#primary.captureState()),
        fallback: controllerStateJson(this.#fallback.captureState())
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
      return;
    }
    if (routed.primary.implementation !== this.#primary.descriptor.implementation
      || routed.fallback.implementation !== this.#fallback.descriptor.implementation) {
      throw new Error("Humanoid capability-routing state belongs to another controller pair");
    }
    this.#primary.restoreState(routed.primary);
    this.#fallback.restoreState(routed.fallback);
    this.#active = routed.active;
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
  if (!fallback.learnedPolicy?.capabilities.includes("joint_reference_tracking")) {
    throw new Error("Reference fallback must support joint-reference tracking");
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

function routingStatePayload(value: JsonValue): {
  active: ControllerBranch | null;
  primary: HumanoidControllerState;
  fallback: HumanoidControllerState;
} | null {
  if (!isRecord(value) || value.protocol !== ROUTING_STATE_PROTOCOL) return null;
  if (value.active !== null && value.active !== "primary" && value.active !== "fallback") {
    throw new Error("Invalid humanoid capability-routing active branch");
  }
  const primary = controllerStateFrom(value.primary);
  const fallback = controllerStateFrom(value.fallback);
  if (!primary || !fallback) {
    throw new Error("Invalid humanoid capability-routing nested controller state");
  }
  return { active: value.active, primary, fallback };
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
