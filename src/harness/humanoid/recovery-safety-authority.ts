import type { NeuralSafetyInterrupt } from "../../domain/neural-hierarchy.js";
import { HUMANOID_NEURAL_AGENT_IDS } from "./neural-hierarchy-contract.js";

export function humanoidRecoverySafetyInterruptIsCurrent(
  interrupt: NeuralSafetyInterrupt | undefined,
  input: {
    worldRevision: number;
    interruptId?: string;
  }
): interrupt is NeuralSafetyInterrupt {
  return interrupt?.kind === "stationary_fall"
    && interrupt.status === "acknowledged"
    && interrupt.source_node_id === HUMANOID_NEURAL_AGENT_IDS.body
    && interrupt.relay_node_id === HUMANOID_NEURAL_AGENT_IDS.reflex
    && interrupt.target_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
    && interrupt.world_revision <= input.worldRevision
    && (input.interruptId === undefined
      || interrupt.interrupt_id === input.interruptId);
}
