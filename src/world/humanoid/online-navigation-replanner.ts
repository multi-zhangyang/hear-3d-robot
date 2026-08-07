export const MAXIMUM_ONLINE_NAVIGATION_REPLANS = 2;

export interface OnlineNavigationReplanDecision {
  replan: boolean;
  failure_class: "dynamic_obstruction" | "unsafe_state" | "semantic_recovery" | "budget_exhausted";
}

export function onlineNavigationReplanDecision(input: {
  reason: string | undefined;
  fallen: boolean;
  attempts: number;
}): OnlineNavigationReplanDecision {
  if (input.fallen) {
    return { replan: false, failure_class: "unsafe_state" };
  }
  if (input.attempts >= MAXIMUM_ONLINE_NAVIGATION_REPLANS) {
    return { replan: false, failure_class: "budget_exhausted" };
  }
  const obstruction = input.reason?.startsWith("environment_contact:")
    || input.reason?.startsWith(
      "contact_while_stopping:environment_contact:"
    );
  return obstruction
    ? { replan: true, failure_class: "dynamic_obstruction" }
    : { replan: false, failure_class: "semantic_recovery" };
}
