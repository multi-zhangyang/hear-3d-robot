export const HUMANOID_NAVIGATION_HORIZON = Object.freeze({
  // Each physical route is capped at 3 m by HumanoidWorld. This keeps one
  // semantic Skill bounded to 96 m without coupling the bound to model turns.
  maximum_segments: 32,
  // A route has a hard 90 s travel budget plus a bounded stopping window at
  // 50 Hz. The complete Skill window remains finite at the segment limit.
  maximum_control_steps: 32 * 100 * 50
});

export function humanoidNavigationSegmentBudgetExhausted(
  completedSegments: number,
  currentProcessSegments = 0
): boolean {
  if (!Number.isSafeInteger(completedSegments) || completedSegments < 0
    || !Number.isSafeInteger(currentProcessSegments)
    || currentProcessSegments < 0) {
    throw new Error("Navigation segment progress must be a non-negative integer");
  }
  return completedSegments + currentProcessSegments
    >= HUMANOID_NAVIGATION_HORIZON.maximum_segments;
}
