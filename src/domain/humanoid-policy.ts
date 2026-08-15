export const HUMANOID_LEARNED_POLICY_CAPABILITIES = [
  "balance",
  "whole_body_recovery",
  "locomotion",
  "joint_reference_tracking",
  "contact_rich_manipulation",
  "bimanual_manipulation"
] as const;

export type HumanoidLearnedPolicyCapability =
  typeof HUMANOID_LEARNED_POLICY_CAPABILITIES[number];

export const HUMANOID_RECOVERY_CONTROL_STEP_SECONDS = 0.02;
export const HUMANOID_RECOVERY_STABLE_STEPS = 20;
export const HUMANOID_RECOVERY_HANDOFF_STEPS = 20;
export const HUMANOID_RECOVERY_MAXIMUM_STEPS = 750;
export const HUMANOID_RECOVERY_WINDOW_SECONDS =
  HUMANOID_RECOVERY_CONTROL_STEP_SECONDS * HUMANOID_RECOVERY_MAXIMUM_STEPS;
