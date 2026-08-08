export const HUMANOID_LEARNED_POLICY_CAPABILITIES = [
  "balance",
  "locomotion",
  "joint_reference_tracking",
  "contact_rich_manipulation",
  "bimanual_manipulation"
] as const;

export type HumanoidLearnedPolicyCapability =
  typeof HUMANOID_LEARNED_POLICY_CAPABILITIES[number];
