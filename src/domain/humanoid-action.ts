export const HUMANOID_ACTION_NAMES = [
  "observe_humanoid",
  "plan_whole_body_motion",
  "plan_whole_body_motion_candidates",
  "execute_whole_body_motion",
  "plan_humanoid_navigation",
  "execute_humanoid_navigation",
  "remove_world_block"
] as const;

export type HumanoidActionName = typeof HUMANOID_ACTION_NAMES[number];
