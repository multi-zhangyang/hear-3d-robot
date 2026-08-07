export const HUMANOID_ACTION_NAMES = [
  "observe_humanoid",
  "submit_humanoid_skill_plan",
  "begin_humanoid_skill",
  "plan_humanoid_skill",
  "execute_humanoid_skill",
  "plan_whole_body_motion",
  "plan_whole_body_motion_candidates",
  "execute_whole_body_motion",
  "plan_humanoid_navigation",
  "execute_humanoid_navigation",
  "remove_world_block"
] as const;

export type HumanoidActionName = typeof HUMANOID_ACTION_NAMES[number];
