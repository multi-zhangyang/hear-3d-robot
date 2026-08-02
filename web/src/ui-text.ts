import type {
  BodyChannel,
  Goal,
  GoalPredicate,
  RunCheckpoint,
  RunListItem,
  StreamState,
  TaskNode,
  VoxelMaterial
} from "./types";

const RUN_STATUS: Record<RunListItem["status"], string> = {
  starting: "启动中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  interrupted: "已暂停",
  local_artifact: "本地记录"
};

const NODE_STATUS: Record<TaskNode["status"], string> = {
  ready: "就绪",
  active: "执行中",
  waiting: "等待中",
  completed: "已完成",
  blocked: "已阻塞",
  failed: "失败"
};

const STREAM_STATUS: Record<StreamState, string> = {
  inactive: "未连接",
  connecting: "连接中",
  connected: "已同步",
  disconnected: "已断开"
};

const BODY_CHANNEL: Record<BodyChannel, string> = {
  base: "底盘",
  head: "传感头",
  arm: "机械臂",
  gripper: "夹爪"
};

const ACTION_LABELS: Record<string, string> = {
  read_proprioception: "读取机器人本体状态",
  sense_scene: "观察可见世界",
  survey_terrain: "勘察附近地形",
  scan_voxels: "扫描可见方块",
  inspect_voxel: "检查体素方块",
  recall_spatial_memory: "检索空间记忆",
  inspect_entity: "检查世界实体",
  query_contacts: "检查物理接触",
  plan_base_path: "规划底盘路线",
  plan_arm_retraction: "规划机械臂收拢",
  plan_joint_targets: "规划关节路径",
  solve_end_effector_position: "求解机械臂位置",
  solve_end_effector_pose: "求解机械臂姿态",
  inspect_command: "检查运动指令",
  execute_base_plan: "执行底盘运动",
  execute_joint_plan: "执行关节运动",
  drive_base: "调整底盘运动",
  set_head_target: "调整传感头",
  set_joint_targets: "调整机器人关节",
  set_gripper_target: "调整夹爪",
  break_voxel: "破坏体素方块",
  place_voxel: "放置体素方块",
  check_mission: "检查任务目标",
  complete_mission: "提交任务完成结果",
  delegate_agent: "委派下级智能体",
  complete_assignment: "提交分配结果",
  tool: "工具"
};

const PHASE_LABELS: Record<string, string> = {
  idle: "空闲",
  active: "执行中",
  starting: "启动中",
  running: "运行中",
  planned: "已规划",
  planning: "规划中",
  queued: "排队中",
  executing: "执行中",
  completed: "已完成",
  blocked: "已阻塞",
  stopped: "已停止",
  stopping: "停止中",
  interrupted: "已暂停",
  consumed: "已使用",
  stale: "已失效",
  accepted: "已接受",
  rejected: "已拒绝"
};

const RESULT_CODES: Record<string, string> = {
  accepted: "已接受",
  action_reused: "已复用有效动作回执",
  arm_retraction_not_required: "机械臂无需收拢",
  arm_retraction_options: "已生成机械臂收拢方案",
  arm_retraction_unavailable: "没有可用的机械臂收拢方案",
  arm_trajectory_unavailable: "没有可用的机械臂轨迹",
  attached_object_missing: "已抓取物体不存在",
  authority_denied: "动作权限校验未通过",
  base_angular_velocity_limit: "底盘角速度超出限制",
  base_duration_limit: "底盘运动时长超出限制",
  base_linear_velocity_limit: "底盘线速度超出限制",
  base_motion_blocked: "底盘运动受阻",
  base_motion_completed: "底盘运动已完成",
  base_path_collision: "底盘路线存在碰撞",
  mission_satisfied: "任务条件已满足",
  mission_incomplete: "任务条件尚未全部满足",
  base_path_planned: "底盘路线已规划",
  base_path_unavailable: "没有可用的底盘路线",
  base_plan_completed: "底盘运动已完成",
  body_channel_busy: "身体通道正忙",
  child_agent_blocked: "下级智能体已阻塞",
  child_agent_failed: "下级智能体执行失败",
  command_completed: "运动指令已完成",
  command_interrupted: "运动指令已中断",
  command_state: "运动指令状态已读取",
  contact_state: "物理接触状态已读取",
  empty_joint_target: "关节目标为空",
  end_effector_solution: "已求得末端执行器解",
  end_effector_verification_failed: "末端执行器目标验证失败",
  end_effector_verified: "末端执行器目标已验证",
  entity_not_visible: "目标实体不可见",
  entity_state: "实体状态已读取",
  grasp_slipped: "抓取物已滑脱",
  grasp_unstable: "抓取状态不稳定",
  gripper_force_limit: "夹爪力度超出限制",
  gripper_joint_limit: "夹爪关节超出限制",
  gripper_motion_blocked: "夹爪运动受阻",
  gripper_motion_timeout: "夹爪运动超时",
  gripper_target_reached: "夹爪已到达目标",
  head_joint_limit: "传感头关节超出限制",
  head_motion_blocked: "传感头运动受阻",
  head_motion_timeout: "传感头运动超时",
  head_target_reached: "传感头已到达目标",
  ik_not_converged: "逆运动学求解未收敛",
  ik_residual_too_large: "逆运动学残差过大",
  ik_solution_outside_limits: "逆运动学解超出关节限制",
  ik_solver_error: "逆运动学求解失败",
  ik_trajectory_endpoint_blocked: "机械臂轨迹终点受阻",
  invalid_base_face_point: "底盘朝向目标无效",
  invalid_base_velocity: "底盘速度无效",
  invalid_end_effector_target: "末端执行器目标无效",
  invalid_gripper_force: "夹爪力度无效",
  invalid_gripper_motion_options: "夹爪运动参数无效",
  invalid_head_motion_options: "传感头运动参数无效",
  invalid_joint_velocity: "关节速度无效",
  invalid_motion_duration: "运动时长无效",
  invalid_planning_transaction: "规划回执无效",
  invalid_skill_input: "动作输入无效",
  joint_limit: "关节超出限制",
  joint_motion_blocked: "关节运动受阻",
  joint_plan_completed: "关节运动已完成",
  joint_target_plan: "关节路线已规划",
  joint_targets_reached: "关节已到达目标",
  joint_trajectory_blocked: "关节轨迹受阻",
  keyed_lock_transition: "钥匙锁状态已更新",
  navigation_projection_failed: "导航投影失败",
  plan_already_consumed: "规划已被使用",
  planning_receipt_missing_plan: "规划回执缺少路线",
  planning_transaction_not_granted: "规划回执未授权给当前智能体",
  proprioception: "机器人本体状态已读取",
  repeated_accepted_action: "检测到重复的已接受动作",
  repeated_denied_action: "检测到重复的已拒绝动作",
  robot_link_state_unavailable: "机器人连杆状态不可用",
  scene_observation: "场景观察已完成",
  scene_observed: "场景观察已完成",
  spatial_memory_context_unavailable: "当前智能体无可用空间记忆上下文",
  spatial_memory_recalled: "空间记忆已读取",
  stale_plan_revision: "规划对应的世界版本已失效",
  target_out_of_bounds: "目标超出世界边界",
  terrain_survey: "地形勘察已完成",
  terrain_unavailable: "地形数据不可用",
  unknown_arm_plan: "机械臂规划不存在",
  unknown_base_plan: "底盘规划不存在",
  unknown_entity: "目标实体不存在",
  unknown_planning_transaction: "规划回执不存在",
  unknown_skill: "请求的机器人动作不存在",
  unknown_tool: "请求的智能体工具不存在",
  unsupported_keyed_lock_geometry: "钥匙锁几何结构不受支持",
  voxel_boundary_protected: "世界边界方块受保护",
  voxel_broken: "体素方块已破坏",
  voxel_chunk_unloaded: "体素分块尚未加载",
  voxel_edit_failed: "体素方块修改失败",
  voxel_empty: "目标体素为空",
  voxel_interaction_ready: "体素交互条件已就绪",
  voxel_interaction_surface_unavailable: "体素交互表面不可用",
  voxel_not_visible: "目标体素不可见",
  voxel_occupied: "目标体素已被占用",
  voxel_out_of_bounds: "目标体素超出世界边界",
  voxel_out_of_reach: "目标体素超出机器人可达范围",
  voxel_placed: "体素方块已放置",
  voxel_placement_blocked: "体素放置受阻",
  voxel_scan: "体素扫描已完成",
  voxel_state: "体素状态已读取",
  voxel_unsupported: "目标体素缺少支撑",
  voxel_world_unavailable: "体素世界不可用"
};

const SCENARIO_LABELS: Record<string, string> = {
  voxel_expanse: "体素原野",
  voxel_highlands: "体素高地",
  voxel_survey: "体素勘察",
  voxel_realm: "体素疆域",
  open_navigation: "开放导航",
  fetch_red_block: "红色方块搬运",
  locked_container: "锁定容器"
};

const MATERIAL_LABELS: Record<VoxelMaterial | "ground" | "air", string> = {
  grass: "草方块",
  dirt: "泥土方块",
  stone: "石头方块",
  sand: "沙方块",
  placed: "已放置方块",
  ground: "地面",
  air: "空"
};

const AGENT_NAMES: Record<string, string> = {
  "mission coordinator": "任务协调智能体",
  "mission supervisor": "任务监督智能体",
  "capability worker": "能力执行智能体",
  "movement supervisor": "运动监督智能体",
  "movement leaf": "运动执行智能体",
  "exploration supervisor": "探索监督智能体",
  "exploration worker": "探索执行智能体",
  "terrain surveyor": "地形勘察智能体",
  "movement executor": "运动执行智能体",
  "frontier mover": "探索边界智能体",
  "observe and unblock": "观察解阻智能体",
  "exploration agent": "探索智能体",
  "block manipulator": "方块操作智能体",
  "carry and release": "搬运释放智能体",
  "nav to green zone": "目标区导航智能体",
  "scene observer": "场景观察智能体",
  navigator: "导航智能体"
};

const ENTITY_NAMES: Record<string, string> = {
  red_block: "红色方块",
  blue_block: "蓝色方块",
  amber_block: "琥珀色方块",
  green_zone: "绿色目标区",
  arrival_zone: "到达区域",
  locked_box: "锁定容器",
  brass_key: "黄铜钥匙",
  barrier_a: "障碍物 A",
  center_column: "中央立柱",
  divider: "隔断"
};

export function runStatusLabel(status: RunListItem["status"]): string {
  return RUN_STATUS[status];
}

export function nodeStatusLabel(status: TaskNode["status"]): string {
  return NODE_STATUS[status];
}

export function streamStatusLabel(status: StreamState): string {
  return STREAM_STATUS[status];
}

export function bodyChannelLabel(channel: BodyChannel): string {
  return BODY_CHANNEL[channel];
}

export function actionLabel(value: string): string {
  return ACTION_LABELS[value] ?? "未识别动作";
}

export function phaseLabel(value: string): string {
  return PHASE_LABELS[value] ?? "状态未知";
}

export function resultCodeLabel(value: string): string {
  return RESULT_CODES[value] ?? "未识别的运行回执";
}

export function scenarioLabel(id: string | null, fallback?: string): string {
  if (id && SCENARIO_LABELS[id]) return SCENARIO_LABELS[id];
  return fallback?.trim() || id || "未知场景";
}

export function materialLabel(material: VoxelMaterial | "ground" | "air" | null): string {
  return MATERIAL_LABELS[material ?? "air"];
}

export function agentNameLabel(name: string): string {
  const normalized = name.trim().toLowerCase().replaceAll("_", " ").replace(/\s+/g, " ");
  const exact = AGENT_NAMES[normalized];
  if (exact) return exact;
  for (const [base, label] of Object.entries(AGENT_NAMES)) {
    const match = normalized.match(new RegExp(`^${escapeRegExp(base)} (\\d+)$`));
    if (match) return `${label} ${match[1]}`;
  }
  if (/[\u3400-\u9fff]/.test(name)) return name;
  const suffix = normalized.match(/(?:^| )(\d+)$/)?.[1];
  return suffix ? `专项智能体 ${suffix}` : "专项智能体";
}

export function entityLabel(id: string): string {
  if (ENTITY_NAMES[id]) return ENTITY_NAMES[id];
  return /[\u3400-\u9fff]/.test(id) ? id : "场景实体";
}

function propertyLabel(value: "locked" | "enabled"): string {
  return value === "locked" ? "锁定" : "启用";
}

export function nodePurposeLabel(node: TaskNode, goal: Goal): string {
  if (node.depth === 0) return goalSummaryLabel(goal);
  if (node.may_delegate) return "协调下级智能体 · 汇总证据";
  const capabilities = [...new Set(node.capabilities.map(actionLabel))].slice(0, 3);
  return capabilities.length > 0
    ? capabilities.join(" · ")
    : "执行当前分配";
}

export function nodeResultLabel(node: TaskNode): string | null {
  if (node.last_result === undefined) return null;
  const output = modelOutputLabel(node.last_result);
  if (output) return output;
  if (node.status === "completed") return "当前分配已完成。";
  if (node.status === "blocked") return "当前分配已被阻塞，等待上级重新规划。";
  if (node.status === "failed") return "当前分配执行失败，等待上级处理。";
  return null;
}

const MODEL_OUTPUT_LIMIT = 240;
const SENSITIVE_ASSIGNMENT = /\b(?:provider(?:[_ -]?id)?|endpoint|base[_ -]?url|api[_ -]?key|authorization|model|response[_ -]?id|request[_ -]?id|trace[_ -]?id|prompt|instructions)\b\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)/giu;
const CREDENTIAL_ASSIGNMENT = /\b[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\b\s*=\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)/giu;
const CREDENTIAL_VALUE = /(?:\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{16,}|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]{5,})?)/giu;
const PROVIDER_ID_VALUE = /\b(?:response|resp|request|req|trace|gen|call|message|msg|transaction)[_-][A-Za-z0-9_-]{8,}\b/giu;
const URL_VALUE = /https?:\/\/[^\s<>"')\]}]+/giu;
const MODEL_ID_VALUE = /\b[a-z0-9][a-z0-9._-]{1,}\/[a-z0-9][a-z0-9._:-]{2,}\b/giu;

export function modelOutputLabel(value: unknown, maxLength = MODEL_OUTPUT_LIMIT): string | null {
  const extracted = extractModelOutput(value, 0);
  if (!extracted) return null;
  let text = extracted
    .replace(/```[\s\S]*?```/gu, " ")
    .replace(SENSITIVE_ASSIGNMENT, " ")
    .replace(CREDENTIAL_ASSIGNMENT, " ")
    .replace(CREDENTIAL_VALUE, "[敏感信息已隐藏]")
    .replace(PROVIDER_ID_VALUE, "[内部标识已隐藏]")
    .replace(URL_VALUE, "[链接已隐藏]")
    .replace(MODEL_ID_VALUE, "[模型标识已隐藏]");
  const rawJsonStart = text.search(/[{[](?=[^{}\[\]]{0,80}"[^"]+"\s*:)/u);
  if (rawJsonStart >= 0) text = text.slice(0, rawJsonStart);
  text = text.replace(/\s+/gu, " ").replace(/\s+([，。！？；：,.!?;:])/gu, "$1").trim();
  if (!text) return null;
  const characters = Array.from(text);
  const limit = Math.max(40, Math.floor(maxLength));
  return characters.length <= limit ? text : `${characters.slice(0, limit - 1).join("")}…`;
}

function extractModelOutput(value: unknown, depth: number): string | null {
  if (depth > 4) return null;
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return null;
    const fenced = text.match(/^```([^\r\n]*)[\r\n]+([\s\S]*?)\s*```$/u);
    if (fenced?.[2]) {
      const language = fenced[1]?.trim().toLowerCase();
      return language === "" || language === "json"
        ? extractModelOutput(fenced[2], depth + 1)
        : null;
    }
    if (text.startsWith("{") || text.startsWith("[")) {
      try {
        return extractModelOutput(JSON.parse(text), depth + 1);
      } catch {
        return text;
      }
    }
    return text;
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const object = value as Record<string, unknown>;
  for (const key of ["summary", "message", "result", "output", "final_output"]) {
    const extracted = extractModelOutput(object[key], depth + 1);
    if (extracted) return extracted;
  }
  return null;
}

export function goalSummaryLabel(goal: Goal): string {
  if (goal.predicates.length === 1) return predicateLabel(goal.predicates[0]!);
  return `完成 ${goal.predicates.length} 项结构化任务条件。`;
}

export function predicateLabel(predicate: GoalPredicate): string {
  switch (predicate.type) {
    case "robot_at":
      return `到达坐标 ${position(predicate.target)}`;
    case "robot_in_zone":
      return `进入区域 ${entityLabel(predicate.zone_id)}`;
    case "terrain_explored":
      return `自主探索并绘制 ${(predicate.minimum_fraction * 100).toFixed(0)}% 的未知地形`;
    case "voxel_at":
      return `坐标 [${predicate.coordinate.column}, ${predicate.coordinate.level}, ${predicate.coordinate.row}] 为${materialLabel(predicate.material)}`;
    case "object_in_zone":
      return `${entityLabel(predicate.object_id)}${predicate.expected ? "位于" : "离开"}区域 ${entityLabel(predicate.zone_id)}`;
    case "object_at":
      return `将${entityLabel(predicate.object_id)}移动到 ${position(predicate.target)}`;
    case "object_property":
      return `${entityLabel(predicate.object_id)}的“${propertyLabel(predicate.property)}”状态为${predicate.expected ? "真" : "假"}`;
    case "object_attached":
      return `${entityLabel(predicate.object_id)}${predicate.expected ? "已被夹爪抓取" : "已从夹爪释放"}`;
  }
}

export function missionResultLabel(checkpoint: RunCheckpoint): string | null {
  if (checkpoint.status === "starting" || checkpoint.status === "running") {
    return "任务正在由层级智能体自主执行。";
  }
  if (checkpoint.status === "succeeded") {
    const exploration = checkpoint.goal.predicates.find((predicate) => predicate.type === "terrain_explored");
    if (exploration && checkpoint.world.explored.total > 0) {
      const { seen, total } = checkpoint.world.explored;
      const percent = seen / total * 100;
      return `机器人已自主探索 ${seen.toLocaleString("zh-CN")}/${total.toLocaleString("zh-CN")} 个地形单元（${percent.toFixed(2)}%），结构化任务检查已通过。`;
    }
    const passed = checkpoint.checker?.checks.filter((check) => check.passed).length ?? 0;
    const total = checkpoint.goal.predicates.length;
    return `${passed}/${total} 项结构化任务条件已通过，任务完成。`;
  }
  if (checkpoint.status === "interrupted") return "任务已暂停，可在条件恢复后继续运行。";
  return "任务未完成，请根据错误信息重新规划或恢复运行。";
}

export function runOptionLabel(run: RunListItem): string {
  return `${scenarioLabel(run.scenario_id)} · ${runStatusLabel(run.status)}`;
}

function position(point: { x: number; z: number }): string {
  return `[${point.x.toFixed(1)}, ${point.z.toFixed(1)}]`;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
