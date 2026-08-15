import { useMemo } from "react";
import type { HumanoidFrameBuffer } from "../stage/humanoid-frame-buffer";
import type {
  GoalPredicate,
  HumanoidBodyChannel,
  HumanoidRunDetails,
  HumanoidWorldSnapshot,
  StreamState
} from "../types";
import { activeContextUsage } from "../context-memory";
import { activeCheckpointGoal } from "../goal-state";
import {
  actionLabel,
  agentNameLabel,
  bodyChannelLabel,
  entityLabel,
  humanoidPolicyCapabilityLabel,
  nodeStatusLabel,
  predicateLabel,
  runStatusLabel,
  streamStatusLabel
} from "../ui-text";
import { HumanoidStage } from "./HumanoidStage";
import { useHumanoidHudFrame } from "./use-humanoid-frame";

interface HumanoidMissionWorkspaceProps {
  details: HumanoidRunDetails;
  frameBuffer: HumanoidFrameBuffer;
  streamState: StreamState;
}

export const HUMANOID_BODY_CHANNELS: readonly HumanoidBodyChannel[] = [
  "locomotion",
  "left_leg",
  "right_leg",
  "torso",
  "left_arm",
  "right_arm"
];

export function HumanoidMissionWorkspace(props: HumanoidMissionWorkspaceProps): React.JSX.Element {
  const { checkpoint } = props.details;
  const live = checkpoint.status === "starting" || checkpoint.status === "running";
  const frame = useHumanoidHudFrame(props.frameBuffer, checkpoint.world);
  const current = checkpoint.active_agent_id
    ? checkpoint.nodes[checkpoint.active_agent_id]
    : checkpoint.nodes[checkpoint.root_id];
  const controlPath = useMemo(
    () => activeControlPath(
      checkpoint.nodes,
      checkpoint.root_id,
      checkpoint.active_agent_ids.length > 0
        ? checkpoint.active_agent_ids
        : current ? [current.id] : []
    ),
    [checkpoint.active_agent_ids, checkpoint.nodes, checkpoint.root_id, current]
  );
  const activeChannels = movingHumanoidChannels(frame);
  const latest = props.details.actions.at(-1);
  const goal = activeCheckpointGoal(checkpoint);
  const inactiveGoalStatus = checkpoint.status === "succeeded"
    ? "已完成"
    : checkpoint.status === "failed"
      ? "已结束"
      : "选择中";
  const inactiveGoalDetail = checkpoint.status === "succeeded"
    ? "任务目标已完成"
    : checkpoint.status === "failed"
      ? "运行已结束"
      : "等待目标管理智能体选择";
  const passed = checkpoint.checker?.checks.filter((check) => check.passed).length ?? 0;
  const total = goal?.predicates.length ?? 0;
  const context = checkpoint.context_memory;
  const contextUsage = activeContextUsage(context);
  const activeGrasps = activeHumanoidGrasps(frame);
  const manipulation = humanoidManipulationTelemetry(
    frame,
    goal,
    checkpoint.checker
  );

  return (
    <section className="mission-world humanoid-mission-world" aria-label={live ? "实时人形任务" : "人形任务回顾"}>
      <HumanoidStage details={props.details} frameBuffer={props.frameBuffer} live={live} />

      <section className="humanoid-agent-hud game-card" aria-label="层级智能体执行状态">
        <header>
          <span className={live ? "status-beacon live" : "status-beacon"} />
          <b>层级控制</b>
          <small>{live ? streamStatusLabel(props.streamState) : runStatusLabel(checkpoint.status)}</small>
        </header>
        <div className="humanoid-agent-focus">
          <span>{current ? agentNameLabel(current.name) : "等待智能体"}</span>
          <strong>{latest ? actionLabel(latest.action) : "正在建立世界状态"}</strong>
        </div>
        <div className="humanoid-agent-chain">
          {controlPath.map((node) => (
            <div
              key={node.id}
              className={`${node.status} ${node.id === current?.id ? "current" : ""}`}
            >
              <i />
              <span><b>{agentNameLabel(node.name)}</b><small>{nodeStatusLabel(node.status)}</small></span>
              <em>{node.model_calls_used}</em>
            </div>
          ))}
        </div>
        <div className="humanoid-channel-grid" aria-label="人形身体通道">
          {HUMANOID_BODY_CHANNELS.map((channel) => (
            <span key={channel} className={activeChannels.includes(channel) ? "active" : ""}>
              <i />{bodyChannelLabel(channel)}
            </span>
          ))}
        </div>
      </section>

      <section className="humanoid-state-hud game-card" aria-label="人形物理状态">
        <header>
          <span>物理</span>
          <b>{frame.robot.controller.learnedPolicy ? "ONNX · MuJoCo" : "MuJoCo"}</b>
        </header>
        {frame.robot.controller.learnedPolicy && (
          <div className="humanoid-policy-capabilities" aria-label="已安装学习策略能力">
            <span>策略</span>
            <div>
              {frame.robot.controller.learnedPolicy.capabilities.map((capability) => (
                <b key={capability}>{humanoidPolicyCapabilityLabel(capability)}</b>
              ))}
            </div>
          </div>
        )}
        <div className="humanoid-state-grid">
          <Metric label="版本" value={frame.worldRevision} />
          <Metric label="时间" value={`${frame.robot.simulatedTime.toFixed(1)}s`} />
          <Metric label="左脚" value={forceLabel(frame.robot.feet.left.normalForce)} active={frame.robot.feet.left.touching} />
          <Metric label="右脚" value={forceLabel(frame.robot.feet.right.normalForce)} active={frame.robot.feet.right.touching} />
        </div>
        <div className="balance-track">
          <span style={{ width: `${Math.max(0, Math.min(100, frame.robot.balance.upright * 100))}%` }} />
        </div>
        {manipulation && (
          <div className="humanoid-manipulation-state" aria-label="实时全身交互闭环">
            <div>
              <small>对象</small>
              <b>{entityLabel(manipulation.objectId)}</b>
            </div>
            <ol>
              <li className={manipulation.present ? "active" : ""}>目标</li>
              <li className={manipulation.contact ? "active" : ""}>接触</li>
              <li className={manipulation.grasped ? "active" : ""}>持握</li>
              <li className={manipulation.placed ? "complete" : ""}>落位</li>
            </ol>
          </div>
        )}
        {activeGrasps.length > 0 && (
          <div className="humanoid-grasp-state" aria-label="实时抓取状态">
            {activeGrasps.map((assessment) => (
              <span
                key={`${assessment.object_id}-${assessment.hand}`}
                className={assessment.grasp_verified ? "verified" : ""}
              >
                <i />
                <b>{graspHandLabel(assessment.hand)} · {entityLabel(assessment.object_id)}</b>
                <small>
                  {assessment.grasp_verified ? "稳定 · " : ""}
                  {assessment.evidence.relative_pose.stable_frames}/{assessment.evidence.lifted_hold_frames} 帧
                </small>
              </span>
            ))}
          </div>
        )}
        <footer>
          <span>{navigationLabel(frame.navigation.status)}</span>
          <b>{frame.robot.fallen ? "失衡" : "稳定"}</b>
        </footer>
      </section>

      <section className="humanoid-goal-hud game-card" aria-label="目标与长期记忆">
        <div className="humanoid-goal-title">
          <span>目标</span>
          <b>{goal ? `${passed}/${total}` : inactiveGoalStatus}</b>
        </div>
        <div className="humanoid-goal-list">
          {goal ? goal.predicates.map((predicate, index) => {
            const graspProgress = predicate.type === "object_grasped"
              ? graspPredicateProgress(frame, predicate)
              : null;
            const blockContact = predicate.type === "block_removed"
              ? blockContactProgress(frame, predicate.block_id)
              : null;
            return (
              <span key={`${predicate.type}-${index}`}>
                <i className={checkpoint.checker?.checks[index]?.passed ? "passed" : ""} />
                <b>{predicateLabel(predicate)}</b>
                {predicate.type === "end_effector_at" && (
                  <em aria-label="连续稳定帧">
                    {checkpoint.goal_progress?.predicate_streaks[index] ?? 0}/{predicate.stable_frames}
                  </em>
                )}
                {graspProgress && (
                  <em aria-label="抓取保持进度">{graspProgress}</em>
                )}
                {blockContact && (
                  <em aria-label="方块接触力">{blockContact}</em>
                )}
              </span>
            );
          }) : (
            <span className="humanoid-goal-awaiting">
              <i />
              <b>{inactiveGoalDetail}</b>
            </span>
          )}
        </div>
        <div
          className="memory-meter"
          aria-label={`当前上下文估算为 ${contextUsage.activeEstimatedTokens} 个令牌，上下文窗口为 ${contextUsage.contextWindowTokens} 个令牌，压缩触发线为 ${contextUsage.compactTriggerTokens} 个令牌`}
        >
          <span>
            <small>上下文</small>
            <b>{compactTokens(contextUsage.activeEstimatedTokens)} / {compactTokens(contextUsage.contextWindowTokens)}</b>
          </span>
          <i><em style={{ width: `${contextUsage.loadFraction * 100}%` }} /></i>
        </div>
      </section>
    </section>
  );
}

function activeControlPath(
  nodes: HumanoidRunDetails["checkpoint"]["nodes"],
  rootId: string,
  activeIds: readonly string[]
): HumanoidRunDetails["checkpoint"]["nodes"][string][] {
  const ordered: HumanoidRunDetails["checkpoint"]["nodes"][string][] = [];
  const included = new Set<string>();
  for (const activeId of activeIds) {
    const branch: HumanoidRunDetails["checkpoint"]["nodes"][string][] = [];
    const visited = new Set<string>();
    let cursor = nodes[activeId];
    while (cursor && !visited.has(cursor.id)) {
      visited.add(cursor.id);
      branch.push(cursor);
      cursor = cursor.parent_id ? nodes[cursor.parent_id] : undefined;
    }
    branch.reverse();
    for (const node of branch) {
      if (included.has(node.id)) continue;
      included.add(node.id);
      ordered.push(node);
    }
  }
  const root = nodes[rootId];
  if (ordered.length === 0 && root) ordered.push(root);
  return ordered;
}

function Metric(props: { label: string; value: string | number; active?: boolean }): React.JSX.Element {
  return <span className={props.active ? "active" : ""}><small>{props.label}</small><b>{props.value}</b></span>;
}

function blockContactProgress(
  frame: HumanoidWorldSnapshot,
  blockId: string
): string | null {
  const maximumForce = frame.robot.contacts.reduce((maximum, contact) => (
    contact.firstSolid === blockId || contact.secondSolid === blockId
      ? Math.max(maximum, contact.normalForce)
      : maximum
  ), 0);
  return maximumForce > 0 ? `接触 ${forceLabel(maximumForce)}` : null;
}

export function movingHumanoidChannels(
  frame: HumanoidWorldSnapshot
): HumanoidBodyChannel[] {
  const active = new Set<HumanoidBodyChannel>();
  const pelvis = frame.robot.links.pelvis;
  if (pelvis && (
    Math.hypot(pelvis.linearVelocity.x, pelvis.linearVelocity.z) >= 0.05
      || Math.abs(pelvis.angularVelocity.y) >= 0.08
  )) {
    active.add("locomotion");
  }
  for (const [name, joint] of Object.entries(frame.robot.joints)) {
    if (Math.abs(joint.velocity) < 0.08) continue;
    if (name.startsWith("left_hip") || name.startsWith("left_knee")
      || name.startsWith("left_ankle")) {
      active.add("left_leg");
    } else if (name.startsWith("right_hip") || name.startsWith("right_knee")
      || name.startsWith("right_ankle")) {
      active.add("right_leg");
    } else if (name.startsWith("waist_")) active.add("torso");
    else if (name.startsWith("left_")) active.add("left_arm");
    else if (name.startsWith("right_")) active.add("right_arm");
  }
  return HUMANOID_BODY_CHANNELS.filter((channel) => active.has(channel));
}

export function activeHumanoidGrasps(
  frame: HumanoidWorldSnapshot
): HumanoidWorldSnapshot["grasp"]["assessments"] {
  return frame.grasp.assessments.filter((assessment) => (
    assessment.frame === frame.frame
      && (assessment.grasp_verified
        || assessment.phase !== "idle"
        || assessment.evidence.contact.status !== "missing")
  ));
}

function graspHandLabel(hand: "left" | "right"): string {
  return hand === "left" ? "左手" : "右手";
}

function graspPredicateProgress(
  frame: HumanoidWorldSnapshot,
  predicate: Extract<GoalPredicate, { type: "object_grasped" }>
): string | null {
  const matching = frame.grasp.assessments.filter((assessment) => (
    assessment.frame === frame.frame
      && assessment.object_id === predicate.object_id
      && (predicate.hand === "either" || assessment.hand === predicate.hand)
  ));
  const assessment = matching.find((candidate) => candidate.grasp_verified)
    ?? matching.find((candidate) => candidate.evidence.contact.status !== "missing");
  return assessment
    ? `${assessment.evidence.relative_pose.stable_frames}/${assessment.evidence.lifted_hold_frames}`
    : null;
}

export interface HumanoidManipulationTelemetry {
  objectId: string;
  present: boolean;
  contact: boolean;
  grasped: boolean;
  placed: boolean;
}

export function humanoidManipulationTelemetry(
  frame: HumanoidWorldSnapshot,
  goal: { predicates: GoalPredicate[] } | null | undefined,
  checker: HumanoidRunDetails["checkpoint"]["checker"]
): HumanoidManipulationTelemetry | null {
  const predicateIndex = goal?.predicates.findIndex((predicate) => (
    predicate.type === "object_grasped"
      || predicate.type === "object_at"
      || predicate.type === "object_in_zone"
      || predicate.type === "object_placed"
  )) ?? -1;
  if (predicateIndex < 0 || !goal) return null;
  const predicate = goal.predicates[predicateIndex]!;
  if (predicate.type !== "object_grasped"
    && predicate.type !== "object_at"
    && predicate.type !== "object_in_zone"
    && predicate.type !== "object_placed") return null;
  const assessments = frame.grasp.assessments.filter((assessment) => (
    assessment.frame === frame.frame && assessment.object_id === predicate.object_id
  ));
  return {
    objectId: predicate.object_id,
    present: frame.robot.objects[predicate.object_id] !== undefined,
    contact: assessments.some((assessment) => (
      assessment.evidence.contact.status !== "missing"
    )),
    grasped: assessments.some((assessment) => assessment.grasp_verified),
    placed: predicate.type === "object_placed"
      && checker?.checks[predicateIndex]?.passed === true
  };
}

function forceLabel(force: number): string {
  return `${Math.round(Math.max(0, force))}N`;
}

function navigationLabel(status: HumanoidWorldSnapshot["navigation"]["status"]): string {
  if (status === "planned") return "路线已验证";
  if (status === "executing") return "双足导航中";
  if (status === "completed") return "导航分块完成";
  if (status === "blocked") return "路线受阻";
  return "等待动作";
}

function compactTokens(tokens: number): string {
  return new Intl.NumberFormat("zh-CN", { notation: "compact", maximumFractionDigits: 1 }).format(tokens);
}
