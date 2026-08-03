import { useMemo } from "react";
import type { HumanoidFrameBuffer } from "../stage/humanoid-frame-buffer";
import type {
  HumanoidBodyChannel,
  HumanoidRunDetails,
  HumanoidWorldSnapshot,
  StreamState
} from "../types";
import {
  actionLabel,
  agentNameLabel,
  bodyChannelLabel,
  humanoidControllerLabel,
  motionGeneratorLabel,
  nodeStatusLabel,
  predicateLabel,
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
  const nodes = useMemo(
    () => Object.values(checkpoint.nodes).sort((left, right) => left.created_at.localeCompare(right.created_at)),
    [checkpoint.nodes]
  );
  const current = checkpoint.active_agent_id
    ? checkpoint.nodes[checkpoint.active_agent_id]
    : checkpoint.nodes[checkpoint.root_id];
  const activeChannels = movingHumanoidChannels(frame);
  const latest = props.details.actions.at(-1);
  const passed = checkpoint.checker?.checks.filter((check) => check.passed).length ?? 0;
  const total = checkpoint.goal.predicates.length;
  const context = checkpoint.context_memory;
  const contextLoad = Math.min(100, context.active_estimated_tokens / context.compact_trigger_tokens * 100);

  return (
    <section className="mission-world humanoid-mission-world" aria-label={live ? "实时人形任务" : "人形任务回顾"}>
      <HumanoidStage details={props.details} frameBuffer={props.frameBuffer} live={live} />

      <section className="humanoid-agent-hud game-card" aria-label="层级智能体执行状态">
        <header>
          <span className={live ? "status-beacon live" : "status-beacon"} />
          <b>{live ? "自主运行中" : "运行记录"}</b>
          <small>{live ? streamStatusLabel(props.streamState) : "历史"}</small>
        </header>
        <div className="humanoid-agent-focus">
          <span>{current ? agentNameLabel(current.name) : "等待智能体"}</span>
          <strong>{latest ? actionLabel(latest.action) : "正在建立世界状态"}</strong>
        </div>
        <div className="humanoid-agent-chain">
          {nodes.map((node) => (
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
          <span>物理闭环</span>
          <b>{motionGeneratorLabel(frame.motionGenerator.implementation)} · {humanoidControllerLabel(frame.robot.controller.implementation)} · MuJoCo</b>
        </header>
        <div className="humanoid-state-grid">
          <Metric label="世界版本" value={frame.worldRevision} />
          <Metric label="物理时间" value={`${frame.robot.simulatedTime.toFixed(1)}s`} />
          <Metric label="左脚" value={forceLabel(frame.robot.feet.left.normalForce)} active={frame.robot.feet.left.touching} />
          <Metric label="右脚" value={forceLabel(frame.robot.feet.right.normalForce)} active={frame.robot.feet.right.touching} />
        </div>
        <div className="balance-track">
          <span style={{ width: `${Math.max(0, Math.min(100, frame.robot.balance.upright * 100))}%` }} />
        </div>
        <footer>
          <span>{navigationLabel(frame.navigation.status)}</span>
          <b>{frame.robot.fallen ? "失衡" : "稳定"}</b>
        </footer>
      </section>

      <section className="humanoid-goal-hud game-card" aria-label="目标与长期记忆">
        <div className="humanoid-goal-title">
          <span>目标</span><b>{passed}/{total}</b>
        </div>
        <div className="humanoid-goal-list">
          {checkpoint.goal.predicates.map((predicate, index) => (
            <span key={`${predicate.type}-${index}`}>
              <i className={checkpoint.checker?.checks[index]?.passed ? "passed" : ""} />
              {predicateLabel(predicate)}
            </span>
          ))}
        </div>
        <div className="memory-meter">
          <span><small>上下文</small><b>{compactTokens(context.active_estimated_tokens)}</b></span>
          <i><em style={{ width: `${contextLoad}%` }} /></i>
          <small>{context.total_compactions} 次压缩 · {checkpoint.embodied_memory.total_episodes} 段经历</small>
        </div>
      </section>
    </section>
  );
}

function Metric(props: { label: string; value: string | number; active?: boolean }): React.JSX.Element {
  return <span className={props.active ? "active" : ""}><small>{props.label}</small><b>{props.value}</b></span>;
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
