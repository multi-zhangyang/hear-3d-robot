import { useMemo } from "react";
import { RobotStage } from "./RobotStage";
import { focusedCommand, liveChannels, liveCommands } from "./active-commands";
import type { AuthoritativeFrameBuffer } from "./stage/authoritative-frame-buffer";
import { useAuthoritativeHudFrame } from "./stage/use-authoritative-frame";
import type {
  BodyChannel,
  RunDetails,
  StreamState,
  TaskNode
} from "./types";
import {
  actionLabel,
  agentNameLabel,
  bodyChannelLabel,
  nodePurposeLabel,
  nodeStatusLabel,
  phaseLabel,
  predicateLabel,
  streamStatusLabel
} from "./ui-text";

interface MissionWorkspaceProps {
  details: RunDetails;
  frameBuffer: AuthoritativeFrameBuffer;
  streamState: StreamState;
}

const BODY_CHANNELS: BodyChannel[] = ["base", "head", "arm", "gripper"];

/**
 * The mission view is the world, not a dashboard around the world. Durable
 * evidence remains in the hotbar panels; this component only overlays the
 * state needed while watching an autonomous body move.
 */
export function MissionWorkspace(props: MissionWorkspaceProps): React.JSX.Element {
  const { checkpoint } = props.details;
  const live = checkpoint.status === "running" || checkpoint.status === "starting";
  const frame = useAuthoritativeHudFrame(props.frameBuffer, checkpoint.world);
  const root = checkpoint.nodes[checkpoint.root_id];
  const current = checkpoint.active_agent_id
    ? checkpoint.nodes[checkpoint.active_agent_id] ?? root
    : root;
  const nodes = useMemo(
    () => Object.values(checkpoint.nodes).sort((left, right) =>
      left.created_at.localeCompare(right.created_at)),
    [checkpoint.nodes]
  );
  const activeIds = checkpoint.active_agent_ids?.length
    ? checkpoint.active_agent_ids
    : nodes.filter((node) => node.status === "active").map((node) => node.id);
  const activeIdSet = new Set(activeIds);
  const lineage = current ? agentLineage(current, checkpoint.nodes) : [];
  const hierarchyNodes = useMemo(
    () => visibleHierarchyNodes(nodes, checkpoint.nodes, checkpoint.root_id, activeIds, current?.id),
    [activeIds, checkpoint.nodes, checkpoint.root_id, current?.id, nodes]
  );
  const passed = checkpoint.checker?.checks.filter((check) => check.passed).length ?? 0;
  const predicates = checkpoint.goal.predicates.length;
  const activeCommands = liveCommands(frame);
  const activeChannels = liveChannels(frame);
  const progress = predicates === 0 ? 0 : passed / predicates * 100;
  const command = focusedCommand(frame);

  return (
    <section className="mission-world" aria-label={live ? "实时任务" : "任务回顾"}>
      <RobotStage
        key={props.details.definition.run_id}
        initialFrame={checkpoint.world}
        frameBuffer={props.frameBuffer}
        scenario={props.details.definition.scenario}
        live={live}
      />

      <section className="agent-hud game-card" aria-label="当前智能体状态">
        <div className="hud-kicker">
          <span className={live ? "status-beacon live" : "status-beacon"} />
          <span>{live ? "自主运行中" : checkpoint.status === "succeeded" ? "任务回顾 · 已完成" : "任务回顾 · 已结束"}</span>
          {activeIds.length > 1 && <span className="parallel-count">{activeIds.length} 个智能体并行</span>}
          <span className="hud-spacer" />
          <span>{live ? streamStatusLabel(props.streamState) : "历史记录"}</span>
        </div>
        <strong className="hud-agent-name">
          {current ? agentNameLabel(current.name) : "暂无活动智能体"}
          {activeIds.length > 1 ? ` +${activeIds.length - 1}` : ""}
        </strong>
        <p className="hud-objective">{current
          ? nodePurposeLabel(current, checkpoint.goal)
          : "等待智能体"}</p>
        <div className="hud-line" aria-label="当前智能体层级路径">
          <span>层级路径</span>
          <b>{lineage.map((node) => agentNameLabel(node.name)).join(" / ") || "—"}</b>
        </div>
        <div className="hud-line" aria-label="机器人身体通道租约">
          <span>控制通道</span>
          <div className="lease-slots">
            {BODY_CHANNELS.map((channel) => (
              <i key={channel} className={activeChannels.includes(channel) ? "active" : ""}>
                {bodyChannelLabel(channel)}
              </i>
            ))}
          </div>
        </div>
        <div className="agent-now">
          <span><small>当前</small><b>{activeCommands.length > 1
            ? `${activeCommands.length} 条身体指令`
            : command ? `${actionLabel(command.skill)} · ${phaseLabel(command.phase)}` : "等待任务"}</b></span>
          <span><small>目标</small><b>{passed}/{predicates}</b></span>
        </div>
        <div className="agent-progress"><i style={{ width: `${progress}%` }} /></div>
      </section>

      <section className="hierarchy-hud game-card" aria-label="实时智能体层级">
        <header>
          <span>智能体流</span>
          <b>{activeIds.length > 0 ? `${activeIds.length} 个执行中` : `${nodes.length} 个智能体`}</b>
        </header>
        <div className="hud-tree">
          {hierarchyNodes.map((node) => (
            <div
              key={node.id}
              className={`hud-tree-node ${node.id === current?.id ? "current" : ""} ${activeIdSet.has(node.id) ? "active" : ""}`}
              style={{ "--agent-depth": node.depth } as React.CSSProperties}
            >
              <span className={`node-light ${node.status}`} />
              <span className="node-name">{agentNameLabel(node.name)}</span>
              <span className="node-state">{nodeStatusLabel(node.status)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="goal-hud game-card" aria-label="任务目标状态">
        <div className="goal-score">
          <span>目标</span>
          <b>{checkpoint.checker ? `${passed}/${predicates}` : `0/${predicates}`}</b>
        </div>
        <div className="goal-copy">
          {checkpoint.goal.predicates.map((predicate, index) => (
            <span key={`${predicate.type}-${index}`}>
              <i className={checkpoint.checker?.checks[index]?.passed ? "passed" : ""} />
              {predicateLabel(predicate)}
            </span>
          ))}
        </div>
      </section>
    </section>
  );
}

export function visibleHierarchyNodes(
  ordered: TaskNode[],
  nodes: Record<string, TaskNode>,
  rootId: string,
  activeIds: string[],
  currentId: string | undefined,
  limit = 9
): TaskNode[] {
  const required = new Set<string>([rootId]);
  for (const id of [...activeIds, ...(currentId ? [currentId] : [])]) {
    let cursor = nodes[id];
    const visited = new Set<string>();
    while (cursor && !visited.has(cursor.id)) {
      required.add(cursor.id);
      visited.add(cursor.id);
      cursor = cursor.parent_id ? nodes[cursor.parent_id] : undefined;
    }
  }
  const selected = new Set(required);
  for (const node of ordered.toReversed()) {
    if (selected.size >= Math.max(limit, required.size)) break;
    selected.add(node.id);
  }
  return ordered.filter((node) => selected.has(node.id));
}

function agentLineage(node: TaskNode, nodes: Record<string, TaskNode>): TaskNode[] {
  const lineage: TaskNode[] = [];
  const visited = new Set<string>();
  let cursor: TaskNode | undefined = node;
  while (cursor && !visited.has(cursor.id)) {
    lineage.unshift(cursor);
    visited.add(cursor.id);
    cursor = cursor.parent_id ? nodes[cursor.parent_id] : undefined;
  }
  return lineage;
}
