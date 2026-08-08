import { useEffect, useMemo, useRef, useState } from "react";
import type { Goal, HumanoidActionReceipt, HumanoidRunCheckpoint, TaskNode } from "../types";
import { activeContextUsage } from "../context-memory";
import { activeCheckpointGoal, goalSelectionLabel } from "../goal-state";
import {
  agentNameLabel,
  goalSummaryLabel,
  nodePurposeLabel,
  nodeStatusLabel,
  runStatusLabel
} from "../ui-text";
import {
  nodeOutput,
  shortTime
} from "./presenter";
import { buildAgentTree, type AgentTreeEntry } from "./agent-tree";
import {
  presentAutonomousCycles,
  type CycleStageKind,
  type PresentedCycle
} from "./cycle-presenter";

interface AgentFlowViewProps {
  checkpoint: HumanoidRunCheckpoint;
  actions: HumanoidActionReceipt[];
  framework: unknown[];
}

export function AgentFlowView(props: AgentFlowViewProps): React.JSX.Element {
  const { checkpoint } = props;
  const tree = useMemo(
    () => buildAgentTree(checkpoint.nodes, checkpoint.root_id),
    [checkpoint.nodes, checkpoint.root_id]
  );
  const nodes = useMemo(() => tree.map((entry) => entry.node), [tree]);
  const [selectedId, setSelectedId] = useState(checkpoint.active_agent_id ?? checkpoint.root_id);
  const selectionLockedRef = useRef(false);
  useEffect(() => {
    if (!checkpoint.nodes[selectedId]) {
      selectionLockedRef.current = false;
      setSelectedId(checkpoint.active_agent_id ?? checkpoint.root_id);
      return;
    }
    if (!selectionLockedRef.current && checkpoint.active_agent_id) {
      setSelectedId(checkpoint.active_agent_id);
    }
  }, [checkpoint.active_agent_id, checkpoint.nodes, checkpoint.root_id, selectedId]);
  const selected = checkpoint.nodes[selectedId] ?? checkpoint.nodes[checkpoint.root_id] ?? nodes[0];
  const selectedUsage = selected ? checkpoint.model_usage?.by_agent[selected.id] : undefined;
  const goal = activeCheckpointGoal(checkpoint);
  const selectionLabel = goalSelectionLabel(checkpoint);
  const inactiveGoalLabel = checkpoint.status === "succeeded"
    ? "任务已完成"
    : checkpoint.status === "failed"
      ? "运行已结束"
      : selectionLabel ?? "等待本轮目标";
  const cycles = useMemo(() => presentAutonomousCycles({
    checkpoint,
    actions: props.actions,
    framework: props.framework
  }), [
    checkpoint.cycle_index,
    checkpoint.embodied_memory.recent_episodes,
    checkpoint.status,
    props.actions,
    props.framework
  ]);
  const passed = checkpoint.checker?.checks.filter((check) => check.passed).length ?? 0;
  const total = goal?.predicates.length ?? 0;
  const progress = total > 0 ? passed / total : 0;
  const contextMemory = checkpoint.context_memory;
  const contextUsage = activeContextUsage(contextMemory);
  const activeCount = checkpoint.active_agent_ids.length;

  return (
    <section className="agent-flow-view" aria-label="实时层级智能体流">
      <header className="flow-hero">
        <div>
          <span className={`flow-live ${isLive(checkpoint.status) ? "active" : ""}`}>
            <i /> {isLive(checkpoint.status) ? "实时运行中" : runStatusLabel(checkpoint.status)}
          </span>
          <p>{goal ? goalSummaryLabel(goal) : inactiveGoalLabel}</p>
        </div>
        <div className="flow-hero-metrics">
          <div
            className="context-memory-card"
            style={{ "--context-load": `${Math.round(contextUsage.loadFraction * 100)}%` } as React.CSSProperties}
            aria-label={`当前上下文估算为 ${contextUsage.activeEstimatedTokens} 个令牌，上下文窗口为 ${contextUsage.contextWindowTokens} 个令牌，压缩触发线为 ${contextUsage.compactTriggerTokens} 个令牌`}
          >
            <span>上下文</span>
            <strong>
              {compactTokens(contextUsage.activeEstimatedTokens)}
              <em> / {compactTokens(contextUsage.contextWindowTokens)}</em>
            </strong>
            <small>
              压缩线 {compactTokens(contextUsage.compactTriggerTokens)}
              {contextMemory.total_compactions > 0
                ? ` · ${contextMemory.total_compactions} 次`
                : ""}
            </small>
            <i />
          </div>
          <div className="flow-progress" style={{ "--flow-progress": `${Math.min(100, progress * 100)}%` } as React.CSSProperties}>
            <strong>{goal ? formatPercent(progress) : checkpoint.status === "succeeded" ? "完成" : "—"}</strong>
            <span>{goal ? "目标进度" : checkpoint.status === "succeeded" ? "任务状态" : "目标选择"}</span>
          </div>
        </div>
      </header>

      <div className="flow-columns">
        <section className="agent-constellation" aria-label="智能体团队">
          <header className="flow-section-heading">
            <div><span>层级结构</span><b>{activeCount > 0 ? `${activeCount} 个执行中 · ` : ""}{nodes.length} 个智能体</b></div>
          </header>
          <div className="agent-branch-list">
            {tree.map((entry) => (
              <AgentCard
                key={entry.node.id}
                entry={entry}
                goal={goal}
                selected={entry.node.id === selected?.id}
                onSelect={() => {
                  selectionLockedRef.current = true;
                  setSelectedId(entry.node.id);
                }}
              />
            ))}
          </div>
          {selected && (
            <article className="agent-output-card">
              <div className="agent-output-title">
                <span className={`agent-avatar ${selected.status}`}>{initials(agentNameLabel(selected.name))}</span>
                <div><small>{nodeOutput(selected) ? "最新结果" : "当前目标"}</small><b>{agentNameLabel(selected.name)}</b></div>
                <StatusPill status={selected.status} />
              </div>
              <p>{nodeOutput(selected) ?? nodePurposeLabel(selected, goal)}</p>
              <div className="agent-output-meta">
                <span>{selected.model_calls_used} 次模型调用</span>
                {selectedUsage && selectedUsage.reported_requests > 0 && (
                  <span>{compactTokens(selectedUsage.total_tokens)} 模型令牌</span>
                )}
                {selectedUsage && selectedUsage.input_tokens > 0 && (
                  <span>
                    缓存读取 {formatPercent(
                      selectedUsage.cached_input_tokens / selectedUsage.input_tokens
                    )}
                  </span>
                )}
                <span>{selected.steps_used} 次工具决策</span>
                <span>{selected.child_ids.length} 个下级角色</span>
              </div>
            </article>
          )}
        </section>

        <section className="execution-feed" aria-label="智能体执行流">
          <header className="flow-section-heading">
            <div><span>自主循环</span><b>感知到记忆的真实因果链</b></div>
            <span className={`stream-signal ${isLive(checkpoint.status) ? "active" : ""}`}>
              <i /> {isLive(checkpoint.status) ? "实时" : "已完成"}
            </span>
          </header>
          <div className="cycle-list" aria-live="polite">
            {cycles.length === 0 ? (
              <div className="flow-empty">等待第一轮自主决策</div>
            ) : cycles.map((cycle, index) => (
              <CycleCard key={cycle.id} cycle={cycle} expanded={index === 0} />
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}

function AgentCard(props: {
  entry: AgentTreeEntry;
  goal: Goal | null;
  selected: boolean;
  onSelect: () => void;
}): React.JSX.Element {
  const { node } = props.entry;
  return (
    <button
      type="button"
      className={`agent-branch-card ${node.status} ${props.selected ? "selected" : ""}`}
      style={{ "--agent-depth": Math.min(props.entry.depth, 6) } as React.CSSProperties}
      aria-pressed={props.selected}
      onClick={props.onSelect}
    >
      <TreeGuides entry={props.entry} />
      <span className={`agent-avatar ${node.status}`}>{initials(agentNameLabel(node.name))}</span>
      <span className="agent-card-copy">
        <b>{agentNameLabel(node.name)}</b>
        <small>{nodePurposeLabel(node, props.goal)}</small>
      </span>
      <StatusPill status={node.status} />
    </button>
  );
}

function TreeGuides({ entry }: { entry: AgentTreeEntry }): React.JSX.Element | null {
  if (entry.depth === 0) return null;
  return (
    <span className="agent-tree-guides" aria-hidden="true">
      {entry.ancestorContinuations.slice(0, -1).map((continues, index) => (
        <i
          className={continues ? "continues" : ""}
          key={index}
          style={{ "--tree-guide": index } as React.CSSProperties}
        />
      ))}
      <i className={`agent-tree-elbow ${entry.isLastSibling ? "last" : ""}`} />
    </span>
  );
}

function CycleCard(props: { cycle: PresentedCycle; expanded: boolean }): React.JSX.Element {
  const { cycle } = props;
  const [open, setOpen] = useState(props.expanded);
  return (
    <details
      className={`cycle-card ${cycle.state}`}
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <span className="cycle-index">{cycle.index}</span>
        <span className="cycle-summary-copy">
          <b>自主 Cycle {cycle.index}</b>
          <small>{cycleStateLabel(cycle)}</small>
        </span>
        {cycle.at && <time>{shortTime(cycle.at)}</time>}
        <i aria-hidden="true" />
      </summary>
      <div className="cycle-card-body">
        {cycle.liveModelOutput && (
          <div className="cycle-model-output">
            <span>模型输出</span>
            <b>{cycle.liveModelOutput.agent}</b>
            <p>{cycle.liveModelOutput.detail}</p>
          </div>
        )}
        <div className="cycle-stages">
          {cycle.stages.map((stage) => (
            <article className={`cycle-stage ${stage.state}`} key={stage.kind}>
              <span className="cycle-stage-mark">{stageGlyph(stage.kind)}</span>
              <div>
                <header><b>{stage.title}</b><small>{stageStateLabel(stage.state)}</small></header>
                <p>{stage.detail}</p>
                {stage.meta && <small>{stage.meta}</small>}
              </div>
            </article>
          ))}
        </div>
      </div>
    </details>
  );
}

function StatusPill({ status }: { status: TaskNode["status"] }): React.JSX.Element {
  return <span className={`agent-status-pill ${status}`}><i />{nodeStatusLabel(status)}</span>;
}

function initials(name: string): string {
  return name.split(/[_\s-]+/).filter(Boolean).slice(0, 2).map((part) => part[0]?.toUpperCase()).join("") || "智";
}

function stageGlyph(kind: CycleStageKind): string {
  if (kind === "sense") return "◉";
  if (kind === "plan") return "⌁";
  if (kind === "execute") return "➜";
  if (kind === "mutate") return "◆";
  if (kind === "verify") return "✓";
  return "◇";
}

function stageStateLabel(state: PresentedCycle["stages"][number]["state"]): string {
  if (state === "success") return "已确认";
  if (state === "warning") return "未通过";
  if (state === "active") return "进行中";
  return "等待";
}

function cycleStateLabel(cycle: PresentedCycle): string {
  if (cycle.state === "active") return cycle.phaseLabel ?? "正在形成下一步动作";
  if (cycle.state === "interrupted") return "本轮尚未完成";
  if (cycle.goalReached) return "验收通过 · 本轮目标达成";
  return "本轮闭环完成 · 继续自主运行";
}

function isLive(status: HumanoidRunCheckpoint["status"]): boolean {
  return status === "starting" || status === "running";
}

function formatPercent(fraction: number): string {
  const percent = Math.min(100, fraction * 100);
  return `${percent < 10 ? percent.toFixed(1) : Math.round(percent)}%`;
}

function compactTokens(tokens: number): string {
  return new Intl.NumberFormat("zh-CN", {
    notation: "compact",
    maximumFractionDigits: 1
  }).format(tokens);
}
