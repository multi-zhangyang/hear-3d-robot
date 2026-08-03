import { useEffect, useMemo, useState } from "react";
import type { Goal, HumanoidActionReceipt, HumanoidRunCheckpoint, TaskNode } from "../types";
import {
  agentNameLabel,
  goalSummaryLabel,
  nodePurposeLabel,
  nodeStatusLabel,
  runStatusLabel
} from "../ui-text";
import {
  nodeOutput,
  presentAction,
  presentEmbodiedEpisode,
  presentFramework,
  shortTime
} from "./presenter";

interface AgentFlowViewProps {
  checkpoint: HumanoidRunCheckpoint;
  actions: HumanoidActionReceipt[];
  framework: unknown[];
}

export function AgentFlowView(props: AgentFlowViewProps): React.JSX.Element {
  const { checkpoint } = props;
  const nodes = useMemo(
    () => Object.values(checkpoint.nodes).sort((left, right) => left.created_at.localeCompare(right.created_at)),
    [checkpoint.nodes]
  );
  const [selectedId, setSelectedId] = useState(checkpoint.active_agent_id ?? checkpoint.root_id);
  useEffect(() => {
    if (checkpoint.active_agent_id) setSelectedId(checkpoint.active_agent_id);
    else if (!checkpoint.nodes[selectedId]) setSelectedId(checkpoint.root_id);
  }, [checkpoint.active_agent_id, checkpoint.nodes, checkpoint.root_id, selectedId]);
  const selected = checkpoint.nodes[selectedId] ?? checkpoint.nodes[checkpoint.root_id] ?? nodes[0];
  const feed = useMemo(() => [
    ...props.actions.map((action) => ({ ...presentAction(action), kind: "action" as const })),
    ...checkpoint.embodied_memory.recent_episodes.map((episode) => ({
      ...presentEmbodiedEpisode(episode),
      kind: "memory" as const
    })),
    ...presentFramework(props.framework).map((moment) => ({ ...moment, kind: "thought" as const, meta: "模型输出" }))
  ].sort((left, right) => right.at.localeCompare(left.at)).slice(0, 24), [props.actions, props.framework]);
  const passed = checkpoint.checker?.checks.filter((check) => check.passed).length ?? 0;
  const total = checkpoint.goal.predicates.length;
  const progress = total > 0 ? passed / total : 0;
  const contextMemory = checkpoint.context_memory;
  const contextLoad = Math.min(
    1,
    contextMemory.active_estimated_tokens / contextMemory.compact_trigger_tokens
  );
  const activeCount = checkpoint.active_agent_ids.length;

  return (
    <section className="agent-flow-view" aria-label="实时层级智能体流">
      <header className="flow-hero">
        <div>
          <span className={`flow-live ${isLive(checkpoint.status) ? "active" : ""}`}>
            <i /> {isLive(checkpoint.status) ? "实时运行中" : runStatusLabel(checkpoint.status)}
          </span>
          <p>{goalSummaryLabel(checkpoint.goal)}</p>
        </div>
        <div className="flow-hero-metrics">
          <div
            className="context-memory-card"
            style={{ "--context-load": `${Math.round(contextLoad * 100)}%` } as React.CSSProperties}
            aria-label={`当前上下文估算为 ${contextMemory.active_estimated_tokens} 个令牌`}
          >
            <span>上下文</span>
            <strong>{compactTokens(contextMemory.active_estimated_tokens)}</strong>
            <small>{contextMemory.total_compactions > 0
              ? `已压缩 ${contextMemory.total_compactions} 次`
              : "实时记忆"}</small>
            <i />
          </div>
          <div className="flow-progress" style={{ "--flow-progress": `${Math.min(100, progress * 100)}%` } as React.CSSProperties}>
            <strong>{formatPercent(progress)}</strong>
            <span>目标进度</span>
          </div>
        </div>
      </header>

      <div className="flow-columns">
        <section className="agent-constellation" aria-label="智能体团队">
          <header className="flow-section-heading">
            <div><span>层级结构</span><b>{activeCount > 0 ? `${activeCount} 个执行中 · ` : ""}{nodes.length} 个智能体</b></div>
          </header>
          <div className="agent-branch-list">
            {nodes.map((node) => (
              <AgentCard
                key={node.id}
                node={node}
                goal={checkpoint.goal}
                selected={node.id === selected?.id}
                onSelect={() => setSelectedId(node.id)}
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
              <p>{nodeOutput(selected) ?? nodePurposeLabel(selected, checkpoint.goal)}</p>
              <div className="agent-output-meta">
                <span>{selected.model_calls_used} 次模型调用</span>
                <span>{selected.steps_used} 次工具决策</span>
                <span>{selected.child_ids.length} 个下级角色</span>
              </div>
            </article>
          )}
        </section>

        <section className="execution-feed" aria-label="智能体执行流">
          <header className="flow-section-heading">
            <div><span>实时活动</span><b>模型决策与物理回执</b></div>
            <span className={`stream-signal ${isLive(checkpoint.status) ? "active" : ""}`}>
              <i /> {isLive(checkpoint.status) ? "实时" : "已完成"}
            </span>
          </header>
          <div className="execution-items" aria-live="polite">
            {feed.length === 0 ? (
              <div className="flow-empty">等待模型决策</div>
            ) : feed.map((item) => (
              <article className={`execution-item ${item.tone}`} key={item.id}>
                <span className="execution-mark">{item.kind === "thought"
                  ? "✦"
                  : item.kind === "memory" ? "◇" : actionGlyph(item.category)}</span>
                <div className="execution-copy">
                  <div><b>{item.title}</b><time>{shortTime(item.at)}</time></div>
                  <span>{item.agent}</span>
                  <p>{item.detail}</p>
                  <small>{item.meta}</small>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}

function AgentCard(props: { node: TaskNode; goal: Goal; selected: boolean; onSelect: () => void }): React.JSX.Element {
  return (
    <button
      type="button"
      className={`agent-branch-card ${props.node.status} ${props.selected ? "selected" : ""}`}
      style={{ "--agent-depth": Math.min(props.node.depth, 6) } as React.CSSProperties}
      aria-pressed={props.selected}
      onClick={props.onSelect}
    >
      <span className={`agent-avatar ${props.node.status}`}>{initials(agentNameLabel(props.node.name))}</span>
      <span className="agent-card-copy">
        <b>{agentNameLabel(props.node.name)}</b>
        <small>{nodePurposeLabel(props.node, props.goal)}</small>
      </span>
      <StatusPill status={props.node.status} />
    </button>
  );
}

function StatusPill({ status }: { status: TaskNode["status"] }): React.JSX.Element {
  return <span className={`agent-status-pill ${status}`}><i />{nodeStatusLabel(status)}</span>;
}

function initials(name: string): string {
  return name.split(/[_\s-]+/).filter(Boolean).slice(0, 2).map((part) => part[0]?.toUpperCase()).join("") || "智";
}

function actionGlyph(category: string): string {
  if (category === "sense") return "◉";
  if (category === "plan") return "⌁";
  if (category === "move") return "➜";
  return "✓";
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
