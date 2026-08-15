import { useEffect, useMemo, useRef, useState } from "react";
import type {
  HumanoidActionReceipt,
  HumanoidRunCheckpoint,
  NeuralHarnessPhase,
  TaskNode
} from "../types";
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
import { buildAgentTree } from "./agent-tree";
import { HierarchyGraph } from "./HierarchyGraph";
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
  const inactiveGoalLabel = selectionLabel ?? "等待本轮目标";
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
  const harnessPhase = checkpoint.neural_hierarchy_state?.harness_phase;

  return (
    <section className="agent-flow-view" aria-label="实时层级智能体流">
      <header className="flow-hero">
        <div>
          <span className={`flow-live ${isLive(checkpoint.status) ? "active" : ""}`}>
            <i /> {isLive(checkpoint.status) ? "实时运行中" : runStatusLabel(checkpoint.status)}
          </span>
          {(goal || isLive(checkpoint.status)) && (
            <p>{goal ? goalSummaryLabel(goal) : inactiveGoalLabel}</p>
          )}
        </div>
        <div className="flow-hero-metrics">
          {harnessPhase && isLive(checkpoint.status) && (
            <div className="harness-phase-card">
              <span>阶段</span>
              <strong>{harnessPhaseLabel(harnessPhase.phase)}</strong>
            </div>
          )}
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
            <i />
          </div>
          <div className="flow-progress" style={{ "--flow-progress": `${Math.min(100, progress * 100)}%` } as React.CSSProperties}>
            <strong>{goal ? formatPercent(progress) : checkpoint.status === "succeeded" ? "完成" : "—"}</strong>
            <span>{goal ? "目标" : "状态"}</span>
          </div>
        </div>
      </header>

      <div className="flow-columns">
        <section className="agent-constellation" aria-label="智能体团队">
          <header className="flow-section-heading">
            <div><span>控制层级</span><b>{nodes.length} 节点{activeCount > 0 ? ` · ${activeCount} 活跃` : ""}</b></div>
            {selected ? (
              <small className="flow-selection-summary">
                <b>{agentNameLabel(selected.name)}</b>
              </small>
            ) : null}
          </header>
          <HierarchyGraph
            nodes={checkpoint.nodes}
            rootId={checkpoint.root_id}
            activeIds={checkpoint.active_agent_ids}
            selectedId={selected?.id ?? null}
            goal={goal}
            onSelect={(nodeId) => {
              selectionLockedRef.current = true;
              setSelectedId(nodeId);
            }}
          />
          {selected && (
            <article className="agent-output-card">
              <div className="agent-output-title">
                <span className={`agent-avatar ${selected.status}`}>{initials(agentNameLabel(selected.name))}</span>
                <div><b>{agentNameLabel(selected.name)}</b></div>
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
              </div>
            </article>
          )}
        </section>

        <section className="execution-feed" aria-label="智能体执行流">
          <header className="flow-section-heading">
            <div><span>执行</span><b>自主闭环</b></div>
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
          <b>闭环 {cycle.index}</b>
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

function harnessPhaseLabel(phase: NeuralHarnessPhase): string {
  if (phase === "bootstrapping") return "初始化";
  if (phase === "goal_valuation") return "目标估值";
  if (phase === "perception") return "感知";
  if (phase === "skill_proposal") return "技能提案";
  if (phase === "commitment_authorization") return "承诺授权";
  if (phase === "motor_assessment") return "动作评估";
  if (phase === "motor_planning") return "运动规划";
  if (phase === "rollout_review") return "预演评估";
  if (phase === "execution") return "物理执行";
  if (phase === "feedback") return "闭环反馈";
  if (phase === "recovery") return "自主恢复";
  if (phase === "cycle_completion") return "周期验收";
  return "已终止";
}
