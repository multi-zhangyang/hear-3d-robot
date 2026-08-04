import { useMemo, useState } from "react";
import type { HumanoidActionReceipt } from "../types";
import { presentAction, receiptFrames, shortTime, type ActionCategory } from "./presenter";

const FILTERS: Array<{ key: "all" | ActionCategory; label: string }> = [
  { key: "all", label: "全部" },
  { key: "move", label: "身体运动" },
  { key: "mutate", label: "世界变化" },
  { key: "plan", label: "规划" },
  { key: "sense", label: "感知" },
  { key: "verify", label: "目标检查" }
];

export function RobotTrailView({ actions }: { actions: HumanoidActionReceipt[] }): React.JSX.Element {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]["key"]>("all");
  const items = useMemo(
    () => actions.map(presentAction).reverse().filter((item) => filter === "all" || item.category === filter),
    [actions, filter]
  );
  const moved = actions.filter((action) => action.accepted && receiptFrames(action) > 0).length;
  const blocked = actions.filter((action) => !action.accepted).length;
  const physicalFrames = actions.reduce((sum, action) => sum + Math.max(0, receiptFrames(action)), 0);

  return (
    <section className="robot-trail-view" aria-label="机器人行动历程">
      <header className="trail-hero">
        <div><span>机器人行动历程</span><h2>动作轨迹</h2></div>
        <div className="trail-summary">
          <Summary value={moved} label="运动" />
          <Summary value={physicalFrames.toLocaleString("zh-CN")} label="物理帧" />
          <Summary value={blocked} label="已拒绝" warning={blocked > 0} />
        </div>
      </header>
      <nav className="story-filters" aria-label="动作筛选">
        {FILTERS.map((option) => (
          <button
            type="button"
            key={option.key}
            className={filter === option.key ? "active" : ""}
            aria-pressed={filter === option.key}
            onClick={() => setFilter(option.key)}
          >
            {option.label}
          </button>
        ))}
      </nav>
      <div className="trail-grid">
        {items.length === 0 ? <div className="flow-empty">当前筛选条件下暂无动作。</div> : items.map((item) => (
          <article className={`trail-card ${item.tone}`} key={item.id}>
            <div className="trail-icon">{glyph(item.category)}</div>
            <div className="trail-copy">
              <div><span>{item.agent}</span><time>{shortTime(item.at)}</time></div>
              <h3>{item.title}</h3>
              <p>{item.detail}</p>
              <footer>
                <small>{item.meta}</small>
                {item.channels.map((channel) => <i key={channel}>{channel}</i>)}
              </footer>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function Summary(props: { value: number | string; label: string; warning?: boolean }): React.JSX.Element {
  return <span className={props.warning ? "warning" : ""}><b>{props.value}</b><small>{props.label}</small></span>;
}

function glyph(category: ActionCategory): string {
  if (category === "sense") return "◉";
  if (category === "plan") return "⌁";
  if (category === "move") return "➜";
  if (category === "mutate") return "◆";
  return "✓";
}
