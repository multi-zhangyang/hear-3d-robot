import { useMemo } from "react";
import { modelActivityLabel, type ModelActivityState } from "../model-activity";
import type { HumanoidRunCheckpoint } from "../types";
import { missionResultLabel } from "../ui-text";
import { presentFramework, shortTime } from "./presenter";

interface ActivityViewProps {
  checkpoint: HumanoidRunCheckpoint;
  framework: unknown[];
  modelActivity: ModelActivityState;
}

export function ActivityView(props: ActivityViewProps): React.JSX.Element {
  const moments = useMemo(
    () => presentFramework(props.framework).reverse().slice(0, 40),
    [props.framework]
  );
  const output = missionResultLabel(props.checkpoint);
  const calls = props.checkpoint.total_model_calls;
  const failed = props.modelActivity.phase === "error";
  const recovering = props.modelActivity.phase === "recovering";
  const online = props.modelActivity.phase === "active"
    || props.modelActivity.phase === "verified";

  return (
    <section className="activity-view" aria-label="模型输出">
      <header
        className={`activity-pulse ${failed ? "failed" : recovering ? "recovering" : ""}`}
        aria-label="模型流状态"
      >
        <span><i className={online ? "online" : ""} />{modelActivityLabel(props.modelActivity.phase)}</span>
        <strong aria-label={`${calls} 次模型调用`}>{calls}<small>模型调用</small></strong>
      </header>

      {output && (
        <article className="final-story-card">
          <span>任务结果</span>
          <h3>{output}</h3>
        </article>
      )}

      <div className="model-moment-list" aria-live="polite">
        {moments.length === 0 ? (
          <div className="flow-empty">等待模型输出。</div>
        ) : moments.map((moment) => (
          <article className={`model-moment ${moment.tone}`} key={moment.id}>
            <span className="model-orb">✦</span>
            <div>
              <header><b>{moment.agent}</b><time>{shortTime(moment.at)}</time></header>
              <h3>{moment.title}</h3>
              <p>{moment.detail}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
