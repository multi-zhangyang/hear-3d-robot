import { useMemo } from "react";
import { latestProviderActivity } from "../stream-state";
import type { HumanoidRunCheckpoint } from "../types";
import { missionResultLabel } from "../ui-text";
import { presentFramework, shortTime } from "./presenter";

interface ActivityViewProps {
  checkpoint: HumanoidRunCheckpoint;
  framework: unknown[];
  provider: unknown[];
}

export function ActivityView(props: ActivityViewProps): React.JSX.Element {
  const moments = useMemo(
    () => presentFramework(props.framework).reverse().slice(0, 40),
    [props.framework]
  );
  const output = missionResultLabel(props.checkpoint);
  const calls = props.checkpoint.total_model_calls;
  const live = props.checkpoint.status === "starting" || props.checkpoint.status === "running";
  const activity = latestProviderActivity(props.provider);
  const usable = activity?.status === "usable_stream";
  const failed = activity?.status === "no_text"
    || activity?.status === "transport_interrupted"
    || activity?.status.includes("error") === true;
  const state = failed
    ? "模型调用异常"
    : live && activity ? "模型调用中"
      : usable ? "模型已响应" : "模型等待中";

  return (
    <section className="activity-view" aria-label="模型输出">
      <header className={`activity-pulse ${failed ? "failed" : ""}`} aria-label="模型流状态">
        <span><i className={usable || live && activity ? "online" : ""} />{state}</span>
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
