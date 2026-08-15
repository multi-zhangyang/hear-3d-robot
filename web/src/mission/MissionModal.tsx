import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import type { Bootstrap, Goal, HumanoidRunMode } from "../types";
import { UiButton } from "../ui/Button";
import { DeferredBoundary } from "../ui/DeferredBoundary";
import { CloseIcon } from "../ui/Icons";
import { LoadingView } from "../ui/LoadingView";
import { goalSummaryLabel, predicateLabel, scenarioLabel } from "../ui-text";
import { validGoal } from "./goal-form";

const loadGoalEditor = () => import("./GoalEditor").then((module) => ({
  default: module.GoalEditor
}));
const GoalEditor = lazy(loadGoalEditor);

interface MissionModalProps {
  open: boolean;
  scenarios: Bootstrap["scenarios"];
  submitting: boolean;
  onCancel: () => void;
  onSubmit: (input: {
    mission: string;
    scenario_id: string;
    goal: Goal;
    run_mode: HumanoidRunMode;
  }) => Promise<void>;
}

export function MissionModal(props: MissionModalProps): React.JSX.Element | null {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const scenarioRef = useRef<HTMLSelectElement>(null);
  const [mission, setMission] = useState("");
  const [scenarioId, setScenarioId] = useState("");
  const [goal, setGoal] = useState<Goal | null>(null);
  const [runMode, setRunMode] = useState<HumanoidRunMode>("continuous");
  const [confirmed, setConfirmed] = useState(false);
  const scenario = useMemo(
    () => props.scenarios.find((candidate) => candidate.id === scenarioId),
    [props.scenarios, scenarioId]
  );
  const valid = mission.trim().length > 0
    && scenario !== undefined
    && goal !== null
    && validGoal(goal)
    && confirmed;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    let focusFrame: number | undefined;
    if (props.open && !dialog.open) {
      dialog.showModal();
      focusFrame = requestAnimationFrame(() => scenarioRef.current?.focus());
    }
    if (!props.open && dialog.open) dialog.close();
    return () => {
      if (focusFrame !== undefined) cancelAnimationFrame(focusFrame);
      if (dialog.open) dialog.close();
    };
  }, [props.open]);

  useEffect(() => {
    if (!props.open) return;
    void loadGoalEditor();
    setMission("");
    setScenarioId("");
    setGoal(null);
    setRunMode("continuous");
    setConfirmed(false);
  }, [props.open]);

  if (!props.open) return null;

  const cancel = (): void => {
    if (!props.submitting) props.onCancel();
  };
  const selectScenario = (value: string): void => {
    const selected = props.scenarios.find((candidate) => candidate.id === value);
    setScenarioId(value);
    if (selected) {
      const suggested = structuredClone(selected.suggested_goal);
      setGoal({ ...suggested, summary: goalSummaryLabel(suggested) });
    } else {
      setGoal(null);
    }
    setConfirmed(false);
  };
  const updateGoal = (value: Goal): void => {
    setGoal(value);
    setConfirmed(false);
  };

  return (
    <dialog
      ref={dialogRef}
      className="mission-dialog"
      aria-labelledby="mission-dialog-title"
      aria-describedby="mission-dialog-state"
      onCancel={(event) => {
        event.preventDefault();
        cancel();
      }}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) cancel();
      }}
    >
      <form
        className="mission-dialog-shell"
        onSubmit={(event) => {
          event.preventDefault();
          if (!valid || !goal || props.submitting) return;
          void props.onSubmit({
            mission: mission.trim(),
            scenario_id: scenarioId,
            goal,
            run_mode: runMode
          });
        }}
      >
        <header className="mission-dialog-header">
          <div>
            <span>自主任务</span>
            <h2 id="mission-dialog-title">新建任务</h2>
          </div>
          <button
            className="mission-dialog-close"
            type="button"
            aria-label="关闭新建任务"
            disabled={props.submitting}
            onClick={cancel}
          >
            <CloseIcon />
          </button>
        </header>

        <div className="mission-dialog-body">
          <span className="sr-only" id="mission-dialog-state">
            {scenario && goal
              ? `已配置 ${goal.predicates.length} 项真实验收条件`
              : "选择世界后配置真实验收条件"}
          </span>
          <div className="mission-basics">
            <label className="mission-control">
              <span>世界场景</span>
              <select
                ref={scenarioRef}
                required
                value={scenarioId}
                onChange={(event) => selectScenario(event.currentTarget.value)}
              >
                <option value="">选择世界场景</option>
                {props.scenarios.map((item) => (
                  <option key={item.id} value={item.id}>
                    {scenarioLabel(item.id, item.title)}
                  </option>
                ))}
              </select>
            </label>
            <fieldset className="mission-run-mode" aria-label="运行模式">
              <legend>运行模式</legend>
              <label className={runMode === "continuous" ? "active" : ""}>
                <input
                  type="radio"
                  name="run-mode"
                  value="continuous"
                  checked={runMode === "continuous"}
                  onChange={() => {
                    setRunMode("continuous");
                    setConfirmed(false);
                  }}
                />
                <span>持续运行</span>
              </label>
              <label className={runMode === "mission" ? "active" : ""}>
                <input
                  type="radio"
                  name="run-mode"
                  value="mission"
                  checked={runMode === "mission"}
                  onChange={() => {
                    setRunMode("mission");
                    setConfirmed(false);
                  }}
                />
                <span>完成后停止</span>
              </label>
            </fieldset>
            <label className="mission-control mission-objective-control">
              <span>任务意图</span>
              <textarea
                aria-label="任务意图"
                rows={3}
                required
                value={mission}
                placeholder="描述机器人要在世界中完成的目标"
                onChange={(event) => {
                  setMission(event.currentTarget.value);
                  setConfirmed(false);
                }}
              />
            </label>
          </div>

          {scenario && goal ? (
            <>
              <DeferredBoundary resetKey={scenario.id}>
                <Suspense fallback={<LoadingView label="正在加载目标编辑器" />}>
                  <GoalEditor scenario={scenario} goal={goal} onChange={updateGoal} />
                </Suspense>
              </DeferredBoundary>
              <MissionContractReview goal={goal} />
            </>
          ) : (
            <div className="mission-goal-placeholder">
              <i aria-hidden="true" />
              <span>选择世界后配置完成条件</span>
            </div>
          )}
        </div>

        <footer className="mission-dialog-footer">
          <label className="mission-confirm">
            <input
              type="checkbox"
              checked={confirmed}
              disabled={!goal || !validGoal(goal)}
              onChange={(event) => setConfirmed(event.currentTarget.checked)}
            />
            <span aria-hidden="true" />
            <b>我确认以上条件是本任务的真实验收标准</b>
          </label>
          <div>
            <UiButton type="button" disabled={props.submitting} onClick={cancel}>
              取消
            </UiButton>
            <UiButton type="submit" tone="primary" busy={props.submitting} disabled={!valid}>
              启动任务
            </UiButton>
          </div>
        </footer>
      </form>
    </dialog>
  );
}

function MissionContractReview({ goal }: { goal: Goal }): React.JSX.Element {
  return (
    <section className="mission-contract-review" aria-labelledby="mission-contract-title">
      <header>
        <div>
          <span>启动核对</span>
          <h3 id="mission-contract-title">真实验收条件</h3>
        </div>
        <b>{goal.predicates.length} 项</b>
      </header>
      <ol>
        {goal.predicates.map((predicate, index) => (
          <li key={`${predicate.type}-${index}`}>
            <span>{index + 1}</span>
            <b>{predicateLabel(predicate)}</b>
          </li>
        ))}
      </ol>
    </section>
  );
}
