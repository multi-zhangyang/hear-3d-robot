import type { Bootstrap, Goal, GoalPredicate } from "../types";
import { PredicateFields } from "./PredicateFields";
import { emptyPredicate, predicateOptions } from "./goal-form";

interface GoalEditorProps {
  scenario: Bootstrap["scenarios"][number];
  goal: Goal;
  onChange: (goal: Goal) => void;
}

export function GoalEditor(props: GoalEditorProps): React.JSX.Element {
  const updatePredicate = (index: number, predicate: GoalPredicate): void => {
    const predicates = [...props.goal.predicates];
    predicates[index] = predicate;
    props.onChange({ ...props.goal, predicates });
  };
  const removePredicate = (index: number): void => {
    props.onChange({
      ...props.goal,
      predicates: props.goal.predicates.filter(
        (_predicate, candidateIndex) => candidateIndex !== index
      )
    });
  };
  const addPredicate = (type: GoalPredicate["type"]): void => {
    props.onChange({
      ...props.goal,
      predicates: [...props.goal.predicates, emptyPredicate(type)]
    });
  };

  return (
    <section className="goal-editor" aria-labelledby="goal-editor-title">
      <div className="mission-section-heading">
        <div>
          <span>完成判定</span>
          <h3 id="goal-editor-title">结构化目标</h3>
        </div>
        <label className="predicate-add">
          <span className="sr-only">添加完成条件</span>
          <select
            value=""
            aria-label="添加完成条件"
            onChange={(event) => {
              const type = event.currentTarget.value as GoalPredicate["type"] | "";
              if (type) addPredicate(type);
            }}
          >
            <option value="">＋ 添加条件</option>
            {predicateOptions.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
      </div>

      <label className="mission-control mission-summary-control">
        <span>条件摘要</span>
        <input
          value={props.goal.summary}
          required
          onChange={(event) => props.onChange({
            ...props.goal,
            summary: event.currentTarget.value
          })}
        />
      </label>

      <div className="predicate-list" aria-live="polite">
        {props.goal.predicates.map((predicate, index) => (
          <article className="predicate-row" key={`${predicate.type}-${index}`}>
            <div className="predicate-row-heading">
              <label>
                <span className="sr-only">条件 {index + 1} 类型</span>
                <select
                  aria-label={`条件 ${index + 1} 类型`}
                  value={predicate.type}
                  onChange={(event) => updatePredicate(
                    index,
                    emptyPredicate(event.currentTarget.value as GoalPredicate["type"])
                  )}
                >
                  {predicateOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </label>
              <button
                className="predicate-delete"
                type="button"
                aria-label={`删除条件 ${index + 1}`}
                onClick={() => removePredicate(index)}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M8 8v10m4-10v10m4-10v10M5 5h14M9 5V3h6v2m3 0-1 16H7L6 5" />
                </svg>
              </button>
            </div>
            <PredicateFields
              predicate={predicate}
              scenario={props.scenario}
              onChange={(next) => updatePredicate(index, next)}
            />
          </article>
        ))}
      </div>
      {props.goal.predicates.length === 0 && (
        <p className="predicate-empty">至少添加一个完成条件</p>
      )}
    </section>
  );
}
