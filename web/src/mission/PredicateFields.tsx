import type {
  Bootstrap,
  GoalPredicate,
  Vec3
} from "../types";
import { entityLabel } from "../ui-text";
import { inputNumber } from "./goal-form";

interface PredicateFieldsProps {
  predicate: GoalPredicate;
  scenario: Bootstrap["scenarios"][number];
  onChange: (predicate: GoalPredicate) => void;
}

export function PredicateFields(props: PredicateFieldsProps): React.JSX.Element {
  const { predicate, scenario, onChange } = props;
  const objectOptions = scenario.objects.map((object) => ({
    value: object.id,
    label: entityLabel(object.id)
  }));
  const zoneOptions = scenario.zones.map((zone) => ({
    value: zone.id,
    label: entityLabel(zone.id)
  }));

  if (predicate.type === "robot_at") {
    return (
      <FieldStack>
        <CoordinateInput
          value={predicate.target}
          onChange={(target) => onChange({ ...predicate, target })}
        />
        <ToleranceInput
          value={predicate.tolerance}
          onChange={(tolerance) => onChange({ ...predicate, tolerance })}
        />
      </FieldStack>
    );
  }
  if (predicate.type === "robot_in_zone") {
    return (
      <FieldStack>
        <SelectControl
          label="目标区域"
          value={predicate.zone_id}
          placeholder="选择区域"
          options={zoneOptions}
          onChange={(zone_id) => onChange({ ...predicate, zone_id })}
        />
        <ToleranceInput
          value={predicate.tolerance}
          onChange={(tolerance) => onChange({ ...predicate, tolerance })}
        />
      </FieldStack>
    );
  }
  if (predicate.type === "object_at") {
    return (
      <FieldStack>
        <SelectControl
          label="目标物体"
          value={predicate.object_id}
          placeholder="选择物体"
          options={objectOptions}
          onChange={(object_id) => onChange({ ...predicate, object_id })}
        />
        <CoordinateInput
          value={predicate.target}
          onChange={(target) => onChange({ ...predicate, target })}
        />
        <ToleranceInput
          value={predicate.tolerance}
          onChange={(tolerance) => onChange({ ...predicate, tolerance })}
        />
      </FieldStack>
    );
  }
  if (predicate.type === "object_in_zone") {
    return (
      <FieldStack>
        <div className="mission-field-pair">
          <SelectControl
            label="目标物体"
            value={predicate.object_id}
            placeholder="选择物体"
            options={objectOptions}
            onChange={(object_id) => onChange({ ...predicate, object_id })}
          />
          <SelectControl
            label="目标区域"
            value={predicate.zone_id}
            placeholder="选择区域"
            options={zoneOptions}
            onChange={(zone_id) => onChange({ ...predicate, zone_id })}
          />
        </div>
        <div className="mission-field-pair mission-field-pair-compact">
          <CheckControl
            checked={predicate.expected}
            label="必须位于区域内"
            onChange={(expected) => onChange({ ...predicate, expected })}
          />
          <ToleranceInput
            value={predicate.tolerance}
            onChange={(tolerance) => onChange({ ...predicate, tolerance })}
          />
        </div>
      </FieldStack>
    );
  }
  return predicate satisfies never;
}

function FieldStack(props: { children: React.ReactNode }): React.JSX.Element {
  return <div className="predicate-fields">{props.children}</div>;
}

function CoordinateInput(props: {
  value: Vec3;
  onChange: (value: Vec3) => void;
}): React.JSX.Element {
  return (
    <fieldset className="mission-coordinate-field">
      <legend>目标坐标</legend>
      <div className="mission-coordinate-grid">
        {(["x", "y", "z"] as const).map((axis) => (
          <NumberControl
            key={axis}
            label={axis.toUpperCase()}
            step={0.1}
            value={props.value[axis]}
            onChange={(value) => props.onChange({ ...props.value, [axis]: value })}
          />
        ))}
      </div>
    </fieldset>
  );
}

function ToleranceInput(props: {
  value: number;
  onChange: (value: number) => void;
}): React.JSX.Element {
  return (
    <NumberControl
      label="容差"
      min={0}
      step={0.05}
      value={props.value}
      onChange={props.onChange}
    />
  );
}

function SelectControl(props: {
  label: string;
  value: string;
  placeholder?: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
}): React.JSX.Element {
  return (
    <label className="mission-control">
      <span>{props.label}</span>
      <select
        value={props.value}
        onChange={(event) => props.onChange(event.currentTarget.value)}
      >
        {props.placeholder && <option value="">{props.placeholder}</option>}
        {props.options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </label>
  );
}

function NumberControl(props: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step: number;
  onChange: (value: number) => void;
}): React.JSX.Element {
  return (
    <label className="mission-control">
      <span>{props.label}</span>
      <span className="mission-number-wrap">
        <input
          type="number"
          value={Number.isFinite(props.value) ? props.value : ""}
          min={props.min}
          max={props.max}
          step={props.step}
          onChange={(event) => props.onChange(inputNumber(event.currentTarget))}
        />
      </span>
    </label>
  );
}

function CheckControl(props: {
  checked: boolean;
  label: string;
  onChange: (checked: boolean) => void;
}): React.JSX.Element {
  return (
    <label className="mission-check">
      <input
        type="checkbox"
        checked={props.checked}
        onChange={(event) => props.onChange(event.currentTarget.checked)}
      />
      <span aria-hidden="true" />
      <b>{props.label}</b>
    </label>
  );
}
