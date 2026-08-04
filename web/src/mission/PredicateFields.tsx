import type {
  Bootstrap,
  GoalPredicate,
  HumanoidEndEffector,
  Vec3
} from "../types";
import { entityLabel } from "../ui-text";
import { inputNumber } from "./goal-form";
import {
  degreesToRadians,
  orientationDegreesToQuaternion,
  quaternionToOrientationDegrees,
  radiansToDegrees,
  type OrientationDegrees
} from "./orientation";

interface PredicateFieldsProps {
  predicate: GoalPredicate;
  scenario: Bootstrap["scenarios"][number];
  onChange: (predicate: GoalPredicate) => void;
}

const END_EFFECTOR_OPTIONS: Array<{
  value: HumanoidEndEffector;
  label: string;
}> = [
  { value: "left_wrist", label: "左手腕" },
  { value: "right_wrist", label: "右手腕" },
  { value: "left_ankle", label: "左脚踝" },
  { value: "right_ankle", label: "右脚踝" }
];

const END_EFFECTOR_FRAME_OPTIONS: Array<{
  value: "pelvis" | "world";
  label: string;
}> = [
  { value: "pelvis", label: "骨盆相对" },
  { value: "world", label: "世界坐标" }
];

const GRASP_HAND_OPTIONS: Array<{
  value: Extract<GoalPredicate, { type: "object_grasped" }>["hand"];
  label: string;
}> = [
  { value: "either", label: "任意手" },
  { value: "left", label: "左手" },
  { value: "right", label: "右手" }
];

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
  const portableObjectOptions = scenario.objects
    .filter((object) => object.portable)
    .map((object) => ({
      value: object.id,
      label: entityLabel(object.id)
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
  if (predicate.type === "block_removed") {
    return (
      <FieldStack>
        <TextControl
          label="目标方块"
          value={predicate.block_id}
          onChange={(block_id) => onChange({ ...predicate, block_id })}
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
  if (predicate.type === "object_placed") {
    return (
      <FieldStack>
        <div className="mission-field-pair">
          <SelectControl
            label="目标物体"
            value={predicate.object_id}
            placeholder="选择可搬动物体"
            options={portableObjectOptions}
            onChange={(object_id) => onChange({ ...predicate, object_id })}
          />
          <SelectControl
            label="放置区域"
            value={predicate.zone_id}
            placeholder="选择区域"
            options={zoneOptions}
            onChange={(zone_id) => onChange({ ...predicate, zone_id })}
          />
        </div>
        <ToleranceInput
          value={predicate.tolerance}
          onChange={(tolerance) => onChange({ ...predicate, tolerance })}
        />
      </FieldStack>
    );
  }
  if (predicate.type === "object_grasped") {
    return (
      <FieldStack>
        <div className="mission-field-pair">
          <SelectControl
            label="目标物体"
            value={predicate.object_id}
            placeholder="选择可抓取物体"
            options={portableObjectOptions}
            onChange={(object_id) => onChange({ ...predicate, object_id })}
          />
          <SelectControl
            label="抓取手"
            value={predicate.hand}
            options={GRASP_HAND_OPTIONS}
            onChange={(hand) => onChange({ ...predicate, hand })}
          />
        </div>
      </FieldStack>
    );
  }
  if (predicate.type === "end_effector_at") {
    const orientationEnabled = predicate.orientation !== undefined
      || predicate.orientation_tolerance_rad !== undefined;
    return (
      <FieldStack>
        <div className="mission-field-pair">
          <SelectControl
            label="关键部位"
            value={predicate.end_effector}
            options={END_EFFECTOR_OPTIONS}
            onChange={(end_effector) => onChange({ ...predicate, end_effector })}
          />
          <SelectControl
            label="坐标系"
            value={predicate.frame}
            options={END_EFFECTOR_FRAME_OPTIONS}
            onChange={(frame) => onChange({ ...predicate, frame })}
          />
        </div>
        <CoordinateInput
          label={predicate.frame === "pelvis" ? "骨盆相对坐标" : "世界坐标"}
          value={predicate.target}
          onChange={(target) => onChange({ ...predicate, target })}
        />
        <div className="mission-field-pair mission-field-pair-compact">
          <ToleranceInput
            value={predicate.tolerance}
            onChange={(tolerance) => onChange({ ...predicate, tolerance })}
          />
          <NumberControl
            label="连续稳定帧"
            min={1}
            max={500}
            step={1}
            value={predicate.stable_frames}
            onChange={(stable_frames) => onChange({ ...predicate, stable_frames })}
          />
        </div>
        <div className="mission-field-pair mission-field-pair-compact">
          <CheckControl
            checked={orientationEnabled}
            label="限定末端姿态"
            onChange={(enabled) => onChange(withEndEffectorOrientation(predicate, enabled))}
          />
          {predicate.orientation_tolerance_rad !== undefined && (
            <NumberControl
              label="姿态容差（°）"
              min={0}
              max={180}
              step={1}
              value={rounded(radiansToDegrees(predicate.orientation_tolerance_rad))}
              onChange={(degrees) => onChange({
                ...predicate,
                orientation_tolerance_rad: degreesToRadians(degrees)
              })}
            />
          )}
        </div>
        {predicate.orientation && (
          <OrientationInput
            value={safeOrientationDegrees(predicate.orientation)}
            onChange={(orientation) => onChange({
              ...predicate,
              orientation: orientationDegreesToQuaternion(orientation)
            })}
          />
        )}
      </FieldStack>
    );
  }
  return predicate satisfies never;
}

function OrientationInput(props: {
  value: OrientationDegrees;
  onChange: (value: OrientationDegrees) => void;
}): React.JSX.Element {
  const axes = [
    { key: "roll", label: "侧倾" },
    { key: "pitch", label: "俯仰" },
    { key: "heading", label: "朝向" }
  ] as const;
  return (
    <fieldset className="mission-coordinate-field">
      <legend>目标姿态（°）</legend>
      <div className="mission-coordinate-grid">
        {axes.map((axis) => (
          <NumberControl
            key={axis.key}
            label={axis.label}
            min={-180}
            max={180}
            step={1}
            value={rounded(props.value[axis.key])}
            onChange={(value) => {
              if (Number.isFinite(value)) {
                props.onChange({ ...props.value, [axis.key]: value });
              }
            }}
          />
        ))}
      </div>
    </fieldset>
  );
}

function withEndEffectorOrientation(
  predicate: Extract<GoalPredicate, { type: "end_effector_at" }>,
  enabled: boolean
): Extract<GoalPredicate, { type: "end_effector_at" }> {
  if (enabled) {
    return {
      ...predicate,
      orientation: predicate.orientation ?? { x: 0, y: 0, z: 0, w: 1 },
      orientation_tolerance_rad: predicate.orientation_tolerance_rad ?? degreesToRadians(10)
    };
  }
  const {
    orientation: _orientation,
    orientation_tolerance_rad: _orientationTolerance,
    ...positionOnly
  } = predicate;
  return positionOnly;
}

function safeOrientationDegrees(
  orientation: NonNullable<Extract<GoalPredicate, { type: "end_effector_at" }>["orientation"]>
): OrientationDegrees {
  try {
    return quaternionToOrientationDegrees(orientation);
  } catch {
    return { roll: 0, pitch: 0, heading: 0 };
  }
}

function rounded(value: number): number {
  return Math.round(value * 100) / 100;
}

function FieldStack(props: { children: React.ReactNode }): React.JSX.Element {
  return <div className="predicate-fields">{props.children}</div>;
}

function CoordinateInput(props: {
  label?: string;
  value: Vec3;
  onChange: (value: Vec3) => void;
}): React.JSX.Element {
  return (
    <fieldset className="mission-coordinate-field">
      <legend>{props.label ?? "目标坐标"}</legend>
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

function SelectControl<T extends string>(props: {
  label: string;
  value: T;
  placeholder?: string;
  options: Array<{ value: T; label: string }>;
  onChange: (value: T) => void;
}): React.JSX.Element {
  return (
    <label className="mission-control">
      <span>{props.label}</span>
      <select
        value={props.value}
        onChange={(event) => props.onChange(event.currentTarget.value as T)}
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

function TextControl(props: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}): React.JSX.Element {
  return (
    <label className="mission-control">
      <span>{props.label}</span>
      <input
        type="text"
        value={props.value}
        onChange={(event) => props.onChange(event.currentTarget.value)}
      />
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
