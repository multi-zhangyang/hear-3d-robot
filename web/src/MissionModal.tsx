import "antd/dist/reset.css";
import { DeleteOutlined, PlusOutlined, RocketOutlined } from "@ant-design/icons";
import {
  Button,
  Checkbox,
  ConfigProvider,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Space,
  Tooltip,
  Typography,
  theme
} from "antd";
import zhCN from "antd/locale/zh_CN";
import { useEffect, useMemo, useState } from "react";
import type {
  Bootstrap,
  Goal,
  GoalPredicate,
  Vec3,
  VoxelCoordinate,
  VoxelMaterial
} from "./types";
import { entityLabel, goalSummaryLabel, scenarioLabel } from "./ui-text";

interface MissionModalProps {
  open: boolean;
  scenarios: Bootstrap["scenarios"];
  submitting: boolean;
  onCancel: () => void;
  onSubmit: (input: { mission: string; scenario_id: string; goal: Goal }) => Promise<void>;
}

export function MissionModal(props: MissionModalProps): React.JSX.Element {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: "#24c8ae",
          colorInfo: "#8e7be8",
          colorSuccess: "#24c8ae",
          colorWarning: "#e6a64b",
          colorError: "#ef6a6a",
          colorBgBase: "#101214",
          colorBgContainer: "#191c1f",
          colorBgElevated: "#22262a",
          colorBorder: "#34393e",
          colorBorderSecondary: "#292d31",
          colorText: "#f0f1f2",
          colorTextSecondary: "#a7adb3",
          colorTextTertiary: "#737a81",
          borderRadius: 6,
          wireframe: false,
          fontFamily: '"Space Grotesk Variable", Inter, ui-sans-serif, system-ui, sans-serif',
          fontFamilyCode: '"JetBrains Mono Variable", "SFMono-Regular", Consolas, monospace'
        }
      }}
    >
      <MissionModalContent {...props} />
    </ConfigProvider>
  );
}

function MissionModalContent(props: MissionModalProps): React.JSX.Element {
  const [mission, setMission] = useState("");
  const [scenarioId, setScenarioId] = useState("");
  const [goal, setGoal] = useState<Goal | null>(null);
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
    if (!props.open) return;
    setMission("");
    setScenarioId("");
    setGoal(null);
    setConfirmed(false);
  }, [props.open]);

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
    <Modal
      title="新建任务"
      open={props.open}
      onCancel={props.onCancel}
      destroyOnHidden
      width={720}
      footer={[
        <Button key="cancel" onClick={props.onCancel}>取消</Button>,
        <Button
          key="start"
          type="primary"
          icon={<RocketOutlined />}
          loading={props.submitting}
          disabled={!valid}
          onClick={() => goal && void props.onSubmit({ mission: mission.trim(), scenario_id: scenarioId, goal })}
        >
          启动任务
        </Button>
      ]}
    >
      <Form layout="vertical" className="mission-form">
        <Form.Item label="世界场景" required>
          <Select
            value={scenarioId || null}
            placeholder="选择世界场景"
            options={props.scenarios.map((item) => ({ value: item.id, label: scenarioLabel(item.id, item.title) }))}
            onChange={selectScenario}
          />
        </Form.Item>
        <Form.Item label="任务目标" required>
          <Input.TextArea
            rows={3}
            value={mission}
            onChange={(event) => {
              setMission(event.target.value);
              setConfirmed(false);
            }}
          />
        </Form.Item>
        {scenario && goal && (
          <GoalEditor scenario={scenario} goal={goal} onChange={updateGoal} />
        )}
        <Checkbox checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)}>
          确认以上完成条件
        </Checkbox>
      </Form>
    </Modal>
  );
}

function GoalEditor(props: {
  scenario: Bootstrap["scenarios"][number];
  goal: Goal;
  onChange: (goal: Goal) => void;
}): React.JSX.Element {
  const updatePredicate = (index: number, predicate: GoalPredicate): void => {
    const predicates = [...props.goal.predicates];
    predicates[index] = predicate;
    props.onChange({ ...props.goal, predicates });
  };
  const removePredicate = (index: number): void => {
    props.onChange({
      ...props.goal,
      predicates: props.goal.predicates.filter((_predicate, candidateIndex) => candidateIndex !== index)
    });
  };
  return (
    <div className="goal-editor">
      <Form.Item label="完成条件摘要" required>
        <Input
          value={props.goal.summary}
          onChange={(event) => props.onChange({ ...props.goal, summary: event.target.value })}
        />
      </Form.Item>
      <div className="predicate-heading">
        <Typography.Text strong>结构化完成条件</Typography.Text>
        <Select<GoalPredicate["type"]>
          className="predicate-add"
          value={null}
          placeholder="添加条件"
          suffixIcon={<PlusOutlined />}
          options={predicateOptions}
          onChange={(type) => props.onChange({
            ...props.goal,
            predicates: [...props.goal.predicates, emptyPredicate(type)]
          })}
        />
      </div>
      <div className="predicate-list">
        {props.goal.predicates.map((predicate, index) => (
          <div className="predicate-row" key={`${predicate.type}-${index}`}>
            <div className="predicate-row-heading">
              <Select<GoalPredicate["type"]>
                value={predicate.type}
                options={predicateOptions}
                onChange={(type) => updatePredicate(index, emptyPredicate(type))}
              />
              <Tooltip title="删除条件">
                <Button
                  aria-label="删除条件"
                  danger
                  type="text"
                  icon={<DeleteOutlined />}
                  onClick={() => removePredicate(index)}
                />
              </Tooltip>
            </div>
            <PredicateFields
              predicate={predicate}
              scenario={props.scenario}
              onChange={(next) => updatePredicate(index, next)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function PredicateFields(props: {
  predicate: GoalPredicate;
  scenario: Bootstrap["scenarios"][number];
  onChange: (predicate: GoalPredicate) => void;
}): React.JSX.Element {
  const { predicate, scenario, onChange } = props;
  const objectOptions = scenario.objects.map((object) => ({ value: object.id, label: entityLabel(object.id) }));
  const zoneOptions = scenario.zones.map((zone) => ({ value: zone.id, label: entityLabel(zone.id) }));
  if (predicate.type === "robot_at") {
    return (
      <Space direction="vertical" size={8} className="predicate-fields">
        <CoordinateInput value={predicate.target} onChange={(target) => onChange({ ...predicate, target })} />
        <ToleranceInput value={predicate.tolerance} onChange={(tolerance) => onChange({ ...predicate, tolerance })} />
      </Space>
    );
  }
  if (predicate.type === "robot_in_zone") {
    return (
      <Space direction="vertical" size={8} className="predicate-fields">
        <Select
          value={predicate.zone_id || null}
          placeholder="选择区域"
          options={zoneOptions}
          onChange={(zone_id) => onChange({ ...predicate, zone_id })}
        />
        <ToleranceInput value={predicate.tolerance} onChange={(tolerance) => onChange({ ...predicate, tolerance })} />
      </Space>
    );
  }
  if (predicate.type === "terrain_explored") {
    return (
      <InputNumber
        min={1}
        max={100}
        step={1}
        addonAfter="% 已探索"
        value={Math.round(predicate.minimum_fraction * 100)}
        onChange={(value) => onChange({
          ...predicate,
          minimum_fraction: Math.min(1, Math.max(0.01, Number(value ?? 1) / 100))
        })}
      />
    );
  }
  if (predicate.type === "voxel_at") {
    return (
      <Space direction="vertical" size={8} className="predicate-fields">
        <VoxelCoordinateInput
          value={predicate.coordinate}
          onChange={(coordinate) => onChange({ ...predicate, coordinate })}
        />
        <Select<VoxelMaterial | "air">
          value={predicate.material ?? "air"}
          options={[
            { value: "air", label: "空" },
            { value: "grass", label: "草方块" },
            { value: "dirt", label: "泥土方块" },
            { value: "stone", label: "石头方块" },
            { value: "sand", label: "沙方块" },
            { value: "placed", label: "已放置方块" }
          ]}
          onChange={(material) => onChange({
            ...predicate,
            material: material === "air" ? null : material
          })}
        />
      </Space>
    );
  }
  if (predicate.type === "object_at") {
    return (
      <Space direction="vertical" size={8} className="predicate-fields">
        <Select
          value={predicate.object_id || null}
          placeholder="选择物体"
          options={objectOptions}
          onChange={(object_id) => onChange({ ...predicate, object_id })}
        />
        <CoordinateInput value={predicate.target} onChange={(target) => onChange({ ...predicate, target })} />
        <ToleranceInput value={predicate.tolerance} onChange={(tolerance) => onChange({ ...predicate, tolerance })} />
      </Space>
    );
  }
  if (predicate.type === "object_in_zone") {
    return (
      <Space direction="vertical" size={8} className="predicate-fields">
        <Space.Compact block>
          <Select
            value={predicate.object_id || null}
            placeholder="选择物体"
            options={objectOptions}
            onChange={(object_id) => onChange({ ...predicate, object_id })}
          />
          <Select
            value={predicate.zone_id || null}
            placeholder="选择区域"
            options={zoneOptions}
            onChange={(zone_id) => onChange({ ...predicate, zone_id })}
          />
        </Space.Compact>
        <Space size={16} wrap>
          <Checkbox checked={predicate.expected} onChange={(event) => onChange({ ...predicate, expected: event.target.checked })}>
            必须位于区域内
          </Checkbox>
          <ToleranceInput value={predicate.tolerance} onChange={(tolerance) => onChange({ ...predicate, tolerance })} />
        </Space>
      </Space>
    );
  }
  if (predicate.type === "object_property") {
    return (
      <Space direction="vertical" size={8} className="predicate-fields">
        <Space.Compact block>
          <Select
            value={predicate.object_id || null}
            placeholder="选择物体"
            options={objectOptions}
            onChange={(object_id) => onChange({ ...predicate, object_id })}
          />
          <Select
            value={predicate.property}
            options={[
              { value: "locked", label: "已锁定" },
              { value: "enabled", label: "已启用" }
            ]}
            onChange={(property) => onChange({ ...predicate, property })}
          />
        </Space.Compact>
        <Checkbox checked={predicate.expected} onChange={(event) => onChange({ ...predicate, expected: event.target.checked })}>
          期望该属性为真
        </Checkbox>
      </Space>
    );
  }
  return (
    <Space direction="vertical" size={8} className="predicate-fields">
      <Select
        value={predicate.object_id || null}
        placeholder="选择物体"
        options={objectOptions}
        onChange={(object_id) => onChange({ ...predicate, object_id })}
      />
      <Checkbox checked={predicate.expected} onChange={(event) => onChange({ ...predicate, expected: event.target.checked })}>
        必须被夹爪抓取
      </Checkbox>
    </Space>
  );
}

function CoordinateInput(props: { value: Vec3; onChange: (value: Vec3) => void }): React.JSX.Element {
  return (
    <Space.Compact block>
      {(["x", "y", "z"] as const).map((axis) => (
        <InputNumber
          key={axis}
          addonBefore={axis.toUpperCase()}
          value={props.value[axis]}
          step={0.1}
          onChange={(value) => props.onChange({ ...props.value, [axis]: value ?? 0 })}
        />
      ))}
    </Space.Compact>
  );
}

function VoxelCoordinateInput(props: {
  value: VoxelCoordinate;
  onChange: (value: VoxelCoordinate) => void;
}): React.JSX.Element {
  return (
    <Space.Compact block>
      {(["column", "level", "row"] as const).map((axis) => (
        <InputNumber
          key={axis}
          addonBefore={axis === "column" ? "X" : axis === "level" ? "Y" : "Z"}
          min={0}
          precision={0}
          value={props.value[axis]}
          onChange={(value) => props.onChange({
            ...props.value,
            [axis]: Math.max(0, Math.trunc(Number(value ?? 0)))
          })}
        />
      ))}
    </Space.Compact>
  );
}

function ToleranceInput(props: { value: number; onChange: (value: number) => void }): React.JSX.Element {
  return (
    <InputNumber
      aria-label="容差"
      addonBefore="容差"
      min={0}
      step={0.05}
      value={props.value}
      onChange={(value) => props.onChange(value ?? 0)}
    />
  );
}

const predicateOptions = [
  { value: "robot_at", label: "机器人到达坐标" },
  { value: "robot_in_zone", label: "机器人进入区域" },
  { value: "terrain_explored", label: "地形探索率" },
  { value: "voxel_at", label: "指定坐标的方块材质" },
  { value: "object_in_zone", label: "物体位于区域" },
  { value: "object_at", label: "物体到达坐标" },
  { value: "object_property", label: "物体属性" },
  { value: "object_attached", label: "物体抓取状态" }
] satisfies Array<{ value: GoalPredicate["type"]; label: string }>;

function emptyPredicate(type: GoalPredicate["type"]): GoalPredicate {
  if (type === "robot_at") return { type, target: { x: 0, y: 0, z: 0 }, tolerance: 0.25 };
  if (type === "robot_in_zone") return { type, zone_id: "", tolerance: 0.2 };
  if (type === "terrain_explored") return { type, minimum_fraction: 0.05 };
  if (type === "voxel_at") {
    return { type, coordinate: { column: 0, level: 0, row: 0 }, material: "placed" };
  }
  if (type === "object_in_zone") return { type, object_id: "", zone_id: "", expected: true, tolerance: 0.05 };
  if (type === "object_at") return { type, object_id: "", target: { x: 0, y: 0, z: 0 }, tolerance: 0.1 };
  if (type === "object_property") return { type, object_id: "", property: "enabled", expected: true };
  return { type, object_id: "", expected: true };
}

function validGoal(goal: Goal): boolean {
  return goal.summary.trim().length > 0
    && goal.predicates.length > 0
    && goal.predicates.every(validPredicate);
}

function validPredicate(predicate: GoalPredicate): boolean {
  if (predicate.type === "robot_at") return finiteVec3(predicate.target) && predicate.tolerance > 0;
  if (predicate.type === "robot_in_zone") return predicate.zone_id.length > 0 && predicate.tolerance >= 0;
  if (predicate.type === "terrain_explored") {
    return predicate.minimum_fraction > 0 && predicate.minimum_fraction <= 1;
  }
  if (predicate.type === "voxel_at") {
    return Number.isInteger(predicate.coordinate.column)
      && Number.isInteger(predicate.coordinate.level)
      && Number.isInteger(predicate.coordinate.row)
      && predicate.coordinate.column >= 0
      && predicate.coordinate.level >= 0
      && predicate.coordinate.row >= 0;
  }
  if (predicate.type === "object_at") {
    return predicate.object_id.length > 0 && finiteVec3(predicate.target) && predicate.tolerance > 0;
  }
  if (predicate.type === "object_in_zone") {
    return predicate.object_id.length > 0 && predicate.zone_id.length > 0 && predicate.tolerance >= 0;
  }
  return predicate.object_id.length > 0;
}

function finiteVec3(value: Vec3): boolean {
  return Number.isFinite(value.x) && Number.isFinite(value.y) && Number.isFinite(value.z);
}
