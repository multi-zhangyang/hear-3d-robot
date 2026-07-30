import type { WorldSnapshot } from "../types";
import { focusedCommand, liveCommands } from "../active-commands";
import type { FirstPersonStatus } from "./stage-interaction";
import { describeWorldSelection, type WorldSelection } from "./world-selection";
import { actionLabel, agentNameLabel } from "../ui-text";

export type CameraMode = "robot" | "sensor" | "world";

interface WorldHudProps {
  frame: WorldSnapshot;
  live: boolean;
  cameraMode: CameraMode;
  selection: WorldSelection | null;
  firstPerson: FirstPersonStatus;
  onCameraMode: (mode: CameraMode) => void;
  onFit: () => void;
  onClearSelection: () => void;
}

export function WorldHud(props: WorldHudProps): React.JSX.Element {
  const active = liveCommands(props.frame);
  const command = focusedCommand(props.frame);
  const activeAgents = [...new Set(active.map((entry) => entry.agent_name))];
  const base = props.frame.robot.links.base;
  const speed = base
    ? Math.hypot(base.linear_velocity.x, base.linear_velocity.z)
    : 0;
  const mapped = props.frame.explored.total === 0
    ? null
    : props.frame.explored.seen / props.frame.explored.total * 100;
  const heading = normalizeDegrees(props.frame.robot.yaw * 180 / Math.PI);
  const selected = props.selection
    ? describeWorldSelection(props.selection, props.frame)
    : null;

  return (
    <div className="world-hud" aria-label="世界运动状态">
      <div className="stage-statusbar game-card">
        <span className={active.length > 0 ? "command-pulse moving" : "command-pulse"} />
        <span className="command-name">
          {active.length > 1
            ? `${active.length} 条身体指令`
            : command ? `${active.length > 0 ? "" : "最近："}${actionLabel(command.skill)}` : "身体空闲"}
        </span>
        <b>{activeAgents.length > 0
          ? activeAgents.slice(0, 2).map(agentNameLabel).join(" + ")
          : command?.agent_name ? agentNameLabel(command.agent_name) : props.live ? "等待智能体" : "任务已结束"}</b>
      </div>

      <div className="camera-bar game-card" role="group" aria-label="相机模式">
        <CameraButton label="跟随视角" shortLabel="跟随" mode="robot" current={props.cameraMode} onSelect={props.onCameraMode} />
        <CameraButton label="第一人称视角" shortLabel="主观" mode="sensor" current={props.cameraMode} onSelect={props.onCameraMode} />
        <CameraButton label="全局视角" shortLabel="全局" mode="world" current={props.cameraMode} onSelect={props.onCameraMode} />
        <button type="button" className="camera-button fit" aria-label="适配相机范围" onClick={props.onFit}>
          适配
        </button>
      </div>

      {props.cameraMode === "sensor" && (
        <>
          <div className="sensor-reticle" aria-hidden="true">
            <i />
            <i />
          </div>
          {props.firstPerson.available && (
            <div
              className={props.firstPerson.locked ? "first-person-status locked" : "first-person-status"}
              aria-label="第一人称鼠标锁定"
            >
              {props.firstPerson.locked ? "移动鼠标观察 · ESC 退出" : "单击进入视角"}
            </div>
          )}
        </>
      )}

      {selected && (
        <div className="world-selection-card game-card" aria-label="已选择的世界目标" aria-live="polite">
          <span>{selected.badge}</span>
          <div><b>{selected.title}</b><small>{selected.detail}</small></div>
          <button type="button" aria-label="清除已选目标" title="清除选择" onClick={props.onClearSelection}>×</button>
        </div>
      )}

      <div className="compass-hud game-card" aria-label="机器人朝向">
        <span>西</span><span>北</span><b>{heading.toFixed(0)}°</b><span>东</span><span>南</span>
      </div>

      <div className="world-state-strip game-card">
        <span className={props.live ? "live-dot" : "archive-dot"} />
        <HudSlot label="位置" value={`${props.frame.robot.position.x.toFixed(1)}, ${props.frame.robot.position.z.toFixed(1)}`} />
        <HudSlot label="朝向" value={`${heading.toFixed(0)}°`} />
        <HudSlot label="速度" value={`${speed.toFixed(2)} 米/秒`} active={speed > 0.01} />
        <HudSlot label="探索率" value={mapped === null ? "暂无" : `${mapped.toFixed(1)}%`} />
      </div>
    </div>
  );
}

function CameraButton(props: {
  label: string;
  shortLabel?: string;
  mode: CameraMode;
  current: CameraMode;
  onSelect: (mode: CameraMode) => void;
}): React.JSX.Element {
  return (
    <button
      type="button"
      className={props.current === props.mode ? "camera-button active" : "camera-button"}
      aria-label={props.label}
      aria-pressed={props.current === props.mode}
      onClick={() => props.onSelect(props.mode)}
    >
      {props.shortLabel ?? props.label}
    </button>
  );
}

function HudSlot(props: {
  label: string;
  value: string;
  active?: boolean;
}): React.JSX.Element {
  return (
    <span className={props.active ? "world-slot active" : "world-slot"}>
      <small>{props.label}</small>
      <b>{props.value}</b>
    </span>
  );
}

function normalizeDegrees(value: number): number {
  return ((value % 360) + 360) % 360;
}
