import { memo, useEffect, useRef, useState } from "react";
import type { AuthoritativeFrameBuffer } from "./stage/authoritative-frame-buffer";
import type { StageController } from "./stage/stage-types";
import type { FirstPersonStatus } from "./stage/stage-interaction";
import { useAuthoritativeHudFrame } from "./stage/use-authoritative-frame";
import { WorldHud, type CameraMode } from "./stage/WorldHud";
import type { WorldSelection } from "./stage/world-selection";
import type { ScenarioDefinition, WorldSnapshot } from "./types";

interface RobotStageProps {
  initialFrame: WorldSnapshot;
  frameBuffer: AuthoritativeFrameBuffer;
  scenario: ScenarioDefinition;
  live: boolean;
}

/** React owns the HUD; the stage controller owns the Three.js lifecycle. */
function RobotStageComponent({
  initialFrame,
  frameBuffer,
  scenario,
  live
}: RobotStageProps): React.JSX.Element {
  const [cameraMode, setCameraMode] = useState<CameraMode>("robot");
  const [stageFailure, setStageFailure] = useState<"load" | "render" | null>(null);
  const [selection, setSelection] = useState<WorldSelection | null>(null);
  const [firstPerson, setFirstPerson] = useState<FirstPersonStatus>({
    available: false,
    locked: false,
    touch: false
  });
  const [stageLoading, setStageLoading] = useState(true);
  const stageRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<StageController | null>(null);
  const liveRef = useRef(live);
  const cameraModeRef = useRef(cameraMode);
  liveRef.current = live;
  cameraModeRef.current = cameraMode;
  const hudFrame = useAuthoritativeHudFrame(frameBuffer, initialFrame);
  const bounds = scenario.bounds;

  useEffect(() => {
    if (!stageRef.current) return;
    const container = stageRef.current;
    let disposed = false;
    let controller: StageController | null = null;
    const initialization = new AbortController();
    setStageLoading(true);
    setStageFailure(null);

    void import("./stage/create-stage")
      .then(async ({ createStage }) => {
        if (disposed) return;
        controller = await createStage(
          container,
          scenario,
          initialFrame,
          frameBuffer,
          liveRef.current,
          {
            onError: (message) => {
              if (!disposed) setStageFailure(message ? "render" : null);
            },
            onSelection: (value) => {
              if (!disposed) setSelection(value);
            },
            onFirstPersonStatus: (value) => {
              if (!disposed) setFirstPerson(value);
            }
          },
          initialization.signal
        );
        if (disposed) {
          controller.dispose();
          return;
        }
        controller.setCameraMode(cameraModeRef.current);
        sceneRef.current = controller;
        setStageLoading(false);
      })
      .catch(() => {
        if (disposed) return;
        setStageLoading(false);
        setStageFailure("load");
      });

    return () => {
      disposed = true;
      initialization.abort();
      controller?.dispose();
      if (sceneRef.current === controller) sceneRef.current = null;
    };
  }, [bounds.depth, bounds.width, frameBuffer, scenario.seed]);

  useEffect(() => sceneRef.current?.setLive(live), [live]);
  useEffect(() => sceneRef.current?.setCameraMode(cameraMode), [cameraMode]);

  return (
    <section className="robot-workspace" aria-label={live ? "实时机器人世界" : "机器人任务回顾"}>
      <div className="three-stage" ref={stageRef}>
        {stageLoading && <div className="stage-loading" role="status" aria-label="正在加载三维世界"><i /><i /><i /></div>}
        {stageFailure && (
          <div className="graphics-error" role="alert">
            {stageFailure === "load"
              ? "3D 场景加载失败，请刷新后重试。"
              : "3D 场景不可用，请检查浏览器图形加速。"}
          </div>
        )}
        <WorldHud
          frame={hudFrame}
          live={live}
          cameraMode={cameraMode}
          selection={selection}
          firstPerson={firstPerson}
          onCameraMode={setCameraMode}
          onFit={() => sceneRef.current?.fit()}
          onClearSelection={() => sceneRef.current?.clearSelection()}
        />
      </div>
    </section>
  );
}

export const RobotStage = memo(RobotStageComponent);
