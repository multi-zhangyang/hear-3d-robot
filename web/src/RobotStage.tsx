import { memo, useEffect, useRef, useState } from "react";
import { createStage } from "./stage/create-stage";
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
  const [webglError, setWebglError] = useState<string | null>(null);
  const [selection, setSelection] = useState<WorldSelection | null>(null);
  const [firstPerson, setFirstPerson] = useState<FirstPersonStatus>({
    available: false,
    locked: false
  });
  const stageRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<StageController | null>(null);
  const hudFrame = useAuthoritativeHudFrame(frameBuffer, initialFrame);
  const bounds = scenario.bounds;

  useEffect(() => {
    if (!stageRef.current) return;
    try {
      const controller = createStage(
        stageRef.current,
        scenario,
        initialFrame,
        frameBuffer,
        live,
        {
          onError: setWebglError,
          onSelection: setSelection,
          onFirstPersonStatus: setFirstPerson
        }
      );
      sceneRef.current = controller;
      return () => {
        controller.dispose();
        sceneRef.current = null;
      };
    } catch (error) {
      setWebglError(error instanceof Error ? error.message : String(error));
      return;
    }
  }, [bounds.depth, bounds.width, frameBuffer, scenario.seed]);

  useEffect(() => sceneRef.current?.setLive(live), [live]);
  useEffect(() => sceneRef.current?.setCameraMode(cameraMode), [cameraMode]);

  return (
    <section className="robot-workspace" aria-label={live ? "实时机器人世界" : "机器人任务回顾"}>
      <div className="three-stage" ref={stageRef}>
        {webglError && <div className="webgl-error" role="alert">3D 场景不可用，请检查浏览器的 WebGL 支持。</div>}
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
