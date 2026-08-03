import { memo, useEffect, useRef, useState } from "react";
import type { HumanoidFrameBuffer } from "../stage/humanoid-frame-buffer";
import type { HumanoidRunDetails } from "../types";
import { useHumanoidHudFrame } from "./use-humanoid-frame";
import type { HumanoidCameraMode, HumanoidStageController } from "./create-humanoid-stage";

interface HumanoidStageProps {
  details: HumanoidRunDetails;
  frameBuffer: HumanoidFrameBuffer;
  live: boolean;
}

function HumanoidStageComponent(props: HumanoidStageProps): React.JSX.Element {
  const [mode, setMode] = useState<HumanoidCameraMode>("follow");
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState(false);
  const hostRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<HumanoidStageController | null>(null);
  const modeRef = useRef(mode);
  const liveRef = useRef(props.live);
  modeRef.current = mode;
  liveRef.current = props.live;
  const frame = useHumanoidHudFrame(props.frameBuffer, props.details.checkpoint.world);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let disposed = false;
    const abort = new AbortController();
    let controller: HumanoidStageController | null = null;
    let startTimer = 0;
    setLoading(true);
    setFailure(false);
    const startFrame = window.requestAnimationFrame(() => {
      startTimer = window.setTimeout(() => {
        void import("./create-humanoid-stage")
          .then(({ createHumanoidStage }) => createHumanoidStage(
            host,
            props.details.definition.scenario,
            props.details.checkpoint.world,
            props.frameBuffer,
            liveRef.current,
            (message) => {
              if (!disposed) setFailure(message !== null);
            },
            abort.signal
          ))
          .then((created) => {
            if (disposed) {
              created.dispose();
              return;
            }
            controller = created;
            controllerRef.current = created;
            created.setCameraMode(modeRef.current);
            setLoading(false);
          })
          .catch((error) => {
            if (disposed || error instanceof DOMException && error.name === "AbortError") return;
            setLoading(false);
            setFailure(true);
          });
      }, 0);
    });
    return () => {
      disposed = true;
      window.cancelAnimationFrame(startFrame);
      window.clearTimeout(startTimer);
      abort.abort();
      controller?.dispose();
      if (controllerRef.current === controller) controllerRef.current = null;
    };
  }, [props.details.definition.run_id, props.details.definition.scenario, props.frameBuffer]);

  useEffect(() => controllerRef.current?.setLive(props.live), [props.live]);
  useEffect(() => controllerRef.current?.setCameraMode(mode), [mode]);

  return (
    <section className="humanoid-stage" aria-label={props.live ? "实时人形世界" : "人形世界回顾"}>
      <div className="three-stage" ref={hostRef}>
        {loading && <div className="stage-loading" role="status" aria-label="正在载入 G1"><i /><i /><i /></div>}
        {failure && <div className="graphics-error" role="alert">三维人形场景不可用</div>}
        <div className="humanoid-stage-bar">
          <div className="camera-switch" role="group" aria-label="观察视角">
            {(["follow", "world", "head"] as const).map((value) => (
              <button
                type="button"
                key={value}
                className={mode === value ? "active" : ""}
                aria-pressed={mode === value}
                onClick={() => setMode(value)}
              >
                {value === "follow" ? "跟随" : value === "world" ? "世界" : "头部"}
              </button>
            ))}
          </div>
          <button type="button" className="fit-world" onClick={() => controllerRef.current?.fit()}>复位视角</button>
        </div>
        <div className={`humanoid-physics-strip ${frame.robot.fallen ? "danger" : ""}`}>
          <span><small>帧</small><b>{frame.frame.toLocaleString("zh-CN")}</b></span>
          <span><small>支撑</small><b>{supportLabel(frame.robot.balance.support)}</b></span>
          <span><small>接触</small><b>{frame.robot.contactCount}</b></span>
          <span><small>直立</small><b>{Math.round(frame.robot.balance.upright * 100)}%</b></span>
        </div>
      </div>
    </section>
  );
}

function supportLabel(value: "double" | "left" | "right" | "none"): string {
  if (value === "double") return "双脚";
  if (value === "left") return "左脚";
  if (value === "right") return "右脚";
  return "腾空";
}

export const HumanoidStage = memo(HumanoidStageComponent);
