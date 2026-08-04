import { lazy, Suspense } from "react";
import type { HumanoidFrameBuffer } from "../stage/humanoid-frame-buffer";
import type { ModelActivityState } from "../model-activity";
import type { HumanoidRunDetails, StreamState } from "../types";
import { DeferredBoundary } from "../ui/DeferredBoundary";
import { LoadingView } from "../ui/LoadingView";
import type { Workspace } from "./GameShell";
import { OverlayPanel } from "./OverlayPanel";

const HumanoidMissionWorkspace = lazy(() => import("../humanoid/HumanoidMissionWorkspace").then((module) => ({
  default: module.HumanoidMissionWorkspace
})));
const ActivityView = lazy(() => import("../flow/ActivityView").then((module) => ({
  default: module.ActivityView
})));
const AgentFlowView = lazy(() => import("../flow/AgentFlowView").then((module) => ({
  default: module.AgentFlowView
})));
const RobotTrailView = lazy(() => import("../flow/RobotTrailView").then((module) => ({
  default: module.RobotTrailView
})));

interface WorkspaceViewProps {
  workspace: Workspace;
  details: HumanoidRunDetails;
  humanoidFrameBuffer: HumanoidFrameBuffer;
  framework: unknown[];
  modelActivity: ModelActivityState;
  streamState: StreamState;
  onClose: () => void;
}

export function WorkspaceView(props: WorkspaceViewProps): React.JSX.Element {
  const runId = props.details.definition.run_id;
  const panel = props.workspace === "world" ? null : props.workspace === "flow"
    ? {
        title: "智能体流",
        body: (
          <AgentFlowView
            checkpoint={props.details.checkpoint}
            actions={props.details.actions}
            framework={props.framework}
          />
        )
      }
    : props.workspace === "journey"
      ? {
          title: "行动历程",
          body: <RobotTrailView actions={props.details.actions} />
        }
      : {
          title: "智能体输出",
          body: (
            <ActivityView
              checkpoint={props.details.checkpoint}
              modelActivity={props.modelActivity}
              framework={props.framework}
            />
          )
        };
  return (
    <div className="game-world-stack">
      <DeferredBoundary resetKey={runId}>
        <Suspense fallback={<LoadingView label="正在载入人形世界" />}>
          <HumanoidMissionWorkspace
            key={runId}
            details={props.details}
            frameBuffer={props.humanoidFrameBuffer}
            streamState={props.streamState}
          />
        </Suspense>
      </DeferredBoundary>
      {panel && (
        <OverlayPanel title={panel.title} onClose={props.onClose}>
          <DeferredBoundary resetKey={`${runId}:${props.workspace}`}>
            <Suspense fallback={<LoadingView label="正在加载视图" />}>
              {panel.body}
            </Suspense>
          </DeferredBoundary>
        </OverlayPanel>
      )}
    </div>
  );
}
