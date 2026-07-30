import {
  PlusOutlined,
  StopOutlined
} from "@ant-design/icons";
import {
  Button,
  ConfigProvider,
  Empty,
  Result,
  Select,
  Space,
  message,
  theme
} from "antd";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  getBootstrap,
  getRun,
  getRuns,
  hasPassword,
  resumeRun,
  setPassword,
  startRun,
  stopRun,
  subscribeToRun
} from "./api";
import { ActivityView } from "./flow/ActivityView";
import { AgentFlowView } from "./flow/AgentFlowView";
import { RobotTrailView } from "./flow/RobotTrailView";
import { MissionModal } from "./MissionModal";
import { MissionWorkspace } from "./MissionWorkspace";
import { GameShell, type ModelConnectionState, type Workspace } from "./game/GameShell";
import { OverlayPanel } from "./game/OverlayPanel";
import { CenteredSpin, FailureAlert, Login, RunStatus } from "./Shell";
import { reduceRunDetails } from "./run-details-reducer";
import { AuthoritativeFrameBuffer } from "./stage/authoritative-frame-buffer";
import {
  appendRecent,
  asRecord,
  isAbortError,
  latestProviderActivity,
  providerActivityFrom,
  updateRunListStatus,
  worldSnapshotsFrom
} from "./stream-state";
import type {
  Bootstrap,
  Goal,
  ProviderActivity,
  RunDetails,
  RunListItem,
  RuntimeEvent,
  StreamState,
  WorldSnapshot
} from "./types";
import { runOptionLabel, runStatusLabel } from "./ui-text";

const FRAMEWORK_HISTORY_LIMIT = 300;
const PROVIDER_HISTORY_LIMIT = 400;
const ACTION_HISTORY_LIMIT = 500;
export function App(): React.JSX.Element {
  const [api, contextHolder] = message.useMessage();
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [details, setDetails] = useState<RunDetails | null>(null);
  const [framework, setFramework] = useState<unknown[]>([]);
  const [workspace, setWorkspace] = useState<Workspace>("world");
  const [loading, setLoading] = useState(true);
  const [authRequired, setAuthRequired] = useState(false);
  const [fatalError, setFatalError] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [streamState, setStreamState] = useState<StreamState>("inactive");
  const [streamEpoch, setStreamEpoch] = useState(0);
  const [providerActivity, setProviderActivity] = useState<ProviderActivity | null>(null);
  const [missionOpen, setMissionOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [stoppingRunId, setStoppingRunId] = useState<string | null>(null);
  const selectedRunRef = useRef<string | null>(null);
  const eventCursorRef = useRef<{ runId: string; eventId?: string } | null>(null);
  const eventFloorRef = useRef<{ runId: string; at: string } | null>(null);
  const loadGenerationRef = useRef(0);
  const loadAbortRef = useRef<AbortController | null>(null);
  const frameBufferRef = useRef<AuthoritativeFrameBuffer | null>(null);
  frameBufferRef.current ??= new AuthoritativeFrameBuffer();
  const frameBuffer = frameBufferRef.current;

  useEffect(() => {
    selectedRunRef.current = selectedRunId;
  }, [selectedRunId]);

  const loadRun = useCallback(async (runId: string): Promise<boolean> => {
    const generation = loadGenerationRef.current + 1;
    loadGenerationRef.current = generation;
    loadAbortRef.current?.abort();
    const controller = new AbortController();
    loadAbortRef.current = controller;
    const isCurrent = (): boolean => !controller.signal.aborted
      && loadGenerationRef.current === generation
      && selectedRunRef.current === runId;

    setLoading(true);
    try {
      const nextDetails = await getRun(runId, {
        signal: controller.signal,
        actionLimit: ACTION_HISTORY_LIMIT,
        providerLimit: PROVIDER_HISTORY_LIMIT,
        frameworkLimit: FRAMEWORK_HISTORY_LIMIT
      });
      if (!isCurrent()) return false;

      setDetails(nextDetails);
      eventCursorRef.current = {
        runId,
        ...(nextDetails.event_cursor ? { eventId: nextDetails.event_cursor } : {})
      };
      eventFloorRef.current = { runId, at: nextDetails.checkpoint.updated_at };
      frameBuffer.reset(nextDetails.checkpoint.world);
      setFramework(nextDetails.framework);
      setProviderActivity(latestProviderActivity(nextDetails.provider));
      setStreamEpoch((current) => current + 1);
      if (nextDetails.checkpoint.status !== "starting" && nextDetails.checkpoint.status !== "running") {
        setStoppingRunId((current) => current === runId ? null : current);
      }
      return true;
    } catch (error) {
      if (!isCurrent() || isAbortError(error)) return false;
      throw error;
    } finally {
      if (loadGenerationRef.current === generation) {
        if (loadAbortRef.current === controller) loadAbortRef.current = null;
        setLoading(false);
      }
    }
  }, [frameBuffer]);

  const refreshRuns = useCallback(async (preferredId?: string): Promise<void> => {
    const nextRuns = await getRuns();
    setRuns(nextRuns);
    const candidate = preferredId
      ?? selectedRunRef.current
      ?? nextRuns.find((run) => run.status !== "local_artifact")?.run_id
      ?? null;
    if (candidate) {
      if (selectedRunRef.current !== candidate) {
        eventCursorRef.current = null;
        eventFloorRef.current = null;
        setDetails(null);
        setFramework([]);
        setProviderActivity(null);
      }
      selectedRunRef.current = candidate;
      setSelectedRunId(candidate);
      await loadRun(candidate);
    }
  }, [loadRun]);

  const initialize = useCallback(async (): Promise<void> => {
    setLoading(true);
    setFatalError(null);
    try {
      const nextBootstrap = await getBootstrap();
      setBootstrap(nextBootstrap);
      setAuthRequired(false);
      await refreshRuns();
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) setAuthRequired(true);
      else setFatalError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }, [refreshRuns]);

  useEffect(() => {
    void initialize();
  }, [initialize]);

  useEffect(() => () => {
    loadGenerationRef.current += 1;
    loadAbortRef.current?.abort();
  }, []);

  const appendWorldFrames = useCallback((incoming: WorldSnapshot[]): void => {
    if (incoming.length === 0) return;
    frameBuffer.push(incoming);
  }, [frameBuffer]);

  const handleRuntimeEvent = useCallback((runId: string, event: RuntimeEvent): void => {
    if (selectedRunRef.current !== runId) return;
    eventCursorRef.current = { runId, eventId: event.event_id };
    const floor = eventFloorRef.current;
    // Events at or before the snapshot the details were loaded from are already
    // reflected in that snapshot; replaying them would double-count steps.
    const historical = floor?.runId === runId && event.at <= floor.at;

    if (event.type === "framework_event") {
      setFramework((current) => appendRecent(current, event.data, FRAMEWORK_HISTORY_LIMIT));
    }

    if (event.type === "provider_event" && !historical) {
      const activity = providerActivityFrom(event.data);
      if (activity) setProviderActivity(activity);
    }

    const worlds = historical ? [] : worldSnapshotsFrom(event.data);
    if (worlds.length > 0) appendWorldFrames(worlds);
    const terminalEvent = event.type === "run_succeeded"
      || event.type === "run_failed"
      || event.type === "run_interrupted";
    const reducerWorlds = !historical && terminalEvent && worlds.length === 0 && frameBuffer.latest
      ? [frameBuffer.latest]
      : worlds;

    // The stage and its local HUD consume high-frequency physics batches from
    // the imperative buffer. Folding a frame-only event through App state would
    // reconcile the entire shell and every open agent panel for no semantic UI
    // change. Receipt, hierarchy and terminal events still update durable React
    // state below their much lower event cadence.
    if (event.type !== "world_frames") {
      setDetails((current) => current === null ? current : reduceRunDetails({
        details: current,
        event,
        worlds: reducerWorlds,
        historical,
        limits: { actions: ACTION_HISTORY_LIMIT, provider: PROVIDER_HISTORY_LIMIT }
      }));
    }

    if (historical) return;
    // The run list is a sibling of the details, not part of them, so terminal
    // transitions still have to be mirrored onto it here.
    if (event.type === "run_started" || event.type === "run_resumed") {
      setRuns((current) => updateRunListStatus(current, runId, "running", null, event.at));
      return;
    }
    if (event.type === "run_succeeded" || event.type === "run_failed" || event.type === "run_interrupted") {
      const data = asRecord(event.data);
      const error = event.type === "run_succeeded"
        ? null
        : typeof data?.error === "string"
          ? data.error
          : typeof data?.reason === "string" ? data.reason : "运行已结束，但服务端未提供原因";
      const status = event.type === "run_succeeded"
        ? "succeeded"
        : event.type === "run_failed" ? "failed" : "interrupted";
      setRuns((current) => updateRunListStatus(current, runId, status, error, event.at));
      setStoppingRunId((currentId) => currentId === runId ? null : currentId);
      setStreamState("inactive");
    }
  }, [appendWorldFrames, frameBuffer]);

  useEffect(() => {
    if (!selectedRunId || details?.definition.run_id !== selectedRunId) {
      setStreamState("inactive");
      return;
    }
    if (details.checkpoint.status !== "starting" && details.checkpoint.status !== "running") {
      // The details response already contains the durable terminal checkpoint
      // and bounded journal tails. A currently terminal run has nothing new to
      // stream, so replaying its full events journal only blocks WebGL.
      setStreamState("inactive");
      return;
    }
    const runId = selectedRunId;
    const generation = loadGenerationRef.current;
    const acceptsStreamUpdate = (): boolean => selectedRunRef.current === runId
      && loadGenerationRef.current === generation;
    const cursor = eventCursorRef.current?.runId === runId
      ? eventCursorRef.current.eventId
      : undefined;
    setStreamError(null);
    return subscribeToRun(
      runId,
      (event) => {
        if (acceptsStreamUpdate()) handleRuntimeEvent(runId, event);
      },
      () => {
        if (acceptsStreamUpdate()) setStreamError("实时事件流连接已断开");
      },
      (state) => {
        if (!acceptsStreamUpdate()) return;
        setStreamState(state);
        if (state === "connected") setStreamError(null);
      },
      cursor
    );
  }, [details?.checkpoint.status, details?.definition.run_id, handleRuntimeEvent, selectedRunId, streamEpoch]);

  const selectRun = async (runId: string): Promise<void> => {
    selectedRunRef.current = runId;
    eventCursorRef.current = null;
    eventFloorRef.current = null;
    setDetails(null);
    setFramework([]);
    setProviderActivity(null);
    setSelectedRunId(runId);
    try {
      await loadRun(runId);
    } catch {
      if (selectedRunRef.current === runId) {
        api.error("无法加载所选运行记录，请稍后重试。");
      }
    }
  };

  const createMission = async (input: {
    mission: string;
    scenario_id: string;
    goal: Goal;
  }): Promise<void> => {
    setSubmitting(true);
    try {
      const runId = await startRun(input);
      setMissionOpen(false);
      setWorkspace("world");
      await refreshRuns(runId);
    } catch {
      api.error("任务启动失败，请检查模型配置和任务条件。");
    } finally {
      setSubmitting(false);
    }
  };

  const resumeSelected = async (): Promise<void> => {
    if (!selectedRunId || !details || resuming
      || (details.checkpoint.status !== "failed" && details.checkpoint.status !== "interrupted")) return;
    const previousStatus = details.checkpoint.status;
    const previousError = details.checkpoint.error;
    const optimisticAt = new Date().toISOString();
    setResuming(true);
    setDetails((current) => current && current.definition.run_id === selectedRunId
      ? {
          ...current,
          checkpoint: {
            ...current.checkpoint,
            status: "starting",
            error: null,
            updated_at: optimisticAt
          }
        }
      : current);
    setRuns((current) => updateRunListStatus(current, selectedRunId, "starting", null, optimisticAt));
    try {
      await resumeRun(selectedRunId);
    } catch {
      setDetails((current) => current && current.definition.run_id === selectedRunId
        ? {
            ...current,
            checkpoint: {
              ...current.checkpoint,
              status: previousStatus,
              error: previousError
            }
          }
        : current);
      setRuns((current) => updateRunListStatus(
        current,
        selectedRunId,
        previousStatus,
        previousError,
        details.checkpoint.updated_at
      ));
      api.error("任务恢复失败，请查看运行状态后重试。");
    } finally {
      setResuming(false);
    }
  };

  const stopSelected = async (): Promise<void> => {
    if (!selectedRunId || stoppingRunId !== null || resuming || submitting) return;
    setStoppingRunId(selectedRunId);
    try {
      await stopRun(selectedRunId);
    } catch {
      setStoppingRunId(null);
      api.error("停止任务失败，请稍后重试。");
    }
  };

  if (loading && !bootstrap && !authRequired) return <CenteredSpin />;
  if (authRequired) {
    return (
      <Login
        hasStoredPassword={hasPassword()}
        onLogin={async (value) => {
          setPassword(value);
          try {
            const nextBootstrap = await getBootstrap();
            setBootstrap(nextBootstrap);
            setAuthRequired(false);
            await refreshRuns();
          } catch (error) {
            setPassword("");
            throw error;
          }
        }}
      />
    );
  }
  if (fatalError || !bootstrap) {
    return <Result
      status="error"
      title="操作服务不可用"
      subTitle="无法连接操作服务，请确认服务已启动并检查网络配置。"
      extra={<Button onClick={() => void initialize()}>重试</Button>}
    />;
  }

  const selectedRun = runs.find((run) => run.run_id === selectedRunId);
  const activeRun = runs.find((run) => run.status === "starting" || run.status === "running");
  const selectedIsActive = details?.checkpoint.status === "starting" || details?.checkpoint.status === "running";
  const missionControlsBusy = submitting || resuming || stoppingRunId !== null;
  const modelState = modelConnectionState(
    bootstrap.provider.configured,
    providerActivity?.status ?? null,
    selectedIsActive
  );
  return (
    <ConfigProvider
      theme={{
        // Neutral graphite keeps the hierarchy and telemetry legible; channel
        // colours are reserved for actual robot control state.
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
        },
        components: { Tabs: { cardBg: "#22262a" } }
      }}
    >
      <GameShell
        workspace={workspace}
        onWorkspace={setWorkspace}
        modelState={modelState}
        onRefresh={() => void refreshRuns()}
        onLogout={hasPassword()
          ? () => {
              setPassword("");
              setAuthRequired(true);
            }
          : null}
        toolbar={(
          <div className="operator-toolbar">
            <Space size={10} wrap className="run-context">
              <Select<string>
                className="run-select"
                value={selectedRunId}
                aria-label="选择运行记录"
                placeholder="选择运行记录"
                onChange={(value) => {
                  if (value) void selectRun(value);
                }}
                options={runs.map((run) => ({
                  value: run.run_id,
                  label: runOptionLabel(run),
                  disabled: run.status === "local_artifact"
                }))}
              />
              {selectedRun && <RunStatus status={selectedRun.status} />}
            </Space>
            <Space className="mission-actions">
              {details && selectedIsActive && (
                <Button
                  danger
                  icon={<StopOutlined />}
                  loading={stoppingRunId === selectedRunId}
                  disabled={resuming || submitting || (stoppingRunId !== null && stoppingRunId !== selectedRunId)}
                  onClick={() => void stopSelected()}
                >
                  停止
                </Button>
              )}
              {details && ["failed", "interrupted"].includes(details.checkpoint.status) && (
                <Button
                  loading={resuming}
                  disabled={missionControlsBusy || activeRun !== undefined}
                  onClick={() => void resumeSelected()}
                >
                  继续运行
                </Button>
              )}
              <Button
                type="primary"
                icon={<PlusOutlined />}
                disabled={missionControlsBusy || activeRun !== undefined || !bootstrap.provider.configured}
                onClick={() => setMissionOpen(true)}
              >
                新建任务
              </Button>
            </Space>
          </div>
        )}
      >
        {contextHolder}
        <div className="game-content">

          {!bootstrap.provider.configured && !loading && !details && (
            <FailureAlert title="模型服务不可用" error={bootstrap.provider.error} />
          )}
          {streamError && (
            <FailureAlert title="实时事件流连接失败" error={streamError} onClose={() => setStreamError(null)} />
          )}
          {details?.checkpoint.error && details.checkpoint.status !== "succeeded" && (
            <FailureAlert title={`运行状态：${runStatusLabel(details.checkpoint.status)}`} error={details.checkpoint.error} />
          )}

          {loading ? <CenteredSpin /> : !details ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description="尚未选择运行记录"
            >
              <Button
                type="primary"
                icon={<PlusOutlined />}
                disabled={missionControlsBusy || activeRun !== undefined || !bootstrap.provider.configured}
                onClick={() => setMissionOpen(true)}
              >
                新建任务
              </Button>
            </Empty>
          ) : (
            <WorkspaceView
              workspace={workspace}
              details={details}
              frameBuffer={frameBuffer}
              framework={framework}
              streamState={streamState}
              onClose={() => setWorkspace("world")}
            />
          )}
        </div>
        <MissionModal
          open={missionOpen}
          scenarios={bootstrap.scenarios}
          submitting={submitting}
          onCancel={() => setMissionOpen(false)}
          onSubmit={createMission}
        />
      </GameShell>
    </ConfigProvider>
  );
}

function WorkspaceView(props: {
  workspace: Workspace;
  details: RunDetails;
  frameBuffer: AuthoritativeFrameBuffer;
  framework: unknown[];
  streamState: StreamState;
  onClose: () => void;
}): React.JSX.Element {
  const world = (
    <MissionWorkspace
      key={props.details.definition.run_id}
      details={props.details}
      frameBuffer={props.frameBuffer}
      streamState={props.streamState}
    />
  );
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
              provider={props.details.provider}
              framework={props.framework}
            />
          )
        };
  return (
    <div className="game-world-stack">
      {world}
      {panel && (
        <OverlayPanel title={panel.title} onClose={props.onClose}>
          {panel.body}
        </OverlayPanel>
      )}
    </div>
  );
}

function modelConnectionState(
  configured: boolean,
  status: string | null,
  runIsActive: boolean
): ModelConnectionState {
  if (!runIsActive && status === "usable_stream") return "verified";
  if (!configured) return "offline";
  if (status === null || status === "configured") return "ready";
  if (status === "no_text" || status.includes("error") || status === "transport_interrupted") return "error";
  if (status === "usable_stream") return runIsActive ? "active" : "verified";
  if (status === "contacted" || status === "streaming_text") return runIsActive ? "active" : "ready";
  return "ready";
}
