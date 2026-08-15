import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  downloadRunTelemetry,
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
import { GameShell, type Workspace } from "./game/GameShell";
import { CenteredSpin, FailureAlert, Login, RunStatus } from "./Shell";
import { reduceHumanoidRunDetails } from "./humanoid-run-details-reducer";
import { HumanoidFrameBuffer } from "./stage/humanoid-frame-buffer";
import {
  asRecord,
  isAbortError,
  nextRuntimeEventCursor,
  providerActivityFrom,
  humanoidWorldSnapshotsFrom,
  upsertRuntimeJournalEntry,
  updateRunListStatus
} from "./stream-state";
import {
  createModelActivity,
  modelActivityFromJournal,
  reduceProviderActivity,
  reduceRuntimeModelActivity,
  settleModelActivity,
  type ModelActivityState
} from "./model-activity";
import type {
  Bootstrap,
  Goal,
  HumanoidRunMode,
  HumanoidRunDetails,
  HumanoidWorldSnapshot,
  RunListItem,
  RuntimeEvent,
  StreamState
} from "./types";
import { UiButton } from "./ui/Button";
import { DeferredBoundary } from "./ui/DeferredBoundary";
import { runOptionLabel, runStatusLabel } from "./ui-text";
import { MissionModal } from "./MissionModal";

const FRAMEWORK_HISTORY_LIMIT = 300;
const PROVIDER_HISTORY_LIMIT = 400;
const ACTION_HISTORY_LIMIT = 500;

interface LoadRunOptions {
  preserveStream?: boolean;
}

const WorkspaceView = lazy(() => import("./game/WorkspaceView").then((module) => ({
  default: module.WorkspaceView
})));

export function App(): React.JSX.Element {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [details, setDetails] = useState<HumanoidRunDetails | null>(null);
  const [framework, setFramework] = useState<unknown[]>([]);
  const [workspace, setWorkspace] = useState<Workspace>("world");
  const [loading, setLoading] = useState(true);
  const [authRequired, setAuthRequired] = useState(false);
  const [fatalError, setFatalError] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [streamState, setStreamState] = useState<StreamState>("inactive");
  const [streamEpoch, setStreamEpoch] = useState(0);
  const [modelActivity, setModelActivity] = useState<ModelActivityState>(
    () => createModelActivity(true)
  );
  const [missionOpen, setMissionOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [stoppingRunId, setStoppingRunId] = useState<string | null>(null);
  const [exportingRunId, setExportingRunId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const selectedRunRef = useRef<string | null>(null);
  const eventCursorRef = useRef<{ runId: string; eventId?: string } | null>(null);
  const loadGenerationRef = useRef(0);
  const loadAbortRef = useRef<AbortController | null>(null);
  const humanoidFrameBufferRef = useRef<HumanoidFrameBuffer | null>(null);
  humanoidFrameBufferRef.current ??= new HumanoidFrameBuffer();
  const humanoidFrameBuffer = humanoidFrameBufferRef.current;
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showError = useCallback((value: string): void => {
    if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    setNotice(value);
    noticeTimerRef.current = setTimeout(() => {
      noticeTimerRef.current = null;
      setNotice(null);
    }, 4_500);
  }, []);

  useEffect(() => {
    selectedRunRef.current = selectedRunId;
  }, [selectedRunId]);

  const loadRun = useCallback(async (
    runId: string,
    options: LoadRunOptions = {}
  ): Promise<HumanoidRunDetails | null> => {
    const generation = options.preserveStream
      ? loadGenerationRef.current
      : loadGenerationRef.current + 1;
    if (!options.preserveStream) loadGenerationRef.current = generation;
    loadAbortRef.current?.abort();
    const controller = new AbortController();
    loadAbortRef.current = controller;
    const isCurrent = (): boolean => !controller.signal.aborted
      && loadGenerationRef.current === generation
      && selectedRunRef.current === runId;

    if (!options.preserveStream) setLoading(true);
    try {
      const nextDetails = await getRun(runId, {
        signal: controller.signal,
        actionLimit: ACTION_HISTORY_LIMIT,
        providerLimit: PROVIDER_HISTORY_LIMIT,
        frameworkLimit: FRAMEWORK_HISTORY_LIMIT
      });
      if (!isCurrent()) return null;

      setDetails(nextDetails);
      eventCursorRef.current = {
        runId,
        ...(nextDetails.event_cursor ? { eventId: nextDetails.event_cursor } : {})
      };
      humanoidFrameBuffer.reset(nextDetails.checkpoint.world);
      setFramework(nextDetails.framework);
      const runIsActive = nextDetails.checkpoint.status === "starting"
        || nextDetails.checkpoint.status === "running";
      setModelActivity(modelActivityFromJournal(false, nextDetails.provider, runIsActive));
      setRuns((current) => updateRunListStatus(
        current,
        runId,
        nextDetails.checkpoint.status,
        nextDetails.checkpoint.error,
        nextDetails.checkpoint.updated_at
      ));
      if (!options.preserveStream) setStreamEpoch((current) => current + 1);
      if (nextDetails.checkpoint.status !== "starting" && nextDetails.checkpoint.status !== "running") {
        setStoppingRunId((current) => current === runId ? null : current);
      }
      return nextDetails;
    } catch (error) {
      if (!isCurrent() || isAbortError(error)) return null;
      throw error;
    } finally {
      if (loadGenerationRef.current === generation) {
        if (loadAbortRef.current === controller) loadAbortRef.current = null;
        if (!options.preserveStream) setLoading(false);
      }
    }
  }, [humanoidFrameBuffer]);

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
        setDetails(null);
        setFramework([]);
        setModelActivity(createModelActivity(true));
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
    if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
  }, []);

  const appendHumanoidFrames = useCallback((incoming: HumanoidWorldSnapshot[]): void => {
    if (incoming.length === 0) return;
    humanoidFrameBuffer.push(incoming);
  }, [humanoidFrameBuffer]);

  const handleRuntimeEvent = useCallback((runId: string, event: RuntimeEvent): void => {
    if (selectedRunRef.current !== runId) return;
    const previousCursor = eventCursorRef.current?.runId === runId
      ? eventCursorRef.current.eventId
      : undefined;
    const nextCursor = nextRuntimeEventCursor(previousCursor, event);
    eventCursorRef.current = {
      runId,
      ...(nextCursor === undefined ? {} : { eventId: nextCursor })
    };
    if (event.type === "framework_event") {
      setFramework((current) => upsertRuntimeJournalEntry(
        current,
        event.data,
        FRAMEWORK_HISTORY_LIMIT
      ));
    }

    if (event.type === "provider_event") {
      const activity = providerActivityFrom(event.data);
      if (activity) setModelActivity((current) => reduceProviderActivity(current, activity));
    }
    if (event.type === "model_request_started") {
      setModelActivity((current) => reduceRuntimeModelActivity(current, event));
    }

    const humanoidWorlds = humanoidWorldSnapshotsFrom(event.data);
    if (humanoidWorlds.length > 0) appendHumanoidFrames(humanoidWorlds);
    const terminalEvent = event.type === "run_succeeded"
      || event.type === "run_failed"
      || event.type === "run_paused"
      || event.type === "run_interrupted";
    setDetails((current) => {
      if (current === null) return current;
      const reducerWorlds = terminalEvent && humanoidWorlds.length === 0 && humanoidFrameBuffer.latest
        ? [humanoidFrameBuffer.latest]
        : humanoidWorlds;
      return reduceHumanoidRunDetails({
        details: current,
        event,
        worlds: reducerWorlds,
        historical: false,
        limits: { actions: ACTION_HISTORY_LIMIT, provider: PROVIDER_HISTORY_LIMIT }
      });
    });

    // The run list is a sibling of the details, not part of them, so terminal
    // transitions still have to be mirrored onto it here.
    if (event.type === "run_started" || event.type === "run_resumed") {
      setRuns((current) => updateRunListStatus(current, runId, "running", null, event.at));
      return;
    }
    if (event.type === "run_succeeded" || event.type === "run_failed"
      || event.type === "run_paused" || event.type === "run_interrupted") {
      const data = asRecord(event.data);
      const error = event.type === "run_succeeded" || event.type === "run_paused"
        ? null
        : typeof data?.error === "string"
          ? data.error
          : typeof data?.reason === "string" ? data.reason : "运行已结束，但服务端未提供原因";
      const status = event.type === "run_succeeded"
        ? "succeeded"
        : event.type === "run_failed"
          ? "failed"
          : event.type === "run_paused" ? "paused" : "interrupted";
      setRuns((current) => updateRunListStatus(current, runId, status, error, event.at));
      setStoppingRunId((currentId) => currentId === runId ? null : currentId);
      setStreamState("inactive");
      setModelActivity((current) => settleModelActivity(current, false));
    }
  }, [appendHumanoidFrames, humanoidFrameBuffer]);

  useEffect(() => {
    if (authRequired) {
      setStreamState("inactive");
      return;
    }
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
      (error) => {
        if (!acceptsStreamUpdate()) return;
        if (error instanceof ApiError && error.status === 401) {
          setPassword("");
          setAuthRequired(true);
          return;
        }
        setStreamError("实时事件流连接已断开");
      },
      (state) => {
        if (!acceptsStreamUpdate()) return;
        setStreamState(state);
        if (state === "connected") setStreamError(null);
      },
      cursor,
      async () => {
        const refreshed = await loadRun(runId, { preserveStream: true });
        if (!refreshed) throw new Error("运行已切换，停止恢复旧事件流");
        return {
          cursor: refreshed.event_cursor,
          active: refreshed.checkpoint.status === "starting"
            || refreshed.checkpoint.status === "running"
        };
      }
    );
  }, [authRequired, details?.checkpoint.status, details?.definition.run_id, handleRuntimeEvent, loadRun, selectedRunId, streamEpoch]);

  const selectRun = async (runId: string): Promise<void> => {
    selectedRunRef.current = runId;
    eventCursorRef.current = null;
    setDetails(null);
    setFramework([]);
    setModelActivity(createModelActivity(true));
    setSelectedRunId(runId);
    try {
      await loadRun(runId);
    } catch {
      if (selectedRunRef.current === runId) {
        showError("无法加载所选运行记录，请稍后重试。");
      }
    }
  };

  const createMission = async (input: {
    mission: string;
    scenario_id: string;
    goal: Goal;
    run_mode: HumanoidRunMode;
  }): Promise<void> => {
    setSubmitting(true);
    try {
      const runId = await startRun(input);
      setMissionOpen(false);
      setWorkspace("world");
      await refreshRuns(runId);
    } catch {
      showError("任务启动失败，请检查模型配置和任务条件。");
    } finally {
      setSubmitting(false);
    }
  };

  const resumeSelected = async (): Promise<void> => {
    if (!selectedRunId || !details || resuming
      || !["paused", "failed", "interrupted"].includes(details.checkpoint.status)) return;
    const previousStatus = details.checkpoint.status;
    const previousError = details.checkpoint.error;
    const optimisticAt = new Date().toISOString();
    setResuming(true);
    setDetails((current) => updateDetailsStatus(
      current,
      selectedRunId,
      "starting",
      null,
      optimisticAt
    ));
    setRuns((current) => updateRunListStatus(current, selectedRunId, "starting", null, optimisticAt));
    try {
      await resumeRun(selectedRunId);
    } catch {
      setDetails((current) => updateDetailsStatus(
        current,
        selectedRunId,
        previousStatus,
        previousError
      ));
      setRuns((current) => updateRunListStatus(
        current,
        selectedRunId,
        previousStatus,
        previousError,
        details.checkpoint.updated_at
      ));
      showError("任务恢复失败，请查看运行状态后重试。");
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
      showError("停止任务失败，请稍后重试。");
    }
  };

  const exportSelectedTelemetry = async (): Promise<void> => {
    if (!selectedRunId || exportingRunId !== null) return;
    setExportingRunId(selectedRunId);
    try {
      await downloadRunTelemetry(selectedRunId);
    } catch {
      showError("遥测导出失败，请稍后重试。");
    } finally {
      setExportingRunId(null);
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
    return (
      <section className="result-state" role="alert">
        <span aria-hidden="true">!</span>
        <h1>操作服务不可用</h1>
        <p>无法连接操作服务，请确认服务已启动并检查网络配置。</p>
        <UiButton onClick={() => void initialize()}>重试</UiButton>
      </section>
    );
  }

  const selectedRun = runs.find((run) => run.run_id === selectedRunId);
  const activeRun = runs.find((run) => run.status === "starting" || run.status === "running");
  const selectedIsActive = details?.checkpoint.status === "starting" || details?.checkpoint.status === "running";
  const missionControlsBusy = submitting || resuming || stoppingRunId !== null;
  const visibleModelActivity = details !== null || bootstrap.provider.configured
    ? modelActivity
    : createModelActivity(false);
  return (
    <GameShell
        workspace={workspace}
        onWorkspace={setWorkspace}
        modelState={visibleModelActivity.phase}
        onRefresh={() => void refreshRuns()}
        onLogout={hasPassword()
          ? () => {
              setPassword("");
              setAuthRequired(true);
            }
          : null}
        toolbar={(
          <div className="operator-toolbar">
            <div className="run-context">
              <select
                className="run-select"
                value={selectedRunId ?? ""}
                aria-label="选择运行记录"
                onChange={(event) => {
                  const value = event.target.value;
                  if (value) void selectRun(value);
                }}
              >
                <option value="" disabled>选择运行记录</option>
                {runs.map((run) => (
                  <option
                    key={run.run_id}
                    value={run.run_id}
                    disabled={run.status === "local_artifact"}
                  >
                    {runOptionLabel(run)}
                  </option>
                ))}
              </select>
              {selectedRun && <RunStatus status={selectedRun.status} />}
            </div>
            <div className="mission-actions">
              {details && (
                <UiButton
                  className="telemetry-export"
                  busy={exportingRunId === selectedRunId}
                  disabled={exportingRunId !== null && exportingRunId !== selectedRunId}
                  title="导出只读 Foxglove MCAP 遥测"
                  onClick={() => void exportSelectedTelemetry()}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 4v11M8 11l4 4 4-4M5 19h14" />
                  </svg>
                  <span className="telemetry-export-label">MCAP</span>
                </UiButton>
              )}
              {details && selectedIsActive && (
                <UiButton
                  tone="danger"
                  busy={stoppingRunId === selectedRunId}
                  disabled={resuming || submitting || (stoppingRunId !== null && stoppingRunId !== selectedRunId)}
                  onClick={() => void stopSelected()}
                >
                  暂停
                </UiButton>
              )}
              {details && ["paused", "failed", "interrupted"].includes(details.checkpoint.status) && (
                <UiButton
                  busy={resuming}
                  disabled={missionControlsBusy || activeRun !== undefined}
                  onClick={() => void resumeSelected()}
                >
                  继续运行
                </UiButton>
              )}
              <UiButton
                tone="primary"
                disabled={missionControlsBusy || activeRun !== undefined || !bootstrap.provider.configured}
                onClick={() => setMissionOpen(true)}
              >
                <b aria-hidden="true">＋</b>新建任务
              </UiButton>
            </div>
          </div>
        )}
      >
        {notice && <div className="operator-notice" role="alert">{notice}</div>}
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
            <section className="empty-state">
              <span aria-hidden="true"><i /><i /><i /></span>
              <p>尚未选择运行记录</p>
              <UiButton
                tone="primary"
                disabled={missionControlsBusy || activeRun !== undefined || !bootstrap.provider.configured}
                onClick={() => setMissionOpen(true)}
              >
                <b aria-hidden="true">＋</b>新建任务
              </UiButton>
            </section>
          ) : (
            <DeferredBoundary resetKey={details.definition.run_id}>
              <Suspense fallback={<CenteredSpin />}>
                <WorkspaceView
                  workspace={workspace}
                  details={details}
                  humanoidFrameBuffer={humanoidFrameBuffer}
                  framework={framework}
                  modelActivity={visibleModelActivity}
                  streamState={streamState}
                  onClose={() => setWorkspace("world")}
                />
              </Suspense>
            </DeferredBoundary>
          )}
        </div>
        {missionOpen && (
          <MissionModal
            open
            scenarios={bootstrap.scenarios}
            submitting={submitting}
            onCancel={() => setMissionOpen(false)}
            onSubmit={createMission}
          />
        )}
    </GameShell>
  );
}

function updateDetailsStatus(
  details: HumanoidRunDetails | null,
  runId: string,
  status: HumanoidRunDetails["checkpoint"]["status"],
  error: string | null,
  updatedAt?: string
): HumanoidRunDetails | null {
  if (!details || details.definition.run_id !== runId) return details;
  return {
    ...details,
    checkpoint: {
      ...details.checkpoint,
      status,
      error,
      ...(updatedAt ? { updated_at: updatedAt } : {})
    }
  };
}
