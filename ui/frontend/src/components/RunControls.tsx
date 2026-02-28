import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { formatDate, truncateDigest } from "../lib/format";
import type {
  ConfigEntry,
  MeetingListItem,
  MeetingRunEntry,
  RunMode,
  RunStageProgress,
  RunStatusResponse,
} from "../lib/types";
import { EmptyState, Panel, StatusBadge } from "./Primitives";

export function RunControls({
  enabled,
  fixedMeetingId,
  availableMeetings,
  title,
  subtitle,
  onRunSettled,
}: {
  enabled: boolean;
  fixedMeetingId?: string;
  availableMeetings: MeetingListItem[];
  title?: string;
  subtitle?: string;
  onRunSettled?: (run: RunStatusResponse) => void;
}) {
  const [configs, setConfigs] = useState<ConfigEntry[]>([]);
  const [selectedMeeting, setSelectedMeeting] = useState(fixedMeetingId ?? "");
  const [selectedConfig, setSelectedConfig] = useState("");
  const [mode, setMode] = useState<RunMode>("validate-only");
  const [showAdvancedConfigs, setShowAdvancedConfigs] = useState(false);
  const [activeRun, setActiveRun] = useState<RunStatusResponse | null>(null);
  const [meetingRuns, setMeetingRuns] = useState<MeetingRunEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [settledRunKey, setSettledRunKey] = useState<string | null>(null);
  const [socketFailed, setSocketFailed] = useState(false);
  const effectiveMeetingId = fixedMeetingId ?? selectedMeeting;

  useEffect(() => {
    if (!enabled) {
      return;
    }
    api.getConfigs().then((rows) => {
      setConfigs(rows);
      const preferred = rows.find((config) => isRecommendedConfig(config.name)) ?? rows[0];
      if (preferred) {
        setSelectedConfig(preferred.name);
      }
    });
  }, [enabled]);

  useEffect(() => {
    if (fixedMeetingId) {
      setSelectedMeeting(fixedMeetingId);
    }
  }, [fixedMeetingId]);

  useEffect(() => {
    if (!enabled || !effectiveMeetingId) {
      return;
    }
    if (!effectiveMeetingId) {
      return;
    }
    api.getMeetingRuns(effectiveMeetingId).then(setMeetingRuns).catch(() => undefined);
  }, [activeRun?.status, effectiveMeetingId, enabled]);

  useEffect(() => {
    if (!enabled || !activeRun || !["queued", "running"].includes(activeRun.status) || !socketFailed) {
      return;
    }
    const timer = window.setInterval(() => {
      api.getRun(activeRun.run_id).then(setActiveRun).catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [activeRun, enabled, socketFailed]);

  useEffect(() => {
    if (!enabled || !activeRun || !["queued", "running"].includes(activeRun.status)) {
      return;
    }
    setSocketFailed(false);
    const socket = new WebSocket(api.runWebSocketUrl(activeRun.run_id));
    socket.onmessage = (event) => {
      try {
        setActiveRun(JSON.parse(event.data) as RunStatusResponse);
      } catch {
        setSocketFailed(true);
      }
    };
    socket.onerror = () => setSocketFailed(true);
    socket.onclose = () => {
      if (["queued", "running"].includes(activeRun.status)) {
        setSocketFailed(true);
      }
    };
    return () => socket.close();
  }, [activeRun?.run_id, activeRun?.status, enabled]);

  useEffect(() => {
    if (!activeRun || !onRunSettled || !["completed", "failed"].includes(activeRun.status)) {
      return;
    }
    const key = `${activeRun.run_id}:${activeRun.status}`;
    if (settledRunKey === key) {
      return;
    }
    setSettledRunKey(key);
    onRunSettled(activeRun);
  }, [activeRun, onRunSettled, settledRunKey]);

  const canStart = enabled && Boolean(effectiveMeetingId) && Boolean(selectedConfig);
  const selectedMeetingSummary = useMemo(
    () => availableMeetings.find((meeting) => meeting.meeting_id === effectiveMeetingId) ?? null,
    [availableMeetings, effectiveMeetingId],
  );
  const recommendedConfigs = useMemo(
    () => configs.filter((config) => isRecommendedConfig(config.name)),
    [configs],
  );
  const visibleConfigs = showAdvancedConfigs || recommendedConfigs.length === 0 ? configs : recommendedConfigs;
  const hiddenCount = Math.max(0, configs.length - visibleConfigs.length);

  useEffect(() => {
    if (!selectedConfig) {
      return;
    }
    if (visibleConfigs.some((config) => config.name === selectedConfig)) {
      return;
    }
    if (visibleConfigs[0]) {
      setSelectedConfig(visibleConfigs[0].name);
    }
  }, [selectedConfig, visibleConfigs]);

  async function handleStart() {
    if (!canStart || !effectiveMeetingId) {
      return;
    }
    setError(null);
    try {
      const run = await api.createRun({
        meeting_id: effectiveMeetingId,
        config: selectedConfig,
        mode,
      });
      setActiveRun(run);
      setSocketFailed(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to start run");
    }
  }

  async function handleCancel() {
    if (!activeRun) {
      return;
    }
    setError(null);
    try {
      const run = await api.cancelRun(activeRun.run_id);
      setActiveRun(run);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to cancel run");
    }
  }

  return (
    <Panel
      title={title ?? "Run Controls"}
      subtitle={subtitle ?? "Start local run or validate-only execution and monitor progress"}
    >
      {!enabled ? (
        <EmptyState message="Run controls are disabled. Set AMI_UI_ENABLE_RUN_CONTROLS=1 to enable local execution from the UI." />
      ) : (
        <>
          <div className="grid gap-4 lg:grid-cols-[1fr_1fr_220px_160px]">
            {!fixedMeetingId ? (
              <label className="text-sm text-textSecondary">
                Meeting
                <select
                  value={selectedMeeting}
                  onChange={(event) => setSelectedMeeting(event.target.value)}
                  className="mt-2 w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-textPrimary"
                >
                  <option value="">Select meeting</option>
                  {availableMeetings.map((meeting) => (
                    <option key={meeting.meeting_id} value={meeting.meeting_id}>
                      {meeting.meeting_id}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <div className="rounded-lg border border-border bg-surface/70 p-3">
                <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">Meeting</p>
                <p className="mt-2 text-base font-semibold text-textPrimary">{fixedMeetingId}</p>
              </div>
            )}
            <label className="text-sm text-textSecondary">
              Pipeline Profile
              <select
                value={selectedConfig}
                onChange={(event) => setSelectedConfig(event.target.value)}
                className="mt-2 w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-textPrimary"
              >
                {visibleConfigs.map((config) => (
                  <option key={config.name} value={config.name}>
                    {config.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-sm text-textSecondary">
              Mode
              <select
                value={mode}
                onChange={(event) => setMode(event.target.value as RunMode)}
                className="mt-2 w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-textPrimary"
              >
                <option value="validate-only">Validate only</option>
                <option value="run">Run pipeline</option>
              </select>
            </label>
            <button
              type="button"
              onClick={handleStart}
              disabled={!canStart}
              className="h-fit self-end rounded-md border border-accent/40 bg-accent/15 px-4 py-2 text-sm text-textPrimary disabled:cursor-not-allowed disabled:border-border disabled:bg-surface disabled:text-textSecondary"
            >
              Start
            </button>
          </div>

          {configs.length > recommendedConfigs.length ? (
            <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-border bg-surface/50 px-3 py-2 text-sm">
              <span className="text-textSecondary">
                {showAdvancedConfigs
                  ? "Advanced profiles are visible."
                  : `${hiddenCount} advanced profile${hiddenCount === 1 ? "" : "s"} hidden by default.`}
              </span>
              <button
                type="button"
                onClick={() => setShowAdvancedConfigs((current) => !current)}
                className="rounded-md border border-border bg-card px-3 py-1.5 text-textPrimary"
              >
                {showAdvancedConfigs ? "Hide advanced" : "Show advanced"}
              </button>
            </div>
          ) : null}

          {selectedMeetingSummary ? (
            <div className="mt-4 rounded-lg border border-border bg-surface/60 p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-textPrimary">{selectedMeetingSummary.meeting_id}</p>
                  <p className="mt-1 text-xs text-textSecondary">{formatDate(selectedMeetingSummary.last_updated)}</p>
                </div>
                <StatusBadge
                  state={selectedMeetingSummary.offline_preflight_ok ? "success" : "warn"}
                  label={selectedMeetingSummary.offline_preflight_ok ? "Audit OK" : "Audit Review"}
                />
              </div>
            </div>
          ) : null}

          {error ? <div className="mt-4 rounded-lg border border-error/40 bg-error/10 p-3 text-sm text-error">{error}</div> : null}

          {activeRun ? (
            <div className="mt-4 grid gap-4 xl:grid-cols-[280px_1fr]">
              <div className="rounded-lg border border-border bg-surface/70 p-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="font-semibold text-textPrimary">Active Run</p>
                  <StatusBadge
                    state={
                      activeRun.status === "completed"
                        ? "success"
                        : activeRun.status === "failed"
                          ? "fail"
                          : activeRun.status === "cancelled"
                            ? "not_run"
                            : "info"
                    }
                    label={formatHistoryStatus(activeRun.status)}
                  />
                </div>
                {["queued", "running"].includes(activeRun.status) ? (
                  <button
                    type="button"
                    onClick={handleCancel}
                    className="mt-3 rounded-md border border-error/40 bg-error/10 px-3 py-2 text-sm text-error"
                  >
                    Cancel run
                  </button>
                ) : null}
                <div className="mt-4 space-y-3 text-sm">
                  <div>
                    <p className="text-textSecondary">Run ID</p>
                    <div className="mt-1 flex items-center gap-3">
                      <p className="text-textPrimary">{activeRun.run_id}</p>
                      <Link
                        to={`/runs/${activeRun.run_id}`}
                        className="text-xs text-accent underline-offset-2 hover:underline"
                      >
                        Open details
                      </Link>
                    </div>
                  </div>
                  <div>
                    <p className="text-textSecondary">Config</p>
                    <p className="mt-1 text-textPrimary">{activeRun.config}</p>
                  </div>
                  <div>
                    <p className="text-textSecondary">Started</p>
                    <p className="mt-1 text-textPrimary">{formatDate(activeRun.started_at)}</p>
                  </div>
                  <div>
                    <p className="text-textSecondary">Artifact Digest</p>
                    <p className="mt-1 text-textPrimary">{truncateDigest(activeRun.artifact_digest)}</p>
                  </div>
                  <div>
                    <p className="text-textSecondary">Current Stage</p>
                    <p className="mt-1 text-textPrimary">{activeRun.progress.current_stage_name ?? "Waiting for stage events"}</p>
                  </div>
                  <div>
                    <p className="text-textSecondary">Stage Progress</p>
                    <p className="mt-1 text-textPrimary">
                      {activeRun.progress.completed_stages}/{activeRun.progress.total_stages} completed
                    </p>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="rounded-lg border border-border bg-surface/70 p-4">
                  <p className="text-sm font-semibold text-textPrimary">Stage Progress</p>
                  <p className="mt-1 text-xs text-textSecondary">
                    {activeRun.progress.last_event ?? "Waiting for pipeline updates"}
                  </p>
                  {socketFailed ? (
                    <p className="mt-2 text-xs text-warning">WebSocket unavailable. Falling back to polling.</p>
                  ) : (
                    <p className="mt-2 text-xs text-textSecondary">Live updates via WebSocket.</p>
                  )}
                  <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                    {activeRun.progress.stages.map((stage) => (
                      <div key={stage.key} className="rounded-md border border-border bg-background/50 p-3">
                        <div className="flex items-center justify-between gap-2">
                          <p className="text-sm font-medium text-textPrimary">{stage.name}</p>
                          <StatusBadge state={mapRunStageState(stage.status)} label={formatRunStageLabel(stage.status)} />
                        </div>
                        <p className="mt-2 text-xs text-textSecondary">
                          {stage.runtime_sec !== null ? `${stage.runtime_sec.toFixed(2)} sec` : "No runtime yet"}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-border bg-surface/70 p-4">
                  <p className="text-sm font-semibold text-textPrimary">Run Log</p>
                  <pre className="mt-3 max-h-72 overflow-y-auto rounded-md bg-background/70 p-3 text-xs text-textPrimary">
                    {activeRun.recent_logs.length ? activeRun.recent_logs.join("\n") : "No log output yet."}
                  </pre>
                </div>
              </div>
            </div>
          ) : null}

          <div className="mt-4 rounded-lg border border-border bg-surface/60 p-4">
            <p className="text-sm font-semibold text-textPrimary">Recent Runs For Meeting</p>
            {meetingRuns.length === 0 ? (
              <p className="mt-3 text-sm text-textSecondary">No runs recorded for this meeting in the current session or batch history.</p>
            ) : (
              <div className="mt-3 space-y-2">
                {meetingRuns.slice(0, 6).map((run, index) => (
                  <div key={`${run.run_id ?? run.started_at ?? index}`} className="rounded-md border border-border bg-card/60 p-3 text-sm">
                    <div className="flex items-center justify-between gap-3">
                      <span className="font-medium text-textPrimary">{run.mode ?? "run"}</span>
                      <StatusBadge
                        state={
                          run.status === "ok" || run.status === "completed"
                            ? "success"
                            : run.status === "failed"
                              ? "fail"
                              : run.status === "not_run"
                                ? "not_run"
                                : "info"
                        }
                        label={formatHistoryStatus(run.status)}
                      />
                    </div>
                    <p className="mt-2 text-textSecondary">{run.config ?? "Unknown config"}</p>
                    <p className="mt-1 text-xs text-textSecondary">
                      {run.source === "live" ? "Live run" : "Batch history"} • {formatDate(run.started_at)}
                    </p>
                    {run.run_id ? (
                      <Link to={`/runs/${run.run_id}`} className="mt-2 inline-block text-xs text-accent underline-offset-2 hover:underline">
                        Open run details
                      </Link>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </Panel>
  );
}

function isRecommendedConfig(name: string) {
  const lower = name.toLowerCase();
  if (lower.includes("sample")) {
    return false;
  }
  if (lower.includes(".asr_") || lower.includes("_only")) {
    return false;
  }
  return true;
}

function mapRunStageState(status: RunStageProgress["status"]): "success" | "in_progress" | "fail" | "not_run" {
  if (status === "completed") {
    return "success";
  }
  if (status === "running") {
    return "in_progress";
  }
  if (status === "failed") {
    return "fail";
  }
  return "not_run";
}

function formatRunStageLabel(status: RunStageProgress["status"]) {
  if (status === "completed") {
    return "Completed";
  }
  if (status === "running") {
    return "Running";
  }
  if (status === "failed") {
    return "Failed";
  }
  if (status === "pending") {
    return "Pending";
  }
  return "Not run";
}

function formatHistoryStatus(status: string) {
  if (status === "ok" || status === "completed") {
    return "Completed";
  }
  if (status === "failed") {
    return "Failed";
  }
  if (status === "cancelled") {
    return "Cancelled";
  }
  if (status === "not_run") {
    return "Not run";
  }
  if (status === "skipped") {
    return "Skipped";
  }
  return status.replace(/_/g, " ");
}
