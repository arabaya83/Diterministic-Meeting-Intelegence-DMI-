import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { formatDate, truncateDigest } from "../lib/format";
import type { RunStatusResponse } from "../lib/types";
import { EmptyState, KeyValueGrid, Panel, StatusBadge } from "./Primitives";

export function RunDetailsPage() {
  const { runId } = useParams();
  const [run, setRun] = useState<RunStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) {
      return;
    }
    api.getRun(runId).then(setRun).catch((reason: Error) => setError(reason.message));
  }, [runId]);

  useEffect(() => {
    if (!run || !["queued", "running"].includes(run.status)) {
      return;
    }
    const timer = window.setInterval(() => {
      api.getRun(run.run_id).then(setRun).catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [run]);

  if (!runId) {
    return <EmptyState message="Run ID is missing." />;
  }
  if (error) {
    return <EmptyState message={`Run details unavailable: ${error}`} />;
  }
  if (!run) {
    return <EmptyState message="Loading run details..." />;
  }

  const meetingIds = run.meeting_ids ?? [run.meeting_id];
  const isBatchRun = meetingIds.length > 1;
  const meetingLabel = isBatchRun ? "Meetings" : "Meeting";
  const meetingValue = isBatchRun ? meetingIds.join(", ") : run.meeting_id;

  return (
    <div className="space-y-6">
      <Panel
        title={`Run ${run.run_id}`}
        subtitle="Live execution details, stage progress, and recent log output"
        actions={
          !isBatchRun ? (
            <Link
              to={`/meetings/${run.meeting_id}`}
              className="rounded-md border border-accent/40 bg-accent/15 px-3 py-2 text-sm text-textPrimary"
            >
              Back to meeting
            </Link>
          ) : undefined
        }
      >
        <KeyValueGrid
          items={[
            { label: meetingLabel, value: meetingValue },
            { label: "Config", value: run.config },
            { label: "Mode", value: run.mode },
            {
              label: "Status",
              value: (
                <StatusBadge
                  state={run.status === "completed" ? "success" : run.status === "failed" ? "fail" : "in_progress"}
                  label={run.status}
                />
              ),
            },
            { label: "Started", value: formatDate(run.started_at) },
            { label: "Ended", value: formatDate(run.ended_at) },
            { label: "Artifact Digest", value: truncateDigest(run.artifact_digest) },
            { label: "Current Stage", value: run.progress.current_stage_name ?? "Waiting for stage events" },
            { label: "Stage Progress", value: `${run.progress.completed_stages}/${run.progress.total_stages}` },
          ]}
        />
      </Panel>

      <Panel title="Stage Progress" subtitle={run.progress.last_event ?? "No stage events recorded yet"}>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {run.progress.stages.map((stage) => (
            <div key={stage.key} className="rounded-lg border border-border bg-surface/70 p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="font-semibold text-textPrimary">{stage.name}</p>
                <StatusBadge
                  state={
                    stage.status === "completed"
                      ? "success"
                      : stage.status === "failed"
                        ? "fail"
                        : stage.status === "running"
                          ? "in_progress"
                          : "not_run"
                  }
                  label={formatStageStatus(stage.status)}
                />
              </div>
              <p className="mt-2 text-sm text-textSecondary">
                Runtime {stage.runtime_sec == null ? "n/a" : `${stage.runtime_sec.toFixed(2)} sec`}
              </p>
              {stage.summary ? (
                <pre className="mt-3 overflow-x-auto rounded-md bg-background/70 p-3 text-xs text-textPrimary">
                  {JSON.stringify(stage.summary, null, 2)}
                </pre>
              ) : null}
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Command" subtitle="Local subprocess invocation used for this run">
        <pre className="overflow-x-auto rounded-md bg-background/70 p-3 text-xs text-textPrimary">
          {run.command.join(" ")}
        </pre>
      </Panel>

      <Panel title="Recent Logs" subtitle="Last captured log lines from the subprocess">
        <pre className="max-h-[30rem] overflow-auto rounded-md bg-background/70 p-3 text-xs text-textPrimary">
          {run.recent_logs.length ? run.recent_logs.join("\n") : "No log output yet."}
        </pre>
      </Panel>

      <Panel title="Recent Stage Events" subtitle="Tail of stage_trace.jsonl as parsed by the backend">
        <pre className="max-h-[30rem] overflow-auto rounded-md bg-background/70 p-3 text-xs text-textPrimary">
          {run.stage_events.length ? JSON.stringify(run.stage_events, null, 2) : "No stage events captured yet."}
        </pre>
      </Panel>
    </div>
  );
}

function formatStageStatus(status: string) {
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
