import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { formatDate, formatNumber } from "../lib/format";
import type { DashboardResponse } from "../lib/types";
import { EmptyState, KeyValueGrid, Panel, StatusBadge } from "./Primitives";
import { RunControls } from "./RunControls";

export function DashboardPage() {
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function refreshDashboard() {
    api.getDashboard().then(setData).catch((reason: Error) => setError(reason.message));
  }

  useEffect(() => {
    refreshDashboard();
  }, []);

  if (error) {
    return <EmptyState message={`Dashboard unavailable: ${error}`} />;
  }
  if (!data) {
    return <EmptyState message="Loading dashboard state..." />;
  }

  return (
    <div className="space-y-6">
      <Panel title="System Health" subtitle="Offline governance and latest pipeline state">
        <div className="grid gap-4 lg:grid-cols-[1.25fr_1fr]">
          <KeyValueGrid
            items={[
              { label: "Offline Mode", value: <StatusBadge state="success" label="ON" /> },
              {
                label: "MLflow Logging",
                value: (
                  <StatusBadge
                    state={data.system_state.mlflow_logging ? "success" : "warn"}
                    label={data.system_state.mlflow_logging ? "Configured" : "Not configured"}
                  />
                ),
              },
              {
                label: "Strict Determinism",
                value: <StatusBadge state={data.system_state.strict_determinism ? "success" : "warn"} label="ON" />,
              },
              {
                label: "Run Controls",
                value: <StatusBadge state={data.system_state.run_controls_enabled ? "warn" : "info"} label={data.system_state.run_controls_enabled ? "Enabled" : "Disabled"} />,
              },
            ]}
          />
          <div className="rounded-lg border border-border bg-surface/70 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-textSecondary">Last Run</p>
            {data.last_run ? (
              <div className="mt-3 space-y-2 text-sm">
                <p className="text-2xl font-semibold text-textPrimary">{data.last_run.meeting_id}</p>
                <p className="text-textSecondary">Updated {formatDate(data.last_run.last_updated)}</p>
                <p>
                  Stage coverage {data.last_run.stages_complete}/{data.last_run.stage_count}
                </p>
              </div>
            ) : (
              <p className="mt-3 text-sm text-textSecondary">No run manifest detected.</p>
            )}
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          <Link to="/runs" className="rounded-md border border-accent/40 bg-accent/15 px-4 py-2 text-sm text-textPrimary">
            Run History
          </Link>
          <Link to="/governance" className="rounded-md border border-accent/40 bg-accent/15 px-4 py-2 text-sm text-textPrimary">
            Evidence Bundles
          </Link>
          <Link to="/reproducibility" className="rounded-md border border-accent/40 bg-accent/15 px-4 py-2 text-sm text-textPrimary">
            Reproducibility Review
          </Link>
        </div>
      </Panel>

      <RunControls
        enabled={Boolean(data.system_state.run_controls_enabled)}
        availableMeetings={data.meetings}
        title="Quick Actions"
        subtitle="Launch a local run or validate-only workflow from the dashboard"
        onRunSettled={() => refreshDashboard()}
      />

      <Panel title="Aggregate Metrics" subtitle="Evaluation summary from local artifact outputs">
        {Object.keys(data.aggregate_metrics).length === 0 ? (
          <EmptyState message="No aggregate evaluation metrics are available yet." />
        ) : (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {Object.entries(data.aggregate_metrics).map(([key, value]) => (
              <div key={key} className="rounded-lg border border-border bg-surface/70 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">{key}</p>
                <p className="mt-3 text-2xl font-semibold text-textPrimary">
                  {typeof value === "number" ? formatNumber(value) : String(value)}
                </p>
              </div>
            ))}
          </div>
        )}
      </Panel>

      <Panel title="Recent Meetings" subtitle="Most recently updated AMI artifact folders">
        {data.meetings.length === 0 ? (
          <EmptyState message="No meetings have been indexed yet." />
        ) : (
          <div className="grid gap-3 lg:grid-cols-2">
            {data.meetings.map((meeting) => (
              <Link
                key={meeting.meeting_id}
                to={`/meetings/${meeting.meeting_id}`}
                className="rounded-lg border border-border bg-surface/70 p-4 transition-colors hover:bg-surface"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-lg font-semibold text-textPrimary">{meeting.meeting_id}</p>
                    <p className="mt-1 text-sm text-textSecondary">{formatDate(meeting.last_updated)}</p>
                  </div>
                  <StatusBadge
                    state={meeting.offline_preflight_ok ? "success" : "warn"}
                    label={meeting.offline_preflight_ok ? "OK" : "Review"}
                  />
                </div>
                <div className="mt-4 flex items-center justify-between text-sm">
                  <span className="text-textSecondary">Stage coverage</span>
                  <span className="text-textPrimary">
                    {meeting.stages_complete}/{meeting.stage_count}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
