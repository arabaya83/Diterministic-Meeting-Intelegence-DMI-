import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { formatDate, truncateDigest } from "../lib/format";
import type { MeetingRunEntry } from "../lib/types";
import { EmptyState, Panel, StatusBadge } from "./Primitives";
import { useEffect } from "react";

export function RunHistoryPage() {
  const [runs, setRuns] = useState<MeetingRunEntry[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getRuns().then(setRuns).catch((reason: Error) => setError(reason.message));
  }, []);

  const filteredRuns = useMemo(() => {
    const lower = query.trim().toLowerCase();
    if (!lower) {
      return runs;
    }
    return runs.filter(
      (run) =>
        run.meeting_id.toLowerCase().includes(lower) ||
        (run.config ?? "").toLowerCase().includes(lower) ||
        (run.status ?? "").toLowerCase().includes(lower),
    );
  }, [query, runs]);

  if (error) {
    return <EmptyState message={`Run history unavailable: ${error}`} />;
  }

  return (
    <Panel title="Run History" subtitle="Live and historical local pipeline runs">
      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Filter by meeting, config, or status"
        className="mb-4 w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-textPrimary outline-none ring-accent focus:ring-2"
      />
      {filteredRuns.length === 0 ? (
        <EmptyState message="No runs recorded yet." />
      ) : (
        <div className="space-y-3">
          {filteredRuns.map((run, index) => (
            <div key={`${run.run_id ?? run.started_at ?? index}`} className="rounded-lg border border-border bg-surface/70 p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-lg font-semibold text-textPrimary">{run.meeting_id}</p>
                  <p className="mt-1 text-sm text-textSecondary">{run.config ?? "Unknown config"}</p>
                </div>
                <StatusBadge
                  state={
                    run.status === "ok" || run.status === "completed"
                      ? "success"
                      : run.status === "failed"
                        ? "fail"
                        : run.status === "cancelled"
                          ? "not_run"
                          : run.status === "not_run"
                            ? "not_run"
                            : "info"
                  }
                  label={formatHistoryStatus(run.status)}
                />
              </div>
              <div className="mt-3 grid gap-3 md:grid-cols-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">Mode</p>
                  <p className="mt-1 text-sm text-textPrimary">{run.mode ?? "run"}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">Started</p>
                  <p className="mt-1 text-sm text-textPrimary">{formatDate(run.started_at)}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">Source</p>
                  <p className="mt-1 text-sm text-textPrimary">{run.source === "live" ? "Live" : "History"}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">Artifact Digest</p>
                  <p className="mt-1 text-sm text-textPrimary">{truncateDigest(run.artifact_digest)}</p>
                </div>
              </div>
              <div className="mt-3 flex gap-3">
                <Link to={`/meetings/${run.meeting_id}`} className="text-sm text-accent underline-offset-2 hover:underline">
                  Open meeting
                </Link>
                {run.run_id ? (
                  <Link to={`/runs/${run.run_id}`} className="text-sm text-accent underline-offset-2 hover:underline">
                    Open run details
                  </Link>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
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
