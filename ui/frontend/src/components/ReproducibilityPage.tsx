import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { formatDate, truncateDigest } from "../lib/format";
import type { MeetingListItem, MeetingReproResponse } from "../lib/types";
import { EmptyState, KeyValueGrid, Panel, StatusBadge } from "./Primitives";

export function ReproducibilityPage() {
  const [meetings, setMeetings] = useState<MeetingListItem[]>([]);
  const [selectedMeeting, setSelectedMeeting] = useState<string>("");
  const [repro, setRepro] = useState<MeetingReproResponse | null>(null);

  useEffect(() => {
    api.getMeetings().then((items) => {
      setMeetings(items);
      if (items[0]) {
        setSelectedMeeting(items[0].meeting_id);
      }
    });
  }, []);

  useEffect(() => {
    if (!selectedMeeting) {
      return;
    }
    api.getMeetingRepro(selectedMeeting).then(setRepro);
  }, [selectedMeeting]);

  const selectedMeetingSummary = useMemo(
    () => meetings.find((meeting) => meeting.meeting_id === selectedMeeting) ?? null,
    [meetings, selectedMeeting],
  );

  if (!meetings.length) {
    return <EmptyState message="No meetings available for reproducibility review." />;
  }

  return (
    <div className="space-y-6">
      <Panel title="Reproducibility" subtitle="Config digests, artifact digests, offline audits, and determinism risk surfaces">
        <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <select
            value={selectedMeeting}
            onChange={(event) => setSelectedMeeting(event.target.value)}
            className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-textPrimary outline-none ring-accent focus:ring-2"
          >
            {meetings.map((meeting) => (
              <option key={meeting.meeting_id} value={meeting.meeting_id}>
                {meeting.meeting_id}
              </option>
            ))}
          </select>
          {selectedMeetingSummary ? (
            <div className="rounded-lg border border-border bg-surface/70 p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="text-lg font-semibold text-textPrimary">{selectedMeetingSummary.meeting_id}</p>
                <StatusBadge
                  state={selectedMeetingSummary.offline_preflight_ok ? "success" : "warn"}
                  label={selectedMeetingSummary.offline_preflight_ok ? "Audit OK" : "Audit Review"}
                />
              </div>
              <div className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
                <div>
                  <p className="text-textSecondary">Updated</p>
                  <p className="mt-1 text-textPrimary">{formatDate(selectedMeetingSummary.last_updated)}</p>
                </div>
                <div>
                  <p className="text-textSecondary">Stage Coverage</p>
                  <p className="mt-1 text-textPrimary">
                    {selectedMeetingSummary.stages_complete}/{selectedMeetingSummary.stage_count}
                  </p>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </Panel>
      {repro ? (
        <>
          <Panel title={`Meeting ${repro.meeting_id}`} subtitle="Digest and audit summary">
            <KeyValueGrid
              items={[
                { label: "Config Digest", value: truncateDigest(repro.config_digest) },
                { label: "Artifact Digest", value: truncateDigest(repro.artifact_digest) },
                {
                  label: "Determinism Risks",
                  value: repro.determinism_risks.length ? repro.determinism_risks.join(", ") : "None recorded",
                },
              ]}
            />
          </Panel>
          <Panel title="Offline Audit">
            <pre className="overflow-x-auto rounded-md bg-background/70 p-4 text-xs">
              {JSON.stringify(repro.offline_audit ?? {}, null, 2)}
            </pre>
          </Panel>
          <Panel title="Run Manifest">
            <pre className="overflow-x-auto rounded-md bg-background/70 p-4 text-xs">
              {JSON.stringify(repro.run_manifest ?? {}, null, 2)}
            </pre>
          </Panel>
          <Panel title="Reproducibility Report">
            <pre className="overflow-x-auto rounded-md bg-background/70 p-4 text-xs">
              {JSON.stringify(repro.reproducibility_report ?? {}, null, 2)}
            </pre>
          </Panel>
        </>
      ) : null}
    </div>
  );
}
