import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { NavLink, useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { formatDate, formatNumber, titleCase, truncateDigest } from "../lib/format";
import type {
  ArtifactEntry,
  MeetingExtractionResponse,
  MeetingEvalResponse,
  MeetingListItem,
  MeetingReproResponse,
  MeetingSpeechResponse,
  MeetingStatusResponse,
  MeetingSummaryResponse,
  MeetingTranscriptResponse,
  StageStatus,
} from "../lib/types";
import { EmptyState, KeyValueGrid, Panel, StatusBadge } from "./Primitives";
import { RunControls } from "./RunControls";

type MeetingTab = "Speech" | "Transcript" | "Summary" | "Extraction" | "Evaluation" | "Reproducibility";

const tabs: MeetingTab[] = ["Speech", "Transcript", "Summary", "Extraction", "Evaluation", "Reproducibility"];

export function MeetingsPage() {
  const navigate = useNavigate();
  const { meetingId } = useParams();
  const [meetings, setMeetings] = useState<MeetingListItem[]>([]);
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search);
  const [selectedTab, setSelectedTab] = useState<MeetingTab>("Speech");
  const [status, setStatus] = useState<MeetingStatusResponse | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactEntry[]>([]);
  const [speechData, setSpeechData] = useState<MeetingSpeechResponse | null>(null);
  const [transcriptData, setTranscriptData] = useState<MeetingTranscriptResponse | null>(null);
  const [summaryData, setSummaryData] = useState<MeetingSummaryResponse | null>(null);
  const [extractionData, setExtractionData] = useState<MeetingExtractionResponse | null>(null);
  const [repro, setRepro] = useState<MeetingReproResponse | null>(null);
  const [evalData, setEvalData] = useState<MeetingEvalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function refreshMeeting(meetingIdToLoad: string) {
    setError(null);
    Promise.all([
      api.getMeetingStatus(meetingIdToLoad),
      api.getArtifacts(meetingIdToLoad),
      api.getMeetingSpeech(meetingIdToLoad),
      api.getMeetingRepro(meetingIdToLoad),
      api.getMeetingEval(meetingIdToLoad),
      api.getMeetings(),
    ])
      .then(([statusResponse, artifactRows, speechResponse, reproResponse, evalResponse, meetingsResponse]) => {
        setStatus(statusResponse);
        setArtifacts(artifactRows);
        setSpeechData(speechResponse);
        setTranscriptData(null);
        setSummaryData(null);
        setExtractionData(null);
        setRepro(reproResponse);
        setEvalData(evalResponse);
        setMeetings(meetingsResponse);
      })
      .catch((reason: Error) => setError(reason.message));
  }

  useEffect(() => {
    api
      .getMeetings()
      .then((items) => {
        setMeetings(items);
        if (!meetingId && items[0]) {
          navigate(`/meetings/${items[0].meeting_id}`, { replace: true });
        }
      })
      .catch((reason: Error) => setError(reason.message));
  }, [meetingId, navigate]);

  useEffect(() => {
    if (!meetingId) {
      return;
    }
    refreshMeeting(meetingId);
  }, [meetingId]);

  useEffect(() => {
    if (!meetingId) {
      return;
    }
    if (selectedTab === "Speech" && !speechData) {
      api.getMeetingSpeech(meetingId).then(setSpeechData).catch(() => undefined);
    }
    if (selectedTab === "Transcript" && !transcriptData) {
      api.getMeetingTranscript(meetingId).then(setTranscriptData).catch(() => undefined);
    }
    if (selectedTab === "Summary" && !summaryData) {
      api.getMeetingSummaryTab(meetingId).then(setSummaryData).catch(() => undefined);
    }
    if (selectedTab === "Extraction" && !extractionData) {
      api.getMeetingExtraction(meetingId).then(setExtractionData).catch(() => undefined);
    }
  }, [extractionData, meetingId, selectedTab, speechData, summaryData, transcriptData]);

  const filteredMeetings = useMemo(() => {
    const query = deferredSearch.trim().toLowerCase();
    if (!query) {
      return meetings;
    }
    return meetings.filter((meeting) => meeting.meeting_id.toLowerCase().includes(query));
  }, [deferredSearch, meetings]);

  const existingArtifactCount = artifacts.filter((artifact) => artifact.exists).length;
  const artifactCoverageLabel = `${existingArtifactCount}/${artifacts.length || 0}`;

  if (error) {
    return <EmptyState message={`Meetings view unavailable: ${error}`} />;
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[320px_1fr]">
      <Panel title="Meetings" subtitle="Searchable AMI meeting inventory" className="h-fit">
        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search meeting_id"
          className="mb-4 w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-textPrimary outline-none ring-accent focus:ring-2"
        />
        <div className="scrollbar-thin max-h-[70vh] space-y-2 overflow-y-auto pr-1">
          {filteredMeetings.map((meeting) => (
            <NavLink
              key={meeting.meeting_id}
              to={`/meetings/${meeting.meeting_id}`}
              className={({ isActive }) =>
                [
                  "block rounded-lg border p-3 transition-colors",
                  isActive ? "border-accent bg-accent/15" : "border-border bg-surface/60 hover:bg-surface",
                ].join(" ")
              }
            >
              <div className="flex items-center justify-between gap-3">
                <span className="font-medium text-textPrimary">{meeting.meeting_id}</span>
                <StatusBadge
                  state={meeting.offline_preflight_ok ? "success" : "warn"}
                  label={meeting.offline_preflight_ok ? "OK" : "Review"}
                />
              </div>
              <p className="mt-2 text-xs text-textSecondary">{formatDate(meeting.last_updated)}</p>
            </NavLink>
          ))}
        </div>
      </Panel>

      <div className="space-y-6">
        {!status ? (
          <EmptyState message="Select a meeting to inspect artifacts and stage outputs." />
        ) : (
          <>
            <Panel title={`Meeting ${status.meeting_id}`} subtitle="Artifact traceability and stage transparency">
              <KeyValueGrid
                items={[
                  { label: "Config Digest", value: truncateDigest(status.summary.config_digest) },
                  { label: "Artifact Digest", value: truncateDigest(status.summary.artifact_digest) },
                  {
                    label: "Offline Audit",
                    value: (
                      <StatusBadge
                        state={status.summary.offline_preflight_ok ? "success" : "warn"}
                        label={status.summary.offline_preflight_ok ? "Passed" : "Review"}
                      />
                    ),
                  },
                  { label: "Artifact Count", value: status.artifact_count },
                  { label: "Artifact Coverage", value: artifactCoverageLabel },
                  { label: "Last Updated", value: formatDate(status.summary.last_updated) },
                  {
                    label: "Determinism Risks",
                    value: status.summary.determinism_risks.length ? status.summary.determinism_risks.join(", ") : "None recorded",
                  },
                ]}
              />
            </Panel>

            <RunControls
              enabled={status.run_controls_enabled}
              fixedMeetingId={status.meeting_id}
              availableMeetings={meetings}
              title="Run And Validate"
              subtitle="Trigger a local run or validate-only pass for the selected meeting"
              onRunSettled={(run) => refreshMeeting(run.meeting_id)}
            />

            <Panel title="Stage Timeline" subtitle="Pipeline stages derived from the artifact contract">
              <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
                {status.stages.map((stage) => (
                  <StageCard key={stage.key} stage={stage} />
                ))}
              </div>
            </Panel>

            <Panel title="Artifact Explorer" subtitle="Speech, transcript, summary, extraction, evaluation, and reproducibility outputs">
              <ArtifactCoverageSummary artifacts={artifacts} />
              <div className="mb-4 flex flex-wrap gap-2">
                {tabs.map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => setSelectedTab(tab)}
                    className={[
                      "rounded-md border px-3 py-2 text-sm",
                      selectedTab === tab
                        ? "border-accent bg-accent/15 text-textPrimary"
                        : "border-border bg-surface text-textSecondary",
                    ].join(" ")}
                  >
                    {tab}
                  </button>
                ))}
              </div>
              <ArtifactTabContent
                tab={selectedTab}
                meetingId={status.meeting_id}
                artifacts={artifacts}
                speechData={speechData}
                transcriptData={transcriptData}
                summaryData={summaryData}
                extractionData={extractionData}
                repro={repro}
                evalData={evalData}
              />
            </Panel>
          </>
        )}
      </div>
    </div>
  );
}

function StageCard({ stage }: { stage: StageStatus }) {
  return (
    <div className="rounded-lg border border-border bg-surface/70 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="font-semibold text-textPrimary">{stage.name}</p>
        <StatusBadge state={stage.status} />
      </div>
      <p className="mt-2 text-sm text-textSecondary">
        Runtime {stage.runtime_sec == null ? "n/a" : `${formatNumber(stage.runtime_sec, 2)} sec`}
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        {stage.artifacts.map((artifact) => (
          <a
            key={artifact.name}
            href={artifact.download_url}
            className={`rounded border px-2 py-1 text-xs ${artifact.exists ? "border-border text-textPrimary" : "border-border/50 text-textSecondary"}`}
          >
            {artifact.name}
          </a>
        ))}
      </div>
    </div>
  );
}

function ArtifactTabContent({
  tab,
  meetingId,
  artifacts,
  speechData,
  transcriptData,
  summaryData,
  extractionData,
  repro,
  evalData,
}: {
  tab: MeetingTab;
  meetingId: string;
  artifacts: ArtifactEntry[];
  speechData: MeetingSpeechResponse | null;
  transcriptData: MeetingTranscriptResponse | null;
  summaryData: MeetingSummaryResponse | null;
  extractionData: MeetingExtractionResponse | null;
  repro: MeetingReproResponse | null;
  evalData: MeetingEvalResponse | null;
}) {
  const findArtifact = (name: string) => artifacts.find((artifact) => artifact.name === name);
  const hasArtifact = (name: string) => Boolean(findArtifact(name)?.exists);
  const availableNames = artifacts.filter((artifact) => artifact.exists).map((artifact) => artifact.name);
  const rawPreview = transcriptData?.raw;
  const normalizedPreview = transcriptData?.normalized;
  const chunkPreview = transcriptData?.chunks;
  const summary = summaryData?.summary;
  const extraction = extractionData?.extraction;
  const extractionReport = extractionData?.validation_report;
  const vad = speechData?.vad_segments;
  const diarization = speechData?.diarization_segments;
  const asr = speechData?.asr_segments;
  const asrConfidence = evalData?.confidence ?? {};
  const audioArtifactName = hasArtifact("staged_audio")
    ? "staged_audio"
    : hasArtifact("raw_audio")
      ? "raw_audio"
      : null;

  if (tab === "Speech") {
    return (
      <div className="space-y-4">
        {audioArtifactName ? (
          <audio controls className="w-full" src={api.artifactDownloadUrl(meetingId, audioArtifactName)} />
        ) : (
          <EmptyState message="No raw or staged audio is available for this meeting." />
        )}
        <SegmentTable title="VAD Segments" rows={vad ?? []} />
        <SegmentTable title="Diarization Segments" rows={diarization ?? []} />
        <SegmentTable title="ASR Segments" rows={asr ?? []} />
      </div>
    );
  }

  if (tab === "Transcript") {
    if (!hasArtifact("transcript_raw.json") && !hasArtifact("transcript_normalized.json") && !hasArtifact("transcript_chunks.jsonl")) {
      return (
        <EmptyState message="Transcript artifacts are not available for this meeting. The pipeline may not have reached canonicalization or chunking." />
      );
    }
    return (
      <div className="space-y-4">
        <Panel title="Raw Transcript Turns" className="border-none bg-surface/60 p-0 shadow-none">
          <PreviewTable rows={rawPreview ?? []} />
        </Panel>
        <Panel title="Normalized Transcript Turns" className="border-none bg-surface/60 p-0 shadow-none">
          <PreviewTable rows={normalizedPreview ?? []} />
        </Panel>
        <Panel title="Transcript Chunks" className="border-none bg-surface/60 p-0 shadow-none">
          <PreviewTable rows={chunkPreview ?? []} />
        </Panel>
      </div>
    );
  }

  if (tab === "Summary") {
    if (!hasArtifact("mom_summary.json")) {
      return (
        <EmptyState message="Summary artifacts are not available for this meeting. Summarization may not have run yet." />
      );
    }
    return (
      <div className="space-y-4">
        <KeyValueGrid
          items={[
            { label: "Backend", value: String(summary?.backend ?? "n/a") },
            { label: "Prompt Template", value: String(summary?.prompt_template_version ?? "n/a") },
            {
              label: "HTML Download",
              value: summaryData?.html_available && summaryData.html_download_url ? (
                <a href={`${api.baseUrl}${summaryData.html_download_url}`} className="text-accent">
                  mom_summary.html
                </a>
              ) : (
                "Not available"
              ),
            },
          ]}
        />
        <div className="rounded-lg border border-border bg-surface/70 p-4">
          <p className="text-sm leading-7 text-textPrimary">{String(summary?.summary ?? "Summary not available.")}</p>
        </div>
        <SummaryList title="Key Points" items={(summary?.key_points as string[] | undefined) ?? []} />
        <EvidenceList title="Discussion Points" items={(summary?.discussion_points as Array<Record<string, unknown>> | undefined) ?? []} />
        <EvidenceList title="Follow Up" items={(summary?.follow_up as Array<Record<string, unknown>> | undefined) ?? []} />
      </div>
    );
  }

  if (tab === "Extraction") {
    if (!hasArtifact("decisions_actions.json") && !hasArtifact("extraction_validation_report.json")) {
      return (
        <EmptyState message="Extraction artifacts are not available for this meeting. Extraction may not have run yet." />
      );
    }
    return (
      <div className="space-y-4">
        <KeyValueGrid
          items={[
            {
              label: "Schema Valid",
              value: <StatusBadge state={extractionReport?.schema_valid ? "success" : "warn"} label={extractionReport?.schema_valid ? "True" : "False"} />,
            },
            { label: "Decision Count", value: String(extractionReport?.decision_count ?? 0) },
            { label: "Action Item Count", value: String(extractionReport?.action_item_count ?? 0) },
          ]}
        />
        <CardList title="Decisions" items={(extraction?.decisions as Array<Record<string, unknown>> | undefined) ?? []} primaryKey="decision" />
        <CardList title="Action Items" items={(extraction?.action_items as Array<Record<string, unknown>> | undefined) ?? []} primaryKey="action" />
        <SummaryList title="Validation Flags" items={(extractionReport?.flags as string[] | undefined) ?? []} />
      </div>
    );
  }

  if (tab === "Evaluation") {
    const metricEntries = Object.entries(evalData?.metrics ?? {});
    const confidenceEntries = Object.entries(asrConfidence ?? evalData?.confidence ?? {});
    if (metricEntries.length === 0 && confidenceEntries.length === 0) {
      return (
        <EmptyState message="Evaluation outputs are not available for this meeting. The evaluation stage may not have completed." />
      );
    }
    return (
      <div className="space-y-4">
        <div className="grid gap-4 lg:grid-cols-2">
          <MetricBars title="Evaluation Metrics" metrics={metricEntries} />
          <MetricBars title="Confidence Distribution" metrics={confidenceEntries} />
        </div>
        <KeyValueGrid
          items={[
            { label: "Quality Checks", value: JSON.stringify(evalData?.quality_checks ?? {}, null, 2) },
            { label: "Chunk Count", value: String((chunkPreview ?? []).length) },
            {
              label: "Downloads",
              value: (
                <div className="flex flex-col gap-1">
                  <a href={`${api.baseUrl}/api/eval/summary`} className="text-accent">Eval Summary API</a>
                  {hasArtifact("eval_wer_breakdown.json") ? (
                    <a href={api.artifactDownloadUrl(meetingId, "eval_wer_breakdown.json")} className="text-accent">wer_breakdown.json</a>
                  ) : (
                    <span className="text-textSecondary">wer_breakdown.json unavailable</span>
                  )}
                </div>
              ),
            },
          ]}
        />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {availableNames.length === 0 ? (
        <EmptyState message="No reproducibility artifacts are available for this meeting yet." />
      ) : null}
      <KeyValueGrid
        items={[
          { label: "Config Digest", value: truncateDigest(repro?.config_digest) },
          { label: "Artifact Digest", value: truncateDigest(repro?.artifact_digest) },
          {
            label: "Determinism Risks",
            value: repro?.determinism_risks.length ? repro.determinism_risks.join(", ") : "None recorded",
          },
        ]}
      />
      <JsonBlock title="Offline Audit" data={repro?.offline_audit} />
      <JsonBlock title="Run Manifest" data={repro?.run_manifest} />
      <JsonBlock title="Reproducibility Report" data={repro?.reproducibility_report} />
    </div>
  );
}

function ArtifactCoverageSummary({ artifacts }: { artifacts: ArtifactEntry[] }) {
  const present = artifacts.filter((artifact) => artifact.exists);
  const missing = artifacts.filter((artifact) => !artifact.exists);
  return (
    <div className="mb-4 grid gap-3 rounded-lg border border-border bg-surface/60 p-4 md:grid-cols-[220px_1fr]">
      <div>
        <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">Coverage</p>
        <p className="mt-2 text-2xl font-semibold text-textPrimary">
          {present.length}/{artifacts.length || 0}
        </p>
        <p className="mt-1 text-sm text-textSecondary">Artifacts available locally</p>
      </div>
      <div>
        <p className="text-xs uppercase tracking-[0.18em] text-textSecondary">Missing From Disk</p>
        {missing.length === 0 ? (
          <p className="mt-2 text-sm text-success">All tracked artifacts are present.</p>
        ) : (
          <div className="mt-2 flex flex-wrap gap-2">
            {missing.slice(0, 8).map((artifact) => (
              <span key={artifact.name} className="rounded border border-border px-2 py-1 text-xs text-textSecondary">
                {artifact.name}
              </span>
            ))}
            {missing.length > 8 ? (
              <span className="rounded border border-border px-2 py-1 text-xs text-textSecondary">
                +{missing.length - 8} more
              </span>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}

function SegmentTable({ title, rows }: { title: string; rows: Array<Record<string, unknown>> }) {
  return (
    <div>
      <h3 className="mb-2 text-sm uppercase tracking-[0.18em] text-textSecondary">{title}</h3>
      {rows.length === 0 ? <EmptyState message={`${title} are not available for this meeting.`} /> : <PreviewTable rows={rows} />}
    </div>
  );
}

function PreviewTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  if (rows.length === 0) {
    return <EmptyState message="No rows available." />;
  }
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="min-w-full text-left text-sm">
        <thead className="bg-surface text-textSecondary">
          <tr>
            {columns.map((column) => (
              <th key={column} className="px-3 py-2 font-medium">
                {titleCase(column)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 20).map((row, index) => (
            <tr key={index} className="border-t border-border bg-card/50">
              {columns.map((column) => (
                <td key={column} className="max-w-[28rem] px-3 py-2 align-top text-textPrimary">
                  <span className="break-words">{formatCell(row[column])}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SummaryList({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-lg border border-border bg-surface/70 p-4">
      <h3 className="mb-3 text-sm uppercase tracking-[0.18em] text-textSecondary">{title}</h3>
      {items.length === 0 ? (
        <EmptyState message="No items available." />
      ) : (
        <ul className="space-y-2 text-sm leading-6 text-textPrimary">
          {items.map((item) => (
            <li key={item} className="rounded-md border border-border bg-card/60 px-3 py-2">
              {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function EvidenceList({ title, items }: { title: string; items: Array<Record<string, unknown>> }) {
  return (
    <div className="rounded-lg border border-border bg-surface/70 p-4">
      <h3 className="mb-3 text-sm uppercase tracking-[0.18em] text-textSecondary">{title}</h3>
      {items.length === 0 ? (
        <EmptyState message="No evidence-linked points available." />
      ) : (
        <div className="space-y-3">
          {items.map((item, index) => (
            <div key={index} className="rounded-md border border-border bg-card/60 p-3">
              <p className="text-sm text-textPrimary">{String(item.text ?? "")}</p>
              <p className="mt-2 text-xs text-textSecondary">
                Evidence: {Array.isArray(item.evidence_chunk_ids) ? item.evidence_chunk_ids.join(", ") : "n/a"}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CardList({
  title,
  items,
  primaryKey,
}: {
  title: string;
  items: Array<Record<string, unknown>>;
  primaryKey: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface/70 p-4">
      <h3 className="mb-3 text-sm uppercase tracking-[0.18em] text-textSecondary">{title}</h3>
      {items.length === 0 ? (
        <EmptyState message={`No ${title.toLowerCase()} detected.`} />
      ) : (
        <div className="space-y-3">
          {items.map((item, index) => (
            <div key={index} className="rounded-md border border-border bg-card/60 p-3">
              <p className="font-medium text-textPrimary">{String(item[primaryKey] ?? "Unspecified")}</p>
              <p className="mt-2 text-xs text-textSecondary">
                Evidence: {Array.isArray(item.evidence_chunk_ids) ? item.evidence_chunk_ids.join(", ") : "n/a"}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MetricBars({ title, metrics }: { title: string; metrics: Array<[string, unknown]> }) {
  if (metrics.length === 0) {
    return <EmptyState message={`${title} unavailable.`} />;
  }
  const informational = metrics.filter(([label, value]) => isInformationalMetric(label, value));
  const measurable = metrics.filter(([label, value]) => !isInformationalMetric(label, value));
  const values = measurable
    .map(([, value]) => Number(value))
    .filter((value) => Number.isFinite(value));
  const max = values.length ? Math.max(...values, 1) : 1;
  return (
    <div className="rounded-lg border border-border bg-surface/70 p-4">
      <h3 className="mb-3 text-sm uppercase tracking-[0.18em] text-textSecondary">{title}</h3>
      {informational.length ? (
        <div className="mb-4 space-y-2 rounded-lg border border-border bg-card/40 p-3">
          {informational.map(([label, value]) => (
            <div key={label} className="flex items-center justify-between gap-3 text-sm">
              <span className="text-textSecondary">{titleCase(label)}</span>
              <span className="font-medium text-textPrimary">{String(value)}</span>
            </div>
          ))}
        </div>
      ) : null}
      <div className="space-y-3">
        {measurable.map(([label, value]) => {
          const numeric = Number(value);
          const tone = metricTone(label, numeric);
          return (
            <div key={label}>
              <div className="mb-1 flex items-center justify-between text-sm">
                <span>{titleCase(label)}</span>
                <span
                  className={
                    tone === "good" ? "text-success" : tone === "medium" ? "text-warning" : "text-error"
                  }
                >
                  {Number.isFinite(numeric) ? formatNumber(numeric) : String(value)}
                </span>
              </div>
              <div className="h-2 rounded-full bg-card">
                <div
                  className={
                    tone === "good"
                      ? "h-2 rounded-full bg-success"
                      : tone === "medium"
                        ? "h-2 rounded-full bg-warning"
                        : "h-2 rounded-full bg-error"
                  }
                  style={{ width: `${Number.isFinite(numeric) ? Math.max(8, (numeric / max) * 100) : 8}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function isInformationalMetric(label: string, value: unknown) {
  const normalized = label.toLowerCase();
  if (normalized.includes("meeting_id") || normalized.includes("meeting id")) {
    return true;
  }
  return !Number.isFinite(Number(value));
}

function metricTone(label: string, value: number): "good" | "medium" | "bad" {
  const normalized = label.toLowerCase();
  if (!Number.isFinite(value)) {
    return "bad";
  }

  if (normalized.includes("wer")) {
    return value <= 0.2 ? "good" : value <= 0.35 ? "medium" : "bad";
  }
  if (normalized.includes("cer")) {
    return value <= 0.12 ? "good" : value <= 0.25 ? "medium" : "bad";
  }
  if (normalized.includes("der") || normalized.includes("cpwer")) {
    return value <= 0.2 ? "good" : value <= 0.35 ? "medium" : "bad";
  }
  if (normalized.includes("confidence") || normalized.includes("reference available")) {
    return value >= 0.75 ? "good" : value >= 0.4 ? "medium" : "bad";
  }
  if (normalized.includes("count")) {
    return value > 0 ? "good" : "bad";
  }

  return value >= 0.75 ? "good" : value >= 0.4 ? "medium" : "bad";
}

function JsonBlock({ title, data }: { title: string; data: unknown }) {
  return (
    <div className="rounded-lg border border-border bg-surface/70 p-4">
      <h3 className="mb-3 text-sm uppercase tracking-[0.18em] text-textSecondary">{title}</h3>
      <pre className="overflow-x-auto rounded-md bg-background/70 p-3 text-xs text-textPrimary">
        {JSON.stringify(data ?? {}, null, 2)}
      </pre>
    </div>
  );
}

function formatCell(value: unknown) {
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value ?? "");
}
