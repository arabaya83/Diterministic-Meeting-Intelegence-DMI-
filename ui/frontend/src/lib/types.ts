export type StageState = "success" | "warn" | "fail" | "not_run" | "in_progress";
export type ArtifactKind =
  | "json"
  | "jsonl"
  | "csv"
  | "text"
  | "html"
  | "audio"
  | "directory"
  | "yaml"
  | "missing";

export interface MeetingListItem {
  meeting_id: string;
  has_raw_audio: boolean;
  has_staged_audio: boolean;
  has_artifacts: boolean;
  last_updated: string | null;
  config_digest: string | null;
  artifact_digest: string | null;
  offline_preflight_ok: boolean | null;
  determinism_risks: string[];
  stages_complete: number;
  stage_count: number;
}

export interface StageStatus {
  name: string;
  key: string;
  status: StageState;
  runtime_sec: number | null;
  artifacts: Array<{
    name: string;
    exists: boolean;
    artifact_url: string;
    download_url: string;
  }>;
  notes: string[];
}

export interface MeetingStatusResponse {
  meeting_id: string;
  summary: MeetingListItem;
  stages: StageStatus[];
  artifact_count: number;
  run_controls_enabled: boolean;
}

export interface ArtifactEntry {
  name: string;
  path: string;
  relative_path: string;
  exists: boolean;
  kind: ArtifactKind;
  size_bytes: number | null;
  download_url: string;
  preview_url: string;
}

export interface ArtifactPreview {
  meeting_id: string;
  artifact: ArtifactEntry;
  content: unknown;
}

export interface EvalSummaryResponse {
  aggregate_metrics: Record<string, number | string>;
  rows: Array<Record<string, string>>;
  latest_meeting: string | null;
}

export interface MeetingEvalResponse {
  meeting_id: string;
  metrics: Record<string, unknown>;
  quality_checks?: Record<string, unknown> | null;
}

export interface MeetingReproResponse {
  meeting_id: string;
  config_digest: string | null;
  artifact_digest: string | null;
  offline_audit?: Record<string, unknown> | null;
  reproducibility_report?: Record<string, unknown> | null;
  run_manifest?: Record<string, unknown> | null;
  determinism_risks: string[];
}

export interface MeetingSpeechResponse {
  meeting_id: string;
  audio: {
    artifact: ArtifactEntry;
    available: boolean;
  };
  vad_segments: Array<Record<string, unknown>>;
  diarization_segments: Array<Record<string, unknown>>;
  asr_segments: Array<Record<string, unknown>>;
}

export interface MeetingTranscriptResponse {
  meeting_id: string;
  raw: Array<Record<string, unknown>>;
  normalized: Array<Record<string, unknown>>;
  chunks: Array<Record<string, unknown>>;
}

export interface MeetingSummaryResponse {
  meeting_id: string;
  summary: Record<string, unknown>;
  html_available: boolean;
  html_download_url: string | null;
}

export interface MeetingExtractionResponse {
  meeting_id: string;
  extraction: Record<string, unknown>;
  validation_report: Record<string, unknown>;
}

export interface DashboardResponse {
  system_state: Record<string, boolean | string>;
  last_run: MeetingListItem | null;
  aggregate_metrics: Record<string, number | string>;
  meetings: MeetingListItem[];
}

export interface ConfigEntry {
  name: string;
  path: string;
  size_bytes: number;
}

export interface ConfigResponse {
  name: string;
  path: string;
  content: string;
}

export interface GovernanceResponse {
  evidence_bundles: Array<Record<string, unknown>>;
  mlflow: {
    configured: boolean;
    runs: Array<Record<string, unknown>>;
  };
}

export interface GovernanceListResponse {
  items: Array<Record<string, unknown>>;
  configured?: boolean | null;
}

export type RunMode = "run" | "validate-only";
export type RunState = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface RunStageProgress {
  key: string;
  name: string;
  status: "pending" | "running" | "completed" | "failed" | "not_run";
  runtime_sec: number | null;
  summary?: Record<string, unknown> | null;
}

export interface RunProgressSummary {
  current_stage_key: string | null;
  current_stage_name: string | null;
  completed_stages: number;
  total_stages: number;
  last_event: string | null;
  stages: RunStageProgress[];
}

export interface RunStatusResponse {
  run_id: string;
  meeting_id: string;
  meeting_ids: string[];
  config: string;
  mode: RunMode;
  status: RunState;
  started_at: string | null;
  ended_at: string | null;
  command: string[];
  recent_logs: string[];
  exit_code: number | null;
  summary?: Record<string, unknown> | null;
  stage_events: Array<Record<string, unknown>>;
  artifact_digest: string | null;
  progress: RunProgressSummary;
}

export interface MeetingRunEntry {
  run_id: string | null;
  meeting_id: string;
  meeting_ids: string[];
  config: string | null;
  mode: RunMode | null;
  status: string;
  started_at: string | null;
  ended_at: string | null;
  artifact_digest: string | null;
  source: "live" | "history";
}
