"""Pydantic models defining the UI backend API contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ArtifactKind = Literal["json", "jsonl", "csv", "text", "html", "audio", "directory", "yaml", "missing"]
StageState = Literal["success", "warn", "fail", "not_run", "in_progress"]


class ArtifactEntry(BaseModel):
    """Metadata describing one artifact exposed through the API."""

    name: str
    path: str
    relative_path: str
    exists: bool
    kind: ArtifactKind
    size_bytes: int | None = None
    download_url: str
    preview_url: str


class StageArtifactLink(BaseModel):
    """UI-facing link metadata for one artifact tied to a pipeline stage."""

    name: str
    exists: bool
    artifact_url: str
    download_url: str


class StageStatus(BaseModel):
    """Computed status summary for a single pipeline stage."""

    name: str
    key: str
    status: StageState
    runtime_sec: float | None = None
    artifacts: list[StageArtifactLink] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class MeetingListItem(BaseModel):
    """Compact meeting summary row used by list and dashboard responses."""

    meeting_id: str
    has_raw_audio: bool
    has_staged_audio: bool
    has_artifacts: bool
    last_updated: datetime | None = None
    config_digest: str | None = None
    artifact_digest: str | None = None
    offline_preflight_ok: bool | None = None
    determinism_risks: list[str] = Field(default_factory=list)
    stages_complete: int = 0
    stage_count: int = 0


class MeetingStatusResponse(BaseModel):
    """Detailed meeting status payload returned by the status endpoint."""

    meeting_id: str
    summary: MeetingListItem
    stages: list[StageStatus]
    artifact_count: int
    run_controls_enabled: bool = False


class ArtifactPreview(BaseModel):
    """Artifact preview payload containing metadata and decoded content."""

    meeting_id: str
    artifact: ArtifactEntry
    content: Any = None


class EvalSummaryResponse(BaseModel):
    """Aggregate evaluation payload for dashboard and summary views."""

    aggregate_metrics: dict[str, Any]
    rows: list[dict[str, Any]]
    latest_meeting: str | None = None


class MeetingEvalResponse(BaseModel):
    """Evaluation details for one meeting."""

    meeting_id: str
    metrics: dict[str, Any]
    quality_checks: dict[str, Any] | None = None


class MeetingReproResponse(BaseModel):
    """Reproducibility artifact bundle for one meeting."""

    meeting_id: str
    config_digest: str | None = None
    artifact_digest: str | None = None
    offline_audit: dict[str, Any] | None = None
    reproducibility_report: dict[str, Any] | None = None
    run_manifest: dict[str, Any] | None = None
    determinism_risks: list[str] = Field(default_factory=list)


class ConfigEntry(BaseModel):
    """Listing entry for one pipeline config file."""

    name: str
    path: str
    size_bytes: int


class ConfigResponse(BaseModel):
    """Config file contents returned by the config-detail endpoint."""

    name: str
    path: str
    content: str


class DashboardResponse(BaseModel):
    """Dashboard payload combining system state, metrics, and recent meetings."""

    system_state: dict[str, Any]
    last_run: MeetingListItem | None = None
    aggregate_metrics: dict[str, Any]
    meetings: list[MeetingListItem]


class GovernanceResponse(BaseModel):
    """Top-level governance response grouping bundle and MLflow listings."""

    evidence_bundles: list[dict[str, Any]]
    mlflow: dict[str, Any]


class RunFeatureResponse(BaseModel):
    """Feature-flag response for run-control capabilities."""

    enabled: bool
    message: str


RunMode = Literal["run", "validate-only"]
RunState = Literal["queued", "running", "completed", "failed", "cancelled"]


class RunCreateRequest(BaseModel):
    """Request payload for launching a new run from the UI."""

    meeting_id: str | None = None
    meeting_ids: list[str] = Field(default_factory=list)
    config: str
    mode: RunMode = "run"


class RunStageProgress(BaseModel):
    """Per-stage progress row included in run status responses."""

    key: str
    name: str
    status: Literal["pending", "running", "completed", "failed", "not_run"]
    runtime_sec: float | None = None
    summary: dict[str, Any] | None = None


class RunProgressSummary(BaseModel):
    """Aggregated progress information derived from stage-trace events."""

    current_stage_key: str | None = None
    current_stage_name: str | None = None
    completed_stages: int = 0
    total_stages: int = 0
    last_event: str | None = None
    stages: list[RunStageProgress] = Field(default_factory=list)


class RunStatusResponse(BaseModel):
    """Detailed run status returned by polling and websocket updates."""

    run_id: str
    meeting_id: str
    meeting_ids: list[str] = Field(default_factory=list)
    config: str
    mode: RunMode
    status: RunState
    started_at: str | None = None
    ended_at: str | None = None
    command: list[str] = Field(default_factory=list)
    recent_logs: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    summary: dict[str, Any] | None = None
    stage_events: list[dict[str, Any]] = Field(default_factory=list)
    artifact_digest: str | None = None
    progress: RunProgressSummary = Field(default_factory=RunProgressSummary)


class RunCancelResponse(BaseModel):
    """Response payload for run cancellation requests."""

    run_id: str
    status: RunState


class MeetingRunEntry(BaseModel):
    """Compact run row used in meeting-specific run listings."""

    run_id: str | None = None
    meeting_id: str
    meeting_ids: list[str] = Field(default_factory=list)
    config: str | None = None
    mode: RunMode | None = None
    status: str
    started_at: str | None = None
    ended_at: str | None = None
    artifact_digest: str | None = None
    source: Literal["live", "history"]


class MeetingSpeechResponse(BaseModel):
    """Speech-artifact payload for a meeting detail page."""

    meeting_id: str
    audio: dict[str, Any]
    vad_segments: list[dict[str, Any]] = Field(default_factory=list)
    diarization_segments: list[dict[str, Any]] = Field(default_factory=list)
    asr_segments: list[dict[str, Any]] = Field(default_factory=list)


class MeetingTranscriptResponse(BaseModel):
    """Transcript-artifact payload for a meeting detail page."""

    meeting_id: str
    raw: list[dict[str, Any]] = Field(default_factory=list)
    normalized: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)


class MeetingSummaryResponse(BaseModel):
    """Summary-artifact payload for a meeting detail page."""

    meeting_id: str
    summary: dict[str, Any] = Field(default_factory=dict)
    html_available: bool = False
    html_download_url: str | None = None


class MeetingExtractionResponse(BaseModel):
    """Extraction-artifact payload for a meeting detail page."""

    meeting_id: str
    extraction: dict[str, Any] = Field(default_factory=dict)
    validation_report: dict[str, Any] = Field(default_factory=dict)


class GovernanceListResponse(BaseModel):
    """Simple list response used by governance sub-endpoints."""

    items: list[dict[str, Any]] = Field(default_factory=list)
    configured: bool | None = None
