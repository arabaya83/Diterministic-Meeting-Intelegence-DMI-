"""Pydantic schema contracts for pipeline artifacts.

These models define the canonical JSON structures persisted by the pipeline.
They are intentionally lightweight and stable because downstream tooling
(`validate-only`, reproducibility audits, and report generation) relies on
field-level compatibility.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TimeSegment(BaseModel):
    """Generic time-aligned segment boundary in seconds."""

    start: float
    end: float


class VADSegment(TimeSegment):
    """Voice activity segment artifact row."""

    label: Literal["speech", "nonspeech"] = "speech"
    source: str = "mock"


class DiarizationSegment(TimeSegment):
    """Speaker-attributed diarization segment artifact row."""

    speaker: str
    source: str = "mock"


class ASRSegment(TimeSegment):
    """Speaker-attributed ASR segment artifact row."""

    speaker: str
    text: str
    confidence: float = 0.0
    source: str = "mock"


class TranscriptTurn(TimeSegment):
    """Canonical transcript turn with raw + normalized text views."""

    speaker: str
    text_raw: str
    text_normalized: str


class TranscriptChunk(BaseModel):
    """Chunked transcript unit used by summarization/extraction stages."""

    chunk_id: str
    meeting_id: str
    turn_indices: list[int]
    start: float
    end: float
    text: str


class DecisionItem(BaseModel):
    """Structured decision extraction record."""

    decision: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    uncertain: bool = False


class ActionItem(BaseModel):
    """Structured action-item extraction record."""

    action: str
    owner: str | None = None
    due_date: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    uncertain: bool = False


class ExtractionOutput(BaseModel):
    """Top-level structured extraction artifact."""

    meeting_id: str
    decisions: list[DecisionItem] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class EvidenceBackedPoint(BaseModel):
    """Summary/discussion/follow-up point with auditable transcript evidence."""

    text: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class MinutesSummary(BaseModel):
    """Structured Minutes of Meeting summary artifact."""

    meeting_id: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    discussion_points: list[EvidenceBackedPoint] = Field(default_factory=list)
    follow_up: list[EvidenceBackedPoint] = Field(default_factory=list)
    prompt_template_version: str = "mock-v1"
    backend: str = "mock"


class CanonicalMeeting(BaseModel):
    """Canonical meeting object used across downstream stages."""

    meeting_id: str
    duration_sec: float
    transcript_turns: list[TranscriptTurn]
    metadata: dict = Field(default_factory=dict)


class QCMetrics(BaseModel):
    """Audio quality-control metrics produced during ingest."""

    meeting_id: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    duration_sec: float
    rms: float
    silence_ratio: float
