from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TimeSegment(BaseModel):
    start: float
    end: float


class VADSegment(TimeSegment):
    label: Literal["speech", "nonspeech"] = "speech"
    source: str = "mock"


class DiarizationSegment(TimeSegment):
    speaker: str
    source: str = "mock"


class ASRSegment(TimeSegment):
    speaker: str
    text: str
    confidence: float = 0.0
    source: str = "mock"


class TranscriptTurn(TimeSegment):
    speaker: str
    text_raw: str
    text_normalized: str


class TranscriptChunk(BaseModel):
    chunk_id: str
    meeting_id: str
    turn_indices: list[int]
    start: float
    end: float
    text: str


class DecisionItem(BaseModel):
    decision: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    uncertain: bool = False


class ActionItem(BaseModel):
    action: str
    owner: str | None = None
    due_date: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    uncertain: bool = False


class ExtractionOutput(BaseModel):
    meeting_id: str
    decisions: list[DecisionItem] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class EvidenceBackedPoint(BaseModel):
    text: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class MinutesSummary(BaseModel):
    meeting_id: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    discussion_points: list[EvidenceBackedPoint] = Field(default_factory=list)
    follow_up: list[EvidenceBackedPoint] = Field(default_factory=list)
    prompt_template_version: str = "mock-v1"
    backend: str = "mock"


class CanonicalMeeting(BaseModel):
    meeting_id: str
    duration_sec: float
    transcript_turns: list[TranscriptTurn]
    metadata: dict = Field(default_factory=dict)


class QCMetrics(BaseModel):
    meeting_id: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    duration_sec: float
    rms: float
    silence_ratio: float
