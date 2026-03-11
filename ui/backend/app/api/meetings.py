"""Meeting-centric API routes for browsing pipeline artifacts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.dependencies import get_indexer, get_runner
from app.config import get_settings
from app.schemas.api_models import (
    ArtifactPreview,
    MeetingRunEntry,
    MeetingExtractionResponse,
    MeetingSpeechResponse,
    MeetingStatusResponse,
    MeetingSummaryResponse,
    MeetingTranscriptResponse,
)
from app.services.artifact_reader import read_artifact_preview

router = APIRouter(prefix="/api/meetings", tags=["meetings"])


@router.get("")
async def list_meetings():
    """List known meetings discovered from audio or artifact directories."""
    indexer = get_indexer()
    return indexer.list_meetings()


@router.get("/{meeting_id}/status", response_model=MeetingStatusResponse)
async def meeting_status(meeting_id: str):
    """Return the computed status summary for one meeting."""
    indexer = get_indexer()
    summary = indexer.build_meeting_summary(meeting_id)
    stages = indexer.compute_stage_status(meeting_id)
    artifacts = indexer.list_artifacts(meeting_id)
    return MeetingStatusResponse(
        meeting_id=meeting_id,
        summary=summary,
        stages=stages,
        artifact_count=sum(1 for artifact in artifacts if artifact.exists),
        run_controls_enabled=get_settings().run_controls_enabled,
    )


@router.get("/{meeting_id}/artifacts")
async def meeting_artifacts(meeting_id: str):
    """List artifact descriptors for a meeting."""
    indexer = get_indexer()
    return indexer.list_artifacts(meeting_id)


@router.get("/{meeting_id}/runs", response_model=list[MeetingRunEntry])
async def meeting_runs(meeting_id: str):
    """Return live and historical run records for a meeting."""
    runner = get_runner()
    return [MeetingRunEntry(**row) for row in runner.list_runs_for_meeting(meeting_id)]


@router.get("/{meeting_id}/speech", response_model=MeetingSpeechResponse)
async def meeting_speech(meeting_id: str):
    """Return staged audio and speech-stage artifacts for a meeting."""
    indexer = get_indexer()
    return MeetingSpeechResponse(**indexer.get_meeting_speech(meeting_id))


@router.get("/{meeting_id}/transcript", response_model=MeetingTranscriptResponse)
async def meeting_transcript(meeting_id: str):
    """Return transcript artifacts produced by canonicalization and chunking."""
    indexer = get_indexer()
    return MeetingTranscriptResponse(**indexer.get_meeting_transcript(meeting_id))


@router.get("/{meeting_id}/summary", response_model=MeetingSummaryResponse)
async def meeting_summary(meeting_id: str):
    """Return summary artifacts for a meeting."""
    indexer = get_indexer()
    return MeetingSummaryResponse(**indexer.get_meeting_summary(meeting_id))


@router.get("/{meeting_id}/extraction", response_model=MeetingExtractionResponse)
async def meeting_extraction(meeting_id: str):
    """Return extraction artifacts for a meeting."""
    indexer = get_indexer()
    return MeetingExtractionResponse(**indexer.get_meeting_extraction(meeting_id))


@router.get("/{meeting_id}/artifact/{name}", response_model=ArtifactPreview)
async def artifact_preview(meeting_id: str, name: str):
    """Return an inline preview payload for a named artifact."""
    indexer = get_indexer()
    artifact = indexer.describe_artifact(meeting_id, name)
    if not artifact.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    content = read_artifact_preview(indexer.resolve_artifact_path(meeting_id, name))
    return ArtifactPreview(meeting_id=meeting_id, artifact=artifact, content=content)


@router.get("/{meeting_id}/artifact/{name}/download")
async def artifact_download(meeting_id: str, name: str):
    """Stream a named artifact back to the browser."""
    indexer = get_indexer()
    path = indexer.resolve_artifact_path(meeting_id, name)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    filename = path.name
    media_type = None
    if path.suffix.lower() == ".wav":
        media_type = "audio/wav"
    return FileResponse(path, filename=filename, media_type=media_type)
