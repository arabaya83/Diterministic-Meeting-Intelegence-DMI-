"""Run-control and run-monitoring API routes for the backend."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.dependencies import get_runner
from app.schemas.api_models import MeetingRunEntry, RunCancelResponse, RunCreateRequest, RunStatusResponse

router = APIRouter(prefix="/api/runs", tags=["runs"])


def serialize_run(runner, record):
    """Project an internal run record into the API response schema."""
    return RunStatusResponse(
        run_id=record.run_id,
        meeting_id=record.meeting_id,
        meeting_ids=record.meeting_ids,
        config=record.config,
        mode=record.mode,
        status=record.status,
        started_at=record.started_at,
        ended_at=record.ended_at,
        command=record.command,
        recent_logs=list(record.recent_logs),
        exit_code=record.exit_code,
        summary=record.summary,
        stage_events=record.stage_events,
        artifact_digest=record.artifact_digest,
        progress=runner.build_progress_summary(record),
    )


@router.get("", response_model=list[MeetingRunEntry])
async def list_runs():
    """Return all live and historical runs known to the backend."""
    runner = get_runner()
    rows = runner.list_runs()
    return [MeetingRunEntry(**row) for row in rows]


@router.post("", response_model=RunStatusResponse)
async def create_run(payload: RunCreateRequest):
    """Start a new batch runner subprocess from the API."""
    runner = get_runner()
    record = runner.create_run(payload)
    return serialize_run(runner, record)


@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str):
    """Return the latest state for a single run identifier."""
    runner = get_runner()
    record = runner.get_run(run_id)
    return serialize_run(runner, record)


@router.post("/{run_id}/cancel", response_model=RunStatusResponse)
async def cancel_run(run_id: str):
    """Request cancellation for a running subprocess-backed run."""
    runner = get_runner()
    record = runner.cancel_run(run_id)
    return serialize_run(runner, record)


@router.get("/meeting/{meeting_id}", response_model=list[MeetingRunEntry])
async def list_meeting_runs(meeting_id: str):
    """Return runs associated with one meeting."""
    runner = get_runner()
    rows = runner.list_runs_for_meeting(meeting_id)
    return [MeetingRunEntry(**row) for row in rows]


@router.websocket("/ws/{run_id}")
async def run_updates(websocket: WebSocket, run_id: str):
    """Stream run status changes over a websocket until completion."""
    runner = get_runner()
    await websocket.accept()
    last_payload = None
    try:
        while True:
            record = runner.get_run(run_id)
            payload = serialize_run(runner, record).model_dump(mode="json")
            if payload != last_payload:
                await websocket.send_json(payload)
                last_payload = payload
            if record.status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
    finally:
        await websocket.close()
