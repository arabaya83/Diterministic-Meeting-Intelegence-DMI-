from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import get_indexer
from app.schemas.api_models import EvalSummaryResponse, MeetingEvalResponse

router = APIRouter(prefix="/api/eval", tags=["evaluation"])


@router.get("/summary", response_model=EvalSummaryResponse)
async def eval_summary():
    indexer = get_indexer()
    return EvalSummaryResponse(**indexer.get_eval_summary())


@router.get("/meeting/{meeting_id}", response_model=MeetingEvalResponse)
async def eval_meeting(meeting_id: str):
    indexer = get_indexer()
    data = indexer.get_meeting_eval(meeting_id)
    return MeetingEvalResponse(meeting_id=meeting_id, **data)
