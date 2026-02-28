from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import get_indexer
from app.schemas.api_models import DashboardResponse, GovernanceListResponse, GovernanceResponse

router = APIRouter(tags=["governance"])


@router.get("/api/dashboard", response_model=DashboardResponse)
async def dashboard():
    indexer = get_indexer()
    return DashboardResponse(**indexer.get_dashboard())


@router.get("/api/meetings/{meeting_id}/repro")
async def meeting_repro(meeting_id: str):
    indexer = get_indexer()
    return indexer.get_meeting_repro(meeting_id)


@router.get("/api/governance", response_model=GovernanceResponse)
async def governance():
    indexer = get_indexer()
    return GovernanceResponse(**indexer.get_governance())


@router.get("/api/governance/evidence-bundles", response_model=GovernanceListResponse)
async def governance_evidence_bundles():
    indexer = get_indexer()
    return GovernanceListResponse(items=indexer.list_evidence_bundles())


@router.get("/api/governance/mlflow/runs", response_model=GovernanceListResponse)
async def governance_mlflow_runs():
    indexer = get_indexer()
    mlflow = indexer.list_mlflow_runs()
    return GovernanceListResponse(items=mlflow["runs"], configured=mlflow["configured"])
