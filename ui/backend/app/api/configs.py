"""Config-file API routes for the offline UI backend."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.dependencies import get_indexer
from app.schemas.api_models import ConfigResponse

router = APIRouter(prefix="/api/configs", tags=["configs"])


@router.get("")
async def list_configs():
    """List YAML config files exposed through the backend."""
    indexer = get_indexer()
    return indexer.list_configs()


@router.get("/{name}", response_model=ConfigResponse)
async def get_config(name: str):
    """Return the raw contents of one named config file."""
    indexer = get_indexer()
    try:
        data = indexer.read_config(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found") from exc
    return ConfigResponse(**data)
