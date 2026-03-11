"""FastAPI application for browsing offline AMI pipeline artifacts.

The backend serves API routes for artifact inspection plus, when available, the
built frontend bundle. Authentication is intentionally minimal and filesystem
access is delegated to dedicated services.
"""

from __future__ import annotations

import base64
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.configs import router as configs_router
from app.api.evaluation import router as evaluation_router
from app.api.governance import router as governance_router
from app.api.meetings import router as meetings_router
from app.api.runs import router as runs_router
from app.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.api_title)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def basic_auth_guard(request: Request, call_next):
    """Apply optional HTTP Basic auth to all non-health requests."""
    if not settings.basic_auth_username or not settings.basic_auth_password:
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        return _basic_auth_challenge()

    try:
        decoded = base64.b64decode(auth_header.split(" ", 1)[1]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        return _basic_auth_challenge()

    if not (
        secrets.compare_digest(username, settings.basic_auth_username)
        and secrets.compare_digest(password, settings.basic_auth_password)
    ):
        return _basic_auth_challenge()

    return await call_next(request)

app.include_router(meetings_router)
app.include_router(evaluation_router)
app.include_router(configs_router)
app.include_router(governance_router)
app.include_router(runs_router)


def _basic_auth_challenge() -> Response:
    """Return the fixed 401 challenge response used by the middleware."""
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="DMI UI"'},
    )


@app.get("/health")
async def health():
    """Return a minimal liveness payload for local orchestration."""
    return {"ok": True, "offline": True}


if settings.ui_dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=settings.ui_dist_dir / "assets"), name="ui-assets")


@app.get("/{full_path:path}")
async def spa_entry(full_path: str):
    """Serve the built SPA or a backend-only placeholder response."""
    if full_path.startswith("api/") or full_path == "health":
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    index_path = settings.ui_dist_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(
        {
            "message": "AMI UI backend running",
            "frontend_built": False,
            "frontend_dist": str(settings.ui_dist_dir),
        }
    )
