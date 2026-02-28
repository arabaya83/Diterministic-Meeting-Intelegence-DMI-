from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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

app.include_router(meetings_router)
app.include_router(evaluation_router)
app.include_router(configs_router)
app.include_router(governance_router)
app.include_router(runs_router)


@app.get("/health")
async def health():
    return {"ok": True, "offline": True}


if settings.ui_dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=settings.ui_dist_dir / "assets"), name="ui-assets")


@app.get("/{full_path:path}")
async def spa_entry(full_path: str):
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
