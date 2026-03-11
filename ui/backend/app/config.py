"""Configuration loading for the read-only FastAPI backend.

The UI backend reads repository-local artifacts and configs directly from disk.
`Settings` centralizes those filesystem locations plus a small set of auth and
run-control toggles sourced from environment variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Resolved backend settings derived from repository defaults and env vars."""

    project_root: Path
    raw_ami_audio_dir: Path
    artifacts_dir: Path
    eval_dir: Path
    configs_dir: Path
    docs_dir: Path
    staged_dir: Path
    ui_dist_dir: Path
    batch_runs_dir: Path
    python_executable: str
    run_controls_enabled: bool = False
    basic_auth_username: str | None = None
    basic_auth_password: str | None = None
    api_title: str = "AMI Offline Meeting Intelligence UI API"
    allowlist_roots: tuple[str, ...] = ("artifacts", "data/staged", "configs", "docs")
    denylist_fragments: tuple[str, ...] = (".git", ".env", ".ssh")
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://app.rabaya-labs.com",
        ]
    )


def _resolve_env_path(project_root: Path, env_name: str, default_rel: str) -> Path:
    """Resolve a path override from the environment relative to the project root."""
    value = os.getenv(env_name)
    candidate = Path(value) if value else project_root / default_rel
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object for the current backend process."""
    project_root_env = os.getenv("AMI_UI_PROJECT_ROOT")
    project_root = Path(project_root_env).resolve() if project_root_env else Path(__file__).resolve().parents[3]
    return Settings(
        project_root=project_root,
        raw_ami_audio_dir=_resolve_env_path(project_root, "AMI_UI_RAW_AMI_AUDIO_DIR", "data/rawa/ami/audio"),
        artifacts_dir=_resolve_env_path(project_root, "AMI_UI_ARTIFACTS_DIR", "artifacts/ami"),
        eval_dir=_resolve_env_path(project_root, "AMI_UI_EVAL_DIR", "artifacts/eval/ami"),
        configs_dir=_resolve_env_path(project_root, "AMI_UI_CONFIGS_DIR", "configs"),
        docs_dir=_resolve_env_path(project_root, "AMI_UI_DOCS_DIR", "docs"),
        staged_dir=_resolve_env_path(project_root, "AMI_UI_STAGED_DIR", "data/staged"),
        ui_dist_dir=_resolve_env_path(project_root, "AMI_UI_FRONTEND_DIST_DIR", "ui/frontend/dist"),
        batch_runs_dir=_resolve_env_path(project_root, "AMI_UI_BATCH_RUNS_DIR", "artifacts/batch_runs"),
        python_executable=os.getenv("AMI_UI_PYTHON_BIN", "python3"),
        run_controls_enabled=os.getenv("AMI_UI_ENABLE_RUN_CONTROLS", "0").lower() in {"1", "true", "yes", "on"},
        basic_auth_username=os.getenv("AMI_UI_BASIC_AUTH_USERNAME"),
        basic_auth_password=os.getenv("AMI_UI_BASIC_AUTH_PASSWORD"),
    )
