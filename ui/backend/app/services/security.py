"""Filesystem access guards for the UI backend.

The backend is intentionally read-mostly and must not expose arbitrary files
from the host. `PathSecurity` constrains requests to repository-local
allowlisted roots while blocking common sensitive fragments.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, status

from app.config import Settings


class PathSecurity:
    """Validate requested paths before they are read by API handlers."""

    def __init__(self, settings: Settings) -> None:
        """Store settings and precompute allowlisted root directories."""
        self.settings = settings
        self.project_root = settings.project_root
        self.allowed_roots = {
            "artifacts": settings.project_root / "artifacts",
            "data/staged": settings.project_root / "data/staged",
            "configs": settings.project_root / "configs",
            "docs": settings.project_root / "docs",
        }

    def validate_relative_input(self, value: str) -> None:
        """Reject path parameters that attempt absolute or parent traversal."""
        if not value or value.startswith("/") or ".." in Path(value).parts:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path input")

    def ensure_readable(self, candidate: Path) -> Path:
        """Resolve a candidate path and ensure it stays inside the allowlist."""
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Path escapes project root") from exc

        parts = set(resolved.parts)
        if any(fragment in parts or fragment in str(resolved) for fragment in self.settings.denylist_fragments):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Path is not readable")

        if not any(resolved.is_relative_to(root.resolve()) for root in self.allowed_roots.values()):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Path is outside allowlist")
        return resolved

    def to_project_relative(self, candidate: Path) -> str:
        """Return a project-root-relative string for an approved path."""
        return str(candidate.resolve().relative_to(self.project_root))
