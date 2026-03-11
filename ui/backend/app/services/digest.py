"""Small helpers for filesystem metadata used by the UI backend."""

from __future__ import annotations

from pathlib import Path


def safe_stat_size(path: Path) -> int | None:
    """Return file size when stat succeeds, otherwise ``None``."""
    try:
        return path.stat().st_size
    except OSError:
        return None
