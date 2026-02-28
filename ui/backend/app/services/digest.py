from __future__ import annotations

from pathlib import Path


def safe_stat_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None
