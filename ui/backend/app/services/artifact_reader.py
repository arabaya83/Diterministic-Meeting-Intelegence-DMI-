"""Helpers for decoding artifact files for API preview endpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.schemas.api_models import ArtifactKind


def infer_kind(path: Path) -> ArtifactKind:
    """Infer the preview/download kind for a filesystem path."""
    if not path.exists():
        return "missing"
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix in {".txt", ".rttm"}:
        return "text"
    if suffix in {".htm", ".html"}:
        return "html"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix in {".wav", ".mp3", ".ogg"}:
        return "audio"
    return "text"


def read_json(path: Path) -> Any:
    """Load a JSON artifact from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path, limit: int | None = None) -> list[Any]:
    """Load a JSONL artifact into memory with an optional row limit."""
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and index + 1 >= limit:
                break
    return rows


def read_csv(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load a CSV artifact into a list of dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(reader):
            rows.append(dict(row))
            if limit is not None and index + 1 >= limit:
                break
    return rows


def read_text(path: Path, limit_chars: int = 20000) -> str:
    """Read a text-like artifact with a conservative preview limit."""
    with path.open("r", encoding="utf-8") as handle:
        return handle.read(limit_chars)


def read_artifact_preview(path: Path) -> Any:
    """Dispatch to the appropriate preview loader for an artifact path."""
    kind = infer_kind(path)
    if kind == "json":
        return read_json(path)
    if kind == "jsonl":
        return read_jsonl(path)
    if kind == "csv":
        return read_csv(path)
    if kind in {"text", "html", "yaml"}:
        return read_text(path)
    if kind == "directory":
        return sorted(child.name for child in path.iterdir())
    return {"message": "Binary content available via download endpoint"}
