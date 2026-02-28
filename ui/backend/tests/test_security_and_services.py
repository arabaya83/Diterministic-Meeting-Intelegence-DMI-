from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.services.artifact_reader import read_jsonl
from app.services.fs_indexer import FilesystemIndexer
from app.services.security import PathSecurity


def test_path_traversal_blocked(synthetic_root: Path):
    settings = get_settings()
    security = PathSecurity(settings)
    with pytest.raises(Exception):
        security.validate_relative_input("../secret.txt")
    with pytest.raises(Exception):
        security.validate_relative_input("/etc/passwd")


def test_meeting_listing_from_raw_audio_directory(synthetic_root: Path):
    settings = get_settings()
    indexer = FilesystemIndexer(settings, PathSecurity(settings))
    meetings = indexer.list_meetings()
    assert meetings
    assert meetings[0].meeting_id == "TEST100a"
    assert meetings[0].has_raw_audio is True


def test_artifact_listing_and_reading(synthetic_root: Path):
    settings = get_settings()
    indexer = FilesystemIndexer(settings, PathSecurity(settings))
    artifacts = indexer.list_artifacts("TEST100a")
    names = {artifact.name for artifact in artifacts if artifact.exists}
    assert "transcript_chunks.jsonl" in names
    preview_path = indexer.resolve_artifact_path("TEST100a", "mom_summary.json")
    assert preview_path.exists()


def test_jsonl_parsing_on_sample_chunk_file(synthetic_root: Path):
    path = synthetic_root / "artifacts/ami/TEST100a/transcript_chunks.jsonl"
    rows = read_jsonl(path)
    assert rows[0]["chunk_id"] == "TEST100a_chunk_0001"
