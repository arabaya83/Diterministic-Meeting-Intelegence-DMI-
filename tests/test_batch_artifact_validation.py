from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ami_mom_pipeline.config import AppConfig


def _load_batch_runner_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_nemo_batch_sequential.py"
    spec = importlib.util.spec_from_file_location("run_nemo_batch_sequential", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validate_meeting_accepts_traceability_artifacts(tmp_path: Path) -> None:
    mod = _load_batch_runner_module()
    cfg = AppConfig()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    meeting_id = "ES2005a"
    artifact_dir = Path(cfg.paths.artifacts_dir) / "ami" / meeting_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Create all required files (including traceability/observability artifacts).
    for name in mod.REQUIRED_MEETING_FILES:
        p = artifact_dir / name
        if name.endswith(".json"):
            if name == "run_manifest.json":
                p.write_text(json.dumps({"meeting_id": meeting_id, "speech_backend": "nemo", "artifact_digest": "abc"}))
            elif name == "asr_segments.json":
                p.write_text(json.dumps([{"text": "hi", "speaker": "speaker_0", "start": 0.0, "end": 1.0, "confidence": 0.5}]))
            else:
                p.write_text("[]")
        elif name.endswith(".jsonl"):
            p.write_text(json.dumps({"event": "stage_start"}) + "\n")
        else:
            p.write_text("x")

    result = mod._validate_meeting(  # noqa: SLF001 - intentional regression test of validator behavior
        cfg=cfg,
        meeting_id=meeting_id,
        canonical_index={meeting_id: {"meeting_id": meeting_id}},
        wer_index={meeting_id: {"meeting_id": meeting_id}},
        rouge_index={meeting_id: {"meeting_id": meeting_id}},
    )
    assert result["valid"] is True
    assert result["missing_files"] == []
    assert result["errors"] == []


def test_validate_meeting_flags_missing_traceability_artifacts(tmp_path: Path) -> None:
    mod = _load_batch_runner_module()
    cfg = AppConfig()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    meeting_id = "ES2005a"
    artifact_dir = Path(cfg.paths.artifacts_dir) / "ami" / meeting_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Only create the legacy core files, omit new traceability artifacts to ensure validator catches them.
    for name in mod.REQUIRED_MEETING_FILES:
        if name in {"preflight_offline_audit.json", "reproducibility_report.json", "stage_trace.jsonl"}:
            continue
        p = artifact_dir / name
        if name.endswith(".json"):
            if name == "run_manifest.json":
                p.write_text(json.dumps({"meeting_id": meeting_id, "speech_backend": "nemo", "artifact_digest": "abc"}))
            elif name == "asr_segments.json":
                p.write_text(json.dumps([{"text": "hi", "speaker": "speaker_0", "start": 0.0, "end": 1.0, "confidence": 0.5}]))
            else:
                p.write_text("[]")
        elif name.endswith(".jsonl"):
            p.write_text(json.dumps({"chunk_id": "x"}) + "\n")
        else:
            p.write_text("x")

    result = mod._validate_meeting(  # noqa: SLF001
        cfg=cfg,
        meeting_id=meeting_id,
        canonical_index={meeting_id: {"meeting_id": meeting_id}},
        wer_index={meeting_id: {"meeting_id": meeting_id}},
        rouge_index={meeting_id: {"meeting_id": meeting_id}},
    )
    assert result["valid"] is False
    assert "preflight_offline_audit.json" in result["missing_files"]
    assert "reproducibility_report.json" in result["missing_files"]
    assert "stage_trace.jsonl" in result["missing_files"]
