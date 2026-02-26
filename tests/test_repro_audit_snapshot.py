from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def _write_snapshot(root: Path, meeting_id: str, artifact_digest: str, config_digest: str) -> None:
    meeting_dir = root / "ami" / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)
    (meeting_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "meeting_id": meeting_id,
                "artifact_digest": artifact_digest,
                "config_digest": config_digest,
                "speech_backend": "nemo",
                "summarization_backend": "llama_cpp",
                "extraction_backend": "llama_cpp",
            }
        ),
        encoding="utf-8",
    )
    (meeting_dir / "reproducibility_report.json").write_text(
        json.dumps({"meeting_id": meeting_id, "config_digest": config_digest}),
        encoding="utf-8",
    )


def _write_min_config(path: Path, artifacts_dir: Path) -> None:
    cfg = {
        "paths": {
            "artifacts_dir": str(artifacts_dir),
            "raw_audio_dir": "data/rawa/ami/audio",
            "annotations_dir": "data/rawa/ami/annotations",
            "staged_dir": "data/staged/ami",
            "models_dir": "models",
        }
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def test_repro_audit_matches_against_snapshot_root(tmp_path: Path) -> None:
    current_root = tmp_path / "current_artifacts"
    snap_root = tmp_path / "snapshot_artifacts"
    _write_snapshot(current_root, "ES2005a", artifact_digest="aaa", config_digest="ccc")
    _write_snapshot(snap_root, "ES2005a", artifact_digest="aaa", config_digest="ccc")
    cfg_path = tmp_path / "cfg.yaml"
    _write_min_config(cfg_path, current_root)

    out_json = tmp_path / "repro_audit.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/repro_audit.py",
            "--config",
            str(cfg_path),
            "--meeting-id",
            "ES2005a",
            "--snapshot-dir",
            str(snap_root),
            "--out-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["summary"]["mismatched_meetings"] == 0
    assert report["rows"][0]["artifact_digest_match"] is True
    assert report["rows"][0]["config_digest_match"] is True


def test_repro_audit_detects_snapshot_mismatch(tmp_path: Path) -> None:
    current_root = tmp_path / "current_artifacts"
    snap_root = tmp_path / "snapshot_artifacts"
    _write_snapshot(current_root, "ES2005a", artifact_digest="aaa", config_digest="ccc")
    _write_snapshot(snap_root, "ES2005a", artifact_digest="bbb", config_digest="ccc")
    cfg_path = tmp_path / "cfg.yaml"
    _write_min_config(cfg_path, current_root)

    out_json = tmp_path / "repro_audit.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/repro_audit.py",
            "--config",
            str(cfg_path),
            "--meeting-id",
            "ES2005a",
            "--snapshot-dir",
            str(snap_root),
            "--out-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 1
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["summary"]["mismatched_meetings"] == 1
    assert report["summary"]["artifact_digest_mismatches"] == 1
