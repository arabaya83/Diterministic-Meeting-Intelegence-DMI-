from __future__ import annotations

import json
from pathlib import Path

import pytest

from ami_mom_pipeline.config import AppConfig
from ami_mom_pipeline.pipeline import run_pipeline


def test_mock_pipeline_artifact_digest_stable_across_repeated_runs(tmp_path: Path) -> None:
    raw_audio = Path("data/rawa/ami/audio/ES2005a.Mix-Headset.wav")
    annotations = Path("data/rawa/ami/annotations")
    if not raw_audio.exists() or not annotations.exists():
        pytest.skip("AMI sample data not available locally")

    cfg = AppConfig.load("configs/pipeline.sample.yaml")
    cfg.paths.raw_audio_dir = str(raw_audio.parent)
    cfg.paths.annotations_dir = str(annotations)
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.staged_dir = str(tmp_path / "staged")
    cfg.runtime.include_nondeterministic_timings_in_manifest = False
    cfg.runtime.fail_on_offline_violations = False

    m1 = run_pipeline(cfg, "ES2005a")
    m2 = run_pipeline(cfg, "ES2005a")

    assert m1["artifact_digest"] == m2["artifact_digest"]
    assert m1["config_digest"] == m2["config_digest"]

    manifest_path = Path(cfg.paths.artifacts_dir) / "ami" / "ES2005a" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_digest"] == m1["artifact_digest"]
    assert "finalization_timings_sec" not in manifest
