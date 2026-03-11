"""Tests for local file-based MLflow logging in the mock pipeline flow."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ami_mom_pipeline.config import AppConfig
from ami_mom_pipeline.pipeline import run_pipeline


def _mlflow_available() -> bool:
    """Return whether MLflow is importable in the current environment."""
    return importlib.util.find_spec("mlflow") is not None


def test_mlflow_local_file_logging_mock_pipeline(tmp_path: Path) -> None:
    """Pipeline runs should log to a local file-based MLflow store when enabled."""
    if not _mlflow_available():
        pytest.skip("mlflow is not installed")

    raw_audio = Path("data/rawa/ami/audio/ES2005a.Mix-Headset.wav")
    annotations = Path("data/rawa/ami/annotations")
    if not raw_audio.exists() or not annotations.exists():
        pytest.skip("AMI sample data not available locally")

    cfg = AppConfig.load("configs/pipeline.sample.yaml")
    cfg.paths.raw_audio_dir = str(raw_audio.parent)
    cfg.paths.annotations_dir = str(annotations)
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.staged_dir = str(tmp_path / "staged")
    cfg.runtime.enable_mlflow_logging = True
    cfg.runtime.mlflow_tracking_uri = f"file:{tmp_path / 'mlruns'}"
    cfg.runtime.mlflow_experiment = "pytest_ami_mom"
    cfg.runtime.offline = True
    cfg.runtime.fail_on_offline_violations = False

    manifest = run_pipeline(cfg, "ES2005a")
    assert manifest["meeting_id"] == "ES2005a"

    repro = json.loads((Path(cfg.paths.artifacts_dir) / "ami" / "ES2005a" / "reproducibility_report.json").read_text())
    assert repro["mlflow_logging"]["enabled"] is True
    assert str(repro["mlflow_logging"]["tracking_uri"]).startswith("file:")

    mlruns_dir = tmp_path / "mlruns"
    assert mlruns_dir.exists()
    # Basic local-file-store evidence: meta files and at least one run directory file.
    assert any(p.name == "meta.yaml" for p in mlruns_dir.rglob("meta.yaml"))

    import mlflow  # type: ignore

    client = mlflow.tracking.MlflowClient(tracking_uri=cfg.runtime.mlflow_tracking_uri)
    exps = [e for e in client.search_experiments() if e.name == cfg.runtime.mlflow_experiment]
    assert exps, "Expected MLflow experiment to be created"
    runs = client.search_runs([exps[0].experiment_id])
    assert runs, "Expected at least one MLflow run"
    run = runs[0]
    # Params
    assert run.data.params.get("meeting_id") == "ES2005a"
    assert run.data.params.get("speech_backend") == "mock"
    # Metrics
    assert "wer" in run.data.metrics
    assert "cer" in run.data.metrics
    assert "asr_segment_count" in run.data.metrics
    # Artifact: run manifest
    artifacts = client.list_artifacts(run.info.run_id, path="meeting_artifacts")
    assert any(a.path.endswith("run_manifest.json") for a in artifacts)
