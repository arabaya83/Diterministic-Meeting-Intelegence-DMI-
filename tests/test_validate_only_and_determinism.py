"""Regression tests for validate-only mode and strict determinism checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ami_mom_pipeline.config import AppConfig
import ami_mom_pipeline.pipeline as pipeline_mod


def _load_batch_runner_module():
    """Import the batch runner script as a module for helper testing."""
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_nemo_batch_sequential.py"
    spec = importlib.util.spec_from_file_location("run_nemo_batch_sequential", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validate_only_record_ok_when_manifest_exists(tmp_path: Path) -> None:
    """Validate-only records should succeed when a manifest already exists."""
    mod = _load_batch_runner_module()
    cfg = AppConfig()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    meeting_id = "ES2005a"
    artifact_dir = Path(cfg.paths.artifacts_dir) / "ami" / meeting_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "run_manifest.json").write_text(
        json.dumps({"meeting_id": meeting_id, "pipeline_version": "0.1.0", "artifact_digest": "abc123"}),
        encoding="utf-8",
    )

    rec = mod._build_validate_only_record(cfg, meeting_id, index=1, resume=True)  # noqa: SLF001
    assert rec["action"] == "validate"
    assert rec["status"] == "ok"
    assert rec["artifact_digest"] == "abc123"
    assert rec["manifest_pipeline_version"] == "0.1.0"


def test_validate_only_record_fails_when_manifest_missing(tmp_path: Path) -> None:
    """Validate-only records should fail when no manifest is present."""
    mod = _load_batch_runner_module()
    cfg = AppConfig()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")

    rec = mod._build_validate_only_record(cfg, "ES2005a", index=1, resume=True)  # noqa: SLF001
    assert rec["action"] == "validate"
    assert rec["status"] == "failed"
    assert rec["error_type"] == "FileNotFoundError"


def test_run_pipeline_strict_determinism_risk_raises_early(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strict determinism mode should raise before stage execution on known risks."""
    cfg = AppConfig()
    cfg.runtime.offline = True
    cfg.runtime.fail_on_determinism_risks = True
    cfg.runtime.write_preflight_audit = False
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.staged_dir = str(tmp_path / "staged")
    cfg.paths.annotations_dir = str(tmp_path / "annotations")
    cfg.paths.raw_audio_dir = str(tmp_path / "audio")

    def fake_configure_determinism(seed: int, strict: bool = True):
        """Inject a deterministic fake report containing one strict-mode risk."""
        return {
            "seed": seed,
            "strict_requested": strict,
            "risks": ["gpu_execution_may_still_be_nondeterministic_for_some_kernels"],
        }

    monkeypatch.setattr(pipeline_mod, "configure_determinism", fake_configure_determinism)

    with pytest.raises(RuntimeError, match="Determinism risks detected in strict mode"):
        pipeline_mod.run_pipeline(cfg, "ES2005a")
