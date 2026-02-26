from __future__ import annotations

import importlib.util
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


def test_build_summary_includes_dvc_template_metadata(tmp_path: Path) -> None:
    mod = _load_batch_runner_module()
    cfg = AppConfig()
    summary = mod.build_summary(
        cfg=cfg,
        config_path=tmp_path / "cfg.yaml",
        records=[{"status": "ok"}],
        total_elapsed=1.23,
        events_path=tmp_path / "events.jsonl",
        timings_csv_path=tmp_path / "timings.csv",
        validation_path=tmp_path / "validation.json",
        speech_eval_summary=None,
        speech_eval_csv_path=None,
        speech_eval_json_path=None,
        dvc_template_info={"mode": "single", "output": "x.yaml", "meeting_count": 1},
    )
    assert summary["dvc_template"]["mode"] == "single"
    assert summary["dvc_template"]["output"] == "x.yaml"


def test_generate_dvc_template_reports_subprocess_failure(monkeypatch) -> None:
    mod = _load_batch_runner_module()

    class FakeProc:
        returncode = 7
        stdout = "oops"
        stderr = "bad"

    def fake_run(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    out = mod._generate_dvc_template(Path("cfg.yaml"), ["ES2005a"], mode="single", output=None)  # noqa: SLF001
    assert out["error"] == "generate_dvc_stage_template_exit_7"
    assert out["stdout"] == "oops"
    assert out["stderr"] == "bad"
