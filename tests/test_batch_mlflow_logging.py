from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ami_mom_pipeline.config import AppConfig


def _mlflow_available() -> bool:
    return importlib.util.find_spec("mlflow") is not None


def _load_batch_runner_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_nemo_batch_sequential.py"
    spec = importlib.util.spec_from_file_location("run_nemo_batch_sequential", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_batch_mlflow_logging_writes_aggregate_metrics_and_summary_artifact(tmp_path: Path) -> None:
    if not _mlflow_available():
        pytest.skip("mlflow is not installed")
    mod = _load_batch_runner_module()
    cfg = AppConfig()
    cfg.runtime.enable_mlflow_logging = True
    cfg.runtime.offline = True
    cfg.runtime.mlflow_tracking_uri = f"file:{tmp_path / 'mlruns'}"
    cfg.runtime.mlflow_experiment = "pytest_ami_batch"

    summary_path = tmp_path / "batch.summary.json"
    summary = {
        "config_path": "configs/pipeline.nemo.llama.yaml",
        "speech_backend": "nemo",
        "meeting_count": 2,
        "counts": {"ok": 2, "failed": 0, "skipped": 0},
        "total_elapsed_sec": 123.456,
        "speech_eval_summary": {
            "summary": {"mean_wer": 0.4, "mean_cpwer": 0.42, "mean_der": 0.61},
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    mod._log_batch_to_mlflow(cfg, "pytest_batch_run", summary, summary_path)  # noqa: SLF001

    import mlflow  # type: ignore

    client = mlflow.tracking.MlflowClient(tracking_uri=cfg.runtime.mlflow_tracking_uri)
    exps = [e for e in client.search_experiments() if e.name == cfg.runtime.mlflow_experiment]
    assert exps
    runs = client.search_runs([exps[0].experiment_id])
    assert runs
    run = runs[0]
    assert run.data.params.get("batch_run_label") == "pytest_batch_run"
    assert run.data.metrics.get("batch_ok_count") == 2.0
    assert run.data.metrics.get("batch_mean_wer") == 0.4
    assert run.data.metrics.get("batch_mean_cpwer") == 0.42
    assert run.data.metrics.get("batch_mean_der") == 0.61
    artifacts = client.list_artifacts(run.info.run_id, path="batch_runs")
    assert any(a.path.endswith("batch.summary.json") for a in artifacts)
