from __future__ import annotations


def test_meetings_endpoint_returns_ok(client):
    response = client.get("/api/meetings")
    assert response.status_code == 200
    assert response.json()[0]["meeting_id"] == "TEST100a"


def test_meeting_status_endpoint_returns_ok(client):
    response = client.get("/api/meetings/TEST100a/status")
    assert response.status_code == 200
    body = response.json()
    assert body["meeting_id"] == "TEST100a"
    assert any(stage["key"] == "asr" for stage in body["stages"])


def test_artifact_preview_endpoint_returns_ok(client):
    response = client.get("/api/meetings/TEST100a/artifact/transcript_chunks.jsonl")
    assert response.status_code == 200
    assert response.json()["content"][0]["chunk_id"] == "TEST100a_chunk_0001"


def test_download_endpoint_returns_ok(client):
    response = asyncio.run(artifact_download("TEST100a", "mom_summary.html"))
    assert response.status_code == 200


def test_eval_and_config_endpoints_return_ok(client):
    eval_summary = client.get("/api/eval/summary")
    assert eval_summary.status_code == 200
    assert eval_summary.json()["rows"][0]["rouge1"] == "0.4"
    assert eval_summary.json()["aggregate_metrics"]["mean_rouge1"] == 0.4
    assert eval_summary.json()["aggregate_metrics"]["mean_cpwer"] == 0.12
    assert eval_summary.json()["aggregate_metrics"]["mean_der"] == 0.2
    eval_meeting = client.get("/api/eval/meeting/TEST100a")
    assert eval_meeting.status_code == 200
    assert eval_meeting.json()["metrics"]["rouge1"] == "0.4"
    assert eval_meeting.json()["metrics"]["cpwer"] == 0.12
    assert eval_meeting.json()["metrics"]["der"] == 0.2
    assert client.get("/api/meetings/TEST100a/repro").status_code == 200
    assert client.get("/api/meetings/TEST100a/speech").status_code == 200
    assert client.get("/api/meetings/TEST100a/transcript").status_code == 200
    assert client.get("/api/meetings/TEST100a/summary").status_code == 200
    assert client.get("/api/meetings/TEST100a/extraction").status_code == 200
    assert client.get("/api/configs").status_code == 200
    assert client.get("/api/configs/pipeline.sample.yaml").status_code == 200
    assert client.get("/api/governance/evidence-bundles").status_code == 200
    assert client.get("/api/governance/mlflow/runs").status_code == 200
    assert client.get("/api/meetings/TEST100a/runs").status_code == 200
    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["system_state"]["mlflow_logging"] is True


def test_invalid_artifact_name_returns_400(client):
    response = client.get("/api/meetings/TEST100a/artifact/../secrets")
    assert response.status_code in {400, 404}


def test_run_endpoints_work_when_enabled(client, monkeypatch):
    monkeypatch.setenv("AMI_UI_ENABLE_RUN_CONTROLS", "1")
    get_runner.cache_clear()

    class DummyProcess:
        def __init__(self):
            self.stdout = iter(["starting\n", "finished\n"])
            self._returncode = None

        def wait(self, timeout=None):
            self._returncode = 0 if self._returncode is None else self._returncode
            return self._returncode

        def poll(self):
            return self._returncode

        def terminate(self):
            self._returncode = -15

        def kill(self):
            self._returncode = -9

    monkeypatch.setattr("app.services.pipeline_runner.subprocess.Popen", lambda *args, **kwargs: DummyProcess())

    response = client.post(
        "/api/runs",
        json_body={"meeting_id": "TEST100a", "config": "pipeline.sample.yaml", "mode": "validate-only"},
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    time.sleep(0.01)
    run_status = client.get(f"/api/runs/{run_id}")
    assert run_status.status_code == 200
    assert run_status.json()["meeting_id"] == "TEST100a"
    assert run_status.json()["progress"]["total_stages"] >= 1
    assert any(stage["key"] == "ingest" for stage in run_status.json()["progress"]["stages"])
    all_runs = client.get("/api/runs")
    assert all_runs.status_code == 200
    assert any(row["meeting_id"] == "TEST100a" for row in all_runs.json())
    meeting_runs = client.get("/api/meetings/TEST100a/runs")
    assert meeting_runs.status_code == 200
    assert any(row["meeting_id"] == "TEST100a" for row in meeting_runs.json())


def test_cancel_run_endpoint_marks_run_cancelled(client, monkeypatch):
    monkeypatch.setenv("AMI_UI_ENABLE_RUN_CONTROLS", "1")
    get_runner.cache_clear()

    class DummyProcess:
        def __init__(self):
            self.stdout = iter([])
            self._returncode = None
            self._done = threading.Event()

        def wait(self, timeout=None):
            self._done.wait(timeout=timeout)
            self._returncode = -15 if self._returncode is None else self._returncode
            return self._returncode

        def poll(self):
            return self._returncode

        def terminate(self):
            self._returncode = -15
            self._done.set()

        def kill(self):
            self._returncode = -9
            self._done.set()

    monkeypatch.setattr("app.services.pipeline_runner.subprocess.Popen", lambda *args, **kwargs: DummyProcess())

    response = client.post(
        "/api/runs",
        json_body={"meeting_id": "TEST100a", "config": "pipeline.sample.yaml", "mode": "run"},
    )
    run_id = response.json()["run_id"]
    cancelled = client.post(f"/api/runs/{run_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


def test_run_registry_survives_runner_recreation(client, monkeypatch):
    monkeypatch.setenv("AMI_UI_ENABLE_RUN_CONTROLS", "1")
    get_runner.cache_clear()

    class DummyProcess:
        def __init__(self):
            self.stdout = iter(["starting\n", "finished\n"])
            self._returncode = None

        def wait(self, timeout=None):
            self._returncode = 0 if self._returncode is None else self._returncode
            return self._returncode

        def poll(self):
            return self._returncode

        def terminate(self):
            self._returncode = -15

        def kill(self):
            self._returncode = -9

    monkeypatch.setattr("app.services.pipeline_runner.subprocess.Popen", lambda *args, **kwargs: DummyProcess())

    response = client.post(
        "/api/runs",
        json_body={"meeting_id": "TEST100a", "config": "pipeline.sample.yaml", "mode": "validate-only"},
    )
    run_id = response.json()["run_id"]
    time.sleep(0.01)

    restored_runner = PipelineRunner(get_runner().settings)
    restored = restored_runner.get_run(run_id)
    assert restored.run_id == run_id
    assert restored.meeting_id == "TEST100a"
import asyncio

from app.api.meetings import artifact_download
import time
import threading

import pytest

from app.dependencies import get_runner
from app.services.pipeline_runner import PipelineRunner
