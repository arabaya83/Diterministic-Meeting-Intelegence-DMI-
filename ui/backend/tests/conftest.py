from __future__ import annotations

import json
from pathlib import Path

import asyncio

import pytest

from app.config import get_settings
from app.dependencies import get_indexer, get_runner, get_security
from app.main import app


@pytest.fixture()
def synthetic_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "artifacts/ami/TEST100a").mkdir(parents=True)
    (tmp_path / "artifacts/eval/ami").mkdir(parents=True)
    (tmp_path / "artifacts/mlruns/1/run1").mkdir(parents=True)
    (tmp_path / "data/rawa/ami/audio").mkdir(parents=True)
    (tmp_path / "data/staged/ami/audio_clean").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "docs").mkdir(parents=True)

    (tmp_path / "data/rawa/ami/audio/TEST100a.Mix-Headset.wav").write_bytes(b"RIFFtest")
    (tmp_path / "data/staged/ami/audio_clean/TEST100a.wav").write_bytes(b"RIFFclean")
    (tmp_path / "configs/pipeline.sample.yaml").write_text("pipeline:\n  retrieval:\n    enabled: false\n", encoding="utf-8")
    (tmp_path / "artifacts/eval/ami/wer_scores.csv").write_text(
        "meeting_id,wer,cer,cpwer,der,speech_backend\nTEST100a,0.1,0.05,0.12,0.2,nemo\n",
        encoding="utf-8",
    )
    (tmp_path / "artifacts/eval/ami/rouge_scores.csv").write_text(
        "meeting_id,rouge1,rouge2,rougeL\nTEST100a,0.4,0.2,0.35\n", encoding="utf-8"
    )
    (tmp_path / "artifacts/eval/ami/wer_breakdown.json").write_text(
        json.dumps({"meeting_id": "TEST100a", "wer": 0.1, "cer": 0.05, "cpwer": 0.12, "der": 0.2}),
        encoding="utf-8",
    )
    (tmp_path / "artifacts/eval/ami/mom_quality_checks.json").write_text(
        json.dumps({"meeting_id": "TEST100a", "schema_valid": True}), encoding="utf-8"
    )
    (tmp_path / "artifacts/mlruns/1/run1/meta.yaml").write_text("artifact_uri: file:///tmp/test\n", encoding="utf-8")

    meeting_dir = tmp_path / "artifacts/ami/TEST100a"
    meeting_dir.joinpath("vad_segments.json").write_text(json.dumps([{"start": 0, "end": 1, "label": "speech", "source": "test"}]), encoding="utf-8")
    meeting_dir.joinpath("vad_segments.rttm").write_text("SPEAKER TEST100a 1 0.0 1.0 <NA> <NA> speech <NA> <NA>\n", encoding="utf-8")
    meeting_dir.joinpath("diarization_segments.json").write_text(json.dumps([{"start": 0, "end": 1, "speaker": "speaker_0", "source": "test"}]), encoding="utf-8")
    meeting_dir.joinpath("diarization.rttm").write_text("SPEAKER TEST100a 1 0.0 1.0 <NA> <NA> speaker_0 <NA> <NA>\n", encoding="utf-8")
    meeting_dir.joinpath("asr_segments.json").write_text(json.dumps([{"start": 0, "end": 1, "speaker": "speaker_0", "text": "hello", "confidence": 0.8, "source": "test"}]), encoding="utf-8")
    meeting_dir.joinpath("full_transcript.txt").write_text("[0.0-1.0] SPEAKER_0: hello\n", encoding="utf-8")
    meeting_dir.joinpath("transcript_raw.json").write_text(json.dumps([{"start": 0, "end": 1, "speaker": "speaker_0", "text_raw": "hello", "text_normalized": "hello"}]), encoding="utf-8")
    meeting_dir.joinpath("transcript_normalized.json").write_text(json.dumps([{"start": 0, "end": 1, "speaker": "speaker_0", "text_raw": "hello", "text_normalized": "hello"}]), encoding="utf-8")
    meeting_dir.joinpath("transcript_chunks.jsonl").write_text(
        json.dumps({"chunk_id": "TEST100a_chunk_0001", "meeting_id": "TEST100a", "turn_indices": [0], "start": 0, "end": 1, "text": "hello"}) + "\n",
        encoding="utf-8",
    )
    meeting_dir.joinpath("mom_summary.json").write_text(json.dumps({"meeting_id": "TEST100a", "summary": "Summary", "key_points": ["Point"], "discussion_points": [], "follow_up": [], "prompt_template_version": "test", "backend": "mock"}), encoding="utf-8")
    meeting_dir.joinpath("mom_summary.html").write_text("<html><body>Summary</body></html>", encoding="utf-8")
    meeting_dir.joinpath("decisions_actions.json").write_text(json.dumps({"meeting_id": "TEST100a", "decisions": [], "action_items": [], "flags": []}), encoding="utf-8")
    meeting_dir.joinpath("extraction_validation_report.json").write_text(json.dumps({"meeting_id": "TEST100a", "schema_valid": True, "decision_count": 0, "action_item_count": 0, "flags": []}), encoding="utf-8")
    meeting_dir.joinpath("preflight_offline_audit.json").write_text(json.dumps({"offline_ok": True}), encoding="utf-8")
    meeting_dir.joinpath("reproducibility_report.json").write_text(json.dumps({"determinism": {"risks": ["gpu_risk"]}}), encoding="utf-8")
    meeting_dir.joinpath("run_manifest.json").write_text(json.dumps({"meeting_id": "TEST100a", "config_digest": "cfg", "artifact_digest": "art", "offline_preflight_ok": True}), encoding="utf-8")
    meeting_dir.joinpath("stage_trace.jsonl").write_text(
        json.dumps({"event": "stage_end", "stage": "ingest", "elapsed_sec": 0.2}) + "\n" +
        json.dumps({"event": "stage_end", "stage": "asr", "elapsed_sec": 1.1}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AMI_UI_PROJECT_ROOT", str(tmp_path))
    get_settings.cache_clear()
    get_security.cache_clear()
    get_indexer.cache_clear()
    get_runner.cache_clear()
    return tmp_path


@pytest.fixture()
def client(synthetic_root: Path):
    get_settings.cache_clear()
    get_security.cache_clear()
    get_indexer.cache_clear()
    get_runner.cache_clear()

    class ASGIClient:
        def get(self, path: str):
            return self.request("GET", path)

        def post(self, path: str, json_body: dict | None = None):
            return self.request("POST", path, json_body=json_body)

        def request(self, method: str, path: str, json_body: dict | None = None):
            async def run_request():
                messages = []
                request_sent = False
                body_bytes = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": method,
                    "headers": [(b"content-type", b"application/json")] if json_body is not None else [],
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode("utf-8"),
                    "query_string": b"",
                    "server": ("testserver", 80),
                    "client": ("testclient", 123),
                    "root_path": "",
                    "extensions": {},
                }

                async def receive():
                    nonlocal request_sent
                    if request_sent:
                        return {"type": "http.disconnect"}
                    request_sent = True
                    return {"type": "http.request", "body": body_bytes, "more_body": False}

                async def send(message):
                    messages.append(message)

                await app(scope, receive, send)
                return messages

            messages = asyncio.run(run_request())
            start = next(message for message in messages if message["type"] == "http.response.start")
            body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
            headers = {
                key.decode("utf-8"): value.decode("utf-8")
                for key, value in start.get("headers", [])
            }
            return SimpleResponse(start["status"], headers, body)

    yield ASGIClient()


class SimpleResponse:
    def __init__(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.content = body
        self.text = body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)
