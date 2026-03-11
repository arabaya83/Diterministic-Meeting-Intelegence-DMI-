"""Subprocess-backed run-control service for the UI backend.

The backend does not execute pipeline stages inline. Instead it spawns the
existing batch runner script, captures logs, and reflects persisted summaries
back into API responses. This preserves the repository's current CLI contracts
and artifact semantics.
"""

from __future__ import annotations

import json
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.config import Settings
from app.schemas.api_models import RunCreateRequest, RunProgressSummary, RunStageProgress
from app.services.fs_indexer import PIPELINE_STAGES


@dataclass
class RunRecord:
    """In-memory state for one UI-triggered or restored run."""

    run_id: str
    run_label: str
    meeting_id: str
    meeting_ids: list[str]
    config: str
    mode: str
    status: str
    started_at: str | None
    ended_at: str | None
    command: list[str]
    recent_logs: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    exit_code: int | None = None
    summary: dict[str, Any] | None = None
    artifact_digest: str | None = None
    process: subprocess.Popen[str] | None = None
    stage_events: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False


class PipelineRunner:
    """Manage subprocess-backed pipeline runs for the UI backend."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the runner registry and restore persisted run metadata."""
        self.settings = settings
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._stage_lookup = {row["key"]: row["name"] for row in PIPELINE_STAGES}
        self._registry_path = self.settings.batch_runs_dir / "ui_runs_registry.json"
        self._load_registry()

    def ensure_enabled(self) -> None:
        """Raise when run controls are disabled for the deployed backend mode."""
        if not self.settings.run_controls_enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run controls are disabled for V1 read-only mode",
            )

    def create_run(self, request: RunCreateRequest) -> RunRecord:
        """Create a batch-run subprocess and register it for UI polling."""
        self.ensure_enabled()
        config_path = self._resolve_config(request.config)
        meeting_ids = self._resolve_requested_meetings(request)

        run_id = uuid.uuid4().hex[:12]
        primary_meeting = meeting_ids[0]
        run_scope = primary_meeting if len(meeting_ids) == 1 else f"batch_{len(meeting_ids)}"
        run_label = f"ui_{request.mode.replace('-', '_')}_{run_scope}_{run_id}"
        command = self._build_command(request, config_path, run_label)
        record = RunRecord(
            run_id=run_id,
            run_label=run_label,
            meeting_id=primary_meeting if len(meeting_ids) == 1 else f"Batch ({len(meeting_ids)})",
            meeting_ids=meeting_ids,
            config=request.config,
            mode=request.mode,
            status="queued",
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=None,
            command=command,
        )
        process = subprocess.Popen(
            command,
            cwd=self.settings.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._build_env(),
        )
        record.process = process
        record.status = "running"
        with self._lock:
            self._runs[run_id] = record
        self._persist_registry()
        thread = threading.Thread(
            target=self._watch_process,
            args=(record, run_label),
            daemon=True,
        )
        thread.start()
        return record

    def get_run(self, run_id: str) -> RunRecord:
        """Return a live run record, refreshing stage events when available."""
        with self._lock:
            record = self._runs.get(run_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
        self._refresh_stage_events(record)
        return record

    def list_runs(self) -> list[dict[str, Any]]:
        """Return live runs merged with persisted historical summaries."""
        live_runs: list[dict[str, Any]] = []
        with self._lock:
            for record in self._runs.values():
                live_runs.append(self._serialize_live_run(record))
        history_runs = self._historical_runs()
        combined = live_runs + history_runs
        combined.sort(key=lambda row: row.get("started_at") or "", reverse=True)
        return combined

    def list_runs_for_meeting(self, meeting_id: str) -> list[dict[str, Any]]:
        """Return runs associated with a particular meeting id."""
        self._validate_meeting_id(meeting_id)
        live_runs: list[dict[str, Any]] = []
        with self._lock:
            for record in self._runs.values():
                if meeting_id in record.meeting_ids:
                    live_runs.append(self._serialize_live_run(record))
        history_runs = self._historical_runs(meeting_id)
        combined = live_runs + history_runs
        combined.sort(key=lambda row: row.get("started_at") or "", reverse=True)
        return combined

    def cancel_run(self, run_id: str) -> RunRecord:
        """Attempt to terminate a queued or running subprocess."""
        self.ensure_enabled()
        record = self.get_run(run_id)
        process = record.process
        if record.status not in {"queued", "running"} or process is None:
            return record

        with self._lock:
            record.cancelled = True
            record.status = "cancelled"
            record.ended_at = datetime.now(timezone.utc).isoformat()
            record.recent_logs.append("Run cancellation requested from UI.")

        terminate = getattr(process, "terminate", None)
        poll = getattr(process, "poll", None)
        kill = getattr(process, "kill", None)
        wait = getattr(process, "wait", None)

        if callable(terminate):
            terminate()
        if callable(wait):
            try:
                wait(timeout=5)
            except TypeError:
                wait()
            except subprocess.TimeoutExpired:
                if callable(kill):
                    kill()
                    wait()

        exit_code = poll() if callable(poll) else None
        with self._lock:
            record.exit_code = exit_code
        self._persist_registry()
        return record

    def _build_command(self, request: RunCreateRequest, config_path: Path, run_label: str) -> list[str]:
        """Build the exact batch-runner command used for a UI launch."""
        meeting_ids = request.meeting_ids or ([request.meeting_id] if request.meeting_id else [])
        command = [
            self.settings.python_executable,
            "scripts/run_nemo_batch_sequential.py",
            "--config",
            str(config_path.relative_to(self.settings.project_root)),
            "--run-label",
            run_label,
            "--skip-speech-eval",
        ]
        for meeting_id in meeting_ids:
            command.extend(["--meeting-id", meeting_id])
        if request.mode == "validate-only":
            command.append("--validate-only")
        else:
            command.append("--no-resume")
        return command

    def _build_env(self) -> dict[str, str]:
        """Build the subprocess environment, preserving the caller's env vars."""
        env = dict(**{key: value for key, value in dict().items()})
        env.update({"PYTHONPATH": str(self.settings.project_root / "src")})
        import os

        merged = os.environ.copy()
        merged.update(env)
        return merged

    def _resolve_config(self, config_name: str) -> Path:
        """Resolve a config filename relative to the backend config directory."""
        path = self.settings.configs_dir / config_name
        if not path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
        return path

    def _validate_meeting_id(self, meeting_id: str) -> None:
        """Reject invalid meeting identifiers before shelling out."""
        if not meeting_id or "/" in meeting_id or ".." in meeting_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid meeting_id")

    def _resolve_requested_meetings(self, request: RunCreateRequest) -> list[str]:
        """Merge primary and repeated meeting ids into a stable unique list."""
        meeting_ids = list(dict.fromkeys(request.meeting_ids or []))
        if request.meeting_id:
            meeting_ids.insert(0, request.meeting_id)
            meeting_ids = list(dict.fromkeys(meeting_ids))
        for meeting_id in meeting_ids:
            self._validate_meeting_id(meeting_id)
        if not meeting_ids:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one meeting_id is required")
        return meeting_ids

    def _watch_process(self, record: RunRecord, run_label: str) -> None:
        """Consume subprocess output, finalize state, and persist the registry."""
        assert record.process is not None
        process = record.process
        if process.stdout:
            for line in process.stdout:
                with self._lock:
                    record.recent_logs.append(line.rstrip())
        exit_code = process.wait()
        summary = self._read_summary(run_label)
        artifact_digest = None
        if summary and len(record.meeting_ids) == 1:
            matching = next((row for row in summary.get("records", []) if row.get("meeting_id") == record.meeting_ids[0]), None)
            artifact_digest = (matching or {}).get("artifact_digest")
        with self._lock:
            record.exit_code = exit_code
            if record.ended_at is None:
                record.ended_at = datetime.now(timezone.utc).isoformat()
            record.summary = summary
            record.artifact_digest = artifact_digest
            if record.cancelled:
                record.status = "cancelled"
            else:
                record.status = "completed" if exit_code == 0 else "failed"
        self._refresh_stage_events(record)
        self._persist_registry()

    def _refresh_stage_events(self, record: RunRecord) -> None:
        """Refresh recent stage-trace events for single-meeting runs."""
        if len(record.meeting_ids) != 1:
            return
        stage_trace = self.settings.artifacts_dir / record.meeting_id / "stage_trace.jsonl"
        if not stage_trace.exists():
            return
        rows = []
        with stage_trace.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        with self._lock:
            record.stage_events = rows[-30:]

    def build_progress_summary(self, record: RunRecord) -> RunProgressSummary:
        """Convert recorded stage events into a UI-friendly progress summary."""
        stage_rows: dict[str, dict[str, Any]] = {}
        current_stage_key: str | None = None
        last_event: str | None = None

        for event in record.stage_events:
            stage_key = event.get("stage")
            if stage_key not in self._stage_lookup:
                continue
            row = stage_rows.setdefault(stage_key, {"status": "pending", "runtime_sec": None, "summary": None})
            if event.get("event") == "stage_start":
                row["status"] = "running"
                current_stage_key = stage_key
                last_event = f"Started {self._stage_lookup[stage_key]}"
                continue
            if event.get("event") == "stage_end":
                status = "completed" if event.get("status") in {None, "ok"} else "failed"
                row["status"] = status
                row["runtime_sec"] = event.get("elapsed_sec")
                row["summary"] = event.get("summary")
                last_event = f"{'Completed' if status == 'completed' else 'Failed'} {self._stage_lookup[stage_key]}"
                if current_stage_key == stage_key:
                    current_stage_key = None

        if record.status == "failed":
            current_stage_key = next(
                (key for key in reversed(list(stage_rows.keys())) if stage_rows[key]["status"] == "failed"),
                current_stage_key,
            )
        elif record.status == "running" and current_stage_key is None:
            current_stage_key = next(
                (key for key in reversed(list(stage_rows.keys())) if stage_rows[key]["status"] == "running"),
                None,
            )

        completed_stages = 0
        stages: list[RunStageProgress] = []
        for definition in PIPELINE_STAGES:
            key = definition["key"]
            row = stage_rows.get(key, {})
            status = row.get("status", "pending")
            if record.status == "completed" and status == "pending":
                status = "not_run"
            if status == "completed":
                completed_stages += 1
            stages.append(
                RunStageProgress(
                    key=key,
                    name=definition["name"],
                    status=status,
                    runtime_sec=row.get("runtime_sec"),
                    summary=row.get("summary"),
                )
            )

        if record.status == "completed" and not last_event:
            last_event = "Run completed"
        if record.status == "failed" and not last_event:
            last_event = "Run failed"

        return RunProgressSummary(
            current_stage_key=current_stage_key,
            current_stage_name=self._stage_lookup.get(current_stage_key) if current_stage_key else None,
            completed_stages=completed_stages,
            total_stages=len(PIPELINE_STAGES),
            last_event=last_event,
            stages=stages,
        )

    def _read_summary(self, run_label: str) -> dict[str, Any] | None:
        """Read the persisted batch summary file for one run label."""
        summary_path = self.settings.batch_runs_dir / f"{run_label}.summary.json"
        if not summary_path.exists():
            return None
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _historical_runs(self, meeting_id: str | None = None) -> list[dict[str, Any]]:
        """Load historical run rows from persisted batch summary files."""
        rows: list[dict[str, Any]] = []
        if not self.settings.batch_runs_dir.exists():
            return rows
        for path in sorted(self.settings.batch_runs_dir.glob("*.summary.json"), reverse=True):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            for record in payload.get("records", []):
                if meeting_id is not None and record.get("meeting_id") != meeting_id:
                    if meeting_id not in (record.get("meeting_ids") or []):
                        continue
                rows.append(
                    {
                        "run_id": None,
                        "meeting_id": record.get("meeting_id") or (record.get("meeting_ids") or ["Unknown"])[0],
                        "meeting_ids": record.get("meeting_ids") or ([record.get("meeting_id")] if record.get("meeting_id") else []),
                        "config": payload.get("config_path"),
                        "mode": "validate-only" if payload.get("validate_only") else "run",
                        "status": self._normalize_history_status(record),
                        "started_at": record.get("started_at_utc"),
                        "ended_at": record.get("ended_at_utc"),
                        "artifact_digest": record.get("artifact_digest"),
                        "source": "history",
                    }
                )
        return rows

    def _normalize_history_status(self, record: dict[str, Any]) -> str:
        """Map persisted batch-runner statuses into UI status labels."""
        status = record.get("status", "unknown")
        if (
            status == "failed"
            and record.get("action") == "validate"
            and record.get("error") == "run_manifest_missing_for_validate_only"
        ):
            return "not_run"
        return status

    def _serialize_live_run(self, record: RunRecord) -> dict[str, Any]:
        """Serialize the in-memory subset used by list endpoints."""
        return {
            "run_id": record.run_id,
            "meeting_id": record.meeting_id,
            "meeting_ids": record.meeting_ids,
            "config": record.config,
            "mode": record.mode,
            "status": record.status,
            "started_at": record.started_at,
            "ended_at": record.ended_at,
            "artifact_digest": record.artifact_digest,
            "source": "live",
        }

    def _persist_registry(self) -> None:
        """Persist the current run registry for backend restarts."""
        self.settings.batch_runs_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {"runs": [self._record_to_json(record) for record in self._runs.values()]}
        self._registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_registry(self) -> None:
        """Restore previously persisted run metadata when the backend starts."""
        if not self._registry_path.exists():
            return
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        loaded: dict[str, RunRecord] = {}
        for row in payload.get("runs", []):
            record = RunRecord(
                run_id=row["run_id"],
                run_label=row.get("run_label", f"ui_{row.get('mode', 'run').replace('-', '_')}_{row.get('meeting_id', 'unknown')}_{row['run_id']}"),
                meeting_id=row["meeting_id"],
                meeting_ids=row.get("meeting_ids", [row["meeting_id"]]),
                config=row["config"],
                mode=row["mode"],
                status=row["status"],
                started_at=row.get("started_at"),
                ended_at=row.get("ended_at"),
                command=row.get("command", []),
                recent_logs=deque(row.get("recent_logs", []), maxlen=200),
                exit_code=row.get("exit_code"),
                summary=row.get("summary"),
                artifact_digest=row.get("artifact_digest"),
                process=None,
                stage_events=row.get("stage_events", []),
                cancelled=row.get("cancelled", False),
            )
            self._reconcile_loaded_record(record)
            loaded[record.run_id] = record

        with self._lock:
            self._runs = loaded

    def _reconcile_loaded_record(self, record: RunRecord) -> None:
        """Refresh a restored record from summary artifacts and stage traces."""
        summary = self._read_summary(record.run_label)
        if summary:
            record.summary = summary
            if len(record.meeting_ids) == 1:
                matching = next((row for row in summary.get("records", []) if row.get("meeting_id") == record.meeting_ids[0]), None)
            else:
                matching = None
            if matching:
                record.artifact_digest = matching.get("artifact_digest")
                record.started_at = matching.get("started_at_utc", record.started_at)
                record.ended_at = matching.get("ended_at_utc", record.ended_at)
                record.status = "completed" if matching.get("status") == "ok" else self._normalize_history_status(matching)
        self._refresh_stage_events(record)
        if record.status in {"queued", "running"} and record.process is None:
            record.status = "failed"
            record.ended_at = record.ended_at or datetime.now(timezone.utc).isoformat()
            record.recent_logs.append("UI backend restarted before active run completion could be confirmed.")

    def _record_to_json(self, record: RunRecord) -> dict[str, Any]:
        """Convert a run record into a JSON-serializable persistence payload."""
        return {
            "run_id": record.run_id,
            "run_label": record.run_label,
            "meeting_id": record.meeting_id,
            "meeting_ids": record.meeting_ids,
            "config": record.config,
            "mode": record.mode,
            "status": record.status,
            "started_at": record.started_at,
            "ended_at": record.ended_at,
            "command": record.command,
            "recent_logs": list(record.recent_logs),
            "exit_code": record.exit_code,
            "summary": record.summary,
            "artifact_digest": record.artifact_digest,
            "stage_events": record.stage_events,
            "cancelled": record.cancelled,
        }
