"""Traceability helpers for reproducibility and offline governance.

These helpers collect hashes, environment metadata, and stage-level execution
events that are written alongside pipeline artifacts. The outputs are designed
to be stable enough for audit and comparison across repeated offline runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .io_utils import ensure_dir


def _sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for an in-memory payload."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    """Stream a file from disk and return its SHA-256 digest."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def config_digest(cfg: AppConfig) -> str:
    """Hash the normalized config payload used for a pipeline run."""
    payload = json.dumps(cfg.model_dump(), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def collect_environment_snapshot() -> dict[str, Any]:
    """Capture a small deterministic environment snapshot for manifests."""
    keys = [
        "PYTHONHASHSEED",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "CUBLAS_WORKSPACE_CONFIG",
        "TOKENIZERS_PARALLELISM",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "env": {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None},
    }


def collect_code_provenance(repo_root: Path) -> dict[str, Any]:
    """Hash the core code files that materially affect pipeline outputs."""
    candidates = [
        repo_root / "src/ami_mom_pipeline/pipeline.py",
        repo_root / "src/ami_mom_pipeline/config.py",
        repo_root / "src/ami_mom_pipeline/backends/nemo_backend.py",
        repo_root / "src/ami_mom_pipeline/backends/llama_cpp_backend.py",
        repo_root / "scripts/nemo_diarize.py",
        repo_root / "scripts/nemo_asr.py",
        repo_root / "scripts/run_nemo_batch_sequential.py",
    ]
    files = []
    for p in candidates:
        if not p.exists():
            continue
        files.append(
            {
                "path": str(p),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
        )
    aggregate = hashlib.sha256()
    for rec in sorted(files, key=lambda r: r["path"]):
        aggregate.update(rec["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(rec["sha256"].encode("utf-8"))
        aggregate.update(b"\0")
    return {"files": files, "aggregate_sha256": aggregate.hexdigest()}


def offline_preflight_audit(cfg: AppConfig) -> dict[str, Any]:
    """Check offline-sensitive config fields before a run starts."""
    violations: list[str] = []
    warnings: list[str] = []

    def _check_local_path(name: str, value: str | None) -> None:
        """Validate that configured model paths are local filesystem paths."""
        if not value:
            return
        if "://" in value:
            violations.append(f"{name}:url_not_allowed:{value}")
        elif not Path(value).expanduser().exists():
            warnings.append(f"{name}:path_missing:{value}")

    _check_local_path("nemo.vad_model_path", cfg.pipeline.speech_backend.nemo.vad_model_path)
    _check_local_path("nemo.diarizer_config_path", cfg.pipeline.speech_backend.nemo.diarizer_config_path)
    _check_local_path("nemo.asr_model_path", cfg.pipeline.speech_backend.nemo.asr_model_path)
    _check_local_path("summarization.llama_cpp.model_path", cfg.pipeline.summarization_backend.llama_cpp.model_path)
    _check_local_path("extraction.llama_cpp.model_path", cfg.pipeline.extraction_backend.llama_cpp.model_path)

    command_fields = {
        "nemo.vad_command": cfg.pipeline.speech_backend.nemo.vad_command,
        "nemo.diarization_command": cfg.pipeline.speech_backend.nemo.diarization_command,
        "nemo.asr_command": cfg.pipeline.speech_backend.nemo.asr_command,
    }
    for name, cmd in command_fields.items():
        if not cmd:
            continue
        lowered = cmd.lower()
        if "http://" in lowered or "https://" in lowered:
            violations.append(f"{name}:contains_url")
        if "wget " in lowered or "curl " in lowered:
            violations.append(f"{name}:network_downloader_detected")

    env_snapshot = collect_environment_snapshot()
    if cfg.runtime.offline:
        if env_snapshot["env"].get("HF_HUB_OFFLINE") not in {"1", "true", "True"}:
            warnings.append("HF_HUB_OFFLINE_not_set")

    return {
        "offline_requested": bool(cfg.runtime.offline),
        "violations": sorted(set(violations)),
        "warnings": sorted(set(warnings)),
        "ok": not violations,
        "environment": env_snapshot,
    }


@dataclass
class StageTraceWriter:
    """Append-only writer for per-stage execution events."""

    path: Path
    enabled: bool = True
    truncate_on_init: bool = True

    def __post_init__(self) -> None:
        """Prepare the output file when tracing is enabled."""
        if self.enabled:
            ensure_dir(self.path.parent)
            if self.truncate_on_init:
                self.path.write_text("", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        """Write one JSONL event to disk when tracing is enabled."""
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, ensure_ascii=True))
            f.write("\n")


def trace_stage(
    writer: StageTraceWriter,
    stage_name: str,
    func,
    *,
    meeting_id: str,
    summarizer=None,
) -> Any:
    """Run a stage, record start/end events, and return the stage output."""
    start_perf = time.perf_counter()
    start_wall = time.time()
    writer.write({"event": "stage_start", "meeting_id": meeting_id, "stage": stage_name, "ts_unix": round(start_wall, 6)})
    try:
        out = func()
        elapsed = time.perf_counter() - start_perf
        payload: dict[str, Any] = {
            "event": "stage_end",
            "meeting_id": meeting_id,
            "stage": stage_name,
            "status": "ok",
            "elapsed_sec": round(elapsed, 6),
        }
        if summarizer is not None:
            try:
                payload["summary"] = summarizer(out)
            except Exception as exc:
                payload["summary_error"] = f"{type(exc).__name__}: {exc}"
        writer.write(payload)
        return out
    except Exception as exc:
        elapsed = time.perf_counter() - start_perf
        writer.write(
            {
                "event": "stage_end",
                "meeting_id": meeting_id,
                "stage": stage_name,
                "status": "error",
                "elapsed_sec": round(elapsed, 6),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
