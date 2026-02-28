from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.schemas.api_models import ArtifactEntry, MeetingListItem, StageArtifactLink, StageStatus
from app.services.artifact_reader import infer_kind, read_csv, read_json, read_jsonl
from app.services.digest import safe_stat_size
from app.services.security import PathSecurity

PIPELINE_STAGES: list[dict[str, Any]] = [
    {"key": "ingest", "name": "Ingest", "artifacts": ["staged_audio"]},
    {"key": "vad", "name": "VAD", "artifacts": ["vad_segments.json", "vad_segments.rttm"]},
    {"key": "diarization", "name": "Diarization", "artifacts": ["diarization_segments.json", "diarization.rttm"]},
    {"key": "asr", "name": "ASR", "artifacts": ["asr_segments.json", "asr_confidence.json", "full_transcript.txt"]},
    {"key": "canonicalization", "name": "Canonicalization", "artifacts": ["transcript_raw.json", "transcript_normalized.json"]},
    {"key": "chunking", "name": "Chunking", "artifacts": ["transcript_chunks.jsonl"]},
    {"key": "retrieval", "name": "Retrieval", "artifacts": ["retrieval_results.json"], "optional": True},
    {"key": "summarization", "name": "Summarization", "artifacts": ["mom_summary.json", "mom_summary.html"]},
    {"key": "extraction", "name": "Extraction", "artifacts": ["decisions_actions.json", "extraction_validation_report.json"]},
    {"key": "evaluation", "name": "Evaluation", "artifacts": ["eval_meeting_metrics"]},
    {"key": "finalize", "name": "Finalize", "artifacts": ["preflight_offline_audit.json", "reproducibility_report.json", "stage_trace.jsonl", "run_manifest.json"]},
]


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    resolver: str


class FilesystemIndexer:
    def __init__(self, settings: Settings, security: PathSecurity) -> None:
        self.settings = settings
        self.security = security

    def list_meetings(self) -> list[MeetingListItem]:
        meeting_ids = self._discover_meeting_ids()
        meetings = [self.build_meeting_summary(meeting_id) for meeting_id in meeting_ids]
        meetings.sort(key=lambda row: (row.last_updated or datetime.min.replace(tzinfo=timezone.utc), row.meeting_id), reverse=True)
        return meetings

    def build_meeting_summary(self, meeting_id: str) -> MeetingListItem:
        meeting_dir = self.settings.artifacts_dir / meeting_id
        run_manifest = self._read_json_if_exists(meeting_dir / "run_manifest.json")
        repro = self._read_json_if_exists(meeting_dir / "reproducibility_report.json")
        last_updated = self._max_mtime(
            [
                meeting_dir / "run_manifest.json",
                meeting_dir / "stage_trace.jsonl",
                self.settings.raw_ami_audio_dir / f"{meeting_id}.Mix-Headset.wav",
                self.settings.project_root / "data/staged/ami/audio_clean" / f"{meeting_id}.wav",
            ]
        )
        stages = self.compute_stage_status(meeting_id)
        return MeetingListItem(
            meeting_id=meeting_id,
            has_raw_audio=(self.settings.raw_ami_audio_dir / f"{meeting_id}.Mix-Headset.wav").exists(),
            has_staged_audio=(self.settings.project_root / "data/staged/ami/audio_clean" / f"{meeting_id}.wav").exists(),
            has_artifacts=meeting_dir.exists(),
            last_updated=last_updated,
            config_digest=run_manifest.get("config_digest") if run_manifest else None,
            artifact_digest=run_manifest.get("artifact_digest") if run_manifest else None,
            offline_preflight_ok=run_manifest.get("offline_preflight_ok") if run_manifest else None,
            determinism_risks=((repro or {}).get("determinism") or {}).get("risks", []),
            stages_complete=sum(1 for stage in stages if stage.status == "success"),
            stage_count=len(stages),
        )

    def compute_stage_status(self, meeting_id: str) -> list[StageStatus]:
        trace_data = self._read_stage_trace(meeting_id)
        results: list[StageStatus] = []
        for definition in PIPELINE_STAGES:
            artifacts = [self.describe_artifact(meeting_id, artifact_name) for artifact_name in definition["artifacts"]]
            present_count = sum(1 for artifact in artifacts if artifact.exists)
            optional = definition.get("optional", False)
            trace_status = trace_data.get(definition["key"], {}).get("status")
            runtime = trace_data.get(definition["key"], {}).get("elapsed_sec")

            if trace_status not in {None, "ok"}:
                status = "fail"
            elif present_count == len(artifacts) and len(artifacts) > 0:
                status = "success"
            elif present_count > 0:
                status = "in_progress"
            elif optional:
                status = "not_run"
            else:
                status = "not_run"

            notes: list[str] = []
            if trace_status not in {None, "ok"}:
                notes.append(f"Stage trace recorded status: {trace_status}")
            elif runtime is None and status in {"success", "warn"}:
                notes.append("No stage trace runtime recorded")

            results.append(
                StageStatus(
                    name=definition["name"],
                    key=definition["key"],
                    status=status,
                    runtime_sec=runtime,
                    artifacts=[
                        StageArtifactLink(
                            name=artifact.name,
                            exists=artifact.exists,
                            artifact_url=artifact.preview_url,
                            download_url=artifact.download_url,
                        )
                        for artifact in artifacts
                    ],
                    notes=notes,
                )
            )
        return results

    def list_artifacts(self, meeting_id: str) -> list[ArtifactEntry]:
        artifact_names = [
            "raw_audio",
            "staged_audio",
            "vad_segments.json",
            "vad_segments.rttm",
            "diarization_segments.json",
            "diarization.rttm",
            "asr_segments.json",
            "asr_confidence.json",
            "full_transcript.txt",
            "transcript_raw.json",
            "transcript_normalized.json",
            "transcript_chunks.jsonl",
            "retrieval_results.json",
            "mom_summary.json",
            "mom_summary.html",
            "decisions_actions.json",
            "extraction_validation_report.json",
            "preflight_offline_audit.json",
            "reproducibility_report.json",
            "stage_trace.jsonl",
            "run_manifest.json",
            "eval_wer_breakdown.json",
        ]
        return [self.describe_artifact(meeting_id, name) for name in artifact_names]

    def describe_artifact(self, meeting_id: str, artifact_name: str) -> ArtifactEntry:
        path = self.resolve_artifact_path(meeting_id, artifact_name)
        kind = infer_kind(path)
        relative = self.security.to_project_relative(path) if path.exists() else self._relative_fallback(path)
        return ArtifactEntry(
            name=artifact_name,
            path=str(path),
            relative_path=relative,
            exists=path.exists(),
            kind=kind,
            size_bytes=safe_stat_size(path),
            download_url=f"/api/meetings/{meeting_id}/artifact/{artifact_name}/download",
            preview_url=f"/api/meetings/{meeting_id}/artifact/{artifact_name}",
        )

    def resolve_artifact_path(self, meeting_id: str, artifact_name: str) -> Path:
        self.security.validate_relative_input(meeting_id)
        self.security.validate_relative_input(artifact_name)
        meeting_dir = self.settings.artifacts_dir / meeting_id
        mapping: dict[str, Path] = {
            "raw_audio": self.settings.raw_ami_audio_dir / f"{meeting_id}.Mix-Headset.wav",
            "staged_audio": self.settings.project_root / "data/staged/ami/audio_clean" / f"{meeting_id}.wav",
            "eval_wer_breakdown.json": self.settings.eval_dir / "wer_breakdown.json",
        }
        if artifact_name in mapping:
            return mapping[artifact_name]
        return meeting_dir / artifact_name

    def get_eval_summary(self) -> dict[str, Any]:
        rows = read_csv(self.settings.eval_dir / "wer_scores.csv") if (self.settings.eval_dir / "wer_scores.csv").exists() else []
        numeric_columns = ("wer", "cer", "cpwer", "der", "mean_confidence")
        aggregate: dict[str, Any] = {"meeting_count": len(rows)}
        for column in numeric_columns:
            values = [float(row[column]) for row in rows if row.get(column) not in (None, "", "null")]
            if values:
                aggregate[f"mean_{column}"] = round(sum(values) / len(values), 6)
        latest_meeting = rows[-1]["meeting_id"] if rows else None
        return {"aggregate_metrics": aggregate, "rows": rows, "latest_meeting": latest_meeting}

    def get_meeting_eval(self, meeting_id: str) -> dict[str, Any]:
        metrics = self._read_json_if_exists(self.settings.eval_dir / "wer_breakdown.json")
        quality = self._read_json_if_exists(self.settings.eval_dir / "mom_quality_checks.json")
        confidence = self._read_json_if_exists(self.settings.artifacts_dir / meeting_id / "asr_confidence.json")
        if metrics and metrics.get("meeting_id") != meeting_id:
            scores = read_csv(self.settings.eval_dir / "wer_scores.csv") if (self.settings.eval_dir / "wer_scores.csv").exists() else []
            matching = next((row for row in scores if row.get("meeting_id") == meeting_id), {})
            metrics = {**matching, **({"meeting_id": meeting_id} if matching else {})}
        if quality and quality.get("meeting_id") != meeting_id:
            quality = None
        return {"metrics": metrics or {}, "confidence": confidence, "quality_checks": quality}

    def get_meeting_speech(self, meeting_id: str) -> dict[str, Any]:
        audio_artifact = self.describe_artifact(
            meeting_id,
            "staged_audio" if self.resolve_artifact_path(meeting_id, "staged_audio").exists() else "raw_audio",
        )
        return {
            "meeting_id": meeting_id,
            "audio": {
                "artifact": audio_artifact.model_dump(),
                "available": audio_artifact.exists,
            },
            "vad_segments": self._read_list_if_exists(self.settings.artifacts_dir / meeting_id / "vad_segments.json"),
            "diarization_segments": self._read_list_if_exists(self.settings.artifacts_dir / meeting_id / "diarization_segments.json"),
            "asr_segments": self._read_list_if_exists(self.settings.artifacts_dir / meeting_id / "asr_segments.json"),
        }

    def get_meeting_transcript(self, meeting_id: str) -> dict[str, Any]:
        return {
            "meeting_id": meeting_id,
            "raw": self._read_list_if_exists(self.settings.artifacts_dir / meeting_id / "transcript_raw.json"),
            "normalized": self._read_list_if_exists(self.settings.artifacts_dir / meeting_id / "transcript_normalized.json"),
            "chunks": self._read_list_if_exists(self.settings.artifacts_dir / meeting_id / "transcript_chunks.jsonl", jsonl=True),
        }

    def get_meeting_summary(self, meeting_id: str) -> dict[str, Any]:
        html_artifact = self.describe_artifact(meeting_id, "mom_summary.html")
        return {
            "meeting_id": meeting_id,
            "summary": self._read_json_if_exists(self.settings.artifacts_dir / meeting_id / "mom_summary.json") or {},
            "html_available": html_artifact.exists,
            "html_download_url": html_artifact.download_url if html_artifact.exists else None,
        }

    def get_meeting_extraction(self, meeting_id: str) -> dict[str, Any]:
        return {
            "meeting_id": meeting_id,
            "extraction": self._read_json_if_exists(self.settings.artifacts_dir / meeting_id / "decisions_actions.json") or {},
            "validation_report": self._read_json_if_exists(self.settings.artifacts_dir / meeting_id / "extraction_validation_report.json") or {},
        }

    def get_meeting_repro(self, meeting_id: str) -> dict[str, Any]:
        meeting_dir = self.settings.artifacts_dir / meeting_id
        run_manifest = self._read_json_if_exists(meeting_dir / "run_manifest.json")
        offline = self._read_json_if_exists(meeting_dir / "preflight_offline_audit.json")
        repro = self._read_json_if_exists(meeting_dir / "reproducibility_report.json")
        return {
            "meeting_id": meeting_id,
            "config_digest": (run_manifest or {}).get("config_digest"),
            "artifact_digest": (run_manifest or {}).get("artifact_digest"),
            "offline_audit": offline,
            "reproducibility_report": repro,
            "run_manifest": run_manifest,
            "determinism_risks": ((repro or {}).get("determinism") or {}).get("risks", []),
        }

    def list_configs(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.settings.configs_dir.glob("*.y*ml")):
            self.security.ensure_readable(path)
            rows.append({"name": path.name, "path": str(path), "size_bytes": path.stat().st_size})
        return rows

    def read_config(self, name: str) -> dict[str, Any]:
        self.security.validate_relative_input(name)
        path = self.settings.configs_dir / name
        self.security.ensure_readable(path)
        if not path.exists():
            raise FileNotFoundError(name)
        return {"name": path.name, "path": str(path), "content": path.read_text(encoding="utf-8")}

    def get_dashboard(self) -> dict[str, Any]:
        meetings = self.list_meetings()
        eval_summary = self.get_eval_summary()
        mlruns_root = self.settings.project_root / "artifacts" / "mlruns"
        return {
            "system_state": {
                "offline_mode": True,
                "mlflow_logging": mlruns_root.exists(),
                "strict_determinism": True,
                "run_controls_enabled": self.settings.run_controls_enabled,
            },
            "last_run": meetings[0] if meetings else None,
            "aggregate_metrics": eval_summary["aggregate_metrics"],
            "meetings": meetings[:10],
        }

    def get_governance(self) -> dict[str, Any]:
        return {"evidence_bundles": self.list_evidence_bundles(), "mlflow": self.list_mlflow_runs()}

    def list_evidence_bundles(self) -> list[dict[str, Any]]:
        evidence_root = self.settings.project_root / "artifacts" / "acceptance_bundles"
        evidence = []
        if evidence_root.exists():
            for path in sorted(evidence_root.iterdir()):
                evidence.append({"name": path.name, "path": str(path), "size_bytes": safe_stat_size(path)})
        return evidence

    def list_mlflow_runs(self) -> dict[str, Any]:
        mlruns_root = self.settings.project_root / "artifacts" / "mlruns"
        mlflow = {"configured": mlruns_root.exists(), "runs": []}
        if mlruns_root.exists():
            for path in sorted(mlruns_root.glob("*/*/meta.yaml")):
                mlflow["runs"].append({"path": str(path)})
        return mlflow

    def _discover_meeting_ids(self) -> list[str]:
        audio_ids = {
            path.name.removesuffix(".Mix-Headset.wav")
            for path in self.settings.raw_ami_audio_dir.glob("*.Mix-Headset.wav")
        }
        artifact_ids = {path.name for path in self.settings.artifacts_dir.iterdir() if path.is_dir()} if self.settings.artifacts_dir.exists() else set()
        return sorted(audio_ids | artifact_ids)

    def _read_json_if_exists(self, path: Path) -> dict[str, Any] | None:
        if path.exists():
            return read_json(path)
        return None

    def _read_list_if_exists(self, path: Path, jsonl: bool = False) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        if jsonl:
            return read_jsonl(path)
        payload = read_json(path)
        return payload if isinstance(payload, list) else []

    def _read_stage_trace(self, meeting_id: str) -> dict[str, dict[str, Any]]:
        path = self.settings.artifacts_dir / meeting_id / "stage_trace.jsonl"
        if not path.exists():
            return {}
        stage_data: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(path):
            if row.get("event") == "stage_end" and row.get("stage"):
                stage_data[row["stage"]] = {
                    "elapsed_sec": row.get("elapsed_sec"),
                    "status": row.get("status"),
                }
        return stage_data

    def _max_mtime(self, candidates: list[Path]) -> datetime | None:
        mtimes = []
        for path in candidates:
            if path.exists():
                mtimes.append(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
        return max(mtimes) if mtimes else None

    def _relative_fallback(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.settings.project_root))
        except ValueError:
            return str(path)
