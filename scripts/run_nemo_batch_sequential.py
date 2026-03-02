#!/usr/bin/env python3
"""Sequential batch runner for AMI pipeline execution and validation.

Primary responsibilities:
- deterministic meeting selection order
- resume/skip semantics via `run_manifest.json`
- per-meeting and aggregate batch logs
- artifact contract validation
- optional post-run speech metrics evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ami_mom_pipeline.config import AppConfig  # noqa: E402
from ami_mom_pipeline.pipeline import list_meetings, run_pipeline  # noqa: E402


REQUIRED_MEETING_FILES = [
    "vad_segments.json",
    "vad_segments.rttm",
    "diarization.rttm",
    "diarization_segments.json",
    "asr_segments.json",
    "full_transcript.txt",
    "transcript_raw.json",
    "transcript_normalized.json",
    "transcript_chunks.jsonl",
    "mom_summary.json",
    "mom_summary.html",
    "decisions_actions.json",
    "extraction_validation_report.json",
    "preflight_offline_audit.json",
    "reproducibility_report.json",
    "stage_trace.jsonl",
    "run_manifest.json",
]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for sequential batch execution workflow."""
    p = argparse.ArgumentParser(
        description="Sequential AMI NeMo batch runner with resume, timing logs, and artifact validation"
    )
    p.add_argument("--config", default="configs/pipeline.nemo.yaml", help="Path to pipeline YAML config")
    p.add_argument("--prefix", default=None, help="Meeting ID prefix filter (e.g., ES2002)")
    p.add_argument("--limit", type=int, default=None, help="Max number of meetings after filtering")
    p.add_argument(
        "--meeting-id",
        dest="meeting_ids",
        action="append",
        default=None,
        help="Explicit meeting ID (repeatable). If set, bypasses list-meetings discovery.",
    )
    p.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip meetings with run_manifest.json")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Rerun meetings even if run_manifest.json exists")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    p.add_argument("--validate-only", action="store_true", help="Do not run pipeline stages; only validate existing artifacts")
    p.add_argument("--require-nemo", action="store_true", default=True, help="Fail unless speech backend is nemo")
    p.add_argument("--allow-non-nemo", dest="require_nemo", action="store_false", help="Allow non-nemo config (debug)")
    p.add_argument(
        "--batch-log-dir",
        default="artifacts/batch_runs",
        help="Directory for batch timing/status/validation logs",
    )
    p.add_argument("--run-label", default=None, help="Optional label for output log filenames")
    p.add_argument("--skip-speech-eval", action="store_true", help="Skip post-run DER/cpWER/WER evaluation")
    p.add_argument("--speech-eval-collar-sec", type=float, default=0.25, help="DER collar seconds for post-run speech eval")
    p.add_argument(
        "--dvc-template",
        choices=["single", "batch"],
        default=None,
        help="Generate a matching DVC stage template for the selected meetings",
    )
    p.add_argument("--dvc-template-output", default=None, help="Optional explicit output path for generated DVC template")
    return p


def main() -> int:
    """CLI entry point for sequential batch execution."""
    args = build_parser().parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = AppConfig.load(str(cfg_path))
    if args.require_nemo and cfg.pipeline.speech_backend.mode != "nemo":
        print(
            f"Refusing to run: speech_backend.mode={cfg.pipeline.speech_backend.mode!r} (expected 'nemo'). "
            "Use --allow-non-nemo to override.",
            file=sys.stderr,
        )
        return 2

    meeting_ids = _select_meetings(cfg, args.meeting_ids, args.prefix, args.limit)
    if not meeting_ids:
        print("No meetings selected after filters.", file=sys.stderr)
        return 1

    log_dir = _resolve_log_dir(args.batch_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_label = args.run_label or datetime.now(timezone.utc).strftime("nemo_batch_%Y%m%dT%H%M%SZ")
    events_path = log_dir / f"{run_label}.events.jsonl"
    timings_csv_path = log_dir / f"{run_label}.timings.csv"
    summary_path = log_dir / f"{run_label}.summary.json"
    validation_path = log_dir / f"{run_label}.validation.json"
    speech_eval_csv_path = log_dir / f"{run_label}.speech_metrics.csv"
    speech_eval_json_path = log_dir / f"{run_label}.speech_metrics.summary.json"

    records: list[dict[str, Any]] = []
    batch_started = time.monotonic()
    print(f"Selected {len(meeting_ids)} meetings")
    print(f"Config: {cfg_path}")
    print(f"Logs: {events_path}")
    dvc_template_info = None
    if args.dvc_template:
        dvc_template_info = _generate_dvc_template(cfg_path, meeting_ids, mode=args.dvc_template, output=args.dvc_template_output)
        if dvc_template_info.get("output"):
            print(f"DVC template: {dvc_template_info['output']}")
        else:
            print(f"DVC template generation error: {dvc_template_info.get('error')}")

    for idx, meeting_id in enumerate(meeting_ids, start=1):
        if args.validate_only:
            record = _build_validate_only_record(cfg, meeting_id, idx, resume=args.resume)
        else:
            record = _process_meeting(cfg, meeting_id, idx, len(meeting_ids), resume=args.resume)
        records.append(record)
        _append_jsonl(events_path, record)
        status = record["status"]
        action = record["action"]
        elapsed = record.get("elapsed_sec", 0.0)
        print(f"[{idx}/{len(meeting_ids)}] {meeting_id} {action}/{status} {elapsed:.3f}s")
        if not args.validate_only and status == "failed" and args.fail_fast:
            print("Stopping due to --fail-fast")
            break

    total_elapsed = round(time.monotonic() - batch_started, 3)
    _write_timings_csv(timings_csv_path, records)

    selected_processed = [r["meeting_id"] for r in records]
    validation = validate_artifacts(cfg, selected_processed)
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    speech_eval_summary = None
    if not args.skip_speech_eval and selected_processed:
        speech_eval_summary = run_speech_eval(
            config_path=cfg_path,
            meeting_ids=selected_processed,
            out_csv=speech_eval_csv_path,
            out_json=speech_eval_json_path,
            collar_sec=args.speech_eval_collar_sec,
        )

    summary = build_summary(
        cfg=cfg,
        config_path=cfg_path,
        records=records,
        total_elapsed=total_elapsed,
        events_path=events_path,
        timings_csv_path=timings_csv_path,
        validation_path=validation_path,
        speech_eval_summary=speech_eval_summary,
        speech_eval_csv_path=speech_eval_csv_path if speech_eval_summary else None,
        speech_eval_json_path=speech_eval_json_path if speech_eval_summary else None,
        dvc_template_info=dvc_template_info,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log_batch_to_mlflow(cfg, run_label, summary, summary_path)

    print(
        "Batch complete: "
        f"ok={summary['counts']['ok']} skipped={summary['counts']['skipped']} failed={summary['counts']['failed']} "
        f"validation_ok={validation['summary']['meetings_valid']}/{validation['summary']['meetings_checked']} "
        f"elapsed={total_elapsed:.3f}s"
    )
    if speech_eval_summary:
        s = speech_eval_summary.get("summary", {})
        print(
            "Speech metrics: "
            f"WER={_fmt_metric(s.get('mean_wer'))} "
            f"cpWER={_fmt_metric(s.get('mean_cpwer'))} "
            f"DER={_fmt_metric(s.get('mean_der'))} "
            f"(n={s.get('meetings_checked', 0)})"
        )
    print(f"Timing CSV: {timings_csv_path}")
    print(f"Validation: {validation_path}")
    if speech_eval_summary:
        print(f"Speech eval CSV: {speech_eval_csv_path}")
        print(f"Speech eval JSON: {speech_eval_json_path}")
    print(f"Summary: {summary_path}")

    if summary["counts"]["failed"] > 0 or validation["summary"]["meetings_invalid"] > 0:
        return 1
    return 0


def _select_meetings(cfg: AppConfig, explicit: list[str] | None, prefix: str | None, limit: int | None) -> list[str]:
    if explicit:
        meetings = list(dict.fromkeys(explicit))
    else:
        meetings = list_meetings(cfg)
    if prefix:
        meetings = [m for m in meetings if m.startswith(prefix)]
    if limit is not None:
        meetings = meetings[:limit]
    return meetings


def _resolve_log_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p


def _meeting_artifact_dir(cfg: AppConfig, meeting_id: str) -> Path:
    return Path(cfg.paths.artifacts_dir) / "ami" / meeting_id


def _process_meeting(cfg: AppConfig, meeting_id: str, index: int, total: int, resume: bool) -> dict[str, Any]:
    del total
    started_dt = datetime.now(timezone.utc)
    started = time.monotonic()
    artifact_dir = _meeting_artifact_dir(cfg, meeting_id)
    manifest_path = artifact_dir / "run_manifest.json"

    base: dict[str, Any] = {
        "order": index,
        "meeting_id": meeting_id,
        "started_at_utc": started_dt.isoformat(),
        "resume_enabled": resume,
        "artifact_dir": str(artifact_dir),
    }

    if resume and manifest_path.exists():
        elapsed = round(time.monotonic() - started, 3)
        base.update(
            {
                "action": "skip",
                "status": "skipped",
                "reason": "run_manifest_exists",
                "manifest_path": str(manifest_path),
                "elapsed_sec": elapsed,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            base["artifact_digest"] = manifest.get("artifact_digest")
            base["manifest_pipeline_version"] = manifest.get("pipeline_version")
        except Exception as exc:
            base["manifest_read_error"] = f"{type(exc).__name__}: {exc}"
        return base

    try:
        manifest = run_pipeline(cfg, meeting_id)
        elapsed = round(time.monotonic() - started, 3)
        base.update(
            {
                "action": "run",
                "status": "ok",
                "elapsed_sec": elapsed,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_path": str(manifest_path),
                "artifact_digest": manifest.get("artifact_digest"),
            }
        )
        return base
    except Exception as exc:
        elapsed = round(time.monotonic() - started, 3)
        base.update(
            {
                "action": "run",
                "status": "failed",
                "elapsed_sec": elapsed,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return base


def _build_validate_only_record(cfg: AppConfig, meeting_id: str, index: int, resume: bool) -> dict[str, Any]:
    artifact_dir = _meeting_artifact_dir(cfg, meeting_id)
    manifest_path = artifact_dir / "run_manifest.json"
    now = datetime.now(timezone.utc).isoformat()
    rec: dict[str, Any] = {
        "order": index,
        "meeting_id": meeting_id,
        "started_at_utc": now,
        "ended_at_utc": now,
        "elapsed_sec": 0.0,
        "resume_enabled": resume,
        "artifact_dir": str(artifact_dir),
        "action": "validate",
        "status": "ok" if manifest_path.exists() else "failed",
    }
    if manifest_path.exists():
        rec["manifest_path"] = str(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rec["artifact_digest"] = manifest.get("artifact_digest")
            rec["manifest_pipeline_version"] = manifest.get("pipeline_version")
        except Exception as exc:
            rec["status"] = "failed"
            rec["manifest_read_error"] = f"{type(exc).__name__}: {exc}"
    else:
        rec["error"] = "run_manifest_missing_for_validate_only"
        rec["error_type"] = "FileNotFoundError"
    return rec

def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
        f.write("\n")


def _write_timings_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "order",
        "meeting_id",
        "action",
        "status",
        "elapsed_sec",
        "started_at_utc",
        "ended_at_utc",
        "artifact_digest",
        "reason",
        "error_type",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})


def validate_artifacts(cfg: AppConfig, meeting_ids: list[str]) -> dict[str, Any]:
    artifacts_root = Path(cfg.paths.artifacts_dir)
    canonical_index = _load_jsonl_index(artifacts_root / "ami" / "meetings_canonical.jsonl", "meeting_id")
    wer_index = _load_csv_index(artifacts_root / "eval" / "ami" / "wer_scores.csv", "meeting_id")
    rouge_index = _load_csv_index(artifacts_root / "eval" / "ami" / "rouge_scores.csv", "meeting_id")

    global_checks = {
        "canonical_jsonl_exists": (artifacts_root / "ami" / "meetings_canonical.jsonl").exists(),
        "wer_scores_csv_exists": (artifacts_root / "eval" / "ami" / "wer_scores.csv").exists(),
        "rouge_scores_csv_exists": (artifacts_root / "eval" / "ami" / "rouge_scores.csv").exists(),
        "wer_breakdown_json_exists": (artifacts_root / "eval" / "ami" / "wer_breakdown.json").exists(),
        "mom_quality_checks_json_exists": (artifacts_root / "eval" / "ami" / "mom_quality_checks.json").exists(),
    }

    meetings: list[dict[str, Any]] = []
    valid_count = 0
    for meeting_id in meeting_ids:
        result = _validate_meeting(cfg, meeting_id, canonical_index, wer_index, rouge_index)
        meetings.append(result)
        if result["valid"]:
            valid_count += 1

    return {
        "summary": {
            "meetings_checked": len(meeting_ids),
            "meetings_valid": valid_count,
            "meetings_invalid": len(meeting_ids) - valid_count,
            "global_checks_ok": all(global_checks.values()),
        },
        "global_checks": global_checks,
        "meetings": meetings,
    }


def run_speech_eval(
    config_path: Path,
    meeting_ids: list[str],
    out_csv: Path,
    out_json: Path,
    collar_sec: float,
) -> dict[str, Any] | None:
    if not meeting_ids:
        return None
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_speech_metrics.py"),
        "--config",
        str(config_path),
        "--collar-sec",
        str(collar_sec),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    for meeting_id in meeting_ids:
        cmd.extend(["--meeting-id", meeting_id])
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "summary": {
                "meetings_checked": len(meeting_ids),
                "mean_wer": None,
                "mean_cpwer": None,
                "mean_der": None,
                "error": f"speech_eval_subprocess_exit_{proc.returncode}",
            },
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    if out_json.exists():
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "summary": {
                    "meetings_checked": len(meeting_ids),
                    "mean_wer": None,
                    "mean_cpwer": None,
                    "mean_der": None,
                    "error": f"speech_eval_summary_read_error:{type(exc).__name__}",
                },
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
    return None


def _validate_meeting(
    cfg: AppConfig,
    meeting_id: str,
    canonical_index: dict[str, dict[str, Any]],
    wer_index: dict[str, dict[str, str]],
    rouge_index: dict[str, dict[str, str]],
) -> dict[str, Any]:
    artifact_dir = _meeting_artifact_dir(cfg, meeting_id)
    missing_files = [name for name in REQUIRED_MEETING_FILES if not (artifact_dir / name).exists()]
    errors: list[str] = []
    manifest_info: dict[str, Any] = {}

    manifest_path = artifact_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_info = {
                "meeting_id": manifest.get("meeting_id"),
                "speech_backend": manifest.get("speech_backend"),
                "artifact_digest": manifest.get("artifact_digest"),
            }
            if manifest.get("meeting_id") != meeting_id:
                errors.append("run_manifest_meeting_id_mismatch")
        except Exception as exc:
            errors.append(f"run_manifest_read_error:{type(exc).__name__}")
            manifest_info["read_error"] = str(exc)

    canonical_row = canonical_index.get(meeting_id)
    wer_row = wer_index.get(meeting_id)
    rouge_row = rouge_index.get(meeting_id)
    if canonical_row is None:
        errors.append("missing_canonical_jsonl_row")
    if wer_row is None:
        errors.append("missing_wer_scores_csv_row")
    if rouge_row is None:
        errors.append("missing_rouge_scores_csv_row")

    if not missing_files and (artifact_dir / "asr_segments.json").exists():
        try:
            asr_segments = json.loads((artifact_dir / "asr_segments.json").read_text(encoding="utf-8"))
            if not isinstance(asr_segments, list):
                errors.append("asr_segments_not_list")
            elif len(asr_segments) == 0:
                errors.append("asr_segments_empty")
        except Exception as exc:
            errors.append(f"asr_segments_read_error:{type(exc).__name__}")

    if (artifact_dir / "transcript_chunks.jsonl").exists():
        line_count = _nonempty_line_count(artifact_dir / "transcript_chunks.jsonl")
    else:
        line_count = 0

    valid = not missing_files and not errors
    return {
        "meeting_id": meeting_id,
        "valid": valid,
        "artifact_dir": str(artifact_dir),
        "missing_files": missing_files,
        "errors": errors,
        "aggregate_presence": {
            "canonical_jsonl": canonical_row is not None,
            "wer_scores_csv": wer_row is not None,
            "rouge_scores_csv": rouge_row is not None,
        },
        "quick_stats": {
            "chunk_lines": line_count,
        },
        "manifest": manifest_info,
    }


def _load_jsonl_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = row.get(key)
            if k is not None:
                out[str(k)] = row
    return out


def _load_csv_index(path: Path, key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row.get(key)
            if k:
                out[str(k)] = row
    return out


def _nonempty_line_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def build_summary(
    cfg: AppConfig,
    config_path: Path,
    records: list[dict[str, Any]],
    total_elapsed: float,
    events_path: Path,
    timings_csv_path: Path,
    validation_path: Path,
    speech_eval_summary: dict[str, Any] | None,
    speech_eval_csv_path: Path | None,
    speech_eval_json_path: Path | None,
    dvc_template_info: dict[str, Any] | None,
) -> dict[str, Any]:
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for rec in records:
        if rec["status"] == "ok":
            counts["ok"] += 1
        elif rec["status"] == "skipped":
            counts["skipped"] += 1
        elif rec["status"] == "failed":
            counts["failed"] += 1

    return {
        "config_path": str(config_path),
        "speech_backend": cfg.pipeline.speech_backend.mode,
        "meeting_count": len(records),
        "counts": counts,
        "total_elapsed_sec": total_elapsed,
        "events_jsonl": str(events_path),
        "timings_csv": str(timings_csv_path),
        "validation_json": str(validation_path),
        "speech_eval_csv": (str(speech_eval_csv_path) if speech_eval_csv_path else None),
        "speech_eval_json": (str(speech_eval_json_path) if speech_eval_json_path else None),
        "speech_eval_summary": speech_eval_summary,
        "dvc_template": dvc_template_info,
        "records": records,
    }


def _generate_dvc_template(config_path: Path, meeting_ids: list[str], mode: str, output: str | None) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_dvc_stage_template.py"),
        "--config",
        str(config_path),
        "--mode",
        mode,
    ]
    for m in meeting_ids:
        cmd.extend(["--meeting-id", m])
    if output:
        cmd.extend(["--output", output])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "error": f"generate_dvc_stage_template_exit_{proc.returncode}",
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    try:
        return json.loads(proc.stdout)
    except Exception as exc:
        return {
            "error": f"dvc_template_output_parse_error:{type(exc).__name__}",
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }


def _log_batch_to_mlflow(cfg: AppConfig, run_label: str, summary: dict[str, Any], summary_path: Path) -> None:
    if not cfg.runtime.enable_mlflow_logging:
        return
    try:
        import mlflow  # type: ignore
    except Exception:
        return
    tracking_uri = cfg.runtime.mlflow_tracking_uri or f"file:{ROOT / 'artifacts' / 'mlruns'}"
    if cfg.runtime.offline and not str(tracking_uri).startswith("file:"):
        return
    try:
        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(cfg.runtime.mlflow_experiment)
        with mlflow.start_run(run_name=f"batch:{run_label}"):
            counts = summary.get("counts", {}) or {}
            mlflow.log_params(
                {
                    "batch_run_label": run_label,
                    "config_path": summary.get("config_path"),
                    "speech_backend": summary.get("speech_backend"),
                    "meeting_count": summary.get("meeting_count"),
                    "offline": cfg.runtime.offline,
                }
            )
            mlflow.log_metrics(
                {
                    "batch_total_elapsed_sec": float(summary.get("total_elapsed_sec") or 0.0),
                    "batch_ok_count": float(counts.get("ok") or 0),
                    "batch_failed_count": float(counts.get("failed") or 0),
                    "batch_skipped_count": float(counts.get("skipped") or 0),
                }
            )
            speech = (summary.get("speech_eval_summary") or {}).get("summary", {})
            for key, metric_name in [
                ("mean_wer", "batch_mean_wer"),
                ("mean_cpwer", "batch_mean_cpwer"),
                ("mean_der", "batch_mean_der"),
            ]:
                if speech.get(key) is not None:
                    mlflow.log_metric(metric_name, float(speech[key]))
            if summary_path.exists():
                mlflow.log_artifact(str(summary_path), artifact_path="batch_runs")
    except Exception:
        # Batch logging is optional and should not fail the run.
        return


def _fmt_metric(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


if __name__ == "__main__":
    raise SystemExit(main())
