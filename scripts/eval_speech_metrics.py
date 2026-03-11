#!/usr/bin/env python3
"""Evaluate speech metrics from persisted AMI pipeline artifacts.

The script compares AMI reference annotations with generated ASR and
diarization artifacts, then writes offline CSV and JSON summaries. It reads
existing artifacts only and does not alter pipeline outputs beyond those
reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ami_mom_pipeline.config import AppConfig  # noqa: E402
from ami_mom_pipeline.pipeline import list_meetings  # noqa: E402
from ami_mom_pipeline.utils.speech_eval import (  # noqa: E402
    compute_cpwer,
    compute_der_approx_nooverlap,
    load_hyp_asr_views,
    load_hyp_diarization,
    load_ref_diarization_from_words,
    load_reference_asr_views,
    wer_from_texts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for speech-metric evaluation."""
    p = argparse.ArgumentParser(description="Evaluate AMI speech metrics: DER, cpWER, WER")
    p.add_argument("--config", default="configs/pipeline.nemo.yaml")
    p.add_argument("--prefix", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--meeting-id", dest="meeting_ids", action="append", default=None)
    p.add_argument("--discover", choices=("artifacts", "raw"), default="artifacts")
    p.add_argument("--collar-sec", type=float, default=0.25, help="DER reference collar (seconds)")
    p.add_argument("--include-overlap", action="store_true", help="Do not skip reference-overlap regions in DER (experimental)")
    p.add_argument(
        "--out-csv",
        default=None,
        help="Default: artifacts/eval/ami/speech_metrics.csv",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Default: artifacts/eval/ami/speech_metrics_summary.json",
    )
    return p


def main() -> int:
    """Run the selected speech evaluation workflow and write reports."""
    args = build_parser().parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = AppConfig.load(str(cfg_path))
    meeting_ids = select_meetings(cfg, args.meeting_ids, args.prefix, args.limit, args.discover)
    if not meeting_ids:
        print("No meetings selected.", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for meeting_id in meeting_ids:
        rows.append(
            evaluate_meeting(
                cfg,
                meeting_id,
                collar_sec=args.collar_sec,
                skip_overlap=not args.include_overlap,
            )
        )

    out_csv = Path(args.out_csv) if args.out_csv else (Path(cfg.paths.artifacts_dir) / "eval" / "ami" / "speech_metrics.csv")
    out_json = Path(args.out_json) if args.out_json else (Path(cfg.paths.artifacts_dir) / "eval" / "ami" / "speech_metrics_summary.json")
    write_csv(out_csv, rows)
    summary = summarize(rows, cfg_path, out_csv, args.collar_sec, skip_overlap=not args.include_overlap)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    s = summary["summary"]
    print(
        "Speech eval complete: "
        f"checked={s['meetings_checked']} "
        f"wer={fmt(s.get('mean_wer'))} cpWER={fmt(s.get('mean_cpwer'))} DER={fmt(s.get('mean_der'))} "
        f"rows_with_errors={s['rows_with_errors']}"
    )
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")
    return 0


def select_meetings(
    cfg: AppConfig,
    explicit: list[str] | None,
    prefix: str | None,
    limit: int | None,
    discover: str,
) -> list[str]:
    """Select meetings from explicit ids or deterministic discovery rules."""
    if explicit:
        meetings = list(dict.fromkeys(explicit))
    elif discover == "raw":
        meetings = list_meetings(cfg)
    else:
        artifacts_ami = Path(cfg.paths.artifacts_dir) / "ami"
        meetings = []
        if artifacts_ami.exists():
            for p in sorted(artifacts_ami.iterdir()):
                if p.is_dir() and (p / "diarization_segments.json").exists() and (p / "asr_segments.json").exists():
                    meetings.append(p.name)
    if prefix:
        meetings = [m for m in meetings if m.startswith(prefix)]
    if limit is not None:
        meetings = meetings[:limit]
    return meetings


def evaluate_meeting(cfg: AppConfig, meeting_id: str, collar_sec: float, skip_overlap: bool) -> dict[str, Any]:
    """Compute speech metrics for one meeting while tolerating partial failures."""
    annotations_dir = Path(cfg.paths.annotations_dir)
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    errors: list[str] = []

    row: dict[str, Any] = {
        "meeting_id": meeting_id,
        "wer": None,
        "cpwer": None,
        "der": None,
        "der_false_alarm_sec": None,
        "der_miss_sec": None,
        "der_confusion_sec": None,
        "der_scored_ref_time_sec": None,
        "der_ignored_ref_overlap_time_sec": None,
        "ref_speaker_count": None,
        "hyp_asr_speaker_count": None,
        "hyp_diar_speaker_count": None,
        "hyp_asr_segment_count": None,
        "hyp_diar_segment_count": None,
        "der_method": "approx_interval_single_label_nooverlap" if skip_overlap else "approx_interval_single_label_overlap_included",
        "der_collar_sec": collar_sec,
        "status": "ok",
        "errors": "",
    }

    try:
        ref_asr = load_reference_asr_views(annotations_dir, meeting_id)
    except Exception as exc:
        ref_asr = {"full_text": "", "speaker_texts": {}, "speaker_count": 0}
        errors.append(f"reference_asr:{type(exc).__name__}")

    try:
        hyp_asr = load_hyp_asr_views(artifacts_dir, meeting_id)
    except Exception as exc:
        hyp_asr = {"segments": [], "full_text": "", "speaker_texts": {}, "speaker_count": 0, "exists": False}
        errors.append(f"hyp_asr:{type(exc).__name__}")

    try:
        ref_diar = load_ref_diarization_from_words(annotations_dir, meeting_id)
    except Exception as exc:
        ref_diar = []
        errors.append(f"reference_diar:{type(exc).__name__}")

    try:
        hyp_diar = load_hyp_diarization(artifacts_dir, meeting_id)
    except Exception as exc:
        hyp_diar = []
        errors.append(f"hyp_diar:{type(exc).__name__}")

    row["ref_speaker_count"] = len(ref_asr.get("speaker_texts", {}))
    row["hyp_asr_speaker_count"] = len(hyp_asr.get("speaker_texts", {}))
    row["hyp_diar_speaker_count"] = len({str(s["speaker"]) for s in hyp_diar})
    row["hyp_asr_segment_count"] = len(hyp_asr.get("segments", []))
    row["hyp_diar_segment_count"] = len(hyp_diar)

    try:
        row["wer"] = wer_from_texts(ref_asr.get("full_text", ""), hyp_asr.get("full_text", ""))
    except Exception as exc:
        errors.append(f"wer:{type(exc).__name__}")

    try:
        cp = compute_cpwer(ref_asr.get("speaker_texts", {}), hyp_asr.get("speaker_texts", {}))
        row["cpwer"] = cp.get("cpwer")
    except Exception as exc:
        errors.append(f"cpwer:{type(exc).__name__}")

    try:
        der = compute_der_approx_nooverlap(ref_diar, hyp_diar, collar_sec=collar_sec, skip_overlap=skip_overlap)
        row["der"] = der.get("der")
        row["der_false_alarm_sec"] = der.get("false_alarm_sec")
        row["der_miss_sec"] = der.get("miss_sec")
        row["der_confusion_sec"] = der.get("confusion_sec")
        row["der_scored_ref_time_sec"] = der.get("scored_ref_time_sec")
        row["der_ignored_ref_overlap_time_sec"] = der.get("ignored_ref_overlap_time_sec")
        row["der_method"] = str(der.get("method", row["der_method"]))
    except Exception as exc:
        errors.append(f"der:{type(exc).__name__}")

    if errors:
        row["status"] = "partial_error"
        row["errors"] = ";".join(errors)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-meeting speech metrics using the repository CSV schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "meeting_id",
        "status",
        "wer",
        "cpwer",
        "der",
        "der_false_alarm_sec",
        "der_miss_sec",
        "der_confusion_sec",
        "der_scored_ref_time_sec",
        "der_ignored_ref_overlap_time_sec",
        "der_method",
        "der_collar_sec",
        "ref_speaker_count",
        "hyp_asr_speaker_count",
        "hyp_diar_speaker_count",
        "hyp_asr_segment_count",
        "hyp_diar_segment_count",
        "errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["errors"] = str(out.get("errors", ""))
            writer.writerow({k: out.get(k, "") for k in fieldnames})


def summarize(
    rows: list[dict[str, Any]],
    cfg_path: Path,
    out_csv: Path,
    collar_sec: float,
    skip_overlap: bool,
) -> dict[str, Any]:
    """Build the aggregate JSON summary accompanying the CSV report."""
    wers = [float(r["wer"]) for r in rows if r.get("wer") is not None]
    cpwers = [float(r["cpwer"]) for r in rows if r.get("cpwer") is not None]
    ders = [float(r["der"]) for r in rows if r.get("der") is not None]

    return {
        "config_path": str(cfg_path),
        "output_csv": str(out_csv),
        "der_settings": {
            "collar_sec": collar_sec,
            "skip_reference_overlap": skip_overlap,
            "method": "approx_interval_single_label_nooverlap",
        },
        "summary": {
            "meetings_checked": len(rows),
            "rows_with_errors": sum(1 for r in rows if r.get("errors")),
            "mean_wer": mean(wers) if wers else None,
            "mean_cpwer": mean(cpwers) if cpwers else None,
            "mean_der": mean(ders) if ders else None,
            "count_wer": len(wers),
            "count_cpwer": len(cpwers),
            "count_der": len(ders),
        },
        "rows": rows,
    }


def fmt(x: float | None) -> str:
    """Format a floating-point metric for console output."""
    return "n/a" if x is None else f"{x:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
