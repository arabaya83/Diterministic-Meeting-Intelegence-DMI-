#!/usr/bin/env python3
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASR confidence QA report for AMI pipeline artifacts")
    p.add_argument("--config", default="configs/pipeline.nemo.yaml")
    p.add_argument("--meeting-id", dest="meeting_ids", action="append", default=None)
    p.add_argument("--prefix", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--discover", choices=("artifacts", "raw"), default="artifacts")
    p.add_argument(
        "--speech-metrics-csv",
        default=None,
        help="Optional speech metrics CSV (WER/cpWER/DER) to join into output rows",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Default: artifacts/eval/ami/asr_confidence_qa.csv",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Default: artifacts/eval/ami/asr_confidence_qa_summary.json",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = AppConfig.load(str(cfg_path))
    meetings = select_meetings(cfg, args.meeting_ids, args.prefix, args.limit, args.discover)
    if not meetings:
        print("No meetings selected.", file=sys.stderr)
        return 1

    speech_metrics_index = load_csv_index(Path(args.speech_metrics_csv), "meeting_id") if args.speech_metrics_csv else {}

    rows = [analyze_meeting(cfg, m, speech_metrics_index.get(m)) for m in meetings]

    out_csv = Path(args.out_csv) if args.out_csv else (Path(cfg.paths.artifacts_dir) / "eval" / "ami" / "asr_confidence_qa.csv")
    out_json = Path(args.out_json) if args.out_json else (Path(cfg.paths.artifacts_dir) / "eval" / "ami" / "asr_confidence_qa_summary.json")
    write_csv(out_csv, rows)
    summary = summarize(rows, cfg_path, out_csv, args.speech_metrics_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    s = summary["summary"]
    print(
        "ASR confidence QA complete: "
        f"checked={s['meetings_checked']} "
        f"with_conf={s['meetings_with_any_nonzero_conf']} "
        f"zero_only={s['meetings_zero_only_conf']} "
        f"mean_nonzero_ratio={s['mean_nonzero_ratio']:.3f}"
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
    if explicit:
        meetings = list(dict.fromkeys(explicit))
    elif discover == "raw":
        meetings = list_meetings(cfg)
    else:
        root = Path(cfg.paths.artifacts_dir) / "ami"
        meetings = []
        if root.exists():
            for p in sorted(root.iterdir()):
                if p.is_dir() and (p / "asr_segments.json").exists():
                    meetings.append(p.name)
    if prefix:
        meetings = [m for m in meetings if m.startswith(prefix)]
    if limit is not None:
        meetings = meetings[:limit]
    return meetings


def analyze_meeting(cfg: AppConfig, meeting_id: str, speech_metrics_row: dict[str, str] | None) -> dict[str, Any]:
    mdir = Path(cfg.paths.artifacts_dir) / "ami" / meeting_id
    asr_segments_path = mdir / "asr_segments.json"
    asr_conf_path = mdir / "asr_confidence.json"

    row: dict[str, Any] = {
        "meeting_id": meeting_id,
        "artifact_dir": str(mdir),
        "asr_segments_exists": asr_segments_path.exists(),
        "asr_confidence_exists": asr_conf_path.exists(),
        "segment_count": 0,
        "nonzero_confidence_count": 0,
        "nonzero_ratio": 0.0,
        "mean_confidence": 0.0,
        "min_confidence": 0.0,
        "max_confidence": 0.0,
        "p10_confidence": 0.0,
        "p50_confidence": 0.0,
        "p90_confidence": 0.0,
        "lt_0_01_count": 0,
        "lt_0_10_count": 0,
        "lt_0_30_count": 0,
        "lt_0_50_count": 0,
        "zero_confidence_count": 0,
        "confidence_source_status": "missing",
        "wer": None,
        "cpwer": None,
        "der": None,
        "cpwer_minus_wer": None,
        "notes": "",
    }

    conf_summary = {}
    if asr_conf_path.exists():
        try:
            conf_summary = json.loads(asr_conf_path.read_text(encoding="utf-8"))
        except Exception as exc:
            row["notes"] = f"asr_confidence_read_error:{type(exc).__name__}"

    confidences: list[float] = []
    if asr_segments_path.exists():
        try:
            segs = json.loads(asr_segments_path.read_text(encoding="utf-8"))
            if isinstance(segs, list):
                for s in segs:
                    if not isinstance(s, dict):
                        continue
                    try:
                        c = float(s.get("confidence", 0.0) or 0.0)
                    except Exception:
                        c = 0.0
                    if c < 0:
                        c = 0.0
                    if c > 1:
                        c = 1.0
                    confidences.append(c)
        except Exception as exc:
            row["notes"] = (row["notes"] + ";" if row["notes"] else "") + f"asr_segments_read_error:{type(exc).__name__}"

    if confidences:
        vals = sorted(confidences)
        n = len(vals)
        nonzero = sum(1 for v in vals if v > 0.0)
        row.update(
            {
                "segment_count": n,
                "nonzero_confidence_count": nonzero,
                "nonzero_ratio": round(nonzero / n, 4),
                "mean_confidence": round(mean(vals), 4),
                "min_confidence": round(vals[0], 4),
                "max_confidence": round(vals[-1], 4),
                "p10_confidence": round(percentile(vals, 0.10), 4),
                "p50_confidence": round(percentile(vals, 0.50), 4),
                "p90_confidence": round(percentile(vals, 0.90), 4),
                "lt_0_01_count": sum(1 for v in vals if v < 0.01),
                "lt_0_10_count": sum(1 for v in vals if v < 0.10),
                "lt_0_30_count": sum(1 for v in vals if v < 0.30),
                "lt_0_50_count": sum(1 for v in vals if v < 0.50),
                "zero_confidence_count": sum(1 for v in vals if v == 0.0),
            }
        )
        row["confidence_source_status"] = "nonzero_present" if nonzero > 0 else "all_zero"

    # Cross-check summary file if present.
    if conf_summary:
        if int(conf_summary.get("segment_count", row["segment_count"]) or 0) != row["segment_count"]:
            row["notes"] = (row["notes"] + ";" if row["notes"] else "") + "segment_count_mismatch_vs_asr_confidence_json"
        if "nonzero_confidence_count" in conf_summary and int(conf_summary.get("nonzero_confidence_count", 0)) != row["nonzero_confidence_count"]:
            row["notes"] = (row["notes"] + ";" if row["notes"] else "") + "nonzero_count_mismatch_vs_asr_confidence_json"

    if speech_metrics_row:
        row["wer"] = as_float_or_none(speech_metrics_row.get("wer"))
        row["cpwer"] = as_float_or_none(speech_metrics_row.get("cpwer"))
        row["der"] = as_float_or_none(speech_metrics_row.get("der"))
        if row["wer"] is not None and row["cpwer"] is not None:
            row["cpwer_minus_wer"] = round(float(row["cpwer"]) - float(row["wer"]), 6)

    return row


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = (len(sorted_vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def as_float_or_none(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_csv_index(path: Path | None, key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if path is None or not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            k = row.get(key)
            if k:
                out[k] = row
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "meeting_id",
        "confidence_source_status",
        "segment_count",
        "nonzero_confidence_count",
        "nonzero_ratio",
        "mean_confidence",
        "min_confidence",
        "max_confidence",
        "p10_confidence",
        "p50_confidence",
        "p90_confidence",
        "zero_confidence_count",
        "lt_0_01_count",
        "lt_0_10_count",
        "lt_0_30_count",
        "lt_0_50_count",
        "wer",
        "cpwer",
        "der",
        "cpwer_minus_wer",
        "asr_segments_exists",
        "asr_confidence_exists",
        "notes",
        "artifact_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def summarize(rows: list[dict[str, Any]], cfg_path: Path, out_csv: Path, speech_metrics_csv: str | None) -> dict[str, Any]:
    nz_ratios = [float(r["nonzero_ratio"]) for r in rows if r["segment_count"] > 0]
    means = [float(r["mean_confidence"]) for r in rows if r["segment_count"] > 0]
    wers = [float(r["wer"]) for r in rows if r.get("wer") is not None]
    cpwer_gaps = [float(r["cpwer_minus_wer"]) for r in rows if r.get("cpwer_minus_wer") is not None]
    return {
        "config_path": str(cfg_path),
        "speech_metrics_csv": speech_metrics_csv,
        "output_csv": str(out_csv),
        "summary": {
            "meetings_checked": len(rows),
            "meetings_with_any_nonzero_conf": sum(1 for r in rows if r["nonzero_confidence_count"] > 0),
            "meetings_zero_only_conf": sum(1 for r in rows if r["segment_count"] > 0 and r["nonzero_confidence_count"] == 0),
            "mean_nonzero_ratio": mean(nz_ratios) if nz_ratios else 0.0,
            "mean_confidence_across_meetings": mean(means) if means else 0.0,
            "mean_wer": mean(wers) if wers else None,
            "mean_cpwer_minus_wer": mean(cpwer_gaps) if cpwer_gaps else None,
        },
        "rows": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
