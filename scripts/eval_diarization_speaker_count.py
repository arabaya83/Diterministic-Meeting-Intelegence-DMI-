#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
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
    p = argparse.ArgumentParser(
        description="Evaluate NeMo diarization speaker-count quality against AMI reference participant count"
    )
    p.add_argument("--config", default="configs/pipeline.nemo.yaml", help="Path to pipeline YAML config")
    p.add_argument("--prefix", default=None, help="Meeting ID prefix filter")
    p.add_argument("--limit", type=int, default=None, help="Max meetings after filtering")
    p.add_argument(
        "--meeting-id",
        dest="meeting_ids",
        action="append",
        default=None,
        help="Explicit meeting ID (repeatable).",
    )
    p.add_argument(
        "--discover",
        choices=("artifacts", "raw"),
        default="artifacts",
        help="Meeting discovery source when --meeting-id is not provided",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: artifacts/eval/ami/diarization_speaker_count.csv)",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Output JSON summary path (default: artifacts/eval/ami/diarization_speaker_count_summary.json)",
    )
    return p


def main() -> int:
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

    reference_index = load_ami_meeting_reference(Path(cfg.paths.annotations_dir))
    rows: list[dict[str, Any]] = []
    for meeting_id in meeting_ids:
        rows.append(evaluate_meeting(cfg, meeting_id, reference_index))

    out_csv = Path(args.out_csv) if args.out_csv else (Path(cfg.paths.artifacts_dir) / "eval" / "ami" / "diarization_speaker_count.csv")
    out_json = Path(args.out_json) if args.out_json else (Path(cfg.paths.artifacts_dir) / "eval" / "ami" / "diarization_speaker_count_summary.json")
    write_csv(out_csv, rows)

    summary = summarize(rows, cfg_path, out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    s = summary["summary"]
    print(
        "Diarization speaker-count eval complete: "
        f"checked={s['meetings_checked']} comparable={s['comparable_meetings']} "
        f"exact={s['exact_match']} over={s['over_clustered']} under={s['under_clustered']} "
        f"missing_pred={s['missing_prediction']} missing_ref={s['missing_reference']}"
    )
    if s["comparable_meetings"] > 0:
        print(
            f"Mean abs error={s['mean_abs_error']:.3f}, "
            f"accuracy(exact count)={s['exact_match_accuracy']:.3f}"
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
        artifacts_ami = Path(cfg.paths.artifacts_dir) / "ami"
        meetings = []
        if artifacts_ami.exists():
            for p in sorted(artifacts_ami.iterdir()):
                if p.is_dir() and (p / "diarization_segments.json").exists():
                    meetings.append(p.name)
    if prefix:
        meetings = [m for m in meetings if m.startswith(prefix)]
    if limit is not None:
        meetings = meetings[:limit]
    return meetings


def localname(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def load_ami_meeting_reference(annotations_dir: Path) -> dict[str, dict[str, Any]]:
    meetings_xml = annotations_dir / "corpusResources" / "meetings.xml"
    index: dict[str, dict[str, Any]] = {}
    if not meetings_xml.exists():
        return index

    root = ET.parse(meetings_xml).getroot()
    for meeting_elem in root:
        if localname(meeting_elem.tag) != "meeting":
            continue
        observation = meeting_elem.attrib.get("observation")
        if not observation:
            continue
        speakers = []
        for child in meeting_elem:
            if localname(child.tag) != "speaker":
                continue
            speakers.append(
                {
                    "nxt_agent": child.attrib.get("nxt_agent"),
                    "role": child.attrib.get("role"),
                    "global_name": child.attrib.get("global_name"),
                    "channel": child.attrib.get("channel"),
                }
            )
        # Deduplicate in case of malformed duplicates while preserving stable ordering.
        unique_agents: list[str] = []
        unique_roles: list[str] = []
        for sp in speakers:
            agent = sp.get("nxt_agent")
            if agent and agent not in unique_agents:
                unique_agents.append(agent)
            role = sp.get("role")
            if role and role not in unique_roles:
                unique_roles.append(role)
        index[observation] = {
            "participant_count": len(speakers),
            "agents": unique_agents,
            "roles": unique_roles,
        }
    return index


def evaluate_meeting(cfg: AppConfig, meeting_id: str, reference_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    artifact_dir = Path(cfg.paths.artifacts_dir) / "ami" / meeting_id
    diar_json = artifact_dir / "diarization_segments.json"
    pred_count: int | None = None
    pred_speakers: list[str] = []
    pred_segment_count = 0
    pred_total_speech_sec: float | None = None
    pred_error: str | None = None

    if diar_json.exists():
        try:
            data = json.loads(diar_json.read_text(encoding="utf-8"))
            if isinstance(data, list):
                pred_segment_count = len(data)
                speaker_set = set()
                total = 0.0
                for seg in data:
                    if not isinstance(seg, dict):
                        continue
                    speaker = seg.get("speaker")
                    if speaker:
                        speaker_set.add(str(speaker))
                    try:
                        start = float(seg.get("start", 0.0))
                        end = float(seg.get("end", 0.0))
                        if end >= start:
                            total += end - start
                    except (TypeError, ValueError):
                        pass
                pred_speakers = sorted(speaker_set)
                pred_count = len(pred_speakers)
                pred_total_speech_sec = round(total, 3)
            else:
                pred_error = "diarization_segments_not_list"
        except Exception as exc:
            pred_error = f"{type(exc).__name__}: {exc}"
    else:
        pred_error = "missing_diarization_segments_json"

    ref = reference_index.get(meeting_id)
    ref_count = ref["participant_count"] if ref else None
    ref_agents = ref["agents"] if ref else []
    ref_roles = ref["roles"] if ref else []

    delta: int | None = None
    abs_delta: int | None = None
    status = "missing_both"
    if pred_count is not None and ref_count is not None:
        delta = pred_count - ref_count
        abs_delta = abs(delta)
        if delta == 0:
            status = "exact"
        elif delta > 0:
            status = "over_clustered"
        else:
            status = "under_clustered"
    elif pred_count is None and ref_count is not None:
        status = "missing_prediction"
    elif pred_count is not None and ref_count is None:
        status = "missing_reference"

    return {
        "meeting_id": meeting_id,
        "status": status,
        "predicted_speaker_count": pred_count,
        "reference_participant_count": ref_count,
        "delta_pred_minus_ref": delta,
        "abs_delta": abs_delta,
        "predicted_segment_count": pred_segment_count,
        "predicted_total_speech_sec": pred_total_speech_sec,
        "predicted_speakers": ";".join(pred_speakers),
        "reference_agents": ";".join(ref_agents),
        "reference_roles": ";".join(ref_roles),
        "artifact_dir": str(artifact_dir),
        "diarization_segments_json_exists": diar_json.exists(),
        "prediction_error": pred_error or "",
        "reference_source": "corpusResources/meetings.xml" if ref else "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "meeting_id",
        "status",
        "predicted_speaker_count",
        "reference_participant_count",
        "delta_pred_minus_ref",
        "abs_delta",
        "predicted_segment_count",
        "predicted_total_speech_sec",
        "predicted_speakers",
        "reference_agents",
        "reference_roles",
        "diarization_segments_json_exists",
        "prediction_error",
        "reference_source",
        "artifact_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def summarize(rows: list[dict[str, Any]], cfg_path: Path, out_csv: Path) -> dict[str, Any]:
    exact = sum(1 for r in rows if r["status"] == "exact")
    over = sum(1 for r in rows if r["status"] == "over_clustered")
    under = sum(1 for r in rows if r["status"] == "under_clustered")
    missing_pred = sum(1 for r in rows if r["status"] == "missing_prediction")
    missing_ref = sum(1 for r in rows if r["status"] == "missing_reference")
    missing_both = sum(1 for r in rows if r["status"] == "missing_both")
    comparable = [r for r in rows if r["delta_pred_minus_ref"] is not None]
    abs_errors = [int(r["abs_delta"]) for r in comparable]
    deltas = [int(r["delta_pred_minus_ref"]) for r in comparable]

    by_delta: dict[str, int] = {}
    for r in comparable:
        key = str(r["delta_pred_minus_ref"])
        by_delta[key] = by_delta.get(key, 0) + 1

    return {
        "config_path": str(cfg_path),
        "output_csv": str(out_csv),
        "summary": {
            "meetings_checked": len(rows),
            "comparable_meetings": len(comparable),
            "exact_match": exact,
            "over_clustered": over,
            "under_clustered": under,
            "missing_prediction": missing_pred,
            "missing_reference": missing_ref,
            "missing_both": missing_both,
            "exact_match_accuracy": (exact / len(comparable)) if comparable else 0.0,
            "mean_abs_error": mean(abs_errors) if abs_errors else 0.0,
            "mean_signed_error": mean(deltas) if deltas else 0.0,
        },
        "delta_histogram": dict(sorted(by_delta.items(), key=lambda kv: int(kv[0]))),
        "rows": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
