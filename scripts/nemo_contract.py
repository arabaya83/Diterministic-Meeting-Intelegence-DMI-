#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def parse_rttm(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            rows.append(
                {
                    "file_id": parts[1],
                    "start": float(parts[3]),
                    "dur": float(parts[4]),
                    "speaker": parts[7],
                }
            )
    return rows


def diarization_json_from_rttm(path: Path) -> list[dict[str, Any]]:
    segs = []
    for row in parse_rttm(path):
        segs.append(
            {
                "start": round(row["start"], 3),
                "end": round(row["start"] + row["dur"], 3),
                "speaker": row["speaker"],
                "source": "nemo_rttm",
            }
        )
    segs.sort(key=lambda s: (s["start"], s["end"], s["speaker"]))
    return segs


def vad_json_from_rttm(path: Path) -> list[dict[str, Any]]:
    diar = diarization_json_from_rttm(path)
    # Merge overlapping/adjacent diarization segments into speech regions.
    merged: list[dict[str, Any]] = []
    for s in diar:
        item = {"start": s["start"], "end": s["end"], "label": "speech", "source": "nemo_derived_from_diarization"}
        if not merged:
            merged.append(item)
            continue
        prev = merged[-1]
        if item["start"] <= prev["end"] + 0.05:
            prev["end"] = round(max(prev["end"], item["end"]), 3)
        else:
            merged.append(item)
    return merged


def write_vad_rttm(path: Path, meeting_id: str, segs: list[dict[str, Any]]) -> None:
    lines = [
        f"SPEAKER {meeting_id} 1 {s['start']:.3f} {(s['end']-s['start']):.3f} <NA> <NA> speech <NA> <NA>"
        for s in segs
    ]
    write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def write_diar_rttm(path: Path, meeting_id: str, segs: list[dict[str, Any]]) -> None:
    lines = [
        f"SPEAKER {meeting_id} 1 {s['start']:.3f} {(s['end']-s['start']):.3f} <NA> <NA> {s['speaker']} <NA> <NA>"
        for s in segs
    ]
    write_text(path, "\n".join(lines) + ("\n" if lines else ""))

