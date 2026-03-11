#!/usr/bin/env python3
"""Normalize VAD outputs into the repository's offline artifact contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nemo_contract import ensure_dir, vad_json_from_rttm, write_json, write_vad_rttm


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the VAD wrapper entrypoint."""
    p = argparse.ArgumentParser(description="NeMo VAD wrapper for AMI pipeline artifact contract")
    p.add_argument("--audio", required=True)
    p.add_argument("--model", required=False, help="Local NeMo VAD model path (.nemo or directory)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--input-vad-json", help="Reuse/normalize an existing VAD JSON artifact")
    p.add_argument("--input-vad-rttm", help="Reuse/normalize an existing VAD RTTM artifact")
    p.add_argument(
        "--delegate-cmd",
        help="Optional local command that performs NeMo VAD and writes files into --out-dir; wrappers will normalize outputs afterward.",
    )
    p.add_argument(
        "--derive-from-diarization-rttm",
        action="store_true",
        help="If VAD output is unavailable, derive speech regions from diarization.rttm in --out-dir.",
    )
    return p.parse_args()


def main() -> int:
    """Materialize VAD artifacts from one supported offline source."""
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    vad_json = out_dir / "vad_segments.json"
    vad_rttm = out_dir / "vad_segments.rttm"

    if args.input_vad_json:
        segs = json.loads(Path(args.input_vad_json).read_text(encoding="utf-8"))
        write_json(vad_json, segs)
        if not vad_rttm.exists():
            write_vad_rttm(vad_rttm, args.meeting_id, segs)
        return 0

    if args.input_vad_rttm:
        src = Path(args.input_vad_rttm)
        segs = vad_json_from_rttm(src)
        write_json(vad_json, segs)
        write_vad_rttm(vad_rttm, args.meeting_id, segs)
        return 0

    if args.delegate_cmd:
        # External execution is delegated; this wrapper only verifies outputs.
        proc = subprocess.run(args.delegate_cmd, shell=True)
        if proc.returncode != 0:
            return proc.returncode
        if vad_json.exists() and vad_rttm.exists():
            return 0

    if args.derive_from_diarization_rttm:
        diar_rttm = out_dir / "diarization.rttm"
        if diar_rttm.exists():
            segs = vad_json_from_rttm(diar_rttm)
            write_json(vad_json, segs)
            write_vad_rttm(vad_rttm, args.meeting_id, segs)
            return 0

    print(
        "No VAD outputs available. Provide --input-vad-json/--input-vad-rttm, "
        "or use --delegate-cmd with a pinned NeMo VAD runner that writes vad_segments.json/vad_segments.rttm.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
