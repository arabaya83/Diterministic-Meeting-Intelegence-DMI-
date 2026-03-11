#!/usr/bin/env python3
"""Generate DVC stage templates that mirror current pipeline entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ami_mom_pipeline.config import AppConfig  # noqa: E402
from ami_mom_pipeline.pipeline import list_meetings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for DVC template generation."""
    p = argparse.ArgumentParser(description="Generate offline DVC stage templates for AMI pipeline runs")
    p.add_argument("--config", default="configs/pipeline.nemo.llama.yaml")
    p.add_argument("--meeting-id", dest="meeting_ids", action="append", default=None, help="Explicit meeting ID (repeatable)")
    p.add_argument("--prefix", default=None, help="Meeting ID prefix filter (e.g. ES2005)")
    p.add_argument("--limit", type=int, default=None, help="Limit selected meetings")
    p.add_argument("--mode", choices=["single", "batch"], default="single", help="Generate per-meeting stages or one batch stage")
    p.add_argument("--output", default=None, help="Output YAML path (default: artifacts/governance/dvc_stage_templates/<auto>.yaml)")
    p.add_argument("--stage-prefix", default="ami", help="Prefix for generated DVC stage names")
    return p


def main() -> int:
    """Generate the requested DVC stage template and print its metadata."""
    args = build_parser().parse_args()
    cfg = AppConfig.load(args.config if Path(args.config).exists() else None)
    meetings = _select_meetings(cfg, args.meeting_ids, args.prefix, args.limit)
    if not meetings:
        print("No meetings selected.", file=sys.stderr)
        return 1

    doc = {"stages": {}}
    if args.mode == "single":
        for m in meetings:
            stage_name = f"{args.stage_prefix}_{m}"
            doc["stages"][stage_name] = _single_stage(args.config, m)
    else:
        stage_name = f"{args.stage_prefix}_batch_{meetings[0]}_{meetings[-1]}_{len(meetings)}"
        doc["stages"][stage_name] = _batch_stage(args.config, meetings)

    out_path = _resolve_output_path(args.output, args.mode, meetings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "mode": args.mode, "meeting_count": len(meetings)}, indent=2))
    return 0


def _select_meetings(cfg: AppConfig, explicit: list[str] | None, prefix: str | None, limit: int | None) -> list[str]:
    """Select meetings from explicit ids or repository discovery."""
    if explicit:
        meetings = list(dict.fromkeys(explicit))
    else:
        meetings = list_meetings(cfg)
    if prefix:
        meetings = [m for m in meetings if m.startswith(prefix)]
    if limit is not None:
        meetings = meetings[:limit]
    return meetings


def _single_stage(config_path: str, meeting_id: str) -> dict:
    """Return the DVC stage definition for a single-meeting run."""
    return {
        "cmd": f"PYTHONPATH=src python3 -m ami_mom_pipeline --config {config_path} run --meeting-id {meeting_id}",
        "deps": [
            "src/",
            "scripts/",
            config_path,
            f"data/rawa/ami/audio/{meeting_id}.Mix-Headset.wav",
            "data/rawa/ami/annotations/",
            "models/",
        ],
        "outs": [
            f"artifacts/ami/{meeting_id}",
            "artifacts/eval/ami",
        ],
    }


def _batch_stage(config_path: str, meetings: list[str]) -> dict:
    """Return the DVC stage definition for a batch-runner invocation."""
    meeting_flags = " ".join(f"--meeting-id {m}" for m in meetings)
    label = f"dvc_batch_{meetings[0]}_{meetings[-1]}_{len(meetings)}"
    return {
        "cmd": (
            "python3 scripts/run_nemo_batch_sequential.py "
            f"--config {config_path} {meeting_flags} --run-label {label}"
        ),
        "deps": [
            "src/",
            "scripts/",
            config_path,
            "data/rawa/ami/annotations/",
            "models/",
            *[f"data/rawa/ami/audio/{m}.Mix-Headset.wav" for m in meetings],
        ],
        "outs": [*[f"artifacts/ami/{m}" for m in meetings], "artifacts/eval/ami", "artifacts/batch_runs"],
    }


def _resolve_output_path(output: str | None, mode: str, meetings: list[str]) -> Path:
    """Resolve the output YAML path, applying the default naming scheme."""
    if output:
        p = Path(output).expanduser()
        return p if p.is_absolute() else ROOT / p
    suffix = f"{mode}_{meetings[0]}_{meetings[-1]}_{len(meetings)}.yaml"
    return ROOT / "artifacts" / "governance" / "dvc_stage_templates" / suffix


if __name__ == "__main__":
    raise SystemExit(main())
