from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig
from .pipeline import list_meetings, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ami-mom", description="Offline-first AMI meeting understanding pipeline")
    p.add_argument("--config", default="configs/pipeline.sample.yaml", help="Path to YAML config")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list-meetings", help="List available AMI meetings from raw audio dir")
    ls.add_argument("--limit", type=int, default=20)

    run = sub.add_parser("run", help="Run pipeline for one meeting")
    run.add_argument("--meeting-id", required=True)

    run_many = sub.add_parser("run-many", help="Run pipeline for multiple meetings")
    run_many.add_argument("--limit", type=int, default=5)
    run_many.add_argument("--prefix", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg_path = args.config if Path(args.config).exists() else None
    cfg = AppConfig.load(cfg_path)

    if args.cmd == "list-meetings":
        meetings = list_meetings(cfg)
        for m in meetings[: args.limit]:
            print(m)
        return 0

    if args.cmd == "run":
        manifest = run_pipeline(cfg, args.meeting_id)
        print(json.dumps({"meeting_id": args.meeting_id, "artifact_digest": manifest["artifact_digest"]}, indent=2))
        return 0

    if args.cmd == "run-many":
        meetings = list_meetings(cfg)
        if args.prefix:
            meetings = [m for m in meetings if m.startswith(args.prefix)]
        for m in meetings[: args.limit]:
            manifest = run_pipeline(cfg, m)
            print(json.dumps({"meeting_id": m, "artifact_digest": manifest["artifact_digest"]}))
        return 0

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
