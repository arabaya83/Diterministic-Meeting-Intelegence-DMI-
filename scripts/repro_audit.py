#!/usr/bin/env python3
"""Compare reproducibility digests across artifact roots.

The audit reads manifests and reproducibility reports from the current artifact
tree plus optional snapshot roots, then reports whether digests agree for each
selected meeting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ami_mom_pipeline.config import AppConfig  # noqa: E402
from ami_mom_pipeline.pipeline import list_meetings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for reproducibility audits."""
    p = argparse.ArgumentParser(description="Reproducibility audit for AMI pipeline manifests/reports")
    p.add_argument("--config", default="configs/pipeline.nemo.llama.yaml")
    p.add_argument("--meeting-id", dest="meeting_ids", action="append", default=None, help="Explicit meeting ID (repeatable)")
    p.add_argument("--prefix", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--snapshot-dir",
        action="append",
        default=None,
        help="Additional artifacts root(s) to compare against current artifacts (repeatable). Expected layout: <root>/ami/<meeting_id>/run_manifest.json",
    )
    p.add_argument("--out-json", default="artifacts/governance/repro_audit.json")
    return p


def main() -> int:
    """Run the reproducibility audit and write the JSON report."""
    args = build_parser().parse_args()
    cfg = AppConfig.load(args.config if Path(args.config).exists() else None)
    meetings = _select_meetings(cfg, args.meeting_ids, args.prefix, args.limit)
    if not meetings:
        print("No meetings selected.", file=sys.stderr)
        return 1
    roots = [Path(cfg.paths.artifacts_dir)]
    if args.snapshot_dir:
        roots.extend(Path(p).expanduser() for p in args.snapshot_dir)

    rows = []
    for meeting_id in meetings:
        snapshots = [_read_snapshot(root, meeting_id) for root in roots]
        rows.append(_compare_snapshots(meeting_id, snapshots))

    summary = _summarize(rows)
    report = {"roots": [str(r) for r in roots], "rows": rows, "summary": summary}
    out = Path(args.out_json).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": str(out), "summary": summary}, indent=2))
    return 0 if summary["mismatched_meetings"] == 0 else 1


def _select_meetings(cfg: AppConfig, explicit: list[str] | None, prefix: str | None, limit: int | None) -> list[str]:
    """Select meetings from explicit ids or deterministic discovery."""
    if explicit:
        meetings = list(dict.fromkeys(explicit))
    else:
        meetings = list_meetings(cfg)
    if prefix:
        meetings = [m for m in meetings if m.startswith(prefix)]
    if limit is not None:
        meetings = meetings[:limit]
    return meetings


def _read_snapshot(root: Path, meeting_id: str) -> dict[str, Any]:
    """Read one meeting snapshot from a given artifact root."""
    manifest_path = root / "ami" / meeting_id / "run_manifest.json"
    repro_path = root / "ami" / meeting_id / "reproducibility_report.json"
    rec: dict[str, Any] = {"root": str(root), "meeting_id": meeting_id}
    if not manifest_path.exists():
        rec["missing_manifest"] = True
        return rec
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        rec["manifest_read_error"] = f"{type(exc).__name__}: {exc}"
        return rec
    rec["artifact_digest"] = manifest.get("artifact_digest")
    rec["config_digest"] = manifest.get("config_digest")
    rec["speech_backend"] = manifest.get("speech_backend")
    rec["summarization_backend"] = manifest.get("summarization_backend")
    rec["extraction_backend"] = manifest.get("extraction_backend")
    if repro_path.exists():
        try:
            repro = json.loads(repro_path.read_text(encoding="utf-8"))
            rec["repro_config_digest"] = repro.get("config_digest")
        except Exception as exc:
            rec["repro_read_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def _compare_snapshots(meeting_id: str, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare digests across all available snapshots for one meeting."""
    present = [s for s in snapshots if not s.get("missing_manifest") and not s.get("manifest_read_error")]
    artifact_digests = sorted({str(s.get("artifact_digest")) for s in present if s.get("artifact_digest") is not None})
    config_digests = sorted({str(s.get("config_digest")) for s in present if s.get("config_digest") is not None})
    return {
        "meeting_id": meeting_id,
        "snapshots": snapshots,
        "present_count": len(present),
        "artifact_digest_count": len(artifact_digests),
        "config_digest_count": len(config_digests),
        "artifact_digest_match": len(artifact_digests) <= 1,
        "config_digest_match": len(config_digests) <= 1,
        "artifact_digests": artifact_digests,
        "config_digests": config_digests,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize mismatch counts across all audited meetings."""
    mismatched = 0
    artifact_mismatch = 0
    config_mismatch = 0
    for r in rows:
        row_bad = False
        if not r.get("artifact_digest_match", True):
            artifact_mismatch += 1
            row_bad = True
        if not r.get("config_digest_match", True):
            config_mismatch += 1
            row_bad = True
        if row_bad:
            mismatched += 1
    return {
        "meetings_checked": len(rows),
        "mismatched_meetings": mismatched,
        "artifact_digest_mismatches": artifact_mismatch,
        "config_digest_mismatches": config_mismatch,
    }


if __name__ == "__main__":
    raise SystemExit(main())
