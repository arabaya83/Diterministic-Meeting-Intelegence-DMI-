#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MEETING_FILES = [
    "run_manifest.json",
    "stage_trace.jsonl",
    "preflight_offline_audit.json",
    "reproducibility_report.json",
    "mom_summary.json",
    "decisions_actions.json",
    "extraction_validation_report.json",
    "asr_confidence.json",
]

DOC_FILES = [
    "README.md",
    "docs/ACCEPTANCE_CHECKLIST_SECTION16.md",
    "docs/PLAN_ALIGNMENT.md",
    "docs/REPRODUCIBILITY_OBSERVABILITY.md",
    "docs/GOVERNANCE_OFFLINE.md",
    "docs/OFFLINE_SETUP.md",
    "docs/HANDOFF_NEXT_SESSION.md",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate an acceptance evidence bundle for AMI pipeline runs")
    p.add_argument("--meeting-id", action="append", dest="meeting_ids", default=None, help="Meeting ID (repeatable)")
    p.add_argument("--bundle-name", default=None, help="Optional fixed bundle folder name")
    p.add_argument("--include-batch-runs", action="store_true", help="Copy recent batch run summaries")
    p.add_argument("--max-batch-files", type=int, default=6, help="Max recent batch summary/speech summary files to include")
    p.add_argument("--out-dir", default="artifacts/governance/evidence_bundle", help="Base output directory")
    return p


def main() -> int:
    args = build_parser().parse_args()
    meeting_ids = list(dict.fromkeys(args.meeting_ids or ["ES2005a"]))

    base_out = _resolve(ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    stamp = args.bundle_name or datetime.now(timezone.utc).strftime("bundle_%Y%m%dT%H%M%SZ")
    bundle_dir = base_out / stamp
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []

    for rel in DOC_FILES:
        _copy_rel(rel, bundle_dir / rel, copied, missing)

    for meeting_id in meeting_ids:
        src_dir = ROOT / "artifacts" / "ami" / meeting_id
        dst_dir = bundle_dir / "artifacts" / "ami" / meeting_id
        for name in MEETING_FILES:
            src = src_dir / name
            dst = dst_dir / name
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(str(dst.relative_to(bundle_dir)))
            else:
                missing.append(str((Path("artifacts") / "ami" / meeting_id / name).as_posix()))

    # Include latest evaluation artifacts if present.
    eval_dir = ROOT / "artifacts" / "eval" / "ami"
    for rel in [
        "artifacts/eval/ami/wer_scores.csv",
        "artifacts/eval/ami/wer_breakdown.json",
        "artifacts/eval/ami/rouge_scores.csv",
        "artifacts/eval/ami/mom_quality_checks.json",
    ]:
        _copy_rel(rel, bundle_dir / rel, copied, missing)

    if args.include_batch_runs:
        batch_dir = ROOT / "artifacts" / "batch_runs"
        patterns = ["*.summary.json", "*.speech_metrics.summary.json", "*.validation.json", "*.timings.csv"]
        recent = []
        for pat in patterns:
            recent.extend(batch_dir.glob(pat))
        recent = sorted(set(recent), key=lambda p: p.stat().st_mtime, reverse=True)[: args.max_batch_files]
        for src in recent:
            rel = Path("artifacts") / "batch_runs" / src.name
            dst = bundle_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str(rel.as_posix()))

    manifest = {
        "bundle_dir": str(bundle_dir),
        "meeting_ids": meeting_ids,
        "copied_count": len(copied),
        "missing_count": len(missing),
        "copied": sorted(copied),
        "missing": sorted(missing),
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"bundle_dir": str(bundle_dir), "copied_count": len(copied), "missing_count": len(missing)}, indent=2))
    return 0


def _resolve(p: Path) -> Path:
    return p.expanduser().resolve()


def _copy_rel(rel: str, dst: Path, copied: list[str], missing: list[str]) -> None:
    src = ROOT / rel
    if not src.exists():
        missing.append(rel)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(Path(rel).as_posix()))


if __name__ == "__main__":
    raise SystemExit(main())
