#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ami_mom_pipeline.config import AppConfig  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Validate local NeMo model/config paths for offline pipeline runs")
    p.add_argument("--config", default="configs/pipeline.nemo.yaml")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = AppConfig.load(str(cfg_path))
    nemo = cfg.pipeline.speech_backend.nemo
    checks = [
        ("vad_model_path", nemo.vad_model_path),
        ("diarizer_config_path", nemo.diarizer_config_path),
        ("asr_model_path", nemo.asr_model_path),
    ]

    print("NeMo model/config path check")
    print(f"config: {cfg_path}")
    print(f"speech_backend.mode: {cfg.pipeline.speech_backend.mode}")

    if cfg.pipeline.speech_backend.mode != "nemo":
        print("WARNING: speech_backend.mode is not 'nemo'")

    missing = []
    for name, value in checks:
        if not value:
            print(f"[MISSING] {name}: not set")
            missing.append(name)
            continue
        if "://" in value:
            print(f"[INVALID] {name}: URL/path not allowed for offline runtime -> {value}")
            missing.append(name)
            continue
        path = Path(value).expanduser()
        status = "OK" if path.exists() else "MISSING"
        kind = "dir" if path.is_dir() else "file" if path.is_file() else "path"
        print(f"[{status}] {name}: {path} ({kind})")
        if not path.exists():
            missing.append(name)

    expected_dirs = [
        ROOT / "models" / "nemo" / "vad",
        ROOT / "models" / "nemo" / "diarizer",
        ROOT / "models" / "nemo" / "asr",
    ]
    print("\nDirectory scaffold:")
    for d in expected_dirs:
        print(f"[{'OK' if d.exists() else 'MISSING'}] {d}")

    if missing:
        print("\nResult: NOT READY (missing/invalid required NeMo paths).", file=sys.stderr)
        return 1

    print("\nResult: READY (required NeMo paths exist).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

