"""Smoke tests for the NeMo wrapper CLI entrypoints."""

import subprocess
import sys
from pathlib import Path


def test_nemo_wrapper_scripts_help():
    """Each NeMo wrapper should respond successfully to ``--help``."""
    root = Path(__file__).resolve().parents[1]
    for script in ["scripts/nemo_vad.py", "scripts/nemo_diarize.py", "scripts/nemo_asr.py"]:
        proc = subprocess.run([sys.executable, str(root / script), "--help"], capture_output=True, text=True)
        assert proc.returncode == 0, f"{script} failed: {proc.stderr}"
