"""Regression tests for NeMo ASR confidence extraction helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("nemo_asr_script", ROOT / "scripts" / "nemo_asr.py")
assert SPEC and SPEC.loader
nemo_asr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nemo_asr)


def test_extract_confidence_prefers_word_confidence_lists() -> None:
    """Word-confidence lists should drive the extracted confidence score."""
    item = {
        "text": "hello world",
        "word_confidence": [0.75, 0.85],
    }
    assert nemo_asr._extract_confidence(item) == 0.8


def test_extract_confidence_flattens_nested_frame_confidence() -> None:
    """Nested frame-confidence lists should be flattened before averaging."""
    item = {
        "text": "hello world",
        "frame_confidence": [[0.2, 0.4], [0.6, 0.8]],
    }
    assert nemo_asr._extract_confidence(item) == 0.5


def test_call_transcribe_compat_requests_hypotheses_when_supported() -> None:
    """Compatibility calls should request hypotheses when the API supports it."""
    calls: list[tuple[tuple, dict]] = []

    class DummyModel:
        """Minimal fake model exposing a compatible ``transcribe`` method."""

        def transcribe(self, paths2audio_files, batch_size: int = 4, return_hypotheses: bool = False):
            """Record invocation kwargs and return a stub hypothesis payload."""
            calls.append((tuple(paths2audio_files), {"batch_size": batch_size, "return_hypotheses": return_hypotheses}))
            return [{"text": "ok", "word_confidence": [0.9]}]

    out = nemo_asr._call_transcribe_compat(DummyModel(), [Path("a.wav")], batch_size=3)

    assert out == [{"text": "ok", "word_confidence": [0.9]}]
    assert calls
    assert calls[0][1]["return_hypotheses"] is True
