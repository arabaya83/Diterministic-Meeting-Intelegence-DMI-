from __future__ import annotations

import audioop
import wave
from pathlib import Path


def wav_metrics(path: Path, frame_ms: int = 30) -> dict:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    duration = nframes / float(sample_rate) if sample_rate else 0.0
    rms = float(audioop.rms(raw, sample_width)) if raw else 0.0

    bytes_per_frame = sample_width * channels
    window_frames = max(1, int(sample_rate * frame_ms / 1000))
    window_bytes = window_frames * bytes_per_frame
    silent = 0
    total = 0
    threshold = max(5, int(rms * 0.08))
    for i in range(0, len(raw), window_bytes):
        chunk = raw[i : i + window_bytes]
        if not chunk:
            continue
        total += 1
        if audioop.rms(chunk, sample_width) <= threshold:
            silent += 1
    silence_ratio = float(silent / total) if total else 0.0
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width_bytes": sample_width,
        "duration_sec": round(duration, 3),
        "rms": round(rms, 3),
        "silence_ratio": round(silence_ratio, 4),
    }

