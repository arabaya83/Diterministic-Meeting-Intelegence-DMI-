#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import tempfile
import wave
from pathlib import Path
from statistics import mean

from nemo_contract import ensure_dir, write_json, write_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeMo ASR wrapper for AMI pipeline artifact contract")
    p.add_argument("--audio", required=True)
    p.add_argument("--model", required=False, help="Local NeMo ASR model path (.nemo preferred)")
    p.add_argument("--diarization-json", required=False, help="Diarization segments JSON to preserve speakers")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for NeMo transcribe() over diarized chunks")
    p.add_argument("--merge-gap-sec", type=float, default=0.2, help="Merge adjacent same-speaker segments if gap <= this")
    p.add_argument(
        "--merge-max-duration-sec",
        type=float,
        default=18.0,
        help="Do not merge if merged segment duration would exceed this",
    )
    p.add_argument(
        "--split-max-duration-sec",
        type=float,
        default=25.0,
        help="Split diarization segments longer than this before ASR",
    )
    p.add_argument(
        "--min-segment-duration-sec",
        type=float,
        default=0.15,
        help="Drop segments shorter than this after optimization",
    )
    p.add_argument("--input-asr-json", help="Reuse existing ASR segments JSON and regenerate summary artifacts")
    p.add_argument("--try-nemo-api", action="store_true", help="Attempt NeMo ASR API (version-dependent)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    asr_json = out_dir / "asr_segments.json"
    conf_json = out_dir / "asr_confidence.json"
    txt_path = out_dir / "full_transcript.txt"

    if args.input_asr_json:
        segs = json.loads(Path(args.input_asr_json).read_text(encoding="utf-8"))
        finalize_outputs(segs, args.meeting_id, asr_json, conf_json, txt_path)
        return 0

    if args.try_nemo_api:
        segs = run_nemo_asr_api(
            audio_path=Path(args.audio),
            model_path=Path(args.model) if args.model else None,
            diarization_json_path=Path(args.diarization_json) if args.diarization_json else None,
            batch_size=max(1, int(args.batch_size)),
            merge_gap_sec=float(args.merge_gap_sec),
            merge_max_duration_sec=float(args.merge_max_duration_sec),
            split_max_duration_sec=float(args.split_max_duration_sec),
            min_segment_duration_sec=float(args.min_segment_duration_sec),
        )
        finalize_outputs(segs, args.meeting_id, asr_json, conf_json, txt_path)
        return 0

    print(
        "No ASR source provided. Use --input-asr-json or --try-nemo-api with a local NeMo ASR model.",
        flush=True,
    )
    return 2


def finalize_outputs(segs: list[dict], meeting_id: str, asr_json: Path, conf_json: Path, txt_path: Path) -> None:
    for s in segs:
        s.setdefault("source", "nemo_asr")
        s.setdefault("confidence", 0.0)
    segs.sort(key=lambda s: (s["start"], s["end"], s.get("speaker", "")))
    write_json(asr_json, segs)
    confs = [float(s.get("confidence", 0.0) or 0.0) for s in segs]
    conf = {
        "meeting_id": meeting_id,
        "segment_count": len(segs),
        "mean_confidence": round(mean(confs), 4) if confs else 0.0,
        "min_confidence": round(min(confs), 4) if confs else 0.0,
        "max_confidence": round(max(confs), 4) if confs else 0.0,
        "nonzero_confidence_count": sum(1 for c in confs if c > 0.0),
    }
    write_json(conf_json, conf)
    lines = [f"[{s['start']:.2f}-{s['end']:.2f}] {s.get('speaker','SPEAKER_1')}: {s.get('text','')}" for s in segs]
    write_text(txt_path, "\n".join(lines) + ("\n" if lines else ""))


def run_nemo_asr_api(
    audio_path: Path,
    model_path: Path | None,
    diarization_json_path: Path | None,
    batch_size: int = 8,
    merge_gap_sec: float = 0.2,
    merge_max_duration_sec: float = 18.0,
    split_max_duration_sec: float = 25.0,
    min_segment_duration_sec: float = 0.15,
) -> list[dict]:
    if model_path is None:
        raise ValueError("--model is required with --try-nemo-api")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    try:
        import torch  # type: ignore
        from nemo.collections.asr.models import ASRModel  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing NeMo ASR dependencies (nemo_toolkit[asr], torch)") from e

    # Load local model only (offline).
    if model_path.is_file():
        model = ASRModel.restore_from(str(model_path))
    else:
        raise RuntimeError("For offline reproducibility, pass a local .nemo file to --model")

    with contextlib.suppress(Exception):
        model.eval()
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            model = model.cuda()

    diar_segments = load_diar_segments(diarization_json_path) if diarization_json_path and diarization_json_path.exists() else None
    if not diar_segments:
        hyp = transcribe_single_hyp(model, audio_path, batch_size=1)
        return [
            {
                "start": 0.0,
                "end": round(wav_duration(audio_path), 3),
                "speaker": "SPEAKER_1",
                "text": hyp["text"],
                "confidence": hyp["confidence"],
                "source": "nemo_asr_whole_file",
            }
        ]

    raw_count = len(diar_segments)
    diar_segments = optimize_diar_segments(
        diar_segments,
        merge_gap_sec=merge_gap_sec,
        merge_max_duration_sec=merge_max_duration_sec,
        split_max_duration_sec=split_max_duration_sec,
        min_segment_duration_sec=min_segment_duration_sec,
    )
    print(
        f"ASR diar-segment optimization: raw={raw_count} optimized={len(diar_segments)} "
        f"(merge_gap={merge_gap_sec}s split_max={split_max_duration_sec}s batch={batch_size})",
        flush=True,
    )

    segs: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="nemo_asr_segments_") as td:
        td_path = Path(td)
        jobs: list[dict] = []
        for i, d in enumerate(diar_segments, start=1):
            chunk = td_path / f"seg_{i:05d}.wav"
            extract_wav_chunk(audio_path, chunk, float(d["start"]), float(d["end"]))
            jobs.append(
                {
                    "chunk": chunk,
                    "start": round(float(d["start"]), 3),
                    "end": round(float(d["end"]), 3),
                    "speaker": d.get("speaker", "SPEAKER_1"),
                }
            )
        for batch in batched(jobs, batch_size):
            hyps = transcribe_many_hyps(model, [j["chunk"] for j in batch], batch_size=batch_size)
            if len(hyps) != len(batch):
                raise RuntimeError(
                    f"NeMo transcribe returned {len(hyps)} outputs for batch of {len(batch)} chunks"
                )
            for j, hyp in zip(batch, hyps):
                segs.append(
                    {
                        "start": j["start"],
                        "end": j["end"],
                        "speaker": j["speaker"],
                        "text": hyp["text"],
                        "confidence": hyp["confidence"],
                        "source": "nemo_asr_diarized_chunk",
                    }
                )
    return segs


def transcribe_single_hyp(model, wav_path: Path, batch_size: int = 1) -> dict:
    out = _call_transcribe_compat(model, [wav_path], batch_size=batch_size)
    normalized = normalize_transcribe_outputs(out)
    if normalized:
        return normalized[0]
    return {"text": "", "confidence": 0.0}


def transcribe_many_hyps(model, wav_paths: list[Path], batch_size: int = 8) -> list[dict]:
    out = _call_transcribe_compat(model, wav_paths, batch_size=batch_size)
    return normalize_transcribe_outputs(out)


def _call_transcribe_compat(model, wav_paths: list[Path], batch_size: int = 1):
    path_list = [str(p) for p in wav_paths]
    try:
        sig = inspect.signature(model.transcribe)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    # Try the most common variants across NeMo versions.
    attempts = []
    if "paths2audio_files" in params:
        attempts.append(((), {"paths2audio_files": path_list, "batch_size": batch_size}))
    if "audio" in params:
        attempts.append(((), {"audio": path_list, "batch_size": batch_size}))
    # Positional list form works in multiple versions.
    attempts.append(((path_list,), {"batch_size": batch_size}))
    attempts.append(((path_list,), {}))

    last_err = None
    for args, kwargs in attempts:
        try:
            return model.transcribe(*args, **kwargs)
        except TypeError as e:
            last_err = e
            continue

    # Re-raise the last TypeError with context if all variants fail.
    raise TypeError(
        f"Unable to call NeMo transcribe() with compatible arguments for files {path_list[:3]}..."
        f"Last error: {last_err}"
    )


def normalize_transcribe_outputs(out) -> list[dict]:
    if isinstance(out, (list, tuple)):
        return [_normalize_transcribe_item(item) for item in out]
    return [_normalize_transcribe_item(out)]


def _normalize_transcribe_item(item) -> dict:
    if isinstance(item, str):
        return {"text": item.strip(), "confidence": 0.0}

    text = _extract_text(item)
    confidence = _extract_confidence(item)
    return {"text": text, "confidence": confidence}


def _extract_text(item) -> str:
    # NeMo hypothesis objects commonly expose `.text`; dict-like outputs may use "text".
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(item, dict):
        t = item.get("text")
        if isinstance(t, str):
            return t.strip()
    return str(item).strip()


def _extract_confidence(item) -> float:
    # Try common NeMo fields in order of preference.
    candidates = []
    if isinstance(item, dict):
        candidates.extend(
            [
                item.get("confidence"),
                item.get("score"),
                item.get("avg_logprob"),
                item.get("y_sequence_score"),
            ]
        )
    for attr in ("confidence", "score", "avg_logprob", "y_sequence_score"):
        with contextlib.suppress(Exception):
            candidates.append(getattr(item, attr))

    # Confidence field may already be a bounded probability.
    for v in candidates:
        c = _as_confidence_if_bounded(v)
        if c is not None:
            return c

    # Score/logprob-like values are often negative; map monotonically into [0,1] with exp(score/len_proxy).
    for v in candidates:
        f = _as_float(v)
        if f is None:
            continue
        if math.isfinite(f):
            if f <= 0:
                # Conservative mapping for negative log-prob-like scores.
                c = math.exp(max(-20.0, f))
                return round(max(0.0, min(1.0, c)), 6)
            # Positive non-bounded scores are ambiguous; skip.

    # As fallback, derive from token/confidence-like lists if present.
    for attr in ("token_confidence", "token_confidences", "char_confidence", "char_confidences"):
        vals = None
        if isinstance(item, dict):
            vals = item.get(attr)
        if vals is None:
            with contextlib.suppress(Exception):
                vals = getattr(item, attr)
        c = _mean_bounded(vals)
        if c is not None:
            return c

    return 0.0


def _as_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _as_confidence_if_bounded(v) -> float | None:
    f = _as_float(v)
    if f is None or not math.isfinite(f):
        return None
    if 0.0 <= f <= 1.0:
        return round(f, 6)
    return None


def _mean_bounded(vals) -> float | None:
    if not isinstance(vals, (list, tuple)) or not vals:
        return None
    xs = []
    for v in vals:
        f = _as_float(v)
        if f is None or not math.isfinite(f):
            continue
        if 0.0 <= f <= 1.0:
            xs.append(f)
    if not xs:
        return None
    return round(sum(xs) / len(xs), 6)


def batched(items: list[dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_diar_segments(path: Path) -> list[dict]:
    segs = json.loads(path.read_text(encoding="utf-8"))
    valid = [s for s in segs if float(s["end"]) > float(s["start"])]
    valid.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    return valid


def optimize_diar_segments(
    segs: list[dict],
    merge_gap_sec: float = 0.2,
    merge_max_duration_sec: float = 18.0,
    split_max_duration_sec: float = 25.0,
    min_segment_duration_sec: float = 0.15,
) -> list[dict]:
    merged = merge_adjacent_same_speaker(segs, merge_gap_sec=merge_gap_sec, merge_max_duration_sec=merge_max_duration_sec)
    split = split_long_segments(merged, max_duration_sec=split_max_duration_sec)
    out = []
    for s in split:
        dur = float(s["end"]) - float(s["start"])
        if dur >= min_segment_duration_sec:
            out.append(
                {
                    "start": round(float(s["start"]), 3),
                    "end": round(float(s["end"]), 3),
                    "speaker": s.get("speaker", "SPEAKER_1"),
                }
            )
    out.sort(key=lambda s: (s["start"], s["end"], s["speaker"]))
    return out


def merge_adjacent_same_speaker(
    segs: list[dict], merge_gap_sec: float = 0.2, merge_max_duration_sec: float = 18.0
) -> list[dict]:
    if not segs:
        return []
    out: list[dict] = []
    cur = {
        "start": float(segs[0]["start"]),
        "end": float(segs[0]["end"]),
        "speaker": segs[0].get("speaker", "SPEAKER_1"),
    }
    for s in segs[1:]:
        s_start = float(s["start"])
        s_end = float(s["end"])
        s_spk = s.get("speaker", "SPEAKER_1")
        gap = s_start - cur["end"]
        merged_dur = max(cur["end"], s_end) - cur["start"]
        can_merge = s_spk == cur["speaker"] and gap >= 0 and gap <= merge_gap_sec and merged_dur <= merge_max_duration_sec
        if can_merge:
            cur["end"] = max(cur["end"], s_end)
        else:
            out.append(cur)
            cur = {"start": s_start, "end": s_end, "speaker": s_spk}
    out.append(cur)
    return out


def split_long_segments(segs: list[dict], max_duration_sec: float = 25.0) -> list[dict]:
    if max_duration_sec <= 0:
        return segs
    out: list[dict] = []
    for s in segs:
        start = float(s["start"])
        end = float(s["end"])
        speaker = s.get("speaker", "SPEAKER_1")
        if end - start <= max_duration_sec:
            out.append({"start": start, "end": end, "speaker": speaker})
            continue
        t = start
        while t < end:
            nxt = min(end, t + max_duration_sec)
            out.append({"start": t, "end": nxt, "speaker": speaker})
            t = nxt
    return out


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def extract_wav_chunk(src: Path, dst: Path, start: float, end: float) -> None:
    start = max(0.0, float(start))
    end = max(start, float(end))
    with wave.open(str(src), "rb") as r:
        params = r.getparams()
        fr = r.getframerate()
        sw = r.getsampwidth()
        ch = r.getnchannels()
        start_frame = int(start * fr)
        end_frame = int(end * fr)
        nframes = max(1, end_frame - start_frame)
        r.setpos(min(start_frame, r.getnframes()))
        raw = r.readframes(nframes)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(fr)
        w.writeframes(raw)


if __name__ == "__main__":
    raise SystemExit(main())
