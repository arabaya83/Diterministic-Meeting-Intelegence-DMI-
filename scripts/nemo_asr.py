#!/usr/bin/env python3
"""Normalize NeMo ASR outputs into stable repository transcript artifacts."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import tempfile
import wave
from pathlib import Path

from nemo_contract import ensure_dir, write_json, write_text


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ASR wrapper."""
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
    """Run the selected ASR path and materialize transcript artifacts."""
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    asr_json = out_dir / "asr_segments.json"
    txt_path = out_dir / "full_transcript.txt"

    if args.input_asr_json:
        segs = json.loads(Path(args.input_asr_json).read_text(encoding="utf-8"))
        finalize_outputs(segs, asr_json, txt_path)
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
        finalize_outputs(segs, asr_json, txt_path)
        return 0

    print(
        "No ASR source provided. Use --input-asr-json or --try-nemo-api with a local NeMo ASR model.",
        flush=True,
    )
    return 2


def finalize_outputs(segs: list[dict], asr_json: Path, txt_path: Path) -> None:
    """Write normalized ASR JSON and transcript text artifacts."""
    for s in segs:
        s.setdefault("source", "nemo_asr")
        s.setdefault("confidence", 0.0)
    segs.sort(key=lambda s: (s["start"], s["end"], s.get("speaker", "")))
    write_json(asr_json, segs)
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
    """Run a local NeMo ASR model and return normalized segment rows."""
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

    enable_confidence_preservation(model)
    disable_cuda_graph_decoder(model)

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


def disable_cuda_graph_decoder(model) -> None:
    """Disable NeMo RNNT/TDT CUDA-graph decoding when the model exposes that knob.

    Some Parakeet/TDT combinations crash in NeMo's CUDA-graph decoder path on
    this workstation stack. Force the safer non-graph greedy decoder instead.
    """
    changed = False
    logged_attr = "_ami_cuda_graph_decoder_disabled_logged"
    with contextlib.suppress(Exception):
        decoding_cfg = getattr(model, "cfg", {}).get("decoding")
        if decoding_cfg and "greedy" in decoding_cfg:
            decoding_cfg.greedy["use_cuda_graph_decoder"] = False
            changed = True
            with contextlib.suppress(Exception):
                model.change_decoding_strategy(decoding_cfg)

    decoding = getattr(model, "decoding", None)
    if decoding is None:
        return

    with contextlib.suppress(Exception):
        cfg = getattr(decoding, "cfg", None)
        if cfg and "greedy" in cfg:
            cfg.greedy["use_cuda_graph_decoder"] = False
            changed = True

    with contextlib.suppress(Exception):
        if hasattr(decoding, "use_cuda_graph_decoder"):
            decoding.use_cuda_graph_decoder = False
            changed = True

    decoding_computer = getattr(decoding, "decoding", None)
    if decoding_computer is not None:
        with contextlib.suppress(Exception):
            if hasattr(decoding_computer, "use_cuda_graph_decoder"):
                decoding_computer.use_cuda_graph_decoder = False
                changed = True
        with contextlib.suppress(Exception):
            if hasattr(decoding_computer, "allow_cuda_graphs"):
                decoding_computer.allow_cuda_graphs = False
                changed = True
        with contextlib.suppress(Exception):
            if hasattr(decoding_computer, "force_cuda_graphs_mode"):
                decoding_computer.force_cuda_graphs_mode("no_graphs")
                changed = True
        with contextlib.suppress(Exception):
            if hasattr(decoding_computer, "disable_cuda_graphs"):
                decoding_computer.disable_cuda_graphs()
                changed = True
        with contextlib.suppress(Exception):
            if hasattr(decoding_computer, "cuda_graphs_mode"):
                decoding_computer.cuda_graphs_mode = None
                changed = True

    if changed and not getattr(model, logged_attr, False):
        print("NeMo ASR: disabled CUDA-graph decoder path for RNNT/TDT decoding", flush=True)
        with contextlib.suppress(Exception):
            setattr(model, logged_attr, True)


def enable_confidence_preservation(model) -> None:
    """Best-effort enablement of NeMo confidence outputs on RNNT/TDT models.

    NeMo exposes confidence-bearing fields such as `word_confidence`,
    `token_confidence`, and `frame_confidence` only when the decoding config
    explicitly preserves them. Older model/runtime combinations may ignore some
    of these settings; in that case the wrapper still falls back to 0.0.
    """
    changed = False
    logged_attr = "_ami_confidence_preservation_logged"

    with contextlib.suppress(Exception):
        decoding_cfg = getattr(model, "cfg", {}).get("decoding")
        if decoding_cfg:
            confidence_cfg = decoding_cfg.get("confidence_cfg")
            if confidence_cfg is None:
                decoding_cfg["confidence_cfg"] = {}
                confidence_cfg = decoding_cfg["confidence_cfg"]
            confidence_cfg["preserve_frame_confidence"] = True
            confidence_cfg["preserve_token_confidence"] = True
            confidence_cfg["preserve_word_confidence"] = True
            method_cfg = confidence_cfg.get("method_cfg") or {}
            method_cfg.setdefault("name", "max_prob")
            confidence_cfg["method_cfg"] = method_cfg

            greedy_cfg = decoding_cfg.get("greedy")
            if greedy_cfg is not None:
                greedy_cfg["preserve_frame_confidence"] = True
                greedy_cfg["preserve_token_confidence"] = True
                greedy_cfg["preserve_word_confidence"] = True
                greedy_cfg["confidence_method_cfg"] = method_cfg
            changed = True
            with contextlib.suppress(Exception):
                model.change_decoding_strategy(decoding_cfg)

    decoding = getattr(model, "decoding", None)
    cfg = getattr(decoding, "cfg", None) if decoding is not None else None
    if cfg is not None:
        with contextlib.suppress(Exception):
            confidence_cfg = getattr(cfg, "confidence_cfg", None)
            if confidence_cfg is None:
                cfg.confidence_cfg = {}
                confidence_cfg = cfg.confidence_cfg
            confidence_cfg["preserve_frame_confidence"] = True
            confidence_cfg["preserve_token_confidence"] = True
            confidence_cfg["preserve_word_confidence"] = True
            method_cfg = confidence_cfg.get("method_cfg") or {}
            method_cfg.setdefault("name", "max_prob")
            confidence_cfg["method_cfg"] = method_cfg
            changed = True
        with contextlib.suppress(Exception):
            greedy_cfg = getattr(cfg, "greedy", None)
            if greedy_cfg is not None:
                greedy_cfg["preserve_frame_confidence"] = True
                greedy_cfg["preserve_token_confidence"] = True
                greedy_cfg["preserve_word_confidence"] = True
                greedy_cfg["confidence_method_cfg"] = confidence_cfg.get("method_cfg") or {"name": "max_prob"}
                changed = True
        with contextlib.suppress(Exception):
            model.change_decoding_strategy(cfg)

    if changed and not getattr(model, logged_attr, False):
        print("NeMo ASR: enabled confidence preservation on decoder outputs", flush=True)
        with contextlib.suppress(Exception):
            setattr(model, logged_attr, True)


def transcribe_single_hyp(model, wav_path: Path, batch_size: int = 1) -> dict:
    """Transcribe one waveform and return normalized text/confidence fields."""
    out = _call_transcribe_compat(model, [wav_path], batch_size=batch_size)
    normalized = normalize_transcribe_outputs(out)
    if normalized:
        return normalized[0]
    return {"text": "", "confidence": 0.0}


def transcribe_many_hyps(model, wav_paths: list[Path], batch_size: int = 8) -> list[dict]:
    """Transcribe multiple waveform chunks and normalize all outputs."""
    out = _call_transcribe_compat(model, wav_paths, batch_size=batch_size)
    return normalize_transcribe_outputs(out)


def _call_transcribe_compat(model, wav_paths: list[Path], batch_size: int = 1):
    """Call ``model.transcribe`` across NeMo API variants."""
    disable_cuda_graph_decoder(model)
    path_list = [str(p) for p in wav_paths]
    try:
        sig = inspect.signature(model.transcribe)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    # Try the most common variants across NeMo versions.
    attempts = []
    if "paths2audio_files" in params:
        attempts.append(((), {"paths2audio_files": path_list, "batch_size": batch_size, "return_hypotheses": True}))
        attempts.append(((), {"paths2audio_files": path_list, "batch_size": batch_size}))
    if "audio" in params:
        attempts.append(((), {"audio": path_list, "batch_size": batch_size, "return_hypotheses": True}))
        attempts.append(((), {"audio": path_list, "batch_size": batch_size}))
    # Positional list form works in multiple versions.
    attempts.append(((path_list,), {"batch_size": batch_size, "return_hypotheses": True}))
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
    """Normalize NeMo transcribe outputs into text/confidence dictionaries."""
    if isinstance(out, (list, tuple)):
        return [_normalize_transcribe_item(item) for item in out]
    return [_normalize_transcribe_item(out)]


def _normalize_transcribe_item(item) -> dict:
    """Normalize one NeMo hypothesis object or dictionary."""
    if isinstance(item, str):
        return {"text": item.strip(), "confidence": 0.0}

    text = _extract_text(item)
    confidence = _extract_confidence(item)
    return {"text": text, "confidence": confidence}


def _extract_text(item) -> str:
    """Extract transcript text from a NeMo hypothesis-like object."""
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
    """Extract a bounded confidence score from NeMo hypothesis metadata."""
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
    for attr in (
        "word_confidence",
        "token_confidence",
        "token_confidences",
        "frame_confidence",
        "char_confidence",
        "char_confidences",
    ):
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
    """Best-effort float conversion used during confidence extraction."""
    try:
        return float(v)
    except Exception:
        return None


def _as_confidence_if_bounded(v) -> float | None:
    """Return a value only when it looks like a valid confidence score."""
    f = _as_float(v)
    if f is None or not math.isfinite(f):
        return None
    if 0.0 <= f <= 1.0:
        return round(f, 6)
    return None


def _mean_bounded(vals) -> float | None:
    """Average bounded confidence values, ignoring invalid entries."""
    if not isinstance(vals, (list, tuple)) or not vals:
        return None
    xs = []
    stack = list(vals)
    while stack:
        v = stack.pop()
        if isinstance(v, (list, tuple)):
            stack.extend(v)
            continue
        f = _as_float(v)
        if f is None or not math.isfinite(f):
            continue
        if 0.0 <= f <= 1.0:
            xs.append(f)
    if not xs:
        return None
    return round(sum(xs) / len(xs), 6)


def batched(items: list[dict], batch_size: int):
    """Yield consecutive batches from a list without reordering it."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_diar_segments(path: Path) -> list[dict]:
    """Load and sort valid diarization segments from JSON."""
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
    """Merge, split, and prune diarization segments before ASR chunking."""
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
    """Merge adjacent same-speaker segments subject to gap and duration limits."""
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
    """Split long diarization segments into smaller ASR chunks."""
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
    """Return WAV duration in seconds using the standard library reader."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def extract_wav_chunk(src: Path, dst: Path, start: float, end: float) -> None:
    """Extract a WAV subsegment into a temporary chunk file."""
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
