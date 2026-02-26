#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from nemo_contract import diarization_json_from_rttm, ensure_dir, write_diar_rttm, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeMo diarization wrapper for AMI pipeline artifact contract")
    p.add_argument("--audio", required=True)
    p.add_argument("--config", required=False, help="Local NeMo diarizer YAML config")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--input-rttm", help="Reuse an existing RTTM and convert to diarization JSON")
    p.add_argument("--input-json", help="Reuse an existing diarization JSON")
    p.add_argument(
        "--try-nemo-api",
        action="store_true",
        help="Attempt NeMo NeuralDiarizer API using --config (version-dependent).",
    )
    p.add_argument(
        "--merge-same-speaker-gap-sec",
        type=float,
        default=0.2,
        help="Merge adjacent same-speaker segments when separated by <= this gap (seconds).",
    )
    p.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.18,
        help="Drop diarization segments shorter than this duration after merge pass (seconds).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    diar_rttm = out_dir / "diarization.rttm"
    diar_json = out_dir / "diarization_segments.json"

    if args.input_json:
        segs = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        segs = postprocess_segments(
            segs,
            merge_same_speaker_gap_sec=args.merge_same_speaker_gap_sec,
            min_segment_sec=args.min_segment_sec,
        )
        write_json(diar_json, segs)
        write_diar_rttm(diar_rttm, args.meeting_id, segs)
        return 0

    if args.input_rttm:
        shutil.copy2(args.input_rttm, diar_rttm)
        segs = diarization_json_from_rttm(diar_rttm)
        segs = postprocess_segments(
            segs,
            merge_same_speaker_gap_sec=args.merge_same_speaker_gap_sec,
            min_segment_sec=args.min_segment_sec,
        )
        write_diar_rttm(diar_rttm, args.meeting_id, segs)
        write_json(diar_json, segs)
        return 0

    if args.try_nemo_api:
        try:
            run_nemo_diarizer_api(
                Path(args.audio),
                Path(args.config) if args.config else None,
                out_dir,
                args.meeting_id,
                merge_same_speaker_gap_sec=args.merge_same_speaker_gap_sec,
                min_segment_sec=args.min_segment_sec,
            )
        except Exception as e:
            print(f"NeMo diarization API execution failed: {e}", file=sys.stderr)
            return 2
        if diar_rttm.exists() and diar_json.exists():
            return 0
        print("NeMo diarization completed but expected outputs were not found.", file=sys.stderr)
        return 2

    print(
        "No diarization source provided. Use --input-rttm/--input-json, or --try-nemo-api with a pinned NeMo config.",
        file=sys.stderr,
    )
    return 2


def run_nemo_diarizer_api(
    audio_path: Path,
    config_path: Path | None,
    out_dir: Path,
    meeting_id: str,
    merge_same_speaker_gap_sec: float,
    min_segment_sec: float,
) -> None:
    if config_path is None:
        raise ValueError("--config is required with --try-nemo-api")
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    try:
        from omegaconf import OmegaConf  # type: ignore
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing NeMo diarization dependencies (nemo_toolkit[asr], omegaconf)") from e

    with tempfile.TemporaryDirectory(prefix="nemo_diarize_") as td:
        td_path = Path(td)
        manifest_path = td_path / "manifest.json"
        manifest = {
            "audio_filepath": str(audio_path.resolve()),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

        cfg = OmegaConf.load(str(config_path))
        # Best-effort field mapping across NeMo diarizer config variants.
        if "diarizer" not in cfg:
            raise RuntimeError("Expected 'diarizer' root in NeMo diarizer config")
        cfg.diarizer.manifest_filepath = str(manifest_path)
        cfg.diarizer.out_dir = str(td_path / "nemo_out")

        diarizer = NeuralDiarizer(cfg=cfg)
        _debug_and_force_clustering_cuda(diarizer, cfg, torch)
        diarizer.diarize()

        produced = list((td_path / "nemo_out").rglob("*.rttm"))
        if not produced:
            raise RuntimeError("NeMo diarizer did not produce any RTTM files")
        # Prefer RTTM matching the meeting id.
        chosen = next((p for p in produced if meeting_id in p.name), produced[0])
        shutil.copy2(chosen, out_dir / "diarization.rttm")
        raw_segs = diarization_json_from_rttm(out_dir / "diarization.rttm")
        segs = postprocess_segments(
            raw_segs,
            merge_same_speaker_gap_sec=merge_same_speaker_gap_sec,
            min_segment_sec=min_segment_sec,
        )
        if len(segs) != len(raw_segs):
            print(
                "NeMo diarization postprocess: "
                f"raw_segments={len(raw_segs)} -> cleaned_segments={len(segs)} "
                f"(merge_gap<={merge_same_speaker_gap_sec:.3f}s, min_seg>={min_segment_sec:.3f}s)",
                flush=True,
            )
        write_diar_rttm(out_dir / "diarization.rttm", meeting_id, segs)
        write_json(out_dir / "diarization_segments.json", segs)


def _debug_and_force_clustering_cuda(diarizer, cfg, torch_mod) -> None:
    cfg_device = str(getattr(cfg, "device", ""))

    def _device_of(obj) -> str:
        try:
            return str(getattr(obj, "device"))
        except Exception:
            return "<unknown>"

    clus_model = None
    spk_model = None
    vad_model = None
    with contextlib.suppress(Exception):
        clus_model = diarizer.clustering_embedding.clus_diar_model
    if clus_model is None:
        print("NeMo diarizer debug: clustering model not found before diarize()", flush=True)
        return

    with contextlib.suppress(Exception):
        spk_model = clus_model._speaker_model
    with contextlib.suppress(Exception):
        vad_model = clus_model._vad_model

    print(
        "NeMo diarizer devices (before): "
        f"cfg.device={cfg_device}, "
        f"speaker_model={_device_of(spk_model) if spk_model is not None else '<none>'}, "
        f"vad_model={_device_of(vad_model) if vad_model is not None else '<none>'}",
        flush=True,
    )

    wants_cuda = cfg_device.startswith("cuda")
    if wants_cuda and torch_mod.cuda.is_available() and spk_model is not None:
        with contextlib.suppress(Exception):
            clus_model._speaker_model = spk_model.to(torch_mod.device("cuda"))
        with contextlib.suppress(Exception):
            diarizer._speaker_model = clus_model._speaker_model
        with contextlib.suppress(Exception):
            diarizer.clustering_embedding._speaker_model = clus_model._speaker_model

    # Optional: ensure VAD model also lands on CUDA when configured for cuda.
    if wants_cuda and torch_mod.cuda.is_available() and vad_model is not None:
        with contextlib.suppress(Exception):
            clus_model._vad_model = vad_model.to(torch_mod.device("cuda"))

    with contextlib.suppress(Exception):
        spk_model = clus_model._speaker_model
    with contextlib.suppress(Exception):
        vad_model = clus_model._vad_model
    print(
        "NeMo diarizer devices (after): "
        f"speaker_model={_device_of(spk_model) if spk_model is not None else '<none>'}, "
        f"vad_model={_device_of(vad_model) if vad_model is not None else '<none>'}",
        flush=True,
    )


def postprocess_segments(
    segs: list[dict],
    merge_same_speaker_gap_sec: float,
    min_segment_sec: float,
) -> list[dict]:
    cleaned = [
        {
            "start": round(float(s["start"]), 3),
            "end": round(float(s["end"]), 3),
            "speaker": str(s["speaker"]),
            "source": str(s.get("source", "nemo_rttm")),
        }
        for s in segs
        if float(s.get("end", 0.0)) > float(s.get("start", 0.0))
    ]
    cleaned.sort(key=lambda s: (s["start"], s["end"], s["speaker"]))
    cleaned = _merge_same_speaker_gaps(cleaned, max_gap_sec=max(0.0, merge_same_speaker_gap_sec))
    if min_segment_sec > 0:
        cleaned = [s for s in cleaned if (s["end"] - s["start"]) >= min_segment_sec]
    # Merge again after pruning micro-fragments to reduce residual fragmentation.
    cleaned = _merge_same_speaker_gaps(cleaned, max_gap_sec=max(0.0, merge_same_speaker_gap_sec))
    return cleaned


def _merge_same_speaker_gaps(segs: list[dict], max_gap_sec: float) -> list[dict]:
    if not segs:
        return []
    out = [dict(segs[0])]
    for s in segs[1:]:
        prev = out[-1]
        gap = float(s["start"]) - float(prev["end"])
        if s["speaker"] == prev["speaker"] and gap <= max_gap_sec:
            prev["end"] = round(max(float(prev["end"]), float(s["end"])), 3)
        else:
            out.append(dict(s))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
