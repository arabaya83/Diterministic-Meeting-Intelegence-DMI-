"""Core stage-based AMI pipeline orchestration.

This module coordinates the offline AMI workflow from ingest through
evaluation. The stage order is intentionally explicit because downstream
artifacts and the UI depend on stable file names and deterministic stage
boundaries:

- speech artifacts are written per meeting under `artifacts/ami/{meeting_id}/`
- aggregate evaluation outputs are written under `artifacts/eval/ami/`
- the final summary pass (`summary_finalize`) merges extraction findings back
  into the MoM before evaluation so the stored summary is the user-facing one
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .backends.llama_cpp_backend import LlamaCppBackend
from .backends.nemo_backend import NemoBackendError, NemoSpeechBackend
from .config import AppConfig
from .schemas.models import (
    ASRSegment,
    ActionItem,
    CanonicalMeeting,
    DecisionItem,
    DiarizationSegment,
    EvidenceBackedPoint,
    ExtractionOutput,
    MinutesSummary,
    QCMetrics,
    TranscriptChunk,
    TranscriptTurn,
    VADSegment,
)
from .utils.ami_annotations import build_utterances, load_abstractive_summary_text, load_word_tokens, reference_plain_text
from .utils.audio_utils import wav_metrics
from .utils.determinism import configure_determinism
from .utils.io_utils import ensure_dir, upsert_csv, upsert_jsonl, write_json, write_jsonl, write_text
from .utils.speech_eval import (
    compute_cpwer,
    compute_der_approx_nooverlap,
    load_hyp_asr_views,
    load_hyp_diarization,
    load_ref_diarization_from_words,
    load_reference_asr_views,
)
from .utils.traceability import (
    StageTraceWriter,
    collect_code_provenance,
    collect_environment_snapshot,
    config_digest,
    offline_preflight_audit,
    trace_stage,
)


@dataclass
class PipelinePaths:
    """Resolved run-time paths for one meeting pipeline execution."""

    raw_audio_dir: Path
    annotations_dir: Path
    staged_dir: Path
    artifacts_dir: Path
    meeting_artifacts_dir: Path
    eval_dir: Path


def resolve_paths(cfg: AppConfig, meeting_id: str) -> PipelinePaths:
    """Resolve and create directory paths used by a meeting run."""
    raw_audio_dir = Path(cfg.paths.raw_audio_dir)
    annotations_dir = Path(cfg.paths.annotations_dir)
    staged_dir = Path(cfg.paths.staged_dir)
    artifacts_root = Path(cfg.paths.artifacts_dir)
    meeting_artifacts_dir = artifacts_root / "ami" / meeting_id
    eval_dir = artifacts_root / "eval" / "ami"
    for p in [staged_dir, meeting_artifacts_dir, eval_dir]:
        ensure_dir(p)
    return PipelinePaths(
        raw_audio_dir=raw_audio_dir,
        annotations_dir=annotations_dir,
        staged_dir=staged_dir,
        artifacts_dir=artifacts_root,
        meeting_artifacts_dir=meeting_artifacts_dir,
        eval_dir=eval_dir,
    )


def list_meetings(cfg: AppConfig) -> list[str]:
    """List AMI meeting ids discovered from raw audio filenames."""
    raw_audio_dir = Path(cfg.paths.raw_audio_dir)
    meetings = []
    for p in sorted(raw_audio_dir.glob("*.Mix-Headset.wav")):
        meeting_id = p.name.replace(".Mix-Headset.wav", "")
        meetings.append(meeting_id)
    return meetings


def run_pipeline(cfg: AppConfig, meeting_id: str) -> dict:
    """Execute the full pipeline for one meeting id.

    Returns:
        dict: Manifest dictionary persisted as `run_manifest.json`.

    Notes:
        The persisted MoM artifacts reflect the post-extraction
        `summary_finalize` stage, not the raw first-pass summary. This matters
        for both UI rendering and ROUGE evaluation.
    """
    mlflow_run = None
    mlflow_mod = None
    mlflow_disabled_reason = None
    if cfg.runtime.enable_mlflow_logging:
        try:
            import mlflow  # type: ignore

            tracking_uri = cfg.runtime.mlflow_tracking_uri or f"file:{Path(cfg.paths.artifacts_dir) / 'mlruns'}"
            if cfg.runtime.offline and not str(tracking_uri).startswith("file:"):
                raise RuntimeError(f"Offline mode requires local file-based MLflow tracking URI, got: {tracking_uri}")
            mlflow.set_tracking_uri(str(tracking_uri))
            mlflow.set_experiment(cfg.runtime.mlflow_experiment)
            mlflow_run = mlflow.start_run(run_name=f"{meeting_id}")
            mlflow_mod = mlflow
            mlflow.log_params(
                {
                    "meeting_id": meeting_id,
                    "seed": cfg.pipeline.seed,
                    "speech_backend": cfg.pipeline.speech_backend.mode,
                    "summarization_backend": cfg.pipeline.summarization_backend.mode,
                    "extraction_backend": cfg.pipeline.extraction_backend.mode,
                    "offline": cfg.runtime.offline,
                }
            )
        except Exception as exc:
            mlflow_disabled_reason = f"{type(exc).__name__}: {exc}"
    det_report = configure_determinism(cfg.pipeline.seed, strict=cfg.runtime.deterministic_mode)
    paths = resolve_paths(cfg, meeting_id)
    try:
        trace_writer = StageTraceWriter(paths.meeting_artifacts_dir / "stage_trace.jsonl", enabled=cfg.runtime.write_stage_trace)
        if cfg.runtime.write_preflight_audit:
            offline_audit = offline_preflight_audit(cfg)
            write_json(paths.meeting_artifacts_dir / "preflight_offline_audit.json", offline_audit)
            if cfg.runtime.offline and cfg.runtime.fail_on_offline_violations and not offline_audit.get("ok", False):
                raise RuntimeError(f"Offline preflight audit failed: {offline_audit.get('violations')}")
        else:
            offline_audit = {"ok": True, "disabled": True}
        repro_report = {
            "meeting_id": meeting_id,
            "config_digest": config_digest(cfg),
            "environment": collect_environment_snapshot(),
            "determinism": det_report,
            "code_provenance": collect_code_provenance(Path.cwd()),
        }
        if mlflow_disabled_reason:
            repro_report["mlflow_logging"] = {"enabled": False, "disabled_reason": mlflow_disabled_reason}
        elif cfg.runtime.enable_mlflow_logging:
            repro_report["mlflow_logging"] = {
                "enabled": True,
                "tracking_uri": cfg.runtime.mlflow_tracking_uri or f"file:{Path(cfg.paths.artifacts_dir) / 'mlruns'}",
                "experiment": cfg.runtime.mlflow_experiment,
            }
        if cfg.runtime.deterministic_mode and cfg.runtime.fail_on_determinism_risks and det_report.get("risks"):
            raise RuntimeError(f"Determinism risks detected in strict mode: {det_report['risks']}")
        write_json(paths.meeting_artifacts_dir / "reproducibility_report.json", repro_report)

        llama_backend = None
        if (
            cfg.pipeline.summarization_backend.mode == "llama_cpp"
            or cfg.pipeline.extraction_backend.mode == "llama_cpp"
        ):
            llama_backend = LlamaCppBackend(cfg)

        token_cache = trace_stage(
            trace_writer,
            "load_annotations",
            lambda: load_word_tokens(paths.annotations_dir, meeting_id),
            meeting_id=meeting_id,
            summarizer=lambda out: {"token_count": len(out) if isinstance(out, list) else None},
        )
        utterances = trace_stage(
            trace_writer,
            "build_utterances",
            lambda: build_utterances(token_cache),
            meeting_id=meeting_id,
            summarizer=lambda out: {"utterance_count": len(out) if isinstance(out, list) else None},
        )

        ingest_out = trace_stage(
            trace_writer,
            "ingest",
            lambda: stage_ingest(cfg, paths, meeting_id),
            meeting_id=meeting_id,
            summarizer=_summarize_ingest_stage,
        )
        if cfg.pipeline.speech_backend.mode == "nemo":
        # NeMo diarization wrapper may internally perform VAD and emit RTTM first; our VAD artifact can then be derived.
            diar_out = trace_stage(
            trace_writer,
            "diarization",
            lambda: stage_diarization(cfg, paths, meeting_id, {"count": 0}, utterances),
            meeting_id=meeting_id,
            summarizer=_summarize_diar_stage,
        )
            vad_out = trace_stage(
            trace_writer,
            "vad",
            lambda: stage_vad(cfg, paths, meeting_id, ingest_out, utterances),
            meeting_id=meeting_id,
            summarizer=_summarize_vad_stage,
        )
            asr_out = trace_stage(
            trace_writer,
            "asr",
            lambda: stage_asr(cfg, paths, meeting_id, diar_out, token_cache),
            meeting_id=meeting_id,
            summarizer=_summarize_asr_stage,
        )
        else:
            vad_out = trace_stage(
            trace_writer,
            "vad",
            lambda: stage_vad(cfg, paths, meeting_id, ingest_out, utterances),
            meeting_id=meeting_id,
            summarizer=_summarize_vad_stage,
        )
            diar_out = trace_stage(
            trace_writer,
            "diarization",
            lambda: stage_diarization(cfg, paths, meeting_id, vad_out, utterances),
            meeting_id=meeting_id,
            summarizer=_summarize_diar_stage,
        )
            asr_out = trace_stage(
            trace_writer,
            "asr",
            lambda: stage_asr(cfg, paths, meeting_id, diar_out, token_cache),
            meeting_id=meeting_id,
            summarizer=_summarize_asr_stage,
        )
        canon_out = trace_stage(
        trace_writer,
        "canonicalization",
        lambda: stage_normalize_and_canonicalize(cfg, paths, meeting_id, asr_out, ingest_out),
        meeting_id=meeting_id,
        summarizer=_summarize_canonical_stage,
    )
        chunk_out = trace_stage(
        trace_writer,
        "chunking",
        lambda: stage_chunking(cfg, paths, meeting_id, canon_out),
        meeting_id=meeting_id,
        summarizer=_summarize_chunk_stage,
    )
        summary_out = trace_stage(
        trace_writer,
        "summarization",
        lambda: stage_summarize(cfg, paths, meeting_id, canon_out, chunk_out, llama_backend=llama_backend),
        meeting_id=meeting_id,
        summarizer=_summarize_summary_stage,
    )
        extract_out = trace_stage(
        trace_writer,
        "extraction",
        lambda: stage_extract(cfg, paths, meeting_id, chunk_out, summary_out, llama_backend=llama_backend),
        meeting_id=meeting_id,
        summarizer=_summarize_extract_stage,
    )
        summary_out = trace_stage(
        trace_writer,
        "summary_finalize",
        lambda: stage_finalize_summary(cfg, paths, meeting_id, summary_out, extract_out),
        meeting_id=meeting_id,
        summarizer=_summarize_summary_stage,
    )
        eval_out = trace_stage(
        trace_writer,
        "evaluation",
        lambda: stage_evaluate(cfg, paths, meeting_id, asr_out, token_cache, summary_out, extract_out),
        meeting_id=meeting_id,
        summarizer=_summarize_eval_stage,
    )

        finalization_started = time.perf_counter()
        digest_started = time.perf_counter()
        artifact_digest = _dir_digest(paths.meeting_artifacts_dir)
        digest_elapsed = time.perf_counter() - digest_started

        manifest = {
        "meeting_id": meeting_id,
        "pipeline_version": "0.1.0",
        "seed": cfg.pipeline.seed,
        "offline": cfg.runtime.offline,
        "offline_preflight_ok": offline_audit.get("ok", None),
        "config_digest": repro_report["config_digest"],
        "speech_backend": cfg.pipeline.speech_backend.mode,
        "summarization_backend": cfg.pipeline.summarization_backend.mode,
        "extraction_backend": cfg.pipeline.extraction_backend.mode,
        "artifact_digest": artifact_digest,
        "stages": {
            "ingest": _summarize_ingest_stage(ingest_out),
            "vad": _summarize_vad_stage(vad_out),
            "diarization": _summarize_diar_stage(diar_out),
            "asr": _summarize_asr_stage(asr_out),
            "canonicalization": _summarize_canonical_stage(canon_out),
            "chunking": _summarize_chunk_stage(chunk_out),
            "summarization": _summarize_summary_stage(summary_out),
            "extraction": _summarize_extract_stage(extract_out),
            "evaluation": _summarize_eval_stage(eval_out),
        },
    }
        write_started = time.perf_counter()
        if cfg.runtime.include_nondeterministic_timings_in_manifest:
            manifest["finalization_timings_sec"] = {
                "artifact_digest": round(digest_elapsed, 6),
                "manifest_write": round(time.perf_counter() - write_started, 6),
                "total_finalization": round(time.perf_counter() - finalization_started, 6),
            }
        write_json(paths.meeting_artifacts_dir / "run_manifest.json", manifest)
        if mlflow_mod is not None and mlflow_run is not None:
            _mlflow_log_pipeline_result(mlflow_mod, cfg, manifest, eval_out)
            mlflow_mod.end_run(status="FINISHED")
        return manifest
    except Exception:
        if mlflow_mod is not None and mlflow_run is not None:
            try:
                mlflow_mod.end_run(status="FAILED")
            except Exception:
                pass
        raise


def stage_ingest(cfg: AppConfig, paths: PipelinePaths, meeting_id: str) -> dict:
    """Ingest raw audio, stage WAV, and persist QC/provenance artifacts."""
    raw_wav = paths.raw_audio_dir / f"{meeting_id}.Mix-Headset.wav"
    if not raw_wav.exists():
        raise FileNotFoundError(f"Missing raw audio: {raw_wav}")
    staged_audio_dir = ensure_dir(paths.staged_dir / "audio_clean")
    staged_wav = staged_audio_dir / f"{meeting_id}.wav"
    if cfg.runtime.overwrite or not staged_wav.exists():
        shutil.copy2(raw_wav, staged_wav)

    metrics = QCMetrics(meeting_id=meeting_id, **wav_metrics(staged_wav))
    upsert_csv(
        paths.staged_dir / "audio_qc_metrics.csv",
        metrics.model_dump(),
        key="meeting_id",
    )
    upsert_jsonl(
        paths.staged_dir / "provenance.jsonl",
        {
            "meeting_id": meeting_id,
            "source_audio": str(raw_wav),
            "staged_audio": str(staged_wav),
            "normalization": {"target_format": "wav", "channels": 1, "sample_rate": 16000, "status": "copied_if_already_compatible"},
        },
        key="meeting_id",
    )
    return {"staged_wav": str(staged_wav), "qc_metrics": metrics.model_dump()}


def stage_vad(cfg: AppConfig, paths: PipelinePaths, meeting_id: str, ingest_out: dict, utterances: list[dict]) -> dict:
    """Produce VAD artifacts using NeMo backend or deterministic mock path."""
    if cfg.pipeline.speech_backend.mode == "nemo":
        backend = NemoSpeechBackend(cfg)
        return backend.run_vad(
            meeting_id=meeting_id,
            audio_path=Path(ingest_out["staged_wav"]),
            output_dir=paths.meeting_artifacts_dir,
        )
    if utterances:
        segments = [VADSegment(start=u["start"], end=u["end"], label="speech", source="mock_ami_annotations") for u in utterances]
    else:
        dur = float(ingest_out["qc_metrics"]["duration_sec"])
        segments = []
        t = 0.0
        while t < dur:
            end = min(dur, t + 20.0)
            segments.append(VADSegment(start=round(t, 3), end=round(end, 3), source="mock_fixed"))
            t = end + 1.0
    json_data = [s.model_dump() for s in segments]
    write_json(paths.meeting_artifacts_dir / "vad_segments.json", json_data)
    rttm_lines = [
        f"SPEAKER {meeting_id} 1 {s.start:.3f} {(s.end - s.start):.3f} <NA> <NA> speech <NA> <NA>"
        for s in segments
    ]
    write_text(paths.meeting_artifacts_dir / "vad_segments.rttm", "\n".join(rttm_lines) + ("\n" if rttm_lines else ""))
    return {"count": len(segments)}


def stage_diarization(cfg: AppConfig, paths: PipelinePaths, meeting_id: str, vad_out: dict, utterances: list[dict]) -> dict:
    """Produce diarization artifacts using NeMo backend or mock fallback."""
    if cfg.pipeline.speech_backend.mode == "nemo":
        backend = NemoSpeechBackend(cfg)
        return backend.run_diarization(
            meeting_id=meeting_id,
            audio_path=paths.staged_dir / "audio_clean" / f"{meeting_id}.wav",
            output_dir=paths.meeting_artifacts_dir,
        )
    diar_segments: list[DiarizationSegment] = []
    speaker_map: dict[str, str] = {}
    if utterances:
        letters = sorted({u["speaker_letter"] for u in utterances})
        speaker_map = {letter: f"SPEAKER_{i+1}" for i, letter in enumerate(letters)}
        diar_segments = [
            DiarizationSegment(
                start=u["start"],
                end=u["end"],
                speaker=speaker_map[u["speaker_letter"]],
                source="mock_ami_annotations",
            )
            for u in utterances
        ]
    else:
        for i in range(vad_out["count"]):
            diar_segments.append(DiarizationSegment(start=float(i) * 10, end=float(i) * 10 + 8, speaker=f"SPEAKER_{(i % 4) + 1}", source="mock_fixed"))

    write_json(paths.meeting_artifacts_dir / "diarization_segments.json", [d.model_dump() for d in diar_segments])
    rttm_lines = [
        f"SPEAKER {meeting_id} 1 {d.start:.3f} {(d.end - d.start):.3f} <NA> <NA> {d.speaker} <NA> <NA>"
        for d in diar_segments
    ]
    write_text(paths.meeting_artifacts_dir / "diarization.rttm", "\n".join(rttm_lines) + ("\n" if rttm_lines else ""))
    ensure_dir(paths.artifacts_dir / "ami" / "speaker_embeddings_cache")
    return {"count": len(diar_segments), "speaker_map": speaker_map}


def stage_asr(cfg: AppConfig, paths: PipelinePaths, meeting_id: str, diar_out: dict, token_cache: list[dict]) -> dict:
    """Produce ASR segments/transcript artifacts using configured speech backend."""
    if cfg.pipeline.speech_backend.mode == "nemo":
        backend = NemoSpeechBackend(cfg)
        return backend.run_asr(
            meeting_id=meeting_id,
            audio_path=paths.staged_dir / "audio_clean" / f"{meeting_id}.wav",
            output_dir=paths.meeting_artifacts_dir,
            diarization_segments=diar_out.get("segments"),
        )
    diar_segments = _load_json_segments(paths.meeting_artifacts_dir / "diarization_segments.json")
    token_by_time = token_cache
    letters_by_speaker = {v: k for k, v in diar_out.get("speaker_map", {}).items()}

    asr_segments: list[ASRSegment] = []
    for seg in diar_segments:
        speaker = seg["speaker"]
        speaker_letter = letters_by_speaker.get(speaker)
        seg_tokens = [
            t
            for t in token_by_time
            if (speaker_letter is None or t["speaker_letter"] == speaker_letter)
            and t["start"] < seg["end"] + 1e-6
            and t["end"] > seg["start"] - 1e-6
        ]
        text = _tokens_to_text(seg_tokens).strip()
        if not text:
            text = "[inaudible]"
            conf = 0.15
            source = "mock_placeholder"
        else:
            conf = 0.99
            source = "mock_ami_annotations"
        asr_segments.append(
            ASRSegment(
                start=seg["start"],
                end=seg["end"],
                speaker=speaker,
                text=text,
                confidence=conf,
                source=source,
            )
        )

    write_json(paths.meeting_artifacts_dir / "asr_segments.json", [s.model_dump() for s in asr_segments])
    full_lines = [f"[{s.start:.2f}-{s.end:.2f}] {s.speaker}: {s.text}" for s in asr_segments if s.text]
    write_text(paths.meeting_artifacts_dir / "full_transcript.txt", "\n".join(full_lines) + ("\n" if full_lines else ""))
    return {"segments": [s.model_dump() for s in asr_segments]}


def stage_normalize_and_canonicalize(cfg: AppConfig, paths: PipelinePaths, meeting_id: str, asr_out: dict, ingest_out: dict) -> dict:
    """Write raw/normalized transcript views and canonical meeting record."""
    turns: list[TranscriptTurn] = []
    raw_records = []
    for seg in asr_out["segments"]:
        raw_text = seg["text"]
        norm_text = normalize_text(raw_text)
        turn = TranscriptTurn(
            start=seg["start"],
            end=seg["end"],
            speaker=seg["speaker"],
            text_raw=raw_text,
            text_normalized=norm_text,
        )
        turns.append(turn)
        raw_records.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text": raw_text,
                "confidence": seg["confidence"],
            }
        )
    write_json(paths.meeting_artifacts_dir / "transcript_raw.json", raw_records)
    write_json(paths.meeting_artifacts_dir / "transcript_normalized.json", [t.model_dump() for t in turns])

    canonical = CanonicalMeeting(
        meeting_id=meeting_id,
        duration_sec=float(ingest_out["qc_metrics"]["duration_sec"]),
        transcript_turns=turns,
        metadata={
            "language": "en",
            "source_corpus": "AMI",
            "speech_backend": cfg.pipeline.speech_backend.mode,
        },
    )
    upsert_jsonl(
        paths.artifacts_dir / "ami" / "meetings_canonical.jsonl",
        canonical.model_dump(),
        key="meeting_id",
    )
    return {"turn_count": len(turns), "canonical": canonical.model_dump()}


def stage_chunking(cfg: AppConfig, paths: PipelinePaths, meeting_id: str, canon_out: dict) -> dict:
    """Split canonical transcript turns into stable overlapping chunks."""
    turns = canon_out["canonical"]["transcript_turns"]
    target = cfg.pipeline.chunk.target_words
    overlap = cfg.pipeline.chunk.overlap_words

    chunks: list[TranscriptChunk] = []
    start_idx = 0
    chunk_num = 1
    while start_idx < len(turns):
        word_count = 0
        end_idx = start_idx
        while end_idx < len(turns):
            wc = _word_count(turns[end_idx]["text_normalized"])
            if end_idx > start_idx and word_count + wc > target:
                break
            word_count += wc
            end_idx += 1
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        chunk_turns = turns[start_idx:end_idx]
        text = "\n".join(f'{t["speaker"]}: {t["text_normalized"]}' for t in chunk_turns)
        chunk = TranscriptChunk(
            chunk_id=f"{meeting_id}_chunk_{chunk_num:04d}",
            meeting_id=meeting_id,
            turn_indices=list(range(start_idx, end_idx)),
            start=float(chunk_turns[0]["start"]),
            end=float(chunk_turns[-1]["end"]),
            text=text,
        )
        chunks.append(chunk)
        chunk_num += 1

        if end_idx >= len(turns):
            break
        # Walk backward to approximate overlap by words while preserving turn boundaries.
        back_words = 0
        next_start = end_idx
        while next_start > start_idx:
            wc = _word_count(turns[next_start - 1]["text_normalized"])
            if back_words + wc > overlap:
                break
            back_words += wc
            next_start -= 1
        if next_start == start_idx:
            next_start = end_idx
        start_idx = next_start

    rows = [c.model_dump() for c in chunks]
    write_jsonl(paths.meeting_artifacts_dir / "transcript_chunks.jsonl", rows)
    return {"chunks": rows, "count": len(rows)}


def _annotate_chunks_with_asr_confidence(paths: PipelinePaths, chunks: list[dict]) -> list[dict]:
    """Attach overlap-weighted ASR confidence estimates to transcript chunks.

    The LLM backend uses these chunk-level scores only as a ranking hint. If
    segment confidence is unavailable or zeroed by the ASR backend, the chunk
    is left unchanged and normal lexical scoring still applies.
    """
    asr_path = paths.meeting_artifacts_dir / "asr_segments.json"
    if not asr_path.exists():
        return chunks
    try:
        asr_segments = json.loads(asr_path.read_text(encoding="utf-8"))
    except Exception:
        return chunks
    if not isinstance(asr_segments, list) or not asr_segments:
        return chunks

    out: list[dict] = []
    for chunk in chunks:
        c = dict(chunk)
        c_start = float(c.get("start", 0.0))
        c_end = float(c.get("end", c_start))
        weighted = 0.0
        covered = 0.0
        for seg in asr_segments:
            try:
                s0 = float(seg.get("start", 0.0))
                s1 = float(seg.get("end", s0))
                conf = float(seg.get("confidence", 0.0))
            except Exception:
                continue
            overlap = min(c_end, s1) - max(c_start, s0)
            if overlap <= 0:
                continue
            weighted += overlap * max(0.0, min(1.0, conf))
            covered += overlap
        if covered > 0:
            c["asr_confidence"] = round(weighted / covered, 4)
        out.append(c)
    return out


def stage_summarize(
    cfg: AppConfig,
    paths: PipelinePaths,
    meeting_id: str,
    canon_out: dict,
    chunk_out: dict,
    llama_backend=None,
) -> dict:
    """Generate and persist first-pass Minutes of Meeting summary artifacts.

    This stage writes the initial `mom_summary.json` and HTML rendering from
    transcript chunks. A later stage may deterministically enrich the summary
    with extraction-grounded follow-up coverage.
    """
    turns = canon_out["canonical"]["transcript_turns"]
    chunks_for_summary = _annotate_chunks_with_asr_confidence(paths, chunk_out["chunks"])
    if cfg.pipeline.summarization_backend.mode == "llama_cpp":
        backend = llama_backend or LlamaCppBackend(cfg)
        summary = backend.summarize(
            meeting_id=meeting_id,
            turns=turns,
            chunks=chunks_for_summary,
        )
    else:
        speaker_counts: dict[str, int] = {}
        for t in turns:
            speaker_counts[t["speaker"]] = speaker_counts.get(t["speaker"], 0) + 1
        top_speakers = ", ".join(f"{k} ({v} turns)" for k, v in sorted(speaker_counts.items()))
        sample_points = []
        for t in turns[:6]:
            txt = t["text_normalized"].strip()
            if txt and txt != "[inaudible]":
                sample_points.append(txt[:180])
        summary_text = (
            f"Meeting {meeting_id} processed offline. "
            f"{len(turns)} speaker-attributed turns across {chunk_out['count']} chunks. "
            f"Detected speakers: {top_speakers or 'unknown'}."
        )
        summary = MinutesSummary(
            meeting_id=meeting_id,
            summary=summary_text,
            key_points=sample_points[:5],
            discussion_points=[],
            follow_up=[],
            prompt_template_version="mock-v1",
            backend="mock",
        )
    write_json(paths.meeting_artifacts_dir / "mom_summary.json", summary.model_dump())
    html_body = [
        "<html><head><meta charset='utf-8'><title>MoM Summary</title></head><body>",
        f"<h1>Meeting {meeting_id}</h1>",
        "<h2>Summary</h2>",
        f"<p>{_html_escape(summary.summary)}</p>",
    ]
    html_body.append("<h2>Discussion Points</h2><ul>")
    for item in summary.discussion_points:
        html_body.append(f"<li><strong>{_html_escape(item.text)}</strong>")
        if item.evidence_snippets:
            html_body.append("<ul>")
            html_body.extend(f"<li><em>{_html_escape(sn)}</em></li>" for sn in item.evidence_snippets)
            html_body.append("</ul>")
        html_body.append("</li>")
    html_body.append("</ul>")
    html_body.append("<h2>Follow Up</h2><ul>")
    for item in summary.follow_up:
        html_body.append(f"<li><strong>{_html_escape(item.text)}</strong>")
        if item.evidence_snippets:
            html_body.append("<ul>")
            html_body.extend(f"<li><em>{_html_escape(sn)}</em></li>" for sn in item.evidence_snippets)
            html_body.append("</ul>")
        html_body.append("</li>")
    html_body.append("</ul>")
    html_body.append("<h2>Key Points</h2><ul>")
    html_body.extend(f"<li>{_html_escape(pt)}</li>" for pt in summary.key_points)
    html_body.append("</ul></body></html>")
    write_text(paths.meeting_artifacts_dir / "mom_summary.html", "\n".join(html_body))
    return summary.model_dump()


def stage_finalize_summary(
    cfg: AppConfig,
    paths: PipelinePaths,
    meeting_id: str,
    summary_out: dict,
    extract_out: dict,
) -> dict:
    """Finalize the MoM after extraction.

    Responsibilities:
    - merge extracted action items back into `follow_up` when they were not
      captured in the initial summarization pass
    - add a small amount of deterministic decision/action coverage to the
      narrative summary when the first-pass summary omitted it
    - rewrite both `mom_summary.json` and `mom_summary.html` so later stages
      and the UI read a single consistent summary artifact
    """
    summary = MinutesSummary.model_validate(summary_out)
    extraction = ExtractionOutput.model_validate(extract_out)

    follow_up = list(summary.follow_up)
    existing_texts = {item.text.strip().lower() for item in follow_up if item.text.strip()}

    for action in extraction.action_items:
        text = action.action.strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in existing_texts:
            continue
        follow_up.append(
            EvidenceBackedPoint(
                text=text[:240],
                evidence_chunk_ids=action.evidence_chunk_ids[:3],
                evidence_snippets=action.evidence_snippets[:3],
                confidence=max(0.45, float(action.confidence or 0.0)),
            )
        )
        existing_texts.add(lowered)

    summary.follow_up = follow_up[:8]
    summary.summary = _finalize_summary_narrative(summary, extraction)
    write_json(paths.meeting_artifacts_dir / "mom_summary.json", summary.model_dump())
    html_body = [
        "<html><head><meta charset='utf-8'><title>MoM Summary</title></head><body>",
        f"<h1>Meeting {meeting_id}</h1>",
        "<h2>Summary</h2>",
        f"<p>{_html_escape(summary.summary)}</p>",
    ]
    html_body.append("<h2>Discussion Points</h2><ul>")
    for item in summary.discussion_points:
        html_body.append(f"<li><strong>{_html_escape(item.text)}</strong>")
        if item.evidence_snippets:
            html_body.append("<ul>")
            html_body.extend(f"<li><em>{_html_escape(sn)}</em></li>" for sn in item.evidence_snippets)
            html_body.append("</ul>")
        html_body.append("</li>")
    html_body.append("</ul>")
    html_body.append("<h2>Follow Up</h2><ul>")
    for item in summary.follow_up:
        html_body.append(f"<li><strong>{_html_escape(item.text)}</strong>")
        if item.evidence_snippets:
            html_body.append("<ul>")
            html_body.extend(f"<li><em>{_html_escape(sn)}</em></li>" for sn in item.evidence_snippets)
            html_body.append("</ul>")
        html_body.append("</li>")
    html_body.append("</ul>")
    html_body.append("<h2>Key Points</h2><ul>")
    html_body.extend(f"<li>{_html_escape(pt)}</li>" for pt in summary.key_points)
    html_body.append("</ul></body></html>")
    write_text(paths.meeting_artifacts_dir / "mom_summary.html", "\n".join(html_body))
    return summary.model_dump()


def stage_extract(
    cfg: AppConfig,
    paths: PipelinePaths,
    meeting_id: str,
    chunk_out: dict,
    summary_out: dict | None = None,
    llama_backend=None,
) -> dict:
    """Generate and persist structured decisions/action items artifacts."""
    if cfg.pipeline.extraction_backend.mode == "llama_cpp":
        backend = llama_backend or LlamaCppBackend(cfg)
        output = backend.extract(meeting_id=meeting_id, chunks=chunk_out["chunks"], summary=summary_out or {})
    else:
        decisions: list[DecisionItem] = []
        actions: list[ActionItem] = []
        flags: list[str] = []

        decision_patterns = [r"\bwe should\b", r"\blet'?s\b", r"\bdecid(?:e|ed)\b", r"\bagree(?:d)?\b"]
        action_patterns = [r"\bi will\b", r"\bwe need to\b", r"\bcan you\b", r"\byou should\b"]
        due_re = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|next week|tomorrow|\d{1,2}/\d{1,2})\b", re.I)

        for chunk in chunk_out["chunks"]:
            text = chunk["text"]
            lower = text.lower()
            for pat in decision_patterns:
                if re.search(pat, lower):
                    decisions.append(
                        DecisionItem(
                            decision=_first_line_excerpt(text),
                            evidence_chunk_ids=[chunk["chunk_id"]],
                            confidence=0.55,
                            uncertain=True,
                        )
                    )
                    break
            for pat in action_patterns:
                if re.search(pat, lower):
                    owner = _infer_owner_from_chunk(text)
                    due = None
                    m = due_re.search(text)
                    if m:
                        due = m.group(1)
                    actions.append(
                        ActionItem(
                            action=_first_line_excerpt(text),
                            owner=owner,
                            due_date=due,
                            evidence_chunk_ids=[chunk["chunk_id"]],
                            confidence=0.5,
                            uncertain=True,
                        )
                    )
                    break

        if not decisions:
            flags.append("no_decisions_detected_by_mock_rules")
        if not actions:
            flags.append("no_actions_detected_by_mock_rules")

        output = ExtractionOutput(meeting_id=meeting_id, decisions=decisions, action_items=actions, flags=flags)
    validation_report = {
        "meeting_id": meeting_id,
        "schema_valid": True,
        "decision_count": len(output.decisions),
        "action_item_count": len(output.action_items),
        "flags": output.flags,
    }
    write_json(paths.meeting_artifacts_dir / "decisions_actions.json", output.model_dump())
    write_json(paths.meeting_artifacts_dir / "extraction_validation_report.json", validation_report)
    return output.model_dump()


def stage_evaluate(
    cfg: AppConfig,
    paths: PipelinePaths,
    meeting_id: str,
    asr_out: dict,
    token_cache: list[dict],
    summary_out: dict,
    extract_out: dict,
) -> dict:
    """Compute per-meeting evaluation artifacts used by the pipeline UI.

    The main pipeline evaluation stage computes:
    - `WER`, `CER`, `cpWER`, and approximate `DER`
    - `ROUGE-1/2/L` against AMI abstractive `abstract` references when present
    - structural MoM quality checks
    """
    hyp = " ".join(seg["text"] for seg in asr_out["segments"])
    ref = reference_plain_text(token_cache) if token_cache else ""
    wer = _wer(ref, hyp) if ref else None
    cer = _cer(ref, hyp) if ref else None
    cpwer = None
    der = None
    cpwer_details: dict[str, object] = {}
    der_details: dict[str, object] = {}
    speech_metric_errors: list[str] = []

    try:
        ref_asr = load_reference_asr_views(paths.annotations_dir, meeting_id)
        hyp_asr = load_hyp_asr_views(paths.artifacts_dir, meeting_id)
        cpwer_details = compute_cpwer(ref_asr.get("speaker_texts", {}), hyp_asr.get("speaker_texts", {}))
        cpwer = cpwer_details.get("cpwer")
    except Exception as exc:
        speech_metric_errors.append(f"cpwer:{type(exc).__name__}")

    try:
        ref_diar = load_ref_diarization_from_words(paths.annotations_dir, meeting_id)
        hyp_diar = load_hyp_diarization(paths.artifacts_dir, meeting_id)
        der_details = compute_der_approx_nooverlap(ref_diar, hyp_diar, collar_sec=0.25, skip_overlap=True)
        der = der_details.get("der")
    except Exception as exc:
        speech_metric_errors.append(f"der:{type(exc).__name__}")

    summary_ref = load_abstractive_summary_text(paths.annotations_dir, meeting_id, section="abstract")
    summary_hyp = str(summary_out.get("summary", "") or "").strip()
    rouge_scores = _rouge_scores(summary_ref, summary_hyp) if summary_ref else None

    wer_row = {
        "meeting_id": meeting_id,
        "wer": "" if wer is None else f"{wer:.6f}",
        "cer": "" if cer is None else f"{cer:.6f}",
        "cpwer": "" if cpwer is None else f"{float(cpwer):.6f}",
        "der": "" if der is None else f"{float(der):.6f}",
        "speech_backend": cfg.pipeline.speech_backend.mode,
    }
    upsert_csv(paths.eval_dir / "wer_scores.csv", wer_row, key="meeting_id")
    speech_metrics_row = {
        "meeting_id": meeting_id,
        "wer": wer_row["wer"],
        "cer": wer_row["cer"],
        "cpwer": wer_row["cpwer"],
        "der": wer_row["der"],
        "der_false_alarm_sec": "" if der_details.get("false_alarm_sec") is None else f"{float(der_details['false_alarm_sec']):.6f}",
        "der_miss_sec": "" if der_details.get("miss_sec") is None else f"{float(der_details['miss_sec']):.6f}",
        "der_confusion_sec": "" if der_details.get("confusion_sec") is None else f"{float(der_details['confusion_sec']):.6f}",
        "der_scored_ref_time_sec": "" if der_details.get("scored_ref_time_sec") is None else f"{float(der_details['scored_ref_time_sec']):.6f}",
        "der_ignored_ref_overlap_time_sec": "" if der_details.get("ignored_ref_overlap_time_sec") is None else f"{float(der_details['ignored_ref_overlap_time_sec']):.6f}",
        "der_method": str(der_details.get("method", "")) if der_details else "",
        "der_collar_sec": "" if der_details.get("collar_sec") is None else f"{float(der_details['collar_sec']):.2f}",
        "cpwer_total_ref_words": "" if cpwer_details.get("total_ref_words") is None else str(cpwer_details["total_ref_words"]),
        "cpwer_total_edits": "" if cpwer_details.get("total_edits") is None else str(cpwer_details["total_edits"]),
        "speech_metric_errors": ";".join(speech_metric_errors),
    }
    upsert_csv(paths.eval_dir / "speech_metrics.csv", speech_metrics_row, key="meeting_id")
    write_json(
        paths.eval_dir / "wer_breakdown.json",
        {
            "meeting_id": meeting_id,
            "wer": wer,
            "cer": cer,
            "cpwer": cpwer,
            "der": der,
            "cpwer_details": cpwer_details,
            "der_details": der_details,
            "speech_metric_errors": speech_metric_errors,
            "reference_available": bool(ref),
            "hypothesis_segment_count": len(asr_out["segments"]),
        },
    )

    rouge_row = {
        "meeting_id": meeting_id,
        "rouge1": "" if not rouge_scores or rouge_scores["rouge1"] is None else f"{rouge_scores['rouge1']:.6f}",
        "rouge2": "" if not rouge_scores or rouge_scores["rouge2"] is None else f"{rouge_scores['rouge2']:.6f}",
        "rougeL": "" if not rouge_scores or rouge_scores["rougeL"] is None else f"{rouge_scores['rougeL']:.6f}",
    }
    upsert_csv(paths.eval_dir / "rouge_scores.csv", rouge_row, key="meeting_id")
    mom_checks = {
        "meeting_id": meeting_id,
        "summary_nonempty": bool(summary_out.get("summary")),
        "key_points_count": len(summary_out.get("key_points", [])),
        "schema_valid": True,
        "extraction_decision_count": len(extract_out.get("decisions", [])),
        "extraction_action_count": len(extract_out.get("action_items", [])),
    }
    write_json(paths.eval_dir / "mom_quality_checks.json", mom_checks)
    return {
        "wer": wer,
        "cer": cer,
        "cpwer": cpwer,
        "der": der,
        **(rouge_scores or {}),
        "mom_quality_checks": mom_checks,
    }


def normalize_text(text: str) -> str:
    """Apply deterministic lightweight normalization for transcript text."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\buh\b|\bum\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_json_segments(path: Path) -> list[dict]:
    """Load a list-like JSON artifact from disk."""
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _tokens_to_text(tokens: list[dict]) -> str:
    """Reconstruct plain text from token records with punctuation markers."""
    parts: list[str] = []
    for t in tokens:
        if t["is_punc"] and parts:
            parts[-1] += t["text"]
        else:
            parts.append(t["text"])
    return " ".join(parts)


def _word_count(text: str) -> int:
    """Count non-empty whitespace-delimited words in a text string."""
    return len([w for w in text.split() if w])


def _first_line_excerpt(text: str, max_len: int = 220) -> str:
    """Return a truncated excerpt from the first line of text."""
    line = text.splitlines()[0].strip() if text else ""
    return line[:max_len]


def _infer_owner_from_chunk(text: str) -> str | None:
    """Infer the leading speaker label from a transcript chunk when present."""
    m = re.search(r"^(SPEAKER_\d+):", text, flags=re.M)
    return m.group(1) if m else None


def _html_escape(s: str) -> str:
    """Escape a small HTML subset for generated summary pages."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _finalize_summary_narrative(summary: MinutesSummary, extraction: ExtractionOutput) -> str:
    """Add deterministic extraction coverage to a summary paragraph.

    The goal is recall, not style. We only append short fallback sentences
    when the first-pass summary omitted the leading decision or action item.
    """
    text = re.sub(r"\s+", " ", str(summary.summary or "")).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences and text:
        sentences = [text]

    additions: list[str] = []
    if extraction.decisions:
        decision = extraction.decisions[0].decision.strip().rstrip(".")
        if decision and not _summary_contains_phrase(text, decision):
            additions.append(f"The group made decisions on {decision[0].lower() + decision[1:]}.")
    if extraction.action_items:
        action = extraction.action_items[0].action.strip().rstrip(".")
        if action and not _summary_contains_phrase(text, action):
            if action.lower().startswith(("prepare ", "review ", "compare ", "evaluate ", "define ", "confirm ", "plan ")):
                additions.append(f"Follow-up work included plans to {action.lower()}.")
            else:
                additions.append(f"Follow-up work included {action[0].lower() + action[1:]}.")

    if not additions:
        return text[:600]

    finalized = " ".join((sentences + additions)[:4]).strip()
    return finalized[:600]


def _summary_contains_phrase(summary_text: str, phrase: str) -> bool:
    """Check whether two texts share enough normalized lexical overlap."""
    summary_words = set(_normalize_for_eval(summary_text).split())
    phrase_words = set(_normalize_for_eval(phrase).split())
    if not phrase_words:
        return False
    overlap = len(summary_words & phrase_words)
    return overlap >= max(2, min(len(phrase_words), 4))


def _normalize_for_eval(s: str) -> str:
    """Normalize text into the lightweight evaluation token space."""
    s = s.lower()
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = re.sub(r"speaker_\d+:", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _wer(ref: str, hyp: str) -> float:
    """Compute word error rate over normalized text."""
    ref_words = _normalize_for_eval(ref).split()
    hyp_words = _normalize_for_eval(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _cer(ref: str, hyp: str) -> float:
    """Compute character error rate over normalized text."""
    ref_chars = list(_normalize_for_eval(ref).replace(" ", ""))
    hyp_chars = list(_normalize_for_eval(hyp).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def _rouge_scores(ref: str, hyp: str) -> dict[str, float | None]:
    """Compute lightweight ROUGE-F1 metrics over normalized token streams.

    This intentionally avoids heavyweight external dependencies so the offline
    pipeline can score summaries directly from local AMI references.
    """
    ref_tokens = _normalize_for_eval(ref).split()
    hyp_tokens = _normalize_for_eval(hyp).split()
    if not ref_tokens:
        return {"rouge1": None, "rouge2": None, "rougeL": None}
    return {
        "rouge1": _rouge_n_f1(ref_tokens, hyp_tokens, n=1),
        "rouge2": _rouge_n_f1(ref_tokens, hyp_tokens, n=2),
        "rougeL": _rouge_l_f1(ref_tokens, hyp_tokens),
    }


def _rouge_n_f1(ref_tokens: list[str], hyp_tokens: list[str], n: int) -> float:
    """Compute ROUGE-N F1 for normalized token sequences."""
    if len(ref_tokens) < n or len(hyp_tokens) < n:
        return 0.0
    ref_counts = _ngram_counts(ref_tokens, n)
    hyp_counts = _ngram_counts(hyp_tokens, n)
    overlap = sum(min(count, hyp_counts.get(gram, 0)) for gram, count in ref_counts.items())
    ref_total = sum(ref_counts.values())
    hyp_total = sum(hyp_counts.values())
    if ref_total == 0 or hyp_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / hyp_total
    recall = overlap / ref_total
    return 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)


def _ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    """Count n-grams in a token sequence."""
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _rouge_l_f1(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    """Compute ROUGE-L F1 from the longest common subsequence length."""
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_len(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Return the longest common subsequence length for two token lists."""
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _dir_digest(path: Path) -> str:
    """Hash persisted meeting artifacts while excluding traceability sidecars."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_dir():
            continue
        if p.name in {
            "run_manifest.json",
            "stage_trace.jsonl",
            "preflight_offline_audit.json",
            "reproducibility_report.json",
        }:
            continue
        rel = p.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _summarize_ingest_stage(out: dict) -> dict:
    """Project ingest outputs into a compact stage-trace summary."""
    qc = out.get("qc_metrics", {}) if isinstance(out, dict) else {}
    return {
        "staged_wav": out.get("staged_wav"),
        "duration_sec": qc.get("duration_sec"),
        "sample_rate_hz": qc.get("sample_rate_hz"),
        "channels": qc.get("channels"),
    }


def _summarize_vad_stage(out: dict) -> dict:
    """Project VAD outputs into a compact stage-trace summary."""
    return {"count": out.get("count")}


def _summarize_diar_stage(out: dict) -> dict:
    """Project diarization outputs into a compact stage-trace summary."""
    segments = out.get("segments") if isinstance(out, dict) else None
    labels = out.get("speaker_labels") if isinstance(out, dict) else None
    if labels is None and isinstance(segments, list):
        labels = sorted({str(s.get("speaker")) for s in segments if isinstance(s, dict) and s.get("speaker")})
    return {
        "count": out.get("count"),
        "speaker_labels": labels or [],
        "speaker_count": len(labels or []),
        "speaker_map": out.get("speaker_map", {}),
    }


def _summarize_asr_stage(out: dict) -> dict:
    """Project ASR outputs into a compact stage-trace summary."""
    segments = out.get("segments") if isinstance(out, dict) else None
    return {
        "segment_count": len(segments) if isinstance(segments, list) else None,
    }


def _summarize_canonical_stage(out: dict) -> dict:
    """Project canonicalization outputs into a compact stage-trace summary."""
    canonical = out.get("canonical", {}) if isinstance(out, dict) else {}
    metadata = canonical.get("metadata", {}) if isinstance(canonical, dict) else {}
    return {
        "turn_count": out.get("turn_count"),
        "duration_sec": canonical.get("duration_sec"),
        "metadata": {
            "language": metadata.get("language"),
            "source_corpus": metadata.get("source_corpus"),
            "speech_backend": metadata.get("speech_backend"),
        },
    }


def _summarize_chunk_stage(out: dict) -> dict:
    """Project chunking outputs into a compact stage-trace summary."""
    return {"count": out.get("count")}


def _summarize_summary_stage(out: dict) -> dict:
    """Project summarization outputs into a compact stage-trace summary."""
    if not isinstance(out, dict):
        return {}
    return {
        "meeting_id": out.get("meeting_id"),
        "backend": out.get("backend"),
        "prompt_template_version": out.get("prompt_template_version"),
        "key_points_count": len(out.get("key_points", []) or []),
        "summary_nonempty": bool(out.get("summary")),
    }


def _summarize_extract_stage(out: dict) -> dict:
    """Project extraction outputs into a compact stage-trace summary."""
    if not isinstance(out, dict):
        return {}
    return {
        "meeting_id": out.get("meeting_id"),
        "decision_count": len(out.get("decisions", []) or []),
        "action_item_count": len(out.get("action_items", []) or []),
        "flag_count": len(out.get("flags", []) or []),
    }


def _summarize_eval_stage(out: dict) -> dict:
    """Project evaluation outputs into a compact stage-trace summary."""
    if not isinstance(out, dict):
        return {}
    mom = out.get("mom_quality_checks", {}) or {}
    return {
        "wer": out.get("wer"),
        "cer": out.get("cer"),
        "mom_quality_checks": {
            "summary_nonempty": mom.get("summary_nonempty"),
            "key_points_count": mom.get("key_points_count"),
            "schema_valid": mom.get("schema_valid"),
            "extraction_decision_count": mom.get("extraction_decision_count"),
            "extraction_action_count": mom.get("extraction_action_count"),
        },
    }


def _mlflow_log_pipeline_result(mlflow, cfg: AppConfig, manifest: dict, eval_out: dict) -> None:
    """Best-effort logging of meeting-level metrics to local MLflow."""
    try:
        mlflow.log_param("config_digest", manifest.get("config_digest"))
        mlflow.log_param("artifact_digest", manifest.get("artifact_digest"))
        mlflow.log_param("offline_preflight_ok", manifest.get("offline_preflight_ok"))
        stages = manifest.get("stages", {}) or {}
        mlflow.log_metrics(
            {
                "asr_segment_count": float(((stages.get("asr") or {}).get("segment_count") or 0)),
                "chunk_count": float(((stages.get("chunking") or {}).get("count") or 0)),
                "summary_key_points_count": float(((stages.get("summarization") or {}).get("key_points_count") or 0)),
                "extraction_decision_count": float(((stages.get("extraction") or {}).get("decision_count") or 0)),
                "extraction_action_count": float(((stages.get("extraction") or {}).get("action_item_count") or 0)),
            }
        )
        if isinstance(eval_out, dict):
            if eval_out.get("wer") is not None:
                mlflow.log_metric("wer", float(eval_out["wer"]))
            if eval_out.get("cer") is not None:
                mlflow.log_metric("cer", float(eval_out["cer"]))
        run_manifest = Path(cfg.paths.artifacts_dir) / "ami" / str(manifest.get("meeting_id")) / "run_manifest.json"
        if run_manifest.exists():
            mlflow.log_artifact(str(run_manifest), artifact_path="meeting_artifacts")
    except Exception:
        # Keep pipeline execution resilient if local MLflow logging has issues.
        return
