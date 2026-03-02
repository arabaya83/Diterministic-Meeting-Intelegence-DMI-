# Pipeline Overview

This document describes the end-to-end AMI processing flow implemented in
`src/ami_mom_pipeline/pipeline.py`.

## Execution model

- Single-meeting orchestrator: `run_pipeline(cfg, meeting_id)`
- Batch orchestration: `scripts/run_nemo_batch_sequential.py`
- Offline-first by default (local model paths, preflight offline audit)
- Determinism controls enabled through config/runtime flags

## Stage flow

1. `load_annotations`
2. `build_utterances`
3. `ingest`
4. `diarization` / `vad` / `asr` (order varies by speech backend mode)
5. `canonicalization`
6. `chunking`
7. `retrieval` (optional)
8. `summarization`
9. `extraction`
10. `summary_finalize`
11. `evaluation`
12. finalization (`run_manifest.json`, digests)

Each stage emits trace events in `stage_trace.jsonl`.

## Stage details

### Ingest

Inputs:
- `data/rawa/ami/audio/{meeting_id}.Mix-Headset.wav`

Outputs:
- `data/staged/ami/audio_clean/{meeting_id}.wav`
- `data/staged/ami/audio_qc_metrics.csv`
- `data/staged/ami/provenance.jsonl`

### VAD

Backend:
- NeMo wrapper (`NemoSpeechBackend.run_vad`) or mock fallback

Outputs:
- `artifacts/ami/{meeting_id}/vad_segments.json`
- `artifacts/ami/{meeting_id}/vad_segments.rttm`

### Diarization

Backend:
- NeMo wrapper (`NemoSpeechBackend.run_diarization`) or mock fallback

Outputs:
- `artifacts/ami/{meeting_id}/diarization_segments.json`
- `artifacts/ami/{meeting_id}/diarization.rttm`
- `artifacts/ami/speaker_embeddings_cache/` (ensured)

### ASR

Backend:
- NeMo wrapper (`NemoSpeechBackend.run_asr`) or mock fallback

Outputs:
- `artifacts/ami/{meeting_id}/asr_segments.json`
- `artifacts/ami/{meeting_id}/full_transcript.txt`

### Canonicalization

Responsibilities:
- preserve raw transcript view
- apply configured normalization (`rule` or `spacy`)
- write canonical meeting object

Outputs:
- `artifacts/ami/{meeting_id}/transcript_raw.json`
- `artifacts/ami/{meeting_id}/transcript_normalized.json`
- `artifacts/ami/meetings_canonical.jsonl`

### Chunking

Responsibilities:
- produce stable chunk IDs (`{meeting_id}_chunk_####`)
- preserve speaker boundaries using turn indices
- deterministic overlap by word-count target

Outputs:
- `artifacts/ami/{meeting_id}/transcript_chunks.jsonl`

### Retrieval (optional)

Config:
- `pipeline.retrieval.enabled`

Modes:
- lexical retrieval (always available, offline-safe)
- optional FAISS + sentence-transformers (`use_faiss: true`)

Outputs:
- `artifacts/ami/{meeting_id}/retrieval_results.json`
- `artifacts/ami/{meeting_id}/faiss_index/` (optional)

### Summarization

Backend:
- mock or local `llama.cpp` backend

Outputs:
- `artifacts/ami/{meeting_id}/mom_summary.json`
- `artifacts/ami/{meeting_id}/mom_summary.html`

Notes:
- the backend now requests internal `abstract`, `actions`, `decisions`, and `problems` buckets
- those buckets are used to compose a more AMI-reference-aligned narrative summary

### Extraction

Backend:
- mock or local `llama.cpp` backend

Outputs:
- `artifacts/ami/{meeting_id}/decisions_actions.json`
- `artifacts/ami/{meeting_id}/extraction_validation_report.json`

### Summary finalize

Responsibilities:
- merge extracted action items back into `follow_up`
- add deterministic narrative coverage when the initial summary omitted a top decision or follow-up
- rewrite the persisted `mom_summary.json` and `mom_summary.html`

### Evaluation

Outputs:
- `artifacts/eval/ami/wer_scores.csv`
- `artifacts/eval/ami/speech_metrics.csv`
- `artifacts/eval/ami/wer_breakdown.json`
- `artifacts/eval/ami/rouge_scores.csv`
- `artifacts/eval/ami/mom_quality_checks.json`

Notes:
- pipeline evaluation currently computes `WER`, `CER`, `cpWER`, approximate `DER`, `ROUGE-1/2/L`, and structural MoM checks
- ROUGE uses the AMI abstractive `abstract` reference when `*.abssumm.xml` is present
- `speech_metrics.csv` stores the expanded speech metrics row, including DER subcomponents

## Determinism and reproducibility

Per-meeting reproducibility artifacts:
- `preflight_offline_audit.json`
- `reproducibility_report.json`
- `stage_trace.jsonl`
- `run_manifest.json`

Notes:
- digest generation excludes audit/trace files to keep artifact digests stable
- strict byte-identical behavior on GPU remains runtime-dependent and is tracked
