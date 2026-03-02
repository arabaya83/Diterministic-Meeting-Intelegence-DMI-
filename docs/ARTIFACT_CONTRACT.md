# Artifact Contract

This document defines artifact paths and high-level schema contracts used by
the AMI pipeline.

## Per-meeting artifacts

Base directory:
- `artifacts/ami/{meeting_id}/`

### Speech artifacts

- `vad_segments.json`
  - list of `VADSegment`
  - required fields: `start`, `end`, `label`, `source`
- `vad_segments.rttm`
  - RTTM view of VAD segments
- `diarization_segments.json`
  - list of `DiarizationSegment`
  - required fields: `start`, `end`, `speaker`, `source`
- `diarization.rttm`
  - RTTM view of diarization segments
- `asr_segments.json`
  - list of `ASRSegment`
  - required fields: `start`, `end`, `speaker`, `text`, `source`
- `full_transcript.txt`
  - line-oriented `[start-end] SPEAKER_X: text`

### Transcript + canonical artifacts

- `transcript_raw.json`
  - segment-level rows with raw ASR text
- `transcript_normalized.json`
  - list of `TranscriptTurn` rows
- `transcript_chunks.jsonl`
  - one `TranscriptChunk` per line
  - invariant: stable `chunk_id` format `{meeting_id}_chunk_####`
- `retrieval_results.json` (when retrieval enabled)
  - required fields: `meeting_id`, `enabled`, `backend`, `top_k`, `results`
  - optional field: `faiss_index`
- `faiss_index/` (optional, FAISS mode only)
  - `index.faiss`, `index_meta.json`

### MoM artifacts

- `mom_summary.json`
  - `MinutesSummary` schema
  - persisted after `summary_finalize`, so `follow_up` may include extraction-grounded action items
- `mom_summary.html`
  - human-readable rendering of summary sections
- `decisions_actions.json`
  - `ExtractionOutput` schema
- `extraction_validation_report.json`
  - required fields: `meeting_id`, `schema_valid`, `decision_count`, `action_item_count`, `flags`

### Traceability / reproducibility artifacts

- `preflight_offline_audit.json`
- `reproducibility_report.json`
- `stage_trace.jsonl`
- `run_manifest.json`

`run_manifest.json` invariants:
- contains compact stage summaries
- includes `artifact_digest`
- excludes nondeterministic trace/audit files from digest calculation

## Aggregate artifacts

- `artifacts/ami/meetings_canonical.jsonl`
- `artifacts/eval/ami/wer_scores.csv`
- `artifacts/eval/ami/speech_metrics.csv`
- `artifacts/eval/ami/wer_breakdown.json`
- `artifacts/eval/ami/rouge_scores.csv`
- `artifacts/eval/ami/mom_quality_checks.json`

Evaluation notes:
- `rouge_scores.csv` is populated from AMI abstractive references under `data/rawa/ami/annotations/abstractive/`
- `wer_scores.csv` now includes the headline per-meeting speech metrics used by the UI: `wer`, `cer`, `cpwer`, `der`
- `speech_metrics.csv` stores the expanded speech-evaluation row, including DER subcomponents and evaluation-method metadata

## Staged data artifacts

- `data/staged/ami/audio_clean/{meeting_id}.wav`
- `data/staged/ami/audio_qc_metrics.csv`
- `data/staged/ami/provenance.jsonl`

## Determinism and ordering invariants

- JSON serialization uses sorted keys.
- JSONL/CSV upsert aggregates are sorted by primary key.
- Chunk IDs are deterministic for identical canonical turn ordering.
- Precomputed artifact reuse is config-controlled and explicit.
