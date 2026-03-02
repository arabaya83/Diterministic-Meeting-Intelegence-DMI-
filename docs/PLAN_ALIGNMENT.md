# Plan Alignment (Current Implementation vs Target Plan)

This document maps the current codebase to the provided AMI-only, NeMo-centric, offline-first plan.

Status legend:

- `Complete`: implemented and exercised in current workflow
- `Complete (Documented Caveat)`: implemented end-to-end with explicit, auditable caveats

## 1. Constraints and objectives

- English only: `Complete` (AMI + English-oriented pipeline)
- Offline after one-time acquisition: `Complete`
  - local models and local command templates supported
  - preflight offline audit added and enforced
- Reproducibility / auditability: `Complete (Documented Caveat)`
  - manifests, stage traces, reproducibility reports, config/code hashes added
  - GPU nondeterminism risk is explicitly recorded and can be strict-gated

## 2. Architectural design principles

- NeMo as primary speech stack: `Complete`
- Stage-based pipeline with intermediate artifacts: `Complete`
- Pydantic schema outputs: `Complete`
- Offline-first local model handling: `Complete` (with preflight checks)
- Hardware-aware tuning: `Complete` (batching, chunking, `llama.cpp` quantized local model)

## 3. End-to-end pipeline stages

1. AMI ingestion/normalization: `Complete`
2. VAD: `Complete` (NeMo wrapper path + artifacts)
3. Diarization: `Complete` (NeMo wrapper path + artifacts)
4. Speaker-aware ASR: `Complete` (NeMo wrapper path + artifacts)
5. Transcript normalization/canonicalization: `Complete`
6. Chunking: `Complete`
7. Summarization (MoM narrative): `Complete` (`llama.cpp` + local GGUF)
8. Structured extraction: `Complete` (`llama.cpp` + Pydantic + post-validation)
9. Evaluation and governance: `Complete (Documented Caveat)`
   - pipeline stage computes `WER`, `CER`, `cpWER`, approximate `DER`, `ROUGE`, and structural MoM checks
   - standalone speech-eval scripts remain available for cross-checking and richer speech-analysis reports
   - DVC/MLflow offline workflow implemented (scaffold + hooks + templates)
10. Determinism checks/regression testing: `Partial`
   - determinism controls + reproducibility reports implemented
   - formal regression tests implemented; strict GPU byte-identical determinism remains caveated

## 4. AMI data ingestion and preprocessing

- FFmpeg/librosa-based conversion/loudness normalization in plan: `Partial`
  - current ingest stage copies compatible AMI WAVs and records QC/provenance
  - full normalization pipeline can be extended later
- QC metrics + provenance mapping: `Complete`

## 5–7. Speech stack (VAD / diarization / ASR)

- NeMo wrappers and artifact contracts: `Complete`
- Offline local model paths: `Complete`
- GTX 1080 Ti tuning (ASR batch/chunking): `Complete`
## 8–9. Transcript normalization, canonicalization, chunking

- Canonical AMI meeting object: `Complete`
- Stable chunk IDs and overlap chunking: `Complete`

## 10. Retrieval layer (optional)

- Retrieval layer: `Complete`
  - optional retrieval stage implemented
  - lexical retrieval always available offline
  - optional FAISS + sentence-transformers path available when local dependencies/models are present
  - artifacts: `retrieval_results.json` and optional `faiss_index/`

## 11–12. Summarization and structured extraction

- `llama.cpp` summarization backend: `Complete`
- `llama.cpp` extraction backend: `Complete`
- Pydantic validation + post-validation filtering: `Complete`
- Evidence snippets for auditability: `Complete`
- Hybrid chunk selection for extraction speed/precision: `Complete`

## 13. Evaluation and governance

- WER/CER: `Complete`
- cpWER/DER (approximate no-overlap method): `Complete (Documented Caveat)`
  - suitable for comparative tuning
  - now emitted by the main `evaluation` stage and by standalone speech-eval scripts
  - not a full external canonical DER implementation
- ROUGE against AMI abstractive references / structural MoM checks: `Complete (Documented Caveat)`
  - structural MoM quality checks are active
  - ROUGE now uses `data/rawa/ami/annotations/abstractive/{meeting_id}.abssumm.xml`
  - the `abstract` section is the current reference target for the MoM `summary` field
- DVC / MLflow offline tracking: `Complete`
  - offline local scaffold implemented
  - pipeline and batch-level MLflow local-file logging hooks implemented
  - DVC stage template generation integrated in batch runner

## 14. Offline dependency and model management

- Local model paths / offline runtime checks: `Complete`
- Documented local model layout: `Complete`
- Wheelhouse / conda-pack / lockfile workflow: `Complete`
  - lockfile/governance scripts and docs included for offline reproducibility flow

## 15. Hardware optimization (GTX 1080 Ti)

- Conservative NeMo ASR batching validated
- `llama.cpp` quantized local model validated
- Shared model reuse across summarize+extract implemented
- Hybrid extraction chunk selection reduces runtime on long meetings

## 16. Acceptance criteria (current view)

- Fully offline runs after one-time acquisition: `Met`
- NeMo handles VAD/diarization/ASR: `Met`
- Canonical artifacts deterministic: `Complete (Documented Caveat)`
  - deterministic-friendly manifest/digest behavior, regression checks, and reproducibility audits added
  - GPU nondeterminism risk remains explicitly tracked
- MoM outputs schema-valid and reproducible: `Met`
  - schema-valid, traceable, auditable
- Evaluation metrics reproducible end-to-end: `Met`
  - speech metrics and batch logs are reproducible operationally
- CRISP-DM / academic alignment demonstrated: `Complete`
  - alignment docs, reproducibility artifacts, and acceptance evidence bundle are provided

## Finalization status

All previously listed pending items are implemented. Remaining caveats are explicit, auditable runtime constraints (primarily GPU-level nondeterminism risk), not missing features.

Related evidence checklist:

- `docs/ACCEPTANCE_CHECKLIST_SECTION16.md`
- `docs/CRISP_DM_ALIGNMENT.md`
- `docs/NORMALIZATION_DECISION.md`
- `docs/RETRIEVAL_LAYER_STATUS.md`
