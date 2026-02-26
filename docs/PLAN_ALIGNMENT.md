# Plan Alignment (Current Implementation vs Target Plan)

This document maps the current codebase to the provided AMI-only, NeMo-centric, offline-first plan.

Status legend:

- `Complete`: implemented and exercised in current workflow
- `Partial`: implemented with known limitations / approximate evaluation / stubs in some subparts
- `Pending`: not yet implemented (or still mock)

## 1. Constraints and objectives

- English only: `Complete` (AMI + English-oriented pipeline)
- Offline after one-time acquisition: `Partial`
  - local models and local command templates supported
  - preflight offline audit added
  - operational discipline still required for one-time downloads
- Reproducibility / auditability: `Partial`
  - manifests, stage traces, reproducibility reports, config/code hashes added
  - GPU nondeterminism risk remains for some kernels

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
9. Evaluation and governance: `Partial`
   - `WER`, `cpWER`, approximate `DER`, confidence QA implemented
   - DVC/MLflow integration still pending
10. Determinism checks/regression testing: `Partial`
   - determinism controls + reproducibility reports implemented
   - formal regression test suite and byte-identical assertions pending

## 4. AMI data ingestion and preprocessing

- FFmpeg/librosa-based conversion/loudness normalization in plan: `Partial`
  - current ingest stage copies compatible AMI WAVs and records QC/provenance
  - full normalization pipeline can be extended later
- QC metrics + provenance mapping: `Complete`

## 5–7. Speech stack (VAD / diarization / ASR)

- NeMo wrappers and artifact contracts: `Complete`
- Offline local model paths: `Complete`
- GTX 1080 Ti tuning (ASR batch/chunking): `Complete`
- ASR confidence extraction: `Complete`

## 8–9. Transcript normalization, canonicalization, chunking

- Canonical AMI meeting object: `Complete`
- Stable chunk IDs and overlap chunking: `Complete`

## 10. Retrieval layer (optional)

- FAISS / sentence-transformers retrieval: `Pending`

## 11–12. Summarization and structured extraction

- `llama.cpp` summarization backend: `Complete`
- `llama.cpp` extraction backend: `Complete`
- Pydantic validation + post-validation filtering: `Complete`
- Evidence snippets for auditability: `Complete`
- Hybrid chunk selection for extraction speed/precision: `Complete`

## 13. Evaluation and governance

- WER/CER: `Complete`
- cpWER/DER (approximate no-overlap method): `Partial`
  - suitable for comparative tuning
  - not a full external canonical DER implementation
- ROUGE placeholders / structural MoM checks: `Partial`
- DVC / MLflow offline tracking: `Partial`
  - offline local scaffold (paths/scripts/docs) implemented
  - pipeline auto-logging and full experiment wiring still pending

## 14. Offline dependency and model management

- Local model paths / offline runtime checks: `Complete`
- Documented local model layout: `Complete`
- Wheelhouse / conda-pack automation: `Pending` (documented but not automated)

## 15. Hardware optimization (GTX 1080 Ti)

- Conservative NeMo ASR batching validated
- `llama.cpp` quantized local model validated
- Shared model reuse across summarize+extract implemented
- Hybrid extraction chunk selection reduces runtime on long meetings

## 16. Acceptance criteria (current view)

- Fully offline runs after one-time acquisition: `Mostly met` (with preflight audit)
- NeMo handles VAD/diarization/ASR: `Met`
- Canonical artifacts deterministic: `Partially met`
  - deterministic-friendly manifest/digest behavior added
  - GPU nondeterminism risk remains
- MoM outputs schema-valid and reproducible: `Mostly met`
  - schema-valid, traceable, auditable
  - quality still requires iterative tuning
- Evaluation metrics reproducible end-to-end: `Mostly met`
  - speech metrics and batch logs are reproducible operationally
- CRISP-DM / academic alignment demonstrated: `Partial`
  - technical traceability/reproducibility evidence now improved
  - formal methodology documentation can be expanded further

## Recommended next upgrades (to close remaining gaps)

1. Add regression tests for artifact contract + parser robustness
2. Add deterministic regression checks over a fixed AMI subset
3. Add pipeline-level DVC/MLflow logging integration on top of the offline scaffold
4. Add optional retrieval layer for extraction/summarization grounding
5. Improve MoM extraction recall while preserving precision

Related evidence checklist:

- `docs/ACCEPTANCE_CHECKLIST_SECTION16.md`
