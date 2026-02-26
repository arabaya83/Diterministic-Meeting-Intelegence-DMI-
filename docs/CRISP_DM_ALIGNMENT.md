# CRISP-DM Alignment (AMI-Only, NeMo-Centric Offline Pipeline)

This document maps the implemented system to CRISP-DM phases and points to concrete evidence artifacts, code, and commands.

## 1. Business Understanding

Objective (implemented target):

- Convert AMI meeting audio into:
  - speaker-attributed transcripts
  - Minutes of Meeting (MoM) outputs
  - reproducible evaluation artifacts

Constraints captured and enforced:

- English-only AMI workflow
- offline-first execution after one-time acquisition
- GTX 1080 Ti / 32GB RAM hardware-aware configuration
- reproducibility / auditability focus

Evidence:

- `README.md`
- `docs/PLAN_ALIGNMENT.md`
- `docs/ACCEPTANCE_CHECKLIST_SECTION16.md`
- `configs/pipeline.nemo.llama.yaml`
- `configs/pipeline.nemo.llama.strict_offline.yaml`

## 2. Data Understanding

Implemented data understanding / profiling:

- AMI annotation ingestion
- utterance/token construction for references
- audio QC metrics and provenance logging
- speech metrics (`WER`, `cpWER`, approximate `DER`)
- ASR confidence QA reporting

Evidence:

- `src/ami_mom_pipeline/utils/ami_annotations.py`
- `src/ami_mom_pipeline/utils/audio_utils.py`
- `scripts/eval_speech_metrics.py`
- `scripts/eval_asr_confidence.py`
- `scripts/eval_diarization_speaker_count.py`
- `data/staged/ami/audio_qc_metrics.csv` (generated)
- `artifacts/eval/ami/*.csv` (generated)

## 3. Data Preparation

Implemented preparation stages:

- audio ingest/staging (AMI -> staged WAV path and QC)
- VAD / diarization / speaker-aware ASR artifacts
- transcript normalization (rule-based)
- canonical meeting object construction (Pydantic)
- stable chunking with overlap and chunk IDs

Evidence:

- `src/ami_mom_pipeline/pipeline.py`
- `src/ami_mom_pipeline/schemas/models.py`
- `artifacts/ami/<meeting_id>/transcript_raw.json`
- `artifacts/ami/<meeting_id>/transcript_normalized.json`
- `artifacts/ami/<meeting_id>/transcript_chunks.jsonl`
- `artifacts/ami/meetings_canonical.jsonl`

Note:

- The plan mentions spaCy as a possible component. Current implementation uses rule-based normalization (documented in `docs/NORMALIZATION_DECISION.md`) to preserve offline simplicity and determinism.

## 4. Modeling

Implemented modeling layers:

- Speech: NeMo (VAD, diarization, ASR)
- Summarization: local `llama.cpp` + GGUF instruct model
- Structured extraction: local `llama.cpp` + Pydantic + post-validation

Implemented model tuning / optimization work:

- ASR model benchmarking (`small` / `medium` / `large` Conformer CTC)
- diarization parameter tuning and evaluation (`DER`, `cpWER`, speaker count checks)
- `llama.cpp` prompt tuning + hybrid chunk selection + evidence-backed MoM outputs

Evidence:

- `scripts/nemo_vad.py`
- `scripts/nemo_diarize.py`
- `scripts/nemo_asr.py`
- `src/ami_mom_pipeline/backends/nemo_backend.py`
- `src/ami_mom_pipeline/backends/llama_cpp_backend.py`
- `models/nemo/diarizer/inference.yaml`
- `configs/pipeline.nemo.llama*.yaml`

## 5. Evaluation

Implemented evaluation and validation:

- `WER`, `CER` (where available), `cpWER`, approximate `DER`
- diarization speaker-count analysis
- ASR confidence QA
- MoM schema validation and extraction validation reports
- reproducibility audits and deterministic regression tests

Evidence:

- `scripts/eval_speech_metrics.py`
- `scripts/eval_diarization_speaker_count.py`
- `scripts/eval_asr_confidence.py`
- `scripts/repro_audit.py`
- `tests/test_deterministic_artifact_digest.py`
- `tests/test_llama_summary_parser.py`
- `tests/test_validate_only_and_determinism.py`

## 6. Deployment / Operations (Offline Research-Grade Pipeline)

Implemented operationalization features:

- sequential batch runner with resume
- `--validate-only` artifact/eval audit mode
- offline preflight audit and enforcement
- stage traces, manifests, reproducibility reports
- optional local-only MLflow logging
- DVC stage template generation and offline governance scaffold
- evidence bundle generation and acceptance bundle Makefile target

Evidence:

- `scripts/run_nemo_batch_sequential.py`
- `src/ami_mom_pipeline/utils/traceability.py`
- `scripts/setup_offline_governance.py`
- `scripts/generate_dvc_stage_template.py`
- `scripts/generate_acceptance_evidence_bundle.py`
- `Makefile`

## CRISP-DM Gaps (Current)

These are the main items still considered partial:

- strict GPU byte-for-byte determinism is not guaranteed
- DVC/MLflow full end-to-end experiment protocol usage is scaffolded but not fully automated in daily workflow
- optional retrieval layer is intentionally deferred (see `docs/RETRIEVAL_LAYER_STATUS.md`)

## Recommended Academic Packaging

For capstone/thesis submission, include:

1. One frozen benchmark config (current production candidate)
2. A versioned benchmark subset definition (meeting IDs)
3. A reproducibility rerun report (`scripts/repro_audit.py`)
4. An acceptance evidence bundle (`make acceptance-bundle`)
5. This CRISP-DM mapping with links to generated artifacts
