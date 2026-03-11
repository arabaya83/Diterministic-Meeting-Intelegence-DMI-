# Developer Guide

Deterministic Meeting Intelligence (DMI)

This guide is for developers working inside this repository. It explains how the
system is organized, how data flows through the pipeline, where artifacts are
written, and what design constraints matter most when making changes.

The repository is built around a few core principles:

- offline-first execution
- deterministic behavior where practical
- explicit artifact contracts between stages
- reproducibility and auditability

For deeper reference material, also see:

- [README.md](/home/arabaya/projects/capstone_2/README.md)
- [docs/PIPELINE_OVERVIEW.md](/home/arabaya/projects/capstone_2/docs/PIPELINE_OVERVIEW.md)
- [docs/ARTIFACT_CONTRACT.md](/home/arabaya/projects/capstone_2/docs/ARTIFACT_CONTRACT.md)
- [docs/REPRODUCIBILITY_OBSERVABILITY.md](/home/arabaya/projects/capstone_2/docs/REPRODUCIBILITY_OBSERVABILITY.md)
- [docs/OFFLINE_SETUP.md](/home/arabaya/projects/capstone_2/docs/OFFLINE_SETUP.md)

---

# 1. Project Overview

Deterministic Meeting Intelligence (DMI) is an offline pipeline that turns AMI
meeting recordings into structured meeting outputs such as:

- transcript artifacts
- summaries
- decisions
- action items
- speech and summary evaluation metrics

The pipeline integrates:

- voice activity detection (VAD)
- speaker diarization
- automatic speech recognition (ASR)
- transcript normalization and canonicalization
- transcript chunking
- optional retrieval artifacts
- local LLM summarization
- structured extraction
- evaluation and traceability reporting

The current implementation is centered on:

- NeMo-backed speech stages
- `llama.cpp`-backed local summarization and extraction
- per-meeting artifact directories under `artifacts/ami/`

---

# 2. High-Level Architecture

The system follows a stage-based pipeline. Each stage reads inputs from either
the dataset, configuration, or prior stage artifacts, then writes explicit
outputs that later stages can consume.

High-level flow:

1. Meeting audio is discovered from the AMI raw audio directory.
2. Audio is staged and basic QC metadata is recorded.
3. VAD identifies speech regions.
4. Diarization assigns speaker labels to speech segments.
5. ASR produces timestamped transcript segments.
6. Transcript canonicalization normalizes the transcript into a stable schema.
7. Chunking divides the transcript into deterministic LLM-sized windows.
8. Optional retrieval artifacts may be produced.
9. Summarization creates MoM-oriented meeting summaries.
10. Extraction produces structured decisions and action items.
11. Evaluation writes speech and summary quality metrics.
12. Final manifest, trace, and reproducibility artifacts are written.

The pipeline implementation lives primarily in
[src/ami_mom_pipeline/pipeline.py](/home/arabaya/projects/capstone_2/src/ami_mom_pipeline/pipeline.py).

---

# 3. Repository Structure

The repository is organized into these main areas:

```text
project_root/
├── src/
├── scripts/
├── configs/
├── models/
├── data/
├── artifacts/
├── docs/
├── tests/
└── ui/
```

Key directories:

- `src/`
  Main Python implementation of the pipeline, backends, schemas, and utilities.
- `scripts/`
  Python and shell entrypoints for batch runs, evaluation, offline setup, and
  NeMo wrapper execution.
- `configs/`
  YAML configuration files controlling model paths, runtime behavior, and stage
  settings.
- `models/`
  Local model assets used for offline execution.
- `data/`
  Raw AMI inputs, annotations, and staged data.
- `artifacts/`
  Generated outputs, evaluation results, batch summaries, governance artifacts,
  and MLflow logs.
- `tests/`
  Regression tests for pipeline helpers, scripts, and reproducibility behavior.
- `ui/`
  Optional backend and frontend for browsing artifacts and launching runs.

---

# 4. Core Pipeline Components

## 4.1 Ingest and Staging

The ingest stage prepares audio for downstream processing and records QC-style
metadata such as sample rate, duration, and channel count.

Typical outputs include:

- staged WAV audio
- ingest QC metadata

## 4.2 VAD

The VAD stage identifies speech regions in the meeting audio.

Typical outputs:

- `vad_segments.json`
- `vad_segments.rttm`

This stage may reuse precomputed artifacts or derive regions from diarization
depending on wrapper configuration.

## 4.3 Speaker Diarization

The diarization stage detects speaker boundaries and assigns speaker labels.

Typical outputs:

- `diarization_segments.json`
- `diarization.rttm`

These artifacts are later used to preserve speaker structure during ASR and
transcript generation.

## 4.4 ASR

The ASR stage converts speech into text.

Typical outputs:

- `asr_segments.json`
- `full_transcript.txt`

Speech evaluation later uses these artifacts to compute:

- WER
- CER
- cpWER
- approximate DER

## 4.5 Transcript Canonicalization

This stage transforms ASR output into a stable internal transcript schema. It
normalizes text, speaker fields, and meeting-level metadata so downstream stages
operate on predictable structures.

Typical outputs:

- `transcript_raw.json`
- `transcript_normalized.json`

## 4.6 Chunking

Chunking splits normalized transcripts into deterministic windows suitable for
LLM prompting.

Typical outputs:

- `transcript_chunks.jsonl`

Deterministic chunking is important because summary and extraction quality can
change if chunk boundaries drift.

## 4.7 Retrieval

Retrieval is optional in this repository. When enabled, it produces artifacts
used to support or inspect chunk selection.

Typical outputs:

- `retrieval_results.json`

## 4.8 Summarization

Summarization uses a local `llama.cpp` backend to produce structured summary
content for a meeting.

Typical outputs include:

- meeting summary narrative
- key points
- discussion points
- follow-up items

Artifacts are typically persisted as:

- `mom_summary.json`
- `mom_summary.html`

## 4.9 Structured Extraction

Extraction converts chunk-level evidence into structured decisions and action
items.

Typical outputs:

- `decisions_actions.json`
- `extraction_validation_report.json`

The extraction stage is designed to preserve evidence snippets and supporting
chunk ids to improve traceability.

## 4.10 Evaluation

The evaluation stage measures both speech and summary quality.

Speech-related metrics include:

- WER
- CER
- cpWER
- approximate DER

Summary-related metrics include:

- ROUGE-1
- ROUGE-2
- ROUGE-L

Typical aggregate outputs live under:

- `artifacts/eval/ami/`

---

# 5. Artifact Contract

One of the most important design concepts in this repository is the artifact
contract.

Each stage writes explicit, named outputs that become the stable interface for:

- later pipeline stages
- validation logic
- the UI backend
- reproducibility audits
- acceptance and governance workflows

Common per-meeting artifacts include:

- `vad_segments.json`
- `vad_segments.rttm`
- `diarization_segments.json`
- `diarization.rttm`
- `asr_segments.json`
- `full_transcript.txt`
- `transcript_raw.json`
- `transcript_normalized.json`
- `transcript_chunks.jsonl`
- `mom_summary.json`
- `mom_summary.html`
- `decisions_actions.json`
- `run_manifest.json`
- `stage_trace.jsonl`
- `preflight_offline_audit.json`
- `reproducibility_report.json`

Per-meeting artifacts are generally written to:

- `artifacts/ami/{meeting_id}/`

Aggregate evaluation artifacts are generally written to:

- `artifacts/eval/ami/`

If you change a stage's outputs, you are changing a cross-cutting contract and
should update validation logic, docs, and any UI readers that depend on it.

---

# 6. Running the Pipeline

Primary CLI entrypoints include:

```bash
PYTHONPATH=src python3 -m ami_mom_pipeline list-meetings --limit 5
PYTHONPATH=src python3 -m ami_mom_pipeline --config configs/pipeline.nemo.llama.final_eval.yaml run --meeting-id ES2005a
```

Batch execution is handled by:

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a
```

Validation-only mode checks existing artifacts without rerunning stages:

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a --validate-only
```

Other important entrypoints:

- `scripts/eval_speech_metrics.py`
- `scripts/eval_diarization_speaker_count.py`
- `scripts/repro_audit.py`
- `scripts/generate_acceptance_evidence_bundle.py`
- `scripts/generate_dvc_stage_template.py`

Shell helpers include:

- `scripts/env_offline.sh`
- `scripts/dvc_offline_init.sh`
- `scripts/run_mlflow_offline.sh`
- `ui/scripts/smoke_test.sh`

---

# 7. Configuration

Configuration is primarily YAML-based and lives under:

- `configs/`

Config files define parameters such as:

- model paths
- backend modes
- runtime offline settings
- determinism strictness
- chunking behavior
- retrieval settings
- evaluation behavior
- MLflow logging settings

Developers should prefer config changes over hardcoding behavior in source code.

Important rule:

- local filesystem paths are preferred for offline execution
- URL-based model paths are generally rejected in offline-sensitive paths

---

# 8. Determinism and Reproducibility

This repository is explicitly designed to support deterministic and auditable
execution wherever feasible.

Key mechanisms include:

- explicit pipeline stage ordering
- deterministic chunking and normalization helpers
- config hashing
- stage-level trace logging
- offline preflight audits
- reproducibility reports
- artifact digests that exclude selected traceability sidecars

Important caveat:

- some GPU-backed libraries can still have nondeterministic kernels even when
  deterministic settings are requested

The repository records these risks rather than pretending they do not exist.

Primary traceability artifacts:

- `run_manifest.json`
- `stage_trace.jsonl`
- `preflight_offline_audit.json`
- `reproducibility_report.json`

---

# 9. UI and Operational Tooling

The `ui/backend/` service reads pipeline artifacts directly from disk and
exposes them through API endpoints for browsing:

- meeting status
- artifact previews
- evaluation summaries
- reproducibility metadata
- run-control status

The UI runner intentionally shells out to the existing batch runner instead of
re-implementing pipeline logic in the web layer.

That design keeps the CLI and UI aligned on the same operational contract.

---

# 10. Adding New Features

When adding or changing functionality:

1. Preserve deterministic behavior where practical.
2. Prefer writing explicit artifacts over hidden in-memory behavior.
3. Keep stage boundaries clear.
4. Document new config fields and artifact outputs.
5. Update validation logic when artifact contracts change.
6. Update docs when architecture or operational workflows change.

Questions to ask before merging a change:

- Does this alter any persisted artifact schema?
- Does this change run-to-run determinism?
- Does this require new offline assumptions?
- Does the UI or validation layer need updates?

---

# 11. Code Style and Documentation Expectations

The repository now expects:

- module-level docstrings in Python files
- function and class docstrings for public and important internal code
- selective inline comments for non-obvious behavior
- shell-script header comments explaining purpose and assumptions

The goal is readability, not comment volume. Comments should explain:

- why a stage or helper exists
- what artifact contract it preserves
- what determinism or offline constraints matter
- what assumptions future maintainers must not accidentally break

---

# 12. Contribution Guidelines

When contributing:

1. Use a feature branch.
2. Avoid broad refactors unless they are explicitly required.
3. Keep behavior changes separate from documentation-only changes when possible.
4. Preserve offline and deterministic guarantees unless a change is deliberate
   and documented.
5. Add or update tests when behavior changes.

---

# 13. Future Improvement Areas

Likely areas for future work include:

- improved diarization quality
- stronger retrieval strategies
- domain-adapted summarization and extraction prompts
- stronger reproducibility reporting
- richer evaluation dashboards
- broader meeting-corpus support

These should be approached carefully because changes in any of these areas can
affect artifact compatibility, evaluation comparability, and determinism.

---

# 14. Maintainer

Ayman Rabaya  
Northwestern University - MS Data Science
