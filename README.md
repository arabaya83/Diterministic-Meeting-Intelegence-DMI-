<<<<<<< HEAD
# meeting_sum_app
meeting summarization Application
=======
# AMI Meeting Understanding Pipeline (Offline-First, NeMo + llama.cpp)

This repository contains a stage-based AMI meeting understanding pipeline aligned to the provided NeMo-centric architecture:

- NeMo for speech stages (VAD, diarization, ASR)
- `llama.cpp` for local/offline summarization and structured extraction
- persisted artifacts per stage for reproducibility and auditability
- optional retrieval layer artifacts (`retrieval_results.json`, optional `faiss_index/`)
- evaluation artifacts (`WER`, `cpWER`, approximate `DER`)
- batch execution with resume + validation

## Current status (implemented)

- End-to-end AMI pipeline runs offline after one-time model acquisition.
- NeMo speech pipeline is integrated and working in sequential mode.
- `llama.cpp` summarization and extraction are integrated and validated with local GGUF models.
- Per-meeting artifact manifests, stage traces, offline preflight audit, and reproducibility reports are written.
- Deterministic controls are applied (best effort) and recorded per run.
- spaCy-backed normalization is supported offline (with `spacy.blank("en")` fallback).

## Quick start

```bash
PYTHONPATH=src python3 -m ami_mom_pipeline list-meetings --limit 5
PYTHONPATH=src python3 -m ami_mom_pipeline --config configs/pipeline.nemo.llama.yaml run --meeting-id ES2005a
```

Or install editable:

```bash
pip install -e .
ami-mom run --meeting-id ES2005a
```

Validate existing artifacts without rerunning meetings:

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.yaml --meeting-id ES2005a --validate-only
```

Generate a matching DVC template while running/validating a selected set:

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.yaml --meeting-id ES2005a --validate-only --dvc-template single
```

Strict offline profile (local-only MLflow enabled, offline checks enforced):

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.strict_offline.yaml --meeting-id ES2005a --validate-only
```

Convenience local regression targets:

```bash
make test-repro
make test-governance
make test-all-local
make acceptance-bundle
```

## Key paths

- Raw AMI audio: `data/rawa/ami/audio/`
- AMI annotations: `data/rawa/ami/annotations/`
- Staged audio/QC: `data/staged/ami/`
- Meeting artifacts: `artifacts/ami/{meeting_id}/`
- Eval artifacts: `artifacts/eval/ami/`

## Traceability / observability artifacts (per meeting)

Written to `artifacts/ami/{meeting_id}/`:

- `run_manifest.json` (stable stage summary + artifact digest)
- `stage_trace.jsonl` (stage-level start/end events + elapsed time)
- `preflight_offline_audit.json` (offline compliance checks)
- `reproducibility_report.json` (config digest, environment snapshot, determinism report, code provenance)
- `retrieval_results.json` (when retrieval is enabled)

## Determinism and reproducibility notes

- The pipeline now applies deterministic settings (Python hash seed, RNG seeding, best-effort NumPy/PyTorch deterministic controls) and records them.
- `run_manifest.json` excludes timing metadata by default to keep manifest content more reproducible.
- Artifact digests exclude nondeterministic trace/audit files.
- GPU inference (NeMo / llama.cpp) can still be nondeterministic for some kernels; this risk is explicitly recorded in `reproducibility_report.json`.

## Documentation

- Offline setup: `docs/OFFLINE_SETUP.md`
- Reproducibility + observability: `docs/REPRODUCIBILITY_OBSERVABILITY.md`
- Pipeline stage flow: `docs/PIPELINE_OVERVIEW.md`
- Artifact schema/contract: `docs/ARTIFACT_CONTRACT.md`
- Plan alignment / implementation status: `docs/PLAN_ALIGNMENT.md`
- Offline governance scaffold (DVC/MLflow): `docs/GOVERNANCE_OFFLINE.md`
- Section 16 acceptance checklist (evidence-based): `docs/ACCEPTANCE_CHECKLIST_SECTION16.md`
- CRISP-DM alignment: `docs/CRISP_DM_ALIGNMENT.md`
- Normalization decision: `docs/NORMALIZATION_DECISION.md`
- Retrieval layer status: `docs/RETRIEVAL_LAYER_STATUS.md`
