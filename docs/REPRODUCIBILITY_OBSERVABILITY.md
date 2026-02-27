# Reproducibility, Traceability, and Observability

This document describes the pipeline controls and artifacts that support:

- traceability (what happened, when, and with what configuration/code)
- observability (stage timings and outcomes)
- reproducibility (best-effort deterministic execution + auditable provenance)
- offline compliance checks

## Per-meeting observability artifacts

For each meeting run (`artifacts/ami/{meeting_id}/`):

- `stage_trace.jsonl`
  - stage-level `stage_start` / `stage_end` events
  - includes stage name, status, elapsed seconds
  - includes compact stage summaries when available
- `run_manifest.json`
  - compact stage outputs summary
  - includes retrieval stage summary when retrieval is enabled
  - artifact digest (`artifact_digest`) over deterministic stage artifacts
  - backend selections and config digest reference
- `preflight_offline_audit.json`
  - checks local-only model/config paths
  - checks NeMo command templates for URLs/downloaders
  - records offline-related environment flags (`HF_HUB_OFFLINE`, etc.)
- `reproducibility_report.json`
  - config digest (stable hash of config)
  - environment snapshot
  - determinism settings report (Python/NumPy/PyTorch best effort)
  - code provenance hashes for core pipeline/backend/scripts
- `retrieval_results.json` (when retrieval is enabled)
  - selected evidence chunks and scores
  - optional FAISS metadata when FAISS mode is enabled

## Determinism controls (best effort)

Configured in `runtime`:

- `deterministic_mode: true`
- `include_nondeterministic_timings_in_manifest: false`

Applied by `src/ami_mom_pipeline/utils/determinism.py`:

- Python RNG seed
- `PYTHONHASHSEED`
- `CUBLAS_WORKSPACE_CONFIG` (best effort)
- `TOKENIZERS_PARALLELISM=false`
- NumPy seed (if available)
- PyTorch seed and deterministic flags (if available)

### Important limitation

GPU inference via NeMo / `llama.cpp` may still exhibit nondeterminism depending on kernels/runtime versions. This is tracked in `reproducibility_report.json` under `determinism.risks`.

## Regression coverage (implemented)

Current automated regression tests include:

- summary parser robustness against repeated / nested JSON model output
- artifact contract validation (including traceability artifacts)
- `stage_trace.jsonl` reset-per-run behavior
- `--validate-only` batch record behavior
- strict determinism failure path (risk gate)
- mock-mode repeated-run `artifact_digest` stability check

## Reproducibility audit command

Use `scripts/repro_audit.py` to compare `artifact_digest` and `config_digest` across one or more artifact roots/snapshots.

Example (current artifacts only):

```bash
python3 scripts/repro_audit.py --config configs/pipeline.nemo.llama.yaml --meeting-id ES2005a
```

Example (compare current artifacts to a saved snapshot root):

```bash
python3 scripts/repro_audit.py --meeting-id ES2005a --snapshot-dir /path/to/previous_artifacts
```

## Offline compliance

Offline checks are enforced before stage execution when:

- `runtime.offline: true`
- `runtime.write_preflight_audit: true`
- `runtime.fail_on_offline_violations: true`

The preflight audit currently checks:

- local-only model/config paths (no `http://` / `https://`)
- command templates for explicit URL/downloader usage (`curl`, `wget`)
- offline-related env flags presence (warning-level)

## Normalization reproducibility

Normalization mode is explicit in config and recorded in canonical metadata:

- `pipeline.normalization.mode` (`rule` or `spacy`)
- `pipeline.normalization.spacy_model`
- per-meeting canonical metadata includes:
  - `metadata.normalization.mode_requested`
  - `metadata.normalization.mode_used`

In `spacy` mode, the pipeline uses `en_core_web_sm` when available and falls back to `spacy.blank("en")` offline.

## Artifact digest behavior

`artifact_digest` in `run_manifest.json` intentionally excludes:

- `run_manifest.json`
- `stage_trace.jsonl`
- `preflight_offline_audit.json`
- `reproducibility_report.json`

Reason:

- these files contain run-time observability metadata and environment snapshots that are useful for tracing but may differ across runs
- excluding them keeps the digest focused on core stage outputs

## Batch runner observability

`scripts/run_nemo_batch_sequential.py` adds:

- batch events JSONL (`*.events.jsonl`)
- per-meeting timings CSV (`*.timings.csv`)
- validation report (`*.validation.json`)
- speech evaluation summaries (`WER`, `cpWER`, `DER`)
