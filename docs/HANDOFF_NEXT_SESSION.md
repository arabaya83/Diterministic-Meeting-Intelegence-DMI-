# Handoff (Current Project State + Next Steps)

This handoff captures the current implementation state of the AMI-only, NeMo-centric, fully offline meeting understanding pipeline and the immediate next tasks requested.

## Environment / execution assumptions

- Project root: `/home/arabaya/projects/capstone_2`
- Conda env: `capstone-gpu`
- Offline shell bootstrap:
  - `source scripts/env_offline.sh`
- Current date context used during implementation: `2026-02-26`

## Core implemented stack (current)

- Speech pipeline:
  - NeMo VAD / diarization / ASR via wrappers and `NemoSpeechBackend`
  - Sequential batch execution is the stable path
  - ASR confidence extraction implemented (real non-zero confidences)
- NLP pipeline:
  - `llama.cpp` summarization backend (local GGUF, Qwen2.5-7B-Instruct Q5_K_M validated)
  - `llama.cpp` extraction backend (decisions/actions) with:
    - stricter prompt constraints
    - post-validation filtering of weak items
    - evidence snippets
    - hybrid summary-guided chunk selection (fewer calls, better precision/speed)
    - shared model reuse across summarize + extract in one run
- Evaluation:
  - `WER`, `CER`
  - `cpWER`
  - approximate `DER` (documented method)
  - ASR confidence QA reports
- Batch operations:
  - sequential runner with resume
  - `--validate-only`
  - artifact validation summary
  - optional DVC template generation (`--dvc-template`)

## Traceability / observability / reproducibility (implemented)

Per meeting (`artifacts/ami/{meeting_id}/`):

- `run_manifest.json` (compact stage summary + `artifact_digest` + `config_digest`)
- `stage_trace.jsonl` (stage-level start/end events + timings + summaries)
- `preflight_offline_audit.json` (offline compliance checks)
- `reproducibility_report.json` (determinism report, environment snapshot, code provenance hashes)

Determinism controls:

- best-effort deterministic setup (Python/NumPy/PyTorch)
- optional strict gate:
  - `runtime.fail_on_determinism_risks: true`
  - fails early on GPU nondeterminism risk (validated)

## Governance / plan alignment upgrades (implemented)

- Offline DVC/MLflow scaffold:
  - `scripts/setup_offline_governance.py`
  - `scripts/dvc_offline_init.sh`
  - `scripts/run_mlflow_offline.sh`
  - `artifacts/governance/offline_governance_manifest.json`
- Optional local-file MLflow logging:
  - pipeline-level (per meeting)
  - batch-level aggregate summary
- DVC stage template generator:
  - `scripts/generate_dvc_stage_template.py`
  - integrated into batch runner via `--dvc-template`
- Reproducibility audit CLI:
  - `scripts/repro_audit.py`
  - compares `artifact_digest` and `config_digest` across roots/snapshots

## Documentation status (important)

Added/updated docs:

- `README.md`
- `docs/OFFLINE_SETUP.md`
- `docs/REPRODUCIBILITY_OBSERVABILITY.md`
- `docs/GOVERNANCE_OFFLINE.md`
- `docs/PLAN_ALIGNMENT.md`
- `docs/ACCEPTANCE_CHECKLIST_SECTION16.md`

`docs/PLAN_ALIGNMENT.md` now explicitly marks complete/partial/pending areas.

## Test coverage added (current)

Regression tests (all passing in current environment):

- `tests/test_llama_summary_parser.py`
- `tests/test_batch_artifact_validation.py`
- `tests/test_stage_trace_writer.py`
- `tests/test_validate_only_and_determinism.py`
- `tests/test_mlflow_local_logging.py` (conditional on `mlflow` + local AMI data)
- `tests/test_batch_mlflow_logging.py` (conditional on `mlflow`)
- `tests/test_deterministic_artifact_digest.py`
- `tests/test_repro_audit_snapshot.py`
- `tests/test_batch_runner_dvc_template.py`

Recent test status:

- targeted suites passed (`10/10`, `12/12`, and later `7/7` subsets during iterative work)

## Known/important caveats

- Strict byte-for-byte determinism is not guaranteed on GPU; risks are tracked and can be gated.
- `DER` implementation is approximate (documented no-overlap interval method) and suitable for comparative tuning, not canonical external scoring.
- DVC/MLflow integration is implemented as scaffold + hooks; full experiment protocol/automation is still partial.
- MLflow file store emits a future deprecation warning in current installed version (test warning only; still works now).

## Last verified commands (examples)

NeMo + llama full run:

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.yaml --meeting-id ES2005a --no-resume
```

Validate-only + DVC template:

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.yaml --meeting-id ES2005a --validate-only --dvc-template single
```

Repro audit:

```bash
python3 scripts/repro_audit.py --config configs/pipeline.nemo.llama.yaml --meeting-id ES2005a
```

## Immediate next tasks (requested now)

1. Add an acceptance evidence bundle generator script
   - collect key outputs (manifests, traces, audits, summaries, eval JSON/CSV, docs references) into `artifacts/governance/evidence_bundle/<timestamp>/`
2. Add local test command targets (`Makefile`)
   - quick subsets for reproducibility/traceability/governance regression tests
3. Add strict offline profile config preset
   - e.g. `configs/pipeline.nemo.llama.strict_offline.yaml`
   - explicitly enable offline checks + MLflow local file store + documented deterministic-risk behavior

## Suggested first commands for next session

```bash
python3 scripts/generate_acceptance_evidence_bundle.py --meeting-id ES2005a
make test-repro
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.strict_offline.yaml --meeting-id ES2005a --validate-only
```
