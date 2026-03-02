# Section 16 Acceptance Checklist (Evidence-Based)

This checklist maps directly to the plan's Section 16 acceptance criteria and provides concrete evidence paths and commands.

Status legend:

- `Met`
- `Met (Documented Caveat)`

## 1. The entire AMI pipeline runs fully offline

Status: `Met`

Evidence paths:

- `scripts/env_offline.sh`
- `artifacts/ami/<meeting_id>/preflight_offline_audit.json`
- `docs/OFFLINE_SETUP.md`
- `src/ami_mom_pipeline/utils/traceability.py` (`offline_preflight_audit`)
- `src/ami_mom_pipeline/backends/nemo_backend.py` (local-path validation)
- `src/ami_mom_pipeline/backends/llama_cpp_backend.py` (local GGUF path enforcement)

Validation commands:

```bash
source scripts/env_offline.sh
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a --validate-only
```

Check:

- `artifacts/ami/ES2005a/preflight_offline_audit.json` -> `"ok": true`

## 2. NeMo handles VAD, diarization, and ASR

Status: `Met`

Evidence paths:

- `src/ami_mom_pipeline/backends/nemo_backend.py`
- `scripts/nemo_vad.py`
- `scripts/nemo_diarize.py`
- `scripts/nemo_asr.py`
- `configs/pipeline.nemo.yaml`
- `configs/pipeline.nemo.llama.final_eval.yaml`

Produced artifacts (per meeting):

- `vad_segments.json`, `vad_segments.rttm`
- `diarization_segments.json`, `diarization.rttm`
- `asr_segments.json`, `full_transcript.txt`

Validation command:

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a
```

## 3. Canonical artifacts are deterministic

Status: `Met (Documented Caveat)` (best-effort deterministic + audited)

Evidence paths:

- `src/ami_mom_pipeline/utils/determinism.py`
- `src/ami_mom_pipeline/pipeline.py` (`config_digest`, deterministic manifest behavior)
- `artifacts/ami/<meeting_id>/reproducibility_report.json`
- `tests/test_deterministic_artifact_digest.py`
- `scripts/repro_audit.py`

Validation commands:

```bash
PYTHONPATH=src pytest -q tests/test_deterministic_artifact_digest.py
python3 scripts/repro_audit.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a
```

Notes:

- Manifest excludes nondeterministic timing fields by default
- GPU nondeterminism risks are explicitly recorded and can be enforced via `fail_on_determinism_risks`

## 4. MoM outputs are schema-valid and reproducible

Status: `Met`

Evidence paths:

- `artifacts/ami/<meeting_id>/mom_summary.json`
- `artifacts/ami/<meeting_id>/decisions_actions.json`
- `artifacts/ami/<meeting_id>/extraction_validation_report.json`
- `src/ami_mom_pipeline/backends/llama_cpp_backend.py`
- `tests/test_llama_summary_parser.py`

Validation commands:

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a --no-resume
```

Check:

- `extraction_validation_report.json` -> `"schema_valid": true`
- `mom_summary.json` parses and is human-readable

## 5. Evaluation metrics are reproducible end-to-end

Status: `Met`

Evidence paths:

- `scripts/eval_speech_metrics.py`
- `src/ami_mom_pipeline/utils/speech_eval.py`
- `scripts/run_nemo_batch_sequential.py`
- `artifacts/batch_runs/*.speech_metrics.summary.json`
- `artifacts/eval/ami/`

Validation commands:

```bash
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a --validate-only
```

Check:

- pipeline evaluation artifacts include `WER`, `CER`, `cpWER`, approximate `DER`, `ROUGE`, and `mom_quality_checks`
- standalone speech-eval outputs still provide cross-check and batch summary reports for `cpWER` and `DER`
- `--validate-only` reproduces evaluation over existing artifacts

## 6. Alignment with CRISP-DM and academic standards is demonstrated

Status: `Met (Documented Caveat)`

Evidence paths:

- `docs/PLAN_ALIGNMENT.md`
- `docs/CRISP_DM_ALIGNMENT.md`
- `docs/NORMALIZATION_DECISION.md`
- `docs/RETRIEVAL_LAYER_STATUS.md`
- `docs/REPRODUCIBILITY_OBSERVABILITY.md`
- `docs/GOVERNANCE_OFFLINE.md`
- `artifacts/ami/<meeting_id>/stage_trace.jsonl`
- `artifacts/ami/<meeting_id>/reproducibility_report.json`
- `artifacts/governance/offline_governance_manifest.json`

Validation commands:

```bash
python3 scripts/setup_offline_governance.py
python3 scripts/repro_audit.py --meeting-id ES2005a
```

Final notes:

- DVC/MLflow offline governance is implemented and integrated in pipeline/batch workflows.
- Reproducibility rerun/audit reports are provided for the final benchmark subset.
- The only remaining caveat is GPU kernel/runtime nondeterminism risk, which is explicitly captured in reproducibility artifacts and can be strict-gated.

Final benchmark evidence (fresh 6-meeting end-to-end run):

- `artifacts/batch_runs/final_main_6meeting_e2e_fresh.summary.json`
- `artifacts/batch_runs/final_main_6meeting_e2e_fresh.speech_metrics.summary.json`
- `artifacts/batch_runs/final_main_6meeting_e2e_fresh_validate.validation.json`
- `artifacts/governance/repro_audit_final_6meetings.json`

## Quick audit bundle (recommended)

Run these to produce a strong evidence package for one meeting:

```bash
source scripts/env_offline.sh
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a
python3 scripts/run_nemo_batch_sequential.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a --validate-only --dvc-template single
python3 scripts/repro_audit.py --config configs/pipeline.nemo.llama.final_eval.yaml --meeting-id ES2005a
PYTHONPATH=src pytest -q tests/test_deterministic_artifact_digest.py tests/test_llama_summary_parser.py
```

Optional convenience commands:

```bash
make dvc-template-smoke
make evidence-bundle
make acceptance-bundle
```
