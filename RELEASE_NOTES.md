# Release Notes

## v1.0.1

Release tag: `v1.0.1`  
Commit: `f4f81cb`

This release finalizes the offline AMI meeting understanding pipeline for deployment and handoff.

### Included

- Completed offline NeMo + `llama.cpp` pipeline under `configs/pipeline.nemo.llama.final_eval.yaml`
- Main pipeline evaluation outputs for `WER`, `CER`, `cpWER`, approximate `DER`, `ROUGE`, and MoM quality checks
- MoM quality improvements for summary fidelity, action-item cleanup, and AMI-reference ROUGE alignment
- UI run controls for both single-meeting and batch execution
- Stage timeline artifact link fix so meeting files open from the backend preview endpoint
- Removal of application-level ASR confidence reporting after confirming the active NeMo path does not yield reliable confidence values
- Documentation refresh across `README.md` and `docs/`

### Acceptance Evidence

- Evidence bundle: `artifacts/governance/evidence_bundle/bundle_20260302T180138Z`
- Repro audit: `artifacts/governance/repro_audit.json`
- Acceptance validation summary: `artifacts/batch_runs/nemo_batch_20260302T180134Z.summary.json`
- Acceptance validation report: `artifacts/batch_runs/nemo_batch_20260302T180134Z.validation.json`

### Acceptance Result

- `ES2005a validate/ok`
- `validation_ok=1/1`
- reproducibility mismatches: `0`
- evidence bundle copied files: `24`
- missing files: `0`
