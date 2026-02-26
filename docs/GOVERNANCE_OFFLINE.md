# Offline Governance (DVC + MLflow Scaffold)

This project now includes a lightweight offline governance scaffold aligned to the plan:

- DVC local cache/remote layout (no network)
- MLflow local file-based tracking store
- local scripts to bootstrap and run both tools offline

## Scaffold setup

Run once:

```bash
python3 scripts/setup_offline_governance.py
```

This creates (if missing):

- `artifacts/mlruns/` (MLflow local tracking store)
- `artifacts/governance/offline_governance_manifest.json`
- `dvc_store/local/` (suggested local DVC remote)
- `.dvc/config` (offline local remote template)
- `.dvcignore`
- `dvc.yaml` (example stage template)
- `scripts/dvc_offline_init.sh`
- `scripts/run_mlflow_offline.sh`

## DVC offline usage (no SCM mode)

Initialize/update local DVC config:

```bash
bash scripts/dvc_offline_init.sh
```

Notes:

- Uses `--no-scm` for local/offline experiment tracking in non-git contexts
- Default remote: `local_offline -> dvc_store/local`

## MLflow offline usage

Start local MLflow UI using a file store:

```bash
bash scripts/run_mlflow_offline.sh
```

Recommended tracking URI for scripts:

```bash
export MLFLOW_TRACKING_URI="file:$(pwd)/artifacts/mlruns"
```

Enable pipeline-level local MLflow logging (optional) in config:

```yaml
runtime:
  enable_mlflow_logging: true
  mlflow_tracking_uri: file:artifacts/mlruns
  mlflow_experiment: ami_mom_offline
```

Ready-made strict profile:

- `configs/pipeline.nemo.llama.strict_offline.yaml`

Notes:

- In offline mode, only `file:` tracking URIs are allowed by the pipeline hook.
- If `mlflow` is not installed, the pipeline continues and records the disable reason in `reproducibility_report.json`.

## Current integration status

- Scaffold + local paths + docs: implemented
- Pipeline auto-logging to MLflow / DVC stage automation: pending (future enhancement)
  - basic pipeline-level local MLflow hook is implemented (optional, behind config flag)
  - full experiment logging coverage is still pending

This intentionally keeps the runtime pipeline lightweight while still making the project governance-ready and aligned with the plan.

## DVC template generation from the batch runner

You can generate a DVC stage template for the exact selected meetings while running (or validating) a batch:

```bash
python3 scripts/run_nemo_batch_sequential.py \
  --config configs/pipeline.nemo.llama.yaml \
  --meeting-id ES2005a --meeting-id ES2005d \
  --validate-only \
  --dvc-template batch
```

The generated template path is printed and stored in the batch summary JSON (`dvc_template` field).
