#!/usr/bin/env bash
# Launch the MLflow UI against the repository-local offline tracking store.
# Use this after pipeline runs when you want to inspect locally logged runs.
# Side effects are limited to creating `artifacts/mlruns` if it is missing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Ensure the file-based backend store exists before handing control to MLflow.
mkdir -p "$ROOT/artifacts/mlruns"
exec mlflow ui --backend-store-uri "file:$ROOT/artifacts/mlruns" --host 127.0.0.1 --port 5001
