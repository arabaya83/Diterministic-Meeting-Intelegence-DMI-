#!/usr/bin/env bash
# Initialize or refresh the repository-local DVC configuration for offline use.
# The script is safe to rerun: it tolerates an existing no-SCM setup and keeps
# the `local_offline` remote pointed at the repository-managed cache directory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# `dvc init --no-scm` exits non-zero when already initialized, which is fine.
dvc init --no-scm || true
# Prefer an update path on reruns so the remote stays aligned with this repo.
dvc remote add -d local_offline dvc_store/local || dvc remote modify local_offline url dvc_store/local
echo "DVC offline remote configured: local_offline -> dvc_store/local"
