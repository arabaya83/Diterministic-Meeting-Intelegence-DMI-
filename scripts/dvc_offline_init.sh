#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
dvc init --no-scm || true
dvc remote add -d local_offline dvc_store/local || dvc remote modify local_offline url dvc_store/local
echo "DVC offline remote configured: local_offline -> dvc_store/local"
