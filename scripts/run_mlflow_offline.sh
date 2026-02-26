#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/artifacts/mlruns"
exec mlflow ui --backend-store-uri "file:$ROOT/artifacts/mlruns" --host 127.0.0.1 --port 5001
