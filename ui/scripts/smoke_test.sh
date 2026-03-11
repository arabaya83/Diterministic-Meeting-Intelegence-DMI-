#!/usr/bin/env bash
# Run a lightweight end-to-end smoke test for the UI backend and frontend.
# The script starts the backend on a local port, probes a small set of API
# endpoints, optionally builds the frontend, and then tears the backend down.
# It is safe to rerun; the only persistent side effect is a backend log file.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${AMI_UI_SMOKE_PORT:-18000}"
HOST="${AMI_UI_SMOKE_HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"
LOG_FILE="${ROOT_DIR}/ui/backend/.smoke_backend.log"

cleanup() {
  # Always stop the background backend process before exiting the script.
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

cd "${ROOT_DIR}"
# Start the backend in the background and capture logs for post-failure review.
PYTHONPATH=ui/backend python3 -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
BACKEND_PID=$!

# Poll health until the backend is ready or the retry loop expires.
for _ in {1..20}; do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS "${BASE_URL}/health" >/dev/null
curl -fsS "${BASE_URL}/api/meetings" >/dev/null
curl -fsS "${BASE_URL}/api/eval/summary" >/dev/null
curl -fsS "${BASE_URL}/api/configs" >/dev/null

# Build the frontend when its package manifest is present so the smoke test
# also catches obvious packaging regressions.
if [[ -f "${ROOT_DIR}/ui/frontend/package.json" ]]; then
  npm --prefix "${ROOT_DIR}/ui/frontend" run build >/dev/null
fi

echo "Smoke test passed"
