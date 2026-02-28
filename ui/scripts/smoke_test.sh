#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${AMI_UI_SMOKE_PORT:-18000}"
HOST="${AMI_UI_SMOKE_HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"
LOG_FILE="${ROOT_DIR}/ui/backend/.smoke_backend.log"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

cd "${ROOT_DIR}"
PYTHONPATH=ui/backend python3 -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
BACKEND_PID=$!

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

if [[ -f "${ROOT_DIR}/ui/frontend/package.json" ]]; then
  npm --prefix "${ROOT_DIR}/ui/frontend" run build >/dev/null
fi

echo "Smoke test passed"
