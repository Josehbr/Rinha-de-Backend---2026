#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K6_BIN="${K6_BIN:-$HOME/.local/bin/k6}"
LOCAL_TEST_BUILD="${LOCAL_TEST_BUILD:-1}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-180}"

cd "$ROOT_DIR"

cleanup() {
  docker compose down
}
trap cleanup EXIT

if [[ "$LOCAL_TEST_BUILD" == "1" ]]; then
  docker compose up --build -d
else
  docker compose up -d
fi

echo "Aguardando health check..."
health_deadline=$((SECONDS + HEALTH_TIMEOUT_SEC))
until curl -sf http://localhost:9999/ready >/dev/null; do
  if (( SECONDS >= health_deadline )); then
    echo "Timeout aguardando /ready depois de ${HEALTH_TIMEOUT_SEC}s"
    docker compose ps
    exit 1
  fi
  sleep 2
done

echo "Rodando k6..."
"$K6_BIN" run test/test.js
