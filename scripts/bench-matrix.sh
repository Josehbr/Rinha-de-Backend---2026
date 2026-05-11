#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/benchmarks"
mkdir -p "$RESULTS_DIR"

cd "$ROOT_DIR"

run_case() {
  local nprobe="$1"
  local workers="$2"
  local tag="nprobe${nprobe}-workers${workers}"
  local result_file="$RESULTS_DIR/${tag}.json"

  echo "[bench] running $tag"
  NPROBE="$nprobe" \
  WORKERS="$workers" \
  LOCAL_TEST_BUILD=0 \
  K6_STAGE_DURATION="${K6_STAGE_DURATION:-20s}" \
  K6_TARGET="${K6_TARGET:-250}" \
  K6_PRE_VUS="${K6_PRE_VUS:-80}" \
  K6_MAX_VUS="${K6_MAX_VUS:-250}" \
  K6_RESULTS_PATH="$result_file" \
  ./scripts/local-test.sh
}

# Fase 1: tuning nprobe com workers fixo
for nprobe in 1 2 4 8; do
  run_case "$nprobe" 2
done

# Fase 2: tuning workers com melhor nprobe candidato (2)
for workers in 1 2 4; do
  run_case 2 "$workers"
done

echo "[bench] completed. Results in $RESULTS_DIR"
