#!/usr/bin/env bash
# Compara A/B duas tags do fraud-api sob stress local que simula Haswell.
# Uso: ./scripts/compare-versions.sh v1 v5

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

A="${1:-v1}"
B="${2:-v5}"
K6_BIN="${K6_BIN:-$HOME/.local/bin/k6}"

run_bench() {
    local tag="$1"
    echo ""
    echo "===== Bench LOCAL STRESS: $tag ====="
    docker compose down -v 2>/dev/null | tail -2

    # Substitui temporariamente as imagens
    export FRAUD_API_TAG="$tag"
    docker tag "josehbr/fraud-api:$tag" "fraud-api:latest"

    # Sobe com stress overlay
    docker compose -f docker-compose.yml -f docker-compose.stress.yml up -d 2>&1 | tail -3

    # Aguarda ready
    local i=0
    until curl -sf http://localhost:9999/ready >/dev/null 2>&1; do
        sleep 1
        i=$((i + 1))
        if [ "$i" -gt 60 ]; then
            echo "[$tag] TIMEOUT aguardando ready"
            return 1
        fi
    done

    # K6 com target menor (CPU 0.2 total não aguenta 900 RPS)
    local results="test/results-stress-$tag.json"
    K6_TARGET="${K6_TARGET:-150}" K6_STAGE_DURATION="${K6_STAGE_DURATION:-60s}" \
        K6_RESULTS_PATH="$results" \
        "$K6_BIN" run test/test.js 2>&1 | tail -2

    python3 -c "
import json
d=json.load(open('$results'))
s=d['scoring']
b=s['breakdown']
print(f'  $tag: p99={d[\"p99\"]:>8s}  score={s[\"final_score\"]:>8.2f}  fp={b[\"false_positive_detections\"]}  fn={b[\"false_negative_detections\"]}  errs={b[\"http_errors\"]}')
"
}

run_bench "$A"
run_bench "$B"

echo ""
echo "===== COMPARAÇÃO FINAL ====="
python3 -c "
import json
A=json.load(open('test/results-stress-$A.json'))
B=json.load(open('test/results-stress-$B.json'))
print(f'  $A: p99={A[\"p99\"]}  score={A[\"scoring\"][\"final_score\"]:.2f}')
print(f'  $B: p99={B[\"p99\"]}  score={B[\"scoring\"][\"final_score\"]:.2f}')
delta = B['scoring']['final_score'] - A['scoring']['final_score']
print(f'  Δ score ($B − $A): {delta:+.2f}')
"
