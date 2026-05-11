#!/usr/bin/env sh
# PGO training workload: hits POST /fraud-score with example payloads to give
# the LLVM profile representative branch + path coverage of the hot loops.
#
# Runs inside the PGO build stage. The fraud-api binary must already be
# listening on the Unix Domain Socket at $UDS.

set -e

UDS="${UDS:-/tmp/pgo-api.sock}"
REPS="${REPS:-200}"
PAYLOADS="/app/resources/example-payloads.json"

if [ ! -f "$PAYLOADS" ]; then
    echo "[pgo-workload] missing $PAYLOADS"
    exit 1
fi

N=$(python3 -c "import json;print(len(json.load(open('$PAYLOADS'))))")
echo "[pgo-workload] $N payloads × $REPS reps = $((N * REPS)) requests via UDS=$UDS"

# Generate one JSON payload per line, then pipe to curl via the UDS socket.
i=0
while [ "$i" -lt "$REPS" ]; do
    python3 -c "
import json, sys
data = json.load(open('$PAYLOADS'))
for p in data:
    # Support both {request: {...}} and flat payload formats
    body = p.get('request', p)
    sys.stdout.write(json.dumps(body) + '\n')
" | while IFS= read -r body; do
        curl -sf --unix-socket "$UDS" http://localhost/fraud-score \
            -H 'Content-Type: application/json' \
            -d "$body" > /dev/null || true
    done
    i=$((i + 1))
done

echo "[pgo-workload] done."
