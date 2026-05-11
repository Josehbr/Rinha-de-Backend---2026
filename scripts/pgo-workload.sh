#!/usr/bin/env sh
# PGO training workload: hits POST /fraud-score with example payloads to give
# the LLVM profile representative branch + path coverage of the hot loops.
#
# Runs inside the PGO build stage. The fraud-api binary must already be
# running on localhost:9998 (UDS not needed for profile-gen).

set -e

URL="${URL:-http://localhost:9998/fraud-score}"
REPS="${REPS:-200}"
PAYLOADS="/app/resources/example-payloads.json"

if [ ! -f "$PAYLOADS" ]; then
    echo "[pgo-workload] missing $PAYLOADS"
    exit 1
fi

# Number of payloads
N=$(python3 -c "import json;print(len(json.load(open('$PAYLOADS'))))")
echo "[pgo-workload] $N payloads × $REPS reps = $((N * REPS)) requests"

i=0
while [ "$i" -lt "$REPS" ]; do
    python3 -c "
import json, urllib.request, sys
data=json.load(open('$PAYLOADS'))
for p in data:
    body = json.dumps(p.get('request', p)).encode()
    req = urllib.request.Request('$URL', body, headers={'Content-Type':'application/json'})
    try:
        urllib.request.urlopen(req, timeout=2).read()
    except Exception as e:
        print('err:', e, file=sys.stderr)
"
    i=$((i + 1))
done

echo "[pgo-workload] done."
