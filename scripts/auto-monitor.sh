#!/usr/bin/env bash
# Auto-monitor: aguarda nova issue rinha/test josehbr-rust no repo oficial,
# poll até bot fechar, registra resultado em benchmarks/oficial-history.tsv.
#
# Quando o bot responde, escreve um marcador em /tmp/rinha-new-result e
# encerra. Pode ser relançado via cron ou ScheduleWakeup para próxima rodada.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/benchmarks"
HISTORY="$ROOT/benchmarks/oficial-history.tsv"
LAST_SEEN="$ROOT/benchmarks/.last-seen-issue"

# Cabeçalho se não existir
if [ ! -f "$HISTORY" ]; then
    printf 'issue\tcreated_at\tclosed_at\tp99\tscore\tfp\tfn\terrs\timage_tag\n' > "$HISTORY"
fi

LAST=$(cat "$LAST_SEEN" 2>/dev/null || echo 0)

echo "[auto-monitor] last seen issue=#$LAST. Polling for new issues..."

DEADLINE=$(( $(date +%s) + 3600 )) # 1h max wait for new issue
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    gh api -X GET search/issues \
        -F 'q=rinha/test josehbr-rust in:body is:issue author:Josehbr repo:zanfranceschi/rinha-de-backend-2026' \
        -F sort=created -F order=desc -F per_page=5 > /tmp/auto-search.json 2>/dev/null
    LATEST=$(python3 -c "
import json
d=json.load(open('/tmp/auto-search.json'))
items=d.get('items', [])
print(items[0]['number'] if items else 0)
")
    if [ "$LATEST" -gt "$LAST" ]; then
        echo "[auto-monitor] new issue #$LATEST detected. Switching to per-issue polling."
        echo "$LATEST" > "$LAST_SEEN"
        break
    fi
    sleep 60
done

if [ -z "$LATEST" ] || [ "$LATEST" -le "$LAST" ]; then
    echo "[auto-monitor] no new issue within 1h, exiting."
    exit 0
fi

# Poll per-issue até fechar
ISS_DEADLINE=$(( $(date +%s) + 3000 )) # 50 min
while [ "$(date +%s)" -lt "$ISS_DEADLINE" ]; do
    gh api repos/zanfranceschi/rinha-de-backend-2026/issues/"$LATEST" > /tmp/auto-issue.json 2>/dev/null
    STATE=$(python3 -c "import json;print(json.load(open('/tmp/auto-issue.json'))['state'])")
    COMM=$(python3 -c "import json;print(json.load(open('/tmp/auto-issue.json'))['comments'])")
    TS=$(date +%H:%M:%S)
    echo "[$TS] issue #$LATEST state=$STATE comments=$COMM"
    if [ "$STATE" = "closed" ] || [ "$COMM" -gt 0 ]; then
        echo "[auto-monitor] bot responded! Fetching official score..."

        # Pega resultado oficial atualizado do leaderboard
        sleep 5
        curl -sf "https://raw.githubusercontent.com/arinhadebackend/arinhadebackend.github.io/2026-preview/results-preview.json" > /tmp/auto-lead.json 2>/dev/null
        python3 << PY
import json
d=json.load(open('/tmp/auto-lead.json'))
my=d.get('Josehbr',{}).get('josehbr-rust',{})
sc=my.get('scoring',{})
b=sc.get('breakdown',{})
created=open('/tmp/auto-issue.json').read()
import json as j
cj=j.loads(created)
row='\t'.join([
    str($LATEST), cj['created_at'], cj.get('closed_at','') or '',
    str(my.get('p99','?')),
    str(sc.get('final_score','?')),
    str(b.get('false_positive_detections','?')),
    str(b.get('false_negative_detections','?')),
    str(b.get('http_errors','?')),
    'v?'
])
with open('$HISTORY','a') as f: f.write(row+'\n')
print(f"\n=== RESULTADO ISSUE #$LATEST ===")
print(f"p99={my.get('p99')} score={sc.get('final_score')} fp={b.get('false_positive_detections')} fn={b.get('false_negative_detections')} errs={b.get('http_errors')}")
PY
        touch /tmp/rinha-new-result
        exit 0
    fi
    sleep 60
done

echo "[auto-monitor] timeout aguardando bot."
exit 1
