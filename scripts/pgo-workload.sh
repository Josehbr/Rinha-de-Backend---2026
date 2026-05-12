#!/usr/bin/env sh
# PGO training workload: hits POST /fraud-score with example payloads to give
# the LLVM profile representative branch + path coverage of the hot loops.
#
# Runs inside the PGO build stage. The fraud-api binary must already be
# listening on the Unix Domain Socket at $UDS.
#
# Uses Python with persistent Unix socket connections (keepalive) to avoid
# the overhead of spawning a new curl process per request.

set -e

UDS="${UDS:-/tmp/pgo-api.sock}"
REPS="${REPS:-200}"
PAYLOADS="/app/resources/example-payloads.json"
TEST_DATA="/app/test/test-data.json"

# Prefer test-data.json (54K entries) for better PGO coverage than example-payloads (32)
if [ -f "$TEST_DATA" ]; then
    PAYLOADS="$TEST_DATA"
    IS_TEST_DATA=1
fi

if [ ! -f "$PAYLOADS" ]; then
    echo "[pgo-workload] missing $PAYLOADS"
    exit 1
fi

N=$(python3 -c "import json; d=json.load(open('$PAYLOADS')); entries=d.get('entries', d); print(len(entries))")
echo "[pgo-workload] $N payloads × $REPS reps = $((N * REPS)) requests via UDS=$UDS"

python3 - "$PAYLOADS" "$UDS" "$REPS" << 'PYEOF'
import json, socket, sys, time

payloads_path, uds_path, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(payloads_path))
entries = data.get('entries', data)  # test-data.json has 'entries' wrapper
payloads = [json.dumps(e.get('request', e)).encode() for e in entries]

def send_request(sock, body):
    req = (
        b"POST /fraud-score HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"\r\n" + body
    )
    sock.sendall(req)
    # Read response (Content-Length delimited)
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += sock.recv(4096)
    header_end = buf.index(b"\r\n\r\n") + 4
    cl = 0
    for line in buf[:header_end].split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            cl = int(line.split(b":", 1)[1].strip())
    body_received = len(buf) - header_end
    while body_received < cl:
        chunk = sock.recv(4096)
        body_received += len(chunk)

def make_conn():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(uds_path)
    return s

conn = make_conn()
sent = 0
for _ in range(reps):
    for body in payloads:
        try:
            send_request(conn, body)
            sent += 1
        except Exception:
            # Reconnect on error
            try: conn.close()
            except: pass
            conn = make_conn()
            send_request(conn, body)
            sent += 1
conn.close()
print(f"[pgo-workload] {sent} requests concluídos")
PYEOF
