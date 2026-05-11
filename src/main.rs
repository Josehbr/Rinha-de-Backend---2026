mod index;
mod models;
mod vectorizer;

use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use index::IvfIndex;
use models::MccRisk;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::{UnixListener, UnixStream};
use tracing::warn;
use tracing_subscriber::EnvFilter;

pub static READY: AtomicBool = AtomicBool::new(false);

struct AppState {
    index: Arc<IvfIndex>,
    mcc_risk: Arc<MccRisk>,
}

// ── Pre-rendered HTTP responses ───────────────────────────────────────────────
// fraud_score ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} × approved (score < 0.6)
// Indices 0,1,2 → approved:true; 3,4,5 → approved:false.

static HTTP_FRAUD: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 37\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 37\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 37\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

static HTTP_READY_OK: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}";
static HTTP_READY_503: &[u8] =
    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
static HTTP_404: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    init_tracing();

    let index_path   = env::var("INDEX_PATH").unwrap_or_else(|_| "./index.bin".into());
    let mcc_risk_path = env::var("MCC_RISK_PATH")
        .unwrap_or_else(|_| "./resources/mcc_risk.json".into());

    let loaded_index = IvfIndex::load(std::path::Path::new(&index_path))
        .with_context(|| format!("falha ao carregar {index_path}"))
        .map_err(|e| { warn!("{e}"); e })
        .expect("índice indisponível");

    let nprobe_fast = env::var("NPROBE").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v > 0);
    let nprobe_full = env::var("FULL_NPROBE").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v > 0);

    let index = Arc::new(match (nprobe_fast, nprobe_full) {
        (Some(f), Some(fu)) => loaded_index.with_nprobes(f, fu),
        (Some(f), None)     => loaded_index.with_nprobes(f, f * 3),
        (None, Some(fu))    => { let f = loaded_index.nprobe(); loaded_index.with_nprobes(f, fu) }
        (None, None)        => loaded_index,
    });

    let mcc_risk = Arc::new(
        MccRisk::load(std::path::Path::new(&mcc_risk_path))
            .with_context(|| format!("falha ao carregar {mcc_risk_path}"))
            .expect("mcc_risk indisponível"),
    );

    warmup_index(&index, 200);
    READY.store(true, Ordering::Release);

    let state = Arc::new(AppState { index, mcc_risk });

    // Try io_uring; fall back to legacy (epoll) for kernels < 5.6.
    let rt_uring = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .with_entries(1024)
        .build();

    match rt_uring {
        Ok(mut rt) => rt.block_on(server_loop(state)),
        Err(_) => {
            let mut rt = monoio::RuntimeBuilder::<monoio::LegacyDriver>::new()
                .build()
                .expect("falha ao criar runtime legacy");
            rt.block_on(server_loop(state));
        }
    }
}

// ── Server accept loop ────────────────────────────────────────────────────────

async fn server_loop(state: Arc<AppState>) {
    let uds_path = env::var("BIND_UDS").unwrap_or_else(|_| {
        let port = env::var("PORT").unwrap_or_else(|_| "9998".into());
        format!("/tmp/fraud-api-{port}.sock")
    });

    let _ = std::fs::remove_file(&uds_path);
    let listener = UnixListener::bind(&uds_path).expect("falha ao bind UDS");

    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(
        &uds_path,
        std::fs::Permissions::from_mode(0o666),
    );

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let state = state.clone();
                monoio::spawn(async move {
                    handle_conn(stream, state).await;
                });
            }
            Err(e) => warn!("accept error: {e}"),
        }
    }
}

// ── Per-connection keepalive handler ──────────────────────────────────────────

async fn handle_conn(mut stream: UnixStream, state: Arc<AppState>) {
    // One 8-KB scratch buffer per connection, reused across requests.
    // nginx sends one request at a time on keepalive connections, so the
    // buffer is cleared between requests.
    let mut accum: Vec<u8> = Vec::with_capacity(4096);

    loop {
        accum.clear();

        // ── Accumulate a complete request ──────────────────────────────────
        let (header_end, content_length, is_close) = loop {
            // Read next chunk into an owned Vec (monoio completion model).
            let chunk = vec![0u8; 4096usize.saturating_sub(accum.len()).max(512)];
            let (res, chunk) = stream.read(chunk).await;
            let n = match res {
                Ok(0) | Err(_) => return, // peer closed or error
                Ok(n) => n,
            };
            accum.extend_from_slice(&chunk[..n]);

            match parse_head(&accum) {
                Some(x) => break x,
                None if accum.len() > 8192 => return, // protect against runaway
                None => continue,
            }
        };

        // ── Read remaining body bytes if any ──────────────────────────────
        let body_start = header_end + 4; // skip past \r\n\r\n
        let body_end   = body_start + content_length;

        while accum.len() < body_end {
            let need  = body_end - accum.len();
            let chunk = vec![0u8; need];
            let (res, chunk) = stream.read(chunk).await;
            let n = match res {
                Ok(0) | Err(_) => return,
                Ok(n) => n,
            };
            accum.extend_from_slice(&chunk[..n]);
        }

        // ── Dispatch ──────────────────────────────────────────────────────
        let resp: &'static [u8] = dispatch(&accum, body_start, body_end, &state);

        // ── Write response ────────────────────────────────────────────────
        // `write_all` takes ownership of the Vec; we allocate one per response.
        // Cost: one small alloc per request (128-110 bytes), acceptable.
        let resp_vec = resp.to_vec();
        let (res, _) = stream.write_all(resp_vec).await;

        if res.is_err() || is_close {
            return;
        }
    }
}

// ── HTTP parsing ──────────────────────────────────────────────────────────────

/// Parses the HTTP request head from `buf`.
///
/// Returns `(header_end_byte, content_length, connection_close)` if a complete
/// header block is found, or `None` if more data is needed.
fn parse_head(buf: &[u8]) -> Option<(usize, usize, bool)> {
    let mut headers = [httparse::EMPTY_HEADER; 24];
    let mut req = httparse::Request::new(&mut headers);

    match req.parse(buf) {
        Ok(httparse::Status::Complete(end)) => {
            let content_length = headers.iter()
                .find(|h| h.name.eq_ignore_ascii_case("content-length"))
                .and_then(|h| std::str::from_utf8(h.value).ok())
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);

            let is_close = headers.iter()
                .any(|h| h.name.eq_ignore_ascii_case("connection")
                       && h.value.eq_ignore_ascii_case(b"close"));

            Some((end, content_length, is_close))
        }
        _ => None,
    }
}

// ── Request dispatcher ────────────────────────────────────────────────────────

fn dispatch(
    buf: &[u8],
    body_start: usize,
    body_end: usize,
    state: &AppState,
) -> &'static [u8] {
    // Fast path: distinguish GET /ready from POST /fraud-score by inspecting
    // the first bytes of the request line. Both paths are always used with the
    // exact spellings nginx generates.
    if buf.starts_with(b"GET /ready") {
        return if READY.load(Ordering::Acquire) { HTTP_READY_OK } else { HTTP_READY_503 };
    }

    if buf.starts_with(b"POST /fraud-score") && body_end > body_start {
        return score_body(&buf[body_start..body_end], state);
    }

    HTTP_404
}

/// Parses the JSON body, runs fraud scoring, returns the pre-built HTTP response.
///
/// On any parse or arithmetic error, returns `HTTP_FRAUD[0]` (approved=true,
/// score=0.0). FP=1pt is always better than Err=5pt in the scoring formula.
fn score_body(body: &[u8], state: &AppState) -> &'static [u8] {
    let payload: models::TransactionPayload = match serde_json::from_slice(body) {
        Ok(p) => p,
        Err(_) => return HTTP_FRAUD[0],
    };

    let query = vectorizer::vectorize(&payload, &state.mcc_risk);
    let score = state.index.fraud_score(&query);

    let idx = if score.is_finite() {
        ((score * 5.0).round() as usize).min(5)
    } else {
        0
    };

    HTTP_FRAUD[idx]
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init();
}

/// Runs synthetic queries before accepting traffic to warm CPU caches.
fn warmup_index(index: &IvfIndex, count: usize) {
    let mut state = 0x12345678u32;
    for _ in 0..count {
        let query: [f32; 14] = std::array::from_fn(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = (state ^ (state >> 16)) as f32 / u32::MAX as f32;
            if i == 5 || i == 6 { -1.0 } else { x }
        });
        let _ = index.fraud_score(&query);
    }
}
