mod index;
mod models;
mod vectorizer;

use std::env;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use index::IvfIndex;
use models::MccRisk;
use tokio::io::{AsyncReadExt, AsyncWriteExt, Interest};
use tokio::net::{TcpStream, UnixListener, UnixStream};
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
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

static HTTP_READY_OK: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}";
static HTTP_READY_503: &[u8] =
    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
static HTTP_404: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    init_tracing();

    let index_path    = env::var("INDEX_PATH").unwrap_or_else(|_| "./index.bin".into());
    let mcc_risk_path = env::var("MCC_RISK_PATH")
        .unwrap_or_else(|_| "./resources/mcc_risk.json".into());

    let index = Arc::new(
        IvfIndex::load(std::path::Path::new(&index_path))
            .with_context(|| format!("falha ao carregar {index_path}"))
            .map_err(|e| { warn!("{e}"); e })
            .expect("índice indisponível"),
    );

    let mcc_risk = Arc::new(
        MccRisk::load(std::path::Path::new(&mcc_risk_path))
            .with_context(|| format!("falha ao carregar {mcc_risk_path}"))
            .expect("mcc_risk indisponível"),
    );

    // mlockall AFTER mmap so MCL_CURRENT only locks what's already resident —
    // avoids EAGAIN when MAP_POPULATE pre-faults pages while a global memlock
    // limit (RLIMIT_MEMLOCK) is in effect. Best-effort: silently ignored if
    // the container lacks CAP_IPC_LOCK or has a memlock ulimit too low.
    unsafe {
        let _ = libc::mlockall(libc::MCL_CURRENT);
    }

    // 5000 queries: enough to drag the bbox arrays + a representative slice of
    // VectorBlocks into L1/L2, plus exercises the sort-and-prune hot path.
    warmup_index(&index, 5000);
    READY.store(true, Ordering::Release);

    let state = Arc::new(AppState { index, mcc_risk });

    // Single-threaded tokio: no context-switch overhead between tasks.
    // With 0.4 CPU, one thread handles all async I/O cooperatively.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("falha ao criar tokio runtime");

    rt.block_on(server_loop(state));
}

// ── Control-socket server: receives accepted TCP fds from fraud-lb ────────────

async fn server_loop(state: Arc<AppState>) {
    let ctrl_path = env::var("CTRL_UDS").unwrap_or_else(|_| "/sockets/api1.ctrl".into());

    let _ = std::fs::remove_file(&ctrl_path);
    let listener = UnixListener::bind(&ctrl_path)
        .unwrap_or_else(|e| panic!("falha ao bind {ctrl_path}: {e}"));

    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(
        &ctrl_path,
        std::fs::Permissions::from_mode(0o666),
    );

    // The LB connects exactly once at boot and keeps the socket open for the
    // process lifetime. If it ever disconnects (crash/restart), we loop back
    // and accept a fresh connection — the same control socket file persists.
    loop {
        match listener.accept().await {
            Ok((lb_conn, _)) => {
                recv_fds_loop(lb_conn, state.clone()).await;
                warn!("LB connection closed; awaiting reconnect");
            }
            Err(e) => warn!("ctrl accept error: {e}"),
        }
    }
}

/// Reads client fds from the LB control socket and spawns one task per fd.
///
/// Returns when the LB closes its side of the control socket — the outer
/// `server_loop` will then accept a new LB connection on the same path.
async fn recv_fds_loop(lb_conn: UnixStream, state: Arc<AppState>) {
    loop {
        match recv_fd_async(&lb_conn).await {
            Ok(fd) => {
                // SAFETY: the LB just transferred ownership of `fd` to us via
                // SCM_RIGHTS; the kernel guarantees it's a valid open file.
                let std_stream = unsafe { std::net::TcpStream::from_raw_fd(fd) };
                if let Err(e) = std_stream.set_nonblocking(true) {
                    warn!("set_nonblocking failed: {e}");
                    continue;
                }
                match TcpStream::from_std(std_stream) {
                    Ok(tcp) => {
                        let _ = tcp.set_nodelay(true);
                        let state = state.clone();
                        tokio::spawn(handle_conn(tcp, state));
                    }
                    Err(e) => warn!("TcpStream::from_std: {e}"),
                }
            }
            Err(e) => {
                warn!("recv_fd: {e}");
                return;
            }
        }
    }
}

/// Awaits readiness on the control socket and pulls one fd out of an
/// `SCM_RIGHTS` ancillary message via blocking `recvmsg`.
async fn recv_fd_async(stream: &UnixStream) -> std::io::Result<RawFd> {
    loop {
        stream.readable().await?;
        match stream.try_io(Interest::READABLE, || recv_fd_sync(stream.as_raw_fd())) {
            Ok(fd) => return Ok(fd),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(e),
        }
    }
}

fn recv_fd_sync(socket_fd: RawFd) -> std::io::Result<RawFd> {
    use nix::sys::socket::{ControlMessageOwned, MsgFlags, recvmsg};

    let mut payload = [0u8; 16];
    let mut iov = [std::io::IoSliceMut::new(&mut payload)];
    let mut cmsg_space = nix::cmsg_space!([RawFd; 1]);
    let msg = recvmsg::<()>(socket_fd, &mut iov, Some(&mut cmsg_space), MsgFlags::empty())
        .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;
    if msg.bytes == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionAborted,
            "LB closed control socket",
        ));
    }
    for cm in msg
        .cmsgs()
        .map_err(|e| std::io::Error::other(format!("cmsg parse: {e}")))?
    {
        if let ControlMessageOwned::ScmRights(fds) = cm {
            if let Some(&fd) = fds.first() {
                return Ok(fd);
            }
        }
    }
    Err(std::io::Error::other("no fd in SCM_RIGHTS cmsg"))
}

// ── Per-connection keepalive handler (now over TcpStream) ────────────────────

async fn handle_conn(mut stream: TcpStream, state: Arc<AppState>) {
    let mut accum: Vec<u8> = Vec::with_capacity(4096);
    let mut chunk_buf = [0u8; 4096]; // stack buffer — evita malloc por chunk

    loop {
        accum.clear();

        // ── Accumulate complete request ────────────────────────────────────
        let (header_end, content_length, is_close) = loop {
            let cap = 4096usize.saturating_sub(accum.len()).max(512);
            let read_buf = &mut chunk_buf[..cap];
            let n = match stream.read(read_buf).await {
                Ok(0) | Err(_) => return,
                Ok(n) => n,
            };
            accum.extend_from_slice(&chunk_buf[..n]);

            match parse_head(&accum) {
                Some(x) => break x,
                None if accum.len() > 8192 => return,
                None => continue,
            }
        };

        // ── Read remaining body bytes ──────────────────────────────────────
        let body_start = header_end;
        let body_end   = body_start + content_length;

        while accum.len() < body_end {
            let need  = body_end - accum.len();
            let read_buf = &mut chunk_buf[..need.min(4096)];
            let n = match stream.read(read_buf).await {
                Ok(0) | Err(_) => return,
                Ok(n) => n,
            };
            accum.extend_from_slice(&chunk_buf[..n]);
        }

        // ── Dispatch ──────────────────────────────────────────────────────
        let resp: &'static [u8] = dispatch(&accum, body_start, body_end, &state);

        // ── Write response ────────────────────────────────────────────────
        if stream.write_all(resp).await.is_err() || is_close {
            return;
        }
    }
}

// ── HTTP parsing ──────────────────────────────────────────────────────────────

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

fn dispatch(buf: &[u8], body_start: usize, body_end: usize, state: &AppState) -> &'static [u8] {
    if buf.starts_with(b"GET /ready") {
        return if READY.load(Ordering::Acquire) { HTTP_READY_OK } else { HTTP_READY_503 };
    }

    if buf.starts_with(b"POST /fraud-score") && body_end > body_start {
        return score_body(&buf[body_start..body_end], state);
    }

    HTTP_404
}

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
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

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
