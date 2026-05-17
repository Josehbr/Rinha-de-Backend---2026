mod index;
mod models;
mod vectorizer;

use std::env;
use std::io::{ErrorKind, IoSliceMut, Read, Write};
use std::net::TcpStream;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use anyhow::Context;
use index::IvfIndex;
use models::MccRisk;
use tracing::warn;
use tracing_subscriber::EnvFilter;

pub static READY: AtomicBool = AtomicBool::new(false);

const WORKER_STACK_BYTES: usize = 256 * 1024;
const WORKER_RT_PRIO: libc::c_int = 10;
const READ_BUF_SIZE: usize = 4096;

struct AppState {
    index:           Arc<IvfIndex>,
    mcc_risk:        Arc<MccRisk>,
    /// When `true` and SCHED_FIFO was granted, workers stay at FIFO only while
    /// blocked in `recv`. Compute+send runs as SCHED_OTHER so a busy worker
    /// does not outrank softirqs / the test client. Mirrors Ronie #2.
    rt_wakeup_only:  bool,
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
    // avoids EAGAIN when MAP_POPULATE pre-faults pages while a low RLIMIT_MEMLOCK
    // is in effect. Best-effort: silently ignored if denied.
    unsafe {
        let _ = libc::mlockall(libc::MCL_CURRENT);
    }

    warmup_index(&index, 5000);
    READY.store(true, Ordering::Release);

    // CPU affinity (best-effort): pin this process + future workers to a single
    // core to avoid cross-core cache misses and reduce scheduler jitter.
    set_cpu_affinity_from_env();

    // SCHED_FIFO setup (best-effort): if granted, new threads created via
    // `pthread_create` inherit FIFO scheduling — workers wake up immediately
    // when their socket becomes readable (no runqueue wait).
    let rt_granted = setup_sched_fifo();
    let want_wakeup_only = env::var("RINHA_RT_MODE").as_deref() == Ok("wakeup");
    let rt_wakeup_only = want_wakeup_only && rt_granted;

    eprintln!(
        "fraud-api: RT mode = {} (prio {})",
        if !rt_granted { "none/SCHED_OTHER" } else if rt_wakeup_only { "wakeup-only" } else { "all" },
        WORKER_RT_PRIO
    );

    let state = Arc::new(AppState { index, mcc_risk, rt_wakeup_only });

    serve_control(state);
}

// ── Control-socket server: receives accepted TCP fds from fraud-lb ────────────

fn serve_control(state: Arc<AppState>) -> ! {
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
        match listener.accept() {
            Ok((lb_conn, _)) => {
                recv_fds_loop(lb_conn, state.clone());
                warn!("LB connection closed; awaiting reconnect");
            }
            Err(e) => warn!("ctrl accept error: {e}"),
        }
    }
}

/// Reads client fds from the LB control socket and spawns one worker thread
/// per fd. Returns when the LB closes its side of the control socket.
fn recv_fds_loop(lb_conn: UnixStream, state: Arc<AppState>) {
    let lb_fd = lb_conn.as_raw_fd();
    loop {
        match recv_fd_blocking(lb_fd) {
            Ok(fd) => {
                // SAFETY: the LB just transferred ownership of `fd` to us via
                // SCM_RIGHTS; the kernel guarantees it's a valid open file.
                let stream = unsafe { TcpStream::from_raw_fd(fd) };
                // CRITICAL: LB is tokio (sets fd nonblocking on accept).
                // SCM_RIGHTS passes the fd verbatim, so without flipping it
                // back, std::net::TcpStream::read returns WouldBlock immediately
                // and the connection drops — ~1000 phantom errors per k6 run.
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_nodelay(true);

                let st = state.clone();
                let spawned = thread::Builder::new()
                    .name("worker".into())
                    .stack_size(WORKER_STACK_BYTES)
                    .spawn(move || handle_conn(stream, st));

                if let Err(e) = spawned {
                    warn!("spawn worker failed: {e}");
                    // `stream` was moved into the closure on success path; on
                    // failure it's dropped here, closing the client fd.
                }
            }
            Err(e) => {
                warn!("recv_fd: {e}");
                return;
            }
        }
    }
}

/// Blocking `recvmsg` that returns the first fd from an `SCM_RIGHTS` ancillary
/// message. Returns `ConnectionAborted` when the peer closed the control socket.
fn recv_fd_blocking(ctrl_fd: RawFd) -> std::io::Result<RawFd> {
    use nix::sys::socket::{ControlMessageOwned, MsgFlags, recvmsg};

    let mut payload = [0u8; 16];
    let mut iov = [IoSliceMut::new(&mut payload)];
    let mut cmsg_space = nix::cmsg_space!([RawFd; 1]);
    let msg = recvmsg::<()>(ctrl_fd, &mut iov, Some(&mut cmsg_space), MsgFlags::empty())
        .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;
    if msg.bytes == 0 {
        return Err(std::io::Error::new(
            ErrorKind::ConnectionAborted,
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

// ── Per-connection blocking handler ──────────────────────────────────────────

fn handle_conn(mut stream: TcpStream, state: Arc<AppState>) {
    let mut buf = [0u8; READ_BUF_SIZE];
    let mut buf_pos: usize = 0;

    loop {
        // Blocking read — the kernel wakes us when bytes are available. With
        // SCHED_FIFO this wakeup is preemptive and skips the runqueue wait.
        let n = match stream.read(&mut buf[buf_pos..]) {
            Ok(0) => return,
            Ok(n) => n,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(_) => return,
        };

        // Wakeup-only mode: drop to SCHED_OTHER now that we've won the wakeup
        // race; compute+send won't outrank softirqs or the test client.
        if state.rt_wakeup_only {
            worker_set_rt(false);
        }

        buf_pos += n;

        // Drain every complete request currently in `buf`. HTTP/1.1 pipelining
        // can deliver multiple in one read; serve them in order.
        let mut drained_at_least_one = false;
        loop {
            let Some((header_end, content_length, is_close)) = parse_head(&buf[..buf_pos]) else {
                if buf_pos >= READ_BUF_SIZE {
                    return; // oversized header → drop the connection
                }
                break;
            };
            let body_end = header_end + content_length;

            // Read remaining body bytes if not all in buffer yet.
            while buf_pos < body_end {
                if body_end > READ_BUF_SIZE {
                    return; // body bigger than buffer — drop
                }
                let n = match stream.read(&mut buf[buf_pos..body_end]) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => n,
                };
                buf_pos += n;
            }

            let resp = dispatch(&buf[..body_end], header_end, body_end, &state);
            if stream.write_all(resp).is_err() {
                return;
            }
            drained_at_least_one = true;
            if is_close {
                return;
            }

            let leftover = buf_pos - body_end;
            if leftover > 0 {
                buf.copy_within(body_end..body_end + leftover, 0);
            }
            buf_pos = leftover;
        }

        let _ = drained_at_least_one;

        // Restore FIFO before blocking in recv again — the next request's
        // wakeup needs to skip the runqueue too.
        if state.rt_wakeup_only {
            worker_set_rt(true);
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

// ── Realtime scheduling helpers ───────────────────────────────────────────────

/// Sets SCHED_FIFO on the calling thread (main). Returns true if granted.
///
/// Subsequent `thread::spawn` calls inherit scheduling from the parent (default
/// `PTHREAD_INHERIT_SCHED`), so worker threads will also be SCHED_FIFO.
///
/// Tries libc wrapper first, falls back to a direct `syscall` — some musl
/// builds + Docker seccomp combos reject the wrapper but accept the raw call.
fn setup_sched_fifo() -> bool {
    unsafe {
        let mut param: libc::sched_param = std::mem::zeroed();
        param.sched_priority = WORKER_RT_PRIO;
        let rc = libc::sched_setscheduler(0, libc::SCHED_FIFO, &param);
        if rc == 0 {
            return true;
        }
        let err1 = *libc::__errno_location();

        // Some setups (musl static + Docker seccomp default) reject the libc
        // wrapper but accept the raw syscall. SYS_sched_setscheduler = 144 on
        // x86_64. Args: pid=0 (self), policy=SCHED_FIFO=1, param ptr.
        let rc2 = libc::syscall(
            libc::SYS_sched_setscheduler,
            0i32,
            libc::SCHED_FIFO,
            &param as *const _,
        );
        if rc2 == 0 {
            eprintln!("fraud-api: SCHED_FIFO via raw syscall (wrapper errno {err1})");
            return true;
        }
        let err2 = *libc::__errno_location();
        eprintln!("fraud-api: SCHED_FIFO denied (wrapper errno {err1}, syscall errno {err2})");
        false
    }
}

/// Switches the *calling thread* between SCHED_FIFO (on) and SCHED_OTHER (off).
///
/// Used in wakeup-only mode: workers spend `recv` blocked at SCHED_FIFO so the
/// kernel wakes them preemptively, then switch to SCHED_OTHER for compute+send.
/// Uses raw syscall — see `setup_sched_fifo` for the libc-wrapper rationale.
fn worker_set_rt(on: bool) {
    unsafe {
        let mut param: libc::sched_param = std::mem::zeroed();
        param.sched_priority = if on { WORKER_RT_PRIO } else { 0 };
        let policy = if on { libc::SCHED_FIFO } else { libc::SCHED_OTHER };
        let _ = libc::syscall(
            libc::SYS_sched_setscheduler,
            0i32,
            policy,
            &param as *const _,
        );
    }
}

/// Pins this process (and inherited worker threads) to the CPU specified by
/// `RINHA_CPU` (e.g. "0", "2"). No-op if the env var is missing or invalid.
fn set_cpu_affinity_from_env() {
    let Ok(cpu_str) = env::var("RINHA_CPU") else { return };
    let Ok(cpu_id) = cpu_str.trim().parse::<usize>() else { return };

    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu_id, &mut set);
        let _ = libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set);
    }
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
