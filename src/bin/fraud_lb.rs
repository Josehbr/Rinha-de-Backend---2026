//! Minimal TCP→UDS load balancer for the fraud-api stack.
//!
//! Listens on `:9999` (configurable via `LB_PORT`), connects to each API's
//! control socket once at boot, then for every accepted TCP client passes the
//! raw fd to one of the APIs via `sendmsg + SCM_RIGHTS` and closes its own
//! reference. The API adopts the fd and speaks HTTP/1.1 directly with the
//! client. This eliminates the per-request parse + proxy that nginx would do.
//!
//! Round-robin across all configured control sockets. WORKERS=1 single-thread
//! tokio current_thread runtime — the LB does only `accept + sendmsg`, no
//! payload work, so one core is more than enough at 900 req/s.

use std::env;
use std::io::IoSlice;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::time::Duration;

use anyhow::{Context, Result};
use nix::sys::socket::{ControlMessage, MsgFlags, sendmsg};
use tokio::net::{TcpListener, TcpStream, UnixStream};
use tracing::warn;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();

    let listen_addr = env::var("LB_LISTEN").unwrap_or_else(|_| "0.0.0.0:9999".into());
    let api_socks = env::var("LB_API_SOCKS")
        .unwrap_or_else(|_| "/sockets/api1.ctrl,/sockets/api2.ctrl".into());

    let api_paths: Vec<String> = api_socks.split(',').map(|s| s.trim().to_owned()).collect();
    anyhow::ensure!(!api_paths.is_empty(), "LB_API_SOCKS empty");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime")?;

    rt.block_on(serve(listen_addr, api_paths))
}

async fn serve(listen_addr: String, api_paths: Vec<String>) -> Result<()> {
    let ctrl_fds = connect_all_apis(&api_paths).await?;

    let listener = TcpListener::bind(&listen_addr)
        .await
        .with_context(|| format!("bind {listen_addr}"))?;

    let mut rr: usize = 0;
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let _ = stream.set_nodelay(true);
                let target_idx = rr % ctrl_fds.len();
                rr = rr.wrapping_add(1);
                if let Err(e) = handoff_fd(&stream, ctrl_fds[target_idx].as_raw_fd()) {
                    warn!("handoff failed (target idx {target_idx}): {e}");
                }
                // `stream` drops here → kernel decrements refcount; the API
                // now owns its copy of the fd via SCM_RIGHTS.
            }
            Err(e) => warn!("accept error: {e}"),
        }
    }
}

/// Connects (with retry) to every API control socket so the APIs are ready
/// to accept fds before the listener opens.
async fn connect_all_apis(paths: &[String]) -> Result<Vec<OwnedFd>> {
    let mut fds = Vec::with_capacity(paths.len());
    for path in paths {
        let fd = connect_with_retry(path).await?;
        fds.push(fd);
    }
    Ok(fds)
}

async fn connect_with_retry(path: &str) -> Result<OwnedFd> {
    for attempt in 0..120 {
        match UnixStream::connect(path).await {
            Ok(stream) => {
                let std_stream = stream
                    .into_std()
                    .with_context(|| format!("UnixStream::into_std for {path}"))?;
                let raw = std_stream.into_raw_fd();
                // SAFETY: raw came from into_raw_fd above, so we own it.
                return Ok(unsafe { OwnedFd::from_raw_fd(raw) });
            }
            Err(_) if attempt < 119 => {
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            Err(e) => {
                return Err(e).with_context(|| format!("connect to {path} after retries"));
            }
        }
    }
    anyhow::bail!("exhausted retries connecting to {path}")
}

/// Sends `stream`'s underlying TCP fd to the control socket via SCM_RIGHTS.
///
/// A single zero byte of payload is required: SOCK_STREAM SCM_RIGHTS won't
/// deliver an ancillary message without at least one byte of data attached.
fn handoff_fd(stream: &TcpStream, ctrl_fd: RawFd) -> Result<()> {
    let client_fd = stream.as_raw_fd();
    let payload = [0u8; 1];
    let iov = [IoSlice::new(&payload)];
    let fds = [client_fd];
    let cmsgs = [ControlMessage::ScmRights(&fds)];
    sendmsg::<()>(ctrl_fd, &iov, &cmsgs, MsgFlags::empty(), None)
        .context("sendmsg SCM_RIGHTS")?;
    Ok(())
}
