mod index;
mod models;
mod routes;
mod vectorizer;

use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use actix_web::{App, HttpServer, web};
use anyhow::Context;
use index::IvfIndex;
use models::MccRisk;
use tracing::warn;
use tracing_subscriber::EnvFilter;

pub static READY: AtomicBool = AtomicBool::new(false);

pub struct AppState {
    pub index: Arc<IvfIndex>,
    pub mcc_risk: Arc<MccRisk>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    if let Err(err) = init_tracing() {
        eprintln!("erro ao inicializar tracing: {err}");
    }

    let index_path = env::var("INDEX_PATH").unwrap_or_else(|_| "./index.bin".to_string());
    let mcc_risk_path =
        env::var("MCC_RISK_PATH").unwrap_or_else(|_| "./resources/mcc_risk.json".to_string());
    let port = env::var("PORT")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(9998);
    let nprobe_override = env::var("NPROBE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0);
    let full_nprobe_override = env::var("FULL_NPROBE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0);
    let workers = env::var("WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or_else(|| num_cpus::get().min(2));

    let loaded_index = IvfIndex::load(std::path::Path::new(&index_path))
        .with_context(|| format!("falha ao carregar índice em {index_path}"))
        .map_err(to_io_error)?;
    let index = Arc::new(match (nprobe_override, full_nprobe_override) {
        (Some(fast), Some(full)) => loaded_index.with_nprobes(fast, full),
        (Some(fast), None) => loaded_index.with_nprobes(fast, fast * 3),
        (None, Some(full)) => {
            let fast = loaded_index.nprobe();
            loaded_index.with_nprobes(fast, full)
        }
        (None, None) => loaded_index,
    });
    let mcc_risk = Arc::new(
        MccRisk::load(std::path::Path::new(&mcc_risk_path))
            .with_context(|| format!("falha ao carregar mcc_risk em {mcc_risk_path}"))
            .map_err(to_io_error)?,
    );

    READY.store(true, Ordering::Release);

    let state = web::Data::new(AppState { index, mcc_risk });

    let server = HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/ready", web::get().to(routes::ready::ready_handler))
            .route(
                "/fraud-score",
                web::post().to(routes::fraud_score::fraud_score_handler),
            )
    })
    .workers(workers);

    // If BIND_UDS is set, bind a Unix Domain Socket (chmod 0666 so nginx can read).
    // Otherwise bind 0.0.0.0:PORT as before.
    let server = if let Ok(uds_path) = env::var("BIND_UDS") {
        let _ = std::fs::remove_file(&uds_path);
        let bound = server.bind_uds(&uds_path)?;
        let _ = std::fs::set_permissions(
            &uds_path,
            <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o666),
        );
        bound
    } else {
        let bind_addr = format!("0.0.0.0:{port}");
        server.bind(&bind_addr)?
    };

    server.run().await
}

fn init_tracing() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .map_err(|err| anyhow::anyhow!("falha ao inicializar tracing: {err}"))?;
    Ok(())
}

fn to_io_error(err: anyhow::Error) -> std::io::Error {
    warn!(error = %err, "startup failure");
    std::io::Error::other(err.to_string())
}
