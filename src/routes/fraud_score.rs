use actix_web::{HttpResponse, web};

use crate::models::{FraudDecision, TransactionPayload};
use crate::vectorizer::vectorize;
use crate::AppState;

/// Fraud scoring endpoint.
///
/// Any internal failure must degrade to `FraudDecision::safe_default()` with HTTP 200.
pub async fn fraud_score_handler(
    state: web::Data<AppState>,
    payload: web::Json<TransactionPayload>,
) -> HttpResponse {
    let query = vectorize(&payload, &state.mcc_risk);
    let score = state.index.fraud_score(&query);
    let decision = if score.is_finite() {
        FraudDecision {
            approved: score < 0.6,
            fraud_score: score,
        }
    } else {
        FraudDecision::safe_default()
    };

    HttpResponse::Ok().json(decision)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use actix_web::{App, http, test, web};
    use bytemuck::bytes_of;

    use super::*;
    use crate::index::IvfIndex;
    use crate::index::layout::{BLOCK_SIZE, IndexHeader, MAGIC, N_DIMS, VERSION, VectorBlock};
    use crate::models::MccRisk;

    fn unique_tmp_file(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{nanos}.bin"))
    }

    /// Builds a minimal valid index.bin in the VectorBlock (SoA int16, VERSION=2) format.
    /// 1 cluster with BLOCK_SIZE (8) vectors; labels [1,1,0,0,0,0,0,0].
    fn build_tiny_index_file(path: &std::path::Path) {
        let n_padded = BLOCK_SIZE; // exactly 1 block
        let nlist: u32 = 1;
        let n_blocks = 1usize;

        let hdr_size = std::mem::size_of::<IndexHeader>();
        let centroids_sz = nlist as usize * N_DIMS * 4;
        let csizes_sz = nlist as usize * 4;
        let csizes_end = hdr_size + centroids_sz + csizes_sz;
        let blocks_start = crate::index::layout::align_up(csizes_end, 32);
        let blocks_sz = n_blocks * std::mem::size_of::<VectorBlock>();
        let labels_start = blocks_start + blocks_sz;
        let total = labels_start + n_padded;

        let n_vb = crate::index::layout::align_up(total, std::mem::size_of::<VectorBlock>())
            / std::mem::size_of::<VectorBlock>();
        let mut backing: Vec<VectorBlock> = vec![VectorBlock::default(); n_vb];

        {
            let buf: &mut [u8] = bytemuck::cast_slice_mut(&mut backing);

            let header = IndexHeader {
                magic: MAGIC,
                version: VERSION,
                nlist,
                nprobe: 1,
                n_vectors: n_padded as u64,
                _padding: [0u8; 8],
            };
            buf[..hdr_size].copy_from_slice(bytes_of(&header));

            let csizes: &mut [u32] = bytemuck::cast_slice_mut(
                &mut buf[hdr_size + centroids_sz..hdr_size + centroids_sz + csizes_sz],
            );
            csizes[0] = n_padded as u32;

            // Labels: [fraud, fraud, legit, legit, legit, legit, legit, legit]
            buf[labels_start] = 1;
            buf[labels_start + 1] = 1;
        }

        let raw = bytemuck::cast_slice::<VectorBlock, u8>(&backing);
        let mut file = fs::File::create(path).expect("create tiny index file");
        file.write_all(&raw[..total]).expect("write tiny index file");
        file.flush().expect("flush tiny index file");
    }

    #[actix_web::test]
    async fn fraud_score_returns_json_content_type_and_200() {
        let tmp = unique_tmp_file("tiny-handler-index");
        build_tiny_index_file(&tmp);

        let index = Arc::new(IvfIndex::load(&tmp).expect("load tiny index"));
        let mcc_risk = Arc::new(
            MccRisk::load(std::path::Path::new("resources/mcc_risk.json")).expect("load mcc risk"),
        );
        let state = web::Data::new(crate::AppState { index, mcc_risk });

        let mut app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/fraud-score", web::post().to(fraud_score_handler)),
        )
        .await;

        let body = fs::read_to_string("resources/example-payloads.json")
            .expect("read example payloads");
        let payloads: serde_json::Value = serde_json::from_str(&body).expect("parse payloads");
        let first_payload = payloads[0].clone();

        let req = test::TestRequest::post()
            .uri("/fraud-score")
            .set_json(&first_payload)
            .to_request();
        let resp = test::call_service(&mut app, req).await;

        assert_eq!(resp.status(), http::StatusCode::OK);
        let content_type = resp
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .unwrap_or_default();
        assert!(
            content_type.starts_with("application/json"),
            "content-type inesperado: {content_type}"
        );

        let _ = fs::remove_file(&tmp);
    }
}
