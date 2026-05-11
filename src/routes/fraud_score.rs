use actix_web::{HttpResponse, web};
use actix_web::http::header;

use crate::models::TransactionPayload;
use crate::vectorizer::vectorize;
use crate::AppState;

// Pre-rendered JSON responses for all 6 possible fraud decisions.
// fraud_score ∈ {0/5, 1/5, 2/5, 3/5, 4/5, 5/5} = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
// approved = score < 0.6  →  indices 0,1,2 → approved:true; 3,4,5 → approved:false.
static RESPONSES: [&[u8]; 6] = [
    br#"{"approved":true,"fraud_score":0.0}"#,
    br#"{"approved":true,"fraud_score":0.2}"#,
    br#"{"approved":true,"fraud_score":0.4}"#,
    br#"{"approved":false,"fraud_score":0.6}"#,
    br#"{"approved":false,"fraud_score":0.8}"#,
    br#"{"approved":false,"fraud_score":1.0}"#,
];

/// Fraud scoring endpoint.
///
/// Any internal failure must degrade to `RESPONSES[0]` (approved:true, score:0.0)
/// with HTTP 200. FP costs 1pt; HTTP 500 costs 5pt — safe_default always wins.
pub async fn fraud_score_handler(
    state: web::Data<AppState>,
    payload: web::Json<TransactionPayload>,
) -> HttpResponse {
    let query = vectorize(&payload, &state.mcc_risk);
    let score = state.index.fraud_score(&query);

    // Map score ∈ {0.0,0.2,0.4,0.6,0.8,1.0} → index 0..5.
    // Non-finite (NaN/Inf) → index 0 (safe default: approved=true, score=0.0).
    let idx = if score.is_finite() {
        ((score * 5.0).round() as usize).min(5)
    } else {
        0
    };

    HttpResponse::Ok()
        .insert_header((header::CONTENT_TYPE, "application/json"))
        .body(actix_web::web::Bytes::from_static(RESPONSES[idx]))
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
