use std::sync::atomic::Ordering;

use actix_web::{HttpResponse, Responder};
use serde::Serialize;

#[derive(Serialize)]
struct ReadyOk {
    status: &'static str,
}

/// Healthcheck endpoint consumed by nginx before routing traffic.
pub async fn ready_handler() -> impl Responder {
    if crate::READY.load(Ordering::Acquire) {
        HttpResponse::Ok().json(ReadyOk { status: "ok" })
    } else {
        HttpResponse::ServiceUnavailable().finish()
    }
}

#[cfg(test)]
mod tests {
    use actix_web::http::StatusCode;

    use super::*;

    #[actix_web::test]
    async fn ready_returns_503_when_not_ready() {
        crate::READY.store(false, Ordering::Release);
        let resp = ready_handler().await.respond_to(&actix_web::test::TestRequest::default().to_http_request());
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[actix_web::test]
    async fn ready_returns_ok_when_ready() {
        crate::READY.store(true, Ordering::Release);
        let resp = ready_handler().await.respond_to(&actix_web::test::TestRequest::default().to_http_request());
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
