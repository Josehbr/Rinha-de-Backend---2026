use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::Context;
use serde::{Deserialize, Serialize};

// ── Request types ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TransactionPayload {
    pub id: String,
    pub transaction: TransactionInfo,
    pub customer: CustomerInfo,
    pub merchant: MerchantInfo,
    pub terminal: TerminalInfo,
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Debug, Deserialize)]
pub struct TransactionInfo {
    pub amount: f32,
    pub installments: u32,
    // ISO 8601 UTC, ex: "2026-03-11T18:45:53Z" — parsed manually in vectorizer
    pub requested_at: String,
}

#[derive(Debug, Deserialize)]
pub struct CustomerInfo {
    pub avg_amount: f32,
    pub tx_count_24h: u32,
    // Vec, not HashSet: duplicates appear in real data (e.g. "MERC-009" twice)
    pub known_merchants: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct MerchantInfo {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Debug, Deserialize)]
pub struct TerminalInfo {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Debug, Deserialize)]
pub struct LastTransaction {
    pub timestamp: String,
    pub km_from_current: f32,
}

// ── Response types ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct FraudDecision {
    pub approved: bool,
    pub fraud_score: f32,
}

impl FraudDecision {
    /// Returns an approved decision with zero score.
    /// Used on internal errors: FP costs 1pt, HTTP 500 costs 5pt in scoring.
    pub fn safe_default() -> Self {
        Self { approved: true, fraud_score: 0.0 }
    }
}

// ── MCC risk table ────────────────────────────────────────────────────────────

/// Merchant Category Code risk scores loaded from mcc_risk.json.
pub struct MccRisk(HashMap<String, f32>);

impl MccRisk {
    /// Loads the MCC risk table from a JSON file.
    ///
    /// Example: `MccRisk::load(Path::new("resources/mcc_risk.json"))?`
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("mcc_risk.json not found at {}", path.display()))?;
        let map: HashMap<String, f32> = serde_json::from_reader(file)
            .context("failed to parse mcc_risk.json")?;
        Ok(Self(map))
    }

    /// Returns the risk score for a MCC code.
    /// Unknown MCCs return 0.5 (mid-range default, per competition spec).
    pub fn score(&self, mcc: &str) -> f32 {
        self.0.get(mcc).copied().unwrap_or(0.5)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const PAYLOAD_NO_LAST_TX: &str = r#"{
        "id": "tx-1329056812",
        "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
        "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
        "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
        "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.2331036248 },
        "last_transaction": null
    }"#;

    const PAYLOAD_WITH_LAST_TX: &str = r#"{
        "id": "tx-3576980410",
        "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
        "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-009", "MERC-001", "MERC-001"] },
        "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
        "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.7090520965 },
        "last_transaction": { "timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8626479774 }
    }"#;

    #[test]
    fn deserialize_payload_without_last_transaction() {
        let payload: TransactionPayload = serde_json::from_str(PAYLOAD_NO_LAST_TX).unwrap();

        assert!(payload.last_transaction.is_none());
        assert_eq!(payload.id, "tx-1329056812");
        assert!((payload.transaction.amount - 41.12).abs() < 0.001);
        assert_eq!(payload.transaction.installments, 2);
        assert_eq!(payload.customer.known_merchants.len(), 2);
        assert_eq!(payload.merchant.mcc, "5411");
        assert!(!payload.terminal.is_online);
        assert!(payload.terminal.card_present);
    }

    #[test]
    fn deserialize_payload_with_last_transaction() {
        let payload: TransactionPayload = serde_json::from_str(PAYLOAD_WITH_LAST_TX).unwrap();

        let last_tx = payload.last_transaction.as_ref().unwrap();
        assert!((last_tx.km_from_current - 18.862).abs() < 0.001);
        assert_eq!(last_tx.timestamp, "2026-03-11T14:58:35Z");
        // Vec preserves duplicates: "MERC-009" appears twice
        assert_eq!(payload.customer.known_merchants.len(), 4);
    }

    #[test]
    fn fraud_decision_safe_default() {
        let d = FraudDecision::safe_default();
        assert!(d.approved);
        assert_eq!(d.fraud_score, 0.0);
    }

    #[test]
    fn mcc_risk_known_and_unknown_codes() {
        let risk = MccRisk::load(Path::new("resources/mcc_risk.json")).unwrap();

        assert!((risk.score("5411") - 0.15).abs() < 0.0001);
        assert!((risk.score("7995") - 0.85).abs() < 0.0001);
        // Unknown MCC → 0.5 default
        assert_eq!(risk.score("9999"), 0.5);
    }
}
