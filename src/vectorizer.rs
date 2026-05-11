use crate::models::{MccRisk, TransactionPayload};

// ── Normalisation constants (hardcoded — never change during a test run) ──────

const MAX_AMOUNT: f32              = 10_000.0;
const MAX_INSTALLMENTS: f32        = 12.0;
const AMOUNT_VS_AVG_RATIO: f32     = 10.0;
const MAX_MINUTES: f32             = 1_440.0;
const MAX_KM: f32                  = 1_000.0;
const MAX_TX_COUNT_24H: f32        = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

// Sentinel for absent last_transaction (indices 5 and 6).
// Outside [0, 1] on purpose — groups no-history transactions together in KNN space.
const SENTINEL: f32 = -1.0;

// ── Public interface ──────────────────────────────────────────────────────────

/// Transforms a transaction payload into the 14-dimensional feature vector.
///
/// ```ignore
/// let v = vectorize(&payload, &mcc_risk);
/// // v[5] == -1.0 when last_transaction is null (sentinel)
/// ```
pub fn vectorize(payload: &TransactionPayload, mcc_risk: &MccRisk) -> [f32; 14] {
    let t    = &payload.transaction;
    let c    = &payload.customer;
    let m    = &payload.merchant;
    let term = &payload.terminal;
    let last = payload.last_transaction.as_ref();

    [
        // [0] amount — raw value relative to max observed amount
        clamp01(t.amount / MAX_AMOUNT),
        // [1] installments — 12 instalments maps to 1.0
        clamp01(t.installments as f32 / MAX_INSTALLMENTS),
        // [2] amount_vs_avg — how many multiples of the customer's average this is
        clamp01((t.amount / c.avg_amount) / AMOUNT_VS_AVG_RATIO),
        // [3] hour_of_day — UTC hour normalised to [0, 1]
        hour_utc(&t.requested_at) as f32 / 23.0,
        // [4] day_of_week — Mon=0 … Sun=6, normalised to [0, 1]
        weekday_utc(&t.requested_at) as f32 / 6.0,
        // [5] minutes_since_last_tx — SENTINEL when no prior transaction exists
        last.map_or(SENTINEL, |l| {
            clamp01(minutes_between(&l.timestamp, &t.requested_at) / MAX_MINUTES)
        }),
        // [6] km_from_last_tx — SENTINEL when no prior transaction exists
        last.map_or(SENTINEL, |l| clamp01(l.km_from_current / MAX_KM)),
        // [7] km_from_home
        clamp01(term.km_from_home / MAX_KM),
        // [8] tx_count_24h — 20+ transactions in 24h maps to 1.0
        clamp01(c.tx_count_24h as f32 / MAX_TX_COUNT_24H),
        // [9] is_online
        term.is_online as u8 as f32,
        // [10] card_present
        term.card_present as u8 as f32,
        // [11] unknown_merchant — 1.0 means the merchant is NOT in the customer's known list
        (!c.known_merchants.iter().any(|k| k == &m.id)) as u8 as f32,
        // [12] mcc_risk — from mcc_risk.json; 0.5 default for unknown MCCs
        mcc_risk.score(&m.mcc),
        // [13] merchant_avg_amount
        clamp01(m.avg_amount / MAX_MERCHANT_AVG_AMOUNT),
    ]
}

// ── Private helpers ───────────────────────────────────────────────────────────

#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Parses a 2–4 digit ASCII decimal slice into a u32.
fn parse_digits(b: &[u8]) -> Option<u32> {
    if b.is_empty() {
        return None;
    }
    let mut acc = 0u32;
    for &d in b {
        if !d.is_ascii_digit() {
            return None;
        }
        acc = acc.saturating_mul(10).saturating_add((d - b'0') as u32);
    }
    Some(acc)
}

#[inline]
fn parse_range_digits(iso: &str, start: usize, end: usize) -> Option<u32> {
    let b = iso.as_bytes();
    parse_digits(b.get(start..end)?)
}

/// Extracts the UTC hour (0–23) from an ISO 8601 string.
/// Input format: "2026-03-11T18:45:53Z" — hour is always at bytes [11..13].
fn hour_utc(iso: &str) -> u32 {
    parse_range_digits(iso, 11, 13).filter(|h| *h <= 23).unwrap_or(0)
}

/// Returns the day of the week as Mon=0 … Sun=6.
/// Input format: "2026-03-11T18:45:53Z" — date is always at bytes [0..10].
///
/// Uses Sakamoto's algorithm (public domain) which returns Sun=0 … Sat=6,
/// then maps to the competition convention with `(day + 6) % 7`.
fn weekday_utc(iso: &str) -> u32 {
    let y = parse_range_digits(iso, 0, 4).unwrap_or(1970) as i32;
    let mo = parse_range_digits(iso, 5, 7).unwrap_or(1) as i32;
    let d = parse_range_digits(iso, 8, 10).unwrap_or(1) as i32;

    // Sakamoto lookup table (months 1–12, 0-indexed)
    let t: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if mo < 3 { y - 1 } else { y };
    let month_idx = mo.clamp(1, 12) - 1;
    let sakamoto = (y + y/4 - y/100 + y/400 + t[month_idx as usize] + d) % 7;

    // Sakamoto: 0=Sun … 6=Sat  →  competition: 0=Mon … 6=Sun
    ((sakamoto + 6) % 7) as u32
}

/// Converts an ISO 8601 UTC timestamp to seconds since the Unix epoch.
/// Input format: "2026-03-11T18:45:53Z" (exactly 20 bytes, always UTC).
fn iso8601_to_secs(iso: &str) -> i64 {
    let y = parse_range_digits(iso, 0, 4).unwrap_or(1970) as i64;
    let mo = parse_range_digits(iso, 5, 7).unwrap_or(1).clamp(1, 12) as i64;
    let d = parse_range_digits(iso, 8, 10).unwrap_or(1).clamp(1, 31) as i64;
    let h = parse_range_digits(iso, 11, 13).unwrap_or(0).clamp(0, 23) as i64;
    let mi = parse_range_digits(iso, 14, 16).unwrap_or(0).clamp(0, 59) as i64;
    let s = parse_range_digits(iso, 17, 19).unwrap_or(0).clamp(0, 59) as i64;

    let days = days_since_unix_epoch(y, mo, d);
    days * 86_400 + h * 3_600 + mi * 60 + s
}

/// Converts a Gregorian date to days since 1970-01-01 via Julian Day Number.
fn days_since_unix_epoch(y: i64, m: i64, d: i64) -> i64 {
    // JDN formula (astronomical algorithms, Meeus)
    let a     = (14 - m) / 12;
    let y_adj = y + 4_800 - a;
    let m_adj = m + 12 * a - 3;
    let jdn   = d + (153 * m_adj + 2) / 5
              + 365 * y_adj
              + y_adj / 4
              - y_adj / 100
              + y_adj / 400
              - 32_045;
    jdn - 2_440_588  // 2_440_588 = JDN of 1970-01-01
}

/// Returns elapsed minutes from `from_iso` to `to_iso` as f32.
/// Assumes to >= from (current transaction is always after the last one).
fn minutes_between(from_iso: &str, to_iso: &str) -> f32 {
    let diff_secs = iso8601_to_secs(to_iso) - iso8601_to_secs(from_iso);
    (diff_secs / 60) as f32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MccRisk, TransactionPayload};
    use std::path::Path;

    fn load_mcc_risk() -> MccRisk {
        MccRisk::load(Path::new("resources/mcc_risk.json")).unwrap()
    }

    fn assert_vec_approx(got: [f32; 14], expected: [f32; 14], tol: f32) {
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < tol,
                "dim[{i}]: got {g:.6}, expected {e:.6} (diff {:.6})",
                (g - e).abs()
            );
        }
    }

    // Payloads hardcoded from CLAUDE.md examples (verified against expected vectors)
    const LEGIT_JSON: &str = r#"{
        "id": "tx-1329056812",
        "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
        "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
        "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
        "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.2331036248 },
        "last_transaction": null
    }"#;

    const FRAUD_JSON: &str = r#"{
        "id": "tx-3330991687",
        "transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
        "customer": { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008","MERC-007","MERC-005"] },
        "merchant": { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
        "terminal": { "is_online": false, "card_present": true, "km_from_home": 952.27 },
        "last_transaction": null
    }"#;

    const WITH_LAST_TX_JSON: &str = r#"{
        "id": "tx-3576980410",
        "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
        "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009","MERC-001"] },
        "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
        "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.7090520965 },
        "last_transaction": { "timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8626479774 }
    }"#;

    #[test]
    fn vectorize_legit_payload_matches_expected() {
        let payload: TransactionPayload = serde_json::from_str(LEGIT_JSON).unwrap();
        let mcc = load_mcc_risk();
        let got = vectorize(&payload, &mcc);

        // Expected values from CLAUDE.md (verified manually for each dimension)
        let expected = [
            0.004112,  // [0]  41.12 / 10000
            0.166667,  // [1]  2 / 12
            0.05,      // [2]  (41.12/82.24)/10 = 0.5/10
            18.0/23.0, // [3]  hour 18
            2.0/6.0,   // [4]  Wednesday (2026-03-11)
            -1.0,      // [5]  sentinel
            -1.0,      // [6]  sentinel
            0.029233,  // [7]  29.2331/1000
            0.15,      // [8]  3/20
            0.0,       // [9]  is_online=false
            1.0,       // [10] card_present=true
            0.0,       // [11] MERC-016 is known
            0.15,      // [12] mcc_risk["5411"]
            0.006025,  // [13] 60.25/10000
        ];

        assert_vec_approx(got, expected, 0.0001);
    }

    #[test]
    fn vectorize_fraud_payload_matches_expected() {
        let payload: TransactionPayload = serde_json::from_str(FRAUD_JSON).unwrap();
        let mcc = load_mcc_risk();
        let got = vectorize(&payload, &mcc);

        let expected = [
            0.950597,  // [0]  9505.97/10000
            10.0/12.0, // [1]  10/12
            1.0,       // [2]  clamped: (9505.97/81.28)/10 >> 1.0
            5.0/23.0,  // [3]  hour 5
            5.0/6.0,   // [4]  Saturday (2026-03-14)
            -1.0,      // [5]  sentinel
            -1.0,      // [6]  sentinel
            0.95227,   // [7]  952.27/1000
            1.0,       // [8]  20/20
            0.0,       // [9]  is_online=false
            1.0,       // [10] card_present=true
            1.0,       // [11] MERC-068 is unknown
            0.75,      // [12] mcc_risk["7802"]
            0.005486,  // [13] 54.86/10000
        ];

        assert_vec_approx(got, expected, 0.0001);
    }

    #[test]
    fn vectorize_dims_5_and_6_with_last_transaction() {
        let payload: TransactionPayload = serde_json::from_str(WITH_LAST_TX_JSON).unwrap();
        let mcc = load_mcc_risk();
        let got = vectorize(&payload, &mcc);

        // 20:23:35 − 14:58:35 = 5h25m = 325 minutes
        let expected_dim5 = 325.0_f32 / 1440.0;
        // km_from_current = 18.8626479774
        let expected_dim6 = 18.8626479774_f32 / 1000.0;

        assert!((got[5] - expected_dim5).abs() < 0.0001, "dim[5] got {:.6}", got[5]);
        assert!((got[6] - expected_dim6).abs() < 0.0001, "dim[6] got {:.6}", got[6]);
        // Sentinels must NOT appear when last_transaction is present
        assert!(got[5] >= 0.0, "dim[5] must not be sentinel");
        assert!(got[6] >= 0.0, "dim[6] must not be sentinel");
    }

    #[test]
    fn hour_utc_extracts_correct_values() {
        assert_eq!(hour_utc("2026-03-11T18:45:53Z"), 18);
        assert_eq!(hour_utc("2026-03-14T00:00:00Z"), 0);
        assert_eq!(hour_utc("2026-03-14T05:15:12Z"), 5);
        assert_eq!(hour_utc("2026-03-11T23:59:59Z"), 23);
    }

    #[test]
    fn weekday_utc_returns_correct_competition_values() {
        // 2026-03-11 = Wednesday = 2 (Mon=0)
        assert_eq!(weekday_utc("2026-03-11T18:45:53Z"), 2);
        // 2026-03-14 = Saturday = 5
        assert_eq!(weekday_utc("2026-03-14T05:15:12Z"), 5);
        // 2026-03-15 = Sunday = 6
        assert_eq!(weekday_utc("2026-03-15T00:00:00Z"), 6);
        // 2026-03-16 = Monday = 0
        assert_eq!(weekday_utc("2026-03-16T00:00:00Z"), 0);
        // 2026-03-17 = Tuesday = 1
        assert_eq!(weekday_utc("2026-03-17T02:04:06Z"), 1);
    }

    #[test]
    fn clamp01_boundary_values() {
        assert_eq!(clamp01(-1.0), 0.0);
        assert_eq!(clamp01(0.0),  0.0);
        assert_eq!(clamp01(0.5),  0.5);
        assert_eq!(clamp01(1.0),  1.0);
        assert_eq!(clamp01(1.5),  1.0);
    }
}
