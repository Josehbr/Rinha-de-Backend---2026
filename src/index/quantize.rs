use crate::index::layout::N_DIMS;

/// Fixed quantization scale: f32 values in [-1.0, 1.0] map to i16 in [-10000, 10000].
///
/// Sentinel -1.0 → -10_000. Normal values [0.0, 1.0] → [0, 10_000].
/// max |diff| between any two quantized values = 20_000 < i16::MAX (32_767), so
/// i16 subtraction never overflows.
pub const QUANT_SCALE: f32 = 10_000.0;

/// Quantizes a single f32 feature value to i16.
#[inline]
pub fn quantize_i16(v: f32) -> i16 {
    (v * QUANT_SCALE).round() as i16
}

/// Quantizes a 14-dim f32 query vector to i16 for the SIMD distance kernel.
#[inline]
pub fn quantize_query(query: &[f32; N_DIMS]) -> [i16; N_DIMS] {
    std::array::from_fn(|i| quantize_i16(query[i]))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinel_quantizes_to_minus_ten_thousand() {
        assert_eq!(quantize_i16(-1.0), -10_000);
    }

    #[test]
    fn zero_quantizes_to_zero() {
        assert_eq!(quantize_i16(0.0), 0);
    }

    #[test]
    fn one_quantizes_to_ten_thousand() {
        assert_eq!(quantize_i16(1.0), 10_000);
    }

    #[test]
    fn round_trip_precision() {
        // With QUANT_SCALE = 10_000, error ≤ 0.5 / 10_000 = 0.00005
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            let back = quantize_i16(v) as f32 / QUANT_SCALE;
            assert!(
                (back - v).abs() < 0.0001,
                "round-trip failed for v={v}: got {back}"
            );
        }
    }

    #[test]
    fn query_quantizes_all_dims() {
        let q = [0.5f32; N_DIMS];
        let qi = quantize_query(&q);
        for &v in &qi {
            assert_eq!(v, 5_000);
        }
    }

    #[test]
    fn max_diff_fits_in_i16() {
        // sentinel vs max: -10_000 - 10_000 = -20_000, fits in i16 (range ±32_767)
        let a = quantize_i16(-1.0);
        let b = quantize_i16(1.0);
        let diff = a as i32 - b as i32;
        assert_eq!(diff, -20_000);
        assert!(diff >= i16::MIN as i32 && diff <= i16::MAX as i32);
    }

    #[test]
    fn half_quantizes_correctly() {
        assert_eq!(quantize_i16(0.5), 5_000);
    }
}
