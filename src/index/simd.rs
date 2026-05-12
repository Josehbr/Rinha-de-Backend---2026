use crate::index::layout::{BLOCK_SIZE, N_DIMS, VectorBlock};

/// Scalar reference implementation of block scan.
///
/// Computes the L2² distance from `query_i16` to each of the BLOCK_SIZE vectors
/// stored in SoA layout inside `block`. Returns `[f32; BLOCK_SIZE]`.
///
/// Distance is computed in quantized i16 space (no dequantization needed since
/// we only need relative ordering for top-k selection).
pub fn scan_block_scalar(query_i16: &[i16; N_DIMS], block: &VectorBlock) -> [f32; BLOCK_SIZE] {
    let mut acc = [0.0f32; BLOCK_SIZE];
    for d in 0..N_DIMS {
        let q = query_i16[d] as f32;
        for slot in 0..BLOCK_SIZE {
            let b = block.data[d * BLOCK_SIZE + slot] as f32;
            let diff = b - q;
            acc[slot] += diff * diff;
        }
    }
    acc
}

/// AVX2 block scan: computes L2² distances from `query_i16` to all BLOCK_SIZE
/// vectors in `block` simultaneously, using 256-bit FMA instructions.
///
/// Uses 4 independent accumulators to hide Haswell's 5-cycle FMA latency
/// (throughput 1/cycle). Each accumulator handles every 4th dimension, so
/// there are 3 independent FMAs between consecutive writes to the same acc.
///
/// Early exit: after processing the first 8 dimensions, if ALL 8 partial
/// distances already exceed `threshold`, returns `[f32::MAX; 8]` — the remaining
/// 6 dimensions can only increase distances.
///
/// Accumulation in f32: 14 × (20_000)² = 5.6×10⁹ > i32::MAX, needs f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_block_avx2(
    query_i16: &[i16; N_DIMS],
    block: &VectorBlock,
    threshold: f32,
) -> [f32; BLOCK_SIZE] {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees avx2+fma available. All pointer arithmetic
    // stays within the 112-element `data` array (N_DIMS × BLOCK_SIZE = 14 × 8).
    unsafe {
        // 4 independent accumulators to break FMA dependency chain.
        // acc0 ← dims 0,4,8,12  acc1 ← dims 1,5,9,13
        // acc2 ← dims 2,6,10    acc3 ← dims 3,7,11
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Inline macro to avoid repeating load/extend/sub/fma for each dim.
        macro_rules! acc_dim {
            ($acc:expr, $d:expr) => {{
                let q = _mm256_set1_ps(query_i16[$d] as f32);
                let raw = _mm_loadu_si128(
                    block.data.as_ptr().add($d * BLOCK_SIZE) as *const __m128i,
                );
                let b = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(raw));
                let diff = _mm256_sub_ps(b, q);
                $acc = _mm256_fmadd_ps(diff, diff, $acc);
            }};
        }

        // First 4 dims: rotate through 4 accumulators (one dim each, breaks FMA chain).
        acc_dim!(acc0, 0);
        acc_dim!(acc1, 1);
        acc_dim!(acc2, 2);
        acc_dim!(acc3, 3);

        // Early exit @4 dims: check if ALL 8 slots already exceed threshold.
        // 4 dimensions is enough to distinguish far blocks when Top5 threshold
        // is tight (late clusters in the probe sequence).
        let partial4 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let thresh_v = _mm256_set1_ps(threshold);
        if _mm256_movemask_ps(_mm256_cmp_ps(partial4, thresh_v, _CMP_GT_OS)) == 0xFF {
            return [f32::MAX; BLOCK_SIZE];
        }

        // Dims 4-7: rotate through 4 accumulators.
        acc_dim!(acc0, 4);
        acc_dim!(acc1, 5);
        acc_dim!(acc2, 6);
        acc_dim!(acc3, 7);

        // Early exit @8 dims on partial sum of all 4 accumulators.
        let partial = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        if _mm256_movemask_ps(_mm256_cmp_ps(partial, thresh_v, _CMP_GT_OS)) == 0xFF {
            return [f32::MAX; BLOCK_SIZE];
        }

        // Remaining 6 dims: 8..14 (acc2 and acc3 get one fewer dim each).
        acc_dim!(acc0, 8); acc_dim!(acc1,  9); acc_dim!(acc2, 10); acc_dim!(acc3, 11);
        acc_dim!(acc0, 12); acc_dim!(acc1, 13);

        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut out = [0.0f32; BLOCK_SIZE];
        _mm256_storeu_ps(out.as_mut_ptr(), acc);
        out
    }
}

/// Computes L2² distances from `query_i16` to all BLOCK_SIZE vectors in `block`.
///
/// Dispatches to AVX2+FMA if available, otherwise scalar fallback.
/// `threshold`: early-exit hint — if all partial distances after 8 dims already
/// exceed this, returns `[f32::MAX; BLOCK_SIZE]`.
pub fn scan_block(
    query_i16: &[i16; N_DIMS],
    block: &VectorBlock,
    threshold: f32,
) -> [f32; BLOCK_SIZE] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { scan_block_avx2(query_i16, block, threshold) };
        }
    }
    let _ = threshold; // scalar path doesn't implement early exit
    scan_block_scalar(query_i16, block)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::quantize::quantize_i16;

    fn make_block(vecs: &[[i16; N_DIMS]]) -> VectorBlock {
        assert!(vecs.len() <= BLOCK_SIZE);
        let mut block = VectorBlock::default();
        for (slot, vec) in vecs.iter().enumerate() {
            for d in 0..N_DIMS {
                block.data[d * BLOCK_SIZE + slot] = vec[d];
            }
        }
        block
    }

    fn uniform_block(val: i16) -> VectorBlock {
        let vecs: Vec<[i16; N_DIMS]> = (0..BLOCK_SIZE).map(|_| [val; N_DIMS]).collect();
        make_block(&vecs)
    }

    #[test]
    fn zero_distance_to_identical_vector() {
        let block = uniform_block(5_000);
        let query = [5_000i16; N_DIMS];
        let dists = scan_block_scalar(&query, &block);
        for d in dists {
            assert_eq!(d, 0.0, "distance to identical vector must be 0");
        }
    }

    #[test]
    fn known_distance_scalar() {
        // All block vectors are [0; 14], query is [1; 14] in i16 units
        // dist = 14 × (1 - 0)² = 14.0
        let block = uniform_block(0);
        let query = [1i16; N_DIMS];
        let dists = scan_block_scalar(&query, &block);
        for d in dists {
            assert!((d - 14.0).abs() < 1e-3, "expected 14.0, got {d}");
        }
    }

    #[test]
    fn avx2_matches_scalar_for_various_inputs() {
        let test_cases: &[([i16; N_DIMS], i16)] = &[
            ([0i16; N_DIMS], 1),
            ([10_000i16; N_DIMS], 0),
            ([-10_000i16; N_DIMS], 10_000),
            ([5_000i16; N_DIMS], -5_000),
        ];

        for (query_arr, block_val) in test_cases {
            let block = uniform_block(*block_val);
            let scalar = scan_block_scalar(query_arr, &block);
            let fast   = scan_block(query_arr, &block, f32::MAX);
            for slot in 0..BLOCK_SIZE {
                assert!(
                    (scalar[slot] - fast[slot]).abs() < 1.0,
                    "mismatch slot={slot}: scalar={} avx2={} (query={:?}, block_val={block_val})",
                    scalar[slot], fast[slot], query_arr
                );
            }
        }
    }

    #[test]
    fn sentinel_same_position_contributes_zero() {
        // Both query and block have sentinel (-10_000) at all dims → dist = 0
        let block = uniform_block(-10_000);
        let query = [-10_000i16; N_DIMS];
        let dists = scan_block(&query, &block, f32::MAX);
        for d in dists {
            assert_eq!(d, 0.0, "sentinel-vs-sentinel should be 0");
        }
    }

    #[test]
    fn early_exit_returns_max_when_all_above_threshold() {
        // Block all-zeros, query all 10_000 → partial after 8 dims: 8 × 10_000² = 8×10⁸
        let block = uniform_block(0);
        let query = [10_000i16; N_DIMS];
        let very_low_threshold = 1.0; // below any real partial distance
        let dists = scan_block(&query, &block, very_low_threshold);
        for d in dists {
            assert_eq!(d, f32::MAX, "all distances should be f32::MAX on early exit");
        }
    }

    #[test]
    fn early_exit_not_triggered_when_some_below_threshold() {
        // Block slot 0: [0; 14] (small distance from query [1; 14])
        // Block slots 1..8: [10_000; 14] (large distance)
        let mut vecs = vec![[10_000i16; N_DIMS]; BLOCK_SIZE];
        vecs[0] = [0i16; N_DIMS];
        let block = make_block(&vecs);
        let query = [1i16; N_DIMS];
        let threshold = 1_000_000.0; // large threshold — no early exit

        let dists = scan_block(&query, &block, threshold);
        // slot 0: 14 × (0 - 1)² = 14.0
        assert!((dists[0] - 14.0).abs() < 1.0, "slot 0 distance should be ~14: {}", dists[0]);
    }

    #[test]
    fn scan_block_quantized_values_match() {
        // Use real-world-like values: amount=0.5, others=0.3
        let q_f32 = {
            let mut v = [0.3f32; N_DIMS];
            v[0] = 0.5;
            v
        };
        let query: [i16; N_DIMS] = std::array::from_fn(|i| quantize_i16(q_f32[i]));

        let b_f32 = [0.3f32; N_DIMS];
        let b_i16: [i16; N_DIMS] = std::array::from_fn(|i| quantize_i16(b_f32[i]));
        let block = make_block(&[b_i16; BLOCK_SIZE]);

        let scalar = scan_block_scalar(&query, &block);
        let fast   = scan_block(&query, &block, f32::MAX);
        for slot in 0..BLOCK_SIZE {
            assert!((scalar[slot] - fast[slot]).abs() < 1.0, "slot {slot}");
        }
    }
}
