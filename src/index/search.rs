use std::collections::BinaryHeap;

use crate::index::layout::{BLOCK_SIZE, N_DIMS, VectorBlock};
use crate::index::simd::scan_block;

// ── Top5 ──────────────────────────────────────────────────────────────────────

/// Fixed-capacity top-k buffer for k=5, sorted ascending by distance.
///
/// Replaces `BinaryHeap<(DistF32, u8)>` on the hot path. k=5 is so small that
/// an insertion-sorted array beats the heap: no allocations, no indirection,
/// and all data fits in a single cache line (5 × 5B ≈ 40B).
pub struct Top5 {
    /// (distance, label) pairs, sorted ascending (nearest first).
    data: [(f32, u8); 5],
    pub len: usize,
}

impl Top5 {
    #[inline]
    pub fn new() -> Self {
        Self { data: [(f32::MAX, 0); 5], len: 0 }
    }

    /// Distance of the worst (farthest) element, or f32::MAX when not full.
    #[inline]
    pub fn worst(&self) -> f32 {
        if self.len < 5 { f32::MAX } else { self.data[4].0 }
    }

    /// Insert `(dist, label)` if it improves the current top-5.
    #[inline]
    pub fn try_insert(&mut self, dist: f32, label: u8) {
        if self.len < 5 {
            // Find insertion point (ascending order).
            let mut pos = self.len;
            while pos > 0 && self.data[pos - 1].0 > dist {
                pos -= 1;
            }
            for i in (pos..self.len).rev() {
                self.data[i + 1] = self.data[i];
            }
            self.data[pos] = (dist, label);
            self.len += 1;
        } else if dist < self.data[4].0 {
            // Evict worst, find insertion point among first 4.
            let mut pos = 4;
            while pos > 0 && self.data[pos - 1].0 > dist {
                pos -= 1;
            }
            for i in (pos..4).rev() {
                self.data[i + 1] = self.data[i];
            }
            self.data[pos] = (dist, label);
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Number of fraud labels (label=1) among the current top-k items.
    #[inline]
    pub fn count_fraud(&self) -> usize {
        self.data[..self.len].iter().filter(|&&(_, l)| l == 1).count()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(f32, u8)> {
        self.data[..self.len].iter()
    }
}

// ── DistF32 ───────────────────────────────────────────────────────────────────

/// Wrapper for non-NaN f32 that implements `Ord`, enabling use as a `BinaryHeap` key.
///
/// Used in a max-heap of (distance, label) pairs: largest distance at the top
/// so we can evict the worst candidate when a closer vector arrives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistF32(pub f32);

impl PartialOrd for DistF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // SAFETY invariant: distances produced by scan_block are never NaN.
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Eq for DistF32 {}

// ── Cluster selection ─────────────────────────────────────────────────────────

/// Returns the indices of the `nprobe` centroids closest to `query` (f32 L2²).
///
/// Uses `select_nth_unstable_by` for O(n) average selection — no full sort.
/// Centroids (4096 × 14 × 4B = 229 KB) fit in Haswell's L2 cache.
pub fn find_nearest_clusters(
    query: &[f32; N_DIMS],
    centroids: &[[f32; N_DIMS]],
    nprobe: usize,
) -> Vec<usize> {
    let k = nprobe.min(centroids.len());
    if k == 0 {
        return vec![];
    }

    let mut idx = vec![0usize; k];
    let mut dists = vec![0.0f32; k];
    let used = find_nearest_clusters_inplace(query, centroids, nprobe, &mut idx, &mut dists);
    idx.truncate(used);
    idx
}

/// Same as `find_nearest_clusters`, but writes into caller-provided stack buffers.
///
/// Avoids allocating a `Vec<(dist, idx)>` on the hot path.
pub fn find_nearest_clusters_inplace(
    query: &[f32; N_DIMS],
    centroids: &[[f32; N_DIMS]],
    nprobe: usize,
    out_indices: &mut [usize],
    out_dists: &mut [f32],
) -> usize {
    let k = nprobe
        .min(centroids.len())
        .min(out_indices.len())
        .min(out_dists.len());
    if k == 0 {
        return 0;
    }

    for i in 0..k {
        out_indices[i] = i;
        out_dists[i] = l2sq_f32(query, &centroids[i]);
    }
    let mut worst_pos = max_pos(&out_dists[..k]);

    for (idx, centroid) in centroids.iter().enumerate().skip(k) {
        let d = l2sq_f32(query, centroid);
        if d < out_dists[worst_pos] {
            out_dists[worst_pos] = d;
            out_indices[worst_pos] = idx;
            worst_pos = max_pos(&out_dists[..k]);
        }
    }

    sort_by_dist(out_indices, out_dists, k);
    k
}

// ── Block scan ────────────────────────────────────────────────────────────────

/// Scans a slice of VectorBlocks and updates a global bounded max-heap of top-k
/// `(distance, label)` results.
///
/// `query_i16`: quantized query in i16 space.
/// `blocks`: the VectorBlocks for this cluster.
/// `labels`: label per vector slot, length = blocks.len() × BLOCK_SIZE.
/// `k`: heap capacity.
/// `heap`: shared heap across probed clusters — avoids per-cluster allocations.
pub fn update_top_k_blocks(
    query_i16: &[i16; N_DIMS],
    blocks: &[VectorBlock],
    labels: &[u8],
    k: usize,
    heap: &mut BinaryHeap<(DistF32, u8)>,
) {
    let mut threshold = if heap.len() >= k {
        heap.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX)
    } else {
        f32::MAX
    };

    for (b_idx, block) in blocks.iter().enumerate() {
        let dists = scan_block(query_i16, block, threshold);

        for slot in 0..BLOCK_SIZE {
            let vec_idx = b_idx * BLOCK_SIZE + slot;
            if vec_idx >= labels.len() {
                break;
            }

            let d = dists[slot];
            // f32::MAX is the early-exit sentinel; skip these.
            if d == f32::MAX {
                continue;
            }

            let dist = DistF32(d);
            let label = labels[vec_idx];

            if heap.len() < k {
                heap.push((dist, label));
            } else if let Some(&(worst, _)) = heap.peek() {
                if dist < worst {
                    heap.pop();
                    heap.push((dist, label));
                }
            }
        }

        // Update threshold after each block so the early exit in the next
        // scan_block call prunes as aggressively as possible.
        if heap.len() >= k {
            if let Some(&(worst, _)) = heap.peek() {
                threshold = worst.0;
            }
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Computes the L2² distance between two 14-dim f32 vectors.
///
/// Dispatches to AVX2+FMA on x86_64 when available:
///   - dims 0-7  → 256-bit `_mm256_fmadd_ps`
///   - dims 8-11 → 128-bit `_mm_fmadd_ps` (via `_mm_mul_ps` + `_mm_add_ps`)
///   - dims 12-13 → scalar
/// This reduces compute cost ~3x vs scalar and improves ILP via FMA pipeline.
#[inline]
fn l2sq_f32(a: &[f32; N_DIMS], b: &[f32; N_DIMS]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2sq_f32_avx2(a, b) };
        }
    }
    l2sq_f32_scalar(a, b)
}

#[inline]
fn l2sq_f32_scalar(a: &[f32; N_DIMS], b: &[f32; N_DIMS]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..N_DIMS {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// AVX2+FMA implementation of L2² for 14-dim f32 vectors.
///
/// Processes dims 0-7 with 256-bit, dims 8-11 with 128-bit, dims 12-13 scalar.
/// Uses 4 independent accumulators to hide 5-cycle FMA latency on Haswell.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2sq_f32_avx2(a: &[f32; N_DIMS], b: &[f32; N_DIMS]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        // Dims 0-7: 256-bit, 2 independent accumulators to break FMA dep chain.
        let a_lo = _mm256_loadu_ps(a.as_ptr());
        let b_lo = _mm256_loadu_ps(b.as_ptr());
        let diff_lo = _mm256_sub_ps(a_lo, b_lo);

        // Split into low 128 (dims 0-3) and high 128 (dims 4-7) for 2 accumulators.
        let diff_lo_lo = _mm256_castps256_ps128(diff_lo);   // dims 0-3
        let diff_lo_hi = _mm256_extractf128_ps(diff_lo, 1); // dims 4-7

        let mut acc0 = _mm_mul_ps(diff_lo_lo, diff_lo_lo);  // dims 0-3 (acc)
        let acc1 = _mm_mul_ps(diff_lo_hi, diff_lo_hi);  // dims 4-7 (acc)

        // Dims 8-11: 128-bit.
        let a_hi = _mm_loadu_ps(a.as_ptr().add(8));
        let b_hi = _mm_loadu_ps(b.as_ptr().add(8));
        let diff_hi = _mm_sub_ps(a_hi, b_hi);
        acc0 = _mm_fmadd_ps(diff_hi, diff_hi, acc0); // reuse acc0 (independent from dims 0-3 done)

        // Dims 12-13: scalar.
        let d12 = a[12] - b[12];
        let d13 = a[13] - b[13];
        let scalar = d12 * d12 + d13 * d13;

        // Horizontal sum of acc0 + acc1 (4+4 = 8 lanes).
        let sum128 = _mm_add_ps(acc0, acc1);
        // Reduce 4 lanes to 1.
        let shuf = _mm_movehdup_ps(sum128);           // [1,1,3,3]
        let sums = _mm_add_ps(sum128, shuf);           // [0+1, _, 2+3, _]
        let shuf2 = _mm_movehl_ps(sums, sums);         // [2+3, ...]
        let total = _mm_add_ss(sums, shuf2);           // [0+1+2+3]

        _mm_cvtss_f32(total) + scalar
    }
}

#[inline]
fn max_pos(values: &[f32]) -> usize {
    let mut max_i = 0usize;
    for i in 1..values.len() {
        if values[i] > values[max_i] {
            max_i = i;
        }
    }
    max_i
}

fn sort_by_dist(indices: &mut [usize], dists: &mut [f32], len: usize) {
    for i in 1..len {
        let idx_key  = indices[i];
        let dist_key = dists[i];
        let mut j = i;
        while j > 0 && dists[j - 1] > dist_key {
            indices[j] = indices[j - 1];
            dists[j]   = dists[j - 1];
            j -= 1;
        }
        indices[j] = idx_key;
        dists[j]   = dist_key;
    }
}

// ── Hot-path block scan (uses Top5) ──────────────────────────────────────────

/// Scans VectorBlocks for the hot path, updating `top5` in place.
///
/// Uses `Top5` instead of `BinaryHeap` to avoid heap overhead for k=5.
/// `update_top_k_blocks` (BinaryHeap) is kept for `search_knn` and tests.
pub fn update_top5_blocks(
    query_i16: &[i16; N_DIMS],
    blocks: &[VectorBlock],
    labels: &[u8],
    top5: &mut Top5,
) {
    let mut threshold = top5.worst();

    for (b_idx, block) in blocks.iter().enumerate() {
        let dists = scan_block(query_i16, block, threshold);

        for slot in 0..BLOCK_SIZE {
            let vec_idx = b_idx * BLOCK_SIZE + slot;
            if vec_idx >= labels.len() {
                break;
            }
            let d = dists[slot];
            if d == f32::MAX {
                continue;
            }
            top5.try_insert(d, labels[vec_idx]);
        }

        threshold = top5.worst();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::kmeans::kmeans;
    use crate::index::quantize::{quantize_i16, quantize_query};

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_block(vecs: &[[i16; N_DIMS]]) -> VectorBlock {
        let mut block = VectorBlock::default();
        for (slot, vec) in vecs.iter().enumerate().take(BLOCK_SIZE) {
            for d in 0..N_DIMS {
                block.data[d * BLOCK_SIZE + slot] = vec[d];
            }
        }
        block
    }

    fn load_example_references() -> Vec<([f32; N_DIMS], u8)> {
        let raw = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/resources/example-references.json"),
        )
        .expect("example-references.json not found");

        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&raw).expect("invalid JSON");

        parsed
            .into_iter()
            .map(|entry| {
                let arr = entry["vector"].as_array().unwrap();
                let mut v = [0.0f32; N_DIMS];
                for (i, x) in arr.iter().enumerate() {
                    v[i] = x.as_f64().unwrap() as f32;
                }
                let label = if entry["label"] == "fraud" { 1u8 } else { 0u8 };
                (v, label)
            })
            .collect()
    }

    // ── unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn find_nearest_clusters_returns_correct_count() {
        let centroids: Vec<[f32; N_DIMS]> = (0..10)
            .map(|i| [i as f32 / 10.0; N_DIMS])
            .collect();
        let query = [0.0f32; N_DIMS];
        let result = find_nearest_clusters(&query, &centroids, 3);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&0), "cluster at 0.0 should be among top-3");
    }

    #[test]
    fn find_nearest_clusters_nprobe_gt_nlist_clamps() {
        let centroids: Vec<[f32; N_DIMS]> = (0..4).map(|_| [0.5f32; N_DIMS]).collect();
        let result = find_nearest_clusters(&[0.5; N_DIMS], &centroids, 100);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn update_top_k_blocks_finds_nearest() {
        // Two vectors in one block: [0; 14] and [100; 14] in i16 units.
        // Query is [0; 14] → slot 0 should be nearer.
        let vecs = {
            let mut v = vec![[0i16; N_DIMS]; BLOCK_SIZE];
            v[1] = [100i16; N_DIMS];
            v
        };
        let block = make_block(&vecs);
        let labels: Vec<u8> = (0..BLOCK_SIZE as u8).collect();

        let query = [0i16; N_DIMS];
        let mut heap: BinaryHeap<(DistF32, u8)> = BinaryHeap::new();

        update_top_k_blocks(&query, &[block], &labels, 1, &mut heap);

        assert_eq!(heap.len(), 1);
        let (dist, label) = heap.pop().unwrap();
        assert_eq!(dist.0, 0.0, "nearest vector has dist 0");
        assert_eq!(label, 0, "nearest vector is slot 0 → label 0");
    }

    #[test]
    fn dist_f32_ordering_is_correct() {
        let a = DistF32(1.0);
        let b = DistF32(2.0);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, DistF32(1.0));
    }

    // ── recall test (integration) ─────────────────────────────────────────────

    /// End-to-end recall test using example-references.json.
    ///
    /// With nprobe = nlist (all clusters probed), recall@5 must be ≥ 97%.
    #[test]
    fn recall_at_5_full_nprobe_is_high() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let mut rng = SmallRng::seed_from_u64(7);

        let records = load_example_references();
        let raw_vecs: Vec<[f32; N_DIMS]> = records.iter().map(|(v, _)| *v).collect();
        let all_labels: Vec<u8> = records.iter().map(|(_, l)| *l).collect();

        // Quantize all vectors to i16
        let all_q16: Vec<[i16; N_DIMS]> = raw_vecs.iter()
            .map(|v| quantize_query(v))
            .collect();

        // Build small IVF with nlist=4
        let nlist = 4;
        let (centroids, assignments) =
            kmeans(&raw_vecs, nlist, 10, &mut rng);

        // Build padded cluster blocks
        let mut cluster_vecs: Vec<Vec<[i16; N_DIMS]>> = vec![vec![]; nlist];
        let mut cluster_labs: Vec<Vec<u8>> = vec![vec![]; nlist];
        for (i, &c) in assignments.iter().enumerate() {
            cluster_vecs[c as usize].push(all_q16[i]);
            cluster_labs[c as usize].push(all_labels[i]);
        }

        // Pad each cluster to multiple of BLOCK_SIZE
        for ci in 0..nlist {
            let rem = cluster_vecs[ci].len() % BLOCK_SIZE;
            if rem != 0 {
                let pad = BLOCK_SIZE - rem;
                cluster_vecs[ci].extend(std::iter::repeat([0i16; N_DIMS]).take(pad));
                cluster_labs[ci].extend(std::iter::repeat(0u8).take(pad));
            }
        }

        // Pack into VectorBlocks
        let cluster_blocks: Vec<Vec<VectorBlock>> = cluster_vecs.iter().map(|vecs| {
            vecs.chunks(BLOCK_SIZE).map(|chunk| {
                let mut block = VectorBlock::default();
                for (slot, v) in chunk.iter().enumerate() {
                    for d in 0..N_DIMS {
                        block.data[d * BLOCK_SIZE + slot] = v[d];
                    }
                }
                block
            }).collect()
        }).collect();

        // Brute-force ground truth.
        // Use scan_block (AVX2 when available) to match the precision of the IVF
        // search path exactly — comparing scalar vs AVX2 distances causes false
        // mismatches on ties when the accumulation order differs.
        let brute_force_top5 = |query: &[i16; N_DIMS]| -> Vec<(DistF32, u8)> {
            use crate::index::simd::scan_block;
            let mut heap: BinaryHeap<(DistF32, u8)> = BinaryHeap::new();
            for ci in 0..nlist {
                for (b_idx, block) in cluster_blocks[ci].iter().enumerate() {
                    let dists = scan_block(query, block, f32::MAX);
                    for slot in 0..BLOCK_SIZE {
                        let vi = b_idx * BLOCK_SIZE + slot;
                        if vi >= cluster_labs[ci].len() { break; }
                        let d = DistF32(dists[slot]);
                        let l = cluster_labs[ci][vi];
                        if heap.len() < 5 {
                            heap.push((d, l));
                        } else if let Some(&(w, _)) = heap.peek() {
                            if d < w {
                                heap.pop();
                                heap.push((d, l));
                            }
                        }
                    }
                }
            }
            heap.into_vec()
        };

        // IVF search (all clusters → perfect recall)
        let k = 5;
        let mut total_hits = 0usize;
        let mut total_possible = 0usize;

        for (qi, query_f32) in raw_vecs.iter().enumerate() {
            let query_i16 = quantize_query(query_f32);
            let gt = brute_force_top5(&query_i16);
            let gt_dists: std::collections::HashSet<u32> = gt.iter()
                .map(|(d, _)| d.0.to_bits())
                .collect();

            let cluster_ids = find_nearest_clusters(query_f32, &centroids, nlist);
            let mut heap: BinaryHeap<(DistF32, u8)> = BinaryHeap::new();
            for ci in &cluster_ids {
                update_top_k_blocks(
                    &query_i16,
                    &cluster_blocks[*ci],
                    &cluster_labs[*ci],
                    k,
                    &mut heap,
                );
            }

            let hits = heap.iter()
                .filter(|(d, _)| gt_dists.contains(&d.0.to_bits()))
                .count();
            total_hits += hits;
            total_possible += gt.len();
            let _ = qi;
        }

        let recall = total_hits as f64 / total_possible as f64;
        assert!(
            recall >= 0.97,
            "recall@5 = {:.3} is below 0.97 ({total_hits}/{total_possible})",
            recall
        );
    }
}
