use std::collections::BinaryHeap;

use crate::index::layout::{BLOCK_SIZE, N_DIMS, VectorBlock};
use crate::index::simd::scan_block;

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
///
/// **Prefetch**: each iteration issues `_mm_prefetch` 2 blocks ahead. Each
/// block is 224 bytes (i16 layout), so 2 blocks ≈ 7 cache lines — well within
/// Haswell's L1 hardware prefetcher capacity. Hides L3/DRAM latency on Haswell
/// where the NPROBE×cluster working set spills past L2 (256 KB).
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

    const PREFETCH_DISTANCE: usize = 2;

    for (b_idx, block) in blocks.iter().enumerate() {
        // Prefetch a few blocks ahead so the L1 has them ready by the time
        // scan_block() touches them. Only valid on x86_64 with SSE present
        // (granted by x86-64-v3 baseline). No-op if the pointer is past the end.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let next = b_idx + PREFETCH_DISTANCE;
            if next < blocks.len() {
                _mm_prefetch(
                    blocks.as_ptr().add(next) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }

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

#[inline]
fn l2sq_f32(a: &[f32; N_DIMS], b: &[f32; N_DIMS]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..N_DIMS {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::kmeans::kmeans;
    use crate::index::quantize::quantize_query;

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

        // Brute-force ground truth using scan_block_scalar
        let brute_force_top5 = |query: &[i16; N_DIMS]| -> Vec<(DistF32, u8)> {
            use crate::index::simd::scan_block_scalar;
            let mut heap: BinaryHeap<(DistF32, u8)> = BinaryHeap::new();
            for ci in 0..nlist {
                for (b_idx, block) in cluster_blocks[ci].iter().enumerate() {
                    let dists = scan_block_scalar(query, block);
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
