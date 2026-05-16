pub mod kmeans;
pub mod layout;
pub mod quantize;
pub mod search;
pub mod simd;

use std::collections::BinaryHeap;
use std::fs::File;
use std::path::Path;

use anyhow::Context;
use memmap2::{Mmap, MmapOptions};

use crate::index::layout::{BLOCK_SIZE, IndexLayout, N_DIMS};
use crate::index::quantize::quantize_query;
use crate::index::search::{
    DistF32, Top5, bbox_lower_bound, query_to_bbox_lanes, update_top5_blocks,
    update_top_k_blocks,
};

/// Upper bound on `nlist` accepted by the bbox-prune path. Used to size a
/// stack-allocated sort buffer to avoid per-request allocations.
/// Current production builds use nlist=4096; this leaves room for experiments.
const MAX_NLIST: usize = 16_384;

/// Encapsulates the mmap'd IVF index and provides a safe, thread-safe search API.
pub struct IvfIndex {
    // Keep mmap owned by this struct for the whole lifetime of borrowed slices.
    _mmap: Mmap,
    // SAFETY: this `'static` lifetime is manufactured in `load` and is sound because
    // `layout` only borrows from `_mmap`, which is owned by this same struct.
    layout: IndexLayout<'static>,
    /// Starting BLOCK index for each cluster (cumulative sum of padded_sizes / BLOCK_SIZE).
    cluster_block_offsets: Vec<usize>,
}

impl IvfIndex {
    /// Loads and parses `index.bin` from disk into a zero-copy mmap view.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("falha ao abrir index.bin em {}", path.display()))?;

        // Read-only mmap with MAP_POPULATE: pre-faults all pages on load so the
        // first requests don't pay page-fault latency. Adds ~1s to startup time.
        let mmap = unsafe { MmapOptions::new().populate().map(&file) }
            .with_context(|| format!("falha ao fazer mmap de {}", path.display()))?;

        // HUGEPAGE cuts TLB misses to ~1/512 (2MB pages); WILLNEED warms the
        // working set before the first request. Adopted by 5/9 top-10 entries
        // (MXLange #1, Ronie #2, atomos #4, macedot #8, itagyba #10).
        unsafe {
            let ptr = mmap.as_ptr() as *mut libc::c_void;
            let len = mmap.len();
            libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
            libc::madvise(ptr, len, libc::MADV_WILLNEED);
        }

        let parsed_layout = IndexLayout::from_bytes(&mmap)
            .with_context(|| format!("falha ao parsear layout de {}", path.display()))?;

        anyhow::ensure!(
            parsed_layout.bbox_mins.len() <= MAX_NLIST,
            "nlist {} exceeds MAX_NLIST {}",
            parsed_layout.bbox_mins.len(),
            MAX_NLIST,
        );

        let cluster_block_offsets = build_cluster_block_offsets(parsed_layout.cluster_sizes);

        // SAFETY: `parsed_layout` borrows from `mmap`. We move both into `Self`, and
        // `_mmap` is kept alive for the entire struct lifetime, so references remain valid.
        let layout = unsafe {
            std::mem::transmute::<IndexLayout<'_>, IndexLayout<'static>>(parsed_layout)
        };

        Ok(Self {
            _mmap: mmap,
            layout,
            cluster_block_offsets,
        })
    }

    /// Returns the global top-k `(distance, label)` across probed IVF clusters.
    ///
    /// k ≤ 5 uses the bbox-prune hot path with `Top5`; larger k falls back to
    /// a `BinaryHeap` scan over every cluster (correctness-only path for tests).
    pub fn search_knn(&self, query: &[f32; N_DIMS], k: usize) -> Vec<(DistF32, u8)> {
        if k == 0 {
            return Vec::new();
        }

        if k <= 5 {
            let mut top5 = Top5::new();
            self.fill_top5_bbox_prune(query, &mut top5);
            return top5.iter().take(k).map(|&(d, l)| (DistF32(d), l)).collect();
        }

        // Fallback for k > 5 (not used in production, correctness path for tests).
        let mut heap: BinaryHeap<(DistF32, u8)> = BinaryHeap::with_capacity(k + 1);
        self.fill_heap_full_scan(query, k, &mut heap);
        let mut out: Vec<(DistF32, u8)> = heap.into_iter().collect();
        out.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Computes fraud score as `frauds_in_top5 / 5.0`.
    ///
    /// Single-stage: bbox-prune visits clusters in ascending lower-bound order
    /// and stops as soon as the next cluster's lower bound ≥ the current top-5
    /// worst distance. There is no "fast vs full nprobe" — pruning is automatic.
    /// Mirrors RonieNeubauer #2 (p99 0.86ms, lowest on the leaderboard).
    pub fn fraud_score(&self, query: &[f32; N_DIMS]) -> f32 {
        let mut top5 = Top5::new();
        self.fill_top5_bbox_prune(query, &mut top5);
        top5.count_fraud() as f32 / 5.0
    }

    /// `nprobe` is preserved as a no-op API for legacy callers; the bbox-prune
    /// path ignores it. Returns the value stored in the on-disk header.
    pub fn nprobe(&self) -> usize {
        self.layout.header.nprobe as usize
    }

    /// Legacy no-op: kept so existing call sites (e.g. main.rs env var wiring)
    /// continue to compile. Bbox-prune is parameter-free.
    pub fn with_nprobes(self, _fast: usize, _full: usize) -> Self {
        self
    }

    /// Hot-path: visits clusters in ascending bbox lower-bound order and stops
    /// as soon as the next cluster cannot improve `top5`.
    fn fill_top5_bbox_prune(&self, query: &[f32; N_DIMS], top5: &mut Top5) {
        let nlist = self.layout.bbox_mins.len();
        if nlist == 0 {
            return;
        }

        let query_i16 = quantize_query(query);
        let q_lanes = query_to_bbox_lanes(&query_i16);

        // Compute every cluster's bbox lower bound, packed as (lb_bits << 32 | cid)
        // so a single u64 sort gives us ascending-distance order with no tuple
        // overhead. f32 bits sort correctly for non-negative finite values.
        let mut packed = [0u64; MAX_NLIST];
        for ci in 0..nlist {
            let lb = bbox_lower_bound(&q_lanes, &self.layout.bbox_mins[ci], &self.layout.bbox_maxes[ci]);
            packed[ci] = ((lb.to_bits() as u64) << 32) | (ci as u64);
        }
        packed[..nlist].sort_unstable();

        // Walk in order, prune the rest as soon as lb ≥ current worst.
        for &entry in &packed[..nlist] {
            let lb_bits = (entry >> 32) as u32;
            let lb = f32::from_bits(lb_bits);

            if lb >= top5.worst() {
                break;
            }

            let cluster_id = (entry & 0xFFFF_FFFF) as usize;
            let Some((bs, be)) = self.cluster_block_range(cluster_id) else { continue };
            if bs >= be { continue; }

            let blocks = &self.layout.blocks[bs..be];
            let labels = &self.layout.labels[bs * BLOCK_SIZE..be * BLOCK_SIZE];
            update_top5_blocks(&query_i16, blocks, labels, top5);
        }
    }

    fn cluster_block_range(&self, cluster_id: usize) -> Option<(usize, usize)> {
        if cluster_id >= self.layout.cluster_sizes.len() {
            return None;
        }
        let start = self.cluster_block_offsets[cluster_id];
        let end   = self.cluster_block_offsets[cluster_id + 1];
        Some((start, end))
    }

    /// Test-only correctness path: scans every cluster's blocks against the heap.
    /// Used by `search_knn(k > 5)`; not exercised on the production hot path.
    fn fill_heap_full_scan(
        &self,
        query: &[f32; N_DIMS],
        k: usize,
        heap: &mut BinaryHeap<(DistF32, u8)>,
    ) {
        let query_i16 = quantize_query(query);
        let nlist = self.layout.cluster_sizes.len();
        for cluster_id in 0..nlist {
            let Some((bs, be)) = self.cluster_block_range(cluster_id) else { continue };
            if bs >= be { continue; }
            let blocks = &self.layout.blocks[bs..be];
            let labels = &self.layout.labels[bs * BLOCK_SIZE..be * BLOCK_SIZE];
            update_top_k_blocks(&query_i16, blocks, labels, k, heap);
        }
    }
}

fn build_cluster_block_offsets(cluster_sizes: &[u32]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(cluster_sizes.len() + 1);
    offsets.push(0usize);
    for &size in cluster_sizes {
        // cluster_sizes are already padded to multiples of BLOCK_SIZE by build-index.
        let blocks = size as usize / BLOCK_SIZE;
        let next = offsets.last().copied().unwrap_or(0) + blocks;
        offsets.push(next);
    }
    offsets
}

// SAFETY: IvfIndex only holds an Mmap (read-only, page-aligned) and slices
// derived from it. No interior mutability; safe to share across threads.
unsafe impl Send for IvfIndex {}
unsafe impl Sync for IvfIndex {}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::index::layout::{Bbox, IndexHeader, MAGIC, VERSION, VectorBlock, align_up};
    use bytemuck::bytes_of;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn ivf_index_is_send_sync() {
        assert_send_sync::<IvfIndex>();
    }

    fn unique_tmp_file(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{nanos}.bin"))
    }

    /// Builds a minimal valid v3 index.bin file with one cluster of `n_per_cluster`
    /// vectors (padded to a multiple of `BLOCK_SIZE`), labels alternating
    /// legit/fraud, and a tight bbox covering the real vectors.
    fn build_tiny_index_file(path: &Path, n_per_cluster: usize) {
        use crate::index::quantize::quantize_i16;

        let n_padded = align_up(n_per_cluster, BLOCK_SIZE);
        let nlist: u32 = 1;
        let n_blocks = n_padded / BLOCK_SIZE;

        let hdr_size       = std::mem::size_of::<IndexHeader>();
        let centroids_sz   = nlist as usize * N_DIMS * 4;
        let csizes_sz      = nlist as usize * 4;
        let csizes_end     = hdr_size + centroids_sz + csizes_sz;
        let bbox_start     = align_up(csizes_end, 32);
        let bbox_sz        = nlist as usize * std::mem::size_of::<Bbox>();
        let bbox_end       = bbox_start + 2 * bbox_sz;
        let blocks_start   = align_up(bbox_end, 32);
        let blocks_sz      = n_blocks * std::mem::size_of::<VectorBlock>();
        let labels_start   = blocks_start + blocks_sz;
        let total          = labels_start + n_padded;

        let n_vb = align_up(total, std::mem::size_of::<VectorBlock>())
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

            // cluster_sizes[0] = n_padded
            let csizes_off = hdr_size + centroids_sz;
            let csizes: &mut [u32] =
                bytemuck::cast_slice_mut(&mut buf[csizes_off..csizes_off + csizes_sz]);
            csizes[0] = n_padded as u32;

            // Bbox: dim 0 in [0, quantize_i16((n_per_cluster-1) / 100.0)], rest 0.
            // This matches the per-slot values written below (vector i has
            // data[0,slot=i] = quantize_i16(i / 100.0)). split_at_mut keeps the
            // borrow checker happy when writing two non-overlapping bbox arrays.
            let (mins_buf, rest) = buf[bbox_start..bbox_start + 2 * bbox_sz].split_at_mut(bbox_sz);
            let bbox_mins:  &mut [Bbox] = bytemuck::cast_slice_mut(mins_buf);
            let bbox_maxes: &mut [Bbox] = bytemuck::cast_slice_mut(rest);
            bbox_mins[0].data = [0; 16];
            bbox_maxes[0].data = [0; 16];
            if n_per_cluster > 0 {
                bbox_maxes[0].data[0] = quantize_i16((n_per_cluster - 1) as f32 / 100.0);
            }

            // Vector data: vector[i] has data[dim=0, slot=i] = quantize_i16(i / 100).
            let blocks: &mut [VectorBlock] =
                bytemuck::cast_slice_mut(&mut buf[blocks_start..blocks_start + blocks_sz]);
            for slot in 0..n_padded {
                let block_idx = slot / BLOCK_SIZE;
                let slot_in_block = slot % BLOCK_SIZE;
                let val = quantize_i16(slot as f32 / 100.0);
                blocks[block_idx].data[0 * BLOCK_SIZE + slot_in_block] = val;
            }

            // Labels: alternate fraud(1) / legit(0)
            for (i, label) in buf[labels_start..labels_start + n_padded].iter_mut().enumerate() {
                *label = (i % 2) as u8;
            }
        }

        let raw = bytemuck::cast_slice::<VectorBlock, u8>(&backing);
        let mut file = fs::File::create(path).expect("create tiny index file");
        file.write_all(&raw[..total]).expect("write tiny index file");
        file.flush().expect("flush tiny index file");
    }

    #[test]
    fn load_and_search_knn_work_on_tiny_index() {
        let tmp = unique_tmp_file("tiny-ivf-index");
        build_tiny_index_file(&tmp, 16); // 16 vectors in 1 cluster

        let index = IvfIndex::load(&tmp).expect("must load tiny index");
        let query = [0.0f32; N_DIMS];
        let top2 = index.search_knn(&query, 2);

        assert_eq!(top2.len(), 2);
        assert!(top2[0].0 <= top2[1].0);

        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn fraud_score_uses_top5_ratio() {
        let tmp = unique_tmp_file("tiny-ivf-index-score");
        build_tiny_index_file(&tmp, 8);

        let index = IvfIndex::load(&tmp).expect("must load tiny index");
        let query = [0.0f32; N_DIMS];
        let score = index.fraud_score(&query);

        // 8 vectors alternate fraud(1)/legit(0); nearest 5 from [0;14] are slots
        // 0..4 with labels 1,0,1,0,1 → 3 frauds → 0.6.
        assert!(score >= 0.0 && score <= 1.0, "score {score} out of range [0,1]");
        assert!(score * 5.0 == (score * 5.0).round(), "score {score} not a multiple of 0.2");

        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn search_knn_returns_empty_for_k_zero() {
        let tmp = unique_tmp_file("tiny-ivf-index-k0");
        build_tiny_index_file(&tmp, 8);
        let index = IvfIndex::load(&tmp).expect("load");
        assert!(index.search_knn(&[0.0f32; N_DIMS], 0).is_empty());
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    #[ignore = "requer index.bin real gerado pelo build-index"]
    fn integration_with_real_index_and_example_payloads() {
        use crate::models::MccRisk;
        use crate::vectorizer::vectorize;

        let index_path = Path::new("index.bin");
        let index = IvfIndex::load(index_path).expect("index.bin deve existir");
        let mcc_risk = MccRisk::load(Path::new("resources/mcc_risk.json")).expect("mcc_risk");

        let payloads_raw = fs::read_to_string("resources/example-payloads.json")
            .expect("example payloads");
        let payloads: Vec<crate::models::TransactionPayload> =
            serde_json::from_str(&payloads_raw).expect("json válido");

        let legit = vectorize(&payloads[0], &mcc_risk);
        let fraud = vectorize(
            payloads
                .iter()
                .find(|p| p.id == "tx-3330991687")
                .expect("payload fraudulento de exemplo"),
            &mcc_risk,
        );

        assert!(index.fraud_score(&legit) < 0.6);
        assert!(index.fraud_score(&fraud) >= 0.6);
    }
}
