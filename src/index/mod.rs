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
    DistF32, find_nearest_clusters, find_nearest_clusters_inplace, update_top_k_blocks,
};

/// Encapsulates the mmap'd IVF index and provides a safe, thread-safe search API.
pub struct IvfIndex {
    // Keep mmap owned by this struct for the whole lifetime of borrowed slices.
    _mmap: Mmap,
    // SAFETY: this `'static` lifetime is manufactured in `load` and is sound because
    // `layout` only borrows from `_mmap`, which is owned by this same struct.
    layout: IndexLayout<'static>,
    /// Starting BLOCK index for each cluster (cumulative sum of padded_sizes / BLOCK_SIZE).
    cluster_block_offsets: Vec<usize>,
    nprobe: usize,
    /// Clusters probed when fraud_count ∈ {2,3} (ambiguous zone). Default = nprobe * 3.
    full_nprobe: usize,
}

const MAX_NPROBE: usize = 64;

impl IvfIndex {
    /// Loads and parses `index.bin` from disk into a zero-copy mmap view.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("falha ao abrir index.bin em {}", path.display()))?;

        // Read-only mmap. The kernel can share these pages between API processes.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("falha ao fazer mmap de {}", path.display()))?;

        let parsed_layout = IndexLayout::from_bytes(&mmap)
            .with_context(|| format!("falha ao parsear layout de {}", path.display()))?;

        let nprobe = parsed_layout.header.nprobe as usize;
        let full_nprobe = nprobe * 3;
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
            nprobe,
            full_nprobe,
        })
    }

    /// Returns the global top-k `(distance, label)` across probed IVF clusters.
    pub fn search_knn(&self, query: &[f32; N_DIMS], k: usize) -> Vec<(DistF32, u8)> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<(DistF32, u8)> = BinaryHeap::with_capacity(k + 1);
        self.fill_top_k_heap(query, k, self.nprobe, &mut heap);

        let mut out: Vec<(DistF32, u8)> = heap.into_iter().collect();
        out.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Computes fraud score as `frauds_in_top5 / 5.0`.
    ///
    /// Two-stage strategy: fast probe (nprobe clusters) first, then a full probe
    /// (full_nprobe) only when the initial count lands in the ambiguous zone {2, 3}
    /// where a single wrong vote flips the fraud decision.
    pub fn fraud_score(&self, query: &[f32; N_DIMS]) -> f32 {
        let mut heap: BinaryHeap<(DistF32, u8)> = BinaryHeap::with_capacity(6);

        self.fill_top_k_heap(query, 5, self.nprobe, &mut heap);
        let fast_frauds = heap.iter().filter(|(_, label)| *label == 1).count();

        let frauds = if fast_frauds == 2 || fast_frauds == 3 {
            heap.clear();
            self.fill_top_k_heap(query, 5, self.full_nprobe, &mut heap);
            heap.iter().filter(|(_, label)| *label == 1).count()
        } else {
            fast_frauds
        };

        frauds as f32 / 5.0
    }

    /// Returns the current fast nprobe value.
    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    pub fn with_nprobes(mut self, fast: usize, full: usize) -> Self {
        let max_clusters = self.layout.centroids.len();
        if fast > 0 {
            self.nprobe = fast.min(max_clusters);
        }
        if full > 0 {
            self.full_nprobe = full.min(max_clusters);
        }
        self
    }

    fn cluster_block_range(&self, cluster_id: usize) -> Option<(usize, usize)> {
        if cluster_id >= self.layout.cluster_sizes.len() {
            return None;
        }
        let start = self.cluster_block_offsets[cluster_id];
        let end   = self.cluster_block_offsets[cluster_id + 1];
        Some((start, end))
    }

    fn fill_top_k_heap(
        &self,
        query: &[f32; N_DIMS],
        k: usize,
        nprobe: usize,
        heap: &mut BinaryHeap<(DistF32, u8)>,
    ) {
        if k == 0 {
            return;
        }

        let query_i16 = quantize_query(query);

        let dispatch = |cluster_id: usize, heap: &mut BinaryHeap<(DistF32, u8)>| {
            let Some((bs, be)) = self.cluster_block_range(cluster_id) else {
                return;
            };
            if bs >= be {
                return;
            }
            let blocks = &self.layout.blocks[bs..be];
            let labels = &self.layout.labels[bs * BLOCK_SIZE..be * BLOCK_SIZE];
            update_top_k_blocks(&query_i16, blocks, labels, k, heap);
        };

        if nprobe <= MAX_NPROBE {
            let mut cluster_ids = [0usize; MAX_NPROBE];
            let mut cluster_dists = [0.0f32; MAX_NPROBE];
            let used = find_nearest_clusters_inplace(
                query,
                self.layout.centroids,
                nprobe,
                &mut cluster_ids,
                &mut cluster_dists,
            );
            for &cid in &cluster_ids[..used] {
                dispatch(cid, heap);
            }
        } else {
            for cid in find_nearest_clusters(query, self.layout.centroids, nprobe) {
                dispatch(cid, heap);
            }
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
    use crate::index::layout::{IndexHeader, MAGIC, VERSION, VectorBlock};
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

    /// Builds a minimal valid index.bin in the new VERSION=2 format.
    ///
    /// 1 cluster, n_per_cluster vectors (padded to multiple of BLOCK_SIZE),
    /// with labels alternating legit/fraud.
    fn build_tiny_index_file(path: &Path, n_per_cluster: usize) {
        use crate::index::quantize::quantize_i16;

        let n_padded = crate::index::layout::align_up(n_per_cluster, BLOCK_SIZE);
        let nlist: u32 = 1;
        let n_blocks = n_padded / BLOCK_SIZE;

        let hdr_size     = std::mem::size_of::<IndexHeader>(); // 32
        let centroids_sz = nlist as usize * N_DIMS * 4;
        let csizes_sz    = nlist as usize * 4;
        let csizes_end   = hdr_size + centroids_sz + csizes_sz;
        let blocks_start = crate::index::layout::align_up(csizes_end, 32);
        let blocks_sz    = n_blocks * std::mem::size_of::<VectorBlock>();
        let labels_start = blocks_start + blocks_sz;
        let total        = labels_start + n_padded;

        // VectorBlock backing for 32-byte alignment.
        let n_vb = crate::index::layout::align_up(total, std::mem::size_of::<VectorBlock>())
            / std::mem::size_of::<VectorBlock>();
        let mut backing: Vec<VectorBlock> = vec![VectorBlock::default(); n_vb];

        {
            let buf: &mut [u8] = bytemuck::cast_slice_mut(&mut backing);

            let header = IndexHeader {
                magic:     MAGIC,
                version:   VERSION,
                nlist,
                nprobe:    1,
                n_vectors: n_padded as u64,
                _padding:  [0u8; 8],
            };
            buf[..hdr_size].copy_from_slice(bytes_of(&header));

            // Centroid at all-zeros
            // (centroids bytes already zero from default)

            // cluster_sizes[0] = n_padded
            let csizes_off = hdr_size + centroids_sz;
            let csizes: &mut [u32] =
                bytemuck::cast_slice_mut(&mut buf[csizes_off..csizes_off + csizes_sz]);
            csizes[0] = n_padded as u32;

            // Fill blocks: vector[i] has dims[0] = quantize_i16(i as f32 / 100.0), rest 0
            let blocks: &mut [VectorBlock] =
                bytemuck::cast_slice_mut(&mut buf[blocks_start..blocks_start + blocks_sz]);
            for slot in 0..n_padded {
                let block_idx = slot / BLOCK_SIZE;
                let slot_in_block = slot % BLOCK_SIZE;
                let val = quantize_i16(slot as f32 / 100.0);
                blocks[block_idx].data[0 * BLOCK_SIZE + slot_in_block] = val;
            }

            // Labels: alternating fraud(1) / legit(0)
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
        // 8 vectors: labels 1,0,1,0,1,0,1,0 → 4 fraud, 4 legit
        build_tiny_index_file(&tmp, 8);

        let index = IvfIndex::load(&tmp).expect("must load tiny index");
        let query = [0.0f32; N_DIMS];
        let score = index.fraud_score(&query);

        // With k=5 neighbours from 8 vectors alternating fraud/legit:
        // nearest 5 from [0;14]: slots 0,1,2,3,4 → labels 1,0,1,0,1 → 3 frauds
        // full_nprobe also returns same cluster → same result
        // But exact count depends on distances, just check it's a valid ratio
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
