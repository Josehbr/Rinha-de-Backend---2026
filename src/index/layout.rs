use anyhow::ensure;
use bytemuck::{Pod, Zeroable};

pub const MAGIC: u32 = 0x49564632; // b"IVF2" little-endian
pub const VERSION: u32 = 3;
pub const N_DIMS: usize = 14;
/// Bbox uses 16 i16 lanes per cluster (14 real dims + 2 zero pad) so a single
/// `_mm256_loadu_si256` reads the whole bbox of one cluster.
pub const BBOX_LANES: usize = 16;
/// Number of vectors packed per VectorBlock (SoA layout).
pub const BLOCK_SIZE: usize = 8;

// ── On-disk types ─────────────────────────────────────────────────────────────

/// One IVF block: BLOCK_SIZE vectors in SoA (dimension-major) layout, quantized as i16.
///
/// `data[dim * BLOCK_SIZE + slot]` is the quantized value of vector[slot] at dim.
///
/// `align(32)` guarantees that a 256-bit AVX2 register can load one dimension
/// across all 8 slots with a single `_mm_loadu_si128(&data[dim * BLOCK_SIZE])`.
/// Size: 14 × 8 × 2 = 224 bytes = 7 × 32.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct VectorBlock {
    pub data: [i16; N_DIMS * BLOCK_SIZE], // index: dim * BLOCK_SIZE + slot
}

impl Default for VectorBlock {
    fn default() -> Self {
        Self { data: [0i16; N_DIMS * BLOCK_SIZE] }
    }
}

// SAFETY: [i16; 112] is a plain array — all bit patterns are valid, no padding, repr(C).
unsafe impl bytemuck::Pod for VectorBlock {}
unsafe impl bytemuck::Zeroable for VectorBlock {}

/// Axis-aligned bounding box of one IVF cluster in quantized (i16) space.
///
/// 14 real dims + 2 zero pad — the pad lets AVX2 load the whole bbox with a
/// single 256-bit instruction and contributes 0 to the lower-bound computation.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct Bbox {
    pub data: [i16; BBOX_LANES],
}

impl Default for Bbox {
    fn default() -> Self {
        Self { data: [0i16; BBOX_LANES] }
    }
}

unsafe impl bytemuck::Pod for Bbox {}
unsafe impl bytemuck::Zeroable for Bbox {}

/// Fixed-size header at byte offset 0 of index.bin — 32 bytes total.
///
/// ```text
/// Offset  Size  Field
///      0     4  magic       (must equal MAGIC = 0x49564632)
///      4     4  version     (must equal VERSION = 3)
///      8     4  nlist       (number of IVF clusters)
///     12     4  nprobe      (legacy default — unused on the v3 bbox-prune hot path)
///     16     8  n_vectors   (total vectors, padded to a multiple of BLOCK_SIZE)
///     24     8  _padding    (zero)
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IndexHeader {
    pub magic:     u32,
    pub version:   u32,
    pub nlist:     u32,
    pub nprobe:    u32,
    pub n_vectors: u64,
    pub _padding:  [u8; 8],
}

// ── Binary layout ─────────────────────────────────────────────────────────────

/// Zero-copy view into the sections of an index.bin v3 buffer (e.g. from mmap).
///
/// ```text
/// [IndexHeader      ]  32 bytes                          offset 0
/// [centroids        ]  nlist × 14 × 4 B                  legacy, kept for compat
/// [cluster_sizes    ]  nlist × 4 B
/// [bbox_mins        ]  nlist × Bbox (32 B)               v3: per-cluster i16 min
/// [bbox_maxes       ]  nlist × Bbox (32 B)               v3: per-cluster i16 max
/// [alignment pad    ]  0–31 bytes                        brings blocks to 32-byte
/// [blocks           ]  (n_vectors/BLOCK_SIZE) × VectorBlock
/// [labels           ]  n_vectors × 1 B                   0=legit, 1=fraud
/// ```
///
/// `cluster_sizes[i]` is the **padded** vector count for cluster i, always a
/// multiple of `BLOCK_SIZE`. Padding slots carry label 0 and all-zero features
/// — they sit at the geometric origin and rarely enter top-5 for real queries.
#[derive(Debug)]
pub struct IndexLayout<'a> {
    pub header:        &'a IndexHeader,
    pub centroids:     &'a [[f32; N_DIMS]], // legacy — present but unused on hot path
    pub cluster_sizes: &'a [u32],
    pub bbox_mins:     &'a [Bbox],          // [nlist]
    pub bbox_maxes:    &'a [Bbox],          // [nlist]
    pub blocks:        &'a [VectorBlock],   // [n_vectors / BLOCK_SIZE]
    pub labels:        &'a [u8],            // [n_vectors] — covers all padded slots
}

impl<'a> IndexLayout<'a> {
    /// Parses an index.bin v3 buffer into zero-copy section slices.
    pub fn from_bytes(buf: &'a [u8]) -> anyhow::Result<Self> {
        const HDR: usize = size_of::<IndexHeader>(); // 32

        ensure!(buf.len() >= HDR, "buffer too small for header ({} bytes)", buf.len());

        let header: &IndexHeader = bytemuck::from_bytes(&buf[..HDR]);

        ensure!(
            header.magic == MAGIC,
            "invalid magic 0x{:08x}, expected 0x{:08x}",
            header.magic,
            MAGIC
        );
        ensure!(
            header.version == VERSION,
            "unsupported index version {}, expected {} — rebuild index.bin",
            header.version,
            VERSION
        );

        let nlist = header.nlist as usize;
        let n_vec = header.n_vectors as usize;

        ensure!(
            n_vec % BLOCK_SIZE == 0,
            "n_vectors {n_vec} is not a multiple of BLOCK_SIZE {BLOCK_SIZE}"
        );
        let n_blocks = n_vec / BLOCK_SIZE;

        let centroids_start = HDR;
        let centroids_end   = centroids_start + nlist * N_DIMS * 4;

        let csizes_start = centroids_end;
        let csizes_end   = csizes_start + nlist * 4;

        // Bbox arrays start at the next 32-byte boundary so Bbox alignment holds.
        let bbox_min_start = align_up(csizes_end, 32);
        let bbox_min_end   = bbox_min_start + nlist * size_of::<Bbox>();
        let bbox_max_start = bbox_min_end;
        let bbox_max_end   = bbox_max_start + nlist * size_of::<Bbox>();

        let blocks_start = align_up(bbox_max_end, 32);
        let blocks_end   = blocks_start + n_blocks * size_of::<VectorBlock>();

        let labels_start = blocks_end;
        let labels_end   = labels_start + n_vec;

        ensure!(
            buf.len() >= labels_end,
            "buffer too small: need {labels_end} bytes, got {}",
            buf.len()
        );

        let centroids: &[[f32; N_DIMS]] =
            bytemuck::try_cast_slice(&buf[centroids_start..centroids_end])
                .map_err(|e| anyhow::anyhow!("centroids cast error: {e:?}"))?;

        let cluster_sizes: &[u32] =
            bytemuck::try_cast_slice(&buf[csizes_start..csizes_end])
                .map_err(|e| anyhow::anyhow!("cluster_sizes cast error: {e:?}"))?;

        let bbox_mins: &[Bbox] =
            bytemuck::try_cast_slice(&buf[bbox_min_start..bbox_min_end])
                .map_err(|e| anyhow::anyhow!("bbox_mins cast error: {e:?}"))?;

        let bbox_maxes: &[Bbox] =
            bytemuck::try_cast_slice(&buf[bbox_max_start..bbox_max_end])
                .map_err(|e| anyhow::anyhow!("bbox_maxes cast error: {e:?}"))?;

        let blocks: &[VectorBlock] =
            bytemuck::try_cast_slice(&buf[blocks_start..blocks_end])
                .map_err(|e| anyhow::anyhow!("blocks cast error: {e:?}"))?;

        let labels = &buf[labels_start..labels_end];

        Ok(Self { header, centroids, cluster_sizes, bbox_mins, bbox_maxes, blocks, labels })
    }
}

#[inline]
pub(crate) fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::bytes_of;

    #[test]
    fn vector_block_is_correct_size_and_alignment() {
        // 14 dims × 8 slots × 2 bytes = 224 bytes = 7 × 32
        assert_eq!(size_of::<VectorBlock>(), 224);
        assert_eq!(align_of::<VectorBlock>(), 32);
        assert_eq!(size_of::<VectorBlock>() % 32, 0);
    }

    #[test]
    fn bbox_is_32_bytes_aligned() {
        assert_eq!(size_of::<Bbox>(), 32);
        assert_eq!(align_of::<Bbox>(), 32);
    }

    #[test]
    fn index_header_is_32_bytes() {
        assert_eq!(size_of::<IndexHeader>(), 32);
        assert_eq!(size_of::<IndexHeader>() % 32, 0);
    }

    /// Builds a minimal valid v3 index buffer for round-trip testing.
    fn build_test_index_v3(nlist: u32, n_per_cluster: u32) -> (Vec<VectorBlock>, usize) {
        assert_eq!(n_per_cluster % BLOCK_SIZE as u32, 0, "n_per_cluster must be multiple of BLOCK_SIZE");
        let n_padded = nlist as usize * n_per_cluster as usize;
        let n_blocks = n_padded / BLOCK_SIZE;

        let hdr_size     = size_of::<IndexHeader>();
        let centroids_sz = nlist as usize * N_DIMS * 4;
        let csizes_sz    = nlist as usize * 4;
        let csizes_end   = hdr_size + centroids_sz + csizes_sz;
        let bbox_min_start = align_up(csizes_end, 32);
        let bbox_sz      = nlist as usize * size_of::<Bbox>();
        let bbox_max_end = bbox_min_start + 2 * bbox_sz;
        let blocks_start = align_up(bbox_max_end, 32);
        let blocks_sz    = n_blocks * size_of::<VectorBlock>();
        let labels_start = blocks_start + blocks_sz;
        let total        = labels_start + n_padded;

        let n_vb = (total + size_of::<VectorBlock>() - 1) / size_of::<VectorBlock>();
        let mut backing: Vec<VectorBlock> = vec![VectorBlock::default(); n_vb];

        {
            let buf: &mut [u8] = bytemuck::cast_slice_mut(&mut backing);

            let hdr = IndexHeader {
                magic: MAGIC,
                version: VERSION,
                nlist,
                nprobe: 8,
                n_vectors: n_padded as u64,
                _padding: [0u8; 8],
            };
            buf[..hdr_size].copy_from_slice(bytes_of(&hdr));

            let csizes: &mut [u32] = bytemuck::cast_slice_mut(
                &mut buf[hdr_size + centroids_sz..hdr_size + centroids_sz + csizes_sz],
            );
            for s in csizes.iter_mut() {
                *s = n_per_cluster;
            }

            for (i, label) in buf[labels_start..labels_start + n_padded].iter_mut().enumerate() {
                *label = (i % 2) as u8;
            }
        }

        (backing, total)
    }

    #[test]
    fn round_trip_index_layout_v3() {
        let (backing, total) = build_test_index_v3(4, 8); // 4 clusters × 8 vectors
        let raw = bytemuck::cast_slice::<VectorBlock, u8>(&backing);
        let layout = IndexLayout::from_bytes(&raw[..total]).unwrap();

        assert_eq!(layout.header.magic,        MAGIC);
        assert_eq!(layout.header.version,      VERSION);
        assert_eq!(layout.header.nlist,        4);
        assert_eq!(layout.header.n_vectors,    32);
        assert_eq!(layout.centroids.len(),     4);
        assert_eq!(layout.cluster_sizes.len(), 4);
        assert_eq!(layout.bbox_mins.len(),     4);
        assert_eq!(layout.bbox_maxes.len(),    4);
        assert_eq!(layout.blocks.len(),        32 / BLOCK_SIZE);
        assert_eq!(layout.labels.len(),        32);
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let (mut backing, total) = build_test_index_v3(1, 8);
        {
            let buf: &mut [u8] = bytemuck::cast_slice_mut(&mut backing);
            buf[0] = 0xFF;
            buf[1] = 0xFF;
        }
        let raw = bytemuck::cast_slice::<VectorBlock, u8>(&backing);
        let err = IndexLayout::from_bytes(&raw[..total]).unwrap_err();
        assert!(err.to_string().contains("invalid magic"), "{err}");
    }

    #[test]
    fn from_bytes_rejects_wrong_version() {
        let (mut backing, total) = build_test_index_v3(1, 8);
        {
            let buf: &mut [u8] = bytemuck::cast_slice_mut(&mut backing);
            let v_bytes = 99u32.to_le_bytes();
            buf[4..8].copy_from_slice(&v_bytes);
        }
        let raw = bytemuck::cast_slice::<VectorBlock, u8>(&backing);
        let err = IndexLayout::from_bytes(&raw[..total]).unwrap_err();
        assert!(err.to_string().contains("unsupported index version"), "{err}");
    }

    #[test]
    fn from_bytes_rejects_truncated_buffer() {
        let result = IndexLayout::from_bytes(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn align_up_is_correct() {
        assert_eq!(align_up(0,  32), 0);
        assert_eq!(align_up(1,  32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(224, 32), 224);
    }
}
