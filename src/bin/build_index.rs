#[path = "../index/mod.rs"]
mod index;

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use bytemuck::{bytes_of, cast_slice};
use flate2::read::GzDecoder;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::Deserializer;
use serde::de::{SeqAccess, Visitor};

use index::IvfIndex;
use index::layout::{BLOCK_SIZE, IndexHeader, MAGIC, N_DIMS, VERSION, VectorBlock};
use index::quantize::quantize_i16;

const INPUT_PATH: &str = "resources/references.json.gz";
const OUTPUT_PATH: &str = "index.bin";
/// NLIST=4096 escolhido para que centroids f32 caibam no L2 do Haswell (256KB).
/// Centroids 4096 × 14 × 4 = 229 KB → cabe em L2 com folga (33 KB para working set IVF).
/// v6 com NLIST=8192 → centroids 448 KB estouravam L2 a cada find_nearest_clusters,
/// causando ~1.7 ms de p99 extra (jairoblatt-rust #5 e macedot-c #8 confirmam o ganho).
/// NPROBE=8 fast / FULL_NPROBE=24 igualam jairoblatt-rust (#5, p99 1.03ms).
const NLIST: usize = 4096;
const NPROBE: u32 = 8;
const NITER: usize = 30;
const LOG_EVERY: usize = 500_000;

#[derive(serde::Deserialize)]
struct ReferenceRecord {
    vector: [f32; N_DIMS],
    label: String,
}

fn main() -> Result<()> {
    let started_at = Instant::now();
    println!("[build] iniciando geração do index.bin");

    let (vectors_f32, labels) = read_references(Path::new(INPUT_PATH))
        .with_context(|| format!("falha ao ler {INPUT_PATH}"))?;
    ensure!(
        vectors_f32.len() == labels.len(),
        "inconsistência: vectors={} labels={}",
        vectors_f32.len(),
        labels.len()
    );

    println!("[build] treinando kmeans (nlist={NLIST}, n_iter={NITER})...");
    let t = Instant::now();
    let mut rng = StdRng::seed_from_u64(42);
    let (centroids, assignments) = index::kmeans::kmeans(&vectors_f32, NLIST, NITER, &mut rng);
    println!("[build] kmeans concluído em {:.2}s", t.elapsed().as_secs_f64());

    println!("[build] ordenando vetores/labels por cluster...");
    let t = Instant::now();
    let mut packed: Vec<(u32, [f32; N_DIMS], u8)> = assignments
        .into_iter()
        .zip(vectors_f32.into_iter())
        .zip(labels.into_iter())
        .map(|((cid, vec), label)| (cid, vec, label))
        .collect();
    packed.par_sort_unstable_by_key(|(cid, _, _)| *cid);

    let mut cluster_sizes_raw = vec![0usize; NLIST];
    let mut sorted_vectors: Vec<[f32; N_DIMS]> = Vec::with_capacity(packed.len());
    let mut sorted_labels: Vec<u8> = Vec::with_capacity(packed.len());
    for (cid, vec, label) in packed {
        cluster_sizes_raw[cid as usize] += 1;
        sorted_vectors.push(vec);
        sorted_labels.push(label);
    }
    println!("[build] ordenação concluída em {:.2}s", t.elapsed().as_secs_f64());

    println!("[build] construindo VectorBlocks (SoA int16)...");
    let (blocks, cluster_sizes_padded, labels_padded) =
        build_blocks(&sorted_vectors, &sorted_labels, &cluster_sizes_raw);

    let n_vectors_padded: u64 = cluster_sizes_padded.iter().map(|&s| s as u64).sum();
    ensure!(
        blocks.len() as u64 * BLOCK_SIZE as u64 == n_vectors_padded,
        "inconsistência: {} blocks × {} ≠ {} vectors padded",
        blocks.len(),
        BLOCK_SIZE,
        n_vectors_padded
    );

    serialize_index(
        Path::new(OUTPUT_PATH),
        &centroids,
        &cluster_sizes_padded,
        &blocks,
        &labels_padded,
        n_vectors_padded,
    )?;

    validate_index(Path::new(OUTPUT_PATH), &sorted_vectors)?;

    let out_size = std::fs::metadata(OUTPUT_PATH)
        .with_context(|| format!("falha ao obter metadata de {OUTPUT_PATH}"))?
        .len();
    println!(
        "[build] finalizado: {} bytes ({:.2} MB) em {:.2}s",
        out_size,
        out_size as f64 / (1024.0 * 1024.0),
        started_at.elapsed().as_secs_f64()
    );

    Ok(())
}

/// Pads each cluster to a multiple of BLOCK_SIZE and packs vectors into SoA VectorBlocks.
///
/// Returns `(blocks, padded_cluster_sizes, padded_labels)`.
/// Padding slots carry label 0 (legit) and all-zero features — geometrically far from
/// real queries under L2², so they don't influence top-k results in practice.
fn build_blocks(
    sorted_vectors: &[[f32; N_DIMS]],
    sorted_labels: &[u8],
    cluster_sizes: &[usize],
) -> (Vec<VectorBlock>, Vec<u32>, Vec<u8>) {
    let n_padded: usize = cluster_sizes
        .iter()
        .map(|&s| align_up(s, BLOCK_SIZE))
        .sum();

    let mut blocks = vec![VectorBlock::default(); n_padded / BLOCK_SIZE];
    let mut padded_labels = vec![0u8; n_padded];
    let mut padded_sizes = vec![0u32; cluster_sizes.len()];

    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    for (ci, &raw_count) in cluster_sizes.iter().enumerate() {
        let padded_count = align_up(raw_count, BLOCK_SIZE);
        padded_sizes[ci] = padded_count as u32;

        for local in 0..raw_count {
            let slot = dst_offset + local;
            let block_idx = slot / BLOCK_SIZE;
            let slot_in_block = slot % BLOCK_SIZE;
            let v = &sorted_vectors[src_offset + local];
            for d in 0..N_DIMS {
                blocks[block_idx].data[d * BLOCK_SIZE + slot_in_block] = quantize_i16(v[d]);
            }
            padded_labels[slot] = sorted_labels[src_offset + local];
        }
        // Padding slots [raw_count..padded_count] remain zero from default init.

        src_offset += raw_count;
        dst_offset += padded_count;
    }

    (blocks, padded_sizes, padded_labels)
}

fn serialize_index(
    output_path: &Path,
    centroids: &[[f32; N_DIMS]],
    cluster_sizes: &[u32],
    blocks: &[VectorBlock],
    labels: &[u8],
    n_vectors: u64,
) -> Result<()> {
    ensure!(
        centroids.len() == cluster_sizes.len(),
        "centroids/cluster_sizes: {} vs {}",
        centroids.len(),
        cluster_sizes.len()
    );

    let header = IndexHeader {
        magic: MAGIC,
        version: VERSION,
        nlist: centroids.len() as u32,
        nprobe: NPROBE,
        n_vectors,
        _padding: [0; 8],
    };

    let hdr_size = std::mem::size_of::<IndexHeader>();
    let centroids_bytes = cast_slice::<[f32; N_DIMS], u8>(centroids);
    let cluster_sizes_bytes = cast_slice::<u32, u8>(cluster_sizes);
    let fixed_end = hdr_size + centroids_bytes.len() + cluster_sizes_bytes.len();
    let blocks_start = align_up(fixed_end, 32); // VectorBlock requires align(32)
    let pad_len = blocks_start - fixed_end;

    let file = File::create(output_path)
        .with_context(|| format!("falha ao criar {}", output_path.display()))?;
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, file);

    writer.write_all(bytes_of(&header))?;
    writer.write_all(centroids_bytes)?;
    writer.write_all(cluster_sizes_bytes)?;
    if pad_len > 0 {
        writer.write_all(&vec![0u8; pad_len])?;
    }
    writer.write_all(cast_slice::<VectorBlock, u8>(blocks))?;
    writer.write_all(labels)?;
    writer.flush()?;

    println!(
        "[build] serializado: {} centroids, {} blocks, {} vectors (padded)",
        centroids.len(),
        blocks.len(),
        n_vectors,
    );
    Ok(())
}

fn validate_index(output_path: &Path, sorted_vectors: &[[f32; N_DIMS]]) -> Result<()> {
    println!("[build] validação pós-build...");
    let index = IvfIndex::load(output_path)
        .with_context(|| format!("falha ao reabrir {}", output_path.display()))?;

    let sanity = 10usize.min(sorted_vectors.len());
    for (i, q) in sorted_vectors[..sanity].iter().enumerate() {
        let score = index.fraud_score(q);
        ensure!(score.is_finite(), "fraud_score não finito em sanity[{i}]");
    }

    let samples = 100usize.min(sorted_vectors.len());
    if samples == 0 {
        println!("[build] sanity check: {sanity} buscas ok");
        return Ok(());
    }

    let step = (sorted_vectors.len() / samples).max(1);
    let valid: usize = (0..sorted_vectors.len())
        .step_by(step)
        .take(samples)
        .filter(|&idx| {
            let top5 = index.search_knn(&sorted_vectors[idx], 5);
            !top5.is_empty() && top5.iter().all(|(d, _)| d.0.is_finite())
        })
        .count();

    println!("[build] validação: {valid}/{samples} buscas com resultados válidos");
    Ok(())
}

fn read_references(path: &Path) -> Result<(Vec<[f32; N_DIMS]>, Vec<u8>)> {
    let file = File::open(path)
        .with_context(|| format!("arquivo não encontrado: {}", path.display()))?;
    let gz = GzDecoder::new(file);
    let mut reader = BufReader::new(gz);

    if is_json_array(&mut reader)? {
        return read_as_json_array(reader);
    }
    read_as_ndjson(reader)
}

fn is_json_array<R: BufRead>(reader: &mut R) -> Result<bool> {
    loop {
        let buf = reader.fill_buf()?;
        ensure!(!buf.is_empty(), "arquivo de referências vazio");
        if let Some(pos) = buf.iter().position(|b| !b.is_ascii_whitespace()) {
            return Ok(buf[pos] == b'[');
        }
        let len = buf.len();
        reader.consume(len);
    }
}

fn read_as_ndjson<R: BufRead>(mut reader: R) -> Result<(Vec<[f32; N_DIMS]>, Vec<u8>)> {
    let mut vectors: Vec<[f32; N_DIMS]> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    let mut line = String::new();
    let started = Instant::now();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: ReferenceRecord = serde_json::from_str(trimmed)
            .with_context(|| format!("NDJSON inválido na linha {}", vectors.len() + 1))?;
        push_record(&mut vectors, &mut labels, rec);
        log_progress(vectors.len());
    }

    ensure!(!vectors.is_empty(), "nenhum registro lido de {}", "ndjson");
    println!("[build] {} vetores em {:.2}s", vectors.len(), started.elapsed().as_secs_f64());
    Ok((vectors, labels))
}

fn read_as_json_array<R: BufRead>(reader: R) -> Result<(Vec<[f32; N_DIMS]>, Vec<u8>)> {
    struct ReferencesVisitor;

    impl<'de> Visitor<'de> for ReferencesVisitor {
        type Value = (Vec<[f32; N_DIMS]>, Vec<u8>);

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("array de { vector, label }")
        }

        fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut vectors = Vec::new();
            let mut labels = Vec::new();
            let started = Instant::now();

            while let Some(rec) = seq.next_element::<ReferenceRecord>()? {
                push_record(&mut vectors, &mut labels, rec);
                log_progress(vectors.len());
            }

            println!(
                "[build] {} vetores em {:.2}s",
                vectors.len(),
                started.elapsed().as_secs_f64()
            );
            Ok((vectors, labels))
        }
    }

    let mut de = serde_json::Deserializer::from_reader(reader);
    let result = de.deserialize_seq(ReferencesVisitor)?;
    ensure!(!result.0.is_empty(), "nenhum registro lido do JSON array");
    Ok(result)
}

#[inline]
fn push_record(vectors: &mut Vec<[f32; N_DIMS]>, labels: &mut Vec<u8>, rec: ReferenceRecord) {
    vectors.push(rec.vector);
    labels.push(if rec.label == "fraud" { 1 } else { 0 });
}

#[inline]
fn log_progress(count: usize) {
    if count.is_multiple_of(LOG_EVERY) {
        let mb = (count * std::mem::size_of::<[f32; N_DIMS]>()) as f64 / (1024.0 * 1024.0);
        println!("[build] lidos {count} vetores ({mb:.1} MB)");
    }
}

#[inline]
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use flate2::Compression;
    use flate2::write::GzEncoder;

    use super::*;

    fn unique_tmp(name: &str, ext: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{nanos}.{ext}"))
    }

    #[test]
    fn read_ndjson_gz_parses_two_records() {
        let path = unique_tmp("refs-ndjson", "json.gz");
        let f = File::create(&path).unwrap();
        let mut gz = GzEncoder::new(f, Compression::default());
        gz.write_all(br#"{"vector":[0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0,0.0,0.0,1.0,0.0,0.5,0.0],"label":"legit"}"#).unwrap();
        gz.write_all(b"\n").unwrap();
        gz.write_all(br#"{"vector":[1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,0.0,1.0,0.8,1.0],"label":"fraud"}"#).unwrap();
        gz.write_all(b"\n").unwrap();
        gz.finish().unwrap();

        let (vectors, labels) = read_references(&path).unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(labels, [0u8, 1u8]);
        assert_eq!(vectors[0][10], 1.0);
        assert_eq!(vectors[1][11], 1.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_json_array_gz_parses_two_records() {
        let path = unique_tmp("refs-array", "json.gz");
        let f = File::create(&path).unwrap();
        let mut gz = GzEncoder::new(f, Compression::default());
        gz.write_all(br#"[{"vector":[0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0,0.0,0.0,1.0,0.0,0.5,0.0],"label":"legit"},{"vector":[1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,0.0,1.0,0.8,1.0],"label":"fraud"}]"#).unwrap();
        gz.finish().unwrap();

        let (vectors, labels) = read_references(&path).unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(labels, [0u8, 1u8]);
        assert_eq!(vectors[0][10], 1.0);
        assert_eq!(vectors[1][11], 1.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn build_blocks_pads_clusters_to_block_size() {
        // cluster 0: 3 vectors → padded to 8; cluster 1: 7 vectors → padded to 8
        let vectors: Vec<[f32; N_DIMS]> = (0..10).map(|i| [i as f32 / 100.0; N_DIMS]).collect();
        let labels: Vec<u8> = (0..10).map(|i| (i % 2) as u8).collect();
        let cluster_sizes = vec![3usize, 7usize];

        let (blocks, padded_sizes, padded_labels) = build_blocks(&vectors, &labels, &cluster_sizes);

        assert_eq!(padded_sizes, [8u32, 8u32]);
        assert_eq!(blocks.len(), 2);        // 16 padded / 8 = 2 blocks
        assert_eq!(padded_labels.len(), 16);

        // First cluster: real labels at slots 0..3, zeros at 3..8.
        for i in 0..3 {
            assert_eq!(padded_labels[i], labels[i], "slot {i}");
        }
        for i in 3..8 {
            assert_eq!(padded_labels[i], 0, "pad slot {i}");
        }
    }

    #[test]
    fn build_blocks_data_layout_is_soa() {
        // 1 cluster, 1 vector: dims [0.0, 1.0, ...], padded to BLOCK_SIZE
        let mut v = [0.0f32; N_DIMS];
        v[1] = 1.0;
        let vectors = vec![v];
        let labels = vec![0u8];
        let cluster_sizes = vec![1usize];

        let (blocks, _, _) = build_blocks(&vectors, &labels, &cluster_sizes);
        assert_eq!(blocks.len(), 1);

        // SoA: block.data[dim * BLOCK_SIZE + slot]
        // dim=0, slot=0 → quantize_i16(0.0) = 0
        assert_eq!(blocks[0].data[0 * BLOCK_SIZE + 0], 0);
        // dim=1, slot=0 → quantize_i16(1.0) = 10_000
        assert_eq!(blocks[0].data[1 * BLOCK_SIZE + 0], 10_000);
    }
}
