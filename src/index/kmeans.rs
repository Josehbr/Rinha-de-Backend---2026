use rand::Rng;
use rayon::prelude::*;

use crate::index::layout::N_DIMS;

/// Squared Euclidean distance between two f32 vectors.
#[inline]
fn l2sq_f32(a: &[f32; N_DIMS], b: &[f32; N_DIMS]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..N_DIMS {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// K-means++ centroid initialisation.
///
/// The first centroid is chosen uniformly at random. Each subsequent centroid
/// is chosen with probability proportional to D²(x) — the squared distance to
/// the nearest already-selected centroid.
pub fn kmeans_plus_plus_init(
    vectors: &[[f32; N_DIMS]],
    nlist: usize,
    rng: &mut impl Rng,
) -> Vec<[f32; N_DIMS]> {
    assert!(!vectors.is_empty());
    assert!(nlist <= vectors.len());

    let mut centroids: Vec<[f32; N_DIMS]> = Vec::with_capacity(nlist);

    let first = rng.random_range(0..vectors.len());
    centroids.push(vectors[first]);

    let mut dists: Vec<f32> = vectors.iter().map(|v| l2sq_f32(v, &centroids[0])).collect();

    for _ in 1..nlist {
        let total: f32 = dists.iter().sum();
        let mut threshold = rng.random::<f32>() * total;
        let mut chosen = dists.len() - 1;
        for (i, &d) in dists.iter().enumerate() {
            threshold -= d;
            if threshold <= 0.0 {
                chosen = i;
                break;
            }
        }
        let new_centroid = vectors[chosen];
        centroids.push(new_centroid);

        for (i, v) in vectors.iter().enumerate() {
            let d = l2sq_f32(v, &new_centroid);
            if d < dists[i] {
                dists[i] = d;
            }
        }
    }

    centroids
}

/// Assigns each vector to its nearest centroid. Returns cluster_id per vector.
///
/// Parallelised with rayon — this is the inner loop of K-means and dominates
/// training time at 3M vectors.
pub fn assign_clusters(
    vectors: &[[f32; N_DIMS]],
    centroids: &[[f32; N_DIMS]],
) -> Vec<u32> {
    vectors
        .par_iter()
        .map(|v| {
            centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (i, l2sq_f32(v, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap()
        })
        .collect()
}

/// Recomputes centroids as the mean of assigned vectors.
///
/// Empty clusters retain their previous centroid to avoid NaN from 0/0.
pub fn update_centroids(
    vectors: &[[f32; N_DIMS]],
    assignments: &[u32],
    nlist: usize,
    prev_centroids: &[[f32; N_DIMS]],
) -> Vec<[f32; N_DIMS]> {
    let mut sums = vec![[0.0f64; N_DIMS]; nlist];
    let mut counts = vec![0u64; nlist];

    for (v, &c) in vectors.iter().zip(assignments) {
        let ci = c as usize;
        for d in 0..N_DIMS {
            sums[ci][d] += v[d] as f64;
        }
        counts[ci] += 1;
    }

    let mut centroids = vec![[0.0f32; N_DIMS]; nlist];
    for i in 0..nlist {
        if counts[i] == 0 {
            centroids[i] = prev_centroids[i];
        } else {
            for d in 0..N_DIMS {
                centroids[i][d] = (sums[i][d] / counts[i] as f64) as f32;
            }
        }
    }
    centroids
}

/// Runs K-means with K-means++ initialisation for `n_iter` iterations.
///
/// Returns `(centroids, assignments)` where `assignments[i]` is the cluster id
/// of `vectors[i]`.
pub fn kmeans(
    vectors: &[[f32; N_DIMS]],
    nlist: usize,
    n_iter: usize,
    rng: &mut impl Rng,
) -> (Vec<[f32; N_DIMS]>, Vec<u32>) {
    let mut centroids = kmeans_plus_plus_init(vectors, nlist, rng);
    let mut assignments = vec![0u32; vectors.len()];

    for iter in 0..n_iter {
        let new_assignments = assign_clusters(vectors, &centroids);
        let new_centroids =
            update_centroids(vectors, &new_assignments, nlist, &centroids);

        let inertia: f32 = vectors
            .iter()
            .zip(&new_assignments)
            .map(|(v, &c)| l2sq_f32(v, &new_centroids[c as usize]))
            .sum();

        println!("[build] kmeans iter={} inertia={:.6}", iter + 1, inertia);

        assignments = new_assignments;
        centroids = new_centroids;
    }

    (centroids, assignments)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn synthetic_f32_vectors(n: usize, rng: &mut impl Rng) -> Vec<[f32; N_DIMS]> {
        (0..n)
            .map(|_| {
                let mut v = [0.0f32; N_DIMS];
                for x in v.iter_mut() {
                    *x = rng.random::<f32>();
                }
                v
            })
            .collect()
    }

    #[test]
    fn assign_clusters_trivial() {
        let near_zero = [0.05f32; N_DIMS];
        let near_one  = [0.95f32; N_DIMS];
        let vectors = vec![near_zero, near_one];

        let c0 = [0.0f32; N_DIMS];
        let c1 = [1.0f32; N_DIMS];
        let centroids = vec![c0, c1];

        let assignments = assign_clusters(&vectors, &centroids);
        assert_eq!(assignments[0], 0, "near_zero should go to cluster 0");
        assert_eq!(assignments[1], 1, "near_one should go to cluster 1");
    }

    #[test]
    fn kmeans_no_empty_clusters_and_inertia_finite() {
        let mut rng = SmallRng::seed_from_u64(42);
        let vectors = synthetic_f32_vectors(500, &mut rng);
        let nlist = 8;

        let (centroids, assignments) = kmeans(&vectors, nlist, 5, &mut rng);

        assert_eq!(centroids.len(), nlist);
        assert_eq!(assignments.len(), vectors.len());

        let mut counts = vec![0u32; nlist];
        for &a in &assignments {
            counts[a as usize] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > 0, "cluster {i} is empty");
        }

        let inertia: f32 = vectors
            .iter()
            .zip(&assignments)
            .map(|(v, &c)| l2sq_f32(v, &centroids[c as usize]))
            .sum();
        assert!(inertia.is_finite(), "inertia should be finite: {inertia}");
    }
}
