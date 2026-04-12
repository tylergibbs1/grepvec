//! Vector search recall benchmark.
//!
//! Tests TurboQuant recall by generating random normalized vectors,
//! computing brute-force ground truth, then measuring recall@k
//! through grepvec's VecIndex. No embedding model needed.
//!
//! Run with: `cargo test --test recall_benchmark -- --ignored --nocapture`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(target_os = "macos")]
extern crate accelerate_src;

use grepvec::search::vector::VecIndex;

/// Generate `n` random unit vectors of dimension `dim` using a fixed seed.
fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // Simple LCG PRNG (not crypto-quality, fine for benchmarks).
    let mut state = seed;
    let mut next_f32 = || -> f32 {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        // Map to [-1, 1].
        ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            // L2-normalize.
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

/// Brute-force cosine similarity search. Returns indices of top-k most similar vectors.
fn brute_force_topk(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = database
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            let dot: f32 = query.iter().zip(doc.iter()).map(|(a, b)| a * b).sum();
            (i, dot)
        })
        .collect();
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    scores.iter().take(k).map(|(i, _)| *i).collect()
}

/// Recall@1 at k: fraction of queries where the true top-1 appears in the top-k results.
fn recall_1_at_k(
    index: &mut VecIndex,
    queries: &[Vec<f32>],
    database: &[Vec<f32>],
    k: usize,
) -> f64 {
    let mut hits = 0;
    for query in queries {
        let true_top1 = brute_force_topk(query, database, 1)[0];
        let results = index.search(query, k);
        if results.iter().any(|(idx, _)| *idx == true_top1) {
            hits += 1;
        }
    }
    hits as f64 / queries.len() as f64
}

#[test]
#[ignore = "run with: cargo test --test recall_benchmark -- --ignored --nocapture"]
fn recall_turboquant_dim384() {
    // Matches bge-small embedding dimension.
    let dim = 384;
    let n_docs = 5_000;
    let n_queries = 100;

    eprintln!("Generating {n_docs} random vectors (dim={dim})...");
    let database = random_unit_vectors(n_docs, dim, 42);
    let queries = random_unit_vectors(n_queries, dim, 123);

    for bit_width in [4, 2] {
        eprintln!("\nBuilding TurboQuant index (dim={dim}, {bit_width}-bit)...");
        let mut index = VecIndex::build(&database, dim, bit_width).expect("build failed");
        eprintln!("  Index size: {} vectors", index.len());

        let mut results = Vec::new();
        for k in [1, 2, 4, 8, 16, 32, 64] {
            let r = recall_1_at_k(&mut index, &queries, &database, k);
            results.push((k, r));
        }

        eprintln!("\n  === dim={dim}, {bit_width}-bit TurboQuant ===");
        for (k, r) in &results {
            eprintln!("  Recall@1 at k={k:>2}: {r:.4}");
        }

        if bit_width == 4 {
            // At k=64, recall should converge toward 1.0.
            let r_at_64 = results.iter().find(|(k, _)| *k == 64).map(|(_, r)| *r).unwrap_or(0.0);
            assert!(
                r_at_64 > 0.90,
                "4-bit recall@1 at k=64 = {r_at_64:.4}, expected > 0.90"
            );
        }
    }
}

#[test]
#[ignore = "run with: cargo test --test recall_benchmark -- --ignored --nocapture"]
fn recall_turboquant_dim1024() {
    // Matches bge-large embedding dimension.
    let dim = 1024;
    let n_docs = 5_000;
    let n_queries = 100;

    eprintln!("Generating {n_docs} random vectors (dim={dim})...");
    let database = random_unit_vectors(n_docs, dim, 42);
    let queries = random_unit_vectors(n_queries, dim, 456);

    for bit_width in [4, 2] {
        eprintln!("\nBuilding TurboQuant index (dim={dim}, {bit_width}-bit)...");
        let mut index = VecIndex::build(&database, dim, bit_width).expect("build failed");

        let mut results = Vec::new();
        for k in [1, 2, 4, 8, 16, 32, 64] {
            let r = recall_1_at_k(&mut index, &queries, &database, k);
            results.push((k, r));
        }

        eprintln!("\n  === dim={dim}, {bit_width}-bit TurboQuant ===");
        for (k, r) in &results {
            eprintln!("  Recall@1 at k={k:>2}: {r:.4}");
        }

        if bit_width == 4 {
            let r_at_64 = results.iter().find(|(k, _)| *k == 64).map(|(_, r)| *r).unwrap_or(0.0);
            assert!(
                r_at_64 > 0.90,
                "4-bit dim=1024 recall@1 at k=64 = {r_at_64:.4}, expected > 0.90"
            );
        }
    }
}
