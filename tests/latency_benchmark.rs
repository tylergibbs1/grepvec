//! Search latency benchmark.
//!
//! Measures per-query search time on a 5000-vector index (dim=384, 4-bit).
//! Reports microseconds per query as a greppable metric.
//!
//! Run with: `cargo test --test latency_benchmark -- --ignored --nocapture`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(target_os = "macos")]
extern crate accelerate_src;

use grepvec::search::vector::VecIndex;
use std::time::Instant;

/// Generate `n` random unit vectors of dimension `dim` using a fixed seed.
fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next_f32 = || -> f32 {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
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

#[test]
#[ignore = "run with: cargo test --test latency_benchmark -- --ignored --nocapture"]
fn search_latency_dim384() {
    let dim = 384;
    let n_docs = 5_000;
    let n_queries = 100;
    let k = 10;

    eprintln!("Building index: {n_docs} vectors, dim={dim}, 4-bit...");
    let database = random_unit_vectors(n_docs, dim, 42);
    let queries = random_unit_vectors(n_queries, dim, 789);

    let mut index = VecIndex::build(&database, dim, 4).expect("build failed");

    // Warmup: run a few searches to populate caches
    for q in queries.iter().take(5) {
        let _ = index.search(q, k);
    }

    // Timed run: search all queries one at a time
    let start = Instant::now();
    for q in &queries {
        let _ = index.search(q, k);
    }
    let elapsed = start.elapsed();

    let us_per_query = elapsed.as_micros() as f64 / n_queries as f64;
    eprintln!("  {n_queries} queries in {:.2?}", elapsed);
    eprintln!("  {us_per_query:.0} us/query (k={k})");

    // Greppable metric
    println!("us_per_query: {us_per_query:.0}");
}
